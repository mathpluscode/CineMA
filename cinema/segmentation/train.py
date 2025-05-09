"""Segmentation model training and evaluation utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch
from monai.losses import DiceLoss
from monai.metrics import compute_dice, compute_hausdorff_distance, compute_iou
from monai.networks.utils import one_hot
from torch import nn
from torch.nn import functional as F  # noqa: N812

from cinema.log import get_logger
from cinema.metric import get_volumes, stability_score
from cinema.segmentation.convunetr import get_model
from cinema.segmentation.unet import UNet
from cinema.transform import aggregate_patches, crop_start, get_patch_grid, patch_grid_sample

if TYPE_CHECKING:
    from collections.abc import Callable

    from omegaconf import DictConfig
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


def get_segmentation_model(config: DictConfig) -> nn.Module:
    """Initialize and load the model.

    Args:
        config: configuration file.

    Returns:
        model: loaded model.
    """
    if hasattr(config.model, "view"):
        # backward compatibility
        views = [config.model.view]
    else:
        views = [config.model.views] if isinstance(config.model.views, str) else config.model.views

    def get_view_config(v: str) -> DictConfig:
        if v == "sax":
            return config.data.sax
        if hasattr(config.data, "lax"):
            return config.data.lax
        return config.data[v]

    if config.model.name == "convunetr":
        model = get_model(config)
    elif config.model.name == "unet":
        if len(views) > 1:
            raise ValueError("UNet only supports single view.")
        view = views[0]
        view_config = get_view_config(view)
        ndim = 3 if view == "sax" else 2
        model = UNet(
            n_dims=len(view_config.spacing),
            in_chans=view_config.in_chans,
            out_chans=config.model.out_chans,
            patch_size=config.model.unet.patch_size[:ndim],
            chans=config.model.unet.chans,
            scale_factor=config.model.unet.scale_factor[:ndim],
            dropout=config.model.unet.dropout,
        )
    else:
        raise ValueError(f"Invalid model name {config.model.name}.")
    if hasattr(model, "set_grad_ckpt"):
        model.set_grad_ckpt(config.grad_ckpt)
    return model


def _segmentation_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss function for one view.

    All classes are mutually exclusive.

    Args:
        logits: (batch, n_classes, ...).
        labels: target labels, (batch, 1, ...).

    Returns:
        loss: total loss scalar.
        metrics: each value is a scalar tensor.
    """
    labels = labels.long()
    mask = one_hot(labels.clamp(min=0), num_classes=logits.shape[1], dtype=logits.dtype)  # (batch, n_classes, ...)
    ce = F.cross_entropy(logits, labels.squeeze(dim=1), ignore_index=-1)
    dice = DiceLoss(
        include_background=False,
        to_onehot_y=False,
        softmax=True,
    )(logits, mask)
    loss = dice + ce
    metrics = {"cross_entropy": ce, "mean_dice_loss": dice, "loss": loss}
    return loss, metrics


def segmentation_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    views: list[str],
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, dict[str, torch.Tensor]]] = _segmentation_loss,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss function for segmentation tasks.

    All classes are mutually exclusive.

    Args:
        model: model to train.
        batch: dictionary with view keys, each value is (batch, channel, *image_size).
        views: list of view keys.
        device: device to use.
        loss_fn: function to compute loss and metrics for one view.

    Returns:
        loss: total loss scalar tensor.
        metrics: each value is a float.
    """
    image_dict = {view: batch[f"{view}_image"].to(device) for view in views}
    label_dict = {view: batch[f"{view}_label"].to(device) for view in views}
    logits_dict = model(image_dict)

    metrics = {}
    losses = []
    for view, logits in logits_dict.items():
        loss_view, metrics_view = loss_fn(logits, label_dict[view])
        metric_keys = list(metrics_view.keys())
        metrics_view[f"{view}_loss"] = loss_view
        losses.append(loss_view)
        metrics.update({f"{view}_{k}": v for k, v in metrics_view.items()})
    loss = sum(losses) / len(logits_dict)
    metrics["loss"] = loss
    metrics = {k: v.item() for k, v in metrics.items()}
    for k in metric_keys:
        metrics[k] = np.mean([metrics[f"{view}_{k}"] for view in logits_dict])
    return loss, metrics


def segmentation_forward(
    model: nn.Module,
    image_dict: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Forward pass for evaluation.

    For overlapped patches, the logits are softmaxed to probabilities and averaged, then log is taken.
    This may not be suitable for other tasks such as landmark detection.

    Args:
        model: model to evaluate.
        image_dict: dictionary with view keys, each value is (1, channel, *image_size).
            image size should larger or equal to patch size.
        patch_size_dict: patch size per view for evaluation.
        amp_dtype: dtype for mixed precision training.

    Returns:
        logits_dict: output logits per view, (1, n_classes, *image_size).
    """
    for view, image in image_dict.items():
        if any(s < p for s, p in zip(image.shape[2:], patch_size_dict[view], strict=False)):
            raise ValueError(
                f"For view {view}, image size {image.shape[2:]} is smaller than patch size {patch_size_dict[view]}."
            )

    views = list(image_dict.keys())
    need_patch_dict = {view: image_dict[view].shape[2:] != patch_size_dict[view] for view in views}

    if not any(need_patch_dict.values()):
        # no need to patch
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            return model(image_dict)

    # need patch, but can only patch on one view
    if sum(need_patch_dict.values()) > 1:
        raise ValueError(f"Only support patching on one view for now, but got {need_patch_dict}.")
    batch_size = image_dict[views[0]].shape[0]
    if batch_size != 1:
        raise ValueError(f"Expected batch size 1 for patching, but got {batch_size}.")

    view_to_patch = next(view for view, need_patch in need_patch_dict.items() if need_patch)
    image_to_patch = image_dict[view_to_patch][0]  # (channel, *image_size)
    patch_size = patch_size_dict[view_to_patch]
    patch_overlap = tuple(s // 2 for s in patch_size)
    patch_start_indices = get_patch_grid(
        image_size=image_to_patch.shape[1:],
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    patches = patch_grid_sample(image_to_patch, patch_start_indices, patch_size)  # (n_patches, channel, *patch_size)
    n_patches = patches.shape[0]

    # segmentation specific
    logits_dict: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for i in range(n_patches):
            patch = patches[i : i + 1, ...]
            patch_image_dict = {view: patch if view == view_to_patch else image_dict[view] for view in views}
            patch_logits_dict = model(patch_image_dict)
            for view in views:
                logits_dict[view].append(patch_logits_dict[view])
    aggregated_logits_dict: dict[str, torch.Tensor] = {}
    for view in views:
        logits = torch.cat(logits_dict[view], dim=0)  # (n_patches, out_chans, *image_size)
        if view == view_to_patch:
            probs = F.softmax(logits, dim=1)
            probs = aggregate_patches(probs, start_indices=patch_start_indices, image_size=image_to_patch.shape[1:])
            logits = torch.log(probs)  # (out_chans, *image_size)
        else:
            logits = torch.log(torch.mean(F.softmax(logits, dim=1), dim=0))  # (out_chans, *image_size)
        aggregated_logits_dict[view] = logits[None, ...]  # (1, out_chans, *image_size)
    return aggregated_logits_dict


def segmentation_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spacing: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Evaluation metrics for segmentation tasks.

    Padded region is considered as background as the metrics are focused on foreground classes.
    Logits are modified to have background class at index 0.

    MONAI only supports up to 3D images.

    Args:
        logits: (batch, 1+n_classes, ...), n_classes is number of foreground classes.
        labels: target labels, (batch, 1, ...).
        spacing: pixel/voxel spacing in mm.

    Returns:
        metrics: each value is of shape (batch,).
    """
    n_classes = logits.shape[1] - 1  # exclude background class
    labels = labels.squeeze(dim=1).long()

    pred_labels = torch.argmax(logits, dim=1)
    pred_mask = F.one_hot(pred_labels, n_classes + 1).moveaxis(-1, 1)
    true_mask = F.one_hot(labels, n_classes + 1).moveaxis(-1, 1)

    # (batch, n_classes+1)
    dice = compute_dice(
        y_pred=pred_mask,
        y=true_mask,
        num_classes=n_classes + 1,
    )
    iou = compute_iou(
        y_pred=pred_mask,
        y=true_mask,
    )
    stability = stability_score(logits=logits)
    hausdorff_dist = compute_hausdorff_distance(
        y_pred=pred_mask,
        y=true_mask,
        percentile=95,
        spacing=spacing,
    )
    true_volumes = get_volumes(mask=true_mask, spacing=spacing)  # (batch, n_classes+1)
    pred_volumes = get_volumes(mask=pred_mask, spacing=spacing)  # (batch, n_classes+1)

    metrics = {}
    for i in range(n_classes):
        cls_idx = i + 1
        metrics[f"class_{cls_idx}_dice_score"] = dice[:, cls_idx]
        metrics[f"class_{cls_idx}_iou_score"] = iou[:, cls_idx]
        metrics[f"class_{cls_idx}_stability_score"] = stability[:, cls_idx]
        metrics[f"class_{cls_idx}_hausdorff_distance_95"] = hausdorff_dist[:, cls_idx - 1]
        metrics[f"class_{cls_idx}_true_volume"] = true_volumes[:, cls_idx]
        metrics[f"class_{cls_idx}_pred_volume"] = pred_volumes[:, cls_idx]

    metrics["mean_dice_score"] = torch.mean(dice[:, 1:], dim=-1)
    metrics["mean_iou_score"] = torch.mean(iou[:, 1:], dim=-1)
    metrics["mean_stability_score"] = torch.mean(stability[:, 1:], dim=-1)
    metrics["mean_hausdorff_distance_95"] = torch.mean(hausdorff_dist, dim=-1)

    return metrics


def segmentation_eval(  # noqa: C901
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
    metrics_fn: Callable[[torch.Tensor, torch.Tensor, tuple[float, ...]], dict[str, torch.Tensor]] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Evaluate segmentation model on a batch.

    Batch size is assumed to be 1 for potential patching.

    Args:
        model: model to task.
        batch: dictionary with view keys, each value is (1, channel, *image_size).
        patch_size_dict: patch size per view for evaluation.
        spacing_dict: spacing of voxels/pixels per view.
        amp_dtype: dtype for mixed precision training.
        device: device to use.
        metrics_fn: function to compute evaluation metrics, needs to be overridden for EMIDEC.

    Returns:
        logits_dict: output logits per view, (1, channel, *image_size).
        metrics: each value is a scalar.
    """
    views = list(patch_size_dict.keys())
    image_dict = {view: batch[f"{view}_image"].to(device) for view in views}

    # (1, n_classes, *image_size)
    logits_dict = segmentation_forward(model, image_dict, patch_size_dict, amp_dtype)

    # crop out padded regions before evaluation
    for view in views:
        width = int(batch[f"{view}_width"][0])
        height = int(batch[f"{view}_height"][0])
        if len(patch_size_dict[view]) == 3:
            n_slices = int(batch["n_slices"][0])
            logits_dict[view] = crop_start(logits_dict[view], (*logits_dict[view].shape[:2], width, height, n_slices))
        elif len(patch_size_dict[view]) == 2:
            logits_dict[view] = crop_start(logits_dict[view], (*logits_dict[view].shape[:2], width, height))
        else:
            raise ValueError(f"Invalid patch size {patch_size_dict[view]}.")

    # for myops dataset test split, ground truth is not available
    if metrics_fn is None:
        return logits_dict, {}

    # crop out padded regions before evaluation
    label_dict = {view: batch[f"{view}_label"].to(device) for view in views}
    for view in views:
        width = int(batch[f"{view}_width"][0])
        height = int(batch[f"{view}_height"][0])
        if len(patch_size_dict[view]) == 3:
            n_slices = int(batch["n_slices"][0])
            label_dict[view] = crop_start(label_dict[view], (*label_dict[view].shape[:2], width, height, n_slices))
        elif len(patch_size_dict[view]) == 2:
            label_dict[view] = crop_start(label_dict[view], (*label_dict[view].shape[:2], width, height))
        else:
            raise ValueError(f"Invalid patch size {patch_size_dict[view]}.")

    metrics = {}
    for view in views:
        metrics_view = metrics_fn(logits_dict[view], label_dict[view], spacing_dict[view])
        metric_keys = list(metrics_view.keys())
        for k, v in metrics_view.items():
            metrics[f"{view}_{k}"] = float(v.cpu().to(dtype=torch.float32).numpy())
    for k in metric_keys:
        metrics[k] = np.mean([metrics[f"{view}_{k}"] for view in views])
    return logits_dict, metrics


def segmentation_eval_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
    metrics_fn: Callable[
        [torch.Tensor, torch.Tensor, tuple[float, ...]], dict[str, torch.Tensor]
    ] = segmentation_metrics,
) -> dict[str, float]:
    """Evaluate segmentation model on a dataloader.

    Args:
        model: model to task.
        dataloader: dataloader to evaluate.
        patch_size_dict: patch size per view for evaluation.
        spacing_dict: spacing of voxels/pixels per view.
        amp_dtype: dtype for mixed precision training.
        device: device to use.
        metrics_fn: function to compute evaluation metrics, needs to be overridden for EMIDEC.

    Returns:
        mean_metrics: mean metrics over the data.
    """
    metrics: dict[str, list[float]] = defaultdict(list)  # each value is a list of floats
    for _, batch in enumerate(dataloader):  # batch size is 1
        _, sample_metrics = segmentation_eval(
            model,
            batch,
            patch_size_dict,
            spacing_dict,
            amp_dtype,
            device,
            metrics_fn,
        )
        for k, v in sample_metrics.items():
            metrics[k].append(v)
    mean_metrics = {k: np.nanmean(v) for k, v in metrics.items()}
    return mean_metrics
