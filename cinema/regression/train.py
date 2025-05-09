"""Regression model training and evaluation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from cinema.log import get_logger
from cinema.transform import get_patch_grid, patch_grid_sample

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


def regression_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    views: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss function for regression tasks.

    Args:
        model: model to train.
        batch: dictionary with view, each view image is (batch, n_classes, *image_size).
        views: list of views.
        device: device to use.

    Returns:
        loss: total loss scalar.
        metrics: each value is a scalar tensor.
    """
    image_dict = {view: batch[f"{view}_image"].to(device) for view in views}
    preds = model(image_dict)  # (batch, n)
    label = batch["label"].to(dtype=preds.dtype, device=device)  # (batch, n)
    mse = F.mse_loss(preds, label)
    mae = F.l1_loss(preds, label)
    max_label, min_label = torch.max(label), torch.min(label)
    max_pred, min_pred = torch.max(preds), torch.min(preds)
    metrics = {
        "mse_loss": mse.item(),
        "loss": mse.item(),
        "mae_loss": mae.item(),
        "max_label": max_label.item(),
        "min_label": min_label.item(),
        "max_pred": max_pred.item(),
        "min_pred": min_pred.item(),
    }
    return mse, metrics


def regression_forward(
    model: nn.Module,
    image_dict: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    """Forward pass for evaluation.

    For overlapped patches, the logits are softmaxed to probabilities and averaged, then log is taken.

    Args:
        model: model to evaluate.
        image_dict: dictionary with view keys, each value is (1, n_classes, *image_size).
            image size should larger or equal to patch size.
        patch_size_dict: patch size per view for evaluation.
        amp_dtype: dtype for mixed precision training.

    Returns:
        logits_dict: output logits per view, (1, 1).
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

    # regression specific
    patch_preds_list = []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for i in range(n_patches):
            patch = patches[i : i + 1, ...]
            patch_image_dict = {view: patch if view == view_to_patch else image_dict[view] for view in views}
            patch_preds = model(patch_image_dict)  # (1, 1)
            patch_preds_list.append(patch_preds)
    preds = torch.mean(torch.cat(patch_preds_list), dim=0, keepdim=True)  # (1, 1)
    return preds


def regression_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        true_labels: true labels, (n_samples,).
        pred_labels: predicted labels, (n_samples,).

    Returns:
        metrics: regression metrics.
    """
    abs_error = np.abs(true_labels - pred_labels)
    return {
        "rmse": np.sqrt(np.mean(abs_error**2)),
        "mae": np.mean(abs_error),
        "max_error": np.max(abs_error),
        "min_error": np.min(abs_error),
        "max_label": np.max(true_labels),
        "min_label": np.min(true_labels),
        "max_pred": np.max(pred_labels),
        "min_pred": np.min(pred_labels),
    }


def regression_eval(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Evaluate segmentation model on a batch.

    Batch size is assumed to be 1 for potential patching.

    Args:
        model: model to task.
        batch: dictionary with view keys, each value is (1, n_classes, *image_size).
        patch_size_dict: patch size per view for evaluation.
        amp_dtype: dtype for mixed precision training.
        device: device to use.

    Returns:
        logits: (1, n_classes).
        metrics: each value is a scalar.
    """
    views = list(patch_size_dict.keys())
    image_dict = {view: batch[f"{view}_image"].to(device) for view in views}
    preds = regression_forward(model, image_dict, patch_size_dict, amp_dtype)
    return preds, {}


def regression_eval_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],  # noqa: ARG001, pylint: disable=unused-argument
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate regression model on a dataloader.

    Args:
        model: model to task.
        dataloader: dataloader to evaluate.
        patch_size_dict: patch size per view for evaluation.
        spacing_dict: spacing of voxels/pixels per view, not used.
        amp_dtype: dtype for mixed precision training.
        device: device to use.

    Returns:
        mean_metrics: mean metrics over the data.
    """
    pred_labels = []
    true_labels = []
    for _, batch in enumerate(dataloader):
        preds, _ = regression_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device
        )
        pred_labels.append(preds)
        true_labels.append(batch["label"])
    pred_labels = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    reg_mean = dataloader.dataset.reg_mean
    reg_std = dataloader.dataset.reg_std
    restored_pred_labels = (pred_labels * reg_std) + reg_mean
    restored_true_labels = (true_labels * reg_std) + reg_mean
    metrics = regression_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
    )
    restored_metrics = regression_metrics(
        true_labels=restored_true_labels,
        pred_labels=restored_pred_labels,
    )
    restored_metrics = {f"restored_{k}": v for k, v in restored_metrics.items()}
    metrics.update(restored_metrics)
    return metrics
