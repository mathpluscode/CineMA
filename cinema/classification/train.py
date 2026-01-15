"""Classification model training and evaluation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score
from torch.nn import functional as F  # noqa: N812

from cinema.convvit import ConvViT
from cinema.log import get_logger
from cinema.resnet import get_resnet2d, get_resnet3d
from cinema.transform import get_patch_grid, patch_grid_sample
from cinema.vit import get_vit_config

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import nn
    from torch.utils.data import DataLoader

logger = get_logger(__name__)


def get_classification_or_regression_model(config: DictConfig) -> nn.Module:
    """Initialize and load the model.

    Args:
        config: configuration file.

    Returns:
        model: loaded model.
    """
    if hasattr(config.data, "class_column"):
        out_chans = len(config.data[config.data.class_column])
    elif hasattr(config.data, "regression_column"):
        out_chans = 1
    else:
        logger.info(f"Using config.model.out_chans {config.model.out_chans}.")
        out_chans = config.model.out_chans
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    in_chans_dict = {v: config.data.sax.in_chans if v == "sax" else config.data.lax.in_chans for v in views}
    if config.model.name == "convvit":
        vit_config = get_vit_config(config.model.convvit.size)
        image_size_dict = {v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views}
        ndim_dict = {v: 3 if v == "sax" else 2 for v in views}
        enc_patch_size_dict = {v: config.model.convvit.enc_patch_size[:n] for v, n in ndim_dict.items()}
        enc_scale_factor_dict = {v: config.model.convvit.enc_scale_factor[:n] for v, n in ndim_dict.items()}
        model = ConvViT(
            image_size_dict=image_size_dict,
            n_frames=config.model.n_frames,
            in_chans_dict=in_chans_dict,
            out_chans=out_chans,
            enc_patch_size_dict=enc_patch_size_dict,
            enc_scale_factor_dict=enc_scale_factor_dict,
            enc_conv_chans=config.model.convvit.enc_conv_chans,
            enc_conv_n_blocks=config.model.convvit.enc_conv_n_blocks,
            enc_embed_dim=vit_config["enc_embed_dim"],
            enc_depth=vit_config["enc_depth"],
            enc_n_heads=vit_config["enc_n_heads"],
            drop_path=config.model.convvit.drop_path,
        )
    elif config.model.name == "resnet":
        if len(views) > 1:
            raise ValueError("ResNet only supports single view.")
        view = views[0]
        get_fn = get_resnet3d if view == "sax" else get_resnet2d
        model = get_fn(
            depth=config.model.resnet.depth,
            in_chans=in_chans_dict[view] * config.model.n_frames,
            out_chans=out_chans,
            layer_inplanes=config.model.resnet.layer_inplanes,
        )
    else:
        raise ValueError(f"Invalid model name {config.model.name}.")
    if hasattr(model, "set_grad_ckpt"):
        model.set_grad_ckpt(config.grad_ckpt)
    return model


def classification_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    views: list[str],
    device: torch.device,
    label_smoothing: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss function for classification tasks.

    All classes are mutually exclusive.

    Args:
        model: model to train.
        batch: dictionary with view, each view image is (batch, n_classes, *image_size).
        views: list of views.
        device: device to use.
        label_smoothing: label smoothing factor.

    Returns:
        loss: total loss scalar.
        metrics: each value is a scalar tensor.
    """
    image_dict = {view: batch[f"{view}_image"].to(device) for view in views}
    logits = model(image_dict)
    label = batch["label"].long().to(device)
    ce = F.cross_entropy(
        logits,
        label,
        label_smoothing=label_smoothing,
    )
    metrics = {"cross_entropy": ce.item(), "loss": ce.item()}
    return ce, metrics


def classification_forward(
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
        logits_dict: output logits per view, (1, n_classes).
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

    # classification specific
    patch_probs_list = []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
        for i in range(n_patches):
            patch = patches[i : i + 1, ...]
            patch_image_dict = {view: patch if view == view_to_patch else image_dict[view] for view in views}
            patch_logits = model(patch_image_dict)
            patch_probs = F.softmax(patch_logits, dim=1)  # (1, n_classes)
            patch_probs_list.append(patch_probs)
    patch_probs = torch.cat(patch_probs_list, dim=0)  # (n_patches, n_classes)
    logits = torch.log(torch.mean(patch_probs, dim=0, keepdim=True))  # (1, n_classes)
    return logits


def binary_classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Compute binary classification evaluation metrics.

    Args:
        true_labels: true labels, (n_samples,).
        pred_labels: predicted labels, (n_samples,).
        pred_probs: predicted probabilities, (n_samples, n_classes).
        n_classes: number of classes.

    Returns:
        metrics: classification metrics.
    """
    if n_classes != 2:
        raise ValueError(f"Expected n_classes=2, but got {n_classes}.")
    labels = list(range(n_classes))

    metrics: dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true=true_labels, y_pred=pred_labels)
    metrics["entropy"] = -np.mean(np.sum(pred_probs * np.log(pred_probs + 1e-6), axis=1))

    cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=labels)
    if cm.shape[0] != n_classes:
        raise ValueError(f"Confusion matrix shape {cm.shape} does not match n_classes {n_classes}.")
    logger.info(f"Confusion matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp)
    metrics["sensitivity"] = tp / (tp + fn)
    metrics["f1"] = f1_score(y_true=true_labels, y_pred=pred_labels, labels=labels)
    if len(np.unique(true_labels)) > 1:
        metrics["mcc"] = matthews_corrcoef(y_true=true_labels, y_pred=pred_labels)
        metrics["roc_auc"] = roc_auc_score(y_true=true_labels, y_score=pred_probs[:, 1], labels=labels)
    else:
        metrics["mcc"] = 0.0
        metrics["roc_auc"] = 0.0
    logger.info("Accuracy | ROC AUC | F1 | MCC | Specificity | Sensitivity")
    logger.info(
        f"{metrics['accuracy'] * 100:.2f} | {metrics['roc_auc'] * 100:.2f} | "
        f"{metrics['f1'] * 100:.2f} | {metrics['mcc'] * 100:.2f} | "
        f"{metrics['specificity'] * 100:.2f} | {metrics['sensitivity'] * 100:.2f}"
    )
    return metrics


def multiclass_classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    """Compute classification evaluation metrics.

    Args:
        true_labels: true labels, (n_samples,).
        pred_labels: predicted labels, (n_samples,).
        pred_probs: predicted probabilities, (n_samples, n_classes).
        n_classes: number of classes.

    Returns:
        metrics: classification metrics.
    """
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=labels)
    logger.info(f"Confusion matrix:\n{cm}")

    metrics: dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true=true_labels, y_pred=pred_labels)
    metrics["entropy"] = -np.mean(np.sum(pred_probs * np.log(pred_probs + 1e-6), axis=1))
    metrics["f1"] = f1_score(y_true=true_labels, y_pred=pred_labels, average="micro", labels=labels)
    if len(np.unique(true_labels)) > 1:
        metrics["mcc"] = matthews_corrcoef(y_true=true_labels, y_pred=pred_labels)
        metrics["roc_auc"] = roc_auc_score(
            y_true=true_labels, y_score=pred_probs, average="macro", multi_class="ovo", labels=labels
        )
    else:
        metrics["mcc"] = 0.0
        metrics["roc_auc"] = 0.0
    logger.info("Accuracy | ROC AUC | F1 | MCC")
    logger.info(
        f"{metrics['accuracy'] * 100:.2f} | {metrics['roc_auc'] * 100:.2f} | "
        f"{metrics['f1'] * 100:.2f} | {metrics['mcc'] * 100:.2f}"
    )
    return metrics


def classification_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray,
) -> dict[str, float]:
    """Compute classification evaluation metrics.

    Args:
        true_labels: true labels, (n_samples,).
        pred_labels: predicted labels, (n_samples,).
        pred_probs: predicted probabilities, (n_samples, n_classes).

    Returns:
        metrics: classification metrics.
    """
    n_classes = pred_probs.shape[1]
    if n_classes == 2:
        return binary_classification_metrics(
            true_labels=true_labels, pred_labels=pred_labels, pred_probs=pred_probs, n_classes=n_classes
        )
    return multiclass_classification_metrics(
        true_labels=true_labels, pred_labels=pred_labels, pred_probs=pred_probs, n_classes=n_classes
    )


def classification_eval(
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
    logits = classification_forward(model, image_dict, patch_size_dict, amp_dtype)
    return logits, {}


def classification_eval_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],  # noqa: ARG001, pylint: disable=unused-argument
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate classification model on a dataloader.

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
    pred_logits = []
    pids = []
    for _, batch in enumerate(dataloader):
        logits, _ = classification_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device
        )
        pred_labels.append(torch.argmax(logits, dim=1))
        true_labels.append(batch["label"])
        pred_logits.append(logits)
        pids += batch["pid"]
    pred_labels = torch.cat(pred_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().to(dtype=torch.float32).numpy()
    pred_logits = torch.cat(pred_logits, dim=0).cpu().to(dtype=torch.float32)
    pred_probs = F.softmax(pred_logits, dim=1).numpy()  # softmax after dtype conversion to ensure sum=1
    metrics = classification_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
        pred_probs=pred_probs,
    )
    return metrics
