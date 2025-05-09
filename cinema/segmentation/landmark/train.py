"""Script to train."""

from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
import torch
from monai.losses import DiceLoss
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
)
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F  # noqa: N812

from cinema.log import get_logger
from cinema.metric import heatmap_soft_argmax
from cinema.segmentation.landmark.dataset import LandmarkDetectionDataset
from cinema.segmentation.train import (
    get_segmentation_model,
    segmentation_eval,
    segmentation_eval_dataloader,
    segmentation_loss,
)
from cinema.train import maybe_subset_dataset, run_train
from cinema.transform import aggregate_patches, get_patch_grid, patch_grid_sample

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset

logger = get_logger(__name__)


def load_dataset(config: DictConfig) -> tuple[Dataset, Dataset]:
    """Load and split the dataset.

    Args:
        config: configuration file.

    Returns:
        train_dataset: dataset for training.
        val_dataset: dataset for validation.
        config: updated config.
    """
    data_dir = Path(config.data.dir).expanduser()
    views = config.model.views
    if not isinstance(views, str):
        raise TypeError(f"Multiple views not supported: {views}")
    train_meta_df = pd.read_csv(data_dir / f"{views}_train.csv")
    val_meta_df = pd.read_csv(data_dir / f"{views}_val.csv")
    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    # transform for label is different from segmentation, where the mode is bilinear instead of nearest
    train_transform = Compose(
        [
            RandAdjustContrastd(keys=f"{views}_image", prob=config.transform.prob, gamma=config.transform.gamma),
            RandGaussianNoised(keys=f"{views}_image", prob=config.transform.prob),
            ScaleIntensityd(keys=f"{views}_image"),
            RandAffined(
                keys=(f"{views}_image", f"{views}_label"),
                mode=("bilinear", "bilinear"),
                prob=config.transform.prob,
                rotate_range=tuple(r / 180 * np.pi for r in config.transform.lax.rotate_range),
                translate_range=config.transform.lax.translate_range,
                scale_range=config.transform.scale_range,
                padding_mode="zeros",
                lazy=True,
            ),
            RandSpatialCropd(keys=(f"{views}_image", f"{views}_label"), roi_size=config.data.lax.patch_size, lazy=True),
            SpatialPadd(
                keys=(f"{views}_image", f"{views}_label"),
                spatial_size=config.data.lax.patch_size,
                method="end",
                lazy=True,
            ),
        ]
    )
    val_transform = Compose(
        [
            ScaleIntensityd(keys=f"{views}_image"),
            SpatialPadd(
                keys=(f"{views}_image", f"{views}_label"),
                spatial_size=config.data.lax.patch_size,
                method="end",
                lazy=True,
            ),
        ]
    )
    train_dataset = LandmarkDetectionDataset(
        data_dir=data_dir, view=views, meta_df=train_meta_df, transform=train_transform
    )
    val_dataset = LandmarkDetectionDataset(data_dir=data_dir, view=views, meta_df=val_meta_df, transform=val_transform)
    return train_dataset, val_dataset


def _landmark_detection_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss function for landmark detection tasks.

    All classes are independent.

    Args:
        logits: (batch, 3, ...).
        labels: (batch, 3, ...).

    Returns:
        loss: total loss scalar.
        metrics: each value is a scalar tensor.
    """
    probs = F.sigmoid(logits)  # (batch, 3, ...)
    dice = DiceLoss(
        include_background=True,
    )(probs, labels)
    bce = BCEWithLogitsLoss()(logits, labels)
    loss = dice + bce
    metrics = {"bce_loss": bce, "dice_loss": dice, "loss": loss}
    return loss, metrics


def landmark_detection_forward(
    model: nn.Module,
    image_dict: dict[str, torch.Tensor],
    patch_size_dict: dict[str, tuple[int, ...]],
    amp_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Forward pass for evaluation.

    For overlapped patches, the logits are sigmoided to probabilities and averaged, then reversed.
    Thus this cannot be just inheriting from segmentation_forward.

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

    # landmark detection specific
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
            probs = F.sigmoid(logits)
            probs = aggregate_patches(probs, start_indices=patch_start_indices, image_size=image_to_patch.shape[1:])
        else:
            probs = torch.mean(F.sigmoid(logits), dim=0)  # (out_chans, *image_size)
        logits = torch.log(probs / (1 - probs))  # (out_chans, *image_size)
        aggregated_logits_dict[view] = logits[None, ...]  # (1, out_chans, *image_size)
    return aggregated_logits_dict


def landmark_detection_coords_metrics(
    pred_labels: torch.Tensor,
    true_labels: torch.Tensor,
    spacing: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Evaluation metrics for landmark coordinates.

    Args:
        pred_labels: predicted coordinates, (batch, 6).
        true_labels: target coordinates, (batch, 6).
        spacing: pixel/voxel spacing in mm, not used.

    Returns:
        metrics: each value is of shape (batch,).
    """
    dx1 = (pred_labels[:, 0] - true_labels[:, 0]) * spacing[0]
    dy1 = (pred_labels[:, 1] - true_labels[:, 1]) * spacing[1]
    dx2 = (pred_labels[:, 2] - true_labels[:, 2]) * spacing[0]
    dy2 = (pred_labels[:, 3] - true_labels[:, 3]) * spacing[1]
    dx3 = (pred_labels[:, 4] - true_labels[:, 4]) * spacing[0]
    dy3 = (pred_labels[:, 5] - true_labels[:, 5]) * spacing[1]

    d1 = torch.sqrt(dx1**2 + dy1**2)  # (batch,)
    d2 = torch.sqrt(dx2**2 + dy2**2)
    d3 = torch.sqrt(dx3**2 + dy3**2)

    return {
        "pred_x1": pred_labels[:, 0],
        "pred_y1": pred_labels[:, 1],
        "pred_x2": pred_labels[:, 2],
        "pred_y2": pred_labels[:, 3],
        "pred_x3": pred_labels[:, 4],
        "pred_y3": pred_labels[:, 5],
        "true_x1": true_labels[:, 0],
        "true_y1": true_labels[:, 1],
        "true_x2": true_labels[:, 2],
        "true_y2": true_labels[:, 3],
        "true_x3": true_labels[:, 4],
        "true_y3": true_labels[:, 5],
        "distance1": d1,
        "distance2": d2,
        "distance3": d3,
        "mean_distance": (d1 + d2 + d3) / 3,
    }


def landmark_detection_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    spacing: tuple[float, ...],
) -> dict[str, torch.Tensor]:
    """Evaluation metrics for landmark detection tasks.

    Args:
        logits: (batch, 3, ...) for 3 landmarks, may be padded at the end.
        labels: target labels, (batch, 3, ...), may be padded at the end.
        spacing: pixel/voxel spacing in mm, not used.

    Returns:
        metrics: each value is of shape (batch,).
    """
    probs = F.sigmoid(logits)

    metrics = landmark_detection_coords_metrics(
        pred_labels=heatmap_soft_argmax(probs),
        true_labels=heatmap_soft_argmax(labels),
        spacing=spacing,
    )

    # dice_loss, cannot use compute_dice as it does not work on soft labels
    dice_loss = DiceLoss(include_background=True, reduction="none")(probs, labels)  # (batch, 3, 1, 1)
    dice_loss = dice_loss.detach().squeeze(-1).squeeze(-1)  # (batch, 3)
    if dice_loss.shape != (logits.shape[0], 3):
        raise ValueError(f"Invalid dice shape: {dice_loss.shape}")

    for i in range(1, 4):
        metrics[f"landmark_{i}_dice_score"] = 1.0 - dice_loss[:, i - 1]
    return metrics


landmark_detection_loss = partial(segmentation_loss, loss_fn=_landmark_detection_loss)

landmark_detection_eval = partial(segmentation_eval, metrics_fn=landmark_detection_metrics)

landmark_detection_eval_dataloader = partial(segmentation_eval_dataloader, metrics_fn=landmark_detection_metrics)


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entrypoint for landmark detection training.

    Args:
        config: config loaded from yaml.
    """
    run_train(
        config=config,
        load_dataset=load_dataset,
        get_model_fn=get_segmentation_model,
        loss_fn=landmark_detection_loss,
        eval_dataloader_fn=landmark_detection_eval_dataloader,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
