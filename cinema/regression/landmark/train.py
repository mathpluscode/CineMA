"""Script to train."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from cinema.classification.train import get_classification_or_regression_model
from cinema.log import get_logger
from cinema.metric import heatmap_argmax
from cinema.regression.train import regression_eval
from cinema.segmentation.landmark.train import (
    landmark_detection_coords_metrics,
    load_dataset,
)
from cinema.train import run_train

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader
logger = get_logger(__name__)


def get_coords_from_batch(batch: dict[str, torch.Tensor], view: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get coordinates from a batch.

    Args:
        batch: dictionary with view, each view image is (batch, n_classes, *image_size).
        view: view name.

    Returns:
        labels: normalised coordinates with values in [0, 1], (batch, 6).
        scales: scales to convert normalised coordinates to pixel coordinates.
    """
    _, _, w, h = batch[f"{view}_image"].shape
    labels = heatmap_argmax(batch[f"{view}_label"])
    scales = torch.tensor([[w, h, w, h, w, h]], dtype=labels.dtype)
    return labels / scales, scales


class WingLoss(nn.Module):
    """Wing loss for landmark detection.

    https://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_Wing_Loss_for_CVPR_2018_paper.pdf
    """

    def __init__(self, w: float = 10, epsilon: float = 2) -> None:
        """Initialise WingLoss.

        Args:
            w: width of the linear part.
            epsilon: epsilon for log.
        """
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.c = w - w * np.log(1 + w / epsilon)

    def forward(self, pred_labels: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """Calculate Wing loss.

        Args:
            pred_labels: predicted coordinates, (batch, n).
            true_labels: ground truth coordinates, (batch, n).
        """
        err = torch.abs(pred_labels - true_labels)
        loss = torch.where(err < self.w, self.w * torch.log(1 + err / self.epsilon), err - self.c)
        return loss.mean()


def get_relative_distances(coords: torch.Tensor) -> torch.Tensor:
    """Get pairwise differences between coordinates.

    The distance is from point 1 to the midpoint of points 2 and 3.

    Args:
        coords: coordinates, (batch, 6).
            x1, y1, x2, y2, x3, y3.

    Returns:
        relative distances, (batch, 6).
            dx1, dy1, dx2, dy2, dx3, dy3.
    """
    return coords @ torch.tensor(
        [
            [1, 0, -0.5, 0, -0.5, 0],
            [0, 1, 0, -0.5, 0, -0.5],
            [-0.5, 0, 1, 0, -0.5, 0],
            [0, -0.5, 0, 1, 0, -0.5],
            [-0.5, 0, -0.5, 0, 1, 0],
            [0, -0.5, 0, -0.5, 0, 1],
        ],
        dtype=coords.dtype,
        device=coords.device,
    )


def landmark_regression_loss(
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
    if len(views) != 1:
        raise ValueError(f"Expected 1 view, got {len(views)}: {views}")
    view = views[0]

    # get predicted and true labels, (batch, 6), x1, y1, x2, y2, x3, y3
    true_labels, scales = get_coords_from_batch(batch, view)
    true_labels, scales = true_labels.to(device), scales.to(device)
    pred_labels = model({view: batch[f"{view}_image"].to(device)})
    true_labels, pred_labels = true_labels * scales, pred_labels * scales

    # https://ieeexplore.ieee.org/abstract/document/9442331
    # not only loss on predicted coordinates
    # but also on the relative distances between the landmarks
    true_rel_dists = get_relative_distances(true_labels)
    pred_rel_dists = get_relative_distances(pred_labels)

    wing_loss = WingLoss()
    landmark_wing_loss = wing_loss(pred_labels, true_labels)
    rel_dist_wing_loss = wing_loss(pred_rel_dists, true_rel_dists)
    loss = landmark_wing_loss + rel_dist_wing_loss

    landmark_mae = F.l1_loss(pred_labels, true_labels)
    rel_dist_mae = F.l1_loss(pred_rel_dists, true_rel_dists)

    metrics = {
        "loss": loss.item(),
        "landmark_wing_loss": landmark_wing_loss.item(),
        "relative_distance_wing_loss": rel_dist_wing_loss.item(),
        "landmark_mae": landmark_mae.item(),
        "relative_distance_mae": rel_dist_mae.item(),
    }
    return loss, metrics


def landmark_regression_eval_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    patch_size_dict: dict[str, tuple[int, ...]],
    spacing_dict: dict[str, tuple[float, ...]],
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
    views = list(patch_size_dict.keys())
    if len(views) != 1:
        raise ValueError(f"Expected 1 view, got {len(views)}: {views}")
    view = views[0]

    pred_labels = []
    true_labels = []
    label_scales = []
    for _, batch in enumerate(dataloader):
        # x1, y1, x2, y2, x3, y3
        labels, scales = get_coords_from_batch(batch, view)
        preds, _ = regression_eval(
            model=model, batch=batch, patch_size_dict=patch_size_dict, amp_dtype=amp_dtype, device=device
        )
        pred_labels.append(preds)
        true_labels.append(labels)
        label_scales.append(scales)

    pred_labels = torch.cat(pred_labels, dim=0)
    true_labels = torch.cat(true_labels, dim=0).to(device=pred_labels.device)
    label_scales = torch.cat(label_scales, dim=0).to(device=pred_labels.device)
    restored_pred_labels = pred_labels * label_scales
    restored_true_labels = true_labels * label_scales
    metrics = landmark_detection_coords_metrics(
        true_labels=true_labels,
        pred_labels=pred_labels,
        spacing=spacing_dict[views[0]],
    )
    metrics = {f"raw_{k}": v for k, v in metrics.items()}
    restored_metrics = landmark_detection_coords_metrics(
        true_labels=restored_true_labels,
        pred_labels=restored_pred_labels,
        spacing=spacing_dict[views[0]],
    )
    metrics.update(restored_metrics)
    metrics = {k: v.mean().item() for k, v in metrics.items()}
    return metrics


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entrypoint for regression training.

    Args:
        config: config loaded from yaml.
    """
    run_train(
        config=config,
        load_dataset=load_dataset,
        get_model_fn=get_classification_or_regression_model,
        loss_fn=landmark_regression_loss,
        eval_dataloader_fn=landmark_regression_eval_dataloader,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
