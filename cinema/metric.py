"""Metrics that do not exist in MONAI."""

from __future__ import annotations

import numpy as np
import torch
from monai.metrics import compute_iou

from cinema.log import get_logger

logger = get_logger(__name__)


# <= 40%: reduced EF, > 55%: normal EF, in between: borderline EF
REDUCED_EF = 40  # reduced EF threshold
NORMAL_EF = 55  # normal EF threshold


def stability_score(
    logits: torch.Tensor,
    threshold: float = 0.0,
    threshold_offset: float = 1.0,
) -> torch.Tensor:
    """Calculate stability of predictions.

    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py

    Args:
        logits: unscaled prediction, (batch, n_classes, ...).
        threshold: threshold for prediction.
        threshold_offset: offset for threshold.

    Returns:
        Stability of shape (batch, n_classes).

    Raises:
        ValueError: if threshold_offset is negative.
    """
    normalized_logits = logits - torch.mean(logits, dim=1, keepdim=True)
    mask_high_threshold = normalized_logits >= (threshold + threshold_offset)
    mask_low_threshold = normalized_logits >= (threshold - threshold_offset)
    return compute_iou(mask_high_threshold, mask_low_threshold)


def heatmap_argmax(heatmap: torch.Tensor) -> torch.Tensor:
    """Extract coordinates from heatmap.

    Args:
        heatmap: (batch, 3, x, y).

    Returns:
        coords: (batch, 6), not normalized.
    """
    _, _, *image_shape = heatmap.shape
    return torch.stack(
        torch.vmap(lambda x: torch.unravel_index(torch.argmax(x[0]), image_shape))(heatmap)
        + torch.vmap(lambda x: torch.unravel_index(torch.argmax(x[1]), image_shape))(heatmap)
        + torch.vmap(lambda x: torch.unravel_index(torch.argmax(x[2]), image_shape))(heatmap)
    ).T


def heatmap_soft_argmax(heatmap: torch.Tensor, beta: float = 1000.0) -> torch.Tensor:
    """Soft argmax.

    https://ojs.aaai.org/index.php/AAAI/article/view/20161

    Args:
        heatmap: (batch, 3, w, h).
        beta: hyper-parameter.

    Returns:
        coords: (batch, 6), not normalized.
    """
    batch, c, w, h = heatmap.shape
    softmax = torch.softmax(heatmap.reshape(batch, c, -1) * beta, dim=2)  # (batch, 3, w, h)

    x = torch.arange(w)
    y = torch.arange(h)
    pred_coords = torch.cartesian_prod(x, y).to(device=heatmap.device)  # (w * h, 2)
    pred_coords = (softmax[..., None] * pred_coords[None, None, ...]).sum(dim=2)  # (batch, 3, 2)
    return pred_coords.reshape((batch, -1)).to(dtype=torch.long)  # (batch, 6)


def get_volumes(mask: torch.Tensor, spacing: tuple[float, ...]) -> torch.Tensor:
    """Get volumes of each class in the mask.

    Args:
        mask: (batch, n_classes, ...).
        spacing: pixel/voxel spacing in mm.

    Returns:
        volumes: (batch, n_classes).
    """
    volumes = torch.sum(mask, dim=tuple(range(2, mask.ndim)))  # (batch, n_classes)
    volumes = volumes * torch.prod(torch.tensor(spacing)) / 1000.0  # ml = 1000 mm^3
    return volumes


def ejection_fraction(
    edv: torch.Tensor | np.ndarray | float,
    esv: torch.Tensor | np.ndarray | float,
) -> torch.Tensor | np.ndarray | float:
    """Calculate ejection fraction (%).

    Args:
        edv: end-diastolic volume.
        esv: end-systolic volume.

    Returns:
        ejection_fraction
    """
    return (edv - esv) / edv * 100.0


def coefficient_of_variance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate coefficient of variance from two measurements.

    https://www-users.york.ac.uk/~mb55/meas/cv.htm

    Args:
        x: first measurement.
        y: second measurement.

    Returns:
        Coefficient of variance.
    """
    s2 = (x - y) ** 2 / 2
    m = (x + y) / 2
    s2m2 = s2 / m**2
    return np.sqrt(np.mean(s2m2))


def get_ef_region(x: float) -> int:
    """Get the EF region based on the EF value.

    Args:
        x: EF value.

    Returns:
        EF region index.
    """
    if x <= REDUCED_EF:
        return 0
    if x <= NORMAL_EF:
        return 1
    return 2
