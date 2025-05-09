"""Transforms for randomly sampling patches from images."""

from __future__ import annotations

import numpy as np
import torch

from cinema.log import get_logger

logger = get_logger(__name__)


def get_patch_grid(
    image_size: tuple[int, ...],
    patch_size: tuple[int, ...],
    patch_overlap: tuple[int, ...],
) -> np.ndarray:
    """Get start_indices per patch following a grid.

    Use numpy due to for and if loops.

    https://github.com/fepegar/torchio/blob/main/src/torchio/data/sampler/grid.py

    Args:
        image_size: image size, (d1, ..., dn).
        patch_size: patch size, (p1, ..., pn),
            patch_size should <= image_size for all dimensions.
        patch_overlap: overlap between patches, (o1, ..., on),
            patch_overlap should <= patch_size for all dimensions.

    Returns:
        Indices grid of shape (n_patches, n).
    """
    indices = []
    for img_size_dim, patch_size_dim, ovlp_size_dim in zip(image_size, patch_size, patch_overlap, strict=False):
        # Example with image_size 10, patch_size 5, overlap 2:
        # [0 1 2 3 4 5 6 7 8 9]
        # [0 0 0 0 0]
        #       [1 1 1 1 1]
        #           [2 2 2 2 2]
        # indices_dim = [0, 3, 5]
        if patch_size_dim > img_size_dim:
            raise ValueError(f"Patch size {patch_size_dim} should be <= image size {img_size_dim}.")
        end = img_size_dim - patch_size_dim + 1
        step = patch_size_dim - ovlp_size_dim
        indices_dim = np.arange(0, end, step)
        if indices_dim[-1] != end - 1:
            indices_dim = np.append(indices_dim, img_size_dim - patch_size_dim)
        indices.append(indices_dim)
    return np.stack(np.meshgrid(*indices, indexing="ij"), axis=-1).reshape(-1, len(image_size))


def patch_grid_sample(
    x: torch.Tensor,
    start_indices: np.ndarray,
    patch_size: tuple[int, ...],
) -> torch.Tensor:
    """Extract patch following a grid.

    Args:
        x: has shape (d1, ..., dn) or (ch, d1, ..., dn).
        start_indices: indices grid of shape (n_patches, n).
        patch_size: patch size, shape = (p1, ..., pn),
            patch_size should <= image_size for all dimensions.

    Returns:
        Patched, has shapes (n_patches, p1, ..., pn)
            or (n_patches, ch, p1, ..., pn).
    """
    if x.ndim == len(patch_size):
        # no channel dimension
        patches = []
        for start in start_indices:
            slices = tuple(slice(start[i], start[i] + patch_size[i]) for i in range(len(patch_size)))
            patches.append(x[slices])
        return torch.stack(patches)

    # with channel dimension
    patches = []
    for start in start_indices:
        slices = (slice(None), *(slice(start[i], start[i] + patch_size[i]) for i in range(len(patch_size))))
        patches.append(x[slices])
    return torch.stack(patches)


def aggregate_patches(
    patches: torch.Tensor,
    start_indices: np.ndarray,
    image_size: tuple[int, ...],
) -> torch.Tensor:
    """Aggregate patches by average on overlapping area following a grid.

    Args:
        patches: array of shape (n_patches, channels, p1, ..., pn).
        start_indices: indices grid of shape (n_patches, n).
        image_size: image size, (d1, ..., dn).

    Returns:
        Aggregated array of shape (channels, d1, ..., dn).
    """
    n_patches, ch, *patch_size = patches.shape
    n_dims = len(image_size)
    if n_patches != start_indices.shape[0]:
        raise ValueError(
            f"n_patches should be the same as start_indices, got {n_patches} and {start_indices.shape[0]}."
        )
    if n_dims != len(patch_size):
        raise ValueError(
            f"image_size and patch_size should have the same length, "
            f"got image_size={image_size} and patches.shape={patches.shape}."
        )

    # x is the summed image
    x = torch.zeros((ch, *image_size), dtype=patches.dtype, device=patches.device)
    count = torch.zeros(image_size, dtype=torch.float32, device=patches.device)

    for i in range(n_patches):
        start = start_indices[i]
        patch = patches[i]
        slices = tuple(slice(start[d], start[d] + patch_size[d]) for d in range(n_dims))
        x[(slice(None), *slices)] += patch
        count[slices] += 1

    return x / count[None, ...]


def crop_start(image: np.ndarray | torch.Tensor, target_shape: tuple[int, ...]) -> np.ndarray | torch.Tensor:
    """Crop the image at the beginning target shape.

    Needed as the SpatialPad is padding at the end.

    Args:
        image: input image.
        target_shape: target shape.

    Returns:
        cropped image.
    """
    if len(image.shape) != len(target_shape):
        raise ValueError(
            f"image.shape and target_shape should have the same length, got {image.shape} and {target_shape}."
        )
    return image[tuple(slice(0, s) for s in target_shape)]
