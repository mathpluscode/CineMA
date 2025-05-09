"""Test the patch related function in the transform module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cinema.transform import (
    aggregate_patches,
    get_patch_grid,
    patch_grid_sample,
)


@pytest.mark.parametrize(
    ("patch_size", "image_size", "patch_overlap", "expected"),
    [
        (
            # 2d - with overlap
            (3, 5),
            (6, 11),
            (1, 3),
            np.array(
                [
                    [0, 0],
                    [0, 2],
                    [0, 4],
                    [0, 6],
                    [2, 0],
                    [2, 2],
                    [2, 4],
                    [2, 6],
                    [3, 0],
                    [3, 2],
                    [3, 4],
                    [3, 6],
                ]
            ),
        ),
        (
            # 2d - patch is image
            (3, 5),
            (3, 5),
            (0, 0),
            np.array(
                [
                    [0, 0],
                ]
            ),
        ),
        (
            # 2d - patch < image
            (3, 5),
            (3, 7),
            (0, 0),
            np.array(
                [
                    [0, 0],
                    [0, 2],
                ]
            ),
        ),
        (
            # 3d
            (4, 5, 6),
            (8, 10, 6),
            (2, 1, 4),
            np.array(
                [
                    [0, 0, 0],
                    [0, 4, 0],
                    [0, 5, 0],
                    [2, 0, 0],
                    [2, 4, 0],
                    [2, 5, 0],
                    [4, 0, 0],
                    [4, 4, 0],
                    [4, 5, 0],
                ]
            ),
        ),
        (
            # 3d
            (128, 128, 128),
            (192, 128, 128),
            (64, 0, 0),
            np.array(
                [
                    [0, 0, 0],
                    [64, 0, 0],
                ]
            ),
        ),
    ],
)
def test_get_patch_grid(
    patch_size: tuple[int, ...],
    image_size: tuple[int, ...],
    patch_overlap: tuple[int, ...],
    expected: np.ndarray,
) -> None:
    """Test get_patch_grid return values.

    Args:
        patch_size: patch size.
        image_size: image spatial shape.
        patch_overlap: overlap between patches.
        expected: expected output.
    """
    got = get_patch_grid(
        image_size=image_size,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    np.testing.assert_equal(got, expected)


@pytest.mark.parametrize(
    ("patch_size", "image_size", "patch_overlap", "n_patches"),
    [
        (
            # 2d - with overlap
            (3, 5),
            (6, 11),
            (1, 3),
            12,
        ),
        (
            # 2d - patch is image
            (3, 5),
            (3, 5),
            (0, 0),
            1,
        ),
        (
            # 2d - patch < image
            (3, 5),
            (3, 7),
            (0, 0),
            2,
        ),
        (
            # 3d
            (4, 5, 6),
            (8, 10, 6),
            (2, 1, 4),
            9,
        ),
        (
            # 3d
            (128, 128, 128),
            (192, 128, 128),
            (64, 0, 0),
            2,
        ),
    ],
)
@pytest.mark.parametrize("n_chans", [-1, 1, 3])
def test_batch_patch_grid_sample(
    n_chans: int,
    patch_size: tuple[int, ...],
    image_size: tuple[int, ...],
    patch_overlap: tuple[int, ...],
    n_patches: int,
) -> None:
    """Test output shapes."""
    image = torch.rand(n_chans, *image_size) if n_chans > 0 else torch.rand(*image_size)
    patch_start_indices = get_patch_grid(
        image_size=image_size,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
    )
    patches = patch_grid_sample(image, patch_start_indices, patch_size)
    if n_chans > 0:
        assert patches.shape == (n_patches, n_chans, *patch_size)
    else:
        assert patches.shape == (n_patches, *patch_size)

    if n_chans > 0:
        aggregated = aggregate_patches(patches, patch_start_indices, image_size)
        np.testing.assert_allclose(aggregated, image, rtol=1.0e-6, atol=1.0e-6)
