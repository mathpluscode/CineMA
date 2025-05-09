"""Test metric functions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cinema.metric import (
    ejection_fraction,
    get_ef_region,
    get_volumes,
    stability_score,
)


@pytest.mark.parametrize(
    ("mask", "spacing", "expected"),
    [
        (np.array([[[0, 1, 0], [0, 0, 1]]]), (1, 1, 1), np.array([[0.001, 0.001]])),
        (np.array([[[0, 1, 0], [0, 0, 1]]]), (1.1, 1.2, 10), np.array([[0.0132, 0.0132]])),
    ],
)
def test_get_volumes(
    mask: np.ndarray,
    spacing: tuple[float, ...],
    expected: np.ndarray,
) -> None:
    """Test values."""
    got = get_volumes(torch.from_numpy(mask), spacing)
    np.testing.assert_allclose(got, expected)


def test_ejection_fraction() -> None:
    """Test values."""
    ed_volumes = np.array([100, 200, 300])
    es_volumes = np.array([50, 150, 250])
    expected = np.array([50, 25, 100.0 / 6])
    got = ejection_fraction(ed_volumes, es_volumes)
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize(
    ("ef", "expected"),
    [
        (30.0, 0),
        (40.0, 0),
        (40.1, 1),
        (50.0, 1),
        (55.0, 1),
        (56.0, 2),
        (60.0, 2),
        (70.0, 2),
    ],
)
def test_get_ef_region(ef: float, expected: int) -> None:
    """Test get_ef_region."""
    assert get_ef_region(ef) == expected


@pytest.mark.parametrize(
    ("logits", "threshold", "threshold_offset", "expected"),
    [
        (
            np.array([[[[0.8, 2.3], [-1.0, 1.1]], [[-1.8, -2.0], [1.5, -1.3]], [[1.0, -0.3], [-0.5, 0.2]]]]),
            0.0,
            1.0,
            np.array([0.5, 1.0, 0.25]),
        ),
        (
            np.array([[[[0.8, 1.3], [-1.0, 1.1]], [[-0.8, -3.0], [1.5, -1.3]], [[2.0, -1.3], [-0.5, 0.2]]]]),
            0.0,
            1.0,
            np.array([0.5, 1.0, 0.25]),
        ),
        (
            np.array([[[[0.8, 2.3], [-1.0, 1.1]], [[-1.8, -2.0], [1.5, -1.3]], [[1.0, -0.3], [-0.5, 0.2]]]]),
            1.0,
            1.0,
            np.array([1.0 / 3.0, 0.0, 0.0]),
        ),
        (
            np.array([[[[0.8, 2.3], [-1.0, 1.1]], [[-1.8, -2.0], [1.5, -1.3]], [[1.0, -0.3], [-0.5, 0.2]]]]),
            2.0,
            0.0,
            np.array([1.0, 0.0, 0.0]),
        ),
    ],
)
def test_stability(
    logits: np.ndarray,
    threshold: float,
    threshold_offset: float,
    expected: np.ndarray,
) -> None:
    """Test output values."""
    got = stability_score(
        logits=torch.from_numpy(logits),
        threshold=threshold,
        threshold_offset=threshold_offset,
    )
    np.testing.assert_allclose(np.nan_to_num(got.numpy()), expected[None, :], rtol=1.0e-5, atol=1.0e-5)
