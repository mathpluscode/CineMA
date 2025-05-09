"""Test training and evaluation."""

import torch

from cinema.regression.landmark.train import get_relative_distances


def test_get_relative_distances() -> None:
    """Test get_relative_distances."""
    batch = 7
    coords = torch.rand(size=(batch, 6), dtype=torch.float32)
    got = get_relative_distances(coords)

    x1, y1, x2, y2, x3, y3 = coords.chunk(6, dim=1)
    dx1 = x1 - (x2 + x3) / 2
    dy1 = y1 - (y2 + y3) / 2
    dx2 = x2 - (x1 + x3) / 2
    dy2 = y2 - (y1 + y3) / 2
    dx3 = x3 - (x1 + x2) / 2
    dy3 = y3 - (y1 + y2) / 2
    expected = torch.concat([dx1, dy1, dx2, dy2, dx3, dy3], dim=1)
    assert torch.allclose(got, expected)
