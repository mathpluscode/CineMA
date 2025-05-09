"""Test training and evaluation."""

import numpy as np
import torch

from cinema.metric import heatmap_argmax, heatmap_soft_argmax
from cinema.segmentation.landmark.dataset import create_circle_2d


def test_heatmap_argmax() -> None:
    """Test heatmap_argmax and heatmap_soft_argmax.

    Sample coordinates and generate heatmap, then extract coordinates from heatmap.
    """
    batch_size = 4
    image_size = (7, 8)

    # generate heatmap
    rng = np.random.default_rng()
    coords = rng.random((batch_size, 6))  # x1, y1, x2, y2, x3, y3
    coords *= np.array(list(image_size) * 3)[None, :]
    coords = coords.astype(int)
    heatmap = torch.Tensor(
        np.stack(
            [
                np.stack(
                    [
                        create_circle_2d(c[:2], image_size),
                        create_circle_2d(c[2:4], image_size),
                        create_circle_2d(c[4:], image_size),
                    ]
                )
                for c in coords
            ]
        )
    )

    # extract coords
    coords_got = heatmap_argmax(heatmap)
    assert np.allclose(coords, coords_got)

    coords_got = heatmap_soft_argmax(heatmap)
    assert np.allclose(coords, coords_got)
