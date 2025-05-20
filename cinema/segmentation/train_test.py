"""Tests for segmentation transform."""

import numpy as np
import pytest
import torch

from cinema.segmentation.train import segmentation_forward, segmentation_metrics


@pytest.mark.parametrize(
    ("image_size_dict", "patch_size_dict"),
    [
        ({"lax": (4, 6)}, {"lax": (2, 3)}),
        ({"sax": (4, 5, 6)}, {"sax": (2, 3, 4)}),
        ({"sax": (4, 5, 6), "lax": (4, 3)}, {"lax": (4, 3), "sax": (3, 4, 5)}),
        ({"sax": (4, 5, 6), "lax": (4, 3)}, {"lax": (2, 3), "sax": (4, 5, 6)}),
    ],
)
@pytest.mark.parametrize("in_chans", [1, 2])
@pytest.mark.parametrize("out_chans", [1, 2])
def test_segmentation_batch_forward(  # noqa: C901
    image_size_dict: dict[str, tuple[int, ...]],
    patch_size_dict: dict[str, tuple[int, ...]],
    in_chans: int,
    out_chans: int,
) -> None:
    """Test output shapes."""
    if len(image_size_dict) == 1:
        view = next(iter(image_size_dict.keys()))
        image_size = image_size_dict[view]
        ndim = len(image_size)
        if ndim == 2:

            class Conv2dModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = torch.nn.Conv2d(in_chans, out_chans, kernel_size=1)

                def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                    return {view: self.conv(x[view])}

            model = Conv2dModel()
        elif ndim == 3:

            class Conv3dModel(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = torch.nn.Conv3d(in_chans, out_chans, kernel_size=1)

                def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                    return {view: self.conv(x[view])}

            model = Conv3dModel()
        else:
            raise ValueError(f"Unsupported ndim: {ndim}")
    else:

        class ConvModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv2d = torch.nn.Conv2d(in_chans, out_chans, kernel_size=1)
                self.conv3d = torch.nn.Conv3d(in_chans, out_chans, kernel_size=1)

            def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                out = {}
                for view, image_size in image_size_dict.items():
                    conv = self.conv2d if len(image_size) == 2 else self.conv3d
                    out[view] = conv(x[view])
                return out

        model = ConvModel()
    image_dict = {view: torch.randn(1, in_chans, *image_size) for view, image_size in image_size_dict.items()}
    got = segmentation_forward(
        model=model,
        image_dict=image_dict,
        patch_size_dict=patch_size_dict,
        amp_dtype=torch.float32,
    )
    for view, image_size in image_size_dict.items():
        assert got[view].shape == (1, out_chans, *image_size)


@pytest.mark.parametrize("n_classes", [1, 2, 3])
@pytest.mark.parametrize(
    ("image_size", "spacing"),
    [
        ((16, 16), (1.0, 1.2)),
        ((8, 16, 16), (1.0, 1.2, 1.5)),
    ],
)
def test_segmentation_eval_metrics(
    n_classes: int,
    image_size: tuple[int, ...],
    spacing: tuple[float, ...],
) -> None:
    """Test output sizes."""
    batch = 2
    logits = torch.rand(size=(batch, n_classes + 1, *image_size), dtype=torch.float32)
    labels = torch.randint(0, n_classes + 1, size=(batch, *image_size))
    labels[..., -2] = 1
    labels[..., -1] = n_classes

    logits_clone = logits.clone()
    labels_clone = labels.clone()

    metrics = segmentation_metrics(logits, labels, spacing)
    for v in metrics.values():
        assert not np.any(np.isnan(v.detach().cpu().numpy()))
        assert v.shape == (batch,)

    # ensure inputs are not modified
    assert torch.allclose(logits, logits_clone)
    assert torch.allclose(labels, labels_clone)
