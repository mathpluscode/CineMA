"""Test unet."""

from __future__ import annotations

import pytest
import torch

from cinema.segmentation.unet import DownsampleEncoder, UNet, UpsampleDecoder


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_chans", [1, 3])
@pytest.mark.parametrize("chans", [[2, 4], [2, 4, 8]])
@pytest.mark.parametrize(
    ("image_size", "patch_size", "scale_factor"),
    [((32, 32), (2, 2), (2, 2)), ((32, 34), (2, 2), (2, 2)), ((30, 34, 32), (2, 2, 2), (2, 2, 2))],
)
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("norm", ["instance", "layer", "group"])
def test_enc_decoder(
    batch_size: int,
    in_chans: int,
    chans: tuple[int, ...],
    image_size: tuple[int, ...],
    patch_size: int | tuple[int, ...],
    scale_factor: int | tuple[int, ...],
    dropout: float,
    norm: str,
) -> None:
    """Test it runs."""
    encoder = DownsampleEncoder(
        n_dims=len(image_size),
        in_chans=in_chans,
        chans=chans,
        patch_size=patch_size,
        scale_factor=scale_factor,
        dropout=dropout,
        norm=norm,
    )
    x = torch.rand(batch_size, in_chans, *image_size)
    got_embeddings = encoder(x)
    assert len(got_embeddings) == len(chans) * (encoder.n_blocks + 1)

    decoder = UpsampleDecoder(
        n_dims=len(image_size),
        in_chans=chans[-1],
        chans=chans,
        patch_size=patch_size,
        scale_factor=scale_factor,
        dropout=dropout,
        norm=norm,
    )
    got = decoder(got_embeddings)
    assert got.shape == (batch_size, chans[0], *image_size)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_chans", [1, 3])
@pytest.mark.parametrize("out_chans", [1, 3])
@pytest.mark.parametrize("chans", [[2, 4], [2, 4, 8]])
@pytest.mark.parametrize(
    ("image_size", "patch_size", "scale_factor"),
    [((32, 32), (2, 2), (2, 2)), ((32, 34), (2, 2), (2, 2)), ((30, 34, 32), (2, 2, 2), (2, 2, 2))],
)
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("norm", ["instance", "layer", "group"])
@pytest.mark.parametrize("grad_ckpt", [False, True])
def test_unet(
    batch_size: int,
    in_chans: int,
    out_chans: int,
    chans: tuple[int, ...],
    image_size: tuple[int, ...],
    patch_size: int | tuple[int, ...],
    scale_factor: int | tuple[int, ...],
    dropout: float,
    norm: str,
    grad_ckpt: bool,
) -> None:
    """Test it runs."""
    unet = UNet(
        n_dims=len(image_size),
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        patch_size=patch_size,
        scale_factor=scale_factor,
        dropout=dropout,
        norm=norm,
    )
    unet.set_grad_ckpt(grad_ckpt)
    x = torch.rand(batch_size, in_chans, *image_size)
    got = unet({"sax": x})["sax"]
    assert got.shape == (batch_size, out_chans, *image_size)
