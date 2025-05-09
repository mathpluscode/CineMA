"""Tests for the conv module."""

import pytest
import torch

from cinema.conv import (
    ConvLayerNorm,
    ConvMlp,
    ConvNormActBlock,
    ConvResBlock,
    MaskedConvBlock,
    get_conv_norm,
)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_features", [1, 8])
@pytest.mark.parametrize("image_size", [(16, 8), (7, 16), (16, 8, 7)])
@pytest.mark.parametrize("grad_ckpt", [True, False])
def test_conv_mlp(batch_size: int, in_features: int, image_size: tuple[int, ...], grad_ckpt: bool) -> None:
    """Test output shapes."""
    block = ConvMlp(n_dims=len(image_size), in_features=in_features)
    block.set_grad_ckpt(grad_ckpt)
    for m in block.children():
        if hasattr(m, "grad_ckpt"):
            assert m.grad_ckpt == grad_ckpt
    x = torch.rand(batch_size, in_features, *image_size)
    got = block(x)
    assert got.shape == x.shape


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_chans", [1, 8])
@pytest.mark.parametrize("image_size", [(16, 8), (7, 16), (16, 8, 7)])
def test_conv_layernorm(batch_size: int, in_chans: int, image_size: tuple[int, ...]) -> None:
    """Test output shapes."""
    block = ConvLayerNorm(in_chans)
    x = torch.rand(batch_size, in_chans, *image_size)
    got = block(x)
    assert got.shape == x.shape


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_chans", [1, 8])
@pytest.mark.parametrize("image_size", [(16, 8), (7, 16), (16, 8, 7)])
@pytest.mark.parametrize("norm", ["instance", "layer", "group"])
def test_get_conv_norm(batch_size: int, in_chans: int, image_size: tuple[int, ...], norm: str) -> None:
    """Test output shapes."""
    block = get_conv_norm(n_dims=len(image_size), in_chans=in_chans, norm=norm)
    x = torch.rand(batch_size, in_chans, *image_size)
    got = block(x)
    assert got.shape == x.shape


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_chans", [1, 8])
@pytest.mark.parametrize("out_chans", [1, 8])
@pytest.mark.parametrize("image_size", [(16, 8), (7, 16), (16, 8, 7)])
@pytest.mark.parametrize("norm", ["instance", "layer", "group"])
@pytest.mark.parametrize("grad_ckpt", [True, False])
def test_conv_norm_act(
    batch_size: int, in_chans: int, out_chans: int, image_size: tuple[int, ...], norm: str, grad_ckpt: bool
) -> None:
    """Test output shapes."""
    block = ConvNormActBlock(n_dims=len(image_size), in_chans=in_chans, out_chans=out_chans, norm=norm)
    block.set_grad_ckpt(grad_ckpt)
    for m in block.children():
        if hasattr(m, "grad_ckpt"):
            assert m.grad_ckpt == grad_ckpt
    x = torch.rand(batch_size, in_chans, *image_size)
    got = block(x)
    assert got.shape == (batch_size, out_chans, *image_size)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_chans", [1, 8])
@pytest.mark.parametrize("out_chans", [1, 8])
@pytest.mark.parametrize("image_size", [(16, 8), (7, 16), (16, 8, 7)])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("norm", ["instance", "layer", "group"])
@pytest.mark.parametrize("grad_ckpt", [True, False])
def test_conv_res(
    batch_size: int,
    in_chans: int,
    out_chans: int,
    image_size: tuple[int, ...],
    dropout: float,
    norm: str,
    grad_ckpt: bool,
) -> None:
    """Test output shapes."""
    block = ConvResBlock(n_dims=len(image_size), in_chans=in_chans, out_chans=out_chans, dropout=dropout, norm=norm)
    block.set_grad_ckpt(grad_ckpt)
    for m in block.children():
        if hasattr(m, "grad_ckpt"):
            assert m.grad_ckpt == grad_ckpt
    x = torch.rand(batch_size, in_chans, *image_size)
    got = block(x)
    assert got.shape == (batch_size, out_chans, *image_size)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("in_chans", [1, 8])
@pytest.mark.parametrize("image_size", [(16, 8), (7, 16), (16, 8, 7)])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("drop_path", [0.0, 0.1])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("norm", ["instance", "layer", "group"])
@pytest.mark.parametrize("grad_ckpt", [True, False])
def test_masked_conv_block(
    batch_size: int,
    in_chans: int,
    image_size: tuple[int, ...],
    dropout: float,
    drop_path: float,
    use_mask: bool,
    norm: str,
    grad_ckpt: bool,
) -> None:
    """Test output shapes."""
    block = MaskedConvBlock(n_dims=len(image_size), in_chans=in_chans, dropout=dropout, drop_path=drop_path, norm=norm)
    block.set_grad_ckpt(grad_ckpt)
    for m in block.children():
        if hasattr(m, "grad_ckpt"):
            assert m.grad_ckpt == grad_ckpt
    x = torch.rand(batch_size, in_chans, *image_size)
    mask = torch.rand(batch_size, *image_size) > 0.5 if use_mask else None
    got = block(x, mask)
    assert got.shape == x.shape
