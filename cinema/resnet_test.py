"""Test resnet class."""

import pytest
import torch

from cinema.resnet import get_resnet2d, get_resnet3d


@pytest.mark.parametrize("depth", [10, 18, 34, 50, 101, 152, 200])
@pytest.mark.parametrize("in_chans", [1, 2])
@pytest.mark.parametrize("out_chans", [1, 2])
def test_get_resnet2d(depth: int, in_chans: int, out_chans: int) -> None:
    """Test get_resnet."""
    batch_size = 2
    layer_inplanes = [4, 8, 16, 32]
    resnet = get_resnet2d(depth=depth, in_chans=in_chans, out_chans=out_chans, layer_inplanes=layer_inplanes)
    x = torch.randn(batch_size, in_chans, 8, 32)
    got = resnet({"x": x})
    assert got.shape == (batch_size, out_chans)


@pytest.mark.parametrize("depth", [10, 18, 34, 50, 101, 152, 200])
@pytest.mark.parametrize("in_chans", [1, 2])
@pytest.mark.parametrize("out_chans", [1, 2])
def test_get_resnet3d(depth: int, in_chans: int, out_chans: int) -> None:
    """Test get_resnet."""
    batch_size = 2
    layer_inplanes = [4, 8, 16, 32]
    resnet = get_resnet3d(depth=depth, in_chans=in_chans, out_chans=out_chans, layer_inplanes=layer_inplanes)
    x = torch.randn(batch_size, in_chans, 8, 32, 32)
    got = resnet({"x": x})
    assert got.shape == (batch_size, out_chans)
