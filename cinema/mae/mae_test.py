"""Tests for ConvMAE model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cinema.mae.mae import (
    CineMA,
    get_batch_random_patch_mask,
)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_patches", [7, 15])
@pytest.mark.parametrize("mask_ratio", [0.1, 0.5, 0.9])
def test_get_batch_random_patch_mask(
    batch_size: int,
    n_patches: int,
    mask_ratio: float,
) -> None:
    """Test random_patch_masking."""
    mask = get_batch_random_patch_mask(
        batch_size=batch_size,
        n_patches=n_patches,
        mask_ratio=mask_ratio,
        device=torch.device("cpu"),
    )
    n_keep = int(n_patches * (1 - mask_ratio))
    assert mask.shape == (batch_size, n_patches)
    assert int(torch.sum(~mask).numpy()) == n_keep * batch_size


@pytest.mark.parametrize("enc_mask_ratio", [0.1, 0.5, 0.9])
@pytest.mark.parametrize(
    ("image_size_dict", "patch_size_dict", "scale_factor_dict", "conv_chans", "in_chans_dict"),
    [
        ({"img1": (32, 32)}, {"img1": (2, 4)}, {"img1": (2, 2)}, [4, 8], {"img1": 1}),
        ({"img1": (32, 64)}, {"img1": (2, 4)}, {"img1": (2, 2)}, [2, 4, 8], {"img1": 1}),
        (
            {"img1": (32, 32), "img2": (32, 16)},
            {"img1": (2, 4), "img2": (2, 2)},
            {"img1": (2, 2), "img2": (2, 2)},
            [4, 8],
            {"img1": 1, "img2": 3},
        ),
        (
            {"img1": (8, 32, 32), "img2": (32, 16)},
            {"img1": (2, 4, 1), "img2": (2, 4)},
            {"img1": (2, 2, 1), "img2": (2, 2)},
            [4, 8],
            {"img1": 1, "img2": 3},
        ),
        (
            {"img1": (8, 16, 16), "img2": (16, 16), "img3": (16, 8)},
            {"img1": (2, 4, 8), "img2": (2, 4), "img3": (2, 4)},
            {"img1": (2, 2, 1), "img2": (2, 2), "img3": (2, 1)},
            [4, 8],
            {"img1": 1, "img2": 3, "img3": 2},
        ),
    ],
)
@pytest.mark.parametrize("norm_target", [True, False])
@pytest.mark.parametrize("cross_attn", [True, False])
@pytest.mark.parametrize("grad_ckpt", [True, False])
def test_conv_mae_size(
    enc_mask_ratio: float,
    image_size_dict: dict[str, tuple[int, ...]],
    patch_size_dict: dict[str, tuple[int, ...]],
    scale_factor_dict: dict[str, tuple[int, ...]],
    conv_chans: list[int],
    in_chans_dict: dict[str, int],
    norm_target: bool,
    cross_attn: bool,
    grad_ckpt: bool,
) -> None:
    """Test output sizes."""
    batch = 2
    mae = CineMA(
        image_size_dict=image_size_dict,
        enc_patch_size_dict=patch_size_dict,
        in_chans_dict=in_chans_dict,
        enc_scale_factor_dict=scale_factor_dict,
        enc_conv_chans=conv_chans,
        enc_conv_n_blocks=1,
        enc_embed_dim=16,
        enc_depth=1,
        enc_n_heads=2,
        dec_embed_dim=16,
        dec_depth=1,
        dec_n_heads=2,
        mlp_ratio=2,
        norm_target=norm_target,
        cross_attn=cross_attn,
    )
    mae.set_grad_ckpt(grad_ckpt)
    for m in mae.children():
        if hasattr(m, "grad_ckpt"):
            assert m.grad_ckpt == grad_ckpt
    image_dict = {
        k: torch.rand(size=(batch, in_chans_dict[k], *image_size_dict[k]), dtype=torch.float32) for k in image_size_dict
    }
    loss, pred_dict, enc_mask_dict, metrics = mae(image_dict, enc_mask_ratio)

    # check
    ns_masked = []
    ns_patches = []
    for k in image_size_dict:
        n_patches = np.prod(mae.enc_down_dict[k].patch_embed.grid_size)
        ns_patches.append(n_patches)
        dim_pred = np.prod(mae.dec_patch_size_dict[k]) * in_chans_dict[k]

        pred = pred_dict[k]
        enc_mask = enc_mask_dict[k]
        assert enc_mask.sum().item() >= 0
        assert pred.shape[0] == batch
        ns_masked.append(pred.shape[1])
        assert pred.shape[2] == dim_pred
        assert enc_mask.shape == (batch, n_patches)

    # check n_masked
    assert sum(ns_masked) == sum(n - int(n * (1 - enc_mask_ratio)) for n in ns_patches)

    # value can be nan if target is empty
    # this is unlikely to happen with large mask_ratio
    if min(ns_masked) > 0:
        assert not np.isnan(loss.detach().numpy())
        for v in metrics.values():
            assert not np.isnan(v.detach())
            assert v.shape == ()
