"""Test for segmentation models."""

from __future__ import annotations

import pytest
import torch

from cinema.segmentation.convunetr import ConvUNetR, check_conv_unetr_enc_dec_compatiblity


@pytest.mark.parametrize(
    (
        "enc_patch_size",
        "enc_scale_factor",
        "enc_n_conv_layers",
        "dec_depth",
        "dec_patch_size",
        "dec_scale_factor",
        "n_layers_wo_skip",
        "n_downsample_layers",
        "fail",
    ),
    [
        ((4, 4), (2, 2), 2, 4, (2, 2), (2, 2), 1, 0, False),
        ((4, 1), (2, 1), 2, 4, (2, 1), (2, 1), 1, 0, False),
        ((4, 4), (2, 2), 2, 5, (2, 2), (2, 2), 1, 1, False),
        ((2, 2), (2, 2), 2, 4, (2, 2), (2, 2), 0, 1, False),
        ((4, 4), (2, 2), 4, 4, (2, 2), (2, 2), 1, 0, True),  # too many conv layers
    ],
)
def test_check_conv_unetr_enc_dec_compatiblity(
    enc_patch_size: tuple[int, ...],
    enc_scale_factor: tuple[int, ...],
    enc_n_conv_layers: int,
    dec_depth: int,
    dec_patch_size: tuple[int, ...],
    dec_scale_factor: tuple[int, ...],
    n_layers_wo_skip: int,
    n_downsample_layers: int,
    fail: bool,
) -> None:
    """Test output sizes."""
    if fail:
        with pytest.raises(ValueError):  # noqa: PT011
            check_conv_unetr_enc_dec_compatiblity(
                enc_patch_size=enc_patch_size,
                enc_scale_factor=enc_scale_factor,
                enc_n_conv_layers=enc_n_conv_layers,
                dec_depth=dec_depth,
                dec_patch_size=dec_patch_size,
                dec_scale_factor=dec_scale_factor,
            )
    else:
        got_n_layers_wo_skip, got_n_downsample_layers = check_conv_unetr_enc_dec_compatiblity(
            enc_patch_size=enc_patch_size,
            enc_scale_factor=enc_scale_factor,
            enc_n_conv_layers=enc_n_conv_layers,
            dec_depth=dec_depth,
            dec_patch_size=dec_patch_size,
            dec_scale_factor=dec_scale_factor,
        )
        assert got_n_layers_wo_skip == n_layers_wo_skip
        assert got_n_downsample_layers == n_downsample_layers


class TestConvUNetR:
    """Test UNetR single and multi view."""

    @pytest.mark.parametrize("in_chans", [1, 3])
    @pytest.mark.parametrize("n_classes", [1, 2, 3])
    @pytest.mark.parametrize(
        (
            "image_size",
            "enc_patch_size",
            "enc_scale_factor",
            "enc_conv_chans",
            "dec_chans",
            "dec_patch_size",
            "dec_scale_factor",
        ),
        [
            ((16, 24), (4, 8), (2, 2), [], (2, 4, 8), (1, 2), (2, 2)),
            ((16, 24), (2, 4), (2, 2), [4], (2, 4, 8), (1, 2), (2, 2)),
            ((16, 24), (1, 2), (2, 2), [4, 8], (2, 4, 8), (1, 2), (2, 2)),
            ((16, 24, 16), (4, 8, 4), (2, 2, 2), [], (2, 4, 8), (1, 2, 1), (2, 2, 2)),
            ((16, 16, 16), (4, 4, 4), (2, 2, 2), [4], (2, 4, 8, 16), (1, 1, 1), (2, 2, 2)),
            ((16, 16, 16), (2, 2, 2), (2, 2, 2), [4, 8], (2, 4, 8, 16), (1, 1, 1), (2, 2, 2)),
            ((16, 16, 1), (2, 2, 1), (2, 2, 1), [4, 8], (2, 4, 8, 16), (1, 1, 1), (2, 2, 1)),
            ((32, 32, 4), (4, 4, 1), (2, 2, 1), [4, 8], (2, 4, 8, 16), (2, 2, 1), (2, 2, 1)),
            ((32, 32, 4), (8, 8, 1), (2, 2, 1), [4, 8], (2, 4, 8, 16, 32), (2, 2, 1), (2, 2, 1)),
            ((32, 32, 4), (4, 4, 1), (2, 2, 1), [4, 8, 16], (2, 4, 8, 16, 32), (2, 2, 1), (2, 2, 1)),
            ((32, 32, 4), (2, 2, 1), (2, 2, 1), [2, 2, 4, 4], (2, 4, 8, 16, 32), (2, 2, 1), (2, 2, 1)),
        ],
    )
    @pytest.mark.parametrize("grad_ckpt", [True, False])
    def test_single_view(
        self,
        n_classes: int,
        image_size: tuple[int, ...],
        in_chans: int,
        enc_patch_size: tuple[int, ...],
        enc_scale_factor: tuple[int, ...],
        enc_conv_chans: list[int],
        dec_chans: tuple[int, ...],
        dec_patch_size: tuple[int, ...],
        dec_scale_factor: tuple[int, ...],
        grad_ckpt: bool,
    ) -> None:
        """Test output sizes."""
        batch = 2
        out_chans = n_classes + 1
        unetr = ConvUNetR(
            image_size_dict={"view": image_size},
            in_chans_dict={"view": in_chans},
            out_chans=out_chans,
            enc_patch_size_dict={"view": enc_patch_size},
            enc_scale_factor_dict={"view": enc_scale_factor},
            enc_conv_chans=enc_conv_chans,
            enc_conv_n_blocks=1,
            enc_embed_dim=16,
            enc_depth=len(dec_chans),
            enc_n_heads=2,
            dec_chans=dec_chans,
            dec_patch_size_dict={"view": dec_patch_size},
            dec_scale_factor_dict={"view": dec_scale_factor},
            mlp_ratio=2,
        )
        unetr.set_grad_ckpt(grad_ckpt)
        for m in unetr.children():
            if hasattr(m, "grad_ckpt"):
                assert m.grad_ckpt == grad_ckpt
        x = torch.rand(size=(batch, in_chans, *image_size), dtype=torch.float32)
        logits = unetr({"view": x})["view"]
        assert logits.shape == (batch, out_chans, *image_size)

    @pytest.mark.parametrize("n_classes", [1, 2, 3])
    @pytest.mark.parametrize(
        (
            "image_size_dict",
            "in_chans_dict",
            "enc_patch_size_dict",
            "enc_scale_factor_dict",
            "enc_conv_chans",
            "dec_chans",
            "dec_patch_size_dict",
            "dec_scale_factor_dict",
        ),
        [
            (
                {"lax": (16, 16)},
                {"lax": 1},
                {"lax": (4, 8)},
                {"lax": (2, 2)},
                [],
                (2, 4, 8),
                {"lax": (1, 2)},
                {"lax": (2, 2)},
            ),
            (
                {"lax": (16, 16), "sax": (16, 24, 16)},
                {"lax": 1, "sax": 1},
                {"lax": (4, 8), "sax": (4, 8, 4)},
                {"lax": (2, 2), "sax": (2, 2, 2)},
                [],
                (2, 4, 8),
                {"lax": (1, 2), "sax": (1, 2, 1)},
                {"lax": (2, 2), "sax": (2, 2, 2)},
            ),
        ],
    )
    @pytest.mark.parametrize("grad_ckpt", [True, False])
    def test_multi_view(
        self,
        n_classes: int,
        image_size_dict: dict[str, tuple[int, ...]],
        in_chans_dict: dict[str, int],
        enc_patch_size_dict: dict[str, tuple[int, ...]],
        enc_scale_factor_dict: dict[str, tuple[int, ...]],
        enc_conv_chans: list[int],
        dec_chans: tuple[int, ...],
        dec_patch_size_dict: dict[str, tuple[int, ...]],
        dec_scale_factor_dict: dict[str, tuple[int, ...]],
        grad_ckpt: bool,
    ) -> None:
        """Test output sizes."""
        batch = 2
        out_chans = n_classes + 1
        unetr = ConvUNetR(
            image_size_dict=image_size_dict,
            in_chans_dict=in_chans_dict,
            out_chans=out_chans,
            enc_patch_size_dict=enc_patch_size_dict,
            enc_scale_factor_dict=enc_scale_factor_dict,
            enc_conv_chans=enc_conv_chans,
            enc_conv_n_blocks=1,
            enc_embed_dim=16,
            enc_depth=len(dec_chans),
            enc_n_heads=2,
            dec_chans=dec_chans,
            dec_patch_size_dict=dec_patch_size_dict,
            dec_scale_factor_dict=dec_scale_factor_dict,
            mlp_ratio=2,
        )
        unetr.set_grad_ckpt(grad_ckpt)
        for m in unetr.children():
            if hasattr(m, "grad_ckpt"):
                assert m.grad_ckpt == grad_ckpt

        x = {
            view: torch.rand(size=(batch, in_chans_dict[view], *image_size), dtype=torch.float32)
            for view, image_size in image_size_dict.items()
        }
        out = unetr(x)
        for view, logits in out.items():
            assert logits.shape == (batch, out_chans, *image_size_dict[view])
