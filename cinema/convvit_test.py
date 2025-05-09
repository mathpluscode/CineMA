"""Test for segmentation models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from cinema.convvit import ConvViT, get_layer_id_for_vit, param_groups_lr_decay, upsample_mask
from cinema.segmentation.convunetr import ConvUNetR


class TestUpsampleMask:
    """Tests for upsample_mask."""

    @pytest.mark.parametrize(
        ("mask", "scale_factor", "expected"),
        [
            (np.array([[0, 1], [1, 0]]), (2, 2), np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]])),
            (np.array([[0, 1], [1, 0]]), (2, 1), np.array([[0, 1], [0, 1], [1, 0], [1, 0]])),
            (
                np.array([[[0, 1, 0], [1, 0, 0]], [[0, 1, 1], [1, 0, 0]]]),
                (1, 1, 2),
                np.array([[[0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], [[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]]]),
            ),
            (
                np.array([[[0, 1, 0], [1, 0, 0]], [[0, 1, 1], [1, 0, 0]]]),
                (3, 1, 2),
                np.array(
                    [
                        [[0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]],
                        [[0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]],
                        [[0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]],
                        [[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]],
                        [[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]],
                        [[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]],
                    ]
                ),
            ),
        ],
    )
    def test_values(self, mask: np.ndarray, scale_factor: tuple[int, ...], expected: np.ndarray) -> None:
        """Test output values."""
        got = upsample_mask(torch.from_numpy(mask)[None, ...], scale_factor)[0]
        assert torch.equal(got, torch.from_numpy(expected))

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize(
        ("mask_shape", "scale_factor"),
        [
            ((8, 16), (2, 2)),
            ((8, 16, 11), (2, 3, 4)),
            ((8, 16, 11, 5), (2, 3, 4, 1)),
        ],
    )
    def test_shape(self, batch_size: int, mask_shape: tuple[int, ...], scale_factor: tuple[int, ...]) -> None:
        """Test output shapes."""
        mask = torch.rand((batch_size, *mask_shape)) < 0.5
        got = upsample_mask(mask, scale_factor)
        expected_shape = tuple(s * f for s, f in zip(mask_shape, scale_factor, strict=False))
        assert got.shape == (batch_size, *expected_shape)


class TestConvViT:
    """Test ConvViT for single and multi-view."""

    @pytest.mark.parametrize("n_frames", [1, 3])
    @pytest.mark.parametrize("in_chans", [1, 3])
    @pytest.mark.parametrize("out_chans", [1, 3])
    @pytest.mark.parametrize(
        (
            "image_size",
            "input_size",
            "enc_patch_size",
            "enc_scale_factor",
            "enc_conv_chans",
        ),
        [
            ((16, 24), (16, 24), (4, 8), (2, 2), []),
            ((16, 24), (16, 24), (2, 4), (2, 2), [4]),
            ((16, 24), (16, 24), (1, 2), (2, 2), [4, 8]),
            ((24, 32), (16, 24), (1, 2), (2, 2), [4, 8]),  # input is smaller than image size
            ((16, 24), (24, 32), (1, 2), (2, 2), [4, 8]),  # input is larger than image size
            ((16, 16, 16), (16, 16, 16), (4, 4, 4), (2, 2, 2), [4]),
            ((16, 16, 16), (16, 16, 16), (2, 2, 2), (2, 2, 2), [4, 8]),
            ((16, 16, 1), (16, 16, 1), (2, 2, 1), (2, 2, 1), [4, 8]),
            ((32, 24, 12), (16, 16, 16), (2, 2, 2), (2, 2, 2), [4]),  # input is smaller/larger
            ((32, 32, 4), (32, 32, 4), (4, 4, 1), (2, 2, 1), [4, 8]),
            ((32, 32, 4), (32, 32, 4), (8, 8, 1), (2, 2, 1), [4, 8]),
            ((32, 32, 4), (32, 32, 4), (4, 4, 1), (2, 2, 1), [4, 8, 16]),
            ((32, 32, 4), (32, 32, 4), (2, 2, 1), (2, 2, 1), [2, 2, 4, 4]),
        ],
    )
    @pytest.mark.parametrize("reduce", ["patch", "all", "cls"])
    @pytest.mark.parametrize("input_mask", [True, False])
    @pytest.mark.parametrize("grad_ckpt", [True, False])
    def test_single_view(
        self,
        n_frames: int,
        image_size: tuple[int, ...],
        input_size: tuple[int, ...],
        in_chans: int,
        out_chans: int,
        enc_patch_size: tuple[int, ...],
        enc_scale_factor: tuple[int, ...],
        enc_conv_chans: list[int],
        reduce: str,
        input_mask: bool,
        grad_ckpt: bool,
    ) -> None:
        """Test output sizes."""
        batch = 2
        view = "sax"
        vit = ConvViT(
            image_size_dict={view: image_size},
            n_frames=n_frames,
            in_chans_dict={view: in_chans},
            out_chans=out_chans,
            enc_patch_size_dict={view: enc_patch_size},
            enc_scale_factor_dict={view: enc_scale_factor},
            enc_conv_chans=enc_conv_chans,
            enc_conv_n_blocks=1,
            enc_embed_dim=16,
            enc_depth=2,
            enc_n_heads=2,
            mlp_ratio=2,
        )
        vit.set_grad_ckpt(grad_ckpt)
        for m in vit.children():
            if hasattr(m, "grad_ckpt"):
                assert m.grad_ckpt == grad_ckpt
        x = torch.rand(size=(batch, n_frames * in_chans, *input_size), dtype=torch.float32)
        grid_size = tuple(s // p for s, p in zip(input_size, vit.enc_down_dict[view].eff_patch_size, strict=False))
        n_patches = math.prod(grid_size)
        mask_dict = None
        if input_mask:
            mask = torch.rand(size=(batch, n_patches), dtype=torch.float32) > 0.5
            mask_dict = {view: mask}
        out = vit({view: x}, mask_dict=mask_dict, reduce=reduce)
        assert out.shape == (batch, out_chans)

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
    @pytest.mark.parametrize("n_frames", [1, 3])
    @pytest.mark.parametrize("out_chans", [1, 3])
    @pytest.mark.parametrize("reduce", ["patch", "all", "cls"])
    @pytest.mark.parametrize("mask", [True, False])
    @pytest.mark.parametrize("grad_ckpt", [True, False])
    def test_multi_view(
        self,
        image_size_dict: dict[str, tuple[int, ...]],
        patch_size_dict: dict[str, tuple[int, ...]],
        scale_factor_dict: dict[str, tuple[int, ...]],
        conv_chans: list[int],
        in_chans_dict: dict[str, int],
        n_frames: int,
        out_chans: int,
        reduce: str,
        mask: bool,
        grad_ckpt: bool,
    ) -> None:
        """Test output sizes."""
        batch = 2
        vit = ConvViT(
            image_size_dict=image_size_dict,
            enc_patch_size_dict=patch_size_dict,
            in_chans_dict=in_chans_dict,
            out_chans=out_chans,
            n_frames=n_frames,
            enc_scale_factor_dict=scale_factor_dict,
            enc_conv_chans=conv_chans,
            enc_conv_n_blocks=1,
            enc_embed_dim=16,
            enc_depth=1,
            enc_n_heads=2,
            mlp_ratio=2,
        )
        vit.set_grad_ckpt(grad_ckpt)
        for m in vit.children():
            if hasattr(m, "grad_ckpt"):
                assert m.grad_ckpt == grad_ckpt
        image_dict = {
            k: torch.rand(size=(batch, n_frames * in_chans_dict[k], *image_size_dict[k]), dtype=torch.float32)
            for k in image_size_dict
        }
        mask_dict = None
        if mask:
            mask_dict = {
                k: torch.rand(size=(batch, vit.enc_down_dict[k].patch_embed.n_patches), dtype=torch.float32) > 0.5
                for k in image_size_dict
            }

        got = vit(image_dict=image_dict, mask_dict=mask_dict, reduce=reduce)
        assert got.shape == (batch, out_chans)


@pytest.mark.parametrize(
    ("name", "n_layers", "expected"),
    [
        ("cls_token", 13, 0),
        ("pos_embed", 13, 0),
        ("patch_embed", 13, 0),
        ("view_embed", 13, 0),
        ("enc_view_embed", 13, 0),
        ("encoder.cls_token", 13, 0),
        ("patch_embed.proj.weight", 13, 0),
        ("patch_embed.proj.bias", 13, 0),
        ("encoder.blocks.0.attn.q.weight", 13, 1),
        ("encoder.blocks.0.attn.kv.weight", 13, 1),
        ("encoder.blocks.0.attn.proj.weight", 13, 1),
        ("encoder.blocks.0.mlp.fc1.weight", 13, 1),
        ("encoder.blocks.0.mlp.fc2.weight", 13, 1),
        ("encoder.blocks.11.attn.q.weight", 13, 12),
        ("encoder.blocks.11.attn.kv.weight", 13, 12),
        ("encoder.blocks.11.attn.proj.weight", 13, 12),
        ("encoder.blocks.11.mlp.fc1.weight", 13, 12),
        ("encoder.blocks.11.mlp.fc2.weight", 13, 12),
    ],
)
def test_get_layer_id_for_vit(name: str, n_layers: int, expected: int) -> None:
    """Test output values."""
    got = get_layer_id_for_vit(name, n_layers)
    assert got == expected


def test_param_groups_lr_decay() -> None:
    """Test json content."""
    expected = {
        "layer_0_decay": {
            "lr_scale": 0.31640625,
            "params": [
                "enc_down_dict.lax.conv_blocks.0.patch_embed.conv.weight",
                "enc_down_dict.lax.conv_blocks.0.conv.0.conv1.weight",
                "enc_down_dict.lax.conv_blocks.0.conv.0.conv2.weight",
                "enc_down_dict.lax.conv_blocks.0.conv.0.dw_conv.weight",
                "enc_down_dict.lax.conv_blocks.0.conv.0.mlp.fc1.weight",
                "enc_down_dict.lax.conv_blocks.0.conv.0.mlp.fc2.weight",
                "enc_down_dict.lax.conv_blocks.1.patch_embed.conv.weight",
                "enc_down_dict.lax.conv_blocks.1.conv.0.conv1.weight",
                "enc_down_dict.lax.conv_blocks.1.conv.0.conv2.weight",
                "enc_down_dict.lax.conv_blocks.1.conv.0.dw_conv.weight",
                "enc_down_dict.lax.conv_blocks.1.conv.0.mlp.fc1.weight",
                "enc_down_dict.lax.conv_blocks.1.conv.0.mlp.fc2.weight",
                "enc_down_dict.lax.patch_embed.proj.weight",
                "enc_down_dict.lax.linear.weight",
                "encoder.cls_token",
            ],
            "weight_decay": 0.05,
        },
        "layer_0_no_decay": {
            "lr_scale": 0.31640625,
            "params": [
                "enc_down_dict.lax.conv_blocks.0.patch_embed.conv.bias",
                "enc_down_dict.lax.conv_blocks.0.patch_embed.norm.weight",
                "enc_down_dict.lax.conv_blocks.0.patch_embed.norm.bias",
                "enc_down_dict.lax.conv_blocks.0.conv.0.norm1.weight",
                "enc_down_dict.lax.conv_blocks.0.conv.0.norm1.bias",
                "enc_down_dict.lax.conv_blocks.0.conv.0.norm2.weight",
                "enc_down_dict.lax.conv_blocks.0.conv.0.norm2.bias",
                "enc_down_dict.lax.conv_blocks.0.conv.0.conv1.bias",
                "enc_down_dict.lax.conv_blocks.0.conv.0.conv2.bias",
                "enc_down_dict.lax.conv_blocks.0.conv.0.dw_conv.bias",
                "enc_down_dict.lax.conv_blocks.0.conv.0.mlp.fc1.bias",
                "enc_down_dict.lax.conv_blocks.0.conv.0.mlp.fc2.bias",
                "enc_down_dict.lax.conv_blocks.1.patch_embed.conv.bias",
                "enc_down_dict.lax.conv_blocks.1.patch_embed.norm.weight",
                "enc_down_dict.lax.conv_blocks.1.patch_embed.norm.bias",
                "enc_down_dict.lax.conv_blocks.1.conv.0.norm1.weight",
                "enc_down_dict.lax.conv_blocks.1.conv.0.norm1.bias",
                "enc_down_dict.lax.conv_blocks.1.conv.0.norm2.weight",
                "enc_down_dict.lax.conv_blocks.1.conv.0.norm2.bias",
                "enc_down_dict.lax.conv_blocks.1.conv.0.conv1.bias",
                "enc_down_dict.lax.conv_blocks.1.conv.0.conv2.bias",
                "enc_down_dict.lax.conv_blocks.1.conv.0.dw_conv.bias",
                "enc_down_dict.lax.conv_blocks.1.conv.0.mlp.fc1.bias",
                "enc_down_dict.lax.conv_blocks.1.conv.0.mlp.fc2.bias",
                "enc_down_dict.lax.patch_embed.proj.bias",
                "enc_down_dict.lax.linear.bias",
            ],
            "weight_decay": 0.0,
        },
        "layer_1_decay": {
            "lr_scale": 0.421875,
            "params": [
                "encoder.blocks.0.attn.q.weight",
                "encoder.blocks.0.attn.kv.weight",
                "encoder.blocks.0.attn.proj.weight",
                "encoder.blocks.0.mlp.fc1.weight",
                "encoder.blocks.0.mlp.fc2.weight",
            ],
            "weight_decay": 0.05,
        },
        "layer_1_no_decay": {
            "lr_scale": 0.421875,
            "params": [
                "encoder.blocks.0.norm1.weight",
                "encoder.blocks.0.norm1.bias",
                "encoder.blocks.0.attn.q.bias",
                "encoder.blocks.0.attn.kv.bias",
                "encoder.blocks.0.attn.proj.bias",
                "encoder.blocks.0.norm2.weight",
                "encoder.blocks.0.norm2.bias",
                "encoder.blocks.0.mlp.fc1.bias",
                "encoder.blocks.0.mlp.fc2.bias",
            ],
            "weight_decay": 0.0,
        },
        "layer_2_decay": {
            "lr_scale": 0.5625,
            "params": [
                "encoder.blocks.1.attn.q.weight",
                "encoder.blocks.1.attn.kv.weight",
                "encoder.blocks.1.attn.proj.weight",
                "encoder.blocks.1.mlp.fc1.weight",
                "encoder.blocks.1.mlp.fc2.weight",
            ],
            "weight_decay": 0.05,
        },
        "layer_2_no_decay": {
            "lr_scale": 0.5625,
            "params": [
                "encoder.blocks.1.norm1.weight",
                "encoder.blocks.1.norm1.bias",
                "encoder.blocks.1.attn.q.bias",
                "encoder.blocks.1.attn.kv.bias",
                "encoder.blocks.1.attn.proj.bias",
                "encoder.blocks.1.norm2.weight",
                "encoder.blocks.1.norm2.bias",
                "encoder.blocks.1.mlp.fc1.bias",
                "encoder.blocks.1.mlp.fc2.bias",
            ],
            "weight_decay": 0.0,
        },
        "layer_3_decay": {
            "lr_scale": 0.75,
            "params": [
                "encoder.blocks.2.attn.q.weight",
                "encoder.blocks.2.attn.kv.weight",
                "encoder.blocks.2.attn.proj.weight",
                "encoder.blocks.2.mlp.fc1.weight",
                "encoder.blocks.2.mlp.fc2.weight",
            ],
            "weight_decay": 0.05,
        },
        "layer_3_no_decay": {
            "lr_scale": 0.75,
            "params": [
                "encoder.blocks.2.norm1.weight",
                "encoder.blocks.2.norm1.bias",
                "encoder.blocks.2.attn.q.bias",
                "encoder.blocks.2.attn.kv.bias",
                "encoder.blocks.2.attn.proj.bias",
                "encoder.blocks.2.norm2.weight",
                "encoder.blocks.2.norm2.bias",
                "encoder.blocks.2.mlp.fc1.bias",
                "encoder.blocks.2.mlp.fc2.bias",
            ],
            "weight_decay": 0.0,
        },
        "layer_4_decay": {
            "lr_scale": 1.0,
            "params": [
                "dec_image_conv_block_dict.lax.conv1.weight",
                "dec_image_conv_block_dict.lax.conv2.weight",
                "dec_image_conv_block_dict.lax.shortcut.weight",
                "dec_conv_blocks_dict.lax.0.conv1.weight",
                "dec_conv_blocks_dict.lax.0.conv2.weight",
                "dec_conv_blocks_dict.lax.0.shortcut.weight",
                "dec_conv_blocks_dict.lax.1.conv1.weight",
                "dec_conv_blocks_dict.lax.1.conv2.weight",
                "dec_conv_blocks_dict.lax.1.shortcut.weight",
                "dec_conv_blocks_dict.lax.2.conv1.weight",
                "dec_conv_blocks_dict.lax.2.conv2.weight",
                "dec_conv_blocks_dict.lax.2.shortcut.weight",
                "decoder_dict.lax.blocks.0.up.weight",
                "decoder_dict.lax.blocks.0.conv.0.conv1.weight",
                "decoder_dict.lax.blocks.0.conv.0.conv2.weight",
                "decoder_dict.lax.blocks.0.conv.1.conv1.weight",
                "decoder_dict.lax.blocks.0.conv.1.conv2.weight",
                "decoder_dict.lax.blocks.1.up.weight",
                "decoder_dict.lax.blocks.1.conv.0.conv1.weight",
                "decoder_dict.lax.blocks.1.conv.0.conv2.weight",
                "decoder_dict.lax.blocks.1.conv.1.conv1.weight",
                "decoder_dict.lax.blocks.1.conv.1.conv2.weight",
                "decoder_dict.lax.blocks.2.up.weight",
                "decoder_dict.lax.blocks.2.conv.0.conv1.weight",
                "decoder_dict.lax.blocks.2.conv.0.conv2.weight",
                "decoder_dict.lax.blocks.2.conv.1.conv1.weight",
                "decoder_dict.lax.blocks.2.conv.1.conv2.weight",
                "pred_head_dict.lax.weight",
            ],
            "weight_decay": 0.05,
        },
        "layer_4_no_decay": {
            "lr_scale": 1.0,
            "params": [
                "encoder.norm.weight",
                "encoder.norm.bias",
                "dec_image_conv_block_dict.lax.norm1.weight",
                "dec_image_conv_block_dict.lax.norm1.bias",
                "dec_image_conv_block_dict.lax.norm2.weight",
                "dec_image_conv_block_dict.lax.norm2.bias",
                "dec_image_conv_block_dict.lax.conv1.bias",
                "dec_image_conv_block_dict.lax.conv2.bias",
                "dec_image_conv_block_dict.lax.shortcut.bias",
                "dec_conv_blocks_dict.lax.0.norm1.weight",
                "dec_conv_blocks_dict.lax.0.norm1.bias",
                "dec_conv_blocks_dict.lax.0.norm2.weight",
                "dec_conv_blocks_dict.lax.0.norm2.bias",
                "dec_conv_blocks_dict.lax.0.conv1.bias",
                "dec_conv_blocks_dict.lax.0.conv2.bias",
                "dec_conv_blocks_dict.lax.0.shortcut.bias",
                "dec_conv_blocks_dict.lax.1.norm1.weight",
                "dec_conv_blocks_dict.lax.1.norm1.bias",
                "dec_conv_blocks_dict.lax.1.norm2.weight",
                "dec_conv_blocks_dict.lax.1.norm2.bias",
                "dec_conv_blocks_dict.lax.1.conv1.bias",
                "dec_conv_blocks_dict.lax.1.conv2.bias",
                "dec_conv_blocks_dict.lax.1.shortcut.bias",
                "dec_conv_blocks_dict.lax.2.norm1.weight",
                "dec_conv_blocks_dict.lax.2.norm1.bias",
                "dec_conv_blocks_dict.lax.2.norm2.weight",
                "dec_conv_blocks_dict.lax.2.norm2.bias",
                "dec_conv_blocks_dict.lax.2.conv1.bias",
                "dec_conv_blocks_dict.lax.2.conv2.bias",
                "dec_conv_blocks_dict.lax.2.shortcut.bias",
                "decoder_dict.lax.blocks.0.up.bias",
                "decoder_dict.lax.blocks.0.conv.0.norm1.weight",
                "decoder_dict.lax.blocks.0.conv.0.norm1.bias",
                "decoder_dict.lax.blocks.0.conv.0.norm2.weight",
                "decoder_dict.lax.blocks.0.conv.0.norm2.bias",
                "decoder_dict.lax.blocks.0.conv.0.conv1.bias",
                "decoder_dict.lax.blocks.0.conv.0.conv2.bias",
                "decoder_dict.lax.blocks.0.conv.1.norm1.weight",
                "decoder_dict.lax.blocks.0.conv.1.norm1.bias",
                "decoder_dict.lax.blocks.0.conv.1.norm2.weight",
                "decoder_dict.lax.blocks.0.conv.1.norm2.bias",
                "decoder_dict.lax.blocks.0.conv.1.conv1.bias",
                "decoder_dict.lax.blocks.0.conv.1.conv2.bias",
                "decoder_dict.lax.blocks.1.up.bias",
                "decoder_dict.lax.blocks.1.conv.0.norm1.weight",
                "decoder_dict.lax.blocks.1.conv.0.norm1.bias",
                "decoder_dict.lax.blocks.1.conv.0.norm2.weight",
                "decoder_dict.lax.blocks.1.conv.0.norm2.bias",
                "decoder_dict.lax.blocks.1.conv.0.conv1.bias",
                "decoder_dict.lax.blocks.1.conv.0.conv2.bias",
                "decoder_dict.lax.blocks.1.conv.1.norm1.weight",
                "decoder_dict.lax.blocks.1.conv.1.norm1.bias",
                "decoder_dict.lax.blocks.1.conv.1.norm2.weight",
                "decoder_dict.lax.blocks.1.conv.1.norm2.bias",
                "decoder_dict.lax.blocks.1.conv.1.conv1.bias",
                "decoder_dict.lax.blocks.1.conv.1.conv2.bias",
                "decoder_dict.lax.blocks.2.up.bias",
                "decoder_dict.lax.blocks.2.conv.0.norm1.weight",
                "decoder_dict.lax.blocks.2.conv.0.norm1.bias",
                "decoder_dict.lax.blocks.2.conv.0.norm2.weight",
                "decoder_dict.lax.blocks.2.conv.0.norm2.bias",
                "decoder_dict.lax.blocks.2.conv.0.conv1.bias",
                "decoder_dict.lax.blocks.2.conv.0.conv2.bias",
                "decoder_dict.lax.blocks.2.conv.1.norm1.weight",
                "decoder_dict.lax.blocks.2.conv.1.norm1.bias",
                "decoder_dict.lax.blocks.2.conv.1.norm2.weight",
                "decoder_dict.lax.blocks.2.conv.1.norm2.bias",
                "decoder_dict.lax.blocks.2.conv.1.conv1.bias",
                "decoder_dict.lax.blocks.2.conv.1.conv2.bias",
                "pred_head_dict.lax.bias",
            ],
            "weight_decay": 0.0,
        },
    }
    no_weight_decay_list: list[str] = []
    weight_decay = 0.05
    layer_decay = 0.75
    model = ConvUNetR(
        image_size_dict={"lax": (16, 24)},
        in_chans_dict={"lax": 1},
        out_chans=3,
        enc_patch_size_dict={"lax": (1, 2)},
        enc_scale_factor_dict={"lax": (2, 2)},
        enc_conv_chans=[4, 8],
        enc_conv_n_blocks=1,
        enc_embed_dim=16,
        enc_depth=3,
        enc_n_heads=2,
        dec_chans=(2, 4, 8),
        dec_patch_size_dict={"lax": (1, 2)},
        dec_scale_factor_dict={"lax": (2, 2)},
        mlp_ratio=2,
    )
    with TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)
        _ = param_groups_lr_decay(model, no_weight_decay_list, weight_decay, layer_decay, out_dir)
        json_path = out_dir / "param_group_names.json"
        with Path.open(json_path, "r", encoding="utf-8") as f:
            got = json.load(f)
        assert got == expected
