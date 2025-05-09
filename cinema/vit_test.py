"""Test for Vision Transformer model."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from timm.layers import Mlp
from timm.models.vision_transformer import Attention as TimmAttention
from timm.models.vision_transformer import Block as TimmBlock
from torch import nn

from cinema.vit import (
    Attention,
    Block,
    PatchEmbed,
    get_1d_sincos_pos_embed_from_grid,
    get_nd_sincos_pos_embed,
    patchify,
    unpatchify,
)


class TestPatchify:
    """Test patchify."""

    @pytest.mark.parametrize(
        ("image_size", "patch_size", "in_chans", "expected_size"),
        [
            ((16, 16), (2, 4), 1, (32, 8)),
            ((16, 16), (2, 4), 3, (32, 24)),
            ((8, 12, 16), (2, 4, 8), 1, (24, 64)),
            ((8, 12, 16), (2, 4, 8), 3, (24, 192)),
            ((8, 12, 16, 9), (2, 4, 8, 3), 1, (72, 192)),
            ((8, 12, 16, 9), (2, 4, 8, 3), 3, (72, 576)),
        ],
    )
    def test_patchify_and_unpatchify(
        self,
        image_size: tuple[int, ...],
        patch_size: tuple[int, ...],
        in_chans: int,
        expected_size: tuple[int, ...],
    ) -> None:
        """Test patchify."""
        batch = 2
        image = torch.rand(size=(batch, in_chans, *image_size), dtype=torch.float32)
        x = patchify(image, patch_size)
        assert x.shape == (batch, *expected_size)
        grid_size = tuple(s // p for s, p in zip(image_size, patch_size, strict=False))
        recon = unpatchify(x, patch_size, grid_size)
        assert recon.shape == (batch, in_chans, *image_size)


@pytest.mark.parametrize(
    (
        "image_size",
        "patch_size",
        "in_chans",
        "embed_dim",
        "expected_shape",
    ),
    [
        ((16, 16), (4, 4), 1, 4, (16, 4)),
        ((16, 16), (4, 2), 3, 4, (32, 4)),
        ((16, 16, 16), (4, 4, 4), 1, 4, (64, 4)),
        ((16, 16, 16), (4, 2, 4), 3, 4, (128, 4)),
        ((16, 16, 16, 9), (4, 4, 4, 3), 1, 5, (192, 5)),
        ((16, 16, 16, 9), (4, 2, 4, 3), 3, 7, (384, 7)),
    ],
)
@pytest.mark.parametrize("grad_ckpt", [True, False])
def test_patch_embed(
    image_size: tuple[int, ...],
    patch_size: tuple[int, ...],
    in_chans: int,
    embed_dim: int,
    expected_shape: tuple[int, ...],
    grad_ckpt: bool,
) -> None:
    """Test output sizes."""
    batch = 2
    patch_embed = PatchEmbed(
        image_size=image_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
    )
    patch_embed.set_grad_ckpt(grad_ckpt)
    for m in patch_embed.children():
        if hasattr(m, "grad_ckpt"):
            assert m.grad_ckpt == grad_ckpt
    x = torch.rand(size=(batch, in_chans, *image_size), dtype=torch.float32)
    out = patch_embed(x)
    assert out.shape == (batch, *expected_shape)


@pytest.mark.parametrize("grid_size", [4, (2, 3)])
def test_get_1d_sincos_pos_embed_from_grid(grid_size: int | tuple[int, ...]) -> None:
    """Test output shapes."""
    embed_dim = 16
    if isinstance(grid_size, int):
        grid_size = (grid_size,)
    rng = np.random.default_rng()
    grid = rng.random(grid_size)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    assert pos_embed.shape == (np.prod(grid_size), embed_dim)


@pytest.mark.parametrize(
    "grid_size",
    [
        (3, 4),
        (2, 3, 4),
    ],
)
def test_get_nd_sincos_pos_embed(grid_size: tuple[int, ...]) -> None:
    """Test output shapes."""
    embed_dim = 16
    pos_embed = get_nd_sincos_pos_embed(embed_dim, grid_size)
    assert pos_embed.shape == (np.prod(grid_size), embed_dim)


class TestAttention:
    """Test for Attention."""

    batch = 2
    dim = 16

    @pytest.mark.parametrize("n_tokens", [1, 5, 16])
    def test_timm(self, n_tokens: int) -> None:
        """Compare with timm implementation."""
        x = torch.rand(self.batch, n_tokens, self.dim, dtype=torch.float32)

        torch.manual_seed(0)
        timm_attention = TimmAttention(self.dim)
        expected = timm_attention(x)

        torch.manual_seed(0)
        attention = Attention(self.dim)
        got = attention(x)

        assert torch.allclose(got, expected)

    @pytest.mark.parametrize("n_q_tokens", [1, 5, 16])
    @pytest.mark.parametrize("n_k_tokens", [1, 5, 16])
    def test_qk(self, n_q_tokens: int, n_k_tokens: int) -> None:
        """Test output shapes."""
        q = torch.rand(self.batch, n_q_tokens, self.dim, dtype=torch.float32)
        k = torch.rand(self.batch, n_k_tokens, self.dim, dtype=torch.float32)

        torch.manual_seed(0)
        attention = Attention(self.dim)
        got = attention(q, k)

        assert got.shape == q.shape


class TestBlock:
    """Test for Block."""

    batch = 2
    dim = 16
    n_heads = 4

    def get_block(self) -> Block:
        """Get Block instance."""
        return Block(
            self.dim,
            self.n_heads,
            mlp_ratio=4,
            qkv_bias=False,
            rotary=False,
            norm_layer=nn.LayerNorm,
            norm_eps=1e-5,
            drop_path=0.0,
            act_layer=nn.GELU,
            mlp_layer=Mlp,
        )

    @pytest.mark.parametrize("n_tokens", [1, 5, 16])
    def test_timm(self, n_tokens: int) -> None:
        """Compare with timm implementation."""
        x = torch.rand(self.batch, n_tokens, self.dim, dtype=torch.float32)

        torch.manual_seed(0)
        timm_block = TimmBlock(self.dim, self.n_heads)
        expected = timm_block(x)

        torch.manual_seed(0)
        block = self.get_block()
        got = block(x)

        assert torch.allclose(got, expected)

    @pytest.mark.parametrize("n_q_tokens", [1, 5, 16])
    @pytest.mark.parametrize("n_k_tokens", [1, 5, 16])
    def test_qk(self, n_q_tokens: int, n_k_tokens: int) -> None:
        """Test output shapes."""
        q = torch.rand(self.batch, n_q_tokens, self.dim, dtype=torch.float32)
        k = torch.rand(self.batch, n_k_tokens, self.dim, dtype=torch.float32)
        block = self.get_block()
        got = block(q, k)
        assert got.shape == q.shape

    @pytest.mark.parametrize("grad_ckpt", [True, False])
    def test_grad_ckpt(self, grad_ckpt: bool) -> None:
        """Test output shapes."""
        n_tokens = 5
        q = torch.rand(self.batch, n_tokens, self.dim, dtype=torch.float32)
        k = torch.rand(self.batch, n_tokens, self.dim, dtype=torch.float32)
        block = self.get_block()
        block.set_grad_ckpt(grad_ckpt)
        for m in block.children():
            if hasattr(m, "grad_ckpt"):
                assert m.grad_ckpt == grad_ckpt
        got = block(q, k)
        assert got.shape == q.shape
