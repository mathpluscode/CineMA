"""Tests for rotary positional embedding."""

import pytest
import torch

from cinema.rotary import RotaryEmbedding, apply_rotary_emb, rotate_half


def test_rotate_half() -> None:
    """Test rotate_half."""
    x = torch.tensor([[[[1, 2], [3, 4]]]])
    expected = torch.tensor([[[[-2, 1], [-4, 3]]]])
    assert torch.allclose(rotate_half(x), expected)


@pytest.mark.parametrize(
    ("n_x_tokens", "n_tokens"),
    [
        (8, 8),
        (8, 10),
    ],
)
def test_apply_rotary_emb(n_x_tokens: int, n_tokens: int) -> None:
    """Test output shapes."""
    batch = 2
    n_heads = 4
    head_dim = 12
    half_rotary_dim = 5
    x = torch.rand(batch, n_x_tokens, n_heads, head_dim)
    cos = torch.rand(n_tokens, half_rotary_dim)
    sin = torch.rand(n_tokens, half_rotary_dim)
    got = apply_rotary_emb(x, cos, sin)
    assert got.shape == (batch, n_x_tokens, n_heads, head_dim)


def test_rotary_embedding() -> None:
    """Test RotaryEmbedding."""
    batch = 2
    n_heads = 4
    head_dim = 12
    rotary = RotaryEmbedding(head_dim)

    n_tokens = 8
    q = torch.rand(batch, n_tokens, n_heads, head_dim)
    k = torch.rand(batch, n_tokens, n_heads, head_dim)
    got_q, got_k = rotary(q, k)
    assert got_q.shape == q.shape
    assert got_k.shape == k.shape

    n_tokens = 12
    q = torch.rand(batch, n_tokens, n_heads, head_dim)
    k = torch.rand(batch, n_tokens, n_heads, head_dim)
    got_q, got_k = rotary(q, k)
    assert got_q.shape == q.shape
    assert got_k.shape == k.shape
