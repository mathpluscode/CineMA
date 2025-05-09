"""Rotary Position Embeddings migrated from ESM3.

https://github.com/evolutionaryscale/esm/blob/main/esm/layers/rotary.py
"""

from __future__ import annotations

import torch
from einops import repeat


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the vector counter-clockwise by 90 degrees.

    Args:
        x: (batch, n_tokens, n_heads, head_dim)

    Returns:
        rotated: (batch, n_tokens, n_heads, head_dim)
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply the rotary position embeddings.

    Args:
        x: (batch, n_x_tokens, n_heads, head_dim)
        cos: (n_tokens, rotary_dim / 2)
        sin: (n_tokens, rotary_dim / 2)

    Returns:
        rotated: (batch, n_tokens, n_heads, head_dim)
    """
    ro_dim = cos.shape[-1] * 2
    if ro_dim > x.shape[-1]:
        raise ValueError(f"Rotary dim {ro_dim} is larger than the last dimension of x {x.shape[-1]}")
    n_tokens = x.size(1)
    cos = cos[:n_tokens]  # (n_x_tokens, ro_dim / 2)
    sin = sin[:n_tokens]
    cos = repeat(cos, "s d -> s 1 (2 d)")  # (n_x_tokens, 1, ro_dim)
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim]) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """The rotary position embeddings."""

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the RotaryEmbedding.

        Args:
            dim: the dimension of the input tensor.
            base: the base of the exponential, theta in paper.
            scaling_factor: extended with linear scaling.
            device: the device to put the tensors on.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.scaling_factor = scaling_factor
        self.device = device

        self.n_tokens = 0
        self.cos = None
        self.sin = None

        inv_freq = 1 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def update_cos_sin(self, n_tokens: int, device: torch.device, dtype: torch.dtype) -> None:
        """Update the cosine and sine cache.

        Reset the tables if the sequence length has changed, if we're on a new device,
        or if we're switching from inference mode to training.

        Args:
            n_tokens: the number of tokens in the sequence.
            device: the device to put the tensors on.
            dtype: the data type of the tensors.
        """
        if (
            (n_tokens > self.n_tokens)  # type: ignore[unreachable]
            or (self.cos is None)
            or (self.cos.device != device)
            or (self.cos.dtype != dtype)
            or (self.training and self.cos.is_inference())
        ):
            self.n_tokens = n_tokens
            t = torch.arange(n_tokens, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
            freqs = torch.outer(t, self.inv_freq)
            self.cos = torch.cos(freqs).to(dtype)
            self.sin = torch.sin(freqs).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            q: (batch, n_tokens, n_heads, head_dim)
            k: (batch, n_tokens, n_heads, head_dim)
            offset: can be used in generation where the qkv being passed in is only the last
            few tokens in the batch.

        Returns:
            q: (batch, n_tokens, n_heads, head_dim)
            k: (batch, n_tokens, n_heads, head_dim)
        """
        if q.shape[1] != k.shape[1]:
            raise ValueError("q and k must have the same sequence length")

        self.update_cos_sin(q.shape[1] + offset, device=q.device, dtype=q.dtype)
        return (
            apply_rotary_emb(q, self.cos[offset:], self.sin[offset:]),  # type: ignore[index]
            apply_rotary_emb(k, self.cos[offset:], self.sin[offset:]),  # type: ignore[index]
        )
