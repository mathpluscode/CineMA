"""Vision Transformer model implementation based on timm.

https://github.com/huggingface/pytorch-image-models/blob/v0.9.16/timm/models/vision_transformer.py
https://github.com/TonyLianLong/CrossMAE/blob/main/transformer_utils.py
"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from timm.layers import DropPath, SwiGLU, use_fused_attn
from timm.models.vision_transformer import LayerScale
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from cinema.conv import Linear
from cinema.log import get_logger
from cinema.rotary import RotaryEmbedding

if TYPE_CHECKING:
    from torch.jit import Final
logger = get_logger(__name__)

checkpoint = partial(torch_checkpoint, use_reentrant=False)


def init_weights(m: nn.Module) -> None:
    """Initialize weights for nn.Linear and nn.LayerNorm.

    Args:
        m: module.
    """
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        # bias and weight may be None if elementwise_affine is False
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)


def get_tokens(embed_dim: int, n_tokens: int) -> nn.Parameter:
    """Return a learnable token of shape (1, n_tokens, embed_dim).

    Args:
        embed_dim: number of embedding channels.
        n_tokens: number of tokens.

    Returns:
        token: learnable token.
    """
    token = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
    # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
    nn.init.normal_(token, std=0.02)
    return token


def patchify_2d(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, H, W)
        patch_size: patch size (p, q).

    Returns:
        x: (batch, h*w, p*q*in_chans), where (p, q) is patch size,
            and H = h * p, W = w * q.
    """
    batch, in_chans, h, w = image.shape
    p, q = patch_size
    if h % p != 0:
        raise ValueError(f"Input height ({h}) cannot be divided by patch size ({p}).")
    if w % q != 0:
        raise ValueError(f"Input width ({w}) cannot be divided by patch size ({q}).")
    h, w = h // p, w // q  # grid size
    x = image.reshape(shape=(batch, in_chans, h, p, w, q))
    x = torch.einsum("nchpwq->nhwpqc", x).contiguous()
    x = x.reshape(shape=(batch, h * w, p * q * in_chans))
    return x


def patchify_3d(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, H, W, D)
        patch_size: patch size (p, q, r).

    Returns:
        x: (batch, h*w*d, p*q*r*in_chans), where (p, q, r) is patch size,
            and H = h * p, W = w * q, D = d * r.
    """
    batch, in_chans, h, w, d = image.shape
    p, q, r = patch_size
    if h % p != 0:
        raise ValueError(f"Input height ({h}) cannot be divided by patch size ({p}).")
    if w % q != 0:
        raise ValueError(f"Input width ({w}) cannot be divided by patch size ({q}).")
    if d % r != 0:
        raise ValueError(f"Input depth ({d}) cannot be divided by patch size ({r}).")
    h, w, d = h // p, w // q, d // r  # grid size
    x = image.reshape(shape=(batch, in_chans, h, p, w, q, d, r))
    x = torch.einsum("nchpwqdr->nhwdpqrc", x).contiguous()
    x = x.reshape(shape=(batch, h * w * d, p * q * r * in_chans))
    return x


def patchify_4d(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, H, W, D, T)
        patch_size: patch size (p, q, r, s).

    Returns:
        x: (batch, h*w*d*t, p*q*r*s*in_chans), where (p, q, r, s) is patch size,
            and H = h * p, W = w * q, D = d * r, T = t * s.
    """
    batch, in_chans, h, w, d, t = image.shape
    p, q, r, s = patch_size
    if h % p != 0:
        raise ValueError(f"Input height ({h}) cannot be divided by patch size ({p}).")
    if w % q != 0:
        raise ValueError(f"Input width ({w}) cannot be divided by patch size ({q}).")
    if d % r != 0:
        raise ValueError(f"Input depth ({d}) cannot be divided by patch size ({r}).")
    if t % s != 0:
        raise ValueError(f"Input time ({t}) cannot be divided by patch size ({s}).")
    h, w, d, t = h // p, w // q, d // r, t // s  # grid size
    x = image.reshape(shape=(batch, in_chans, h, p, w, q, d, r, t, s))
    x = torch.einsum("nchpwqdrts->nhwdtpqrsc", x).contiguous()
    x = x.reshape(shape=(batch, h * w * d * t, p * q * r * s * in_chans))
    return x


def patchify(image: torch.Tensor, patch_size: tuple[int, ...]) -> torch.Tensor:
    """Patchify input images.

    Args:
        image: (batch, in_chans, ...).
        patch_size: corresponding patch size.

    Returns:
        x: (batch, n_patches, out_chans).
    """
    if len(patch_size) == 2:
        return patchify_2d(image, patch_size)
    if len(patch_size) == 3:
        return patchify_3d(image, patch_size)
    if len(patch_size) == 4:
        return patchify_4d(image, patch_size)
    raise ValueError(f"Patchify only supports 2D, 3D, and 4D images, got {len(patch_size)}D.")


def unpatchify_2d(x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, h*w, p*q*c), where (p, q) is patch size,
            and H = h * p, W = w * q.
        patch_size: patch size (p, q).
        grid_size: grid size (h, w).

    Returns:
        image of shape (batch, in_chans, H, W)
    """
    batch = x.shape[0]
    p, q = patch_size
    h, w = grid_size
    x = x.reshape(shape=(batch, h, w, p, q, -1))
    x = torch.einsum("nhwpqc->nchpwq", x).contiguous()
    x = x.reshape(shape=(batch, -1, h * p, w * q))
    return x


def unpatchify_3d(x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, h*w*d, p*q*r*c), where (p, q, r) is patch size,
            and H = h * p, W = w * q, D = d * r.
        patch_size: patch size (p, q, r).
        grid_size: grid size (h, w, d).

    Returns:
        image of shape (batch, in_chans, H, W, D)
    """
    batch = x.shape[0]
    p, q, r = patch_size
    h, w, d = grid_size
    x = x.reshape(shape=(batch, h, w, d, p, q, r, -1))
    x = torch.einsum("nhwdpqrc->nchpwqdr", x).contiguous()
    x = x.reshape(shape=(batch, -1, h * p, w * q, d * r))
    return x


def unpatchify_4d(x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, h*w*d*t, p*q*r*s*c), where (p, q, r, s) is patch size,
            and H = h * p, W = w * q, D = d * r, T = t * s.
        patch_size: patch size (p, q, r, s).
        grid_size: grid size (h, w, d, t).

    Returns:
        image of shape (batch, in_chans, H, W, D, T)
    """
    batch = x.shape[0]
    p, q, r, s = patch_size
    h, w, d, t = grid_size
    x = x.reshape(shape=(batch, h, w, d, t, p, q, r, s, -1))
    x = torch.einsum("nhwdtpqrsc->nchpwqdrts", x).contiguous()
    x = x.reshape(shape=(batch, -1, h * p, w * q, d * r, t * s))
    return x


def unpatchify(x: torch.Tensor, patch_size: tuple[int, ...], grid_size: tuple[int, ...]) -> torch.Tensor:
    """Unpatchify to input images.

    Args:
        x: (batch, n_patches, chans).
        patch_size: patch size.
        grid_size: grid size.

    Returns:
        image: (batch, in_chans, ...)
    """
    _, n_patches, chans = x.shape
    if n_patches != math.prod(grid_size):
        raise ValueError(
            f"Number of patches {n_patches} != product of grid size {math.prod(grid_size)} for {grid_size}."
        )
    if chans % math.prod(patch_size) != 0:
        raise ValueError(
            f"Number of channels {chans} is not divisible by product of patch size {math.prod(patch_size)} "
            f"for {patch_size}."
        )
    if len(patch_size) != len(grid_size):
        raise ValueError(f"Patch size {patch_size} and grid size {grid_size} do not match.")
    if len(patch_size) == 2:
        return unpatchify_2d(x, patch_size, grid_size)
    if len(patch_size) == 3:
        return unpatchify_3d(x, patch_size, grid_size)
    if len(patch_size) == 4:
        return unpatchify_4d(x, patch_size, grid_size)
    raise ValueError(f"Unpatchify only supports 2D, 3D, and 4D images, got {len(patch_size)}D.")


class PatchEmbed(nn.Module):
    """Transforms 2D/3D/4D image to patch embeddings."""

    def __init__(
        self,
        image_size: tuple[int, ...],
        patch_size: tuple[int, ...],
        in_chans: int,
        embed_dim: int,
        norm_layer: nn.Module = None,
        bias: bool = True,
        strict_image_size: bool = False,
        dynamic_img_pad: bool = False,
    ) -> None:
        """Initialize the module.

        Args:
            image_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input channels.
            embed_dim: Number of output channels.
            norm_layer: Normalization layer.
            bias: Whether to include bias in the projection layer.
            strict_image_size: Whether to strictly enforce the input image size.
            dynamic_img_pad: Whether to dynamically pad the input image.
        """
        super().__init__()
        self.grad_ckpt = False
        self.n_dims = len(image_size)
        self.patch_size = patch_size
        self.image_size = image_size
        self.grid_size = tuple(s // p for s, p in zip(self.image_size, self.patch_size, strict=False))
        self.n_patches = math.prod(self.grid_size)
        self.strict_image_size = strict_image_size
        self.dynamic_img_pad = dynamic_img_pad
        self.proj = Linear(
            in_features=in_chans * math.prod(patch_size),
            out_features=embed_dim,
            bias=bias,
        )
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.proj.set_grad_ckpt(enable)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            image: of size (N, C, ...).

        Returns:
            Patched tensor, of size (N, L, C).
        """
        _, _, *image_size = image.shape
        if self.image_size is not None:
            if self.strict_image_size:
                for i in range(self.n_dims):
                    if self.image_size[i] != image_size[i]:
                        raise ValueError(
                            f"Input size ({image.shape}) doesn't match config (batch, channel) + {self.image_size}.",
                        )
            elif not self.dynamic_img_pad:
                for i in range(self.n_dims):
                    if image_size[i] % self.patch_size[i] != 0:
                        raise ValueError(
                            f"Input size ({image_size}) should be divisible by patch size ({self.patch_size}).",
                        )
        if self.dynamic_img_pad:
            pad = sum(
                [(0, (p - image_size[i] % p) % p) for i, p in enumerate(self.patch_size)],
                (),
            )
            image = F.pad(image, pad)
        x = patchify(
            image=image,
            patch_size=self.patch_size,
        )
        x = self.proj(x)
        x = self.norm(x)
        return x


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    grid: np.ndarray,
    max_period: int = 10000,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Get 1d sin/cos positional embedding for a grid.

    Half defined by sin, half by cos.
    For position x, the embeddings are (for i = 0,...,half_dim-1)
        sin(x / (max_period ** (i/half_dim)))
        cos(x / (max_period ** (i/half_dim)))

    Args:
        embed_dim: output dimension for each position, E.
        grid: a grid of M positions to be encoded, ndim can be 1 or larger.
        max_period: the maximum frequency.
        dtype: the data type of the output.

    Return:
        out: size (M, E).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be divisible by 2, got {embed_dim}.")
    half_dim = embed_dim // 2

    omega = np.arange(half_dim, dtype=dtype)  # (E/2,)
    omega = np.exp(-np.log(max_period) * omega / half_dim)

    grid = grid.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", grid, omega)  # (M, E/2), outer product

    emb_sin = np.sin(out)  # (M, E/2)
    emb_cos = np.cos(out)  # (M, E/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, E)
    return emb


def get_nd_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """Get Nd sin/cos positional embedding for a grid.

    Zero padding is applied if the embedding dimension is not divisible evenly.

    Args:
        embed_dim: output dimension for each position, E.
        grid: a grid of positions to be encoded, of size (n, ...).

    Return:
        out: size (M, E).
    """
    n = grid.shape[0]
    d = embed_dim // n
    d = d - d % 2  # ensure even division
    pad = embed_dim - d * n
    emb = np.concatenate([get_1d_sincos_pos_embed_from_grid(d, grid[i]) for i in range(n)], axis=1)
    if pad > 0:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], pad))], axis=1)
    return emb


def get_nd_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, ...],
) -> np.ndarray:
    """Get Nd sin/cos positional embedding for a certain size of grid.

    Args:
        embed_dim: output dimension for each position, E.
        grid_size: (N,).

    Returns:
        pos_embed: (M, embed_dim) with M = np.prod(grid_size).
    """
    grid = np.stack(np.meshgrid(*[np.arange(size, dtype=np.float32) for size in grid_size]), axis=0)  # (N, ...)
    pos_embed = get_nd_sincos_pos_embed_from_grid(embed_dim, grid)  # (M, E)
    return pos_embed


def get_pos_embed(embed_dim: int, grid_size: tuple[int, ...]) -> nn.Parameter:
    """Get nd sin/cos positional embedding for any size of grid, without flattening the grid.

    Args:
        embed_dim: output dimension for each position, E.
        grid_size: any grid, in total of size N.

    Returns:
        pos_embed: (1, N, E).
    """
    pos_embed_np = get_nd_sincos_pos_embed(
        embed_dim=embed_dim,
        grid_size=grid_size,
    )
    n_patches = math.prod(grid_size)
    pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim), requires_grad=False)
    pos_embed.data.copy_(torch.from_numpy(pos_embed_np).float().unsqueeze(0))
    return pos_embed


class Attention(nn.Module):
    """Attention layer supporting attention mask and different query and key."""

    fused_attn: Final[bool]  # pylint: disable=invalid-name

    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_eps: float = 1e-5,
        rotary: bool = False,
    ) -> None:
        """Initialize the attention."""
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} should be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, eps=norm_eps) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rotary = RotaryEmbedding(self.head_dim) if rotary else None

    def forward(self, q: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            q: query tokens, (batch, n_q_tokens, ch).
            k: optional key tokens, (batch, n_k_tokens, ch), if None, use q for both query and key.

        Returns:
            q: query tokens, (batch, n_q_tokens, ch).
        """
        if k is None:
            k = q
        elif self.rotary:
            raise ValueError("Rotary positional embedding is not supported with different query and key.")
        batch, n_q_tokens, ch = q.shape
        n_k_tokens = k.shape[1]
        q = self.q(q).reshape(batch, n_q_tokens, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(k).reshape(batch, n_k_tokens, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # q, k, v: (batch, n_heads, n_tokens, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rotary:
            q, k = self.rotary(q, k)

        if self.fused_attn:
            q = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            q = attn @ v

        q = q.transpose(1, 2).reshape(batch, n_q_tokens, ch)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q


class Block(nn.Module):
    """Transformer block."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: int,
        norm_layer: nn.Module,
        norm_eps: float,
        drop_path: float,
        qkv_bias: bool,
        rotary: bool,
        act_layer: nn.Module,
        mlp_layer: nn.Module,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: float | None = None,
    ) -> None:
        """Initialize the block."""
        super().__init__()
        self.grad_ckpt = False

        self.norm1 = norm_layer(dim, eps=norm_eps)
        self.attn = Attention(
            dim,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            norm_eps=norm_eps,
            rotary=rotary,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim, eps=norm_eps)
        hidden_features = int(dim * mlp_ratio)
        if mlp_layer == SwiGLU:
            # swiglu has more parameters, so we reduce the hidden_features
            hidden_features = int(((hidden_features * 2.0 / 3.0) + 255) // 256 * 256)
            logger.info(f"Using SwiGLU with hidden_features {hidden_features}.")
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=hidden_features,
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        if enable and (self.attn.attn_drop.p > 0.0 or self.attn.proj_drop.p > 0.0):
            logger.error("Gradient checkpointing is not supported with dropout.")
            enable = False
        self.grad_ckpt = enable

    def path1(self, q: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
        """Path 1 without drop path."""
        return self.ls1(self.attn(self.norm1(q), k))

    def path2(self, q: torch.Tensor) -> torch.Tensor:
        """Path 2 without drop path."""
        return self.ls2(self.mlp(self.norm2(q)))

    def forward(self, q: torch.Tensor, k: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            q: query tokens, (batch, n_q_tokens, ch), with positional and view embeddings added without norm.
            k: optional key tokens after norm, (batch, n_k_tokens, ch), if None, use q for both query and key.

        Returns:
            q: query tokens, (batch, n_q_tokens, ch).
        """
        h = checkpoint(self.path1, q, k) if self.grad_ckpt else self.path1(q, k)
        q = q + self.drop_path1(h)
        h = checkpoint(self.path2, q) if self.grad_ckpt else self.path2(q)
        q = q + self.drop_path2(h)
        return q


class ViTEncoder(nn.Module):
    """VisionTransformer encoder."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: int,
        qkv_bias: bool,
        norm_layer: nn.Module,
        norm_eps: float,
        rotary: bool,
        act_layer: nn.Module,
        mlp_layer: nn.Module,
        drop_path: float,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        self.grad_ckpt = False
        self.cls_token = get_tokens(embed_dim=embed_dim, n_tokens=1)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    norm_eps=norm_eps,
                    rotary=rotary,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ],
        )
        self.norm = norm_layer(embed_dim, eps=norm_eps)

        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        for block in self.blocks:
            block.set_grad_ckpt(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tokens with positional embedding added, (batch, n_enc_keep, emb_dim).

        Returns:
            x: latent tensor, (batch, 1+n_enc_keep, emb_dim).
        """
        # append cls token
        # (batch, 1, emb_dim)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1).contiguous()
        # (batch, 1+n_enc_keep, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def feature_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for feature extraction, returns all intermediate features.

        Args:
            x: tokens with positional embedding added, (batch, n_enc_keep, emb_dim).

        Returns:
            x: stacked tensor, (batch, 1+n_enc_keep, emb_dim, n_layers).
        """
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1).contiguous()  # (batch, 1, emb_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch, 1+n_enc_keep, emb_dim)
        xs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i != len(self.blocks) - 1:  # the last layer is not appended
                xs.append(x)
        x = self.norm(x)
        xs.append(x)
        return torch.stack(xs, dim=-1)  # (batch, 1+n_enc_keep, emb_dim, n_layers)


class ViTDecoder(nn.Module):
    """VisionTransformer decoder."""

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: int,
        qkv_bias: bool,
        norm_layer: nn.Module,
        norm_eps: float,
        rotary: bool,
        act_layer: nn.Module,
        mlp_layer: nn.Module,
        drop_path: float,
    ) -> None:
        """Initialize the module."""
        super().__init__()
        self.grad_ckpt = False
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    norm_eps=norm_eps,
                    rotary=rotary,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    drop_path=drop_path,
                )
                for _ in range(depth)
            ],
        )
        self.norm = norm_layer(embed_dim)

        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        for block in self.blocks:
            block.set_grad_ckpt(enable)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None,
        n_enc_masked: int,
    ) -> torch.Tensor:
        """Forward pass of the decoder.

        https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_pretrain.py
        https://github.com/TonyLianLong/CrossMAE/blob/main/models_cross.py for cross attention.

        Args:
            x_q: query patches.
              if not cross attention, it's all patched with masked ones at the end, (batch, 1+n_patches, dec_emb_dim).
              else, it's cls token and masked patches, (batch, 1+n_enc_masked, dec_emb_dim).
            x_k: key patches.
              if not cross attention, it's None.
              else, it's visible patches, (batch, 1+n_enc_keep, dec_emb_dim).
            n_enc_masked: number of masked patches.

        Returns:
            pred: predicted masked patches, (batch, n_enc_masked, dec_emb_dim).
        """
        # (batch, 1+n_patches, dec_emb_dim)
        for block in self.blocks:
            x_q = block(x_q, x_k)

        # remove cls token, visible patches, return only masked patches
        # (batch, n_enc_masked, dec_emb_dim)
        x_q = x_q[:, -n_enc_masked:, :]
        x_q = self.norm(x_q)

        return x_q


def get_vit_config(size: str) -> dict[str, int]:
    """Get VisionTransformer configuration.

    Except from tiny, other configurations are from MAE.
    https://github.com/facebookresearch/mae/blob/main/models_mae.py

    Args:
        size: size of the model, must be in ['tiny', 'base', 'large', 'huge'].

    Returns:
        config_dict: configuration dictionary.
    """
    if size not in ["tiny", "base", "large", "huge"]:
        raise ValueError(f"size must be in ['tiny', 'base', 'large', 'huge'], got {size}.")
    return {
        "tiny": {
            "enc_embed_dim": 16,
            "enc_depth": 1,
            "enc_n_heads": 2,
            "dec_embed_dim": 16,
            "dec_depth": 1,
            "dec_n_heads": 2,
        },
        "base": {
            "enc_embed_dim": 768,
            "enc_depth": 12,
            "enc_n_heads": 12,
            "dec_embed_dim": 512,
            "dec_depth": 8,
            "dec_n_heads": 16,
        },
        "large": {
            "enc_embed_dim": 1024,
            "enc_depth": 24,
            "enc_n_heads": 16,
            "dec_embed_dim": 512,
            "dec_depth": 8,
            "dec_n_heads": 16,
        },
        "huge": {
            "enc_embed_dim": 1280,
            "enc_depth": 32,
            "enc_n_heads": 16,
            "dec_embed_dim": 512,
            "dec_depth": 8,
            "dec_n_heads": 16,
        },
    }[size]
