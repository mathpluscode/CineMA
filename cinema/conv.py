"""Layers for convolutional neural networks.

https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
"""

from __future__ import annotations

from functools import partial

import torch
from timm.layers import DropPath, Mlp, to_2tuple
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

checkpoint = partial(torch_checkpoint, use_reentrant=False)

KernelSizeType = tuple[int, ...] | int


class Linear(nn.Linear):
    """Linear with gradient checkpointing support."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize the module."""
        super().__init__(*args, **kwargs)
        self.grad_ckpt = False

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return checkpoint(super().forward, x) if self.grad_ckpt else super().forward(x)


class Conv2d(nn.Conv2d):
    """Conv2d with gradient checkpointing support."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize the module."""
        super().__init__(*args, **kwargs)
        self.grad_ckpt = False

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return checkpoint(super().forward, x) if self.grad_ckpt else super().forward(x)


class Conv3d(nn.Conv3d):
    """Conv3d with gradient checkpointing support."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize the module."""
        super().__init__(*args, **kwargs)
        self.grad_ckpt = False

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return checkpoint(super().forward, x) if self.grad_ckpt else super().forward(x)


class ConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose2d with gradient checkpointing support."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize the module."""
        super().__init__(*args, **kwargs)
        self.grad_ckpt = False

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable

    def forward(self, x: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        """Forward pass."""
        return checkpoint(super().forward, x, output_size) if self.grad_ckpt else super().forward(x, output_size)


class ConvTranspose3d(nn.ConvTranspose3d):
    """ConvTranspose3d with gradient checkpointing support."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Initialize the module."""
        super().__init__(*args, **kwargs)
        self.grad_ckpt = False

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable

    def forward(self, x: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        """Forward pass."""
        return checkpoint(super().forward, x, output_size) if self.grad_ckpt else super().forward(x, output_size)


class ConvMlp(Mlp):
    """Extend Mlp to ConvMlp for 2d and 3d convolutions.

    https://github.com/Alpha-VL/ConvMAE/blob/main/vision_transformer.py
    """

    def __init__(
        self,
        n_dims: int,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module | None = None,
        bias: tuple[bool, bool] | bool = True,
        drop: tuple[float, float] | float = 0.0,
    ) -> None:
        """Initialize the module.

        Args:
            n_dims: Number of dimensions, 2 or 3.
            in_features: Number of input features.
            hidden_features: Number of hidden features.
            out_features: Number of output features.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            bias: Whether to include bias, if tuple, separate for each layer.
            drop: Dropout rate, if tuple, separate for each layer.
        """
        if n_dims not in {2, 3}:
            raise ValueError(f"Invalid n_dims, must be 2 or 3, got {n_dims}.")
        self.grad_ckpt = False
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        bias_tuple: tuple[bool, bool] = to_2tuple(bias)
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=act_layer,
            norm_layer=norm_layer,
            bias=bias_tuple,
            drop=drop,
            use_conv=True,
        )
        del self.fc1, self.fc2  # type: ignore[has-type]
        conv_cls = Conv2d if n_dims == 2 else Conv3d
        self.fc1 = conv_cls(in_features, hidden_features, kernel_size=1, bias=bias_tuple[0])
        self.fc2 = conv_cls(hidden_features, out_features, kernel_size=1, bias=bias_tuple[1])

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.fc1.set_grad_ckpt(enable)
        self.fc2.set_grad_ckpt(enable)


class ConvLayerNorm(nn.LayerNorm):
    """Convolutional layer with normalization."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape, (batch, in_chans, *spatial_shape).

        Returns:
            Output tensor.
        """
        # (batch, in_chans, *spatial_shape) -> (batch, *spatial_shape, in_chans)
        x = x.permute(0, *range(2, x.ndim), 1)
        # LayerNorm performs normalization over the last dimension
        x = super().forward(x)
        # (batch, *spatial_shape, in_chans) -> (batch, in_chans, *spatial_shape)
        x = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))
        return x.contiguous()


def get_conv_norm(n_dims: int, in_chans: int, norm: str, eps: float = 1e-6, n_groups: int = 32) -> nn.Module:
    """Get a normalization layer for convolutional networks.

    Args:
        n_dims: Number of dimensions, 2 or 3.
        in_chans: Number of input channels.
        norm: Normalization layer, either 'instance' or 'layer' or 'group'.
        n_groups: Number of groups for group normalization.
        eps: Epsilon value for normalization layers.

    Returns:
        Normalization layer.
    """
    if norm == "instance":
        return nn.InstanceNorm2d(in_chans, eps=eps) if n_dims == 2 else nn.InstanceNorm3d(in_chans, eps=eps)
    if norm == "layer":
        return ConvLayerNorm(in_chans, eps=eps)
    if norm == "group":
        return nn.GroupNorm(num_groups=min(n_groups, in_chans), num_channels=in_chans, eps=eps, affine=True)
    raise ValueError(f"Invalid norm type, got {norm}, must be 'instance' or 'layer' or 'group'.")


class ConvNormActBlock(nn.Module):
    """Block with conv-norm-act."""

    def __init__(
        self,
        n_dims: int,
        in_chans: int,
        out_chans: int,
        norm: str,
        kernel_size: KernelSizeType = 3,
        stride: KernelSizeType = 1,
        padding: str = "same",
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialize the module.

        Args:
            n_dims: Number of dimensions, 2 or 3.
            in_chans: Number of input channels.
            out_chans: Number of output channels.
            norm: Normalization layer, either 'instance' or 'layer' or 'group'.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
            act_layer: Activation layer.
        """
        if n_dims not in {2, 3}:
            raise ValueError(f"Invalid n_dims, must be 2 or 3, got {n_dims}.")
        if not isinstance(kernel_size, int) and len(kernel_size) != n_dims:
            raise ValueError(f"Invalid kernel_size {kernel_size}, must be an integer or a tuple of {n_dims} integers.")
        if not isinstance(stride, int) and len(stride) != n_dims:
            raise ValueError(f"Invalid stride {stride}, must be an integer or a tuple of {n_dims} integers.")
        super().__init__()
        self.grad_ckpt = False
        conv_cls = Conv2d if n_dims == 2 else Conv3d
        self.conv = conv_cls(in_chans, out_chans, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = get_conv_norm(n_dims=n_dims, in_chans=out_chans, norm=norm)
        self.act = act_layer()

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.conv.set_grad_ckpt(enable)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, in_chans, *spatial_shape).

        Returns:
            (batch, out_chans, *spatial_shape),
            output spatial shape may be different from input.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvResBlock(nn.Module):
    """Residue conv block.

    The activation function is GELU instead of Swish.

    https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py
    """

    def __init__(
        self,
        n_dims: int,
        in_chans: int,
        out_chans: int,
        norm: str,
        dropout: float = 0.0,
        kernel_size: KernelSizeType = 3,
        act_layer: nn.Module = nn.GELU,
    ) -> None:
        """Initialize the module.

        Args:
            n_dims: Number of dimensions, 2 or 3.
            in_chans: Number of input channels.
            out_chans: Number of output channels.
            dropout: Dropout rate.
            kernel_size: Convolution kernel size.
            norm: Normalization layer, either 'instance' or 'layer' or 'group'.
            act_layer: Activation layer.
        """
        if n_dims not in {2, 3}:
            raise ValueError(f"Invalid n_dims, must be 2 or 3, got {n_dims}.")
        if not isinstance(kernel_size, int) and len(kernel_size) != n_dims:
            raise ValueError(f"Invalid kernel_size {kernel_size}, must be an integer or a tuple of {n_dims} integers.")
        super().__init__()
        self.grad_ckpt = False
        conv_cls = Conv2d if n_dims == 2 else Conv3d
        self.norm1 = get_conv_norm(n_dims=n_dims, in_chans=in_chans, norm=norm)
        self.norm2 = get_conv_norm(n_dims=n_dims, in_chans=out_chans, norm=norm)
        self.conv1 = conv_cls(in_chans, out_chans, kernel_size=kernel_size, padding="same")
        self.conv2 = conv_cls(out_chans, out_chans, kernel_size=kernel_size, padding="same")
        self.dropout = nn.Dropout(dropout)
        self.act = act_layer()
        self.shortcut = conv_cls(in_chans, out_chans, kernel_size=1) if in_chans != out_chans else nn.Identity()

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.conv1.set_grad_ckpt(enable)
        self.conv2.set_grad_ckpt(enable)
        if hasattr(self.shortcut, "set_grad_ckpt"):
            self.shortcut.set_grad_ckpt(enable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, in_chans, *spatial_shape).

        Returns:
            (batch, out_chans, *spatial_shape).
        """
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class MaskedConvBlock(nn.Module):
    """Masked Convolutional Block.

    https://github.com/Alpha-VL/ConvMAE/blob/main/vision_transformer.py
    """

    def __init__(
        self,
        n_dims: int,
        in_chans: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm: str = "layer",
    ) -> None:
        """Initialize the module.

        Args:
            n_dims: Number of dimensions, 2 or 3.
            in_chans: Number of input channels.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            dropout: Dropout rate.
            drop_path: Stochastic depth probability.
            act_layer: Activation layer.
            norm: Normalization layer, either 'instance' or 'layer' or 'group'.
        """
        if n_dims not in {2, 3}:
            raise ValueError(f"Invalid n_dims, must be 2 or 3, got {n_dims}.")
        super().__init__()
        self.grad_ckpt = False
        self.norm1 = get_conv_norm(n_dims=n_dims, in_chans=in_chans, norm=norm)
        self.norm2 = get_conv_norm(n_dims=n_dims, in_chans=in_chans, norm=norm)
        conv_cls = Conv2d if n_dims == 2 else Conv3d
        self.conv1 = conv_cls(in_chans, in_chans, kernel_size=1, padding="same")
        self.conv2 = conv_cls(in_chans, in_chans, kernel_size=1, padding="same")
        self.dw_conv = conv_cls(in_chans, in_chans, kernel_size=5, padding="same", groups=in_chans)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = ConvMlp(
            n_dims=n_dims, in_features=in_chans, hidden_features=in_chans * mlp_ratio, act_layer=act_layer, drop=dropout
        )

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.conv1.set_grad_ckpt(enable)
        self.conv2.set_grad_ckpt(enable)
        self.dw_conv.set_grad_ckpt(enable)
        self.mlp.set_grad_ckpt(enable)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, in_chans, *spatial_shape).
            mask: (batch, *spatial_shape) to mask certain regions for all channels, 1 is visible, 0 is masked.

        Returns:
            (batch, out_chans, *spatial_shape).
        """
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.dw_conv(mask.unsqueeze(1).to(x.dtype) * self.conv1(self.norm1(x)))))
        else:
            x = x + self.drop_path(self.conv2(self.dw_conv(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
