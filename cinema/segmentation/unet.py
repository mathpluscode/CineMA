"""UNet for segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from cinema.conv import Conv2d, Conv3d, ConvNormActBlock, ConvResBlock, ConvTranspose2d, ConvTranspose3d


class DownsampleEncoder(nn.Module):
    """Down-sample encoder module with convolutions for unet."""

    def __init__(
        self,
        n_dims: int,
        in_chans: int,
        chans: tuple[int, ...],
        patch_size: int | tuple[int, ...],
        scale_factor: int | tuple[int, ...],
        norm: str,
        kernel_size: int = 3,
        n_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the module.

        Args:
            n_dims: Number of dimensions, 2 or 3.
            in_chans: Number of input channels.
            chans: Number of output channels for each layer.
            patch_size: Patch size for the first down-sampling layer.
            scale_factor: Scale factor for other down-sampling layers.
            kernel_size: Convolution kernel size for residual blocks.
            n_blocks: Number of blocks for each layer.
            dropout: Dropout rate.
            norm: Normalization layer, either 'instance' or 'layer' or 'group'.
        """
        if n_dims not in {2, 3}:
            raise ValueError(f"Invalid n_dims, must be 2 or 3, got {n_dims}.")
        super().__init__()
        self.grad_ckpt = False

        # init layers
        conv_cls = Conv2d if n_dims == 2 else Conv3d
        self.in_conv = ConvNormActBlock(
            n_dims=n_dims,
            in_chans=in_chans,
            out_chans=chans[0],
            kernel_size=kernel_size,
            norm=norm,
        )
        self.blocks = nn.ModuleList()
        for i, ch in enumerate(chans):
            block = nn.Module()
            block.conv = nn.ModuleList(
                [
                    ConvResBlock(
                        n_dims=n_dims,
                        in_chans=ch,
                        out_chans=ch,
                        dropout=dropout,
                        kernel_size=kernel_size,
                        norm=norm,
                    )
                    for _ in range(n_blocks)
                ]
            )
            if i < len(chans) - 1:
                down_kernel_size = patch_size if i == 0 else scale_factor
                block.down = conv_cls(
                    ch, chans[i + 1], kernel_size=down_kernel_size, stride=down_kernel_size, padding="valid"
                )
            self.blocks.append(block)

        # store attributes
        self.n_blocks = n_blocks

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.in_conv.set_grad_ckpt(enable)
        for block in self.blocks:
            for conv in block.conv:
                conv.set_grad_ckpt(enable)
            if hasattr(block, "down"):
                block.down.set_grad_ckpt(enable)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encoder the image via downsampling.

        Args:
            x: array of shape (batch, in_chans, *spatial_shape).

        Returns:
            List of embeddings from each layer.
        """
        # encoder raw input
        x = self.in_conv(x)

        # encoding
        embeddings = [x]
        for i, block in enumerate(self.blocks):
            for j in range(self.n_blocks):
                x = block.conv[j](x)
                embeddings.append(x)

            if i < len(self.blocks) - 1:
                x = block.down(x)
                embeddings.append(x)

        return embeddings


class UpsampleDecoder(nn.Module):
    """Up-sample decoder module with convolutions for unet."""

    def __init__(
        self,
        n_dims: int,
        in_chans: int,
        chans: tuple[int, ...],
        patch_size: int | tuple[int, ...],
        scale_factor: int | tuple[int, ...],
        norm: str,
        kernel_size: int = 3,
        n_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the module.

        Args:
            n_dims: Number of dimensions, 2 or 3.
            in_chans: Number of input channels.
            chans: Number of output channels for each layer.
            patch_size: Patch size for the first down-sampling layer.
            scale_factor: Scale factor for other down-sampling layers.
            kernel_size: Convolution kernel size for residual blocks.
            n_blocks: Number of blocks for each layer.
            dropout: Dropout rate.
            norm: Normalization layer, either 'instance' or 'layer' or 'group'.
        """
        if n_dims not in {2, 3}:
            raise ValueError(f"Invalid n_dims, must be 2 or 3, got {n_dims}.")
        super().__init__()
        self.grad_ckpt = False

        # init layers
        conv_cls = ConvTranspose2d if n_dims == 2 else ConvTranspose3d
        self.in_conv = ConvNormActBlock(
            n_dims=n_dims,
            in_chans=in_chans,
            out_chans=chans[0],
            kernel_size=kernel_size,
            norm=norm,
        )
        self.blocks = nn.ModuleList()
        for i, ch in enumerate(chans[::-1]):
            block = nn.Module()
            block.conv = nn.ModuleList(
                [
                    ConvResBlock(
                        n_dims=n_dims,
                        in_chans=ch,
                        out_chans=ch,
                        dropout=dropout,
                        kernel_size=kernel_size,
                        norm=norm,
                    )
                    for _ in range(n_blocks)
                ]
            )
            if i < len(chans) - 1:
                up_kernel_size = patch_size if i == len(chans) - 2 else scale_factor
                block.up = conv_cls(ch, chans[-i - 2], kernel_size=up_kernel_size, stride=up_kernel_size)
            self.blocks.append(block)

        # store attributes
        self.n_blocks = n_blocks

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.in_conv.set_grad_ckpt(enable)
        for block in self.blocks:
            for conv in block.conv:
                conv.set_grad_ckpt(enable)
            if hasattr(block, "up"):
                block.up.set_grad_ckpt(enable)

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Decode the image via upsampling.

        Args:
            embeddings: List of embeddings from encoder.

        Returns:
            List of embeddings from each layer.
        """
        x = embeddings.pop()
        for i, block in enumerate(self.blocks):
            for j in range(self.n_blocks):
                x = block.conv[j](x) + embeddings.pop()

            if i < len(self.blocks) - 1:
                x = block.up(x)
                skipped = embeddings.pop()
                if x.shape != skipped.shape:
                    # skipped may have larger spatial shape
                    # pad x to match the spatial shape of skipped
                    # pad needs to be defined from last dimension to first
                    pad = [(0, s - x) for s, x in zip(skipped.shape, x.shape, strict=False)]
                    pad = sum(pad[::-1], ())  # type: ignore[arg-type]
                    x = F.pad(x, pad)
                x = x + skipped
        return x


class UNet(nn.Module):
    """UNet with optional mask and timesteps inputs."""

    def __init__(
        self,
        n_dims: int,
        in_chans: int,
        out_chans: int,
        chans: tuple[int, ...],
        dropout: float = 0.0,
        patch_size: int | tuple[int, ...] = 2,
        scale_factor: int | tuple[int, ...] = 2,
        n_blocks: int = 2,
        kernel_size: int = 3,
        norm: str = "instance",
    ) -> None:
        """Initialize the module.

        Args:
            n_dims: number of spatial dimensions.
            in_chans: number of input channels.
            out_chans: number of output channels.
            chans: number of channels in each layer.
            dropout: dropout rate.
            patch_size: patch size for the first down-sampling layer.
            scale_factor: scale factor for other down-sampling layers.
            n_blocks: number of blocks for each layer.
            kernel_size: convolution kernel size for residual blocks.
            norm: normalization layer, either 'instance' or 'layer' or 'group'.
        """
        if n_dims not in {2, 3}:
            raise ValueError(f"Invalid n_dims, must be 2 or 3, got {n_dims}.")
        super().__init__()
        self.grad_ckpt = False
        self.encoder = DownsampleEncoder(
            n_dims=n_dims,
            in_chans=in_chans,
            chans=chans,
            patch_size=patch_size,
            scale_factor=scale_factor,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            dropout=dropout,
            norm=norm,
        )
        self.decoder = UpsampleDecoder(
            n_dims=n_dims,
            in_chans=chans[-1],
            chans=chans,
            patch_size=patch_size,
            scale_factor=scale_factor,
            kernel_size=kernel_size,
            n_blocks=n_blocks,
            dropout=dropout,
            norm=norm,
        )
        conv_cls = Conv2d if n_dims == 2 else Conv3d
        self.out_conv = conv_cls(chans[0], out_chans, kernel_size=1)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.encoder.set_grad_ckpt(enable)
        self.decoder.set_grad_ckpt(enable)

    def forward(self, image_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        For vanilla UNet, mask and t are None.

        Args:
            image_dict: dict with one view, (batch, in_chans, *image_size).

        Returns:
            logits_dict: dict with one view, (batch, out_chans, *image_size).
        """
        if len(image_dict) != 1:
            raise ValueError(f"Only one view is supported, got {len(image_dict)} views.")
        view = next(iter(image_dict.keys()))
        image = image_dict[view]

        embeddings = self.encoder(image)
        x = self.decoder(embeddings)
        logits = self.out_conv(x)

        return {view: logits}
