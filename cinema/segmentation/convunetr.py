"""Segmentation models.

https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/output_adapters.py#L359
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from timm.layers import Mlp
from torch import nn

from cinema.conv import Conv2d, Conv3d, ConvResBlock, ConvTranspose2d, ConvTranspose3d
from cinema.convvit import DownsampleEncoder, load_pretrain_weights
from cinema.log import get_logger
from cinema.vit import ViTEncoder, get_vit_config, init_weights

logger = get_logger(__name__)


class UpsampleDecoder(nn.Module):
    """Up-sample decoder module with convolutions for unet."""

    def __init__(
        self,
        n_dims: int,
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
        deconv_cls = ConvTranspose2d if n_dims == 2 else ConvTranspose3d
        self.blocks = nn.ModuleList()
        for i, ch in enumerate(chans[::-1]):
            block = nn.Module()
            up_kernel_size = patch_size if i == len(chans) - 1 else scale_factor
            out_chans = chans[-i - 2] if i < len(chans) - 1 else ch
            block.up = deconv_cls(ch, out_chans, kernel_size=up_kernel_size, stride=up_kernel_size)
            block.conv = nn.ModuleList(
                [
                    ConvResBlock(
                        n_dims=n_dims,
                        in_chans=out_chans,
                        out_chans=out_chans,
                        dropout=dropout,
                        kernel_size=kernel_size,
                        norm=norm,
                    )
                    for _ in range(n_blocks)
                ]
            )
            self.blocks.append(block)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        for block in self.blocks:
            block.up.set_grad_ckpt(enable)
            for conv in block.conv:
                conv.set_grad_ckpt(enable)

    def forward(self, embeddings: list[torch.Tensor | None]) -> torch.Tensor:
        """Decode the image via upsampling.

        Args:
            embeddings: List of embeddings from encoder, if None, the skip connection is not used.

        Returns:
            List of embeddings from each layer.
        """
        x = embeddings.pop()
        for block in self.blocks:
            x = block.up(x)
            skip = embeddings.pop()
            if skip is not None:
                x = x + skip
            for conv in block.conv:
                x = conv(x)
        return x


def check_conv_unetr_enc_dec_compatiblity(
    enc_patch_size: tuple[int, ...],
    enc_scale_factor: tuple[int, ...],
    enc_n_conv_layers: int,
    dec_depth: int,
    dec_patch_size: tuple[int, ...],
    dec_scale_factor: tuple[int, ...],
) -> tuple[int, int]:
    """Check ConvUNetR encoder and decoder compatibility.

    Args:
        enc_patch_size: patch size for the first layer.
        enc_scale_factor: scale factor for other layers.
        enc_n_conv_layers: number of conv layers in the encoder.
        dec_depth: number of layers for decoder.
        dec_patch_size: patch size for the last decoder layer.
        dec_scale_factor: scale factor for each non-last decoder layer.
    """
    if enc_n_conv_layers >= dec_depth:
        raise ValueError(f"enc_n_conv_layers {enc_n_conv_layers} must be less than dec_depth {dec_depth}.")
    if any(f < s for f, s in zip(enc_patch_size, dec_patch_size, strict=False)):
        raise ValueError(f"enc_patch_size {enc_patch_size} must be greater than dec_patch_size {dec_patch_size}.")
    enc_patch_size = tuple(enc_patch_size)
    enc_scale_factor = tuple(enc_scale_factor)
    dec_patch_size = tuple(dec_patch_size)
    dec_scale_factor = tuple(dec_scale_factor)

    enc_factor = enc_patch_size
    for _ in range(enc_n_conv_layers):
        enc_factor = tuple(f * s for f, s in zip(enc_factor, enc_scale_factor, strict=False))

    dec_factor = dec_patch_size
    n_layers_wo_skip = None
    n_downsample_layers = None
    for i in range(dec_depth):
        if dec_factor == enc_patch_size:
            n_layers_wo_skip = i
        if dec_factor == enc_factor:
            n_downsample_layers = dec_depth - 1 - i
        dec_factor = tuple(f * s for f, s in zip(dec_factor, dec_scale_factor, strict=False))

    if n_layers_wo_skip is None:
        raise ValueError(
            f"enc_patch_size {enc_patch_size} must be equal to "
            f"dec_patch_size {dec_patch_size} times certain number of {dec_scale_factor} ."
        )
    if n_downsample_layers is None:
        raise ValueError(
            f"enc_factor {enc_factor} must be equal to "
            f"dec_patch_size {dec_patch_size} times certain number of {dec_scale_factor} ."
        )

    return n_layers_wo_skip, n_downsample_layers


def get_model(
    config: DictConfig,
) -> ConvUNetR:
    """Get model from config.

    Args:
        config: configuration file.

    Returns:
        ConvUNetR model.
    """

    def get_view_config(v: str) -> DictConfig:
        if v == "sax":
            return config.data.sax
        if hasattr(config.data, "lax"):
            return config.data.lax
        return config.data[v]

    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    vit_config = get_vit_config(config.model.convunetr.size)
    image_size_dict = {v: get_view_config(v).patch_size for v in views}
    in_chans_dict = {v: get_view_config(v).in_chans for v in views}
    ndim_dict = {v: 3 if v == "sax" else 2 for v in views}
    enc_patch_size_dict = {v: config.model.convunetr.enc_patch_size[:n] for v, n in ndim_dict.items()}
    enc_scale_factor_dict = {v: config.model.convunetr.enc_scale_factor[:n] for v, n in ndim_dict.items()}
    dec_patch_size_dict = {v: config.model.convunetr.dec_patch_size[:n] for v, n in ndim_dict.items()}
    dec_scale_factor_dict = {v: config.model.convunetr.dec_scale_factor[:n] for v, n in ndim_dict.items()}
    model = ConvUNetR(
        image_size_dict=image_size_dict,
        in_chans_dict=in_chans_dict,
        out_chans=config.model.out_chans,
        enc_patch_size_dict=enc_patch_size_dict,
        enc_scale_factor_dict=enc_scale_factor_dict,
        enc_conv_chans=config.model.convunetr.enc_conv_chans,
        enc_conv_n_blocks=config.model.convunetr.enc_conv_n_blocks,
        enc_embed_dim=vit_config["enc_embed_dim"],
        enc_depth=vit_config["enc_depth"],
        enc_n_heads=vit_config["enc_n_heads"],
        dec_chans=config.model.convunetr.dec_chans,
        dec_patch_size_dict=dec_patch_size_dict,
        dec_scale_factor_dict=dec_scale_factor_dict,
        dropout=config.model.convunetr.dropout,
        drop_path=config.model.convunetr.drop_path,
    )
    model.set_grad_ckpt(config.grad_ckpt)
    return model


class ConvUNetR(nn.Module):
    """UNetR inspired by ConvMAE, supporting multiple views."""

    def __init__(
        self,
        image_size_dict: dict[str, tuple[int, ...]],
        in_chans_dict: dict[str, int],
        out_chans: int,
        enc_patch_size_dict: dict[str, tuple[int, ...]],
        enc_scale_factor_dict: dict[str, tuple[int, ...]],
        enc_conv_chans: list[int],
        enc_conv_n_blocks: int,
        enc_embed_dim: int,
        enc_depth: int,
        enc_n_heads: int,
        dec_chans: tuple[int, ...],
        dec_patch_size_dict: dict[str, tuple[int, ...]],
        dec_scale_factor_dict: dict[str, tuple[int, ...]],
        dec_kernel_size: int = 3,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_eps: float = 1e-5,
        rotary: bool = False,
        act_layer: nn.Module = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        norm: str = "layer",
    ) -> None:
        """Initialize the module.

        Args:
            image_size_dict: input image size per view.
            in_chans_dict: number of input channels per view.
            out_chans: number of output channels.
            enc_patch_size_dict: patch size for the first layer per view.
            enc_scale_factor_dict: scale factor for other layers per view.
            enc_conv_chans: number of channels for each conv layer, if empty, no conv layers.
            enc_conv_n_blocks: number of MaskedConvBlock for each enc_conv_block.
            enc_embed_dim: number of embedding channels for ViT encoder.
            enc_depth: number of layers for ViT encoder.
            enc_n_heads: number of heads for ViT encoder.
            dec_chans: number of channels for decoder layers, from last to first.
            dec_patch_size_dict: patch size for the last decoder layer (largest image size) per view.
            dec_scale_factor_dict: scale factor for each non-last decoder layer per view.
            dec_kernel_size: kernel size for decoder convolutions.
            act_layer: activation layer.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: whether to include bias in the qkv projection layer.
            norm_layer: normalization layer.
            norm_eps: epsilon for normalization layer.
            rotary: use rotary position embedding in attention.
            mlp_layer: mlp layer in transformer.
            dropout: dropout rate.
            drop_path: drop path rate.
            norm: normalization layer for conv blocks, 'instance' or 'layer' or 'group'.
        """
        super().__init__()

        self.grad_ckpt = False

        self.views = list(image_size_dict.keys())
        for view in self.views:
            if len(image_size_dict[view]) not in {2, 3}:
                raise ValueError(f"Invalid image_size for {view}, must be 2D or 3D, got {image_size_dict[view]}.")

        n_layers_wo_skip_list = []
        n_downsample_layers_list = []
        for view in self.views:
            n_layers_wo_skip, n_downsample_layers = check_conv_unetr_enc_dec_compatiblity(
                enc_patch_size=enc_patch_size_dict[view],
                enc_scale_factor=enc_scale_factor_dict[view],
                enc_n_conv_layers=len(enc_conv_chans),
                dec_depth=len(dec_chans),
                dec_patch_size=dec_patch_size_dict[view],
                dec_scale_factor=dec_scale_factor_dict[view],
            )
            n_layers_wo_skip_list.append(n_layers_wo_skip)
            n_downsample_layers_list.append(n_downsample_layers)
        if len(set(n_layers_wo_skip_list)) != 1:
            raise ValueError(f"n_layers_wo_skip_list {n_layers_wo_skip_list} must be the same for all views.")
        if len(set(n_downsample_layers_list)) != 1:
            raise ValueError(f"n_downsample_layers_list {n_downsample_layers_list} must be the same for all views.")
        self.n_layers_wo_skip = n_layers_wo_skip_list[0]
        n_downsample_layers = n_downsample_layers_list[0]

        # per view conv encoder
        self.enc_down_dict = nn.ModuleDict(
            {
                view: DownsampleEncoder(
                    image_size=image_size_dict[view],
                    in_chans=in_chans_dict[view],
                    patch_size=enc_patch_size_dict[view],
                    scale_factor=enc_scale_factor_dict[view],
                    conv_chans=enc_conv_chans,
                    conv_n_blocks=enc_conv_n_blocks,
                    embed_dim=enc_embed_dim,
                    norm=norm,
                )
                for view in self.views
            }
        )
        # shared Transformer encoder
        self.encoder = ViTEncoder(
            embed_dim=enc_embed_dim,
            depth=enc_depth,
            n_heads=enc_n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            norm_eps=norm_eps,
            rotary=rotary,
            act_layer=act_layer,
            mlp_layer=mlp_layer,
            drop_path=drop_path,
        )

        # per view layers
        self.dec_image_conv_block_dict = nn.ModuleDict()
        self.dec_down_blocks_dict = nn.ModuleDict()
        self.dec_conv_blocks_dict = nn.ModuleDict()
        self.decoder_dict = nn.ModuleDict()
        self.pred_head_dict = nn.ModuleDict()
        for view in self.views:
            n_dims_view = len(image_size_dict[view])

            # raw image encoder
            self.dec_image_conv_block_dict[view] = ConvResBlock(
                n_dims=n_dims_view,
                in_chans=in_chans_dict[view],
                out_chans=dec_chans[0],
                kernel_size=dec_kernel_size,
                dropout=dropout,
                act_layer=act_layer,
                norm=norm,
            )

            # per view downsampling layers
            conv_cls = Conv2d if n_dims_view == 2 else Conv3d
            self.dec_down_blocks_dict[view] = nn.ModuleList(
                [
                    conv_cls(
                        enc_embed_dim,
                        enc_embed_dim,
                        kernel_size=dec_scale_factor_dict[view],
                        stride=dec_scale_factor_dict[view],
                        padding="valid",
                    )
                    for _ in range(n_downsample_layers)
                ]
            )

            # channel adjustment
            self.dec_conv_blocks_dict[view] = nn.ModuleList()
            for i, ch in enumerate(enc_conv_chans):
                # for skips from conv layers
                self.dec_conv_blocks_dict[view].append(
                    ConvResBlock(
                        n_dims=n_dims_view,
                        in_chans=ch,
                        out_chans=dec_chans[self.n_layers_wo_skip + i],
                        kernel_size=dec_kernel_size,
                        dropout=dropout,
                        act_layer=act_layer,
                        norm=norm,
                    )
                )
            for i in range(n_downsample_layers + 1):
                # for ViT outputs and downsampling layers
                self.dec_conv_blocks_dict[view].append(
                    ConvResBlock(
                        n_dims=n_dims_view,
                        in_chans=enc_embed_dim,
                        out_chans=dec_chans[self.n_layers_wo_skip + len(enc_conv_chans) + i],
                        kernel_size=dec_kernel_size,
                        dropout=dropout,
                        act_layer=act_layer,
                        norm=norm,
                    )
                )

            # decoder
            self.decoder_dict[view] = UpsampleDecoder(
                n_dims=n_dims_view,
                chans=dec_chans,
                patch_size=dec_patch_size_dict[view],
                scale_factor=dec_scale_factor_dict[view],
                norm=norm,
            )
            self.pred_head_dict[view] = conv_cls(dec_chans[0], out_chans, kernel_size=1)

        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        self.encoder.set_grad_ckpt(enable)
        for view in self.views:
            self.enc_down_dict[view].set_grad_ckpt(enable)
            self.dec_image_conv_block_dict[view].set_grad_ckpt(enable)
            for block in self.dec_down_blocks_dict[view]:
                block.set_grad_ckpt(enable)
            for block in self.dec_conv_blocks_dict[view]:
                block.set_grad_ckpt(enable)
            self.decoder_dict[view].set_grad_ckpt(enable)
            self.pred_head_dict[view].set_grad_ckpt(enable)

    def forward(
        self,
        image_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            image_dict: dictionary of images, (batch, in_chans, *image_size).

        Returns:
            dict of logits, each has shape (batch, out_chans, *image_size).
        """
        views = list(image_dict.keys())
        if any(x not in self.views for x in views):
            raise ValueError(f"views {views} must be in self.input_keys {self.views}.")

        # encode per view
        xs = []
        skips = []
        ns = []
        for view in views:
            # skip_view (batch, chans, *spatial_shape) for each
            # x_view (batch, n_patches_view, enc_emb_dim)
            skips_view, x_view = self.enc_down_dict[view](image_dict[view], mask=None)
            skips.append(skips_view)
            xs.append(x_view)
            ns.append(x_view.shape[1])

        # joint Transformer encoder
        # (batch, 1+n_patches, enc_emb_dim)
        x = self.encoder(torch.cat(xs, dim=1))

        # split x into chunks per view for decoder
        xs = torch.split(x, [1, *ns], dim=1)  # include cls token
        xs = xs[1:]  # exclude cls token

        # decoder per view
        preds = {}
        for i, view in enumerate(views):
            # (batch, n_patches_view, enc_emb_dim)
            x_view = xs[i]
            # (batch, enc_emb_dim, n_patches_view)
            x_view = x_view.permute(0, 2, 1)
            # (batch, enc_emb_dim, *grid_size)
            x_view = x_view.reshape(x_view.shape[0], x_view.shape[1], *self.enc_down_dict[view].patch_embed.grid_size)

            # skip_view (batch, chans, *spatial_shape) for each
            skips_view = skips[i] + [x_view]

            # add downsampled tensors
            for block in self.dec_down_blocks_dict[view]:
                x_view = block(x_view)
                skips_view.append(x_view)

            # match tensors to layer lengths
            embeddings_view = [self.dec_image_conv_block_dict[view](image_dict[view])] + [None] * self.n_layers_wo_skip
            for j, block in enumerate(self.dec_conv_blocks_dict[view]):
                # adjust channels
                embeddings_view.append(block(skips_view[j]))

            # decoder
            x = self.decoder_dict[view](embeddings_view)
            preds[view] = self.pred_head_dict[view](x)
        return preds

    @classmethod
    def from_finetuned(  # type: ignore[no-untyped-def]
        cls,
        repo_id: str,
        model_filename: str,
        config_filename: str,
        **kwargs,
    ) -> ConvUNetR:
        """Load finetuned weights."""
        # download weights
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            **kwargs,
        )
        logger.info(f"Cached model weights to {model_path}.")
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                state_dict[key] = f.get_tensor(key)

        # download config
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            **kwargs,
        )
        logger.info(f"Cached model config to {config_path}.")
        config = OmegaConf.load(config_path)

        # init model and load state dict
        model = get_model(config)
        model.load_state_dict(state_dict)
        logger.info("Loaded pretrained weights.")
        return model

    @classmethod
    def from_pretrained(cls, config: DictConfig, freeze: bool, **kwargs) -> ConvUNetR:  # type: ignore[no-untyped-def]
        """Load pretrained weights."""
        # download weights
        model_path = hf_hub_download(
            repo_id="mathpluscode/CineMA",
            filename="pretrained/cinema.safetensors",
            **kwargs,
        )
        logger.info(f"Cached model weights to {model_path}.")

        # init model and load state dict
        model = get_model(config)
        model = load_pretrain_weights(
            model=model,
            views=config.model.views,
            ckpt_path=Path(model_path),
            freeze=freeze,
        )
        return model
