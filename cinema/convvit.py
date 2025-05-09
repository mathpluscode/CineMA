"""ConvVit model for classification/regression/dino."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from timm.layers import Mlp
from torch import nn
from torch.nn import functional as F  # noqa: N812

from cinema.conv import Conv2d, Conv3d, ConvNormActBlock, Linear, MaskedConvBlock
from cinema.log import get_logger
from cinema.vit import PatchEmbed, ViTEncoder, get_pos_embed, get_vit_config, init_weights

logger = get_logger(__name__)


def upsample_mask(mask: torch.Tensor, scale_factor: tuple[int, ...]) -> torch.Tensor:
    """Upsample mask.

    Args:
        mask: binary mask, (batch, *spatial_shape).
        scale_factor: scale factor for each spatial dimension.

    Returns:
        mask: upsampled mask, (batch, *upsampled_spatial_shape).
    """
    if mask.ndim != len(scale_factor) + 1:
        raise ValueError(
            f"mask must have the same number of dimensions as scale_factor except batch, "
            f"got {mask.ndim} and {len(scale_factor)}."
        )
    expand_shape = (*(-1 for _ in range(mask.ndim)), np.prod(scale_factor))
    permute_dim: tuple[int, ...] = (0,)
    for i, _ in enumerate(scale_factor):
        permute_dim = (*permute_dim, i + 1, len(scale_factor) + i + 1)
    upsampled_shape = (mask.shape[0], *(s * f for s, f in zip(mask.shape[1:], scale_factor, strict=False)))
    mask = (
        mask.unsqueeze(-1)
        .expand(*expand_shape)
        .reshape(*mask.shape, *scale_factor)
        .permute(*permute_dim)
        .reshape(upsampled_shape)
    )
    return mask


class DownsampleEncoder(nn.Module):
    """Down-sample encoder module from ConvMAE before ViT, with masking support."""

    def __init__(
        self,
        image_size: tuple[int, ...],
        in_chans: int,
        patch_size: tuple[int, ...],
        scale_factor: tuple[int, ...],
        conv_chans: list[int],
        conv_n_blocks: int,
        embed_dim: int,
        norm: str,
    ) -> None:
        """Initialize the module.

        Args:
            image_size: input image size.
            in_chans: number of input channels.
            patch_size: patch size for the first layer.
            scale_factor: scale factor for other layers.
            conv_chans: number of channels for each conv layer, if empty, no conv layers.
            conv_n_blocks: number of MaskedConvBlock for each conv_block.
            embed_dim: number of embedding channels for ViT encoder.
            norm: normalization layer, either 'instance' or 'layer' or 'group'.
        """
        super().__init__()
        self.grad_ckpt = False

        n_dims = len(image_size)
        n_conv_layers = len(conv_chans)
        self.patch_sizes = [patch_size] + [scale_factor] * n_conv_layers

        # conv encoder
        conv_emb_size: tuple[int, ...] = image_size
        eff_patch_size: tuple[int, ...] = (1,) * n_dims
        conv_emb_in_chans = in_chans
        self.conv_blocks = nn.ModuleList()
        for patch_size_i, chans_i in zip(self.patch_sizes[:-1], conv_chans, strict=False):
            block = nn.Module()
            block.patch_embed = ConvNormActBlock(
                n_dims=n_dims,
                in_chans=conv_emb_in_chans,
                out_chans=chans_i,
                norm=norm,
                kernel_size=patch_size_i,
                stride=patch_size_i,
                padding="valid",
            )
            conv_emb_size = tuple(s // p for s, p in zip(conv_emb_size, patch_size_i, strict=False))
            eff_patch_size = tuple(s * p for s, p in zip(eff_patch_size, patch_size_i, strict=False))
            conv_emb_in_chans = chans_i

            block.conv = nn.ModuleList(
                [MaskedConvBlock(n_dims=n_dims, in_chans=chans_i, norm=norm) for _ in range(conv_n_blocks)]
            )
            self.conv_blocks.append(block)

        # effective patch size after conv layers and ViT patch embedding
        self.eff_patch_size = tuple(s * p for s, p in zip(eff_patch_size, self.patch_sizes[-1], strict=False))
        # embedding before ViT encoder
        self.patch_embed = PatchEmbed(
            image_size=conv_emb_size,
            patch_size=self.patch_sizes[-1],
            in_chans=conv_emb_in_chans,
            embed_dim=embed_dim,
        )
        self.linear = Linear(embed_dim, embed_dim)  # original MAE does not have this layer
        self.pos_embed = get_pos_embed(
            embed_dim=embed_dim,
            grid_size=self.patch_embed.grid_size,
        )

        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        for block in self.conv_blocks:
            block.patch_embed.set_grad_ckpt(enable)
            for conv in block.conv:
                conv.set_grad_ckpt(enable)
        self.patch_embed.set_grad_ckpt(enable)
        self.linear.set_grad_ckpt(enable)

    def interpolate_pos_encoding(self, grid_size: tuple[int, ...]) -> torch.Tensor:
        """Interpolate positional encoding to match the input size.

        https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py

        Args:
            grid_size: grid size of the input, image size divided by patch size.

        Returns:
            pos_embed: (1+n_patches, emb_dim).
        """
        if grid_size == self.patch_embed.grid_size:
            # no need to interpolate
            return self.pos_embed

        n_dims = len(grid_size)
        mode = {2: "bicubic", 3: "trilinear"}[n_dims]
        pos_embed = self.pos_embed.float()  # (1, N, emb_dim)
        emb_dim = pos_embed.shape[-1]
        pos_embed = pos_embed.reshape(1, *self.patch_embed.grid_size, emb_dim)
        pos_embed = torch.moveaxis(pos_embed, -1, 1)  # (1, emb_dim, *grid_size)
        pos_embed = F.interpolate(pos_embed, size=grid_size, mode=mode, antialias=False)
        pos_embed = torch.moveaxis(pos_embed, 1, -1).reshape(1, -1, emb_dim)
        return pos_embed.to(dtype=self.pos_embed.dtype)

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            image: (batch, in_chans, ...).
            mask: (batch, n_patches) at ViT grid size, None if no masking.

        Returns:
            skips: list of skipped features from each conv layer, each is (batch, chans, *spatial_shape).
            x: (batch, n_keep, emb_dim), input to ViT encoder.
        """
        batch_size, _, *image_size = image.shape
        grid_size = tuple(s // p for s, p in zip(image_size, self.eff_patch_size, strict=False))

        if mask is None:
            conv_masks = [None] * len(self.conv_blocks)
        else:
            # upsample mask to conv input resolution
            # each element is (batch, n_patches), 0 is keep, 1 is remove
            conv_masks = []
            conv_mask = mask.reshape(batch_size, *grid_size)
            for patch_size in self.patch_sizes[:0:-1]:  # drop the first one and reverse
                conv_mask = upsample_mask(conv_mask, scale_factor=patch_size)
                conv_masks.insert(0, ~conv_mask)  # 1 is visible

        # conv encoder
        skips = []
        x = image
        for block, conv_mask in zip(self.conv_blocks, conv_masks, strict=False):
            x = block.patch_embed(x)
            for conv in block.conv:
                x = conv(x, conv_mask)
            skips.append(x)

        # patch embedding
        # (batch, n_patches, emb_dim)
        x = self.linear(self.patch_embed(x)) + self.interpolate_pos_encoding(grid_size)

        return skips, x


class MultiScaleFusion(nn.Module):
    """Multi-scale fusion module from ConvMAE, with masking support."""

    def __init__(
        self,
        image_size: tuple[int, ...],
        patch_size: tuple[int, ...],
        scale_factor: tuple[int, ...],
        conv_chans: list[int],
        embed_dim: int,
        norm_layer: nn.Module,
        norm_eps: float,
    ) -> None:
        """Initialize the module.

        Args:
            image_size: input image size.
            patch_size: patch size for the first layer.
            scale_factor: scale factor for other layers.
            conv_chans: number of channels for each conv layer, if empty, no conv layers.
            embed_dim: number of embedding channels for ViT encoder.
            norm_layer: normalization layer.
            norm_eps: epsilon for normalization layer.
        """
        super().__init__()
        self.grad_ckpt = False

        n_dims = len(image_size)
        patch_sizes = [patch_size] + [scale_factor] * len(conv_chans)

        # shape pre-calculation
        grid_size: tuple[int, ...] = image_size
        for patch_size_i in patch_sizes:
            grid_size = tuple(s // p for s, p in zip(grid_size, patch_size_i, strict=False))

        # downsample blocks
        conv_emb_size: tuple[int, ...] = image_size
        conv_cls = Conv2d if n_dims == 2 else Conv3d
        self.down_convs = nn.ModuleList()
        for i, ch in enumerate(conv_chans):
            conv_emb_size = tuple(s // p for s, p in zip(conv_emb_size, patch_sizes[i], strict=False))
            down_kernel_size = tuple(s // p for s, p in zip(conv_emb_size, grid_size, strict=False))
            conv = conv_cls(ch, embed_dim, kernel_size=down_kernel_size, stride=down_kernel_size, padding="valid")
            self.down_convs.append(conv)
        self.norm = norm_layer(embed_dim, eps=norm_eps)

        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        for conv in self.down_convs:
            conv.set_grad_ckpt(enable)

    def forward(
        self,
        skips: list[torch.Tensor],
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass.

        Args:
            skips: list of skipped features from each conv layer, each is (batch, chans, *spatial_shape).
            x: (batch, n_keep, emb_dim), output from ViT encoder.
            mask: (batch, n_patches) at ViT grid size, None if no masking.
                If mask is given, masked skip features are removed before fusion.

        Returns:
            x: (batch, n_keep, emb_dim).
        """
        for skip, conv in zip(skips, self.down_convs, strict=False):
            # (batch, emb_dim, *spatial_shape)
            down = conv(skip)
            down = down.flatten(2).transpose(1, 2)  # (batch, n_patches, emb_dim)
            # (batch, n_keep, emb_dim)
            if mask is not None:
                down = down[~mask].reshape(x.shape[0], -1, x.shape[-1])
            x = x + down
        x = self.norm(x)
        return x


def get_model(config: DictConfig) -> ConvViT:
    """Load model from config.

    Args:
        config: configuration file.

    Returns:
        ConvViT model.
    """
    views = [config.model.views] if isinstance(config.model.views, str) else config.model.views
    vit_config = get_vit_config(config.model.convvit.size)
    in_chans_dict = {v: config.data.sax.in_chans if v == "sax" else config.data.lax.in_chans for v in views}
    if hasattr(config.data, "class_column"):
        out_chans = len(config.data[config.data.class_column])
    elif hasattr(config.data, "regression_column"):
        out_chans = 1
    else:
        logger.info(f"Using config.model.out_chans {config.model.out_chans}.")
        out_chans = config.model.out_chans
    image_size_dict = {v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views}
    ndim_dict = {v: 3 if v == "sax" else 2 for v in views}
    enc_patch_size_dict = {v: config.model.convvit.enc_patch_size[:n] for v, n in ndim_dict.items()}
    enc_scale_factor_dict = {v: config.model.convvit.enc_scale_factor[:n] for v, n in ndim_dict.items()}
    model = ConvViT(
        image_size_dict=image_size_dict,
        n_frames=config.model.n_frames,
        in_chans_dict=in_chans_dict,
        out_chans=out_chans,
        enc_patch_size_dict=enc_patch_size_dict,
        enc_scale_factor_dict=enc_scale_factor_dict,
        enc_conv_chans=config.model.convvit.enc_conv_chans,
        enc_conv_n_blocks=config.model.convvit.enc_conv_n_blocks,
        enc_embed_dim=vit_config["enc_embed_dim"],
        enc_depth=vit_config["enc_depth"],
        enc_n_heads=vit_config["enc_n_heads"],
        drop_path=config.model.convvit.drop_path,
    )
    model.set_grad_ckpt(config.grad_ckpt)
    return model


class ConvViT(nn.Module):
    """Multi-view ViT with convolution layers from ConvMAE for classification/regression."""

    def __init__(
        self,
        image_size_dict: dict[str, tuple[int, ...]],
        in_chans_dict: dict[str, int],
        n_frames: int,
        out_chans: int,
        enc_patch_size_dict: dict[str, tuple[int, ...]],
        enc_scale_factor_dict: dict[str, tuple[int, ...]],
        enc_conv_chans: list[int],
        enc_conv_n_blocks: int,
        enc_embed_dim: int,
        enc_depth: int,
        enc_n_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_eps: float = 1e-5,
        rotary: bool = False,
        act_layer: nn.Module = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        drop_path: float = 0.0,
        norm: str = "layer",
        head_layer: nn.Module | None = nn.Linear,
    ) -> None:
        """Initialize the module.

        Args:
            image_size_dict: input image size per view.
            in_chans_dict: number of input channels per view.
            n_frames: number of frames.
            out_chans: number of output channels.
            enc_patch_size_dict: patch size for the first layer per view.
            enc_scale_factor_dict: scale factor for other layers per view.
            enc_conv_chans: number of channels for each conv layer, if empty, no conv layers.
            enc_conv_n_blocks: number of MaskedConvBlock for each enc_conv_block.
            enc_embed_dim: number of embedding channels for ViT encoder.
            enc_depth: number of layers for ViT encoder.
            enc_n_heads: number of heads for ViT encoder.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: whether to include bias in the qkv projection layer.
            norm_layer: normalization layer.
            norm_eps: epsilon for normalization layer.
            rotary: use rotary position embedding in attention.
            act_layer: activation layer.
            mlp_layer: mlp layer in transformer.
            drop_path: drop path rate.
            norm: normalization layer for conv blocks, 'instance' or 'layer' or 'group'.
            head_layer: class for prediction head.
        """
        super().__init__()

        self.grad_ckpt = False

        self.views = list(image_size_dict.keys())
        self.n_frames = n_frames

        # view specific encoding
        self.enc_down_dict = nn.ModuleDict(
            {
                view: DownsampleEncoder(
                    image_size=image_size_dict[view],
                    in_chans=n_frames * in_chans_dict[view],
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
        self.enc_fusion_dict = nn.ModuleDict(
            {
                view: MultiScaleFusion(
                    image_size=image_size_dict[view],
                    patch_size=enc_patch_size_dict[view],
                    scale_factor=enc_scale_factor_dict[view],
                    conv_chans=enc_conv_chans,
                    embed_dim=enc_embed_dim,
                    norm_layer=norm_layer,
                    norm_eps=norm_eps,
                )
                for view in self.views
            }
        )

        # shared vit encoder
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
        self.apply(init_weights)

        # prediction head
        self.pred_head_dict = nn.ModuleDict()
        if head_layer is not None:
            for view in [*self.views, "cls"]:
                self.pred_head_dict[view] = head_layer(enc_embed_dim, out_chans)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        for view in self.views:
            self.enc_down_dict[view].set_grad_ckpt(enable)
            self.enc_fusion_dict[view].set_grad_ckpt(enable)
        self.encoder.set_grad_ckpt(enable)
        for view in [*self.views, "cls"]:
            if view in self.pred_head_dict and hasattr(self.pred_head_dict[view], "set_grad_ckpt"):
                self.pred_head_dict[view].set_grad_ckpt(enable)

    def feature_forward(
        self,
        image_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            image_dict: dictionary of images, (batch, in_chans, ...).
            mask_dict: dictionary of masks, (batch, n_patches) at ViT grid size, None if no masking.
                mask is only used to mask out patches in the encoder, the output size is not affected.

        Returns:
            x_dict: dictionary of embeddings, (batch, 1, enc_emb_dim) for cls token,
                (batch, n_patches_view, enc_emb_dim) for each view.
        """
        views = list(image_dict.keys())
        if any(x not in self.views for x in views):
            raise ValueError(f"views {views} must be in self.input_keys {self.views}.")

        # patch embedding
        xs = []
        ns_patch = []
        skips_view = {}
        for view in views:
            # (batch, n_patches_view, enc_emb_dim)
            mask_view = mask_dict[view] if mask_dict is not None else None
            skip_view, x_view = self.enc_down_dict[view](image_dict[view], mask=mask_view)
            ns_patch.append(x_view.shape[1])
            skips_view[view] = skip_view
            xs.append(x_view)

        # encoder
        # (batch, 1+n_patches, enc_emb_dim)
        x = self.encoder(torch.cat(xs, dim=1))

        # fuse skipped features for each view
        xs = list(torch.split(x, [1, *ns_patch], dim=1))

        x_dict = dict(zip(["cls", *views], xs, strict=False))
        for view in views:
            # (batch, n_patches_view, enc_emb_dim)
            x_dict[view] = self.enc_fusion_dict[view](skips_view[view], x_dict[view], mask=None)

        return x_dict

    def forward(
        self,
        image_dict: dict[str, torch.Tensor],
        mask_dict: dict[str, torch.Tensor] | None = None,
        reduce: str = "all",
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            image_dict: dictionary of images, (batch, in_chans, ...).
            mask_dict: dictionary of masks, (batch, n_patches) at ViT grid size, None if no masking.
            reduce: reduction method, "patch" or "all" or "cls" or "none".
                patch: average over patches.
                all: average over all tokens including cls token.
                cls: use cls token only.

        Returns:
            logits, (batch, out_chans).
        """
        x_dict = self.feature_forward(image_dict=image_dict, mask_dict=mask_dict)
        if reduce == "patch":
            # (batch, n_views, out_chans)
            logits = torch.concat(
                [
                    # (batch, 1, out_chans)
                    self.pred_head_dict[view](x_dict[view].mean(dim=1, keepdim=True))
                    for view in self.views
                ],
                dim=1,
            )
            # (batch, out_chans)
            logits = logits.mean(dim=1)
            return logits
        if reduce == "all":
            # (batch, n_views+1, out_chans)
            logits = torch.concat(
                [
                    # (batch, 1, out_chans)
                    self.pred_head_dict[view](x_dict[view].mean(dim=1, keepdim=True))
                    for view in self.views
                ]
                + [self.pred_head_dict["cls"](x_dict["cls"])],
                dim=1,
            )
            # (batch, out_chans)
            logits = logits.mean(dim=1)
            return logits
        if reduce == "cls":
            logits = self.pred_head_dict["cls"](x_dict["cls"])[:, 0]
            return logits

        raise NotImplementedError(f"Unsupported reduce method {reduce}.")

    @classmethod
    def from_finetuned(  # type: ignore[no-untyped-def]
        cls,
        repo_id: str,
        model_filename: str,
        config_filename: str,
        **kwargs,
    ) -> ConvViT:
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
    def from_pretrained(cls, config: DictConfig, freeze: bool, **kwargs) -> ConvViT:  # type: ignore[no-untyped-def]
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


def load_pretrain_weights(  # noqa: C901
    model: nn.Module,
    views: str | list[str],
    ckpt_path: Path,
    freeze: bool,
) -> nn.Module:
    """Load weights from MAE.

    Args:
        model: model to load weights.
        views: one or more view name, sax, lax_2c, or lax_4c.
        ckpt_path: path to the checkpoint.
        freeze: freeze the loaded weights.

    Returns:
        model: loaded model.
    """
    if ckpt_path.suffix == ".pt":
        pretrained_state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
    elif ckpt_path.suffix == ".safetensors":
        pretrained_state_dict = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                pretrained_state_dict[key] = f.get_tensor(key)
    keys_to_drop = [
        "mask",
        "decoder",
        "_head",
        "sax",
        "lax_2c",
        "lax_3c",
        "lax_4c",
        "fusion",
        "dec_linear",
        "pos_embed",
    ]
    if hasattr(model, "enc_fusion_dict"):
        keys_to_drop.remove("fusion")
    views = [views] if isinstance(views, str) else views
    expected_missing_keys = []
    for view in views:
        keys_to_drop.remove(view)
        expected_missing_keys.append(f"enc_down_dict.{view}.pos_embed")
    state_dict = {}
    for k, v in pretrained_state_dict.items():
        # throw away not needed weights
        if any(x in k for x in keys_to_drop):
            continue
        skip = False
        for view in views:
            if k == f"enc_down_dict.{view}.conv_blocks.0.patch_embed.conv.weight":
                # for video classification model
                # weight in checkpoint has shape (embed_dim, in_chans, *kernel_size)
                # weight in model has shape (embed_dim, n_frames*in_chans, *kernel_size)
                # or for Myops2020 dataset, input has three modalities
                chans = model.enc_down_dict[view].conv_blocks[0].patch_embed.conv.weight.shape[1]
                if v.shape[1] != chans:
                    logger.info(f"Duplicate the weights for input channels {chans} vs {v.shape[1]}.")
                    if len(v.shape) == 5:
                        v_extended = v.repeat(1, chans, 1, 1, 1)
                    elif len(v.shape) == 4:
                        v_extended = v.repeat(1, chans, 1, 1)
                    else:
                        raise ValueError(f"Unsupported weight shape {v.shape}.")
                    state_dict[k] = v_extended
                    skip = True
                    break
        if skip:
            continue
        state_dict[k] = v

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    missing_keys = [
        x
        for x in incompatible_keys.missing_keys
        if ("decoder" not in x) and (not x.startswith("dec_")) and ("head" not in x)
    ]
    if set(missing_keys) != set(expected_missing_keys):
        raise ValueError(f"Missing keys from checkpoint: {missing_keys}, expected {expected_missing_keys}")
    if len(incompatible_keys.unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}")

    # freeze the weights for any loaded layers
    if freeze:
        logger.info("Freezing pretrained weights.")
        for name, param in model.named_parameters():
            if name in state_dict:
                param.requires_grad = False
    return model


def get_layer_id_for_vit(name: str, n_layers: int) -> int:
    """Assign a parameter with its layer id.

    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33

    Args:
        name: parameter name
        n_layers: number of layers

    Returns:
        layer_id: layer id, the first layer is 1.
    """
    if name.startswith("enc_"):
        # conv specific
        return 0
    if any(x in name for x in ["cls_token", "pos_embed", "patch_embed", "view_embed"]):
        # Example names:
        #   enc_view_embed
        #   encoder.cls_token
        #   patch_embed.proj.weight
        #   patch_embed.proj.bias
        return 0
    if name.startswith("encoder.blocks"):
        # Example names:
        #   encoder.blocks.0.attn.q.weight
        #   encoder.blocks.0.attn.kv.weight
        #   encoder.blocks.0.attn.proj.weight
        #   encoder.blocks.0.mlp.fc1.weight
        #   encoder.blocks.0.mlp.fc2.weight
        return int(name.split(".")[2]) + 1
    return n_layers


def param_groups_lr_decay(
    model: nn.Module,
    no_weight_decay_list: list[str],
    weight_decay: float,
    layer_decay: float,
    out_dir: Path | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    """Parameter groups for layer-wise lr decay.

    The code is specific to model architecture and should be modified accordingly.

    Following
    - https://github.com/facebookresearch/mae
    - https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58

    Args:
        model: model.
        no_weight_decay_list: list of parameter names without weight decay.
        weight_decay: weight decay.
        layer_decay: layer decay.
        out_dir: output directory to save param_group_names.json

    Returns:
        param_groups: list of parameter groups
    """
    param_group_names = {}  # for debugging
    param_groups = {}

    n_layers = len(model.encoder.blocks) + 1

    layer_scales = [layer_decay ** (n_layers - i) for i in range(n_layers + 1)]

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if p.ndim == 1 or n in no_weight_decay_list:
            # no decay for all 1D parameters and model specific ones
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, n_layers)
        group_name = f"layer_{layer_id}_{g_decay}"

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with Path.open(out_dir / "param_group_names.json", "w", encoding="utf-8") as f:
            json.dump(param_group_names, f, indent=2)
        logger.info(f"Saved param_group_names to {out_dir / 'param_group_names.json'}.")
    return list(param_groups.values())
