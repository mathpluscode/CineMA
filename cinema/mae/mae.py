"""Cine Masked Autoencoder."""

from __future__ import annotations

import math

import torch
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from timm.layers import Mlp
from torch import nn

from cinema.conv import Linear
from cinema.convvit import DownsampleEncoder, MultiScaleFusion
from cinema.log import get_logger
from cinema.vit import (
    ViTDecoder,
    ViTEncoder,
    get_pos_embed,
    get_tokens,
    get_vit_config,
    init_weights,
    patchify,
)

logger = get_logger(__name__)


def get_batch_random_patch_mask(
    batch_size: int, n_patches: int, mask_ratio: float, device: torch.device
) -> torch.Tensor:
    """Get a per-sample random mask for a tensor of shape (batch, n_patches, emb_dim).

    Per-sample shuffling is done by argsort random noise.

    https://github.com/EPFL-VILAB/MultiMAE/blob/66910f5b5ba236f5e731883db85fe4f24ee01106/multimae/multimae.py#L164

    Args:
        batch_size: batch size.
        n_patches: number of patches in total.
        mask_ratio: ratio of patches to remove, in [0, 1]. For each sample in the batch, ratio is the same.
        device: device to store the results.

    Returns:
        mask: binary mask, (batch, n_patches), 0 is keep, 1 is remove.
    """
    if mask_ratio < 0:
        raise ValueError(f"mask_ratio must be positive, got {mask_ratio}.")
    if mask_ratio == 0:
        return torch.zeros((batch_size, n_patches), dtype=torch.bool, device=device)

    # sort noise for each sample
    noise = torch.rand(batch_size, n_patches, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep, 1 is remove
    n_keep = int(n_patches * (1 - mask_ratio))
    mask = torch.ones([batch_size, n_patches], device=device, dtype=torch.bool)
    mask[:, :n_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask


def add_pos_embed_and_append_mask_token(
    x_vis: torch.Tensor,
    enc_mask: torch.Tensor,
    dec_pos_embed: nn.Parameter,
    mask_token: nn.Parameter,
    concat: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Add mask tokens and position embeddings.

    Args:
        x_vis: visible tokens without class token, (batch, n_enc_keep, dec_emb_dim).
        enc_mask: binary mask, (batch, n_patches),
            0 is keep/visible to encoder, 1 is remove.
        dec_pos_embed: positional embedding for decoder, (n_patches, dec_emb_dim).
        mask_token: learnable mask token, (1, 1, dec_emb_dim).
        concat: whether to concatenate mask tokens to the sequence.

    Returns:
        if concat:
            x: tensor with mask tokens and position embeddings, (batch, 1+n_patches, dec_emb_dim).
        else:
            x_vis: visible tokens with position embeddings, (batch, n_enc_keep, dec_emb_dim).
            x_mask: mask tokens with position embeddings, (batch, n_enc_masked, dec_emb_dim).
    """
    batch, n_enc_keep, dec_emb_dim = x_vis.shape
    _, n_patches = enc_mask.shape
    n_enc_masked = n_patches - n_enc_keep

    # shuffle the pos embedding
    dec_pe = dec_pos_embed.expand(batch, -1, -1).contiguous()  # (batch, n_patches, dec_emb_dim)
    vis_pe = dec_pe[~enc_mask].reshape(batch, n_enc_keep, dec_emb_dim)  # (batch, n_enc_keep, dec_emb_dim)
    mask_pe = dec_pe[enc_mask].reshape(batch, n_enc_masked, dec_emb_dim)  # (batch, n_enc_masked, dec_emb_dim)

    # append mask tokens to sequence
    if concat:
        return torch.cat([x_vis + vis_pe, mask_token + mask_pe], dim=1)  # (batch, n_patches, dec_emb_dim)
    return x_vis + vis_pe, mask_token + mask_pe


def mse_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    enc_mask: torch.Tensor,
    norm_target: bool,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward pass of the loss, only calculated on masked patches.

    Args:
        target: target patches, (batch, n_patches, out_chans).
        pred: predicted patches, (batch, n_enc_masked, out_chans).
        enc_mask: binary mask, (batch, n_patches),
            0 is keep/visible to encoder, 1 is to be predicted.
        norm_target: whether to normalize target values for loss.
        epsilon: small value to avoid division by zero.

    Returns:
        loss: MSE loss on masked patches.
        metrics: metrics.
    """
    metrics: dict[str, torch.Tensor] = {}
    mean = target.mean(dim=-1, keepdim=True)  # (batch, n_patches, 1)
    var = target.var(dim=-1, keepdim=True)
    std = var**0.5
    metrics.update(
        {
            "target_mean": mean.mean(),
            "target_std": std.mean(),
        }
    )
    if norm_target:
        target = (target - mean) / (std + epsilon)
    target = target[enc_mask].reshape(pred.shape)  # (batch, n_enc_masked, out_chans)

    loss = nn.MSELoss(reduction="none")(pred, target.detach())  # squared error, (batch, n_enc_masked, out_chans)
    loss = loss.mean()  # scalar
    metrics["mse_loss"] = loss

    if norm_target and target.shape[1] > 0:
        # when normalizing target
        # pred_max is a good indicator of whether the model is learning
        metrics["normed_target_max"] = target.max()
        metrics["pred_max"] = pred.max()

    return loss, metrics


class DecoderEmbedding(nn.Module):
    """Decoder embedding module."""

    def __init__(
        self,
        enc_grid_size: tuple[int, ...],
        dec_embed_dim: int,
        add_embed_token: bool,
    ) -> None:
        """Initialize the module.

        Args:
            enc_grid_size: grid size of encoder.
            dec_embed_dim: number of embedding channels for decoder.
            add_embed_token: whether to add an embedding to all tokens.
        """
        super().__init__()
        self.pos_embed = get_pos_embed(
            embed_dim=dec_embed_dim,
            grid_size=enc_grid_size,
        )
        self.embed_token = get_tokens(embed_dim=dec_embed_dim, n_tokens=1) if add_embed_token else None
        self.mask_token = get_tokens(embed_dim=dec_embed_dim, n_tokens=1)

    def forward(
        self,
        x: torch.Tensor,
        enc_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (batch, 1+n_enc_keep, dec_emb_dim).
            enc_mask: (batch, n_patches), 0 is keep/visible to encoder, 1 is remove.

        Returns:
            x_vis: visible tokens with position embeddings, (batch, n_enc_keep, dec_emb_dim).
            x_mask: mask tokens with position embeddings, (batch, n_enc_masked, dec_emb_dim).
        """
        x_vis, x_mask = add_pos_embed_and_append_mask_token(
            x_vis=x,
            enc_mask=enc_mask,
            dec_pos_embed=self.pos_embed,
            mask_token=self.mask_token,
            concat=False,
        )
        if self.embed_token is not None:
            x_vis = x_vis + self.embed_token
            x_mask = x_mask + self.embed_token
        return x_vis, x_mask


def get_decoder_patch_size(
    image_size: tuple[int, ...],
    n_conv_layers: int,
    enc_patch_size: tuple[int, ...],
    enc_scale_factor: tuple[int, ...],
) -> tuple[int, ...]:
    """Get decoder patch size based on encoder settings.

    Args:
        image_size: input image size.
        n_conv_layers: number of conv layers in encoder.
        enc_patch_size: patch size for the first layer.
        enc_scale_factor: scale factor for other layers.

    Returns:
        dec_patch_size: patch size for the top layer in decoder.
    """
    dec_patch_size = (1,) * len(image_size)
    for i in range(1 + n_conv_layers):
        patch_size = enc_patch_size if i == 0 else enc_scale_factor
        dec_patch_size = tuple(s * p for s, p in zip(dec_patch_size, patch_size, strict=False))
    return dec_patch_size


def get_model(
    config: DictConfig,
) -> CineMA:
    """Get model from config.

    Args:
        config: config.

    Returns:
        MAE model.
    """
    image_size_dict = {
        "sax": config.data.sax.patch_size,
        "lax_2c": config.data.lax.patch_size,
        "lax_3c": config.data.lax.patch_size,
        "lax_4c": config.data.lax.patch_size,
    }
    in_chans_dict = {
        "sax": config.data.sax.in_chans,
        "lax_2c": config.data.lax.in_chans,
        "lax_3c": config.data.lax.in_chans,
        "lax_4c": config.data.lax.in_chans,
    }
    patch_size_dict = {
        "sax": config.model.patch_size,
        "lax_2c": config.model.patch_size[:2],
        "lax_3c": config.model.patch_size[:2],
        "lax_4c": config.model.patch_size[:2],
    }
    scale_factor_dict = {
        "sax": config.model.scale_factor,
        "lax_2c": config.model.scale_factor[:2],
        "lax_3c": config.model.scale_factor[:2],
        "lax_4c": config.model.scale_factor[:2],
    }
    vit_config = get_vit_config(config.model.size)
    model = CineMA(
        image_size_dict=image_size_dict,
        in_chans_dict=in_chans_dict,
        enc_patch_size_dict=patch_size_dict,
        enc_scale_factor_dict=scale_factor_dict,
        enc_conv_chans=config.model.enc_conv_chans,
        enc_conv_n_blocks=config.model.enc_conv_n_blocks,
        enc_embed_dim=vit_config["enc_embed_dim"],
        enc_depth=vit_config["enc_depth"],
        enc_n_heads=vit_config["enc_n_heads"],
        dec_embed_dim=vit_config["dec_embed_dim"],
        dec_depth=vit_config["dec_depth"],
        dec_n_heads=vit_config["dec_n_heads"],
    )
    model.set_grad_ckpt(config.grad_ckpt)
    return model


class CineMA(nn.Module):
    """Cine masked autoencoder."""

    def __init__(
        self,
        image_size_dict: dict[str, tuple[int, ...]],
        in_chans_dict: dict[str, int],
        enc_patch_size_dict: dict[str, tuple[int, ...]],
        enc_scale_factor_dict: dict[str, tuple[int, ...]],
        enc_conv_chans: list[int],
        enc_conv_n_blocks: int,
        enc_embed_dim: int,
        enc_depth: int,
        enc_n_heads: int,
        dec_embed_dim: int,
        dec_depth: int,
        dec_n_heads: int,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        norm_target: bool = False,
        cross_attn: bool = True,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_eps: float = 1e-5,
        rotary: bool = False,
        act_layer: nn.Module = nn.GELU,
        mlp_layer: nn.Module = Mlp,
        drop_path: float = 0.0,
        norm: str = "layer",
    ) -> None:
        """Initialize the module.

        Args:
            image_size_dict: input image size per view.
            in_chans_dict: number of input channels per view.
            enc_patch_size_dict: patch size for the first layer per view.
            enc_scale_factor_dict: scale factor for other layers per view.
            enc_conv_chans: number of channels for each conv layer, if empty, no conv layers.
            enc_conv_n_blocks: number of MaskedConvBlock for each enc_conv_block.
            enc_embed_dim: number of embedding channels for ViT encoder.
            enc_depth: number of layers for ViT encoder.
            enc_n_heads: number of heads for ViT encoder.
            dec_embed_dim: number of embedding channels for ViT decoder.
            dec_depth: number of layers for ViT decoder.
            dec_n_heads: number of heads for ViT decoder.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: whether to include bias in the qkv projection layer.
            norm_layer: normalization layer.
            norm_eps: epsilon for normalization layer.
            rotary: whether to use rotary position embeddings.
            act_layer: activation layer.
            mlp_layer: mlp layer in transformer.
            norm_target: whether to normalize target values for loss.
            cross_attn: whether to use cross attention.
            drop_path: drop path rate, only used for fine-tuning.
            norm: normalization layer, either 'instance' or 'layer'.
        """
        super().__init__()

        self.grad_ckpt = False

        self.norm_target = norm_target
        self.views = list(image_size_dict.keys())

        # view specific encoding
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

        # shared decoder embedding
        self.dec_linear = Linear(enc_embed_dim, dec_embed_dim)

        # view specific decoder embedding
        self.dec_embed_dict = nn.ModuleDict(
            {
                view: DecoderEmbedding(
                    enc_grid_size=self.enc_down_dict[view].patch_embed.grid_size,
                    dec_embed_dim=dec_embed_dim,
                    add_embed_token=False,
                )
                for view in self.views
            }
        )

        # shared decoder
        self.cross_attn = cross_attn
        self.decoder = ViTDecoder(
            embed_dim=dec_embed_dim,
            depth=dec_depth,
            n_heads=dec_n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            norm_eps=norm_eps,
            rotary=rotary,
            act_layer=act_layer,
            mlp_layer=mlp_layer,
            drop_path=drop_path,
        )

        # prediction head
        self.dec_patch_size_dict = {
            view: get_decoder_patch_size(
                image_size=image_size_dict[view],
                n_conv_layers=len(enc_conv_chans),
                enc_patch_size=enc_patch_size_dict[view],
                enc_scale_factor=enc_scale_factor_dict[view],
            )
            for view in self.views
        }
        self.pred_head_dict = nn.ModuleDict(
            {
                view: Linear(dec_embed_dim, math.prod(patch_size) * in_chans_dict[view])
                for view, patch_size in self.dec_patch_size_dict.items()
            }
        )

        self.apply(init_weights)

    @torch.jit.ignore
    def set_grad_ckpt(self, enable: bool = True) -> None:
        """Set gradient checkpointing."""
        self.grad_ckpt = enable
        for view in self.views:
            self.enc_down_dict[view].set_grad_ckpt(enable)
            self.enc_fusion_dict[view].set_grad_ckpt(enable)
        self.encoder.set_grad_ckpt(enable)
        self.dec_linear.set_grad_ckpt(enable)
        self.decoder.set_grad_ckpt(enable)
        for view in self.views:
            self.pred_head_dict[view].set_grad_ckpt(enable)

    def feature_forward(  # pylint:disable=too-many-statements
        self,
        image_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass for feature extraction only.

        Args:
            image_dict: dictionary of images, (batch, in_chans, ...).

        Returns:
            dict of features, (batch, n_patches, enc_emb_dim).
        """
        views = list(image_dict.keys())
        if any(x not in self.views for x in views):
            raise ValueError(f"views {views} must be in self.input_keys {self.views}.")
        batch_size = image_dict[views[0]].shape[0]

        # sample mask at ViT input resolution for each view
        # each mask is (batch_size, n_patches_view)
        ns_patches = [self.enc_down_dict[view].patch_embed.n_patches for view in views]

        # patch embedding and masking
        xs = []
        ns_enc_keep = []
        ns_enc_masked = []
        skips_view = []
        for i, view in enumerate(views):
            # (batch, n_patches_view, enc_emb_dim)
            skip_view, x_view = self.enc_down_dict[view](image_dict[view], mask=None)
            x_view = x_view.reshape(batch_size, -1, x_view.shape[-1])
            skips_view.append(skip_view)
            ns_enc_keep.append(x_view.shape[1])
            ns_enc_masked.append(ns_patches[i] - x_view.shape[1])
            xs.append(x_view)

        # encoder
        # (batch, 1+n_enc_keep, enc_emb_dim)
        x = self.encoder(torch.cat(xs, dim=1))

        # fuse skipped features for each view
        xs = list(torch.split(x, [1, *ns_enc_keep], dim=1))
        for i, view in enumerate(views):
            xs[i + 1] = self.enc_fusion_dict[view](skips_view[i], xs[i + 1], mask=None)

        views = ["cls", *views]
        return dict(zip(views, xs, strict=True))

    def forward(  # pylint:disable=too-many-statements
        self,
        image_dict: dict[str, torch.Tensor],
        enc_mask_ratio: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            image_dict: dictionary of images, (batch, in_chans, ...).
            enc_mask_ratio: masking ratio for encoder.

        Returns:
            loss: MSE loss on masked patches.
            pred_dict: dict of predicted patches, (batch, n_patches, out_chains).
            enc_mask_dict: dict of binary mask, (batch, n_patches), 0 is keep/visible to encoder, 1 is remove.
            metrics: metrics, each value is a scalar tensor.
        """
        views = list(image_dict.keys())
        if any(x not in self.views for x in views):
            raise ValueError(f"views {views} must be in self.input_keys {self.views}.")
        batch_size = image_dict[views[0]].shape[0]
        device = image_dict[views[0]].device
        metrics = {}

        # sample mask at ViT input resolution for each view
        # each mask is (batch_size, n_patches_view)
        ns_patches = [self.enc_down_dict[view].patch_embed.n_patches for view in views]
        enc_masks = [
            get_batch_random_patch_mask(
                batch_size=batch_size,
                n_patches=n_patches,
                mask_ratio=enc_mask_ratio,
                device=device,
            )
            for n_patches in ns_patches
        ]

        # patch embedding and masking
        xs = []
        ns_enc_keep = []
        ns_enc_masked = []
        skips_view = []
        for i, view in enumerate(views):
            # (batch, n_patches_view, enc_emb_dim)
            skip_view, x_view = self.enc_down_dict[view](image_dict[view], mask=enc_masks[i])
            # (batch, n_patches_view_keep, enc_emb_dim)
            x_view = x_view[~enc_masks[i]].reshape(batch_size, -1, x_view.shape[-1])
            skips_view.append(skip_view)
            ns_enc_keep.append(x_view.shape[1])
            ns_enc_masked.append(ns_patches[i] - x_view.shape[1])
            xs.append(x_view)

        # encoder
        # (batch, 1+n_enc_keep, enc_emb_dim)
        x = self.encoder(torch.cat(xs, dim=1))

        # fuse skipped features for each view
        xs = list(torch.split(x, [1, *ns_enc_keep], dim=1))
        for i, view in enumerate(views):
            xs[i + 1] = self.enc_fusion_dict[view](skips_view[i], xs[i + 1], enc_masks[i])

        # project to decoder space
        # (batch, 1+n_enc_keep, dec_emb_dim)
        x = self.dec_linear(torch.cat(xs, dim=1))

        # split x into x_cls and x_view chunks and append mask tokens
        xs = torch.split(x, [1, *ns_enc_keep], dim=1)  # include cls token
        xs_vis, xs_mask = [], []
        for i, view in enumerate(views):
            x_vis_view, x_mask_view = self.dec_embed_dict[view](x=xs[i + 1], enc_mask=enc_masks[i])
            xs_vis.append(x_vis_view)
            xs_mask.append(x_mask_view)

        # decoder
        # (batch, n_enc_masked, dec_emb_dim)
        if self.cross_attn:
            x_q = torch.cat([xs[0], *xs_mask], dim=1)
            x_view = torch.cat(xs_vis, dim=1)
            x = self.decoder(x_q, x_view, sum(ns_enc_masked))  # without cls token
        else:
            x = torch.cat([xs[0], *xs_vis, *xs_mask], dim=1)
            x = self.decoder(x, None, sum(ns_enc_masked))  # without cls token
        # (batch, n_enc_masked_view, dec_emb_dim) for each element
        xs = torch.split(x, ns_enc_masked, dim=1)

        # loss
        preds = []
        losses = []
        for i, view in enumerate(views):
            # (batch, n_enc_masked_view, out_chans)
            pred_view = self.pred_head_dict[view](xs[i])
            preds.append(pred_view)
            loss_view, metrics_view = mse_loss(
                target=patchify(image=image_dict[view], patch_size=self.dec_patch_size_dict[view]),
                pred=pred_view,
                enc_mask=enc_masks[i],
                norm_target=self.norm_target,
            )
            metrics_view = {f"{view}_{m}": v for m, v in metrics_view.items()}
            metrics.update(metrics_view)
            if torch.isfinite(loss_view):
                # loss maybe nan if certain view is not masked at all
                losses.append(loss_view)

        loss = sum(losses) / len(losses) if len(losses) > 0 else torch.tensor(float("nan"), device=device)
        metrics["loss"] = loss
        pred_dict = dict(zip(views, preds, strict=False))
        enc_mask_dict = dict(zip(views, enc_masks, strict=False))
        return loss, pred_dict, enc_mask_dict, metrics

    @classmethod
    def from_pretrained(cls, **kwargs) -> CineMA:  # type: ignore[no-untyped-def]
        """Load pretrained weights."""
        # download weights
        model_path = hf_hub_download(
            repo_id="mathpluscode/CineMA",
            filename="pretrained/cinema.safetensors",
            **kwargs,
        )
        logger.info(f"Cached model weights to {model_path}.")
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                state_dict[key] = f.get_tensor(key)

        # download config
        config_path = hf_hub_download(
            repo_id="mathpluscode/CineMA",
            filename="pretrained/cinema.yaml",
            **kwargs,
        )
        logger.info(f"Cached model config to {config_path}.")
        config = OmegaConf.load(config_path)

        # init model and load state dict
        model = get_model(config)
        model.load_state_dict(state_dict)
        logger.info("Loaded pretrained weights.")
        return model
