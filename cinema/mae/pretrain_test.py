"""Test training.

mocker.patch, https://pytest-mock.readthedocs.io/en/latest/
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cinema.convvit import ConvViT, load_pretrain_weights
from cinema.device import get_free_port
from cinema.mae.mae import get_model
from cinema.mae.pretrain import pretrain
from cinema.segmentation.convunetr import ConvUNetR
from cinema.vit import get_vit_config

if TYPE_CHECKING:
    from pytest_mock import MockFixture


def get_test_config() -> DictConfig:
    """Get test config."""
    config_path = Path(__file__).parent.resolve() / "config.yaml"
    config = OmegaConf.load(config_path)
    config.train.batch_size = 2
    config.train.batch_size_per_device = 1
    config.train.n_epochs = 1
    config.train.n_warmup_epochs = 1
    config.data.max_n_samples = 2
    config.model.size = "tiny"
    config.model.enc_conv_chans = [2, 4]
    config.model.enc_conv_n_blocks = 1
    return config


@pytest.mark.parametrize(
    "views",
    [
        "sax",
        "lax_2c",
        "lax_4c",
        ["sax", "lax_2c", "lax_3c", "lax_4c"],
        ["lax_2c", "lax_4c"],
    ],
)
def test_get_model(views: str | list[str]) -> None:
    """Test pretraining with one time frame."""
    config = get_test_config()
    if isinstance(views, str):
        views = [views]
    config.model.views = views

    get_model(config=config)


class TestLoadMaeWeights:
    """Test MAE can be loaded into ConvViT and ConvUNetR."""

    @pytest.mark.parametrize(
        "views",
        [
            "sax",
            "lax_2c",
            "lax_4c",
            ["sax", "lax_2c", "lax_3c", "lax_4c"],
            ["lax_2c", "lax_4c"],
        ],
    )
    @pytest.mark.parametrize("n_frames", [1, 2])
    def test_vit(self, views: str | list[str], n_frames: int) -> None:
        """Ensure MAE weight can be loaded into ViT."""
        config = get_test_config()
        if isinstance(views, str):
            views = [views]
        config.model.views = views

        mae = get_model(config=config)
        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / "ckpt.pt"
            to_save = {
                "model": mae.state_dict(),
            }
            torch.save(to_save, ckpt_path)
            for view in views:
                view_key = "sax" if view == "sax" else "lax"
                ndim = len(config.data[view_key].patch_size)
                vit_config = get_vit_config(config.model.size)
                vit = ConvViT(
                    image_size_dict={view: config.data[view_key].patch_size},
                    in_chans_dict={view: config.data[view_key].in_chans},
                    n_frames=n_frames,
                    out_chans=2,
                    enc_patch_size_dict={view: config.model.patch_size[:ndim]},
                    enc_scale_factor_dict={view: config.model.scale_factor[:ndim]},
                    enc_conv_chans=config.model.enc_conv_chans,
                    enc_conv_n_blocks=config.model.enc_conv_n_blocks,
                    enc_embed_dim=vit_config["enc_embed_dim"],
                    enc_depth=vit_config["enc_depth"],
                    enc_n_heads=vit_config["enc_n_heads"],
                    drop_path=0.1,
                )

                load_pretrain_weights(vit, view, ckpt_path, freeze=True)

    @pytest.mark.parametrize(
        "views",
        [
            "sax",
            "lax_2c",
            "lax_4c",
            ["sax", "lax_2c", "lax_3c", "lax_4c"],
            ["lax_2c", "lax_4c"],
        ],
    )
    def test_convunetr(self, views: str | list[str]) -> None:
        """Ensure MAE weight can be loaded into ConvUNetR."""
        config = get_test_config()
        if isinstance(views, str):
            views = [views]
        config.model.views = views

        mae = get_model(config=config)
        with TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / "ckpt.pt"
            to_save = {
                "model": mae.state_dict(),
            }
            torch.save(to_save, ckpt_path)

            # load per view
            for view in views:
                view_key = "sax" if view == "sax" else "lax"
                ndim = len(config.data[view_key].patch_size)
                vit_config = get_vit_config(config.model.size)
                convunetr = ConvUNetR(
                    image_size_dict={view: config.data[view_key].patch_size},
                    in_chans_dict={view: config.data[view_key].in_chans},
                    out_chans=3,
                    enc_patch_size_dict={view: config.model.patch_size[:ndim]},
                    enc_scale_factor_dict={view: config.model.scale_factor[:ndim]},
                    enc_conv_chans=config.model.enc_conv_chans,
                    enc_conv_n_blocks=config.model.enc_conv_n_blocks,
                    enc_embed_dim=vit_config["enc_embed_dim"],
                    enc_depth=vit_config["enc_depth"],
                    enc_n_heads=vit_config["enc_n_heads"],
                    dec_chans=(32, 64, 128, 256, 512),
                    dec_patch_size_dict={view: (2, 2, 1)[:ndim]},
                    dec_scale_factor_dict={view: (2, 2, 1)[:ndim]},
                )

                load_pretrain_weights(convunetr, view, ckpt_path, freeze=True)

            # load all views
            if len(views) > 1:
                image_size_dict = {
                    v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views
                }
                in_chans_dict = {v: config.data.sax.in_chans if v == "sax" else config.data.lax.in_chans for v in views}
                ndim_dict = {v: 3 if v == "sax" else 2 for v in views}
                enc_patch_size_dict = {v: config.model.patch_size[: ndim_dict[v]] for v in views}
                enc_scale_factor_dict = {v: config.model.scale_factor[: ndim_dict[v]] for v in views}
                dec_patch_size_dict = {v: (2, 2, 1)[: ndim_dict[v]] for v in views}
                dec_scale_factor_dict = {v: (2, 2, 1)[: ndim_dict[v]] for v in views}
                convunetr = ConvUNetR(
                    image_size_dict=image_size_dict,
                    in_chans_dict=in_chans_dict,
                    out_chans=3,
                    enc_patch_size_dict=enc_patch_size_dict,
                    enc_scale_factor_dict=enc_scale_factor_dict,
                    enc_conv_chans=config.model.enc_conv_chans,
                    enc_conv_n_blocks=config.model.enc_conv_n_blocks,
                    enc_embed_dim=vit_config["enc_embed_dim"],
                    enc_depth=vit_config["enc_depth"],
                    enc_n_heads=vit_config["enc_n_heads"],
                    dec_chans=(32, 64, 128, 256, 512),
                    dec_patch_size_dict=dec_patch_size_dict,
                    dec_scale_factor_dict=dec_scale_factor_dict,
                )

                load_pretrain_weights(convunetr, views, ckpt_path, freeze=True)


@pytest.mark.integration
@pytest.mark.parametrize(
    "views",
    [
        "sax",
        "lax_2c",
        ["sax", "lax_2c", "lax_3c", "lax_4c"],
    ],
)
def test_pretrain_image_mae(mocker: MockFixture, views: str | list[str]) -> None:
    """Test pretraining with one time frame."""
    port = get_free_port()
    config = get_test_config()
    if isinstance(views, str):
        views = [views]
    config.model.views = views
    cache_dir = Path(os.environ.get("CINEMA_DATA_DIR", "~/.cache/cinema_datasets")).expanduser()
    config.data.dir = cache_dir / "ukb" / "processed"

    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        config.logging.dir = temp_dir
        pretrain(rank=0, world_size=1, port=port, config=config)
        shutil.rmtree(temp_dir)
