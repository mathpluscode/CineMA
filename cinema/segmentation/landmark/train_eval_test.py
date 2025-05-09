"""Test training and evaluation.

mocker.patch, https://pytest-mock.readthedocs.io/en/latest/
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from cinema.log import get_logger
from cinema.metric import heatmap_argmax, heatmap_soft_argmax
from cinema.segmentation.landmark.dataset import create_circle_2d
from cinema.segmentation.landmark.eval import landmark_detection_eval_dataset
from cinema.segmentation.landmark.train import (
    landmark_detection_eval_dataloader,
    landmark_detection_loss,
    load_dataset,
)
from cinema.segmentation.train import get_segmentation_model
from cinema.train import run_train

if TYPE_CHECKING:
    from pytest_mock import MockFixture

logger = get_logger(__name__)


def test_heatmap_argmax() -> None:
    """Test heatmap_argmax and heatmap_soft_argmax.

    Sample coordinates and generate heatmap, then extract coordinates from heatmap.
    """
    batch_size = 4
    image_size = (7, 8)

    # generate heatmap
    rng = np.random.default_rng()
    coords = rng.random((batch_size, 6))  # x1, y1, x2, y2, x3, y3
    coords *= np.array(list(image_size) * 3)[None, :]
    coords = coords.astype(int)
    heatmap = torch.Tensor(
        np.stack(
            [
                np.stack(
                    [
                        create_circle_2d(c[:2], image_size),
                        create_circle_2d(c[2:4], image_size),
                        create_circle_2d(c[4:], image_size),
                    ]
                )
                for c in coords
            ]
        )
    )

    # extract coords
    coords_got = heatmap_argmax(heatmap)
    assert np.allclose(coords, coords_got)

    coords_got = heatmap_soft_argmax(heatmap)
    assert np.allclose(coords, coords_got)


def get_test_config() -> DictConfig:
    """Get test config."""
    config_path = Path(__file__).parent.resolve() / "config.yaml"
    config = OmegaConf.load(config_path)
    config.train.batch_size = 2
    config.train.batch_size_per_device = 1
    config.train.n_epochs = 1
    config.train.n_warmup_epochs = 1
    config.train.eval_interval = 1
    config.data.max_n_samples = 2
    return config


@pytest.mark.integration
@pytest.mark.parametrize("model", ["unet", "convunetr"])
def test_seg_train_eval(mocker: MockFixture, model: str) -> None:
    """Test training.

    Args:
        model: model name.
        mocker: mocker, a wrapper of unittest.mock.
    """
    config = get_test_config()
    config.model.name = model
    if model == "unet":
        config.model.unet.chans = [2, 2, 4, 4]
        config.model.unet.patch_size = [4, 4, 1]
    elif model == "convunetr":
        config.model.convunetr.enc_conv_chans = [2, 4]
        config.model.convunetr.dec_chans = [2, 2, 4, 4]
        config.model.convunetr.size = "tiny"
    tfds_dir = Path(os.environ.get("CINEMA_DATA_DIR", "~/.cache/cinema_datasets")).expanduser()
    config.data.dir = tfds_dir / "landmark" / "processed"

    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        config.logging.dir = temp_dir

        run_train(
            config=config,
            load_dataset=load_dataset,
            get_model_fn=get_segmentation_model,
            loss_fn=landmark_detection_loss,
            eval_dataloader_fn=landmark_detection_eval_dataloader,
        )

        run_dir = next(Path.iterdir(Path(temp_dir)))
        ckpt_path = Path(temp_dir) / run_dir / "ckpt_0.pt"
        assert ckpt_path.exists()

        landmark_detection_eval_dataset(
            config=config,
            ckpt_path=ckpt_path,
            out_dir=Path(temp_dir),
            split="test",
            save=True,
        )
        shutil.rmtree(temp_dir)
