"""Test training and evaluation.

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

from cinema.classification.train import get_classification_or_regression_model
from cinema.log import get_logger
from cinema.regression.landmark.eval import landmark_regression_eval_dataset
from cinema.regression.landmark.train import (
    get_relative_distances,
    landmark_regression_eval_dataloader,
    landmark_regression_loss,
    run_train,
)
from cinema.segmentation.landmark.train import load_dataset

if TYPE_CHECKING:
    from pytest_mock import MockFixture

logger = get_logger(__name__)


def test_get_relative_distances() -> None:
    """Test get_relative_distances."""
    batch = 7
    coords = torch.rand(size=(batch, 6), dtype=torch.float32)
    got = get_relative_distances(coords)

    x1, y1, x2, y2, x3, y3 = coords.chunk(6, dim=1)
    dx1 = x1 - (x2 + x3) / 2
    dy1 = y1 - (y2 + y3) / 2
    dx2 = x2 - (x1 + x3) / 2
    dy2 = y2 - (y1 + y3) / 2
    dx3 = x3 - (x1 + x2) / 2
    dy3 = y3 - (y1 + y2) / 2
    expected = torch.concat([dx1, dy1, dx2, dy2, dx3, dy3], dim=1)
    assert torch.allclose(got, expected)


def get_test_config() -> DictConfig:
    """Get test config."""
    config_path = Path(__file__).parent.resolve() / "config.yaml"
    config = OmegaConf.load(config_path)
    config.train.batch_size = 4
    config.train.batch_size_per_device = 4
    config.train.n_epochs = 1
    config.train.n_warmup_epochs = 1
    config.train.eval_interval = 1
    config.data.max_n_samples = 4
    return config


@pytest.mark.integration
@pytest.mark.parametrize("model", ["resnet", "convvit"])
def test_reg_train_eval(mocker: MockFixture, model: str) -> None:
    """Test training.

    Args:
        model: model name.
        mocker: mocker, a wrapper of unittest.mock.
    """
    config = get_test_config()
    config.model.name = model
    if model == "resnet":
        config.model.resnet.depth = 10
    elif model == "convvit":
        config.model.convvit.size = "tiny"
    tfds_dir = Path(os.environ.get("CINEMA_DATA_DIR", "~/.cache/cinema_datasets"))
    tfds_dir = tfds_dir.expanduser()
    config.data.dir = tfds_dir / "landmark" / "processed"

    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        config.logging.dir = temp_dir

        run_train(
            config=config,
            load_dataset=load_dataset,
            get_model_fn=get_classification_or_regression_model,
            loss_fn=landmark_regression_loss,
            eval_dataloader_fn=landmark_regression_eval_dataloader,
        )

        run_dir = next(Path.iterdir(Path(temp_dir)))
        ckpt_path = Path(temp_dir) / run_dir / "ckpt_0.pt"
        assert ckpt_path.exists()

        landmark_regression_eval_dataset(
            config=config,
            split="test",
            ckpt_path=ckpt_path,
            out_dir=Path(temp_dir),
            save=True,
        )
        shutil.rmtree(temp_dir)
