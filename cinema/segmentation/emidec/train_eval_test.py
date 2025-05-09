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
from omegaconf import DictConfig, OmegaConf

from cinema.log import get_logger
from cinema.segmentation.emidec.eval import segmentation_eval_emidec_dataset
from cinema.segmentation.emidec.train import (
    emidec_segmentation_eval_dataloader,
    load_dataset,
)
from cinema.segmentation.train import (
    get_segmentation_model,
    segmentation_loss,
)
from cinema.train import run_train

if TYPE_CHECKING:
    from pytest_mock import MockFixture

logger = get_logger(__name__)


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
    cache_dir = Path(os.environ.get("CINEMA_DATA_DIR", "~/.cache/cinema_datasets")).expanduser()
    config.data.dir = cache_dir / "emidec" / "processed"

    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        config.logging.dir = temp_dir

        run_train(
            config=config,
            load_dataset=load_dataset,
            get_model_fn=get_segmentation_model,
            loss_fn=segmentation_loss,
            eval_dataloader_fn=emidec_segmentation_eval_dataloader,
        )

        run_dir = next(Path.iterdir(Path(temp_dir)))
        ckpt_path = Path(temp_dir) / run_dir / "ckpt_0.pt"
        assert ckpt_path.exists()

        segmentation_eval_emidec_dataset(
            config=config,
            split="test",
            ckpt_path=ckpt_path,
            out_dir=Path(temp_dir),
            save=True,
        )
        shutil.rmtree(temp_dir)
