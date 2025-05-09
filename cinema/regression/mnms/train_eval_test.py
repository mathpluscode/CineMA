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

from cinema.classification.train import get_classification_or_regression_model
from cinema.log import get_logger
from cinema.regression.eval import regression_eval_dataset
from cinema.regression.mnms.train import load_mnms_dataset, run_train
from cinema.regression.train import regression_eval_dataloader, regression_loss

if TYPE_CHECKING:
    from pytest_mock import MockFixture

logger = get_logger(__name__)


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
@pytest.mark.parametrize(
    ("model", "regression_column"),
    [
        ("resnet", "ef"),
        ("convvit", "age"),
        ("resnet", "age"),
    ],
)
def test_reg_train_eval(mocker: MockFixture, model: str, regression_column: str) -> None:
    """Test training.

    Args:
        model: model name.
        mocker: mocker, a wrapper of unittest.mock.
        regression_column: regression column name.
    """
    config = get_test_config()
    config.data.regression_column = regression_column
    config.model.name = model
    if model == "resnet":
        config.model.resnet.depth = 10
    elif model == "convvit":
        config.model.convvit.size = "tiny"
    cache_dir = Path(os.environ.get("CINEMA_DATA_DIR", "~/.cache/cinema_datasets")).expanduser()
    config.data.dir = cache_dir / "mnms" / "processed"

    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        config.logging.dir = temp_dir

        run_train(
            config=config,
            load_dataset=load_mnms_dataset,
            get_model_fn=get_classification_or_regression_model,
            loss_fn=regression_loss,
            eval_dataloader_fn=regression_eval_dataloader,
        )

        run_dir = next(Path.iterdir(Path(temp_dir)))
        ckpt_path = Path(temp_dir) / run_dir / "ckpt_0.pt"
        assert ckpt_path.exists()

        regression_eval_dataset(
            config=config,
            split="test",
            ckpt_path=ckpt_path,
            out_dir=Path(temp_dir),
        )
        shutil.rmtree(temp_dir)
