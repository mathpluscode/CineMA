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

from cinema.classification.eval import classification_eval_dataset
from cinema.classification.mnms2.train import load_mnms2_dataset, run_train
from cinema.classification.train import (
    classification_eval_dataloader,
    classification_loss,
    get_classification_or_regression_model,
)
from cinema.log import get_logger

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
    config.data.max_n_samples = 6
    return config


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "class_column"),
    [
        ("resnet", "pathology"),
        ("convvit", "pathology"),
        ("resnet", "sex"),
    ],
)
@pytest.mark.parametrize("views", ["sax", "lax_4c", ["sax", "lax_4c"]])
def test_clf_train_eval(mocker: MockFixture, model: str, views: str | list[str], class_column: str) -> None:
    """Test training.

    Args:
        model: model name.
        mocker: mocker, a wrapper of unittest.mock.
        views: sax or lax_4c or both.
        class_column: class column name.
    """
    config = get_test_config()
    config.model.views = views
    config.data.class_column = class_column
    config.model.name = model
    if views != "lax_4c" and class_column != "pathology":
        # skip other classification tasks on non lax_4c views
        return
    if model == "resnet":
        if len(views) > 1:
            return  # skip multi-view for resnet
        config.model.resnet.depth = 10
    elif model == "convvit":
        config.model.convvit.size = "tiny"
    cache_dir = Path(os.environ.get("CINEMA_DATA_DIR", "~/.cache/cinema_datasets")).expanduser()
    config.data.dir = cache_dir / "mnms2" / "processed"

    with TemporaryDirectory() as temp_dir:
        mocker.resetall()
        mocker.patch.dict("os.environ", {"WANDB_MODE": "offline"})
        config.logging.dir = temp_dir

        run_train(
            config=config,
            load_dataset=load_mnms2_dataset,
            get_model_fn=get_classification_or_regression_model,
            loss_fn=classification_loss,
            eval_dataloader_fn=classification_eval_dataloader,
        )

        run_dir = next(Path.iterdir(Path(temp_dir)))
        ckpt_path = Path(temp_dir) / run_dir / "ckpt_0.pt"
        assert ckpt_path.exists()

        classification_eval_dataset(
            config=config,
            split="test",
            ckpt_path=ckpt_path,
            out_dir=Path(temp_dir),
        )
        shutil.rmtree(temp_dir)
