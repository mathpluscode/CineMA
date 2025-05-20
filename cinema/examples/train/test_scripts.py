"""Test all scripts."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from omegaconf import OmegaConf

from cinema.examples.train.classification import run as run_clf
from cinema.examples.train.pretrain import run as run_pretrain
from cinema.examples.train.regression import run as run_reg
from cinema.examples.train.segmentation import run as run_seg


@pytest.mark.integration
def test_classification() -> None:
    """Execute script."""
    config_path = Path(__file__).parent.resolve() / "classification.yaml"
    config = OmegaConf.load(config_path)
    config.train.batch_size = 2
    config.train.n_epochs = 1
    config.train.n_warmup_epochs = 1
    config.train.eval_interval = 1
    config.data.max_n_samples = 2
    with TemporaryDirectory() as temp_dir:
        config.logging.dir = temp_dir
        run_clf(config)


@pytest.mark.integration
def test_regression() -> None:
    """Execute script."""
    config_path = Path(__file__).parent.resolve() / "regression.yaml"
    config = OmegaConf.load(config_path)
    config.train.batch_size = 2
    config.train.n_epochs = 1
    config.train.n_warmup_epochs = 1
    config.train.eval_interval = 1
    config.data.max_n_samples = 2
    with TemporaryDirectory() as temp_dir:
        config.logging.dir = temp_dir
        run_reg(config)


@pytest.mark.integration
def test_segmentation() -> None:
    """Execute script."""
    config_path = Path(__file__).parent.resolve() / "segmentation.yaml"
    config = OmegaConf.load(config_path)
    config.train.batch_size = 2
    config.train.n_epochs = 1
    config.train.n_warmup_epochs = 1
    config.train.eval_interval = 1
    config.data.max_n_samples = 2
    with TemporaryDirectory() as temp_dir:
        config.logging.dir = temp_dir
        run_seg(config)


@pytest.mark.integration
def test_pretrain() -> None:
    """Execute script."""
    config_path = Path(__file__).parent.resolve() / "pretrain.yaml"
    config = OmegaConf.load(config_path)
    config.train.batch_size = 2
    config.train.n_epochs = 1
    config.train.n_warmup_epochs = 1
    config.train.eval_interval = 1
    with TemporaryDirectory() as temp_dir:
        config.logging.dir = temp_dir
        run_pretrain(config)
