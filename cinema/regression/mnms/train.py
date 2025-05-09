"""Script to train."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pandas as pd

from cinema.classification.dataset import get_image_transforms
from cinema.classification.train import get_classification_or_regression_model
from cinema.log import get_logger
from cinema.regression.dataset import EndDiastoleEndSystoleDataset
from cinema.regression.train import regression_eval_dataloader, regression_loss
from cinema.train import maybe_subset_dataset, run_train

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset
logger = get_logger(__name__)


def load_mnms_dataset(config: DictConfig) -> tuple[Dataset, Dataset]:
    """Load and split the dataset.

    Args:
        config: configuration file.

    Returns:
        train_dataset: dataset for training.
        val_dataset: dataset for validation.
    """
    data_dir = Path(config.data.dir)
    reg_col = config.data.regression_column
    reg_mean = config.data[reg_col].mean
    reg_std = config.data[reg_col].std
    train_meta_df = pd.read_csv(data_dir / "train_metadata.csv")
    val_meta_df = pd.read_csv(data_dir / "val_metadata.csv")
    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    train_transform, val_transform = get_image_transforms(config)
    train_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train",
        meta_df=train_meta_df,
        reg_col=reg_col,
        reg_mean=reg_mean,
        reg_std=reg_std,
        views=config.model.views,
        transform=train_transform,
    )
    val_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "val",
        meta_df=val_meta_df,
        reg_col=reg_col,
        reg_mean=reg_mean,
        reg_std=reg_std,
        views=config.model.views,
        transform=val_transform,
    )
    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entrypoint for regression training.

    Args:
        config: config loaded from yaml.
    """
    run_train(
        config=config,
        load_dataset=load_mnms_dataset,
        get_model_fn=get_classification_or_regression_model,
        loss_fn=regression_loss,
        eval_dataloader_fn=regression_eval_dataloader,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
