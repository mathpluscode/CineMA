"""Script to train."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pandas as pd

from cinema.classification.dataset import EndDiastoleEndSystoleDataset, get_image_transforms
from cinema.classification.train import (
    classification_eval_dataloader,
    classification_loss,
    get_classification_or_regression_model,
)
from cinema.log import get_logger
from cinema.train import maybe_subset_dataset, run_train

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset
logger = get_logger(__name__)


def load_mnms2_dataset(config: DictConfig) -> tuple[Dataset, Dataset]:
    """Load and split the dataset.

    Args:
        config: configuration file.

    Returns:
        train_dataset: dataset for training.
        val_dataset: dataset for validation.
    """
    data_dir = Path(config.data.dir).expanduser()
    class_col = config.data.class_column
    classes = config.data[class_col]
    train_meta_df = pd.read_csv(data_dir / "train_metadata.csv", dtype={"pid": str})
    val_meta_df = pd.read_csv(data_dir / "val_metadata.csv", dtype={"pid": str})
    # certain class does not exist in the validation set
    n_val = len(val_meta_df)
    val_meta_df = val_meta_df[val_meta_df[class_col].isin(classes)].reset_index(drop=True)
    if len(val_meta_df) < n_val:
        logger.warning(f"Removed {n_val - len(val_meta_df)} samples from validation split.")
    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)

    train_transform, val_transform = get_image_transforms(config)
    train_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "train",
        meta_df=train_meta_df,
        class_col=class_col,
        classes=classes,
        views=config.model.views,
        transform=train_transform,
    )
    val_dataset = EndDiastoleEndSystoleDataset(
        data_dir=data_dir / "val",
        meta_df=val_meta_df,
        class_col=class_col,
        classes=classes,
        views=config.model.views,
        transform=val_transform,
    )
    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entrypoint for classification training.

    Args:
        config: config loaded from yaml.
    """
    run_train(
        config=config,
        load_dataset=load_mnms2_dataset,
        get_model_fn=get_classification_or_regression_model,
        loss_fn=classification_loss,
        eval_dataloader_fn=classification_eval_dataloader,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
