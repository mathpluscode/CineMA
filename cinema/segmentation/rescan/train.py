"""Script to train."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import pandas as pd

from cinema.log import get_logger
from cinema.segmentation.dataset import get_segmentation_transforms
from cinema.segmentation.rescan.dataset import CineSegmentationDataset
from cinema.segmentation.train import (
    get_segmentation_model,
    segmentation_eval_dataloader,
    segmentation_loss,
)
from cinema.train import maybe_subset_dataset, run_train

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset
logger = get_logger(__name__)


def load_dataset(config: DictConfig) -> tuple[Dataset, Dataset]:
    """Load and split the dataset.

    Args:
        config: configuration file.

    Returns:
        train_dataset: dataset for training.
        val_dataset: dataset for validation.
    """
    data_dir = Path(config.data.dir).expanduser()
    train_meta_df = pd.read_csv(data_dir / "train_metadata.csv")

    # split train and val
    # G/s_0053 -> G
    train_meta_df["group"] = train_meta_df["pid"].apply(lambda x: x.split("/")[0])
    train_meta_df = train_meta_df.sort_values("pid").reset_index(drop=True)
    val_meta_df = train_meta_df.drop_duplicates("group").copy()
    train_meta_df = train_meta_df.loc[~train_meta_df.index.isin(val_meta_df.index)].reset_index(drop=True)

    train_meta_df, val_meta_df = maybe_subset_dataset(config, train_meta_df, val_meta_df)
    train_transform, val_transform = get_segmentation_transforms(config)
    train_dataset = CineSegmentationDataset(
        data_dir=data_dir / "train",
        meta_df=train_meta_df,
        views=config.model.views,
        has_labels=True,
        transform=train_transform,
    )
    val_dataset = CineSegmentationDataset(
        data_dir=data_dir / "train",
        meta_df=val_meta_df,
        views=config.model.views,
        has_labels=True,
        transform=val_transform,
    )
    return train_dataset, val_dataset


@hydra.main(version_base=None, config_path="", config_name="config")
def main(config: DictConfig) -> None:
    """Entrypoint for fine-tuning.

    Args:
        config: config loaded from yaml.
    """
    run_train(
        config=config,
        load_dataset=load_dataset,
        get_model_fn=get_segmentation_model,
        loss_fn=segmentation_loss,
        eval_dataloader_fn=segmentation_eval_dataloader,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
