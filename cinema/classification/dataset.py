"""Dataset for end-diastole and end-systole frame classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
    Transform,
)
from torch.utils.data import Dataset

from cinema.log import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from omegaconf import DictConfig
logger = get_logger(__name__)


class EndDiastoleEndSystoleDataset(Dataset):
    """Dataset for ED/ES frame classification.

    The data should be organized as follows with different views.
    LAX images are stored in 3d, of size (x, y, 1).
    SAX images are stored in 3d, of size (x, y, z).

    ```
    data_dir/
    ├── pid1/
    │   ├── pid1_lax_4c_ed.nii.gz
    │   ├── pid1_lax_4c_ed_gt.nii.gz
    │   ├── pid1_lax_4c_es.nii.gz
    │   ├── pid1_lax_4c_es_gt.nii.gz
    │   ├── pid1_sax_ed.nii.gz
    │   ├── pid1_sax_ed_gt.nii.gz
    │   ├── pid1_sax_es.nii.gz
    │   └── pid1_sax_es_gt.nii.gz
    ├── pid2/
    ├── .../
    ```
    """

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        class_col: str,
        classes: list[str],
        views: str | list[str],
        transform: Transform | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: path to the data directory.
            meta_df: metadata dataframe, including pid and n_slices.
            class_col: column name for the class label.
            classes: list of class names, needed as there may be missing classes in the dataset.
                If int, it is the number of classes.
            views: view of the data, sax, lax_2c, lax_3c, or lax_4c.
            transform: data augmentation to be applied per sample.
            dtype: data type for image.
        """
        for col in ["pid", "n_slices"]:
            if col not in meta_df.columns:
                raise ValueError(f"Column {col} is required in meta_df.")
        if isinstance(views, str):
            views = [views]
        self.data_dir = data_dir
        self.meta_df = meta_df
        self.views = views
        self.transform = transform
        self.dtype = dtype
        self.classes = classes
        self.class_col = class_col

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Number of samples.
        """
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a sample.

        Args:
            idx: index of the sample.

        Returns:
            A dictionary having ED/ES image and label.
            Particularly, label is of type int8, not uint8.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.meta_df.iloc[idx]
        pid = row["pid"]
        pid_dir = self.data_dir / pid
        cls = row[self.class_col]
        data = {
            "pid": pid,
            "class": cls,
            "label": torch.tensor(self.classes.index(cls)),
        }

        for view in self.views:
            ed_image_path = pid_dir / f"{pid}_{view}_ed.nii.gz"
            es_image_path = pid_dir / f"{pid}_{view}_es.nii.gz"
            ed_image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(ed_image_path)))  # (x, y, z)
            es_image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(es_image_path)))
            image = np.stack([ed_image, es_image], axis=0)  # (2, x, y, 1) or (2, x, y, z)
            if view != "sax":
                image = image[..., 0]  # (2, x, y, 1) -> (2, x, y)
            data[f"{view}_image"] = torch.from_numpy(image)

        if self.transform:
            data = self.transform(data)
        return data


def get_image_transforms(config: DictConfig) -> tuple[Transform, Transform]:
    """Get the data augmentation transforms for classification and regression.

    Args:
        config: config for data augmentation.

    Returns:
        train_transforms: transforms for training.
        val_transforms: transforms for validation.
    """
    views = config.model.views
    if isinstance(views, str):
        views = [views]
    patch_size_dict = {v: config.data.sax.patch_size if v == "sax" else config.data.lax.patch_size for v in views}
    rotate_range_dict = {
        v: config.transform.sax.rotate_range if v == "sax" else config.transform.lax.rotate_range for v in views
    }
    translate_range_dict = {
        v: config.transform.sax.translate_range if v == "sax" else config.transform.lax.translate_range for v in views
    }
    train_transforms = []
    val_transforms = []
    for view in views:
        train_transforms += [
            RandAdjustContrastd(keys=f"{view}_image", prob=config.transform.prob, gamma=config.transform.gamma),
            RandGaussianNoised(keys=f"{view}_image", prob=config.transform.prob),
            ScaleIntensityd(keys=f"{view}_image"),
            RandAffined(
                keys=f"{view}_image",
                mode="bilinear",
                prob=config.transform.prob,
                rotate_range=tuple(r / 180 * np.pi for r in rotate_range_dict[view]),
                translate_range=translate_range_dict[view],
                scale_range=config.transform.scale_range,
                padding_mode="zeros",
                lazy=True,
            ),
            RandSpatialCropd(keys=f"{view}_image", roi_size=patch_size_dict[view], lazy=True),
            SpatialPadd(keys=f"{view}_image", spatial_size=patch_size_dict[view], method="end", lazy=True),
        ]
        val_transforms += [
            ScaleIntensityd(keys=f"{view}_image"),
            SpatialPadd(keys=f"{view}_image", spatial_size=patch_size_dict[view], method="end", lazy=True),
        ]
    return Compose(train_transforms), Compose(val_transforms)
