"""Dataset for end-diastole and end-systole frame segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandAffined,
    RandCoarseDropoutd,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityd,
    SpatialPadd,
)
from torch.utils.data import Dataset

from cinema.log import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from monai.transforms import Transform
    from omegaconf import DictConfig
logger = get_logger(__name__)


class EndDiastoleEndSystoleDataset(Dataset):
    """Dataset for ED/ES frame segmentation.

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
        views: str | list[str],
        transform: Transform | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: path to the data directory.
            meta_df: metadata dataframe, including pid and n_slices.
            views: view of the data, sax, lax_2c, lax_3c, or lax_4c.
            transform: data augmentation to be applied per sample per view.
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

    def __len__(self) -> int:
        """Return the number of samples.

        ED and ES are paired, so the number of samples is doubled.

        Returns:
            Number of samples.
        """
        return len(self.meta_df) * 2

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

        row_idx = idx // 2
        is_ed = idx % 2 == 0
        row = self.meta_df.iloc[row_idx]
        pid = row["pid"]
        pid_dir = self.data_dir / pid
        data = {
            "pid": pid,
            "is_ed": is_ed,
        }

        frame_name = "ed" if is_ed else "es"
        for view in self.views:
            image_path = pid_dir / f"{pid}_{view}_{frame_name}.nii.gz"
            label_path = pid_dir / f"{pid}_{view}_{frame_name}_gt.nii.gz"

            image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))  # (x, y, z)
            label = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))  # same shape as image

            data[f"{view}_width"], data[f"{view}_height"] = image.shape[:2]
            if view == "sax":
                data["n_slices"] = int(row["n_slices"])
            else:
                image = image[..., 0]  # (x, y, 1) -> (x, y)
                label = label[..., 0]  # (x, y, 1) -> (x, y)

            data[f"{view}_image"] = torch.from_numpy(image[None, ...])  # (1, x, y) or (1, x, y, z)
            data[f"{view}_label"] = torch.from_numpy(label[None, ...].astype(np.int8))

        if self.transform:
            data = self.transform(data)
        return data


def get_segmentation_transforms(config: DictConfig) -> tuple[Transform, Transform]:
    """Get the data augmentation transforms for segmentation.

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
    dropout_size_dict = {
        v: config.transform.sax.get("dropout_size", None)
        if v == "sax"
        else config.transform.lax.get("dropout_size", None)
        for v in views
    }
    train_transforms = []
    val_transforms = []
    for view in views:
        train_transforms += [
            RandAdjustContrastd(keys=f"{view}_image", prob=config.transform.prob, gamma=config.transform.gamma),
            RandGaussianNoised(keys=f"{view}_image", prob=config.transform.prob),
            ScaleIntensityd(keys=f"{view}_image"),
            RandAffined(
                keys=(f"{view}_image", f"{view}_label"),
                mode=("bilinear", "nearest"),
                prob=config.transform.prob,
                rotate_range=tuple(r / 180 * np.pi for r in rotate_range_dict[view]),
                translate_range=translate_range_dict[view],
                scale_range=config.transform.scale_range,
                padding_mode="zeros",
                lazy=True,
                allow_missing_keys=True,  # Rescan test_retest_100 has no segmentation labels
            ),
        ]
        if dropout_size_dict[view]:
            train_transforms.append(
                RandCoarseDropoutd(
                    keys=f"{view}_image",
                    prob=config.transform.prob,
                    holes=1,
                    fill_value=0,
                    spatial_size=dropout_size_dict[view],
                )
            )
        train_transforms += [
            RandSpatialCropd(
                keys=(f"{view}_image", f"{view}_label"),
                roi_size=patch_size_dict[view],
                lazy=True,
                allow_missing_keys=True,  # Rescan test_retest_100 has no segmentation labels
            ),
            SpatialPadd(
                keys=(f"{view}_image", f"{view}_label"),
                spatial_size=patch_size_dict[view],
                method="end",
                lazy=True,
                allow_missing_keys=True,  # Rescan test_retest_100 has no segmentation labels
            ),
        ]
        val_transforms += [
            ScaleIntensityd(keys=f"{view}_image"),
            SpatialPadd(
                keys=(f"{view}_image", f"{view}_label"),
                spatial_size=patch_size_dict[view],
                method="end",
                lazy=True,
                allow_missing_keys=True,  # Myops2020 test set does not have labels
            ),
        ]
    return Compose(train_transforms), Compose(val_transforms)
