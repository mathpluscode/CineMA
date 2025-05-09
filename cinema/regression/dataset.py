"""Dataset for end-diastole and end-systole frame regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from torch.utils.data import Dataset

from cinema.log import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from monai.transforms import Transform
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
        reg_col: str,
        reg_mean: float,
        reg_std: float,
        views: str | list[str],
        transform: Transform | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: path to the data directory.
            meta_df: metadata dataframe, including pid and n_slices.
            reg_col: column name for the regression.
            reg_mean: mean value for regression.
            reg_std: standard deviation value for regression.
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
        self.reg_col = reg_col
        self.reg_mean = reg_mean
        self.reg_std = reg_std

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
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.meta_df.iloc[idx]
        pid = row["pid"]
        pid_dir = self.data_dir / pid
        data = {
            "pid": pid,
            "label": torch.tensor([(row[self.reg_col] - self.reg_mean) / self.reg_std], dtype=torch.float32),
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
