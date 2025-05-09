"""Dataset for landmark localization regression."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from monai.transforms import Transform


class LandmarkRegressionDataset(Dataset):
    """Dataset for landmark localization detection."""

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        view: str,
        transform: Transform | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: path to the data directory.
            meta_df: metadata dataframe, including pid and n_slices.
            view: view to use, lax_2c or lax_4c.
            transform: data augmentation to be applied per sample per view.
            dtype: data type for image.
        """
        self.data_dir = data_dir
        self.meta_df = meta_df
        self.views = [view]
        self.transform = transform
        self.dtype = dtype

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
        uid = row["uid"]
        view = self.views[0]

        x1, y1, x2, y2, x3, y3 = row["x1"], row["y1"], row["x2"], row["y2"], row["x3"], row["y3"]
        image = np.transpose(np.array(Image.open(self.data_dir / view / "images" / f"{uid}.png")))  # (x, y)
        w, h = image.shape[:2]
        label = np.array([x1, y1, x2, y2, x3, y3]).astype(self.dtype)
        label_scale = np.array([w, h, w, h, w, h]).astype(self.dtype)
        label /= label_scale

        data = {"uid": uid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "x3": x3, "y3": y3}
        data[f"{view}_width"], data[f"{view}_height"] = image.shape
        data[f"{view}_image"] = torch.from_numpy(image[None, ...])  #  (1, x, y)
        data["label"] = torch.from_numpy(label)  # (3, x, y)
        data["label_scale"] = torch.from_numpy(label_scale)

        if self.transform:
            data = self.transform(data)
        return data
