"""Dataset for end-diastole and end-systole frame segmentation."""

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


class CineSegmentationDataset(Dataset):
    """Dataset for cine segmentation.

    The data should be organized as follows with different views.
    LAX images are stored in 3d, of size (x, y, 1, t).
    SAX images are stored in 3d, of size (x, y, z, t).
    """

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        views: str | list[str],
        has_labels: bool,
        transform: Transform | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: path to the data directory.
            meta_df: metadata dataframe, including pid and n_slices.
            views: view of the data, sax, lax_2c, lax_3c, or lax_4c.
            has_labels: whether have segmentation labels.
            transform: data augmentation to be applied per sample per view.
            dtype: data type for image.
        """
        self.views = [views] if isinstance(views, str) else views
        if has_labels and set(self.views) != {"sax"}:
            # currently there are only labels for SAX view
            raise ValueError(f"Invalid view: {views}, currently only SAX view is supported for having labels.")
        self.has_labels = has_labels
        self.data_dir = data_dir
        self.meta_df = meta_df.reset_index(drop=True)
        self.transform = transform
        self.dtype = dtype

        # each SAX has multiple frames
        idx_map: dict[int, tuple[int, int]] = {}
        i = 0
        for row_idx, row in self.meta_df.iterrows():
            for j in range(row["n_frames"]):
                idx_map[i] = (int(row_idx), j)
                i += 1
        self.idx_map = idx_map

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            Number of samples.
        """
        return len(self.idx_map)

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

        row_idx, frame_idx = self.idx_map[idx]
        row = self.meta_df.iloc[row_idx]
        pid = row["pid"]
        pid_dir = self.data_dir / pid
        data = {
            "pid": pid,
            "frame": frame_idx,
        }

        for view in self.views:
            image_path = pid_dir / f"{view}_t.nii.gz"

            image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))  # (..., t)
            image = image[..., frame_idx]  # (...)

            # normalize image
            image = image.astype(self.dtype)
            v_min, v_max = np.min(image), np.max(image)
            image = (image - v_min) / (v_max - v_min)

            if view == "sax":
                data[f"{view}_width"], data[f"{view}_height"], data["n_slices"] = image.shape
            else:
                image = image[..., 0]  # (x, y, 1) -> (x, y)
                data[f"{view}_width"], data[f"{view}_height"] = image.shape
                data["n_slices"] = 1
            data[f"{view}_image"] = torch.from_numpy(image[None, ...])  # (1, ...)

            if self.has_labels:
                label_path = pid_dir / f"{view}_gt_t.nii.gz"
                label = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(label_path)))  # same shape as image
                label = label[..., frame_idx]  # (...)
                data[f"{view}_label"] = torch.from_numpy(label[None, ...].astype(np.int8))
            else:
                # test_retest_100, these metrics are same per time frame
                data["edv"] = row["edv"]
                data["esv"] = row["esv"]
                data["ef"] = row["ef"]

            if self.transform is not None:
                data = self.transform(data)
        return data
