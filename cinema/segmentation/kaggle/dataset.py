"""ACDC dataset for downstream tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813
import torch
from torch.utils.data import Dataset

from cinema.log import get_logger
from cinema.metric import ejection_fraction

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from monai.transforms import Transform

logger = get_logger(__name__)


class KaggleVideoDataset(Dataset):
    """Kaggle dataset that returns videos."""

    def __init__(
        self,
        data_dir: Path,
        meta_df: pd.DataFrame,
        view: str,
        max_n_frames: int,
        transform: Transform | None = None,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_dir: path to the data directory.
            meta_df: metadata dataframe, including pid and n_slices.
            view: "sax" or "lax_2c" or "lax_4c".
            max_n_frames: maximum number of frames to load.
            transform: data augmentation.
            dtype: data type for image.
        """
        if view not in {"sax", "lax_2c", "lax_4c"}:
            raise ValueError(f"Invalid view {view}.")
        self.data_dir = data_dir
        self.view = view
        self.meta_df = meta_df
        self.max_n_frames = max_n_frames
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
            A dictionary having required views and metadata.
            Video shapes are (t, 1, x, y, z) for SAX and (t, 1, x, y) for LAX.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.meta_df.iloc[idx]
        pid = int(row["pid"])
        pid_dir = self.data_dir / str(pid)

        edv = row["diastole_volume"]
        esv = row["systole_volume"]
        ef = ejection_fraction(edv, esv)
        data = {
            "pid": pid,
            "n_slices": int(row["n_slices"]),
            "n_frames": int(row["n_frames"]),
            "edv": edv,
            "esv": esv,
            "ef": ef,
        }

        view_path = pid_dir / f"{pid}_{self.view}_t.nii.gz"
        # (t, z, y, x) for SAX and (t, y, x) for LAX
        view_video = sitk.GetArrayFromImage(sitk.ReadImage(view_path))
        # (t, x, y, z) for SAX and (t, x, y) for LAX
        view_video = np.transpose(view_video, (0, *range(1, view_video.ndim)[::-1]))
        # t = max_n_frames
        view_video = view_video[: self.max_n_frames]
        data[f"{self.view}_image"] = torch.from_numpy(view_video)

        if self.transform:
            data = self.transform(data)

        if view_video.shape[0] < self.max_n_frames:
            data[f"{self.view}_image"] = torch.cat(
                [
                    data[f"{self.view}_image"],
                    torch.zeros((self.max_n_frames - view_video.shape[0], *view_video.shape[1:])),
                ],
                dim=0,
            )
        data[f"{self.view}_image"] = data[f"{self.view}_image"][:, None, ...]
        return data
