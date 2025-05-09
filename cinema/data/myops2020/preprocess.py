"""Preprocess the EMIDEC dataset."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import SimpleITK as sitk  # noqa: N813
from tqdm import tqdm

from cinema.data.myops2020 import MYOPS2020_LABEL_MAP, MYOPS2020_SLICE_SIZE
from cinema.data.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_3d,
)
from cinema.log import get_logger

logger = get_logger(__name__)


def preprocess_pid(  # pylint:disable=too-many-statements
    pid: str,
    split: str,
    image_dir: Path,
    out_dir: Path,
    label_dir: Path | None = None,
) -> dict[str, str | int | float]:
    """Preprocess the myops2020 data without cropping.

    No resampling as, for inference, needs to resample back and it's hard to control the size.

    Args:
        pid: case name.
        split: training or test.
        image_dir: input directory.
        out_dir: output directory.
        label_dir: label directory.

    Returns:
        dictionary of the config file, including the patient id and class volumes.
    """
    out_dir = out_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, str | int | float] = {"pid": pid}

    c0_path = image_dir / f"myops_{split}_{pid}_C0.nii.gz"  # bSSFP
    de_path = image_dir / f"myops_{split}_{pid}_DE.nii.gz"  # LGE
    t2_path = image_dir / f"myops_{split}_{pid}_T2.nii.gz"

    image_c0 = sitk.ReadImage(str(c0_path))
    image_de = sitk.ReadImage(str(de_path))
    image_t2 = sitk.ReadImage(str(t2_path))

    orig_spacing = image_c0.GetSpacing()
    data["orig_spacing_x"] = orig_spacing[0]
    data["orig_spacing_y"] = orig_spacing[1]
    data["orig_spacing_z"] = orig_spacing[2]

    current_size = image_c0.GetSize()  # after resampling
    data["n_slices"] = current_size[-1]
    crop_lower_x = (current_size[0] - MYOPS2020_SLICE_SIZE[0]) // 2
    crop_upper_x = current_size[0] - MYOPS2020_SLICE_SIZE[0] - crop_lower_x
    crop_lower_y = (current_size[1] - MYOPS2020_SLICE_SIZE[1]) // 2
    crop_upper_y = current_size[1] - MYOPS2020_SLICE_SIZE[1] - crop_lower_y
    crop_lower = (crop_lower_x, crop_lower_y, 0)
    crop_upper = (crop_upper_x, crop_upper_y, 0)
    data["crop_lower_x"] = crop_lower_x
    data["crop_lower_y"] = crop_lower_y
    data["crop_upper_x"] = crop_upper_x
    data["crop_upper_y"] = crop_upper_y

    image_c0 = sitk.Crop(image_c0, crop_lower, crop_upper)
    image_de = sitk.Crop(image_de, crop_lower, crop_upper)
    image_t2 = sitk.Crop(image_t2, crop_lower, crop_upper)

    image_c0 = clip_and_normalise_intensity_3d(image_c0, intensity_range=None)
    image_de = clip_and_normalise_intensity_3d(image_de, intensity_range=None)
    image_t2 = clip_and_normalise_intensity_3d(image_t2, intensity_range=None)

    image_c0_path = out_dir / f"{pid}_c0.nii.gz"
    image_de_path = out_dir / f"{pid}_de.nii.gz"
    image_t2_path = out_dir / f"{pid}_t2.nii.gz"

    sitk.WriteImage(image=cast_to_uint8(image_c0), fileName=image_c0_path, useCompression=True)
    sitk.WriteImage(image=cast_to_uint8(image_de), fileName=image_de_path, useCompression=True)
    sitk.WriteImage(image=cast_to_uint8(image_t2), fileName=image_t2_path, useCompression=True)

    if label_dir is not None:
        label_path = label_dir / f"myops_{split}_{pid}_gd.nii.gz"
        label = sitk.ReadImage(str(label_path))
        label = sitk.ChangeLabel(label, MYOPS2020_LABEL_MAP)
        label = sitk.Crop(label, crop_lower, crop_upper)
        out_path = out_dir / f"{pid}_gt.nii.gz"
        sitk.WriteImage(image=label, fileName=out_path, useCompression=True)

    return data


def preprocess() -> None:
    """Preprocess the data."""
    cache_dir = Path(os.environ.get("CINEMA_DATA_DIR", "~/.cache/cinema_datasets")).expanduser()
    data_dir = cache_dir / "myops2020"
    out_dir = cache_dir / "myops2020" / "processed"

    train_pids = sorted({int(x.name.split("_")[2]) for x in (data_dir / "train25").glob("*.nii.gz")})
    train_meta_df = pd.DataFrame(
        [
            preprocess_pid(
                pid=str(x),
                split="training",
                image_dir=data_dir / "train25",
                label_dir=data_dir / "train25_myops_gd",
                out_dir=out_dir / "train",
            )
            for x in tqdm(train_pids)
        ]
    )
    train_meta_df_path = out_dir / "train_metadata.csv"
    train_meta_df.to_csv(train_meta_df_path, index=False)
    logger.info(f"Saved train metadata to {train_meta_df_path}.")

    test_pids = sorted({int(x.name.split("_")[2]) for x in (data_dir / "test20").glob("*.nii.gz")})
    test_meta_df = pd.DataFrame(
        [
            preprocess_pid(pid=str(x), split="test", image_dir=data_dir / "test20", out_dir=out_dir / "test")
            for x in tqdm(test_pids)
        ]
    )
    test_meta_df_path = out_dir / "test_metadata.csv"
    test_meta_df.to_csv(test_meta_df_path, index=False)
    logger.info(f"Saved test metadata to {test_meta_df_path}.")


def main() -> None:
    """Preprocess the dataset."""
    preprocess()


if __name__ == "__main__":
    main()
