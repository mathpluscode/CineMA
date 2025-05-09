"""Preprocess the EMIDEC dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
from tqdm import tqdm

from cinema.data.emidec import EMIDEC_SLICE_SIZE, EMIDEC_SPACING
from cinema.data.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_3d,
    get_binary_mask_bounding_box,
    get_center_crop_size_from_bbox,
    resample_spacing_3d,
)
from cinema.log import get_logger

logger = get_logger(__name__)
TEST_PROPORTION = 0.2
VAL_PROPORTION = 0.2


def preprocess_pid(  # pylint:disable=too-many-statements
    pid: str,
    data_dir: Path,
    out_dir: Path,
) -> dict[str, str | int | float]:
    """Preprocess the EMIDEC data.

    sex, age, tobacco (Y/N/former smoker), overweight (BMI > 25),
    arterial hypertension (Y/N), diabetes (Y/N), familial history of coronary artery disease (Y/N),
    ECG (ST+ (STEMI) or not), troponin (value), Killip max (between 1 and 4),
    ejection fraction of the left ventricle from echography (value), NTproBNP (value).

    Args:
        pid: case name.
        data_dir: data directory.
        out_dir: output directory.

    Returns:
        dictionary of the config file, including the patient id and class volumes.
    """
    # process metadata
    with Path.open(data_dir / f"Case {pid}.txt", "r", encoding="unicode_escape") as f:
        lines = f.read().splitlines()
    lines = [x.split(":") for x in lines if ":" in x]
    raw_data: dict[str, str | int | float] = {x[0].strip(): x[1].strip() for x in lines}
    data = {
        "pid": pid,
        "sex": raw_data["Sex"],
        "age": int(raw_data["Age"]),
        "tobacco": {1: "Y", 2: "N", 3: "Former"}[int(raw_data["Tobacco"])],
        "overweight": raw_data["Overweight"],
        "arterial_hypertension": raw_data["Arterial hypertension"],
        "diabetes": raw_data["Diabetes"],
        "family_history": raw_data["Familial history of coronary artery disease"],
        "ecg": raw_data["ECG (ST +)"],
        "troponin": float(raw_data["Troponin"]),
        "killip_max": int(raw_data["Killip Max"]),
        "ef": int(raw_data["FEVG"]),
        "ntprobnp": int(raw_data["NTProBNP"]),
        "pathology": pid[0],
    }

    # process image/labels
    image_path = data_dir / f"Case_{pid}" / "Images" / f"Case_{pid}.nii.gz"
    label_path = data_dir / f"Case_{pid}" / "Contours" / f"Case_{pid}.nii.gz"
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path), outputPixelType=sitk.sitkUInt8)

    orig_spacing = image.GetSpacing()
    data["orig_spacing_x"] = orig_spacing[0]
    data["orig_spacing_y"] = orig_spacing[1]
    data["orig_spacing_z"] = orig_spacing[2]

    image = resample_spacing_3d(image=image, is_label=False, target_spacing=EMIDEC_SPACING)
    label = resample_spacing_3d(image=label, is_label=True, target_spacing=EMIDEC_SPACING)

    # crop based on cavity
    label_arr = np.transpose(sitk.GetArrayFromImage(label))  # (x, y, z)
    if label_arr.min() < 0 or label_arr.max() > 4:
        raise ValueError(f"Invalid label values: {np.unique(label_arr)} for {pid}.")
    n_slices = label_arr.shape[-1]
    data["n_slices"] = n_slices
    bbox_min, bbox_max = get_binary_mask_bounding_box(mask=label_arr == 2)  # crop to center of cavity
    crop_lower, crop_upper = get_center_crop_size_from_bbox(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        current_size=label.GetSize(),
        target_size=(*EMIDEC_SLICE_SIZE, n_slices),
    )
    image = sitk.Crop(image, crop_lower, crop_upper)
    label = sitk.Crop(label, crop_lower, crop_upper)
    for cls_idx in range(1, 5):
        data[f"cls_{cls_idx}_proportion"] = (label_arr == cls_idx).mean()

    # normalise intensity
    image = clip_and_normalise_intensity_3d(image, intensity_range=None)

    # out
    out_dir = out_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{pid}.nii.gz"
    label_path = out_dir / f"{pid}_gt.nii.gz"
    sitk.WriteImage(
        image=cast_to_uint8(image),
        fileName=image_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=label,
        fileName=label_path,
        useCompression=True,
    )
    return data


def split_pids(pids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split the patient ids into training, validation, and testing sets.

    Args:
        pids: list of patient ids.

    Returns:
        tuple of training, validation, and testing patient ids.
    """
    test_size = int(len(pids) * TEST_PROPORTION)
    val_size = int(len(pids) * VAL_PROPORTION)
    train_size = len(pids) - test_size - val_size
    logger.info(f"Using {train_size} samples for training, {val_size} for validation, and {test_size} for testing.")
    rng = np.random.default_rng(0)
    rng.shuffle(pids)
    train_pids = pids[:train_size]
    val_pids = pids[train_size : train_size + val_size]
    test_pids = pids[train_size + val_size :]
    return train_pids, val_pids, test_pids


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Folder of the unzipped database folder.",
        default="emidec-dataset-1.0.1",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Folder saving output files.",
        default="processed",
    )
    args = parser.parse_args()

    return args


def preprocess(data_dir: Path, out_dir: Path) -> None:
    """Preprocess the data.

    Args:
        data_dir: path to the data directory.
        out_dir: path to the output directory.
    """
    case_names = [x.stem.split(" ")[1] for x in data_dir.glob("Case *.txt")]
    case_names = sorted(case_names, key=lambda x: int(x[1:]))
    meta_df = pd.DataFrame([preprocess_pid(x, data_dir, out_dir / "train") for x in tqdm(case_names)])

    normal_pids = meta_df[meta_df["pathology"] == "N"]["pid"].to_list()
    pathological_pids = meta_df[meta_df["pathology"] == "P"]["pid"].to_list()
    logger.info("Splitting normal patients into training, validation, and testing sets.")
    train_normal_pids, val_normal_pids, test_normal_pids = split_pids(normal_pids)
    logger.info("Splitting pathological patients into training, validation, and testing sets.")
    train_pathological_pids, val_pathological_pids, test_pathological_pids = split_pids(pathological_pids)

    train_pids = train_normal_pids + train_pathological_pids
    val_pids = val_normal_pids + val_pathological_pids
    test_pids = test_normal_pids + test_pathological_pids
    meta_df[meta_df["pid"].isin(train_pids)].to_csv(out_dir / "train_metadata.csv", index=False)
    meta_df[meta_df["pid"].isin(val_pids)].to_csv(out_dir / "val_metadata.csv", index=False)
    meta_df[meta_df["pid"].isin(test_pids)].to_csv(out_dir / "test_metadata.csv", index=False)

    meta_df_path = out_dir / "all_metadata.csv"
    meta_df.to_csv(meta_df_path, index=False)
    logger.info(f"Saved train metadata to {meta_df_path}.")


def main() -> None:
    """Preprocess the dataset."""
    args = parse_args()
    preprocess(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
