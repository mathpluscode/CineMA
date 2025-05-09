"""Preprocess the M&Ms dataset.

The original classes are:
LV, left ventricle, class 1
MYO, myocardium, class 2
RV, right ventricle, class 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
from tqdm import tqdm

from cinema import LV_LABEL, RV_LABEL
from cinema.data.mnms2 import MNMS2_LABEL_MAP, MNMS2_LAX_SLICE_SIZE, MNMS2_SAX_SLICE_SIZE, MNMS2_SPACING
from cinema.data.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_3d,
    get_binary_mask_bounding_box,
    get_center_crop_size_from_bbox,
    resample_spacing_3d,
)
from cinema.log import get_logger
from cinema.metric import ejection_fraction

logger = get_logger(__name__)


def preprocess_pid(  # pylint:disable=too-many-statements
    data_dir: Path,
    pid: str,
    out_dir: Path,
) -> dict[str, str | int | float]:
    """Preprocess the ACDC data.

    Args:
        data_dir: path to the data directory.
        pid: patient id.
        out_dir: output directory.

    Returns:
        dictionary of the config file, including the patient id and class volumes.
    """
    lax_4c_ed_image_path = data_dir / pid / f"{pid}_LA_ED.nii.gz"
    lax_4c_ed_label_path = data_dir / pid / f"{pid}_LA_ED_gt.nii.gz"
    lax_4c_es_image_path = data_dir / pid / f"{pid}_LA_ES.nii.gz"
    lax_4c_es_label_path = data_dir / pid / f"{pid}_LA_ES_gt.nii.gz"
    sax_ed_image_path = data_dir / pid / f"{pid}_SA_ED.nii.gz"
    sax_ed_label_path = data_dir / pid / f"{pid}_SA_ED_gt.nii.gz"
    sax_es_image_path = data_dir / pid / f"{pid}_SA_ES.nii.gz"
    sax_es_label_path = data_dir / pid / f"{pid}_SA_ES_gt.nii.gz"

    # load
    lax_4c_ed_image = sitk.ReadImage(str(lax_4c_ed_image_path))
    lax_4c_ed_label = sitk.ReadImage(str(lax_4c_ed_label_path), outputPixelType=sitk.sitkUInt8)
    lax_4c_es_image = sitk.ReadImage(str(lax_4c_es_image_path))
    lax_4c_es_label = sitk.ReadImage(str(lax_4c_es_label_path), outputPixelType=sitk.sitkUInt8)
    sax_ed_image = sitk.ReadImage(str(sax_ed_image_path))
    sax_ed_label = sitk.ReadImage(str(sax_ed_label_path), outputPixelType=sitk.sitkUInt8)
    sax_es_image = sitk.ReadImage(str(sax_es_image_path))
    sax_es_label = sitk.ReadImage(str(sax_es_label_path), outputPixelType=sitk.sitkUInt8)

    # unify labels
    lax_4c_ed_label = sitk.ChangeLabel(lax_4c_ed_label, MNMS2_LABEL_MAP)
    lax_4c_es_label = sitk.ChangeLabel(lax_4c_es_label, MNMS2_LABEL_MAP)
    sax_ed_label = sitk.ChangeLabel(sax_ed_label, MNMS2_LABEL_MAP)
    sax_es_label = sitk.ChangeLabel(sax_es_label, MNMS2_LABEL_MAP)

    # resample
    data: dict[str, str | int | float] = {"pid": pid}
    orig_lax_4c_spacing = lax_4c_ed_image.GetSpacing()[:2]  # (x, y)
    data["orig_lax_4c_spacing_x"] = orig_lax_4c_spacing[0]
    data["orig_lax_4c_spacing_y"] = orig_lax_4c_spacing[1]
    lax_4c_ed_image = resample_spacing_3d(
        image=lax_4c_ed_image,
        is_label=False,
        target_spacing=(*MNMS2_SPACING[:2], lax_4c_ed_image.GetSpacing()[-1]),
    )
    lax_4c_ed_label = resample_spacing_3d(
        image=lax_4c_ed_label,
        is_label=True,
        target_spacing=(*MNMS2_SPACING[:2], lax_4c_ed_label.GetSpacing()[-1]),
    )
    lax_4c_es_image = resample_spacing_3d(
        image=lax_4c_es_image,
        is_label=False,
        target_spacing=(*MNMS2_SPACING[:2], lax_4c_es_image.GetSpacing()[-1]),
    )
    lax_4c_es_label = resample_spacing_3d(
        image=lax_4c_es_label,
        is_label=True,
        target_spacing=(*MNMS2_SPACING[:2], lax_4c_es_label.GetSpacing()[-1]),
    )

    orig_sax_spacing = sax_ed_image.GetSpacing()  # (x, y, z)
    data["orig_sax_spacing_x"] = orig_sax_spacing[0]
    data["orig_sax_spacing_y"] = orig_sax_spacing[1]
    data["orig_sax_spacing_z"] = orig_sax_spacing[2]
    sax_ed_image = resample_spacing_3d(
        image=sax_ed_image,
        is_label=False,
        target_spacing=MNMS2_SPACING,
    )
    sax_ed_label = resample_spacing_3d(
        image=sax_ed_label,
        is_label=True,
        target_spacing=MNMS2_SPACING,
    )
    sax_es_image = resample_spacing_3d(
        image=sax_es_image,
        is_label=False,
        target_spacing=MNMS2_SPACING,
    )
    sax_es_label = resample_spacing_3d(
        image=sax_es_label,
        is_label=True,
        target_spacing=MNMS2_SPACING,
    )

    # crop based on ed_label, crop ED/ES the same way
    lax_4c_ed_label_arr = np.transpose(sitk.GetArrayFromImage(lax_4c_ed_label))  # (x, y, 1)
    bbox_min, bbox_max = get_binary_mask_bounding_box(mask=lax_4c_ed_label_arr == LV_LABEL)  # crop to center of LV
    crop_lower, crop_upper = get_center_crop_size_from_bbox(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        current_size=lax_4c_ed_label.GetSize(),
        target_size=(*MNMS2_LAX_SLICE_SIZE, 1),
    )
    lax_4c_ed_image = sitk.Crop(lax_4c_ed_image, crop_lower, crop_upper)
    lax_4c_ed_label = sitk.Crop(lax_4c_ed_label, crop_lower, crop_upper)
    lax_4c_es_image = sitk.Crop(lax_4c_es_image, crop_lower, crop_upper)
    lax_4c_es_label = sitk.Crop(lax_4c_es_label, crop_lower, crop_upper)

    sax_ed_label_arr = np.transpose(sitk.GetArrayFromImage(sax_ed_label))  # (x, y, z)
    n_slices = sax_ed_label_arr.shape[-1]
    data["n_slices"] = n_slices
    bbox_min, bbox_max = get_binary_mask_bounding_box(mask=sax_ed_label_arr == LV_LABEL)  # crop to center of LV
    crop_lower, crop_upper = get_center_crop_size_from_bbox(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        current_size=sax_ed_label.GetSize(),
        target_size=(*MNMS2_SAX_SLICE_SIZE, n_slices),
    )
    sax_ed_image = sitk.Crop(sax_ed_image, crop_lower, crop_upper)
    sax_ed_label = sitk.Crop(sax_ed_label, crop_lower, crop_upper)
    sax_es_image = sitk.Crop(sax_es_image, crop_lower, crop_upper)
    sax_es_label = sitk.Crop(sax_es_label, crop_lower, crop_upper)

    # calculate EDV, ESV, EF for LV and RV, ml = 1000 mm^3
    data["lv_edv"] = sitk.GetArrayFromImage(sax_ed_label == LV_LABEL).sum() * np.prod(MNMS2_SPACING) / 1000.0
    data["lv_esv"] = sitk.GetArrayFromImage(sax_es_label == LV_LABEL).sum() * np.prod(MNMS2_SPACING) / 1000.0
    data["lv_ef"] = ejection_fraction(edv=data["lv_edv"], esv=data["lv_esv"])
    data["rv_edv"] = sitk.GetArrayFromImage(sax_ed_label == RV_LABEL).sum() * np.prod(MNMS2_SPACING) / 1000.0
    data["rv_esv"] = sitk.GetArrayFromImage(sax_es_label == RV_LABEL).sum() * np.prod(MNMS2_SPACING) / 1000.0
    data["rv_ef"] = ejection_fraction(edv=data["rv_edv"], esv=data["rv_esv"])

    # normalise intensity
    lax_4c_ed_image = clip_and_normalise_intensity_3d(lax_4c_ed_image, intensity_range=None)
    lax_4c_es_image = clip_and_normalise_intensity_3d(lax_4c_es_image, intensity_range=None)
    sax_ed_image = clip_and_normalise_intensity_3d(sax_ed_image, intensity_range=None)
    sax_es_image = clip_and_normalise_intensity_3d(sax_es_image, intensity_range=None)

    # out
    # LAX images are stored as 3D for better visualisation in 3D slicer
    # such that the labels are defined in the same space across different views
    out_dir = out_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    lax_4c_ed_image_path = out_dir / f"{pid}_lax_4c_ed.nii.gz"
    lax_4c_ed_label_path = out_dir / f"{pid}_lax_4c_ed_gt.nii.gz"
    lax_4c_es_image_path = out_dir / f"{pid}_lax_4c_es.nii.gz"
    lax_4c_es_label_path = out_dir / f"{pid}_lax_4c_es_gt.nii.gz"
    sax_ed_image_path = out_dir / f"{pid}_sax_ed.nii.gz"
    sax_ed_label_path = out_dir / f"{pid}_sax_ed_gt.nii.gz"
    sax_es_image_path = out_dir / f"{pid}_sax_es.nii.gz"
    sax_es_label_path = out_dir / f"{pid}_sax_es_gt.nii.gz"
    sitk.WriteImage(
        image=cast_to_uint8(lax_4c_ed_image),
        fileName=lax_4c_ed_image_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=lax_4c_ed_label,
        fileName=lax_4c_ed_label_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(lax_4c_es_image),
        fileName=lax_4c_es_image_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=lax_4c_es_label,
        fileName=lax_4c_es_label_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(sax_ed_image),
        fileName=sax_ed_image_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=sax_ed_label,
        fileName=sax_ed_label_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(sax_es_image),
        fileName=sax_es_image_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=sax_es_label,
        fileName=sax_es_label_path,
        useCompression=True,
    )
    return data


def preprocess_split(meta_df: pd.DataFrame, data_dir: Path, split: str, out_dir: Path) -> None:
    """Preprocess the split data.

    Args:
        meta_df: metadata dataframe.
        data_dir: path to the data directory.
        split: split name, train or test.
        out_dir: output directory.
    """
    meta_df["pid"] = meta_df["pid"].apply(lambda x: f"{x:03d}")
    data_lst = []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        pid = str(row["pid"])
        try:
            data = preprocess_pid(
                data_dir=data_dir,
                pid=pid,
                out_dir=out_dir / split,
            )
        except RuntimeError:
            logger.exception(f"Failed to preprocess {pid}.")
            continue
        data_lst.append(data)
    data_df = pd.DataFrame(data_lst)
    meta_df = meta_df.merge(data_df, on="pid")
    meta_df = meta_df.sort_values("pid")
    meta_df_path = out_dir / f"{split}_metadata.csv"
    meta_df.to_csv(meta_df_path, index=False)
    logger.info(f"Saved metadata to {meta_df_path}.")


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Folder of the unzipped database folder.",
        default="MnM2",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Folder saving output files.",
        default="processed",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Preprocess the dataset."""
    args = parse_args()

    # read metadata
    meta_df = pd.read_csv(args.data_dir / "dataset_information.csv").dropna()
    meta_df = meta_df.rename(
        columns={
            "SUBJECT_CODE": "pid",
            "DISEASE": "pathology",
            "VENDOR": "vendor",
            "SCANNER": "scanner",
            "FIELD": "field",
        },
        errors="raise",
    )
    meta_df["pid"] = meta_df["pid"].astype(int)
    if len(meta_df) != 360:
        raise ValueError(f"Expected 360 patients, got {len(meta_df)}.")

    # train / val / test split
    # The challenge 160 training subjects, 40 validation subjects and 160 testing subjects are included sequentially.
    train_meta_df = meta_df[meta_df["pid"] <= 160].copy()
    val_meta_df = meta_df[(meta_df["pid"] > 160) & (meta_df["pid"] <= 200)].copy()
    test_meta_df = meta_df[meta_df["pid"] > 200].copy()

    # preprocess each split
    data_dir = args.data_dir / "dataset"
    preprocess_split(train_meta_df, data_dir, "train", args.out_dir)
    preprocess_split(val_meta_df, data_dir, "val", args.out_dir)
    preprocess_split(test_meta_df, data_dir, "test", args.out_dir)


if __name__ == "__main__":
    main()
