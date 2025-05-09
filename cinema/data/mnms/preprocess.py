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
from cinema.data.mnms import MNMS_LABEL_MAP, MNMS_SAX_SLICE_SIZE, MNMS_SPACING
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
    image_path: Path,
    label_path: Path,
    ed_index: int,
    es_index: int,
    out_dir: Path,
) -> dict[str, str | int | float]:
    """Preprocess the data.

    Args:
        image_path: path to the image file, e.g. E9H1U4/E9H1U4_sa.nii.gz.
        label_path: path to the label file, e.g. E9H1U4/E9H1U4_sa_gt.nii.gz.
        ed_index: end-diastolic frame index, starting from 0.
        es_index: end-systolic frame index, starting from 0.
        out_dir: output directory.

    Returns:
        dictionary of the config file, including the patient id and class volumes.
    """
    pid = label_path.parent.name

    # load
    image = sitk.ReadImage(str(image_path))  # (x, y, z, t)
    label = sitk.ReadImage(str(label_path), outputPixelType=sitk.sitkUInt8)

    ed_image = image[:, :, :, ed_index]
    ed_label = label[:, :, :, ed_index]
    es_image = image[:, :, :, es_index]
    es_label = label[:, :, :, es_index]

    # unify labels
    ed_label = sitk.ChangeLabel(ed_label, MNMS_LABEL_MAP)
    es_label = sitk.ChangeLabel(es_label, MNMS_LABEL_MAP)

    # resample
    data: dict[str, str | int | float] = {"pid": pid}
    orig_sax_spacing = image.GetSpacing()[:-1]  # (x, y, z)
    data["orig_sax_spacing_x"] = orig_sax_spacing[0]
    data["orig_sax_spacing_y"] = orig_sax_spacing[1]
    data["orig_sax_spacing_z"] = orig_sax_spacing[2]
    ed_image = resample_spacing_3d(
        image=ed_image,
        is_label=False,
        target_spacing=MNMS_SPACING,
    )
    ed_label = resample_spacing_3d(
        image=ed_label,
        is_label=True,
        target_spacing=MNMS_SPACING,
    )
    es_image = resample_spacing_3d(
        image=es_image,
        is_label=False,
        target_spacing=MNMS_SPACING,
    )
    es_label = resample_spacing_3d(
        image=es_label,
        is_label=True,
        target_spacing=MNMS_SPACING,
    )

    # crop based on ed_label, crop ED/ES the same way
    ed_label_arr = np.transpose(sitk.GetArrayFromImage(ed_label))  # (x, y, z)
    n_slices = ed_label_arr.shape[-1]
    data["n_slices"] = n_slices
    bbox_min, bbox_max = get_binary_mask_bounding_box(mask=ed_label_arr == LV_LABEL)  # crop to center of LV
    crop_lower, crop_upper = get_center_crop_size_from_bbox(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        current_size=ed_label.GetSize(),
        target_size=(*MNMS_SAX_SLICE_SIZE, n_slices),
    )
    ed_image = sitk.Crop(ed_image, crop_lower, crop_upper)
    ed_label = sitk.Crop(ed_label, crop_lower, crop_upper)
    es_image = sitk.Crop(es_image, crop_lower, crop_upper)
    es_label = sitk.Crop(es_label, crop_lower, crop_upper)

    # calculate EDV, ESV, EF for LV and RV, ml = 1000 mm^3
    data["lv_edv"] = sitk.GetArrayFromImage(ed_label == LV_LABEL).sum() * np.prod(MNMS_SPACING) / 1000.0
    data["lv_esv"] = sitk.GetArrayFromImage(es_label == LV_LABEL).sum() * np.prod(MNMS_SPACING) / 1000.0
    data["lv_ef"] = ejection_fraction(edv=data["lv_edv"], esv=data["lv_esv"])
    data["rv_edv"] = sitk.GetArrayFromImage(ed_label == RV_LABEL).sum() * np.prod(MNMS_SPACING) / 1000.0
    data["rv_esv"] = sitk.GetArrayFromImage(es_label == RV_LABEL).sum() * np.prod(MNMS_SPACING) / 1000.0
    data["rv_ef"] = ejection_fraction(edv=data["rv_edv"], esv=data["rv_esv"])

    # normalise intensity
    ed_image = clip_and_normalise_intensity_3d(ed_image, intensity_range=None)
    es_image = clip_and_normalise_intensity_3d(es_image, intensity_range=None)

    # out
    out_dir = out_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    ed_image_path = out_dir / f"{pid}_sax_ed.nii.gz"
    ed_label_path = out_dir / f"{pid}_sax_ed_gt.nii.gz"
    es_image_path = out_dir / f"{pid}_sax_es.nii.gz"
    es_label_path = out_dir / f"{pid}_sax_es_gt.nii.gz"
    sitk.WriteImage(
        image=cast_to_uint8(ed_image),
        fileName=ed_image_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=ed_label,
        fileName=ed_label_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(es_image),
        fileName=es_image_path,
        useCompression=True,
    )
    sitk.WriteImage(
        image=es_label,
        fileName=es_label_path,
        useCompression=True,
    )

    return data


def preprocess_split(meta_df: pd.DataFrame, split_dir: Path, split: str, out_dir: Path) -> None:
    """Preprocess the split data.

    Args:
        meta_df: metadata dataframe.
        split_dir: path to the split directory.
        split: split name, train or test.
        out_dir: output directory.
    """
    data_lst = []
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        pid = str(row["pid"])
        label_path = split_dir / pid / f"{pid}_sa_gt.nii.gz"
        image_path = split_dir / pid / f"{pid}_sa.nii.gz"
        if not image_path.exists():
            logger.error(f"Image for {pid} not found: {image_path}.")
            continue
        if not label_path.exists():
            logger.error(f"Label for {pid} not found: {label_path}.")
            continue
        try:
            data = preprocess_pid(
                image_path=image_path,
                label_path=label_path,
                ed_index=int(row["ed_index"]),
                es_index=int(row["es_index"]),
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
        default="OpenDataset",
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
    meta_df = pd.read_csv(args.data_dir / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv").iloc[:, 1:]
    meta_df = meta_df.rename(
        columns={
            "External code": "pid",
            "Pathology": "pathology",
            "VendorName": "vendor_name",
            "Vendor": "vendor",
            "Centre": "center",
            "ED": "ed_index",
            "ES": "es_index",
            "Age": "age",
            "Sex": "sex",
            "Height": "height",  # may have missing values
            "Weight": "weight",  # may have missing values
        },
        errors="raise",
    )
    int_columns = ["ed_index", "es_index", "age"]
    meta_df[int_columns] = meta_df[int_columns].astype(int)

    # preprocess each split
    preprocess_split(meta_df, args.data_dir / "Training" / "Labeled", "train", args.out_dir)
    preprocess_split(meta_df, args.data_dir / "Validation", "val", args.out_dir)
    preprocess_split(meta_df, args.data_dir / "Testing", "test", args.out_dir)


if __name__ == "__main__":
    main()
