"""Preprocess the ACDC dataset.

RV, right ventricle, class 1
MYO, myocardium, class 2
LV, left ventricle, class 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
from tqdm import tqdm

from cinema import LV_LABEL, RV_LABEL
from cinema.data.acdc import ACDC_LABEL_MAP, ACDC_SAX_SLICE_SIZE, ACDC_SPACING
from cinema.data.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_3d,
    clip_and_normalise_intensity_4d,
    crop_4d,
    get_binary_mask_bounding_box,
    get_center_crop_size_from_bbox,
    resample_spacing_3d,
    resample_spacing_4d,
)
from cinema.log import get_logger
from cinema.metric import ejection_fraction

logger = get_logger(__name__)


def load_config(config_path: Path) -> dict[str, str | int | float]:
    """Load the config file.

    Config file example:
        ED: 1
        ES: 12
        Group: DCM
        Height: 184.0
        NbFrame: 30
        Weight: 95.0

    Args:
        config_path: path to the config file.

    Returns:
        dictionary of the config file, including the patient id.
    """
    with Path.open(config_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    d: dict[str, str | int | float] = {}
    for x in lines:
        k, v = x.split(": ")
        d[k] = v
    height = float(d["Height"]) / 100.0  # meter
    weight = float(d["Weight"])  # kg
    bmi = weight / height**2
    return {
        "pid": config_path.parent.name,
        "pathology": d["Group"],
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "n_frames": int(d["NbFrame"]),
        "ed_frame": int(d["ED"]),
        "es_frame": int(d["ES"]),
    }


def preprocess_pid(  # pylint:disable=too-many-statements
    config_path: Path,
    out_dir: Path,
) -> dict[str, str | int | float]:
    """Preprocess the ACDC data.

    Args:
        config_path: path to the config file.
        out_dir: output directory.

    Returns:
        dictionary of the config file, including the patient id and class volumes.
    """
    data = load_config(config_path)
    pid = str(data["pid"])
    ed = int(data["ed_frame"])  # it's frame not index, starting at 1
    es = int(data["es_frame"])

    # load
    image_path = config_path.parent / f"{pid}_4d.nii.gz"
    ed_image_path = config_path.parent / f"{pid}_frame{ed:02d}.nii.gz"
    ed_label_path = config_path.parent / f"{pid}_frame{ed:02d}_gt.nii.gz"
    es_image_path = config_path.parent / f"{pid}_frame{es:02d}.nii.gz"
    es_label_path = config_path.parent / f"{pid}_frame{es:02d}_gt.nii.gz"

    image = sitk.ReadImage(str(image_path))
    ed_image = sitk.ReadImage(str(ed_image_path))  # (x, y, z)
    ed_label = sitk.ReadImage(str(ed_label_path), outputPixelType=sitk.sitkUInt8)
    es_image = sitk.ReadImage(str(es_image_path))
    es_label = sitk.ReadImage(str(es_label_path), outputPixelType=sitk.sitkUInt8)

    # unify labels
    ed_label = sitk.ChangeLabel(ed_label, ACDC_LABEL_MAP)
    es_label = sitk.ChangeLabel(es_label, ACDC_LABEL_MAP)

    # resample
    image = resample_spacing_4d(
        image=image,
        is_label=False,
        target_spacing=ACDC_SPACING,
    )
    orig_sax_spacing = ed_image.GetSpacing()
    data["original_sax_spacing_x"] = orig_sax_spacing[0]
    data["original_sax_spacing_y"] = orig_sax_spacing[1]
    data["original_sax_spacing_z"] = orig_sax_spacing[2]
    ed_image = resample_spacing_3d(
        image=ed_image,
        is_label=False,
        target_spacing=ACDC_SPACING,
    )
    ed_label = resample_spacing_3d(
        image=ed_label,
        is_label=True,
        target_spacing=ACDC_SPACING,
    )
    es_image = resample_spacing_3d(
        image=es_image,
        is_label=False,
        target_spacing=ACDC_SPACING,
    )
    es_label = resample_spacing_3d(
        image=es_label,
        is_label=True,
        target_spacing=ACDC_SPACING,
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
        target_size=(*ACDC_SAX_SLICE_SIZE, n_slices),
    )
    image = crop_4d(image, crop_lower, crop_upper)
    ed_image = sitk.Crop(ed_image, crop_lower, crop_upper)
    ed_label = sitk.Crop(ed_label, crop_lower, crop_upper)
    es_image = sitk.Crop(es_image, crop_lower, crop_upper)
    es_label = sitk.Crop(es_label, crop_lower, crop_upper)

    # calculate EDV, ESV, EF for LV and RV, ml = 1000 mm^3
    data["lv_edv"] = sitk.GetArrayFromImage(ed_label == LV_LABEL).sum() * np.prod(ACDC_SPACING) / 1000.0
    data["lv_esv"] = sitk.GetArrayFromImage(es_label == LV_LABEL).sum() * np.prod(ACDC_SPACING) / 1000.0
    data["lv_ef"] = ejection_fraction(edv=data["lv_edv"], esv=data["lv_esv"])
    data["rv_edv"] = sitk.GetArrayFromImage(ed_label == RV_LABEL).sum() * np.prod(ACDC_SPACING) / 1000.0
    data["rv_esv"] = sitk.GetArrayFromImage(es_label == RV_LABEL).sum() * np.prod(ACDC_SPACING) / 1000.0
    data["rv_ef"] = ejection_fraction(edv=data["rv_edv"], esv=data["rv_esv"])

    # normalise intensity
    image = clip_and_normalise_intensity_4d(image, intensity_range=None)
    ed_image = clip_and_normalise_intensity_3d(ed_image, intensity_range=None)
    es_image = clip_and_normalise_intensity_3d(es_image, intensity_range=None)

    # out
    out_dir = out_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / f"{pid}_sax_t.nii.gz"
    ed_image_path = out_dir / f"{pid}_sax_ed.nii.gz"
    ed_label_path = out_dir / f"{pid}_sax_ed_gt.nii.gz"
    es_image_path = out_dir / f"{pid}_sax_es.nii.gz"
    es_label_path = out_dir / f"{pid}_sax_es_gt.nii.gz"
    sitk.WriteImage(
        image=cast_to_uint8(image),
        fileName=image_path,
        useCompression=True,
    )
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


def preprocess_split(split_dir: Path, split: str, out_dir: Path) -> None:
    """Preprocess the split data.

    Args:
        split_dir: path to the split directory.
        split: split name, train or test.
        out_dir: output directory.
    """
    config_paths = list(split_dir.glob("*/Info.cfg"))
    meta_df = pd.DataFrame([preprocess_pid(x, out_dir / split) for x in tqdm(config_paths)])
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
        default="database",
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
    """Preprocess the ACDC dataset."""
    args = parse_args()
    preprocess_split(args.data_dir / "training", "train", args.out_dir)
    preprocess_split(args.data_dir / "testing", "test", args.out_dir)


if __name__ == "__main__":
    main()
