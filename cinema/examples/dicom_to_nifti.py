"""Script to convert UK Biobank DICOM to NIfTI and crop.

Reference: https://github.com/baiwenjia/ukbb_cardiac
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813

from cinema import UKB_LAX_SLICE_SIZE, UKB_SAX_SLICE_SIZE, UKB_SPACING
from cinema.data.dicom import load_dicom_folder
from cinema.data.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_4d,
    crop_xy_4d,
    get_origin_for_crop,
    get_sax_center,
    resample_spacing_4d,
)
from cinema.log import get_logger

logger = get_logger(__name__)


def point_to_plane_projection(
    point: np.ndarray,
    plane_origin: np.ndarray,
    plane_norm_vec: np.ndarray,
) -> np.ndarray:
    """Project a point to a plane.

    Args:
        point: a point, (3,).
        plane_origin: origin on the plane, (3,).
        plane_norm_vec: norm vector of the plane, (3,).

    Returns:
        projected_point: a point on the plane, (3,).
    """
    distance = np.dot(point - plane_origin, plane_norm_vec)
    return point - distance * plane_norm_vec


def date_repl(m: re.Match[str]) -> str:
    """Function for reformatting the date.

    Aug 30, 2015 -> 30-Aug-2015
    group1: A
    group2: ug
    group3: 30
    group4: 15

    Args:
        m: re.Match object.

    Returns:
        reformatted date.
    """
    return f"{m.group(3)}-{m.group(1)}{m.group(2)}-20{m.group(4)}"


def fix_manifest(manifest_path: Path, fixed_manifest_path: Path) -> None:
    """Fix the date format in the manifest file.

    Read the lines in the manifest.csv file and check whether the date format contains
    a comma, which needs to be removed since it causes problems in parsing the file.

    Args:
        manifest_path: path to the manifest file.
        fixed_manifest_path: path to the fixed manifest file.
    """
    with (
        Path.open(fixed_manifest_path, "w", encoding="utf-8") as f_fixed,
        Path.open(manifest_path, encoding="utf-8") as f,
    ):
        for line in f:
            line_fixed = re.sub(r"([A-Z])(\w{2}) (\d{1,2}), 20(\d{2})", date_repl, line)
            f_fixed.write(line_fixed)


def find_fix_and_read_manifest(unzip_dir: Path, out_path: Path) -> pd.DataFrame | None:
    """Find, fix and read manifest file.

    Args:
        unzip_dir: directory to search for manifest file.
        out_path: path to store fixed manifest file.

    Returns:
        manifest dataframe or None.
    """
    manifest_paths = list(unzip_dir.glob("manifest.*"))
    if len(manifest_paths) == 0:
        logger.error(f"Failed to find manifest in {unzip_dir}.")
        return None
    if len(manifest_paths) > 1:
        logger.error(f"Found multiple manifest in {unzip_dir}, using the first found {manifest_paths[0]}.")
    manifest_path = manifest_paths[0]
    fix_manifest(manifest_path, out_path)
    return pd.read_csv(out_path)


def split_dicom_files_and_convert_to_nifti(
    dicom_dir: Path,
    nifti_dir: Path,
    eid: str,
    instance_id: str,
    suffix: str,
) -> tuple[dict[str, sitk.Image], pd.DataFrame]:
    """Move dicom files to separate folders based on series discription.

    Args:
        dicom_dir: directory having manifest.cvs or manifest.csv
        nifti_dir: directory to store nifti files for the participant.
        eid: participant identifier, e.g. 6000182.
        instance_id: instance id, 2_0 or 3_0.
        suffix: suffix to add to manifest file.

    Returns:
        - dict mapping series name to image.
        - manifest dataframe.
    """
    fixed_manifest_path = dicom_dir / f"{eid}_{instance_id}_manifest_{suffix}.csv"
    manifest_df = find_fix_and_read_manifest(dicom_dir, fixed_manifest_path)
    if manifest_df is None:
        raise ValueError(f"Failed to find manifest in {dicom_dir}.")

    # split dicom files
    for series_name, series_df in manifest_df.groupby("series discription"):
        # known problems
        if "InlineVF" in series_name:
            continue
        if "Inline_VF_Results" in series_name:
            continue
        series_dir = dicom_dir / str(series_name)
        series_dir.mkdir(parents=True, exist_ok=True)
        for fname in series_df["filename"]:
            shutil.copy(dicom_dir / fname, series_dir / fname)

    # load dicom folder
    series_name_to_image = {}
    if suffix == "lax":
        for series_name in manifest_df["series discription"].unique():
            series_dir = dicom_dir / str(series_name)
            nii_path = nifti_dir / f"{eid}_{instance_id}_{series_name}.nii.gz"
            image = load_dicom_folder([series_dir], nii_path)
            series_name_to_image[series_name] = image
    else:
        series_names = [x for x in manifest_df["series discription"].unique() if "CINE_segmented_SAX_b" in x]
        series_names = sorted(series_names, key=lambda x: int(re.match(r"CINE_segmented_SAX_b(\d*)$", x).group(1)))  # type: ignore[union-attr]
        series_dirs = [dicom_dir / x for x in series_names]
        nii_path = nifti_dir / f"{eid}_{instance_id}_CINE_segmented_SAX.nii.gz"
        image = load_dicom_folder(series_dirs, nii_path)
        series_name_to_image["CINE_segmented_SAX"] = image

    # store the fixed manifest file in the same directory as the nifti files
    # only after the conversion is successful
    fixed_manifest_path = nifti_dir / f"{eid}_{instance_id}_manifest_{suffix}.csv"
    manifest_df.to_csv(fixed_manifest_path, index=False)

    return series_name_to_image, manifest_df


@dataclass
class EIDData:
    """Data for a single participant."""

    eid: str
    instance_id: str
    lax_2c_image: sitk.Image  # (x, y, 1, t)
    lax_3c_image: sitk.Image  # (x, y, 1, t)
    lax_4c_image: sitk.Image  # (x, y, 1, t)
    sax_image: sitk.Image  # (x, y, z, t)


def get_sax_series(sax_manifest_df: pd.DataFrame, folder_id: str) -> list[int]:
    """Get the series number for SAX images.

    Args:
        sax_manifest_df: manifest dataframe for SAX images.
        folder_id: folder identifier, e.g. 6000182_2_0.

    Returns:
        list of series numbers.
    """
    series = sax_manifest_df["series discription"].unique()
    nums = sorted([int(x.replace("CINE_segmented_SAX_b", "")) for x in series if x.startswith("CINE_segmented_SAX_b")])
    if set(nums) != set(range(1, len(nums) + 1)):
        raise ValueError(f"SAX files are not continuous for {folder_id}: got series discription for {nums}.")
    return nums


def transform_to_nifti(
    lax_dicom_dir: Path,
    sax_dicom_dir: Path,
    out_dir: Path,
) -> EIDData:
    """Transform dicom to nifti for a single participant.

    Args:
        lax_dicom_dir: path to dicom images of LAX views.
        sax_dicom_dir: path to dicom images of SAX views.
        out_dir: path to output nifti directory.

    Returns:
        EIDData object.
    """
    # eid_20208_2_0 -> eid and 2
    eid = lax_dicom_dir.stem.split("_")[0]
    instance_id = lax_dicom_dir.stem.split("_")[-2]

    folder_id = f"{eid}_{instance_id}"
    nifti_dir = out_dir / folder_id
    nifti_dir.mkdir(parents=True, exist_ok=True)

    # load manifest, split dicom files and convert to nifti
    lax_series_name_to_image, _ = split_dicom_files_and_convert_to_nifti(
        dicom_dir=lax_dicom_dir,
        nifti_dir=nifti_dir,
        eid=eid,
        instance_id=instance_id,
        suffix="lax",
    )
    for i in [2, 3, 4]:
        if f"CINE_segmented_LAX_{i}Ch" not in lax_series_name_to_image:
            raise ValueError(f"LAX {i}C file for {folder_id} is not loaded.")
    lax_2c_image = lax_series_name_to_image["CINE_segmented_LAX_2Ch"]
    lax_3c_image = lax_series_name_to_image["CINE_segmented_LAX_3Ch"]
    lax_4c_image = lax_series_name_to_image["CINE_segmented_LAX_4Ch"]

    sax_series_name_to_image, _ = split_dicom_files_and_convert_to_nifti(
        dicom_dir=sax_dicom_dir,
        nifti_dir=nifti_dir,
        eid=eid,
        instance_id=instance_id,
        suffix="sax",
    )
    sax_image = sax_series_name_to_image["CINE_segmented_SAX"]

    return EIDData(
        eid=eid,
        instance_id=instance_id,
        lax_2c_image=lax_2c_image,
        lax_3c_image=lax_3c_image,
        lax_4c_image=lax_4c_image,
        sax_image=sax_image,
    )


def crop_nifti(
    data: EIDData,
    out_dir: Path,
    spacing: tuple[float, ...] = UKB_SPACING,
    lax_slice_size: tuple[int, int] = UKB_LAX_SLICE_SIZE,
    sax_slice_size: tuple[int, int] = UKB_SAX_SLICE_SIZE,
) -> None:
    """Crop nifti files for a single participant.

    Args:
        data: EIDData object.
        out_dir: output directory.
        spacing: spacing at X, Y, Z axis.
        lax_slice_size: size of the slice at X, Y axis for LAX images.
        sax_slice_size: size of the slice at X, Y axis for SAX images.
    """
    if len(spacing) != 3:
        raise ValueError(f"Spacing should have 3 elements, got {spacing}.")

    lax_2c_image = data.lax_2c_image
    lax_3c_image = data.lax_3c_image
    lax_4c_image = data.lax_4c_image
    sax_image = data.sax_image

    # resample images
    lax_2c_image = resample_spacing_4d(
        image=lax_2c_image,
        is_label=False,
        target_spacing=(*spacing[:2], lax_2c_image.GetSpacing()[-2]),
    )
    lax_3c_image = resample_spacing_4d(
        image=lax_3c_image,
        is_label=False,
        target_spacing=(*spacing[:2], lax_3c_image.GetSpacing()[-2]),
    )
    lax_4c_image = resample_spacing_4d(
        image=lax_4c_image,
        is_label=False,
        target_spacing=(*spacing[:2], lax_4c_image.GetSpacing()[-2]),
    )
    sax_image = resample_spacing_4d(
        image=sax_image,
        is_label=False,
        target_spacing=spacing,
    )

    # get 2C/4C/SAX center
    sax_center = get_sax_center(
        sax_image=sax_image,
        lax_2c_image=lax_2c_image,
        lax_4c_image=lax_4c_image,
    )
    if sax_center is None:
        raise ValueError("Failed to get the center of 2C/4C/SAX images for cropping.")
    lax_3c_rot = np.array(lax_3c_image.GetDirection()).reshape((4, 4))[:3, :3]
    lax_3c_center = point_to_plane_projection(
        point=sax_center,
        plane_origin=lax_3c_image.GetOrigin()[:3],
        plane_norm_vec=lax_3c_rot[:, -1],
    )

    # crop 2C, 3C, 4C and SAX
    lax_2c_origin_indices = get_origin_for_crop(
        center=sax_center,
        image=lax_2c_image,
        slice_size=lax_slice_size,
    )
    lax_2c_image = crop_xy_4d(
        image=lax_2c_image,
        origin_indices=lax_2c_origin_indices,
        slice_size=lax_slice_size,
    )
    lax_3c_origin_indices = get_origin_for_crop(
        center=lax_3c_center,
        image=lax_3c_image,
        slice_size=lax_slice_size,
    )
    lax_3c_image = crop_xy_4d(
        image=lax_3c_image,
        origin_indices=lax_3c_origin_indices,
        slice_size=lax_slice_size,
    )
    lax_4c_origin_indices = get_origin_for_crop(
        center=sax_center,
        image=lax_4c_image,
        slice_size=lax_slice_size,
    )
    lax_4c_image = crop_xy_4d(
        image=lax_4c_image,
        origin_indices=lax_4c_origin_indices,
        slice_size=lax_slice_size,
    )
    sax_origin_indices = get_origin_for_crop(
        center=sax_center,
        image=sax_image,
        slice_size=sax_slice_size,
    )
    sax_image = crop_xy_4d(
        image=sax_image,
        origin_indices=sax_origin_indices,
        slice_size=sax_slice_size,
    )

    # normalise the intensity
    lax_2c_image = clip_and_normalise_intensity_4d(lax_2c_image, intensity_range=None)
    lax_3c_image = clip_and_normalise_intensity_4d(lax_3c_image, intensity_range=None)
    lax_4c_image = clip_and_normalise_intensity_4d(lax_4c_image, intensity_range=None)
    sax_image = clip_and_normalise_intensity_4d(sax_image, intensity_range=None)

    # save the cropped nifti file
    folder_id = f"{data.eid}_{data.instance_id}"
    nifti_dir = out_dir / folder_id
    nifti_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(
        image=cast_to_uint8(lax_2c_image),
        fileName=nifti_dir / f"{folder_id}_lax_2c.nii.gz",
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(lax_3c_image),
        fileName=nifti_dir / f"{folder_id}_lax_3c.nii.gz",
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(lax_4c_image),
        fileName=nifti_dir / f"{folder_id}_lax_4c.nii.gz",
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(sax_image),
        fileName=nifti_dir / f"{folder_id}_sax.nii.gz",
        useCompression=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lax_dicom_dir",
        type=Path,
        help="Folder saving dicom files, folder name such as eid_20208_2_0 or eid_20208_3_0.",
        required=True,
    )
    parser.add_argument(
        "--sax_dicom_dir",
        type=Path,
        help="Folder saving dicom files, folder name such as eid_20209_2_0 or eid_20209_3_0.",
        required=True,
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Folder saving output nifti files.",
        required=True,
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function.

    This script will store
    - intermediate files under out_dir / eid_instance_id.
    - processed nifti files under out_dir, with names such as eid_2_lax_2c.nii.gz
    """
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = transform_to_nifti(
        lax_dicom_dir=args.lax_dicom_dir,
        sax_dicom_dir=args.sax_dicom_dir,
        out_dir=out_dir,
    )
    crop_nifti(
        data=data,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
