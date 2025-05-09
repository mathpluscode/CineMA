"""Script to preprocess kaggle DICOM data."""

from __future__ import annotations

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813

from cinema.data.dicom import load_dicom_folder
from cinema.data.kaggle import KAGGLE_LAX_SLICE_SIZE, KAGGLE_SAX_SLICE_SIZE, KAGGLE_SPACING
from cinema.data.sitk import (
    cast_to_uint8,
    clip_and_normalise_intensity_4d,
    crop_xy_4d,
    get_origin_for_crop,
    get_sax_center,
    resample_spacing_4d,
)
from cinema.log import get_logger
from cinema.metric import ejection_fraction

logger = get_logger(__name__)

PIDS_TO_SKIP = [
    761,  # all black images
]


def find_longest_consecutive_subseq_with_same_values(
    lst: list[float | np.ndarray] | np.ndarray,
) -> tuple[int, int]:
    """Find longest consecutive subsequence with same values.

    Args:
        lst: list of values.

    Returns:
        start index and length of the longest consecutive subsequence.
    """
    best_n = 0
    n = 0
    best_start = -1
    start = -1
    for i, x in enumerate(lst):
        if i > 0 and np.all(x == lst[i - 1]):
            n += 1
        else:
            n = 1
            start = i
        if n > best_n:
            best_n = n
            best_start = start
    return best_start, best_n


def filter_sax_images(sax_images: list[sitk.Image], decimals: int) -> list[sitk.Image]:
    """Filter SAX images.

    Need to find consecutive slices such that
    the difference between the origins are consistent, this equals the spacing at z-axis,
    then the spacing should be the same,
    then need to align the directions, select the most comment direction.

    Args:
        sax_images: list of SAX images, GetSize() = (x, y, t).
        decimals: number of decimals to round the values.

    Returns:
        list of filtered SAX images, GetSize() = (x, y, t).
    """
    # check SAX image sizes
    sax_image_sizes = np.array([sax_image.GetSize() for sax_image in sax_images])
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_image_sizes)
    sax_images = sax_images[start_index : start_index + slice_len]

    # check SAX pixel spacing
    # (n_slices, 3)
    sax_pixel_spacings = np.round(np.array([image.GetSpacing() for image in sax_images]), decimals)
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_pixel_spacings)
    sax_images = sax_images[start_index : start_index + slice_len]

    # check SAX slice direction
    # (n_slices, 9)
    sax_directions = np.round(np.array([image.GetDirection() for image in sax_images]), decimals)
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_directions)
    sax_images = sax_images[start_index : start_index + slice_len]

    # check SAX slice spacing
    # (n_slices, 3)
    sax_origins = np.array([image.GetOrigin() for image in sax_images])
    # (n_slices-1,)
    sax_slice_spacings = np.round(np.linalg.norm(np.diff(sax_origins, axis=0), axis=-1), decimals)
    start_index, slice_len = find_longest_consecutive_subseq_with_same_values(sax_slice_spacings)
    sax_images = sax_images[start_index : start_index + slice_len + 1]  # +1 as the seq is on difference

    return sax_images


def process_study(  # pylint:disable=too-many-statements
    study_dir: Path,
    pid: str,
    out_dir: Path,
    spacing: tuple[float, ...] = KAGGLE_SPACING,
    lax_slice_size: tuple[int, int] = KAGGLE_LAX_SLICE_SIZE,
    sax_slice_size: tuple[int, int] = KAGGLE_SAX_SLICE_SIZE,
) -> dict[str, int]:
    """Process DICOM in study and save nifti files.

    Args:
        study_dir: directory of the study.
        pid: unique id of the study.
        out_dir: output directory.
        spacing: target spacing.
        lax_slice_size: slice size for cropping LAX images.
        sax_slice_size: slice size for cropping SAX images.

    Returns:
        dictionary of metadata, including pid, the number of slices and number of time frames.
    """
    # load 2c/4c images
    dir_2c = next(study_dir.glob("2ch_*"))
    dir_4c = next(study_dir.glob("4ch_*"))
    lax_2c_image = load_dicom_folder([dir_2c], study_dir / "lax_2c.nii.gz")  # GetSize() = (x, y, 1, t)
    lax_4c_image = load_dicom_folder([dir_4c], study_dir / "lax_4c.nii.gz")  # GetSize() = (x, y, 1, t)

    # load sax images
    sax_dirs = sorted(study_dir.glob("sax_*"), key=lambda x: int(x.name.split("sax_")[1]))
    sax_image = load_dicom_folder(sax_dirs, study_dir / "sax.nii.gz")

    # resample images
    lax_2c_image = resample_spacing_4d(
        image=lax_2c_image,
        is_label=False,
        target_spacing=(*spacing[:2], lax_2c_image.GetSpacing()[-2]),
    )
    lax_4c_image = resample_spacing_4d(
        image=lax_4c_image,
        is_label=False,
        target_spacing=(*spacing[:2], lax_4c_image.GetSpacing()[-2]),
    )
    orig_sax_spacing = sax_image.GetSpacing()[:3]  # (x, y, z)
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
        raise ValueError(f"Failed to get the center of 2C/4C/SAX images for cropping for {pid}.")
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
    lax_4c_image = clip_and_normalise_intensity_4d(lax_4c_image, intensity_range=None)
    sax_image = clip_and_normalise_intensity_4d(sax_image, intensity_range=None)

    # save the cropped nifti file
    out_dir = out_dir / pid
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(
        image=cast_to_uint8(lax_2c_image),
        fileName=out_dir / f"{pid}_lax_2c_t.nii.gz",
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(lax_4c_image),
        fileName=out_dir / f"{pid}_lax_4c_t.nii.gz",
        useCompression=True,
    )
    sitk.WriteImage(
        image=cast_to_uint8(sax_image),
        fileName=out_dir / f"{pid}_sax_t.nii.gz",
        useCompression=True,
    )

    return {
        "pid": int(pid),
        "n_slices": sax_image.GetSize()[2],
        # for sample 416, the SAX image has 30 frames, but the LAX images have 21 frames
        "n_frames": min(sax_image.GetSize()[-1], lax_4c_image.GetSize()[-1], lax_2c_image.GetSize()[-1]),
        "original_sax_spacing_x": orig_sax_spacing[0],
        "original_sax_spacing_y": orig_sax_spacing[1],
        "original_sax_spacing_z": orig_sax_spacing[2],
    }


def try_process_study(study_dir: Path, pid: str, out_dir: Path) -> dict[str, int]:
    """Try to process a study and log error if failed.

    Args:
        study_dir: directory of the study.
        pid: unique id of the study.
        out_dir: output directory.

    Returns:
        dictionary of metadata, including pid, the number of slices and number of time frames.
    """
    try:
        return process_study(study_dir, pid, out_dir)
    except Exception:  # pylint: disable=broad-except
        logger.exception(f"Failed to process {pid} for {study_dir}.")
    return {}


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Folder of data.",
        default="second-annual-data-science-bowl",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Folder saving output files.",
        default="processed",
    )
    parser.add_argument(
        "--max_n_cpus",
        type=int,
        help="Maximum number of cpus to use.",
        default=4,
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Main function."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_cpus = min(cpu_count(), args.max_n_cpus)

    for split in ["train", "validate", "test"]:
        out_split = "val" if split == "validate" else split
        logger.info(f"Processing {split} split.")
        split_dir = args.data_dir / split / split
        study_dirs = list(split_dir.glob("*/study"))
        study_dirs = [study_dir for study_dir in study_dirs if int(study_dir.parent.name) not in PIDS_TO_SKIP]

        with Pool(n_cpus) as p:
            results = p.starmap_async(
                try_process_study,
                [
                    (
                        study_dir,
                        study_dir.parent.name,
                        args.out_dir / out_split,
                    )
                    for study_dir in study_dirs
                ],
            )
            data = [x for x in results.get() if len(x) > 0]

        # merge label and save metadata
        if split == "test":
            label_df = pd.read_csv(args.data_dir / "solution.csv")
            label_df["phase"] = label_df["Id"].apply(lambda x: x.split("_")[1])
            label_df["Id"] = label_df["Id"].apply(lambda x: int(x.split("_")[0]))
            label_df = label_df.pivot_table(index="Id", columns="phase", values="Volume").reset_index()
        else:
            label_df = pd.read_csv(args.data_dir / f"{split}.csv")
        label_df = label_df.rename(
            columns={
                "Id": "pid",
                "Systole": "systole_volume",
                "Diastole": "diastole_volume",
            },
            errors="raise",
        )
        label_df["ef"] = ejection_fraction(edv=label_df["diastole_volume"], esv=label_df["systole_volume"])
        meta_df = pd.DataFrame(data).sort_values("pid")
        meta_df = meta_df.merge(label_df, on="pid", how="left")
        meta_df_path = args.out_dir / f"{out_split}_metadata.csv"
        meta_df.to_csv(meta_df_path, index=False)
        logger.info(f"Saved metadata to {meta_df_path}.")


if __name__ == "__main__":
    main()
