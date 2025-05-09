"""Script to parse pickle files."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk  # noqa: N813
from tqdm import tqdm

from cinema import LV_LABEL
from cinema.data.dicom import dicom_orientation_to_rotation_matrix
from cinema.data.rescan import RESCAN_LABEL_MAP, RESCAN_LAX_SLICE_SIZE, RESCAN_SAX_SLICE_SIZE, RESCAN_SPACING
from cinema.data.sitk import (
    clip_and_normalise_intensity_4d,
    crop_xy_4d,
    get_origin_for_crop,
    get_sax_center,
    resample_spacing_3d,
)
from cinema.log import get_logger
from cinema.metric import ejection_fraction

logger = get_logger(__name__)


def load_pickle(path: Path) -> dict[str, np.ndarray]:
    """Load pickle file.

    Args:
        path: path to the pickle file.

    Returns:
        dictionary of numpy arrays.
    """
    with Path.open(path, mode="rb") as f:  # pylint: disable=unspecified-encoding
        return pickle.load(f)


def sax_image_label_to_nifti(
    sax: dict[str, np.ndarray],
    sax_label: dict[str, np.ndarray],
) -> tuple[sitk.Image, sitk.Image]:
    """Convert SAX image and label to sitk Image and store as nifti files.

    Args:
        sax: dictionary of numpy arrays for image.
        sax_label: dictionary of numpy arrays for label.

    Returns:
        SAX image and label.
    """
    # (z, t, y, x)
    # (11, 25, 256, 208)
    image_voxels = sax["image_voxels"]
    image_segmentation = sax_label["image_segmentation"]
    num_frames = image_voxels.shape[1]

    image_frames = []
    label_frames = []
    for i in range(num_frames):
        # (11, 256, 208)
        image_slice = image_voxels[::-1, i, :, :]
        label_slice = image_segmentation[::-1, i, :, :]

        # process image
        origin = sax["ImagePositionPatient"][-1, :]
        rotation = dicom_orientation_to_rotation_matrix(sax["ImageOrientationPatient"])
        spacing = np.array([sax["PixelSpacing"][0], sax["PixelSpacing"][1], sax["SliceSpacing"]])  # (x, y, z)

        image = sitk.GetImageFromArray(image_slice)  # image.GetSize() = (208, 256, 11)
        image.SetOrigin(origin)
        image.SetDirection(rotation.reshape(-1))
        image.SetSpacing(spacing)

        # process label
        label = sitk.GetImageFromArray(label_slice)  # label.GetSize() = (208, 256, 11)
        label = sitk.Cast(label, sitk.sitkUInt8)
        label.CopyInformation(image)
        label = sitk.ChangeLabel(label, RESCAN_LABEL_MAP)

        # resample
        image = resample_spacing_3d(
            image=image,
            is_label=False,
            target_spacing=RESCAN_SPACING,
        )
        label = resample_spacing_3d(
            image=label,
            is_label=True,
            target_spacing=RESCAN_SPACING,
        )

        image_frames.append(image)
        label_frames.append(label)

    image = sitk.JoinSeries(image_frames)  # (x, y, z, t)
    label = sitk.JoinSeries(label_frames)
    return image, label


def sax_image_to_nifti(sax: dict[str, np.ndarray]) -> sitk.Image:
    """Convert SAX image to sitk Image and store as nifti files.

    Args:
        sax: dictionary of numpy arrays for image.

    Returns:
        SAX image.
    """
    # (z, t, y, x)
    # (11, 25, 256, 208)
    image_voxels = sax["image_voxels"]
    num_frames = image_voxels.shape[1]

    image_frames = []
    for i in range(num_frames):
        # (11, 256, 208)
        image_slice = image_voxels[::-1, i, :, :]

        # process image
        origin = sax["ImagePositionPatient"][-1, :]
        rotation = dicom_orientation_to_rotation_matrix(sax["ImageOrientationPatient"])
        spacing = np.array([sax["PixelSpacing"][0], sax["PixelSpacing"][1], sax["SliceSpacing"]])  # (x, y, z)

        image = sitk.GetImageFromArray(image_slice)  # image.GetSize() = (208, 256, 11)
        image.SetOrigin(origin)
        image.SetDirection(rotation.reshape(-1))
        image.SetSpacing(spacing)

        # resample
        image = resample_spacing_3d(
            image=image,
            is_label=False,
            target_spacing=RESCAN_SPACING,
        )
        image_frames.append(image)

    return sitk.JoinSeries(image_frames)  # (x, y, z, t)


def lax_to_nifti(
    x: dict[str, np.ndarray],
    slice_spacing: float = 1.0,
) -> sitk.Image:
    """Convert 2C/4C image to sitk Image and store as nifti files.

    Args:
        x: dictionary of numpy arrays.
        slice_spacing: default slice spacing.

    Returns:
        LAX image.
    """
    # (t, y, x)
    # (25, 256, 208)
    image_voxels = x["image_voxels"]
    num_frames = image_voxels.shape[0]

    image_frames = []
    for i in range(num_frames):
        # (1, 256, 208)
        image_slice = image_voxels[i : i + 1, :, :]

        origin = x["ImagePositionPatient"]
        rotation = dicom_orientation_to_rotation_matrix(x["ImageOrientationPatient"])
        spacing = np.array([x["PixelSpacing"][0], x["PixelSpacing"][1], slice_spacing])

        # image.GetSize() = (208, 256, 1)
        image = sitk.GetImageFromArray(image_slice)
        image.SetOrigin(origin)
        image.SetDirection(rotation.reshape(-1))
        image.SetSpacing(spacing)

        # resample
        image = resample_spacing_3d(
            image=image,
            is_label=False,
            target_spacing=(*RESCAN_SPACING[:2], image.GetSpacing()[-1]),
        )

        image_frames.append(image)

    image = sitk.JoinSeries(image_frames)
    return image


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Folder of data.",
        default="pickle",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Folder for output.",
        default="processed",
    )
    args = parser.parse_args()

    return args


def crop(
    sax_image: sitk.Image, sax_label: sitk.Image | None, lax_2c_image: sitk.Image, lax_4c_image: sitk.Image
) -> tuple[sitk.Image, sitk.Image, sitk.Image, sitk.Image]:
    """Crop the images to 2C, 4C and SAX intersection center.

    Args:
        sax_image: 4D image of SAX, (x, y, z, t).
        sax_label: 4D label of SAX, (x, y, z, t), or None
        lax_2c_image: 4D image of 2C, (x, y, 1, t).
        lax_4c_image: 4D image of 4C, (x, y, 1, t).

    Returns:
        cropped images.
    """
    sax_center = get_sax_center(
        sax_image=sax_image[..., 0],
        lax_2c_image=lax_2c_image[..., 0],
        lax_4c_image=lax_4c_image[..., 0],
    )
    if sax_center is None:
        raise ValueError("Failed to get SAX center.")

    # crop 2C, 3C, 4C and SAX
    lax_2c_origin_indices = get_origin_for_crop(
        center=sax_center,
        image=lax_2c_image[..., 0],
        slice_size=RESCAN_LAX_SLICE_SIZE,
    )
    lax_2c_image = crop_xy_4d(
        image=lax_2c_image,
        origin_indices=lax_2c_origin_indices,
        slice_size=RESCAN_LAX_SLICE_SIZE,
    )
    lax_4c_origin_indices = get_origin_for_crop(
        center=sax_center,
        image=lax_4c_image[..., 0],
        slice_size=RESCAN_LAX_SLICE_SIZE,
    )
    lax_4c_image = crop_xy_4d(
        image=lax_4c_image,
        origin_indices=lax_4c_origin_indices,
        slice_size=RESCAN_LAX_SLICE_SIZE,
    )
    sax_origin_indices = get_origin_for_crop(
        center=sax_center,
        image=sax_image[..., 0],
        slice_size=RESCAN_SAX_SLICE_SIZE,
    )
    sax_image = crop_xy_4d(
        image=sax_image,
        origin_indices=sax_origin_indices,
        slice_size=RESCAN_SAX_SLICE_SIZE,
    )
    if sax_label:  # may be None
        sax_label = crop_xy_4d(
            image=sax_label,
            origin_indices=sax_origin_indices,
            slice_size=RESCAN_SAX_SLICE_SIZE,
        )

    # normalise the intensity
    lax_2c_image = clip_and_normalise_intensity_4d(lax_2c_image, intensity_range=None)
    lax_4c_image = clip_and_normalise_intensity_4d(lax_4c_image, intensity_range=None)
    sax_image = clip_and_normalise_intensity_4d(sax_image, intensity_range=None)

    return sax_image, sax_label, lax_2c_image, lax_4c_image


def process(data_dir: Path, out_dir: Path, split: str) -> None:
    """Process pickle files.

    Args:
        data_dir: folder of data.
        out_dir: folder for output.
        split: train or test.
    """
    data_df_path = out_dir / f"{split}_metadata.csv"
    data_dir = data_dir / split
    out_dir = out_dir / split

    data_lst = []
    paths_lax_2c = list(data_dir.glob("**/2C.pickle"))
    paths_lax_4c = list(data_dir.glob("**/4C.pickle"))
    paths_sax = list(data_dir.glob("**/SAX.pickle"))
    paths_sax_label = list(data_dir.glob("**/SAX_segs.pickle"))

    # see if any files are missing
    folder_paths = sorted(
        set(
            [path.parent for path in paths_lax_2c]
            + [path.parent for path in paths_lax_4c]
            + [path.parent for path in paths_sax]
            + [path.parent for path in paths_sax_label]
        )
    )
    for folder_path in tqdm(folder_paths):
        # scan_02_B or G/s_0053
        relative_path = folder_path.relative_to(data_dir)

        path_2c = folder_path / "2C.pickle"
        path_4c = folder_path / "4C.pickle"
        path_sax = folder_path / "SAX.pickle"
        path_sax_label = folder_path / "SAX_segs.pickle"

        if not path_2c.exists():
            logger.error(f"{path_2c} does not exist.")
            continue
        if not path_4c.exists():
            logger.error(f"{path_4c} does not exist.")
            continue
        if not path_sax.exists():
            logger.error(f"{path_sax} does not exist.")
            continue
        if not path_sax_label.exists():
            logger.error(f"{path_sax_label} does not exist.")
            continue

        lax_2c = load_pickle(path_2c)
        lax_4c = load_pickle(path_4c)
        sax = load_pickle(path_sax)
        sax_label = load_pickle(path_sax_label)
        if any(len(x) == 0 for x in [lax_2c, lax_4c, sax, sax_label]):
            logger.error(f"Failed to load pickle files for scan {relative_path}.")
            continue
        slice_spacing = float(sax["SliceSpacing"])

        sax_image, sax_label = sax_image_label_to_nifti(sax, sax_label)  # (x, y, z, t)
        lax_2c_image = lax_to_nifti(lax_2c, slice_spacing)  # (x, y, 1, t)
        lax_4c_image = lax_to_nifti(lax_4c, slice_spacing)  # (x, y, 1, t)

        sax_image, sax_label, lax_2c_image, lax_4c_image = crop(
            sax_image=sax_image,
            sax_label=sax_label,
            lax_2c_image=lax_2c_image,
            lax_4c_image=lax_4c_image,
        )
        # get ed/es indices
        sax_label_np = sitk.GetArrayFromImage(sax_label)  # (t, z, y, x)
        lv_volumes = np.sum(sax_label_np == LV_LABEL, axis=(1, 2, 3))
        ed_index = np.argmax(lv_volumes)
        es_index = np.argmin(lv_volumes)
        data_lst.append(
            {
                "pid": str(relative_path),
                "orig_sax_spacing_x": float(sax["PixelSpacing"][0]),
                "orig_sax_spacing_y": float(sax["PixelSpacing"][1]),
                "orig_sax_spacing_z": float(sax["SliceSpacing"]),
                "orig_lax_spacing_x": float(lax_2c["PixelSpacing"][0]),
                "orig_lax_spacing_y": float(lax_2c["PixelSpacing"][1]),
                "n_slices": sax_image.GetSize()[2],
                "n_frames": sax_image.GetSize()[3],
                "ed_index": ed_index,
                "es_index": es_index,
            }
        )

        out_dir_i = out_dir / relative_path
        out_dir_i.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(
            image=sax_image,
            fileName=out_dir_i / "sax_t.nii.gz",
            useCompression=True,
        )
        sitk.WriteImage(
            image=sax_label,
            fileName=out_dir_i / "sax_gt_t.nii.gz",
            useCompression=True,
        )
        sitk.WriteImage(
            image=lax_2c_image,
            fileName=out_dir_i / "lax_2c_t.nii.gz",
            useCompression=True,
        )
        sitk.WriteImage(
            image=lax_4c_image,
            fileName=out_dir_i / "lax_4c_t.nii.gz",
            useCompression=True,
        )

    data_df = pd.DataFrame(data_lst)
    data_df.to_csv(data_df_path, index=False)
    logger.info(f"Saved metadata to {data_df_path}.")


def process_paired(data_dir: Path, out_dir: Path, split: str) -> None:
    """Process pickle files.

    Args:
        data_dir: folder of data.
        out_dir: folder for output.
        split: train or test.
    """
    data_df_path = out_dir / f"{split}_metadata.csv"
    data_dir = data_dir / split
    out_dir = out_dir / split

    label_df = pd.read_csv(data_dir / "labels.csv")
    data_lst = []
    for i, row in tqdm(label_df.iterrows(), total=len(label_df)):
        id_a, id_b1 = int(row["A"]), int(row["B1"])
        ids = [id_a, id_b1]
        vs = "AB"
        if not np.isnan(row["B2"]):
            ids.append(int(row["B2"]))
            vs += "B"

        for j, v in zip(ids, vs, strict=False):
            pid = f"scan_{i:02d}_{v}"
            path_2c = data_dir / str(j) / "2C.pickle"
            path_4c = data_dir / str(j) / "4C.pickle"
            path_sax = data_dir / str(j) / "SAX.pickle"

            if not path_2c.exists():
                logger.error(f"{path_2c} does not exist.")
                continue
            if not path_4c.exists():
                logger.error(f"{path_4c} does not exist.")
                continue
            if not path_sax.exists():
                logger.error(f"{path_sax} does not exist.")
                continue

            lax_2c = load_pickle(path_2c)
            lax_4c = load_pickle(path_4c)
            sax = load_pickle(path_sax)
            if any(len(x) == 0 for x in [lax_2c, lax_4c, sax]):
                logger.error(f"Failed to load pickle files for scan {pid}.")
                continue
            slice_spacing = float(sax["SliceSpacing"])

            sax_image = sax_image_to_nifti(sax)  # (x, y, z, t)
            lax_2c_image = lax_to_nifti(lax_2c, slice_spacing)  # (x, y, 1, t)
            lax_4c_image = lax_to_nifti(lax_4c, slice_spacing)  # (x, y, 1, t)

            sax_image, _, lax_2c_image, lax_4c_image = crop(
                sax_image=sax_image,
                sax_label=None,
                lax_2c_image=lax_2c_image,
                lax_4c_image=lax_4c_image,
            )

            if v == "A":
                edv = row["EDV_A"]
                esv = row["ESV_A"]
            else:
                edv = np.nanmean(row[["EDV_B1", "EDV_B2"]].to_numpy())
                esv = np.nanmean(row[["ESV_B1", "ESV_B2"]].to_numpy())

            data_lst.append(
                {
                    "pid": pid,
                    "orig_sax_spacing_x": float(sax["PixelSpacing"][0]),
                    "orig_sax_spacing_y": float(sax["PixelSpacing"][1]),
                    "orig_sax_spacing_z": float(sax["SliceSpacing"]),
                    "orig_lax_spacing_x": float(lax_2c["PixelSpacing"][0]),
                    "orig_lax_spacing_y": float(lax_2c["PixelSpacing"][1]),
                    "n_slices": sax_image.GetSize()[2],
                    "n_frames": sax_image.GetSize()[3],
                    "edv": edv,
                    "esv": esv,
                    "ef": ejection_fraction(edv, esv),
                }
            )

            out_dir_i = out_dir / pid
            out_dir_i.mkdir(parents=True, exist_ok=True)

            sitk.WriteImage(
                image=sax_image,
                fileName=out_dir_i / "sax_t.nii.gz",
                useCompression=True,
            )
            sitk.WriteImage(
                image=lax_2c_image,
                fileName=out_dir_i / "lax_2c_t.nii.gz",
                useCompression=True,
            )
            sitk.WriteImage(
                image=lax_4c_image,
                fileName=out_dir_i / "lax_4c_t.nii.gz",
                useCompression=True,
            )

    data_df = pd.DataFrame(data_lst)
    data_df.to_csv(data_df_path, index=False)
    logger.info(f"Saved metadata to {data_df_path}.")


def main() -> None:
    """Main function."""
    args = parse_args()
    # process(args.data_dir, args.out_dir, split="train")
    # process(args.data_dir, args.out_dir, split="test")
    process_paired(args.data_dir, args.out_dir, split="test_retest_100")


if __name__ == "__main__":
    main()
