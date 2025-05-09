"""Utilities for working with DICOM files.

Code adapted from https://github.com/baiwenjia/ukbb_cardiac.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import SimpleITK as sitk  # noqa: N813
from pydicom import dcmread

from cinema.log import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def find_series(dcm_dir: Path) -> list[Path]:
    """Find the series with the most files.

    Args:
        dcm_dir: path to the DICOM directory.

    Returns:
        list of DICOM file paths, sorted.
    """
    files = sorted(dcm_dir.glob("*.dcm"))
    # Sort the files according to their series UIDs
    series: dict[str, list[Path]] = {}
    for f in files:
        d = dcmread(f)
        # Kaggle dataset does not have SeriesInstanceUID
        suid = d.SeriesInstanceUID if hasattr(d, "SeriesInstanceUID") else "suid"
        if suid in series:
            series[suid] += [f]
        else:
            series[suid] = [f]

    # use the last series.
    choose_suid = sorted(series.keys())[-1]
    files = sorted(series[choose_suid])
    return files


def load_dicom_folder(slice_dirs: list[Path], path: Path) -> sitk.Image:  # noqa: C901
    """Load dicom folder and return sitk image.

    Args:
        slice_dirs: list of dir paths, each containing a dicom series.
        path: path to save the nifti file.

    Returns:
        sitk image.
    """
    z = len(slice_dirs)
    first_slice_dcm_paths = find_series(slice_dirs[0])

    # Read a dicom file from the correct series
    d = dcmread(first_slice_dcm_paths[0])
    x = d.Columns
    y = d.Rows
    t = d.CardiacNumberOfImages
    dx = float(d.PixelSpacing[1])
    dy = float(d.PixelSpacing[0])

    # DICOM coordinate (LPS)
    #  x: left
    #  y: posterior
    #  z: superior
    # Nifti coordinate (RAS)
    #  x: right
    #  y: anterior
    #  z: superior
    # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
    # Refer to
    # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

    # The coordinate of the upper-left voxel of the first and second slices
    pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
    pos_ul[:2] = -pos_ul[:2]

    # Image orientation
    axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
    axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
    axis_x[:2] = -axis_x[:2]
    axis_y[:2] = -axis_y[:2]

    if z >= 2:
        # Read a dicom file at the second slice
        second_slice_dcm_paths = find_series(slice_dirs[1])
        d2 = dcmread(second_slice_dcm_paths[0])
        pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
        pos_ul2[:2] = -pos_ul2[:2]
        axis_z = pos_ul2 - pos_ul
        axis_z = axis_z / np.linalg.norm(axis_z)
    else:
        axis_z = np.cross(axis_x, axis_y)

    # Determine the z spacing
    if hasattr(d, "SpacingBetweenSlices"):
        dz = float(d.SpacingBetweenSlices)
    elif z >= 2:
        # can not find attribute SpacingBetweenSlices. Calculate from two successive slices
        dz = float(np.linalg.norm(pos_ul2 - pos_ul))
    else:
        # can not find attribute SpacingBetweenSlices. Use attribute SliceThickness instead
        dz = float(d.SliceThickness)

    # Affine matrix which converts the voxel coordinate to world coordinate
    affine = np.eye(4)
    affine[:3, 0] = axis_x * dx
    affine[:3, 1] = axis_y * dy
    affine[:3, 2] = axis_z * dz
    affine[:3, 3] = pos_ul

    # The 4D volume
    volume = np.zeros((x, y, z, t), dtype="float32")

    # Go through each slice
    for k in range(z):
        files = sorted(slice_dirs[k].iterdir())

        # Now for this series, sort the files according to the trigger time.
        files_time = []
        for f in files:
            d = dcmread(f)
            j = d.TriggerTime
            files_time += [[f, j]]
        files_time = sorted(files_time, key=lambda x: x[1])

        # Read the images
        for j in range(t):
            # http://nipy.org/nibabel/dicom/dicom_orientation.html#i-j-columns-rows-in-dicom
            # The dicom pixel_array has dimension (Y,X), i.e. X changing faster.
            # However, the nibabel data array has dimension (X,Y,Z,T), i.e. X changes the slowest.
            # We need to flip pixel_array so that the dimension becomes (X,Y), to be consistent
            # with nibabel's dimension.
            try:
                f = files_time[j][0]
                d = dcmread(f)
                volume[:, :, k, j] = d.pixel_array.transpose()
            except IndexError:
                logger.warning(
                    f"dicom file missing for {slice_dirs[k]}: time point {j}. "
                    "Image will be copied from the previous time point."
                )
                volume[:, :, k, j] = volume[:, :, k, j - 1]
            except (ValueError, TypeError, AttributeError):
                logger.warning(
                    f"failed to read pixel_array from file {files[j]}. "
                    "Image will be copied from the previous time point."
                )
                volume[:, :, k, j] = volume[:, :, k, j - 1]
            except NotImplementedError:
                f = files_time[j][0]
                logger.warning(
                    f"failed to read pixel_array from file {f}. "
                    "pydicom cannot handle compressed dicom files. "
                    "Switch to SimpleITK instead."
                )
                reader = sitk.ImageFileReader()
                reader.SetFileName(f)
                img = sitk.GetArrayFromImage(reader.Execute())
                volume[:, :, k, j] = np.transpose(img[0], (1, 0))
    # Temporal spacing
    try:
        dt = (files_time[1][1] - files_time[0][1]) * 1e-3
    except IndexError:
        dt = 1

    nim = nib.Nifti1Image(volume, affine)
    nim.header["pixdim"][4] = dt
    nim.header["sform_code"] = 1

    nib.save(nim, path)
    return sitk.ReadImage(str(path))


def load_dicom_files(paths: list[Path]) -> sitk.Image:
    """Load dicom files and return sitk image.

    Args:
        paths: paths to the dicom files.

    Returns:
        sitk image.
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    image = reader.Execute()
    return image


def concatenate_sax_images(sax_images: list[sitk.Image], slice_spacing: float) -> sitk.Image:
    """Concatenate SAX images along z-axis.

    Args:
        sax_images: list of SAX images, GetSize() = (x, y, t).
        slice_spacing: spacing at z-axis.

    Returns:
        SAX images, GetSize() = (x, y, z, t).
    """
    # (z, t, y, x)
    image_array = np.stack([sitk.GetArrayFromImage(image_sax) for image_sax in sax_images])
    # GetSize() = (x, y, t)
    image_xyt = sax_images[0]
    origin = image_xyt.GetOrigin()
    pixel_spacing = image_xyt.GetSpacing()[:2]
    spacing = np.array([pixel_spacing[0], pixel_spacing[1], slice_spacing])
    direction = image_xyt.GetDirection()
    image_slices = []
    for t in range(image_array.shape[1]):
        # GetSize() = (x, y, z)
        image_xyz = sitk.GetImageFromArray(image_array[:, t, :, :], isVector=False)
        image_xyz.SetOrigin(origin)
        image_xyz.SetSpacing(spacing)
        image_xyz.SetDirection(direction)
        image_slices.append(image_xyz)
    # GetSize() = (x, y, z, t)
    return sitk.JoinSeries(image_slices)


def check_lax_sax_images(
    lax_2c_image: sitk.Image,
    lax_3c_image: sitk.Image | None,
    lax_4c_image: sitk.Image,
    sax_images: list[sitk.Image],
    image_id: str,
    decimals: int = 4,
) -> float:
    """Sanity check for image sizes, also calculate the slice spacing for SAX images.

    Args:
        lax_2c_image: 2C image, GetSize() = (x, y, t).
        lax_3c_image: 3C image, GetSize() = (x, y, t), can be None as not all images have 3C.
        lax_4c_image: 4C image, GetSize() = (x, y, t).
        sax_images: list of SAX images, each one has GetSize() = (x, y, t).
        image_id: folder identifier, e.g. 6000182_2_0, for error message.
        decimals: number of decimals to round.

    Returns:
        SAX slice spacing.
    """
    # check LAX image number of time frames
    lax_image_times = [
        lax_2c_image.GetSize()[-1],
        lax_4c_image.GetSize()[-1],
    ]
    lax_image_names = "2C/4C"
    if lax_3c_image is not None:
        lax_image_times.append(lax_3c_image.GetSize()[-1])
        lax_image_names = "2C/3C/4C"
    if len(set(lax_image_times)) != 1:
        raise ValueError(
            f"LAX images have different number of time frames: {lax_image_times} for {lax_image_names} for {image_id}.",
        )

    # check SAX image sizes
    sax_image_sizes = [sax_image.GetSize() for sax_image in sax_images]
    if len(set(sax_image_sizes)) != 1:
        raise ValueError(f"SAX image having different sizes {sax_image_sizes} for {image_id}.")

    # check SAX pixel spacing
    sax_pixel_spacings = np.array([image.GetSpacing() for image in sax_images])
    sax_pixel_spacings = np.round(sax_pixel_spacings, decimals=decimals)
    if len(set(map(tuple, sax_pixel_spacings))) != 1:
        raise ValueError(f"SAX images have different pixel spacings {sax_pixel_spacings} for {image_id}.")

    # check SAX slice direction
    # (n_slices, 9)
    sax_directions = np.array([sax_image.GetDirection() for sax_image in sax_images])
    sax_directions = np.round(sax_directions, decimals=decimals)
    if len(set(map(tuple, sax_directions))) != 1:
        raise ValueError(f"SAX images have different directions {sax_directions} for {image_id}.")

    # check SAX slice spacing
    # (n_slices, 3)
    sax_origins = np.array([sax_image.GetOrigin() for sax_image in sax_images])
    # (n_slices-1,)
    sax_slice_spacings = np.linalg.norm(np.diff(sax_origins, axis=0), axis=-1)
    sax_slice_spacings = np.round(sax_slice_spacings, decimals=decimals)
    if len(set(sax_slice_spacings)) != 1:
        raise ValueError(f"SAX images have different slice distances {sax_slice_spacings} for {image_id}.")
    return sax_slice_spacings[0]


def dicom_orientation_to_rotation_matrix(orientation: np.ndarray) -> np.ndarray:
    """Convert DICOM orientation to rotation matrix.

    https://gist.github.com/agirault/60a72bdaea4a2126ecd08912137fe641
        Ax  Bx  Cx
    R = Ay  By  Cy
        Az  Bz  Cz
    where [C] = [A] x [B] (cross-product)

    Args:
        orientation: DICOM orientation, (6,).
    """
    a = orientation[:3]
    b = orientation[3:]
    c = np.cross(a, b)
    r = np.asarray([a, b, c]).T
    return r
