"""Utilities for working with DICOM files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813

if TYPE_CHECKING:
    from pathlib import Path


def load_dicom_folder(path: Path) -> sitk.Image:
    """Load dicom folder and return sitk image.

    Args:
        path: path to the dicom folder.

    Returns:
        sitk image.
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(path)))
    image = reader.Execute()
    return image


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
