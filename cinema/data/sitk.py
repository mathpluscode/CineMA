"""Preprocess functions using sitk."""

from __future__ import annotations

from functools import partial
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk  # noqa: N813

from cinema.log import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


def plane_plane_intersection(
    rot1: np.ndarray,
    origin1: np.ndarray,
    rot2: np.ndarray,
    origin2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the intersection of two planes.

    https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
    https://gist.github.com/marmakoide/79f361dd613f2076ece544070ddae6ab

    Coordinates in a 2D image is the plane z=0.
    Each coordinate in the image corresponds to a coordinate in the real space.
    The real coordinate is calculated as rot() + origin.

    rotation matrix is represented as
            Ax  Bx  Cx
        R = Ay  By  Cy
            Az  Bz  Cz
    where [Cx, Cy, Cz] = [C] = [A] x [B] (cross-product)

    So the norm vector (x, y, z) = (0, 0, 1) is rotated to [C].
    The plane in real space is represented by the origin and norm vector [C].
    Let [C] = (Cx, Cy, Cz), origin be (ox, oy, oz), the plane is
        (x-ox)*Cx + (y-oy)*Cy + (z-oz)*Cz = 0
    which can be rewritten to
        x*Cx + y*Cy + zCz = ox*Cx + oy*Cy + oz*Cz

    Consider the intersection of two planes,
    any line on the plane is orthogonal to its norm vector.
    So the intersection line is orthogonal to both norm vectors.
    So the intersection line is parallel to the cross product of the norm vectors.

    Then we just need to find one point on the line.
    Let the real space zero origin pointing to this point, be orthorgonal to the line vector.

    Args:
        rot1: rotation matrix, (3, 3).
        origin1: origin of the plane, (3,).
        rot2: rotation matrix, (3, 3).
        origin2: origin of the plane, (3,).

    Returns:
        line_point: a point on the intersection line, (3,).
        line_vec: vector of the intersection line, (3,).
    """
    # (3,)
    plane_norm_vec1 = rot1[:, -1]
    plane_norm_vec1 /= np.linalg.norm(plane_norm_vec1)  # renorm just in case
    plane_norm_vec2 = rot2[:, -1]
    plane_norm_vec2 /= np.linalg.norm(plane_norm_vec2)
    line_vec = np.cross(plane_norm_vec1, plane_norm_vec2)
    line_vec /= np.linalg.norm(line_vec)
    # (3, 3)
    a = np.array([plane_norm_vec1, plane_norm_vec2, line_vec])
    a_cond = np.linalg.cond(a)
    if a_cond > 1 / np.finfo(a.dtype).eps:
        logger.error(f"matrix a is ill-conditioned, np.linalg.cond(a)={a_cond}")
    # (3,)
    b = np.array([np.dot(origin1, plane_norm_vec1), np.dot(origin2, plane_norm_vec2), 0.0])
    # (3,), a @ line_point = b
    line_point = np.linalg.solve(a, b)
    return line_point, line_vec


def plane_line_intersection(
    rot: np.ndarray,
    origin: np.ndarray,
    line_point: np.ndarray,
    line_vec: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray | None:
    """Calculate the intersection of a plane and a line.

    rotation matrix is represented as
            Ax  Bx  Cx
        R = Ay  By  Cy
            Az  Bz  Cz
    where [Cx, Cy, Cz] = [C] = [A] x [B] (cross-product)

    Let [C] = (Cx, Cy, Cz), origin be (ox, oy, oz), the plane is
        (x-ox)*Cx + (y-oy)*Cy + (z-oz)*Cz = 0
    which can be rewritten to
        x*Cx + y*Cy + zCz = ox*Cx + oy*Cy + oz*Cz
    so
        [C] @ (x, y, z) = [C] @ origin

    Assume the intersection point is line_point + t * line_vec,
    where t is a scalar, so
        [C] @ (line_point + t * line_vec) = [C] @ origin
    so
        t = ([C] @ (origin - line_point)) / ([C] @ line_vec)

    Args:
        rot: rotation matrix, (3, 3).
        origin: origin of the plane, (3,).
        line_point: a point on the intersection line, (3,).
        line_vec: vector of the intersection line, (3,).
        epsilon: a small number to avoid division by zero.

    Returns:
        intersection_point: a point on the intersection, (3,),
            returns None if the line is parallel to the plane.
    """
    # (3,)
    plane_norm_vec = rot[:, -1]
    plane_norm_vec /= np.linalg.norm(plane_norm_vec)  # renorm just in case
    # scalar
    nominator = np.dot(plane_norm_vec, (origin - line_point))
    denominator = np.dot(plane_norm_vec, line_vec)
    t = nominator / denominator
    if np.abs(denominator) < epsilon:
        logger.info(
            f"plane_norm_vec {plane_norm_vec} is orthogonal to line_vec {line_vec},"
            f"denominator np.dot(plane_norm_vec, line_vec)={denominator} is too small.",
        )
        return None
    return line_point + t * line_vec


def process_4d(
    image: sitk.Image,
    func: Callable[[sitk.Image], sitk.Image],
) -> sitk.Image:
    """Process 3D+t image along the last axis.

    This function assumes that the last axis is the time axis.
    Therefore this axis contains non-spatial information, such as
    origin, direction, spacing, etc.

    Args:
        image: image to resample.
        func: a function applied to a 3d image.

    Returns:
        Resampled image.
    """
    image_size = image.GetSize()
    if len(image_size) != 4:
        raise ValueError(f"Image should have 4 dimensions, got {image_size}.")

    n_frames = image_size[-1]
    try:
        # split and stack on the last axis
        return sitk.JoinSeries([func(image[..., i]) for i in range(n_frames)])
    except Exception:  # pylint: disable=broad-except
        logger.exception(f"Failed to join series for {func}.")
        raise


def resample_spacing_3d(image: sitk.Image, is_label: bool, target_spacing: tuple[float, ...]) -> sitk.Image:
    """Resample 3D image to the target spacing.

    Args:
        image: image to resample.
        is_label: True if it represents a label,
            thus nearest neighbour for interpolation.
        target_spacing: target dimension per axis.

    Returns:
        Resampled image.
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()

    if len(original_spacing) != 3 or len(target_spacing) != 3:
        raise ValueError(
            f"This function supports 3D image only. "
            f"Original spacing {original_spacing} "
            f"and target spacing {target_spacing} should have 3 elements.",
        )

    # calculate size after resampling
    # round to integers to be robust
    # otherwise, ceiling is sensitive to spacing
    resample_target_size = tuple(
        int(np.round(orig_sh * orig_sp / trg_sp))
        for orig_sh, orig_sp, trg_sp in zip(original_size, original_spacing, target_spacing, strict=False)
    )
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    # No transform because we do not want to change the represented
    # physical size of the objects in the image
    transform = sitk.Transform()
    # The origin is middle of the voxel/pixel
    # https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html
    target_origin = [x + 0.5 * (target_spacing[i] - original_spacing[i]) for i, x in enumerate(original_origin)]
    # Do not rotate
    target_direction = image.GetDirection()
    return sitk.Resample(
        image,
        size=resample_target_size,
        transform=transform,
        interpolator=interpolator,
        outputOrigin=target_origin,
        outputSpacing=target_spacing,
        outputDirection=target_direction,
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID(),
        useNearestNeighborExtrapolator=False,
    )


def resample_spacing_4d(
    image: sitk.Image,
    is_label: bool,
    target_spacing: tuple[float, ...],
) -> sitk.Image:
    """Resample 4D image to the target spacing along the last axis.

    Args:
        image: image to resample.
        is_label: True if it represents a label, thus nearest neighbour for interpolation.
        target_spacing: target dimension, excluding an axis.

    Returns:
        Resampled image.
    """
    return process_4d(
        image=image,
        func=partial(resample_spacing_3d, is_label=is_label, target_spacing=target_spacing),
    )


def clip_and_normalise_intensity_3d(
    image: sitk.Image,
    intensity_range: tuple[float, float] | None,
) -> sitk.Image:
    """Clip and normalise the intensity of the image.

    Args:
        image: image to clip and normalise.
        intensity_range: intensity range to clip to.
            If None, clip to 0.95 and 99.5 percentiles.

    Returns:
        Image with clipped and normalised intensity.
    """
    if len(image.GetSize()) != 3:
        raise ValueError(f"This function supports 3D image only. Image size {image.GetSize()} should have 3 elements.")
    # clip intensity
    if intensity_range is None:
        # if not configured, clip to 0.95 and 99.5 percentiles
        # https://arxiv.org/abs/2304.12306
        image_array = sitk.GetArrayFromImage(image)
        intensity_range = (
            np.percentile(image_array, 0.95),
            np.percentile(image_array, 99.5),
        )
    image = sitk.Clamp(
        image,
        lowerBound=intensity_range[0],
        upperBound=intensity_range[1],
    )

    # normalise the intensity
    image = sitk.Normalize(image)
    image = sitk.RescaleIntensity(image, outputMinimum=0, outputMaximum=1)
    return image


def clip_and_normalise_intensity_4d(
    image: sitk.Image,
    intensity_range: tuple[float, float] | None,
) -> sitk.Image:
    """Clip and normalise the intensity of the image.

    Args:
        image: image to clip and normalise.
        intensity_range: intensity range to clip to.
            If None, clip to 0.95 and 99.5 percentiles.

    Returns:
        Image with clipped and normalised intensity.
    """
    return process_4d(
        image=image,
        func=partial(clip_and_normalise_intensity_3d, intensity_range=intensity_range),
    )


def get_center_pad_size(
    current_size: tuple[int, ...],
    target_size: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Get pad sizes for sitk.ConstantPad.

    The padding is added symmetrically.

    Args:
        current_size: current size of the image.
        target_size: target size of the image.

    Returns:
        pad_lower: size to pad on the lower side.
        pad_upper: size to pad on the upper side.
    """
    pad_lower = []
    pad_upper = []
    for i, size_i in enumerate(current_size):
        pad_i = max(target_size[i] - size_i, 0)
        pad_lower_i = pad_i // 2
        pad_upper_i = pad_i - pad_lower_i
        pad_lower.append(pad_lower_i)
        pad_upper.append(pad_upper_i)
    return tuple(pad_lower), tuple(pad_upper)


def pad_4d(
    image: sitk.Image,
    pad_lower: tuple[int, ...],
    pad_upper: tuple[int, ...],
) -> sitk.Image:
    """Pad 4D volume along the last axis.

    Args:
        image: image to pad.
        pad_lower: lower bound for pad, 3d.
        pad_upper: upper bound for pad, 3d.

    Returns:
        Padded image.
    """
    if min(pad_lower) < 0:
        raise ValueError(f"Pad lower {pad_lower} should be non-negative.")
    if min(pad_upper) < 0:
        raise ValueError(f"Pad upper {pad_upper} should be non-negative.")
    return process_4d(
        image=image,
        func=lambda x: sitk.ConstantPad(x, pad_lower, pad_upper, 0),
    )


def crop_4d(
    image: sitk.Image,
    crop_lower: tuple[int, ...],
    crop_upper: tuple[int, ...],
) -> sitk.Image:
    """Pad 4D volume along the last axis.

    Args:
        image: image to pad.
        crop_lower: lower bound of crop, 3d.
        crop_upper: upper bound of crop, 3d.

    Returns:
        Cropped image.
    """
    if min(crop_lower) < 0:
        raise ValueError(f"Crop lower {crop_lower} should be non-negative.")
    if min(crop_upper) < 0:
        raise ValueError(f"Crop upper {crop_upper} should be non-negative.")
    return process_4d(
        image=image,
        func=lambda x: sitk.Crop(x, crop_lower, crop_upper),
    )


def crop_xy_3d(
    image: sitk.Image,
    origin_indices: tuple[int, int],
    slice_size: tuple[int, int],
) -> sitk.Image:
    """Crop 3D volume along the first two axes.

    Perform padding if the crop is out of the image boundary.

    Args:
        image: image to crop.
        origin_indices: origin indices, (x, y),
            cropping is (x: x+slice_size[0], y: y+slice_size[1]).
            indices may be negative, indicating the crop is out of the image boundary.
        slice_size: size of the slice at X, Y axis.

    Returns:
        Cropped image.
    """
    image_size = image.GetSize()  # (x, y, t)
    if len(image_size) != 3:
        raise ValueError(f"This function supports 3D image only. Image size {image_size} should have 3 elements.")
    x_start, y_start = origin_indices  # inclusive
    if x_start < 0 or y_start < 0:
        pad_lower = (-min(x_start, 0), -min(y_start, 0), 0)
        pad_upper = (0, 0, 0)
        image = sitk.ConstantPad(image, pad_lower, pad_upper, 0)
        x_start, y_start = max(x_start, 0), max(y_start, 0)
    sax_x_end, sax_y_end = x_start + slice_size[0], y_start + slice_size[1]  # exclusive
    if sax_x_end > image_size[0] or sax_y_end > image_size[1]:
        pad_lower = (0, 0, 0)
        pad_upper = (max(sax_x_end - image_size[0], 0), max(sax_y_end - image_size[1], 0), 0)
        image = sitk.ConstantPad(image, pad_lower, pad_upper, 0)
    return image[x_start:sax_x_end, y_start:sax_y_end, :]


def crop_xy_4d(
    image: sitk.Image,
    origin_indices: tuple[int, int],
    slice_size: tuple[int, int],
) -> sitk.Image:
    """Crop 4D volume along the first two axes.

    Perform padding if the crop is out of the image boundary.

    Args:
        image: image to crop.
        origin_indices: origin indices, (x, y),
            cropping is (x: x+slice_size[0], y: y+slice_size[1]).
            indices may be negative, indicating the crop is out of the image boundary.
        slice_size: size of the slice at X, Y axis.

    Returns:
        Cropped image.
    """
    image_size = image.GetSize()  # (x, y, z, t)
    if len(image_size) != 4:
        raise ValueError(f"This function supports 4D image only. Image size {image_size} should have 4 elements.")
    x_start, y_start = origin_indices  # inclusive
    if x_start < 0 or y_start < 0:
        pad_lower = (-min(x_start, 0), -min(y_start, 0), 0)
        pad_upper = (0, 0, 0)
        image = pad_4d(image, pad_lower, pad_upper)
        x_start, y_start = max(x_start, 0), max(y_start, 0)
    sax_x_end, sax_y_end = x_start + slice_size[0], y_start + slice_size[1]  # exclusive
    if sax_x_end > image_size[0] or sax_y_end > image_size[1]:
        pad_lower = (0, 0, 0)
        pad_upper = (max(sax_x_end - image_size[0], 0), max(sax_y_end - image_size[1], 0), 0)
        image = pad_4d(image, pad_lower, pad_upper)
    return image[x_start:sax_x_end, y_start:sax_y_end, :, :]


def cast_to_uint8(image: sitk.Image) -> sitk.Image:
    """Cast image to uint16 to save disk space.

    Args:
        image: image to cast with values in [0, 1].

    Returns:
        Casted image with values in uint16.
    """
    ndim = len(image.GetSize())
    if ndim <= 3:
        return sitk.Cast(image * 255, sitk.sitkUInt8)
    if ndim == 4:
        return process_4d(image, lambda x: sitk.Cast(x * 255, sitk.sitkUInt8))
    raise ValueError(f"Image should have 3 or 4 dimensions, got {ndim}.")


def load_subimage(file_path: str, extract_index: list[int], extract_size: list[int]) -> sitk.Image:
    """Load a sub-image.

    Args:
        file_path: path to the image file.
        extract_index: start index of the sub-image.
        extract_size: size of the sub-image, if -1, extract the whole axis.

    Returns:
        Sub-image.
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(file_path)
    file_reader.ReadImageInformation()
    image_size = file_reader.GetSize()
    for i, s in enumerate(extract_size):
        if s == -1:
            extract_index[i] = 0
            extract_size[i] = image_size[i]
    file_reader.SetExtractIndex(extract_index)
    file_reader.SetExtractSize(extract_size)
    return file_reader.Execute()


def pad_array(arr: np.ndarray, dim: int, n: int, value: int) -> np.ndarray:
    """Index and pad the input array to the desired number of frames and slices.

    If the array is larger than needed, only the first n frames or slices are used.

    Args:
        arr: input array.
        dim: dimension to pad.
        n: number of frames or slices to pad.
        value: value to pad.

    Returns:
        arr: indexed and padded array.
    """
    if arr.shape[dim] == n:
        return arr
    if arr.shape[dim] < n:
        pad_size = list(arr.shape)
        pad_size[dim] = n - arr.shape[dim]
        pad = (np.ones(pad_size) * value).astype(arr.dtype)
        arr = np.concatenate([arr, pad], axis=dim)
        return arr
    logger.error(f"Array has size {arr.shape[dim]}, which is larger than needed {n}, truncating.")
    return np.take(arr, range(n), axis=dim)


def get_invalid_bounding_box(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return all -1 values as an invalid bounding box.

    Args:
        mask: boolean mask, with n spatial axes.

    Returns:
        bbox_min: [-1] * n.
        bbox_min: [-1] * n.
    """
    ndim_spatial = len(mask.shape)
    bbox_min = -np.ones(ndim_spatial, np.int32)
    bbox_max = -np.ones(ndim_spatial, np.int32)
    return bbox_min, bbox_max


def get_valid_binary_mask_bounding_box(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of foreground with start-end positions.

    If there is no foreground, return -1 for all outputs.

    Args:
        mask: boolean mask, with only spatial axes.

    Returns:
        bbox_min: [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        bbox_max: [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
    """
    ndim_spatial = len(mask.shape)
    bbox_min = []
    bbox_max = []
    for axes_to_reduce in combinations(reversed(range(ndim_spatial)), ndim_spatial - 1):
        mask_reduced = np.amax(mask, axis=axes_to_reduce)
        bbox_min_axis = np.argmax(mask_reduced)
        bbox_max_axis = mask_reduced.shape[0] - np.argmax(np.flip(mask_reduced))
        bbox_min.append(bbox_min_axis)
        bbox_max.append(bbox_max_axis)
    return np.stack(bbox_min), np.stack(bbox_max)


def get_binary_mask_bounding_box(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the bounding box of foreground with start-end positions.

    If there is no foreground, return -1 for all outputs.

    Args:
        mask: boolean mask, with only spatial axes.

    Returns:
        bbox_min: [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        bbox_max: [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
    """
    if mask.dtype != np.bool_:
        mask = mask > 0

    if np.any(mask):
        return get_valid_binary_mask_bounding_box(mask)
    return get_invalid_bounding_box(mask)


def get_center_crop_size_from_1d_bbox(
    bbox_min: int,
    bbox_max: int,
    current_length: int,
    target_length: int,
) -> tuple[int, int]:
    """Try to crop at the center of bounding box, 1D.

    Args:
        bbox_min: bounding box index minimum, inclusive.
        bbox_max: bounding box index maximum, exclusive.
        current_length: current image length.
        target_length: target image length.

    Returns:
        crop_lower: shape to pad on the lower side.
        crop_upper: shape to pad on the upper side.

    Raises:
        ValueError: if label min max is out of range.
    """
    if bbox_min < 0 or bbox_max > current_length:
        raise ValueError("Label index out of range.")

    if current_length <= target_length:
        # no need of crop
        return 0, 0

    # attempt to perform crop centered at label center
    label_center = (bbox_max - 1 + bbox_min) / 2.0
    bbox_lower = int(np.ceil(label_center - target_length / 2.0))
    bbox_upper = bbox_lower + target_length
    # if lower is negative, then have to shift the window to right
    bbox_lower = max(bbox_lower, 0)
    # if upper is too large, then have to shift the window to left
    if bbox_upper > current_length:
        bbox_lower -= bbox_upper - current_length
    # calculate crop
    crop_lower = bbox_lower  # bbox index starts at 0
    crop_upper = current_length - target_length - crop_lower
    return crop_lower, crop_upper


def get_center_crop_size_from_bbox(
    bbox_min: tuple[int, ...] | np.ndarray,
    bbox_max: tuple[int, ...] | np.ndarray,
    current_size: tuple[int, ...],
    target_size: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Get crop sizes for sitk.Crop from label bounding box.

    The crop is not necessarily performed symmetrically.

    Args:
        bbox_min: [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        bbox_max: [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
        current_size: current shape of the image.
        target_size: target shape of the image.

    Returns:
        crop_lower: shape to crop on the lower side.
        crop_upper: shape to crop on the upper side.
    """
    crop_lower = []
    crop_upper = []
    for i, current_length in enumerate(current_size):
        crop_lower_i, crop_upper_i = get_center_crop_size_from_1d_bbox(
            bbox_min=bbox_min[i],
            bbox_max=bbox_max[i],
            current_length=current_length,
            target_length=target_size[i],
        )
        crop_lower.append(crop_lower_i)
        crop_upper.append(crop_upper_i)
    return tuple(crop_lower), tuple(crop_upper)


def save_image(
    image_np: np.ndarray,
    reference_image_path: Path | None,
    out_path: Path | str,
) -> None:
    """Save 3d image.

    Args:
        image_np: (width, height, depth) or (width, height, depth, n_frames), the values are integers, may be padded.
        reference_image_path: path to reference image for copy meta data.
        out_path: output path.
    """
    if isinstance(out_path, str):
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.transpose(image_np), isVector=False)
    if reference_image_path is not None:
        ref_image = sitk.ReadImage(str(reference_image_path))
        if (ref_image.GetDimension() == 4) and (image_np.ndim == 3):
            # rescan data
            ref_image = ref_image[:, :, :, 0]
        if ref_image.GetSize() != image.GetSize():
            # for Kaggle, some images have >30 frames
            logger.error(
                f"Reference image {reference_image_path} has different size from the input image, "
                f"{ref_image.GetSize()} != {image.GetSize()}"
            )
            min_size = min(ref_image.GetSize()[-1], image.GetSize()[-1])
            if image_np.ndim == 4:
                ref_image = ref_image[:, :, :, :min_size]
                image = image[:, :, :, :min_size]
            elif image_np.ndim == 3:
                ref_image = ref_image[:, :, :min_size]
                image = image[:, :, :min_size]
            else:
                raise ValueError(
                    f"Reference image and input image have different dimensions, "
                    f"{image_np.shape} != {ref_image.GetSize()}"
                )
        if ref_image.GetSize() != image.GetSize():
            # unexpected
            raise ValueError(
                f"Reference image {reference_image_path} has different size from the input image, "
                f"{ref_image.GetSize()} != {image.GetSize()}"
            )
        image.CopyInformation(ref_image)
    sitk.WriteImage(
        image=image,
        fileName=out_path,
        useCompression=True,
    )


def get_lax_2c_4c_plane_intersection(
    lax_2c_image: sitk.Image,
    lax_4c_image: sitk.Image,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the intersection of 2c and 4c planes.

    Args:
        lax_2c_image: 2c image, GetSize() = (x, y, t).
        lax_4c_image: 4c image, GetSize() = (x, y, t).

    Returns:
        line_point: point on the intersection line.
        line_vec: direction of the intersection line.
    """
    origin_2c = np.array(lax_2c_image.GetOrigin())
    rot_2c = np.array(lax_2c_image.GetDirection()).reshape((3, 3))
    origin_4c = np.array(lax_4c_image.GetOrigin())
    rot_4c = np.array(lax_4c_image.GetDirection()).reshape((3, 3))
    return plane_plane_intersection(rot_2c, origin_2c, rot_4c, origin_4c)


def get_sax_center(
    sax_image: sitk.Image,
    lax_2c_image: sitk.Image,
    lax_4c_image: sitk.Image,
) -> np.ndarray | None:
    """Get the center of 2C/4C/SAX image.

    Args:
        sax_image: SAX image, GetSize() = (x, y, z).
        lax_2c_image: 2C image, GetSize() = (x, y, t).
        lax_4c_image: 4C image, GetSize() = (x, y, t).

    Returns:
        center coordinates in real space.
    """
    # 2C/4C intersection
    line_point, line_vec = get_lax_2c_4c_plane_intersection(lax_2c_image, lax_4c_image)
    sax_origin = sax_image.GetOrigin()
    sax_rot = np.array(sax_image.GetDirection()).reshape((3, 3))
    # 2C/4C/SAX intersection
    sax_center = plane_line_intersection(
        rot=sax_rot,
        origin=sax_origin,
        line_point=line_point,
        line_vec=line_vec,
    )
    if sax_center is None:
        logger.error(
            "2C/4C intersection line is parallel to SAX plane, cannot identify the intersection point, return None.",
        )
    return sax_center


def get_origin_for_crop(
    center: np.ndarray,
    image: sitk.Image,
    slice_size: tuple[int, int],
) -> tuple[int, int]:
    """Get the origin for image cropping at the given center.

    Args:
        center: center of the cropping, (x, y, z).
        image: GetSize() = (x, y, *).
        slice_size: size of the slice at X, Y axis.

    Returns:
        origin indices, (x, y), cropping is (x: x+slice_size[0], y: y+slice_size[1]).
    """
    origin = np.array(image.GetOrigin())
    rot = np.array(image.GetDirection()).reshape((3, 3))
    # center = rotation @ (coords * spacing) + origin
    indices = np.linalg.solve(rot, center - origin)[:2]
    indices /= np.array(image.GetSpacing()[:2])
    indices[0] -= (slice_size[0] - 1) / 2.0
    indices[1] -= (slice_size[1] - 1) / 2.0
    return int(indices[0]), int(indices[1])
