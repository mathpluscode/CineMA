"""Tests for sitk utility functions."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np
import pytest
import SimpleITK as sitk  # noqa: N813

from cinema.data.sitk import (
    clip_and_normalise_intensity_3d,
    clip_and_normalise_intensity_4d,
    crop_4d,
    crop_xy_3d,
    crop_xy_4d,
    get_binary_mask_bounding_box,
    get_center_crop_size_from_1d_bbox,
    get_center_crop_size_from_bbox,
    get_center_pad_size,
    load_subimage,
    pad_4d,
    pad_array,
    plane_plane_intersection,
    resample_spacing_3d,
    resample_spacing_4d,
    save_image,
)


class TestPlanePlaneIntersection:
    """Test plane_plane_intersection."""

    @pytest.mark.parametrize(
        (
            "rot1",
            "origin1",
            "rot2",
            "origin2",
            "expected_line_point",
            "expected_line_vec",
        ),
        [
            (
                # x-y plane
                np.eye(3),
                np.zeros(3),
                # y-z plane
                np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
                np.zeros(3),
                np.zeros(3),
                np.array([0.0, 1.0, 0.0]),
            ),
            (
                # x-y plane
                np.eye(3),
                np.zeros(3),
                # x-z plane
                np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
                np.zeros(3),
                np.zeros(3),
                np.array([-1.0, 0.0, 0.0]),
            ),
        ],
    )
    def test_plane_plane_intersection(
        self,
        rot1: np.ndarray,
        origin1: np.ndarray,
        rot2: np.ndarray,
        origin2: np.ndarray,
        expected_line_point: np.ndarray,
        expected_line_vec: np.ndarray,
    ) -> None:
        """Test plane_plane_intersection.

        Args:
            rot1: rotation matrix, (3, 3).
            origin1: origin of the plane, (3,).
            rot2: rotation matrix, (3, 3).
            origin2: origin of the plane, (3,).
            expected_line_point: a point on the intersection line, (3,).
            expected_line_vec: vector of the intersection line, (3,).
        """
        got_line_point, got_line_vec = plane_plane_intersection(rot1, origin1, rot2, origin2)
        np.testing.assert_allclose(got_line_point, expected_line_point)
        np.testing.assert_allclose(got_line_vec, expected_line_vec)


class TestResampleSpacing:
    """Test resample_spacing functions."""

    @pytest.mark.parametrize("is_label", [True, False])
    @pytest.mark.parametrize(
        ("image_size", "target_spacing", "expected_size"),
        [
            ((10, 11, 12), (1.0, 2.0, 3.0), (10, 11, 12)),
            ((10, 11, 12), (1.0, 4.0, 3.0), (10, 6, 12)),
            ((10, 11, 12), (1.0, 2.0, 1.5), (10, 11, 24)),
        ],
    )
    def test_3d(
        self,
        image_size: tuple[int, ...],
        is_label: bool,
        target_spacing: tuple[float, ...],
        expected_size: tuple[int, ...],
    ) -> None:
        """Test output sizes."""
        source_spacing = (1.0, 2.0, 3.0)
        dtype = np.uint8 if is_label else np.float32
        x = np.ones(image_size, dtype=dtype)
        x = np.transpose(x)
        image = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        image.SetSpacing(source_spacing)
        out = resample_spacing_3d(image, is_label, target_spacing)
        assert out.GetSize() == expected_size

    @pytest.mark.parametrize("is_label", [True, False])
    @pytest.mark.parametrize(
        ("image_size", "source_spacing", "target_spacing", "expected_size"),
        [
            ((8, 10, 11, 12), (1.0, 1.0, 2.0, 3.0), (1.5, 2.5, 3.5), (5, 4, 6, 12)),
        ],
    )
    def test_4d(
        self,
        image_size: tuple[int, ...],
        is_label: bool,
        source_spacing: tuple[float, ...],
        target_spacing: tuple[float, ...],
        expected_size: tuple[int, ...],
    ) -> None:
        """Test output sizes."""
        dtype = np.uint8 if is_label else np.float32
        x = np.ones(image_size, dtype=dtype)
        x = np.transpose(x)
        image = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        image.SetSpacing(source_spacing)
        out = resample_spacing_4d(image, is_label, target_spacing)
        assert out.GetSize() == expected_size


class TestClipAndNormaliseIntensity:
    """Test clip and normalise intensity functions."""

    @pytest.mark.parametrize(
        ("image_size", "intensity_range"),
        [
            ((10, 11, 12), None),
            ((10, 11, 12), (0.2, 0.8)),
        ],
    )
    def test_3d(
        self,
        image_size: tuple[int, ...],
        intensity_range: tuple[float, float] | None,
    ) -> None:
        """Test output sizes."""
        rng = np.random.default_rng()
        x = rng.random(image_size)
        x = np.transpose(x)
        image = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        out = clip_and_normalise_intensity_3d(image, intensity_range)
        assert out.GetSize() == image_size

    @pytest.mark.parametrize(
        ("image_size", "intensity_range"),
        [
            ((9, 10, 11, 12), None),
            ((9, 10, 11, 12), (0.2, 0.8)),
        ],
    )
    def test_4d(
        self,
        image_size: tuple[int, ...],
        intensity_range: tuple[float, float] | None,
    ) -> None:
        """Test output sizes."""
        rng = np.random.default_rng()
        x = rng.random(image_size)
        x = np.transpose(x)
        image = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        out = clip_and_normalise_intensity_4d(image, intensity_range)
        assert out.GetSize() == image_size


@pytest.mark.parametrize(
    ("current_size", "target_size", "expected_lower", "expected_upper"),
    [
        (
            (64, 44, 40),
            (64, 44, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (64, 44, 40),
            (40, 44, 30),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (64, 44, 40),
            (64, 64, 40),
            (0, 10, 0),
            (0, 10, 0),
        ),
        (
            (63, 43, 39),
            (64, 64, 40),
            (0, 10, 0),
            (1, 11, 1),
        ),
        (
            (44, 40),
            (64, 40),
            (10, 0),
            (10, 0),
        ),
        (
            (43, 39),
            (64, 40),
            (10, 0),
            (11, 1),
        ),
    ],
    ids=[
        "3d_same",
        "3d_no_pad",
        "3d_even",
        "3d_odd",
        "2d_even",
        "2d_odd",
    ],
)
def test_get_center_pad_size(
    current_size: tuple[int, ...],
    target_size: tuple[int, ...],
    expected_lower: tuple[int, ...],
    expected_upper: tuple[int, ...],
) -> None:
    """Test get_center_pad_size.

    Args:
        current_size: current size of the image.
        target_size: target size of the image.
        expected_lower: size to pad on the lower side.
        expected_upper: size to pad on the upper side.
    """
    got_lower, got_upper = get_center_pad_size(current_size, target_size)
    assert got_lower == expected_lower
    assert got_upper == expected_upper


@pytest.mark.parametrize(
    ("image_size", "pad_lower", "pad_upper", "expected_size"),
    [
        ((8, 10, 11, 12), (0, 0, 0), (0, 0, 0), (8, 10, 11, 12)),
        ((8, 10, 11, 12), (1, 2, 0), (0, 1, 2), (9, 13, 13, 12)),
    ],
)
def test_pad_4d(
    image_size: tuple[int, ...],
    pad_lower: tuple[int, ...],
    pad_upper: tuple[int, ...],
    expected_size: tuple[int, ...],
) -> None:
    """Test output sizes."""
    x = np.ones(image_size, dtype=np.float32)
    x = np.transpose(x)
    volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
    out = pad_4d(volume, pad_lower, pad_upper)
    assert out.GetSize() == expected_size


@pytest.mark.parametrize(
    ("image_size", "crop_lower", "crop_upper", "expected_size"),
    [
        ((8, 10, 11, 12), (0, 0, 0), (0, 0, 0), (8, 10, 11, 12)),
        ((8, 10, 11, 12), (1, 2, 0), (0, 1, 2), (7, 7, 9, 12)),
    ],
)
def test_crop_4d(
    image_size: tuple[int, ...],
    crop_lower: tuple[int, ...],
    crop_upper: tuple[int, ...],
    expected_size: tuple[int, ...],
) -> None:
    """Test output sizes."""
    x = np.ones(image_size, dtype=np.float32)
    x = np.transpose(x)
    volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
    out = crop_4d(volume, crop_lower, crop_upper)
    assert out.GetSize() == expected_size


class TestCropXY:
    """Test crop_xy."""

    @pytest.mark.parametrize(
        ("image_size", "origin_indices", "slice_size"),
        [
            ((8, 10, 11), (0, 0), (5, 6)),
            ((8, 10, 11), (4, 6), (5, 6)),
        ],
    )
    def test_3d_sizes(
        self,
        image_size: tuple[int, ...],
        origin_indices: tuple[int, int],
        slice_size: tuple[int, int],
    ) -> None:
        """Test output sizes."""
        x = np.ones(image_size, dtype=np.float32)
        x = np.transpose(x)
        volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        out = crop_xy_3d(volume, origin_indices, slice_size)
        assert out.GetSize() == (*slice_size, image_size[2])

    @pytest.mark.parametrize(
        ("image_size", "origin_indices", "slice_size"),
        [
            ((8, 10, 11, 12), (0, 0), (5, 6)),
            ((8, 10, 11, 12), (4, 6), (5, 6)),
        ],
    )
    def test_4d_sizes(
        self,
        image_size: tuple[int, ...],
        origin_indices: tuple[int, int],
        slice_size: tuple[int, int],
    ) -> None:
        """Test output sizes."""
        x = np.ones(image_size, dtype=np.float32)
        x = np.transpose(x)
        volume = sitk.GetImageFromArray(x, isVector=False)  # axes are reversed
        out = crop_xy_4d(volume, origin_indices, slice_size)
        assert out.GetSize() == (*slice_size, image_size[2], image_size[3])


@pytest.mark.parametrize(
    ("arr_size", "dim", "n", "expected_size"),
    [
        ((8, 10), 0, 8, (8, 10)),
        ((8, 10), 1, 20, (8, 20)),
        ((8, 10, 11), 0, 8, (8, 10, 11)),
        ((8, 10, 11), 0, 20, (20, 10, 11)),
        ((8, 10, 11), 1, 10, (8, 10, 11)),
        ((8, 10, 11), 1, 20, (8, 20, 11)),
        ((8, 10, 11), -2, 10, (8, 10, 11)),
        ((8, 10, 11), -2, 20, (8, 20, 11)),
    ],
)
def test_pad_array(
    arr_size: tuple[int, ...],
    dim: int,
    n: int,
    expected_size: tuple[int, ...],
) -> None:
    """Test output sizes."""
    x = np.ones(arr_size, dtype=np.float32)
    out = pad_array(x, dim, n, value=0)
    assert out.shape == expected_size


@pytest.mark.parametrize(
    ("image_size", "extract_index", "extract_size"),
    [
        ((8, 10), [0, 5], [2, 3]),
        ((8, 10, 11), [3, 4, 5], [-1, 2, 4]),
        ((8, 10, 11, 3), [4, 5, 0, 1], [2, 3, 3, -1]),
    ],
)
def test_load_subimage(
    image_size: tuple[int, ...],
    extract_index: list[int],
    extract_size: list[int],
) -> None:
    """Test return values."""
    with TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/test.nii.gz"
        rng = np.random.default_rng()
        img = sitk.GetImageFromArray(rng.random(image_size[::-1]), isVector=False)
        assert img.GetSize() == image_size
        sitk.WriteImage(img, file_path)
        img_full = sitk.ReadImage(file_path)
        assert img_full.GetSize() == image_size

        img_slice = load_subimage(file_path, extract_index, extract_size)

        arr_slice = np.transpose(sitk.GetArrayFromImage(img_slice))
        arr_full = np.transpose(sitk.GetArrayFromImage(img_full))
        expected = arr_full.copy()
        for i, s in enumerate(extract_size):
            if s == -1:
                extract_index[i] = 0
                extract_size[i] = image_size[i]
            expected = np.split(expected, [extract_index[i], extract_index[i] + extract_size[i]], axis=i)[1]
        np.testing.assert_allclose(arr_slice, expected)


@pytest.mark.parametrize(
    "image_size",
    [(8, 9, 10), (8, 10, 9, 11)],
)
def test_save_image(
    image_size: tuple[int, ...],
) -> None:
    """Test image shapes."""
    with TemporaryDirectory() as temp_dir:
        file_path = f"{temp_dir}/test.nii.gz"
        rng = np.random.default_rng()
        image = rng.random(image_size)
        save_image(image_np=image, reference_image_path=None, out_path=file_path)

        loaded_image = np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(file_path)))
        assert loaded_image.shape == image_size
        np.testing.assert_allclose(loaded_image, image)


@pytest.mark.parametrize(
    ("mask", "expected_bbox_min", "expected_bbox_max"),
    [
        (
            # 1d-int
            np.array([0, 1, 0, 1, 0]),
            np.array([1]),
            np.array([4]),
        ),
        (
            # 1d-bool
            np.array([False, True, False, True, False]),
            np.array([1]),
            np.array([4]),
        ),
        (
            # 1d-all-true
            np.array([True, True, True, True, True]),
            np.array([0]),
            np.array([5]),
        ),
        (
            # 1d-all-false
            np.array([False, False, False, False, False]),
            np.array([-1]),
            np.array([-1]),
        ),
        (
            # 2d-1x5
            np.array([[0, 1, 0, 1, 0]]),
            np.array([0, 1]),
            np.array([1, 4]),
        ),
        (
            # 2d-2x5
            np.array([[0, 1, 0, 1, 0], [1, 1, 0, 1, 0]]),
            np.array([0, 0]),
            np.array([2, 4]),
        ),
        (
            # 2d-2x5-all-false
            np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
            np.array([-1, -1]),
            np.array([-1, -1]),
        ),
    ],
)
def test_get_binary_mask_bounding_box(
    mask: np.ndarray,
    expected_bbox_min: np.ndarray,
    expected_bbox_max: np.ndarray,
) -> None:
    """Test dice loss values.

    Args:
        mask: binary mask with only spatial axes.
        expected_bbox_min: expected bounding box min, inclusive.
        expected_bbox_max: expected bounding box max, exclusive.
    """
    got_bbox_min, got_bbox_max = get_binary_mask_bounding_box(
        mask=mask,
    )
    np.testing.assert_allclose(got_bbox_min, expected_bbox_min)
    np.testing.assert_allclose(got_bbox_max, expected_bbox_max)


@pytest.mark.parametrize(
    (
        "bbox_min",
        "bbox_max",
        "current_length",
        "target_length",
        "expected_lower",
        "expected_upper",
    ),
    [
        (0, 5, 6, 6, 0, 0),
        (0, 5, 6, 7, 0, 0),
        (0, 5, 6, 4, 0, 2),
        (1, 5, 6, 4, 1, 1),
        (2, 6, 6, 4, 2, 0),
        (0, 3, 7, 4, 0, 3),
        (5, 7, 7, 4, 3, 0),
    ],
    ids=[
        "no_crop_same_length",
        "no_crop_too_short",
        "center_crop_no_shift_no_left_crop",
        "center_crop_no_shift_both_sides_crop",
        "center_crop_no_shift_no_right_crop",
        "shift_right",
        "shift_left",
    ],
)
def test_get_center_crop_size_from_1d_bbox(
    bbox_min: int,
    bbox_max: int,
    current_length: int,
    target_length: int,
    expected_lower: int,
    expected_upper: int,
) -> None:
    """Test try_to_get_center_crop_size."""
    got_lower, got_upper = get_center_crop_size_from_1d_bbox(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        current_length=current_length,
        target_length=target_length,
    )
    assert got_lower == expected_lower
    assert got_upper == expected_upper


@pytest.mark.parametrize(
    (
        "bbox_min",
        "bbox_max",
        "current_size",
        "target_size",
        "expected_lower",
        "expected_upper",
    ),
    [
        (
            (0, 0, 0),
            (64, 44, 40),
            (64, 44, 40),
            (64, 44, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (0, 0, 0),
            (64, 44, 40),
            (64, 44, 40),
            (64, 64, 40),
            (0, 0, 0),
            (0, 0, 0),
        ),
        (
            (0, 30, 0),
            (20, 44, 40),
            (64, 44, 40),
            (20, 30, 30),
            (0, 14, 5),
            (44, 0, 5),
        ),
        (
            (0, 30, 0),
            (20, 44, 40),
            (65, 45, 41),
            (20, 30, 30),
            (0, 15, 5),
            (45, 0, 6),
        ),
    ],
    ids=[
        "3d_same",
        "3d_no_crop",
        "3d_even",
        "3d_odd",
    ],
)
def test_get_center_crop_size_from_bbox(
    bbox_min: tuple[int, ...],
    bbox_max: tuple[int, ...],
    current_size: tuple[int, ...],
    target_size: tuple[int, ...],
    expected_lower: tuple[int, ...],
    expected_upper: tuple[int, ...],
) -> None:
    """Test get_center_crop_size_from_bbox.

    Args:
        bbox_min: [start_in_1st_spatial_dim, ...], inclusive, starts at zero.
        bbox_max: [end_in_1st_spatial_dim, ...], exclusive, starts at zero.
        current_size: current shape of the image.
        target_size: target shape of the image.
        expected_lower: shape to crop on the lower side.
        expected_upper: shape to crop on the upper side.
    """
    got_lower, got_upper = get_center_crop_size_from_bbox(bbox_min, bbox_max, current_size, target_size)
    assert got_lower == expected_lower
    assert got_upper == expected_upper
