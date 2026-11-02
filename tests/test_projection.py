"""Unit tests for ``range_image_generator.projection`` public functions."""

import numpy as np

from range_image_generator import projection as P


# ---------------------------------------------------------------------------
# range_projection_idx_only
# ---------------------------------------------------------------------------


def test_range_projection_shape_and_dtype():
    point_cloud = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    proj_idx = P.range_projection_idx_only(
        point_cloud, fov_up=25.0, fov_down=-25.0, proj_H=8, proj_W=8, max_range=10.0
    )

    assert proj_idx.shape == (8, 8)
    assert proj_idx.dtype == np.int32


def test_range_projection_maps_points_to_pixels():
    point_cloud = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    proj_idx = P.range_projection_idx_only(
        point_cloud, fov_up=25.0, fov_down=-25.0, proj_H=8, proj_W=8, max_range=10.0
    )

    # The +x point (index 0) sits at yaw 0 -> mid column; the +y point
    # (index 1) sits at yaw -90 deg. The origin point is dropped (depth 0).
    assert proj_idx[4, 4] == 0
    assert proj_idx[4, 2] == 1
    assert set(proj_idx[proj_idx >= 0].tolist()) == {0, 1}


def test_range_projection_drops_points_beyond_max_range():
    point_cloud = np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    proj_idx = P.range_projection_idx_only(
        point_cloud, fov_up=25.0, fov_down=-25.0, proj_H=8, proj_W=8, max_range=10.0
    )

    assert proj_idx[4, 4] == 0
    assert 1 not in proj_idx.tolist()
    # Only the in-range point is present.
    assert np.count_nonzero(proj_idx >= 0) == 1


def test_range_projection_all_empty_returns_minus_one():
    point_cloud = np.array([[0.0, 0.0, 0.0]])

    proj_idx = P.range_projection_idx_only(
        point_cloud, fov_up=25.0, fov_down=-25.0, proj_H=4, proj_W=4, max_range=10.0
    )

    assert np.all(proj_idx == -1)


# ---------------------------------------------------------------------------
# proj_range_from_idx
# ---------------------------------------------------------------------------


def _sample_cloud():
    cloud = np.zeros((3, 8))
    cloud[0] = [1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 4.0, 7.0]
    cloud[1] = [2.0, 0.0, 0.0, 20.0, 5.0, 2.0, 5.0, 8.0]
    return cloud


def _sample_proj_idx():
    return np.array([[-1, 0], [-1, 1]], dtype=np.int32)


def test_proj_range_from_idx():
    pi = _sample_proj_idx()
    result = P.proj_range_from_idx(pi, _sample_cloud(), np.zeros(3))

    np.testing.assert_allclose(result, [[-1.0, 1.0], [-1.0, 2.0]])
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# proj_intensity_from_idx
# ---------------------------------------------------------------------------


def test_proj_intensity_from_idx_default_column():
    pi = _sample_proj_idx()
    result = P.proj_intensity_from_idx(pi, _sample_cloud())

    np.testing.assert_allclose(result, [[-1.0, 10.0], [-1.0, 20.0]])


def test_proj_intensity_from_idx_custom_field_mapping():
    pi = _sample_proj_idx()
    result = P.proj_intensity_from_idx(pi, _sample_cloud(), {"intensity": 4})

    np.testing.assert_allclose(result, [[-1.0, 0.0], [-1.0, 5.0]])


# ---------------------------------------------------------------------------
# proj_class_from_idx
# ---------------------------------------------------------------------------


def test_proj_class_from_idx_offset_by_one():
    pi = _sample_proj_idx()
    result = P.proj_class_from_idx(pi, _sample_cloud())

    # class column + 1 so 0 always means background.
    np.testing.assert_array_equal(result, [[0, 1], [0, 6]])
    assert result.dtype == np.uint8


def test_proj_class_from_idx_custom_field_mapping():
    pi = _sample_proj_idx()
    result = P.proj_class_from_idx(pi, _sample_cloud(), {"classification": 3})

    np.testing.assert_array_equal(result, [[0, 11], [0, 21]])


# ---------------------------------------------------------------------------
# proj_color_from_idx
# ---------------------------------------------------------------------------


def test_proj_color_from_idx_bgr_order():
    pi = _sample_proj_idx()
    result = P.proj_color_from_idx(pi, _sample_cloud())

    # Output is stored as BGR: point0 (r=1,g=4,b=7) -> [7,4,1].
    expected = np.array(
        [[[0, 0, 0], [7, 4, 1]], [[0, 0, 0], [8, 5, 2]]],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(result, expected)


def test_proj_color_from_idx_custom_field_mapping():
    pi = _sample_proj_idx()
    result = P.proj_color_from_idx(
        pi, _sample_cloud(), {"r": 3, "g": 4, "b": 0}
    )

    expected = np.array(
        [[[0, 0, 0], [1, 0, 10]], [[0, 0, 0], [2, 5, 20]]],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# filter_invalid_points
# ---------------------------------------------------------------------------


def test_filter_invalid_points_fills_single_hole():
    image = np.array(
        [[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
    )

    result = P.filter_invalid_points(
        image.copy(), kernel_size=3, threshold=3, max_passes=1
    )

    # The single hole is averaged from its valid neighbours (all 1.0).
    np.testing.assert_allclose(result, np.ones((3, 3)))


def test_filter_invalid_points_respects_zero_passes():
    image = np.array([[1.0, 1.0], [-1.0, 1.0]])

    result = P.filter_invalid_points(
        image.copy(), kernel_size=3, threshold=1, max_passes=0
    )

    np.testing.assert_array_equal(result, image)


def test_filter_invalid_points_all_invalid_stays_invalid():
    image = np.full((3, 3), -1.0)

    result = P.filter_invalid_points(
        image.copy(), kernel_size=3, threshold=3, max_passes=3
    )

    np.testing.assert_array_equal(result, -1.0)
