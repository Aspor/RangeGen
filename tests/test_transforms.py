"""Unit tests for ``range_image_generator.transforms`` public functions."""

import numpy as np
from scipy.spatial.transform import Rotation as R

from range_image_generator import transforms as T


# ---------------------------------------------------------------------------
# transform_points
# ---------------------------------------------------------------------------


def test_transform_points_identity_is_translation_only():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    position = np.array([1.0, 1.0, 1.0])

    result = T.transform_points(points, position, R.identity())

    np.testing.assert_allclose(result, points - position)


def test_transform_points_applies_rotation():
    points = np.array([[1.0, 0.0, 0.0]])
    rotation = R.from_euler("z", 90, degrees=True)

    result = T.transform_points(points, np.zeros(3), rotation)

    # +x axis rotated 90 degrees about z -> +y axis.
    np.testing.assert_allclose(result, [[0.0, 1.0, 0.0]], atol=1e-6)


def test_transform_points_combined_translation_and_rotation():
    points = np.array([[1.0, 0.0, 0.0]])
    position = np.array([10.0, 0.0, 0.0])
    rotation = R.from_euler("z", 90, degrees=True)

    result = T.transform_points(points, position, rotation)

    expected = (rotation.as_matrix() @ (points - position).T).T
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# perturb_position
# ---------------------------------------------------------------------------


def test_perturb_position_zero_ranges_is_identity():
    position = np.array([1.0, 2.0, 3.0])

    result = T.perturb_position(position, (0, 0), (0, 0), (0, 0))

    np.testing.assert_allclose(result, position)


def test_perturb_position_stays_within_ranges():
    position = np.array([0.0, 0.0, 0.0])
    ranges = ((1.0, 2.0), (-1.0, 1.0), (0.0, 5.0))

    for _ in range(50):
        result = T.perturb_position(position, *ranges)
        assert result[0] >= 1.0 and result[0] <= 2.0
        assert result[1] >= -1.0 and result[1] <= 1.0
        assert result[2] >= 0.0 and result[2] <= 5.0


# ---------------------------------------------------------------------------
# perturb_orientation
# ---------------------------------------------------------------------------


def test_perturb_orientation_zero_ranges_is_identity():
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])

    result = T.perturb_orientation(quaternion, (0, 0), (0, 0), (0, 0))

    assert result.shape == (4,)
    np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 1.0], atol=1e-6)


def test_perturb_orientation_returns_quaternion():
    quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    np.random.seed(42)

    result = T.perturb_orientation(quaternion, (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0))

    assert result.shape == (4,)
    # A quaternion is (approximately) unit length.
    np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# crop_points_by_distance
# ---------------------------------------------------------------------------


def test_crop_points_by_distance_selects_square_window():
    cloud = np.array(
        [
            [0.0, 0.0, 0.0], #kept
            [5.0, 0.0, 0.0], #kept
            [10.0, 0.0, 0.0], #discarded
            [0.0, 5.0, 0.0], #kept
            [0.0, 0.0, 9.0], #kept
            [100.0, 100.0, 100.0], #discaded
        ]
    )

    cropped, indices = T.crop_points_by_distance(cloud, np.array([0.0, 0.0, 0.0]), 6.0)

    # Only x/y within +/-6 are kept; z is ignored.
    assert list(indices.flatten()) == [0, 1, 3, 4]
    np.testing.assert_allclose(cropped, cloud[[0, 1, 3, 4], :3])
    assert cropped.shape == (4, 3)
    assert indices.dtype == np.uint32


def test_crop_points_by_distance_empty_when_none_in_range():
    cloud = np.array([[100.0, 100.0, 0.0]])

    cropped, indices = T.crop_points_by_distance(cloud, np.array([0.0, 0.0, 0.0]), 6.0)

    assert cropped.shape == (0, 3)
    assert indices.shape == (0, 1)


# ---------------------------------------------------------------------------
# batch_crop_points_by_distance
# ---------------------------------------------------------------------------


def test_batch_crop_points_by_distance_covers_all_positions():
    cloud = np.array(
        [
            [3.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [7.0, 7.0, 0.0],
        ]
    )
    positions = np.array([[4.0, 0.0, 0.0], [6.0, 0.0, 0.0]])

    cropped, indices = T.batch_crop_points_by_distance(cloud, positions, 2.0)

    # x within [min_x-2, max_x+2] = [2, 8]; y within [-2, 2].
    assert list(indices.flatten()) == [0, 1]
    assert cropped.shape == (2, 3)
