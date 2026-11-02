"""Unit tests for ``rangegen.samplers.generate_from_random_points``.

Only the public API is tested: ``random_sampling`` and
``calculate_random_sampling_points`` (the ``_``-prefixed helpers are internal).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from range_image_generator.utils import Pose
from samplers.generate_from_random_points import (
    calculate_random_sampling_points,
    random_sampling,
)


def _cloud(n: int = 4, cls: float = 2.0) -> np.ndarray:
    points = np.zeros((n, 8))
    for i in range(n):
        points[i, 0] = float(i * 10)
        points[i, 1] = float(i)
        points[i, 2] = 0.0
        points[i, 3] = 100.0
        points[i, 4] = cls
    return points


# ---------------------------------------------------------------------------
# calculate_random_sampling_points
# ---------------------------------------------------------------------------


def test_calculate_random_sampling_points_count_and_ids():
    poses, base_ids = calculate_random_sampling_points(
        _cloud(), (0, 1), (0, 1), (0, 1), 5, sensor_orientation=[0, 0, 0]
    )
    poses = list(poses)
    base_ids = list(base_ids)

    assert len(poses) == 5
    assert base_ids == ["0", "1", "2", "3", "4"]
    assert all(isinstance(p, Pose) for p in poses)


def test_calculate_random_sampling_points_positions_come_from_cloud():
    cloud = _cloud()
    poses, _ = calculate_random_sampling_points(
        cloud, (0, 1), (0, 1), (0, 1), 12, sensor_orientation=[0, 0, 0]
    )
    allowed_x = {float(i * 10) for i in range(len(cloud))}

    for pose in poses:
        assert float(pose.position[0]) in allowed_x


def test_calculate_random_sampling_points_default_orientation_identity():
    poses, _ = calculate_random_sampling_points(
        _cloud(), (0, 1), (0, 1), (0, 1), 3, sensor_orientation=[0, 0, 0]
    )

    for pose in poses:
        np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0])


def test_calculate_random_sampling_points_applies_sensor_orientation():
    # NOTE: this sampler interprets sensor_orientation in radians (no degrees flag).
    expected = R.from_euler("xyz", [90.0, 0.0, 0.0]).as_quat()
    poses, _ = calculate_random_sampling_points(
        _cloud(), (0, 1), (0, 1), (0, 1), 3, sensor_orientation=[90.0, 0.0, 0.0]
    )

    for pose in poses:
        np.testing.assert_allclose(pose.orientation, expected)


def test_calculate_random_sampling_points_filters_by_class():
    cloud = np.zeros((4, 8))
    for i in range(4):
        cloud[i, 0] = float(i * 10)
        cloud[i, 4] = 3.0 if i < 2 else 5.0

    poses, _ = calculate_random_sampling_points(
        cloud, (0, 1), (0, 1), (0, 1), 12, classes=[3]
    )

    # Only the class-3 rows (x = 0 and 10) may be sampled.
    allowed_x = {0.0, 10.0}
    for pose in poses:
        assert float(pose.position[0]) in allowed_x


# ---------------------------------------------------------------------------
# random_sampling
# ---------------------------------------------------------------------------


def test_random_sampling_reads_processing_and_sensor_cfg():
    cfg = {
        "processing": {"sampling_points": 4},
        "sensor": {"sensor_orientation": [0, 0, 0]},
    }

    poses, base_ids = random_sampling(_cloud(), cfg)

    assert len(list(poses)) == 4
    assert list(base_ids) == ["0", "1", "2", "3"]


def test_random_sampling_forwards_classes_argument():
    cfg = {
        "processing": {"sampling_points": 6},
        "sensor": {"sensor_orientation": [0, 0, 0]},
    }
    cloud = np.zeros((4, 8))
    for i in range(4):
        cloud[i, 0] = float(i * 10)
        cloud[i, 4] = 3.0 if i < 2 else 5.0

    poses, _ = random_sampling(cloud, cfg, classes=[3])

    allowed_x = {0.0, 10.0}
    for pose in poses:
        assert float(pose.position[0]) in allowed_x
