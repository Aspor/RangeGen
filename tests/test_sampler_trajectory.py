"""Unit tests for ``rangegen.samplers.generate_from_trajectory``.

Only the public API is tested: ``generate_poses_around_trajectory``. This
sampler is light (pure numpy/scipy, no multiprocessing or heavy I/O) so it is
exercised for real behaviour rather than mocked.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from range_image_generator.utils import Pose
from samplers.generate_from_trajectory import generate_poses_around_trajectory


def _trajectory(xs, orientation=(0.0, 0.0, 0.0)):
    """Build a straight-line trajectory along the X axis."""
    return [
        (float(i), np.array([float(x), 0.0, 0.0]), list(orientation))
        for i, x in enumerate(xs)
    ]


def _cfg(processing=None, sensor=None):
    cfg = {"sensor": sensor or {}, "processing": processing or {}}
    return cfg


# ---------------------------------------------------------------------------
# Stride sampling / counting / IDs
# ---------------------------------------------------------------------------


def test_generate_poses_around_trajectory_count_and_ids():
    # 5 points 5 m apart with a 4 m stride emits one pose per 5 m segment.
    poses, basenames = generate_poses_around_trajectory(
        _cfg(processing={"stride_distance": 4}, sensor={"sensor_orientation": [0, 0, 0]}),
        _trajectory([0, 5, 10, 15, 20]),
    )

    assert len(poses) == 3
    assert basenames == ["1", "2", "3"]
    assert all(isinstance(p, Pose) for p in poses)


def test_generate_poses_around_trajectory_positions_come_from_trajectory():
    poses, _ = generate_poses_around_trajectory(
        _cfg(processing={"stride_distance": 4}, sensor={"sensor_orientation": [0, 0, 0]}),
        _trajectory([0, 5, 10, 15, 20]),
    )

    allowed_x = {5.0, 10.0, 15.0}
    for pose in poses:
        assert float(pose.position[0]) in allowed_x


def test_generate_poses_around_trajectory_skip_start_and_end():
    poses, basenames = generate_poses_around_trajectory(
        _cfg(
            processing={"stride_distance": 4, "skip_start": 1, "skip_end": 1},
            sensor={"sensor_orientation": [0, 0, 0]},
        ),
        _trajectory([0, 5, 10, 15, 20]),
    )

    assert len(poses) == 2
    assert basenames == ["2", "3"]
    assert {float(p.position[0]) for p in poses} == {10.0, 15.0}


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


def test_generate_poses_around_trajectory_default_orientation_identity():
    poses, _ = generate_poses_around_trajectory(
        _cfg(processing={"stride_distance": 4}, sensor={"sensor_orientation": [0, 0, 0]}),
        _trajectory([0, 5, 10, 15, 20]),
    )

    for pose in poses:
        np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0])


def test_generate_poses_around_trajectory_applies_sensor_orientation():
    # NOTE: this sampler interprets sensor_orientation in DEGREES (degrees=True),
    # unlike the random-points sampler which uses radians.
    expected = R.from_euler("xyz", [90.0, 0.0, 0.0], degrees=True).as_quat()
    poses, _ = generate_poses_around_trajectory(
        _cfg(
            processing={"stride_distance": 4},
            sensor={"sensor_orientation": [90.0, 0.0, 0.0]},
        ),
        _trajectory([0, 5, 10, 15, 20]),
    )

    for pose in poses:
        np.testing.assert_allclose(pose.orientation, expected)


def test_generate_poses_around_trajectory_applies_trajectory_orientation():
    # use_trajectory_orientation is always truthy in this sampler, so the
    # trajectory orientation (about Z here) is applied on top of the (identity)
    # sensor orientation.
    expected = R.from_euler("xyz", [0.0, 0.0, 90.0], degrees=True).as_quat()
    poses, _ = generate_poses_around_trajectory(
        _cfg(processing={"stride_distance": 4}, sensor={"sensor_orientation": [0, 0, 0]}),
        _trajectory([0, 5, 10, 15, 20], orientation=(0.0, 0.0, 90.0)),
    )

    for pose in poses:
        np.testing.assert_allclose(pose.orientation, expected)
