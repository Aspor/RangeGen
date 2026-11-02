"""Unit tests for ``rangegen.samplers.generate_around_objects``.

Only the public API is tested: ``generate_poses_around_objects``. This sampler
is light (numpy/scipy + ``random``) so it is exercised for real. Positions are
jittered randomly per sample, so assertions cover pose count, base IDs,
orientation and the source label — never the exact position.
"""

import numpy as np

from range_image_generator.utils import Pose
from samplers.generate_around_objects import generate_poses_around_objects


def _trajectory(point=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0)):
    return [(0.0, np.array(point), list(orientation))]


def _cfg(categories=None, sampling_points=2, sensor_orientation=(0.0, 0.0, 0.0)):
    return {
        "sensor": {"sensor_orientation": list(sensor_orientation)},
        "image": {"max_range": 50},
        "processing": {
            "categories": categories,
            "sampling_points": sampling_points,
        },
    }


def _detections():
    # instance 101 is class 1, instance 102 is class 3.
    return [
        (np.array([1.0, 2.0, 0.5]), 101, 1),
        (np.array([3.0, 4.0, 0.5]), 102, 3),
    ]


def test_generate_poses_around_objects_filters_by_category():
    poses, base_ids, label = generate_poses_around_objects(
        _cfg(categories=[1, 2], sampling_points=2),
        _detections(),
        _trajectory(),
    )

    # Class 3 is dropped, so only the class-1 object (instance 101) is sampled
    # twice.
    assert label == "objects"
    assert len(poses) == 2
    assert base_ids == ["0101_0", "0101_1"]
    assert all(isinstance(p, Pose) for p in poses)


def test_generate_poses_around_objects_keeps_all_when_categories_none():
    poses, base_ids, label = generate_poses_around_objects(
        _cfg(categories=None, sampling_points=2),
        _detections(),
        _trajectory(),
    )

    assert label == "objects"
    assert len(poses) == 4
    assert base_ids == ["0101_0", "0101_1", "0102_0", "0102_1"]


def test_generate_poses_around_objects_identity_orientation():
    # identity trajectory orientation * identity sensor orientation.
    poses, _, _ = generate_poses_around_objects(
        _cfg(categories=None, sampling_points=1),
        _detections(),
        _trajectory(orientation=(0.0, 0.0, 0.0)),
    )

    for pose in poses:
        np.testing.assert_allclose(pose.orientation, [0.0, 0.0, 0.0, 1.0])
