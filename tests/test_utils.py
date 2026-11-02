"""Unit tests for ``range_image_generator.utils`` public API and methods."""

import os

import numpy as np

from range_image_generator import utils as U


# ---------------------------------------------------------------------------
# MemmapMetadata
# ---------------------------------------------------------------------------


def test_memmap_metadata_reopens_array(tmp_path):
    path = tmp_path / "points.bin"
    writer = np.memmap(path, dtype=np.float64, mode="w+", shape=(2, 3))
    writer[0] = [1.0, 2.0, 3.0]
    writer[1] = [4.0, 5.0, 6.0]
    writer.flush()
    del writer

    metadata = U.MemmapMetadata(
        filepath=str(path), dtype=np.dtype(np.float64), shape=(2, 3)
    )

    reopened = metadata.memmap(mode="r")
    np.testing.assert_array_equal(reopened, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


# ---------------------------------------------------------------------------
# FrameIntrinsics
# ---------------------------------------------------------------------------


def test_frame_intrinsics_as_tuple():
    intrinsics = U.FrameIntrinsics(
        fov_down=-20.0, fov_up=20.0, height=16, width=16, max_range=10.0
    )

    assert intrinsics.as_tuple() == (-20.0, 20.0, 16, 16, 10.0)


def test_frame_intrinsics_write_read_roundtrip(tmp_path):
    intrinsics = U.FrameIntrinsics(
        fov_down=-20.0, fov_up=20.0, height=16, width=16, max_range=10.0
    )
    path = tmp_path / "intrinsics.npy"

    intrinsics.write(str(path))
    loaded = U.FrameIntrinsics.read(str(path))

    assert isinstance(loaded, U.FrameIntrinsics)
    np.testing.assert_allclose(
        loaded.as_tuple(), (-20.0, 20.0, 16, 16, 10.0)
    )


# ---------------------------------------------------------------------------
# make_output_dirs
# ---------------------------------------------------------------------------


def test_make_output_dirs_creates_subdirectories(tmp_path):
    U.make_output_dirs(str(tmp_path), save_pc=True)

    expected = {
        "pose",
        "class",
        "rgb",
        "range",
        "intensity",
        "range_filtered",
        "intensity_filtered",
        "pc_index",
        "frame_instricts",
        "pc",
        "pc_transformed",
    }
    for sub in expected:
        assert os.path.isdir(os.path.join(str(tmp_path), sub))


def test_make_output_dirs_without_point_cloud(tmp_path):
    U.make_output_dirs(str(tmp_path), save_pc=False)

    assert os.path.isdir(os.path.join(str(tmp_path), "range"))
    assert not os.path.isdir(os.path.join(str(tmp_path), "pc"))


# ---------------------------------------------------------------------------
# Pose / Frame_Task
# ---------------------------------------------------------------------------


def test_pose_holds_position_and_orientation():
    position = np.array([1.0, 2.0, 3.0])
    orientation = np.array([0.0, 0.0, 0.0, 1.0])

    pose = U.Pose(position=position, orientation=orientation)

    np.testing.assert_array_equal(pose.position, position)
    np.testing.assert_array_equal(pose.orientation, orientation)


def test_frame_task_defaults():
    pose = U.Pose(position=np.zeros(3), orientation=np.array([0.0, 0.0, 0.0, 1.0]))
    points = U.MemmapMetadata(
        filepath="points.bin", dtype=np.dtype(np.float64), shape=(4, 8)
    )
    intrinsics = U.FrameIntrinsics(-20.0, 20.0, 16, 16, 10.0)

    task = U.Frame_Task(
        output_dir="out",
        basename="1234",
        pose=pose,
        points=points,
        frame_intrinsics=intrinsics,
    )

    assert task.output_dir == "out"
    assert task.basename == "1234"
    assert task.pose is pose
    assert task.frame_intrinsics is intrinsics
    assert task.save_pc is False
    assert task.number_of_pos_perturbations == 0
    assert task.field_mapping is None


# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------


def test_instance_num_points():
    instance = U.Instance(
        instance_id=1,
        class_id=2,
        point_indices=np.array([0, 1, 2]),
        centroid=np.array([1.0, 2.0, 3.0]),
    )

    assert instance.num_points() == 3


# ---------------------------------------------------------------------------
# parse_pose
# ---------------------------------------------------------------------------


def test_parse_pose_from_wrapped_values():
    text = (
        "np.float64(1.0) np.float64(2.0) np.float64(3.0) "
        "np.float64(4.0) np.float64(5.0) np.float64(6.0)"
    )

    pose = U.parse_pose(text)

    assert list(pose.position) == [1.0, 2.0, 3.0]
    assert list(pose.orientation) == [4.0, 5.0, 6.0]


def test_parse_pose_from_plain_values():
    pose = U.parse_pose("1 2 3 4 5 6")

    assert list(pose.position) == [1.0, 2.0, 3.0]
    assert list(pose.orientation) == [4.0, 5.0, 6.0]
