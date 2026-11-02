"""Unit tests for ``range_image_generator.frame_generator``.

The two public entry points (``frame_randomizer`` / ``batched_frame_randomizer``)
are tested as pure orchestration: their heavy internals are mocked, and every
mocked internal has its own real-behaviour test elsewhere in this suite
(``randomize_frames``, ``transform_and_project`` and ``save_images`` here;
``crop_points_by_distance`` / ``perturb_*`` in test_transforms;
``make_output_dirs`` in test_utils).
"""

import os
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation as R

from range_image_generator import frame_generator as FG
from range_image_generator import utils as U

# A valid ETRS-TM35FIN position (Oulu, Finland) so EXIF/GPS transforms succeed.
VALID_POSITION = np.array([348961.7, 6710252.8, 0.0])
INTRINSICS = U.FrameIntrinsics(
    fov_down=-20.0, fov_up=20.0, height=16, width=16, max_range=10.0
)
IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _make_task(output_dir, pose, basename):
    """Build a task test-double exposing exactly the attributes the drivers read.

    ``pose`` and ``basename`` may be a single value (``frame_randomizer``) or a
    list of values (``batched_frame_randomizer``), so they are left untyped.
    """
    task = mock.Mock()
    task_points = mock.Mock()
    task_points.memmap.return_value = np.zeros((4, 8))
    task_points.field_mapping = None
    task.output_dir = output_dir
    task.save_pc = False
    task.points = task_points
    task.pose = pose
    task.frame_intrinsics = INTRINSICS
    task.basename = basename
    task.number_of_pos_perturbations = 0
    task.number_of_orientation_perturbations = 0
    task.position_perturbation = None
    task.orientation_perturbation = None
    return task


# ---------------------------------------------------------------------------
# frame_randomizer (orchestration)
# ---------------------------------------------------------------------------


def test_frame_randomizer_no_perturb_delegates_and_returns(tmp_path):
    pose = U.Pose(position=np.array([1.0, 2.0, 3.0]), orientation=IDENTITY_QUAT)
    task = _make_task(str(tmp_path), pose, basename="1234")

    points_mmap = np.zeros((4, 8))
    original_indices = np.arange(4).reshape(-1, 1)
    sentinel = (["IMG"], ["ANN"], "OUT")

    with (
        mock.patch.object(FG, "make_output_dirs") as m_dirs,
        mock.patch.object(
            FG, "crop_points_by_distance", return_value=(points_mmap, original_indices)
        ) as m_crop,
        mock.patch.object(FG, "randomize_frames", return_value=sentinel) as m_rand,
    ):
        result = FG.frame_randomizer(task)

    assert result == sentinel
    m_dirs.assert_called_once_with(str(tmp_path), False)
    m_crop.assert_called_once()
    # Crop is centred on the pose position with range = max_range * 2.5.
    assert m_crop.call_args.kwargs["range"] == 25.0
    np.testing.assert_array_equal(m_crop.call_args.args[1], pose.position)
    # Delegates to randomize_frames with no_perturb and the pose fields.
    assert m_rand.call_args.args[0] is True
    assert m_rand.call_args.args[6] == "1234"
    assert m_rand.call_args.kwargs["field_mapping"] is None


# ---------------------------------------------------------------------------
# batched_frame_randomizer (orchestration)
# ---------------------------------------------------------------------------


def test_batched_frame_randomizer_calls_per_pose(tmp_path):
    pose0 = U.Pose(position=np.array([0.0, 0.0, 0.0]), orientation=IDENTITY_QUAT)
    pose1 = U.Pose(position=np.array([9.0, 8.0, 7.0]), orientation=IDENTITY_QUAT)
    task = _make_task(
        str(tmp_path), pose=[pose0, pose1], basename=["1111", "2222"]
    )

    points_mmap = np.zeros((4, 8))
    original_indices = np.arange(4).reshape(-1, 1)

    with (
        mock.patch.object(FG, "make_output_dirs") as m_dirs,
        mock.patch.object(
            FG,
            "batch_crop_points_by_distance",
            return_value=(points_mmap, original_indices),
        ) as m_crop,
        mock.patch.object(FG, "randomize_frames") as m_rand,
    ):
        # The batched driver unpacks exactly two values per call.
        m_rand.side_effect = [(["i0"], ["a0"]), (["i1"], ["a1"])]
        result = FG.batched_frame_randomizer(task)

    assert m_rand.call_count == 2
    first, second = m_rand.call_args_list
    assert first.args[2] is pose0.position and first.args[6] == "1111"
    assert second.args[2] is pose1.position and second.args[6] == "2222"
    # The batch crop covers both positions.
    assert len(m_crop.call_args.args[1]) == 2
    assert m_crop.call_args.kwargs["range"] == 25.0
    # Returns the last pose's (images, annotations).
    assert result == (["i1"], ["a1"])
    m_dirs.assert_called_once()


# ---------------------------------------------------------------------------
# randomize_frames (real, heavy transform_and_project mocked)
# ---------------------------------------------------------------------------


def test_randomize_frames_no_perturb(tmp_path):
    points_mmap = np.zeros((4, 8))
    original_indices = np.arange(4).reshape(-1, 1)
    original_pc = np.zeros((6, 8))

    with mock.patch.object(
        FG, "transform_and_project", return_value=(True, ["IMG"], ["ANN"])
    ) as m_tp:
        result = FG.randomize_frames(
            True,
            points_mmap,
            VALID_POSITION,
            IDENTITY_QUAT,
            str(tmp_path),
            INTRINSICS,
            "1234",
            original_indices,
            original_pc,
            False,
            0,
            0,
            None,
            None,
            field_mapping=None,
        )

    assert result == (["IMG"], ["ANN"], str(tmp_path))
    m_tp.assert_called_once()
    assert m_tp.call_args.kwargs["original_pc"] is original_pc
    assert m_tp.call_args.kwargs["field_mapping"] is None


def test_randomize_frames_position_perturbations(tmp_path):
    with (
        mock.patch.object(
            FG,
            "perturb_position",
            side_effect=[np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])],
        ),
        mock.patch.object(
            FG, "transform_and_project", return_value=(True, ["I"], ["A"])
        ) as m_tp,
    ):
        result = FG.randomize_frames(
            False,
            np.zeros((4, 8)),
            VALID_POSITION,
            IDENTITY_QUAT,
            str(tmp_path),
            INTRINSICS,
            "base",
            np.zeros((4, 1)),
            np.zeros((4, 8)),
            False,
            2,
            0,
            ((0, 1), (0, 1), (0, 1)),
            None,
            field_mapping=None,
        )

    assert m_tp.call_count == 2
    assert m_tp.call_args_list[0].args[5] == "base_0"
    assert m_tp.call_args_list[1].args[5] == "base_1"
    # images/annotations are accumulated across positions.
    assert result[0] == ["I", "I"]
    assert result[1] == ["A", "A"]


def test_randomize_frames_position_times_orientation(tmp_path):
    with (
        mock.patch.object(FG, "perturb_position", side_effect=[np.zeros(3)] * 2),
        mock.patch.object(FG, "perturb_orientation", side_effect=[IDENTITY_QUAT] * 6),
        mock.patch.object(
            FG, "transform_and_project", return_value=(True, ["I"], ["A"])
        ) as m_tp,
    ):
        FG.randomize_frames(
            False,
            np.zeros((4, 8)),
            VALID_POSITION,
            IDENTITY_QUAT,
            str(tmp_path),
            INTRINSICS,
            "base",
            np.zeros((4, 1)),
            np.zeros((4, 8)),
            False,
            2,
            3,
            ((0, 1), (0, 1), (0, 1)),
            ((0, 1), (0, 1), (0, 1)),
            field_mapping=None,
        )

    # 2 positions x 3 orientations = 6 frames.
    assert m_tp.call_count == 6
    assert m_tp.call_args_list[0].args[5] == "base_0_0"
    assert m_tp.call_args_list[-1].args[5] == "base_1_2"


def test_randomize_frames_orientation_only(tmp_path):
    with (
        mock.patch.object(FG, "perturb_orientation", side_effect=[IDENTITY_QUAT] * 3),
        mock.patch.object(
            FG, "transform_and_project", return_value=(True, ["I"], ["A"])
        ) as m_tp,
    ):
        FG.randomize_frames(
            False,
            np.zeros((4, 8)),
            VALID_POSITION,
            IDENTITY_QUAT,
            str(tmp_path),
            INTRINSICS,
            "base",
            np.zeros((4, 1)),
            np.zeros((4, 8)),
            False,
            0,
            3,
            None,
            ((0, 1), (0, 1), (0, 1)),
            field_mapping=None,
        )

    assert m_tp.call_count == 3
    assert [c.args[5] for c in m_tp.call_args_list] == ["base_0", "base_1", "base_2"]


# ---------------------------------------------------------------------------
# transform_and_project (real transform/project, save_images mocked)
# ---------------------------------------------------------------------------


def test_transform_and_project_writes_pose_and_delegates(tmp_path):
    points = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 5.0]]
    ).reshape(3, 3)
    points = np.column_stack([points, np.zeros((3, 5))])
    original_pc = points.copy()
    original_indices = np.arange(3).reshape(-1, 1)

    U.make_output_dirs(str(tmp_path), save_pc=False)

    with mock.patch.object(FG, "save_images", return_value=(["IMG"], ["ANN"])) as m_save:
        ok, images, annotations = FG.transform_and_project(
            points,
            VALID_POSITION,
            IDENTITY_QUAT,
            str(tmp_path),
            INTRINSICS,
            "1234",
            original_indices,
            original_pc,
            timestamp=None,
            save_pc=False,
            field_mapping=None,
        )

    assert ok is True
    assert images == ["IMG"]
    assert annotations == ["ANN"]
    assert os.path.exists(os.path.join(str(tmp_path), "pose", "1234.txt"))
    assert os.path.exists(os.path.join(str(tmp_path), "frame_instricts", "1234.npy"))

    save_kwargs = m_save.call_args.kwargs
    assert save_kwargs["points"] is original_pc
    assert save_kwargs["filename"] == "1234"
    assert save_kwargs["output_dir"] == str(tmp_path)
    assert save_kwargs["position"] is VALID_POSITION
    assert save_kwargs["proj_idx"].shape == (16, 16)


# ---------------------------------------------------------------------------
# save_images (real end-to-end with a small image)
# ---------------------------------------------------------------------------


def _save_images_setup():
    H = W = 16
    proj_idx = np.full((H, W), -1, dtype=np.int32)
    proj_idx[3:9, 3:9] = np.arange(36).reshape(6, 6)  # cropped indices 0..35
    original_indices = (np.arange(36) + 10).reshape(-1, 1)
    n = 46
    points = np.zeros((n, 8))
    for k in range(36):
        row = 10 + k
        points[row, 0] = float(row)
        points[row, 3] = 100.0  # intensity
        points[row, 4] = 2.0    # class -> class image value 3 (> 1)
    return proj_idx, original_indices, points


def test_save_images_writes_outputs_and_builds_coco(tmp_path):
    U.make_output_dirs(str(tmp_path), save_pc=True)
    proj_idx, original_indices, points = _save_images_setup()

    image_infos, annotations = FG.save_images(
        points,
        proj_idx.copy(),
        str(tmp_path),
        "1234",
        VALID_POSITION,
        R.identity(),
        None,
        original_indices,
        save_pc=True,
        field_mapping=None,
    )

    # One image record with the projection dimensions.
    assert len(image_infos) == 1
    assert image_infos[0]["width"] == 16
    assert image_infos[0]["height"] == 16
    assert image_infos[0]["file_name"] == "1234.png"
    # The 6x6 class block produces exactly one COCO annotation.
    assert len(annotations) == 1
    assert annotations[0]["image_id"] == image_infos[0]["id"]

    for sub, ext in [
        ("pc_index", "npy"),
        ("range", "png"),
        ("intensity", "png"),
        ("rgb", "png"),
        ("class", "png"),
        ("pc", "pcd"),
        ("pc_transformed", "pcd"),
    ]:
        assert os.path.exists(os.path.join(str(tmp_path), sub, f"1234.{ext}"))


def test_save_images_no_classes_returns_empty(tmp_path):
    U.make_output_dirs(str(tmp_path), save_pc=False)
    proj_idx, original_indices, points = _save_images_setup()
    # Zero out the classification column -> no foreground class.
    points[:, 4] = 0.0

    image_infos, annotations = FG.save_images(
        points,
        proj_idx.copy(),
        str(tmp_path),
        "1234",
        VALID_POSITION,
        R.identity(),
        None,
        original_indices,
        save_pc=False,
        field_mapping=None,
    )

    assert image_infos == []
    assert annotations == []
