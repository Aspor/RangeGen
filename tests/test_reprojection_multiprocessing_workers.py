"""Unit tests for ``rangegen.reprojection.multiprocessing_workers``.

``calculate_annotation_indexes`` is a multiprocessing orchestrator (always runs
a ``Pool``), so the pool itself is mocked and its orchestration is asserted; the
per-image ``deprojection_worker`` it maps over is tested separately with its
heavy dependencies mocked. ``make_tasks`` and ``decode_segmentation`` are light
enough to be exercised for real.
"""

from unittest import mock

import numpy as np
from pycocotools import mask as maskUtils

from reprojection import multiprocessing_workers as MW
from range_image_generator import utils as U


# ---------------------------------------------------------------------------
# calculate_annotation_indexes (multiprocessing orchestrator)
# ---------------------------------------------------------------------------


def test_calculate_annotation_indexes_merges_worker_results():
    tasks = [object(), object()]

    with (
        mock.patch.object(MW, "Pool") as m_pool,
        mock.patch.object(MW, "deprojection_worker") as m_worker,
    ):
        cm = m_pool.return_value.__enter__.return_value
        cm.imap_unordered.return_value = [{1: "a"}, {2: "b"}]
        result = MW.calculate_annotation_indexes(tasks)

    # Worker outputs are merged into a single mapping.
    assert result == {1: "a", 2: "b"}
    m_pool.assert_called_once_with(6)
    cm.imap_unordered.assert_called_once()
    args = cm.imap_unordered.call_args.args
    assert args[0] is m_worker
    assert args[1] == tasks


# ---------------------------------------------------------------------------
# make_tasks
# ---------------------------------------------------------------------------


def _coco(image_ids, names, annotations, ann_ids):
    coco = mock.Mock()
    coco.getImgIds.return_value = image_ids
    coco.loadImgs.side_effect = lambda img_id: [
        {"id": img_id, "extra": {"name": names[img_id]}, "height": 4, "width": 4}
    ]
    coco.getAnnIds.return_value = ann_ids
    coco.loadAnns.return_value = annotations
    return coco


def test_make_tasks_builds_deprojection_tasks(tmp_path):
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    (pose_dir / "frameA.txt").write_text("0.0 0.0 0.0 0.0 0.0 0.0")
    (pose_dir / "frameB.txt").write_text("1.0 2.0 3.0 0.0 0.0 0.0")

    annotations = [{"id": 100, "category_id": 1}, {"id": 101, "category_id": 2}]
    coco = _coco([1, 2], {1: "frameA.png", 2: "frameB.png"}, annotations, [100, 101])
    meta = mock.Mock()
    cfg = {
        "processing": {},
        "paths": {"datasetroot": str(tmp_path)},
        "image": {"width": 4, "height": 4, "max_range": 10, "elevation_range": [-20, 20]},
    }

    tasks = MW.make_tasks(coco, meta, cfg)

    assert len(tasks) == 2
    assert all(isinstance(t, U.DeprojectionTask) for t in tasks)
    # Poses are parsed from the matched pose files.
    np.testing.assert_array_equal(tasks[0].pose.position, [0, 0, 0])
    np.testing.assert_array_equal(tasks[1].pose.position, [1, 2, 3])
    assert tasks[0].annotations == annotations
    assert tasks[0].points_memmap is meta
    # Missing .npy intrinsics fall back to the default constructed above.
    assert tasks[0].frame_intrinsics.height == 4
    assert tasks[0].frame_intrinsics.max_range == 10


# ---------------------------------------------------------------------------
# decode_segmentation
# ---------------------------------------------------------------------------


def _block_mask():
    m = np.zeros((4, 4), dtype=np.uint8)
    m[1:3, 1:3] = 1
    return np.asfortranarray(m)


def test_decode_segmentation_decodes_rle():
    m = _block_mask()
    rle = maskUtils.encode(m)

    np.testing.assert_array_equal(MW.decode_segmentation(rle, 4, 4), m)


# ---------------------------------------------------------------------------
# deprojection_worker
# ---------------------------------------------------------------------------


def _worker_task(original_pc, annotations, img_metadata):
    pose = U.Pose(
        position=np.array([0.0, 0.0, 0.0]), orientation=np.array([0.0, 0.0, 0.0])
    )
    task = mock.Mock()
    task.annotations = annotations
    task.image_metadata = img_metadata
    task.pose = pose
    task.points_memmap.memmap.return_value = original_pc
    task.frame_intrinsics = U.FrameIntrinsics(
        fov_down=-20.0, fov_up=20.0, height=4, width=4, max_range=10.0
    )
    return task


def test_deprojection_worker_builds_instance():
    original_pc = np.array(
        [
            [0.0, 0, 0, 0, 0, 0],
            [1.0, 0, 0, 0, 0, 0],
            [0.0, 1.0, 0, 0, 0, 0],
            [0.0, 0, 0, 0, 0, 0],
        ]
    )
    original_indices = np.arange(4).reshape(-1, 1)
    point_cloud = original_pc.copy()
    proj_idx = np.full((4, 4), -1, dtype=np.int64)
    proj_idx[1, 1] = 0
    proj_idx[1, 2] = 1
    proj_idx[2, 1] = 2
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[1, 2] = True
    mask[2, 1] = True
    mask[2, 2] = True
    object_indices = np.array([0, 1, 2])
    progress = [np.array([0, 1, 2]), np.array([0, 1, 2])]

    annotations = [{"id": 42, "category_id": 7, "segmentation": "S"}]
    img_metadata = {"id": 1, "height": 4, "width": 4}
    task = _worker_task(original_pc, annotations, img_metadata)

    with (
        mock.patch.object(
            MW, "crop_points_by_distance", return_value=(point_cloud, original_indices)
        ) as m_crop,
        mock.patch.object(MW, "transform_points", return_value=point_cloud),
        mock.patch.object(MW, "range_projection_idx_only", return_value=proj_idx),
        mock.patch.object(MW, "decode_segmentation", return_value=mask),
        mock.patch.object(
            MW, "expand_labels", return_value=(object_indices, progress)
        ) as m_expand,
    ):
        poles = MW.deprojection_worker(task)

    assert set(poles) == {42}
    inst = poles[42]
    assert isinstance(inst, U.Instance)
    assert inst.instance_id == 42
    assert inst.class_id == 7
    np.testing.assert_array_equal(inst.point_indices, [0, 1, 2])
    np.testing.assert_allclose(inst.centroid, [1 / 3, 1 / 3, 0.0])
    # Crop is centred on the pose with range = max_range * 1.5.
    assert m_crop.call_args.args[2] == 15.0
    # Seed indices forwarded to the label-expansion pipeline.
    np.testing.assert_array_equal(m_expand.call_args.kwargs["mask_indices"], [0, 1, 2])
    np.testing.assert_array_equal(inst.extra["unexpanded"], [0, 1, 2])
    np.testing.assert_array_equal(inst.extra["expanded"], [0, 1, 2])
    assert len(inst.extra["labelExpansion"]) == 2


def test_deprojection_worker_empty_point_cloud_returns_empty():
    original_pc = np.zeros((4, 6))
    empty_cloud = np.zeros((0, 6))
    annotations = [{"id": 1, "category_id": 1, "segmentation": "S"}]
    img_metadata = {"id": 1, "height": 4, "width": 4}
    task = _worker_task(original_pc, annotations, img_metadata)

    with (
        mock.patch.object(
            MW,
            "crop_points_by_distance",
            return_value=(empty_cloud, np.zeros((0, 1), dtype=np.int64)),
        ) as m_crop,
        mock.patch.object(MW, "transform_points"),
        mock.patch.object(MW, "range_projection_idx_only"),
        mock.patch.object(MW, "decode_segmentation"),
        mock.patch.object(MW, "expand_labels"),
    ):
        poles = MW.deprojection_worker(task)

    assert poles == {}
    m_crop.assert_called_once()


def test_deprojection_worker_no_projection_returns_empty():
    original_pc = np.array([[0.0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0]])
    point_cloud = original_pc.copy()
    annotations = [{"id": 1, "category_id": 1, "segmentation": "S"}]
    img_metadata = {"id": 1, "height": 4, "width": 4}
    task = _worker_task(original_pc, annotations, img_metadata)

    with (
        mock.patch.object(
            MW,
            "crop_points_by_distance",
            return_value=(point_cloud, np.arange(2).reshape(-1, 1)),
        ),
        mock.patch.object(MW, "transform_points", return_value=point_cloud),
        mock.patch.object(MW, "range_projection_idx_only", return_value=np.zeros(0)),
        mock.patch.object(MW, "decode_segmentation"),
        mock.patch.object(MW, "expand_labels"),
    ):
        poles = MW.deprojection_worker(task)

    assert poles == {}
