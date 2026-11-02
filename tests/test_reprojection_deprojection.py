"""Unit tests for ``reprojection.deprojection``.

``label_pointcloud_coco`` is the package's heavy orchestrator (its only external
entry point, called from ``main.py``), so its heavy dependencies are mocked and
its orchestration asserted. The internal helpers it relies on —
``project_to_pointcloud`` and ``combine_ids`` — are then tested for real
behaviour, satisfying "every mocked internal gets its own real test".
"""

import os
from unittest import mock

import numpy as np
from scipy.spatial import KDTree

from reprojection import deprojection as DEP
from range_image_generator import utils as U


# ---------------------------------------------------------------------------
# label_pointcloud_coco (orchestration)
# ---------------------------------------------------------------------------


def test_label_pointcloud_coco_orchestrates(tmp_path):
    memmap_arr = np.zeros((4, 8))
    meta = mock.Mock()
    meta.memmap.return_value = memmap_arr
    meta.field_mapping = None

    coco_inst = mock.Mock()
    coco_inst.loadCats.return_value = [{"id": 1, "name": "pole"}]
    sentinel_tasks = ["T1"]
    sentinel_poles = {1: {"id": 1}}
    cfg = {"paths": {"output_dir": str(tmp_path / "out"), "laz_dir": "lazdir"}}

    with (
        mock.patch.object(DEP, "COCO", return_value=coco_inst) as m_coco,
        mock.patch.object(DEP, "make_tasks", return_value=sentinel_tasks) as m_tasks,
        mock.patch.object(
            DEP, "calculate_annotation_indexes", return_value=sentinel_poles
        ) as m_calc,
        mock.patch.object(DEP, "project_to_pointcloud", return_value=memmap_arr) as m_proj,
        mock.patch.object(DEP, "update_classification") as m_upd,
    ):
        poles, categories = DEP.label_pointcloud_coco(meta, "coco.json", cfg)

    # Return value passes straight through the pipeline.
    assert poles is sentinel_poles
    assert categories == [{"id": 1, "name": "pole"}]

    # Orchestration order and arguments.
    m_coco.assert_called_once_with("coco.json")
    m_tasks.assert_called_once_with(coco_inst, meta, cfg)
    m_calc.assert_called_once_with(sentinel_tasks)
    m_proj.assert_called_once_with(sentinel_poles, memmap_arr, None)

    m_upd.assert_called_once()
    up_args = m_upd.call_args.args
    assert up_args[0] is memmap_arr
    assert up_args[1] == "lazdir"
    out_dir = str(tmp_path / "out")
    assert up_args[2] == os.path.join(out_dir, "classificated_pointcloud.laz")

    # Side-effect outputs are written.
    assert os.path.exists(os.path.join(out_dir, "pole.pickle"))
    assert os.path.exists(os.path.join(out_dir, "categories.pickle"))


# ---------------------------------------------------------------------------
# project_to_pointcloud
# ---------------------------------------------------------------------------


def _pole(indices, instance_id, class_id, cloud):
    return U.Instance(
        instance_id=instance_id,
        class_id=class_id,
        point_indices=np.asarray(indices),
        centroid=cloud[np.asarray(indices), :3].mean(axis=0),
        extra={
            "unexpanded": np.asarray(indices),
            "expanded": np.asarray(indices),
            "labelExpansion": [np.asarray(indices), np.asarray(indices)],
        },
    )


def test_project_to_pointcloud_labels_cloud_and_builds_progress():
    pc = np.zeros((6, 6))
    pc[0, 0] = 0.0
    pc[1, 0] = 0.1  # points 0 and 1 are the instance's (close) points
    poles = {1: _pole([0, 1], 1, 2, pc)}

    result = DEP.project_to_pointcloud(poles, pc)  # field_mapping None -> class_col 4

    assert result.shape == (6, 8)
    np.testing.assert_array_equal(result[:, :3], pc[:, :3])
    # Instance id stamped into the "unexpanded" column and the final column.
    np.testing.assert_array_equal(result[[0, 1], 3], [1, 1])
    np.testing.assert_array_equal(result[[0, 1], -1], [1, 1])
    # The original cloud is labelled: classification column -> category,
    # instance-id column -> id.
    np.testing.assert_array_equal(pc[[0, 1], 4], [2, 2])
    np.testing.assert_array_equal(pc[[0, 1], -1], [1, 1])


# ---------------------------------------------------------------------------
# combine_ids
# ---------------------------------------------------------------------------


def test_combine_ids_no_existing_id_returns_own_instance():
    pc = np.zeros((4, 6))
    pc[0, 0] = 1.0
    pc[1, 0] = 1.0
    pc[2, 0] = 50.0  # far away, not a spatial neighbour
    kd = KDTree(pc[:, :3])
    pole = U.Instance(
        instance_id=7,
        class_id=3,
        point_indices=np.array([0, 1]),
        centroid=pc[[0, 1], :3].mean(axis=0),
    )
    poles = {7: pole}

    idxs, category, instance_id = DEP.combine_ids(pole, poles, pc, kd, class_col=4)

    assert list(np.sort(idxs)) == [0, 1]
    assert category == 3
    assert instance_id == 7
    assert poles[7].instance_id == 7


def test_combine_ids_merges_into_existing_instance():
    pc = np.zeros((6, 6))
    pc[0, 0] = 0.0
    pc[1, 0] = 0.1
    pc[0, -1] = 2  # poleA's point already carries instance id 2 (poleB)
    kd = KDTree(pc[:, :3])
    pole_a = U.Instance(
        instance_id=1, class_id=2, point_indices=np.array([0]), centroid=pc[0, :3].copy()
    )
    pole_b = U.Instance(
        instance_id=2, class_id=9, point_indices=np.array([1]), centroid=pc[1, :3].copy()
    )
    poles = {1: pole_a, 2: pole_b}

    idxs, category, instance_id = DEP.combine_ids(pole_a, poles, pc, kd, class_col=4)

    assert list(np.sort(idxs)) == [0, 1]
    assert category == 9  # adopts the merged (existing) instance's class
    assert instance_id == 2  # adopts the existing instance id
    assert 1 not in poles  # the current pole is removed
    assert list(np.sort(poles[2].point_indices)) == [0, 1]
    assert poles[2].class_id == 9
