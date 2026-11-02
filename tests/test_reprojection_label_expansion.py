"""Unit tests for ``rangegen.reprojection.label_expansion``.

Covers the public, externally-called API of the label-expansion pipeline:
``load_config``, ``register_step``, the ``expand_labels`` runner, and each of
the registered pipeline steps. All are light (numpy/scipy/sklearn), so they are
exercised for real behaviour rather than mocked.
"""

import json

import numpy as np
import pytest
from scipy.spatial import KDTree

from reprojection import label_expansion as LE


# ---------------------------------------------------------------------------
# load_config / register_step
# ---------------------------------------------------------------------------


def test_load_config_returns_default_pipeline():
    cfg = LE.load_config()
    assert cfg is LE.DEFAULT_CONFIG
    assert isinstance(cfg["pipeline"], list)
    assert all("step" in stage for stage in cfg["pipeline"])


def test_load_config_reads_json_file(tmp_path):
    custom = {"pipeline": [{"step": "expand_cluster", "radii": [[40, 0.2]]}]}
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(custom))

    assert LE.load_config(str(path)) == custom


def test_register_step_registers_and_returns_function():
    name = "ut_step"

    def step(points, idxs, **kwargs):
        return idxs

    returned = LE.register_step(name)(step)

    assert returned is step
    assert LE.PIPELINE_REGISTRY[name] is step
    del LE.PIPELINE_REGISTRY[name]


def test_default_steps_are_registered():
    expected = [
        "remove_3d_outliers",
        "expand_cluster",
        "select_closest_cluster",
        "select_cluster_dbscan",
    ]
    for name in expected:
        assert name in LE.PIPELINE_REGISTRY


# ---------------------------------------------------------------------------
# expand_labels (pipeline runner)
# ---------------------------------------------------------------------------


def _seed_cloud():
    return np.array([[0.0, 0, 0], [0.1, 0, 0], [0.15, 0, 0], [10.0, 0, 0]])


def test_expand_labels_empty_mask_returns_empty():
    points = np.zeros((3, 3))
    kd = KDTree(points[:, :3])

    final, progress = LE.expand_labels(np.array([], dtype=np.int64), points, kd)

    assert final == []
    assert progress == []


def test_expand_labels_runs_configured_steps_and_records_progress():
    points = _seed_cloud()
    kd = KDTree(points[:, :3])
    cfg = {"pipeline": [{"step": "expand_cluster", "radii": [[np.inf, 0.2]]}]}

    final, progress = LE.expand_labels(np.array([0]), points, kd, config=cfg)

    assert sorted(np.asarray(final).tolist()) == [0, 1, 2]
    # progress holds the initial mask plus one entry per executed step.
    assert len(progress) == 2
    assert np.asarray(progress[0]).tolist() == [0]


def test_expand_labels_unknown_step_raises():
    points = np.zeros((2, 3))
    kd = KDTree(points[:, :3])
    cfg = {"pipeline": [{"step": "does_not_exist"}]}

    with pytest.raises(ValueError):
        LE.expand_labels(np.array([0]), points, kd, config=cfg)


# ---------------------------------------------------------------------------
# remove_3d_outliers
# ---------------------------------------------------------------------------


def test_remove_3d_outliers_drops_far_point():
    points = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.0, 0.1, 0], [10.0, 0, 0]])
    idxs = np.array([0, 1, 2, 3])

    result = LE.remove_3d_outliers(
        points, idxs, distance_threshold=0.5, number_of_neighhbors=3, max_iterations=5
    )

    # The tight cluster survives; the far outlier (index 3) is removed.
    assert sorted(result.tolist()) == [0, 1, 2]


def test_remove_3d_outliers_all_close_keeps_all():
    points = np.array([[0.0, 0, 0], [0.1, 0, 0], [0.0, 0.1, 0]])
    idxs = np.array([0, 1, 2])

    result = LE.remove_3d_outliers(
        points, idxs, distance_threshold=0.5, number_of_neighhbors=3, max_iterations=5
    )

    assert sorted(result.tolist()) == [0, 1, 2]


# ---------------------------------------------------------------------------
# select_closest_cluster
# ---------------------------------------------------------------------------


def test_select_closest_cluster_picks_nearest_cluster():
    # Two distance groups (near origin and far) separated by a large gap.
    points = np.array(
        [
            [1.0, 0, 0],
            [1.1, 0, 0],
            [1.2, 0, 0],
            [100.0, 0, 0],
            [101.0, 0, 0],
            [102.0, 0, 0],
        ]
    )
    idxs = np.arange(6)

    result = LE.select_closest_cluster(
        points, idxs, multiplier=1, min_size=2, min_dist=0.0, mode=LE.CLUSTER_CLOSEST
    )

    assert sorted(result.tolist()) == [0, 1, 2]


def test_select_closest_cluster_single_cluster_returns_all():
    points = np.array([[1.0, 0, 0], [0.0, 1.0, 0], [0.0, 0.0, 1.0]])
    idxs = np.array([0, 1, 2])

    result = LE.select_closest_cluster(
        points, idxs, multiplier=1, min_size=1, min_dist=0.0, mode=LE.CLUSTER_CLOSEST
    )

    assert sorted(result.tolist()) == [0, 1, 2]


# ---------------------------------------------------------------------------
# expand_cluster
# ---------------------------------------------------------------------------


def test_expand_cluster_radius_branch_gathers_near_points():
    points = _seed_cloud()
    kd = KDTree(points[:, :3])

    result = LE.expand_cluster(points, np.array([0]), kd_tree=kd, radii=[(np.inf, 0.2)])

    assert sorted(result.tolist()) == [0, 1, 2]


def test_expand_cluster_knn_branch():
    points = _seed_cloud()
    kd = KDTree(points[:, :3])

    result = LE.expand_cluster(points, np.array([0]), kd_tree=kd, radii=[(2, 0.5)])

    # k=2 nearest of the seed (itself + index 1).
    assert sorted(result.tolist()) == [0, 1]


# ---------------------------------------------------------------------------
# select_cluster_dbscan
# ---------------------------------------------------------------------------


def test_select_cluster_dbscan_closest():
    points = np.array([[0.1, 0, 0], [0.2, 0, 0], [10.0, 0, 0], [11.0, 0, 0]])
    idxs = np.arange(4)

    result = LE.select_cluster_dbscan(
        points, idxs, eps=1.0, min_samples=2, mode=LE.CLUSTER_CLOSEST
    )

    assert sorted(result.tolist()) == [0, 1]


def test_select_cluster_dbscan_largest():
    points = np.array(
        [[0.1, 0, 0], [0.2, 0, 0], [10.0, 0, 0], [11.0, 0, 0], [12.0, 0, 0]]
    )
    idxs = np.arange(5)

    result = LE.select_cluster_dbscan(
        points, idxs, eps=1.0, min_samples=2, mode=LE.CLUSTER_LARGEST
    )

    assert sorted(result.tolist()) == [2, 3, 4]


def test_select_cluster_dbscan_empty_indices():
    result = LE.select_cluster_dbscan(np.zeros((3, 3)), np.array([], dtype=np.int64))

    assert len(result) == 0
