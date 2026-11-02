"""Unit tests for ``io_utils.classification`` public functions."""

import laspy
import numpy as np

from range_image_generator.io_utils import (
    create_progress_pountclouds,
    update_classification,
)


# ---------------------------------------------------------------------------
# update_classification
# ---------------------------------------------------------------------------


def test_update_classification_writes_ids_and_classification(las_data, tmp_path):
    n = las_data["n"]
    # point_cloud layout: x, y, z, intensity, classification, id
    point_cloud = np.zeros((n, 6))
    point_cloud[:, 4] = np.array([5, 1, 40, 2, 3], dtype=np.uint8)
    point_cloud[:, 5] = np.array([7, 8, 9, 10, 11], dtype=np.uint32)

    target = tmp_path / "updated.las"
    update_classification(point_cloud, las_data["path"], str(target))

    result = laspy.read(str(target))
    np.testing.assert_array_equal(
        np.asarray(result.id), [7, 8, 9, 10, 11]
    )
    # classification from column 4, with values >= 32 reset to 0.
    np.testing.assert_array_equal(
        np.asarray(result.classification), [5, 1, 0, 2, 3]
    )


def test_update_classification_keeps_xyz(las_data, tmp_path):
    n = las_data["n"]
    point_cloud = np.zeros((n, 6))
    point_cloud[:, 4] = np.zeros(n, dtype=np.uint8)
    point_cloud[:, 5] = np.ones(n, dtype=np.uint32)

    target = tmp_path / "updated_xyz.las"
    update_classification(point_cloud, las_data["path"], str(target))

    result = laspy.read(str(target))
    np.testing.assert_array_equal(
        np.asarray(result.x).astype(int), las_data["x"].astype(int)
    )
    np.testing.assert_array_equal(
        np.asarray(result.y).astype(int), las_data["y"].astype(int)
    )
    np.testing.assert_array_equal(
        np.asarray(result.z).astype(int), las_data["z"].astype(int)
    )


# ---------------------------------------------------------------------------
# create_progress_pountclouds
# ---------------------------------------------------------------------------


def test_create_progress_pountclouds_appends_expansion_dims(las_data, tmp_path):
    n = las_data["n"]
    # Columns beyond xyz (3) become i_0, i_1, i_2.
    point_cloud = np.zeros((n, 6))
    point_cloud[:, 3] = [1, 2, 3, 4, 5]
    point_cloud[:, 4] = [10, 20, 30, 40, 50]
    point_cloud[:, 5] = [100, 200, 300, 400, 500]

    target = tmp_path / "progress.las"
    create_progress_pountclouds(point_cloud, las_data["path"], str(target))

    result = laspy.read(str(target))
    names = [d.name for d in result.header.point_format.dimensions]
    assert "i_0" in names
    assert "i_1" in names
    assert "i_2" in names

    np.testing.assert_array_equal(np.asarray(result.i_0), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.asarray(result.i_1), [10, 20, 30, 40, 50])
    np.testing.assert_array_equal(
        np.asarray(result.i_2), [100, 200, 300, 400, 500]
    )
