"""Unit tests for ``io_utils.pointcloud`` public functions."""

import numpy as np
import pypcd4
import pytest

from range_image_generator.io_utils import (
    LasSchema,
    PointCloudResult,
    iter_las_chunks,
    load_las,
    load_las_memmap,
    load_pointcloud,
    process_las_in_chunks,
    print_schema,
    write_pointcloud,
)


# ---------------------------------------------------------------------------
# load_las
# ---------------------------------------------------------------------------


def test_load_las_selected_fields_and_xyz(las_data):
    result = load_las(las_data["path"], fields=["intensity", "classification"])

    assert result.shape[0] == las_data["n"]
    assert "intensity" in result.dtype.names
    assert "classification" in result.dtype.names
    assert "xyz" in result.dtype.names

    expected_xyz = np.column_stack(
        (las_data["x"], las_data["y"], las_data["z"])
    )
    np.testing.assert_array_equal(result["xyz"], expected_xyz)
    np.testing.assert_array_equal(result["intensity"], las_data["intensity"])
    np.testing.assert_array_equal(
        result["classification"], las_data["classification"]
    )


def test_load_las_default_fields_include_all_dimensions(las_data):
    result = load_las(las_data["path"])

    assert result.shape[0] == las_data["n"]
    assert "xyz" in result.dtype.names
    assert "intensity" in result.dtype.names
    assert "classification" in result.dtype.names


def test_load_las_exclude_xyz(las_data):
    result = load_las(
        las_data["path"],
        fields=["intensity"],
        include_xyz=False,
    )

    assert "xyz" not in result.dtype.names
    assert "intensity" in result.dtype.names


# ---------------------------------------------------------------------------
# iter_las_chunks
# ---------------------------------------------------------------------------


def test_iter_las_chunks_total_and_dtype(las_data):
    chunks = list(
        iter_las_chunks(
            las_data["path"],
            chunk_size=2,
            fields=["intensity", "classification"],
        )
    )

    assert sum(len(chunk) for chunk in chunks) == las_data["n"]
    for chunk in chunks:
        assert "intensity" in chunk.dtype.names
        assert "xyz" in chunk.dtype.names

    concatenated = np.concatenate([chunk["xyz"] for chunk in chunks])
    expected_xyz = np.column_stack(
        (las_data["x"], las_data["y"], las_data["z"])
    )
    np.testing.assert_array_equal(concatenated, expected_xyz)


# ---------------------------------------------------------------------------
# load_las_memmap
# ---------------------------------------------------------------------------


def test_load_las_memmap_values_and_dtype(las_data, tmp_path):
    output = tmp_path / "memmap.bin"
    mmap = load_las_memmap(
        las_data["path"],
        output,
        chunk_size=2,
        fields=["intensity", "classification"],
    )

    assert output.exists()
    assert mmap.shape[0] == las_data["n"]
    assert "intensity" in mmap.dtype.names
    np.testing.assert_array_equal(
        mmap["classification"], las_data["classification"]
    )


def test_load_las_memmap_no_overwrite_raises(las_data, tmp_path):
    output = tmp_path / "memmap.bin"
    load_las_memmap(
        las_data["path"],
        output,
        fields=["intensity"],
    )

    with pytest.raises(FileExistsError):
        load_las_memmap(
            las_data["path"],
            output,
            fields=["intensity"],
            overwrite=False,
        )


# ---------------------------------------------------------------------------
# process_las_in_chunks
# ---------------------------------------------------------------------------


def test_process_las_in_chunks_applies_func(las_data):
    results = process_las_in_chunks(
        las_data["path"],
        func=len,
        chunk_size=2,
    )

    assert results == [2, 2, 1]
    assert sum(results) == las_data["n"]


# ---------------------------------------------------------------------------
# print_schema
# ---------------------------------------------------------------------------


def test_print_schema_structured_logs_fields(las_data, caplog):
    arr = load_las(las_data["path"], fields=["intensity"])

    with caplog.at_level("INFO"):
        print_schema(arr)

    message = caplog.text
    assert "intensity" in message
    assert "xyz" in message


def test_print_schema_unstructured(caplog):
    with caplog.at_level("INFO"):
        print_schema(np.array([1.0, 2.0, 3.0]))

    assert "not structured" in caplog.text.lower()


# ---------------------------------------------------------------------------
# load_pointcloud
# ---------------------------------------------------------------------------


def test_load_pointcloud_layout_and_mapping(las_data):
    result = load_pointcloud(las_data["path"])

    # xyz + intensity + classification -> 5 columns.
    assert result.array.shape == (las_data["n"], 5)
    assert result.field_mapping == {"intensity": 3, "classification": 4}
    assert result.metadata is not None
    assert result.metadata.shape == (las_data["n"], 5)

    # xyz is preserved verbatim.
    expected_xyz = np.column_stack(
        (las_data["x"], las_data["y"], las_data["z"])
    )
    np.testing.assert_array_equal(result.array[:, :3], expected_xyz)
    # Classification is preserved verbatim.
    np.testing.assert_array_equal(
        result.array[:, 4], las_data["classification"]
    )


def test_load_pointcloud_intensity_is_normalized(las_data):
    result = load_pointcloud(las_data["path"])

    maximum = float(np.max(las_data["intensity"]))
    expected = 65534.0 * (las_data["intensity"] / maximum)
    np.testing.assert_allclose(result.array[:, 3], expected)


def test_load_pointcloud_reopens_from_metadata(las_data):
    result = load_pointcloud(las_data["path"])

    reopened = result.metadata.memmap(mode="r")
    np.testing.assert_array_equal(
        reopened[:, :3], result.array[:, :3]
    )


def test_load_pointcloud_custom_fields(las_data):
    result = load_pointcloud(las_data["path"], fields=["classification"])

    assert result.array.shape == (las_data["n"], 4)
    assert result.field_mapping == {"classification": 3}
    np.testing.assert_array_equal(
        result.array[:, 3], las_data["classification"]
    )


# ---------------------------------------------------------------------------
# write_pointcloud
# ---------------------------------------------------------------------------


def test_write_pointcloud_basic(tmp_path):
    points = np.array([[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]])
    output = tmp_path / "cloud.pcd"

    write_pointcloud(points, str(output))

    cloud = pypcd4.PointCloud.from_path(str(output))
    assert cloud.fields == ("x", "y", "z", "intensity")
    np.testing.assert_allclose(cloud.numpy(), points)


def test_write_pointcloud_with_transform(tmp_path):
    position = np.array([10.0, 0.0, 0.0])
    rotation = np.eye(3)
    input_points = np.array([[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]])
    output = tmp_path / "transformed.pcd"

    # Note: write_pointcloud mutates its input in place (reshape returns a
    # view), so capture the expected result before the call.
    expected = input_points.copy()
    expected[:, :3] = (rotation.T @ input_points[:, :3].T).T + position

    write_pointcloud(
        input_points, str(output), transform=(position, rotation)
    )

    cloud = pypcd4.PointCloud.from_path(str(output))
    np.testing.assert_allclose(cloud.numpy(), expected)


def test_write_pointcloud_save_labels(tmp_path):
    points = np.array([[1.0, 2.0, 3.0, 10.0, 7.0], [4.0, 5.0, 6.0, 20.0, 8.0]])
    output = tmp_path / "labeled.pcd"

    write_pointcloud(points, str(output), save_labels=True)

    cloud = pypcd4.PointCloud.from_path(str(output))
    assert "label" in cloud.fields
    np.testing.assert_allclose(cloud.numpy(), points)


def test_write_pointcloud_sets_metadata_from_image_shape(tmp_path):
    image = np.zeros((24, 32, 4))
    output = tmp_path / "image.pcd"

    write_pointcloud(image, str(output))

    cloud = pypcd4.PointCloud.from_path(str(output))
    assert cloud.metadata.height == 24
    assert cloud.metadata.width == 32


# ---------------------------------------------------------------------------
# dataclasses
# ---------------------------------------------------------------------------


def test_las_schema_dataclass():
    dtype = np.dtype([("intensity", np.uint16), ("xyz", (np.float64, 3))])
    schema = LasSchema(dtype=dtype, fields=["intensity"])

    assert schema.dtype == dtype
    assert schema.fields == ["intensity"]


def test_point_cloud_result_defaults():
    array = np.zeros((2, 3))
    result = PointCloudResult(array=array)

    assert result.array is array
    assert result.metadata is None
    assert result.field_mapping is None
