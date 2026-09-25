"""Utilities for reading, writing and processing LAS/LAZ point clouds.

Supports loading point clouds fully into memory, streaming them
chunk-by-chunk, converting them into memory-mapped structured arrays,
and writing them back out as PCD files.
"""

from __future__ import annotations

import logging
import os
import tempfile

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import laspy
import numpy as np
from pypcd4 import PointCloud

from ..utils import MemmapMetadata, points_dtype

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================


@dataclass(slots=True)
class LasSchema:
    """Description of the target structured-array schema for a LAS file.

    Attributes
    ----------
    dtype : numpy.dtype
        Structured dtype that converted chunks are built with.
    fields : list of str
        Names of the LAS dimensions to include.
    """

    dtype: np.dtype
    fields: list[str]


@dataclass(slots=True)
class PointCloudResult:
    """Result of loading a point cloud into the project processing format.

    Attributes
    ----------
    array : numpy.ndarray
        Point cloud array (``xyz`` followed by any requested fields).
    metadata : MemmapMetadata | None
        Metadata describing a memory-mapped backing file, if any.
    field_mapping : dict[str, int] | None
        Mapping of field name to its column index in ``array``.
    """

    array: np.ndarray
    metadata: MemmapMetadata | None = None
    field_mapping: dict[str, int] | None = None


# ============================================================================
# Internal helpers
# ============================================================================


def _create_schema(
    point_format,
    fields: list[str] | None = None,
    include_xyz: bool = True,
) -> LasSchema:
    """Create a NumPy structured schema from a LAS point format.

    Parameters
    ----------
    point_format : laspy PointFormat
        LAS point format whose dimensions define the available fields.
    fields : list of str, optional
        Dimensions to include. If ``None``, all available dimensions are
        used.
    include_xyz : bool, optional
        Whether to append a ``xyz`` (float64, 3) column.

    Returns
    -------
    LasSchema
        Schema with the resolved structured ``dtype`` and field list.
    """

    available = {d.name: d for d in point_format.dimensions}

    selected_fields = (
        list(available.keys())
        if fields is None
        else list(fields)
    )

    dtype_fields: list[tuple] = []

    for field in selected_fields:
        dimension = available.get(field)

        if dimension is None:
            logger.warning("Field '%s' does not exist", field)
            dtype_fields.append((field, np.float64))
            continue

        dtype_fields.append((field, dimension.dtype))

    if include_xyz:
        dtype_fields.append(("xyz", (np.float64, 3)))

    return LasSchema(
        dtype=np.dtype(dtype_fields),
        fields=selected_fields,
    )


def _copy_fields(
    destination: np.ndarray,
    points,
    fields: list[str],
    include_xyz: bool,
) -> None:
    """Copy LAS dimensions into a structured array.

    Parameters
    ----------
    destination : numpy.ndarray
        Structured array that receives the copied fields.
    points : laspy PointCloud
        Point cloud whose dimensions are read.
    fields : list of str
        Field names to copy (only those present in ``destination``).
    include_xyz : bool
        Whether to also populate the ``xyz`` column from x/y/z.
    """

    for field in fields:
        if field not in destination.dtype.names:
            continue

        try:
            destination[field] = points[field]
        except Exception as exc:
            logger.debug(
                "Failed to read field '%s': %s",
                field,
                exc,
            )

    if include_xyz:
        destination["xyz"] = np.column_stack(
            (points.x, points.y, points.z)
        )


def _convert_chunk(
    points,
    schema: LasSchema,
    include_xyz: bool,
) -> np.recarray:
    """Convert a LAS chunk into a NumPy recarray.

    Parameters
    ----------
    points : laspy PointCloud
        Point cloud chunk to convert.
    schema : LasSchema
        Target structured schema.
    include_xyz : bool
        Whether to also populate the ``xyz`` column.

    Returns
    -------
    numpy.recarray
        Structured array view of the converted chunk.
    """

    result = np.empty(len(points), dtype=schema.dtype)

    _copy_fields(
        destination=result,
        points=points,
        fields=schema.fields,
        include_xyz=include_xyz,
    )

    return result.view(np.recarray)


# ============================================================================
# LAS Reading
# ============================================================================


def load_las(
    file_path: str | Path,
    fields: list[str] | None = None,
    include_xyz: bool = True,
) -> np.recarray:
    """Load a complete LAS/LAZ file into memory.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the LAS/LAZ file to read.
    fields : list of str, optional
        Dimensions to include. If ``None``, all available dimensions are
        used.
    include_xyz : bool, optional
        Whether to append a ``xyz`` (float64, 3) column.

    Returns
    -------
    numpy.recarray
        All points as a structured array.
    """

    las = laspy.read(file_path)

    schema = _create_schema(
        las.header.point_format,
        fields=fields,
        include_xyz=include_xyz,
    )

    return _convert_chunk(
        las,
        schema=schema,
        include_xyz=include_xyz,
    )


def iter_las_chunks(
    file_path: str | Path,
    chunk_size: int = 1_000_000,
    fields: list[str] | None = None,
    include_xyz: bool = True,
) -> Iterator[np.recarray]:
    """Stream a LAS/LAZ file chunk-by-chunk.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the LAS/LAZ file to read.
    chunk_size : int, optional
        Number of points per chunk.
    fields : list of str, optional
        Dimensions to include. If ``None``, all available dimensions are
        used.
    include_xyz : bool, optional
        Whether to append a ``xyz`` (float64, 3) column.

    Yields
    ------
    numpy.recarray
        The next chunk of points as a structured array.
    """

    with laspy.open(file_path) as reader:

        schema = _create_schema(
            reader.header.point_format,
            fields=fields,
            include_xyz=include_xyz,
        )

        for chunk in reader.chunk_iterator(chunk_size):
            yield _convert_chunk(
                chunk,
                schema=schema,
                include_xyz=include_xyz,
            )


def load_las_memmap(
    file_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 1_000_000,
    fields: list[str] | None = None,
    include_xyz: bool = True,
    overwrite: bool = True,
) -> np.memmap:
    """Convert a LAS/LAZ file into a memory-mapped structured array.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the source LAS/LAZ file.
    output_path : str or pathlib.Path
        Path of the memory-mapped file to write.
    chunk_size : int, optional
        Number of points per read chunk.
    fields : list of str, optional
        Dimensions to include. If ``None``, all available dimensions are
        used.
    include_xyz : bool, optional
        Whether to append a ``xyz`` (float64, 3) column.
    overwrite : bool, optional
        If ``False``, raise ``FileExistsError`` when ``output_path``
        already exists.

    Returns
    -------
    numpy.memmap
        Memory-mapped structured array backed by ``output_path``.
    """

    output_path = str(output_path)

    with laspy.open(file_path) as reader:

        schema = _create_schema(
            reader.header.point_format,
            fields=fields,
            include_xyz=include_xyz,
        )

        point_count = reader.header.point_count

        if os.path.exists(output_path):
            if not overwrite:
                raise FileExistsError(output_path)

            os.remove(output_path)

        mmap = np.memmap(
            output_path,
            dtype=schema.dtype,
            mode="w+",
            shape=(point_count,),
        )

        offset = 0

        for chunk in reader.chunk_iterator(chunk_size):

            converted = _convert_chunk(
                chunk,
                schema=schema,
                include_xyz=include_xyz,
            )

            count = len(converted)

            mmap[offset : offset + count] = converted
            offset += count

        mmap.flush()

    return mmap


# ============================================================================
# Processing
# ============================================================================


def process_las_in_chunks(
    file_path: str | Path,
    func: Callable,
    chunk_size: int = 1_000_000,
    fields: list[str] | None = None,
):
    """Map an operation over the chunks of a LAS/LAZ file.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the LAS/LAZ file to process.
    func : callable
        Function applied to each chunk.
    chunk_size : int, optional
        Number of points per chunk.
    fields : list of str, optional
        Dimensions to include. If ``None``, all available dimensions are
        used.

    Returns
    -------
    list
        The result of applying ``func`` to each chunk, in order.
    """

    return [
        func(chunk)
        for chunk in iter_las_chunks(
            file_path=file_path,
            fields=fields,
            chunk_size=chunk_size,
        )
    ]


def print_schema(arr: np.ndarray) -> None:
    """Log the structured dtype schema of an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Structured array whose dtype fields are logged.
    """

    if arr.dtype.names is None:
        logger.info("Array is not structured.")
        return

    for name in arr.dtype.names:
        logger.info("%s: %s", name, arr.dtype[name])

# ============================================================================
# Point Cloud Processing
# ============================================================================


def _normalize_intensity(
    values: np.ndarray,
) -> np.ndarray:
    """Scale intensity values into the 0-65534 range.

    Values are scaled linearly so the maximum maps to 65534. If the
    array is empty or its maximum is not positive, it is returned
    unchanged.

    Parameters
    ----------
    values : numpy.ndarray
        Intensity values to scale.

    Returns
    -------
    numpy.ndarray
        Scaled intensity values.
    """

    if len(values) == 0:
        return values

    maximum = np.max(values)

    if maximum <= 0:
        return values

    return 65534 * (values / maximum)


def load_pointcloud(
    file_path: str,
    fields: list[str] | None = None,
) -> PointCloudResult:
    """Load a point cloud into the project processing format.

    The cloud is read from a LAS file and written to a temporary
    memory-mapped file: an ``xyz`` column followed by the requested
    scalar fields (intensity normalised to 0-65534, classification, and
    id when present).

    Parameters
    ----------
    file_path : str
        Path to the LAS/LAZ file to read.
    fields : list of str, optional
        Scalar fields to include. Defaults to ``["intensity",
        "classification"]``.

    Returns
    -------
    PointCloudResult
        The point cloud array, its memmap metadata, and the field-to-column
        mapping.
    """

    fields = fields or [
        "intensity",
        "classification",
    ]

    point_cloud = load_las(
        file_path,
        fields=fields,
    )

    column_count = 3 + len(fields)

    _, mmap_path = tempfile.mkstemp()

    mmap_array = np.memmap(
        mmap_path,
        dtype=points_dtype,
        mode="w+",
        shape=(point_cloud.shape[0], column_count),
    )

    mmap_array[:, :3] = point_cloud["xyz"]

    field_mapping: dict[str, int] = {}

    column = 3

    if "intensity" in point_cloud.dtype.names:
        mmap_array[:, column] = _normalize_intensity(
            point_cloud["intensity"]
        )

        field_mapping["intensity"] = column
        column += 1

    if "classification" in point_cloud.dtype.names:
        mmap_array[:, column] = point_cloud["classification"]

        field_mapping["classification"] = column
        column += 1

    if "id" in point_cloud.dtype.names:
        mmap_array[:, column] = point_cloud["id"]

        field_mapping["id"] = column

    metadata = MemmapMetadata(
        filepath=mmap_path,
        dtype=mmap_array.dtype,
        shape=mmap_array.shape,
    )

    return PointCloudResult(
        array=mmap_array,
        metadata=metadata,
        field_mapping=field_mapping,
    )


# ============================================================================
# Point Cloud Writing
# ============================================================================


def write_pointcloud(
    points: np.ndarray,
    file_path: str,
    transform=None,
    save_labels: bool = False,
) -> None:
    """Save a point cloud as a PCD file.

    Parameters
    ----------
    points : numpy.ndarray
        Point cloud data, shaped ``(*, 4)`` for x/y/z/intensity or
        ``(*, 5)`` when ``save_labels`` is set.
    file_path : str
        Output PCD file path.
    transform : tuple (position, rotation), optional
        ``(position, rotation)`` pair applied to the xyz before saving.
    save_labels : bool, optional
        Whether the cloud carries a label column (saved as ``xyzil``).
    """

    original_shape = points.shape

    points = points.reshape(
        (-1, 5) if save_labels else (-1, 4)
    )

    if transform is not None:
        position, rotation = transform

        points[:, :3] = (
            rotation.T @ points[:, :3].T
        ).T + position

    cloud = (
        PointCloud.from_xyzil_points(points)
        if save_labels
        else PointCloud.from_xyzi_points(points)
    )

    if len(original_shape) >= 2:
        cloud.metadata.height = original_shape[0]
        cloud.metadata.width = original_shape[1]

    cloud.save(file_path)
