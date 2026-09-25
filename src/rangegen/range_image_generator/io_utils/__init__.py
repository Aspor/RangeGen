"""Input/output utilities for LiDAR datasets.

This package groups the I/O helpers into focused submodules:

- ``pointcloud``     -- LAS/LAZ reading, point cloud processing, PCD writing.
- ``trajectory``     -- trajectory / NMEA / YAML reading and LAS<->trajectory
                        matching.
- ``image_export``   -- range / RGB / intensity saving and classification export.
- ``classification`` -- writing labels / progress back into LAS/LAZ files.

All public names are re-exported here so callers can simply do::

    from range_image_generator.io_utils import (
        load_pointcloud,
        write_pointcloud,
        read_trajectory_file,
        match_las_and_trajectory,
    )
"""

from .pointcloud import (
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
from .trajectory import (
    match_las_and_trajectory,
    parse_nmea_file,
    parse_pose_yaml,
    read_trajectory_file,
    save_csv,
)
from .image_export import save_classification_image, save_image
from .classification import create_progress_pountclouds, update_classification

__all__ = [
    # pointcloud
    "LasSchema",
    "PointCloudResult",
    "iter_las_chunks",
    "load_las",
    "load_las_memmap",
    "load_pointcloud",
    "process_las_in_chunks",
    "print_schema",
    "write_pointcloud",
    # trajectory
    "match_las_and_trajectory",
    "parse_nmea_file",
    "parse_pose_yaml",
    "read_trajectory_file",
    "save_csv",
    # image_export
    "save_classification_image",
    "save_image",
    # classification
    "create_progress_pountclouds",
    "update_classification",
]
