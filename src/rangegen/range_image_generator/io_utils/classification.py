"""Utilities for writing LAS point clouds with updated classification and id data.

Reads a base LAS file, updates the per-point classification and instance
ids from a processed point cloud array, and writes the result to a new
LAS file.
"""
import laspy
import numpy as np

def update_classification(point_cloud, orginal_file, target_path):
    """Write a copy of a LAS file with updated classification and ids.

    The per-point classification is taken from the fifth column of
    ``point_cloud`` (values >= 32 are reset to 0), and an ``id`` extra
    dimension is added from the last column.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Point cloud array; column 4 holds the classification and the last
        column holds the instance id.
    orginal_file : str or pathlib.Path
        Source LAS/LAZ file to read and base the output on.
    target_path : str or pathlib.Path
        Output LAS file to write.
    """
    las = laspy.read(orginal_file)
    classification = point_cloud[:, 4].astype(np.uint8)
    classification[classification >= 32] = 0
    las.classification = classification  # point_cloud[:, 4]
    laspy.convert(las, file_version="1.4")

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="id", type=np.uint32, description="Instance id from coco"
        )
    )
    ids = point_cloud[:, -1].astype(np.uint32)
    las.id = ids
    las.write(target_path)


def create_progress_pountclouds(point_cloud, orginal_file, target_path):
    """Write a LAS file with per-point expansion columns appended.

    Copies the base LAS file and appends one ``i_<n>`` uint32 extra
    dimension for each column beyond the first three (xyz) of
    ``point_cloud``.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Point cloud array; columns beyond xyz become ``i_<n>`` dimensions.
    orginal_file : str or pathlib.Path
        Source LAS/LAZ file to read and base the output on.
    target_path : str or pathlib.Path
        Output LAS file to write.
    """
    las = laspy.read(orginal_file)
    laspy.convert(las, file_version="1.4")

    dims_to_add = []
    for i in range(point_cloud.shape[1] - 3):
        dims_to_add.append(
            laspy.ExtraBytesParams(
                name=f"i_{i}", type=np.uint32, description=f"expansions{i}"
            )
        )
    las.add_extra_dims(dims_to_add)

    for i in range(point_cloud.shape[1] - 3):
        ids = point_cloud[:, i + 3].astype(np.uint32)
        las.__setattr__(f"i_{i}", ids)

    las.write(target_path)
