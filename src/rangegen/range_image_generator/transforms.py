"""Transformation utilities for LiDAR point clouds, including pose-based
coordinate transformations and controlled perturbations to position and
orientation.

This module contains no I/O code and no projection logic — only pure math
operations on positions, orientations, and point arrays.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# -------------------------------------------------------------------------
# Direct coordinate transformation
# -------------------------------------------------------------------------


def transform_points(points, position, orientation):
    """Transform 3D points from global/world coordinates into the local sensor
    coordinate frame using a given pose.

    Parameters
    ----------
    points : ndarray (N, 3)
        3D points in world coordinates.

    position : ndarray (3,)
        Sensor position in world coordinates.

    orientation : Rotation
        scipy.spatial.transform.Rotation instance representing the
        orientation of the sensor in world coordinates.

    Returns
    -------
    ndarray (N, 3)
        Transformed points in the sensor frame.
    """
    rot_matrix = orientation.as_matrix()
    # return (rot_matrix @ (points - position).T).T

    tmp = points - position
    tmp = tmp.T
    tmp = rot_matrix @ tmp
    return tmp.T


# -------------------------------------------------------------------------
# Pose perturbations
# -------------------------------------------------------------------------


def perturb_position(position, range_x, range_y, range_z):
    """Apply a random perturbation to a 3D position.

    Perturbation is drawn independently in x/y/z from uniform ranges:
        range_x = (min_x, max_x)
        range_y = (min_y, max_y)
        range_z = (min_z, max_z)

    Parameters
    ----------
    position : ndarray (3,)
        Original position.

    range_x, range_y, range_z : tuple(float, float)
        Ranges for generating random deltas.

    Returns
    -------
    ndarray (3,)
        Perturbed position.
    """
    delta = np.random.uniform(
        (range_x[0], range_y[0], range_z[0]),
        (range_x[1], range_y[1], range_z[1]),
        3,
    )
    return position + delta


def perturb_orientation(orientation, range_x, range_y, range_z):
    """Apply a random Euler-angle perturbation to a Rotation object.

    Perturbation angles are drawn independently for x/y/z axes over the
    specified uniform ranges, in degrees.

    Parameters
    ----------
    orientation : Rotation
        Base orientation.

    range_x, range_y, range_z : tuple(float, float)
        Ranges of rotational perturbation around each axis in degrees.

    Returns
    -------
    Rotation
        A new Rotation instance representing:
            orientation' = R(perturbation) * orientation
    """
    euler_delta = np.random.uniform(
        (range_x[0], range_y[0], range_z[0]),
        (range_x[1], range_y[1], range_z[1]),
        3,
    )
    orientation = R(orientation)
    delta_rot = R.from_euler("xyz", euler_delta, degrees=True)
    return (delta_rot * orientation).as_quat()


def crop_points_by_distance(points, position, range):
    """Crop points to a square region centred on the sensor position.

    A square crop centred at ``position`` with sides of ``2 * range`` is
    used rather than a true distance (norm) crop, which is faster and
    more memory efficient.

    Parameters
    ----------
    points : ndarray (N, >= 3)
        Point cloud array.
    position : ndarray (3,)
        Sensor position in world coordinates; the crop is centred here.
    range : float
        Half the side length of the square crop window.

    Returns
    -------
    tuple of (ndarray (M, 3), ndarray (M, 1))
        * Cropped points (first three columns) within the crop window.
        * Original row indices (uint32) of the cropped points.
    """
    mask = (
        (points[:, 0] > position[0] - range)
        & (points[:, 0] < range + position[0])
        & (points[:, 1] > position[1] - range)
        & (points[:, 1] < range + position[1])
    )
    # mask = np.array(np.linalg.norm(position - points[:, :3], axis=1) < (range), dtype=np.bool_ )
    original_indices = np.argwhere(mask).astype(np.uint32)

    return points[mask, :3], original_indices


def batch_crop_points_by_distance(points, positions, range):
    """Crop points to a rectangular region covering a set of positions.

    A rectangular crop is used whose bounds enclose all of ``positions``
    expanded by ``range`` in x and y, i.e. sides equal to ``2 * range``
    plus the x/y spread between the farthest positions.

    Parameters
    ----------
    points : ndarray (N, >= 3)
        Point cloud array.
    positions : ndarray (P, 3)
        Sensor positions to be covered by the crop.
    range : float
        Margin added on each side of the min/max position bounds.

    Returns
    -------
    tuple of (ndarray (M, 3), ndarray (M, 1))
        * Cropped points (first three columns) within the crop window.
        * Original row indices (uint32) of the cropped points.
    """
    min_x, min_y, _ = np.min(positions, axis=0)
    max_x, max_y, _ = np.max(positions, axis=0)

    mask = (
        (points[:, 0] > min_x - range)
        & (points[:, 0] < range + max_x)
        & (points[:, 1] > min_y - range)
        & (points[:, 1] < range + max_y)
    )
    original_indices = np.argwhere(mask).astype(np.uint32)
    return points[mask, :3], original_indices
