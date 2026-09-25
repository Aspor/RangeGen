"""Generate random sensor poses by sampling inside the point cloud.

Selects candidate sensor positions from within the point cloud (optionally
restricted to a bounding segment and a set of classes) and pairs them with the
configured sensor orientation, returning ``(poses, basenames)``.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from range_image_generator.utils import Pose
# -------------------------------------------------------------------------
# Sampling helpers
# -------------------------------------------------------------------------


def calculate_sampling_range_average(minimum, maximum, avg, segment):
    """Shrink a per-axis bounding range to a central segment of the cloud.

    Parameters
    ----------
    minimum : float
        Minimum value of the axis across the cloud.
    maximum : float
        Maximum value of the axis across the cloud.
    avg : float
        Average offset of the points from ``minimum`` along the axis.
    segment : tuple[float, float]
        Normalized ``(start, end)`` span (0 to 1) of the average to keep.

    Returns
    -------
    tuple[float, float]
        ``(low, high)`` bounds for the sampling window on this axis.
    """
    span = avg
    cutoff_x_min = segment[0] * span
    cutoff_x_max = (1 - segment[1]) * span
    return minimum + cutoff_x_min, maximum - cutoff_x_max


def array_in_bounds(points, bounds):
    """Return a boolean mask of points inside a per-axis bounding box.

    Parameters
    ----------
    points : numpy.ndarray
        Points of shape ``(N, 3)`` or wider; only the first three columns are
        used.
    bounds : tuple
        Per-axis ``(low, high)`` bounds for ``x``, ``y``, and ``z``.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(N,)``, ``True`` where the point is in bounds.
    """
    return (
        (points[:, 0] >= bounds[0][0])
        & (points[:, 0] <= bounds[0][1])
        & (points[:, 1] >= bounds[1][0])
        & (points[:, 1] <= bounds[1][1])
        & (points[:, 2] >= bounds[2][0])
        & (points[:, 2] <= bounds[2][1])
    )


def calculate_random_sampling_points(
    points,
    x_segment,
    y_segment,
    z_segment,
    number_of_sampling_points,
    sensor_orientation=None,
    classes=None,
):
    """Sample random sensor poses inside the point cloud's central region.

    Parameters
    ----------
    points : numpy.ndarray
        Point cloud of shape ``(N, 8)`` in the ``(x, y, z, intensity, class,
        r, g, b)`` layout.
    x_segment, y_segment, z_segment : tuple[float, float]
        Normalized ``(start, end)`` span (0 to 1) of each axis to sample from.
    number_of_sampling_points : int
        How many poses to generate.
    sensor_orientation : numpy.ndarray or None, optional
        Euler orientation (degrees) applied to every pose. Default is ``None``.
    classes : array-like, optional
        If given, only points whose classification (column 4) is in this set are
        sampled from. Default is ``None``.

    Returns
    -------
    tuple
        ``(poses, base_ids)`` — generators of ``Pose`` objects and of
        zero-padded string IDs.
    """
    minimums = np.amin(points[:, :3], axis=0)
    maximums = np.amax(points[:, :3], axis=0)
    avgs = np.average(points[:, :3] - minimums, axis=0)
    sensor_orientation = (
        R.from_euler("xyz", sensor_orientation).as_quat()
        if sensor_orientation is not None
        else sensor_orientation
    )

    x_range = calculate_sampling_range_average(
        minimums[0], maximums[0], avgs[0], x_segment
    )
    y_range = calculate_sampling_range_average(
        minimums[1], maximums[1], avgs[1], y_segment
    )
    z_range = calculate_sampling_range_average(
        minimums[2], maximums[2], avgs[2], z_segment
    )

    bounds = (x_range, y_range, z_range)

    pts = points.copy()

    # if we are intrested in only some classes
    if classes is not None:
        pts = pts[np.isin(pts[:, 4], classes)]

    sampling_points = pts[array_in_bounds(pts[:, :3], bounds)]
    rng = np.random.default_rng()
    orientation = sensor_orientation if sensor_orientation is not None else [0, 0, 0]
    traj_len_str = len(str(number_of_sampling_points))
    poses = (
        Pose(position=rng.choice(sampling_points)[:3], orientation=orientation)
        for i in range(number_of_sampling_points)
    )
    base_ids = (str(i).zfill(traj_len_str) for i in range(number_of_sampling_points))
    return poses, base_ids


def random_sampling(points, cfg, classes=None):
    """Draw random sampling poses using the random-sampling config section.

    Parameters
    ----------
    points : numpy.ndarray
        Point cloud of shape ``(N, 8)``.
    cfg : dict
        Parsed pipeline configuration (``processing`` and ``sensor`` sections).
    classes : array-like, optional
        If given, restrict sampling to these classification labels. Default is
        ``None``.

    Returns
    -------
    tuple
        ``(poses, base_ids)`` from ``calculate_random_sampling_points``.
    """
    x_segment = cfg["processing"].get("x_segment", (0, 1))
    y_segment = cfg["processing"].get("y_segment", (0, 1))
    z_segment = cfg["processing"].get("z_segment", (0, 1))
    number_of_sampling_points = cfg["processing"].get("sampling_points", 25)
    sensor_orientation = cfg["sensor"].get("sensor_orientation", [0, 0, 0])

    return calculate_random_sampling_points(
        points,
        x_segment,
        y_segment,
        z_segment,
        number_of_sampling_points,
        sensor_orientation=sensor_orientation,
        classes=classes,
    )
