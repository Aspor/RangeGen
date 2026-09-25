"""Generate sensor poses spaced along a trajectory.

Walks a trajectory at a configurable stride distance, applying the sensor
mounting and (optionally) the trajectory orientation, and returns one pose per
sampled point as ``(poses, basenames)``.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from range_image_generator import Pose
import logging


def generate_poses_around_trajectory(cfg, trajectory):
    """Sample poses along a trajectory at a fixed travel distance.

    Steps forward along the trajectory, accumulating travel distance until the
    configured ``stride_distance`` is exceeded, then emits a pose for that point
    with the sensor mounting (and optionally the trajectory orientation) applied.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration (``sensor`` and ``processing`` sections).
    trajectory : list
        Sequence of ``(timestamp, position, orientation)`` entries.

    Returns
    -------
    tuple
        ``(poses, basenames)`` — a list of ``Pose`` objects and a matching list
        of zero-padded trajectory-index IDs.
    """
    sensor_orientation = cfg["sensor"].get("sensor_orientation", [0.0, 0.0, 0.0])
    use_trajectory_orientation = (cfg["sensor"].get("use_trajectory_orientation", True),)
    skip_start = cfg["processing"].get("skip_start", 0)
    skip_end = cfg["processing"].get("skip_end", 0)
    stride_distance = cfg["processing"].get("stride_distance", 30)

    i = skip_start
    stride = 1

    # maximum lenght of trajectory index used for zero padding
    traj_len_str = len(str(len(trajectory)))

    if sensor_orientation is not None:
        sensor_rot = R.from_euler("xyz", sensor_orientation, degrees=True)
    poses = []
    basenames = []

    while i < len(trajectory) - skip_end:
        _, position, orientation = trajectory[i]
        travel = 0

        # Move forward along the trajectory until we exceed stride_distance
        # every tenth point to smooth gps jitter
        for j, (_ts, p2, _o2) in enumerate(trajectory[i:-1:stride]):
            travel += np.linalg.norm(p2[:2] - position[:2])

            if travel > stride_distance:
                logging.debug((f"{j=}, {travel=}, {stride_distance=}"))
                i += j * stride
                _, position, orientation = _ts, p2, _o2
                logging.debug(("Position:", position))
                break
        else:
            break

        traj_index = str(i).zfill(traj_len_str)
        orientation_to_use = R.identity()

        # Apply sensor mounting
        if sensor_orientation is not None:
            orientation_to_use = sensor_rot * orientation_to_use

        # Apply trajectory orientation if desired
        if use_trajectory_orientation:
            traj_rot = R.from_euler("xyz", orientation, degrees=True)
            orientation_to_use = traj_rot * orientation_to_use

        # Generate a single frame (or randomized frames)
        orientation_to_use = orientation_to_use.as_quat()
        pose = Pose(position=position, orientation=orientation_to_use)

        basenames.append(traj_index)
        poses.append(pose)

    return poses, basenames
