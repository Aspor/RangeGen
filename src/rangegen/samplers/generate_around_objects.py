from scipy.spatial.transform import Rotation as R
import numpy as np
import random
import logging

from scipy.spatial import KDTree
from range_image_generator import Pose


def generate_poses_around_objects(cfg, detection_results, trajectory=None):
    """Generate sampled sensor poses around detected objects.

    For each detected object, places the sensor along the vector to its nearest
    trajectory point (with per-sample jitter) and orients it using the trajectory
    orientation and the sensor mounting.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration (``sensor``, ``image`` and ``processing``
        sections).
    detection_results : list
        Object detections as ``(centroid, instance_id, class_id)`` tuples.
    trajectory : list or None, optional
        Trajectory providing the reference point and orientation for each
        object; may be ``None`` (trajectory alignment skipped). Default is
        ``None``.

    Returns
    -------
    tuple
        ``(poses, base_ids, "objects")`` — the generated ``Pose`` objects, their
        IDs, and the source label.
    """
    sensor_orientation = cfg["sensor"].get("sensor_orientation", [0, 0, 0])

    max_range = cfg["image"].get("max_range", 50)
    categories_of_interest = cfg["processing"].get("categories", None)

    sampling_points = cfg["processing"].get("sampling_points", 1)

    sensor_orientation = R.from_euler("xyz", sensor_orientation, degrees=True)
    orientation = sensor_orientation
    traj = None
    if trajectory:
        traj = np.zeros((len(trajectory), 3))
        for i, (_, position, orientation) in enumerate(trajectory):
            traj[i] = position
            logging.debug(("TRAJECTORY", i, position, trajectory[i]))
        logging.debug((traj.shape))
        traj_kd = KDTree(traj)
    base_ids = []
    poses = []

    for centroid, instance_id, category in detection_results:
        if categories_of_interest is not None and category not in categories_of_interest:
            continue

        pos = centroid
        if traj is not None:
            dist, closest_index = traj_kd.query(pos, p=2)
            closest_trajectory_point = traj[closest_index]
            orientation = trajectory[closest_index][2]
            orientation = R.from_euler("xyz", orientation, degrees=True)
            orientation = orientation * sensor_orientation
            diff = closest_trajectory_point - pos

        for i in range(sampling_points):
            change = diff.copy()

            for ii in range(2):
                change[ii] *= random.normalvariate(0.5, 0.5)

            change *= random.uniform(-0.25, 0.75)

            if change[-1] < 0:
                change[-1] = 0

            pos = closest_trajectory_point + change

            logging.debug(
                f"OBJECTS {closest_trajectory_point=} {change=}   {pos=}  {dist/2 > max_range/3 =}"
            )

            base_id = f"{instance_id:04}_{i}"
            pose = Pose(position=pos, orientation=orientation.as_quat())
            base_ids.append(base_id)
            poses.append(pose)

    return (poses, base_ids, "objects")
