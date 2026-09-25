"""Generate range image poses around poles detected along a trajectory.

This module provides the pipeline for detecting poles in a point cloud
along a vehicle trajectory and deriving camera poses positioned around
each detected pole. The main entry point is
:func:`generate_poses_around_poles`, which:

* Runs pole detection over the trajectory
  (see :func:`run_pole_detection`) to find pole locations.
* Filters out poles that are too close together to avoid redundant
  viewpoints.
* Builds a :class:`range_image_generator.Pose`
  (position + sensor orientation) for each accepted pole.

Pole detection itself is parallelised with a ``multiprocessing.Pool``:
the trajectory is split into local area crops, and each crop is
processed by :func:`_worker`, which crops points by distance, removes
the ground plane, voxelises the remainder, and finally detects the
pole candidates.

The detected pole points are also re-labelled in the original point
cloud with a category ID so downstream consumers can identify them.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation as R
from multiprocessing import Pool, TimeoutError

import time
import logging


# --- Pole detection imports ---
from poledetection import detect_poles, remove_ground
from poledetection.utils import voxelize_pointcloud

from range_image_generator import Pose

from range_image_generator.transforms import crop_points_by_distance


@dataclass(frozen=True, slots=True)
class PoleDetectionTask:
    """Immutable description of one pole-detection task for a local area.

    Each task captures the point cloud crop parameters and the
    detection hyper-parameters that the worker process (see
    :func:`_worker`) needs to find pole candidates around a single
    trajectory position.

    Attributes
    ----------
    points : Any
        Full point cloud array (shared between tasks).
    position : Any
        Trajectory position around which to crop points.
    range : float
        Cropping distance around ``position``.
    voxel_size : float
        Voxelisation cell size in metres.
    cell_size : float
        Ground-removal cell size.
    z_percentile : float
        Percentile used to estimate the ground plane.
    clearance : float
        Clearance above the ground plane (m).
    fill_holes : bool
        Whether to fill holes in the ground mask.
    eps_xy : float
        DBSCAN-like neighbourhood radius in the XY plane.
    min_samples : float
        Minimum samples for pole candidates.
    r_max : float
        Maximum pole radius (m).
    h_min : float
        Minimum pole height (m).
    verticality_min : float
        Minimum verticality score for a pole.
    index : int
        Index of the trajectory entry for this task.
    """
    points: Any
    position: Any
    range: float
    voxel_size: float
    cell_size: float
    z_percentile: float
    clearance: float
    fill_holes: bool
    eps_xy: float
    min_samples: float
    r_max: float
    h_min: float
    verticality_min: float
    index: int


# =======================================================================
# Pole detection
# =======================================================================
def generate_poses_around_poles(cfg, trajectory, points, category=4):
    """Generate frames around detected pole positions.

    ####Deprecated use generate_around_objects instead ####
    Runs pole detection along the trajectory, then builds a camera
    :class:`~range_image_generator.Pose` at each
    detected pole, skipping any pole closer than ``min_dist`` metres to
    the previously accepted one to avoid redundant viewpoints.

    Parameters
    ----------
    cfg : dict
        Configuration dict. Expects at least a ``"processing"``
        section (with optional ``skip_start`` / ``skip_end``), a
        ``"sensor"`` section (with optional ``sensor_orientation``
        Euler angles in degrees) and a ``"pole_detection"`` section.
    trajectory : iterable of (timestamp, position, orientation)
        Entries describing the vehicle trajectory.
    points : numpy.ndarray
        Point cloud array; mutated in place so that the points
        belonging to detected poles are re-labelled with ``category``.
    category : int, optional
        Category ID used to label the detected pole points.

    Returns
    -------
    tuple
        Tuple of ``(poses, base_ids, label, categories)`` where
        ``poses`` is the list of generated poses, ``base_ids`` a list of
        unique identifiers per pole (``"<trajectory_index>_<pole_id>"``),
        ``label`` is the fixed string ``"poles"``, and ``categories``
        repeats ``category`` once per pose.
    """
    skip_start = cfg["processing"].get("skip_start", 0)
    skip_end = cfg["processing"].get("skip_end", 0)
    min_dist = 5
    traj_len_str = len(str(len(trajectory)))
    sensor_orientation = cfg["sensor"].get("sensor_orientation", [0, 0, 0])

    sensor_orientation = R.from_euler("xyz", sensor_orientation).as_quat()
    found_poles = run_pole_detection(
        trajectory,
        skip_start,
        skip_end,
        points,
        traj_len_str,
        cfg["pole_detection"],
        visualize=False,
        category=category,
    )

    prev_pos = np.array([0, 0, 0])
    base_ids = []
    poses = []

    for pole in found_poles:
        pos = pole["position"]
        dist = np.linalg.norm(pos - prev_pos)

        if dist < min_dist:
            continue

        prev_pos = pos
        base_id = f"{pole['trajectory_index']}_{pole['pole_id']}"
        pose = Pose(position=pos, orientation=sensor_orientation)
        base_ids.append(base_id)
        poses.append(pose)

    categories = [category] * len(poses)
    return (poses, base_ids, "poles", categories)


def run_pole_detection(
    trajectory,
    skip_start,
    skip_end,
    points,
    traj_str_len,
    cfg,
    visualize=False,
    category=4,
):
    """Run pole detection along a vehicle trajectory.

    The trajectory is sampled into local area crops spaced roughly
    ``stride_distance`` metres apart, and each crop is processed in
    parallel by :func:`_worker` via a ``multiprocessing.Pool``. Worker
    results are collected as they complete, and the detected pole points
    are re-labelled in ``points`` with the given ``category``.

    Parameters
    ----------
    trajectory : iterable of (timestamp, position, orientation)
        Entries describing the vehicle trajectory.
    skip_start : int
        Number of leading trajectory entries to skip.
    skip_end : int
        Number of trailing trajectory entries to skip.
    points : numpy.ndarray
        Point cloud array; mutated in place so that the points
        belonging to detected poles are re-labelled with ``category``.
    traj_str_len : int
        Length used to zero-pad trajectory indices in the
        returned pole dictionaries.
    cfg : dict
        Pole-detection configuration dict holding the individual
        hyper-parameters.
    visualize : bool, optional
        Unused flag reserved for optional visualisation.
    category : int, optional
        Category ID used to label the detected pole points.

    Returns
    -------
    list of dict
        List of detected pole dictionaries, each annotated with a
        zero-padded ``trajectory_index`` and a ``category``.
    """
    found_poles = []

    stride_distance = cfg.get("stride_distance", 25)
    cell_size = cfg.get("cell_size", 5.5)
    z_percentile = cfg.get("z_percentile", 50)
    clearance = cfg.get("clearance_m", 0.2)
    fill_holes = cfg.get("fill_holes", False)
    eps_xy = cfg.get("eps_xy", 1.0)
    min_samples = cfg.get("min_samples", 25)
    r_max = cfg.get("r_max_m", 0.75)
    h_min = cfg.get("h_min_m", 0.7)
    vertical_min = cfg.get("verticality_min", 0.85)
    local_dist = cfg.get("local_area_distance", 55)
    number_of_processes = cfg.get("number_of_processes", 4)

    i = skip_start
    stride = 1
    tasks = []

    while i < len(trajectory) - skip_end:
        _, position, orientation = trajectory[i]
        travel = 0

        for j, (_ts, p2, _o2) in enumerate(trajectory[i:-1:stride]):
            travel += np.linalg.norm(p2[:2] - position[:2])

            if travel > stride_distance:
                logging.debug((f"{j=}, {travel=}, {stride_distance=}"))
                i += j * stride
                _, position, orientation = _ts, p2, _o2
                logging.debug(("Position:", position, f"{i=}"))
                break
        else:
            break
        task = PoleDetectionTask(
            points=points,
            position=position,
            range=local_dist,
            voxel_size=0.06,
            cell_size=cell_size,
            z_percentile=z_percentile,
            clearance=clearance,
            fill_holes=fill_holes,
            eps_xy=eps_xy,
            min_samples=min_samples,
            r_max=r_max,
            h_min=h_min,
            verticality_min=vertical_min,
            index=i,
        )
        tasks.append(task)

    pole_indexes = np.empty(0, dtype=np.uint32)
    found_poles = []

    prcosses_running = 0
    async_results = []
    with Pool(number_of_processes) as pool:
        for task in tasks:
            while prcosses_running > number_of_processes + 2:
                for res in async_results:
                    try:
                        idx, poles = res.get(0.02)
                        prcosses_running -= 1
                        pole_indexes = np.append(pole_indexes, idx)
                        found_poles += poles

                        async_results.remove(res)
                    except TimeoutError:
                        time.sleep(0.02)
                        pass

            async_results.append(pool.apply_async(_worker, (task,)))
            prcosses_running += 1

        pool.close()
        pool.join()

    for res in async_results:
        idx, poles = res.get()
        prcosses_running -= 1
        pole_indexes = np.append(pole_indexes, idx)
        found_poles += poles

    logging.debug((pole_indexes))

    # Add classification information
    points[pole_indexes, 4] = category

    for pole in found_poles:
        pole["trajectory_index"] = str(pole["trajectory_index"]).zfill(traj_str_len)
        pole["category"] = category
    return found_poles


def _worker(task: PoleDetectionTask) -> tuple[np.ndarray, list[dict]]:
    """Process one :class:`PoleDetectionTask` in a worker process.

    Pipeline:

    1. Crop the point cloud to points within ``task.range`` of
       ``task.position``.
    2. Voxelise the local points into sparse voxels.
    3. Remove the ground plane from the voxel centroids.
    4. Detect pole candidates in the remaining points.
    5. Remap the surviving pole voxel indices back into the original
       point cloud.

    Parameters
    ----------
    task : PoleDetectionTask
        The pole detection task describing the local crop and the
        detection hyper-parameters.

    Returns
    -------
    tuple of (numpy.ndarray, list of dict)
        * The indices (in the *original* point cloud) of the points
          that belong to detected poles.
        * The list of detected pole dictionaries, each annotated with
          ``trajectory_index`` and a zero-padded ``pole_id``.
    """

    local_pts, orginal_indexes = crop_points_by_distance(
        task.points, task.position, range=task.range
    )

    voxels = voxelize_pointcloud(local_pts, 0.06)
    local_xyz = voxels["centroids"]

    filtered_xyz, mask_kept, ground_z = remove_ground(
        local_xyz,
        cell_size=task.cell_size,
        z_percentile=task.z_percentile,
        clearance_m=task.clearance,
        fill_holes=task.fill_holes,
    )

    poles, labels = detect_poles(
        filtered_xyz,
        eps_xy=task.eps_xy,
        min_samples=task.min_samples,
        r_max_m=task.r_max,
        h_min_m=task.h_min,
        verticality_min=task.verticality_min,
        ground_z=ground_z,
    )

    pole_id = 0
    pole_id_strlen = len(str((len(pole_id) + 1)))
    # Index for naming
    for p in poles:
        p["trajectory_index"] = task.index
        p["pole_id"] = f"{pole_id:0{pole_id_strlen}}"
        pole_id += 1

    # remap from ground_removed->voxels->cropped->orginal
    kept_indices = np.flatnonzero(mask_kept)
    label_indices = kept_indices[labels > -1]

    mask = np.zeros_like(mask_kept, dtype=bool)
    mask[label_indices] = True

    return orginal_indexes[mask[voxels["point_to_voxel"]], 0], poles
