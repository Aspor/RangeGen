import sys
import shutil
import datetime
import os
import tomllib
import logging
from enum import Flag, auto
from multiprocessing import Pool
from pycocotools.coco import COCO

from range_image_generator import (
    read_trajectory_file,
    match_las_and_trajectory,
    FrameIntrinsics,
    Frame_Task,
    frame_randomizer,
    write_coco_json,
)
from range_image_generator.io_utils import (
    load_pointcloud,
)

from samplers.generate_from_random_points import random_sampling
from samplers.generate_from_trajectory import generate_poses_around_trajectory

from reprojection.deprojection import label_pointcloud_coco
from samplers.generate_around_objects import generate_poses_around_objects


logger = logging.getLogger(__name__)


class pipelineenum(Flag):
    random = auto()
    trajectory = auto()
    detected = auto()
    poles = auto()
    coco = auto()


def label_pointcloud(cfg, points, pipeline, trajectory=None):
    """Detect objects in the point cloud for the enabled pipeline stages.

    Based on the flags set in ``pipeline``, runs pole detection (using the
    trajectory) or reads existing COCO annotations, collecting object
    descriptions to drive downstream pose sampling.

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.
    points : MemmapMetadata
        Loaded point cloud.
    pipeline : pipelineenum
        Flags selecting which detection stages to run.
    trajectory : list or None
        Trajectory consumed by the poles stage; may be ``None``.

    Returns
    -------
    list
        Detection results. The poles stage appends a zipped sequence and the
        COCO stage appends one ``(centroid, instance_id, class_id)`` tuple per
        detected instance.
    """

    ret = []

    if pipelineenum.poles in pipeline:
        from samplers.generate_from_trajectory_with_pole_detection import (
            generate_poses_around_poles,
        )
        result = generate_poses_around_poles(cfg, trajectory, points.memmap("r+"))
        ret.append(zip((result[0], result[1], result[3])))

    if pipelineenum.coco in pipeline:
        detection_results, categories = label_pointcloud_coco(
            points, coco_file=cfg["paths"]["coco_file"], cfg=cfg
        )
        for key, instance in detection_results.items():
            result = (instance.centroid, instance.instance_id, instance.class_id)
            ret.append(result)
    return ret


def calculate_sampling_points(cfg, points, trajectory, pipeline, detection_results=None):
    """Compute candidate sampling poses for the enabled pipeline stages.

    Each enabled stage contributes a ``(poses, basenames, label)`` entry, where
    ``label`` identifies the sampling source (``"random"``, ``"trajectory"``,
    or ``"objects"``).

    Parameters
    ----------
    cfg : dict
        Parsed pipeline configuration.
    points : MemmapMetadata
        Loaded point cloud.
    trajectory : list or None
        Trajectory for the trajectory/poles stages; may be ``None``.
    pipeline : pipelineenum
        Flags selecting which sampling stages to run.
    detection_results : list, optional
        Object detections consumed by the ``detected`` stage.

    Returns
    -------
    list of tuple
        ``[(sampling_points, basenames, label), ...]`` for every enabled stage.
    """
    # match pipeline:
    ret = []
    if pipelineenum.random in pipeline:
        sampling_points, basenames = random_sampling(points=points, cfg=cfg)
        ret.append((sampling_points, basenames, "random"))
    if pipelineenum.trajectory in pipeline:
        sampling_points, basenames = generate_poses_around_trajectory(cfg, trajectory)
        ret.append((sampling_points, basenames, "trajectory"))
    if pipelineenum.detected in pipeline:
        ret.append(generate_poses_around_objects(cfg, detection_results, trajectory))
    return ret


def generate_tasks(poses, names, config, memmap_meta, output_dir, batched=False):
    """Build the per-frame ``Frame_Task`` list from sampled poses and config.

    Reads the processing and image settings (perturbation ranges and counts,
    frame intrinsics) from ``config`` and wraps each ``(pose, name)`` pair into
    a ``Frame_Task``
    ready to be handed to ``frame_randomizer``.

    Parameters
    ----------
    poses : iterable
        Sampled poses to generate frames for.
    names : iterable
        Basenames aligned with ``poses``.
    config : dict
        Parsed pipeline configuration.
    memmap_meta : MemmapMetadata
        Loaded point cloud shared by all tasks.
    output_dir : str
        Directory the generated frames should be written to.
    batched : bool, optional
        When ``True``, group poses into tasks of ``batch_size``; otherwise one
        task per pose. Default is ``False``.

    Returns
    -------
    list of Frame_Task
        One task per pose (or per batch when ``batched`` is ``True``).
    """

    pos_perturbations = config["processing"].get("pos_perturbations", 0)
    orientation_perturbations = config["processing"].get("orientation_perturbations", 0)
    rand_range_x = config["processing"].get("rand_range_x", (0, 0))
    rand_range_y = config["processing"].get("rand_range_y", (0, 0))
    rand_range_z = config["processing"].get("rand_range_z", (0, 0))
    rand_range_roll = config["processing"].get("rand_range_roll", (0, 0))
    rand_range_pitch = config["processing"].get("rand_range_pitch", (0, 0))
    rand_range_yaw = config["processing"].get("rand_range_yaw", (0, 0))

    width = config["image"].get("width", 128)
    height = config["image"].get("height", 64)
    max_range = config["image"].get("max_range", 65)
    elevation_range = tuple(config["image"].get("elevation_range", [-45, 45]))

    intrinsics = FrameIntrinsics(
        fov_down=elevation_range[0],
        fov_up=elevation_range[1],
        height=height,
        width=width,
        max_range=max_range,
    )

    tasks = []
    if batched:
        task_poses = []
        base_ids = []
        batch_size = config["processing"].get("batch_size", (0, 0))
    for pose, base_id in zip(poses, names):
        if batched:
            task_poses.append(poses)
            base_id.append(base_ids)
            if len(task_poses) < batch_size:
                continue
            tasks.append(
                Frame_Task(
                    basename=base_ids,
                    pose=poses,
                    frame_intrinsics=intrinsics,
                    points=memmap_meta,
                    output_dir=output_dir,
                    # randomize=randomize,
                    position_perturbation=(rand_range_x, rand_range_y, rand_range_z),
                    orientation_perturbation=(
                        rand_range_roll,
                        rand_range_pitch,
                        rand_range_yaw,
                    ),
                    number_of_pos_perturbations=pos_perturbations,
                    number_of_orientation_perturbations=orientation_perturbations,
                )
            )
            task_poses = []
            base_ids = []
            continue

        tasks.append(
            Frame_Task(
                basename=base_id,
                pose=pose,
                frame_intrinsics=intrinsics,
                points=memmap_meta,
                output_dir=output_dir,
                # randomize=randomize,
                position_perturbation=(rand_range_x, rand_range_y, rand_range_z),
                orientation_perturbation=(
                    rand_range_roll,
                    rand_range_pitch,
                    rand_range_yaw,
                ),
                number_of_pos_perturbations=pos_perturbations,
                number_of_orientation_perturbations=orientation_perturbations,
            )
        )

    # final batch if samples are not divided equally
    if batched and len(base_ids) > 0:
        tasks.append(
            Frame_Task(
                basename=base_ids,
                pose=poses,
                frame_intrinsics=intrinsics,
                points=memmap_meta,
                output_dir=output_dir,
                # randomize=randomize,
                position_perturbation=(rand_range_x, rand_range_y, rand_range_z),
                orientation_perturbation=(
                    rand_range_roll,
                    rand_range_pitch,
                    rand_range_yaw,
                ),
                number_of_pos_perturbations=pos_perturbations,
                number_of_orientation_perturbations=orientation_perturbations,
            )
        )

    return tasks


def main():

    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    logging.debug(("Using config:", config_file))

    with open(config_file, "rb") as config_fh:
        config = tomllib.load(config_fh)

    output_dir = config["paths"].get(
        "output_dir",
        f"output_images_TEST_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    logging.debug(("Output directory:", output_dir))

    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(config_file, os.path.join(output_dir, "config.toml"))

    pipelinestr = config["processing"].get(
        "pipeline",
        "random",
    )
    pipeline = None  # 0# pipelineenum.random
    for pipe in pipelinestr.split(","):
        pipe = pipe.strip()
        pipeline = (
            pipelineenum[pipe] if pipeline is None else pipeline | pipelineenum[pipe]
        )
        logging.debug((pipeline, pipelineenum[pipe]))

    laz_dir = config["paths"]["las_path"]
    traj_dir = config["paths"].get("trajectory_path", None)
    if traj_dir is None:
        logger.info("Trajectory not provided, defaulting to random sampling")
        pipeline = pipelineenum.random

    pairs = match_las_and_trajectory(las_dir=laz_dir, traj_dir=traj_dir)

    batched = config["processing"].get("batched", False)
    images = {}
    annotations = {}

    #Loop trough all the files in the directory
    for las_file_, traj_file_ in pairs:
        if not os.path.isdir(laz_dir):
            las_file = laz_dir  # os.path.join(laz_dir, las_file_)
            traj_file = traj_dir  # os.path.join(traj_dir, traj_file_)
        else:
            las_file = os.path.join(laz_dir, las_file_)
            traj_file = os.path.join(traj_dir, traj_file_)

        out_dir = os.path.join(
            output_dir, os.path.splitext(os.path.basename(las_file))[0]
        )
        logging.debug((f"Output directory: {out_dir}"))
        try:
            point_cloud = load_pointcloud(
                las_file, fields=config["processing"]["fields"]
            )
            memmap_meta = point_cloud.metadata

            trajectory = read_trajectory_file(traj_file)

            detected_objects = label_pointcloud(
                cfg=config, points=memmap_meta, trajectory=trajectory, pipeline=pipeline
            )
            samples = calculate_sampling_points(
                config, memmap_meta.memmap("r+"), trajectory, pipeline, detected_objects
            )

            tasks = []
            for sampling_points, basenames, outdir in samples:
                output_dir_ = os.path.join(out_dir, outdir)
                tasks += generate_tasks(
                    sampling_points, basenames, config, memmap_meta, output_dir_
                )

            number_of_processes = config["processing"].get("number_of_processes", 1)
            if number_of_processes > 1:
                with Pool(number_of_processes) as p:
                    results = p.map(frame_randomizer, tasks)
                for images_, annotations_, output_dir in results:
                    if output_dir not in images:
                        images[output_dir] = []
                        annotations[output_dir] = []

                    annotations[output_dir] += annotations_
                    images[output_dir] += images_

            else:
                for t in tasks:
                    annotations_, images_, output_dir = frame_randomizer(t)
                    if output_dir not in images:
                        images[output_dir] = []
                        annotations[output_dir] = []
                    annotations[output_dir] += annotations_
                    images[output_dir] += images_

            try:
                coco = COCO(config["paths"]["coco_file"])
                cats = coco.loadCats(coco.getCatIds())
            except:
                cats = [
                    {"id": 0, "name": "pole-Fgrm"},
                    {"id": 1, "name": "deer_fence", "supercategory": "pole-Fgrm"},
                    {"id": 2, "name": "deer_fence_pole", "supercategory": "pole-Fgrm"},
                    {"id": 3, "name": "gantry_sign_pole", "supercategory": "pole-Fgrm"},
                    {"id": 4, "name": "light_pole", "supercategory": "pole-Fgrm"},
                    {"id": 5, "name": "traffic_pole", "supercategory": "pole-Fgrm"},
                    {"id": 7, "name": "unclassified_pole", "supercategory": "pole-Fgrm"},
                    # Add more categories if needed
                ]
            for out_dir in images:
                os.makedirs(out_dir, exist_ok=True)
                for i, annotation in enumerate(annotations[out_dir]):
                    annotation["id"] = i
                write_coco_json(
                    os.path.join(out_dir, "coco.json"),
                    images=images[out_dir],
                    annotations=annotations[out_dir],
                    categories=cats,
                )

        finally:
            os.remove(memmap_meta.filepath)


if __name__ == "__main__":
    main()
