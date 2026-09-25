"""Multiprocessing workers for the COCO deprojection pipeline.

Builds per-image deprojection tasks from COCO annotations and runs the
deprojection and cluster-expansion work in parallel over a
``multiprocessing.Pool``, collecting the resulting instance mappings.
"""
import os
import logging

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from pycocotools import mask as maskUtils


from tqdm import tqdm

from multiprocessing.pool import Pool


from range_image_generator.projection import range_projection_idx_only
from range_image_generator.transforms import transform_points, crop_points_by_distance
from range_image_generator.utils import DeprojectionTask, Instance

from range_image_generator.utils import FrameIntrinsics

from .label_expansion import expand_labels

from .filenamematcher import find_matches
from range_image_generator.utils import parse_pose


def calculate_annotation_indexes(annotation_tasks):
    """Process deprojection tasks in parallel to compute instance mappings.

    Each task is executed in a multiprocessing pool and the results are
    incrementally collected into a dictionary of instances.

    Parameters
    ----------
    annotation_tasks : list of DeprojectionTask
        Tasks to process.

    Returns
    -------
    dict
        Mapping of instance IDs to :class:`Instance` objects.
    """

    poles = {}

    worker_count = 6
    if worker_count == 1:
        for i, task in enumerate(annotation_tasks):
            new_poles = deprojection_worker(task)
            poles.update(new_poles)

        return poles

    with Pool(worker_count) as p:
        for new_poles in tqdm(
            p.imap_unordered(deprojection_worker, annotation_tasks),
            total=len(annotation_tasks),
        ):
            poles.update(new_poles)
    return poles


def make_tasks(coco, orginal_pc_mmap_meta, cfg):
    """Create deprojection tasks from COCO annotations.

    For each image in the dataset, this function loads the image
    metadata, pose information, and annotations, and bundles them into
    :class:`DeprojectionTask` objects.

    Parameters
    ----------
    coco : COCO
        Loaded COCO dataset object.
    orginal_pc_mmap_meta : MemmapMetadata
        Metadata object for point cloud access.
    cfg : dict
        Configuration with parameters such as FOV and resolution.

    Returns
    -------
    list of DeprojectionTask
        List of tasks ready for parallel processing.
    """

    image_ids = coco.getImgIds()

    # If user wants specific indices, filter here
    if cfg["processing"].get("image_indices", False):
        image_ids = [image_ids[i] for i in cfg.image_indices]
    annotation_tasks = []

    posefile_mapping = find_matches(coco, cfg["paths"]["datasetroot"])

    width = cfg["image"].get("width", 128)
    height = cfg["image"].get("height", 64)
    max_range = cfg["image"].get("max_range", 65)
    elevation_range = tuple(cfg["image"].get("elevation_range", [-45, 45]))

    frame_intrinsics = FrameIntrinsics(
        fov_down=elevation_range[0],
        fov_up=elevation_range[1],
        height=height,
        width=width,
        max_range=max_range,
    )

    for image_id in image_ids:
        try:
            frame_intrinsics_path = (
                posefile_mapping[image_id]
                .replace("pose", "frame_intrinsics")
                .replace(".txt", ".npy")
            )
            frame_intrinsics = FrameIntrinsics.read(frame_intrinsics_path)
        except Exception as e:
            logging.debug((" ", e, frame_intrinsics_path))
            frame_intrinsics = frame_intrinsics

        img_metadata = coco.loadImgs(image_id)[0]

        img_name = img_metadata["extra"]["name"]

        pose_path = posefile_mapping[
            image_id
        ]  #  os.path.join("pose", img_name[:-3] + "txt")
        if not os.path.isfile(pose_path):
            continue
        # Load pose
        with open(pose_path) as p:
            pose = parse_pose(p.read())

        # Load annotations
        ann_ids = coco.getAnnIds(imgIds=img_metadata["id"])
        annotations = coco.loadAnns(ann_ids)
        task = DeprojectionTask(
            annotations=annotations,
            image_metadata=img_metadata,
            pose=pose,
            frame_intrinsics=frame_intrinsics,
            points_memmap=orginal_pc_mmap_meta,
        )
        annotation_tasks.append(task)

    return annotation_tasks


def decode_segmentation(segmentation, height, width):
    """Decode COCO-style segmentation into a binary mask.

    Parameters
    ----------
    segmentation : list | dict
        COCO segmentation.
    height : int
        Image height.
    width : int
        Image width.

    Returns
    -------
    mask : np.ndarray
        Binary segmentation mask.
    """
    if isinstance(segmentation, list):
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation["counts"], list):
        rle = maskUtils.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation

    return maskUtils.decode(rle)


def deprojection_worker(
    task,
):
    """Perform deprojection and clustering for all annotations in an image.

    Crops and transforms the point cloud into the sensor frame, projects
    it to the image plane, and for each annotation decodes the
    segmentation mask, expands the resulting point cluster, and records
    the instance together with its expansion progress.

    Parameters
    ----------
    task : DeprojectionTask
        Bundle of the annotations, image metadata, sensor pose, frame
        intrinsics, and point cloud memmap for one image.

    Returns
    -------
    dict
        Mapping from annotation ID to a detected :class:`Instance`.
    """

    annotations = task.annotations
    img_metadata = task.image_metadata
    pos = task.pose.position
    orientation = task.pose.orientation
    frame_intrinsics = task.frame_intrinsics
    original_pc_mmap = task.points_memmap.memmap(mode="r")

    image_id = img_metadata["id"]

    poles = {}

    # Crop to the area of interest
    point_cloud, original_indices = crop_points_by_distance(
        original_pc_mmap, pos, frame_intrinsics.max_range * 1.5
    )

    if len(point_cloud) == 0:
        return poles

    # Transform to sensor origin
    point_cloud[:, :] = transform_points(
        point_cloud,
        position=pos,
        orientation=R.from_euler("xyz", orientation, degrees=True),
    )

    # Calculate the indexes from the pointcloud. We could also just read the saved indexes
    # but that was later addition.
    proj_idx = range_projection_idx_only(
        point_cloud,
        fov_up=frame_intrinsics.fov_up,
        fov_down=frame_intrinsics.fov_down,
        proj_H=frame_intrinsics.height,
        proj_W=frame_intrinsics.width,
        max_range=frame_intrinsics.max_range,
    )
    if len(proj_idx) == 0:
        return poles

    # Create kd_tree for outlier removal and cluster expansion
    densities = None  # precompute_density(kd_tree, point_cloud[:,:3],0.05)
    kd_tree = KDTree(point_cloud[:, :3], 25)

    for ann in annotations:
        category = int(ann["category_id"])
        ann_id = int(ann["id"])

        logging.debug((f"{image_id=} {ann_id=} {category=}"))

        mask = decode_segmentation(
            ann["segmentation"],
            img_metadata["height"],
            img_metadata["width"],
        )

        classes = np.argwhere(mask)
        idx = proj_idx[classes[:, 0], classes[:, 1]]
        object_indices = idx[idx != -1].astype(int)  # .tolist()
        extra = {}

        extra["unexpanded"] = original_indices[
            object_indices[object_indices < original_indices.shape[0]], 0
        ]

        object_indices, progress = expand_labels(
            mask_indices=object_indices,
            point_cloud=point_cloud,
            kd_tree=kd_tree,
            densities=densities,
        )

        if len(object_indices) == 0:
            continue

        object_indices = original_indices[
            object_indices[object_indices < original_indices.shape[0]], 0
        ]

        positions = original_pc_mmap[object_indices, :3]
        average_position = np.mean(positions, axis=0)

        extra["expanded"] = object_indices

        for i, p in enumerate(progress):
            progress[i] = original_indices[p[p < original_indices.shape[0]], 0]

        extra["labelExpansion"] = progress

        poles[int(ann_id)] = Instance(
            instance_id=ann_id,
            class_id=category,
            centroid=average_position,
            point_indices=object_indices,
            extra=extra,
        )

    return poles
