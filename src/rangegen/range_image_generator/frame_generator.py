"""High‑level frame generation pipeline for LiDAR data.

This module provides the orchestration logic for:
- Applying optional perturbations to poses
- Transforming point clouds into the sensor coordinate frame
- Projecting them into spherical range images
- Saving all output images and metadata to disk

The module ties together the utilities from:
    io_utils.py
    projection.py
    transforms.py
    utils.py
"""

import os
import time
import gc
import logging
import numpy as np

from scipy.spatial.transform import Rotation as R

import random

from .utils import FrameIntrinsics, make_output_dirs

from .transforms import (
    transform_points,
    perturb_position,
    perturb_orientation,
    batch_crop_points_by_distance,
    crop_points_by_distance,
)
from .projection import (
    range_projection_idx_only,
    proj_intensity_from_idx,
    proj_class_from_idx,
    proj_color_from_idx,
    proj_range_from_idx,
    filter_invalid_points,
)
from .io_utils import (
    write_pointcloud,
    save_image,
    save_classification_image,
)


from .coco_writer import generate_coco

# -------------------------------------------------------------------------
# Frame randomization driver
# -------------------------------------------------------------------------


def batched_frame_randomizer(task):
    """Generate frames for a batch of nearby poses, sharing one crop.

    Batched counterpart of ``frame_randomizer``. Because the poses in a batch
    are assumed to be close together, the point cloud is cropped once around
    all of them and that subset is reused for every frame, which is faster when
    working with very large point clouds.

    Parameters
    ----------
    task : Frame_Task
        Task bundle holding the pose list, the shared point cloud, frame
        intrinsics, output directory, and the position/orientation
        perturbation settings.

    Returns
    -------
    tuple
        ``(images, annotations)`` collected across every pose in the batch.
    """

    make_output_dirs(task.output_dir, task.save_pc)

    # Crop points to near sensor
    t1 = time.time()
    points = task.points.memmap("r")

    points_mmap, original_indices = batch_crop_points_by_distance(
        points,
        [pose.position for pose in task.pose],
        range=task.frame_intrinsics.max_range * 2.5,
    )
    # No perturbations → run standard single-thread path
    no_perturb = (
        task.number_of_pos_perturbations + task.number_of_orientation_perturbations == 0
    )

    for basename, pose in zip(task.basename, task.pose):
        images, annotations = randomize_frames(
            no_perturb,
            points_mmap,
            pose.position,
            pose.orientation,
            task.output_dir,
            task.frame_intrinsics,
            basename,
            original_indices,
            points,
            task.save_pc,
            task.number_of_pos_perturbations,
            task.number_of_orientation_perturbations,
            task.position_perturbation,
            task.orientation_perturbation,
        )
        gc.collect()
    return images, annotations


def frame_randomizer(task):
    """Crop the cloud to a pose's neighbourhood, then generate its frames.

    Crops the point cloud to the neighbourhood of ``task.pose`` and delegates to
    ``randomize_frames``, which applies any configured perturbations and writes
    the frame outputs. Kept as a separate function so the batched variant can be
    swapped in for groups of nearby poses.

    Parameters
    ----------
    task : Frame_Task
        Task bundle holding the pose, the shared point cloud, frame intrinsics,
        output directory, and the position/orientation perturbation settings.

    Returns
    -------
    tuple
        ``(images, annotations, output_dir)`` produced for this pose.
    """
    make_output_dirs(task.output_dir, task.save_pc)
    points = task.points.memmap("r")
    # Crop points to near sensor
    t1 = time.time()
    points_mmap, original_indices = crop_points_by_distance(
        points, task.pose.position, range=task.frame_intrinsics.max_range * 2.5
    )
    # No perturbations → run standard single-thread path
    no_perturb = (
        task.number_of_pos_perturbations + task.number_of_orientation_perturbations == 0
    )
    return randomize_frames(
        no_perturb,
        points_mmap,
        task.pose.position,
        task.pose.orientation,
        task.output_dir,
        task.frame_intrinsics,
        task.basename,
        original_indices,
        points,
        task.save_pc,
        task.number_of_pos_perturbations,
        task.number_of_orientation_perturbations,
        task.position_perturbation,
        task.orientation_perturbation,
    )


def randomize_frames(
    no_perturb,
    points_mmap,
    position,
    orientation,
    output_dir,
    frame_intrinsics,
    basename,
    original_indices,
    points,
    save_pc,
    number_of_pos_perturbations,
    number_of_orientation_perturbations,
    position_perturbation,
    orientation_perturbation,
):
    """Generate one or more frames by perturbing a single base pose.

    When no perturbations are configured a single frame is written directly.
    Otherwise ``number_of_pos_perturbations`` new positions are sampled (and,
    if ``number_of_orientation_perturbations`` is set, each is combined with that
    many orientation samples) and ``transform_and_project`` is run from every
    resulting pose.

    Parameters
    ----------
    no_perturb : bool
        When ``True`` (no perturbations requested) run the single fast path.
    points_mmap : numpy.ndarray
        Cropped point cloud, shape ``(N, 8)``, in the ``(x, y, z, intensity,
        class, r, g, b)`` layout.
    position : numpy.ndarray
        Base sensor position, shape ``(3,)``.
    orientation : numpy.ndarray
        Base sensor orientation (quaternion or Euler angles).
    output_dir : str
        Directory to write the frame outputs into.
    frame_intrinsics : FrameIntrinsics
        Spherical projection parameters (fov, size, max range).
    basename : str
        Base name for the generated output files.
    original_indices : numpy.ndarray
        Indices of the cropped points in the original cloud, shape ``(N, 1)``.
    points : numpy.ndarray or numpy.memmap
        The original (uncropped) point cloud, re-read via ``proj_idx`` for
        colour/classification reconstruction.
    save_pc : bool
        Whether to also export PCD point clouds.
    number_of_pos_perturbations : int
        How many position perturbations to sample.
    number_of_orientation_perturbations : int
        How many orientation perturbations to sample per position.
    position_perturbation : tuple
        Per-axis ``(min, max)`` ranges for position offsets.
    orientation_perturbation : tuple
        Per-axis ``(min, max)`` ranges for orientation offsets.

    Returns
    -------
    tuple
        ``(images, annotations, output_dir)`` collected from the generated
        frames.
    """

    if no_perturb:
        _, images, annotations = transform_and_project(
            points_mmap,
            position,
            orientation,
            output_dir,
            frame_intrinsics,
            basename,
            original_indices,
            original_pc=points,
            save_pc=save_pc,
        )
        return images, annotations, output_dir

    images = []
    annotations = []

    # Position perturbations
    for i in range(number_of_pos_perturbations):
        new_pos = perturb_position(position, *position_perturbation)

        if number_of_orientation_perturbations > 0:
            # double loop (pos × ori)
            for j in range(number_of_orientation_perturbations):
                new_ori = perturb_orientation(orientation, *orientation_perturbation)
                fname = f"{basename}_{i}_{j}"
                _, images_, annotations_ = transform_and_project(
                    points_mmap,
                    new_pos,
                    new_ori,
                    output_dir,
                    frame_intrinsics,
                    fname,
                    original_indices,
                    points,
                    None,
                    0,
                    save_pc,
                )
                annotations += annotations_
                images += images_

        else:
            # only pos perturbation
            fname = f"{basename}_{i}"
            _, images_, annotations_ = transform_and_project(
                points_mmap,
                new_pos,
                orientation,
                output_dir,
                frame_intrinsics,
                fname,
                original_indices,
                points,
                None,
                0,
                save_pc,
                images,
                annotations,
            )
            annotations += annotations_
            images += images_

    # Orientation‑only perturbations
    if number_of_pos_perturbations == 0:
        for j in range(number_of_orientation_perturbations):
            new_ori = perturb_orientation(orientation, *orientation_perturbation)
            fname = f"{basename}_{j}"
            _, images_, annotations_ = transform_and_project(
                points_mmap,
                position,
                new_ori,
                output_dir,
                frame_intrinsics,
                fname,
                original_indices,
                points,
                None,
                0,
                save_pc,
                images,
                annotations,
            )
            annotations += annotations_
            images += images_

    # Collect results (mainly needed for COCO writer images/annotations)
    return images, annotations, output_dir


# -------------------------------------------------------------------------
# Main per-frame processing pipeline
# -------------------------------------------------------------------------


def transform_and_project(
    points,
    position,
    orientation_to_use,
    output_dir,
    frame_intrinsics: FrameIntrinsics,
    filename,
    original_indices,
    original_pc,
    timestamp=None,
    valid_proportion_limit=0,
    save_pc=False,
):
    """Transform a point cloud to a pose, project it, and save all frame outputs.

    Runs the core per-frame pipeline: brings the point cloud into the sensor
    frame, projects it to a spherical range image, and writes all derived
    outputs (pose, intrinsics, RGB, intensity, range, classification, and
    optional PCD data).

    Parameters
    ----------
    points : numpy.ndarray
        Cropped point cloud used for projection, shape ``(N, 8)``.
    position : numpy.ndarray
        Sensor position, shape ``(3,)``.
    orientation_to_use : scipy.spatial.transform.Rotation or numpy.ndarray
        Rotation applied to bring the cloud into the sensor frame.
    output_dir : str
        Directory to write the frame outputs into.
    frame_intrinsics : FrameIntrinsics
        Spherical projection parameters (fov_down, fov_up, height, width,
        max_range).
    filename : str
        Base name for the generated output files.
    original_indices : numpy.ndarray
        Indices of the cropped points in the original cloud, shape ``(N, 1)``.
    original_pc : numpy.ndarray or numpy.memmap
        The original (uncropped) point cloud re-read for colour/classification.
    timestamp : float or None, optional
        Acquisition timestamp embedded in exported images. Default is ``None``.
    valid_proportion_limit : float, optional
        Reserved: minimum proportion of valid projected pixels. Default is
        ``0``.
    save_pc : bool, optional
        Whether to also export PCD point clouds. Default is ``False``.

    Returns
    -------
    tuple
        ``(ok, image_info, annotations)`` where ``ok`` is ``True`` and the
        other two are the COCO image record and the annotation list.
    """

    orientation_to_use = R(orientation_to_use)

    # Store pose as Euler angles
    pose_array = np.asarray(
        (*position, *orientation_to_use.as_euler("xyz", degrees=True))
    )

    # Write pose file
    pose_path = os.path.join(output_dir, "pose", f"{filename}.txt")
    with open(pose_path, "w") as f:
        f.write(str(pose_array).replace("\n", "")[1:-1])

    # Save frame intrinsics
    intr_path = os.path.join(output_dir, "frame_instricts", f"{filename}.npy")
    frame_intrinsics.write(intr_path)

    # Transform point cloud
    start = time.time()

    transformed = transform_points(
        points[:, :3],
        position,
        orientation_to_use,
    )

    # Projection
    start_proj = time.time()

    # Indexses of points to be projected onto image. Only Indexses are saved in memory
    proj_idx = range_projection_idx_only(
        transformed,
        fov_up=frame_intrinsics.fov_up,
        fov_down=frame_intrinsics.fov_down,
        proj_H=frame_intrinsics.height,
        proj_W=frame_intrinsics.width,
        max_range=frame_intrinsics.max_range,
    )

    del transformed

    image_info, annotations = save_images(
        points=original_pc,
        proj_idx=proj_idx,
        output_dir=output_dir,
        filename=filename,
        orientation_to_use=orientation_to_use,
        position=position,
        timestamp=timestamp,
        save_pc=save_pc,
        original_indices=original_indices,
    )

    logging.debug((f"Saved frame: {filename}"))
    return True, image_info, annotations


def save_images(
    points,
    proj_idx,
    output_dir,
    filename,
    position,
    orientation_to_use,
    timestamp,
    original_indices,
    save_pc=False,
):
    """Write all derived per-frame outputs to ``output_dir``.

    Given the projected pixel-index map, reconstructs the range, intensity,
    colour, and classification images and writes them (plus the pose index map
    and optional PCD exports). When classes are present it also builds the COCO
    image record and annotations from the classification image.

    #TODO Add option to choose which images are saved.
    # Read fields from pointcloud metadata instead of hardcoding

    Parameters
    ----------
    points : numpy.ndarray or numpy.memmap
        Original point cloud indexed by ``proj_idx`` to read per-pixel values.
    proj_idx : numpy.ndarray
        Projected ``(H, W)`` pixel-to-point index map, ``-1`` for empty pixels.
    output_dir : str
        Directory to write the frame outputs into.
    filename : str
        Base name for the generated output files.
    position : numpy.ndarray
        Sensor position, shape ``(3,)``.
    orientation_to_use : scipy.spatial.transform.Rotation
        Sensor orientation, used for the transformed PCD export.
    timestamp : float or None
        Acquisition timestamp embedded in exported images.
    original_indices : numpy.ndarray
        Indices mapping cropped points back to the original cloud, shape
        ``(N, 1)``.
    save_pc : bool, optional
        Whether to also export PCD point clouds. Default is ``False``.

    Returns
    -------
    tuple
        ``([image_info], annotations)`` when a classification image was written,
        otherwise ``([], [])``.
    """

    mask = proj_idx >= 0
    proj_idx[mask] = original_indices[proj_idx[mask], 0]
    np.save(
        os.path.join(output_dir, "pc_index", f"{filename}.npy"),
        proj_idx,
    )
    if save_pc:
        vertex_img = points[proj_idx]
        write_pointcloud(
            vertex_img,
            os.path.join(output_dir, "pc", f"{filename}.pcd"),
        )
        write_pointcloud(
            vertex_img,
            os.path.join(output_dir, "pc_transformed", f"{filename}.pcd"),
            transform=(position, orientation_to_use.as_matrix()),
        )
        del vertex_img

    range_img = proj_range_from_idx(proj_idx, points, position)
    filtered_range = filter_invalid_points(range_img.copy())
    try:
        save_image(
            range_img,
            os.path.join(output_dir, "range", f"{filename}.png"),
            timestamp,
            position,
            multiplier=1000,
        )

        save_image(
            filtered_range,
            os.path.join(output_dir, "range_filtered", f"{filename}.png"),
            timestamp,
            position,
            multiplier=1000,
        )
    except:
        logging.debug(("RANGE IMAGE FAILURE?"))

    del range_img, filtered_range

    intensity_img = proj_intensity_from_idx(proj_idx, points)
    filtered_intensity = filter_invalid_points(intensity_img.copy())
    try:
        save_image(
            intensity_img,
            os.path.join(output_dir, "intensity", f"{filename}.png"),
            timestamp,
            position,
        )

        save_image(
            filtered_intensity,
            os.path.join(output_dir, "intensity_filtered", f"{filename}.png"),
            timestamp,
            position,
        )
    except Exception:
        logging.debug(("INTENSITY IMAGE FAILURE?"))
    del intensity_img, filtered_intensity

    try:
        proj_color = proj_color_from_idx(proj_idx, points)
        # Save RGB, range, intensity
        save_image(
            proj_color,
            os.path.join(output_dir, "rgb", f"{filename}.png"),
            timestamp,
            position,
            color=True,
        )
        del proj_color
    except Exception as e:
        logging.debug(("COLOR ERROR", e))
        pass

    try:
        class_img = proj_class_from_idx(proj_idx, points)

        # Classification image (only if classes >1 exist)
        if np.any(class_img > 1):
            save_classification_image(
                class_img,
                os.path.join(output_dir, "class", f"{filename}.png"),
                color=True,
            )

            # Leading zeros will be discarded so we add some randomness to the end to
            # mitigate duplicate IDs (filenames 301_1_0_0 and 030_11_0_0 will map to same id )
            image_id = int(f"{filename}_{random.randint(1, 10**4):0{5}}")

            annotations = generate_coco(class_img, image_id=image_id)
            height, width = proj_idx.shape

            image_info = {
                "id": int(image_id),
                "file_name": f"{filename}.png",
                "width": int(width),
                "height": int(height),
            }

            return [image_info], annotations
    except Exception as e:
        logging.debug(("CLASS EXEPT", e))
        pass
    return [], []
