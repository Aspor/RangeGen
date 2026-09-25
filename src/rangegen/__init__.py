"""rangegen — LiDAR range-image generation toolkit.

A modular toolkit that turns LiDAR point clouds and trajectories into
spherical range images and COCO annotations:

- Reading LAS/LAZ point clouds and NMEA/YAML trajectories
- Transforming and perturbing camera/sensor poses
- Projecting point clouds into spherical range images
- Saving RGB, intensity, classification, range, and pose outputs
- Generating augmented frames via position/orientation perturbations

Top-level entry points
----------------------
main.py            config-driven CLI pipeline (``rangegen``)
generate_from_*    pose-sampling strategies (random, trajectory, poles, objects)

Subpackages
-----------
range_image_generator/  core math, IO, projection, frame generation, COCO
reprojection/           deprojection and COCO label expansion back to 3D
"""

from range_image_generator import (
    FrameIntrinsics,
    enumerate2,
    make_output_dirs,
    Frame_Task,
    Pose,
    MemmapMetadata,
)

from range_image_generator import (
    load_pointcloud,
    write_pointcloud,
    read_trajectory_file,
    save_image,
    save_classification_image,
    match_las_and_trajectory,
)


from range_image_generator import (
    transform_points,
    perturb_position,
    perturb_orientation,
)

from range_image_generator import (
    frame_randomizer,
    batched_frame_randomizer,
)

from range_image_generator import generate_coco, write_coco_json

__all__ = [
    "FrameIntrinsics",
    "enumerate2",
    "make_output_dirs",
    "Frame_Task",
    "Pose",
    "MemmapMetadata",
    "load_pointcloud",
    "write_pointcloud",
    "read_trajectory_file",
    "save_image",
    "save_classification_image",
    "match_las_and_trajectory",
    "transform_points",
    "perturb_position",
    "perturb_orientation",
    "frame_randomizer",
    "batched_frame_randomizer",
    "generate_coco",
    "write_coco_json",
]
