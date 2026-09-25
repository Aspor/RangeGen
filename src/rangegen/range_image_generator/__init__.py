"""range_image_generator — core LiDAR range-image building blocks.

The core, dependency-light layer of the top-level ``rangegen``
package. It provides the typed helpers, IO, spherical projection, pose
transforms, and the high-level frame-export pipeline, plus COCO annotation
generation.

Submodules
----------
utils.py                        FrameIntrinsics/Pose/Frame_Task helpers, dir creation
io_utils/                       point cloud & trajectory IO, image export, classification
projection.py                   spherical projection + hole-filling filter
transforms.py                   pose transforms, perturbations, distance cropping
frame_generator.py              high-level frame export pipeline
coco_writer.py                  COCO annotation + JSON writing
pyprojCoordinateTranformer.py   CRS conversions (WGS84 <-> ETRS-TM35FIN/UTM), NMEA GGA parsing
read_exif.py                    EXIF helpers for exported images
"""

from .utils import (
    FrameIntrinsics,
    enumerate2,
    make_output_dirs,
    Frame_Task,
    Pose,
    MemmapMetadata,
)

from .io_utils import (
    load_pointcloud,
    write_pointcloud,
    read_trajectory_file,
    save_image,
    save_classification_image,
    match_las_and_trajectory,
)


from .transforms import (
    transform_points,
    perturb_position,
    perturb_orientation,
)

from .frame_generator import (
    frame_randomizer,
    batched_frame_randomizer,
)

from .projection import range_projection_idx_only

from .coco_writer import generate_coco, write_coco_json

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
    "range_projection_idx_only",
    "generate_coco",
    "write_coco_json",
]
