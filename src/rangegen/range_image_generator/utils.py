"""Utility structures and helper functions for LiDAR frame generation,
including directory creation, typed frame metadata, and custom iteration
helpers.

This module is intentionally lightweight and shared across the package.
"""

import os
import re

import numpy as np
from dataclasses import dataclass

from typing import List, Dict, Any
# -------------------------------------------------------------------------
# Frame metadata structures
# -------------------------------------------------------------------------

#: Namedtuple describing intrinsic parameters for a projected LiDAR frame.

#: NumPy dtype list for storing frame intrinsic information.
FrameIntrinsics_dtype = [
    ("fov_down", "<f8"),
    ("fov_up", "<f8"),
    ("height", "<u4"),
    ("width", "<u4"),
    ("max_range", "<f8"),
]

points_dtype = np.float64


# -------------------------------------------------------------------------
# Directory creation utilities
# -------------------------------------------------------------------------


def make_output_dirs(baseoutput_dir, save_pc=False):
    """Create all required subdirectories for storing projection outputs.

    Parameters
    ----------
    baseoutput_dir : str
        Base directory where all subfolders will be created.

    save_pc : bool
        Whether to also create point cloud export folders.
    """
    subdirs = [
        "pose",
        "class",
        "rgb",
        "range",
        "intensity",
        "range_filtered",
        "intensity_filtered",
        "pc_index",
        "frame_instricts",
    ]

    for sd in subdirs:
        os.makedirs(os.path.join(baseoutput_dir, sd), exist_ok=True)

    if save_pc:
        os.makedirs(os.path.join(baseoutput_dir, "pc"), exist_ok=True)
        os.makedirs(os.path.join(baseoutput_dir, "pc_transformed"), exist_ok=True)


# -------------------------------------------------------------------------
# Custom enumeration helper
# -------------------------------------------------------------------------


def enumerate2(xs, skip_start=0, skip_end=0, stride=1):
    """Enumerate a sequence with optional starting offset, ending offset, and
    stride (step size). Useful when selecting every Nth element or skipping
    leading/trailing sections.

    Parameters
    ----------
    xs : sequence
        Iterable or indexable sequence.

    skip_start : int
        Number of elements to skip from the beginning.

    skip_end : int
        Number of elements to skip at the end.

    stride : int
        Step size.

    Yields
    ------
    tuple
        (index, element), where the index advances by stride.
    """
    index = skip_start
    length = len(xs) - skip_end

    for x in xs[skip_start:length:stride]:
        yield index, x
        index += stride


@dataclass(slots=True)
class MemmapMetadata:
    """Metadata describing a memmapped point cloud stored on disk.

    Holds just enough information to re-open the underlying array with
    ``numpy.memmap`` without keeping it resident. The default layout produced by
    ``load_pointcloud`` is ``(x, y, z, intensity, classification, r, g, b)``;
    ``field_mapping`` records the meaning/order of the non-xyz columns when a
    different field set is used.

    Attributes
    ----------
    filepath : str
        Path to the memmapped array file.
    dtype : numpy.dtype
        Struct dtype of the array.
    shape : tuple
        Array shape ``(N, num_fields)``.
    field_mapping : dict, optional
        Mapping from field name to column; ``None`` for the default layout.
    """

    filepath: str
    dtype: np.dtype
    shape: any
    field_mapping: dict = None

    def memmap(self, mode="r"):
        return np.memmap(
            self.filepath,
            dtype=self.dtype,
            shape=self.shape,
            mode=mode,
        )


@dataclass(slots=True)
class FrameIntrinsics:
    """Spherical LiDAR range image parameters.
    """

    fov_down: float
    fov_up: float
    height: int
    width: int
    max_range: float

    def write(self, file):
        """Serialize the intrinsics to a NumPy structured array on disk.

        Parameters
        ----------
        file : str
            Destination path (``.npy``) for the saved array.
        """
        to_save = np.asarray(self.as_tuple(), dtype=FrameIntrinsics_dtype)
        np.save(file=file, arr=to_save)

    @classmethod
    def read(cls, file):
        """Load intrinsics previously written by ``write``.

        Parameters
        ----------
        file : str
            Path (``.npy``) of the saved intrinsics.

        Returns
        -------
        FrameIntrinsics
            The reconstructed intrinsics.
        """
        data = np.load(file)
        try:
            fov_down, fov_up, height, width, max_range = data[:]
        except:
            fov_down, fov_up, height, width, max_range = (
                data["fov_down"],
                data["fov_up"],
                data["height"],
                data["width"],
                data["max_range"],
            )
        return cls(fov_down, fov_up, height, width, max_range)

    def as_tuple(self):
        """Return the intrinsics as a ``(fov_down, fov_up, height, width, max_range)`` tuple."""
        return (self.fov_down, self.fov_up, self.height, self.width, self.max_range)


NUM_RE = re.compile(
    r"np\.float64\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
)


def parse_pose(text: str):
    """Parse a 6-DoF pose from a text blob containing six numbers.

    Accepts either ``numpy.float64(...)``-wrapped values or plain
    whitespace-separated numbers; the first three become the position and the
    last three the orientation.

    Parameters
    ----------
    text : str
        Raw pose text containing exactly six numeric values.

    Returns
    -------
    Pose
        The parsed 6-DoF pose.
    """
    matches = NUM_RE.findall(text)
    if len(matches) == 6:
        pose = [float(m) for m in matches]
        return Pose(pose[:3], pose[3:])
    else:
        text = text.replace("\n", " ")
        matches = text.split()
        pose = [float(m) for m in matches]
        return Pose(pose[:3], pose[3:])


@dataclass(slots=True)
class Pose:
    """A 6-DoF sensor pose.

    Attributes
    ----------
    position : numpy.ndarray
        XYZ position in world coordinates, shape ``(3,)``.
    orientation : numpy.ndarray
        Orientation as Euler angles (xyz, degrees) or a compatible quaternion.
    """

    position: np.ndarray  # shape (3,)
    orientation: np.ndarray  # shape (3,), Euler xyz in degrees


@dataclass(slots=True)
class Frame_Task:
    """A single frame-generation unit of work.

    Bundles everything ``frame_randomizer`` needs to produce one (or a batch of)
    frames: the pose, the shared point cloud, the frame intrinsics, the output
    location, and optional position/orientation perturbation settings.

    Attributes
    ----------
    output_dir : str
        Directory to write the frame outputs into.
    basename : str
        Base name for the generated output files.
    pose : Pose
        The pose (or poses) to generate frames from.
    points : MemmapMetadata
        Shared point cloud.
    frame_intrinsics : FrameIntrinsics
        Spherical projection parameters.
    position_perturbation : tuple, optional
        Per-axis ``(min, max)`` position offsets. Default is ``None``.
    orientation_perturbation : tuple, optional
        Per-axis ``(min, max)`` orientation offsets. Default is ``None``.
    number_of_pos_perturbations : int, optional
        Number of position perturbations. Default is ``0``.
    number_of_orientation_perturbations : int, optional
        Number of orientation perturbations per position. Default is ``0``.
    save_pc : bool, optional
        Whether to export PCD point clouds. Default is ``False``.
    """

    output_dir: any
    basename: str
    pose: Pose
    points: MemmapMetadata
    frame_intrinsics: FrameIntrinsics
    position_perturbation: tuple = None
    orientation_perturbation: tuple = None
    number_of_pos_perturbations: int = 0
    number_of_orientation_perturbations: int = 0
    save_pc: bool = False


@dataclass
class DeprojectionTask:
    """Task passed to a deprojection worker."""

    annotations: List[Dict[str, Any]]
    image_metadata: Dict[str, Any]
    pose: Pose
    points_memmap: MemmapMetadata
    # kdtree: KDTree
    frame_intrinsics: FrameIntrinsics


@dataclass
class Instance:
    """A refined 3D object instance extracted from LiDAR points."""

    instance_id: int
    class_id: int
    point_indices: np.ndarray  # indices into the original point cloud
    centroid: np.ndarray  # shape (3,)

    extra: Any = None

    def num_points(self) -> int:
        return int(self.point_indices.shape[0])
