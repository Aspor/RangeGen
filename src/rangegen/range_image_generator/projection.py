"""Range image projection utilities for LiDAR point clouds.

This module provides:
- A high‑performance spherical projection of LiDAR data into a 2D range
  image representation compatible with RangeNet++ / Range‑MCL style
  algorithms.
- A neighborhood‑based post‑filter for filling invalid pixels (-1 values)
  in range or intensity images.

Column layout
-------------
Currently the projection helpers assume a fixed 8-column point cloud layout::

    [x, y, z, intensity, classification, r, g, b]

i.e. ``x, y, z`` at columns 0-2, ``intensity`` at column 3,
``classification`` at column 4, and RGB at columns 5-7. This matches the output
of ``load_pointcloud`` for the default ``fields=["intensity", "classification"]``.
``proj_class_from_idx`` and ``proj_color_from_idx`` rely on these fixed column
positions, so pass point clouds with that layout.

This could be changes in the future to use field_mapping in pointcloud metadata
"""

import numpy as np

# -------------------------------------------------------------------------
# Spherical projection
# -------------------------------------------------------------------------


def range_projection_idx_only(
    point_cloud,
    fov_up=25.0,
    fov_down=-25.0,
    proj_H=64,
    proj_W=900,
    max_range=50.0,
):
    """Project a point cloud to a spherical range image and return only proj_idx.

    Parameters
    ----------
    point_cloud : ndarray (N, 8)
        (x, y, z, intensity, class, r, g, b)

    Returns
    -------
    proj_idx : ndarray (H, W), int32
        Pixel → index into original point_cloud, -1 if empty
    """

    # FOV to radians
    fov_up = np.deg2rad(fov_up)
    fov_down = np.deg2rad(fov_down)
    fov = abs(fov_down) + abs(fov_up)

    # Compute depth
    xyz = point_cloud[:, :3]
    depth = xyz * xyz
    depth = np.add.reduce(depth, axis=1, keepdims=False)
    depth = np.sqrt(depth)
    # depth = np.linalg.norm(xyz, axis=1).astype(np.float32)

    # Memory optimization, only one temporary arary in memory
    mask = depth > 0.0
    mask &= depth <= max_range

    # Pitch
    pitch = np.arcsin(xyz[:, 2] / depth).astype(np.float32)

    # Memory optimization, only one temporary arary in memory
    mask &= pitch >= fov_down
    mask &= pitch <= fov_up

    # Filter
    depth = depth[mask]
    order = np.argsort(depth)[::-1]

    del depth
    pitch = pitch[mask]
    # current = point_cloud[mask]

    (
        x,
        y,
    ) = point_cloud[mask, :2].T

    # Yaw
    # yaw = (-np.arctan2(y, x))

    # Normalize to image plane
    proj_x = ((-np.arctan2(y, x)) / np.pi + 1.0) * 0.5
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov
    del pitch

    proj_x = np.floor(proj_x * proj_W)
    proj_y = np.floor(proj_y * proj_H)

    proj_x = np.clip(proj_x, 0, proj_W - 1).astype(np.int32)
    proj_y = np.clip(proj_y, 0, proj_H - 1).astype(np.int32)

    # Sort so closest point wins
    proj_x = proj_x[order]
    proj_y = proj_y[order]
    # indices = np.flatnonzero(mask)[order]
    indices = np.argwhere(mask)[order]
    del order
    # Allocate output
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)

    # Write indices
    proj_idx[proj_y, proj_x] = indices[:, 0]

    return proj_idx


def proj_range_from_idx(proj_idx, point_cloud, position):
    """Reconstruct a projected range image from a pixel-index map.

    Parameters
    ----------
    proj_idx : numpy.ndarray
        Projected ``(H, W)`` pixel-to-point index map, ``-1`` for empty.
    point_cloud : numpy.ndarray
        Point cloud indexed by ``proj_idx``; uses the xyz columns.
    position : numpy.ndarray
        Sensor position, shape ``(3,)``, used to compute per-pixel depth.

    Returns
    -------
    numpy.ndarray
        ``proj_range`` ``(H, W)`` float32 range image, ``-1`` where empty.
    """
    H, W = proj_idx.shape
    proj_range = np.full((H, W), -1.0, dtype=np.float32)

    mask = proj_idx >= 0
    pts = point_cloud[proj_idx[mask], :3]

    proj_range[mask] = np.linalg.norm(pts - position, axis=1)
    return proj_range


def proj_intensity_from_idx(proj_idx, point_cloud):
    """Reconstruct a projected intensity image from a pixel-index map.

    Parameters
    ----------
    proj_idx : numpy.ndarray
        Projected ``(H, W)`` pixel-to-point index map, ``-1`` for empty.
    point_cloud : numpy.ndarray
        Point cloud indexed by ``proj_idx``; uses the intensity column (3).

    Returns
    -------
    numpy.ndarray
        ``proj_intensity`` ``(H, W)`` float32 image, ``-1`` where empty.
    """
    H, W = proj_idx.shape
    proj_intensity = np.full((H, W), -1.0, dtype=np.float32)

    mask = proj_idx >= 0
    proj_intensity[mask] = point_cloud[proj_idx[mask], 3]

    return proj_intensity


def proj_class_from_idx(proj_idx, point_cloud):
    """Reconstruct a projected classification image from a pixel-index map.

    The classification label is read from column 4 of ``point_cloud`` and
    offset by 1 so that 0 always means "empty/background".

    Parameters
    ----------
    proj_idx : numpy.ndarray
        Projected ``(H, W)`` pixel-to-point index map, ``-1`` for empty.
    point_cloud : numpy.ndarray
        Point cloud indexed by ``proj_idx``; uses the classification column (4).

    Returns
    -------
    numpy.ndarray
        ``proj_class`` ``(H, W)`` uint8 image, 0 where empty.
    """
    H, W = proj_idx.shape
    proj_class = np.zeros((H, W), dtype=np.uint8)

    mask = proj_idx >= 0
    proj_class[mask] = point_cloud[proj_idx[mask], 4].astype(np.uint8) + 1

    return proj_class


def proj_color_from_idx(proj_idx, point_cloud):
    """Reconstruct a projected colour image from a pixel-index map.

    RGB is read from columns 5-7 of ``point_cloud`` and stored as BGR so the
    result can be handed directly to OpenCV.

    Parameters
    ----------
    proj_idx : numpy.ndarray
        Projected ``(H, W)`` pixel-to-point index map, ``-1`` for empty.
    point_cloud : numpy.ndarray
        Point cloud indexed by ``proj_idx``; uses the RGB columns (5-7).

    Returns
    -------
    numpy.ndarray
        ``proj_color`` ``(H, W, 3)`` uint16 image in BGR order, 0 where empty.
    """
    H, W = proj_idx.shape
    proj_color = np.zeros((H, W, 3), dtype=np.uint16)

    mask = proj_idx >= 0
    colors = point_cloud[proj_idx[mask], 5:8]

    # Store as (B, G, R) for opencv
    proj_color[mask, 0] = colors[:, 2]
    proj_color[mask, 1] = colors[:, 1]
    proj_color[mask, 2] = colors[:, 0]

    return proj_color


# # -------------------------------------------------------------------------
# # Hole filling for invalid pixels
# # -------------------------------------------------------------------------


def filter_invalid_points(image, kernel_size=5, threshold=17, max_passes=3):
    """Fill missing values (-1) in a 2D image by averaging nearby valid pixels.

    The algorithm uses repeated passes (up to `max_passes`). For each
    invalid pixel, a kernel-sized region is scanned; if the number of
    valid samples >= threshold, the pixel is replaced with their average.

    Parameters
    ----------
    image : ndarray (H, W)
        2D array containing depth or intensity values, with holes set to -1.

    kernel_size : int
        Size of the scanning window. Must be odd.

    threshold : int
        Minimum count of valid neighbors required before filling a pixel.

    max_passes : int
        Maximum number of refinement passes.

    Returns
    -------
    ndarray
        Modified image with some holes filled.
    """
    modified = True
    passes = 0

    while modified and passes < max_passes:
        passes += 1
        invalid = np.where(image == -1)
        modified = False

        for i, j in zip(*invalid):
            valid = 0
            val_sum = 0
            eff_thresh = threshold

            for dx in range(kernel_size):
                for dy in range(kernel_size):
                    xx = i - kernel_size // 2 + dx
                    yy = j - kernel_size // 2 + dy

                    try:
                        v = image[xx, yy]
                        if v != -1:
                            valid += 1
                            val_sum += v
                    except IndexError:
                        eff_thresh -= 0.5  # close to edge → lower threshold
                        continue

            if valid >= eff_thresh:
                image[i, j] = val_sum / valid
                modified = True

    return image
