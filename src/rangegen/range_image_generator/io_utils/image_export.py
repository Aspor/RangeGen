
"""Utilities for exporting range, intensity, and classification images.

Saves generated images to disk via OpenCV and can embed GPS and
``DateTime`` EXIF metadata when a source position is available. Also
exports classification images, optionally applying a matplotlib colormap.
"""
import logging
import numpy as np
import cv2
from PIL import Image
import matplotlib as mpl

from PIL import ExifTags
from PIL.ExifTags import GPS
from ..pyprojCoordinateTranformer import ETRSTM35FINxy_to_WGS84lalo




# -------------------------------------------------------------------------
# Image saving
# -------------------------------------------------------------------------


def save_image(
    image, output_path, timestamp=None, position=None, multiplier=1, color=False
):
    """Save a range/intensity/RGB image and optionally embed minimal GPS EXIF.

    Parameters
    ----------
    image : ndarray
        Image data to save.

    output_path : str
        Output image file path.

    timestamp : float or None
        Optional timestamp embedded as EXIF DateTime.

    position : ndarray or None
        (x, y, z) in ETRS-TM35FIN coordinates. Converted to WGS84 EXIF GPS.

    multiplier : float
        Scaling factor applied to 16‑bit grayscale data.

    color : bool
        If True, image is assumed to already be 3‑channel RGB.
    """

    if not color:
        image = multiplier * image
        image[image < 0] = -1
        image = image.astype(np.uint16)

    cv2.imwrite(output_path, image)

    img = Image.open(output_path)
    exif = img.getexif()

    # Encode GPS EXIF if position given
    if position is not None:
        lat, lon = ETRSTM35FINxy_to_WGS84lalo(*position[:2])
        exif[ExifTags.Base.GPSInfo] = {
            GPS.GPSLatitudeRef: "N" if lat >= 0 else "S",
            GPS.GPSLatitude: abs(lat),
            GPS.GPSLongitudeRef: "E" if lon >= 0 else "W",
            GPS.GPSLongitude: abs(lon),
            GPS.GPSAltitudeRef: 0,
            GPS.GPSAltitude: float(position[2]),
        }

    if timestamp is not None:
        exif[ExifTags.Base.DateTime] = str(timestamp)
    # img.save(output_path, exif=exif)


# -------------------------------------------------------------------------
# Classification pseudo‑color export
# -------------------------------------------------------------------------


def save_classification_image(
    image, output_path, timestamp=None, position=None, color=False
):
    """Save a classification image.

    When ``color`` is ``True``, a matplotlib ``tab20b`` colormap is
    applied and the result is written as a 3-channel image; otherwise the
    raw classification array is written directly.

    Parameters
    ----------
    image : ndarray
        Integer classification values (0–255).
    output_path : str
        Output image file path.
    color : bool, optional
        If ``True``, apply the matplotlib colormap.
    timestamp : float or None, optional
        Optional timestamp; accepted for consistency with
        :func:`save_image` (not embedded in this variant).
    position : ndarray or None, optional
        Optional position; accepted for consistency with
        :func:`save_image` (not embedded in this variant).
    """
    logging.debug(("CLASSIFICATION IMAGE", output_path))

    if color:
        colors = np.asarray(mpl.color_sequences["tab20b"])
        color_img = np.zeros((*image.shape, 3), dtype=np.uint8)

        for idx, _ in np.ndenumerate(image):
            cid = image[idx] % len(colors)
            color_img[idx] = (colors[cid] * 255).astype(np.uint8)

        cv2.imwrite(output_path, color_img)
        return

    cv2.imwrite(output_path, image)
