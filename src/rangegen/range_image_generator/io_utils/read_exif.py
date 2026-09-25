"""Utilities for reading EXIF metadata from PIL images.

Provides a helper to extract the full EXIF tag dictionary of an image,
with the nested ``GPSInfo`` tag expanded into a readable sub-dictionary
of decoded GPS tags.
"""
from PIL.ExifTags import TAGS, GPSTAGS


def get_exif_data(image):
    """Return the EXIF data of a PIL image as a dictionary.

    Top-level EXIF tags are decoded by name. The nested ``GPSInfo`` tag
    is expanded into its own sub-dictionary of decoded GPS tags.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image to read EXIF metadata from.

    Returns
    -------
    dict
        Mapping of decoded EXIF tag names to their values. The
        ``GPSInfo`` entry, when present, maps decoded GPS tag names to
        their values.
    """
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value

    return exif_data
