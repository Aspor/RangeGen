"""Unit tests for ``io_utils.read_exif`` public functions."""

from PIL.ExifTags import TAGS, GPSTAGS

from range_image_generator.io_utils import read_exif


class _FakeImage:
    """Minimal stand-in exposing only the ``_getexif`` attribute used."""

    def __init__(self, info):
        self._info = info

    def _getexif(self):
        return self._info


def _tag(name: str) -> int:
    for tag_id, decoded in TAGS.items():
        if decoded == name:
            return tag_id
    raise KeyError(name)


def _gps_tag(name: str) -> int:
    for tag_id, decoded in GPSTAGS.items():
        if decoded == name:
            return tag_id
    raise KeyError(name)


def test_get_exif_data_decodes_top_level_and_gps():
    info = {
        _tag("Make"): "TestMake",
        _tag("Model"): "TestModel",
        _tag("GPSInfo"): {
            _gps_tag("GPSLatitudeRef"): "N",
            _gps_tag("GPSLatitude"): 60.5,
            _gps_tag("GPSLongitudeRef"): "E",
            _gps_tag("GPSLongitude"): 24.25,
        },
    }

    result = read_exif.get_exif_data(_FakeImage(info))

    assert result["Make"] == "TestMake"
    assert result["Model"] == "TestModel"
    assert result["GPSInfo"] == {
        "GPSLatitudeRef": "N",
        "GPSLatitude": 60.5,
        "GPSLongitudeRef": "E",
        "GPSLongitude": 24.25,
    }


def test_get_exif_data_keeps_unknown_tag_id():
    info = {99999: "mystery"}

    result = read_exif.get_exif_data(_FakeImage(info))

    assert result[99999] == "mystery"


def test_get_exif_data_none_returns_empty():
    assert read_exif.get_exif_data(_FakeImage(None)) == {}


def test_get_exif_data_empty_returns_empty():
    assert read_exif.get_exif_data(_FakeImage({})) == {}
