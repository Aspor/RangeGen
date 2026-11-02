"""Unit tests for ``io_utils.image_export`` public functions."""

import cv2
import matplotlib as mpl
import numpy as np

from range_image_generator.io_utils import (
    save_classification_image,
    save_image,
)


def _read(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------


def test_save_image_grayscale_normalizes_dtype(tmp_path):
    image = np.array([[100.0, -5.0, 300.0], [400.0, 0.0, 500.0]])
    output = tmp_path / "gray.png"

    save_image(image, str(output))

    saved = _read(str(output))
    assert saved.dtype == np.uint16
    # Negative values are clamped to -1, which wraps to 65535 as uint16.
    expected = np.array([[100, 65535, 300], [400, 0, 500]], dtype=np.uint16)
    np.testing.assert_array_equal(saved, expected)


def test_save_image_grayscale_applies_multiplier(tmp_path):
    image = np.array([[10.0, 20.0]])
    output = tmp_path / "scaled.png"

    save_image(image, str(output), multiplier=2)

    saved = _read(str(output))
    assert saved.dtype == np.uint16
    np.testing.assert_array_equal(saved, np.array([[20, 40]], dtype=np.uint16))


def test_save_image_color_passthrough(tmp_path):
    image = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]],
        dtype=np.uint8,
    )
    output = tmp_path / "rgb.png"

    save_image(image, str(output), color=True)

    saved = _read(str(output))
    assert saved.shape == (2, 2, 3)
    assert saved.dtype == np.uint8
    # A plain imwrite/imread round-trip preserves channel order.
    np.testing.assert_array_equal(saved[0, 0], np.array([255, 0, 0], dtype=np.uint8))


def test_save_image_with_position_and_timestamp(tmp_path):
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Valid ETRS-TM35FIN coordinates (Oulu, Finland).
    position = np.array([348961.7, 6710252.8, 10.0])
    output = tmp_path / "exif.png"

    save_image(image, str(output), timestamp=1234.5, position=position)

    assert output.exists()
    saved = _read(str(output))
    assert saved.dtype == np.uint16
    np.testing.assert_array_equal(
        saved, np.array([[1, 2], [3, 4]], dtype=np.uint16)
    )


# ---------------------------------------------------------------------------
# save_classification_image
# ---------------------------------------------------------------------------


def test_save_classification_image_raw(tmp_path):
    image = np.array([[0, 5], [10, 25]], dtype=np.uint8)
    output = tmp_path / "class_raw.png"

    save_classification_image(image, str(output))

    saved = _read(str(output))
    assert saved.shape == (2, 2)
    assert saved.dtype == np.uint8
    np.testing.assert_array_equal(saved, image)


def test_save_classification_image_color_applies_colormap(tmp_path):
    image = np.array([[0, 1]], dtype=np.uint8)
    output = tmp_path / "class_color.png"

    save_classification_image(image, str(output), color=True)

    saved = _read(str(output))
    assert saved.shape == (1, 2, 3)
    assert saved.dtype == np.uint8

    colors = np.asarray(mpl.color_sequences["tab20b"])
    expected_0 = (colors[0] * 255).astype(np.uint8)
    expected_1 = (colors[1] * 255).astype(np.uint8)
    np.testing.assert_array_equal(saved[0, 0], expected_0)
    np.testing.assert_array_equal(saved[0, 1], expected_1)


def test_save_classification_image_color_wraps_class_id(tmp_path):
    # class id 25 wraps to index 5 (25 % 20) of the 20-colour table.
    image = np.array([[25]], dtype=np.uint8)
    output = tmp_path / "class_wrap.png"

    save_classification_image(image, str(output), color=True)

    saved = _read(str(output))
    colors = np.asarray(mpl.color_sequences["tab20b"])
    expected = (colors[5] * 255).astype(np.uint8)
    np.testing.assert_array_equal(saved[0, 0], expected)
