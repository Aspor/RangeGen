"""Unit tests for ``range_image_generator.coco_writer`` public functions."""

import json

import numpy as np

from range_image_generator import coco_writer as C


def _block_image(size: int = 6, block_class: int = 3, image_size: int = 16) -> np.ndarray:
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    start = 3
    image[start : start + size, start : start + size] = block_class
    return image


# ---------------------------------------------------------------------------
# generate_coco (classification-image path)
# ---------------------------------------------------------------------------


def test_generate_coco_class_image_produces_annotation():
    classification = _block_image(size=6, block_class=3)

    annotations = C.generate_coco(classification, annotation_id=0, image_id=42)

    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation["image_id"] == 42
    assert annotation["category_id"] == 2  # block class (3) minus 1
    assert annotation["id"] == 0
    assert annotation["iscrowd"] == 0
    assert len(annotation["bbox"]) == 4
    assert annotation["area"] > 0
    assert annotation["segmentation"]["counts"]


def test_generate_coco_small_component_is_dropped():
    classification = _block_image(size=3, block_class=3)

    assert C.generate_coco(classification, image_id=1) == []


def test_generate_coco_all_background_has_no_annotations():
    classification = np.zeros((16, 16), dtype=np.uint8)

    assert C.generate_coco(classification, image_id=1) == []


# ---------------------------------------------------------------------------
# generate_coco (explicit instance-map path)
# ---------------------------------------------------------------------------


def test_generate_coco_with_instance_map():
    classification = np.zeros((16, 16), dtype=np.uint8)
    classification[3:10, 3:10] = 4
    instance_map = np.zeros((16, 16), dtype=np.uint32)
    instance_map[3:10, 3:10] = 7

    annotations = C.generate_coco(
        classification, annotation_id=0, image_id=9, id_image=instance_map
    )

    assert len(annotations) == 1
    assert annotations[0]["category_id"] == 3  # class (4) minus 1
    assert annotations[0]["image_id"] == 9


def test_generate_coco_multiple_instances():
    classification = np.zeros((16, 16), dtype=np.uint8)
    classification[1:9, 1:9] = 4
    classification[9:16, 9:16] = 6
    instance_map = np.zeros((16, 16), dtype=np.uint32)
    instance_map[1:9, 1:9] = 11
    instance_map[9:16, 9:16] = 12

    annotations = C.generate_coco(
        classification, annotation_id=5, image_id=9, id_image=instance_map
    )

    assert len(annotations) == 2
    assert [a["category_id"] for a in annotations] == [3, 5]
    # Annotation ids continue from the starting annotation_id.
    assert [a["id"] for a in annotations] == [5, 6]


# ---------------------------------------------------------------------------
# write_coco_json
# ---------------------------------------------------------------------------


def test_write_coco_json_roundtrip(tmp_path):
    classification = _block_image(size=6, block_class=3)
    annotations = C.generate_coco(classification, image_id=1)
    images = [{"id": 1, "file_name": "a.png", "width": 16, "height": 16}]
    categories = [{"id": 1, "name": "pole", "supercategory": "obj"}]
    path = tmp_path / "coco.json"

    C.write_coco_json(str(path), images, annotations, categories)

    with open(path) as handle:
        data = json.load(handle)

    assert set(data.keys()) == {"info", "images", "annotations", "categories"}
    assert data["images"] == images
    assert data["categories"] == categories
    assert len(data["annotations"]) == 1
    assert isinstance(data["annotations"][0]["segmentation"]["counts"], str)


def test_write_coco_json_empty_dataset(tmp_path):
    path = tmp_path / "empty.json"

    C.write_coco_json(str(path), [], [], [])

    with open(path) as handle:
        data = json.load(handle)

    assert data["images"] == []
    assert data["annotations"] == []
    assert data["categories"] == []
