"""Unit tests for ``rangegen.reprojection.filenamematcher``.

``find_matches`` is a pure function over a COCO-like object and a dataset tree,
so it is tested for real using a lightweight COCO stand-in and a temporary
dataset directory.
"""

from unittest import mock

from reprojection import filenamematcher as FM


def _coco(image_ids, names=None, file_names=None):
    """Build a minimal COCO stand-in exposing the two methods used."""
    coco = mock.Mock()
    coco.getImgIds.return_value = image_ids

    def load_imgs(img_id):
        img = {"id": img_id}
        if names is not None and img_id in names:
            img["extra"] = {"name": names[img_id]}
        if file_names is not None and img_id in file_names:
            img["file_name"] = file_names[img_id]
        return [img]

    coco.loadImgs.side_effect = load_imgs
    return coco


def test_find_matches_maps_pose_files_via_extra_name(tmp_path):
    pose_dir = tmp_path / "scene" / "pose"
    pose_dir.mkdir(parents=True)
    (pose_dir / "frameA.txt").write_text("0 0 0 0 0 0")
    (pose_dir / "frameB.txt").write_text("0 0 0 0 0 0")
    # A file outside any "pose" directory must be ignored.
    (tmp_path / "pose.txt").write_text("x")

    coco = _coco([1, 2, 3], names={1: "frameA.png", 2: "frameB.png", 3: "frameC.png"})

    mapping = FM.find_matches(coco, tmp_path)

    assert set(mapping) == {1, 2}
    assert mapping[1] == str((pose_dir / "frameA.txt").resolve())
    assert mapping[2] == str((pose_dir / "frameB.txt").resolve())


def test_find_matches_uses_file_name_fallback(tmp_path):
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    (pose_dir / "shot123.txt").write_text("0 0 0 0 0 0")

    # No "extra.name" -> stem is derived from file_name via the roboflow pattern.
    coco = _coco([7], file_names={7: "shot123_png.rf.12345"})

    mapping = FM.find_matches(coco, tmp_path)

    assert mapping[7] == str((pose_dir / "shot123.txt").resolve())


def test_find_matches_skips_unmatched_images(tmp_path):
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    (pose_dir / "frameA.txt").write_text("0 0 0 0 0 0")

    coco = _coco([1], names={1: "frameB.png"})  # stem "frameB" has no pose file

    mapping = FM.find_matches(coco, tmp_path)

    assert mapping == {}
