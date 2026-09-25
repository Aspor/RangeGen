"""Matching of COCO images to their pose files.

Roboflow renames files so a helper module to rematch the files to
originals.
Discovers pose text files under any ``pose`` directory in a dataset and
maps each COCO image ID to the corresponding pose file by comparing
filename stems. Can be extended to include metadata for later use
"""
from pathlib import Path
import logging


def find_matches(coco, dataset_root):
    """Map COCO image IDs to their pose files by matching filename stems.

    Discovers all pose files under any directory named ``pose`` and
    matches them to COCO images by comparing the image name stem (from
    the ``extra.name`` field when present, otherwise derived from
    ``file_name``) against the pose file stems.

    Parameters
    ----------
    coco : COCO
        Loaded COCO annotation object.
    dataset_root : str or pathlib.Path
        Root directory containing the dataset.

    Returns
    -------
    dict
        Mapping of COCO image ID to the resolved pose file path (str).
    """

    # Root directory containing the dataset
    dataset_root = Path(dataset_root)

    # ------------------------------------------------------------------
    # Find all pose files under directories named "pose"
    # ------------------------------------------------------------------
    pose_files = {}

    for pose_dir in dataset_root.rglob("pose"):
        if pose_dir.is_dir():
            for txt_file in pose_dir.glob("*.txt"):
                pose_files[txt_file.stem] = txt_file.resolve()

    # ------------------------------------------------------------------
    # Build mapping
    # ------------------------------------------------------------------
    mapping = {}
    for id in coco.getImgIds():
        image = coco.loadImgs(id)[0]
        original_name = image.get("extra", {}).get("name")
        if original_name:
            stem = Path(original_name).stem
        else:
            stem = image["file_name"].split("_png.rf.")[0]

        if stem in pose_files:
            mapping[image["id"]] = str(pose_files[stem])

    logging.debug((f"Created {len(mapping)} mappings"))
    return mapping
