"""COCO-based point cloud labelling and instance deprojection.

Orchestrates the pipeline that loads COCO annotations, computes
object-to-point-cloud mappings, projects instance labels into the point
cloud using spatial proximity, and writes the labelled point cloud plus
per-instance metadata to disk.
"""
import os

import pickle

import numpy as np
from scipy.spatial import KDTree
from pycocotools.coco import COCO
import logging

from range_image_generator.utils import Instance
from .multiprocessing_workers import calculate_annotation_indexes, make_tasks
from range_image_generator.io_utils import (
    update_classification,
    create_progress_pountclouds,
)


def label_pointcloud_coco(orginal_pc_mmap_meta, coco_file, cfg):
    """Label a point cloud using COCO annotations and projection results.

    This function orchestrates the full pipeline:

    * Loads annotations from a COCO file.
    * Creates deprojection tasks.
    * Computes object-to-point cloud mappings.
    * Projects labels into the point cloud.
    * Writes the updated point cloud and metadata to disk.

    Parameters
    ----------
    orginal_pc_mmap_meta : MemmapMetadata
        Metadata object providing access to the point cloud memmap.
    coco_file : str
        Path to the COCO annotation file.
    cfg : dict
        Configuration containing processing parameters.

    Returns
    -------
    tuple of (dict, list)
        * ``poles``: mapping of instance IDs to :class:`Instance` objects.
        * COCO category definitions.
    """

    orginal_pc_mmap = orginal_pc_mmap_meta.memmap("r+")

    # ---------------------------------------------------------------
    # Process images
    # ---------------------------------------------------------------
    coco = COCO(coco_file)

    annotation_tasks = make_tasks(coco, orginal_pc_mmap_meta, cfg)
    poles = calculate_annotation_indexes(annotation_tasks)

    tmp_pointcloud = project_to_pointcloud(poles, orginal_pc_mmap)

    output_dir = cfg["paths"].get("output_dir", "out")
    os.makedirs(output_dir, exist_ok=True)

    logging.debug(
        (
            "output_dir",
            output_dir,
            os.path.join(output_dir, "classificated_pointcloud.laz"),
        )
    )

    update_classification(
        orginal_pc_mmap,
        cfg["paths"]["laz_dir"],
        os.path.join(output_dir, "classificated_pointcloud.laz"),
    )
    #For debugging / publication
    # create_progress_pountclouds(
    #     tmp_pointcloud,
    #     cfg["paths"]["laz_dir"],
    #     os.path.join(output_dir, "progress_pointcloud.laz"),
    # )

    with open(os.path.join(output_dir, "pole.pickle"), "wb") as p:
        pickle.dump(poles, p)
    with open(os.path.join(output_dir, "categories.pickle"), "wb") as c:
        pickle.dump(coco.loadCats(ids=coco.getCatIds()), c)

    return poles, coco.loadCats(ids=coco.getCatIds())


def project_to_pointcloud(poles, orginal_pc_mmap):
    """Assign instance labels to the point cloud using spatial proximity.

    A KD-tree is used to find neighboring points around each detected
    instance, and labels are propagated accordingly.

    Parameters
    ----------
    poles : dict
        Mapping of instance IDs to :class:`Instance` objects.
    orginal_pc_mmap : numpy.memmap
        Memory-mapped point cloud array.

    Returns
    -------
    numpy.ndarray
        A new point cloud array holding the xyz columns, per-instance
        classification/expansion columns, and a final instance-id column.
    """

    logging.debug(("Create KD_tree"))

    kd_tree = KDTree(orginal_pc_mmap[:, :3], 25)
    logging.debug(("KD_tree created"))
    pole_values = list(poles.values())

    # xyz, orginal, expanded, progress
    progress_pc_datasize = 3 + 1 + 1 + len(pole_values[0].extra["labelExpansion"]) + 1

    logging.debug((f"{progress_pc_datasize=}"))

    progress_pc = np.zeros(
        (orginal_pc_mmap.shape[0], progress_pc_datasize), dtype=orginal_pc_mmap.dtype
    )
    progress_pc[:, :3] = orginal_pc_mmap[:, :3]
    for pole in pole_values:
        if pole.instance_id not in poles:
            logging.debug(("DELETED POLE?", pole.instance_id))
            # pole was already overwritten so we skip it. This shouldn't ever happen?
            continue

        extra_fields = pole.extra

        progress_pc[extra_fields["unexpanded"], 3] = int(pole.instance_id)
        progress_pc[extra_fields["expanded"], 4] = int(pole.instance_id)
        i = 5

        for progress in extra_fields["labelExpansion"]:
            progress_pc[progress, i] = int(pole.instance_id)
            i += 1

        idxs, category, id = combine_ids(pole, poles, orginal_pc_mmap, kd_tree)
        orginal_pc_mmap[idxs, 4] = int(category)
        if orginal_pc_mmap.shape[1] > 5:
            orginal_pc_mmap[idxs, -1] = int(id)

    # Final result
    progress_pc[:, -1] = orginal_pc_mmap[:, -1]

    return progress_pc


def combine_ids(pole: Instance, poles, orginal_pc_mmap, kd_tree):
    """Merge overlapping or neighboring instances in the point cloud.

    This function examines spatial neighbors and resolves conflicts
    between overlapping instance IDs by merging them and choosing
    the most frequent category.

    TODO better voting when there are more detection results ie. same object in several
    labeled frames

    Parameters
    ----------
    pole : Instance
        The current instance to process.
    poles : dict
        Dictionary of all instances.
    orginal_pc_mmap : numpy.ndarray
        Point cloud data.
    kd_tree : KDTree
        KD-tree for spatial queries.

    Returns
    -------
    tuple of (numpy.ndarray, int, int)
        * Indices of the points belonging to the merged instance.
        * The final class/category ID.
        * The final instance ID.
    """

    category = pole.class_id
    id = pole.instance_id
    orginal_id = pole.instance_id

    idxs = pole.point_indices

    positions = orginal_pc_mmap[idxs, :3]
    indexes = kd_tree.query_ball_point(positions, r=0.2, p=2)
    indexes = [idx for list in indexes for idx in list]
    # indexes = np.array(indexes)

    indexes = np.reshape(indexes, -1)
    neigbor_idxs = np.append(idxs, indexes)
    neigbor_idxs = idxs[idxs < orginal_pc_mmap.shape[0]]
    # existing_category =  np.max(orginal_pc_mmap[idxs,4])
    existing_ids = orginal_pc_mmap[neigbor_idxs, -1]
    unique_ids = np.unique(existing_ids)

    counts = {}
    counts[category] = np.count_nonzero(idxs)
    for old_id in unique_ids:
        if old_id <= 0:
            continue
        logging.debug(("EXISTING ID", orginal_id, id, old_id))
        del poles[id]
        id = int(old_id)
        old_pole = poles[id]
        idxs = np.unique(np.append(idxs, old_pole.point_indices))
        category = old_pole.class_id

        if category in counts:
            counts[category] += np.sum(existing_ids == old_id)
        else:
            counts[category] = np.sum(existing_ids == old_id)
        # idxs += old_pole[3]
    max_count = 0
    for cat in counts:
        if counts[cat] > max_count:
            max_count = counts[cat]
            category = cat
    averange_position_ = np.mean(orginal_pc_mmap[idxs, :3], axis=0)
    id = int(id)
    poles[id] = Instance(
        centroid=averange_position_, class_id=category, instance_id=id, point_indices=idxs
    )

    return idxs, category, id
