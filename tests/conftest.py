"""Shared pytest fixtures for the ``io_utils`` unit tests.

Provides a small, deterministic LAS file (with known point values) that the
point-cloud, classification, and trajectory tests read from.
"""

import numpy as np
import laspy
import pytest


def _build_las(path: str) -> dict:
    """Create a LAS file at ``path`` and return its known contents.

    The file holds ``n`` points with simple, predictable values so tests can
    assert exact column contents after loading.
    """
    n = 5
    x = np.arange(1, n + 1, dtype=np.int32)                # 1..5
    y = (np.arange(1, n + 1) * 10).astype(np.int32)         # 10..50
    z = (np.arange(1, n + 1) * 100).astype(np.int32)        # 100..500
    intensity = (np.arange(1, n + 1) * 10).astype(np.uint16)  # 10..50
    classification = np.arange(n, dtype=np.uint8)           # 0..4

    las = laspy.LasData(header=laspy.LasHeader(version="1.2", point_format=1))
    las.x = x
    las.y = y
    las.z = z
    las.intensity = intensity
    las.classification = classification
    las.write(path)

    return {
        "path": path,
        "n": n,
        "x": x,
        "y": y,
        "z": z,
        "intensity": intensity,
        "classification": classification,
    }


@pytest.fixture(scope="session")
def las_data(tmp_path_factory) -> dict:
    """A LAS file path plus the expected per-point values it contains."""
    directory = tmp_path_factory.mktemp("las_data")
    return _build_las(str(directory / "sample.las"))
