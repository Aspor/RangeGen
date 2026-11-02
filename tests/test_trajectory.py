"""Unit tests for ``io_utils.trajectory`` public functions."""

import numpy as np
import yaml

from range_image_generator.io_utils import (
    match_las_and_trajectory,
    parse_nmea_file,
    parse_pose_yaml,
    read_trajectory_file,
    save_csv,
)
from range_image_generator.pyprojCoordinateTranformer import (
    WGS84lalo_to_ETRSTM35FINxy,
)


def _write(path, text: str) -> None:
    path.write_text(text)


def _assert_entry_lists_equal(a: list, b: list) -> None:
    """Compare trajectory entry lists element-wise (they hold numpy arrays)."""
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2] == right[2]


# ---------------------------------------------------------------------------
# read_trajectory_file (text branch)
# ---------------------------------------------------------------------------


def test_read_trajectory_file_text(tmp_path):
    path = tmp_path / "traj.txt"
    _write(
        path,
        "header line\n"
        "1.0 100 200 300 400 500 600 700\n"
        "2.0 1 2 3 4 5 6 7\n",
    )

    trajectory = read_trajectory_file(str(path))

    assert len(trajectory) == 2
    (ts1, pos1, orient1), (ts2, pos2, orient2) = trajectory

    assert ts1 == 1.0
    # position = (parts[2], parts[1], parts[3]) -> (200, 100, 300)
    np.testing.assert_array_equal(pos1, [200.0, 100.0, 300.0])
    np.testing.assert_array_equal(orient1, [400.0, 500.0, 600.0])

    assert ts2 == 2.0
    np.testing.assert_array_equal(pos2, [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(orient2, [4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# parse_pose_yaml
# ---------------------------------------------------------------------------


def test_parse_pose_yaml(tmp_path):
    data = {
        "poseList": {
            "poseListSize": 2,
            "0": [10.0, 1.0, 2.0, 3.0],
            "1": [20.0, 4.0, 5.0, 6.0],
        }
    }
    path = tmp_path / "poses.yaml"
    _write(path, yaml.safe_dump(data))

    poses = parse_pose_yaml(str(path))

    # poseListSize is skipped; two entries remain.
    assert len(poses) == 2
    assert poses[0][0] == 10.0
    np.testing.assert_array_equal(poses[0][1], [1.0, 2.0, 3.0])
    assert poses[0][2] == [0, 0, 0]

    assert poses[1][0] == 20.0
    np.testing.assert_array_equal(poses[1][1], [4.0, 5.0, 6.0])


# ---------------------------------------------------------------------------
# parse_nmea_file
# ---------------------------------------------------------------------------


def test_parse_nmea_file(tmp_path):
    path = tmp_path / "fix.dfnav"
    _write(
        path,
        "$GNRMC,120000,A,6030.0000,N,02415.0000,E,0.0,360.0,120326,,,A*hh\n"
        "$GNGGA,120000,6030.0000,N,02415.0000,E,1,08,1.0,123.4,M,46.0,M,,*hh\n",
    )

    data = parse_nmea_file(str(path))

    assert len(data) == 1
    timestamp, position, orientation = data[0]

    assert timestamp == 120000.0
    assert orientation == (0, 0, 0)

    # lat/lon: 6030.0000 N -> 60.5, 02415.0000 E -> 24.25
    expected_x, expected_y = WGS84lalo_to_ETRSTM35FINxy(60.5, 24.25)
    np.testing.assert_allclose(position[0], expected_x)
    np.testing.assert_allclose(position[1], expected_y)
    # altitude + 18 offset (123.4 + 18 = 141.4)
    np.testing.assert_allclose(position[2], 141.4)


def test_parse_nmea_file_without_rmc_date(tmp_path):
    path = tmp_path / "no_date.dfnav"
    # A GGA fix with no preceding RMC date is ignored.
    _write(
        path,
        "$GNGGA,120000,6030.0000,N,02415.0000,E,1,08,1.0,123.4,M,46.0,M,,*hh\n",
    )

    assert parse_nmea_file(str(path)) == []


# ---------------------------------------------------------------------------
# read_trajectory_file dispatch to NMEA / YAML
# ---------------------------------------------------------------------------


def test_read_trajectory_file_dispatch_nmea(tmp_path):
    path = tmp_path / "route.dfnav"
    _write(
        path,
        "$GNRMC,120000,A,6030.0000,N,02415.0000,E,0.0,360.0,120326,,,A*hh\n"
        "$GNGGA,120000,6030.0000,N,02415.0000,E,1,08,1.0,123.4,M,46.0,M,,*hh\n",
    )

    _assert_entry_lists_equal(
        read_trajectory_file(str(path)), parse_nmea_file(str(path))
    )


def test_read_trajectory_file_dispatch_yaml(tmp_path):
    path = tmp_path / "poses.yaml"
    _write(
        path,
        yaml.safe_dump(
            {"poseList": {"poseListSize": 1, "0": [1.0, 2.0, 3.0, 4.0]}}
        ),
    )

    _assert_entry_lists_equal(
        read_trajectory_file(str(path)), parse_pose_yaml(str(path))
    )


# ---------------------------------------------------------------------------
# match_las_and_trajectory
# ---------------------------------------------------------------------------


def test_match_las_and_trajectory_no_traj_dir(tmp_path):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    (las_dir / "a.las").write_text("x")
    (las_dir / "b.laz").write_text("x")
    (las_dir / "ignore.txt").write_text("x")

    result = match_las_and_trajectory(str(las_dir))

    matched = sorted(name for name, in result)
    assert matched == ["a.las", "b.laz"]


def test_match_las_and_trajectory_single_file(tmp_path):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    (las_dir / "Record012320240101_1200.las").write_text("x")

    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    (traj_dir / "Traj20240101_1200.txt").write_text("x")

    result = match_las_and_trajectory(str(las_dir), str(traj_dir))

    assert result == ("Record012320240101_1200.las", "Traj20240101_1200.txt")


def test_match_las_and_trajectory_multiple_files(tmp_path):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    (las_dir / "Record012320240101_1200.las").write_text("x")
    (las_dir / "Record999920250202_2300.las").write_text("x")

    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    (traj_dir / "Traj20240101_1200.txt").write_text("x")
    (traj_dir / "Traj20250202_2300.txt").write_text("x")

    result = sorted(match_las_and_trajectory(str(las_dir), str(traj_dir)))

    assert result == [
        ("Record012320240101_1200.las", "Traj20240101_1200.txt"),
        ("Record999920250202_2300.las", "Traj20250202_2300.txt"),
    ]


def test_match_las_and_trajectory_single_file_shortcut(tmp_path):
    las_file = tmp_path / "single.las"
    las_file.write_text("x")

    result = match_las_and_trajectory(
        str(las_file), "TrajSomething.txt"
    )

    assert result == [("single.las", "TrajSomething.txt")]
