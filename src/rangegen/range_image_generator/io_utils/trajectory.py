
"""Utilities for reading and matching LiDAR trajectory files.

Reads vehicle trajectories from several text, YAML, and NMEA formats,
matches LAS/LAZ point cloud files with their trajectory files by
filename, and provides helpers for NMEA coordinate conversion and CSV
export.
"""

from datetime import datetime
import csv
import os

import yaml
import numpy as np

import logging

from ..pyprojCoordinateTranformer import (
    WGS84lalo_to_ETRSTM35FINxy,
)
# -------------------------------------------------------------------------
# Trajectory reading
# -------------------------------------------------------------------------


def read_trajectory_file(file_path):
    """Read a trajectory file formatted as:

        timestamp  unused  y  x  z  roll  pitch  yaw

    The function returns tuples of (timestamp, position, orientation_vector).

    Parameters
    ----------
    file_path : str
        Path to the trajectory file to read.

    Returns
    -------
    list of tuple
        Each element is (timestamp, position(3,), orientation_rpy(3,)).
    """
    trajectory = []
    if file_path.endswith("dfnav"):
        return parse_nmea_file(file_path)
    if file_path.endswith("yaml"):
        return parse_pose_yaml(file_path)
    with open(file_path, "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue

            parts = line.strip().split()
            timestamp = float(parts[0])

            # File order is X/Y swapped → convert to array in x,y,z order
            position = np.array(
                [
                    float(parts[2]),  # x
                    float(parts[1]),  # y
                    float(parts[3]),  # z
                ]
            )

            orientation = np.array(
                [
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                ]
            )

            trajectory.append((timestamp, position, orientation))

    return trajectory



def match_las_and_trajectory(las_dir, traj_dir=None, sort=False):
    """Match LAS/LAZ point cloud files with trajectory text files by timestamp
    pattern appearing in both filenames.

    Parameters
    ----------
    las_dir : str
        Directory containing LAS/LAZ files, or single LAS/LAZ file.
    traj_dir : str
        Directory containing trajectory TXT files.
    sort : bool
        Sort LAS files alphabetically.

    Returns
    -------
    list[tuple(str, str)]
        List of (las_file, traj_file) pairs.
    """

    # Single-file shortcut
    if os.path.isfile(las_dir):
        try:
            return [(os.path.basename(las_dir), os.path.basename(traj_dir))]
        except FileNotFoundError:
            return ((os.path.basename(las_dir),),)

    las_files = [
        f for f in os.listdir(las_dir) if f.endswith(".las") or f.endswith(".laz")
    ]

    if sort:
        las_files.sort()
    if traj_dir is None:
        # return tuple of tuples
        return ((las_file,) for las_file in las_files)

    traj_files = [
        f for f in os.listdir(traj_dir) if f.endswith(".txt") and f.startswith("Traj")
    ]

    # One LAS → one trajectory
    if len(las_files) == 1:
        return (las_files[0], traj_files[0])

    pairs = []

    # Extract identifier: 'RecordXXXX_YYYYMMDD_HHMMSS'
    sample = las_files[0]
    id_start = sample.find("Record") + 10
    id_end = id_start + 13

    for lasf in las_files:
        ident = lasf[id_start:id_end]
        for trajf in traj_files:
            if ident in trajf:
                pairs.append((lasf, trajf))
                traj_files.remove(trajf)
                break

    return pairs




def dm_to_dd(dm, direction):
    """Convert NMEA degree-minute format to decimal degrees.

    Parameters
    ----------
    dm : str
        Degree-minute value as a string, e.g. ``ddmm.mmmm`` for latitude
        or ``dddmm.mmmm`` for longitude.
    direction : str
        Hemisphere direction: ``N``/``S`` for latitude, ``E``/``W`` for
        longitude.

    Returns
    -------
    float or None
        The value in decimal degrees, negated for ``S``/``W``. ``None``
        is returned when ``dm`` is empty.
    """
    if not dm:
        return None

    if len(dm.split(".")[0]) > 4:
        deg_len = 3  # longitude
    else:
        deg_len = 2  # latitude

    degrees = int(dm[:deg_len])
    minutes = float(dm[deg_len:])
    dd = degrees + (minutes / 60)

    if direction in ["S", "W"]:
        dd *= -1

    return dd


def parse_nmea_file(filepath):
    """Parse an NMEA file into a list of trajectory entries.

    ``$GNRMC`` sentences establish the current date and ``$GNGGA``
    sentences yield the fixes, whose GPS coordinates are converted to
    ETRS-TM35FIN (x, y) with a height offset applied.

    Parameters
    ----------
    filepath : str
        Path to the NMEA file to parse.

    Returns
    -------
    list
        One ``[timestamp, position, orientation]`` entry per GGA fix,
        where ``position`` is an ``(x, y, z)`` array and ``orientation``
        is ``(0, 0, 0)``.
    """
    data = []
    initial_pos = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        current_date = None

        for line in f:
            line = line.strip()

            if line.startswith("$GNRMC"):
                parts = line.split(",")

                time_raw = parts[1]
                date_raw = parts[9]

                if time_raw and date_raw:
                    current_date = datetime.strptime(
                        date_raw + time_raw[:6], "%d%m%y%H%M%S"
                    )
                continue
                lat = dm_to_dd(parts[3], parts[4])
                lon = dm_to_dd(parts[5], parts[6])

                data.append(
                    {
                        "timestamp": current_date,
                        "latitude": lat,
                        "longitude": lon,
                        "altitude_m": None,
                        "source": "RMC",
                    }
                )

            elif line.startswith("$GNGGA"):
                parts = line.split(",")

                if current_date is None:
                    continue

                time_raw = parts[1]

                dt = datetime.strptime(
                    current_date.strftime("%Y%m%d") + time_raw[:6], "%Y%m%d%H%M%S"
                )

                lat = dm_to_dd(parts[2], parts[3])
                lon = dm_to_dd(parts[4], parts[5])
                # OFFSETS IN LIDAR DATA: 333816.611, 7222578.964, 35.087
                x, y = WGS84lalo_to_ETRSTM35FINxy(lat, lon)

                # x,y = to_tm.transform(lon,lat )

                # x= x
                # y= -y

                fix_quality = int(parts[6]) if parts[6] else None
                altitude = float(parts[9]) if parts[9] else None

                if initial_pos is None:
                    initial_pos = np.array([x, y, altitude + 20.087])

                pos = np.array([x, y, altitude + 18])  # - initial_pos

                data.append([float(time_raw), pos, (0, 0, 0)])

                # data.append({
                #     "timestamp": dt,
                #     "latitude": lat,
                #     "longitude": lon,
                #     "altitude_m": altitude,
                #     "fix_quality": fix_quality,
                #     "source": "GGA"
                # })

    return data


def parse_pose_yaml(filepath):
    """Parse a pose-list YAML file into a list of pose entries.

    Parameters
    ----------
    filepath : str
        Path to the YAML file containing a ``poseList`` mapping.

    Returns
    -------
    list
        One ``[timestamp, position(3,), [0, 0, 0]]`` entry per pose,
        skipping the ``poseListSize`` key.
    """
    with open(filepath) as f:
        pose_list = yaml.safe_load(f)["poseList"]

    return [
        [
            v[0],
            np.array(
                [
                    v[1],
                    v[2],
                    v[3],
                ]
            ),
            [0, 0, 0],
        ]
        for k, v in pose_list.items()
        if k != "poseListSize"
    ]


def save_csv(data, filename="parsed_rtk_with_height.csv"):
    """Write a list of records to a CSV file. Used for debugging

    Parameters
    ----------
    data : list of dict
        Records to write; the keys of the first record define the columns.
    filename : str, optional
        Output CSV file path.
    """
    keys = data[0].keys()

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    file_path = "20260526-100457_Rtk (copy).txt"

    parsed = parse_nmea_file(file_path)

    for row in parsed[:]:
        # if row["source"] == "GGA":
        logging.debug((row))

    # save_csv(parsed)
