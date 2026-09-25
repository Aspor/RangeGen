#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordinate-system transforms and NMEA helpers for LiDAR/GPS data.

Provides the pyproj transformers used to convert between WGS84 lat/lon and the
local metric frames (ETRS-TM35FIN and UTM) used by the pipeline, plus small
helpers to parse NMEA GGA sentences into decimal lat/lon/altitude.

EPSG codes used
---------------
4326    WGS84 (GPS lat/lon)
3067    ETRS89 / TM35FIN (metric coordinates, Finland)
32633   WGS84 / UTM zone 33N
"""

from pyproj import CRS, Transformer
import logging

# ￼epsg.io to search coordinate systems
# 4326 is WGS84 ie. GPS position
# 3067 is ETHR89 ie metric coordinates in Finland

transformerWGS_84_to_ETRS89_TM35FIN_E_N = Transformer.from_crs(4326, 3067)
transformerETRS89_TM35FIN_E_N_to_WGS_84 = Transformer.from_crs(3067, 4326)

transformerWGS_84_to_UTM35 = Transformer.from_crs(4326, 32633)
# 32633
# 32635
# +proj=utm +zone=35 +datum=WGS84 +units=m +no_defs


def WGS84lalo_to_ETRSTM35FINxy(lat, lon):
    """Convert WGS84 lat/lon to ETRS-TM35FIN easting/northing.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.

    Returns
    -------
    tuple[float, float]
        ``(easting, northing)`` in metres.
    """
    # X = EASTING , y = NORHING
    return transformerWGS_84_to_ETRS89_TM35FIN_E_N.transform(lat, lon)


def WGS84lalo_to_UTM35(lat, lon):
    """Convert WGS84 lat/lon to the configured UTM projection (EPSG:32633).

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.

    Returns
    -------
    tuple[float, float]
        ``(easting, northing)`` in metres.
    """
    return transformerWGS_84_to_UTM35.transform(lat, lon)


def ETRSTM35FINxy_to_WGS84lalo(easting, northing):
    """Convert ETRS-TM35FIN easting/northing back to WGS84 lat/lon.

    Parameters
    ----------
    easting : float
        Easting in metres.
    northing : float
        Northing in metres.

    Returns
    -------
    tuple[float, float]
        ``(latitude, longitude)`` in degrees.
    """
    return transformerETRS89_TM35FIN_E_N_to_WGS_84.transform(easting, northing)



# --- Define CRS ---
wgs84 = CRS.from_epsg(4326)

custom_tm = CRS.from_proj4(
    "+proj=tmerc +lat_0=0 +lon_0=29 +k=1 "
    "+x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
)

# Transformers
to_tm = Transformer.from_crs(wgs84, custom_tm, always_xy=True)
to_wgs = Transformer.from_crs(custom_tm, wgs84, always_xy=True)


# --- NMEA parsing ---
def nmea_to_decimal(coord, direction):
    """Convert an NMEA degree/minute coordinate to signed decimal degrees.

    Parameters
    ----------
    coord : str
        Coordinate in ``DDMM.MMMMM`` (or ``DDDMM.MMMMM``) format.
    direction : str
        Hemisphere, one of ``"N"``, ``"S"``, ``"E"``, ``"W"``.

    Returns
    -------
    float
        Decimal degrees, negative for ``"S"``/``"W"``.
    """
    # coord format: DDMM.MMMMM
    deg = int(coord[: 2 if direction in ["N", "S"] else 3])
    minutes = float(coord[2 if direction in ["N", "S"] else 3 :])
    value = deg + minutes / 60.0
    return -value if direction in ["S", "W"] else value


def parse_gga(line):
    """Parse an NMEA ``$GPGGA`` sentence into latitude, longitude, and altitude.

    Parameters
    ----------
    line : str
        A single NMEA GGA sentence.

    Returns
    -------
    tuple[float, float, float]
        ``(latitude, longitude, altitude)`` with lat/lon in degrees and
        altitude in metres.
    """
    parts = line.split(",")
    lat = nmea_to_decimal(parts[2], parts[3])
    lon = nmea_to_decimal(parts[4], parts[5])
    alt = float(parts[9])
    return lat, lon, alt
