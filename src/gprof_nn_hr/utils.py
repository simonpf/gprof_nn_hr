"""
gprof_nn_hr.utils
=================

A collection of utility functions for the 'gprof_nn_hr' package.
"""
from typing import Union

import numpy as np


def great_circle_distance(
        lats_1: Union[np.ndarray, float],
        lons_1: Union[np.ndarray, float],
        lats_2: Union[np.ndarray, float],
        lons_2: Union[np.ndarray, float],
):
    """
    Approximate distance between locations on earth.

    Uses haversine formulat with an earth radius of 6 371 km.

    Args:
        lats_1: Latitude coordinates of starting points.
        lons_1: Longitude coordinates of starting points.
        lats_2: Latitude coordinates of target points.
        lons_2: Longitude coordinates of target points.

    Return:
        The distance between the points described by the input
        arrays in m.
    """
    lats_1 = np.deg2rad(lats_1)
    lons_1 = np.deg2rad(lons_1)
    lats_2 = np.deg2rad(lats_2)
    lons_2 = np.deg2rad(lons_2)

    d_lons = lons_2 - lons_1
    d_lats = lats_2 - lats_1

    a = np.sin(d_lats / 2.0) ** 2 + np.cos(lats_1) * np.cos(lats_2) * np.sin(
        d_lons / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371e3
    return R * c
