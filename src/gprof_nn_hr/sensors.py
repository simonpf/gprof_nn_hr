"""
gprof_nn_hr.sensors
===================

Defines the sensors supported by the high-resolution GPROF retrieval.
"""
from pathlib import Path
from typing import Callable

import numpy as np
from pansat.products import Product
from pansat.products.satellite.gpm import l1c_xcal2026v_gosatgw_amsr3_v08a
from pyresample import kd_tree
from pyresample.geometry import SwathDefinition
import xarray as xr


class Sensor:
    """
    A passive microwave sensor and the means to read its L1C data.

    Attributes:
        pansat_product: The pansat product representing the sensor's GPM L1C
            product.
        read_l1c: Function that reads an L1C file and interpolates its
            observations to a common swath.
    """

    def __init__(
        self,
        pansat_product: Product,
        read_l1c: Callable[[Path], xr.Dataset],
    ):
        """
        Args:
            pansat_product: The pansat product representing the sensor's GPM
                L1C product.
            read_l1c: Function that takes the path of an L1C file and returns
                an xarray.Dataset containing the file's observations
                interpolated to a common swath.
        """
        self.pansat_product = pansat_product
        self.read_l1c = read_l1c


def _mask_fill_values(var: xr.Variable) -> np.ndarray:
    """
    Replace the fill values of a variable, which pansat doesn't mask, with NaN.
    """
    values = var.data
    fill_value = var.attrs.get("_FillValue")
    if fill_value is None:
        return values
    return np.where(values == fill_value, np.nan, values).astype(values.dtype)


def remap_swaths(
    data: xr.Dataset,
    target_swath: int,
    radius_of_influence: float = 10e3,
    scan_margin: int = 50,
) -> xr.Dataset:
    """
    Map the observations from all swaths of an L1C file to the pixels of a
    single swath using nearest-neighbor interpolation.

    The swaths of a sensor are observed by different feedhorns, so pixels with
    the same scan and pixel index can be several tens of kilometers apart.
    The remapping is therefore based on the geolocation of each pixel.

    Because the beginning and end of an orbit overlap, the remapping is
    performed separately for the first and the second half of the target
    scans. Each half only uses source scans from the same half extended by
    'scan_margin' scans, which prevents target pixels at the beginning of the
    orbit from being matched with source pixels at its end and vice versa.

    Args:
        data: An xarray.Dataset containing L1C data as loaded by pansat, i.e.,
            with the variables of swath 'i' suffixed with '_s<i>'.
        target_swath: The index of the swath to map the observations to.
        radius_of_influence: The maximum distance in meters between a target
            pixel and its nearest neighbor. Target pixels without a neighbor
            within this distance, or whose nearest neighbor is missing, are
            set to NaN.
        scan_margin: The number of source scans from the other half of the
            orbit to include in the remapping of each half. Must be larger
            than the along-track offset between the swaths in scans.

    Return:
        An xarray.Dataset with dimensions 'scans' and 'pixels' containing the
        latitude and longitude of the target swath and all swath-specific
        variables mapped to the target swath.
    """
    lats_t = _mask_fill_values(data[f"latitude_s{target_swath}"].variable)
    lons_t = _mask_fill_values(data[f"longitude_s{target_swath}"].variable)

    n_scans = lats_t.shape[0]
    half = n_scans // 2
    # Pairs of target scans and the source scans used to remap them.
    halves = [
        (slice(0, half), slice(0, min(half + scan_margin, n_scans))),
        (slice(half, n_scans), slice(max(half - scan_margin, 0), n_scans)),
    ]

    remapped = {
        "latitude": (("scans", "pixels"), lats_t),
        "longitude": (("scans", "pixels"), lons_t),
    }
    for name, var in data.variables.items():
        if var.dims == ("scans",) and name != "scans":
            remapped[name] = (("scans",), _mask_fill_values(var))

    swaths = sorted(
        int(name[len("latitude_s"):])
        for name in data.variables
        if name.startswith("latitude_s")
    )
    for swath in swaths:
        lats = _mask_fill_values(data[f"latitude_s{swath}"].variable)
        lons = _mask_fill_values(data[f"longitude_s{swath}"].variable)
        swath_vars = {
            name: var
            for name, var in data.data_vars.items()
            if name.endswith(f"_s{swath}")
            and var.dims[:2] == data[f"latitude_s{swath}"].dims
        }
        results = {
            name: np.full(lats_t.shape + var.shape[2:], np.nan, dtype=var.dtype)
            for name, var in swath_vars.items()
        }

        for target_scans, source_scans in halves:
            target = SwathDefinition(
                lons=lons_t[target_scans], lats=lats_t[target_scans]
            )
            source = SwathDefinition(
                lons=lons[source_scans], lats=lats[source_scans]
            )
            valid_in, valid_out, inds, _ = kd_tree.get_neighbour_info(
                source, target, radius_of_influence=radius_of_influence, neighbours=1
            )
            for name, var in swath_vars.items():
                results[name][target_scans] = kd_tree.get_sample_from_neighbour_info(
                    "nn",
                    target.shape,
                    _mask_fill_values(var.variable)[source_scans],
                    valid_in,
                    valid_out,
                    inds,
                    fill_value=np.nan,
                )

        for name, var in swath_vars.items():
            remapped[name] = (("scans", "pixels") + var.dims[2:], results[name])

    return xr.Dataset(remapped)


def read_l1c_amsr3(path: Path) -> xr.Dataset:
    """
    Read AMSR3 L1C observations and map them to the pixels of swath 8.

    Args:
        path: Path to a GOSAT-GW AMSR3 L1C file.

    Return:
        An xarray.Dataset containing the brightness temperatures and incidence
        angles from all swaths mapped to the pixels of swath 8.
    """
    data = l1c_xcal2026v_gosatgw_amsr3_v08a.open(Path(path))
    return remap_swaths(data, target_swath=8)


AMSR3 = Sensor(l1c_xcal2026v_gosatgw_amsr3_v08a, read_l1c_amsr3)
