from hygeos_core.geo.naming import names
from typing import Literal

import dask.array as da 
import xarray as xr


def convert_latlon_2D(lat: da.array, lon: da.array) -> tuple:
    """
    Convert latitude and longitude vectors into 2D representation
    """
    
    lat_size = lat.shape
    lon_size = lon.shape
    assert len(lat_size) == 1 and len(lon_size) == 1, \
    "Latitude and longitude should be 1-dimensional"
    
    return da.meshgrid(lon, lat)

def center_longitude(ds: xr.Dataset, center: Literal[0, 180]=0, lon_name: str=names.lon):
    """
    Center longitudes from [0, 360] to [-180, 180] or from [-180, 180] to [0, 360]
    """
    
    assert (center == 0.0) or (center == 180.0)
    
    lon = None
    if center == 0.0:
        lon = (ds[lon_name].values + 180) % 360 - 180
    elif center == 180.0:
        lon = (ds[lon_name].values) % 360
    
    ds = ds.assign_coords({lon_name:lon})
    ds = ds.sortby(lon_name)
    
    return ds