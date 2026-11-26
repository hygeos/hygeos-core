import pytest

import numpy as np
import xarray as xr

from hygeos_core.geo.convert import convert_latlon_2D, center_longitude


def test_convert_latlon():
    lat = np.linspace(-10,20,300)
    lon = np.linspace(-30,30,600)

    lat, lon = convert_latlon_2D(lat, lon)
    
    assert lat.shape == lon.shape