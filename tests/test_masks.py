from hygeos_core.masks import gen_bitmask, explicit_bitmask
from numpy import random

import xarray as xr
import pytest

@pytest.fixture
def masks(): 
    return [xr.DataArray(random.randint(0,2,(20,20),dtype=bool), name=f'Mask{i}') 
            for i in range(3)]

@pytest.mark.parametrize("index", [None, [0,1,3]])
def test_bitmask_conversion(masks, index):
    bitmask = gen_bitmask(*masks, bit_index=index)
    retrieve = explicit_bitmask(bitmask)
    assert all((retrieve[m.name] == m).all() for m in masks)