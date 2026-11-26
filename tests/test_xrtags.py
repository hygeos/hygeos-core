import numpy as np
import xarray as xr

from hygeos_core.xrtags import tag_add, tag_filter


def _roundtrip(ds, tmp_path):
    p = tmp_path / 'ds.nc'
    ds.to_netcdf(p)
    with xr.open_dataset(p) as d:
        return d.load()


def test_tags_after_netcdf_roundtrip(tmp_path):
    ds = xr.Dataset({
        'a': ('x', np.zeros(3)),
        'b': ('x', np.zeros(3)),
    })
    tag_add(ds['a'], 'level2')
    tag_add(ds['b'], 'level')
    ds = _roundtrip(ds, tmp_path)
    assert isinstance(ds['a'].attrs['tags'], str)  # single tag becomes a str

    # substring must not match
    assert list(tag_filter(ds, 'level')) == ['b']
    assert list(tag_filter(ds, 'level2')) == ['a']

    # adding a tag to a str-valued attribute
    tag_add(ds['a'], 'ancillary')
    assert sorted(ds['a'].attrs['tags']) == ['ancillary', 'level2']


def test_tag_add_filter():
    ds = xr.Dataset({'a': ('x', np.zeros(3)), 'b': ('x', np.zeros(3))})
    tag_add(ds['a'], ['level2', 'debug'])
    assert list(tag_filter(ds, ['debug'])) == ['a']
    assert list(tag_filter(ds, 'other')) == []
