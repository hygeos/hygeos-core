        
from hygeos_core.geo.naming import names


def test_names():
    
    # usage
    names.lat
    names.lat.attrs['desc']
    names.lat.attrs['minv']
    names.lat.attrs['maxv']
    
    # theses attributes should not have any max or min values
    assert names.rows.attrs.get('minv') is None
    assert names.rows.attrs.get('maxv') is None
    assert names.columns.attrs.get('minv') is None
    assert names.columns.attrs.get('maxv') is None
    
    assert isinstance(names.lon.attrs['minv'], (int, float))
    assert isinstance(names.lon.attrs['maxv'], (int, float))
    assert names.lon.attrs['minv'] < names.lon.attrs['maxv']
