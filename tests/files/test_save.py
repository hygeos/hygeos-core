
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from hygeos_core.files.save import to_netcdf, to_zarr
import xarray as xr


def ds_example(): return xr.Dataset({'test': range(10)})
def da_example(): return xr.DataArray(range(10))


def test_to_netcdf():
    with TemporaryDirectory() as tmpdir:
        to_netcdf(ds_example(), Path(tmpdir)/'test.nc')

def test_to_netcdf_format_check(tmpdir):
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            to_netcdf('inp', Path(tmpdir)/'test.nc')

def test_to_netcdf_format_check_da(tmpdir):
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            to_netcdf(da_example(), Path(tmpdir)/'test.nc')


def test_to_zarr():
    with TemporaryDirectory() as tmpdir:
        to_zarr(ds_example(), Path(tmpdir)/'test.zarr')

def test_to_zarr_format_check(tmpdir):
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            to_zarr('inp', Path(tmpdir)/'test.zarr')

def test_to_zarr_format_check_da(tmpdir):
    with TemporaryDirectory() as tmpdir:
        with pytest.raises(AssertionError):
            to_zarr(da_example(), Path(tmpdir)/'test.zarr')


@pytest.mark.parametrize("versions,attr", [
    (["pytest"], "pytest_version"),
    ({"test_runner": "pytest"}, "test_runner_version"),
])
def test_to_netcdf_versions(versions, attr):
    with TemporaryDirectory() as tmpdir:
        to_netcdf(ds_example(), Path(tmpdir)/'test.nc', versions=versions)
        attrs = xr.open_dataset(Path(tmpdir)/'test.nc').attrs
    assert attrs[attr] == pytest.__version__


def test_to_netcdf_git_commit_deprecated():
    with TemporaryDirectory() as tmpdir:
        to_netcdf(ds_example(), Path(tmpdir)/'test1.nc')
        assert 'git_commit' not in xr.open_dataset(Path(tmpdir)/'test1.nc').attrs
        with pytest.warns(DeprecationWarning):
            to_netcdf(ds_example(), Path(tmpdir)/'test2.nc', git_comit=True)
        assert 'git_commit' in xr.open_dataset(Path(tmpdir)/'test2.nc').attrs


def test_to_zarr_versions():
    with TemporaryDirectory() as tmpdir:
        to_zarr(ds_example(), Path(tmpdir)/'test.zarr', versions=["pytest"])
        attrs = xr.open_zarr(Path(tmpdir)/'test.zarr').attrs
    assert attrs["pytest_version"] == pytest.__version__