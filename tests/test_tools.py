import pytest
import numpy as np
import xarray as xr
from dask import array as dask_arr
from hygeos_core.tools import (
    locate,
    reglob,
    xr_filter_decorator,
    xr_unfilter,
    xr_filter,
    is_numpy_backed,
)


@pytest.mark.parametrize("regexp", [".*.py", "[a-z]+.*"])
def test_reglob(regexp):
    p = "hygeos_core"
    l = reglob(p, regexp)
    assert len(l) != 0


def sample_dataset(with_xy_coords: bool):
    coords = {"bands": [400, 500, 600]}
    if with_xy_coords:
        coords["x"] = [0, 1, 2, 3]
        coords["y"] = [0, 1, 2, 3, 4]

    return xr.Dataset(
        {
            "A": (("x", "y"), np.random.rand(4, 5)),
            "B": (("bands", "x", "y"), np.ones((3, 4, 5), dtype="int")),
        },
        coords=coords,
    )


@pytest.mark.parametrize("with_xy_coords", [True, False])
def test_xr_filter_unfilter(with_xy_coords: bool):
    ds = sample_dataset(with_xy_coords)

    ok = ds.A < 0.5

    # Extract valid pixels
    sub = xr_filter(ds, ok)

    # Reformat sub to full array
    full = xr_unfilter(sub, ok)

    # Check that the coords are preserved
    assert len(ds.coords) == len(full.coords)


@pytest.mark.parametrize("transparent", [True, False])
def test_xr_filter_unfilter_passthrough_preserves_original(transparent: bool):
    """Pass-through variables (present in both sub and ds_original) should preserve
    their original values where condition is False, instead of being overwritten
    with fill values."""
    ds = sample_dataset(True)
    # Add a pass-through integer variable (e.g. flags) with known values
    ds = ds.assign(flags=xr.DataArray(np.arange(20, dtype="uint16").reshape(4, 5), dims=("x", "y")))

    ok = ds.A < 0.5

    # Filter
    sub = xr_filter(ds, ok, transparent=transparent)

    # Unfilter WITHOUT ds_original: pass-through vars get fill_value_int (0)
    full_no_orig = xr_unfilter(sub, ok, transparent=transparent)
    # flags where ~ok should be 0 (fill_value_int)
    assert (full_no_orig.flags.values[~ok.values] == 0).all()

    # Unfilter WITH ds_original: pass-through vars preserve original values
    full_with_orig = xr_unfilter(sub, ok, transparent=transparent, ds_original=ds)
    # flags where ~ok should match original
    np.testing.assert_array_equal(
        full_with_orig.flags.values[~ok.values],
        ds.flags.values[~ok.values],
    )
    # flags where ok should come from sub (which is same as original for pass-through)
    np.testing.assert_array_equal(
        full_with_orig.flags.values[ok.values],
        ds.flags.values[ok.values],
    )
    # Dimension order should be preserved
    assert full_with_orig.flags.dims == ds.flags.dims


@pytest.mark.parametrize("with_xy_coords", [True, False])
def test_xr_unfilter_transparent_preserves_dim_order(with_xy_coords: bool):
    """xr_unfilter in transparent mode should preserve the original dimension order."""
    ds = sample_dataset(with_xy_coords)
    ds = ds.transpose('x', 'y', 'bands')

    # Create a condition with non-trivial dimension order (bands, x, y)
    ok = ds.A > 0

    # Filter with transparent mode activated
    sub = xr_filter(ds, ok, transparent=True)
    sub = sub * 2.0

    # Unfilter should restore original dimension order
    full = xr_unfilter(sub, ok, transparent=True)

    for var in ds.data_vars:
        if var in full.data_vars:
            assert full[var].dims == ds[var].dims, f"Dimension order mismatch for {var}"


@pytest.mark.parametrize("transparent", [True, False])
@pytest.mark.parametrize("with_xy_coords", [True, False])
def test_xr_filter_decorator(transparent: bool, with_xy_coords: bool):
    def myfunc(ds: xr.Dataset) -> xr.Dataset:
        return xr.merge(
            [
                ds.A.rename("A1"),
                ds.B.rename("B1"),
            ]
        )

    ds = sample_dataset(with_xy_coords)

    res = xr_filter_decorator(
        0,
        lambda x: x.A < 0.5,
        fill_value_int=-1,
        transparent=transparent,
    )(myfunc)(ds)
    print(res)

    # Check that the coords are preserved
    assert len(ds.coords) == len(res.coords)


@pytest.mark.parametrize("transparent", [True, False])
def test_xr_filter_decorator_passthrough_preserves_original(transparent: bool):
    """Pass-through variables should preserve original values where condition is False
    when using xr_filter_decorator."""
    ds = sample_dataset(True)
    ds = ds.assign(flags=xr.DataArray(np.arange(20, dtype="uint16").reshape(4, 5), dims=("x", "y")))

    ok = ds.A < 0.5
    original_flags = ds.flags.values.copy()

    def myfunc_inplace(ds: xr.Dataset) -> None:
        ds["C"] = ds.A * 2

    xr_filter_decorator(
        0,
        lambda x: x.A < 0.5,
        fill_value_int=65535,
        transparent=transparent,
    )(myfunc_inplace)(ds)

    # flags is a pass-through variable: original values should be preserved where ~ok
    np.testing.assert_array_equal(
        ds.flags.values[~ok.values],
        original_flags[~ok.values],
    )
    # Dimension order should be preserved
    assert ds.flags.dims == ("x", "y")


def test_xr_filter_transparent_preserves_dim_order():
    """xr_filter in transparent mode should retain the order of dimensions of the initial array."""
    # Create a dataset where a variable has dimensions in a specific order ("y", "x")
    ds = xr.Dataset(
        {
            "A": (("y", "x"), np.random.rand(5, 4)),
            "B": (("y", "x", "bands"), np.ones((5, 4, 3), dtype="int")),
        },
        coords={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3, 4], "bands": [400, 500, 600]},
    )

    # Apply filter in transparent mode
    sub = xr_filter(ds, ds.A < 0.5, transparent=True)

    # The dimension order should be preserved from the original dataset
    assert sub.A.dims == ds.A.dims, f"Expected {ds.A.dims}, got {sub.A.dims}"
    assert sub.B.dims == ds.B.dims, f"Expected {ds.B.dims}, got {sub.B.dims}"


def test_locate():
    lat_min = 0
    lat_max = 10
    lon_min = 0
    lon_max = 10
    lat, lon = xr.broadcast(
        xr.DataArray(np.linspace(lat_min, lat_max, 100), dims=["lat"]),
        xr.DataArray(np.linspace(lon_min, lon_max, 100), dims=["lon"]),
    )
    locate(lat, lon, 5.0, 5.0)
    locate(lat, lon, 15.0, 15.0)

    locate(lat, lon, 5.0, 5.0, dist_min_km=10)
    with pytest.raises(ValueError):
        locate(lat, lon, 15.0, 15.0, dist_min_km=10)


@pytest.mark.parametrize(
    "obj,expected",
    [
        ("numpy_da", True),
        ("dask_da", False),
        ("numpy_ds", True),
        ("dask_ds", False),
        ("mixed_ds", False),
        ("numpy_ds_coords", True),
    ],
)
def test_is_numpy_backed(obj, expected):
    numpy_da = xr.DataArray(np.array([1, 2, 3]))
    dask_da = xr.DataArray(dask_arr.from_array(np.array([1, 2, 3])))
    numpy_ds = xr.Dataset({'a': (('x',), np.array([1, 2, 3]))})
    dask_ds = xr.Dataset({'a': (('x',), dask_arr.from_array(np.array([1, 2, 3])))})
    mixed_ds = xr.Dataset({
        'a': (('x',), np.array([1, 2, 3])),
        'b': (('x',), dask_arr.from_array(np.array([4, 5, 6]))),
    })
    numpy_ds_coords = xr.Dataset(
        {'a': (('x',), np.array([1, 2, 3]))},
        coords={'x': np.array([0, 1, 2])},
    )
    fixtures = {
        "numpy_da": numpy_da,
        "dask_da": dask_da,
        "numpy_ds": numpy_ds,
        "dask_ds": dask_ds,
        "mixed_ds": mixed_ds,
        "numpy_ds_coords": numpy_ds_coords,
    }
    assert is_numpy_backed(fixtures[obj]) is expected