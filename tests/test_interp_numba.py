"""
Differential tests of the numba backend of hygeos_core.interpolate against the numpy backend

(kept apart from test_interp.py, whose autouse fixture runs every test with each
backend: here, both backends are used explicitly in each test)
"""

import inspect

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("numba")  # optional dependency

from hygeos_core.interpolate import Interpolator, Linear, Nearest, Spline, interp
from hygeos_core.tests.pytest_utils import parametrize_dict


def random_coords(rng, vmin, vmax, dims, chunk=False):
    sizes = {"x": 10, "y": 15, "z": 20}
    da = xr.DataArray(
        vmin + rng.random([sizes[d] for d in dims]) * (vmax - vmin), dims=dims
    )
    return da.chunk({d: 7 for d in dims}) if chunk else da


# interpolation kwargs, as a function of the interpolation class (Linear or Spline),
# a random generator and whether the coordinates are dask-backed
CASES = {
    "same_dims": lambda K, rng, ch: dict(
        a=K(random_coords(rng, 1, 4, ["x", "y"], ch)),
        b=K(random_coords(rng, 11, 14, ["x", "y"], ch)),
        c=K(random_coords(rng, 101, 105, ["x", "y"], ch)),
    ),
    "mixed_dims_nearest": lambda K, rng, ch: dict(
        a=Nearest(random_coords(rng, 1, 4, ["z"], ch).round(), tolerance=1.),
        b=K(random_coords(rng, 11, 14, ["x", "y"], ch)),
        c=K(random_coords(rng, 101, 105, ["x", "z"], ch)),
    ),
    "scalar_selection": lambda K, rng, ch: dict(
        a=Nearest(2),
        b=K(random_coords(rng, 11, 14, ["x", "y"], ch)),
        c=K(random_coords(rng, 101, 105, ["x", "z"], ch)),
    ),
    "missing_dim": lambda K, rng, ch: dict(
        b=K(random_coords(rng, 11, 14, ["x", "y"], ch)),
        c=K(random_coords(rng, 101, 105, ["x", "z"], ch)),
    ),
    "oob_nan_clip": lambda K, rng, ch: dict(
        a=K(random_coords(rng, 0, 5, ["x", "y"], ch), bounds="nan"),
        c=K(random_coords(rng, 99, 107, ["x", "z"], ch), bounds="clip"),
    ),
    "irregular": lambda K, rng, ch: dict(
        a=K(random_coords(rng, 1, 4, ["x", "y"], ch), spacing="irregular"),
        b=K(random_coords(rng, 11, 14, ["x"], ch), spacing="irregular"),
    ),
    "nearest_only": lambda K, rng, ch: dict(  # no weights: numpy fallback
        a=Nearest(random_coords(rng, 1, 4, ["x", "y"], ch).round(), tolerance=1.),
    ),
}


def set_default_parallel(monkeypatch, parallel: bool):
    """
    `interp` does not expose the `parallel` argument of Interpolator: change its
    default value
    """
    init = Interpolator.__init__
    names = [
        p.name for p in inspect.signature(init).parameters.values()
        if p.default is not inspect.Parameter.empty
    ]
    defaults = list(init.__defaults__)
    defaults[names.index("parallel")] = parallel
    monkeypatch.setattr(init, "__defaults__", tuple(defaults))


@pytest.mark.parametrize("parallel", **parametrize_dict({
    "serial": False,
    "parallel": True,
}))
@pytest.mark.parametrize("chunk", **parametrize_dict({
    "numpy_coords": False,
    "dask_coords": True,
}))
@pytest.mark.parametrize("dtype,nan_in_data", **parametrize_dict({
    "float64": ("float64", False),
    "float32": ("float32", False),
    "int32": ("int32", False),
    "float64_nan": ("float64", True),
    "float32_nan": ("float32", True),
}))
@pytest.mark.parametrize("kind", [Linear, Spline])
@pytest.mark.parametrize("case", CASES)
def test_numba_vs_numpy(monkeypatch, case, kind, dtype, nan_in_data, chunk,
                        parallel):
    """
    The numba backend gives the same results as the numpy backend (values, dims,
    coords and dtype) on random data and coordinates
    """
    rng = np.random.default_rng(0)
    values = rng.random((3, 4, 5)) * 100
    if nan_in_data:
        values[1, 2, :] = np.nan
    data = xr.DataArray(
        values.astype(dtype),
        dims=["a", "b", "c"],
        coords={
            "a": [1.0, 2.0, 4.0],
            "b": [11.0, 12.0, 13.0, 14.0],
            "c": [105.0, 104.0, 103.0, 102.0, 101.0],
        },
    )

    # same coordinates for both backends
    kwargs = CASES[case](kind, rng, chunk)
    ref = interp(data, backend="numpy", **kwargs).compute()

    set_default_parallel(monkeypatch, parallel)
    res = interp(data, backend="numba", **kwargs).compute()

    assert res.dims == ref.dims
    assert res.dtype == ref.dtype
    assert set(res.coords) == set(ref.coords)
    for d in ref.coords:
        np.testing.assert_array_equal(res.coords[d].values, ref.coords[d].values)
    np.testing.assert_allclose(res.values, ref.values, rtol=1e-6, equal_nan=True)


def test_default_backend_auto():
    """
    The default backend is "auto": numba when it is installed, and it is
    stored in the attributes of the interpolated variables
    """
    from hygeos_core.interpolate import Interpolator, resolve_backend

    assert resolve_backend("auto") == "numba"
    data = xr.DataArray(
        np.arange(10, dtype="float32"),
        dims=["x"],
        coords={"x": np.arange(10.)},
        attrs={"units": "m"},
    )
    res = interp(data, x=Linear(xr.DataArray([1.5, 2.5], dims=["y"])))
    assert res.attrs["interpolation_backend"] == "numba"
    assert res.attrs["units"] == "m"
    assert Interpolator(data.to_dataset(name="a"), x=Linear("x")).backend == "numba"
    res = interp(data, backend="numpy", x=Linear(xr.DataArray([1.5], dims=["y"])))
    assert res.attrs["interpolation_backend"] == "numpy"
