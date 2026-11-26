from matplotlib import pyplot as plt
import importlib.util
import numpy as np
import pytest
from typing import Callable

from hygeos_core.monitor import Chrono

from hygeos_core.interpolate import (
    Index,
    Interpolator,
    Linear_Indexer,
    Nearest_Indexer,
    Spline_Indexer,
    align_lists,
    find_indices,
    interp,
    Linear,
    Nearest,
    Spline,
)
from tests.test_blockwise import create_ancillary_dataset, create_sample_dataset
try:
    from luts import luts 
except ImportError:
    pass

from hygeos_core.tests import conftest
import xarray as xr
from hygeos_core.tests.pytest_utils import parametrize_dict

# numba is an optional dependency: skip the numba backend tests if not installed
HAS_NUMBA = importlib.util.find_spec("numba") is not None
NUMBA = pytest.param(
    "numba", marks=pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
)

# scipy is an optional dependency: skip the scipy backend tests if not installed
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
SCIPY = pytest.param(
    "scipy", marks=pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
)


@pytest.fixture(autouse=True, params=["numpy", NUMBA])
def default_backend(request, monkeypatch):
    """
    Run all tests with each backend of hygeos_core.interpolate, by changing the default
    value of the `backend` argument
    """
    import inspect
    import hygeos_core.interpolate as ci
    for func in [
        ci.Interpolator.__init__, ci.interp,
        ci.Linear.get_indexer, ci.Spline.get_indexer,
        ci.Linear_Indexer.__init__, ci.Spline_Indexer.__init__,
        ci.create_locator, ci.Locator.__init__, ci.Locator_Regular.__init__,
    ]:
        names = [
            p.name for p in inspect.signature(func).parameters.values()
            if p.default is not inspect.Parameter.empty
        ]
        defaults = list(func.__defaults__)
        defaults[names.index("backend")] = request.param
        monkeypatch.setattr(func, "__defaults__", tuple(defaults))
    return request.param


list_interp_functions = {
    "xarray": lambda data, l1: data.interp(
        lat=l1.latitude,
        lon=l1.longitude,
    ),
    # "lut": lambda data, l1: xr.DataArray(
    #     luts.from_xarray(data)[
    #         luts.Idx(l1.latitude.values), luts.Idx(l1.longitude.values)
    #     ]
    # ),
    "interp": lambda data, l1: interp(
        data,
        lat=Linear(l1.latitude),
        lon=Linear(l1.longitude),
    ),
    "interp_non_regular": lambda data, l1: interp(
        data,
        lat=Linear(l1.latitude, spacing='irregular'),
        lon=Linear(l1.longitude, spacing='irregular'),
    ),
}


@pytest.mark.parametrize("reverse", **parametrize_dict({
    "increasing": False,
    "decreasing": True,
}))
@pytest.mark.parametrize("coords", **parametrize_dict({
    'regular': np.linspace(10, 15, 6),
    'irregular': np.array([10., 11., 12., 13., 15.]),
}))
@pytest.mark.parametrize("values,oob", **parametrize_dict({
    "scalar": (np.array(11.2), False),
    "scalar-edge1": (np.array(10.), False),
    "scalar-edge2": (np.array(15.), False),
    "scalar-oob": (np.array(9.), True),
    "array": (np.array([np.nan, 9., 10., 10.5, 14.5, 15., 16.]), True),
    "array2D": (np.array([[np.nan, 9., 10.], [14.5, 15., 16.]]), True),
}))
@pytest.mark.parametrize("indexer_factory", **parametrize_dict({ 
    'nearest': lambda c: Nearest_Indexer(c, tolerance=3),
    'linear_nan': lambda c: Linear_Indexer(c, bounds="nan", spacing="auto"),
    'linear_error': lambda c: Linear_Indexer(c, bounds="error", spacing="auto"),
    'linear_clip': lambda c: Linear_Indexer(c, bounds="clip", spacing="auto"),
    'linear_cycle': lambda c: Linear_Indexer(c, bounds="cycle", spacing="auto"),
    'spline_nan': lambda c: Spline_Indexer(c, bounds="nan", spacing="auto", tension=0.5),
    'spline_error': lambda c: Spline_Indexer(c, bounds="error", spacing="auto", tension=0.5),
    'spline_clip': lambda c: Spline_Indexer(c, bounds="clip", spacing="auto", tension=0.5),
}))
def test_indexer(indexer_factory, coords, values, oob, reverse):
    """
    Thoroughly test the indexers
    """
    if reverse:
        coords = coords[::-1]

    # instantiate the indexer
    try:
        indexer = indexer_factory(coords)
    except RuntimeError:
        # in case of bounds="cycle" and irregular cooefs
        return
    
    if hasattr(indexer, 'bounds') and (indexer.bounds == "error") and oob:
        with pytest.raises(ValueError):
            indexer(values)
    else:
        for idx, w in indexer(values):
            assert (idx >= 0).all()
            assert (idx < len(coords)).all()
            if (w is not None) and not isinstance(indexer, Spline_Indexer):
                w = np.array(w)
                ok = ~np.isnan(w)
                assert (w[ok] >= 0).all()
                assert (w[ok] <= 1).all()

        # check bracketing values (for scalars - linear only)
        if (not oob) and (values.ndim == 0) and isinstance(indexer, Linear_Indexer):
            i1, w1 = indexer(values)[0]
            i2, w2 = indexer(values)[1]
            v1 = coords[i1]
            v2 = coords[i2]
            assert np.isclose(values, v1*w1 + v2*w2)
            if indexer.bounds != "cycle":
                assert v1 <= values
                assert v2 >= values


def test_indexer_time():
    indexer = Linear_Indexer(
        np.datetime64('2025-01-01') + np.arange(10)*np.timedelta64(1, 'D'),
        bounds="error",
        spacing="auto",
    )
    value = np.array(np.datetime64('2025-01-02 10:30'))
    indexer(value)


def test_cycle_period():
    coords = np.linspace(-180, 179, 360)
    Linear_Indexer(coords, bounds="cycle", spacing="auto", period=360)
    with pytest.raises(AssertionError):
        Linear_Indexer(coords, bounds="cycle", spacing="auto", period=359)


@pytest.mark.parametrize("apply_function", **parametrize_dict(list_interp_functions))
@pytest.mark.parametrize("mode", [
    'numpy',
    'dask',
    ])
def test_perf_interp(mode: str, apply_function: Callable, request):
    factor = 1 # 1 for fast, 10 for slower
    l1 = xr.Dataset()
    shp = (500*factor, 500*factor)
    l1['latitude'] = xr.DataArray(np.random.random(shp)*20 - 10, dims=['x', 'y'])
    l1['longitude'] = xr.DataArray(np.random.random(shp)*20 - 10, dims=['x', 'y'])
    
    if mode == "dask":
        # numpy mode: pre-compute the array of coordinates
        l1 = l1.chunk({'x':50*factor, 'y': 50*factor})

    # dummy ancillary data
    shp_anc = (180, 360)
    data = xr.DataArray(
        np.random.random(shp_anc),
        dims=["lat", "lon"],
        coords={
            "lat": np.linspace(-90, 90, shp_anc[0]),
            "lon": np.linspace(-180, 180, shp_anc[1]),
        },
    )

    with Chrono("INST"):
        interpolated = apply_function(data, l1)

    with Chrono("COMP"):
        interpolated.compute()



def sample(vmin: float, vmax: float, dims: list):
    sizes = {
        "x": 10,
        "y": 15,
        "z": 20,
    }
    shp = [sizes[dim] for dim in dims]
    return xr.DataArray(
        vmin + np.random.random(np.prod(shp)).reshape(shp) * (vmax - vmin),
        dims=dims,
        coords={dim: np.linspace(0, 50, shp[i]) for i, dim in enumerate(dims)},
    ).chunk({d: sizes[d] for d in dims})


@pytest.mark.parametrize(
    "kwargs",
    [
        {  # same dimensions
            "a": Linear(sample(1, 3, ["x", "y"])),
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "y"])),
        },
        {  # mixed dimensions
            "a": Nearest(sample(1, 3, ["z"]), tolerance=1.),
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # scalar selection
            "a": Nearest(2),
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "b": Linear(sample(11, 14, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "a": Linear(sample(1, 3, ["x", "y"])),
            "c": Linear(sample(101, 105, ["x", "z"])),
        },
        {  # same dimensions
            "a": Spline(sample(1, 3, ["x", "y"])),
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "y"])),
        },
        {  # mixed dimensions
            "a": Nearest(sample(1, 3, ["z"]), tolerance=1.),
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
        {  # scalar selection
            "a": Nearest(2),
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "b": Spline(sample(11, 14, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
        {  # missing dimension (slice)
            "a": Spline(sample(1, 3, ["x", "y"])),
            "c": Spline(sample(101, 105, ["x", "z"])),
        },
    ],
)
def test_interp(kwargs):
    data = xr.DataArray(
        np.zeros((3, 4, 5), dtype="float32"),
        dims=["a", "b", "c"],
        coords={
            "a": [1.0, 2.0, 4.0],
            "b": [11.0, 12.0, 13.0, 14.0],
            "c": [105.0, 104.0, 103.0, 102.0, 101.0],
        },
    )
    res = interp(data, **kwargs)
    # all dims should have coordinates
    for dim in res.dims:
        assert dim in res.coords
    res.compute()



def test_interp_dataset():
    """
    Test the interpolation of a full dataset with `Interpolator()`
    # TODO: test Interpolator on either Dataset or DataArray
    """
    coords = {
        "a": [1.0, 2.0, 4.0],
        "b": [11.0, 12.0, 13.0, 14.0],
        "c": [105.0, 104.0, 103.0, 102.0, 101.0],
    }
    data = xr.Dataset({
        "var1": (["a"], np.zeros(3, dtype="float32")),
        "var2": (["a", "c"], np.zeros((3, 5), dtype="float32")),
        "var3": (["a", "b", "c"], np.zeros((3, 4, 5), dtype="float32")),
    }, coords=coords)
    
    # Example interpolation kwargs
    kwargs = {
        "a": Linear("a"),
        "b": Linear("b"),
        "c": Linear("c"),
    }
    ds_coords = xr.Dataset({
        "a": sample(1, 3, ["z"]),
        "b": sample(11, 14, ["x", "y"]),
        "c": sample(101, 105, ["x", "z"]),
    })
    res = Interpolator(data, **kwargs).map_blocks(ds_coords)
    res.compute()


def test_interp_real_case():
    ds = create_sample_dataset()
    ret = Interpolator(
        create_ancillary_dataset(),
        lat=Linear("latitude"),
        lon=Linear("longitude"),
    ).map_blocks(ds)

    # Check that attributes are preserved
    assert 'units' in ret.wind.attrs
    assert 'units' in ret.ozone.attrs

    ret.compute()


def test_interp_noop():
    ds = create_ancillary_dataset()
    ret = Interpolator(
        ds,
        lat=Linear("lat"),
        lon=Linear("lon"),
    ).map_blocks(ds)
    ret.compute()

def test_interp_commondims():
    """
    Test common dimensions between interpolated coords and interpolating variables
    """
    ds = create_sample_dataset()
    ds = ds.rename(rho_toa='WV', band='pressure', latitude='p')[['WV', 'p']]
    res = Interpolator(ds[['WV']].compute(), pressure=Linear('p', bounds='clip')).map_blocks(ds)
    res.compute()


def test_interp_real_case2():
    '''
    test avec deux ds comportant une même dimension (bands)
    les coordonnées sont perdues (?)
    TODO
    '''
    ds = create_sample_dataset()
    anc = create_ancillary_dataset()
    anc["dummy"] = xr.DataArray(
        np.random.uniform(0, 1, size=(anc.lat.size, anc.lon.size, 3)),
        dims=["lat", "lon", "band"],
        coords={"band": [443, 560, 670]},
    )
    ret = Interpolator(
        anc,
        lat=Linear("latitude"),
        lon=Linear("longitude"),
    ).map_blocks(ds)

    # Check that attributes are preserved
    assert 'units' in ret.wind.attrs
    assert 'units' in ret.ozone.attrs
    assert 'band' in ret.coords

    ret.compute()


@pytest.mark.parametrize('regular', [True, False])
def test_decreasing(request, regular):
    if regular:
        a = [105.0, 104.0, 103.0, 102.0, 101.0]
    else:
        a = [106.0, 104.0, 103.0, 102.0, 101.0]
    data = xr.DataArray(
        np.array(a),
        dims=["a"],
        coords={
            "a": a,
        },
    )
    assert np.isclose(interp(data, a = Linear(xr.DataArray(104.2))),104.2)
    interpolated = interp(
        data,
        a = Linear(xr.DataArray(np.linspace(90, 110, 200), dims=["x"]), bounds="clip"),
    )

    # Plot some interpolated data
    interpolated.plot()
    conftest.savefig(request)


@pytest.fixture(params=[1, 2])
def fixed_sample(request) -> xr.DataArray:
    A = np.arange(10)
    if request.param == 2:
        # irregular grid
        A[1] += 0.1
    return xr.DataArray(
        np.eye(10),
        dims=["a", "b"],
        coords={
            "a": A,
            "b": A,
        },
    )


def test_oob_sel(fixed_sample):
    # Selection failing because of out of bounds
    with pytest.raises(ValueError):
        interp(
            fixed_sample,
            b = Nearest(xr.DataArray([-2]), tolerance=1e-4),
        )


def test_oob_sel_nearest(fixed_sample):
    # This should pass
    interp(
        fixed_sample,
        b = Nearest(xr.DataArray([-2]), tolerance=None)
    )


@pytest.mark.parametrize("kind", [Linear, Spline])
@pytest.mark.parametrize("value", [-2, 10])
def test_oob_interp_error(fixed_sample, kind, value):
    # Interpolation failing because of out of bounds (one call per pytest.raises,
    # otherwise the calls after the first raising one are not executed)
    with pytest.raises(ValueError):
        interp(
            fixed_sample,
            b = kind(xr.DataArray([value])),
        )


def test_oob_interp(fixed_sample):
    # Interpolation at the upper edge (in bounds)
    interp(
        fixed_sample,
        b = Linear(xr.DataArray([9])),
    )
    interp(
        fixed_sample,
        b = Spline(xr.DataArray([9])),
    )


def test_oob_interp_clip(fixed_sample):
    # Passes because clip=True
    interp(
        fixed_sample,
        b=Linear(xr.DataArray([-2]), bounds="clip"),
    )
    interp(
        fixed_sample,
        b=Spline(xr.DataArray([-2]), bounds="clip"),
    )



@pytest.mark.parametrize("A", **parametrize_dict({
    'float_array': np.array([1., 2., 4.]), 
    'int_array': np.array([1, 2, 4]), 
    'int_decreasing': np.array([1, 2, 4])[::-1], 
}))
def test_nearest_indexer(A):
    
    # basic exact indexing
    indexer = Nearest_Indexer(A, 1e-8)
    assert (indexer(A)[0][0] == np.array([0, 1, 2])).all()

    for value in [0, 1.5 , 5.]:
        with pytest.raises(ValueError):
            indexer(value)

    # inexact indexing
    indexer = Nearest_Indexer(A, 0.1)
    assert (indexer(A+0.05)[0][0] == np.array([0, 1, 2])).all()
    assert (indexer(A-0.05)[0][0] == np.array([0, 1, 2])).all()
    

@pytest.mark.parametrize('spacing', ["regular", "irregular"])
@pytest.mark.parametrize('kind', [Linear, Spline])
def test_interp_2D(request, spacing, kind):
    
    A = xr.DataArray(np.eye(3), dims=['x', 'y'])
    N = 100

    interpolated = interp(
        A,
        x=kind(
            xr.DataArray(np.linspace(-1, 3, N), dims=["new_x"]),
            bounds="clip",
            spacing=spacing,
        ),
        y=kind(
            xr.DataArray(np.linspace(-1, 3, N), dims=["new_Y"]),
            bounds="nan",
            spacing=spacing,
        ),
    )

    plt.imshow(interpolated)
    conftest.savefig(request)


@pytest.mark.parametrize("spacing", ["regular", "irregular"])
def test_spline(request, spacing):
    # a spline through constant values is constant
    if spacing == "irregular":
        # irregular coordinates (only valid with spacing="irregular")
        Y = xr.DataArray([1.0, 1.0, 1.0], dims=["X"], coords=[np.array([0,1,3])])
        assert np.isclose(interp(Y, X=Spline(xr.DataArray([0.5]), tension=0.5, spacing=spacing)), 1.0)
        assert np.isclose(interp(Y, X=Spline(xr.DataArray([1.5]), tension=0.5, spacing=spacing)), 1.0)

    Y = xr.DataArray([1.0, 1.0, 1.0, 1.0], dims=["X"], coords=[np.array([0,1,2,3])])
    assert np.isclose(interp(Y, X=Spline(xr.DataArray([0.5]), tension=0.5, spacing=spacing)), 1.0)
    assert np.isclose(interp(Y, X=Spline(xr.DataArray([1.5]), tension=0.5, spacing=spacing)), 1.0)
    assert np.isclose(interp(Y, X=Spline(xr.DataArray([2.5]), tension=0.5, spacing=spacing)), 1.0)
    

def test_inverse_func():
    indexer = Linear_Indexer([0., 1., 4., 9.], bounds="nan", spacing=np.sqrt)
    indexer(0.5)

    
@pytest.mark.parametrize('spacing', ["auto", lambda x: np.sqrt(x)])
def test_nearest_func(request, spacing):
    Y = xr.DataArray([1.0, 1.0, 1.0, 2.0], dims=["X"], coords=[np.array([0,1,4,16])])
    lp = np.linspace(0, 16, 1000)
    Ys = interp(Y, X=Nearest(xr.DataArray(lp), tolerance=None, spacing=spacing))
    plt.plot(Y.X, Y, 'ro')
    plt.plot(lp,Ys, 'b-')
    plt.grid()
    plt.show()
    conftest.savefig(request)
    plt.clf()
    
    
def test_nearest_func_values_invert(request):
    Y = xr.DataArray([1.0, 1.0, 1.0, 2.0], dims=["X"], coords=[np.array([0,1,4,16])])
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([8.9]), tolerance=None, spacing=lambda x: np.sqrt(x))), 1.0)
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([9.1]), tolerance=None, spacing=lambda x: np.sqrt(x))), 2.0)
    
    Y = xr.DataArray([1.0, 1.0, 1.0, 2.0], dims=["X"], coords=[np.array([0,1,4,16])])
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([9.9]), tolerance=None, spacing="auto")), 1.0)
    assert np.isclose(interp(Y, X=Nearest(xr.DataArray([10.1]), tolerance=None, spacing="auto")), 2.0)


@pytest.mark.parametrize("indexer_factory", **parametrize_dict({
    'nearest': lambda x: Nearest(x, tolerance=10),
    'linear_nan': lambda x: Linear(x, bounds="nan"),
    'spline_nan': lambda x: Spline(x, bounds="nan"),
    'linear_clip': lambda x: Linear(x, bounds="clip"),
    'linear_cycle': lambda x: Linear(x, bounds="cycle"),
    'spline_clip': lambda x: Spline(x, bounds="clip"),
}))
def test_interp_1D(request, indexer_factory):
    Y = xr.DataArray(
        [1.0, 2.0, 0.0, 3.0], dims=["X"], coords=[np.array([0.0, 1.0, 2.0, 3.0])]
    )
    Xi = xr.DataArray(np.linspace(-2, 6, 100))
    plt.plot(Y.X, Y, 'ro')
    Ys = interp(Y, X=indexer_factory(Xi))
    plt.plot(Xi.values, Ys.values, '-')
    plt.grid(True)
    conftest.savefig(request)


def test_interp_same_dimensions():
    data = xr.DataArray(
        np.zeros(10), dims=["a"], coords={"a": np.linspace(100, 110, 10)}
    )
    interp(
        data, a=Linear(xr.DataArray([105.0, 104.0, 103.0, 102.0, 101.0], dims=["a"]))
    )


def test_align_lists():
    # test interleaving of several lists
    result = align_lists([["x"], ["x", "z"], ["y", "z"], ["x", "y"]])
    assert result == ["x", "y", "z"]

    with pytest.raises(ValueError):
        align_lists([["x", "y"], ["y", "x"]])


# ---------------------------------------------------------------------------
# find_indices tests
# ---------------------------------------------------------------------------


BACKENDS = [SCIPY, NUMBA]


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_basic(backend):
    """Basic 2D regular grid — indices and distances for known query points."""
    grid = (np.linspace(0, 10, 11), np.linspace(-5, 5, 21))
    # Query points: (dim0, dim1) for each column — use non-exact values to avoid
    # backend discrepancies on exact grid-point matches
    xi = np.array([
        [2.5, 7.3, 0.1, 9.9],
        [-2.3, 3.7, -4.9, 4.9],
    ])
    indices, distances = find_indices(grid, xi, backend=backend)

    assert indices.shape == (2, 4)
    assert distances.shape == (2, 4)

    # Point 0: dim0=2.5 between grid[0][2]=2 and [3]=3 → index=2, dist=0.5
    assert indices[0, 0] == 2
    assert np.isclose(distances[0, 0], 0.5)
    # dim1=-2.3 between grid[1][5]=-2.5 and [6]=-2.0 → index=5, dist=0.4
    assert indices[1, 0] == 5
    assert np.isclose(distances[1, 0], 0.4)

    # Point 1: dim0=7.3 between grid[0][7]=7 and [8]=8 → index=7, dist=0.3
    assert indices[0, 1] == 7
    assert np.isclose(distances[0, 1], 0.3)

    # Point 2: dim0=0.1 between grid[0][0]=0 and [1]=1 → index=0, dist=0.1
    assert indices[0, 2] == 0
    assert np.isclose(distances[0, 2], 0.1)

    # Point 3: dim0=9.9 between grid[0][9]=9 and [10]=10 → index=9, dist=0.9
    assert indices[0, 3] == 9
    assert np.isclose(distances[0, 3], 0.9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_decreasing_grid(backend):
    """Grid coordinates in decreasing order — scipy requires increasing grids."""
    grid = (np.linspace(10, 0, 11)[::-1], np.linspace(5, -5, 11)[::-1])
    xi = np.array([
        [5.0, 5.0],
        [0.0, 0.0],
    ])
    if backend == "scipy":
        # scipy's rgi_cython requires monotonically increasing coordinates
        with pytest.raises(ValueError):
            find_indices(grid, xi, backend=backend)
    else:
        indices, distances = find_indices(grid, xi, backend=backend)
        assert indices.shape == (2, 2)
        assert distances.shape == (2, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_nan_values(backend):
    """NaN query points should produce safe fallback indices and NaN distances."""
    grid = (np.linspace(0, 10, 11), np.linspace(-5, 5, 11))
    xi = np.array([
        [np.nan, 5.0],
        [0.0, np.nan],
    ])
    indices, distances = find_indices(grid, xi, backend=backend)
    assert np.isnan(distances[0, 0])
    assert np.isnan(distances[1, 1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_out_of_bounds(backend):
    """Out-of-bounds values should be clamped to valid interval indices."""
    grid = (np.linspace(0, 10, 11), np.linspace(-5, 5, 11))
    xi = np.array([
        [-1.0, 11.0, 5.0],
        [-6.0, 6.0, 0.0],
    ])
    indices, distances = find_indices(grid, xi, backend=backend)
    # Below lower bound → index 0
    assert indices[0, 0] == 0
    assert indices[1, 0] == 0
    # Above upper bound → last valid interval
    assert indices[0, 1] == 9
    assert indices[1, 1] == 9


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_single_element_grid(backend):
    """Grid dimension with a single element."""
    grid = (np.array([5.0]), np.linspace(0, 10, 11))
    xi = np.array([
        [5.0, 5.0],
        [3.0, 7.0],
    ])
    indices, distances = find_indices(grid, xi, backend=backend)
    # Single-element dimension should get special index (-1)
    assert (indices[0] == -1).all()
    assert (distances[0] == 0.0).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_3d_grid(backend):
    """Three-dimensional grid."""
    grid = (np.linspace(0, 5, 6), np.linspace(0, 10, 11), np.linspace(-1, 1, 21))
    xi = np.array([
        [1.5, 3.0],
        [5.5, 10.0],
        [0.0, -0.5],
    ])
    indices, distances = find_indices(grid, xi, backend=backend)
    assert indices.shape == (3, 2)
    assert distances.shape == (3, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_irregular_grid(backend):
    """Irregularly spaced grid coordinates."""
    grid = (np.array([0.0, 1.0, 3.0, 6.0, 10.0]), np.linspace(0, 1, 5))
    xi = np.array([
        [2.0, 5.0, 9.9],
        [0.25, 0.5, 1.0],
    ])
    indices, distances = find_indices(grid, xi, backend=backend)
    # 2.0 is between 1.0 (idx 1) and 3.0 (idx 2), dist = (2-1)/(3-1) = 0.5
    assert indices[0, 0] == 1
    assert np.isclose(distances[0, 0], 0.5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_dtype(backend):
    """Check return dtypes — scipy returns int64/float64, numba returns intp/float32."""
    grid = (np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    xi = np.array([[5.0], [5.0]])
    indices, distances = find_indices(grid, xi, backend=backend)
    if backend == "scipy":
        assert indices.dtype == np.int64
        assert distances.dtype == np.float64
    else:
        assert indices.dtype == np.intp
        assert distances.dtype == np.float32


def test_find_indices_backend_invalid():
    """Unknown backend should raise ValueError."""
    grid = (np.linspace(0, 10, 11),)
    xi = np.array([[5.0]])
    with pytest.raises(ValueError, match="Unknown find_indices backend"):
        find_indices(grid, xi, backend="unknown")


@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed (reference backend)")
@pytest.mark.parametrize("backend", BACKENDS)
def test_find_indices_consistency_with_scipy(backend):
    """All backends should return results consistent with scipy for non-NaN points."""
    grid = (np.linspace(-10, 10, 101), np.linspace(-20, 20, 81))
    np.random.seed(42)
    n = 500
    xi = np.column_stack([
        np.random.uniform(-10, 10, n),
        np.random.uniform(-20, 20, n),
    ]).T

    ref_indices, ref_distances = find_indices(grid, xi, backend="scipy")
    indices, distances = find_indices(grid, xi, backend=backend)

    np.testing.assert_array_equal(indices, ref_indices)
    np.testing.assert_allclose(distances, ref_distances, rtol=1e-5)


# ---------------------------------------------------------------------------
# find_indices performance benchmarks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_points", [1_000_000])
def test_perf_find_indices(backend, n_points):
    """Benchmark find_indices across backends and query point counts."""
    grid = (
        np.linspace(-90, 90, 1800),    # latitude-like
        np.linspace(-180, 180, 3600),   # longitude-like
    )
    np.random.seed(42)
    xi = np.column_stack([
        np.random.uniform(-90, 90, n_points),
        np.random.uniform(-180, 180, n_points),
    ]).T

    # Warm-up (important for numba JIT)
    find_indices(grid, xi, backend=backend)

    with Chrono(f"find_indices ({backend}, {n_points} pts)"):
        find_indices(grid, xi, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n_points", [1_000_000])
def test_perf_find_indices_correlated(backend, n_points):
    """Benchmark find_indices with spatially correlated query points (random walk).

    This tests the adaptive walk strategy where successive points are close,
    making the running-index approach O(1) amortized per query.
    """
    grid = (
        np.linspace(-90, 90, 1800),    # latitude-like
        np.linspace(-180, 180, 3600),   # longitude-like
    )
    np.random.seed(42)
    xi = np.zeros((2, n_points))
    xi[0, 0] = np.random.uniform(-90, 90)
    xi[1, 0] = np.random.uniform(-180, 180)
    for j in range(1, n_points):
        xi[0, j] = xi[0, j - 1] + np.random.normal(0, 0.5)
        xi[1, j] = xi[1, j - 1] + np.random.normal(0, 0.5)
    # Wrap back into grid bounds
    xi[0] = np.mod(xi[0] + 90, 180) - 90
    xi[1] = np.mod(xi[1] + 180, 360) - 180

    # Warm-up (important for numba JIT)
    find_indices(grid, xi, backend=backend)

    with Chrono(f"find_indices_correlated ({backend}, {n_points} pts)"):
        find_indices(grid, xi, backend=backend)


@pytest.mark.parametrize("backend", **parametrize_dict({
    "numpy": False,
    "dask": True,
}))
def test_interp_preserves_non_interpolated_coords(backend):
    """Non-interpolated dimensions preserve their original coordinates."""
    data = xr.DataArray(
        np.random.rand(3, 4, 5),
        dims=["a", "b", "c"],
        coords={"a": [1.0, 2.0, 4.0], "b": [11.0, 12.0, 13.0, 14.0], "c": [101.0, 102.0, 103.0, 104.0, 105.0]},
    )
    a_coord = xr.DataArray(np.linspace(1.0, 4.0, 10), dims=["x"])
    b_coord = xr.DataArray(np.linspace(11.0, 14.0, 8 * 6).reshape(8, 6), dims=["y", "z"])
    if backend:
        a_coord = a_coord.chunk({"x": 5})
        b_coord = b_coord.chunk({"y": 4, "z": 3})

    coords = xr.Dataset({"a": a_coord, "b": b_coord})
    interpolator = Interpolator(data.to_dataset(name="dummy"), a=Linear(a_coord), b=Linear(b_coord))

    if backend:
        result = interpolator.map_blocks(coords)["dummy"]
    else:
        interpolator.process_block(coords)
        result = coords["dummy"]

    np.testing.assert_array_equal(result.coords["c"].values, data.coords["c"].values)
    assert "x" in result.coords and "y" in result.coords and "z" in result.coords
    assert "a" not in result.coords and "b" not in result.coords


# Regression tests (run with each backend through the autouse `default_backend` fixture)

@pytest.mark.parametrize("tension", [0.2, 0.5, 0.8])
def test_spline_tension(tension):
    """A cardinal spline is exact at the nodes and smooth for any tension"""
    x = np.arange(6.)
    data = xr.DataArray(x**2, dims=["x"], coords={"x": x})
    res = interp(data, x=Spline(xr.DataArray([2., 2.9999, 3.], dims=["p"]),
                                tension=tension))
    np.testing.assert_allclose(res.values, [4., 9., 9.], atol=1e-2)


@pytest.mark.parametrize("regular", [True, False])
def test_spline_decreasing(regular):
    """Spline interpolation on descending coords"""
    coords = np.array([10., 8., 6., 4., 2.] if regular else [10., 8., 5., 4., 0.])
    data = xr.DataArray(2 * coords, dims=["x"], coords={"x": coords})
    values = np.array([5., 8.]) if not regular else np.array([6., 8., 3.])
    res = interp(data, x=Spline(xr.DataArray(values, dims=["p"])))
    np.testing.assert_allclose(res.values, 2 * values)


def test_interp_shared_dim_order():
    """
    Dimension `time` shared between the data and the coordinate variable of the
    second interpolated dimension (not carried by the first one)
    """
    rng = np.random.default_rng(0)
    data = xr.DataArray(
        rng.random((3, 4, 5)),
        dims=["time", "lat", "lon"],
        coords={"lat": np.arange(4.), "lon": np.arange(5.)},
    )
    lat = xr.DataArray(rng.random((3, 6)) * 3, dims=["time", "x"])
    lon = xr.DataArray(rng.random(6) * 4, dims=["x"])
    ref = data.interp(lat=lat, lon=lon)
    for kwargs in [dict(lon=Linear(lon), lat=Linear(lat)),
                   dict(lat=Linear(lat), lon=Linear(lon))]:
        res = interp(data, **kwargs).transpose(*ref.dims)
        np.testing.assert_allclose(res.values, ref.values)


def test_regular_tiny_spacing():
    """An irregular grid with a tiny spacing is not detected as regular"""
    coords = np.array([0, 1e-9, 5e-9, 6e-9])
    data = xr.DataArray([0., 1., 5., 6.], dims=["x"], coords={"x": coords})
    res = interp(data, x=Linear(xr.DataArray([5e-9], dims=["p"])))
    np.testing.assert_allclose(res.values, [5.])
    with pytest.raises(ValueError):
        Linear_Indexer(coords, "error", "regular")


def test_nearest_nan_query():
    """NaN queries give NaN outputs with Nearest on float data, as with Linear"""
    data = xr.DataArray(np.array([10, 20, 30], dtype="float64"), dims=["x"],
                        coords={"x": [1., 2., 3.]})
    values = xr.DataArray([2.1, np.nan, 2.9], dims=["p"])
    res = interp(data, x=Nearest(values))
    np.testing.assert_array_equal(res.values, [20., np.nan, 30.])
    res_lin = interp(data, x=Linear(values))
    assert np.isnan(res_lin.values[1])


def test_nearest_nan_query_2d():
    """NaN query mask is broadcast against the output dims"""
    data = xr.DataArray(np.arange(6, dtype="float64").reshape(2, 3),
                        dims=["y", "x"], coords={"x": [1., 2., 3.]})
    values = xr.DataArray([[1., np.nan], [3., 2.]], dims=["a", "b"])
    res = interp(data, x=Nearest(values))
    assert res.dims == ("y", "a", "b")
    expected = np.array([[[0., np.nan], [2., 1.]],
                         [[3., np.nan], [5., 4.]]])
    np.testing.assert_array_equal(res.values, expected)


def test_nearest_nan_query_int_dtype():
    """Integer data keeps its dtype with Nearest, whatever the query values"""
    data = xr.DataArray(np.array([10, 20, 30], dtype="int64"), dims=["x"],
                        coords={"x": [1., 2., 3.]})
    for values in [[2.1, 2.9], [2.1, np.nan, 2.9]]:
        res = interp(data, x=Nearest(xr.DataArray(values, dims=["p"])))
        assert res.dtype == data.dtype
    # dtype is also consistent across dask chunks with and without NaN queries
    values = xr.DataArray([2.1, 2.9, np.nan, 1.2], dims=["p"]).chunk(p=2)
    res = interp(data, x=Nearest(values))
    assert res.dtype == data.dtype
    assert res.compute().dtype == data.dtype
    np.testing.assert_array_equal(res.values[[0, 1, 3]], [20, 30, 10])


def test_linear_decreasing_unsigned():
    """Descending unsigned int coords are detected as descending"""
    coords = np.array([3, 2, 1], dtype="uint8")
    data = xr.DataArray([30., 20., 10.], dims=["x"], coords={"x": coords})
    res = interp(data, x=Linear(xr.DataArray([2.5], dims=["p"])))
    np.testing.assert_allclose(res.values, [25.])
    res = interp(data, x=Nearest(xr.DataArray([2.9], dims=["p"])))
    np.testing.assert_allclose(res.values, [30.])


def test_interpolator_index():
    """Interpolator accepts Index (not only interp)"""
    ds = xr.Dataset({"v": (("x", "y"), np.arange(12.).reshape(3, 4))},
                    coords={"y": np.arange(4.)})
    coords = xr.Dataset({"x": xr.DataArray([2, 0], dims=["p"]),
                         "y": xr.DataArray([1.5, 2.5], dims=["p"])})
    interpolator = Interpolator(ds, x=Index(coords.x), y=Linear("y"))
    interpolator.process_block(coords)
    np.testing.assert_allclose(coords["v"].values, [9.5, 2.5])
