from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pytest
import xarray as xr

from hygeos_core.files.save import to_netcdf
from hygeos_core.interpolate import Interpolator, Linear
from hygeos_core.monitor import Chrono
from hygeos_core.process.blockwise import BlockProcessor, CompoundProcessor
from hygeos_core.tools import Var


class InitProcessor(BlockProcessor):
    def input_vars(self) -> List[Var]:
        return [Var('latitude'), Var('rho_toa', tags=['level1'])]

    def created_vars(self) -> List[Var]:
        return [Var('flags', dtype='uint16', dims_like='latitude', tags=['level2', 'flags'])]

    def process_block(self, block: xr.Dataset):
        block['flags'] = xr.zeros_like(block.latitude, dtype='uint16')
        

class Cloudmask(BlockProcessor):
    """Example cloud mask processor"""
    
    def input_vars(self) -> List[Var]:
        return [Var('rho_toa')]

    def modified_vars(self) -> List[Var]:
        return [Var('flags', flags={'CLOUD': 1})]
    
    def process_block(self, block: xr.Dataset) -> None:
        """Compute cloud mask"""
        nir = block.rho_toa.isel(band=-1)

        self.raiseflag(block, 'flags', 'CLOUD', nir > 0.1)


def InterpolatorFactory() -> BlockProcessor:
    return Interpolator(
        create_ancillary_dataset(),
        lat=Linear("latitude"),
        lon=Linear("longitude"),
    )


class Rayleigh(BlockProcessor):
    def input_vars(self) -> List[Var]:
        return [Var("rho_toa"), Var("ozone")]

    def created_vars(self) -> List[Var]:
        return [Var("rho_rc")]
    
    def auto_template(self) -> bool:
        return True
    
    def check(self, ds: xr.Dataset):
        assert ds.ozone.units == 'DU'

    def process_block(self, block: xr.Dataset):
        assert block.ozone.units == 'DU'
        block["rho_rc"] = xr.zeros_like(block.rho_toa, dtype="float32")

class Aerosol(BlockProcessor):
    def input_vars(self) -> List[Var]:
        return [Var("rho_rc")]

    def modified_vars(self) -> List[Var]:
        return [Var("flags", flags={'AC_ERROR': 2}, tags=['level2'])]

    def created_vars(self) -> List[Var]:
        return [
            Var(
                "rho_w",
                dtype="float32",
                dims_like="rho_rc",
                tags=["level2"],
                attrs={
                    "desc": "Water reflectance",
                    "units": "dimensionless",
                },
            ),
            Var("rho_aer", dtype="float32", dims_like="rho_rc"),
        ]
    def global_attrs(self) -> Dict[str, Any]:
        return {"algorithm_version": "1.0"}

    def process_block(self, block: xr.Dataset):
        self.raiseflag(block, "flags", "AC_ERROR", block.rho_rc.sum(dim='band') > 0.5)
        block["rho_aer"] = xr.zeros_like(block.rho_rc, dtype="float32")
        block["rho_w"] = xr.zeros_like(block.rho_rc, dtype="float32")


class Noop(BlockProcessor):
    def process_block(self, block: xr.Dataset) -> None:
        pass

def create_sample_dataset(
    small: bool = True
) -> xr.Dataset:
    """
    Create a sample dataset with latitude, longitude, and rho_toa variables.
    
    Parameters
    ----------
    x_size, y_size : dimensions of the product
    n_bands : int
        Number of spectral bands (default: 4)
    chunks : dict, optional
        Chunk sizes for dask arrays. Example: {'lat': 45, 'lon': 90, 'band': 2}
    seed : int
        Random seed for reproducible data
        
    Returns
    -------
    xr.Dataset
        Dataset with latitude, longitude coordinates and rho_toa variable
    """
    if small:
        x_size = 200
        y_size = 200
        n_bands = 10
        chunks = {"x": 50, "y": 50, "band": -1}
    else:
        x_size = 2000
        y_size = 2000
        n_bands = 10
        chunks = {"x": 500, "y": 500, "band": -1}

    seed = 42
    np.random.seed(seed)

    # Create latitude and longitude 2D variables as random arrays
    latitude = np.random.uniform(-90, 90, size=(y_size, x_size))
    longitude = np.random.uniform(-180, 180, size=(y_size, x_size))

    ds = xr.Dataset(
        {
            "rho_toa": (
                ["band", "y", "x"],
                np.random.uniform(0.0, 1.0, size=(n_bands, y_size, x_size)),
                {
                    "long_name": "Top of Atmosphere Reflectance",
                    "units": "dimensionless",
                    "valid_range": [0.0, 1.0],
                },
            ),
            "latitude": (["y", "x"], latitude),
            "longitude": (["y", "x"], longitude),
        },
        coords={
            "band": np.arange(n_bands),
            "y": np.arange(y_size),
            "x": np.arange(x_size),
        },
        attrs={
            "title": "Sample satellite reflectance dataset",
            "description": "Simulated top-of-atmosphere reflectance data",
            "created_by": "create_sample_dataset function",
        },
    )
    if chunks is not None:
        ds = ds.chunk(chunks)
    return ds


def create_ancillary_dataset() -> xr.Dataset:
    """
    Create a sample dataset with ozone and wind variables, and lat/lon coordinates.
        
    Returns
    -------
    xr.Dataset
        Dataset with ozone and wind variables, lat/lon coordinates
    """
    lat_size = 90
    lon_size = 180

    seed = 42
    np.random.seed(seed)

    # Create lat and lon coordinates
    lat = np.linspace(-90, 90, lat_size)
    lon = np.linspace(-180, 180, lon_size)

    # Create ozone variable (e.g., ozone concentration in DU)
    ozone = np.random.uniform(200, 400, size=(lat_size, lon_size))
    
    # Create wind variable (e.g., wind speed in m/s)
    wind = np.random.uniform(0, 20, size=(lat_size, lon_size))

    ds = xr.Dataset(
        {
            "ozone": (
                ["lat", "lon"],
                ozone,
                {
                    "long_name": "Ozone concentration",
                    "units": "DU",
                },
            ),
            "wind": (
                ["lat", "lon"],
                wind,
                {
                    "long_name": "Wind speed",
                    "units": "m/s",
                },
            ),
        },
        coords={
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": "Sample ozone and wind dataset",
            "description": "Simulated ozone concentration and wind speed data",
            "created_by": "create_ozone_wind_dataset function",
        },
    )
    return ds


def test_sample_dataset():
    """Test the sample dataset creation."""
    # Create basic dataset
    ds = create_sample_dataset()

    # Check variables
    assert 'rho_toa' in ds.data_vars
    assert 'latitude' in ds.data_vars
    assert 'longitude' in ds.data_vars

    # Check coordinates
    assert 'band' in ds.coords
    assert 'y' in ds.coords
    assert 'x' in ds.coords

    # Check data ranges
    assert ds.rho_toa.min() >= 0.0
    assert ds.rho_toa.max() <= 1.0

    assert hasattr(ds.rho_toa.data, 'chunks')


@pytest.fixture(scope="session")
def sample_netcdf_file(request):
    """Create a sample dataset for performance tests."""
    small: bool = request.param
    if small:
        ds = create_sample_dataset(small=True)
        yield ds
    else:
        file_path = Path('/tmp/') / "large_sample_data.nc"
        if not file_path.exists():
            # Create large sample dataset
            ds = create_sample_dataset(small=False)
            
            # Save to NetCDF
            to_netcdf(ds, file_path)
        
        # Load dataset with chunks
        chunks = {"x": 500, "y": 500, "band": -1}
        ds_loaded = xr.open_dataset(file_path, chunks=chunks)
        
        yield ds_loaded


@pytest.mark.parametrize(
    "compound_kwargs,expected_outputs",
    [
        (
            {"outputs": "all"},
            {
                "rho_rc",
                "rho_aer",
                "wind",
                "ozone",
                "rho_w",
                "flags",
                "latitude",
                "rho_toa",
                "longitude",
            },
        ),
        (
            {"outputs": "tags", "outputs_tags": ["level1", "level2"]},
            {"flags", "rho_toa", "rho_w"},
        ),
        (
            {"outputs": "named", "outputs_names": ["rho_w", "flags", "rho_toa"]},
            {"flags", "rho_toa", "rho_w"},
        ),
        (
            {"outputs": "created_modified"},
            {'flags', 'wind', 'ozone', 'rho_w', 'rho_rc', 'rho_aer'},
        ),
    ],
)
def test_blockwise(compound_kwargs: dict, expected_outputs: set):
    """
    Test blockwise processing with different output selection modes.
    """
    ds = create_sample_dataset()
    
    # Apply blockwise processing to the compound processor
    compound = CompoundProcessor(
        [
            InitProcessor(),
            Cloudmask(),
            InterpolatorFactory(),
            Rayleigh(),
            Aerosol(),
        ],
        **compound_kwargs,
    )
    result = compound.map_blocks(ds)

    # Check the output variables
    assert set([str(x) for x in result.data_vars]) == expected_outputs
    
    # Verify results
    assert 'rho_w' in result.data_vars
    assert 'flags' in result.data_vars

    # Check that it computes
    result.compute()

    # Check var attrs
    assert result['rho_w'].attrs == {'desc': 'Water reflectance', 'units': 'dimensionless'}
    assert len(result['flags'].attrs)
    
    # Check global attributes
    assert result.attrs['algorithm_version'] == '1.0'
    # Original attributes should be preserved
    assert result.attrs['title'] == 'Sample satellite reflectance dataset'
    if 'rho_toa' in expected_outputs:
        assert result.rho_toa.attrs['units'] == "dimensionless"


def test_blockwise_missing_var():
    """
    Check that KeyError is raised when the input dataset does not contain the
    required variable
    """
    with pytest.raises(KeyError):
        InitProcessor().map_blocks(xr.Dataset())

def test_blockwise_dtype_error():
    # When the actual dtype does not match the declared one
    class BuggedInitProcessor(InitProcessor):
        def process_block(self, block: xr.Dataset):
            # dtype does not match created_vars
            block['flags'] = xr.zeros_like(block.latitude, dtype='uint8')

    ds = create_sample_dataset()
    with pytest.raises(TypeError):
        BuggedInitProcessor().map_blocks(ds).compute()

def test_blockwise_wrong_order():
    # Apply modules in the wrong order
    ds = create_sample_dataset()
    with pytest.raises(KeyError):
        CompoundProcessor([
            Aerosol(),
            InitProcessor(),
        ]).map_blocks(ds)

def test_blockwise_missing_out_var():
    # When the processor does not create the expected variable
    ds = create_sample_dataset()
    class BuggedInitProcessor(InitProcessor):
        def process_block(self, block: xr.Dataset):
            # dtype does not match created_vars
            pass

    ds = create_sample_dataset()
    with pytest.raises(ValueError):
        BuggedInitProcessor().map_blocks(ds).compute()


@pytest.mark.parametrize('compound', [True, False])
def test_blockwise_newdims(compound: bool):
    """
    Test a processor which creates new dimensions
    """
    class NewdimensionProcessorA(BlockProcessor):

        def input_vars(self) -> List[Var]:
            return [Var('latitude')]
            
        def created_dims(self):
            return {'new_dim1': 2, 'new_dim2': [1, 2, 3]}

        def created_vars(self) -> List[Var]:
            return [
                Var("A", dtype="float32", dims=("new_dim1", "new_dim2")),
            ]
        
        def process_block(self, block: xr.Dataset):
            block["A"] = xr.DataArray(
                np.zeros((2, 3), dtype="float32"), dims=("new_dim1", "new_dim2")
            )
    class NewdimensionProcessorB(BlockProcessor):

        def input_vars(self) -> List[Var]:
            return [Var('latitude')]
            
        def created_dims(self):
            return {'new_dim1': 2}

        def created_vars(self) -> List[Var]:
            return [
                Var("B", dtype="float32", dims=("new_dim1", "y", "x")),
            ]
        
        def process_block(self, block: xr.Dataset):
            block["B"] = xr.DataArray(
                np.zeros((2, block.y.size, block.x.size), dtype="float32"),
                dims=("new_dim1", "y", "x"),
            )

    ds = create_sample_dataset()

    if compound:
        result = CompoundProcessor(
            [NewdimensionProcessorA(), NewdimensionProcessorB()]
        ).map_blocks(ds)
    else:
        result = NewdimensionProcessorA().map_blocks(ds)
    assert 'new_dim1' in result.dims
    assert 'new_dim2' in result.dims
    assert 'new_dim2' in result.coords

    result.compute()



@pytest.mark.parametrize("fail,kwargs", [
    (False, {'outputs': 'named', 'outputs_names': ['flags']}),
    (True, {'outputs': 'named', 'outputs_names': ['dummy']}),
    (True, {'outputs': 'created_modified'}),
    (True, {'outputs': 'all'}),
    ])
def test_blockwise_partial(fail: bool, kwargs: dict):
    """
    Test partial processing: unnecessary Processors shall not be applied
    """
    class Invalid(BlockProcessor):
        def input_vars(self) -> List[Var]:
            return [Var('flags')]
        def created_vars(self) -> List[Var]:
            return [Var('dummy', dims_like='flags', dtype='float32')]
        def process_block(self, block: xr.Dataset):
            raise RuntimeError

    ds = create_sample_dataset()
    res = CompoundProcessor([
        InitProcessor(),
        Cloudmask(),
        Invalid(),
    ], **kwargs).map_blocks(ds)
    
    if fail:
        # This should fail because of the Invalid processor.
        with pytest.raises(RuntimeError):
            res.compute()
    else:
        # And not fail when we skip the variable "dummy" for which
        # the processor raises an error
        res.compute()


def test_blockwise_noop():
    """
    Test a BlockProcessor that does nothing
    """
    ds = create_sample_dataset()
    Noop().map_blocks(ds)


def test_processor_describe():
    Aerosol().describe()


def test_compound_describe():
    CompoundProcessor([
        InitProcessor(),
        Cloudmask(),
        InterpolatorFactory(),
        Noop(),
        Rayleigh(),
        Aerosol(),
    ]).describe()


@pytest.mark.parametrize(
    "sample_netcdf_file",
    [
        # False,  # large
        True,  # small
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "scheduler",
    [
        "sync",
        "threads",
        "processes",
        "distributed",
    ],
)
@pytest.mark.parametrize("chained", [True, False])
def test_perf(chained: bool, scheduler: str, sample_netcdf_file):
    """
    Check the performance of chained processing, compared with successive map_blocks

    """
    ds = sample_netcdf_file

    processors = [
        InitProcessor(),
        Cloudmask(),
        InterpolatorFactory(),
        Rayleigh(),
        Aerosol(),
    ]

    with Chrono('Init'):
        ds = CompoundProcessor(processors).map_blocks(ds, chained=chained)

    with Chrono('Process'):
        if scheduler == "distributed":
            from dask.distributed import Client, LocalCluster
            cluster = LocalCluster()
            client = Client(cluster)
            ds.compute()

        else:
            ds.compute(scheduler=scheduler)
