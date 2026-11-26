#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Various utility functions for modifying xarray object
'''

import re
from collections import defaultdict
from enum import Enum
from functools import wraps
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, overload

import numpy as np
import xarray as xr
from dask import array as da
from numpy import arcsin as asin
from numpy import cos, radians, sin, sqrt, where

try:
    from shapely.geometry import Point, Polygon
except ImportError:
    pass
from collections import OrderedDict

from dateutil.parser import parse

from hygeos_core import log

flags          = 'flags'
flags_dtype    = 'uint16'
flags_meanings = 'flag_meanings'
flags_masks    = 'flag_masks'
flags_meanings_separator = ' '

footprint_lat = 'footprint_lat'
footprint_lon = 'footprint_lon'


def align_lists(lists: list[list]) -> list:
    """
    Align items from several lists into a single list that respects the order in each sublist.
    Raises ValueError if the lists cannot be aligned (e.g., due to cycles).

    Args:
        lists: List of lists, where each sublist defines a partial order.

    Returns:
        A single list with items aligned according to the partial orders.

    Example:
        align_lists([['x'], ['x', 'y'], ['y', 'z']]) -> ['x', 'y', 'z']
        align_lists([['x', 'y'], ['y', 'x']]) -> ValueError
    """
    graph = defaultdict(list)
    indegree = defaultdict(int)
    all_items = set()

    # Build graph and indegree
    for lst in lists:
        for item in lst:
            all_items.add(item)
        for i in range(len(lst) - 1):
            graph[lst[i]].append(lst[i + 1])
            indegree[lst[i + 1]] += 1

    # Topological sort using Kahn's algorithm with priority queue for deterministic order
    queue = []
    for item in all_items:
        if indegree[item] == 0:
            heappush(queue, item)
    result = []

    while queue:
        item = heappop(queue)
        result.append(item)
        for neighbor in graph[item]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heappush(queue, neighbor)

    if len(result) != len(all_items):
        raise ValueError("Cannot align: cycle detected in the order constraints")

    return result


def datetime(ds: xr.Dataset):
    '''
    Parse datetime (in isoformat) from `ds` attributes
    '''
    if ('start_time' in ds.attrs) and ('end_time' in ds.attrs):
        st = ds.start_time
        et = ds.end_time
        if isinstance(st, str):
            st = parse(st).replace(tzinfo=None)
        if isinstance(et, str):
            et = parse(et).replace(tzinfo=None)
        return st + (et - st)/2
    elif 'datetime' in ds.attrs:
        return parse(ds.attrs['datetime']).replace(tzinfo=None)
    else:
        raise AttributeError


def haversine(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float,
    lon2: float,
    radius: float = 6371,
):
    '''
    Calculate the great circle distance between two points (specified in
    decimal degrees) on a sphere of a given radius

    Returns the distance in the same unit as radius (defaults to earth radius in km)
    '''
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = [radians(x) for x in [lon1, lat1, lon2, lat2]]

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    dist = radius * c

    return dist


def locate(
    lat: xr.DataArray,
    lon: xr.DataArray,
    lat0: float,
    lon0: float,
    dist_min_km: float | None = None,
    verbose: bool = False,
) -> Dict:
    """
    Locate `lat0`, `lon0` within `lat`, `lon` (xr.DataArrays)

    if dist_min_km is specified and if the minimal distance
    exceeds it, a ValueError is raised

    returns a dictionary of the pixel coordinates
    """
    if verbose:
        print(f'Locating lat={lat0}, lon={lon0}')
    dist = haversine(lat.values, lon.values, lat0, lon0)
    dist_min = np.array(np.nanmin(dist))

    if np.isnan(dist_min):
        raise ValueError('No valid input coordinate')

    if (dist_min_km is not None) and (dist_min > dist_min_km):
        raise ValueError(f'locate: minimal distance is {dist_min}, '
                         f'should be at most {dist_min_km}')

    coords = [x[0] for x in np.where(dist == dist_min)]

    return {dim: coords[idim] for idim, dim in enumerate(lat.dims)}


def drop_unused_dims(ds): 
    """Simple function to remove unused dimensions in a xarray.Dataset"""
    return ds.drop_vars([var for var in ds.coords if var not in ds.dims])

def contains(ds: xr.Dataset, lat: float, lon: float):
    pt = Point(lat, lon)
    area = Polygon(zip(
        ds.attrs[footprint_lat],
        ds.attrs[footprint_lon]
    ))
    # TODO: proper inclusion test
    # TODO: make it work with arrays
    return area.contains(pt)


def sub(ds: xr.Dataset, 
        cond: xr.DataArray, 
        drop_invalid: bool = True, 
        int_default_value: int = 0):
    '''
    Creates a Dataset based on the conditions passed in parameters

    cond : a DataArray of booleans that defines which pixels are kept

    drop_invalid, bool
        if True invalid pixels will be replace by nan for floats and
        int_default_value for other types

    int_default_value, int
        for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    res = xr.Dataset()

    if drop_invalid:
        assert 'mask_valid' not in res
        res['mask_valid'] = cond.where(cond, drop=True)
        res['mask_valid'] = res['mask_valid'].where(~np.isnan(res['mask_valid']), 0).astype(bool)

    slice_dict = dict()
    for dim in cond.dims:
        s = cond.any(dim=[d for d in cond.dims if d != dim])
        wh = where(s)[0]
        if len(wh) == 0:
            slice_dict[dim] = slice(2,1)
        else:
            slice_dict[dim] = slice(wh[0], wh[-1]+1)

    for var in ds.variables:
        if set(cond.dims) == set(ds[var].dims).intersection(set(cond.dims)):
            if drop_invalid:
                if ds[var].dtype in ['float16', 'float32', 'float64']:
                    res[var] = ds[var].where(cond, drop=True)
                else:
                    res[var] = ds[var].isel(slice_dict).where(res['mask_valid'], int_default_value)

            else:
                res[var] = ds[var].isel(slice_dict)

    res.attrs.update(ds.attrs)

    return res


def sub_rect(ds: xr.Dataset, lat_min, lon_min, lat_max, lon_max, 
             drop_invalid: bool = True, int_default_value: int = 0):
    '''
    Returns a Dataset based on the coordinates of the rectangle passed in parameters

    lat_min, lat_max, lon_min, lon_max : delimitations of the region of interest

    drop_invalid, bool : if True, invalid pixels will be replace by nan
    for floats and int_default_value for other types

    int_default_value, int : for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    lat = ds.latitude.compute()
    lon = ds.longitude.compute()
    cond = (lat < lat_max) & (lat > lat_min) & (lon < lon_max) & (lon > lon_min)
    cond = cond.compute()

    return sub(ds, cond, drop_invalid, int_default_value)


def sub_pt(ds: xr.Dataset, pt_lat, pt_lon, rad, 
           drop_invalid: bool = True, int_default_value: int = 0):
    '''
    Creates a Dataset based on the circle specified in parameters

    pt_lat, pt_lon : Coordonates of the center of the point

    rad : radius of the circle in km

    drop_invalid, bool
        if True invalid pixels will be replace by nan for floats
        and int_default_value for other types

    int_default_value, int
        for DataArrays of type int, this value is assigned on non-valid pixels
    '''
    lat = ds["latitude"].compute()
    lon = ds["longitude"].compute()
    cond = haversine(lat, lon, pt_lat, pt_lon) < rad
    cond = cond.compute()

    return sub(ds, cond, drop_invalid, int_default_value)


def split(d: xr.Dataset | xr.DataArray, dim: str, sep: str = '_'):
    '''
    Returns a Dataset where a given dimension is split into as many variables

    d: Dataset or DataArray
    '''
    assert dim in d.dims
    assert dim in d.coords, f'The split dimension "{dim}" must have coordinates.'

    if isinstance(d, xr.DataArray):
        m = xr.merge([
            d.isel(**{dim: i}).rename(f'{d.name}{sep}{d[dim].data[i]}').drop_vars(dim)
            for i in range(len(d[dim]))
            ])
    elif isinstance(d, xr.Dataset):
        m = xr.merge(
            [split(d[x], dim, sep=sep)
             if dim in d[x].dims
             else d[x]
             for x in d])
    else:
        raise Exception('`split` expects Dataset or DataArray.')

    m.attrs.update(d.attrs)
    m.attrs['split_dimension'] = dim
    m = m.assign_coords(**d.coords)
    return m


def merge(ds: xr.Dataset,
          dim: str = None,
          varname: str = None,
          pattern: str = r'(.+)_(\d+)',
          dtype: type = int):
    r"""
    Merge DataArrays in `ds` along dimension `dim`.

    ds: xr.Dataset

    dim: str or None
        name of the new or existing dimension
        if None, use the attribute `split_dimension`
    
    varname: str or None
        name of the variable to create
        if None, detect variable name from regular expression

    pattern: str
        Regular expression for matching variable names and coordinates
        if varname is None:
            First group represents the new variable name.
            Second group represents the coordinate value
            Ex: r'(.+)_(\d+)'
                    First group matches all characters.
                    Second group matches digits.
                r'(\D+)(\d+)'
                    First group matches non-digit.
                    Second group matches digits.
        if varname is not None:
            Match a single group representing the coordinate value

    dtype: data type
        data type of the coordinate items
    """
    copy = ds.copy()

    if dim is None:
        dim = copy.attrs['split_dimension']

    mapping = {}   # {new_name: [(old_name, value), ...], ...}
    for x in copy:
        m = re.findall(pattern, x)
        if not m:
            continue  # does not match
        assert len(m) == 1, 'Expecting a single match'

        if varname is None:
            assert len(m[0]) == 2, 'Expecting two groups in regular expression'
            var, coord = m[0]
        else:
            assert not isinstance(m[0], tuple), 'Expecting a single group in regular expression'
            coord = m[0]
            var = varname
        c = dtype(coord)

        if var not in mapping:
            mapping[var] = []

        mapping[var].append((x, c))

    for var in mapping:
        data = xr.concat([copy[x] for x, c in mapping[var]], dim)
        coords = [c for x, c in mapping[var]]
        if dim in copy.coords:
            # check that the coordinates are matching
            existing_coords = list(copy.coords[dim].data)
            assert existing_coords == coords, \
                f'Error: {existing_coords} != {coords} (in variable {var})'
        else:
            copy = copy.assign_coords(**{dim: coords})
        copy[var] = data
        copy = copy.drop_vars([x for x, c in mapping[var]])

    return copy


def getflags(A=None, meanings=None, masks=None, sep=None):
    """
    returns the flags in attributes of `A` as a dictionary {meaning: value}

    Arguments:
    ---------

    provide either:
        A: Dataarray
    or:
        meanings: flag meanings 'FLAG1 FLAG2'
        masks: flag values [1, 2]
        sep: string separator
    """
    try:
        meanings = meanings if (meanings is not None) else A.attrs[flags_meanings]
        masks = masks if (masks is not None) else A.attrs[flags_masks]
        sep = sep or flags_meanings_separator
    except KeyError:
        return OrderedDict()
    return OrderedDict(zip(meanings.split(sep), masks))


def getflag(A: xr.DataArray, name: str):
    """
    Return the binary flag with given `name` as a boolean array

    A: DataArray
    name: str

    example: getflag(flags, 'LAND')
    """
    flags = getflags(A)

    assert name in flags, f'Error, {name} no in {list(flags)}'

    return (A & flags[name]) != 0


def raiseflag(A: xr.DataArray, flag_name: str | Enum, flag_value: int, condition=None):
    """
    Raise a flag in DataArray `A` with name `flag_name`, value `flag_value` and `condition`
    The name and value of the flag is recorded in the attributes of `A`

    Arguments:
    ----------
    A: DataArray of integers

    flag_name: str
        Name of the flag
    flag_value: int
        Value of the flag
    condition: boolean array-like of same shape as `A`
        Condition to raise flag.
        If None, the flag values are unchanged ; the flag is simple registered in the
        attributes.
    """
    if isinstance(flag_name, Enum):
        flag_name = flag_name.name
    assert isinstance(flag_name , str)
    if not np.issubdtype(A.dtype, np.integer):
        raise ValueError(f"raiseflag can only be used on integer DataArrays, got {A.dtype}")
    flags = getflags(A)
    dtype_flag_masks = A.dtype

    if flags_meanings not in A.attrs:
        A.attrs[flags_meanings] = ''
    if flags_masks not in A.attrs:
        A.attrs[flags_masks] = np.array([], dtype=dtype_flag_masks)

    # update the attributes if necessary
    if flag_name in flags:
        # existing flag: check value
        assert flags[flag_name] == flag_value, \
            f'Flag {flag_name} already exists with a different value'
    else:
        assert flag_value not in flags.values(), \
            f'Flag value {flag_value} is already assigned to a different flags (assigned flags are {flags.values()})'

        flags[flag_name] = flag_value

        # sort the flags by values
        keys, values = zip(*sorted(flags.items(), key=lambda y: y[1]))

        A.attrs[flags_meanings] = flags_meanings_separator.join(keys)
        A.attrs[flags_masks] = np.array(values, dtype=dtype_flag_masks)

    if condition is not None:
        notraised = (A & flag_value) == 0
        A += flag_value * ((condition != 0) & notraised).astype(A.dtype)


def wrap(ds: xr.Dataset, dim: str, vmin: float, vmax: float):
    """
    Wrap and reorder a cyclic dimension between vmin and vmax.
    The border value is duplicated at the edges.
    The period is (vmax-vmin)

    Example:
    * Dimension [0, 359] -> [-180, 180]
    * Dimension [-180, 179] -> [-180, 180]
    * Dimension [0, 359] -> [0, 360]

    Arguments:
    ----------

    ds: xarray.Dataset
    dim: str
        Name of the dimension to wrap
    vmin, vmax: float
        new values for the edges
    """

    pivot = vmax if (vmin < ds[dim][0]) else vmin

    left = ds.sel({dim: slice(None, pivot)})
    right = ds.sel({dim: slice(pivot, None)})

    if right[dim][-1] > vmax:
        # apply the offset at the right part
        right = right.assign_coords({dim: right[dim] - (vmax-vmin)})
    else:
        # apply the offset at the left part
        left = left.assign_coords({dim: left[dim] + (vmax-vmin)})

    # swaps the two parts
    return xr.concat([right, left], dim=dim)


def chunk(ds: xr.Dataset, **kwargs):
    """
    Apply rechunking to a xr.Dataset `ds` along dimensions provided as kwargs

    Works like `ds.chunk` but works also for Datasets with repeated dimensions.
    """

    for var in ds:
        chks = [kwargs[d] if d in kwargs else None for d in ds[var].dims]
        if hasattr(ds[var].data, 'chunks') and len([c for c in chks if c is not None]):
            ds[var].data = ds[var].data.rechunk(chks)
            
    return ds


def trim_dims(A: xr.Dataset):
    """
    Trim the dimensions of Dataset A
    
    Rename all possible dimensions to avoid duplicate dimensions with same sizes
    Avoid any DataArray with duplicate dimensions
    """
    # list of lists of dimensions that should be grouped together
    groups = []
    
    # loop over all dimensions sizes
    for size in set(A.sizes.values()):
        # list all dimensions with current size
        groups_current = []
        dims_current = [k for k, v in A.sizes.items()
                        if v == size]

        # for each variable, add its dimensions (intersecting dims_current)
        # to separate groups to avoid duplicated
        for var in A:
            for i, d in enumerate(
                [x for x in A[var].dims
                 if x in dims_current]
                ):
                if len(groups_current) <= i:
                    groups_current.append([])
                if d not in groups_current[i]:
                    groups_current[i].append(d)

        groups += groups_current

    # check that groups are pairwise disjoint (no dimension in several groups)
    all_dims = sum(groups, [])
    assert len(all_dims) == len(set(all_dims))

    rename_dict = dict(sum([[(dim, 'new_'+group[0])
                             for dim in group]
                            for group in groups
                            if len(group) > 1  # don't rename if not useful
                            ], []))
    return A.rename_dims(rename_dict)


def only(iterable):
    """If *iterable* has only one item, return it.
    Otherwise raise a ValueError
    """
    x = list(iterable)
    if len(x) != 1:
        raise ValueError
    return x[0] 

def reglob(path: Path|str, regexp: str) -> List[Path]:
    """
    Returns the sorted list of paths in directory `path` whose file name
    matches the regular expression `regexp` (using `re.search`)

    Ex: reglob(dir, r'^S2_\d+\.nc$')
    """
    assert regexp[0] != '*', 'Avoid using wildcard in regular expression'
    return sorted(p for p in Path(path).iterdir()
                  if re.search(regexp, p.name) is not None)

@overload
def xrcrop(A: xr.Dataset, **kwargs) -> xr.Dataset: ...
@overload
def xrcrop(A: xr.DataArray, **kwargs) -> xr.DataArray: ...


def xrcrop(
    A: Union[xr.Dataset, xr.DataArray], **kwargs
) -> Union[xr.Dataset, xr.DataArray]:
    """
    Crop a Dataset or DataArray along dimensions based on min/max values.
    
    For each dimension provided as kwarg, the min/max values along that dimension
    can be provided:
        - As a min/max tuple
        - As a DataArrat, for which the min/max are computed

    Ex: crop dimensions `latitude` and `longitude` of `gsw` based on the min/max
        of ds.lat and ds.lon
        gsw = xrcrop(
            gsw,
            latitude=ds.lat,
            longitude=ds.lon,
        )
    
    Note: the purpose of this function is to make it possible to .compute() the result
    of the cropped data, thus allowing to perform a sel over large arrays (otherwise
    extremely slow with dask based arrays).
    """
    isel_dict = {}
    for k, v in kwargs.items():
        if isinstance(v, tuple):
            (vmin, vmax) = v
            assert vmin < vmax
        elif isinstance(v, xr.DataArray):
            vmin = v.min().compute().item()
            vmax = v.max().compute().item()
        else:
            raise TypeError
        index = A.indexes[k]

        # Get bracketing indices for vmin/vmax
        if index.is_monotonic_increasing:
            imin = max(0, index.get_slice_bound(vmin, "right") - 1)
            imax = min(len(index), index.get_slice_bound(vmax, "left") + 1)
        elif index.is_monotonic_decreasing:
            imin = max(0, index.get_slice_bound(vmax, "right") - 1)
            imax = min(len(index), index.get_slice_bound(vmin, "left") + 1)
        else:
            raise ValueError

        assert imin < imax
        isel_dict[k] = slice(imin, imax)

    return A.isel(isel_dict)


def _model_name(m) -> str:
    """Name of a MapBlocksOutput model item (Var or DataArray)"""
    return m.name if isinstance(m, xr.DataArray) else str(m)


class MapBlocksOutput:
    def __init__(
        self,
        model: List,
        new_dims: Dict | None = None,
    ) -> None:
        """
        Describe a Dataset structure, for use with xr.map_blocks to ensure consistency
        between the output of the blockwise function and the call to xr.map_blocks.

        Args:
            model: list of DataArrays of Var objects describing the output
            new_dims: dictionary providing the new dimensions
                ex: new_dims={'new_dim': xr.DataArray([0, 2, 3])}
                or simply the dimension size if new_dim has no coordinate

        Example:
            model = MapBlockOutput([
                # Either describe the variable with a `Var` object
                Var('latitude', 'float32', ['x', 'y']),
                # or with a renamed DataArray
                ds.rho_toa.rename('rho_gc')
            ])

        """
        self.model = model
        self.new_dims = new_dims or {}

    def __add__(self, other):
        concatenated = MapBlocksOutput(self.model + other.model,
                               new_dims={**self.new_dims, **other.new_dims})
        names = [_model_name(x) for x in concatenated.model]
        assert len(set(names)) == len(names)
        return concatenated

    def template(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Return an empty template for this model, to be provided to xr.map_blocks
        """
        return xr.merge(
            [
                m.to_template(ds, new_dims=self.new_dims) if isinstance(m, Var) else m
                for m in self.model
            ]
        )

    def subset(self, ds: xr.Dataset) -> xr.Dataset:
        return ds[[_model_name(var) for var in self.model]]

    def conform(self, ds: xr.Dataset, transpose: bool = False) -> xr.Dataset:
        """
        Conform dataset `ds` to this model

        transpose: whether to automatically transpose the variables in `ds` to conform
            to the specified dimensions.
        """
        list_vars = []
        for var in self.model:
            if isinstance(var, xr.DataArray):
                var = Var(var.name, dtype=var.dtype, dims=var.dims)
            da = ds[var]
            list_vars.append(var.conform(da, transpose=transpose))

        return xr.merge(list_vars)


class Var(str):
    dtype: str | None
    dims: Tuple | None
    dims_like: str | None
    flags: Dict[str, int] | None
    tags: list[str] | None
    attrs: Dict[str, Any]

    def __new__(
        cls,
        name: str,
        *,
        dtype: str | None = None,
        dims: Tuple | None = None,
        dims_like: str | None = None,
        flags: Dict[str, int] | None = None,
        tags: list[str] | None = None,
        attrs: Dict[str, Any] | None = None,
    ):
        """
        Create a `Var` descriptor defining the expected structure of a variable

        `Var` is a lightweight descriptor (a `str` subclass) whose string
        value is the variable name. It carries metadata about the intended
        data type, dimensions and arbitrary attributes, and is used by
        helpers such as `map_blocks`for blockwise processing.

        Args:
            name: The variable name. The `Var` instance itself is a `str` equal
                to this name.
            dtype: Expected NumPy dtype for the variable (e.g. 'float32',
                'uint16'). Used by template creation and validation.
            dims: Expected dimensions as a tuple of dimension names (optional)
                (e.g. ('x', 'y')).
            dims_like: Name of an existing variable in a reference dataset;
                the dimensions will be taken from that variable in the dataset 
                (see `getdims`). Use this to define dimensions "like another Var".
            flags: mapping for bitmask flags (optional),
                e.g. {'CLEAR': 1, 'CLOUDY': 2}.
            tags: Variable tags, used for further selection.
                e.g. tags=['debug', 'level2']
            attrs: Optional dictionary of attributes to attach to the variable
                descriptor, including 'desc' (description), 'units',
                'minv' (minimum valid value), 'maxv' (maximum valid value).
        """
        instance = super().__new__(cls, name)
        # Use object.__setattr__ because str is immutable
        for k, v in {
            'dtype': dtype,
            'dims': dims,
            'dims_like': dims_like,
            'flags': flags,
            'tags': tags,
        }.items():
            object.__setattr__(instance, k, v)
        # Store attrs
        object.__setattr__(instance, 'attrs', attrs or {})
        return instance

    def __repr__(self):
        return f"Var({str(self)!r})"
    
    def unit(self) -> str:
        """
        Return unit
        For backward compatibility
        """
        assert "units" in self.attrs
        assert isinstance(self.attrs["units"], str)
        return self.attrs["units"]

    def desc(self) -> str:
        """
        Returns variable description
        For backward compatibility
        """
        assert "desc" in self.attrs
        assert isinstance(self.attrs["desc"], str)
        return self.attrs["desc"]

    def describe(self):
        """
        Prints the description of the variable.
        """
        lines = [f"Variable: {self}"]
        for key, value in self.attrs.items():
            if value is not None:
                lines.append(f"  {key}: {value}")
        print('\n'.join(lines))

    def getdims(self, ds: xr.Dataset | None = None):
        """
        Get the actual dimensions for that variable.
        If dims is defined as a tuple, it is returned as is.
        If dims_like is defined, the dimensions of the corresponding 
        variable in ds are returned.
        """
        if self.dims is not None:
            return self.dims
        elif self.dims_like is not None:
            assert ds is not None
            return ds[self.dims_like].dims
        elif (ds is not None) and (str(self) in ds.data_vars):
            return ds[str(self)].dims
        else:
            raise TypeError(
                f"Either dims or dims_like must be specified for created variables ({self})"
            )

    def to_template(
        self,
        ds: xr.Dataset,
        meta: xr.Dataset | None = None,
        new_dims: Dict | None = None,
    ):
        """
        Convert to a DataArray with dims infos provided by `ds`

        Args:
            ds: Dataset providing dimension and coordinate information
            meta: Optional dataset containing the variable with the same name,
                used to infer dtype and dims if not specified in the Var object
            new_dims: Dictionary mapping dimension names to their size or coordinates.
                For dimensions not present in `ds`, this parameter is required.
                Values can be:
                - An integer specifying the dimension size (no coordinates)
                - An array-like object (list, numpy array, etc.) providing coordinate
                  values. The dimension size is inferred from the length.
                Example: {'new_dim': 5} or {'new_dim': [0, 1, 2, 3, 4]}

        Returns:
            xr.DataArray: Empty DataArray with appropriate dimensions, chunks, and coordinates
        """
        new_dims = new_dims or {}
        if (meta is not None) and (self in meta):
            actual_dims = meta[self].dims
        else:
            actual_dims = self.getdims(ds)
        shape = []
        chunks = []
        coords = {}
        for d in actual_dims:
            if d in ds.dims:
                shape.append(len(ds[d]))
                if d in ds.chunks:
                    chunks.append(ds.chunks[d])
                else:
                    chunks.append(-1)
            else:
                if d not in new_dims:
                    raise RuntimeError(f'dimension "{d}" has not been described.')
                if hasattr(new_dims[d], "__len__"):
                    n = len(new_dims[d])
                else:
                    n = new_dims[d]
                shape.append(n)
                chunks.append((n,))

            if d in ds.coords and len(ds.coords[d]) == len(ds[d]):
                coords[d] = ds.coords[d].values
            elif (d in new_dims) and hasattr(new_dims[d], "__len__"):
                coords[d] = new_dims[d]
            elif d in ds.dims:
                coords[d] = np.arange(len(ds[d]))

        if self.dtype is None:
            if (meta is not None) and (self in meta):
                dtype = meta[self].dtype
            else:
                raise TypeError(f'Please define the dtype for created variable {self}')
        else:
            dtype = self.dtype

        return xr.DataArray(
            da.empty(shape=shape, dtype=dtype, chunks=chunks),
            dims=actual_dims,
            name=self,
            coords=coords,
        )

    def merge_with(self, other: "Var") -> "Var":
        """
        Merge this Var with another Var, combining their attributes.

        Attributes are merged by preferring non-None values. If both have
        non-None values for the same attribute and they differ, a ValueError
        is raised.
        """
        merged_kwargs = {}
        for attr in [
            "dtype",
            "dims",
            "dims_like",
            "flags",
            "tags",
        ]:
            self_val = getattr(self, attr)
            other_val = getattr(other, attr)
            if self_val is not None and other_val is not None:
                if isinstance(self_val, dict) and isinstance(other_val, dict):
                    # merge dicts
                    merged_dict = self_val.copy()
                    for k, v in other_val.items():
                        if k in merged_dict and merged_dict[k] != v:
                            raise ValueError(
                                f"Conflicting {attr} key '{k}': {merged_dict[k]} vs {v}"
                            )
                        merged_dict[k] = v
                    merged_kwargs[attr] = merged_dict
                elif isinstance(self_val, list) and isinstance(other_val, list):
                    # merge lists
                    merged_list = list(set(self_val + other_val))
                    merged_kwargs[attr] = merged_list
                elif self_val == other_val:
                    merged_kwargs[attr] = self_val
                else:
                    raise ValueError(f"Conflicting {attr}: {self_val} vs {other_val}")
            elif self_val is not None:
                merged_kwargs[attr] = self_val
            elif other_val is not None:
                merged_kwargs[attr] = other_val

        # Merge attrs dict with proper conflict resolution
        merged_attrs = {}
        all_keys = set(self.attrs.keys()) | set(other.attrs.keys())
        for key in all_keys:
            self_val = self.attrs.get(key)
            other_val = other.attrs.get(key)
            if self_val is not None and other_val is not None:
                if isinstance(self_val, dict) and isinstance(other_val, dict):
                    # merge dicts
                    merged_dict = self_val.copy()
                    for k, v in other_val.items():
                        if k in merged_dict and merged_dict[k] != v:
                            raise ValueError(
                                f"Conflicting {key} key '{k}': {merged_dict[k]} vs {v}"
                            )
                        merged_dict[k] = v
                    merged_attrs[key] = merged_dict
                elif isinstance(self_val, list) and isinstance(other_val, list):
                    # merge lists
                    merged_list = list(set(self_val + other_val))
                    merged_attrs[key] = merged_list
                elif self_val == other_val:
                    merged_attrs[key] = self_val
                else:
                    raise ValueError(f"Conflicting {key}: {self_val} vs {other_val}")
            elif self_val is not None:
                merged_attrs[key] = self_val
            elif other_val is not None:
                merged_attrs[key] = other_val

        return Var(self, attrs=merged_attrs, **merged_kwargs)

    def conform(self, da: xr.DataArray, transpose: bool = False) -> xr.DataArray:
        """
        Conform a DataArray to the variable definition
        """
        if self.dims is None:
            raise ValueError(f"dims must be set for Var {self}")
        
        # type check
        if self.dtype is not None and da.dtype != self.dtype:
            raise TypeError(
                f'Expected type "{self.dtype}" for "{self}" but '
                f'encountered "{da.dtype}"'
            )
        
        # dimensions check
        if da.dims != self.dims:
            if set(da.dims) != set(self.dims):
                raise RuntimeError(
                    f'Expected dimensions "{self.dims}" for variable "{self}" '
                    f'but encountered "{da.dims}".'
                )
            if transpose:
                return da.transpose(*self.dims)
            else:
                raise RuntimeError(
                    f'Expected dimensions "{self.dims}" for variable "{self}" '
                    f'but encountered "{da.dims}". Please consider `transpose=True`'
                )
        else:
            return da


def xr_filter(
    ds: xr.Dataset,
    condition: xr.DataArray,
    stackdim: str | None = None,
    transparent: bool = False,
) -> xr.Dataset:
    """
    Extracts a subset of the dataset where the condition is True, stacking the
    `condition` dimensions. Equivalent to numpy's boolean indexing, A[condition].

    Parameters:
    ds (xr.Dataset): The input dataset.
    condition (xr.DataArray): A boolean DataArray indicating where the condition is True.
    stackdim (str, optional): The name of the new stacked dimension. If None, it will be
        determined automatically from the condition dimensions.
    transparent (bool, optional): whether to reassign the original dimension names to
        the Dataset (expanding with length-one dimensions).

    Returns:
    xr.Dataset: A new dataset with the subset of data where the condition is True.
    """
    stackdim = stackdim or "_".join([str(x) for x in condition.dims])
    assert stackdim not in ds.dims
    ok = condition.stack({stackdim: condition.dims})

    # Extract sub Dataset
    stacked = ds.stack({stackdim: condition.dims})
    sub = xr.Dataset()
    for var in stacked:
        da = stacked[var]
        sub[var] = xr.DataArray(
            da.values[
                tuple(ok if (dim == stackdim) else slice(None) for dim in da.dims)
            ],
            dims=da.dims,
            attrs=da.attrs
        )

    # Assign required coordinates to sub
    sub = sub.assign_coords(
        {dim: ds.coords[dim] for dim in ds.coords if dim not in condition.coords}
    )
    
    if transparent:
        # reassign the initial dimension names to the Dataset
        sub = sub.rename({stackdim: condition.dims[0]}).expand_dims(condition.dims[1:])
        sub = sub.transpose(*align_lists([list(ds[x].dims) for x in ds]))
    return sub


def xr_unfilter(
    sub: xr.Dataset,
    condition: xr.DataArray,
    stackdim: str | None = None,
    fill_value_float: float = np.nan,
    fill_value_int: int = 0,
    transparent: bool = False,
    ds_original: xr.Dataset | None = None,
) -> xr.Dataset:
    """
    Reconstructs the original dataset from a subset dataset where the condition is True,
    unstacking the condition dimensions.

    Parameters:
    sub (xr.Dataset): The subset dataset where the condition is True.
    condition (xr.DataArray): A boolean DataArray indicating where the condition is True.
    stackdim (str, optional): The name of the stacked dimension. If None, it will be
        determined automatically from the condition dimensions.
    fill_value_float (float, optional): The fill value for floating point data types.
        Default is np.nan.
    fill_value_int (int, optional): The fill value for integer data types. Default is 0.
    transparent (bool, optional): whether to revert the transparent compatibility
        conversion applied in xrwhere.
    ds_original (xr.Dataset, optional): The original dataset before filtering.
        If provided, variables present in both `sub` and `ds_original` will preserve
        their original values where `condition` is False, instead of being filled
        with fill_value_float/fill_value_int. This is important for pass-through
        variables (e.g. flags) that should not be overwritten.

    Returns:
    xr.DataArray: The reconstructed dataset with the specified dimensions unstacked.
    """
    stackdim = stackdim or "_".join(str(d) for d in condition.dims)

    if transparent:
        sub_dims = align_lists([list(sub[x].dims) for x in sub])
        sub = sub.rename({condition.dims[0]: stackdim}).squeeze([*condition.dims[1:]])
    else:
        sub_dims = []

    assert stackdim in sub.dims
    ok = condition.stack({stackdim: condition.dims})

    # Drop coordinates that are not dimensions
    sub = sub.drop_vars([c for c in sub.coords if c not in sub.dims])

    # remove eventual coords for "stackdim"
    # otherwise these coords would be re-added with "full.assign_coords(sub.coords)""
    if stackdim in sub:
        sub = sub.drop_vars(stackdim)

    # Determine pass-through variables (present in both sub and ds_original)
    passthrough_vars = set()
    if ds_original is not None:
        passthrough_vars = set(sub.data_vars) & set(ds_original.data_vars)

    stacked = xr.Dataset()
    for var in sub:
        # For pass-through variables, stack ds_original[var] directly
        # so original values are preserved where condition is False
        if var in passthrough_vars:
            orig_var = ds_original[var]
            # Only stack condition dims that actually exist in this variable
            cond_dims_in_var = [d for d in condition.dims if d in orig_var.dims]
            if cond_dims_in_var:
                orig_stacked = orig_var.stack({stackdim: cond_dims_in_var})
            else:
                # Variable has no condition dims - just use as-is
                orig_stacked = orig_var
            # In transparent mode, squeeze extra condition dims
            if transparent and len(condition.dims) > 1:
                squeeze_dims = [d for d in condition.dims
                               if d != condition.dims[0] and d in orig_stacked.dims]
                if squeeze_dims:
                    orig_stacked = orig_stacked.squeeze(squeeze_dims)
            var_dims = orig_stacked.dims
            stacked[var] = orig_stacked
        else:
            var_dims = sub[var].dims

            # determine the shape of the stacked array
            stacked_shape = tuple(
                condition.size if dim == stackdim else sub[dim].size
                for dim in var_dims
            )

            # determine fill value
            dtype = sub[var].dtype
            if np.issubdtype(dtype, np.floating):
                fill_value = fill_value_float
            elif np.issubdtype(dtype, np.integer):
                fill_value = fill_value_int
            elif dtype == np.dtype(bool):
                fill_value = False
            else:
                raise ValueError(f"Undefined fill value for type {dtype}")

            # initialize the stacked array
            stacked[var] = xr.DataArray(
                np.full(stacked_shape, fill_value, dtype=dtype),
                dims=var_dims,
                attrs=sub[var].attrs,
            )

        # Transpose sub[var] to match var_dims order, then assign values
        sub_var = sub[var].transpose(*var_dims)
        stacked[var].values[
            tuple(ok if (dim == stackdim) else slice(None) for dim in var_dims)
        ] = sub_var.values

    # unstack the array
    full = stacked.assign_coords({stackdim: ok[stackdim]}).unstack(stackdim)

    # remove all coords
    full = full.drop_vars(full.coords)

    # reassign all required coords
    full = full.assign_coords(sub.coords).assign_coords(condition.coords)

    # reorder the dimensions when in transparent mode
    if transparent:
        # Build explicit dimension list: replace stackdim with condition.dims
        # at the same position, then append any extra dims from coordinates
        full_dims = []
        for d in sub_dims:
            if d == stackdim:
                full_dims.extend(condition.dims)
            else:
                full_dims.append(d)
        # Append any dimensions introduced by assign_coords that aren't already listed
        for d in full.dims:
            if d not in full_dims:
                full_dims.append(d)
        full = full.transpose(*full_dims)

    return full


def xr_filter_decorator(
    argpos: int,
    condition: Callable,
    fill_value_float: float = np.nan,
    fill_value_int: int = 0,
    transparent: bool = False,
    stackdim: str | None = None,
):
    """
    A decorator which applies the decorated function only where the condition is True.

    Args:
        argpos (int): Position index of the input dataset in the decorated function call.
        condition (Callable): A callable taking the Dataset as input and returning a
            boolean DataArray.
        fill_value_float (float, optional): Fill value for floating point data types.
            Default is np.nan.
        fill_value_int (int, optional): Fill value for integer data types.
            Default is 0
        transparent (bool, optional): Whether to reassign the original dimension names
            to the Dataset (expanding with length-one dimensions). Default is False.
        stackdim (str | None, optional): The name of the new stacked dimension.
            If None, it will be determined automatically from the condition dimensions. Default is None.

    Example:
        @xr_filter_decorator(0, lambda x: x.flags == 0)
        def my_func(ds: xr.Dataset) -> xr.Dataset:
            # my_func is applied only where ds.flags == 0
            ...

    The decorator works by:
    1. Extracting a subset of the dataset where the condition is True using `xr_filter`.
    2. Applying the decorated function to the subset.
    3. Reconstructing the original dataset from the subset using `xr_unfilter`.

    Behavior with in-place vs. non-in-place modifications:
    - If the decorated function returns a Dataset (non-in-place), the decorator returns
      the unfiltered result.
    - If the decorated function returns None (in-place modification), the decorator
      updates the original input dataset in-place with the unfiltered modified subset.

    NOTE: this decorator does not guarantee that the order of dimensions is maintained.
    When using this decorator with `xr.apply_blocks`, you may want to wrap your
    xr_filter_decorator decorated method with the `conform` decorator.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ds = args[argpos]
            ok = condition(ds)
            sub = xr_filter(
                ds,
                ok,
                stackdim=stackdim,
                transparent=transparent,
            )
            new_args = tuple(sub if i == argpos else a for i, a in enumerate(args))
            result = func(*new_args, **kwargs)

            # unfilter the result
            # if `func` does in-place modification, then modify the input dataset `ds`
            # in-place with the unfiltered modified input `sub`.
            # otherwise, return the unfiltered result.
            unf = xr_unfilter(
                result or sub,
                ok,
                fill_value_float=fill_value_float,
                fill_value_int=fill_value_int,
                stackdim=stackdim,
                transparent=transparent,
                ds_original=ds,
            )
            if result is not None:
                return unf
            else:
                ds.update(unf)

        return wrapper

    return decorator


def conform(attrname: str, transpose: bool = True):
    """
    A method decorator which applies MapBlocksOutput.conform to the method output.

    The MapBlocksOutput should be an attribute `attrname` of the class.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # get the MapBlocksOutput model
            model = getattr(self, attrname)

            # apply the function
            result = func(self, *args, **kwargs)

            # return the conformed result
            return model.conform(result, transpose=transpose)

        return wrapper

    return decorator

def xr_flat(ds: xr.Dataset) -> xr.Dataset:
    """
    A method which flat a xarray.Dataset on a new dimension named 'index'

    Args:
        ds (xr.Dataset): Dataset to flat
    """
    dims = list(ds.dims)
    assert 'index' not in dims
    flat_ds = ds.stack(index=dims)
    return flat_ds.reset_index(dims).reset_coords(dims)

def xr_sample(ds: xr.Dataset, nb_sample: int|float, seed: int | None = None) -> xr.Dataset:
    """
    A method to extract a subset of sample from a flat xarray.Dataset

    Args:
        ds (xr.Dataset): Input flat dataset
        nb_sample (int|float): Number or percentage of sample to extract
        seed (int, optional): Random seed to use. Defaults to None.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Retrieve index dimension
    size = ds.sizes
    assert len(size) == 1, f'Input dataset should be flatten, got sizes: {len(size)}'
    index_dim = list(size)[0]
    
    # Sample input dataset
    length = size[index_dim]
    if isinstance(nb_sample, float): 
        log.check(nb_sample >= 0 and nb_sample <= 1, 'Invalid number of sample.'
                  f'If float, should be between 0 and 1, got {nb_sample}')
        nb_sample = int(length*nb_sample)
    if nb_sample > length:
        nb_sample = length
    selec = np.random.choice(length, nb_sample, replace=False)
    return ds.isel({index_dim: selec})


def str_to_bool(value: str) -> bool:
    """
    Convert a string representation to a boolean value.
    
    Args:
        value: String value to convert. Case-insensitive comparison with 'true'.
        
    Returns:
        True if the lowercase value equals 'true', False otherwise.
        
    Example:
        >>> str_to_bool('True')
        True
        >>> str_to_bool('false')
        False
        >>> str_to_bool('TRUE')
        True
    """
    return value.lower() == 'true'


def is_dask_backed(data: xr.Dataset, strict: bool = False) -> bool:
    """Check if data variables in a Dataset are dask-backed.

    Returns True if all variables are dask-backed, False if all are numpy-backed.
    In the mixed case (some dask, some numpy), returns True if ``strict=False``
    (any variable is dask-backed), or raises ValueError if ``strict=True``.

    Args:
        data: The input Dataset to check.
        strict: If True, raise ValueError when the Dataset has a mix of
            dask-backed and numpy-backed variables. If False (default),
            return True if any variable is dask-backed.

    Returns:
        True if any/all variables are dask-backed, False if all are numpy-backed.
    """
    if not data.data_vars:
        return False

    dask_vars = []
    numpy_vars = []

    for var in data.data_vars:
        if hasattr(data[var].data, "compute"):
            dask_vars.append(var)
        else:
            numpy_vars.append(var)

    if dask_vars and numpy_vars:
        if strict:
            raise ValueError(
                f"Dataset has a mix of dask-backed and numpy-backed variables. "
                f"Dask variables: {dask_vars}. Numpy variables: {numpy_vars}."
            )
        return True

    return bool(dask_vars)


def is_numpy_backed(obj: xr.Dataset | xr.DataArray) -> bool:
    """
    Check if an xarray Dataset or DataArray is backed by numpy arrays
    (as opposed to dask arrays or other backends).

    For Datasets, recursively checks all data variables and coordinates.

    Note: even when not dask-backed, DataArrays can be backed by 'xarray.core.indexing.MemoryCachedArray',
    which may be inefficient compared to np.ndarray.

    Args:
        obj: An xarray Dataset or DataArray.

    Returns:
        True if all data variables are numpy-backed, False otherwise.
    """
    from xarray.core.indexing import PandasIndexingAdapter

    if isinstance(obj, xr.DataArray):
        return isinstance(obj.variable._data, (np.ndarray, PandasIndexingAdapter))
    elif isinstance(obj, xr.Dataset):
        return all(
            is_numpy_backed(obj[var])
            for var in list(obj.data_vars) + list(obj.coords)
        )
    else:
        raise TypeError(f"Expected xr.Dataset or xr.DataArray, got {type(obj)}")