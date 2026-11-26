from functools import reduce
from itertools import product
from typing import Any, Dict, Iterable, List, Literal, Callable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from hygeos_core.process.blockwise import BlockProcessor
from hygeos_core.tools import Var, is_dask_backed, align_lists, is_numpy_backed


class Interpolator(BlockProcessor):
    """
    Interpolate/select a Dataset onto new coordinates.

    This class is similar to xr.interp and xr.sel, but:
        - Supports dask-based coordinates inputs without triggering immediate
          computation as is done by xr.interp
        - Supports combinations of selection and interpolation. This is faster and more
          memory efficient than performing independently the selection and interpolation.
        - Supports pointwise indexing/interpolation using dask arrays
          (see https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing)
        - Supports per-dimension options (nearest neighbour selection, linear/spline
          interpolation, out-of-bounds behaviour, cyclic dimensions...)
        - Works with full Datasets, allowing interpolation of multiple variables at once.

    The `interp` function is a convenience wrapper around this class for single DataArrays.

    Args:
        data (xr.Dataset): The input Dataset containing variables to interpolate.
        backend (str): computation backend, either "numpy" (default) or "numba".
            The numpy backend uses scipy's find_indices for the index lookup on
            irregular grids. The numba backend accelerates both the index lookup and
            the interpolation (it requires numba to be installed).
        parallel (bool): with the numba backend, parallelize the interpolation over
            the output points. Should not be used when the Interpolator is run from
            multiple threads (e.g. dask's threaded scheduler), to avoid
            oversubscription.
        **kwargs: definition of the selection/interpolation coordinates for each
            dimension, using the following classes:
                - Linear: linear interpolation (like xr.DataArray.interp)
                - Nearest: nearest neighbour selection (like xr.DataArray.sel)
                - Index: integer index selection (like xr.DataArray.isel)
            These classes store the coordinate data in their `.values` attribute and have
            a `.get_indexer` method which returns an indexer for the passed coordinates.

    Example:
        >>> interpolator = Interpolator(
        ...     # input Dataset with variables ozone and wind, dimensions (lat, lon)
        ...     data,
        ...     # perform linear interpolation along dimension `lat`
        ...     # using variable "latitude" in coords_dataset
        ...     # clip out-of-bounds values to the axis min/max.
        ...     lat = Linear("latitude", bounds='clip'),    
        ...     # perform nearest neighbour selection along
        ...     # dimension `lon`, using variable "longitude"
        ...     # in coords 
        ...     lon = Nearest("longitude"),
        ... )
        >>> result = interpolator.map_blocks(coords_dataset)  # coords_dataset has lat and lon variables
        ... # returns a Dataset with variables "ozone" and "wind", interpolated over the
        ... # coordinates lat and lon provided in coords_dataset
    """

    def __init__(
        self,
        data: xr.Dataset,
        backend: Literal["numpy", "numba"] = "numpy",
        parallel: bool = False,
        **kwargs,
    ):
        check_backend(backend)
        self.backend = backend
        self.parallel = parallel

        # get interpolators along all dimensions
        self.indexers = {}
        self.varnames = {}  # mapping of data coords to block variables
        for k, v in kwargs.items():
            varname = v.varname or k
            self.indexers[k] = v.get_indexer(data[k], backend=backend)
            self.varnames[k] = varname

        # ensure data is numpy-backed
        if not is_numpy_backed(data):
            raise TypeError(
                "Interpolator requires numpy-backed data. Use .compute() on the input Dataset."
            )
        self.data: xr.Dataset = data
    
    def input_vars(self) -> List[Var]:
        return [Var(x) for x in self.varnames.values()]

    def created_vars(self) -> List[Var]:
        return [Var(str(x), attrs=self.data[x].attrs) for x in self.data]
    
    def created_dims(self) -> Dict[str, Any]:
        cdims = {}
        for dim in [d for d in self.data.dims if d not in self.indexers]:
            if dim in self.data.coords:
                cdims[dim] = self.data.coords[dim].values
            else:
                cdims[dim] = self.data.dims[dim]
                
        return cdims

    def auto_template(self) -> bool:
        return True
    
    def check(self, ds: xr.Dataset):
        # Check that shared (non-interpolated) dimensions have consistent coordinates
        for dim in self.data.dims:
            if dim not in self.indexers and dim in ds.dims:
                if dim in self.data.coords and dim in ds.coords:
                    if not np.array_equal(self.data.coords[dim].values, ds.coords[dim].values):
                        raise ValueError(
                            f"Shared dimension '{dim}' has inconsistent coordinates "
                            f"between the interpolation data and the input dataset."
                        )

    def process_block(self, block: xr.Dataset):
        """
        ex: block is a xr.Dataset with
           latitude (x, y)
           longitude (x, y)
        Returns interpolated_data, interpolated over the coordinates provided in `block`.
        interpolated_data is of the same type as data
        """
        # Check that all indexers are present as variables in block_coords
        for dim in self.varnames.values():
            if dim not in block:
                raise KeyError(
                    f"Interpolated dimension '{dim}' is not present in the "
                    f"provided coordinate dataset {list(block)}"
                )
        # take the subset of block that corresponds to used coordinate variables
        block_sub = block[list(self.varnames.values())]

        # Determine output dimensions for each variable
        out_dims = {
            k: determine_output_dimensions(self.data[k], self.varnames, block_sub)
            for k in self.data.data_vars
        }
        global_out_dims = align_lists(list(out_dims.values()))
        new_out_dims = [x for x in global_out_dims if x not in self.data]

        # get broadcasted data from block_coords (all with the same number of dimensions)
        np_indexers = broadcast_numpy(block_sub, global_out_dims)

        # apply index searching over all dimensions (ie, v(values))
        indices_weights = {
            k: v(np_indexers[self.varnames[k]]) for k, v in self.indexers.items()
        }

        # copy relevant dims from block_sub (interpolating coords) first,
        # then from self.data for shared non-interpolated dimensions.
        coords = xr.Dataset()
        for dim in global_out_dims:
            if (dim not in coords) and (dim in block_sub.coords):
                coords[dim] = block_sub.coords[dim]
            elif (dim not in coords) and (dim in self.data.coords):
                coords[dim] = self.data.coords[dim]
            elif (dim not in coords) and (dim in block_sub.dims):
                coords[dim] = np.arange(block_sub.sizes[dim])

        # Order of coordinate dims as they appear in the index arrays
        # (global_out_dims filtered to dims present in block_sub)
        coord_dims_ordered = [d for d in global_out_dims if d in block_sub.dims]

        # numba backend: pack the indices and weights once for all variables
        packed = None
        if self.backend == "numba":
            packed = pack_terms(
                indices_weights,
                dict(self.data.sizes),
                coord_dims_ordered,
                {
                    d: block_sub.sizes[d] for d in coord_dims_ordered
                    if (d in self.data.dims) and (d not in self.indexers)
                },
            )

        # var loop
        for var in self.data:

            # list current "indexing" dimensions
            current_dims = [
                d
                for d in global_out_dims
                if d in list(self.data[var].dims) + new_out_dims
            ]

            w_shape = {
                k: broadcast_shapes(block_sub, current_dims)[self.varnames[k]]
                for k in self.indexers
            }

            # Shared dims: non-interpolated dims of this variable that are also
            # dims of the coordinate variables. These need element-wise range
            # indexing instead of slice(None) to avoid broadcasting all values.
            shared_dims_info = {
                d: coord_dims_ordered.index(d)
                for d in self.data[var].dims
                if d not in self.indexers and d in coord_dims_ordered
            }

            # get the shapes for broadcasting each weight against the output dimensions
            result = None
            if packed is not None:
                result = interp_var_numba(
                    self.data[var],
                    packed,
                    out_dims[var],
                    coord_dims_ordered,
                    shared_dims_info,
                    parallel=self.parallel,
                )
            if result is None:
                result = interp_var(
                    self.data[var],
                    indices_weights,
                    w_shape,
                    out_dims[var],
                    current_dims,
                    shared_dims_info,
                )

            block[var] = xr.DataArray(
                result,
                dims=out_dims[var],
                coords={k: v for k, v in coords.coords.items() if k in out_dims[var]},
                attrs=self.data[var].attrs,
            )


def check_backend(backend: str):
    """
    Check that `backend` is valid, and that numba is installed for the numba backend
    """
    if backend == "numpy":
        return
    elif backend == "numba":
        try:
            import hygeos_core.interpolate_numba  # noqa: F401
        except ImportError:
            raise ImportError(
                'numba backend requested but numba is not installed. '
                'Install with `pip install "hygeos-core[numba]"` or use backend="numpy".'
            )
    else:
        raise ValueError(
            f'Unknown backend "{backend}". Expected "numpy" or "numba".'
        )


def find_indices(grid, xi, backend: str = "scipy") -> tuple[NDArray[np.intp], NDArray[np.float32]]:
    """
    Multi-dimensional grid interpolation index finding.

    Args:
        grid: Tuple of 1D arrays defining grid coordinates for each dimension.
        xi: 2D array where each row represents a dimension and each column a query point.
        backend: One of "scipy" (default) or "numba".

    Returns:
        indices: Grid interval indices for each query point in each dimension.
        distances: Normalized distances within each interval.

    Raises:
        ImportError: If backend is "numba" but numba is not installed.
        ValueError: If backend is not recognized.
    """
    if backend == "scipy":
        from scipy.interpolate._rgi_cython import find_indices as _fi
    elif backend == "numba":
        try:
            from hygeos_core.interpolate_numba import find_indices_numba as _fi
        except ImportError:
            raise ImportError(
                'numba backend requested but numba is not installed. '
                'Install with `pip install "hygeos-core[numba]"` or use backend="scipy".'
            )
    else:
        raise ValueError(
            f'Unknown find_indices backend "{backend}". '
            'Expected "scipy" or "numba".'
        )
    return _fi(grid, xi)


def interp_var(
    da: xr.DataArray,
    indices_weights: dict,
    w_shape: dict,
    current_out_dims: list,
    current_dims: list,
    shared_dims_info: dict | None = None,
) -> NDArray[Any]:
    """
    Interpolate a single variable `da`, with indices and weights

    shared_dims_info maps dim names that are shared between `da` and the
    coordinate variables to their position in the index arrays. These dims
    need element-wise range indexing instead of slice(None).
    """
    # cartesian product of the combination of lower and upper indices (in case of
    # linear interpolation) for each dimension
    result = None
    data_values = da.values
    shared_dims_info = shared_dims_info or {}

    current_indices_weights = {
        k: v for k, v in indices_weights.items() if k in da.dims
    }

    if not current_indices_weights:
        # current `da` contains no interpolated dimension: return as-is
        return da.data
    
    # get indices/weights for current DataArray
    for iw in product_dict(**current_indices_weights):
        weights = [
            1 if iw[dim][1] is None else iw[dim][1].reshape(w_shape[dim])
            for dim in iw
        ]
        w = reduce(lambda x, y: x*y, weights)
        # Build indexing keys for each dim of da:
        # - interpolated dims: use the fancy index from iw
        # - shared dims (also coord dims): use element-wise range index to
        #   avoid selecting all elements along that dim
        # - other dims: slice(None) to keep all elements
        any_idx = iw[next(iter(iw))][0]
        keys = []
        for dim in da.dims:
            if dim in iw:
                keys.append(iw[dim][0])
            elif dim in shared_dims_info:
                pos = shared_dims_info[dim]
                size = any_idx.shape[pos] if pos < len(any_idx.shape) else 1
                n_dims = len(any_idx.shape)
                shape = tuple(size if i == pos else 1 for i in range(n_dims))
                keys.append(np.arange(size).reshape(shape))
            else:
                keys.append(slice(None))
        term = data_values[tuple(keys)] * w
        if result is None:
            result = term
        else:
            result += term

    # result has dimensions as in global_out_dims
    # squeeze it so that we keep dimensions out_dims[var]
    assert result is not None
    result = result.squeeze(
        axis=tuple(
            [
                i
                for (i, dim) in enumerate(current_dims)
                if dim not in current_out_dims
            ]
        )
    )

    return result


class PackedTerms:
    """
    Indices and weights of all interpolation terms, packed in flat buffers for the
    numba interpolation kernel (see `pack_terms`)

    The index and weight arrays of each term are stored with their own (compact)
    shape, with `rank` dimensions (the dimensions of the output points), and are
    broadcasted in the kernel by using zero strides.
    """
    def __init__(self, rank: int):
        self.rank = rank
        self.ibuf = []        # list of flat index arrays (concatenated at the end)
        self.wbuf = [np.ones(1)]  # flat weight arrays; element 0 is the weight 1
        self.isize = 0
        self.wsize = 1
        self.iptr, self.wptr = [], []  # start of each term in ibuf/wbuf
        self.ishape, self.wshape = [], []  # shapes of each term in ibuf/wbuf
        self.dim_terms = {}   # list of term indices, for each dimension
        self.dim_dtypes = {}  # list of weight dtypes, for each dimension

    def add(self, dim, idx, w):
        idx = np.ascontiguousarray(idx, dtype=np.intp)
        assert idx.ndim <= self.rank
        self.iptr.append(self.isize)
        self.ishape.append((1,) * (self.rank - idx.ndim) + idx.shape)
        self.ibuf.append(idx.ravel())
        self.isize += idx.size
        if w is None:
            self.wptr.append(0)
            self.wshape.append((1,) * self.rank)
        else:
            self.dim_dtypes.setdefault(dim, []).append(np.asarray(w).dtype)
            w = np.ascontiguousarray(w, dtype=np.float64)
            assert w.ndim <= self.rank
            self.wptr.append(self.wsize)
            self.wshape.append((1,) * (self.rank - w.ndim) + w.shape)
            self.wbuf.append(w.ravel())
            self.wsize += w.size
        self.dim_terms.setdefault(dim, []).append(len(self.iptr) - 1)

    def finalize(self):
        self.ibuf = np.concatenate(self.ibuf)
        self.wbuf = np.concatenate(self.wbuf)
        self.iptr = np.array(self.iptr, dtype=np.intp)
        self.wptr = np.array(self.wptr, dtype=np.intp)
        self.ishape = np.array(self.ishape, dtype=np.intp).reshape(-1, self.rank)
        self.wshape = np.array(self.wshape, dtype=np.intp).reshape(-1, self.rank)


def _broadcast_strides(shapes: NDArray) -> NDArray:
    """
    C-order element strides of arrays of shapes `shapes` (one array per row), with
    0 for the dimensions of length 1 (broadcasting)
    """
    st = np.ones_like(shapes)
    for a in range(shapes.shape[1] - 2, -1, -1):
        st[:, a] = st[:, a + 1] * shapes[:, a + 1]
    return np.where(shapes == 1, 0, st)


def pack_terms(
    indices_weights: dict,
    sizes: dict,
    coord_dims: list,
    shared_dims: dict,
) -> PackedTerms | None:
    """
    Pack the indices and weights of all interpolated dimensions (and of the shared
    dimensions) for the numba interpolation kernel.

    Args:
        indices_weights: for each interpolated dimension, list of (index, weight)
            tuples. The index and weight arrays have the dimensions `coord_dims`
            (possibly with length 1).
        sizes: size of each dimension of the interpolated data
        coord_dims: dimensions of the index arrays
        shared_dims: dimensions that are shared between the interpolated data and
            the coordinate variables, and their size in the coordinate variables.
            They are indexed element-wise (as in `interp_var`).

    Returns None if this case is not supported by the numba kernel.
    """
    # rank of the output points (at least 1, for scalar coordinates)
    rank = max(len(coord_dims), 1)
    packed = PackedTerms(rank)
    for dim, iw in indices_weights.items():
        n = sizes[dim]
        for idx, w in iw:
            idx = np.asarray(idx)
            if (idx.dtype.kind not in "iu") or (idx.ndim > rank) or (
                (w is not None) and (np.ndim(w) > rank)
            ):
                return None
            if idx.size and (idx.min() < 0):
                # negative indices: same as numpy indexing
                idx = np.where(idx < 0, idx + n, idx)
            if idx.size and ((idx.min() < 0) or (idx.max() >= n)):
                raise IndexError(
                    f"index out of bounds for dimension {dim} with size {n}"
                )
            packed.add(dim, idx, w)

    # shared dims: element-wise index, with weight 1
    for dim, size in shared_dims.items():
        if size > sizes[dim]:
            raise IndexError(
                f"index out of bounds for dimension {dim} with size {sizes[dim]}"
            )
        pos = coord_dims.index(dim)
        shp = [1] * len(coord_dims)
        shp[pos] = size
        packed.add(dim, np.arange(size).reshape(shp), None)

    packed.finalize()
    return packed


def interp_var_numba(
    da: xr.DataArray,
    packed: PackedTerms,
    current_out_dims: list,
    coord_dims: list,
    shared_dims_info: dict | None = None,
    parallel: bool = False,
) -> NDArray[Any] | None:
    """
    Interpolate a single variable `da` (numba backend)

    Same result as `interp_var` (up to floating point rounding), but the weighted sum
    over the corners of the interpolation cell is computed in a single pass by a
    numba kernel, which gathers the data directly with their strides (no
    intermediate arrays for each corner).

    Args:
        da: the data to interpolate
        packed: indices and weights of all interpolation terms (see `pack_terms`)
        current_out_dims: output dimensions of this variable
        coord_dims: dimensions of the index arrays
        shared_dims_info: maps dim names that are shared between `da` and the
            coordinate variables to their position in the index arrays.
        parallel: use the parallel kernel

    Returns None if this case is not supported by the numba kernel (the caller
    should then use `interp_var`).
    """
    from hygeos_core.interpolate_numba import interp_kernel_parallel, interp_kernel_serial

    shared_dims_info = shared_dims_info or {}
    interp_dims = [d for d in packed.dim_terms if d in da.dims]
    interp_dims_weighted = [d for d in interp_dims if d in packed.dim_dtypes]
    if not interp_dims:
        # current `da` contains no interpolated dimension: return as-is
        return da.data

    # Unsupported cases
    if not interp_dims_weighted:
        # pure selection: no weights
        return None
    if da.dtype.kind not in "fiu":
        return None

    data = np.ascontiguousarray(da.values)
    strides = [s // data.itemsize for s in data.strides]

    # dimensions to expand: interpolated dims, then shared dims
    dims = [d for d in interp_dims if d not in shared_dims_info] + [
        d for d in shared_dims_info if d in da.dims
    ]
    terms = {d: packed.dim_terms[d] for d in dims}
    allterms = [t for d in dims for t in terms[d]]

    # shape of the output points: broadcast of the shapes of all terms
    R = packed.rank
    S = tuple(int(x) for x in np.broadcast_shapes(
        *[tuple(packed.ishape[t]) for t in allterms],
        *[tuple(packed.wshape[t]) for t in allterms],
    ))

    # choose the inner axis of the kernel loop: the one along which most dimensions
    # have invariant terms (their expansion is then done once for all points along
    # this axis), defaulting to the last axis. It is moved last (perm).
    def invariant(d, a):
        return all(
            (packed.ishape[t, a] == 1) and (packed.wshape[t, a] == 1)
            for t in terms[d]
        )
    ninv_max, inner_axis = 0, R - 1
    for a in range(R):
        if S[a] > 1:
            ninv_a = sum(invariant(d, a) for d in dims)
            if ninv_a > ninv_max:
                ninv_max, inner_axis = ninv_a, a
    perm = [a for a in range(R) if a != inner_axis] + [inner_axis]

    # dims invariant along the inner axis first
    dims = [d for d in dims if invariant(d, inner_axis)] + [
        d for d in dims if not invariant(d, inner_axis)
    ]
    ninv = sum(invariant(d, inner_axis) for d in dims)
    allterms = np.array([t for d in dims for t in terms[d]], dtype=np.intp)
    term_count = np.array([len(terms[d]) for d in dims], dtype=np.intp)
    term_start = np.concatenate([[0], np.cumsum(term_count)[:-1]]).astype(np.intp)
    t_dstride = np.array(
        [strides[da.dims.index(d)] for d in dims for _ in terms[d]], dtype=np.intp
    )

    # non-interpolated dimensions of `da` (kept in the output)
    kept_dims = [
        d for d in da.dims
        if (d not in packed.dim_terms) and (d not in shared_dims_info)
    ]
    kept_shape = tuple(da.sizes[d] for d in kept_dims)
    kept_offsets = np.zeros(kept_shape, dtype=np.intp)
    for i, d in enumerate(kept_dims):
        shp = [1] * len(kept_dims)
        shp[i] = da.sizes[d]
        kept_offsets = kept_offsets + (
            np.arange(da.sizes[d]) * strides[da.dims.index(d)]
        ).reshape(shp)
    kept_offsets = kept_offsets.ravel()
    M = len(kept_offsets)

    dtype = np.result_type(
        data.dtype, *[dt for d in interp_dims_weighted for dt in packed.dim_dtypes[d]]
    )

    # output, of shape S + kept_shape (C-order)
    out = np.empty(int(np.prod(S)) * M, dtype=np.float64)
    out_str = np.array(
        [int(np.prod(S[a + 1:])) * M for a in range(R)], dtype=np.intp
    )

    kernel = interp_kernel_parallel if parallel else interp_kernel_serial
    kernel(
        data.reshape(-1),
        packed.ibuf,
        packed.wbuf,
        packed.iptr[allterms],
        np.ascontiguousarray(_broadcast_strides(packed.ishape)[allterms][:, perm]),
        packed.wptr[allterms],
        np.ascontiguousarray(_broadcast_strides(packed.wshape)[allterms][:, perm]),
        t_dstride,
        term_start,
        term_count,
        ninv,
        kept_offsets,
        np.array(S, dtype=np.intp)[perm],
        out,
        out_str[perm],
    )

    # reshape to (coord_dims..., kept_dims...), then to current_out_dims
    # (for scalar coordinates, the output points have a single dimension of length 1)
    result = out.reshape(S[R - len(coord_dims):] + kept_shape)
    dims_all = list(coord_dims) + kept_dims
    squeeze = tuple(i for i, d in enumerate(dims_all) if d not in current_out_dims)
    result = result.squeeze(axis=squeeze)
    dims_all = [d for d in dims_all if d in current_out_dims]
    result = result.transpose([dims_all.index(d) for d in current_out_dims])

    return result.astype(dtype, copy=False)


def interp(da: xr.DataArray, backend: str = "numpy", **kwargs) -> xr.DataArray:
    """
    Interpolate/select a DataArray onto new coordinates.

    This function is similar to xr.interp and xr.sel, but:
        - Supports dask-based coordinates inputs without triggering immediate
          computation as is done by xr.interp
        - Supports combinations of selection and interpolation. This is faster and more
          memory efficient than performing independently the selection and interpolation.
        - Supports pointwise indexing/interpolation using dask arrays
          (see https://docs.xarray.dev/en/latest/user-guide/indexing.html#more-advanced-indexing)
        - Supports per-dimension options (nearest neighbour selection, linear/spline
          interpolation, out-of-bounds behaviour, cyclic dimensions...)

    Args:
        da (xr.DataArray): The input DataArray
        backend (str): computation backend, either "numpy" (default) or "numba"
            (see Interpolator).
        **kwargs: definition of the selection/interpolation coordinates for each
            dimension, using the following classes:
                - Linear: linear interpolation (like xr.DataArray.interp)
                - Nearest: nearest neighbour selection (like xr.DataArray.sel)
                - Index: integer index selection (like xr.DataArray.isel)
            These classes store the coordinate data in their `.values` attribute and have
            a `.get_indexer` method which returns an indexer for the passed coordinates.

    Example:
        >>> interp(
        ...     data,  # input DataArray with dimensions (a, b, c)
        ...     a = Linear(           # perform linear interpolation along dimension `a`
        ...          a_values,        # `a_values` is a DataArray with dimension (x, y);
        ...          bounds='clip'),  # clip out-of-bounds values to the axis min/max.
        ...     b = Nearest(b_values), # perform nearest neighbour selection along
        ...                            # dimension `b`; `b_values` is a DataArray
        ...                            # with dimension (x, y)
        ... ) # returns a DataArray with dimensions (x, y, c)
        No interpolation or selection is performed along dimension `c` thus it is
        left as-is.

    Returns:
        xr.DataArray: DataArray on the new coordinates.
    """
    # Build ds_coords from the interpolating DataArrays, preserving their coordinates
    ds_coords_data = {}
    for k, v in kwargs.items():
        v.varname = k
        if hasattr(v, 'values') and isinstance(v.values, xr.DataArray):
            ds_coords_data[k] = v.values
        else:
            ds_coords_data[k] = v.values

    ds_coords = xr.Dataset(ds_coords_data)

    # Ensure dimensions of interpolating DataArrays become coordinates
    # (use the dimension values as coordinates if not already present)
    for var_name, var in ds_coords.data_vars.items():
        for dim in var.dims:
            if dim not in ds_coords.coords:
                ds_coords = ds_coords.assign_coords({dim: np.arange(ds_coords.sizes[dim])})

    interpolator = Interpolator(da.to_dataset(name="dummy"), backend=backend, **kwargs)

    # Use map_blocks for dask-backed coords (lazy/chunked evaluation),
    # or process_block directly for numpy-backed coords (no dask overhead).
    if is_dask_backed(ds_coords):
        result = interpolator.map_blocks(ds_coords)["dummy"]
    else:
        interpolator.process_block(ds_coords)
        result = ds_coords["dummy"]

    return result


def product_dict(**kwargs) -> Iterable[Dict]:
    """
    Cartesian product of a dictionary of lists
    """
    # https://stackoverflow.com/questions/5228158/
    keys = kwargs.keys()
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def _nan_like(values: NDArray):
    """
    NaN with the type of `values` if it is a floating point array (float64 otherwise,
    as when NaNs are introduced by numpy in an integer array)
    """
    return values.dtype.type(np.nan) if values.dtype.kind == "f" else np.float64(np.nan)


class Locator:
    """
    The purpose of these classes is to locate values in coordinate axes.
    """
    def __init__(self, coords: NDArray, bounds: str, backend: str = "numpy"):
        self.coords = coords.astype("double")
        self.bounds = bounds
        self.backend = backend

        if coords[0] < coords[-1]:
            self.vmin = coords[0]
            self.vmax = coords[-1]
        else:
            self.vmin = coords[-1]
            self.vmax = coords[0]

        if bounds == "cycle":
            raise RuntimeError('Cannot use bounds="cycle" with irregular locator')


    def handle_oob(self, values: np.ndarray):
        """
        handle out of bound values

        Note: when bounds == "cycle", does nothing
        """
        if self.bounds == "clip":
            # FIXME:
            # manage the case where coords and values
            # have incompatible types (eg, int and float)
            values = values.clip(self.vmin, self.vmax)
        elif self.bounds in ["nan", "error"]:
            # bounds is either "nan" or "error"
            oob = (values < self.vmin) | (values > self.vmax)
            if oob.any():
                if self.bounds == "error":
                    raise ValueError(f"Some values (range {values.min()} to {values.max()}) "
                                     f"are out of bounds for the coordinate axis (range {self.vmin} to {self.vmax}).")
                if self.bounds == "nan":
                    values = np.where(oob, np.nan, values)

        return values
        
    
    def locate_index_weight(self, values):
        """
        Find indices and dist of values for linear and spline interpolation in self.coords

        Returns a list of indices, dist (float 0 to 1) and oob
        """
        if self.backend == "numba":
            from hygeos_core.interpolate_numba import locate_irregular, OOB_MODES
            values = np.asarray(values)
            indices, dist, n_oob = locate_irregular(
                values.ravel(), self.coords, self.vmin, self.vmax,
                OOB_MODES[self.bounds], _nan_like(values),
            )
            if n_oob:
                self.handle_oob(values)  # raises the error
            return indices.reshape(values.shape), dist.reshape(values.shape)

        values = self.handle_oob(values)

        shp = values.shape
        indices, dist = find_indices(
            (self.coords,), values.astype("double").ravel()[None, :],
            backend="scipy",
        )
        indices = indices.reshape(shp).clip(0, None) # Note: a clip is performed because
                                                     # -1 is obtained in presence of NaN
        dist = dist.reshape(shp)

        return indices, dist

    def locate_index(self, values):
        raise NotImplementedError
    
    
class Locator_Regular(Locator):
    def __init__(
        self,
        coords,
        bounds: str,
        inversion_func: Callable | None = None,
        period=None,
        backend: str = "numpy",
    ):
        self.inversion_func = inversion_func
        if inversion_func is None:
            self.coords = coords
        else:
            self.coords = inversion_func(coords)
        self.bounds = bounds
        self.backend = backend

        # Check that the coords are regular, otherwise raise a ValueError
        diff = np.diff(self.coords)
        self.is_time = 'timedelta' in diff.dtype.name
        if self.is_time:
            regular = np.allclose(diff[0], diff, rtol=1e-5, atol=np.timedelta64(0, "s"))
        else:
            regular = np.allclose(diff[0], diff)
        if not regular:
            raise ValueError

        self.vstart = self.coords[0]
        self.vend = self.coords[-1]
        if self.vstart < self.vend:
            self.vmin = self.vstart
            self.vmax = self.vend
        else:
            self.vmin = self.vend
            self.vmax = self.vstart
        self.N = len(self.coords)

        deltat = (self.vend - self.vstart)
        if self.is_time:
            deltat = deltat / np.timedelta64(1, "s")
        self.scal = (self.N - 1) / deltat

        if (period is not None) and (not np.isclose(period, self.N / self.scal)):
            raise AssertionError(
                f"Expected a period of {period}, but a period of {self.N/self.scal} "
                "has been derived from the coords"
            )
    
    def locate_index_weight(self, values):
        """
        Optimized version of locate_index_weight for regular grid coords
        """
        if self.inversion_func is not None:
            values = self.inversion_func(values)

        if (self.backend == "numba") and not self.is_time:
            from hygeos_core.interpolate_numba import locate_regular, OOB_MODES
            values = np.asarray(values)
            i_inf, dist, n_oob = locate_regular(
                values.ravel(), self.vstart, self.vmin, self.vmax, self.scal,
                self.N, OOB_MODES[self.bounds], _nan_like(values),
            )
            if n_oob:
                self.handle_oob(values)  # raises the error
            return i_inf.reshape(values.shape), dist.reshape(values.shape)

        values = self.handle_oob(values)
        
        # calculate floating index (scale to [0, N-1])
        if self.is_time:
            x = (values - self.vstart) * self.scal / np.timedelta64(1, "s")
        else:
            x = (values - self.vstart) * self.scal
        i_inf = np.floor(np.nan_to_num(x)).astype("int")

        dist = x - i_inf

        if self.bounds != "cycle":
            # Deals with indexing the last coordinate item
            mask_idk = i_inf == self.N - 1
            i_inf = np.where(mask_idk, i_inf - 1, i_inf)
            dist = np.where(mask_idk, np.float64(1), dist)
        
        return i_inf, dist
    
    #For Nearest ?
    def locate_index(self, values):
        raise NotImplementedError


def create_locator(
    coords, bounds: str, spacing, period: float | None = None,
    backend: str = "numpy"
) -> Locator:
    """
    Locator factory

    The purpose of this method is to instantiate the appropriate "Locator" class.

    The args are passed from the indexers.
    """
    if spacing != "irregular":
        # regular, auto, or callable
        ifunc = spacing if callable(spacing) else None
        try:
            return Locator_Regular(coords, bounds, inversion_func=ifunc,
                                   period=period, backend=backend)
        except ValueError:
            if spacing == "auto":
                pass
            else:  # regular or callable: raise the exception
                raise

    return Locator(coords, bounds, backend=backend)


class Spline:
    def __init__(
        self, 
        var: str | xr.DataArray | None, 
        tension = 0.5, 
        bounds: Literal["error", "nan", "clip"] = "error", 
        spacing: Literal["regular", "irregular", "auto"]|Callable[[float],float] = "auto", 
    ):
        """
        A proxy class for Spline indeting.

        The purpose of this class is to provide a convenient interface to the
        interp function, by initializing an indexer class.

        Args:
            bounds (str): how to deal with out-of-bounds values:
                - error: raise a ValueError
                - nan: replace by NaNs
                - clip: clip values to the extrema
            spacing: how to deal with regular grids
                - "regular": assume a regular grid, raise an error if not
                - "irregular": assume an irregular grid
                - "auto": detect if grid is regular or not
                - if a function is provided, it is assumed that the grid is regular
                after applying this function. 
                (for example if the coords follow x² you need to feed sqrt)
            tension (float): how tight the rope between points is (1 very tight, 0 very loose)
                - 0: Very tight rope
                - 1: Very lose rope
                The points between indexes 0 and 1 and N - 1 and N will have a tightness of 0.5
        """
        if isinstance(var, str):
            self.varname = var
            self.values = None
        else:
            self.varname = None
            self.values = var
        self.bounds = bounds
        self.tension = tension
        self.spacing = spacing
    
    def get_indexer(self, coords: xr.DataArray, backend: str = "numpy"):
        cval = coords.values
        if len(cval) < 3 :
            raise ValueError("You need to have at least 3 points to spline in between") #AJOUTER LA BONNE ERREUR
        return Spline_Indexer(cval, self.bounds, self.spacing, self.tension, backend=backend)


class Spline_Indexer:
    def __init__(self, coords: NDArray, bounds: str, spacing, tension: float, backend: str = "numpy"):
        self.coords = coords
        self.spacing = spacing
        self.bounds = bounds

        self.tension_matrix_full = np.array([
            [0,             1,              0,                  0],
            [-tension,      0,              tension,            0],
            [2 * tension,   -5 * tension,   4 * tension,        -tension],
            [-tension,      3 * tension,    -3 * tension,       tension],
        ])
        
        self.tension_matrix_early = np.array([
            [1, 0, 0, 0],
            [-1.5, 2, -0.5, 0],
            [0.5, -1, 0.5, 0],
            [0, 0, 0, 0],
        ])
        
        self.tension_matrix_late = np.array([
            [0, 1, 0, 0],
            [-0.5, 0, 0.5, 0],
            [0.5, -1, 0.5, 0],
            [0, 0, 0, 0],
        ])

        self.locator = create_locator(
            coords=self.coords,
            bounds=self.bounds,
            spacing=self.spacing,
            backend=backend,
        )
        
    def __call__(self, values):
        """
        Find indices of values for linear interpolation in self.coords

        Returns a list of tuples [(idx_inf, weights), (idx_sup, weights)]
        """
        
        N_min_1, dist = self.locator.locate_index_weight(values)
        
        N = len(self.coords)
        
        #init indices (-2 -1 +1 +2)
        indices = np.array([
            np.clip(N_min_1 - 1, -1, N), 
            N_min_1,
            np.clip(N_min_1 + 1, 0, N),
            np.clip(N_min_1 + 2, 0, N),
        ])
        
        mask_first_negative = indices[0] < 0 #early (-1 +1 +2)
        
        mask_last_oob = indices[3] > (N - 1) #late (-2 -1 +1)

        mask_normal = ~(mask_first_negative | mask_last_oob) #middle (-2 -1 +1 +2)
        
        # Initialize weights matrix
        weights = np.zeros((self.tension_matrix_full.shape[1], *dist.shape))
        # Calculate and assign weights based on masks
        if mask_normal.any():
            t_normal = dist[mask_normal]
            x_vector_full = np.vstack((np.ones_like(t_normal), t_normal, t_normal ** 2, t_normal ** 3)).T
            weights[:, mask_normal] = np.dot(self.tension_matrix_full.T, x_vector_full.T)
        
        if mask_first_negative.any():
            t_early = dist[mask_first_negative]
            x_vector_half_early = np.vstack((np.ones_like(t_early), t_early, t_early ** 2, np.zeros_like(t_early))).T
            weights[:, mask_first_negative] = np.dot(self.tension_matrix_early.T, x_vector_half_early.T)
        
        if mask_last_oob.any():
            t_late = dist[mask_last_oob]
            x_vector_half_late = np.vstack((np.ones_like(t_late), t_late, t_late ** 2, np.zeros_like(t_late))).T
            weights[:, mask_last_oob] = np.dot(self.tension_matrix_late.T, x_vector_half_late.T)

        #we change indices like we changed the weights
        if len(indices.shape) == 1:
            for i in range(1, 4):
                if(mask_first_negative):
                    indices[i - 1] = indices[i]
            if(mask_last_oob):
                 indices[3] = 0
        else:
            for i in range(1, 4):
                indices[i - 1][mask_first_negative] = indices[i][mask_first_negative]
            indices[3][mask_last_oob] = 0
        
            
        return [(indices[i], weights[i]) for i in range(len(weights))]
    
        

class Linear:
    def __init__(
        self,
        var: str | xr.DataArray | None,
        bounds: Literal["error", "nan", "clip", "cycle"] = "error",
        spacing: Literal["regular", "irregular", "auto"]
        | Callable[[float], float] = "auto",
        period: float | None = None,
    ):
        """
        A proxy class for Linear indexing.

        The purpose of this class is to provide a convenient interface to the
        interp function, by initializing an indexer class.

        Args:
            var (str or DataArray): the variable name or values to interpolate along.
            bounds (str): how to deal with out-of-bounds values:
                - "error": raise a ValueError
                - "nan": replace by NaNs
                - "clip": clip values to the extrema
                - "cycle": the axis is considered cyclic
                    e.g.: longitudes between -180 and 179
                    allows indexing values in the range of [-180, 180] or even [0, 360].
                    The "period" argument can be used to check the period inferred
                    from the coord values, for example period=360.
            spacing: how to deal with regular grids
                - "regular": assume a regular grid, raise an error if not
                - "irregular": assume an irregular grid
                - "auto": detect if grid is regular or not
                - if a function is provided, it is assumed that the grid is regular
                  after applying this function. 
                  (for example if the coords follow x² you need to feed sqrt)
                The index lookup is optimized for regular grids, using the
                Locator_Regular class.
            period (optional, float): if provided, we verify that the period inferred
                from the coordinates is equal to this value.
                (only when bounds="cycle")
        """
        if isinstance(var, str):
            self.varname = var
            self.values = None
        else:
            self.varname = None
            self.values = var
        self.bounds = bounds
        self.spacing = spacing
        self.period = period

    def get_indexer(self, coords: xr.DataArray, backend: str = "numpy"):
        cval = coords.values
        if len(cval) < 2 :
            raise ValueError(
                f"Cannot apply linear indexing to an axis of {len(cval)} values"
            )
        return Linear_Indexer(cval, self.bounds, self.spacing, self.period, backend=backend)


class Linear_Indexer:
    def __init__(
        self, coords: NDArray, bounds: str, spacing, period=None, backend: str = "numpy"
    ):
        self.bounds = bounds
        self.N = len(coords)
        self.coords = coords
        self.spacing = spacing
        self.period = period

        # check ascending/descending order
        diff = np.diff(coords).astype(float)
        zero = np.zeros(1, dtype=diff.dtype)
        if (diff > zero).all():
            self.ascending = True
            self.coords = coords
        elif (diff < zero).all():
            self.ascending = False
            self.coords = coords[::-1].copy()
        else:
            raise ValueError('Input coords should be monotonous.')

        self.locator = create_locator(
            coords=self.coords,
            bounds=self.bounds,
            spacing=self.spacing,
            period=period,
            backend=backend,
        )
        
        # print(self.locator)
        

    def __call__(self, values: NDArray) -> List:
        """
        Find indices of `values` for linear interpolation in self.coords

        Returns a list of tuples [(idx_inf, weights), (idx_sup, weights)]
        """
        
        indices, dist = self.locator.locate_index_weight(values)

        if self.ascending:
            iinf = indices
            winf = 1 - dist
            isup = indices + 1
            wsup = dist
        else:
            iinf = self.N - 1 - indices
            isup = self.N - 2 - indices
            winf = 1 - dist
            wsup = dist
        
        if self.bounds == "cycle":
            iinf = iinf % self.N
            isup = isup % self.N

        return [(iinf, winf), (isup, wsup)]


class Index:
    def __init__(self, values):
        """
        Proxy class for integer index-based selection (isel)

        Does not require a separate Indexer class, since the coords are not used;
        the values are returned as-is, they are already integer indices.
        """
        self.values = values
    
    def get_indexer(self, coords, backend: str = "numpy"):
        # We can simply return self to avoid another class, since coords are not used.
        # The __call__ method of the Indexer is implemented in this class.
        return self
        
    def __call__(self, values: NDArray):
        return [(values, None)]


class Nearest:
    def __init__(
        self,
        var: str | xr.DataArray | None, 
        tolerance: float | None = None,
        spacing: Literal["auto"] | Callable[[float], float] = "auto",
    ):
        """
        Proxy class for value selection (sel)

        Args:
            values (xr.DataArray): values for selection
            tolerance (float, optional): absolute tolerance for inexact search
            spacing(str | Callable): 
                - "auto" : will take the value of the nearest valid value
                - lambda x: f(x) : will take the value of the nearest valid value after
                    inverting the x axis values based on lambda
        """
        if isinstance(var, str):
            self.varname = var
            self.values = None
        else:
            self.varname = None
            self.values = var
        self.tolerance = tolerance
        self.spacing = spacing

    def get_indexer(self, coords: xr.DataArray, backend: str = "numpy"):
        return Nearest_Indexer(coords.values, self.tolerance, self.spacing)


class Nearest_Indexer:
    def __init__(self, coords: NDArray, tolerance: float|None, spacing: str|Callable = "auto"):
        self.tolerance = tolerance
        
        if not (isinstance(spacing, str) or isinstance(spacing, Callable)):
            raise TypeError("Spacing is of the wrong type, waiting for str or Callable")
            
        self.spacing = spacing
        if (np.diff(coords) > 0).all():
            self.ascending = True
            self.coords = coords
        elif (np.diff(coords) < 0).all():
            self.ascending = False
            self.coords = coords[::-1]
        else:
            raise ValueError('Input coords should be monotonous.')
    
    def __call__(self, values: NDArray):
        mvalues = None
        if self.spacing == "auto":
            mvalues = values
            coords = self.coords
        elif isinstance(self.spacing, Callable):
            mvalues = self.spacing(values)
            coords = self.spacing(self.coords)
        else:
            raise ValueError("spacing isn't 'auto' or a Callable (lambda x: f(x))")
        
        
        idx = np.searchsorted(coords, mvalues)
        idx = idx.clip(0, len(coords) - 1)

        # distance to the inf/sup bounds
        dist_inf = np.abs(mvalues - coords[idx-1]) # NOTE: EDGE CASE TO BE HANDLED ? [idx = 0]
        dist_sup = np.abs(coords[idx] - mvalues)
        
        # NOTE: EDGE CASE HANDLING ?
        mask_idx_neg = (idx -1) < 0

        dist_inf = np.where(mask_idx_neg, dist_sup + 1, dist_inf)


        if (self.tolerance is not None) and (
            np.minimum(dist_inf, dist_sup) > self.tolerance
        ).any():
            raise ValueError

        idx_closest = np.where(dist_inf < dist_sup, idx-1, idx)

        if self.ascending:
            return [(idx_closest, None)]
        else:
            return [(len(coords) - 1 - idx_closest, None)]


def broadcast_numpy(ds: xr.Dataset, out_dims: list) -> Dict:
    """
    Returns all data variables in `ds` as numpy arrays
    broadcastable against each other, with dimensions orderes as `out_dims`.
    (with new single-element dimensions)

    This requires the input to be broadcasted to common dimensions.
    """
    result = {}
    for var in ds.variables:
        result[var] = ds[var].data[
            tuple(
                [
                    slice(None) if d in ds[var].dims else None
                    for d in out_dims
                    if d in ds.dims
                ]
            )
        ]
    return result


def broadcast_shapes(ds: xr.Dataset, dims: list) -> Dict:
    """
    For each data variable in `ds`, returns the shape for broadcasting
    in the dimensions defined by dims
    """
    result = {}
    for var in ds.variables:
        result[var] = tuple(
            [
                ds[var].shape[ds[var].dims.index(d)] if d in ds[var].dims else 1
                for d in dims
            ]
        )
    return result


def determine_output_dimensions(data: xr.DataArray, mapping: dict, coordinates: xr.Dataset) -> list:
    """
    Determine output dimensions for an interpolated/selected DataArray.
    
    This function implements NumPy's advanced indexing rules to determine the final
    dimension ordering of the output DataArray after interpolation/selection operations.
    
    The key principle is that when advanced indexing is applied to some dimensions
    of an array, those dimensions are replaced by the dimensions of the indexing arrays,
    and these new dimensions are inserted at the position of the first indexed dimension.
    
    Parameters
    ----------
    data : xr.DataArray
        The input DataArray being interpolated/selected.
    mapping : dict
        A dictionary mapping dimension names in `data` (keys) to variable names in
        `coordinates` (values). Only dimensions being interpolated/selected should be
        included.
    coordinates : xr.Dataset
        The Dataset containing coordinate variables referenced in `mapping`.
    
    Returns
    -------
    list
        A list of dimension names for the output DataArray, in the order they will appear.
    """
    out_dims = []
    # dimensions introduced by the coords (aligned, to preserve their relative order)
    dims_coords = align_lists(
        [coordinates[v].dims for k, v in mapping.items() if k in data.dims]
    )
    dims_coords_added = False
    for dim in data.dims:
        if dim in mapping:
            # dimension is interpolated: replace it (once) with dims_coords
            if not dims_coords_added:
                for d in dims_coords:
                    if d not in out_dims:
                        out_dims.append(d)
                dims_coords_added = True
        else:
            # dimension is not interpolated: it remains as-is
            # (skip if already present from dims_coords, e.g. when coord vars
            # share dimensions with the data being interpolated)
            if dim not in out_dims:
                out_dims.append(dim)

    return out_dims
