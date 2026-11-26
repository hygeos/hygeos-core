"""
Numba-accelerated functions for core.interpolate

Index lookup:
    - find_indices_numba: drop-in replacement for scipy's find_indices
    - locate_regular: index lookup on regular grids (including out-of-bounds handling)
    - locate_irregular: index lookup on irregular grids (including out-of-bounds handling)

Interpolation:
    - interp_kernel_serial/interp_kernel_parallel: weighted sum over the corners of
      the interpolation cell, gathered directly from the flat data array

All functions release the GIL (nogil=True), so that they run in parallel when called
from multiple threads (e.g. by the dask threaded scheduler). The interpolation kernel
has a serial and a parallel variant: the parallel variant should not be used from
within dask threads, to avoid oversubscription.
"""

from typing import Tuple

import numpy as np
from numba import njit, prange

# Out-of-bounds handling modes (see Locator.handle_oob)
OOB_NONE = 0  # no handling (bounds="cycle")
OOB_CLIP = 1  # clip to the axis extrema
OOB_NAN = 2   # replace by NaN
OOB_ERROR = 3 # count the out-of-bounds values (the caller raises an error)

# Number of points per chunk in find_indices_numba (parallelized over the chunks)
CHUNK_SIZE = 4096

# Maximum number of steps of the walk from the previous interval in the index lookup
# on irregular grids, before falling back to a binary search (see _walk_interval)
MAX_WALK = 8

# intp range, as floats (exact: powers of 2), to saturate the float -> intp casts
INTP_MIN = np.iinfo(np.intp).min
INTP_MAX = np.iinfo(np.intp).max
INTP_FLOAT_LIMIT = -float(INTP_MIN)  # 2**63 on 64-bit platforms

OOB_MODES = {
    "cycle": OOB_NONE,
    "clip": OOB_CLIP,
    "nan": OOB_NAN,
    "error": OOB_ERROR,
}


@njit(nogil=True, cache=True, inline="always")
def _search_interval(coords, value):
    """
    Index i of the interval [coords[i], coords[i+1]) containing `value`, clipped to
    [0, N-2] (same as scipy's find_indices). `coords` are ascending.
    """
    n = coords.shape[0]
    lo = 0
    hi = n - 2
    # find the last i in [0, n-2] such that coords[i] <= value
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if coords[mid] <= value:
            lo = mid
        else:
            hi = mid - 1
    return lo


@njit(nogil=True, cache=True, inline="always")
def _walk_interval(coords, value, i):
    """
    Same as `_search_interval`, starting from the interval index `i` (generally, the
    one of the previous value).

    Successive values are generally close (e.g. in an image): the interval is first
    searched by walking from `i`, by at most MAX_WALK steps, which is faster than a
    binary search. If the interval is further away, a binary search is used, so that
    the worst case remains O(log N) (instead of O(N) for an unbounded walk).
    """
    last = coords.shape[0] - 2
    k = MAX_WALK
    while (k > 0) and (i > 0) and (value < coords[i]):
        i -= 1
        k -= 1
    while (k > 0) and (i < last) and (value >= coords[i + 1]):
        i += 1
        k -= 1
    if k == 0:
        # walk budget exhausted: the interval may be further away
        i = _search_interval(coords, value)
    return i


@njit(nogil=True, cache=True, parallel=True)
def find_indices_numba(grid, xi) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-dimensional grid interpolation preprocessing.

    Drop-in replacement for scipy.interpolate._rgi_cython.find_indices, with the same
    outputs: indices clipped to [0, N-2] (-1 for NaN), and distances extrapolated
    outside of the grid. The distances are float32 (float64 for scipy). Length-1 grids
    give an index of -1 and a distance of 0. The grid coordinates should be ascending
    (not checked).

    The points are processed in contiguous chunks (in parallel), and the search of
    each point starts from the interval of the previous point (see _walk_interval).

    Args:
        grid: Tuple of 1D arrays defining grid coordinates for each dimension
        xi: 2D array where each row represents a dimension and each column a query point

    Returns:
        indices: Grid interval indices for each query point in each dimension
        distances: Normalized distances within each interval
    """
    n_dims, n_points = xi.shape

    indices = np.empty((n_dims, n_points), dtype=np.intp)
    distances = np.empty((n_dims, n_points), dtype=np.float32)

    for dim in range(n_dims):
        coordinates = grid[dim]
        if coordinates.shape[0] == 1:
            # length-1 grid: special value for downstream processing
            for j in range(n_points):
                indices[dim, j] = -1
                distances[dim, j] = 0.0
            continue
        # parallel over chunks of points (the number of dimensions is generally 1)
        nchunks = (n_points + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk in prange(nchunks):
            i = 0  # interval index of the previous point
            for j in range(chunk * CHUNK_SIZE,
                           min((chunk + 1) * CHUNK_SIZE, n_points)):
                value = xi[dim, j]
                if value != value:  # NaN
                    indices[dim, j] = -1
                    distances[dim, j] = np.nan
                    continue
                i = _walk_interval(coordinates, value, i)
                indices[dim, j] = i
                distances[dim, j] = (value - coordinates[i]) / (
                    coordinates[i + 1] - coordinates[i]
                )

    return (indices, distances)


@njit(nogil=True, cache=True)
def locate_regular(values, vstart, vmin, vmax, scal, N, mode, nan):
    """
    Index lookup on a regular grid (see Locator_Regular.locate_index_weight)

    Args:
        values: 1D array of values to locate
        vstart: first coordinate value
        vmin, vmax: coordinate extrema
        scal: (N-1)/(vend-vstart)
        N: number of coordinates
        mode: one of the OOB_* modes
        nan: NaN, with the type of `values` (so that the computations are done in the
            same type as in Locator_Regular)

    Returns:
        i_inf (intp), dist (float64), number of out-of-bounds values
    """
    n = values.shape[0]
    i_inf = np.empty(n, dtype=np.intp)
    dist = np.empty(n, dtype=np.float64)
    n_oob = 0
    for j in range(n):
        v = values[j]
        if mode == OOB_CLIP:
            # same as np.clip (NaN are kept)
            if v < vmin:
                v = vmin
            elif v > vmax:
                v = vmax
        elif (mode == OOB_NAN) or (mode == OOB_ERROR):
            if (v < vmin) or (v > vmax):
                if mode == OOB_NAN:
                    v = nan
                else:
                    n_oob += 1
        x = (v - vstart) * scal
        # floor(nan_to_num(x)), saturating the cast to intp: casting a float beyond
        # the intp range is undefined behaviour
        if x != x:
            i = 0
        elif x >= INTP_FLOAT_LIMIT:
            i = INTP_MAX
        elif x < -INTP_FLOAT_LIMIT:
            i = INTP_MIN
        else:
            i = np.intp(np.floor(x))
        d = x - i
        if (mode != OOB_NONE) and (i == N - 1):
            # Deals with indexing the last coordinate item
            i = N - 2
            d = 1.0
        i_inf[j] = i
        dist[j] = d
    return i_inf, dist, n_oob


@njit(nogil=True, cache=True)
def locate_irregular(values, coords, vmin, vmax, mode, nan):
    """
    Index lookup on an irregular ascending grid (see Locator.locate_index_weight)

    Same outputs as handle_oob + scipy's find_indices + clip(0, None). The search of
    each value starts from the interval of the previous value (see _walk_interval).

    Returns:
        indices (intp), dist (float64), number of out-of-bounds values
    """
    n = values.shape[0]
    indices = np.empty(n, dtype=np.intp)
    dist = np.empty(n, dtype=np.float64)
    n_oob = 0
    i = 0  # interval index of the previous value
    for j in range(n):
        v = values[j]
        if mode == OOB_CLIP:
            if v < vmin:
                v = vmin
            elif v > vmax:
                v = vmax
        elif (mode == OOB_NAN) or (mode == OOB_ERROR):
            if (v < vmin) or (v > vmax):
                if mode == OOB_NAN:
                    v = nan
                else:
                    n_oob += 1
        value = np.float64(v)
        if value != value:  # NaN
            indices[j] = 0
            dist[j] = np.nan
            continue
        i = _walk_interval(coords, value, i)
        indices[j] = i
        dist[j] = (value - coords[i]) / (coords[i + 1] - coords[i])
    return indices, dist, n_oob


@njit(nogil=True, cache=True)
def _outer_offsets(outer, shape, t_iptr, t_istr, t_wptr, t_wstr, out_str, ibase,
                   wbase):
    """
    Offsets, for the outer index `outer` (flattened index over all axes of `shape`
    but the last one, the inner axis), of each term in the index and weight buffers
    (ibase, wbase), and in the output (returned).
    """
    R = shape.shape[0]
    nterms = t_iptr.shape[0]
    for t in range(nterms):
        ibase[t] = t_iptr[t]
        wbase[t] = t_wptr[t]
    obase = 0
    rem = np.intp(outer)  # (prange gives an unsigned integer)
    for a in range(R - 2, -1, -1):
        ia = rem % shape[a]
        rem = rem // shape[a]
        for t in range(nterms):
            ibase[t] += ia * t_istr[t, a]
            wbase[t] += ia * t_wstr[t, a]
        obase += ia * out_str[a]
    return obase


@njit(nogil=True, cache=True)
def _interp_inner(data, ibuf, wbuf, t_istr, t_wstr, t_dstride, term_start,
                  term_count, ninv, kept_offsets, out, out_str, n_inner, ibase,
                  wbase, obase, off, w, boff, bw, coff, cw):
    """
    Interpolate all points along the inner axis (last axis of the output points),
    for one outer index (whose offsets are ibase, wbase, obase)

    For each point, the offsets and weights of the corners of the interpolation cell
    are expanded dimension by dimension: after dimension k, the partial offsets and
    weights of all combinations of the terms of dimensions 0..k are available (in
    the same order as itertools.product).

    The terms of the first `ninv` dimensions are invariant along the inner axis: they
    are expanded once per outer index (in boff, bw), and reused for all points along
    the inner axis.
    """
    R = t_istr.shape[1]
    M = kept_offsets.shape[0]
    ndims = term_start.shape[0]
    inner = R - 1

    # invariant dimensions: load their terms and expand them once
    nb = 1
    boff[0] = 0
    bw[0] = 1.0
    for k in range(ninv):
        sk = term_start[k]
        nk = term_count[k]
        for j in range(nk):
            t = sk + j
            off[t] = ibuf[ibase[t]] * t_dstride[t]
            w[t] = wbuf[wbase[t]]
        for c in range(nb - 1, -1, -1):
            o = boff[c]
            ww = bw[c]
            for j in range(nk):
                boff[c * nk + j] = o + off[sk + j]
                bw[c * nk + j] = ww * w[sk + j]
        nb = nb * nk

    for i in range(n_inner):
        # other dimensions: load their terms at this point, and expand them
        n = nb
        for c in range(nb):
            coff[c] = boff[c]
            cw[c] = bw[c]
        for k in range(ninv, ndims):
            sk = term_start[k]
            nk = term_count[k]
            for j in range(nk):
                t = sk + j
                off[t] = ibuf[ibase[t] + i * t_istr[t, inner]] * t_dstride[t]
                w[t] = wbuf[wbase[t] + i * t_wstr[t, inner]]
            for c in range(n - 1, -1, -1):
                o = coff[c]
                ww = cw[c]
                for j in range(nk):
                    coff[c * nk + j] = o + off[sk + j]
                    cw[c * nk + j] = ww * w[sk + j]
            n = n * nk

        # weighted sum over the corners
        o_out = obase + i * out_str[inner]
        for m in range(M):
            ko = kept_offsets[m]
            acc = 0.0
            for c in range(n):
                acc += data[coff[c] + ko] * cw[c]
            out[o_out + m] = acc


@njit(nogil=True, cache=True)
def interp_kernel_serial(data, ibuf, wbuf, t_iptr, t_istr, t_wptr, t_wstr,
                         t_dstride, term_start, term_count, ninv, kept_offsets,
                         shape, out, out_str):
    """
    Interpolation as a weighted sum over the corners of the interpolation cell

    The output points have an arbitrary number R of axes, given by `shape`. The loop
    has two levels: an outer index (flattened index over all axes but the last one),
    and the inner axis (the last axis of `shape`, chosen by the caller as the one
    along which most interpolated dimensions are invariant).

    Args:
        data: flat (1D) data array
        ibuf, wbuf: flat buffers of the indices (intp) and weights (float64) of all
            terms. A term is one of the possible values of an interpolated dimension
            (e.g. 2 for linear interpolation: lower and upper).
        t_iptr, t_wptr: for each term, start of its indices/weights in ibuf/wbuf
        t_istr, t_wstr: (nterms, R) strides of the indices/weights of each term in
            ibuf/wbuf, along each axis of the output points (0 for broadcasted axes)
        t_dstride: for each term, stride of its dimension in `data` (in elements)
        term_start, term_count: for each interpolated dimension, index of its first
            term, and number of terms
        ninv: number of (first) dimensions whose terms are invariant along the inner
            axis
        kept_offsets: offsets in `data` of each element of the non-interpolated
            dimensions (M elements)
        shape: (R,) shape of the output points
        out: flat output array (float64)
        out_str: (R,) strides of the output points in `out`. The M elements of the
            non-interpolated dimensions are contiguous.
    """
    R = shape.shape[0]
    n_outer = 1
    for a in range(R - 1):
        n_outer *= shape[a]
    nterms = t_dstride.shape[0]
    ncorners = 1
    for k in range(term_count.shape[0]):
        ncorners *= term_count[k]
    ibase = np.empty(nterms, dtype=np.intp)
    wbase = np.empty(nterms, dtype=np.intp)
    off = np.empty(nterms, dtype=np.intp)
    w = np.empty(nterms, dtype=np.float64)
    boff = np.empty(ncorners, dtype=np.intp)
    bw = np.empty(ncorners, dtype=np.float64)
    coff = np.empty(ncorners, dtype=np.intp)
    cw = np.empty(ncorners, dtype=np.float64)
    for outer in range(n_outer):
        obase = _outer_offsets(outer, shape, t_iptr, t_istr, t_wptr, t_wstr,
                               out_str, ibase, wbase)
        _interp_inner(data, ibuf, wbuf, t_istr, t_wstr, t_dstride, term_start,
                      term_count, ninv, kept_offsets, out, out_str, shape[R - 1],
                      ibase, wbase, obase, off, w, boff, bw, coff, cw)


@njit(nogil=True, cache=True, parallel=True)
def interp_kernel_parallel(data, ibuf, wbuf, t_iptr, t_istr, t_wptr, t_wstr,
                           t_dstride, term_start, term_count, ninv, kept_offsets,
                           shape, out, out_str):
    """
    Parallel version of `interp_kernel_serial` (parallelized over the outer index)
    """
    R = shape.shape[0]
    n_outer = 1
    for a in range(R - 1):
        n_outer *= shape[a]
    nterms = t_dstride.shape[0]
    ncorners = 1
    for k in range(term_count.shape[0]):
        ncorners *= term_count[k]
    for outer in prange(n_outer):
        # scratch arrays (per outer index)
        ibase = np.empty(nterms, dtype=np.intp)
        wbase = np.empty(nterms, dtype=np.intp)
        off = np.empty(nterms, dtype=np.intp)
        w = np.empty(nterms, dtype=np.float64)
        boff = np.empty(ncorners, dtype=np.intp)
        bw = np.empty(ncorners, dtype=np.float64)
        coff = np.empty(ncorners, dtype=np.intp)
        cw = np.empty(ncorners, dtype=np.float64)
        obase = _outer_offsets(outer, shape, t_iptr, t_istr, t_wptr, t_wstr,
                               out_str, ibase, wbase)
        _interp_inner(data, ibuf, wbuf, t_istr, t_wstr, t_dstride, term_start,
                      term_count, ninv, kept_offsets, out, out_str, shape[R - 1],
                      ibase, wbase, obase, off, w, boff, bw, coff, cw)
