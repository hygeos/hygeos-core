# Changelog

All notable changes to this project will be documented in this file.

## v2.0.0

### Added

- **Package renamed to `hygeos-core`** (distribution name and import module, previously `core`).
- Selectable interpolation backend: numba-accelerated implementation alongside the numpy one (`backend="auto"` by default).
- `provenance` module: record versions of the imported modules; stored in output attributes by `to_netcdf` / `to_zarr`.
- `save.to_zarr`: write a Dataset to a zarr store (compression, filegen wrapping, provenance).
- `process.timegate`: time-gated iterator.
- New product patterns: MERIS N1 and SAFE 4th reprocessing (`eoread.meris`), HICO; `.` accepted as field separator in product names.
- blockwise: more flexible definition of outputs.
- `tools.is_dask_based` / `tools.is_numpy_backed` helpers.
- Optional peak-memory monitoring in pytest-html reports.

### Changed

- **Breaking**: imports must use `hygeos_core` instead of `core`; the PyPI package is now `hygeos-core`.
- numba, scipy and zarr are optional extras (`hygeos-core[numba]`, `[scipy]`, `[zarr]`): numba accelerates interpolation (used automatically when installed), scipy is required for irregular-grid interpolation with the numpy backend, zarr by `save.to_zarr`.
- **Breaking**: removed the unused `tools.convert` function (use `eotools.units.convert`).
- Better dask handling: `Interpolator` raises on dask-backed data; `map_blocks` warns when passed non-dask data.
- `xrimshow` defaults to `yincrease=False`.
- Documentation rebuilt with quartodoc.

### Fixed

- interpolate: non-interpolated dimensions preserve their original coordinates.
- `xr_filter` / `xr_unfilter`: dimension order and handling in transparent mode; `xr_unfilter` no longer modifies variables where the condition is False.
- monitor: formatting fixes in `Chrono.display` and byte-size display below 1 KB.
- Cached figure: removed an extra bounding box around the plot.
