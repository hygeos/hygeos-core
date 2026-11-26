# Changelog

All notable changes to this project will be documented in this file.

## v2.0.2

### Changed

- `tools.reglob` matches on file names and returns a sorted list of `Path`.
- `uncompress_decorator` returns a file for single-file archives (`foo.nc.gz` → `foo.nc`).
- top-level `deptree.gen_executor` returns the task result and raises its errors.
- `dates.time_range` no longer skips the last step; `interpolate.Nearest` gives NaN for NaN
  queries on float data; the `zarr` extra requires `zarr>=3`.

### Fixed

- Several bugs found in a code review, notably: circular imports (`log`/`env`/`progressbar`),
  spline interpolation (tension, descending coords), numpy/numba interpolation mismatch,
  `gen_executor` not awaiting shared dependencies, `.gz`/`.bz2` decompression and tar path
  traversal, `to_zarr` with zarr 3, non-atomic FTP uploads, csv `cache_dataframe` checks.

## v2.0.1

### Changed

- objects must be imported from the module that defines them (no re-exports in
  `__init__.py`, deprecated top-level modules removed), e.g. `hygeos_core.files.fileutils.filegen`,
  `hygeos_core.network.ftp`, `hygeos_core.process.deptree`. Importing a submodule no longer
  loads xarray and pandas.

### Fixed

- matplotlib and Pillow are optional dependencies.

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
