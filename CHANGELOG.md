# Changelog

All notable changes to this project will be documented in this file.

## v2.0.1

### Changed

- **Breaking**: objects must be imported from the module that defines them; the `__init__.py`
  files no longer re-export anything, and the deprecated top-level modules are removed.
  `import hygeos_core.log` (and any other submodule) no longer loads xarray and pandas
  (import time ~630 ms → ~110 ms).

  | Old import | New import |
  |---|---|
  | `hygeos_core.filegen`, `hygeos_core.mdir` | `hygeos_core.files.fileutils.filegen`, `.mdir` |
  | `hygeos_core.to_netcdf` | `hygeos_core.files.save.to_netcdf` |
  | `hygeos_core.rgb` | `hygeos_core.log.rgb` |
  | `hygeos_core.files.{cache_*, filegen, get_git_commit, mdir, LockFile, to_netcdf, uncompress}` | `hygeos_core.files.{cache, fileutils, lock, save, uncompress}.<name>` |
  | `hygeos_core.geo.{get_pattern, get_level, retrieve_product}` | `hygeos_core.geo.product_name.<name>` |
  | `hygeos_core.geo.n` | `from hygeos_core.geo.naming import names as n` |
  | `hygeos_core.geo.<convert functions>` | `hygeos_core.geo.convert.<name>` |
  | `hygeos_core.static.{interface, abstract, singleton, virtual, parametrized}` | `hygeos_core.static.<name>.<name>`, e.g. `from hygeos_core.static.interface import interface` |
  | `hygeos_core.{auth, download, ftp}` | `hygeos_core.network.{auth, download, ftp}` |
  | `hygeos_core.{cache, fileutils, lock, save, uncompress}` | `hygeos_core.files.{cache, fileutils, lock, save, uncompress}` |
  | `hygeos_core.{condor, deptree, deptree_prefect}` | `hygeos_core.process.{condor, deptree, deptree_prefect}` |
  | `hygeos_core.{conftest, pytest_utils}` | `hygeos_core.tests.{conftest, pytest_utils}` |

  Note: `from hygeos_core.static import interface` now returns the *module*, so using it as
  a decorator fails at import time with `TypeError: 'module' object is not callable`.

### Fixed

- matplotlib and Pillow are optional: they are imported lazily by `files.cache.cache_figure`
  (importing `hygeos_core` previously failed without matplotlib).

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
