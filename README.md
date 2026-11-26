# hygeos-core
Common Operations for Robust Engineering

A collection of common procedures for generic programming purposes

## Installation

To install or update this module from PyPI in your current python environment, use the following command:

    pip install hygeos-core

To install it from the github repository (latest development version), use the following command:

    pip install git+https://github.com/hygeos/hygeos-core.git

To install it from a cloned folder, in editable mode:

    pip install -e <dir/to/hygeos-core/>

### Optional dependencies

- `numba`: accelerated backend for `hygeos_core.interpolate`
  (`backend="numba"`). Install with `pip install "hygeos-core[numba]"`, or
  `conda install numba`. With the default `backend="auto"`, it is used
  automatically when installed.
- `scipy`: required by the numpy interpolation backend for irregular grids
  (`backend="numpy"`). Install with `pip install "hygeos-core[scipy]"`, or
  `conda install scipy`.
- `zarr`: required by `hygeos_core.files.save.to_zarr`. Install with
  `pip install "hygeos-core[zarr]"`, or `conda install zarr`.
