#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional

import xarray as xr
from dask.diagnostics.progress import ProgressBar

from hygeos_core.files.fileutils import filegen, get_git_commit
from hygeos_core.provenance import version_attrs
from hygeos_core import log

def to_netcdf(
    ds: xr.Dataset,
    filename: Path,
    *,
    engine: str = "h5netcdf",
    zlib: bool = True,
    complevel: int = 5,
    verbose: bool = True,
    tmpdir: Optional[Path] = None,
    lock_timeout: int = 0,
    git_comit: bool = False,
    versions: Optional[Iterable[str] | Mapping[str, str]] = None,
    if_exists: Literal["skip", "overwrite", "backup", "error"] = "error",
    clean_attrs: bool = True,
    **kwargs
):
    """
    Write an xarray Dataset `ds` using `.to_netcdf` with several additional features:

    - Use file compression
    - Wrapped by filegen: use temporary files, detect existing output files...

    Args:
        ds (xr.Dataset): Input dataset
        filename (Path): Output file path
        engine (str, optional): Engine driver to use. Defaults to 'h5netcdf'.
        zlib (bool, optional): activate zlib. Defaults to True.
        complevel (int, optional): Compression level. Defaults to 5.
        verbose (bool, optional): Verbosity. Defaults to True.
        tmpdir (Path, optional): use a given temporary directory. Defaults to None.
        lock_timeout (int, optional): timeout in case of existing lock file
        git_comit (bool, optional): deprecated, use `versions` instead.
            Add the git version of the current working directory to the
            dataset attributes (`git_commit`). Defaults to False.
        versions (optional): modules whose versions are added to the dataset
            attributes, as "<name>_version" (see hygeos_core.provenance.version_attrs):
            a list of module names, or a dict {name: module name}.
            Example: ["core", "numpy"] -> core_version, numpy_version
        if_exists (str, optional): what to do if output file exists. Defaults to 'error'.
        clean_attrs: whether to remove attributes in the xarray object, that cannot
            be written to netcdf.
        other kwargs are passed to ds.to_netcdf
    """
    soluce = ', got an xarray DataArray. Please use .to_dataset method and ' \
             'specify a variable name' if isinstance(ds, xr.DataArray) else ''
    if not isinstance(ds, xr.Dataset):
        log.error("to_netcdf expects an xarray Dataset" + soluce, 
                  e=AssertionError)

    encoding = (
        {var: dict(zlib=True, complevel=complevel) for var in ds.data_vars}
        if zlib
        else None
    )

    if clean_attrs:
        clean_attributes(ds)

    PBar = {True: ProgressBar, False: nullcontext}[verbose]

    with PBar():
        if verbose:
            log.info("Writing:", filename)
            
        _add_git_commit(ds, git_comit)
        if versions: ds.attrs.update(version_attrs(versions))

        filegen(
            0,
            tmpdir=tmpdir,
            lock_timeout=lock_timeout,
            if_exists=if_exists,
            verbose=verbose,
        )(ds.to_netcdf)(filename, engine=engine, encoding=encoding, **kwargs)


def _add_git_commit(ds: xr.Dataset, git_comit: bool):
    """
    Deprecated `git_comit` option of to_netcdf and to_zarr
    """
    if git_comit:
        warnings.warn(
            "The `git_comit` option is deprecated: the `git_commit` attribute "
            "is the git version of the current working directory, not of the "
            "code used for the processing. Use `versions` instead "
            "(e.g. versions=['core', 'numpy']).",
            DeprecationWarning,
            stacklevel=3,
        )
        ds.attrs.update(git_commit=get_git_commit())


def clean_attributes(obj: xr.Dataset|xr.DataArray):
    """
    Remove attributes that can not be written to netcdf
    """
    import numpy as np
    for attr in list(obj.attrs):
        if isinstance(obj.attrs[attr], (bool,)):
            obj.attrs[attr] = str(obj.attrs[attr])
        elif not isinstance(obj.attrs[attr], (str, float, int, np.ndarray, np.number)):
            del obj.attrs[attr]
    
    # recursively clean attributes in the individual variables
    if isinstance(obj, xr.Dataset):
        for var in list(obj):
            clean_attributes(obj[var])


def to_zarr(
    ds: xr.Dataset,
    filename: Path,
    *,
    zlib: bool = True,
    complevel: int = 5,
    verbose: bool = True,
    tmpdir: Optional[Path] = None,
    lock_timeout: int = 0,
    git_comit: bool = False,
    versions: Optional[Iterable[str] | Mapping[str, str]] = None,
    if_exists: Literal["skip", "overwrite", "backup", "error"] = "error",
    clean_attrs: bool = True,
    **kwargs
):
    """
    Write an xarray Dataset `ds` using `.to_zarr` with several additional features:

    - Use compression
    - Wrapped by filegen: use temporary files, detect existing output files...

    Args:
        ds (xr.Dataset): Input dataset
        filename (Path): Output zarr store path (directory or .zip file)
        zlib (bool, optional): activate zlib compression. Defaults to True.
        complevel (int, optional): Compression level. Defaults to 5.
        verbose (bool, optional): Verbosity. Defaults to True.
        tmpdir (Path, optional): use a given temporary directory. Defaults to None.
        lock_timeout (int, optional): timeout in case of existing lock file
        git_comit (bool, optional): deprecated, use `versions` instead.
            Add the git version of the current working directory to the
            dataset attributes (`git_commit`). Defaults to False.
        versions (optional): modules whose versions are added to the dataset
            attributes, as "<name>_version" (see hygeos_core.provenance.version_attrs):
            a list of module names, or a dict {name: module name}.
        if_exists (str, optional): what to do if output file exists. Defaults to 'error'.
        clean_attrs: whether to remove attributes in the xarray object that cannot
            be written to zarr.
        other kwargs are passed to ds.to_zarr
    """
    soluce = ', got an xarray DataArray. Please use .to_dataset method and ' \
             'specify a variable name' if isinstance(ds, xr.DataArray) else ''
    if not isinstance(ds, xr.Dataset):
        log.error("to_zarr expects an xarray Dataset" + soluce,
                  e=AssertionError)

    encoding = (
        {var: dict(compressor=get_zarr_compressor(complevel))
         for var in ds.data_vars}
        if zlib
        else None
    )

    if clean_attrs:
        clean_attributes(ds)

    PBar = {True: ProgressBar, False: nullcontext}[verbose]
    
    def to_zarr_with_filegen(ds, filename, encoding, **kwargs):
        ds.to_zarr(filename, encoding=encoding, **kwargs)
    
    with PBar():
        if verbose:
            log.info("Writing:", filename)

        _add_git_commit(ds, git_comit)
        if versions:
            ds.attrs.update(version_attrs(versions))

        filegen(
            1,
            tmpdir=tmpdir,
            lock_timeout=lock_timeout,
            if_exists=if_exists,
            verbose=verbose,
        )(to_zarr_with_filegen)(ds, filename, encoding=encoding, **kwargs)


def get_zarr_compressor(complevel: int = 5):
    """
    Return a Blosc compressor for zarr encoding.

    Args:
        complevel (int): Compression level (1-9). Defaults to 5.

    Returns:
        zarr.Blosc compressor instance
    """
    import zarr
    return zarr.Blosc(cname="zstd", clevel=complevel, shuffle=zarr.Blosc.SHUFFLE)
