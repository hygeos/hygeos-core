#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import bz2
import gzip
import shutil
import subprocess
import tarfile
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from hygeos_core import log
from hygeos_core.tools import only


def uncompress_decorator(target_name_func=None, verbose=True):
    """
    A decorator that uncompresses the result of function `f`
    
    Signature of `f` is assumed to be as follows:
        f(identifier: str, dirname: Path, *args, **kwargs)
    
    The file returned by `f` is uncompressed to dirname
    
    Parameters:
    -----------
    target_name_func : callable, optional
        A function that takes an identifier and returns the target name/path
        for the uncompressed file. The identifier can be a URL or any string.
        If None, defaults to using Path(identifier).stem.
    verbose : bool
        Whether to display verbose output during uncompression
    """
    # Provide default target_name_func if none is given
    if target_name_func is None:
        def default_target_name_func(identifier):
            return Path(identifier).stem
        target_name_func = default_target_name_func
    
    def decorator(f):
        @wraps(f)
        def wrapper(identifier, dirname, *args, **kwargs):
            # Get target name from the provided function
            target_name = target_name_func(identifier)
            target = Path(dirname) / target_name
            
            # Check if target already exists
            if target.exists():
                return target
            
            # Download and uncompress in a temporary directory, then move the
            # result to target on success only (no partial target left behind)
            Path(dirname).mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix='tmp_uncompress_', dir=dirname) as tmpdir:
                dl_dir = Path(tmpdir) / 'download'
                dl_dir.mkdir()
                f_compressed = Path(f(identifier, dl_dir, *args, **kwargs))
                out_dir = Path(tmpdir) / 'out'
                uncompress(f_compressed, out_dir, verbose=verbose, extract_to='target_dir')
                
                if is_single_file_compression(f_compressed):
                    # single compressed file (.gz, .bz2, .Z): the result is a file
                    result = only(list(out_dir.iterdir()))
                else:
                    # archive: the result is a directory with the archive content
                    result = out_dir
                shutil.move(result, target)
                
            assert target.exists(), f"Target {target} does not exist after uncompression"
            return target
        return wrapper
    return decorator


def _check_tar_members(filename, target_dir):
    """
    Raise an error if a member of tar archive `filename` would be extracted
    outside of `target_dir` (fallback for Pythons without tar extraction filters)
    """
    target = Path(target_dir).resolve()
    with tarfile.open(filename) as tar:
        for m in tar.getmembers():
            dest = (target / m.name).resolve()
            if (target not in dest.parents) and (dest != target):
                raise ValueError(f'Unsafe path in archive {filename}: {m.name}')
            if (m.issym() or m.islnk()) and (
                Path(m.linkname).is_absolute() or '..' in Path(m.linkname).parts
            ):
                raise ValueError(f'Unsafe link in archive {filename}: {m.name}')


def is_single_file_compression(filename) -> bool:
    """
    Whether `filename` is a single compressed file (.gz, .bz2, .Z), as opposed to
    an archive (.zip, .tar, .tar.gz...).

    The check is case-insensitive, except for '.Z' (unix compress) which is
    distinct from '.z' (pack).
    """
    name = Path(filename).name
    lname = name.lower()
    return ((lname.endswith('.gz') and not lname.endswith('.tar.gz')) or
            (lname.endswith('.bz2') and not lname.endswith('.tar.bz2')) or
            name.endswith('.Z'))


def uncompress_single(filename, output_path, verbose=False):
    """
    Decompress a single compressed file (not an archive).
    
    Parameters:
    -----------
    filename : Path
        Path to the compressed file
    output_path : Path  
        Path where the decompressed file should be written
    verbose : bool
        Whether to display verbose output
        
    Returns:
    --------
    Path
        Path to the decompressed file
    """
    filename = Path(filename)
    fname = str(filename)
    lname = filename.name.lower()
    
    if lname.endswith('.gz') and not any(lname.endswith(x) for x in ['.tar.gz', '.tgz']):
        # Single gzip file
        with gzip.open(fname, 'rb') as f_in, open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    elif lname.endswith('.bz2') and not lname.endswith('.tar.bz2'):
        # Single bzip2 file  
        with bz2.BZ2File(fname) as f_in, open(output_path, 'wb') as f_out:
            data = f_in.read()
            f_out.write(data)
            
    elif filename.name.endswith('.Z'):
        # Unix compress format
        cmd = f'gunzip {fname}'
        if verbose:
            log.info('Executing:', cmd)
        if subprocess.call(cmd.split()):
            raise Exception(f'Error executing command {cmd}')
        # For .Z files, gunzip creates the output file directly
        expected_output = filename.parent / filename.stem
        if expected_output != output_path:
            shutil.move(expected_output, output_path)
    else:
        raise ValueError(f"Unsupported single file compression format: {filename.suffix}")
    
    return output_path


def uncompress(filename: str | Path,
               target_dir: str | Path,
               create_out_dir=True,
               extract_to: Literal['auto', 'subdir', 'target_dir'] = 'auto',
               verbose=False) -> Path:
    """
    Uncompress `filename` to `dirname`

    Parameters
    ----------

    filename : str or Path
        Path to the file to uncompress

    target_dir : str or Path
        Directory to uncompress the file to

    create_out_dir : bool, optional
        If True, create the output directory if it does not exist. Default is True.

    extract_to : {'auto', 'subdir', 'target_dir'}, default 'auto'
        'auto': if the root of the archive contains a single file or directory, uncompress
            it to `target_dir`. Otherwise uncompress it to <target_dir>/<archivefolder>, where
            `archivefolder` is a directory named after `filename` but without the extension.
        'subdir': Uncompress the content of the archive to <target_dir>/<archivefolder>, where
            `archivefolder` is a directory named after `filename` but without the extension.
        'target_dir': uncompress the content of the archive to <target_dir> regardless of the
            content of the archive.

    verbose : bool, optional
        If True, print verbose output. Default is False.

    Returns
    -------

    Path
        Path to the uncompressed file or directory
    """
    filename = Path(filename)
    fname = filename.name.lower()
    if verbose:
        log.info(f'Uncompressing {filename} to {target_dir}')
    if not Path(target_dir).exists():
        if create_out_dir:
            Path(target_dir).mkdir(parents=True)
        else:
            raise IOError(f'Directory {target_dir} does not exist.')

    with TemporaryDirectory(prefix='tmp_uncompress', dir=target_dir) as tmpdir:
        extracted_path = None
        
        # Check if it's a single compressed file first
        if is_single_file_compression(filename):
            # Single compressed file
            extracted_path = Path(tmpdir) / filename.stem
            uncompress_single(filename, extracted_path, verbose)
        else:
            # Try archive formats with shutil.unpack_archive
            # (format detected case-insensitively from the extension)
            fmt = None
            for name, exts, _ in shutil.get_unpack_formats():
                if any(fname.endswith(e.lower()) for e in exts):
                    fmt = name
                    break
            if fmt == 'zip':
                # zip members are sanitized by zipfile (no '..' or absolute paths)
                shutil.unpack_archive(filename, tmpdir, format=fmt)
            elif hasattr(tarfile, 'data_filter'):
                # filter='data' rejects tar members with absolute paths or '..'
                # components (path traversal), links outside the target...
                shutil.unpack_archive(filename, tmpdir, format=fmt, filter='data')
            else:
                # Python versions without extraction filters (< 3.10.12, 3.11.4)
                _check_tar_members(filename, tmpdir)
                shutil.unpack_archive(filename, tmpdir, format=fmt)
            extracted_path = Path(tmpdir)
        
        # Determine whether the archive shall be uncompressed to a subdirectory or
        if extracted_path.is_file():
            content = [extracted_path]
        else:
            content = list(extracted_path.glob('*'))
        assert len(content)
        if extract_to == 'target_dir':
            extract_to_subdir = False
        elif extract_to == 'subdir':
            extract_to_subdir = True
        elif extract_to == 'auto':
            extract_to_subdir = len(content) >= 2
        else:
            raise ValueError(extract_to)

        # Determine final target path
        if extract_to_subdir:
            # Handle compound extensions properly
            if fname.endswith(('.tar.gz', '.tar.bz2', '.tar.xz')):
                # For compound extensions, remove both parts
                target_name = Path(filename.stem).stem
            else:
                target_name = filename.stem
            
            target = Path(target_dir) / target_name
            assert not target.exists(), f'Error, {target} exists.'
            # Move to final destination
            shutil.move(extracted_path, target)
        else:
            target = Path(target_dir)
            # Move the contents of extracted_path to target_dir
            items = [extracted_path] if extracted_path.is_file() else list(extracted_path.iterdir())
            for item in items:
                dest = target / item.name
                assert not dest.exists(), f'Error, {dest} exists.'
                shutil.move(item, dest)
            if extract_to == 'auto':
                # a single file was moved: we shall return its path instead of the
                # containing folder
                target = target / only(items).name
        
    assert target.exists()
    return target


def get_compression_ext(f: str|Path):
    """
    Detect the compression format of a file using the system 'file' command.
    
    This function uses the Unix 'file' command to determine the compression format
    of a file based on its content (magic numbers), not just its extension.
    
    Parameters:
    -----------
    f : str or Path
        Path to the file to analyze
        
    Returns:
    --------
    str or None
        The detected compression extension ('.zip', '.tar', '.tar.gz', '.tgz', 
        '.gz', '.bz2', '.Z') or None if no compression is detected
    """
    # Use file module to deduce format
    result = subprocess.run(['file', str(f)], capture_output=True, text=True)
    file_type = result.stdout.lower()
    
    # Check for various compression formats based on file command output
    if 'zip archive' in file_type: 
        return '.zip'
    if 'gzip compressed' in file_type:
        # Check if it's a tar.gz by looking for tar content
        if 'tar archive' in file_type or str(f).endswith(('.tar.gz', '.tgz')):
            return '.tar.gz' if str(f).endswith('.tar.gz') else '.tgz'
        return '.gz'
    if 'bzip2 compressed' in file_type:
        # Check if it's a tar.bz2
        if 'tar archive' in file_type or str(f).endswith('.tar.bz2'):
            return '.tar.bz2'
        return '.bz2'
    if 'compress\'d data' in file_type or 'unix compress' in file_type:
        return '.Z'
    if 'tar archive' in file_type:
        return '.tar'
    
    # Return None if no compression is detected
    return None