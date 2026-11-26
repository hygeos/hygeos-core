#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import bz2
import gzip
import shutil
import subprocess
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
            
            # Uncompress to target location
            with TemporaryDirectory() as tmpdir:
                f_compressed = f(identifier, tmpdir, *args, **kwargs)
                uncompress(f_compressed, target, verbose=verbose, extract_to='target_dir')
                
                # Find the uncompressed result in dirname and rename if needed
                uncompressed_files = [p for p in Path(dirname).iterdir() 
                                    if p.is_file() or p.is_dir()]
                
                # Find the most recently created file/directory
                if uncompressed_files:
                    newest = max(uncompressed_files, key=lambda p: p.stat().st_mtime)
                    if newest.name != target_name:
                        newest.rename(target)
                
            assert target.exists(), f"Target {target} does not exist after uncompression"
            return target
        return wrapper
    return decorator


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
    fname = str(filename)
    
    if fname.endswith('.gz') and not any(fname.endswith(x) for x in ['.tar.gz', '.tgz']):
        # Single gzip file
        with gzip.open(fname, 'rb') as f_in, open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    elif fname.endswith('.bz2') and not fname.endswith('.tar.bz2'):
        # Single bzip2 file  
        with bz2.BZ2File(fname) as f_in, open(output_path, 'wb') as f_out:
            data = f_in.read()
            f_out.write(data)
            
    elif fname.endswith('.Z'):
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
        if ((fname.endswith('.gz') and not fname.endswith('.tar.gz')) or
            (fname.endswith('.bz2') and not fname.endswith('.tar.bz2')) or
            fname.endswith('.Z')):
            # Single compressed file
            extracted_path = Path(tmpdir) / filename.stem
            uncompress_single(filename, extracted_path, verbose)
        else:
            # Try archive formats with shutil.unpack_archive
            shutil.unpack_archive(filename, tmpdir)
            extracted_path = Path(tmpdir)
        
        # Determine whether the archive shall be uncompressed to a subdirectory or
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