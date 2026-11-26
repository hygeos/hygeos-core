#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import contextmanager
import getpass
import inspect
import json
import os
import shutil
from datetime import datetime
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Optional, Union

from hygeos_core.files.lock import LockFile
from hygeos_core import log


def safe_move(src, dst, makedirs: bool = True, verbose: bool = True):
    """
    Move `src` file to `dst`

    if `makedirs`: create directory if necessary
    """
    pdst = Path(dst)
    psrc = Path(src)

    if pdst.exists():
        raise IOError(f'Error, {dst} exists')
    if not pdst.parent.exists():
        if makedirs:
            pdst.parent.mkdir(parents=True)
        else:
            raise IOError(f'Error, directory {pdst.parent} does not exist')
    
    if verbose:
        log.debug(f'Moving "{psrc}" to "{pdst}"...')

    with TemporaryDirectory(prefix='copying_'+psrc.name+'_', dir=pdst.parent) as tmpdir:
        tmp = Path(tmpdir)/psrc.name
        shutil.move(psrc, tmp)
        shutil.move(tmp, pdst)

    assert pdst.exists()


class PersistentList(list):
    """
    A list that saves its content in `filename` on each modification. The extension
    must be `.json`.
    
    `concurrent`: whether to activate concurrent mode. In this mode, the
        file is also read before each access.
    """
    def __init__(self,
                 filename,
                 timeout=0,
                 concurrent=True):
        self._filename = Path(filename)
        self.concurrent = concurrent
        self.timeout = timeout
        assert str(filename).endswith('.json')
        self._read()

        # use `_autosave` decorator on all of these methods
        for attr in ('append', 'extend', 'insert', 'pop',
                     'remove', 'reverse', 'sort', 'clear'):
            setattr(self, attr, self._autosave(getattr(self, attr)))

    def __getitem__(self, index):
        # dunder methods are looked up on the class, so the read must be
        # triggered here (an instance attribute would be ignored by `b[0]`)
        if self.concurrent:
            self._read()
        return list.__getitem__(self, index)

    def __len__(self):
        # for some reason, len() does not work with _autoread wrapper
        if self.concurrent:
            self._read()
        return list.__len__(self)

    def _autoread(self, func):
        @wraps(func)
        def _func(*args, **kwargs):
            if self.concurrent:
                self._read()
            return func(*args, **kwargs)
        return _func

    def _autosave(self, func):
        @wraps(func)
        def _func(*args, **kwargs):
            with LockFile(self._filename,
                          timeout=self.timeout):
                if self.concurrent:
                    self._read()
                ret = func(*args, **kwargs)
                self._save()
                return ret
        return _func

    def _read(self):
        list.clear(self)
        if self._filename.exists():
            with open(self._filename) as fp:
                # don't call .extend directly, as it would
                # recursively trigger read and save
                list.extend(self, json.load(fp))

    def _save(self):
        tmpfile = self._filename.parent/(self._filename.name+'.tmp')
        with open(tmpfile, 'w') as fp:
            json.dump(self.copy(), fp, indent=4)
        shutil.move(tmpfile, self._filename)


def skip(filename: Path,
         if_exists: str='skip'):
    """
    Utility function to check whether to skip an existing file
    
    if_exists:
        'skip': skip the existing file
        'error': raise an error on existing file
        'overwrite': overwrite existing file
        'backup': move existing file to a backup '*.1', '*.2'...
    """
    if Path(filename).exists():
        if if_exists == 'skip':
            return True
        elif if_exists == 'error':
            raise FileExistsError(f'File {filename} exists.')
        elif if_exists == 'overwrite':
            if Path(filename).is_dir():
                shutil.rmtree(filename)
            else:
                os.remove(filename)
            return False
        elif if_exists == 'backup':
            i = 0
            while True:
                i += 1
                file_backup = str(filename)+'.'+str(i)
                if not Path(file_backup).exists():
                    break
                if i >= 100:
                    raise FileExistsError()
            shutil.move(filename, file_backup)
        else:
            raise ValueError(f'Invalid argument if_exists={if_exists}')
    else:
        return False


class filegen:
    def __init__(self,
                 arg: Union[int, str]=0,
                 tmpdir: Optional[Path] = None,
                 lock_timeout: int = 0,
                 if_exists: Literal['skip', 'overwrite', 'backup', 'error'] = 'error',
                 verbose: bool = True,
                 ):
        """
        A decorator for functions generating an output file.
        The path to this output file should is defined through `arg`.


        This decorator adds the following features to the function:
        - Use temporary file in a configurable directory, moved afterwards to final location
        - Detect existing file (if_exists='skip', 'overwrite', 'backup' or 'error')
        - Use output file lock when multiple functions may produce the same file
        The timeout for this lock is determined by argument `lock_timeout`.
        
        Args:
            arg: int ot str (default 0)
                if int, defines the position of the positional argument defining the output file
                    (warning, starts at 1 for methods)
                if str, defines the argname of the keyword argument defining the output file
            tmpdir: which temporary directory to use
            lock_timeout (int): timeout (in second) in case of existing lock file
            if_exists (str): what to do in case of existing file
            verbose (bool): verbosity control

        Example:
            @filegen(arg=0)
            def f(path):
                open(path, 'w').write('test')
            f('/path/to/file.txt')

        Note:
            The arguments can be modified at runtime.
            Example:
                f.if_exists = 'overwrite'
                f.verbose = False
        """
        self._arg = arg
        self.tmpdir = tmpdir
        self.lock_timeout = lock_timeout
        self.if_exists = if_exists
        self.verbose = verbose
        
    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if isinstance(self._arg, int):
                assert args, 'Error, no positional argument have been provided'
                assert (self._arg >= 0) and (self._arg < len(args))
                path = args[self._arg]
            elif isinstance(self._arg, str):
                assert self._arg in kwargs, \
                    f'Error, function should have keyword argument "{self._arg}"'
                path = kwargs[self._arg]
            else:
                raise TypeError(f'Invalid argumnt {self._arg}')
                
            ofile = Path(path)

            if skip(ofile, self.if_exists):
                if self.verbose:
                    log.debug(f'Skipping existing file {ofile.name}')
                return
            
            with TemporaryDirectory(dir=self.tmpdir) as tmpd:
                
                # target (intermediary) file
                tfile = Path(tmpd)/ofile.name

                with LockFile(ofile,
                            timeout=self.lock_timeout,
                            ):
                    if skip(ofile, self.if_exists):
                        return
                    if isinstance(self._arg, int):
                        updated_args = list(args)
                        updated_args[self._arg] = tfile
                        updated_kwargs = kwargs
                    elif isinstance(self._arg, str):
                        updated_args = args
                        updated_kwargs = {**kwargs, self._arg: tfile}
                    else:
                        raise ValueError(f'Invalid argumnt {self._arg}')
                        
                    ret = f(*updated_args, **updated_kwargs)
                    assert tfile.exists()

                    # check that the function does not return anything,
                    # because the function call may be skipped upon existing file
                    assert ret is None

                    safe_move(tfile, ofile, verbose=self.verbose)
            return
        return wrapper


def get_git_commit():
    """
    Returns the git version (`git describe --always --dirty`) of the current
    working directory, or a placeholder if it cannot be obtained.

    Note: this identifies the checkout from which the processing is launched;
    the versions of the modules used are given by `hygeos_core.provenance`.
    """
    from hygeos_core.provenance import git_describe
    return (git_describe(Path.cwd(), tracked_only=False, tags=False)
            or '<could not get git commit>')


def mdir(directory: Union[Path,str],
         mdir_filename: str='mdir.json',
         strict: bool=False,
         create: bool=True,
         **kwargs
         ) -> Path:
    """
    Create or access a managed directory with path `directory`
    Returns the directory path, so that it can be used in directories definition:
        dir_data = mdir('/path/to/data/')

    tag it with a file `mdir.json`, containing:
        - The creation date
        - The last access date
        - The python file and module that was run during access
        - The username
        - The current git commit if available
        - Any other kwargs, such as:
            - project
            - version
            - description
            - etc

    mdir_filename: default='mdir.json'

    strict: boolean
        False: metadata is updated
        True: metadata is checked or added (default)
           (remove file content to override)

    create: whether directory is automatically created (default True)
    """
    d = Path(directory)
    mdir_file = d/mdir_filename

    caller = inspect.stack()[1]

    # Attributes to check
    attrs = {
        'caller_file': caller.filename,
        'caller_function': caller.function,
        'git_commit': get_git_commit(),
        'username': getpass.getuser(),
        **kwargs,
    }

    data_init = {
        '__comment__': 'This file has been automatically created '
                        'by mdir() upon managed directory creation, '
                        'and stores metadata.',
        'creation_date': str(datetime.now()),
        **attrs,
    }

    modified = False
    if not d.exists():
        if not create:
            raise FileNotFoundError(
                f'Directory {d} does not exist, '
                'please create it [mdir(..., create=False)]')
        d.mkdir(parents=True)
        data = data_init
        modified = True
    else:
        if not mdir_file.exists():
            if strict:
                raise FileNotFoundError(
                    f'Directory {d} has been wrapped by mdir '
                    'but does not contain a mdir file.')
            else:
                data = data_init
                modified = True
        else:
            with open(mdir_file, encoding='utf-8') as fp:
                data = json.load(fp)

        for k, v in attrs.items():
            if k in data:
                if v != data[k]:
                    if strict:
                        raise ValueError(f'Mismatch of "{k}" in {d} ({v} != {data[k]})')
                    else:
                        data[k] = v
                        modified = True
            else:
                data[k] = v
                modified = True

    if modified:
        with open(mdir_file, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, indent=4)

    return d


@contextmanager
def temporary_copy(src: Path, enable: bool = True, **kwargs):
    """
    Context manager to copy a file/folder to a temporary directory.

    Args:
        src (Path): Path to the source file/folder to copy.
        enable (bool): whether to enable the copy, otherwise returns the input
        Other **kwargs are passed to TemporaryDirectory

    Yields:
        Path: Path to the temporary file/folder
    """
    src = Path(src)
    if not enable:
        yield src
    else:
        with TemporaryDirectory(**kwargs) as temp_dir:
            temp_path = Path(temp_dir) / src.name
            if src.is_dir():
                shutil.copytree(src, temp_path)
            elif src.is_file():
                shutil.copy2(src, temp_path)
            else:
                raise ValueError(f"Source path {src} is neither a file nor a directory.")
            yield temp_path
