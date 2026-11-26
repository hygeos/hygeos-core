from dotenv import load_dotenv, find_dotenv
from os import environ
from pathlib import Path
from typing import Optional
from hygeos_core import log


def find_dotenvs(
    filename: str = '.env',
    start_path: Optional[Path] = None,
) -> list[Path]:
    """
    Find all dotenv files named `filename` in the start directory and every
    parent directory up to the filesystem root.

    Unlike :func:`dotenv.find_dotenv`, which stops at the first match, this
    function collects **all** matches along the directory tree, using
    :func:`dotenv.find_dotenv` to resolve the starting directory.

    Args:
        filename: dotenv filename to look for (default: ``'.env'``).
        usecwd: if ``True`` (default), start from the current working directory.
            Ignored when `start_path` is provided.
            Note: unlike :func:`dotenv.find_dotenv`, ``usecwd=False`` here is
            equivalent to ``usecwd=True`` because ``find_dotenv`` would
            otherwise use *this* module's frame, not the caller's.
        start_path: explicit starting directory, takes priority over `usecwd`.

    Returns:
        List of :class:`~pathlib.Path` objects for every dotenv file found,
        ordered from **closest** (start directory) to **farthest** (root).
    """
    if start_path is not None:
        current = Path(start_path).resolve()
    else:
        # find_dotenv with usecwd=True resolves from CWD; when called from
        # inside another function usecwd=False would point to this module, so
        # we always forward usecwd=True and let start_path handle other cases.
        first = find_dotenv(filename=filename, usecwd=True)
        current = Path(first).parent if first else Path.cwd()

    dotenvs: list[Path] = []
    while True:
        candidate = current / filename
        if candidate.is_file():
            dotenvs.append(candidate)
        parent = current.parent
        if parent == current:
            break
        current = parent

    return dotenvs


def load_dotenvs(
    filename: str = '.env',
    start_path: Optional[Path] = None,
    override: bool = True,
) -> list[Path]:
    """
    Load all dotenv files found by :func:`find_dotenvs`, from the farthest
    ancestor to the closest directory so that inner ``.env`` files take
    precedence over outer ones.

    ``load_dotenv`` is called once per file found; variables already present
    in the environment are only overwritten when ``override=True`` (the
    default here, unlike the dotenv library default, to honour the
    closest-wins precedence).

    Args:
        filename: dotenv filename to look for (default: ``'.env'``).
        usecwd: if ``True``, start from the current working directory.
        start_path: explicit starting directory.
        override: whether a closer file's value overrides a value already
            loaded from a farther file (default: ``True``).

    Returns:
        The list of dotenv files that were loaded, closest first.
    """
    dotenvs = find_dotenvs(filename=filename, start_path=start_path)
    # Load farthest -> closest so that inner files override outer ones
    for dotenv in reversed(dotenvs):
        load_dotenv(dotenv_path=dotenv, override=override)
    return dotenvs

load_dotenvs()

def getvar(
    envvar: str,
    default: str|None = None,
    ) -> str:
    """
    Returns the value of environment variable `envvar`. If this variable is not defined, returns default.

    The environment variable can be defined in the users `.bashrc`, or in a file `.env`
    in the current working directory.

    Args:
        envvar: the input environment variable
        default: the default return, if the environment variable is not defined
    
    Returns:
        the requested environment variable or the default if the var is not defined and a default has been provided.
    """
    variable = None
    if envvar in environ:
        variable = environ[envvar]
        
    elif default is None:
        raise KeyError(f"{envvar} is not defined, and no default has been provided.")
    else:
        variable = default

    return variable

def getdir(
    envvar: str,
    default: Optional[Path] = None,
    create: Optional[bool] = None,
) -> Path:
    """
    Returns the value of environment variable `envvar`, assumed to represent a
    directory path. If this variable is not defined, returns default.

    The environment variable can be defined in the users `.bashrc`, or in a file `.env`
    in the current working directory.

    Args:
        envvar: the input environment variable
        default: the default path, if the environment variable is not defined
            default values are predefined for the following variables:
                - DIR_DATA : "data" (in current working directory)
                - DIR_STATIC : DIR_DATA/"static"
                - DIR_SAMPLES : DIR_DATA/"sample_products"
                - DIR_ANCILLARY : DIR_DATA/"ancillary"
        create: whether to silently create the directory if it does not exist.
            If not provided this parameter defaults to False except for DIR_STATIC,
            DIR_SAMPLES and DIR_ANCILLARY.
    
    Returns:
        the path to the directory.
    """
    use_default = envvar not in environ
    default_create = False
    if envvar in environ:
        dirname = Path(environ[envvar])
    else:
        if default is None:
            # use a predefined default value
            if envvar == 'DIR_DATA':
                # Root of the data directory
                # All data in this directory are assumed disposable, and should be
                # downloaded on the fly
                # defaults to 'data' in the current working directory
                dirname = Path('data')
            elif envvar == 'DIR_STATIC':
                # static data files, required for processing
                dirname = getdir('DIR_DATA')/"static"
                default_create = True
            elif envvar == 'DIR_SAMPLES':
                # sample products, used for testing
                dirname = getdir('DIR_DATA')/"sample_products"
                default_create = True
            elif envvar == 'DIR_ANCILLARY':
                # ancillary data (downloaded on the fly)
                dirname = getdir('DIR_DATA')/"ancillary"
                default_create = True
            else:
                raise KeyError(f"{envvar} is not defined, and no default has been "
                               "provided.")
        else:
            dirname = Path(default)

    if create is None:
        create = default_create
    
    if not dirname.exists():
        if create:
            dirname.mkdir(exist_ok=True)
        else:
            raise NotADirectoryError(
                (f"Environment variable '{envvar}' is undefined, using default "
                 f"value '{dirname}'. " if use_default else "") + 
                f"Directory '{dirname}' does not exist. You may want to initialize it "
                f"with the following command: 'mkdir {dirname}'")

    return dirname


if __name__ == "__main__":
    dotenvs = find_dotenvs()
    log.info(f"all dotenv files (closest first): {dotenvs}")
    loaded = load_dotenvs()
    log.info(f"loaded {len(loaded)} dotenv file(s)")
