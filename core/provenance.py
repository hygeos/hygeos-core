"""
Provenance information: versions of the Python modules used in a processing

The version of a module is determined from the code that is actually imported:

- If the module is imported from a git checkout (e.g. an editable install),
  its version is given by `git describe --tags --always --dirty`, which
  identifies the exact commit and whether there are uncommitted changes.
- If the module was installed from a git repository (pip/pixi git
  dependency), its version is the distribution version, followed by the
  installed commit, e.g. "1.2.4+g2fcb634a1".
- Otherwise, its version is the distribution version (importlib.metadata),
  or the `__version__` attribute of the module.

Example:
    >>> get_versions(["core", "numpy"])
    {'core': 'v1.3.0-58-g2fcb634-dirty', 'numpy': '2.3.4'}
"""

import importlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

UNKNOWN = "unknown"


def _module_dir(module: str) -> Optional[Path]:
    """
    Directory of the top-level module `module`, or None if it can not be found
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:  # package
        locations = list(spec.submodule_search_locations)
        return Path(locations[0]) if locations else None
    if spec.origin and spec.origin not in ("built-in", "frozen"):
        return Path(spec.origin).parent
    return None


def _git(directory: Path, *args: str) -> Optional[str]:
    """
    Run a git command in `directory`, return its stripped stdout or None
    """
    try:
        res = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def git_describe(directory: Path) -> Optional[str]:
    """
    Version of the git checkout containing `directory`, provided that the
    files of `directory` are tracked in this checkout (so that a module
    installed in an environment located within an unrelated git repository is
    not mistaken for a checkout).

    Returns `git describe --tags --always --dirty`, or None.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    tracked = _git(directory, "ls-files", "--", ".")
    if not tracked:
        return None
    return _git(directory, "describe", "--tags", "--always", "--dirty")


def _distribution(module: str) -> Optional[importlib.metadata.Distribution]:
    """
    Distribution providing the top-level module `module`
    """
    names = importlib.metadata.packages_distributions().get(module, [])
    for name in [*names, module]:
        try:
            return importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def _vcs_commit(dist: importlib.metadata.Distribution) -> Optional[str]:
    """
    Commit of a distribution installed from a version control system (PEP 610)
    """
    try:
        direct_url = dist.read_text("direct_url.json")
        if not direct_url:
            return None
        return json.loads(direct_url).get("vcs_info", {}).get("commit_id")
    except (OSError, ValueError, AttributeError):
        return None


@lru_cache
def get_version(module: str) -> str:
    """
    Version of the top-level module `module` (import name), determined from
    the code that is actually imported (see module docstring).

    Returns "unknown" if the version can not be determined.
    """
    if module == "python":
        return platform.python_version()

    # module imported from a git checkout
    directory = _module_dir(module)
    if directory is not None:
        describe = git_describe(directory)
        if describe:
            return describe

    # installed distribution
    dist = _distribution(module)
    if dist is not None:
        commit = _vcs_commit(dist)
        if commit:
            return f"{dist.version}+g{commit[:9]}"
        return dist.version

    # module attribute
    if directory is not None:
        try:
            version = getattr(importlib.import_module(module), "__version__", None)
        except Exception:
            version = None
        if version:
            return str(version)

    return UNKNOWN


def get_versions(modules: Iterable[str]) -> dict[str, str]:
    """
    Versions of the top-level modules `modules` (import names), as a dict
    {module: version}. The special name "python" gives the Python version.
    """
    return {module: get_version(module) for module in modules}
