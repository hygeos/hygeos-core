import platform
import subprocess
from pathlib import Path

import pytest

from core.provenance import (
    UNKNOWN, get_version, get_versions, git_describe, version_attrs)


@pytest.fixture(autouse=True)
def clear_cache():
    get_version.cache_clear()
    yield
    get_version.cache_clear()


def git(directory: Path, *args: str):
    subprocess.run(
        ["git", "-C", str(directory),
         "-c", "user.name=test", "-c", "user.email=test@example.com",
         *args],
        check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """
    A git checkout containing a package `provtestpkg`, imported from there
    """
    pkg = tmp_path / "provtestpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("x = 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    git(tmp_path, "tag", "v0.1.0")
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path


def test_git_checkout(repo):
    assert get_version("provtestpkg") == "v0.1.0"


def test_git_checkout_dirty(repo):
    (repo / "provtestpkg" / "__init__.py").write_text("x = 2\n")
    assert get_version("provtestpkg") == "v0.1.0-dirty"


def test_git_checkout_commits_after_tag(repo):
    (repo / "provtestpkg" / "__init__.py").write_text("x = 2\n")
    git(repo, "commit", "-q", "-am", "change")
    version = get_version("provtestpkg")
    assert version.startswith("v0.1.0-1-g")
    assert not version.endswith("-dirty")


def test_untracked_in_git_repo(repo, monkeypatch):
    """
    A module located in an untracked directory of an unrelated git repository
    (e.g. an environment within a project) is not versioned by this repository
    """
    env = repo / "env"
    (env / "provtestother").mkdir(parents=True)
    (env / "provtestother" / "__init__.py").write_text("__version__ = '3.2.1'\n")
    monkeypatch.syspath_prepend(str(env))
    assert get_version("provtestother") == "3.2.1"


def test_distribution():
    import pytest as module
    assert get_version("pytest") == module.__version__


def test_python():
    assert get_version("python") == platform.python_version()


def test_unknown():
    assert get_version("provtest_not_a_module") == UNKNOWN


def test_get_versions(repo):
    versions = get_versions(["provtestpkg", "python"])
    assert versions == {
        "provtestpkg": "v0.1.0",
        "python": platform.python_version(),
    }


def test_version_attrs(repo):
    assert version_attrs(["provtestpkg"]) == {"provtestpkg_version": "v0.1.0"}
    assert version_attrs({"pkg": "provtestpkg"}) == {"pkg_version": "v0.1.0"}


def test_get_git_commit(repo, monkeypatch):
    from core.files.fileutils import get_git_commit
    monkeypatch.chdir(repo)
    expected = git_describe(repo, tracked_only=False, tags=False)
    assert expected  # (commit hash: lightweight tags are not used)
    assert get_git_commit() == expected
    monkeypatch.chdir(repo.parent)  # not a git repository
    assert get_git_commit() == "<could not get git commit>"
