from pathlib import Path

import pytest

from hygeos_core.static import constraint
from hygeos_core.static.constraint import ConstraintError


def test_int_maximum_only():
    constraint.int(maximum=5).check(3)
    with pytest.raises(ValueError):
        constraint.int(maximum=5).check(6)


def test_int_no_bounds():
    with pytest.raises(ValueError):
        constraint.int().check(3)


def test_path_default_mode_checks_existence(tmp_path):
    with pytest.raises(ConstraintError):
        constraint.path(exists=True).check(Path('/nonexistent'))
    constraint.path(exists=True).check(tmp_path)
    constraint.path(exists=False).check(tmp_path / "missing")
    with pytest.raises(ConstraintError):
        constraint.path(exists=False).check(tmp_path)


def test_path_modes(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    constraint.path(mode="file").check(f)
    constraint.path(mode="dir").check(tmp_path)
    with pytest.raises(ConstraintError):
        constraint.path(mode="dir").check(f)
    with pytest.raises(ConstraintError):
        constraint.path(mode="file").check(tmp_path)
