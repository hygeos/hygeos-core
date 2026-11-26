import pytest

from hygeos_core.progressbar import progressbar, msg_stack


@pytest.fixture(autouse=True)
def clean_stack():
    msg_stack.stack = []
    yield
    msg_stack.stack = []


def test_empty_iterable():
    assert list(progressbar([], "empty")) == []
    assert msg_stack.stack == []


def test_iterate_twice():
    p = progressbar([1, 2], "twice")
    assert list(p) == [1, 2]
    assert list(p) == [1, 2]
    assert msg_stack.stack == []


def test_break_removes_from_stack():
    for x in progressbar([1, 2, 3], "break"):
        break
    assert msg_stack.stack == []


def test_nested_break():
    outer = progressbar([1, 2], "outer")
    for x in outer:
        for y in progressbar([1, 2, 3], "inner"):
            break
        assert msg_stack.stack == [outer]
    assert msg_stack.stack == []
