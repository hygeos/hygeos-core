import threading
from dataclasses import dataclass, field

import pytest

from hygeos_core.process.deptree import Task, gen_executor, gen_sync


class _Root(Task):
    def init(self, deps):
        self.deps = deps
        return self


class Slow(Task):
    """A dependency that blocks until released (or a short timeout)"""

    def __init__(self):
        self.release = threading.Event()
        self.finished = False
        self.nruns = 0

    def run(self):
        self.nruns += 1
        # Under the bug, a parent runs while this is blocked and releases it.
        # When fixed, nobody releases it and the timeout is reached.
        self.release.wait(0.5)
        self.finished = True


class Parent(Task):
    def __init__(self, dep: Slow):
        self.deps = [dep]
        self.saw_finished = None

    def run(self):
        dep = self.deps[0]
        self.saw_finished = dep.finished
        dep.release.set()


def test_executor_shared_dependency_waits():
    """Every parent of a shared dependency waits for its completion"""
    slow = Slow()
    p1, p2 = Parent(slow), Parent(slow)
    gen_executor(_Root().init([p1, p2]))
    assert p1.saw_finished is True
    assert p2.saw_finished is True
    assert slow.nruns == 1


@dataclass
class Leaf(Task):
    x: int
    runs: list = field(default_factory=list, compare=False)

    def run(self):
        self.runs.append(self.x)


def test_executor_equal_distinct_tasks():
    """Distinct task objects that compare equal are not merged"""
    runs = []
    a, b = Leaf(0, runs), Leaf(0, runs)
    assert a == b
    root = _Root().init([a, b])
    gen_executor(root)
    assert runs == [0, 0]
    assert a.status == "success"
    assert b.status == "success"


cleaned = []


class CleanLeaf(Task):
    def __init__(self, i):
        self.i = i

    def cleanup(self):
        cleaned.append(self.i)


class FreshParent(Task):
    """dependencies() builds new objects at each call"""

    def dependencies(self):
        return [CleanLeaf(0), CleanLeaf(1)]


@pytest.mark.parametrize("gen", [gen_sync, gen_executor])
def test_cleanup_fresh_dependencies(gen):
    cleaned.clear()
    gen(FreshParent())
    assert sorted(cleaned) == [0, 1]


class Return42(Task):
    def run(self):
        return 42


class Failing(Task):
    def run(self):
        raise ValueError("failed")


def test_executor_root_result():
    assert gen_sync(Return42()) == 42
    assert gen_executor(Return42()) == 42


def test_executor_root_exception():
    with pytest.raises(ValueError):
        gen_sync(Failing())
    with pytest.raises(ValueError):
        gen_executor(Failing())


def test_executor_root_canceled_raises():
    """The root task is canceled because of a failed dependency: raise"""
    root = _Root().init([Failing()])
    with pytest.raises(Exception):
        gen_executor(root)
    assert root.status == "canceled"
