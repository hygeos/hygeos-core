import pytest

from hygeos_core.static.interface import interface
from hygeos_core.static.Exceptions import InterfaceException


@pytest.fixture
def enabled_interface():
    was_disabled = hasattr(interface, "disabled")
    if was_disabled:
        interface.enable()
    yield
    if was_disabled:
        interface.disable()


def test_union_error_message(enabled_interface):

    @interface
    def f(x: int | float):
        return x

    assert f(1) == 1
    with pytest.raises(InterfaceException) as excinfo:
        f("a")
    assert "or" in str(excinfo.value)
