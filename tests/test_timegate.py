from datetime import datetime, time, timedelta
from unittest.mock import patch

import pytest

from core.process.timegate import TimeGate, _IntraDayTime


class FakeDatetime(datetime):
    """Datetime subclass with a controllable 'now'."""
    _now_value = None

    @classmethod
    def now(cls, tz=None):
        return cls._now_value


def set_now(h, m, weekday=0):
    """Set the fake current time. weekday: 0=Mon ... 6=Sun."""
    # 2026-08-17 is a Monday
    base = datetime(2026, 8, 17, h, m)
    FakeDatetime._now_value = base + timedelta(days=weekday)


# ---------------------------------------------------------------------------
# _IntraDayTime
# ---------------------------------------------------------------------------

def test_intradaytime_from_string():
    t = _IntraDayTime("13:45")
    assert t.h == 13
    assert t.m == 45


def test_intradaytime_from_string_h_separator():
    t = _IntraDayTime("13h45")
    assert t.h == 13
    assert t.m == 45


def test_intradaytime_from_datetime_time():
    t = _IntraDayTime(time(8, 5))
    assert t.h == 8
    assert t.m == 5


def test_intradaytime_invalid_type():
    with pytest.raises(TypeError):
        _IntraDayTime(123)


def test_intradaytime_invalid_string():
    with pytest.raises(AssertionError):
        _IntraDayTime("1345")


def test_intradaytime_now():
    with patch("core.process.timegate.datetime", FakeDatetime):
        set_now(10, 30)
        t = _IntraDayTime.now()
    assert t.h == 10
    assert t.m == 30


def test_intradaytime_eq():
    assert _IntraDayTime("10:00") == _IntraDayTime("10:00")
    assert _IntraDayTime("10:00") != _IntraDayTime("10:01")
    assert _IntraDayTime("10:00") != _IntraDayTime("11:00")


def test_intradaytime_lt():
    assert _IntraDayTime("09:59") < _IntraDayTime("10:00")
    assert _IntraDayTime("10:00") < _IntraDayTime("10:01")
    assert not _IntraDayTime("10:00") < _IntraDayTime("10:00")


def test_intradaytime_gt():
    assert _IntraDayTime("10:01") > _IntraDayTime("10:00")
    assert _IntraDayTime("11:00") > _IntraDayTime("10:59")
    assert not _IntraDayTime("10:00") > _IntraDayTime("10:00")


def test_intradaytime_comparison_invalid_type():
    t = _IntraDayTime("10:00")
    with pytest.raises(TypeError):
        t == "10:00"
    with pytest.raises(TypeError):
        t < "10:00"


def test_intradaytime_str_repr():
    # both separators are accepted as input, output is normalized to "h"
    for s in ("08:05", "08h05"):
        t = _IntraDayTime(s)
        assert str(t) == "08h05"
        assert repr(t) == "_IntraDayTime(08h05)"


# ---------------------------------------------------------------------------
# TimeGatedIter
# ---------------------------------------------------------------------------

def test_iter_inner_mode_allowed():
    """start < end: iteration allowed inside the window."""
    with patch("core.process.timegate.datetime", FakeDatetime):
        set_now(12, 0)  # inside [10:00 - 14:00]
        it = TimeGate(range(3), start="10:00", end="14:00")
        assert list(it) == [0, 1, 2]


def test_iter_inner_mode_gated_then_allowed():
    """Gated at first, then time moves inside the window."""
    with patch("core.process.timegate.datetime", FakeDatetime), \
         patch("core.process.timegate.sleep") as mock_sleep:
        set_now(9, 0)  # outside [10:00 - 14:00]
        it = TimeGate([1, 2], start="10:00", end="14:00")

        # First call: gated, sleeps, then time moves inside the window
        def advance(*args, **kwargs):
            set_now(11, 0)

        mock_sleep.side_effect = advance
        assert next(it) == 1
        assert next(it) == 2
        mock_sleep.assert_called_once()


def test_iter_outer_mode_allowed():
    """start > end: iteration allowed outside the window."""
    with patch("core.process.timegate.datetime", FakeDatetime):
        set_now(23, 0)  # outside [20:00 - 06:00]
        it = TimeGate(range(2), start="20:00", end="06:00")
        assert list(it) == [0, 1]


def test_iter_outer_mode_gated():
    """Inside the excluded window: gated until time moves out."""
    with patch("core.process.timegate.datetime", FakeDatetime), \
         patch("core.process.timegate.sleep") as mock_sleep:
        set_now(12, 0)  # inside the excluded window of [20:00 - 06:00]
        it = TimeGate([42], start="20:00", end="06:00")

        def advance(*args, **kwargs):
            set_now(21, 0)

        mock_sleep.side_effect = advance
        assert next(it) == 42
        mock_sleep.assert_called_once()


def test_iter_weekend_bypass():
    """On a weekend, the gate is bypassed regardless of time."""
    with patch("core.process.timegate.datetime", FakeDatetime):
        set_now(12, 0, weekday=5)  # Saturday, inside the excluded window
        it = TimeGate(range(2), start="20:00", end="06:00",
                           bypass_during_weekends=True)
        assert list(it) == [0, 1]


def test_iter_weekend_bypass_disabled():
    """With bypass disabled, weekends are gated like any other day."""
    with patch("core.process.timegate.datetime", FakeDatetime), \
         patch("core.process.timegate.sleep") as mock_sleep:
        set_now(12, 0, weekday=6)  # Sunday, inside the excluded window
        it = TimeGate([1], start="20:00", end="06:00",
                           bypass_during_weekends=False)

        def advance(*args, **kwargs):
            set_now(21, 0, weekday=6)

        mock_sleep.side_effect = advance
        assert next(it) == 1
        mock_sleep.assert_called_once()


def test_iter_exhaustion():
    with patch("core.process.timegate.datetime", FakeDatetime):
        set_now(12, 0)
        it = TimeGate([], start="10:00", end="14:00")
        with pytest.raises(StopIteration):
            next(it)
