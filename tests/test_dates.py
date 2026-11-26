from datetime import datetime, timedelta

from hygeos_core.dates import time_range


def test_time_range_non_multiple_step():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 10)
    res = time_range(start, end, timedelta(hours=3))
    assert [d.hour for d in res] == [0, 3, 6, 9, 10]


def test_time_range_multiple_step():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 9)
    res = time_range(start, end, timedelta(hours=3))
    assert [d.hour for d in res] == [0, 3, 6, 9]
