from datetime import datetime, timedelta, date


def round_date(date, h):
    """
    Round a date to the bracketing hours, by steps of `h` hours
    """
    assert isinstance(h, int)
    day = datetime(date.year, date.month, date.day)
    d0 = day + timedelta(hours=h*(date.hour//h))
    d1 = d0 + timedelta(hours=h)
    return (d0, d1)


def closest(date, h):
    """
    Round a date to the closest hour, by steps of `h` hours
    """
    assert isinstance(h, int)
    d0 = round_date(date, h)[0]
    if date - d0 < timedelta(hours=float(h)/2.):
        return d0
    else:
        return d0 + timedelta(hours=h)


def date_range(date_start: date, date_end: date) -> list[date]:
    """
    Returns a list of days starting from date_start, up to date_end included
    """
    days_list = [date_start + i*timedelta(days=1) for i in range((date_end - date_start).days + 1)]
    return days_list


def time_range(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    """
    Returns a list of datetime starting from start, up to end included
    """
    assert end > start
    dt = (end - start) // step
    datelist = [start + i*step for i in range(dt)]
    if end not in datelist: datelist += [end]
    return datelist

def now_isofmt():
    """
    Returns now in iso format
    """
    return datetime.now().isoformat()

def duration(s):
    """
    Returns a timedelta from a string `s`
    """
    if s.endswith('w'):
        return timedelta(weeks=float(s[:-1]))
    elif s.endswith('d'):
        return timedelta(days=float(s[:-1]))
    elif s.endswith('h'):
        return timedelta(hours=float(s[:-1]))
    else:
        raise Exception(f'Can not convert "{s}"')

def get_month_range(year: int, month: int):
    """
    Returns a couple of date corresponding to start and end of the month 
    """
    assert month <= 12, f'Invalid month, got {month}'
    start = date(year, month, 1)
    next_month = date(year+1,1,1) if month==12 else date(year,month+1,1)
    return start, next_month-timedelta(1)