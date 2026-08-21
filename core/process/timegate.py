from datetime import datetime, time as dtime
from functools import total_ordering
from time import sleep
from core import log

@total_ordering
class _IntraDayTime:
    """Helper for comparing intra-day time values."""
    def __init__(self, time: str|dtime):

        if isinstance(time, str):

            assert (":" in time or "h" in time)
            c = ":" if ":" in time else "h"
            h, m = time.split(c)

        elif isinstance(time, dtime):
            h, m = time.hour, time.minute

        else:
            raise TypeError("Invalid type for IntraDayTime object constructor time param")

        self.h, self.m = int(h), int(m)

    @staticmethod
    def now():
        """Return the current time of day."""
        return _IntraDayTime(datetime.now().time())

    def __eq__(self, other: "_IntraDayTime"):
        """Check equality with another `_IntraDayTime`."""

        if not isinstance(other, _IntraDayTime):
            raise TypeError("Invalid type for IntraDayTime object comparison")
        
        return self.h == other.h and self.m == other.m

    def __lt__(self, other):
        """Check if less than another `_IntraDayTime`."""

        if not isinstance(other, _IntraDayTime):
            raise TypeError("Invalid type for IntraDayTime object comparison")
        
        if self.h == other.h:
            return self.m < other.m
        return self.h < other.h
        
    def __repr__(self):
        return f"_IntraDayTime({self.h:02}h{self.m:02})"
        
    def __str__(self):
        return f"{self.h:02d}h{self.m:02d}"
        

class TimeGate:
    
    def __init__(self, iterable, start: str = "20h00", end: str = "06h00", bypass_during_weekends=True, polling_period=60*5):
        """
        Args:
            iterable: The iterable to wrap.
            start: Start time in "HH:MM" or "HHhMM" format.
            end: End time in "HH:MM" or "HHhMM" format.
            bypass_during_weekends: If True, bypass the time gate on weekends.
            polling_period: Seconds between checks while gated.
        """
        self.iterator = iter(iterable)
        self.start = _IntraDayTime(start)
        self.end = _IntraDayTime(end)
        self.bypass_during_weekends = bypass_during_weekends
        self.polling_period = polling_period

        self.outer_mode = self.start > self.end
    
    def __iter__(self):
        return self

    def __next__(self):

        allow_iteration = False
        while not allow_iteration:

            # get current time
            t = _IntraDayTime.now()
            
            # conditions to allow the iterations
            if self.bypass_during_weekends and datetime.now().weekday() >= 5:
                allow_iteration = True
                
            if self.outer_mode:
                if t >= self.start or t <= self.end:
                    allow_iteration = True
            else:
                if t >= self.start and t <= self.end:
                    allow_iteration = True

            # if so iterate
            if allow_iteration:
                return next(self.iterator)

            # otherwise wait and retry later
            else:
                log.debug(log.rgb.gray, f"TimeGatedIterator: Current time {t} is outside [{self.start} - {self.end}]. Waiting {self.polling_period}s..")
                sleep(self.polling_period)
        