import math
import os
import subprocess
import time
import tracemalloc
from datetime import timedelta
from typing import Literal
from multiprocessing import Process, Event, Queue

import pandas as pd

from hygeos_core import log


class Chrono:
    """
    - name: str
    - unit "m" | "s" | "ms" | "us"
    """
    
    def __init__(self, name = 'chrono object', unit="m"):
        assert unit in ["m", "s", "ms", "us"]
        self.unit = unit
        self.name = name
        self.paused = False
        self.start_t: float = time.time()
        self.total_t: float = 0.
    
    def __enter__(self):
        self.restart()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.display(self.unit)
        return False
    
    def restart(self):
        self.paused = False
        self.start_t = time.time()
    
    def pause(self):
        if self.paused:
            raise RuntimeError('Cannot pause already paused chrono object')
        self.paused = True
        self.total_t += time.time()- self.start_t

    def elapsed(self) -> timedelta:
        if self.paused:
            return timedelta(seconds=(self.total_t))
        else:
            return timedelta(seconds=(time.time() - self.start_t))
        
    def reset(self) -> timedelta:
        dt = self.elapsed()
        self.start_t: float = time.time()
        self.total_t: float = 0.
        return dt
    
    def laps(self) -> timedelta:
        return self.reset()
        
    def stop(self) -> timedelta:
        self.paused = True
        self.total_t += time.time() - self.start_t
        return timedelta(seconds=self.total_t)

    def display(self, unit: Literal["m", "s", "ms", "us"] | None = None):
        if unit:
            assert unit in ["m", "s", "ms", "us"]
        else:
            unit = self.unit
        
        t = self.elapsed().total_seconds()
        if unit == "m":
            m = int(t) // 60
            s = t % 60
            log.info(f"Chrono: [{self.name}] took {m:02}m{s:06.3f}s to complete.")
        else:
            coefs = {
                "s":  1, 
                "ms": 1e3, 
                "us": 1e6
            }
            coef = coefs[unit]
            
            log.info(f"Chrono: [{self.name}] took {t * coef:.3f}{unit} to complete.")


class RAM:
    """
    - name: str
    """
    
    def __init__(self, name = 'ram object'):
        self.name = name
        self.paused = False
        self.peak: float = 0.
        self.current: float = 0.
        tracemalloc.start()
    
    def __enter__(self):
        self.restart()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.display()
        return False
    
    def restart(self):
        self.paused = False
        tracemalloc.start()
    
    def pause(self):
        if self.paused:
            raise RuntimeError('Cannot pause already paused chrono object')
        self.paused = True
        current, peak = self.stop()
        self.peak = max(self.peak,peak)
        self.current = current
 
    def elapsed(self):
        if not self.paused: 
            self.stop()
            self.restart()
        return self.peak, self.current
               
    def reset(self):
        current, peak = self.stop()
        self.restart()  
        self.peak: float = 0.
        self.current: float = 0.      
        return current, peak
    
    def laps(self):
        return self.reset()
        
    def stop(self):
        current, peak = tracemalloc.get_traced_memory()
        self.peak = max(self.peak, peak)
        self.current = current
        tracemalloc.stop()
        return self.current, self.peak

    def display(self):
        str_peak = self._format_size(self.peak)
        str_current = self._format_size(self.current)
        msg = "RAM usage: [{}] use {} at peak and {} at the end of execution."         
        log.info(msg.format(self.name, str_peak, str_current))
    
    def _format_size(self, byte_size):
        scale = {
            0: "B",
            3: "KB",  
            6: "MB", 
            9: "GB", 
            12: "TB", 
        }
        
        sc = None
        if 0 <= byte_size < 10**3: 
            sc = 0
        elif 10**3 <= byte_size < 10**6: 
            sc = 3
        elif 10**6 <= byte_size < 10**9: 
            sc = 6
        elif 10**9 <= byte_size < 10**12: 
            sc = 9
        elif 10**12 <= byte_size:
            sc = 12
        else:
            raise Exception('ERROR ', f"byte_size: {byte_size}")
            
        number = byte_size / (1024**(sc/3))
        unit = scale[sc]
        
        return f"{number:0.1f}{unit}"


class Monitor:
    
    """
    Meta-structure to monitor some variables in a script
    """
    
    def __init__(self, name: str = 'monitor object', 
                 time: Chrono | None = None, 
                 ram: RAM | None = None):
        self.name = name
        self.trackers = []
        
        if ram is None: ram = RAM(self.name)
        if time is None: time = Chrono(self.name)
        
        if ram : self.trackers.append(ram)
        if time: self.trackers.append(time)
    
    def __enter__(self):
        for t in self.trackers:
            t.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for t in self.trackers:
            t.__exit__(exc_type, exc_val, exc_tb)
        return False
    
    def restart(self):
        for t in self.trackers:
            t.restart()
    def pause(self):
        for t in self.trackers:
            t.pause()
 
    def elapsed(self):
        out = []
        for t in self.trackers:
            out.append(t.elapsed())
        return out
               
    def reset(self):
        out = []
        for t in self.trackers:
            out.append(t.reset())
        return out
    
    def laps(self):
        for t in self.trackers:
            t.laps()
        
    def stop(self):
        out = []
        for t in self.trackers:
            out.append(t.stop())
        return out

    def display(self):
        for t in self.trackers:
            t.display()


def dask_graph_stats(ds) -> pd.DataFrame:
    """
    Get statistics about the dask graph for each variable in the dataset `ds`.

    Returns a pandas DataFrame with the following columns:
        - var: The name of the variable.
        - graph_len: The length of the dask graph for the variable.
        - n_chunks: The number of chunks in the dask graph for the variable.
        - per_chunk: graph_len/n_chunks.

    Example:
    >>> print(dask_graph_stats(ds).to_string(index=False))
    """
    df = pd.DataFrame()

    for var in ds:
        data = ds[var].data

        if hasattr(data, 'chunks'):
            len_graph = len(data.__dask_graph__())
            nchunks = math.prod([len(x) for x in data.chunks])
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "var": [var],
                            "graph_len": [len_graph],
                            "n_chunks": [nchunks],
                            "per_chunk": [len_graph / nchunks],
                        }
                    ),
                ]
            )
        else:
            pass
    
    return df
            

def ram_probe(pid: int, period: float, stop_event, result_queue: Queue):

    def _get_page_size_kb():
        # cf so Question 4888067
        cmd = "cat /proc/meminfo | grep -E '^Mapped:' | awk '{print $2}' && cat /proc/vmstat | grep nr_mapped | cut -d' ' -f2"
        mapped, nr_mapped = subprocess.check_output(cmd, shell=True).decode().split()
        return int(int(mapped) / int(nr_mapped))
    page_size_kb = _get_page_size_kb() 
    
    def _get_process_ram_usage():
        cmd = f"cat /proc/{pid}/statm"
        ram = int(subprocess.check_output(cmd, shell=True).decode().split(" ")[1]) * page_size_kb * 1024 
        return ram
        
    # reference = _get_process_ram_usage()
    
    def get_mem_usage():
        return _get_process_ram_usage() # - reference
    
    # probe for mem usage 4x per second
    while not stop_event.is_set():
        time.sleep(period)
        result_queue.put(get_mem_usage())

    # def display(self):
        # str_peak = self._format_size(self.peak)
        # str_current = self._format_size(self.current)
        # msg = "RAM usage: [{}] use {} at peak and {} at the end of execution."         
        # log.info(msg.format(self.name, str_peak, str_current))

class MemHist:
    
    def __init__(self, tickrate: int = 5):
        assert tickrate != 0 
        assert tickrate < 100 
        
        self.period = float(1 / tickrate) 
        self.stop_event = Event()
        self.result_queue = Queue()
        self.pid = os.getpid()
        
    def __enter__(self):
        self.probe = Process(target=ram_probe, args=(self.pid, self.period, self.stop_event, self.result_queue))
        self.probe.start()

        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.probe.join()
        
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get())

        peak = max(results)
        width = 40
        
        print("RAM Histogram:")
        print(results)
        for val in results:
            print('#' * math.ceil(val/peak * width))
        print(f"peak: {MemHist._format_size(peak)}")
        # print(f"peak (raw): {peak}")
        
    @staticmethod
    def _format_size(byte_size: int) -> str:
        scale = {
            0: "B",
            3: "KB",  
            6: "MB", 
            9: "GB", 
            12: "TB", 
        }
        
        sc = None
        if 0 <= byte_size < 10**3: 
            sc = 0
        elif 10**3 <= byte_size < 10**6: 
            sc = 3
        elif 10**6 <= byte_size < 10**9: 
            sc = 6
        elif 10**9 <= byte_size < 10**12: 
            sc = 9
        elif 10**12 <= byte_size:
            sc = 12
        else:
            raise Exception('ERROR ', f"byte_size: {byte_size}")
            
        number = byte_size / (1024**(sc/3))
        unit = scale[sc]
        
        return f"{number:0.1f}{unit}"
