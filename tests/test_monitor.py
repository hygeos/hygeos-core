# standard library imports
import numpy as np
import time
        
# third party imports
import pytest
        
# sub package imports
from hygeos_core.monitor import *


def test_chrono_base():
    
    def big_function():
        time.sleep(0.01)
        return 42
    
    with Chrono("Computation N°1", unit="ms"):
        res = big_function()
        print("result", res)
    
    print('end')

def test_chrono_nonpaused():
    
    c = Chrono()
    dt = c.elapsed()
    time.sleep(0.01)
    dt2 = c.elapsed()
    
    assert not dt == dt2
    

def test_chrono_pause():
    c = Chrono()
    c.pause()
    dt = c.elapsed()
    time.sleep(0.01)
    dt2 = c.elapsed()
    
    assert dt == dt2

def test_chrono_loop():
    c = Chrono(unit='s')
    for i in range(10):
        c.restart()
        time.sleep(0.01)
        c.pause()
        time.sleep(0.01)
    
    assert pytest.approx(c.total_t, rel=1e-1) == 10*0.01
    c.display()

def test_chrono_context():
    with Chrono() as c:
        c.stop()    
    
    
def test_ram_base():
    
    def big_function():
        a = [np.random.randint(0,10,(20,20)) for i in range(20)]
        return len(a)
    
    with RAM("Computation N°1"):
        res = big_function()
        print("result", res)
    
    print('end')

def test_ram_nonpaused():
    
    r = RAM()
    dt = r.elapsed()
    a = [np.random.randint(0,10,(20,20)) for i in range(1000)]
    dt2 = r.elapsed()
    
    assert not dt2 == dt

def test_ram_pause():
    r = RAM()
    r.pause()
    dt = r.elapsed()
    a = [np.random.randint(0,10,(20,20)) for i in range(20)]
    dt2 = r.elapsed()
    
    assert dt == dt2

def test_ram_loop():
    a,b = [], []
    r = RAM()
    for i in range(10):
        r.restart()
        a.append(np.random.randint(0,10,(20,20)))
        r.pause()
        b.append(np.random.randint(0,10,(20,20)))
    
    assert pytest.approx(r.current, 0.01) == 3300
    assert pytest.approx(r.peak, 0.01) == 3700
    r.display()
    
def test_ram_context():
    with RAM() as r:
        r.stop()   
    
    
def test_monitor_base():
    
    def big_function():
        a = [np.random.randint(0,10,(20,20)) for i in range(20)]
        time.sleep(1)
        return len(a)
    
    with Monitor():
        res = big_function()
        print("result", res)
    
    print('end')

def test_monitor_nonpaused():
    
    m = Monitor()
    out1 = m.elapsed()
    time.sleep(0.01)
    a = [np.random.randint(0,10,(20,20)) for i in range(20)]
    out2 = m.elapsed()
    
    assert all(dt1 != dt2 for dt1, dt2 in zip(out1, out2))
    

def test_monitor_pause():
    m = Monitor()
    m.pause()
    out1 = m.elapsed()
    time.sleep(0.01)
    a = [np.random.randint(0,10,(20,20)) for i in range(20)]
    out2 = m.elapsed()
    
    assert all(dt1 == dt2 for dt1, dt2 in zip(out1, out2))

def test_monitor_loop():
    m = Monitor()
    for i in range(10):
        m.restart()
        time.sleep(0.01)
        m.pause()
        time.sleep(0.01)
    m.display()

def test_monitor_context():
    with Monitor() as m:
        m.stop()   