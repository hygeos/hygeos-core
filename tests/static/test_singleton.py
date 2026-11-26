import pytest 

from hygeos_core.static.singleton import singleton


def test_base_usage():
    
    @singleton
    class Object:
        def __init__(self):
            self.ok = "OK"
    
    a = Object()
    b = Object.instance() or Object() # get instance if already present, instanciate otherwise 
    
    assert a is b 

    return

def test_duplicate():
    
    @singleton
    class Object:
        def __init__(self):
            self.ok = "OK"
        def sing(self): print("I dont want to set the world on fiiire")
            
    obj = Object()
    # the instance is stored inside the class
    assert Object.instance() is not None
    
    # the instance has not yet been cleared, cannot make another
    with pytest.raises(Exception):
        obj2 = Object()
    
    return
    

def test_free():
    
    @singleton
    class Object:
        def __init__(self):
            self.ok = "OK"
        def sing(self): print("I dont want to set the world on fiiire")
            
    obj = Object()
    
    del obj # free the only reference for the instance (internal ref is weak)
    assert Object.instance() is None
    
    obj2 = Object() # can now create a second instance

    return
    
    
def test_not_a_class():
    
    with pytest.raises(Exception):
        @singleton
        def function(): return
    
    return

def test_init_called_once():
    calls = []

    @singleton
    class Object:
        def __init__(self, x):
            calls.append(x)

    obj = Object(1)
    assert calls == [1]
    assert Object.instance() is obj
