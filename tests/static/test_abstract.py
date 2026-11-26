import pytest

from hygeos_core.static.Exceptions import ClassIsAbstract, MethodIsAbstract
from hygeos_core.static.abstract import abstract


def test_base():
    
    @abstract
    class Object(object):
        
        @abstract
        def allocate(self):
            pass
    
    with pytest.raises(ClassIsAbstract):
        obj = Object()  
    
    class Thing(Object):
        pass    
    
    with pytest.raises(MethodIsAbstract):
        thing = Thing()


def test_abstract_specified():
    @abstract
    class Object(object):
        
        @abstract
        def allocate(self):
            pass
    
    class Thing(Object):
        def allocate(self):
            pass    
    
    thing = Thing()

def test_overiden_with_abstract():
    """
    Verifies that the recursive check is working according to depth levels
    """
    
    class Object(object):
        
        def allocate(self):
            pass
    
    @abstract
    class Thing(Object):
        
        @abstract
        def allocate(self):
            pass
    
    class Truc(Thing):
        pass
    
    with pytest.raises(MethodIsAbstract):
        truc = Truc()