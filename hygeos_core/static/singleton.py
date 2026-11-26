from inspect import isclass

from .Exceptions import WrongUsage

import weakref

def singleton(myclass):
    """
    Declare a Class as a singleton, which can only have a single instance at the same time
        - adds a instance() class method which returns the current instanced object
    """
    setattr(myclass, "_instance", weakref.ref(lambda: None)) # instantiate dead weakref
    
    if not isclass(myclass):
        raise WrongUsage(f"Argument '{myclass.__name__}' is should be a class, instead got {type(myclass)}")
    
    def _singleton_wrapper():
        # currently replace the new function
        def inner(self, *args, **kwargs):
            
            # return instance if exists
            if myclass._instance() is not None:
                raise WrongUsage(f"class '{myclass.__name__}' is already instanced, and is declared as a strict singleton")
                
            new_instance = None
            new_instance = object.__new__(myclass)
            
            new_instance.__init__(*args, **kwargs)
            
            setattr(myclass, "_instance", weakref.ref(new_instance))
            return new_instance
            
        return inner
    
    # add instance() class method, which returns the current instance of the class
    def return_instance():
        return myclass._instance()
    myclass.instance = return_instance
    
    myclass.__new__ = _singleton_wrapper()
    return myclass

