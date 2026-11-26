from inspect import isclass

from hygeos_core.static.Exceptions import ClassIsAbstract, MethodIsAbstract


def abstract(myclass_or_method):
    """
    Declare a Class or method as abstract
    Raise an error if trying to call __init__() or the method
      - Abstract Classes are meant to be derived in subclasses
      - Abstract Methods are meant to be overriden in subclasses
    """
    
    def _init_abstract_decorator(init_method):
        def inner(self, *args, **kwargs):
            
            if myclass_or_method is type(self): # didn't use issubclass to allow subclass usage of superclass
                raise ClassIsAbstract( f"\n\tClass '{myclass_or_method.__name__}' is abstract and is not meant be instanced")
                
            init_method(self, *args, **kwargs)
            
            # func: (isabstract, lvl)
            abstract_map = {}
            
            def fill_abstract_map(cls, abstract_map, lvl):
                # look recursively through functions, and base classes functions
                # for abstract classes
                for attr, value in cls.__dict__.items():
                    if callable(value) and hasattr(value, "__isAbstractMethod"):
                        abstract_map[attr] = (value.__isAbstractMethod, lvl)
                # recursive call
                for base_cls in cls.__bases__:
                    fill_abstract_map(base_cls, abstract_map, lvl+1)
            
            def unfill_abstract_map(cls, abstract_map, lvl):
                # look recursively through functions, and base classes functions
                # for abstract classes that have been implemented in sub-classes
                for attr, value in cls.__dict__.items():
                    if callable(value) and attr in abstract_map:
                        # if function is not marked as abstract, and if it is from a subclass that the abstract one
                        if not hasattr(value, "__isAbstractMethod") and abstract_map[attr][1] > lvl:
                            abstract_map[attr] = (False, lvl)
                # recursive call
                for base_cls in cls.__bases__:
                    unfill_abstract_map(base_cls, abstract_map, lvl+1)
            
            # kickstart recursive exploring at current class, level 0
            fill_abstract_map(self.__class__, abstract_map, 0)
            unfill_abstract_map(self.__class__, abstract_map, 0)
            
            filtered_abstract_list = [key for key, value in abstract_map.items() if value[0] is True]
            
            if len(filtered_abstract_list) > 0:
                mess = f"Can't instantiate abstract class {self.__class__} without an implementation for abstract methods:\n"
                for func in filtered_abstract_list:
                    mess += f"  '{func}'\n"
                raise MethodIsAbstract(mess)
                            
        return inner
    
    def _abstract_method_decorator(my_method):
        def inner(self, *args, **kwargs):
            raise MethodIsAbstract( f"\n\tMethod '{my_method.__name__}' from class '{type(self).__name__}' is abstract and is meant to be overriden")
        return inner
    
    if isclass(myclass_or_method):
        myclass_or_method.__init__ = _init_abstract_decorator(myclass_or_method.__init__)
        return myclass_or_method
    # else: # is a method
        
    method = _abstract_method_decorator(myclass_or_method)
    method.__isAbstractMethod = True
    
    return method
