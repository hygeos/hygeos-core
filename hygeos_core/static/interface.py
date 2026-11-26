from inspect import isclass, signature, _empty
from typing import get_origin, get_args

from hygeos_core.static.Exceptions import WrongUsage, InterfaceException
from typing import Any
from hygeos_core import log

def _compile_passed_signature(
    expected_signature,
    args,
    kwargs,
    default_params,
):
    passed_signature = {}
    
    args = list(args)

    for i, p in enumerate(expected_signature):
        passed_signature[p[0]] = _param(name=p[0], hints=p[1], index=i)
    
    for n, v in kwargs.items():
        passed_signature[n].set_value(v)
    
    unmatched_params = [p for p in passed_signature.values() if not p.has_value()]
    unmatched_params = sorted(unmatched_params, key=lambda p: p.index)
    
    for i, value in enumerate(args):
        unmatched_params[i].set_value(value)
    
    # add default values to param which are missing their values
    default_params = {k: v for k, v in default_params}
    for p in passed_signature.values():
        if not p.has_value():
            p.set_value(default_params[p.name])
            
        if p.name in default_params and default_params[p.name] == None: 
            p.hints = p.hints | None
    
    for param in passed_signature.values():
        assert param.has_value() # should always be true
            
    
    return passed_signature.values()

class _param:
    
    def __init__(self, name, hints, index):
        self.name = name
        self.hints = hints
        self.index = index
    
    def set_value(self, value):
        self.value = value
    
    def __str__(self):
        ret = f"({self.name}, {self.hints}"
        if hasattr(self, "value"):
            ret += f", {self.value}"
        else: 
            ret += ", NOT MATCHED YET"
        ret += ")"
        return ret
    
    def __repr__(self):
        return self.__str__()
        
    def has_value(self):
        return hasattr(self, "value")
    

def interface(function):
    """
    Declare a function or method as an Interface
    Raise an error if types passed do not match definition
    """
    
    if isclass(function):
        mess = f'\n\tCannot declare class \'{function.__name__}\' as an interface, only functions or methods can be'
        log.error(mess, e=WrongUsage)
        
    
    def wrapper(*args, **kwargs):
        
        if hasattr(interface, "disabled"):
            return function(*args, **kwargs)
        
        # retrieve meta infos about function signature and passed parameters
        # construct datastructures used
        expected_signature = [(i.name, i.annotation) for i in signature(function).parameters.values()]
        expected_return_type = signature(function).return_annotation
        
        full_default_values = [(k, v.default) for k, v in signature(function).parameters.items()]
        default_params = [item for item in full_default_values if item[1] is not _empty]
        
        passed_signature = _compile_passed_signature(
            expected_signature,
            args,
            kwargs,
            default_params,
        )
        
        errors = []
        
        # Type checking
        for p in passed_signature:
            
            ref = expected_signature.pop(0)
            assert ref[0] == p.name # if not true => Reconstruction is desynced.
            
            hints = p.hints
            if hints == _empty: # no type hint provided at function declaration
                continue
            
            # Inheritance management (subclass should be allowed in place of superclass)
            vtypes = [type(p.value)] # vtypes is a list of types for the value (namely its type and all superclasses)
            if hasattr(p.value, "__class__"):
                superclasses = list(p.value.__class__.__bases__)
                if object in superclasses:
                    superclasses.remove(object)
                vtypes += superclasses

            # process type hint
            if get_origin(hints) == type(int|float):
                hints = get_args(hints)
            else: hints = [hints]
            # removes the subtyping list list[str] -> list
            exploded_hints = [get_origin(hint) or hint for hint in hints] 
            # list[str]|float -> [class list, class float]

            matched = False
            for vtype in vtypes: # iterate value's class and superclasses
                for hint in exploded_hints:
                    if issubclass(vtype, hint):
                        matched = True
                        break
                if matched:
                    break
            
            if not matched:
                errors.append(p)
        
        # Error management:
        # raise error if at least one mismatch
        if len(errors) != 0: # error on at least one parameter
            mess = f'\n\tFunction \'{function.__name__}\': Invalid parameters types passed:'  
            for p in errors:
                param, expect, actual, value = p.name, p.hints, type(p.value), p.value
                if type(p.hints) is tuple: # better message for unions e.g: int | float
                    expect = " or ".join([str(i) for i in p[1]])
                
                mess += (f'\n\t\tParameter \'{param}\' expected: {expect} got {actual} with value: {value}')
            log.error(mess, e=InterfaceException)

        result = function(*args, **kwargs)

        if      (expected_return_type != _empty) \
            and (expected_return_type != Any) \
            and not isinstance(result, expected_return_type):
            
            mess = f'\n\tFunction \'{function.__name__}\': Invalid type returned: '
            mess += f"Expected {expected_return_type} got {type(result)}"  
            log.error(mess, e=InterfaceException)
        
        return result
        
    return wrapper

interface.disable = lambda : setattr(interface, "disabled", True)
interface.enable =  lambda : delattr(interface, "disabled")

interface.disable()

# def debug_infos():
#     print("unnamed_params      ", unnamed_params)
#     print("named_params        ", named_params)
#     print("full_default_values ", full_default_values)
#     print("args                ", args)
#     print("kwargs              ", kwargs)
#     print("default_params      ", default_params)
#     print("-" * 22)
#     print("expected      ", expected_signature)
#     print("-" * 22)
#     print("compiled      ", passed_signature)