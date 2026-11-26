
import inspect
from pathlib import Path
from typing import Iterable, Literal

from hygeos_core.static.interface import interface
from hygeos_core.static.abstract import abstract
from hygeos_core import log

from math import isnan


class ConstraintError(Exception):
    pass

@abstract
class Constraint:
    context = ""
    
    @abstract
    def check(param):
        return

    def contextstr(self):
        if self.context == "": return self.context
        return f"(Context: {self.context}) "

    def throw(self, *args):
        log.error("ConstraintError: ", self.contextstr(), *args, e=ConstraintError)
    
    
    def __str__(self):
        subconstraints = vars(self).copy()
        if "context" in subconstraints: del subconstraints["context"]
        
        strings = []
        string = f"[{self.context}] " if self.context != "" else ""
        string += f"{self.__class__.__name__} constraint: "
        for s, v in subconstraints.items(): 
            strings.append(f"{s}: {v}")
        string += ", ".join(strings)
        
        return string

class path(Constraint):
    
    @interface
    def __init__(self, *, exists: bool=True, mode=Literal["dir", "file"], context:str=""):
        self.exists = exists
        self.mode = mode
        self.context = context

    @interface
    def check(self, p: Path):
        
        if self.mode == "dir":
            if p.exists() != self.exists:
                if self.exists:          log.error(self.contextstr(), f"Path \"{p}\" does not exists. Expected existing directory path", e=ConstraintError)
                elif not self.exists:    log.error(self.contextstr(), f"Path \"{p}\" exist. Expected non existing directory path", e=ConstraintError)
            if p.is_file():
                log.error(self.contextstr(), f"Path \"{p}\" is a file. Expected a directory.", e=ConstraintError)
                
        if self.mode == "file":
            if p.exists() != self.exists:
                if self.exists:          log.error(self.contextstr(), f"Path \"{p}\" does not exists. Expected existing file path", e=ConstraintError)
                elif not self.exists:    log.error(self.contextstr(), f"Path \"{p}\" exist. Expected non existing file path", e=ConstraintError)
            if p.is_dir():
                log.error(self.contextstr(), f"Path \"{p}\" is directory. Expected a file", e=ConstraintError)

        return

class none(Constraint):
    """
    Empty constraint
    """
    def __init__(self, context:str=""):
        self.context = context

class float(Constraint):
    
    @interface
    def __init__(self, *, minimum: float=None, maximum: float=None, nan_allowed=False, context:str=""):
        self.minimum = minimum
        self.maximum = maximum
        self.nan_allowed = nan_allowed
        self.context = context

    @interface
    def check(self, f: float):
        
        if isnan(f):
            if self.nan_allowed: return
            elif not self.nan_allowed:
                log.error(self.contextstr(), f"Float value {f} expected to not be NaN value", e=ValueError)
        
        if self.minimum is not None and f < self.minimum: log.error(self.contextstr(), f"Float value {f} expected to be greater or equal than {self.minimum}", e=ValueError)
        if self.maximum is not None and f > self.maximum: log.error(self.contextstr(), f"Float value {f} expected to be lower or equal than {self.maximum}", e=ValueError)
        
        return

class int(Constraint):
    
    @interface
    def __init__(self, *, minimum: int=None, maximum: int=None, context:str=""):
        self.minimum = minimum
        self.maximum = maximum
        self.context = context

    @interface
    def check(self, i: int):
        
        if self.minimum is not None and i < self.minimum: log.error(self.contextstr(), f"Int value {i} expected to be greater or equal than {self.minimum}", e=ValueError)
        if self.maximum is not None and i > self.maximum: log.error(self.contextstr(), f"Int value {i} expected to be lower or equal than {self.maximum}", e=ValueError)
        
        if self.minimum is None and self.minimum is None:
            log.error(self.contextstr(), "Either minimum or maximum parameter must be set", e=ValueError)

        return

class bool(Constraint):

    @interface
    def __init__(self, context:str=""):
        self.context = context
    
    @interface
    def check(self, b: bool):
        return


class literal(Constraint):
    
    def __init__(self, values: Iterable, context:str=""):
        self.values = values
        self.context = context
    
    @interface
    def check(self, value):
        
        found_match = False
        for auth_val in self.values:
            if value == auth_val:
                found_match = True
                break
        
        if not found_match:
            log.error(self.contextstr(), f"Value \"{value}\" does not match any literal value passed: {str(self.values)}", e=ConstraintError)