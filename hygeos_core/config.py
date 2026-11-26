from pathlib import Path
from typing import Optional, Any
from copy import deepcopy
import toml

from hygeos_core.static import constraint
from hygeos_core.static import interface
from hygeos_core import log

from warnings import warn

warn('hygeos_core.config will be deprecated.', DeprecationWarning)

"""
Generic project configuration module.
Simpler version of hygeos_core.config module, more generic
Config path need to be passed as argument
"""

class Config:
    
    @interface
    def new_from_toml(config_path: Path, allow_empty_if_no_file: bool = False):
        
        file_found = config_path.is_file() and config_path.name.endswith(".toml")
        
        if not file_found and not allow_empty_if_no_file:
            raise RuntimeError(f"Invalid config file path {config_path}")
        
        configuration = {} if not file_found else toml.load(config_path)
        
        return Config(configuration)
    
    @interface
    def new_from_dict(config_dict: dict):
        return Config(config_dict)
    
    @interface
    def copy(self):
        return Config(deepcopy(self.config_dict))
    
    @interface
    def __init__(self, config_dict: dict):  
        self.config_dict: dict = config_dict
    
    @interface
    def get_subsection(self, subsection: str=None, default:dict=None) -> dict:
        """
        Returns the specified toml subsection
        returns the root if None provided
        """
        config_ptr = self.config_dict # get config content
        
        # Section or nested subsection
        if subsection is not None: 
            sections = subsection.split('.')
            
            # find subsection
            for sub in sections:
                if sub in config_ptr:
                    config_ptr = config_ptr[sub]
                else:
                    if default is not None:
                        return default
                    raise KeyError(f"Missing section [{subsection}] in config file")
        return config_ptr


    @interface
    def ingest(self, config_dict, override=True):
        """
        Ingest a configuration, if override is true then doubled keys are overriden by new config
        otherwise only add missing keys from new config
        keeping the keys defined in configuration and not overriden by itself.
        """
        
        if isinstance(config_dict, dict): # already a dict
            pass
        elif isinstance(config_dict, Config):
            config_dict = config_dict.config_dict
        else:
            log.error(f"Unexpected argument type \'{type(config_dict)}\', expected dict or Config object", e=ValueError)  
            
        for k, v in config_dict.items():  # iterate provided defaults keys, vals
            if k in self.config_dict:      # if the key is not already defined
                if not override:
                    continue
            self.config_dict[k] = v        
            
        
    @interface
    def get(
        self,
        key: str,
        subsection: str = None,
        *,
        default=None,
    ) -> Any:
        """
        Returns the key if present in the config file.
        If not present and no default provided: raises an error
        if default is provided (different than None): return default instead
        
        auto cast values to Path objects if its key starts with 'dir_' and its type is str
        """

        config_dict_ptr = self.get_subsection(subsection=subsection)
            
        # key managing
        if key not in config_dict_ptr:
            if default is not None:
                return default
            else:
                mess = "root section" if subsection is None else f"section [{subsection}]"
                raise KeyError( f"Missing key \'{key}\' in {mess} of in config file")
                
        # get value
        value = config_dict_ptr[key]
        
        # auto convert value to path if its key starts with 'dir_'
        if type(value) is str and key.startswith("dir_"):
            value = Path(value)

        return value
    
    def __str__(self):
        return str(self.config_dict)
    
    @interface
    def __getitem__(self, key: str, subsection=None):
        return self.get(key=key, subsection=subsection)