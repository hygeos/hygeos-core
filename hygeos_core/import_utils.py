import importlib
from typing import Any

def import_module(module_path: str) -> Any:
    """Import a module or attribute from a dotted module path.

    Args:
        module_path (str): The dotted path to the module or attribute,
            e.g., 'package.module.myclass'.

    Returns:
        The imported object (module, class, function, or any other attribute).
    """
    if "." in module_path:
        p, m = module_path.rsplit(".", 1)
        mod = importlib.import_module(p)
        obj = getattr(mod, m)
    else:
        obj = importlib.import_module(module_path)

    return obj