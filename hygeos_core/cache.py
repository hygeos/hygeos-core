import warnings
from hygeos_core.files.cache import (
    cache_dataframe,
    cache_json,
    cache_pickle,
    cache_dataset,
    cachefunc,
)

warnings.warn("Please import from hygeos_core.files.cache", DeprecationWarning)
