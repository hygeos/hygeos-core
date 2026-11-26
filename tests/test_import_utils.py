import os

from hygeos_core.import_utils import import_module


def test_import_module():
    # Test importing a module
    imported = import_module('os')
    assert imported is os

    # Test importing an attribute
    imported_path = import_module('os.path')
    assert imported_path is os.path

    # Test importing a function
    imported_join = import_module('os.path.join')
    assert imported_join is os.path.join