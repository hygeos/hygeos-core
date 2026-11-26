import hygeos_core
from hygeos_core import log
from time import sleep

import pytest

"""
only sensible in debug mode
"""

def test_base():
    
    log.silence(hygeos_core, log.lvl.ERROR)
    
    log.warning("test string")

def test_str_conversion():
    log.log(log.lvl.INFO, 1)
    
    
def test_box():
    
    log.info(
        "Let's put emphasis on ",
        log.rgb.orange("THIS"),
        " word"
    )

# TODO: rewrite this test ? 
# This should have never passed with log.pbar since the interfaces are wrong here..
 
# def test_progress_bar():
#     pbar = log.pbar(log.lvl.INFO, range(20), desc='test')
#     for i in pbar:
#         pbar.set_description(str(i))
#         pbar.write(str(i))
#         sleep(0.1)

def test_warning_invalid_type_raises_runtimeerror():
    with pytest.raises(RuntimeError):
        log.warning("msg", w=int)
    with pytest.raises(RuntimeError):
        log.warning("msg", w="not a class")


def test_error_invalid_type_raises_runtimeerror():
    with pytest.raises(RuntimeError):
        log.error("msg", e="not a class")
