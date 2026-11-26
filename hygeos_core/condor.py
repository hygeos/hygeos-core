import warnings
from hygeos_core.process.condor import (
    get_htcondor_logdir,
    get_htcondor_params,
    get_htcondor_runner,
    get_htcondor_client,
)

warnings.warn("Please import from hygeos_core.process.condor", DeprecationWarning)
