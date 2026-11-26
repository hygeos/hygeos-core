import os
from datetime import datetime
from pathlib import Path
from typing import Dict


def get_htcondor_logdir(logdir: Path | str | None) -> Path:
    """
    A default log directory for condor
    """
    if logdir is None:
        username = os.getlogin()
        return Path(f"/tmp/condor_logs_{username}/{datetime.now().isoformat()}")
    else:
        return Path(logdir)


def get_htcondor_params(
    cores=1,
    memory="1GB",
    disk="1GB",
    death_timeout="600",
    logdir: Path | str | None = None,
) -> Dict:
    """
    Parameters passed to HTCondorCluster
    """
    logdir = get_htcondor_logdir(logdir)
    print("HTCondor log directory is", logdir)
    return {
        "cores": cores,
        "memory": memory,
        "disk": disk,
        "death_timeout": death_timeout,
        "log_directory": logdir,
        # 'job_extra_directives': {
        #     # "log": "logs/dask_job_output.log",
        #     # "output": "logs/dask_job_output.out",
        #     # "error": "logs/dask_job_output.err",
        #     "sdefsdfshould_transfer_files": "YES",
        #     "initialdir": "$(LOCAL_DIR)",
        #     "environment": "TMP=$(LOCAL_DIR)/tmp",
        # },
    }


def get_htcondor_runner(maximum_jobs: int = 100, **kwargs):
    """
    Initialize a prefect DaskTaskRunner based on HTCondorCluster
    """
    from dask_jobqueue.htcondor import HTCondorCluster
    from prefect_dask import DaskTaskRunner

    task_runner = DaskTaskRunner(
        cluster_class=HTCondorCluster,
        cluster_kwargs=get_htcondor_params(**kwargs),
        adapt_kwargs={"maximum_jobs": maximum_jobs},
    )
    return task_runner


def get_htcondor_client(maximum_jobs=100, **kwargs):
    """
    Get a client for HTCondor cluster
    """
    from dask_jobqueue.htcondor import HTCondorCluster
    from dask.distributed import Client

    cluster = HTCondorCluster(**get_htcondor_params(**kwargs))
    cluster.adapt(maximum_jobs=maximum_jobs)
    client = Client(cluster)
    print("Dashboard address is", client.dashboard_link)
    return client
