import asyncio
from prefect import task as prefect_task
from prefect import get_client
from prefect.futures import PrefectFuture
from hygeos_core.process.deptree import _ensure_ref_count_initialized, _try_cleanup_dependency


"""
This module is imported by the deptree module.
It is used to isolate the part which requires prefect.
"""

def gen_prefect(
    task,
    concurrency_limit: dict | None = None,
    task_runner=None,
    log_prints: bool = True,
):
    """
    Generate a task in a flow using Prefect

    task (Task): the task to generate

    Note: a prefect server will be started automatically. To use an existing server,
    you can use the following variable:
        PREFECT_API_URL=http://127.0.0.1:4200/api
    (here with a local server started with `prefect server start`)

    task_runner can be a DaskTaskRunner() or DaskTaskRunner(get_htcondor_cluster())
    """
    from prefect import flow

    flowname = f'{task.__class__.__name__}'
    if concurrency_limit is None:
        concurrency_limit = {}
    return decorator_concurrency_limit(concurrency_limit)(
        flow(name=flowname, task_runner=task_runner, log_prints=log_prints)(
            gen_prefect_node
        )
    )(task)

async def set_concurrency_limit(d: dict):
    """
    Asynchronous function to set concurrency limits for tasks
    """
    async with get_client() as client:
        for item in d :
            await client.create_concurrency_limit(
                tag=item, 
                concurrency_limit=d[item]
                )


async def remove_concurrency_limit(d: dict):
    """
    Asynchronous function to remove concurrency limits for tasks
    """
    for item in d:
        async with get_client() as client:
            await client.delete_concurrency_limit_by_tag(tag=item)


def decorator_concurrency_limit(d: dict|None):
    """
    Decorator to apply concurrency limits to a function
    """
    def actual_decorator(func):
        def wrapper(*args, **kwargs):
            try:
                asyncio.run(set_concurrency_limit(d))  # Set concurrency limits
                result = func(*args, **kwargs)         # Execute the function
            finally:
                asyncio.run(remove_concurrency_limit(d))  # Remove concurrency limits
            return result
        if (d is None) or len(d) == 0:
            # bypass the decorator if useless
            return func
        else:
            return wrapper
    return actual_decorator


def wraptask(task, deps: list):
    """
    Wrap a task so that its dependencies are passed as futures to the wrapped
    function arguments

    `deps` is the result of `task.dependencies()`, which is called only once.
    """
    classname = str(task.__class__.__name__)
    def wrapper(*args):
        # Ensure reference counts are initialized for this task and its dependencies
        _ensure_ref_count_initialized(task, deps)

        # execute the run method
        result = task.run()
        # Try to cleanup each dependency (will only cleanup if all dependent tasks are done)
        for dep in deps:
            _try_cleanup_dependency(dep)
        return result
    return prefect_task(name=classname, tags=[classname])(wrapper)


def gen_prefect_node(
    task, root=True, all_futures=None, submitted=None
) -> PrefectFuture | None:
    """
    Generate a Task with Prefect

    If `root`, this is the root node: must wait for the root task to finish, otherwise
    futures are garbage collected before they finish.

    `submitted` maps id(task) to (task, future), so that a task shared by several
    parents is submitted only once (identity comparison).
    """
    if submitted is None:
        submitted = {}
    if id(task) in submitted:
        return submitted[id(task)][1]

    if task.done():
        return None

    if all_futures is None:
        all_futures = []

    # recursively trigger dependencies
    # call dependencies() only once: it may build new objects at each call
    deps = task.dependencies()
    futures = [
        gen_prefect_node(x, root=False, all_futures=all_futures, submitted=submitted)
        for x in deps
        if not x.done()
    ]

    # submit the current task by passing the futures dependencies as arguments
    future = wraptask(task, deps).submit(*futures)
    # keep a reference to the task so that its id() is not reused
    submitted[id(task)] = (task, future)

    # this list avoids that future are garbage collected before they are finished, which
    # raises a warning
    all_futures.append(future)

    if root:
        # avoid warning
        future.wait()
    
    return future
