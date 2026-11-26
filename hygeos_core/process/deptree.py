from datetime import datetime
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sys import argv
from time import sleep, time
from typing import Literal

from hygeos_core.interpolate import product_dict
from hygeos_core import log
from hygeos_core.ascii_table import ascii_table
import pandas as pd
import threading


class Task:
    """
    Base class for tasks implementing a dependency tree structure for execution workflow.

    This class provides the foundation for creating tasks that can be organized in a
    dependency graph and executed in the correct order. Tasks can be generated and run
    using the `gen` function with different execution modes (sync, executor, or prefect).

    Attributes:
        output: Optional attribute for storing the task's output path or result.
        deps: Optional list of Task instances that this task depends on.
        status: Task execution status

    Methods:
        dependencies(): Returns the list of dependent tasks. The default implementation
            returns the `deps` attribute, if it exists.
        run(): Implements the actual execution code
        done(): Checks if the task is completed. The default implementation checks if
            the `output` Path exists, if defined.
        cleanup(): Optional cleanup method executed when all parent tasks have finished

    Usage:
        Subclass Task and implement at minimum:
        - The `run()` method with your task's execution logic (if any)
        - Optionally override `dependencies()` or self.deps if your task has dependencies
        - Optionally override `cleanup()` if your task needs cleanup after execution
        - Set `output` attribute if your task produces output files

    Example:
        See the `sample` function for a complete example of Task definition and usage.
    """

    def dependencies(self) -> list:
        """
        Returns a list of dependencies, each being a `Task`.
        """
        if hasattr(self, "deps"):
            return getattr(self, "deps")
        else:
            return []

    def run(self):
        """
        Implements the actual execution code
        """
        pass

    def cleanup(self):
        """
        Optional cleanup method executed after the task finishes running
        """
        pass

    def done(self) -> bool:
        """
        Whether the task is done
        """
        if hasattr(self, 'output'):
            return Path(self.output).exists() # type: ignore
        else:
            return False


def gen(
    task: Task,
    mode: Literal['sync', 'executor', 'prefect'] = 'sync',
    **kwargs
):
    """
    Run a task and its dependencies

    Arguments:
        mode (str):
            - 'sync': sequential
            - 'executor': using executors
            - 'prefect': using prefect as a worflow manager
    """
    if mode == 'sync':
        return gen_sync(task, **kwargs)
    elif mode == 'executor':
        return gen_executor(task, **kwargs)
    elif mode == 'prefect':
        from hygeos_core.process.deptree_prefect import gen_prefect
        return gen_prefect(task, **kwargs)
    else:
        raise ValueError(f"Invalid mode '{mode}'")


def gen_sync(
    task: Task,
    verbose: bool = False,
):
    """Run a Task in sequential mode

    Args:
        task (Task): The task to generate
        verbose (bool): Whether to print verbose output
    """
    if not task.done():
        # call dependencies() only once: it may build new objects at each call
        deps = task.dependencies()
        # Ensure reference counts are initialized for this task and its dependencies
        _ensure_ref_count_initialized(task, deps)

        for d in deps:
            gen_sync(d, verbose=verbose)
        if verbose:
            log.info(f"Generating {task}...")
        try:
            result = task.run()
            return result
        finally:
            # Try to cleanup each dependency (will only cleanup if all dependent tasks are done)
            for d in deps:
                _try_cleanup_dependency(d)

def format_time(t):
    if t == "?": 
        return t
    if t < 60:
        return f"{t:.0f}s"
    elif t < 3600:
        m = t // 60
        s = t % 60
        return f"{m:.0f}m {s:.0f}s"
    else:
        h = t // 3600
        m = (t % 3600) // 60
        s = t % 60
        
        parts = []
        if h > 0:
            parts.append(f"{h:.0f}h")
            
        if m > 0:
            parts.append(f"{m:02.0f}m")
            
        if s > 0 and h == 0: # only show seconds if no hours
            parts.append(f"{s:02.0f}s")
            
        return "".join(parts)

def print_execution_summary(
    tasks: list, start_time: float, log_file: Path | None = None, finished: bool = False
):
    """
    Print a summary table showing count of tasks by status and class name.
    
    Args:
        tasks: List of Task instances to summarize
    """
    
    if not finished and not log_file:
        return
    
    l, r = 16, 15
    msg = log.rgb.orange("=======[ Execution Log ]=======") + "\n"
    
    # dislay status of the overall execution
    s = "Finished" if finished else "Running"
    # find color based on errors
    c = log.rgb.orange 
    if any(getattr(t, 'status', None) == 'error' for t in tasks):
        c = log.rgb.red
    else:
        if finished:
            c = log.rgb.green  # finished without errors

    msg +=  log.rgb.orange("Status: ".ljust(l))  + c(s.rjust(r)) + "\n"
    elapsed = time() - start_time
    
    # datetime from epoch
    start = datetime.fromtimestamp(start_time)
    
    msg += log.rgb.orange("Time spent: ".ljust(l)) + format_time(elapsed).rjust(r) + "\n"
    msg += log.rgb.orange("Start:".ljust(l)) + start.strftime('%H:%M:%S').rjust(r) + "\n"
    msg += log.rgb.orange("Date: ".ljust(l))  + datetime.now().strftime('%Y-%m-%d').rjust(r) + "\n"
    
    # Get unique statuses and classes directly from list comprehensions + set
    all_statuses = sorted(set(getattr(task, 'status', 'unknown') for task in tasks))
    all_classes = sorted(set(task.__class__.__name__ for task in tasks))
    
    if all_classes and all_statuses:
        # Create DataFrame directly using double nested loop
        columns = ['Class'] + all_statuses + ['Total']
        
        # Initialize DataFrame with the right shape
        df = pd.DataFrame(index=range(len(all_classes) + 1), columns=columns)
        
        # Fill rows for each class
        for i, class_name in enumerate(all_classes):
            df.iloc[i, 0] = class_name  # Class column
            class_total = 0
            
            for j, status in enumerate(all_statuses):
                # Count directly from tasks list
                count = sum(1 for task in tasks 
                           if task.__class__.__name__ == class_name 
                           and getattr(task, 'status', 'unknown') == status)
                df.iloc[i, j + 1] = str(count) if count > 0 else ''
                class_total += count
            
            df.iloc[i, -1] = str(class_total)  # Total column
        
        # Fill totals row
        totals_row_idx = len(all_classes)
        df.iloc[totals_row_idx, 0] = 'TOTAL'
        grand_total = 0
        
        for j, status in enumerate(all_statuses):
            # Count directly from tasks list
            status_total = sum(1 for task in tasks 
                             if getattr(task, 'status', 'unknown') == status)
            df.iloc[totals_row_idx, j + 1] = str(status_total)
            grand_total += status_total
        
        df.iloc[totals_row_idx, -1] = str(grand_total)
        
        # Replace NaN with empty strings
        df = df.fillna('')
        colors = dict(
            Class       = log.rgb.default,
            Pending     = log.rgb.orange,
            Running     = log.rgb.blue,
            Success     = log.rgb.green,
            Error       = log.rgb.red,
            Skipped     = log.rgb.cyan,
            Canceled    = log.rgb.orange,
            Total       = log.rgb.cyan,
        )
        
        sides = dict(
            Class       = "left",
            Pending     = "right",
            Running     = "right",
            Success     = "right",
            Error       = "right",
            Skipped     = "right",
            Canceled    = "right",
            Total       = "right",
        )
        
        table = ascii_table(df, colors=colors, sides=sides)
        table_msg = table.to_string(live_print=False, no_color=False)
        
        msg += table_msg + "\n"

    
    # Count and display failed tasks
    failed_tasks = [t for t in tasks if getattr(t, 'status', None) == 'error']
    nfailed = 5  # number of displayed failed tasks
    if failed_tasks:
        msg += log.rgb.orange(f"Failed tasks ({len(failed_tasks)}):")  + "\n"
        for t in failed_tasks[:nfailed]:
            e_msg = "" if not hasattr(t, "error_msg") else f"{t.error_msg}"
            msg += f"  - {t.__class__} {log.rgb.red(e_msg)}" + "\n"
        if len(failed_tasks) > 5:
            msg += "..."  + "\n"
        msg += str(log.rgb.orange) + "========================="  + "\n"

    # pad the output by one space left
    msg = msg.split("\n")
    msg = "\n ".join(msg)
    msg = " " + msg
    
    # using a lock shared accross threads to avoid race conditions
    if log_file is not None:
        file_lock = threading.Lock()
        with file_lock:
            # overwrite the output to a local file called /tmp/deptreelog
            with open(log_file, "w") as f:
                f.write(msg)
    else:
        log.disp(msg)
        

def run_with_status_update(task, verbose):
    """Wrapper function that sets status to running when actually executing"""
    setattr(task, "status", "running")
    if verbose:
        log.info(f"Generating {task}...")
    result = task.run()
    return result


def gen_executor(
    task: Task,
    executors: dict | None = None,
    default_executor=None,
    graph_executor=None,
    verbose: bool = False,
    _tasks: list | None = None,
    log_file: Path|None = None,
    _nodes: dict | None = None,
):
    """
    Recursively generate dependencies of `task`, then call its `run` method in a
    custom executor.

    Args:
        task: a Task object
        executors: dictionary of executors
        default_executor: the executor to use for all tasks not included in executors. Defaults
            to ThreadPoolExecutor
        graph_executor: an instance of ThreadPoolExecutor used for traversing the
            dependency tree with the `gen` function. It is useful to override the
            default ThreadPoolExecutor(512) if the tree is too large.
        _tasks: internal parameter to track tasks status and provide an execution 
            summary.
        _nodes: internal parameter mapping id(task) to its execution state, used to
            avoid duplicate task submissions.

    Example:
        from concurrent.futures import ThreadPoolExecutor
        from distributed import Client # dask executor

        gen(task,
            executors={'Level1' : ThreadPoolExecutor(2)},
            default_executor=Client()  # dask executor
                # Note: can also use dask_jobqueue to instantiate
                # an executor backed by a job queuing system like
                # HTCondor
        })

    For details on how to implement `task`, please check function hygeos_core.process.deptree.sample.
    """
    # Note: `gen` is defined as a function instead of an task method because it includes
    # non-serializable code, and task can be serialized (for example by the dask
    # distributed executor).
    
    _start_time = time()
    is_root_call = _tasks is None
    if _tasks is None:
        _tasks = []
        _start_time = time()
    if _nodes is None:
        _nodes = {}

    if graph_executor is None:
        graph_executor = ThreadPoolExecutor(512)
    
    if default_executor is None:
        default_executor = ThreadPoolExecutor()

    # Check if task is already being processed, using object identity (tasks may
    # compare equal by value, e.g. dataclasses, while being distinct objects).
    # Tasks are kept alive in _tasks, so their id() can not be reused.
    with _nodes_lock:
        node = _nodes.get(id(task))
        is_new = node is None
        if is_new:
            node = _nodes[id(task)] = _Node()
            # Set initial status
            if not hasattr(task, "status"):
                setattr(task, "status", "pending")
            _tasks.append(task)

    if not is_new:
        # Task is already submitted by another parent: wait for its completion
        node.event.wait()
        if node.exception is not None:
            raise node.exception
        return node.result

    try:
        node.result = _gen_executor_task(
            task,
            executors=executors,
            default_executor=default_executor,
            graph_executor=graph_executor,
            verbose=verbose,
            _tasks=_tasks,
            log_file=log_file,
            _nodes=_nodes,
            _start_time=_start_time,
            is_root_call=is_root_call,
        )
    except BaseException as e:
        node.exception = e
        raise
    finally:
        node.event.set()
    return node.result


class _Node:
    """Execution state of a task within a `gen_executor` call"""
    def __init__(self):
        self.event = threading.Event()  # set when the task is finished
        self.result = None
        self.exception = None


_nodes_lock = threading.Lock()


def _gen_executor_task(
    task: Task,
    executors,
    default_executor,
    graph_executor,
    verbose: bool,
    _tasks: list,
    log_file: Path | None,
    _nodes: dict,
    _start_time: float,
    is_root_call: bool,
):
    """
    Process a single (deduplicated) task for `gen_executor`
    """
    print_execution_summary(_tasks, _start_time, log_file)

    if not task.done():
        # call dependencies() only once: it may build new objects at each call
        deps = task.dependencies()
        # Ensure reference counts are initialized for this task and its dependencies
        _ensure_ref_count_initialized(task, deps)

        # run all dependencies on the graph executor
        if len(deps) > graph_executor._max_workers:
            raise RuntimeError(
                f"Task {task} has {len(deps)} dependencies, but graph_executor only "
                f"has {graph_executor._max_workers} maximum workers. Please use a "
                f"graph_executor with additional workers."
            )
        futures = [
            graph_executor.submit(
                gen_executor,
                d,
                executors=executors,
                default_executor=default_executor,
                graph_executor=graph_executor,
                verbose = verbose,
                _tasks=_tasks,
                log_file=log_file,
                _nodes=_nodes,
            )
            for d in deps
        ]

        # wait for all tasks to finish
        # if any error occurred in the task, it is raised now
        for t in futures:
            t.result()

        # Check if any dependency has an error status
        for dep in deps:
            if dep.status in ["error", "canceled"]:
                setattr(task, "status", "canceled")
                if verbose:
                    log.info(f"Task {task} canceled due to dependency error")
    
        # print_execution_summary(_tasks, _start_time)

        if getattr(task, "status") != "canceled":
            # get the executor for current `task`
            clsname = task.__class__.__name__
            if (executors is not None) and (clsname in executors):
                executor = executors[clsname]
            else:
                executor = default_executor

            # Keep status as pending until the task actually starts running
            # execute wrapped function on this executor and wait for result
            future = executor.submit(run_with_status_update, task, verbose)
            try:
                result = future.result()
                setattr(task, "status", "success")
                # Try to cleanup each dependency (will only cleanup if all dependent tasks are done)
                for dep in deps:
                    _try_cleanup_dependency(dep)
                print_execution_summary(_tasks, _start_time, log_file,
                                        finished=is_root_call)
                return result
            except Exception as e:
                setattr(task, "status", "error")
                setattr(task, "error_msg", str(e))
                # Try to cleanup each dependency even if the task failed
                for dep in deps:
                    _try_cleanup_dependency(dep)
                print_execution_summary(_tasks, _start_time, log_file)
                if verbose:
                    log.warning(f"Task {task.__class__} failed with error {str(e.__class__)}")
                if is_root_call:
                    print_execution_summary(_tasks, _start_time, log_file, finished=True)
                    raise
                return None
        elif is_root_call:
            print_execution_summary(_tasks, _start_time, log_file, finished=True)
            raise RuntimeError(f"Task {task} canceled due to failed dependencies")
    else:
        # Task is already done, mark as skipped if status is pending
        if getattr(task, "status", None) == "pending":
            setattr(task, "status", "skipped")
            # print_execution_summary(_tasks, _start_time)
    
    print_execution_summary(_tasks, _start_time, log_file, finished=is_root_call)



class TaskList(Task):
    def __init__(self, deps: list):
        """
        Construct a Task that depends on a list of dependencies (other tasks)

        Ex: TaskList([MyTask1(), MyTask2()])
        depends on [
            MyTask1(),
            MyTask2(),
        ]
        """
        self.deps = deps

    def dependencies(self):
        return self.deps


class TaskProduct(Task):
    def __init__(self, cls, others={}, **kwargs):
        """
        Construct a Task that depends on the cartesian product of
        the given arguments.
        The `other` arguments are passed directly to the class.

        cls: a class that inherits from Task

        Ex: TaskProduct(MyTask, {
            'a': [1, 2],
            'b': [3, 4],
        })
        depends on [
            MyTask(a=1, b=3),
            MyTask(a=1, b=4),
            MyTask(a=2, b=3),
            MyTask(a=2, b=4),
        ]
        """
        self.cls = cls
        for k, v in kwargs.items():
            assert isinstance(v, list)
            assert k not in others
        self.kwargs = kwargs
        self.others = others

    def dependencies(self):
        return [self.cls(**d, **self.others) for d in product_dict(**self.kwargs)]


def _ensure_ref_count_initialized(task, deps: list):
    """
    Ensure that a task's immediate dependencies have reference counts initialized.
    This is called lazily when a task is about to be processed.
    Only initializes the immediate dependencies, not their sub-dependencies.

    `deps` is the result of `task.dependencies()`; it is passed explicitly so that
    reference counts are set on the same objects that are later cleaned up.
    """
    if not hasattr(task, '_ref_count_initialized'):
        # Mark this task as having its dependencies processed
        task._ref_count_initialized = True

        for dep in deps:
            # Initialize ref count for dependency if not already done
            if not hasattr(dep, '_ref_count'):
                dep._ref_count = 0
            # Count this task as a dependent of the dependency
            dep._ref_count += 1


def _try_cleanup_dependency(dep):
    """
    Try to cleanup a dependency. Only cleanup if all tasks that depend on it have finished.
    """
    if hasattr(dep, '_ref_count'):
        dep._ref_count -= 1
        if dep._ref_count <= 0:
            # All dependent tasks have finished, safe to cleanup
            if hasattr(dep, 'cleanup') and callable(getattr(dep, 'cleanup')):
                dep.cleanup()


def sample():
    """
    Generate a sample Product for demonstration purposes
    """
    class Level1(Task):
        def __init__(self, id):
            self.id = id

        def run(self):
            log.info(f"Downloading level1 {self.id}")
            sleep(1)
        
    class Level2(Task):
        def __init__(self, id, **kwargs):
            self.id = id
            self.kwargs = kwargs

        def dependencies(self) -> list:
            # each Level2 depends on a single Level1
            return [Level1(self.id, **self.kwargs)]

        def run(self):
            log.info(f"Generating level2 {self.id} on {socket.gethostname()}")
            sleep(1)

    class Composite(Task):
        def __init__(self, id, **kwargs):
            self.id = id
            self.kwargs = kwargs

        def dependencies(self) -> list:
            # list required level 2 products for current composite
            return [Level2(i, **self.kwargs) for i in range(5)]

        def run(self):
            log.info(
                f"Generating monthly composite {self.id} from "
                f"{len(self.dependencies())} dependencies."
            )
            sleep(1)

    return Composite("202410")


if __name__ == "__main__":
    """
    python -m hygeos_core.process.deptree sync
    python -m hygeos_core.process.deptree executor
    python -m hygeos_core.process.deptree prefect
    """

    composite = sample()

    if argv[1] == 'sync':
        gen_sync(composite)
    elif argv[1] == 'executor':
        gen_executor(
            composite,
            executors={"Level1": ThreadPoolExecutor(2)},
            verbose=True,
            # default_executor=get_htcondor_client(),
        )
    else:
        from hygeos_core.process.deptree_prefect import gen_prefect
        if argv[1] == 'prefect':
            task_runner = None
        elif argv[1] == 'prefect_dask':
            from prefect_dask import DaskTaskRunner
            task_runner = DaskTaskRunner
        elif argv[1] == 'prefect_condor':
            from hygeos_core.process.condor import get_htcondor_runner
            task_runner=get_htcondor_runner()
        else:
            raise ValueError(f"Invalid mode {argv[1]}")

        gen_prefect(
            composite,
            task_runner=task_runner,
            # concurrency_limit={"Level1": 2},
        )
