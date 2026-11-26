from contextlib import contextmanager
from pathlib import Path
from time import sleep


@contextmanager
def LockFile(locked_file: Path,
             ext='.lock',
             interval=1,
             timeout=0,
             create_dir=True,
             ):
    """
    Create a blocking context with a lock file

    timeout: timeout in seconds, waiting to the lock to be released.
        If negative, disable lock files entirely.
    
    interval: interval in seconds

    Example:
        with LockFile('/dir/to/file.txt'):
            # create a file '/dir/to/file.txt.lock' including a filesystem lock
            # the context will enter once the lock is released
    """
    lock_file = Path(str(locked_file)+ext)
    disable = timeout < 0
    if create_dir and not disable:
        lock_file.parent.mkdir(exist_ok=True, parents=True)
    
    if disable:
        yield lock_file
    else:
        # wait untile the lock file does not exist anymore
        i = 0
        while lock_file.exists():
            if i > timeout:
                raise TimeoutError(f'Timeout on Lockfile "{lock_file}"')
            sleep(interval)
            i += 1

        # create the lock file
        with open(lock_file, 'w') as fd:
            fd.write('')

        try:
            yield lock_file
        finally:
            # remove the lock file
            lock_file.unlink()