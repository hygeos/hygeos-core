import subprocess
from pathlib import Path
from typing import Any, Callable, Literal

from hygeos_core.files.fileutils import filegen
from hygeos_core import log
import urllib.request
import urllib.error


def download_url(
    url: str,
    dirname: Path,
    wget_opts: str | None = None,
    check_function: Callable | None = None,
    verbose: bool = True,
    if_exists: Literal['skip', 'overwrite', 'backup', 'error'] = 'skip',
    **kwargs: Any
) -> Path:
    """
    Download `url` to `dirname`

    Uses wget when `wget_opts` is provided, otherwise uses urllib.request
    Options `wget_opts` are added to wget command
    Uses a `filegen` wrapper
    Other kwargs are passed to `filegen` (lock_timeout, tmpdir, if_exists)

    Returns the path to the downloaded file
    """
    target = Path(dirname)/(Path(url).name)
    @filegen(if_exists=if_exists, verbose=verbose, **kwargs)
    def download_target(path):
        if verbose:
            log.info('Downloading:', url)
            log.info('To: ', target)

        if wget_opts is not None:
            # Use wget when options are provided
            cmd = f'wget {wget_opts} {url} -O {path}'
            ret = subprocess.call(cmd.split())
            if ret:
                raise FileNotFoundError(cmd)
        else:
            # Use urllib.request when no wget options
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=60) as response:
                    with open(path, 'wb') as f:
                        while chunk := response.read(65536):
                            f.write(chunk)
            except urllib.error.URLError as e:
                raise FileNotFoundError(f"Failed to download {url}: {e}")

        if check_function is not None:
            check_function(path)

    download_target(target)

    return target

