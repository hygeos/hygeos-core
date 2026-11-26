import json
import pickle
import pandas as pd
import xarray as xr

from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Literal, Optional
from pandas.testing import assert_frame_equal

from hygeos_core import log
from hygeos_core.files.save import to_netcdf
from hygeos_core.files.fileutils import filegen, safe_move
from matplotlib.figure import Figure
from numpy import array
from PIL import Image

import matplotlib.pyplot as plt
import hashlib
import re


def cachefunc(
    cache_file: Path | str,
    reader: Callable,
    writer: Callable,
    check_in: Optional[Callable] = None,
    check_out: Optional[Callable] = None,
    fg_kwargs=None,
):
    """
    A decorator that caches the return of a function in a file, with
    customizable format

    reader: a function that reads inputs/output from the cache file
        reader(filename) -> {'output': ..., 'input': ...}

    writer: a function that writes the inputs/output to the cache file
        writer(filename, output, input_args, input_kwargs)

    check_in: a custom function to test the equality of the inputs
        checker(obj1, obj2) -> bool
        (defaults to None -> no checking)

    check_out: a custom function to test the equality of the outputs
        checker(obj1, obj2) -> bool
        (defaults to ==)

    fg_kwargs: kwargs passed to filegen (ex: lock_timeout=-1)
    """
    # default input/output checkers
    check_out = check_out or (lambda x, y: x == y)

    cache_file = Path(cache_file)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if not cache_file.exists():
                result = f(*args, **kwargs)

                with TemporaryDirectory(
                    dir=cache_file.parent, prefix="cachefunc_"
                ) as tmpdir:
                    cache_file_tmp = Path(tmpdir) / cache_file.name

                    # write the temporary cache file
                    filegen(**(fg_kwargs or {}))(writer)(
                        cache_file_tmp, result, args, kwargs
                    )

                    # check that the output read back is identical
                    # to the original output (defaults to ==)
                    chk = reader(cache_file_tmp)
                    assert "output" in chk

                    assert check_out(result, chk["output"])

                    # check successful: move the file
                    safe_move(cache_file_tmp, cache_file)
                    assert cache_file.exists()

                    return chk["output"]
            else:
                content = reader(cache_file)

                # check function inputs
                if check_in:
                    with TemporaryDirectory() as tmpdir:
                        # write and read back the inputs for checking
                        tmpchk = Path(tmpdir) / "tmp_check"
                        writer(tmpchk, None, args, kwargs)
                        chk = reader(tmpchk)
                        if not check_in(content["input"], chk["input"]):
                            raise ValueError(
                                f"Input parameters do not match stored inputs.\n"
                                f"  Input params:  {content['input']}\n"
                                f"  Stored inputs: {chk['input']}"
                            )

                return content["output"]

        return wrapper

    return decorator


def cache_dataframe(
    cache_file: Path | str, inputs: Literal["check", "store", "ignore"] = "check"
):
    """
    A decorator that caches the result of a function, which is a pandas DataFrame

    inputs:
        "check" [default]: store and check the function inputs
        "store": store but don't check the function inputs
        "ignore": ignore the function inputs
    """
    extension = Path(cache_file).suffix
    if extension == ".csv":
        return cachefunc(
            cache_file,
            writer=lambda filename, df, args, kwargs: df.to_csv(filename, index=False),
            reader=lambda filename: {
                "output": pd.read_csv(filename, parse_dates=["time"])
            },
            check_out=lambda x, y: assert_frame_equal,
        )
    elif extension == ".pickle":
        return cache_pickle(cache_file, inputs, check_out=(lambda x, y: x.equals(y)))
    else:
        raise ValueError(f"Invalid cache file extension, got ({extension})")


def cache_json(
    cache_file: Path | str,
    inputs: Literal["check", "store", "ignore"] = "check",
):
    """
    A decorator that caches the result of a function to a json file.

    inputs:
        "check" [default]: store and check the function inputs
        "store": store but don't check the function inputs
        "ignore": ignore the function inputs
    """

    # assert isinstance(cache_file, str) or isinstance(cache_file, Path)

    def reader(filename):
        with open(filename) as fp:
            return json.load(fp)

    def writer(filename, output, input_args, input_kwargs):
        with open(filename, "w") as fp:
            content = {}
            if inputs in ["store", "check"]:
                content["input"] = {"args": input_args, "kwargs": input_kwargs}
            content["output"] = output
            json.dump(content, fp, indent=4, default=str)

    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
        check_in=(lambda x, y: x == y) if (inputs == "check") else None,
    )


def cache_pickle(
    cache_file: Path | str,
    inputs: Literal["check", "store", "ignore"] = "check",
    check_out=(lambda x, y: x == y),
):
    """
    A decorator that caches the result of a function to a pickle file.

    inputs:
        "check" [default]: store and check the function inputs
        "store": store but don't check the function inputs
        "ignore": ignore the function inputs
    """

    def reader(filename):
        with open(filename, "rb") as fp:
            return pickle.load(fp)

    def writer(filename, output, input_args, input_kwargs):
        with open(filename, "wb") as fp:
            content = {}
            content["input"] = {"args": input_args, "kwargs": input_kwargs}
            content["output"] = output
            pickle.dump(content, fp)

    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
        check_in=(lambda x, y: x == y) if (inputs == "check") else None,
        check_out=check_out,
    )


def cache_dataset(cache_file: Path | str, attrs=None, **kwargs):
    """
    A decorator that caches the dataset returned by a function in a netcdf file

    The attribute dictionary `attrs` is stored in the file, and verified upon
    reading.

    Other kwargs (ex: chunks) are passed to xr.open_dataset
    """

    def reader(filename):
        ds = xr.open_dataset(filename, **kwargs)

        # check attributes in loaded file
        if attrs is not None:
            for k, v in attrs.items():
                assert ds.attrs[k] == v, (
                    f"Error when checking attribute {k}: {ds.attrs[k]} != {v}"
                )
        return {"output": ds}

    def writer(filename, ds_out, input_args, input_kwargs):
        if attrs is not None:
            ds_out.attrs.update(attrs)
        to_netcdf(ds_out, filename=filename)

    return cachefunc(
        cache_file,
        reader=reader,
        writer=writer,
    )


def cache_figure(
    cache_dir: Path | str, overwrite: bool = False, verbose: bool = False, **kwargs
):
    """
    A decorator that caches the figure returned by a function in a png file

    The name of the png file is deduced by the decorator and is the concatenation
    of the function name and the hash of its params (e.g. `function_a8f6d`)

    Other kwargs are passed to fig.savefig()

    Returns the path to the cached png file.
    """
    cache_dir = Path(cache_dir)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **fkwargs):
            # Determine cache file path
            if len(args) == 0:
                cache_file = cache_dir / f"{f.__name__}.png"
            else:
                h = hashparams(*args, **fkwargs)
                cache_file = cache_dir / f"{f.__name__}_{h}.png"

            # If exists, load image and turn it into plt.Figure
            if not overwrite and cache_file.exists():
                img = array(Image.open(cache_file))
                ratio = img.shape[0] / img.shape[1]  # rows over columns
                fig = plt.figure(figsize=(15, int(ratio * 15)))
                plt.imshow(img)
                plt.axis("off")
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                return fig

            # Apply function and save result into png file
            cache_dir.mkdir(parents=True, exist_ok=True)
            fig = f(*args, **fkwargs)
            assert isinstance(fig, Figure), "Function should return a plt.Figure"
            params = dict(if_exists="overwrite", verbose=verbose)
            filegen(**params)(lambda p: fig.savefig(p, bbox_inches="tight", **kwargs))(
                cache_file
            )
            return fig

        return wrapper

    return decorator


# pre-compile the regex pattern
_MEMORY_ADDRESS_PATTERN = re.compile(r"<.*?\bat\b\s+0x[0-9a-fA-F]+>")


def hashparams(*args, hash_digest_size=16, **kwargs) -> str:
    """
    Generates a deterministic hash based on arguments.
    Simplifies caching based on inputs

    Args:
        *args: Positional arguments (hashed with index-based keys).
        **kwargs: Keyword arguments.

    Returns:
        str: 16-character hexadecimal hash. (or specified digest size)
    """
    params_str_parts = []
    memory_footprint_detected = []

    # Hash positional args with index-based keys
    for i, v in enumerate(args):
        value_repr = repr(v)
        key = f"arg{i}"
        params_str_parts.append(f"{key}:{value_repr}")

        if _MEMORY_ADDRESS_PATTERN.search(value_repr):
            memory_footprint_detected.append(f"- {key} repr: {value_repr}")

    # Hash keyword args with their names
    for k, v in sorted(kwargs.items()):
        value_repr = repr(v)
        params_str_parts.append(f"{k}:{value_repr}")

        if _MEMORY_ADDRESS_PATTERN.search(value_repr):
            memory_footprint_detected.append(f"- {k} repr: {value_repr}")

    # Error messaging for cache misses caused by memory addresses
    if memory_footprint_detected:
        offenders_list = "\n".join(memory_footprint_detected)
        log.error(
            f"Volatile information detected in cache parameters:\n"
            f"{offenders_list}\n"
            "This will cause cache misses because memory addresses change every run."
        )

    params_str = "_".join(params_str_parts)
    hash = hashlib.blake2b(
        params_str.encode(), digest_size=hash_digest_size
    ).hexdigest()

    return hash
