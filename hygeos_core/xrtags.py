#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Union

import xarray as xr

"""
Tag variable in xarray objects in view of filtering them.

Example: create variables in a dataset, with tags "level2", "ancillary", "intermediate".
The final product can be written by filtering the Dataset for only keeping the
"level2" tagged variables.
"""


def _get_tags(da: xr.DataArray) -> List[str]:
    """
    Return the tags of `da` as a list

    A single tag may be stored as a plain str (e.g. after a netCDF round-trip)
    """
    tags = da.attrs.get("tags", [])
    if isinstance(tags, str):
        return [tags]
    return list(tags)


def tag_add(da: xr.DataArray, tag: Union[List[str], str]):
    """
    Add one or several tags to DataArray A

    tag: either a single tag (ex: "level2")
         or multiple tags (ex: ["level2", "ancillary"])
    """
    existing_tags = _get_tags(da)

    if isinstance(tag, str):
        new_tags = [tag]
    elif isinstance(tag, list):
        new_tags = tag
    else:
        raise ValueError

    updated_tags = list(set(existing_tags + new_tags))
    da.attrs.update({"tags": updated_tags})


def tag_filter(ds: xr.Dataset, tag: Union[List[str], str]):
    """
    Returns the filtered Dataset, containing variables tagged with any of the
    tag(s) provided
    """
    if isinstance(tag, str):
        tags = [tag]
    elif isinstance(tag, list):
        tags = tag
    else:
        raise ValueError

    def match(da: xr.DataArray) -> bool:
        da_tags = _get_tags(da)
        for t in tags:
            if t in da_tags:
                return True

        return False

    return ds[[x for x in ds if match(ds[x])]]
