#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Block processing framework for xarray datasets with lazy evaluation support.

This module provides a framework for processing large xarray datasets in chunks
using lazy evaluation, leveraging xarray's map_blocks functionality and dask's chunked
arrays.

Key components:
- BlockProcessor: Abstract base class for defining individual processing operations
  that modify or create dataset variables within data blocks.
- CompoundProcessor: Class for chaining multiple processors into optimized pipelines
  with flexible output selection

The framework supports:
- Variable metadata management (dimensions, dtypes, attributes, flags)
- Inputs and outputs description, with automatic validation
- Performance optimization through processor chaining

Example
-------
Create a simple processor that adds two variables:

>>> class AddProcessor(BlockProcessor):
...     def input_vars(self):
...         return [Var('a'), Var('b')]
...
...     def created_vars(self):
...         return [Var('sum', 'float64', ('x', 'y'))]
...
...     def process_block(self, block):
...         block['sum'] = block['a'] + block['b']
...
>>> result = AddProcessor().map_blocks(dataset)

Chain multiple processors:

>>> compound = CompoundProcessor([processor1, processor2])
>>> result = compound.map_blocks(dataset)
"""

from abc import ABC, abstractmethod
from functools import reduce
from typing import Any, Literal
import warnings

import pandas as pd
import xarray as xr
from core.ascii_table import ascii_table
from core.tools import Var, raiseflag, is_dask_backed
from xarray.core.parallel import make_meta


class BlockProcessor(ABC):
    """
    Abstract base class for block processing operations.  Processors are intended to be
    executed by their method `map_blocks`. Use `CompoundProcessor` to combine multiple
    processors into a single processing pipeline and minimize dask graph management
    overhead.

    Each processor defines:
    - process_block: Method that performs the actual processing on a data block, by
      adding or modifying Dataset variables in place.
    - input_vars: List of variable names required as input (optional)
    - modified_vars: List of Var objects describing the variables to be modified (optional)
    - created_vars: List of Var objects describing newly created variables (optional)
    - output_vars: List of Var objects describing the variables to be output by the
        map_blocks method (optional)
    - created_dims: Dictionary of new dimensions to be created by this processor (optional)
    - global_attrs: Dictionary of the global attributes set by this processor (optional)
    - auto_template: Method that returns whether automatic templating should be activated (optional)
    - check: Method to validate the input dataset before processing (optional)

    Examples
    --------
    >>> class AddProcessor(BlockProcessor):
    ...     def input_vars(self):
    ...         return [Var('a'), Var('b')]
    ...
    ...     def created_vars(self):
    ...         return [Var('sum', 'float64', ('x', 'y'))]
    ...
    ...     def process_block(self, block, **kwargs):
    ...         block['sum'] = block['a'] + block['b']
    ...
    >>> result = AddProcessor().map_blocks(ds)
    """
    _preserve_attrs: bool = True

    def input_vars(self) -> list[Var]:
        """
        Input variables required by this processor.

        Example:
        >>> def input_vars(self):
        ...     return [Var('a'), Var('b'), Var('flags')]
        """
        return []

    def modified_vars(self) -> list[Var]:
        """
        Variables to be modified in-place.

        Use `Var` with only `name` (leave `dims`/`dtype` as None). Optionally
        provide other attributes or `flags` to register flag metadata
        used by `raiseflag()`.

        Example:
        >>> def modified_vars(self):
        ...     return [Var('flags', flags={'CLOUD': 1})]

        Note: flags are raised like so:
        >>> def process_block(self, block, **kwargs):
        ...     self.raiseflag(block, 'flags', 'CLOUD', block['rho_toa'] > 0.2)
        """
        return []

    def created_vars(self) -> list[Var]:
        """
        Variables that this processor will create.

        Provide Var definition: `name`, `dtype`, `dims` or `dims_like` (plus optionally
        `flags` or other attributes). Declare any new dims in `created_dims()`.
        If `auto_template()` returns `True` and `dtype` or `dims`/`dims_like`
        are not specified, they will be assessed by running `process_block` on mockup data.

        Use `dims_like` to specify that the dimensions should match those of an existing
        variable in the dataset (e.g., `dims_like='input_var'`).

        Example:
        >>> def created_vars(self):
        ...     return [Var('sum', 'float64', ('x', 'y'))]
        """
        return []

    def output_vars(self) -> list[Var]:
        """
        Defines the list of variables to be returned by `map_blocks`

        By default, the created and modified variables are included as output_vars.

        This method can be overridden to adjust this behaviour.
        """
        return self.created_vars() + self.modified_vars()

    def created_dims(self) -> dict[str, Any]:
        """
        Describes new dimensions to be created by this processor.

        Each dimension is defined either by:
        - An int: specifying the dimension size
        - An array-like: providing coordinates for the dimension

        Returns
        -------
        Dict[str, int or array-like]
            Dictionary mapping dimension names to their size (int) or coordinates (array-like)

        Examples
        --------
        >>> def created_dims(self):
        ...     return {'z': 10, 'wavelength': [400, 500, 600, 700]}
        """
        return {}

    def global_attrs(self) -> dict[str, Any]:
        """
        Return dictionary of global attributes to be set by this processor.
        For variable attributes, use the `attrs` argument of the created/modified variables.
        """
        return {}

    def raiseflag(
        self, block: xr.Dataset, var_name: str, flag_name: str, condition: xr.DataArray
    ):
        """
        Apply a flag to a variable in the block where condition is True.

        This is a convenience wrapper around core.tools.raiseflag that automatically
        looks up the flag bit position from the processor's flag definitions.

        Note that the flags must be registered in the Var definition for the variable
        in modified_vars() or created_vars().

        Parameters
        ----------
        block : xr.Dataset
            Dataset block containing the flag variable
        var_name : str
            Name of the flag variable (e.g., 'flags', 'quality_flags')
        flag_name : str
            Name of the flag to raise (e.g., 'CLOUDY', 'BAD_DATA')
        condition : xr.DataArray
            Boolean array indicating where to raise the flag

        Raises
        ------
        ValueError
            If the variable has no flags defined or the flag name is not defined

        Examples
        --------
        >>> def modified_vars(self):
        ...     return [Var('flags', flags={'CLOUDY': 1})]
        >>> def process_block(self, block, **kwargs):
        ...     cloud_mask = block['reflectance'] > 0.8
        ...     self.raiseflag(block, 'flags', 'CLOUDY', cloud_mask)
        """

        # Get flag mappings (cached after first call)
        flag_mappings = self.get_flag_mappings()

        if var_name not in flag_mappings:
            raise ValueError(
                f"No flags defined for variable '{var_name}'. "
                f"Define flags in modified_vars() or created_vars()."
            )
        if flag_name not in flag_mappings[var_name]:
            available = list(flag_mappings[var_name].keys())
            raise ValueError(
                f"Flag '{flag_name}' not defined for variable '{var_name}'. "
                f"Available flags: {available}"
            )

        flag_value = flag_mappings[var_name][flag_name]

        # check that condition has the same dimensions as the flags variable
        if condition.dims != block[var_name].dims:
            raise ValueError(
                f"Dimensions of condition {condition.dims} do not match "
                f"dimensions of variable '{var_name}' {block[var_name].dims}"
            )

        raiseflag(block[var_name], flag_name, flag_value, condition)

    def get_flag_mappings(self) -> dict[str, dict[str, int]]:
        """Extract flag bit mappings (var -> flags) from modified_vars() and
        created_vars(). Results are cached."""
        if not hasattr(self, "_flag_mappings"):
            flag_mappings = {}
            for var in self.modified_vars() + self.created_vars():
                if hasattr(var, "flags") and var.flags:
                    flag_mappings[var] = var.flags
            self._flag_mappings = flag_mappings
        return self._flag_mappings
    
    @abstractmethod
    def process_block(self, block: xr.Dataset) -> None:
        """
        Process a single block of data (in place).

        Parameters
        ----------
        block : xr.Dataset
            Input data block containing the required input variables. This block is
            modified in place.
        """
        pass

    def describe(self) -> None:
        """Print a description of this processor for logging/debugging."""
        print(f"Processor: {self.__class__.__name__}")
        
        # Collect all variables with their roles
        all_vars = {}
        for v in self.input_vars():
            all_vars[str(v)] = ('I', v)
        for v in self.modified_vars():
            all_vars[str(v)] = ('M', v)
        for v in self.created_vars():
            all_vars[str(v)] = ('C', v)
        
        if not all_vars:
            print("  No variables defined")
            return
        
        # Prepare data for table
        table_data = []
        for name, (role, v) in sorted(all_vars.items()):
            dtype_str = v.dtype if v.dtype else ''
            dims_str = str(v.dims) if v.dims else (v.dims_like if v.dims_like else '')
            table_data.append({
                'Variable': name,
                'dtype': dtype_str,
                'dims': dims_str,
                'role': role
            })
        
        # Create DataFrame and print table
        df = pd.DataFrame(table_data)
        ascii_table(df).print()
        
        # Additional info
        if self.created_dims():
            print()
            print("Created dims:")
            for k, v in self.created_dims().items():
                print(f"  {k}: {v}")
        if self.global_attrs():
            print()
            print("Global attrs:")
            for k, v in self.global_attrs().items():
                print(f"  {k}: {v}")

    def check(self, ds: xr.Dataset) -> None:
        """
        Validate the input dataset for this processor.

        This method can be overridden by subclasses to perform custom validation
        on the input dataset before processing. By default, it does nothing.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to validate.
        """
        pass
        
    def auto_template(self) -> bool:
        """
        Returns whether "automatic templating" should be activated
        
        Automatic templating calls process_block on mocked up (empty) data to determine
        the output variables types and dimensions.
        
        auto_template is disactivated by default.
        """
        return False
    
    def build_template(self, ds_template: xr.Dataset):
        """
        Modifies a template dataset for xr.map_blocks operations.

        This method builds a template dataset that defines the structure and metadata
        of the output dataset after processing. The template includes:
        - All output variables (created and modified)
        - Appropriate data types and dimensions for created variables
        - Flag metadata for variables with defined flags
        - Global attributes from the processor

        If some dimensions are not fully described, the method runs the processing
        code on mockup data to assess the data types and dimensions of created variables.

        Parameters
        ----------
        ds_template : xr.Dataset
            Input template, modified in place.

        Returns
        -------
        None
            The input dataset is modified in place.
        """
        self.check(ds_template)
        
        # Add created vars to the template
        if not self.auto_template():
            for var in self.created_vars():
                ds_template[str(var)] = var.to_template(
                    ds_template, new_dims=self.created_dims()
                )
        else:
            # auto_template mode: apply process_block on mocked up data to assess the
            # dtype and dimensions # of created variables
            meta: xr.Dataset = make_meta(ds_template) # type: ignore
            self.process_block(meta)
            for var in self.created_vars():
                ds_template[str(var)] = var.to_template(
                    ds_template, meta, new_dims=self.created_dims()
                )
        
        # Set all global and variable attributes
        self.set_attributes(ds_template)

    def process_and_validate(self, block: xr.Dataset) -> xr.Dataset:
        """
        Process a block and validate the created output variables.

        This method wraps the call to `process_block`, and is called by xr.map_map_blocks
        """
        # standard block call
        self.process_block(block)
        
        # validate outputs
        self.validate(block)

        # set attributes
        self.set_attributes(block)

        # Return only output variables
        return block[self.output_vars()] 
    
    def validate(self, block: xr.Dataset):
        """
        Validate that all advertised created output variables exist with correct specs
        """
        for var in [x for x in self.output_vars() if x in self.created_vars()]:
            if var not in block.data_vars:
                raise ValueError(
                    f"Processor {self.__class__.__name__} failed to create variable '{var}'"
                )

            actual_var = block[var]
            if (var.dtype is not None) and (actual_var.dtype != var.dtype):
                raise TypeError(
                    f"Processor {self.__class__.__name__} created variable '{var}' "
                    f"with dtype {actual_var.dtype}, expected {var.dtype}"
                )

            var_dims = var.getdims(block)
            if actual_var.dims != var_dims:
                raise ValueError(
                    f"Processor {self.__class__.__name__} created variable '{var}' "
                    f"with dims {actual_var.dims}, expected {var_dims}"
                )

    def set_attributes(self, ds: xr.Dataset):
        """
        Set attributes defined in created/modified vars in `ds`
        
        This method is called:
        - During the template creation
        - After each execution of process_block (in process_and_validate)
        """
        # Set all variable attributes
        for var in self.modified_vars() + self.created_vars():
            ds[var].attrs.update(var.attrs)

        # Register flags for all variables that have been defined
        flag_mappings = self.get_flag_mappings()
        for var, flags_dict in flag_mappings.items():
            var_name = str(var)
            for flag_name, flag_value in flags_dict.items():
                raiseflag(ds[var_name], flag_name, flag_value)

        # Update global attributes
        ds.attrs.update(self.global_attrs())
        
    def map_blocks(
        self,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """
        Apply this processor to a dataset using xarray's map_blocks.

        This method validates that all required input and modified variables are present
        in the dataset, creates a subset containing only the necessary variables, builds
        a template dataset defining the output structure, and applies the processor's
        process_block method to each chunk using xr.map_blocks.

        The output includes only the variables specified by output_vars() (by default,
        created and modified variables).

        Parameters
        ----------
        ds : xr.Dataset
            Input dataset to process. Must contain all variables required by
            input_vars() and modified_vars().

        Returns
        -------
        xr.Dataset
            Processed dataset containing the output variables as defined by output_vars().
        """
        # Warn if the dataset is not dask-backed, as map_blocks is designed for
        # chunked arrays. The user may want to use process_block() directly
        # for better performance.
        if not is_dask_backed(ds):
            warnings.warn(
                f"Dataset passed to {self.__class__.__name__}.map_blocks() is not "
                "dask-backed. Processing would be more efficient by using "
                f"{self.__class__.__name__}.process_block instead.",
                UserWarning,
            )

        # Validate processor inputs (can be coords as well)
        current_vars = set(ds.data_vars) | set(ds.coords)
        missing_inputs = [
            var
            for var in self.input_vars() + self.modified_vars()
            if var not in current_vars
        ]
        if missing_inputs:
            raise KeyError(
                f"Processor {self} requires "
                f"variables {missing_inputs} which are not available. "
                f"Current variables: {list(current_vars)}"
            )

        # Check that there are output variables
        output_vars = self.output_vars()
        if not output_vars:
            return ds[[]]

        # take a subset of the input
        sub = ds[self.input_vars() + self.modified_vars()]

        # create a template
        template = sub.copy()
        if not self._preserve_attrs:
            template.attrs.clear()
        self.build_template(template)
        template = template[self.output_vars()]

        if not template:
            raise ValueError('Missing output variables. Make sure to define at least '
                             f'one output or created variable ({self}).')

        # Ensure coordinates from the template are added to the subset
        # This is necessary because xr.map_blocks doesn't preserve coordinates
        # from the template when they're not present in the input dataset
        for coord_name in template.coords:
            if coord_name not in sub.coords:
                sub = sub.assign_coords({coord_name: template.coords[coord_name]})

        # Include processor identity in the input dataset attrs so that
        # xr.map_blocks generates a unique dask graph token for each processor.
        # Without this, different processor instances receiving the same input
        # dataset produce identical tokens (because xr.map_blocks tokenizes
        # only the input data, args, and kwargs — not the function/method itself).
        sub.attrs['_processor_token'] = id(self)

        # Call "map_blocks"
        result = xr.map_blocks(self.process_and_validate, sub, template=template)

        return result


class CompoundProcessor(BlockProcessor):
    """
    Combines multiple BlockProcessor instances into a single processor.

    This class allows chaining multiple processors together, automatically filtering
    out processors that are not required to produce the specified outputs. It provides
    flexible output selection based on variable roles (created/modified/input) or
    custom criteria like tags or specific names.

    The compound processor optimizes execution by only running processors that
    contribute to the final output, reducing unnecessary computation in processing
    pipelines.

    Parameters
    ----------
    list_processors : list[BlockProcessor]
        List of processors to combine in execution order
    outputs : Literal["created_modified", "all", "tags", "named"], default "created_modified"
        How to select output variables:
        - "created_modified": output variables that are created or modified by the processors
        - "all": output all variables (created, modified, and input variables)
        - "tags": output variables that have tags matching outputs_tags
        - "named": output variables with names matching outputs_names
    outputs_tags : list[str] | None, default None
        Required when outputs="tags". List of tags that variables must have to be included
        in the output. Variables are included if they have at least one matching tag.
    outputs_names : list[str] | None, default None
        Required when outputs="named". List of variable names to include in the output.
        All specified names must exist in the processor chain.
    preserve_attrs : bool, default True
        Whether to preserve global attributes from the input dataset in the output.

    Examples
    --------
    Combine two processors with default output (created and modified variables):

    >>> p1 = CloudMaskProcessor()
    >>> p2 = QualityFlagProcessor()
    >>> compound = CompoundProcessor([p1, p2])
    >>> result = compound.map_blocks(dataset)

    Output only variables with specific tags:

    >>> compound = CompoundProcessor([p1, p2],
    ...                              outputs="tags",
    ...                              outputs_tags=["quality"])
    >>> result = compound.map_blocks(dataset)

    Output specific named variables:

    >>> compound = CompoundProcessor([p1, p2], outputs="named", 
    ...                              outputs_names=["cloud_mask", "quality_flags"])
    >>> result = compound.map_blocks(dataset)
    """
    def __init__(
        self,
        list_processors: list[BlockProcessor],
        outputs: Literal[
            "created_modified", "all", "tags", "named"
        ] = "created_modified",
        outputs_tags: list[str] | None = None,
        outputs_names: list[str] | None = None,
        preserve_attrs: bool = True,
    ):
        self.list_processors = list_processors
        self.outputs = outputs
        self.output_tags = outputs_tags
        self.output_names = outputs_names
        self._preserve_attrs = preserve_attrs
        if outputs == 'named':
            assert outputs_names is not None
        if outputs == 'tags':
            assert outputs_tags is not None

    def get_required_processors(self) -> list[BlockProcessor]:
        """
        Filter the processor chain to keep only those required to produce allowed_vars.
        
        A processor is kept if:
        - It creates any variables in allowed_vars, OR
        - It modifies any variables in allowed_vars, OR
        - It creates variables needed by a later kept processor
        
        This avoids running unnecessary processors that don't contribute to the output.
        
        Parameters
        ----------
        processors : list[BlockProcessor]
            Full list of processors
            
        Returns
        -------
        list[BlockProcessor]
            Filtered list containing only required processors
        """
        # Track which variables we need to produce (starts with allowed_vars)
        needed_vars = set(self.output_vars())
        # Track which processors are required
        required_processors = []
        
        # Work backwards through the processor chain
        for processor in reversed(self.list_processors):
            created_vars = set(processor.created_vars())
            modified_vars = set(processor.modified_vars())
            
            # Keep this processor if it creates/modifies any needed variables
            if (created_vars & needed_vars) or (modified_vars & needed_vars):
                required_processors.append(processor)
                # Add this processor's inputs to the set of needed variables
                needed_vars.update(processor.input_vars())
                # Add modified vars to needed (they must exist before modification)
                needed_vars.update(processor.modified_vars())
        
        # Reverse to restore original order
        return list(reversed(required_processors))

    def input_vars(self) -> list[Var]:
        """
        Return the list of input variables for the compound processor.

        The compound input vars are the variables being input in at least one processor,
        but not modified nor created by any processor.

        Returns
        -------
        list[Var]
            List of input variables required by the compound processor
        """
        input_vars = []
        for processor in self.list_processors:
            for var in processor.input_vars():
                if var not in input_vars:
                    input_vars.append(var)
        exclude_vars = []
        for processor in self.list_processors:
            for var in processor.modified_vars() + processor.created_vars():
                if var not in exclude_vars:
                    exclude_vars.append(var)
        result = [var for var in input_vars if var not in exclude_vars]
        return result

    def modified_vars(self) -> list[Var]:
        """
        Return the list of variables modified by the compound processor.

        The compound modified vars are the variables that are modified vars for all
        processors (not created, not input).

        Returns
        -------
        list[Var]
            List of variables modified by the compound processor
        """
        # Collect all modified vars, including their multiple definitions
        modified_vars = {}
        for processor in self.list_processors:
            for var in processor.modified_vars():
                svar = str(var)
                if svar not in modified_vars:
                    modified_vars[svar] = []
                modified_vars[svar].append(var)
        # Remove vars created by any processor, which will be listed in the
        # method `created_vars`
        for processor in self.list_processors:
            for var in processor.created_vars():
                svar = str(var)
                if svar in modified_vars:
                    del modified_vars[svar]
        modified_vars_merged = [
            reduce(lambda x, y: x.merge_with(y), v) for v in modified_vars.values()
        ]
        return modified_vars_merged

    def created_vars(self) -> list[Var]:
        """
        Return the list of variables created by the compound processor.

        The compound created vars is the union of all created vars from sub-processors.

        Returns
        -------
        list[Var]
            List of variables created by the compound processor
        """
        created_vars = {}
        # Collect all created vars, including their multiple definitions
        for processor in self.list_processors:
            for var in processor.created_vars():
                svar = str(var)
                if svar not in created_vars:
                    created_vars[svar] = []
                created_vars[svar].append(var)
            # append modified vars
            for var in processor.modified_vars():
                svar = str(var)
                if svar in created_vars:
                    created_vars[svar].append(var)
        
        # merge all vars in created_vars
        created_vars_merged = [
            reduce(lambda x, y: x.merge_with(y), v) for v in created_vars.values()
        ]

        return created_vars_merged

    def output_vars(self) -> list[Var]:
        if self.outputs == "created_modified":
            return self.created_vars() + self.modified_vars()
        else:
            all_vars = self.created_vars() + self.modified_vars() + self.input_vars()
            if self.outputs == 'all':
                return all_vars
            elif self.outputs == 'named':
                assert self.output_names is not None
                named_outputs = [x for x in all_vars if x in self.output_names]
                assert len(named_outputs) == len(self.output_names), 'Could not find all required outputs in the compound'
                return named_outputs
            elif self.outputs == 'tags':
                # return all variables that have a non-void intersection between their
                # tags and self.outputs_tags
                assert self.output_tags is not None
                filtered_vars = [
                    x
                    for x in all_vars
                    if hasattr(x, "tags")
                    and x.tags
                    and set(x.tags) & set(self.output_tags)
                ]
                if not filtered_vars:
                    # Fail loudly when no Var were selected
                    all_tags = set()
                    for var in all_vars:
                        if hasattr(var, "tags") and var.tags:
                            all_tags.update(var.tags)
                    available_tags_str = ", ".join(sorted(all_tags)) if all_tags else "none"
                    raise ValueError(f"No variable matching tags {self.output_tags} "
                                     "have been selected. Input variables contain the "
                                     f"following tags: {available_tags_str}")
                return filtered_vars
            else:
                raise ValueError(self.outputs)

    def created_dims(self) -> dict[str, Any]:
        created_dims = {}
        for processor in self.list_processors:
            for dim_name, dim_value in processor.created_dims().items():
                if dim_name in created_dims:
                    if created_dims[dim_name] != dim_value:
                        raise ValueError(
                            f"Incompatible dimension '{dim_name}' defined by multiple processors. "
                            f"Existing: {created_dims[dim_name]}, New: {dim_value}"
                        )
                else:
                    created_dims[dim_name] = dim_value
        return created_dims

    def global_attrs(self) -> dict[str, Any]:
        """
        Safe merge of the global attributes from all processors
        """
        merged_attrs = {}
        for processor in self.list_processors:
            for key, value in processor.global_attrs().items():
                if key not in merged_attrs:
                    merged_attrs[key] = value
                else:
                    assert merged_attrs[key] == value
        return merged_attrs

    def describe(self) -> None:
        """
        Print a detailed description of the compound processor.
        """
        processors = self.list_processors
        
        # Compute compound roles
        compound_created = {str(v) for v in self.created_vars()}
        compound_modified = {str(v) for v in self.modified_vars()}
        compound_input = {str(v) for v in self.input_vars()}
        
        # Prepare data for table
        table_data = []
        all_vars = self.input_vars() + self.modified_vars() + self.created_vars()
        for v in all_vars:
            v_name = str(v)
            row = {
                'Variable': v_name,
                'dtype': v.dtype if v.dtype else '',
                'dims': str(v.dims) if v.dims else (str(v.dims_like) if v.dims_like else '')
            }
            
            # Add processor columns
            for p in processors:
                p_name = p.__class__.__name__
                if any(str(x) == v_name for x in p.created_vars()):
                    cell = 'C'
                elif any(str(x) == v_name for x in p.modified_vars()):
                    cell = 'M'
                elif any(str(x) == v_name for x in p.input_vars()):
                    cell = 'I'
                else:
                    cell = ''
                row[p_name] = cell
            
            # Add compound column
            if v_name in compound_created:
                cell = 'C'
            elif v_name in compound_modified:
                cell = 'M'
            elif v_name in compound_input:
                cell = 'I'
            else:
                cell = ''
            row['Compound'] = cell
            
            table_data.append(row)
        
        # Create DataFrame and print table
        df = pd.DataFrame(table_data)
        ascii_table(df).print()

    def process_block(self, block: xr.Dataset) -> None:
        for p in self.get_required_processors():
            p.process_block(block)
    
    def process_and_validate(self, block: xr.Dataset) -> xr.Dataset:
        """
        Process a single block of data by applying all required processors, hereby
        implementing the compound processor.

        This method iterates through the list of required processors obtained from
        `get_required_processors()` and calls their `process_block` method on the
        provided block, modifying it in place.

        Parameters
        ----------
        block : xr.Dataset
            The xarray Dataset representing the block to be processed.
        """
        for p in self.get_required_processors():
            p.process_and_validate(block)
            
        # Return only output variables
        return block[self.output_vars()]

    def build_template(self, ds_template: xr.Dataset):
        """
        Modifies a template dataset for xr.map_blocks operations for compound processors.

        This method builds a template dataset that defines the structure and metadata
        of the output dataset after processing by all processors in the compound.
        The template includes:
        - All output variables (created and modified) from all processors
        - Appropriate data types and dimensions for created variables
        - Flag metadata for variables with defined flags
        - Global attributes from all processors

        Parameters
        ----------
        ds_template : xr.Dataset
            Input dataset, modified in place.

        Returns
        -------
        None
            The input dataset is modified in place.
        """
        for processor in self.get_required_processors():
            processor.build_template(ds_template)

    def map_blocks(self, ds: xr.Dataset, chained: bool = True) -> xr.Dataset:
        """
        Apply processors to dataset blocks with optional chaining.

        Parameters
        ----------
        ds : xr.Dataset
            Input xarray Dataset to process.
        chained : bool, default True
            If True, use processor chaining for efficient computation.
            If False, don't chain processors; this may involve additional dask graph
            overhead.

        Returns
        -------
        xr.Dataset
            Processed dataset containing only the output variables.
        """
        if chained:
            return super().map_blocks(ds)
        else:
            # Naive implementation without processor chaining
            # Involves dask graph overhead.
            for p in self.list_processors:
                result = p.map_blocks(ds)
                for x in result:
                    ds[x] = result[x]
            return ds[self.output_vars()]


