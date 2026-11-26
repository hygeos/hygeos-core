import xarray as xr

def gen_bitmask(*masks, bit_index: list = None) -> xr.DataArray:
    """
    Concatenate several boolean masks into a single bitmask

    Args:
        bit_index (list, optional): Index of the bit to fill. If not specified, it completes the bits in order. Defaults to None.
    """
    
    # Initialise and check bit_index list
    if bit_index is None: 
        bit_index = range(len(masks))
    else: 
        assert len(bit_index) == len(masks), 'length of masks and bit index lists are different'
        
    # Check masks format
    assert all(isinstance(m, xr.DataArray) for m in masks), 'Masks should be of type xarray.DataArray'
    names = [m.name for m in masks]
    assert all(n is not None for n in names), f'Mask should be named, got {names}'
        
    # Generate bitmask
    size = masks[0].shape
    bitmask = 2**bit_index[0]*masks[0]
    for mask, bit in zip(masks[1:], bit_index[1:]):
        assert mask.shape == size, 'Masks have different shape'
        bitmask += 2**bit*mask
    
    bitmask.attrs = {'index': list(bit_index), 'mask_name': names}
    return bitmask

def explicit_bitmask(bitmask: xr.DataArray) -> xr.Dataset:
    """
    Retrieve masks used to generate the provided bitmask

    Args:
        bit_mask (xr.DataArray): Bitmask obtained using gen_bitmask
    """
    ds = xr.Dataset()
    for bit, name in zip(bitmask.index, bitmask.mask_name):
        ds[name] = (bitmask >> bit) & 1   
    return ds.astype(bool)