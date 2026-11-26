#!/usr/bin/env python3

from hygeos_core.tools import Var


class names:
    
    # Dimensions
    rows        = Var("y", attrs={'desc': "Vertical dimension of rasters", 'units': "pixels"})
    columns     = Var("x", attrs={'desc': "Horizontal dimension of rasters", 'units': "pixels"})
    lon         = Var("longitude", attrs={'desc': "Horizontal earth coordinate axis", 'units': "Degrees", 'minv': -180, 'maxv': 180}, dtype='float32')
    lat         = Var("latitude", attrs={'desc': "Vertical earth coordinate axis", 'units': "Degrees", 'minv': -90, 'maxv': 90}, dtype='float32')

    bands       = Var("bands", attrs={'desc': "spectral dimension of the acquisition"})
    bands_nvis  = Var("bands_nvis", attrs={'desc': "Visible spectral dimension of the acquisition"})
    bands_ir    = Var("bands_ir", attrs={'desc': "Infrared spectral dimension of the acquisition"})
    bnames      = Var("bandnames", attrs={'desc': "Name of bands"})
    bgroup      = Var("bands_group", attrs={'desc': "List of group of bands"})
    detector    = Var("detectors", attrs={'desc': "Index of detector"})
    
    # Radiometry 
    rtoa        = Var("Rtoa", attrs={'desc': "Top of Atmosphere reflectance"}, dtype='float32')
    ltoa        = Var("Ltoa", attrs={'desc': "Top of Atmosphere radiance", 'units': "W.m-2.sr-1"})            
    bt          = Var("BT", attrs={'desc': "Brightness Temperature", 'units': "K", 'minv': 0})
    rho_w       = Var("rho_w", attrs={'desc': "Water Reflectance"})

    wav         = Var("wav", attrs={'desc': "Effective wavelength", 'units': "nm"}) 
    cwav        = Var("cwav", attrs={'desc': "Central (nominal) wavelength", 'units': "nm"})

    F0          = Var("F0", attrs={'desc': "Solar flux irradiance", 'units': "W.m-2.sr-1"})
    
    # Angles 
    sza = Var("sza", attrs={'desc': "Sun zenith angle", 'units': "Degrees", 'minv': 0, 'maxv': 90}, dtype='float32')
    vza = Var("vza", attrs={'desc': "View zenith angle", 'units': "Degrees", 'minv': 0, 'maxv': 90}, dtype='float32')
    saa = Var("saa", attrs={'desc': "Sun azimuth angle", 'units': "Degrees", 'minv': 0, 'maxv': 180}, dtype='float32')
    vaa = Var("vaa", attrs={'desc': "View azimuth angle", 'units': "Degrees", 'minv': 0, 'maxv': 180}, dtype='float32')
    raa = Var("raa", attrs={'desc': "Relative azimuth angle", 'units': "Degrees", 'minv': -180, 'maxv': 180}, dtype='float32')
    mus = Var("mus", attrs={'desc': "Cosine of the sun zenith angle"}, dtype='float32')
    muv = Var("muv", attrs={'desc': "Cosine of the view zenith angle"}, dtype='float32')
    
    # Flags
    flags   = Var("flags", attrs={'desc': "Bitmask describing the pixel data"}, dtype='uint16')
    quality = Var("quality", attrs={'desc': "Boolean mask describing the pixel quality"}, dtype='bool')
    
    # Attributes
    crs             = Var("crs", attrs={'desc': "Projection"})
    unit            = Var("unit")
    datetime        = Var("datetime")
    platform        = Var("platform")
    sensor          = Var("sensor")
    shortname       = Var("shortname")
    resolution      = Var("resolution", attrs={'desc': "Resolution in meter"})
    description     = Var("description")
    product_name    = Var("product_name")
    input_directory = Var("input_directory")

