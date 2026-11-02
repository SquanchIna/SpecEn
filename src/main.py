import sys, os
from SpecEn import SPECS
import json
import numpy as np
import xarray as xr 
def main():
    """
    Workflow for decomposing data into frequency space
    """
    with( open(sys.argv[1]) ) as f:
        params = json.load(f)
    out   = params["outputdir"]
    indir = params["inputdir"]
    print(f"\n Starting analysis...")
    
    specs = SPECS(min_lat = params['min_lat'],
                  max_lat = params['max_lat'],
                  min_lon = params['min_lon'],
                  max_lon = params['max_lon'],
                  CentTime = params['CentTime'],
                  anzT = params['anzT'],
                  varname = params['varname']
                 )
    
    #load data and smooth boundaries 
    data = specs.prepare_dataset(indir)
    #enshure that x and y dimensions are equal
    data = specs.ensure_odd_and_equal(data)
    # data with zeros
    data_pad = specs.pad_data(data)
    
    if 'lon' in data.sizes: 
        coordX = 'lon'
        coordY = 'lat'
    elif 'x' in data.sizes: 
        coordX = 'x'
        coordY = 'y'
    else:
        raise ValueError(
            f'coordinates not implementet yet.'
        ) from None
    
    n_spat = int(np.log2(data_pad.sizes[coordX]))
    n_temp = int(np.log2(data_pad.sizes['time']))
    S = 2**np.arange(0,n_spat)
    T = 2**np.arange(0,n_temp)
    
    # 1D: Time
    print(f"\n 1D Wavelet Transform...")
    decomp1   = specs.decompose_1D(data_pad)
    # 2D: Space
    print(f"\n 2D Wavelet Transform...")
    decomp2   = specs.decompose_2D(data_pad)
    # 3D: Space-Time
    print(f"\n 3D Wavelet Transform...")
    decomp3   = specs.decompose_3D(data_pad)
    #  specs.plot_output(decomp)    
    print(f"\n Analysis finished.")    
    data = specs.saveFile(data, data_pad, decomp1,decomp2,decomp3,S,T,out)
    print(f"\n File saved.")
    #data = xr.open_dataset("/work/bm1159/XCES/xces-work/b381225/evaluation_system/output/specen/20241029_093225/935210SpecEns.nc",engine="netcdf4")
    specs.plotEvent(data, out)
    specs.plotDec1(data, out)
    specs.plotDec2(data, out)
    specs.plotDec3(data, out)
    

if __name__ == "__main__":
    main()
