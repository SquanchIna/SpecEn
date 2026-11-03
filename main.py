import sys, os
from SpecEn import SPECS
import json
import numpy as np
import xarray as xr 
def main():
    """
    Workflow for decomposing data into frequency space
    """
    # ds_transf, ds_stacked = tpdm.load_data()
    config_file = "config/params.json"
    # config_file = "config/params.json"
    spec._load_config(config_file)

    out = spec.outdir
    indir = spec.files

    # load data and smooth boundaries
    data = spec.prepare_dataset(indir)
    ## data with zeros
    data_pad = spec.pad_data(data)

    # 1D: Time
    print(f"\n 1D Wavelet Transform...")
    decomp1 = spec.decompose_1D(data_pad)
    # 2D: Space
    print(f"\n 2D Wavelet Transform...")
    decomp2 = spec.decompose_2D(data_pad)
    # 3D: Space-Time
    print(f"\n 3D Wavelet Transform...")
    decomp3 = spec.decompose_3D(data_pad)
    print(f"\n Analysis finished.")

    data_new = spec.saveFile(data,decomp1,decomp2,decomp3)
    print(f"\n File saved.")


    spec.plotEvent(data_new)
    spec.plotDec1(data_new)#    specs.plotDec2(data, out)
    spec.plotDec3(data_new)
if __name__ == "__main__":
    main()
