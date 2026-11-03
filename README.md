#SpecEn – Spatiotemporal Spectral Energy Analysis
> Svenja Szemkus (University of Bonn) 
> Version from 2026-11-03

SpecEn is a Python-based tool for the analysis of spatiotemporal structures in gridded datasets using the Dual-Tree Complex Wavelet Transform (DT-CWT).
The method decomposes a three-dimensional signal into its spatial and temporal scales, allowing for a separate investigation of the characteristic spatial and temporal structures of an event.

The implementation combines a one-dimensional wavelet transform along the temporal dimension with a two-dimensional wavelet transform along the spatial dimensions. This hybrid approach enables an independent analysis of spatial and temporal scales, which is difficult to achieve using a conventional three-dimensional wavelet transform.

The method is particularly suited for analysing precipitation fields and identifying characteristic properties of heavy precipitation events.

## Perform Computation 

1. Create a directory where the output should be saved

2. In this directory: Create a file named 'params.json' in which the input parameters are specified. (Example: see config)

3. Update & run either the main.py or run_analysis.ipynb file with the folder created in (1).

## Input Parameters
- `outputdir`: Directory where the output should be saved.
- `varname`: Name of the variable within the xarray.
- `files`: Input Data. Should be an Xarray with lon-lat coordinates.
- `cent_lon`: Center Longitude which will be used to select a specific region from the dataset. 
- `cent_lat`: Center Latitude which will be used to select a specific region from the dataset. 
- `cent_time`: Center Time which will be used to select a specific time-span from the dataset.
- `anzLon`: Number of gridpoints in longitude direction which is used. Must be same as `anz_lat` 
- `anzLat`: Number of gridpoints in latitude direction which is used. Must be same as `anz_lon` 
- `anzT`: Number of timesteps which is used.


## References

[sszemkus@uni-bonn.de](mailto:sszemkus@uni-bonn.de)

Szemkus, S. and Buschow, S. and Friederichs, P.: Revealing the structure of heavy precipitation events: a spatio-temporal wavelet approach., Natural Hazards and Earth System Sciences, 26.8, 3863-3879, https://doi.org/10.5194/nhess-26-3863-2026, 2026.


