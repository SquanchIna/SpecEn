######################
### Calculates spectral Energies
###########################


#################################################################################################################
############################# Calculation spectral scale
#################################################################################################################


######Temporal Transform
WT_T = wavelet_En_T(event,coord = coordX)

WT_T = WT_T.reshape((n_temp,event.shape[1], event.shape[2]))  ##Rescale to x,y grid
WT_T = WT_T[:,pad_lat:(WT_T.shape[1]-pad_lat),pad_lon:(WT_T.shape[2]-pad_lon)]

####### Spatial Transform
WT_S = wavelet_En_S(event, coord=coordX)
## recut region which has previously been added
WT_S = WT_S[:, pad_time:(WT_S.shape[1]-pad_time)]

####### Spatio-Temporal Transform
WT = wavelet_En_ST(event, coord=coordX)


output = xr.Dataset(
    data_vars = dict(
        WT_T= (["resT", "y","x"], WT_T ),
        WT_S= (["resS", "time"], WT_S),
        WT  = (["resT", "resS"], WT),
        Data= (["time", "y","x"], weight_area.values),
    ),
    coords = dict(
        x = ("x", weight_area[coordX].values),
        y = ("y", weight_area[coordY].values),
        time = ("time", weight_area.time.values),
        resT = ("resT", T),
        resS = ("resS", S),
    )
)

output.to_netcdf(path=path_save + "SpecEns.nc")
