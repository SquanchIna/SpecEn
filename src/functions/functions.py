import xarray as xr ## data handling
import numpy as np  ## maths
import sys, os
from matplotlib.pylab import *
import matplotlib.animation as animation
import pylab as py 
import sys, os
import matplotlib.animation as animation
import pylab as py 
from matplotlib.colors import ListedColormap
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools  ##nestet loop
from scipy.ndimage import uniform_filter
import dtcwt
import dask  #
import dask.array as da
import cartopy.crs as ccrs
import cartopy  as cp
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors

def wavelet_En_T(var,coord):     
    gp_spat = var.sizes[coord]
    gp_temp = var.sizes['time']
    n_spat = int(np.log2(gp_spat)) 
    n_temp = int(np.log2(gp_temp)) 
                    
    #### First step: 1D wavelet-transform on each gridpoint
    # Initialize DTCWT transform object
    trans = dtcwt.Transform1d()  
    X = var.values.reshape((gp_temp,gp_spat*gp_spat))
        
    test = trans.forward(X, nlevels=n_temp)
    
    scales = np.empty((n_temp, gp_spat**2))
    for scale in (range(0,n_temp)):
        scales[scale,] = np.sum(np.abs(test.highpasses[scale].real)**2 + np.abs(test.highpasses[scale].imag)**2, axis = 0) 
    #scales = scales/(np.sum(scales,axis=0)+0.0000001)
    return scales



##################################### 2) Investigate Spatial Scales:
##############################################################


def wavelet_En_S(var,coord):     
    gp_spat = var.sizes[coord]
    gp_temp = var.sizes['time']
    n_spat = int(np.log2(gp_spat)) 
    n_temp = int(np.log2(gp_temp)) 
                        
    #### First step: 1D wavelet-transform on each gridpoint
    # Initialize DTCWT transform object
    trans = dtcwt.Transform2d()  
    
    scales = np.empty((n_spat, gp_temp))
    
    for temp in range(0, gp_temp):
        test =  trans.forward(var.isel(time = temp), nlevels=n_spat)
        for scale in (range(0,n_spat)):
            scales[scale,temp] = np.sum(np.abs(test.highpasses[scale].real)**2 + np.abs(test.highpasses[scale].imag)**2)
    
    #scales = scales/(np.sum(scales,axis=0)+0.0000001)
    return(scales)



##################################### 1) Investigate Spatio-Temporal Scales:
##############################################################


def wavelet_En_ST(var, coord):
    
    gp_spat = var.sizes[coord]
    gp_temp = var.sizes['time']
    n_spat = int(np.log2(gp_spat)) 
    n_temp = int(np.log2(gp_temp)) 
                
    #### First step: 1D wavelet-transform on each gridpoint
    # Initialize DTCWT transform object
    trans = dtcwt.Transform1d()  
    X = var.values.reshape((gp_temp,gp_spat*gp_spat))
    
    test = trans.forward(X, nlevels=n_temp)
    
    scales_RE = [test.highpasses[scale].real.reshape(2**(n_temp-scale-1),gp_spat,gp_spat) for scale in (range(0,n_temp))]
    scales_IM = [test.highpasses[scale].imag.reshape(2**(n_temp-scale-1),gp_spat,gp_spat) for scale in (range(0,n_temp))]
       
    # Initialize DTCWT transform object
    trans = dtcwt.Transform2d()
            
    spacetime =  {(scaleT, scaleS): np.empty((2**(n_temp-scaleT-1), 2**(n_spat-scaleS-1), 2 **(n_spat-scaleS-1)), dtype=np.float32)
        for scaleT, scaleS in itertools.product(range(0, n_temp), range( n_spat))
    }
            
    for scT in range(0, n_temp):
        for temp in range(0,2**(n_temp-scT-1)): 
            #data = scales_RE[scT-1][temp,:,:]
            traRE = trans.forward(scales_RE[scT][temp,:,:], nlevels=n_spat)
            traIM = trans.forward(scales_IM[scT][temp,:,:], nlevels=n_spat)
            for spar in range(0,n_spat):
                spacetime_RE =  np.sum(np.abs(traRE.highpasses[spar].real)**2 + np.abs(traRE.highpasses[spar].imag)**2, axis = 2)
                spacetime_IM =  np.sum(np.abs(traIM.highpasses[spar].real)**2 + np.abs(traIM.highpasses[spar].imag)**2, axis = 2)
                spacetime[scT,spar][temp,:,:] = spacetime_RE + spacetime_IM
    
    
    res = np.empty([n_temp,n_spat])
    for sc1,sc2 in spacetime:
        res[(-sc1-1), sc2] = np.sum(spacetime[sc1,sc2]) 
        # print('scale1: ' + str(sc1)+ ' scale2: ' +str(sc2)+' value: '+ str(sum(spacetime[sc1,sc2])) )
        
    
    #return(spacetime)    
    return(res)




def rotated_to_geographic(lon_r, lat_r, pole_lon, pole_lat):
    """
    Convert rotated grid coordinates to geographic coordinates.
    
    Parameters:
    lon_r (float or array): Longitude in the rotated grid
    lat_r (float or array): Latitude in the rotated grid
    pole_lon (float): Longitude of the rotated pole (geographic)
    pole_lat (float): Latitude of the rotated pole (geographic)
    
    Returns:
    lon (float or array): Geographic longitude
    lat (float or array): Geographic latitude
    """
    
    # Convert degrees to radians
    lon_r = np.radians(lon_r)
    lat_r = np.radians(lat_r)
    pole_lon = np.radians(pole_lon)
    pole_lat = np.radians(pole_lat)
    
    # Calculate geographic latitude
    sin_lat = np.sin(lat_r) * np.sin(pole_lat) + np.cos(lat_r) * np.cos(pole_lat) * np.cos(lon_r)
    lat = np.arcsin(sin_lat)
    
    # Calculate geographic longitude
    cos_lon_diff = (np.cos(lat_r) * np.sin(lon_r)) / np.cos(lat)
    lon_diff = np.arccos(cos_lon_diff)
    
    # Adjust the longitude based on the quadrant
    lon = pole_lon + np.where(np.sin(lon_r) >= 0, lon_diff, -lon_diff)
    
    # Convert radians back to degrees
    lon = np.degrees(lon)
    lat = np.degrees(lat)
    
    return lon, lat



def geographic_to_rotated(lon, lat, pole_lon, pole_lat):
    """
    Convert geographic coordinates to rotated grid coordinates.
    
    Parameters:
    lon (float or array): Geographic longitude
    lat (float or array): Geographic latitude
    pole_lon (float): Longitude of the rotated pole (geographic)
    pole_lat (float): Latitude of the rotated pole (geographic)
    
    Returns:
    lon_r (float or array): Rotated grid longitude
    lat_r (float or array): Rotated grid latitude
    """
    
    # Convert degrees to radians
    lon = np.radians(lon)
    lat = np.radians(lat)
    pole_lon = np.radians(pole_lon)
    pole_lat = np.radians(pole_lat)
    
    # Calculate rotated latitude
    sin_lat_r = np.sin(lat) * np.sin(pole_lat) + np.cos(lat) * np.cos(pole_lat) * np.cos(lon - pole_lon)
    lat_r = np.arcsin(sin_lat_r)
    
    # Calculate rotated longitude
    cos_lon_r = (np.cos(lat) * np.sin(lon - pole_lon)) / np.cos(lat_r)
    lon_r = np.arcsin(cos_lon_r)
    
    # Adjust the longitude for the correct quadrant
    lon_r = np.where(np.cos(lat) * np.sin(lon - pole_lon) < 0, -lon_r, lon_r)
    
    # Convert radians back to degrees
    lon_r = np.degrees(lon_r)
    lat_r = np.degrees(lat_r)
    
    return lon_r, lat_r
