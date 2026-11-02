import xarray as xr ## data handling
import scipy
import numpy as np  ## maths
import sys, os
from matplotlib.pylab import *
import matplotlib.animation as animation
import pylab as py 
import xarray as xr ## data handling
import numpy as np  ## maths
import sys, os
from matplotlib.pylab import *
import matplotlib.animation as animation
import pylab as py 
from matplotlib.colors import ListedColormap
import multiprocessing as mp
#from joblib import Parallel, delayed
import itertools  ##nestet loop
from scipy.ndimage import uniform_filter
import dtcwt
import dask  #
import dask.array as da
import cartopy.crs as ccrs
import cartopy  as cp
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates


dask.config.set({"array.slicing.split_large_chunks": True})



class SPECS():
    
    def __init__( self,
                  min_lat,
                  max_lat,
                  min_lon,
                  max_lon,
                  CentTime,
                  anzT,
                  varname
                ):
        self.min_lat = min_lat
        self.min_lon = min_lon
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.CentTime = CentTime
        self.anzT    = anzT
        self.varname = varname
    
    def prepare_dataset(self,inputdir):
        data = xr.open_mfdataset(inputdir+"*.nc",engine="netcdf4")
        self.mapproj = self._check_map_proj(data)
        data = data[self.varname]
        data = self._calculate_mm(data)
        data = data.chunk("auto")
        # Find index of Center coordinates         
        
        mask_lon = (data.lon >= self.min_lon) & (data.lon <= self.max_lon)
        mask_lat = (data.lat >= self.min_lat) & (data.lat <= self.max_lat)
        mask = mask_lon & mask_lat
        data = data.where(mask.compute(), drop=True)
        
        inds = data.sel(time = self.CentTime, method = 'nearest')
        #index_of_CentLat = np.where(data.lat.values == inds.lat.values)[0][0] 
        #index_of_CentLon = np.where(data.lon.values == inds.lon.values)[0][0] 
        index_of_CentTime = np.where(data.time.values == inds.time.values)[0][0] 
        # 1) Cut file around center coordinates 
        data = data.isel(time = slice(index_of_CentTime-self.anzT, index_of_CentTime+self.anzT))
        
        print(data.sizes)
        # 2) Smooth data at outmost 10% of gridpoints
        if 'lon' in data.sizes: 
            self.coordX = 'lon'
            self.coordY = 'lat'
        elif 'x' in data.sizes: 
            self.coordX = 'x'
            self.coordY = 'y'
        else:
            raise ValueError(
                f'coordinates not implementet yet.'
            ) from None
        
        nt, nx, ny = [int((data.sizes['time']/100)*20),int((data.sizes[self.coordX]/100) *20),int((data.sizes[self.coordY]/100)*20)]
        t = np.append(np.linspace(0, 1, nt),np.append(np.ones(data.time.shape[0]-2*nt),np.linspace(1, 0, nt)))
        y = np.append(np.linspace(0, 1, ny),np.append(np.ones(data[self.coordY].shape[0]-2*ny),np.linspace(1, 0, ny)))
        x = np.append(np.linspace(0, 1, nx),np.append(np.ones(data[self.coordX].shape[0]-2*nx),np.linspace(1, 0, nx)))    
        # Create a 3D grid of linear gradients
        weight_matrix = np.outer(t, np.ones((x.shape[0],y.shape[0])))
        weight_matrix = weight_matrix.reshape((t.shape[0],x.shape[0], y.shape[0]))
        # Apply the linear gradients along y and x dimensions
        weight_matrix *= np.outer(x, y)
        dat = data*weight_matrix
        
        return dat

    
    def ensure_odd_and_equal(self,data):
        print(data.sizes)
        # Get the original dimensions
        x_len, y_len, time_len = data.sizes[self.coordX], data.sizes[self.coordY], data.sizes['time']
        # Check if x, y, and time dimensions are odd; if even, slice off one element
        if x_len % 2 != 0:
            cut_kwargs = {self.coordX: slice(0, x_len-1)}
            data = data.isel(**cut_kwargs)
            print(f"\n X dimension not even and cutted")  
        if y_len % 2 != 0:
            cut_kwargs = {self.coordY: slice(0, y_len-1)}
            data = data.isel(**cut_kwargs)
            print(f"\n Y dimension not even and cutted")  
        if time_len % 2 != 0:
            data = data.isel(time=slice(0, time_len - 1))
            print(f"\n Time dimension not even and cutted")   
        
        return data

    
    def pad_data(self, data):       
        # 3) Pad with zero at the edges 
        ### define numer of values to be added
        # Get updated dimensions after possible trimming
        x_len, y_len, time_len = data.sizes[self.coordX], data.sizes[self.coordY], data.sizes['time']
        # Make sure x and y dimensions are equal by trimming the larger one
        if x_len > y_len:
            padY = int((x_len - y_len)/2) 
            padX = 0    
        elif y_len > x_len:
            padX = int((y_len - x_len)/2) 
            padY = 0  
        
        pad_kwargs = {self.coordX: padX, self.coordY: padY}
        data = data.pad(**pad_kwargs, constant_values=0)
        print(data.sizes)
        pad_test = (2**np.arange(0,12) - data.sizes[self.coordY] *2)
        ind     = next((i for i, num in enumerate(pad_test) if num > 0), -1)
        self.pad_lat = int((2**np.arange(0,12)[ind]-data.sizes[self.coordY])/2)
        
        pad_test = (2**np.arange(0,12) - data.sizes[self.coordX] *2)
        ind     = next((i for i, num in enumerate(pad_test) if num > 0), -1)
        self.pad_lon = int((2**np.arange(0,12)[ind]-data.sizes[self.coordX])/2)
        
        pad_test = (2**np.arange(0,12) - data.sizes['time'] *2)
        ind     = next((i for i, num in enumerate(pad_test) if num > 0), -1)
        self.pad_time = int((2**np.arange(0,12)[ind]-data.sizes['time'])/2)
        
        pad_kwargs = {self.coordX: self.pad_lon, self.coordY: self.pad_lat, "time": self.pad_time}
        dat = data.pad(**pad_kwargs, constant_values=0)
        dat = dat.fillna(0)
        
        self.pad_lat = self.pad_lat + padY
        self.pad_lon = self.pad_lon + padX
        print(dat.sizes)
        return dat
    
    def decompose_1D(self,var):     
        gp_spat = var.sizes[self.coordX]
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
        
        scales = scales.reshape(n_temp, gp_spat, gp_spat)
        return scales
    
    def decompose_2D(self,var):     
        gp_spat = var.sizes[self.coordX]
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
    
    def decompose_3D(self,var):
        
        gp_spat = var.sizes[self.coordX]
        gp_temp = var.sizes['time']
        n_spat = int(np.log2(gp_spat)) 
        n_temp = int(np.log2(gp_temp)) 
        
        #### First step: 1D wavelet-transform on each gridpoint
        # Initialize DTCWT transform object
        trans = dtcwt.Transform1d()  
        X = var.values.reshape((gp_temp,gp_spat*gp_spat))
        
        test = trans.forward(X, nlevels=n_temp)
        
        scales_RE = [test.highpasses[scale].real.reshape(2**(n_temp-scale-1),gp_spat,gp_spat) for scale in (range(0,n_temp))]
        scales_IM = [test.highpasses[scale].real.reshape(2**(n_temp-scale-1),gp_spat,gp_spat) for scale in (range(0,n_temp))]
        
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
        
        return(res)
    

    def saveFile(self,data, data_pad, decomp1,decomp2,decomp3,S,T,out):
        print(decomp1.shape)
        decomp1 = decomp1[:,self.pad_lat:(decomp1.shape[1]-self.pad_lat),self.pad_lon:(decomp1.shape[2]-self.pad_lon)]
        decomp2 = decomp2[:, self.pad_time:(decomp2.shape[1]-self.pad_time)]
        print(decomp1.shape)
        output = xr.Dataset(
            data_vars = dict(
                WT_T= (["resT", "y","x"], decomp1),
                WT_S= (["resS", "time"], decomp2),
                WT  = (["resT", "resS"], decomp3),
                Data= (["time", "y","x"], data.values),
            ),
            coords = dict(
                x = ("x", data[self.coordX].values),
                y = ("y", data[self.coordY].values),
                time = ("time", data['time'].values),
                resT = ("resT", T),
                resS = ("resS", S),
            )
        )
        output.to_netcdf(path=out + "/SpecEns.nc")
        
        return(output)

    
    def plotEvent(self,data,out):
        fig = plt.figure(figsize=(10, 5))
        data.Data.sum(dim='time').plot(subplot_kws=dict(projection=self.mapproj), cmap= 'magma_r',cbar_kwargs={'label': "[mm]"})
        plt.title('Precipitation sum: '+ np.datetime_as_string(data.time.values[0], unit='h') + ' - ' + np.datetime_as_string(data.time.values[-1], unit='h') )
        ax = plt.gca()
        ax.add_feature(cp.feature.BORDERS, linestyle='-', alpha=1)
        ax.gridlines(draw_labels=False, y_inline=False)
        plt.savefig(out +"/Niederschlagsmenge.png",bbox_inches='tight')
        plt.close()
    
    
    def plotDec1(self,data,out):
        n_temp = data.sizes['resT']
        WT_T_norm = data.WT_T.values
        WT_T_norm = WT_T_norm/(np.sum(WT_T_norm,axis=0)+0.0000001)                   ## Normalise (still needed?)
        ##calculate dominant scale
        prec_sum    = data.Data.sum(dim='time')
        scale       = np.argmax(WT_T_norm, axis=0)
        #scale       = np.where(prec_sum<10, np.nan, scale)
        scale       = (2**(scale))
        scale_spart = xr.full_like(data.Data.isel(time=0), fill_value= np.nan)
        scale_spart.values = scale
        
        #Set seaborn style
        sns.set(style="whitegrid")       
        ##define colors for the filled areas
        cmap = plt.cm.jet  # define the colormap
        cmaplist = [cmap(i) for i in np.linspace(0,1,n_temp+1)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, n_temp+1)
        cmap.set_bad(color='grey')  # Color for NaN values
        ##plotting
        fig = plt.figure(figsize = (10, 5))
        gs1 = gridspec.GridSpec(1,1, figure=fig)
        gs1.update(left=0.1, right=1, top=.9, bottom=0.1, wspace=0.2, hspace=0.2)
        ax1 = fig.add_subplot(gs1[0,0], projection=self.mapproj)
        scale_spart.plot(ax=ax1,levels= (2**(np.arange(0,n_temp+1)-.5)),cmap=cmap,cbar_kwargs={'orientation':'vertical','ticks': np.round(2**(np.arange(0,n_temp+1)),2),'label': 'temporal scale'})
        ax1 = plt.gca()
        ax1.add_feature(cp.feature.BORDERS, linestyle='-', alpha=1)
        ax1.gridlines(draw_labels=False, y_inline=False)
        # Add markers for Munich, Augsburg, and Regensburg
        ax1.set_title('Dominant temporal scale')
        plt.savefig(out +"/Dominant_temp_scale.png",bbox_inches='tight')
        plt.close()
    

    def plotDec2(self,data,out):
        n_spat = data.sizes['resS']
        squared_data = data.Data ** 2
        ##define colors for the filled areas
         ##define colors for the filled areas
        cmap = plt.cm.jet  # define the colormap
        cmaplist = [cmap(i) for i in np.linspace(0,1,n_spat+1)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, n_spat)
        cmap.set_bad(color='grey')  # Color for NaN values
        bounds = np.arange(0, n_spat + 1)
        norm = mcolors.BoundaryNorm(bounds, n_spat)
        ##plotting
        fig = plt.figure(figsize = (10, 6))
        gs2 = gridspec.GridSpec(1, 1, figure=fig)
        gs2.update(left=0.1, right=.9, top=0.74, bottom=0.35, wspace=0.2, hspace=0.2)
        ax2 = fig.add_subplot(gs2[0,0])
        
        ax2.plot(data.time.values , np.sum(data.WT_S, axis=0).values, color='black')
        #ax2.plot(output.time.values,squared_data.sum(dim=('x','y'))/output.Data.mean(dim=('x','y','time')))
        ax2.fill_between(data.time.values, data.WT_S[0, :], 0, facecolor=cmaplist[0], alpha=0.8)
        for i in range(1, n_spat):
            ax2.fill_between(data.time.values, np.sum(data.WT_S[0:(i+1), :], axis=0), np.sum(data.WT_S[0:i, :], axis=0), facecolor=cmaplist[i], alpha=0.8)
        # Create the colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for the colorbar
        myFmt = mdates.DateFormatter('%m-%d')
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.yaxis.set_ticklabels([])
        ax2.set_title("spatial scale")
        # Create the colorbar
        cbar_ax = fig.add_axes([0.38, 0.18, 0.24, 0.04]) 
        cbar = plt.colorbar(sm, cax=cbar_ax, ticks=np.arange(0, n_spat) + 0.5, orientation='horizontal')
        cbar.set_label('Spatial Scale')
        cbar.set_ticklabels(2 ** np.arange(0, n_spat))
        
        plt.savefig(out +"/spat_scale.png",bbox_inches='tight')
        plt.close()
    
    
    def plotDec3(self,data,out):
        n_spat = data.sizes['resS']
        n_temp = data.sizes['resT']
        
        fig = plt.figure(figsize = (5, 5))
        gs3 = gridspec.GridSpec(1, 1, figure=fig)
        gs3.update(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.4)
        ax3 = fig.add_subplot(gs3[0,0])
        
        ax3.imshow(data.WT, extent=[0, n_spat, 0, n_temp], origin="upper",cmap='magma_r')
        #plt.plot(log2((np.arange(0,n_spat) + 0.5)), log2(1/36 *np.arange(0,n_spat) + 0.5) )
        # Customize ticks and labels for the main plot
        ax3.set_xticks((np.arange(0,n_spat)[::2] + 0.5),np.arange(0,n_spat)[::2])
        ax3.set_yticks((np.arange(0,n_temp) + 0.5),np.arange(0,n_temp))
        ax3.set_xlabel('spat. scale')
        ax3.set_ylabel('temp. scale')
        #ax3.colorbar()        
        ax3.set_ylim(0,n_temp)
        ax3.set_xlim(0,n_spat)
        
        plt.savefig(out +"/spat_temp_scale.png",bbox_inches='tight')
        plt.close()
    
        
    def _check_map_proj(self, output):
        if output.attrs['model_id'] == 'radklim':
            projection = ccrs.Stereographic(true_scale_latitude=0.0, central_latitude=51.0, central_longitude= 9.0)
        else:
            projection = ccrs.PlateCarree()
        return projection

    def _calculate_mm(self, output):
        if output.attrs['units']=='kg m^-2 s^-1':
            output = output*60*60
        return output


    



