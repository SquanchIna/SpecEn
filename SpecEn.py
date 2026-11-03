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
import json
from pathlib import Path


dask.config.set({"array.slicing.split_large_chunks": True})



class SPECS():

    def _load_config(self, config_file):
        """
        Opens json file and loads configuration parameters
        as class attributes
        """
        with open(config_file) as f:
            config = json.load(f)#, object_hook=self._decode_datetime)

        self.outdir = config["outputdir"]
        self.files =  config["files"]
        self.varname = config["varname"]
        self.cent_lon = float(config["cent_lon"])
        self.cent_lat = float(config["cent_lat"])
        self.cent_time = datetime64(config["CentTime"])
        self.anz_lon = int(config["anzLon"])
        self.anz_lat = int(config["anzLat"])
        self.anz_time = int(config["anzT"])

        log_file =  self.outdir +  "/Data/logger.log"
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s: %(message)s",
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        logFormatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        )
        rootLogger = logging.getLogger()
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    def show_config(self):
        """
        Prints the configuration parameters
        """
        attributes = vars(self)
        # print("Configuration:")
        logging.info("Configuration:")
        for attr, value in attributes.items():
            # print(f"{attr}: {value}")
            logging.info(f"{attr}: {value}")

    def prepare_dataset(self,inputdir):

        if Path(inputdir).suffix != ".nc":
            data = xr.open_mfdataset(inputdir + "*.nc", engine="netcdf4")
        else:
            data = xr.open_dataset(inputdir, engine="netcdf4")

        self.mapproj = self._check_map_proj(data)
        data = data[self.varname]
        data = self._calculate_mm(data)
        data = data.chunk("auto")
        # Find index of Center coordinates         

        if self.anz_lon % 2 != 0:
            raise ValueError(f"anzLon muss gerade sein, erhalten: {self.anz_lat}")

        if self.anz_lat % 2 != 0:
            raise ValueError(f"anzLat muss gerade sein, erhalten: {self.anz_lat}")

        if self.anz_time % 2 != 0:
            raise ValueError(f"anzLat muss gerade sein, erhalten: {self.anz_time}")

        # calculate total number of gridpoints
        nx_total = 2 ** int(np.ceil(np.log2(self.anz_lon)))
        ny_total = 2 ** int(np.ceil(np.log2(self.anz_lat)))
        nt_total = 2 ** int(np.ceil(np.log2(self.anz_time)))

        self.N_x = (nx_total - self.anz_lon) // 2
        self.N_y = (ny_total - self.anz_lat) // 2
        self.N_t = (nt_total - self.anz_time) // 2

        # Define coordinates in Wavelet space
        n_spat = int(np.log2(nx_total))
        n_temp = int(np.log2(nt_total))
        self.S = 2 ** np.arange(0, n_spat)
        self.T = 2 ** np.arange(0, n_temp)

        # Check whether central coordinates are within the dataset range
        if not data.lon.min() < self.cent_lon < data.lon.max():
            raise ValueError(
                f"cent_lon = {self.cent_lon} is outside the dataset range. "
                f"Please choose a value between {data.lon.min().compute().item()} "
                f"and {data.lon.max().compute().item()}."
            )

        if not data.lat.min() < self.cent_lat < data.lat.max():
            raise ValueError(
                f"cent_lat = {self.cent_lat} is outside the dataset range. "
                f"Please choose a value between {data.lat.min().compute().item()} "
                f"and {data.lat.max().compute().item()}."
            )

        if not data.time.min() <= self.cent_time <= data.time.max():
            raise ValueError(
                f"cent_time = {self.cent_time} is outside the dataset range. "
                f"Please choose a value between {data.time.min().values} "
                f"and {data.time.max().values}."
            )

        # find Index of cent_lon, cent_lat
        dist = (data.lon - self.cent_lon) ** 2 + (data.lat - self.cent_lat) ** 2
        idx_y, idx_x = np.unravel_index( dist.argmin().compute().item(), dist.shape)
        idx_time =np.abs(data.time - self.cent_time).argmin().compute().item()

        # Ausschnitt definieren
        if "lon" in data.dims and "lat" in data.dims:
            self.x_dim, self.y_dim = "lon", "lat"
        elif "x" in data.dims and "y" in data.dims:
            self.x_dim, self.y_dim = "x", "y"
        else:
            raise ValueError("Could not identify spatial dimensions (lon/lat or x/y).")

        #logging.info( f"Dims of dataset identified: {self.x_dim}, {self.y_dim} " )

        # Select data subset
        data = data.isel(
            {
                self.x_dim: slice(idx_x   - self.anz_lon  // 2, idx_x  + self.anz_lon  // 2),
                self.y_dim: slice(idx_y   - self.anz_lat  // 2, idx_y  + self.anz_lat  // 2),
                "time": slice(idx_time - self.anz_time // 2, idx_time + self.anz_time // 2)
            }
        )

        nt = max(1, int(data.sizes["time"] * 0.2))
        nx = max(1, int(data.sizes[self.x_dim] * 0.2))
        ny = max(1, int(data.sizes[self.y_dim] * 0.2))

        ## smooth edges
        t = np.append(np.linspace(0, 1, nt),np.append(np.ones(data.time.shape[0]       -2*nt),np.linspace(1, 0, nt)))
        y = np.append(np.linspace(0, 1, ny),np.append(np.ones(data[self.y_dim].shape[0]-2*ny),np.linspace(1, 0, ny)))
        x = np.append(np.linspace(0, 1, nx),np.append(np.ones(data[self.x_dim].shape[0]-2*nx),np.linspace(1, 0, nx)))
        # Create a 3D grid of linear gradients

        weight_matrix = np.einsum("i,j,k->ijk", t, x, y)
        weights = xr.DataArray(
            weight_matrix,
            dims=("time", self.x_dim, self.y_dim),
            coords={
                "time": data.time,
                self.x_dim: data[self.x_dim],
                self.y_dim: data[self.y_dim]
            }
        )

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
        x_len, y_len, time_len = data.sizes[self.x_dim], data.sizes[self.y_dim], data.sizes['time']

        pad_kwargs = {self.x_dim: self.N_x, self.y_dim: self.N_y, 'time': self.N_t}
        data = data.pad(**pad_kwargs, constant_values=0)
        data = data.fillna(0)

        return data
    
    def decompose_1D(self,var):     
        gp_spat = var.sizes[self.x_dim]
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
        gp_spat = var.sizes[self.x_dim]
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
        
        gp_spat = var.sizes[self.x_dim]
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
        
        return(res)
    

    def saveFile(self,data, decomp1,decomp2,decomp3):
        #print(decomp1.shape)
        decomp1 = decomp1[:,self.N_y:(decomp1.shape[1]-self.N_y),self.N_x:(decomp1.shape[2]-self.N_x)]
        decomp2 = decomp2[:, self.N_t:(decomp2.shape[1]-self.N_t)]
        #print(decomp1.shape)
        output = xr.Dataset(
            data_vars = dict(
                WT_T= (["resT", "y","x"], decomp1),
                WT_S= (["resS", "time"], decomp2),
                WT  = (["resT", "resS"], decomp3),
                Data= (["time", "y","x"], data.values),
            ),
            coords = dict(
                x = ("x", data[self.x_dim].values),
                y = ("y", data[self.y_dim].values),
                time = ("time", data['time'].values),
                resT = ("resT", self.T),
                resS = ("resS", self.S),
            )
        )
        output.to_netcdf(path=  self.outdir + "/Data/specEns.nc")
        
        return(output)

    
    def plotEvent(self,data):
        fig = plt.figure(figsize=(10, 5))
        data.Data.sum(dim='time').plot(subplot_kws=dict(projection=self.mapproj), cmap= 'magma_r',cbar_kwargs={'label': "[mm]"})
        plt.title('Precipitation sum: '+ np.datetime_as_string(data.time.values[0], unit='h') + ' - ' + np.datetime_as_string(data.time.values[-1], unit='h') )
        ax = plt.gca()
        ax.add_feature(cp.feature.BORDERS, linestyle='-', alpha=1)
        ax.gridlines(draw_labels=False, y_inline=False)
        plt.savefig(self.outdir + "/Data/PrecipitationSum.png",bbox_inches='tight')
        plt.close()
    
    
    def plotDec1(self,data):
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
        plt.savefig(self.outdir + "/Data/Dominant_temp_scale.png",bbox_inches='tight')
        plt.close()
    

    def plotDec2(self,data):
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
        
        plt.savefig(self.outdir + "/Data/spat_scale.png",bbox_inches='tight')
        plt.close()
    
    
    def plotDec3(self,data):
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
        
        plt.savefig(self.outdir + "/Data/spat_temp_scale.png",bbox_inches='tight')
        plt.close()
    
        
    def _check_map_proj(self, output):
        if "RADKLIM" in output.attrs["title"]:
            projection = ccrs.Stereographic(true_scale_latitude=0.0, central_latitude=51.0, central_longitude= 9.0)
        else:
            projection = ccrs.PlateCarree()
        return projection

    def _calculate_mm(self, output):
        if output.attrs['units']=='kg m^-2 s^-1':
            output = output*60*60
        return output


    



