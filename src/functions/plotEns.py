######################
### plot spectral Energies
###########################
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

import matplotlib.gridspec as gridspec


#fig = plt.figure(figsize=(10, 5))
#output.Data.sum(dim='time').plot(subplot_kws=dict(projection=map_proj), cmap= 'magma_r',cbar_kwargs={'label': "[mm]"})
#plt.title('Precipitation sum: '+ np.datetime_as_string(output.time.values[0], unit='h') + ' - ' + np.datetime_as_string(output.time.values[-1], unit='h') )
#ax = plt.gca()
#ax.coastlines()
#ax.add_feature(cp.feature.BORDERS, linestyle='-', alpha=1)
#ax.gridlines(draw_labels=False, y_inline=False)
#plt.savefig(path_save +"Niederschlagsmenge.png",bbox_inches='tight')
#plt.close()




WT_T_norm = output.WT_T.values
WT_T_norm = WT_T_norm/(np.sum(WT_T_norm,axis=0)+0.0000001)                   ## Normalise (still needed?)

##calculate dominant scale
prec_sum    = output.Data.sum(dim='time')
scale       = np.argmax(WT_T_norm, axis=0)
scale       = np.where(prec_sum<10, np.nan, scale)
scale       = (res_T* 2**(scale))
scale_spart = xr.full_like(output.Data.isel(time=0), fill_value= np.nan)
scale_spart.values = scale


munich_lat = 48.1351
munich_lon = 11.5820
augsburg_lat = 48.3705
augsburg_lon = 10.8978
regensburg_lat = 49.0134
regensburg_lon = 12.1016
stuttgart_lat  = 48.7758
stuttgart_lon  = 9.1829

#################################################################################################################
############################# Plotting
#################################################################################################################
# Set seaborn style
sns.set(style="whitegrid")

# Your data processing and plotting code
#fig = plt.figure(figsize = (15, 5))
fig = plt.figure(figsize = (10, 10))
##fine colors for the filled areas
cmap = plt.cm.jet  # define the colormap
cmaplist = [cmap(i) for i in np.linspace(0,1,n_temp+1)]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, n_temp+1)
cmap.set_bad(color='grey')  # Color for NaN values

gs1 = gridspec.GridSpec(1, 1, figure=fig)
gs1.update(left=0.1, right=1, top=.9, bottom=0.1, wspace=0.2, hspace=0.2)

ax1 = fig.add_subplot(gs1[0,0], projection=map_proj)
scale_spart.plot(ax=ax1,levels= (res_T* 2**(np.arange(0,n_temp+1)-.5)),cmap=cmap,cbar_kwargs={'orientation':'horizontal','ticks': np.round(res_T* 2**(np.arange(0,n_temp+1)),2),'label': 'temporal scale [h]'})
ax1 = plt.gca()
ax1.coastlines()
ax1.add_feature(cp.feature.BORDERS, linestyle='-', alpha=1)
ax1.gridlines(draw_labels=False, y_inline=False)
# Add markers for Munich, Augsburg, and Regensburg
ax1.plot(munich_lon, munich_lat, marker='o', color='black', markersize=8, transform=ccrs.PlateCarree())
ax1.text(munich_lon + 0.1, munich_lat, 'Munich', transform=ccrs.PlateCarree(), fontsize=10)
ax1.plot(augsburg_lon, augsburg_lat, marker='o', color='black', markersize=8, transform=ccrs.PlateCarree())
ax1.text(augsburg_lon + 0.1, augsburg_lat, 'Augsburg', transform=ccrs.PlateCarree(), fontsize=10)
ax1.plot(regensburg_lon, regensburg_lat, marker='o', color='black', markersize=8, transform=ccrs.PlateCarree())
ax1.text(regensburg_lon + 0.1, regensburg_lat, 'Regensburg', transform=ccrs.PlateCarree(), fontsize=10)
ax1.plot(stuttgart_lon, stuttgart_lat, marker='o', color='black', markersize=8, transform=ccrs.PlateCarree())
ax1.text(stuttgart_lon + 0.1, stuttgart_lat, 'Stuttgart', transform=ccrs.PlateCarree(), fontsize=10)
ax1.set_title('Dominant temporal scale')


plt.savefig(path_save +"Temp_scale.png",bbox_inches='tight')
plt.close()



fig = plt.figure(figsize = (10, 10))

##define colors for the filled areas
cmap = plt.cm.jet  # define the colormap
cmaplist = [cmap(i) for i in np.linspace(0,1,n_spat+1)]
cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, n_spat)
cmap.set_bad(color='grey')  # Color for NaN values
bounds = np.arange(0, n_spat + 1)
norm = mcolors.BoundaryNorm(bounds, n_spat)

squared_data = output.Data ** 2


gs2 = gridspec.GridSpec(1, 1, figure=fig)
gs2.update(left=0.1, right=.9, top=0.74, bottom=0.35, wspace=0.2, hspace=0.2)
ax2 = fig.add_subplot(gs2[0,0])

ax2.plot(output.time.values , np.sum(output.WT_S, axis=0).values, color='black')
#ax2.plot(output.time.values,squared_data.sum(dim=('x','y'))/output.Data.mean(dim=('x','y','time')))
ax2.fill_between(output.time.values, output.WT_S[0, :], 0, facecolor=cmaplist[0], alpha=0.8)
for i in range(1, n_spat):
    ax2.fill_between(output.time.values, np.sum(output.WT_S[0:(i+1), :], axis=0), np.sum(output.WT_S[0:i, :], axis=0), facecolor=cmaplist[i], alpha=0.8)

# Create the colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Only needed for the colorbar
myFmt = mdates.DateFormatter('%m-%d')
ax2.xaxis.set_major_formatter(myFmt)
ax2.yaxis.set_ticklabels([])
ax2.set_title("Dominant spatial scale")
# Create the colorbar
cbar_ax = fig.add_axes([0.38, 0.18, 0.24, 0.04]) 
cbar = plt.colorbar(sm, cax=cbar_ax, ticks=np.arange(0, n_spat) + 0.5, orientation='horizontal')
cbar.set_label('Spatial Scale [km]')
cbar.set_ticklabels(res_S * 2 ** np.arange(0, n_spat))

plt.savefig(path_save +"Spat_scale.png",bbox_inches='tight')
plt.close()



WT_norm = output.WT
y = output.resT
x = output.resS
#WT_norm = WT_norm/np.sum(WT_norm)

fig = plt.figure(figsize=(5,3)) 
gs3 = gridspec.GridSpec(1, 1, figure=fig)
gs3.update(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.4)

ax3 = fig.add_subplot(gs3[0,0])

ax3.imshow(WT_norm, extent=[0, n_spat, 0, n_temp], origin="upper",cmap='magma_r')
#plt.plot(log2((np.arange(0,n_spat) + 0.5)), log2(1/36 *np.arange(0,n_spat) + 0.5) )
# Customize ticks and labels for the main plot
ax3.set_xticks((np.arange(0,n_spat) + 0.5), x.values)
ax3.set_yticks((np.arange(0,n_temp) + 0.5),y.values)
ax3.plot(np.arange(0,(n_spat+1))+.5,log2(res_S/(36))+(np.arange(0,(n_spat+1)))+.5,c='grey', ls='-.')        #*6.666/10
#plt.yticklabels(y)
ax3.set_xlabel('Spat[km]')
ax3.set_ylabel('Temp[h]')

#ax3.colorbar()        
ax3.set_ylim(0,n_temp)
ax3.set_xlim(0,n_spat)

plt.savefig(path_save +"Spat_Temp_scale.png",bbox_inches='tight')
plt.close()

