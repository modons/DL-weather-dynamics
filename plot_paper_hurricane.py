"""
Plot results for hurricane simulations.

NOTE: simulations plotted here were mostly run using hurricane_loop.py

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023
"""

# save figures
savefig = True
#savefig = False

# hard hack for the initial the disturbance location (same as in run_hurricane.py)
ylat = 15. # degrees North
xlon = 40. # degrees West
amps = [1,5,7,8,10,20]

# lat,lon indices:
bplat = int((90.-ylat)*4); bplon = int((360.-xlon)*4) 

import numpy as np
import yaml
import h5py
import panguweather_utils as pw
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import warnings
warnings.filterwarnings("ignore")

# load config
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# experiment output path
opath = config['path_output']

zm = False
mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],season='JAS',zm=zm)
nlat = len(lat)
nlon = len(lon)
lat_2d = np.repeat(lat[:,np.newaxis],lon.shape[0],axis=1)
lon_2d = np.repeat(lon[np.newaxis,:],lat.shape[0],axis=0)

"""

process simulations that loop over IC amplitude

"""

def tc_vitals(tc_sfc,lat,lon):
    latmin = []
    lonmin = []
    pmin = []
    for it in range(tc_sfc.shape[0]):
        minp = np.where(tc_sfc[it,0,:,:] == np.min(tc_sfc[it,0,:,:]))
        latmin.append(lat[minp[0][0]])
        lonmin.append(lon[minp[1][0]])
        pmin.append(np.min(tc_sfc[it,0,:,:])/100.)
    return latmin,lonmin,pmin

# set a list of default colors to loop through in the plots 
cols = []
for key in mcolors.BASE_COLORS:
    cols.append(key)
    
# set global font properties
font = {'family':'DejaVu Sans','weight':'bold','size': 12}
matplotlib.rc('font', **font)

projection = ccrs.Robinson(central_longitude=-90.)

fig, ax = plt.subplots(figsize=(9,6),subplot_kw={'projection': projection})

ax.add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.set_extent([270, 330, 5, 55],crs=ccrs.PlateCarree()) # Atlantic (it=5+)
ax.coastlines(color='gray')

k = -1
pmsave = []
for h in amps:
    k+=1
    hfile = 'hurricane_amp'+str(h)+'.h5'
    h5f = h5py.File(opath+hfile,'r')
    TC_sfc = h5f['TC_sfc'][:]
    h5f.close()
    print('processing: ',h)
    latm,lonm,pmin = tc_vitals(TC_sfc-mean_sfc[np.newaxis,:,:,:],lat,lon)
    # hack to insert the initial condition
    latm.insert(0,lat[bplat]); lonm.insert(0,lon[bplon])
    pmsave.append(pmin)
    # fix outliers
    if k == 1:
        latm[8] = latm[7]; lonm[8] = lonm[7]
    elif k == 0:
        print(len(pmin))
        for p in range(3,len(latm)):
            print(p,latm[p],lonm[p])
            if pmin[p-1] > -1.: # 1 hPa threshold for noise
                latm[p] = latm[p-1]; lonm[p] = lonm[p-1]
        
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.plot(lonm,latm,marker='o', markersize=6, linestyle='-', linewidth=2, color=cols[k], transform=ccrs.PlateCarree(),label='x '+str(h))

    # add no-q experiment (have to handle this one as a special case)
    if h == 10: 
        hfile = 'hurricane_amp'+str(h)+'_noq.h5'
        h5f = h5py.File(opath+hfile,'r')
        TC_sfc = h5f['TC_sfc'][:]
        h5f.close()
        pmsave_noq = []
        for it in range(TC_sfc.shape[0]):
            diff = TC_sfc[it,0,:,:]/100. - mean_sfc[0,:,:]/100.
            expt = np.zeros_like(diff)
            expt[250:330,1100:1280] = diff[250:330,1100:1280]
            pmsave_noq.append(np.min(expt))
        
if savefig:
    plt.legend()
    plt.savefig('hurricane_map.pdf',dpi=300,bbox_inches='tight')

days = range(1,TC_sfc.shape[0]+1)
lw = 2
fig, ax = plt.subplots()
for h in range(len(amps)):
    ax.plot(days,pmsave[h],linewidth=lw,color=cols[h],label='x '+str(amps[h]))
    if amps[h] == 10:
        ax.plot(days[:2],pmsave_noq[:2],'k--',linewidth=lw,color=cols[h],label='x '+str(amps[h])+'_noq')
  
xl = ax.get_xlim()
ax.plot(xl,[0,0],'k-',linewidth=1)
ax.set_xlim(1,12)
plt.xlabel('time (days)',weight='bold')
plt.ylabel('minimum MSLP anomaly (hPa)',weight='bold')
plt.setp(ax.spines.values(),linewidth=lw)

if savefig:
    plt.legend()
    plt.savefig('hurricane_timeseries.pdf',dpi=300,bbox_inches='tight')



