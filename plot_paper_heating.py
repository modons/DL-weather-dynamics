
"""
Plot results for steady heating simulations.

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

"""

savefig = True
#savefig = False
grav = 9.81

import numpy as np
import yaml
import h5py
import panguweather_utils as pw
import onnxruntime as ort
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
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
zm = config['zm']

mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],zm=zm)

# 500Z plot
fsize = 14
font = {'family':'DejaVu Sans','weight':'bold','size':fsize}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 2.0 #set the value globally

plot_vec = False
#plot_vec = True

projection = ccrs.Robinson(central_longitude=120.)
fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

panel_label = ['(A)','(B)','(C)']

axi = -1
for it in [120,240,480]:
    axi+=1

    infile = opath+'heating_DJF_'+str(it)+'h.h5'
    print('reading from: ',infile)
    h5f = h5py.File(infile,'r')
    ivp_pl_save = h5f['ivp_pl_save'][:]
    heating = h5f['heating'][:]
    h5f.close()

    pzdat = (ivp_pl_save[0,5,:,:]- mean_pl[0,5,:,:])/grav
    udat = (ivp_pl_save[3,5,:,:]- mean_pl[3,5,:,:])
    vdat = (ivp_pl_save[4,5,:,:]- mean_pl[4,5,:,:])
    basefield = mean_pl[0,5,:,:]/grav
    
    if it == 0:
        dcint = .00001; ncint=5
    elif it == 120:
        dcint = .3; ncint=5
        vscale = 50 # vector scaling (counterintuitive:smaller=larger arrows)
    elif it == 240:
        dcint = 2; ncint=5        
        vscale = 100 # vector scaling (counterintuitive:smaller=larger arrows)
    else:
        dcint = 20; ncint=5
        vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)
    
    if plot_vec:
        # Plot vectors on the map
        latskip = 10
        lonskip = 10
        alpha = 0.75
        col = 'g'
        cs = ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
        qk = ax[axi].quiverkey(cs, 0.65, 0.01, 10., r'$10~ m/s$', labelpos='E',coordinates='figure',color=col)

    # mean state or full field
    alpha = 1.0
    cints = np.arange(4800,6000,60.)
    cs = ax[axi].contour(lon,lat,basefield,levels=cints,colors='0.5',transform=ccrs.PlateCarree(),alpha=alpha)
    # perturbations
    alpha = 1.0
    cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
    cints_neg = list(np.arange(-ncint*dcint,-dcint+.001,dcint))
    cints_pos = list(np.arange(dcint,ncint*dcint+.001,dcint))
    lw = 2.
    cs = ax[axi].contour(lon,lat,pzdat,levels=cints_neg,colors='b',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha)
    cs = ax[axi].contour(lon,lat,pzdat,levels=cints_pos,colors='r',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha)

    # plot heating
    cs = ax[axi].contour(lon,lat,heating,levels=[.05],colors='r',linestyles='dashed',linewidths=4,transform=ccrs.PlateCarree(),alpha=alpha)

    # colorize land
    ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

    gl = ax[axi].gridlines(crs=ccrs.PlateCarree(),linewidth=1.0,color='gray', alpha=0.5,linestyle='--', draw_labels=True)
    gl.top_labels = False
    if axi != 2:
        gl.bottom_labels = False
    gl.xlabels_left = True

    ax[axi].text(-0.02,0.02,panel_label[axi],transform=ax[axi].transAxes)

if savefig:
    fig.tight_layout()
    plt.savefig('heating_500z_day20.pdf',dpi=300,bbox_inches='tight')

"""

plot 850hPa wind field at day = 20

"""

fsize = 22
font = {'family':'DejaVu Sans','weight':'bold','size':fsize}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 2.0 #set the value globally

projection = ccrs.Robinson(central_longitude=120.)
fig, ax = plt.subplots(figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

#infile = opath+'heating_DJF_480h.h5'
"""
infile = opath+'heating_DJF_240h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = h5f['ivp_pl_save'][:]
heating = h5f['heating'][:]
h5f.close()
infile = opath+'heating_DJF_360h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
ivp_pl_save = (ivp_pl_save + h5f['ivp_pl_save'][:])/2.
heating = h5f['heating'][:]
h5f.close()
"""
infile = opath+'heating_DJF_480h.h5'
print('reading from: ',infile)
h5f = h5py.File(infile,'r')
#ivp_pl_save = (ivp_pl_save + h5f['ivp_pl_save'][:])/2.
ivp_pl_save = h5f['ivp_pl_save'][:]
heating = h5f['heating'][:]
h5f.close()

udat = ivp_pl_save[3,2,:,:]- mean_pl[3,2,:,:]
vdat = ivp_pl_save[4,2,:,:]- mean_pl[4,2,:,:]

vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)

# Plot vectors on the map
latskip = 8
lonskip = 8
alpha = 1.0
col = 'g'
cs = ax.quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
qk = ax.quiverkey(cs,0.85,-.03, 10., r'$10~ m/s$', labelpos='E',coordinates='axes',color=col,fontproperties={'weight':'bold','size':fsize})

# plot heating
cs = ax.contour(lon,lat,heating,levels=[.05],colors='r',linestyles='dashed',linewidths=8,transform=ccrs.PlateCarree(),alpha=alpha)

# colorize land
ax.add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

gl = ax.gridlines(crs=ccrs.PlateCarree(),linewidth=2.0,color='0.75',alpha=1.0,linestyle='--',draw_labels=True)
gl.xlabel_style = {'size':fsize, 'weight':'bold'}

ax.set_extent([60,180,-25,25],crs=ccrs.PlateCarree()) # Tropical Pacific

if savefig:
    fig.tight_layout()
    plt.savefig('heating_day20_850wind.pdf',dpi=300,bbox_inches='tight')

