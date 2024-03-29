"""
Plot results for geostrophic adjustment simulations.

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

"""

savefig = True
#savefig = False
grav = 9.81

import numpy as np
import h5py
import yaml
import panguweather_utils as pw
import onnxruntime as ort
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cartopy.crs as ccrs
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
mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],zm=zm)

# 500Z plot
font = {'family':'DejaVu Sans','weight':'bold','size': 12}
matplotlib.rc('font', **font)

#plot_vec = False
plot_vec = True

projection = ccrs.Robinson(central_longitude=-90.)
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

panel_label = ['(A)','(B)','(C)','(D)']
figfile = 'geo_adjust_500_40N.pdf'
    
axi = -1
for it in [0,1,3,6]:
    axi+=1

    infile = opath+'geo_adjust_DJF_'+str(it)+'h.h5'
    print('reading from: ',infile)
    h5f = h5py.File(infile,'r')
    ivp_pl_save = h5f['ivp_pl_save'][:]
    h5f.close()

    pzdat = (ivp_pl_save[0,5,:,:]- mean_pl[0,5,:,:])/grav
    udat = (ivp_pl_save[3,5,:,:]- mean_pl[3,5,:,:])
    vdat = (ivp_pl_save[4,5,:,:]- mean_pl[4,5,:,:])
    basefield = mean_pl[0,5,:,:]/grav
        
    dcint = 20; ncint=5
    vscale = 150 # vector scaling (counterintuitive:smaller=larger arrows)

    if plot_vec:
        # Plot vectors on the map
        latskip = 7
        lonskip = 7
        alpha = 1.0
        col = 'g'
        cs = ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
        if '_40N' in infile:
            qk = ax[axi].quiverkey(cs, 0.65, 0.01, 10., r'$10~ m/s$', labelpos='E',coordinates='figure',color=col)
        elif '_0N' in infile:
            qk = ax[axi].quiverkey(cs, .5, .01, 10., r'$10~ m/s$', labelpos='E',coordinates='figure',color=col)
            
    # mean state or full field
    alpha = 1.0
    cints = np.arange(4800,6000,60.)
    cs = ax[axi].contour(lon,lat,basefield,levels=cints,colors='0.5',linewidths=2.0,transform=ccrs.PlateCarree(),alpha=alpha,zorder=0)
    # perturbations
    alpha = 0.75
    cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
    cints_neg = list(np.arange(-ncint*dcint,-dcint+.001,dcint))
    cints_pos = list(np.arange(dcint,ncint*dcint+.001,dcint))
    cs = ax[axi].contour(lon,lat,pzdat,levels=cints_neg,colors='b',linestyles='solid',transform=ccrs.PlateCarree(),alpha=alpha)
    cs = ax[axi].contour(lon,lat,pzdat,levels=cints_pos,colors='r',linestyles='solid',transform=ccrs.PlateCarree(),alpha=alpha)

    # colorize land
    ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

    ax[axi].coastlines(color='gray')
    gl = ax[axi].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.bottom_labels = False
    if axi ==3:
        gl.bottom_labels = True
        
    ax[axi].set_extent([120, 200, 25, 55],crs=ccrs.PlateCarree()) # Pacific
    ax[axi].text(113,25,panel_label[axi],transform=ccrs.PlateCarree())
    
    print('zmin for t=',it,':',np.min(pzdat),'m')
        
if savefig:
    print('saving figure to:',figfile)
    fig.tight_layout()
    plt.savefig(figfile,dpi=300,bbox_inches='tight')
    
