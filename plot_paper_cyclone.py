"""
Plot results for baroclinic development simulations.

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

zm = False
mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],zm=zm)

"""

plot winter cyclone case

"""

# 500Z plot
font = {'family':'DejaVu Sans','weight':'bold','size': 12}
matplotlib.rc('font', **font)

#plot_vec = False
plot_vec = True

projection = ccrs.Robinson(central_longitude=-90.)
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

panel_label = ['(A)','(B)','(C)','(D)']

axi = -1
for it in [0,48,72,96]:
    axi+=1

    infile = opath+'cyclone_DJF_'+str(it)+'h.h5'
    print('reading from: ',infile)
    h5f = h5py.File(infile,'r')
    ivp_pl_save = h5f['ivp_pl_save'][:]
    h5f.close()

    pzdat = (ivp_pl_save[0,5,:,:]- mean_pl[0,5,:,:])/grav
    udat = (ivp_pl_save[3,5,:,:]- mean_pl[3,5,:,:])
    vdat = (ivp_pl_save[4,5,:,:]- mean_pl[4,5,:,:])
    #basefield = ivp_pl_save[it,0,1,:,:]/grav
    basefield = mean_pl[0,5,:,:]/grav

    dcint = 20; ncint=5
    vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)

    if plot_vec:
        # Plot vectors on the map
        latskip = 10
        lonskip = 10
        alpha = 1.0
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

    # colorize land
    ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

    # gridlines
    gl = ax[axi].gridlines(crs=ccrs.PlateCarree(),linewidth=1.0,color='gray', alpha=0.5,linestyle='--', draw_labels=True)

    ax[axi].set_extent([140, 260, 20, 70],crs=ccrs.PlateCarree()) # Pacific

    ax[axi].text(130,20,panel_label[axi],transform=ccrs.PlateCarree())

if savefig:
    fig.tight_layout()
    plt.savefig('IVP_500.pdf',dpi=300,bbox_inches='tight')

# surface plots with MSLP and T_2m anomalies
font = {'family':'DejaVu Sans','weight':'bold','size': 12}
matplotlib.rc('font', **font)

#plot_vec = False
plot_vec = True

fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

panel_label = ['(A)','(B)','(C)','(D)']

axi = -1
for it in [0,48,72,96]:
    axi+=1

    infile = opath+'cyclone_DJF_'+str(it)+'h.h5'
    print('reading from: ',infile)
    h5f = h5py.File(infile,'r')
    ivp_sfc_save = h5f['ivp_sfc_save'][:]
    h5f.close()

    udat = ivp_sfc_save[1,:,:]- mean_sfc[1,:,:]
    vdat = ivp_sfc_save[2,:,:]- mean_sfc[2,:,:]
    tdat = ivp_sfc_save[3,:,:]- mean_sfc[3,:,:]
    zdat = (ivp_sfc_save[0,:,:]- mean_sfc[0,:,:])/100.

    projection = ccrs.Robinson(central_longitude=-90.)

    if plot_vec:
        # Plot vectors on the map
        latskip = 10
        lonskip = 10
        vscale = 200 # vector scaling (counterintuitive:smaller=larger arrows)
        alpha = 1.0
        col = 'g'
        cs =ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
        qk = ax[axi].quiverkey(cs, 0.65, 0.01, 5., r'$5~ m/s$', labelpos='E',coordinates='figure',color=col)

    # add MSLP
    alpha = 1.0
    dcint = 2; ncint=10
    cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
    cs = ax[axi].contour(lon,lat,zdat,levels=cints,colors='k',linewidths=2.0,transform=ccrs.PlateCarree(),alpha=alpha)

    # add T_2m
    alpha = 0.75 
    dcint = 0.5; ncint = 4
    cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
    cints_neg = list(np.arange(-ncint*dcint,-dcint+.001,dcint))
    cints_pos = list(np.arange(dcint,ncint*dcint+.001,dcint))
    csc = ax[axi].contour(lon,lat,tdat,levels=cints_neg,colors='b',linestyles='solid',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0)
    csc = ax[axi].contour(lon,lat,tdat,levels=cints_pos,colors='r',transform=ccrs.PlateCarree(),alpha=alpha,zorder=-1)

    # colorize land
    ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

    # add gridlines
    gl = ax[axi].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--', draw_labels=True)
    ax[axi].set_extent([140, 260, 20, 70],crs=ccrs.PlateCarree()) # Pacific

    ax[axi].text(130,20,panel_label[axi],transform=ccrs.PlateCarree())

if savefig:
    fig.tight_layout()
    plt.savefig('IVP_SFC.pdf',dpi=300,bbox_inches='tight')

# plot with MSLP and 850 hPa water vapor mixing ratio anomalies
font = {'family':'DejaVu Sans','weight':'bold','size': 12}
matplotlib.rc('font', **font)

qscale = 1e3

plot_vec = False
#plot_vec = True

projection = ccrs.Robinson(central_longitude=-90.)
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

panel_label = ['(A)','(B)','(C)','(D)']

axi = -1
for it in [0,48,72,96]:
    axi+=1

    infile = opath+'cyclone_DJF_'+str(it)+'h.h5'
    print('reading from: ',infile)
    h5f = h5py.File(infile,'r')
    ivp_pl_save = h5f['ivp_pl_save'][:]
    ivp_sfc_save = h5f['ivp_sfc_save'][:]
    h5f.close()

    udat = ivp_sfc_save[1,:,:]- mean_sfc[1,:,:]
    vdat = ivp_sfc_save[2,:,:]- mean_sfc[2,:,:]
    tdat = ivp_sfc_save[3,:,:]- mean_sfc[3,:,:]
    zdat = (ivp_sfc_save[0,:,:]- mean_sfc[0,:,:])/100.
    qdat = (ivp_pl_save[1,3,:,:]- mean_pl[1,3,:,:])*qscale

    projection = ccrs.Robinson(central_longitude=-90.)

    if plot_vec:
        # Plot vectors on the map
        latskip = 10
        lonskip = 10
        vscale = 200 # vector scaling (counterintuitive:smaller=larger arrows)
        alpha = 1.0
        col = 'g'
        cs =ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
        qk = ax[axi].quiverkey(cs, 0.65, 0.01, 5., r'$5~ m/s$', labelpos='E',coordinates='figure',color=col)

    # add MSLP
    alpha = 0.75
    dcint = 2; ncint=10
    cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
    cints_neg = list(np.arange(-ncint*dcint,-dcint+.001,dcint))
    cints_pos = list(np.arange(dcint,ncint*dcint+.001,dcint))
    lw = 1.0
    csc = ax[axi].contour(lon,lat,zdat,levels=cints_neg,colors='b',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha,zorder=1)
    csc = ax[axi].contour(lon,lat,zdat,levels=cints_pos,colors='r',linestyles='solid',linewidths=lw,transform=ccrs.PlateCarree(),alpha=alpha,zorder=1)

    # q
    vmax = 1.5; vmin = -vmax
    alpha=1.0
    cs = ax[axi].pcolormesh(lon,lat,qdat,vmin=-vmax,vmax=vmax,cmap='BrBG',edgecolors='none',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0,rasterized=True)
    #fig.colorbar(cs,ax=ax[axi],orientation='horizontal',pad =0.01,shrink=0.5)
    fig.colorbar(cs,ax=ax[axi],fraction=0.046, pad=0.005)#,orientation='horizontal',pad =0.01,shrink=0.5)
    
    # colorize land
    ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

    ax[axi].coastlines(color='gray')
    # add gridlines
    gl = ax[axi].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--', draw_labels=True)
    gl.bottom_lables = False
    if axi != 0:
        gl.top_labels = False
    
    ax[axi].set_extent([140, 260, 20, 70],crs=ccrs.PlateCarree()) # Pacific

    ax[axi].text(130,20,panel_label[axi],transform=ccrs.PlateCarree())

savefig = True
#savefig = False
if savefig:
    plt.savefig('IVP_SFC_700q_raster.pdf',dpi=100,bbox_inches='tight')
    

"""

plot DJF zonal-mean cyclone case

"""

zm = True
mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],zm=zm)

# 500Z plot
font = {'family':'DejaVu Sans','weight':'bold','size': 12}
matplotlib.rc('font', **font)

#plot_vec = False
plot_vec = True

projection = ccrs.Robinson(central_longitude=-90.)
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

panel_label = ['(A)','(B)','(C)','(D)']

axi = -1
for it in [0,96,168,240]:
    axi+=1

    infile = opath+'cyclone_DJF_zm_'+str(it)+'h.h5'
    print('reading from: ',infile)
    h5f = h5py.File(infile,'r')
    ivp_pl_save = h5f['ivp_pl_save'][:]
    h5f.close()

    pzdat = (ivp_pl_save[0,5,:,:]- mean_pl[0,5,:,:])/grav
    udat = (ivp_pl_save[3,5,:,:]- mean_pl[3,5,:,:])
    vdat = (ivp_pl_save[4,5,:,:]- mean_pl[4,5,:,:])
    basefield = ivp_pl_save[0,5,:,:]/grav

    dcint = 15; ncint=10
    vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)

    if plot_vec:
        # Plot vectors on the map
        latskip = 15
        lonskip = 15
        alpha = 1.0
        col = 'g'
        cs = ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
        qk = ax[axi].quiverkey(cs, 0.7, 0.01, 10., r'$10~ m/s$', labelpos='E',coordinates='figure',color=col)

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

    # colorize land
    ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

    # gridlines
    gl = ax[axi].gridlines(crs=ccrs.PlateCarree(),linewidth=1.0,color='gray', alpha=0.5,linestyle='--', draw_labels=True)

    ax[axi].set_extent([140, 360, 10, 70],crs=ccrs.PlateCarree()) # Pacific

    ax[axi].text(130,10,panel_label[axi],transform=ccrs.PlateCarree())

if savefig:
    fig.tight_layout()
    plt.savefig('IVP_500_zm.pdf',dpi=300,bbox_inches='tight')

# surface plots
font = {'family':'DejaVu Sans','weight':'bold','size': 12}
matplotlib.rc('font', **font)

#plot_vec = False
plot_vec = True

projection = ccrs.Robinson(central_longitude=-90.)
fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(11*2,8.5*2),subplot_kw={'projection': projection}, layout='constrained')

panel_label = ['(A)','(B)','(C)','(D)']

axi = -1
for it in [0,96,168,240]:
    axi+=1

    infile = opath+'cyclone_DJF_zm_'+str(it)+'h.h5'
    print('reading from: ',infile)
    h5f = h5py.File(infile,'r')
    ivp_sfc_save = h5f['ivp_sfc_save'][:]
    udat = ivp_sfc_save[1,:,:]- mean_sfc[1,:,:]
    vdat = ivp_sfc_save[2,:,:]- mean_sfc[2,:,:]
    tdat = ivp_sfc_save[3,:,:]- mean_sfc[3,:,:]
    zdat = (ivp_sfc_save[0,:,:]- mean_sfc[0,:,:])/100.
    h5f.close()

    if plot_vec:
        # Plot vectors on the map
        latskip = 15
        lonskip = 15
        vscale = 250 # vector scaling (counterintuitive:smaller=larger arrows)
        alpha = 1.0
        col = 'g'
        cs =ax[axi].quiver(lon[::lonskip],lat[::latskip],udat[::latskip,::lonskip],vdat[::latskip,::lonskip],transform=ccrs.PlateCarree(),scale=vscale,color=col,alpha=alpha)
        qk = ax[axi].quiverkey(cs, 0.7, 0.01, 5., r'$5~ m/s$', labelpos='E',coordinates='figure',color=col)

    # add MSLP
    alpha = 1.0
    dcint = 2; ncint=10
    cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
    cs = ax[axi].contour(lon,lat,zdat,levels=cints,colors='k',linewidths=2.0,transform=ccrs.PlateCarree(),alpha=alpha)

    # add T_2m
    alpha = 1.0 
    if it>3:
        dcint = 2.0; ncint = 10
    else:
        dcint = 1.0; ncint = 10
    cints = list(np.arange(-ncint*dcint,-dcint+.001,dcint))+list(np.arange(dcint,ncint*dcint+.001,dcint))
    cints_neg = list(np.arange(-ncint*dcint,-dcint+.001,dcint))
    cints_pos = list(np.arange(dcint,ncint*dcint+.001,dcint))
    csc = ax[axi].contour(lon,lat,tdat,levels=cints_neg,colors='b',linestyles='solid',transform=ccrs.PlateCarree(),alpha=alpha,zorder=0)
    csc = ax[axi].contour(lon,lat,tdat,levels=cints_pos,colors='r',transform=ccrs.PlateCarree(),alpha=alpha,zorder=-1)

    # colorize land
    ax[axi].add_feature(cfeature.LAND,edgecolor='0.5',linewidth=0.5,zorder=-1)

    # add gridlines
    gl = ax[axi].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='gray', alpha=0.5, linestyle='--', draw_labels=True)
    ax[axi].set_extent([140, 360, 10, 70],crs=ccrs.PlateCarree()) # Pacific

    ax[axi].text(130,10,panel_label[axi],transform=ccrs.PlateCarree())

if savefig:
    fig.tight_layout()
    plt.savefig('IVP_SFC_zm.pdf',dpi=300,bbox_inches='tight')
