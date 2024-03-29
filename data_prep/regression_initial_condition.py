"""
compute the regression initial conditions for the extratropical and tropical cyclone cases.

you must download ERA5 sample data to compute the regression. specifically, to repeat results in the Hakim & Masanam (2023) paper, ERA5 data are sampled every 10 days at 00UTC from 1979 to 2020.

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

"""

#
# START: parameters and setup
#

#nsamp = 200
nsamp = 1e6

# select DJF or JAS initial conditions
#ic = 'DJF'
ic = 'JAS'

# grib data lives here:
dpath = '/absolute/path/here/'

# write regression results here:
opath = '/absolute/path/here/'

# pangu-weather state info:
param_level_pl = (
    ["z", "q", "t", "u", "v"],
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
)
param, level = param_level_pl
nvars_pl = len(param)
nlevs = len(level)
param_sfc = ["msl", "10u", "10v", "2t"]
nvars_sfc = len(param_sfc)

#
# END: parameters and setup
#

import numpy as np
import h5py
import panguweather_utils as pw
import onnxruntime as ort
import climetlab as cml
from scipy.stats import linregress
import glob

print('computing the climo regression against one var at one point')

# ERA5 lat,lon grid
lat = 90 - np.arange(721) * 0.25
lon = np.arange(1440) * 0.25
nlat = len(lat)
nlon = len(lon)
lat_2d = np.repeat(lat[:,np.newaxis],lon.shape[0],axis=1)
lon_2d = np.repeat(lon[np.newaxis,:],lat.shape[0],axis=0)

if ic == 'DJF':
    dates_december = glob.glob(dpath+'*12??_*.grib')
    dates_january = glob.glob(dpath+'*01??_*.grib')
    dates_february = glob.glob(dpath+'*02??_*.grib')
    dates = dates_december+dates_january+dates_february
    # set the lat,lon of the location to define the perturbation (longitude degrees east)
    ylat = 40.; xlon = 150.; xlev = 5
    # localization radius in km for the scale of the initial perturbation
    locrad = 2000.
    # scaling amplitude for initial condition (1=climo varaince at the base point)
    amp = -1.
elif ic == 'JAS':    
    dates_july = glob.glob(dpath+'*07??_*.grib')
    dates_august = glob.glob(dpath+'*08??_*.grib')
    dates_september = glob.glob(dpath+'*09??_*.grib')
    dates = dates_july+dates_august+dates_september
    # set the lat,lon of the location to define the perturbation (longitude degrees east)
    ylat = 15.; xlon = 360.-40.; xlev = 5
    # localization radius in km for the scale of the initial perturbation
    locrad = 1000.
    # scaling amplitude for initial condition (1=climo varaince at the base point)
    amp = -1.
else:
    raise('not a valid season. set ic to DJF or JAS')

# base point indices
bplat = int((90.-ylat)*4); bplon = int(xlon)*4
print('lat, lon=',lat[bplat],lon[bplon])

locfunc = pw.perturbations(lat_2d,lon_2d,bplat,bplon,1.0,locRad=locrad)
print('locfunc max:',np.max(locfunc))

# indices where this function is greater than zero
nonzeros = np.argwhere(locfunc>0.)

# indices of rectangle bounding the region (fast array access)
iminlat = np.min(nonzeros[:,0])
imaxlat = np.max(nonzeros[:,0])
iminlon = np.min(nonzeros[:,1])
imaxlon = np.max(nonzeros[:,1])
latwin = imaxlat-iminlat
lonwin = imaxlon-iminlon
print(iminlat,imaxlat,lat[iminlat],lat[imaxlat])
print(iminlon,imaxlon,lon[iminlon],lon[imaxlon])
print(latwin,lonwin)

# get appropriate date files
ndates = len(dates)
print(ndates)

# get counts of files in each category (should be the same)
n_pl = 0
n_sfc = 0
for infile in dates:
    if '_pl' in infile:
       n_pl+=1
    elif '_sfc' in infile:
        n_sfc +=1

if nsamp < n_pl:
    n_pl = nsamp
    n_sfc = nsamp
else:
    nsamp = n_pl+n_sfc
    
print('number of files:',n_pl,n_sfc,nsamp)

# subset of data in the small volume defined by locfunc
regdat_pl = np.zeros([nvars_pl,n_pl,nlevs,latwin,lonwin])
k_pl = 0
for infile in dates[:nsamp*2]:
    if '_pl' in infile:
        print('\r working on ',k_pl,infile,end= ' ')
        fields_pl = cml.load_source("file",infile).sel(levtype="pl")
        fields_pl = fields_pl.sel(param=param, level=level)
        fields_pl = fields_pl.order_by(param=param, level=level)
        fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
        fields_pl_numpy = fields_pl_numpy.reshape((5, 13, 721, 1440))
        for var in range(nvars_pl):
            regdat_pl[var,k_pl,:] = fields_pl_numpy[var,:,iminlat:imaxlat,iminlon:imaxlon]
            #print(var,regdat_pl[var,k_pl,5,int(latwin/2),int(lonwin/2)])

        k_pl+=1

# get surface data in the volume
regdat_sfc = np.zeros([nvars_sfc,n_sfc,latwin,lonwin])
k_sfc = 0
for infile in dates[:nsamp*2]:
    if '_sfc' in infile:
        print('\r working on ',k_sfc,infile,end= ' ')
        #try:
        #    fields_sfc = cml.load_source("file",infile).sel(levtype="sfc")
        #except:
        #    print('filesystem glitch...skipping this one.')
        #    continue
        fields_sfc = cml.load_source("file",infile).sel(levtype="sfc")
        fields_sfc = fields_sfc.sel(param=param_sfc)
        fields_sfc = fields_sfc.order_by(param=param_sfc)
        fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)
        for var in range(nvars_sfc):
            regdat_sfc[var,k_sfc,:] = fields_sfc_numpy[var,iminlat:imaxlat,iminlon:imaxlon]
        k_sfc+=1

    
# center the data
regdat_pl = regdat_pl - np.mean(regdat_pl,axis=1,keepdims=True)
regdat_sfc = regdat_sfc - np.mean(regdat_sfc,axis=1,keepdims=True)

for var in range(nvars_pl):
    print(var,regdat_pl[var,:,5,int(latwin/2),int(lonwin/2)])

# define the independent variable: sample at the chosen point (middle of domain)
if ic == 'DJF':
    xvar = regdat_pl[0,:,xlev,int(latwin/2)+1,int(lonwin/2)+1] # upper level
elif ic == 'JAS':
    xvar = regdat_sfc[0,:,int(latwin/2)+1,int(lonwin/2)+1] # surface

# standardize
xvar = xvar/np.std(xvar)

print('xvar shape:',xvar.shape)
print('xvar min,max:',np.min(xvar),np.max(xvar))

# regress pressure variables
regf_pl = np.zeros([nvars_pl,len(level),latwin,lonwin])
for var in range(nvars_pl):
    for k in range(len(level)):
        print('k=',k)
        for j in range(latwin):
            for i in range(lonwin):
                yvar = regdat_pl[var,:,k,j,i]
                slope,intercept,r_value,p_value,std_err = linregress(xvar,yvar)
                regf_pl[var,k,j,i] = slope*amp + intercept
                if j==latwin/2 and i == lonwin/2 and k == 5 and var == 0:
                    cov = np.matmul(xvar,yvar.T)/np.matmul(xvar,xvar.T)
                    #print('base point:',cov,slope,intercept,regf_pl[var,k,j,i])
                    
        # spatially localize
        regf_pl[var,k,:] = locfunc[iminlat:imaxlat,iminlon:imaxlon]*regf_pl[var,k,:]

# regress surface variables
regf_sfc = np.zeros([nvars_sfc,latwin,lonwin])
for var in range(nvars_sfc):
    for j in range(latwin):
        for i in range(lonwin):
            yvar = regdat_sfc[var,:,j,i]
            slope,intercept,r_value,p_value,std_err = linregress(xvar,yvar)
            regf_sfc[var,j,i] = slope*amp + intercept

    # spatially localize
    regf_sfc[var,:] = locfunc[iminlat:imaxlat,iminlon:imaxlon]*regf_sfc[var,:]

# save the regression field for later simulations
if ic == 'DJF':
    rgfile = opath+'cyclone_DJF_40N_150E_regression.h5'
elif ic == 'JAS':
    rgfile = opath+'hurricane_JAS_15N_40W_regression.h5'
    
h5f = h5py.File(rgfile, 'w')
h5f.create_dataset('regf_pl',data=regf_pl)
h5f.create_dataset('regf_sfc',data=regf_sfc)
h5f.create_dataset('lat',data=lat)
h5f.create_dataset('lon',data=lon)
h5f.create_dataset('iminlat',data=iminlat)
h5f.create_dataset('imaxlat',data=imaxlat)
h5f.create_dataset('iminlon',data=iminlon)
h5f.create_dataset('imaxlon',data=imaxlon)
h5f.close()
