"""
compute a climatological mean state from a sample drawn from ERA5

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

Revisions:
           GJH, February 2024: refactor for public release; tests on NCAR machines
"""

import numpy as np
import glob
import climetlab as cml
import h5py

with open("../config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# write results here
path_mean = config['path_mean']

# select DJF or JAS initial conditions
mean_state_season = config['mean_state_season']

# grib data lives here:
dpath = '/absolute/path/here/'

if mean_state_season == 'DJF':
    dates_december = glob.glob(dpath+'*12??_*.grib')
    dates_january = glob.glob(dpath+'*01??_*.grib')
    dates_february = glob.glob(dpath+'*02??_*.grib')
    dates = dates_december+dates_january+dates_february
elif mean_state_season == 'JAS':    
    dates_july = glob.glob(dpath+'*07??_*.grib')
    dates_august = glob.glob(dpath+'*08??_*.grib')
    dates_september = glob.glob(dpath+'*09??_*.grib')
    dates = dates_july+dates_august+dates_september
else:
    raise('not a valid season. set mean_state_season to DJF or JAS')

#
# time-average over input files
#

param_level_pl = (
    ["z", "q", "t", "u", "v"],
    [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50],
)
param, level = param_level_pl
param_sfc = ["msl", "10u", "10v", "2t"]

k_pl = 0
k_sfc = 0
for infile in dates:
    print('\r working on ',k_pl+k_sfc,infile,end= ' ')
    if '_pl' in infile:
        fields_pl = cml.load_source("file",infile).sel(levtype="pl")
        fields_pl = fields_pl.sel(param=param, level=level)
        fields_pl = fields_pl.order_by(param=param, level=level)
        fields_pl_numpy = fields_pl.to_numpy(dtype=np.float32)
        fields_pl_numpy = fields_pl_numpy.reshape((5, 13, 721, 1440))
        if k_pl == 0:
            mean_pl = np.zeros_like(fields_pl_numpy)
        mean_pl = mean_pl + fields_pl_numpy
        k_pl+=1
    elif '_sfc' in infile:
        fields_sfc = cml.load_source("file",infile).sel(levtype="sfc")
        fields_sfc = fields_sfc.sel(param=param_sfc)
        fields_sfc = fields_sfc.order_by(param=param_sfc)
        fields_sfc_numpy = fields_sfc.to_numpy(dtype=np.float32)
        if k_sfc == 0:
            mean_sfc = np.zeros_like(fields_sfc_numpy)
        mean_sfc = mean_sfc + fields_sfc_numpy
        k_sfc+=1
        
mean_pl=mean_pl/k_pl
mean_sfc = mean_sfc/k_sfc
print(mean_pl.shape,mean_sfc.shape)
# -

mean_pl=mean_pl/k_pl
mean_sfc = mean_sfc/k_sfc

# ERA5 lat,lon grid
lat = 90 - np.arange(721) * 0.25
lon = np.arange(1440) * 0.25
nlat = len(lat)
nlon = len(lon)
lat_2d = np.repeat(lat[:,np.newaxis],lon.shape[0],axis=1)
lon_2d = np.repeat(lon[np.newaxis,:],lat.shape[0],axis=0)

outfile = path_mean+'mean_'+mean_state_season+'.h5'
print('writing output here: ',outfile)
h5f = h5py.File(outfile, 'w')
h5f.create_dataset('mean_pl',data=mean_pl)
h5f.create_dataset('mean_sfc',data=mean_sfc)
h5f.create_dataset('lat',data=lat)
h5f.create_dataset('lon',data=lon)
h5f.create_dataset('k_pl',data=k_pl)
h5f.create_dataset('k_sfc',data=k_sfc)
h5f.close()


