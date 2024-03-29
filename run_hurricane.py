"""
simulate an Atlantic tropical cyclone, using an initial condition based on climo regression.

this code loops over a range of initial amplitudes, and saves only surface fields.

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

Revisions:
           GJH, February 2024: refactor for public release; tests on NCAR machines
"""

# scaling amplitude for IC (1=climo amp in MSLP at the vortex point)
amps = [1.,5.,7.,8.,10.,10.,20.] 

import numpy as np
import logging
import yaml
import h5py
import panguweather_utils as pw
import torch
import onnxruntime as ort
import glob
import climetlab as cml
from scipy.stats import linregress

# initialize logger
logger = pw.get_logger()

# load config
logger.info('reading configuration file...')
with open("config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# convenience vars
Nvars_pl = len(config['var_pl'])
Nvars_sfc = len(config['var_sfc'])
Nlevels = len(config['levels'])
nhours = config['nhours']
ohr = config['ohr']
only_500 = config['only_500']

# output path
opath = config['path_output']

logger.info('reading mean state...')
zm = False
mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],season='JAS',zm=zm)

logger.info('reading mean state tendencies...')
mean_pl_dt_24h,mean_sfc_dt_24h = pw.fetch_tendency(config['path_mean'],'24',season='JAS',zm=zm)

# read initial condition
infile_iv = config['path_input']+'hurricane_JAS_15N_40W_regression.h5'
logger.info('reading perturbation file...'+infile_iv)
h5f = h5py.File(infile_iv, 'r')
regf_pl = h5f['regf_pl'][:]
regf_sfc = h5f['regf_sfc'][:]
lat = h5f['lat'][:]
lon = h5f['lon'][:]
iminlat = h5f['iminlat'][()]
imaxlat = h5f['imaxlat'][()]
iminlon = h5f['iminlon'][()]
imaxlon = h5f['imaxlon'][()]
h5f.close()

# set up lat and lon arrays
nlat = len(lat)
nlon = len(lon)
lat_2d = np.repeat(lat[:,np.newaxis],lon.shape[0],axis=1)
lon_2d = np.repeat(lon[np.newaxis,:],lat.shape[0],axis=0)

logger.info('checking on GPU availability...')
try:
    device_index = torch.cuda.current_device()
    providers = [("CUDAExecutionProvider",{"device_id": device_index,},)]
    logger.info('Got a GPU---no waiting!')
except:
    providers = ["CPUExecutionProvider"]
    logger.info('Using CPU---no problem!')

# paths to model weights
pangu24 = config['path_model']+'pangu_weather_24.onnx'

num_threads = 1
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = num_threads

logger.info('starting ONNX session for 24h model...')
ort_session_24 = ort.InferenceSession(pangu24,sess_options=options,providers=providers)

noq = False # flag for second amp=10 experiment w/o water vapor
for amp in amps:
    logger.info('running simulation for amp='+str(amp))
    # add the perturbation to the mean state; must do it this way since vector results are messed up
    ivp_reg_pl = np.copy(mean_pl)
    for var in range(Nvars_pl):
        for k in range(Nlevels):
            ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] + amp*regf_pl[var,k,:,:]

    ivp_reg_sfc = np.copy(mean_sfc)
    for var in range(Nvars_sfc):
        ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] + amp*regf_sfc[var,:,:]

    if amp == 10 and noq:
        logger.info('running no water vapor experiment...')
        ivp_reg_pl[1,:,:,:] = 0.
        
    # regression IVP simulation
    make_steady = True
    ndays = 12
    ivp_reg_pl_solution,ivp_reg_sfc_solution = pw.run_panguweather(ort_session_24,ndays,ivp_reg_pl,ivp_reg_sfc,make_steady=make_steady,pl_dt=mean_pl_dt_24h,sfc_dt=mean_sfc_dt_24h)

    # save the results
    if amp == 10 and noq:
        outfile = opath+'hurricane_amp'+str(int(amp))+'_noq.h5'
    else:
        outfile = opath+'hurricane_amp'+str(int(amp))+'.h5'
    h5f = h5py.File(outfile, 'w')
    h5f.create_dataset('TC_sfc',data=ivp_reg_sfc_solution)
    h5f.create_dataset('lat',data=lat)
    h5f.create_dataset('lon',data=lon)
    h5f.close()

    if amp == 10 and not noq:
        # second experiment w/o water vapor
        noq = True
