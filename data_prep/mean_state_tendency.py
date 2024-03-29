"""
compute and store one-step inference difference for the mean state

Originator: Greg Hakim
            ghakim@uw.edu
            University of Washington
            July 2023

"""

import numpy as np
import onnxruntime as ort
import logging
import yaml
import torch
import h5py
import panguweather_utils as pw

# define the model time step
dt = 24 

# initialize logger
logger = pw.get_logger()

# softlink config.yml to a machine-specific version to avoid edits here
logger.info('reading configuration...')
with open("../config.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# load mean state
infile_mean = config['path_mean']+'mean_'+config['mean_state_season']+'.h5'
logger.info('reading mean state file: '+infile_mean)
h5f = h5py.File(infile_mean,'r')
lat = h5f['lat'][:]
lon = h5f['lon'][:]
mean_pl = h5f['mean_pl'][:]
mean_sfc = h5f['mean_sfc'][:]    
h5f.close()

# option for zonal-mean
if config['zm']:
    nlon = len(lon)
    zm = np.mean(mean_pl,axis=3,keepdims=True)
    zm_pl = np.tile(zm,nlon)
    zm = np.mean(mean_sfc,axis=2,keepdims=True)
    zm_sfc = np.tile(zm,nlon)
    # save to a file
    outfile = config['path_mean']+'mean_'+config['mean_state_season']+'_zm.h5'
    h5f = h5py.File(outfile, 'w')
    h5f.create_dataset('zm_pl',data=zm_pl)
    h5f.create_dataset('zm_sfc',data=zm_sfc)
    h5f.create_dataset('lat',data=lat)
    h5f.create_dataset('lon',data=lon)
    h5f.close()
    # reset variable names for tendency calculation
    mean_pl = zm_pl
    mean_sfc = zm_sfc

pangu = config['path_model']+'pangu_weather_'+str(dt)+'.onnx'
logger.info('using model: '+pangu)
num_threads = 1
try:
    device_index = torch.cuda.current_device()
    providers = [("CUDAExecutionProvider",{"device_id": device_index,},)]
    logger.info('Got a GPU---no waiting!')
except:
    providers = ["CPUExecutionProvider"]
    logger.info('Using CPU---no problem!')

options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = num_threads

# generalize this to loop over all models...starting here?
logger.info('starting ONNX session...')
ort_session = ort.InferenceSession(pangu,sess_options=options,providers=providers)

# mean-state tendency from one-step inference
logger.info('starting inference...')
nsteps = 1
make_steady = False
mean_pl_dt,mean_sfc_dt = pw.run_panguweather(ort_session,nsteps,mean_pl,mean_sfc,make_steady=make_steady)
mean_pl_dt = mean_pl_dt[0,:] - mean_pl
mean_sfc_dt = mean_sfc_dt[0,:] - mean_sfc

if config['zm']:
#    outfile = config['path_mean']+'mean_'+config['mean_state_season']+'_zm_dt_'+str(dt)+'h.h5'
    outfile = config['path_mean']+'mean_'+config['mean_state_season']+'_zm_dt_'+str(dt)+'h_casper.h5'
    logger.info('writing results to :'+outfile)
    h5f = h5py.File(outfile, 'w')
    h5f.create_dataset('zm_pl_dt_'+str(dt)+'h',data=mean_pl_dt)
    h5f.create_dataset('zm_sfc_dt_'+str(dt)+'h',data=mean_sfc_dt)
    h5f.create_dataset('lat',data=lat)
    h5f.create_dataset('lon',data=lon)
    h5f.close()
else:
#    outfile = config['path_mean']+'mean_'+config['mean_state_season']+'_dt_'+str(dt)+'h.h5'
    outfile = config['path_mean']+'mean_'+config['mean_state_season']+'_dt_'+str(dt)+'h_casper.h5'
    logger.info('writing results to :'+outfile)
    h5f = h5py.File(outfile, 'w')
    h5f.create_dataset('mean_pl_dt_'+str(dt)+'h',data=mean_pl_dt)
    h5f.create_dataset('mean_sfc_dt_'+str(dt)+'h',data=mean_sfc_dt)
    h5f.create_dataset('lat',data=lat)
    h5f.create_dataset('lon',data=lon)
    h5f.close()
