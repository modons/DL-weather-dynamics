"""
geostrophic & hydrostatic adjustment, using a subset of the cyclone initial condition

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

Revisions:
           GJH, February 2024: refactor for public release; tests on NCAR machines
"""

import numpy as np
import logging
import yaml
import h5py
import panguweather_utils as pw
import torch
import onnxruntime as ort

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

# model time step is 1hr for these experiments 
dhr = 1

# set flag for zonal-mean simulations
zm = config['zm']

# output path
opath = config['path_output']

# output data file name
outfile = opath+'geo_adjust_'+config['mean_state_season']+'.h5'

logger.info('reading mean state...')
mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],zm=zm)

logger.info('reading mean state tendencies...')
mean_pl_dt_24h,mean_sfc_dt_24h = pw.fetch_tendency(config['path_mean'],'24',zm=zm)
mean_pl_dt_6h,mean_sfc_dt_6h = pw.fetch_tendency(config['path_mean'],'6',zm=zm)
mean_pl_dt_3h,mean_sfc_dt_3h = pw.fetch_tendency(config['path_mean'],'3',zm=zm)
mean_pl_dt_1h,mean_sfc_dt_1h = pw.fetch_tendency(config['path_mean'],'1',zm=zm)

# read initial condition
infile_iv = config['path_input']+'cyclone_DJF_40N_150E_regression.h5'
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
pangu6 = config['path_model']+'pangu_weather_6.onnx'
pangu3 = config['path_model']+'pangu_weather_3.onnx'
pangu1 = config['path_model']+'pangu_weather_1.onnx'

num_threads = 1
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = num_threads

logger.info('starting ONNX session for 24h model...')
ort_session_24 = ort.InferenceSession(pangu24,sess_options=options,providers=providers)
ort_session_6 = ort.InferenceSession(pangu6,sess_options=options,providers=providers)
ort_session_3 = ort.InferenceSession(pangu3,sess_options=options,providers=providers)
ort_session_1 = ort.InferenceSession(pangu1,sess_options=options,providers=providers)

# initial conditions (perturb 500hPa Z only)
logger.info('adding perturbation to the mean state...')
ivp_reg_pl = np.copy(mean_pl)
ivp_reg_pl[0,5,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_pl[0,5,iminlat:imaxlat,iminlon:imaxlon] + regf_pl[0,5,:,:]
ivp_reg_sfc = np.copy(mean_sfc)

t = 0
ofile = outfile[:-3]+'_'+str(t)+'h.h5'
logger.info('writing the initial condition to: '+ofile)

if only_500:
    h5f = h5py.File(ofile[:-3]+'_500hPa.h5', 'w')
    h5f.create_dataset('vars_500',data=ivp_reg_pl[(0,3,4),5,:,:])
else:
    h5f = h5py.File(ofile, 'w')
    h5f.create_dataset('ivp_pl_save',data=ivp_reg_pl)
    h5f.create_dataset('ivp_sfc_save',data=ivp_reg_sfc)
    h5f.create_dataset('lat',data=lat)
    h5f.create_dataset('lon',data=lon)
h5f.close()
 
# initialize 'old' states for each model with the IC for their first step
pl_last_24 = np.copy(ivp_reg_pl)
sfc_last_24 = np.copy(ivp_reg_sfc)
pl_last_6 = np.copy(ivp_reg_pl)
sfc_last_6 = np.copy(ivp_reg_sfc)
pl_last_3 = np.copy(ivp_reg_pl)
sfc_last_3 = np.copy(ivp_reg_sfc)

for t in np.arange(dhr,nhours+1,dhr):

    if t == 1:
        logger.info('first step: 1h model')
        ivp_pl_run = ivp_reg_pl
        ivp_sfc_run = ivp_reg_sfc
        ort_session = ort_session_1
        mean_pl_dt = mean_pl_dt_1h
        mean_sfc_dt = mean_sfc_dt_1h
    elif np.mod(t,24)==0:
        logger.info(str(t)+': 24 hr model')
        ivp_pl_run = pl_last_24
        ivp_sfc_run = sfc_last_24
        ort_session = ort_session_24
        mean_pl_dt = mean_pl_dt_24h
        mean_sfc_dt = mean_sfc_dt_24h
    elif np.mod(t,6)==0:
        logger.info(str(t)+': 6 hr model')
        ivp_pl_run = pl_last_6
        ivp_sfc_run = sfc_last_6
        ort_session = ort_session_6
        mean_pl_dt = mean_pl_dt_6h
        mean_sfc_dt = mean_sfc_dt_6h
    elif np.mod(t,3)==0:
        logger.info(str(t)+': 3 hr model')
        ivp_pl_run = pl_last_3
        ivp_sfc_run = sfc_last_3
        ort_session = ort_session_3
        mean_pl_dt = mean_pl_dt_3h
        mean_sfc_dt = mean_sfc_dt_3h
    else:
        logger.info(str(t)+': 1 hr model')
        ivp_pl_run = pl_last
        ivp_sfc_run = sfc_last
        ort_session = ort_session_1
        mean_pl_dt = mean_pl_dt_1h
        mean_sfc_dt = mean_sfc_dt_1h
  
    make_steady = True
    pl_tmp,sfc_tmp = pw.run_panguweather(ort_session,1,ivp_pl_run,ivp_sfc_run,make_steady=make_steady,pl_dt=mean_pl_dt,sfc_dt=mean_sfc_dt)
    
    pl_last = pl_tmp[-1,:]
    sfc_last = sfc_tmp[-1,:]
    
    # save old states for the appropriate model
    if np.mod(t,24)==0:
        pl_last_24 = np.copy(pl_last)
        sfc_last_24 = np.copy(sfc_last)
        pl_last_6 = np.copy(pl_last)
        sfc_last_6 = np.copy(sfc_last)
        pl_last_3 = np.copy(pl_last)
        sfc_last_3 = np.copy(sfc_last)
    elif np.mod(t,6)==0:
        pl_last_6 = np.copy(pl_last)
        sfc_last_6 = np.copy(sfc_last)
        pl_last_3 = np.copy(pl_last)
        sfc_last_3 = np.copy(sfc_last)
            
    # write to a file (no lat,lon; that's in the IC file)
    ofile = outfile[:-3]+'_'+str(t)+'h.h5'
    if only_500:
        logger.info('writing 500hPa (z,u,v) only...')
        h5f = h5py.File(ofile[:-3]+'_500hPa.h5', 'w')
        h5f.create_dataset('vars_500',data=pl_last[(0,3,4),5,:,:])
        h5f.close()
    else:
        logger.info('writing the entire state...')
        h5f = h5py.File(ofile, 'w')
        h5f.create_dataset('ivp_pl_save',data=pl_last)
        h5f.close()
