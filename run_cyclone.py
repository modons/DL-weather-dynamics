"""
simulate an extratropical cyclone, using an initial condition based on climo regression

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

# model time step defaults to 24h unless 6h output is requested
if ohr == 6:
    dhr = 6
else:
    dhr = 24

# set flag for zonal-mean simulations
zm = config['zm']

# output path
opath = config['path_output']

# output data file name
if zm:
    outfile = opath+'cyclone_'+config['mean_state_season']+'_zm.h5'
else:
    outfile = opath+'cyclone_'+config['mean_state_season']+'.h5'

logger.info('reading mean state...')
mean_pl,mean_sfc,lat,lon = pw.fetch_mean_state(config['path_mean'],zm=zm)

logger.info('reading mean state tendencies...')
mean_pl_dt_24h,mean_sfc_dt_24h = pw.fetch_tendency(config['path_mean'],'24',zm=zm)
# always need 24h model; check if 6hr is needed too
if ohr == '6':
    mean_pl_dt_6h,mean_sfc_dt_6h = pw.fetch_tendency(config['path_mean'],'6',zm=zm)

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

num_threads = 1
options = ort.SessionOptions()
options.enable_cpu_mem_arena = False
options.enable_mem_pattern = False
options.enable_mem_reuse = False
options.intra_op_num_threads = num_threads

logger.info('starting ONNX session for 24h model...')
ort_session_24 = ort.InferenceSession(pangu24,sess_options=options,providers=providers)
if ohr == '6':
    logger.info('starting ONNX session for 6h model...')
    ort_session_6 = ort.InferenceSession(pangu6,sess_options=options,providers=providers)

# add the perturbation to the mean state; must do it this way since vector results are messed up
logger.info('adding perturbation to the mean state...')
ivp_reg_pl = np.copy(mean_pl)
for var in range(Nvars_pl):
    for k in range(Nlevels):
        ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_pl[var,k,iminlat:imaxlat,iminlon:imaxlon] + regf_pl[var,k,:,:]

ivp_reg_sfc = np.copy(mean_sfc)
for var in range(Nvars_sfc):
    ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] = ivp_reg_sfc[var,iminlat:imaxlat,iminlon:imaxlon] + regf_sfc[var,:,:]
    
t = 0
ofile = outfile[:-3]+'_'+str(t)+'h.h5'
logger.info('writing the initial condition to: '+ofile)

h5f = h5py.File(ofile, 'w')
if only_500:
    h5f.create_dataset('ivp_pl_save',data=ivp_reg_pl[:,5,:,:])    
else:
    h5f.create_dataset('ivp_pl_save',data=ivp_reg_pl)
    h5f.create_dataset('ivp_sfc_save',data=ivp_reg_sfc)
    h5f.create_dataset('lat',data=lat)
    h5f.create_dataset('lon',data=lon)
h5f.close()
 
# initialize 'old' states for each model with the IC for their first step
pl_last_24 = np.copy(ivp_reg_pl)
sfc_last_24 = np.copy(ivp_reg_sfc)

# loop over forecast lead time
for t in np.arange(dhr,nhours+1,dhr):

    if t == 6:
        logger.info('first step: 6h model')
        ivp_pl_run = ivp_reg_pl
        ivp_sfc_run = ivp_reg_sfc
        ort_session = ort_session_6
        mean_pl_dt = mean_pl_dt_6h
        mean_sfc_dt = mean_sfc_dt_6h
    elif np.mod(t,24)==0:
        logger.info(str(t)+' 24h model')
        ivp_pl_run = pl_last_24
        ivp_sfc_run = sfc_last_24
        ort_session = ort_session_24
        mean_pl_dt = mean_pl_dt_24h
        mean_sfc_dt = mean_sfc_dt_24h
    else:
        logger.info(str(t)+' 6h model')
        ivp_pl_run = pl_last
        ivp_sfc_run = sfc_last
        ort_session = ort_session_6
        mean_pl_dt = mean_pl_dt_6h
        mean_sfc_dt = mean_sfc_dt_6h
  
    make_steady = True
    pl_tmp,sfc_tmp = pw.run_panguweather(ort_session,1,ivp_pl_run,ivp_sfc_run,make_steady=make_steady,pl_dt=mean_pl_dt,sfc_dt=mean_sfc_dt)
    
    pl_last = pl_tmp[-1,:]
    sfc_last = sfc_tmp[-1,:]
    
    if np.mod(t,24)==0:
        print('copying 24 hour output for the next 24 step IC...')
        pl_last_24 = np.copy(pl_last)
        sfc_last_24 = np.copy(sfc_last)
        
    # write to a file (no lat,lon; that's in the IC file)
    ofile = outfile[:-3]+'_'+str(t)+'h.h5'
    print('writing to: ',ofile)
    h5f = h5py.File(ofile, 'w')
    if only_500:
        h5f.create_dataset('ivp_pl_save',data=pl_last[:,5,:,:])        
    else:
        h5f.create_dataset('ivp_pl_save',data=pl_last)
        h5f.create_dataset('ivp_sfc_save',data=sfc_last)
    h5f.close()
