"""
Support functions for atmospheric dynamics experiments using the Pangu-Weather model.

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

Revisions:
           GJH, February 2024: refactor for public release; tests on NCAR machines
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging
import os,sys
import h5py

def fetch_tendency(opath,hr,season='DJF',zm=False):
    #
    # get mean-state tendency
    # opath: string absolute path to directory containing the file
    # hr: string for the model time step
    # 
    # optional:
    # season: string in file name for months ('DJF' or 'JAS')
    # zm: True for zonal-mean tendencies
    
    print('reading '+hr+'h tendencies for '+season)
    
    if zm:
        infile_mean_dt = opath+'mean_'+season+'_zm_dt_'+hr+'h.h5'
        h5f = h5py.File(infile_mean_dt,'r')
        mean_pl_dt = h5f['zm_pl_dt_'+hr+'h'][:]
        mean_sfc_dt = h5f['zm_sfc_dt_'+hr+'h'][:]    
    else:
        infile_mean_dt = opath+'mean_'+season+'_dt_'+hr+'h.h5'
        h5f = h5py.File(infile_mean_dt,'r')
        mean_pl_dt = h5f['mean_pl_dt_'+hr+'h'][:]
        mean_sfc_dt = h5f['mean_sfc_dt_'+hr+'h'][:]
        
    h5f.close()
        
    return mean_pl_dt,mean_sfc_dt

def fetch_mean_state(opath,season='DJF',zm=False):
    #
    # get mean-state grids
    # opath: string absolute path to directory containing the file
    # 
    # optional:
    # season: string in file name for months ('DJF' or 'JAS')
    # zm: True for zonal-mean tendencies
    
    print('reading mean state for '+season)
    
    if zm:
        infile_mean = opath+'mean_'+season+'_zm.h5'
        h5f = h5py.File(infile_mean,'r')
        mean_pl = h5f['zm_pl'][:]
        mean_sfc = h5f['zm_sfc'][:]
        lat = h5f['lat'][:]
        lon = h5f['lon'][:]
    else:
        infile_mean = opath+'mean_'+season+'.h5'
        h5f = h5py.File(infile_mean,'r')
        mean_pl = h5f['mean_pl'][:]
        mean_sfc = h5f['mean_sfc'][:]
        lat = h5f['lat'][:]
        lon = h5f['lon'][:]
        
    h5f.close()
        
    return mean_pl,mean_sfc,lat,lon

def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def run_panguweather(ort_session,ndays,input_pl,input_sfc,make_steady=False,heating=[0.],pl_dt=None,sfc_dt=None,pl_archive_var=-1,pl_archive_level=-1,sfc_archive_var=-1):

    if np.max(np.abs(heating)) > 0.:
        print('applying steady heating...')
    else:
        print('no heating...initial-value problem only')
        
    # if single archive variables are not specified, archive all
    if pl_archive_var == -1:
        print('archiving all variables...')
        archive_all = True
        tmp = np.zeros_like(input_sfc)
        save_sfc = np.repeat(tmp[None,:,:,:],ndays,axis=0)
        tmp = np.zeros_like(input_pl)
        save_pl = np.repeat(tmp[None,:,:,:,:],ndays,axis=0)
    else:
        print('archiving selected variables...')
        archive_all = False
        save_sfc = np.zeros([ndays,input_pl.shape[2],input_pl.shape[3]]) 
        save_pl = np.zeros([ndays,input_sfc.shape[1],input_sfc.shape[2]])

    # initialize
    input_test = np.copy(input_pl)
    input_surface_test = np.copy(input_sfc)

    # run nsteps
    for k in range(ndays):
        print('working on day=',k)
        output, output_surface = ort_session.run(
            None,
            {
                "input": input_test,
                "input_surface": input_surface_test,
            },
        )
                
        # output->input for next step
        input_test = output
        input_surface_test = output_surface

        # apply steady heating
        if np.max(np.abs(heating)) > 0.:
            input_test[2,:9,:,:] = input_test[2,:9,:,:] + heating[np.newaxis,np.newaxis,:,:]
#            input_test[2,5,:,:] = input_test[2,5,:,:] + heating[np.newaxis,np.newaxis,:,:]

        if make_steady:
            input_test = input_test - pl_dt
            input_surface_test = input_surface_test - sfc_dt
            
        if archive_all:
            save_sfc[k,:] = input_surface_test
            save_pl[k,:] = input_test
        else:
            save_sfc[k,:,:] = input_surface_test[sfc_archive_var,:,:]
            save_pl[k,:,:] = input_test[pl_archive_var,pl_archive_level,:,:]
            
    return save_pl,save_sfc

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(np.radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km

# function to make the fixed perturbations
def perturbations(lat_2d,lon_2d,ilat,ilon,amp,locRad=1000.,Z500=False):
    grav = 9.81
    nlat = lat_2d.shape[0]
    nlon = lon_2d.shape[1]
    site_lat = lat_2d[ilat,0]
    site_lon = lon_2d[0,ilon]
    lat_vec = np.reshape(lat_2d,[nlat*nlon])
    lon_vec = np.reshape(lon_2d,[nlat*nlon])
    dists = np.zeros(shape=[nlat*nlon])
    dists = np.array(haversine(site_lon,site_lat,lon_vec,lat_vec),dtype=np.float64)

    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    covLoc = np.ones(shape=[nlat*nlon],dtype=np.float64)

    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0
    
    # prevent negative values: calc. above may produce tiny negative
    covLoc[covLoc < 0.0] = 0.0
    
    if Z500:
        # 500Z:
        print('500Z perturbation...')
        perturb = np.reshape(covLoc*grav*amp,[nlat,nlon])
    else:
        # heating:
        print('heating perturbation...')
        perturb = np.reshape(covLoc*amp,[nlat,nlon])

    return perturb

def perturbations_ellipse(lat,lon,k,ylat,xlon,locRad):

    """
    center a localized ellipse at (xlat,xlon)
    k: meridional wavenumber; disturbance is non-zero up to first zero crossing in cos
    xlat: latitude, in degrees to center the function
    xlon: longitude, in degrees to center the function
    locRad: zonal GC distance, in km
    """
    km = 1.e3
    nlat = len(lat)
    nlon = len(lon)
 
    ilon = xlon*4. #lon index of center
    ilat = int((90.-ylat)*4.) #lat index of center
    yfunc = np.cos(np.deg2rad(k*(lat-ylat)))

    # first zero-crossing
    crit = np.cos(np.deg2rad(k*(lat[ilat]-ylat)))
    ll = np.copy(ilat)
    while crit>0:
        ll-=1
        crit = yfunc[ll]

    yfunc[:ll+1] = 0.
    yfunc[2*ilat-ll:] = 0.

    # gaspari-cohn in logitude only, at the equator
    dx = 6380.*km*2*np.pi/(360.) #1 degree longitude at the equator
    dists = np.zeros_like(lon)
    for k in range(len(lon)):
        dists[k] = dx*np.min([np.abs(lon[k]-xlon),np.abs(lon[k]-360.-xlon)])

    #locRad = 10000.*km
    hlr = 0.5*locRad; # work with half the localization radius
    r = dists/hlr;

    # indexing w.r.t. distances
    ind_inner = np.where(dists <= hlr)    # closest
    ind_outer = np.where(dists >  hlr)    # close
    ind_out   = np.where(dists >  2.*hlr) # out

    # Gaspari-Cohn function
    covLoc = np.ones(nlon)

    # for pts within 1/2 of localization radius
    covLoc[ind_inner] = (((-0.25*r[ind_inner]+0.5)*r[ind_inner]+0.625)* \
                         r[ind_inner]-(5.0/3.0))*(r[ind_inner]**2)+1.0
    # for pts between 1/2 and one localization radius
    covLoc[ind_outer] = ((((r[ind_outer]/12. - 0.5) * r[ind_outer] + 0.625) * \
                          r[ind_outer] + 5.0/3.0) * r[ind_outer] - 5.0) * \
                          r[ind_outer] + 4.0 - 2.0/(3.0*r[ind_outer])
    # Impose zero for pts outside of localization radius
    covLoc[ind_out] = 0.0

    # prevent negative values: calc. above may produce tiny negative
    covLoc[covLoc < 0.0] = 0.0

    # make the function
    [a,b] = np.meshgrid(covLoc,yfunc)
    perturb = a*b

    return perturb
