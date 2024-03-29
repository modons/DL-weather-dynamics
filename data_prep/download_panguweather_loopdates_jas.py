"""
download all data needed for panguweather from CDS in grib format. 

Originator: Greg Hakim 
            ghakim@uw.edu
            University of Washington
            July 2023

Revisions:
           GJH, February 2024: refactor for public release; tests on NCAR machines

"""

import datetime
import numpy as np
import sys
import cdsapi

# set time variables for downloads here
#years = range(1979,1980)
#years = range(1980,2020)
years = range(2016,2021)
smonth = 7
emonth = 9

# path to write data
output_path = '/absolute/path/here/'

# Specify the variables, pressure levels, date, and output file path
sfc_variables = ["mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature"]
pl_variables = ["geopotential", 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']

def download_cds_grib_panguweather_pl(c,variables, pressure_levels, year, month, day, output_file):

    request = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': variables,
        'pressure_level': pressure_levels,
        'year': year,
        'month': month,
        'day': day,
        'time': '00:00'
    }

    print('retrieving pressure-level GRIB data from CDS for date:',year,month,day)
    print('writing the output here:',output_file)
    c.retrieve('reanalysis-era5-pressure-levels', request, output_file)
    
    return

def download_cds_grib_panguweather_sfc(c,variables, year, month, day, output_file):

    request = {
        'product_type': 'reanalysis',
        'format': 'grib',
        'variable': variables,
        'year': year,
        'month': month,
        'day': day,
        'time': '00:00',
        'levtype': 'sfc'
    }

    print('retrieving surface GRIB data from CDS for date:',year,month,day)
    print('writing the output here:',output_file)
    c.retrieve('reanalysis-era5-single-levels', request, output_file)

    return

# start the cds client and pass to the download functions
c = cdsapi.Client()

for year in years:
    print('working on year = ',year)
    start_date = datetime.date(year, smonth, 1)
    end_date = datetime.date(year, emonth, 30) # september of current year
            
    dates = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y%m%d")
        dates.append(date_str)
        current_date += datetime.timedelta(days=10)

    print(dates)

    for date in dates:
        # parse the date string
        year = date[:4]
        month = date[4:6]
        day = date[6:8]

        output_file_sfc = output_path+date+'_sfc.grib'
        output_file_pl = output_path+date+'_pl.grib'
        print(output_file_sfc,output_file_pl)
        
        download_cds_grib_panguweather_sfc(c,sfc_variables, year, month, day, output_file_sfc)
        download_cds_grib_panguweather_pl(c,pl_variables, pressure_levels, year, month, day, output_file_pl)
