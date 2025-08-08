#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import pickle
from eofs.xarray import Eof
import datetime
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature



def concat_with_monthids(ds):
    
    t1 = ds.time[0]; tn = ds.time[-1]
    ds = ds.sel(time=slice(t1, tn)) 

    arrays = []
    n = 0
    month_ids = [] # integer identifiers for month associated with each timepoint (e.g. 1=Jan, 2=Feb, etc)

    for ens in ds.ensemble:

        vals = ds.sel(ensemble=ens)

        months = vals['time.month'].values.tolist()

        month_ids.extend(months)

        tstart = n
        n = n + len(vals.time)
        tfinish = n

        vals['time'] = np.arange(tstart, tfinish)

        arrays.append(vals)

    concatted = xr.concat(arrays, dim='time') 
    concatted = concatted.to_dataset(name='anoms').drop('ensemble')
    concatted['month'] = month_ids
    
    return concatted

def calc_weights(ds):
    '''
    Construct area-weighting matrix in shape of eof function input data
    Cosine(latitude)
    '''
    calc_weights = list(np.cos(np.deg2rad(ds.lat).values))
    shape_weights = np.tile(calc_weights, (len(ds.lon), 1))
    weights = shape_weights.T
    
    weights = xr.DataArray(weights, coords=[ds['lat'], ds['lon']], 
                           dims=['lat', 'lon'])
    return weights

def calc_EOFs(ds, path, filename):
    
    from pathlib import Path
    
    # if the file already exists
    if (Path(path + filename)).is_file():
        print('already calculated')
        
        # Load EOF solver object
        with open(path + filename, "rb") as fp:
            solver = pickle.load(fp)
    else:
        print('calculating EOFs')
 
        weights = calc_weights(ds)
        solver = Eof(ds, weights=weights) #, center=False)

        # Save solver as pickled object
        with open(path + filename, "wb") as fp: 
            pickle.dump(solver, fp)
        print('done')
          
    return solver

def areaweighted_mean(ds):    
    
    if hasattr(ds, 'lat'): # if latitude dimension has label 'lat'
        weights = np.cos(np.deg2rad(ds.lat)) 
        weights.name = "weights"
        array_weighted = ds.weighted(weights)
        weighted_mean = array_weighted.mean(("lon", "lat"))
        
    elif hasattr(ds, 'latitude'): # if latitude dimension has label 'latitude'
        weights = np.cos(np.deg2rad(ds.latitude)) 
        weights.name = "weights"
        array_weighted = ds.weighted(weights)
        weighted_mean = array_weighted.mean(("longitude", "latitude"))
        
    return weighted_mean


def add_gmean(data_array, var_name):
    '''
    For a gridded data array, calculate the area-weighted mean of the gridded data and save it 
    under variable name 'gmean'
    
    convert the data array into a dataset with the gridded variable name as var_name
    and gmean as the variable for the global mean
    '''
    ds = data_array.to_dataset(name=var_name)
    
    ds['gmean'] = areaweighted_mean(ds[var_name])
    
    return ds

def cdo_regrid(targetgrid, file_to_regrid, regridded):
    '''
    Regrid with CDO.
    all input objects are strings: paths to files of target grid (to regrid fild_to_regrid to),
    file to be regridded, and where to save/what to name regridded file
    *** DOES NOT WORK ON MULTIPLE ENSEMBLE MEMBERS. ONLY DIMENSIONS CAN BE LAT, LON, AND TIME ***
    '''
    import subprocess
    
    targetgrid = '-remapbil,' + targetgrid
    
    subprocess.run(['cdo', targetgrid, file_to_regrid, regridded]) 

def dateshift_netCDF(fp):
    """
    Function to shift and save netcdf with dates at midpoint of month.
    :param fp: dataarray to be reoriented
    """
    f = fp
    if np.unique(fp.indexes['time'].day)[0]==1 & len(np.unique(fp.indexes['time'].day))==1:
        new_time = fp.indexes['time']-datetime.timedelta(days=16)
        f = f.assign_coords({'time':new_time})
    return f

def check_for_nans(ds):
    """
    Checks if there are any NaNs in an xarray dataset.
    Args:
        ds (xarray.Dataset): The dataset to check.
    Returns:
        bool: True if there are any NaNs, False otherwise.
    """
    # Stack all data variables into a single array
    stacked_data = ds.stack(z=ds.dims)

    # Check for NaNs using NumPy's isnan and any
    has_nans = np.isnan(stacked_data.values).any() 

    return has_nans


# In[4]:


# # DONE

# path = '/home/msaenger/InVERT/CESM_LENS2/SSP370/raw_data/'

# keys = [] 

# # Combine .nc files using xr.open_mfdataset i.e. combine all
# # of the time chunks for a given run

# for file in os.listdir(path): # iterate over all files in folder of raw data
    
#     if file[-3:] == '.nc': # only select .nc files 

#         # select unique key for ensemble member
#         # (this may be different depending on the experiment file names)

#         key = file.split('LE2-')[1][:8]

#         keys.append(key)
        
# keys_list = np.unique(keys).tolist() # sort runs by unique keys
        
# print(keys_list) 

# print(len(keys_list))


# In[20]:


# # DONE

# ds_list = []

# for i, key in enumerate(keys_list):
    
#     print(i)  # track progress
    
#     ds = xr.open_mfdataset(path + '*LE2-' + key + '.cam*.nc') # Combine files by ensemble member
    
#     ds = dateshift_netCDF(ds['TREFHT']) # Shift dates to mid-month and isolate TREFHT
    
#     ds = ds.to_dataset(name='TREFHT') # convert from dataarray to dataset
    
#     ds_list.append(ds)


# In[21]:


# # DONE in Python Screen 

# path = '/home/msaenger/InVERT/CESM_LENS2/Historical/TREFHT/combined_by_ensemble_member/'

# for i, ds in enumerate(ds_list):
    
#     print(i)
    
#     print('saving')
        
#     ds.to_netcdf(path + 'LENS2_Historical_' + keys_list[i] + '.nc')
    
#     print('regridding')
    
#     cdo_regrid(targetgrid = '/home/msaenger/InVERT/targetgrid.nc',

#            file_to_regrid = path + 'LENS2_Historical_' + keys_list[i] + '.nc',

#            regridded = path + 'regridded/' + 'LENS2_Historical_' + keys_list[i] + '_regridded.nc')


# # LENS2 
# 
# ### Load regridded temperature data and combine into one dataset

# In[5]:


# ## DONE

# path = '/home/msaenger/InVERT/CESM_LENS2/SSP370/combined_by_ensemble_member/regridded/'

# # combine 50 regridded ensemble member files into one dataset with 'ensemble' dimension
# ds = xr.open_mfdataset(path + '*.nc', concat_dim='ensemble', 
#                          combine='nested', parallel=True)

# # assign ensemble coordinate
# ds = ds.assign_coords(ensemble=np.arange(len(ds['ensemble'])))

# ds


# In[7]:


def remove_seasonal_cycle(data):
    """
    Removes the seasonal cycle from a dataset of gridded temperature anomalies.
    Args:
        ds (xarray.Dataset): The dataset containing the temperature anomaly data.
    Returns:
        xarray.Dataset: The dataset with the seasonal cycle removed.
    """
    # Calculate the climatological monthly means
    climatology = data.groupby('time.month').mean(dim='time')

    # Subtract the climatological monthly means from the original data
    ds_deseasonalized = data.groupby('time.month') - climatology

    return ds_deseasonalized


# In[6]:


# # DONE

# ## Calculate ensemble mean and add to dataset as a variable
# ds['emean'] = ds.TREFHT.mean('ensemble')

# ## subtract ensemble mean to get anomalies 
# anoms = ds['TREFHT'] - ds['emean']

# # remove seasonal cycle from anomalies
# anoms_deseasoned = remove_seasonal_cycle(anoms)

# anoms_deseasoned


# In[7]:


# # DONE

# # Concatenate over time, preserve months

# anoms_deseasoned_concatted = concat_with_monthids(anoms_deseasoned)
# anoms_deseasoned_concatted['gmean'] = areaweighted_mean(anoms_deseasoned_concatted.anoms)
# anoms_deseasoned_concatted


# path = '/home/msaenger/InVERT/CESM_LENS2/SSP370/'




# ## DONE

# anoms_deseasoned_concatted.to_netcdf(path + \
#     'SSP370_regridded_monthly_TREFHT_anoms_deseasoned_concatted_with_gmean.nc')

## These are the temperature anomalies used in EOF decomposition and compared to InVERT


lpath = '/home/msaenger/InVERT/Vector_autoregression/LENS2_SSP370_info/'


# In[11]:


anoms_concatted = xr.open_dataset(lpath + \
    'SSP370_regridded_monthly_TREFHT_anoms_deseasoned_concatted_with_gmean.nc')



# adjust time from integers 1 - len(time)
anoms_concatted['time'] = np.arange(0, len(anoms_concatted.time))

# Save month IDs from original T anomaly time series
month_da = xr.DataArray(anoms_concatted.month.values,
                        coords={'time': anoms_concatted.time.values, 
                                'month': ('time', anoms_concatted.month.values)},
                        dims=['time'])

anoms_concatted['month'] = month_da



# ## Month-specific EOFs


#### DONE

# path = '/home/msaenger/InVERT/Vector_autoregression/LENS2_SSP370_info/EOFs_by_month/'

# for month in range(1,13):
    
#     month_EOFs = calc_EOFs(anoms_concatted.groupby('month')[month].anoms, path=path,
#                       filename = 'LENS2_HIST_regridded_monthly_deseasoned_Tanom_EOFs_month='+str(month))
    
#     print('month ' + str(month) + ' EOFs saved')




##### Re-introduce seasonal cycle (ONLY for return time figures )

# anoms_concatted = xr.open_dataset(lpath + \
#     'SSP370_regridded_monthly_TREFHT_anoms_deseasoned_concatted_with_gmean.nc')


# # adjust time from integers 1 - len(time)
# anoms_concatted['time'] = np.arange(0, len(anoms_concatted.time))


# # Save month IDs from original T anomaly time series
# month_da = xr.DataArray(anoms_concatted.month.values,
#                         coords={'time': anoms_concatted.time.values, 
#                                 'month': ('time', anoms_concatted.month.values)},
#                         dims=['time'])

from InVERT_functions import stack_time

## Re-introduce seasonal cycle from de-meaned LENS2 T anomalies

scenario = 'SSP370'

hpath = '/home/msaenger/InVERT/CESM_LENS2/'+scenario+\
        '/combined_by_ensemble_member/regridded/'

# combine 50 regridded ensemble member files into one dataset with 'ensemble' dimension

ds = xr.open_mfdataset(hpath + '*.nc', concat_dim='ensemble', 
                         combine='nested', parallel=True)

# assign ensemble coordinate
ds = ds.assign_coords(ensemble=np.arange(len(ds['ensemble'])))

## Calculate ensemble mean and add to dataset as a variable
ds['emean'] = ds.TREFHT.mean('ensemble')

## subtract ensemble mean to get anomalies 
anoms = ds['TREFHT'] - ds['emean']

## Stack over time dimension
LENS_anoms_stacked = stack_time(anoms)

# Add 'month' coordinate
LENS_anoms_stacked['month'] = month_da



lpath = '/home/msaenger/InVERT/Vector_autoregression/LENS2_SSP370_info/'

## DONE -- # Calculate monthly climatologies of de-meaned T anomalies
monthly_means = LENS_anoms_stacked.groupby('month').mean('time')



print('saving LENS monthly climatologies')
monthly_means.to_netcdf(lpath + 'monthly_means.nc') 


# Tanoms_lens['month'] = month_da

monthly_means = xr.open_dataarray(lpath + 'monthly_means.nc')

# Re-introduce seasonal cycle to the de-seasoned anomalies
# LENS
LENS_anoms_seasonal = LENS_anoms_stacked.groupby('month') + monthly_means


print('saving LENS with seasonal cycle re-introduced')
# DONE
LENS_anoms_seasonal.to_netcdf(lpath + 'LENS_'+scenario+'_Tanoms_with_seasonal_cycle_reintroduced.nc')
