#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title: Comprehensive Estuary Discharge Analysis and Visualization
Description: This script contains functions to extract and analyze estuary data.

Author: Marloes Bonenkamp
Date: April 22, 2025
"""
#%%
import h5py
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import rasterio
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from coordinate_transformation import (
    transform_coordinates, 
    create_spatial_index, 
    find_nearest_point_kdtree,
    efficient_basin_area_search
)
#%% 

def matlab_datenum_to_datetime(datenum):
    """
    Converts Matlab datenum into Python datetime
    """
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)
#%%

def load_data_once(mat_file_path, tif_path):
    """
    Load all necessary data from files once to avoid repeated file operations.
    
    Returns:
        dict: Dictionary containing all required data
    """
    data_cache = {}
    
    # Load mat file data
    with h5py.File(mat_file_path, 'r') as d:
        data_cache['rm_lat'] = np.array(d['rm_lat']).flatten()
        data_cache['rm_lon'] = np.array(d['rm_lon']).flatten()
        data_cache['lon_grid'] = np.array(d['lon_grid']).flatten()
        data_cache['lat_grid'] = np.array(d['lat_grid']).flatten()
        data_cache['discharge_series'] = np.array(d['discharge_series'])
        data_cache['sed_series'] = np.array(d['sed_series'])
        data_cache['t'] = np.array(d['t']).flatten()
    
    # Load basin area data
    with rasterio.open(tif_path) as src:
        data_cache['basin_area'] = np.flipud(src.read(1))
        data_cache['basin_area'][data_cache['basin_area'] < 0] = np.nan
    
    # Create datetime objects once
    data_cache['datetimes'] = [matlab_datenum_to_datetime(dn) for dn in data_cache['t']]
    
    return data_cache
#%%
def extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series):
    # Create spatial index once
    spatial_index = create_spatial_index((rm_lon, rm_lat))
    
    estuary_discharge_timeseries = {}
    estuary_rm_coords = {}

    for estuary, (lat, lon) in estuary_coords.items():
        # Transform lat/lon to rm_lon/rm_lat
        estuary_rm_lon, estuary_rm_lat = transform_coordinates(lon, lat)
        
        # Find nearest point using spatial index (much faster)
        nearest_index = find_nearest_point_kdtree((estuary_rm_lon, estuary_rm_lat), spatial_index)
        
        # Extract discharge time series
        estuary_discharge_timeseries[estuary] = discharge_series[nearest_index, :]
        estuary_rm_coords[estuary] = (rm_lon[nearest_index], rm_lat[nearest_index])

    return estuary_discharge_timeseries, estuary_rm_coords

#%%
def get_Qriver_timeseries(rm_lat, rm_lon, basinarea, mat_file_path, tif_path):
    """
    Python implementation of the MATLAB get_Qriver_timeseries function.
    Extracts and scales river discharge time series based on basin area.
    
    Parameters:
        rm_lat (float): Grid latitude index
        rm_lon (float): Grid longitude index
        basinarea (float): Basin area in km²
        mat_file_path (str): Path to the .mat file with discharge data
        tif_path (str): Path to the basin area GeoTIFF
        
    Returns:
        tuple: (sed_series, discharge_series, t) - Sediment and discharge time series with time values
    """
    # Adjust rm_lon if it's greater than 180 (MATLAB compatibility)
    if rm_lon > 180:
        rm_lon -= 360
    
    # Load basin area data
    with rasterio.open(tif_path) as src:
        a = src.read(1)
        a[a < 0] = np.nan
        a = np.flipud(a)
    
    # Load discharge data
    with h5py.File(mat_file_path, 'r') as d:
        wbm_lat = np.array(d['rm_lat']).flatten()
        wbm_lon = np.array(d['rm_lon']).flatten()
        lon_grid = np.array(d['lon_grid']).flatten()
        lat_grid = np.array(d['lat_grid']).flatten()
        
        # Get basin areas for all river mouths
        wbm_basinarea = np.zeros(len(wbm_lat))
        for i in range(len(wbm_lat)):
            r, c = int(wbm_lat[i]), int(wbm_lon[i])
            if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
                wbm_basinarea[i] = a[r, c]
            else:
                wbm_basinarea[i] = np.nan
        
        # Calculate complex coordinates for distance calculation
        delta_coor = rm_lat + 1j * rm_lon
        wbm_coor = np.zeros(len(wbm_lat), dtype=complex)
        for i in range(len(wbm_lat)):
            lat_idx = int(wbm_lat[i])
            lon_idx = int(wbm_lon[i])
            if 0 <= lat_idx < len(lat_grid) and 0 <= lon_idx < len(lon_grid):
                wbm_coor[i] = lat_grid[lat_idx] + 1j * lon_grid[lon_idx]
        
        # Calculate blub (distance metric)
        blub = np.abs(wbm_coor - delta_coor)
        
        # Add basin area scaling term
        valid_idx = ~np.isnan(wbm_basinarea)
        scaling_term = np.ones(len(wbm_lat)) * 1000
        if basinarea > 0:
            scaling_term[valid_idx] = np.abs((wbm_basinarea[valid_idx] - basinarea) / basinarea * np.log10(basinarea))
        
        # Add penalty for points far away
        blub = blub + scaling_term + (np.abs(wbm_coor - delta_coor) > 8) * 10
        
        # Find closest match
        idx = np.nanargmin(blub)
        
        # Calculate scaling factor
        fac = wbm_basinarea[idx] / basinarea if basinarea > 0 and wbm_basinarea[idx] > 0 else 1.0
        
        # Get time series data
        sed_series = np.array(d['sed_series'])[idx, :] * fac
        discharge_series = np.array(d['discharge_series'])[idx, :] * fac
        t = np.array(d['t']).flatten()
        
    return sed_series, discharge_series, t

#%%
def calculate_vectorized_daily_means(series_dict, datetimes):
    """
    Calculate daily means using pandas vectorized operations.
    
    Parameters:
        series_dict (dict): Dictionary of time series data {name: values}
        datetimes (list): List of datetime objects
        
    Returns:
        pd.DataFrame: DataFrame with daily means
    """
    # Create DataFrame with all series
    df = pd.DataFrame(series_dict)
    df['datetime'] = datetimes
    
    # Add day of year column
    df['dayofyear'] = pd.DatetimeIndex(df['datetime']).dayofyear
    
    # Group by day of year and calculate mean for all columns at once
    return df.groupby('dayofyear').mean().drop(columns=['datetime'])