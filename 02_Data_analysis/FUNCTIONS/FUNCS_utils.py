"""Utility functions for data analysis in estuary project."""
#%% --- IMPORTS ---
import numpy as np
import h5py
from datetime import datetime, timedelta
import rasterio

#%% Define utility functions

def transform_coordinates(lon, lat, return_indices=False):
    """
    Transforms geographic coordinates to grid coordinates or indices.
    
    Parameters:
        lon (float): Longitude in decimal degrees, range [-180, 180]
        lat (float): Latitude in decimal degrees, range [-90, 90]
        return_indices (bool): If True, return row/col indices; if False, return rm coordinates
        
    Returns:
        tuple: Either (rm_lon, rm_lat) or (row, col) depending on return_indices

    Notes:
        - This is a simple linear transformation to shift and scale geographic coordinates
          into a positive grid system for indexing or mapping purposes.
        - Each increment of 1 in rm_lat or rm_lon corresponds to 0.1 degree in latitude or longitude.
        - The output values are NOT standard geographic coordinates (not degrees, radians, or meters).
        - This is NOT a map projection; it's only for internal grid use.
    """
    # Calculate rm coordinates
    rm_lon_val = (lon + 180) * 10
    rm_lat_val = (lat + 90) * 10
    
    if return_indices:
        # Convert to row/col indices with latitude flipped
        row = 1800 - int(rm_lat_val)  # Convert to row index
        col = int(rm_lon_val)         # Convert to col index
        return row, col
    else:
        return rm_lon_val, rm_lat_val


def matlab_datenum_to_datetime(datenum):
    """
    Converts Matlab datenum into Python datetime
    
    Parameters:
        datenum (float): Matlab datenum value
        
    Returns:
        datetime: Corresponding Python datetime object
    """
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)


def load_data(mat_file_path, tif_path=None):
    """
    Load all necessary data from files.
    
    Parameters:
        mat_file_path (str): Path to the .mat file containing estuary data
        tif_path (str, optional): Path to the TIF file containing basin area data
        
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
    
    # Load basin area data if provided
    if tif_path is not None:
        with rasterio.open(tif_path) as src:
            data_cache['basin_area'] = np.flipud(src.read(1))
            data_cache['basin_area'][data_cache['basin_area'] < 0] = np.nan
    
    # Create datetime objects from MATLAB date numbers
    data_cache['datetimes'] = [matlab_datenum_to_datetime(dn) for dn in data_cache['t']]
    
    print("Data loaded successfully.")
    return data_cache
