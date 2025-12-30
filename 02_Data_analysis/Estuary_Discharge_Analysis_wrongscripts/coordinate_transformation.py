#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title: Comprehensive Estuary Discharge Analysis and Visualization
Description: This script contains functions to transform coordinates. 

Author: Marloes Bonenkamp
Date: April 22, 2025
"""
from scipy.spatial import cKDTree
from scipy import ndimage
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#%% 
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
#%% Preliminary functions, not sure if to be used.

def create_spatial_index(points):
    """Create a spatial index for fast nearest neighbor queries."""
    return cKDTree(np.column_stack(points))

def find_nearest_point_kdtree(target_coords, kdtree):
    """
    Find the index of the nearest point using KD-Tree for efficient spatial search.
    
    Parameters:
        target_coords (tuple): (x, y) of target point
        kdtree (cKDTree): Spatial index of reference points
        
    Returns:
        int: Index of nearest point
    """
    distance, index = kdtree.query([target_coords])
    return index[0]

def efficient_basin_area_search(basin_area_matrix, row, col, search_radius=10):
    """
    Efficiently find the nearest non-NaN basin area value using distance transform.
    
    Parameters:
        basin_area_matrix (np.ndarray): Basin area data
        row, col (int): Grid indices
        search_radius (int): Maximum search radius
        
    Returns:
        tuple: (basin_area, row, col) - Basin area value and the grid indices where it was found
    """
    # Check if the exact location has a valid value
    if 0 <= row < basin_area_matrix.shape[0] and 0 <= col < basin_area_matrix.shape[1]:
        if not np.isnan(basin_area_matrix[row, col]):
            return basin_area_matrix[row, col], row, col
    
    # Create a mask of valid values
    mask = ~np.isnan(basin_area_matrix)
    
    # Create a small window around the point to search
    window_rows = slice(max(0, row - search_radius), min(basin_area_matrix.shape[0], row + search_radius + 1))
    window_cols = slice(max(0, col - search_radius), min(basin_area_matrix.shape[1], col + search_radius + 1))
    
    window = basin_area_matrix[window_rows, window_cols]
    window_mask = mask[window_rows, window_cols]
    
    if not np.any(window_mask):
        return np.nan, row, col
    
    # Calculate distance to nearest valid point in window
    distance_map = ndimage.distance_transform_edt(~window_mask)
    
    # Find the closest valid point
    local_row, local_col = np.unravel_index(np.argmin(distance_map), window.shape)
    
    # Convert back to global indices
    global_row = local_row + max(0, row - search_radius)
    global_col = local_col + max(0, col - search_radius)
    
    return basin_area_matrix[global_row, global_col], global_row, global_col