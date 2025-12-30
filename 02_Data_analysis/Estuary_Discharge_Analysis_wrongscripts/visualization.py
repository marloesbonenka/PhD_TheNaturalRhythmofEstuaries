#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title: Comprehensive Estuary Discharge Analysis and Visualization
Description: This script contains functions to visualize estuary data.

Author: Marloes Bonenkamp
Date: April 22, 2025
"""
#%%

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from coordinate_transformation import (
    transform_coordinates, 
    create_spatial_index, 
    find_nearest_point_kdtree,
    efficient_basin_area_search
)
#%%
def plot_timeseries(estuaries, timeseries_dict, datetimes, output_dir, prefix, ylabel, title_prefix, savefig):
    """
    Plot time series for a list of estuaries.
    
    Parameters:
        estuaries (list): List of estuary names
        timeseries_dict (dict): Dictionary of time series data {estuary_name: timeseries}
        datetimes (list): List of datetime objects
        output_dir (str): Directory to save figures
        prefix (str): Prefix for saved filenames
        ylabel (str): Y-axis label
        title_prefix (str): Prefix for plot titles
        savefig (bool): Whether to save figures
    """
    for estuary in estuaries:
        if estuary not in timeseries_dict:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(datetimes, timeseries_dict[estuary], label=estuary)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.title(f'{title_prefix} Time Series for {estuary}')
        plt.grid(True, alpha=0.3)
        
        if savefig:
            save_dir = os.path.join(output_dir, f'{prefix}_per_estuary')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, f'{prefix}_{estuary}.png'),
                dpi=200, bbox_inches='tight'
            )
        
        plt.close()

def plot_estuary_locations(estuary_coords, estuary_rm_coords, output_dir, savefig):
    """
    Plot satellite view of estuary locations with original and matched coordinates.
    
    Parameters:
        estuary_coords (dict): Dictionary of original coordinates {name: (lat, lon)}
        estuary_rm_coords (dict): Dictionary of matched coordinates {name: (rm_lon, rm_lat)}
        output_dir (str): Directory to save figures
        savefig (bool): Whether to save figures
    """
    for estuary, (lat, lon) in estuary_coords.items():
        if estuary not in estuary_rm_coords:
            continue
            
        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Set a zoom level
        extent = [lon - 5, lon + 5, lat - 5, lat + 5]
        ax.set_extent(extent)

        # Add natural earth features for context
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.OCEAN)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.RIVERS)
        ax.stock_img()

        # Original estuary location
        ax.plot(lon, lat, 'bo', markersize=8, transform=ccrs.PlateCarree(), label='Original Location')

        # Nearest grid point
        rm_lon_estuary, rm_lat_estuary = estuary_rm_coords[estuary]
        lon_index = (rm_lon_estuary / 10) - 180
        lat_index = (rm_lat_estuary / 10) - 90
        ax.plot(lon_index, lat_index, 'ro', markersize=8, transform=ccrs.PlateCarree(), label='Nearest Grid Point')

        ax.set_title(f'Satellite View of {estuary}')
        ax.legend()
        
        if savefig:
            save_dir = os.path.join(output_dir, 'estuary_location_validation')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{estuary}.png'), dpi=200, bbox_inches='tight')
        
        plt.close()

def create_world_map_of_estuaries(rm_lon, rm_lat, mean_discharge=None, output_dir=None, savefig=True):
    """
    Create a world map showing estuary locations, optionally with discharge values.
    
    Parameters:
        rm_lon (np.ndarray): Longitude grid indices
        rm_lat (np.ndarray): Latitude grid indices
        mean_discharge (np.ndarray, optional): Mean discharge values for coloring
        output_dir (str, optional): Directory to save figures
        savefig (bool): Whether to save figures
    """
    # Convert grid indices to geographic coordinates
    lon_coords = (rm_lon / 10) - 180
    lat_coords = (rm_lat / 10) - 90
    
    # Create figure with map projection
    plt.figure(figsize=(14, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Add base map features
    ax.add_feature(cfeature.LAND, alpha=0.5)
    ax.add_feature(cfeature.OCEAN, alpha=0.5)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Plot estuary locations
    if mean_discharge is not None:
        # Color points by discharge
        vmin = np.percentile(mean_discharge, 5)
        vmax = np.percentile(mean_discharge, 95)
        sc = ax.scatter(lon_coords, lat_coords, c=mean_discharge, 
                     transform=ccrs.PlateCarree(), 
                     cmap='viridis', s=15, alpha=0.7,
                     vmin=vmin, vmax=vmax)
        plt.colorbar(sc, label='Mean Discharge', shrink=0.6)
        plt.title('Global Distribution of Estuaries with Mean Discharge')
    else:
        # Simple point plot
        ax.scatter(lon_coords, lat_coords, 
                transform=ccrs.PlateCarree(),
                color='blue', s=10, alpha=0.7)
        plt.title('Global Distribution of Estuaries')
    
    if savefig and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        title = 'global_estuary_map_with_discharge.png' if mean_discharge is not None else 'global_estuary_map.png'
        plt.savefig(os.path.join(output_dir, title), dpi=200, bbox_inches='tight')
    
    plt.close()