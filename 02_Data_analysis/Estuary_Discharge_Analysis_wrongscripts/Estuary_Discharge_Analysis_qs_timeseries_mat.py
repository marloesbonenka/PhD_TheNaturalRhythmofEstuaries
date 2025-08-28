#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title: Estuary Discharge Analysis and Visualization
Description: This script loads estuary data from a .mat file, plots discharge time series,
             and visualizes the global distribution of estuaries using scatterplots and density maps.
Author: Marloes Bonenkamp
Date: April 14, 2025
"""
#%%
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import pandas as pd
import rasterio

from coordinate_transformation import (
    transform_coordinates)

#%% Specify directories of retrieving and storing input and output files
input_dir = r"C:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Data\01_Discharge_var_int_flash"
output_dir = r"C:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Data\01_Discharge_var_int_flash\01_Analysis_smallselection_estuaries"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Do you want to save the plotted figures/dataframes? True / False
savefig = True
savedata = True


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
#%% Load the .mat file
data = h5py.File(os.path.join(input_dir, "qs_timeseries_Nienhuis2020.mat"), "r")

# Print the keys in the data to understand the structure
print("Keys in the MAT file:", data.keys())

discharge_series = np.array(data['discharge_series'])
sed_series = np.array(data['sed_series'])
lat_grid = np.array(data['lat_grid']).flatten()
lon_grid = np.array(data['lon_grid']).flatten()
rm_lat = np.array(data['rm_lat']).flatten()
rm_lon = np.array(data['rm_lon']).flatten()
t = np.array(data['t']).flatten()

print("Data loaded successfully.")

#%% Convert Matlab datenum to Python datetimes

def matlab_datenum_to_datetime(datenum):
    # Converts Matlab datenum into Python datetime
    return datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)

# Use it to convert the array of MATLAB datenums to Python datetimes
python_datetimes = [matlab_datenum_to_datetime(dn) for dn in t]

print("Matlab datenum to Python datetimes conversion successfull.")

#%% Define the (approximate) coordinates of the estuaries

estuary_coords = {
    'Eel': (40.63, -124.31),
    'Klamath': (41.54, -124.08),
    'Cacipore': (3.6, -51.2),
    'Suriname': (5.84, -55.11),
    'Demerara': (6.79, -58.18),
    'Yangon': (16.52, 96.29),
    'Sokyosen': (36.9, 126.9),
    'Wai Bian': (-8.10, 139.97),
    'Thames': (51.5, 0.6)
}

#%% Assess global distribution and density of estuaries/deltas

# Scatterplot: Global distribution of estuaries
plt.figure(figsize=(12, 8))
plt.scatter(rm_lon, rm_lat, alpha=0.5, s=5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Global Distribution of Estuaries')
plt.grid(True, alpha=0.3)
plt.show()

# Density visualization using hexbin
plt.figure(figsize=(12, 8))
hb = plt.hexbin(rm_lon, rm_lat, gridsize=50, cmap='viridis')
cb = plt.colorbar(hb, label='Number of Estuaries')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Density of Estuaries Worldwide (Hexbin)')
plt.show()

# Average discharge per estuary and scatterplot with color-coded discharge values
mean_discharge = np.nanmean(discharge_series, axis=1)  # Average discharge per estuary

# Adjust the colormap based on the distribution of mean discharge values
vmin = np.percentile(mean_discharge, 5)  # 5th percentile
vmax = np.percentile(mean_discharge, 95)  # 95th percentile

plt.figure(figsize=(12, 8))
scatter = plt.scatter(rm_lon, rm_lat, c=mean_discharge, cmap='viridis',
                    alpha=0.7, s=10, edgecolors='none', vmin=vmin, vmax=vmax)
cb = plt.colorbar(scatter, label='Average Discharge')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Global Distribution of Estuaries with Average Discharge Values')
plt.grid(True, alpha=0.3)
plt.show()

# %% Extract information from fluvial sediment load qs [kg/s] on estuaries in which I am interested:

def extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series, sed_series):
    """
    Extracts discharge time series values for given estuary coordinates by
    finding the nearest grid point in rm_lon/rm_lat space.

    Args:
        estuary_coords (dict): Dictionary of estuary coordinates (lat, lon).
        rm_lon (np.ndarray): Flattened rm_lon grid.
        rm_lat (np.ndarray): Flattened rm_lat grid.
        discharge_series (np.ndarray): Discharge time series data (grid x time).

    Returns:
        dict: Dictionary of discharge time series for each estuary and
        dict: Dictionary of rm_lon, rm_lat river mouth coordinates for each estuary.
    """
    estuary_discharge_timeseries = {}
    estuary_sed_timeseries = {}
    estuary_rm_coords = {}  # Store rm_lon, rm_lat for plotting

    for estuary, (lat, lon) in estuary_coords.items():
        # Transform lat/lon to rm_lon/rm_lat
        estuary_rm_lon, estuary_rm_lat = transform_coordinates(lat, lon)

        # Find the index of the nearest grid point
        distances = np.sqrt((rm_lon - estuary_rm_lon)**2 + (rm_lat - estuary_rm_lat)**2)
        nearest_index = np.argmin(distances)

        # Extract the discharge time series for that grid point
        estuary_discharge_timeseries[estuary] = discharge_series[nearest_index, :]
        estuary_sed_timeseries[estuary] = sed_series[nearest_index, :]
        estuary_rm_coords[estuary] = (rm_lon[nearest_index], rm_lat[nearest_index]) #Use the grid values

    return estuary_discharge_timeseries, estuary_rm_coords, estuary_sed_timeseries

# Plotting
if __name__ == "__main__":
    estuary_discharge_data, estuary_rm_coords, estuary_sed_data = extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series, sed_series)
    
    # Calculate and store mean discharge values
    mean_discharge_values = {}
    mean_sediment_values = {} 
    
    # Plot the discharge time series for each estuary
    for estuary, discharge_series in estuary_discharge_data.items():
        sed_series = estuary_sed_data[estuary]  # Get the corresponding sediment data

        # --- Discharge plot ---
        mean_discharge = np.mean(discharge_series)
        mean_discharge_values[estuary] = mean_discharge

        plt.figure(figsize=(10, 6))
        plt.plot(python_datetimes, discharge_series, label=estuary)  # Use the actual time series
        plt.axhline(mean_discharge, linestyle='dashed', color='orange', label=f'mean = {mean_discharge:.2f}')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('River Discharge $Q_{river}$ [m3/s]')
        plt.title(f'River Discharge $Q_{{river}}$ Time Series for {estuary}')  # Title with estuary name
        plt.grid(True, alpha=0.3)
        plt.legend(loc = 'upper right')
        # plt.legend()
        
        if savefig:
            save_path = os.path.join(output_dir, f'01_River_discharge_Qriver_per_estuary')
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'Q_{estuary}.png'), 
                        dpi = 200, bbox_inches = 'tight')
        plt.show()
    
        # --- Sediment plot ---
        mean_sediment = np.mean(sed_series)
        mean_sediment_values[estuary] = mean_sediment

        plt.figure(figsize=(10, 6))
        plt.plot(python_datetimes, sed_series, label='Sediment', color='green')
        plt.axhline(mean_sediment, linestyle='dashed', color='red', label=f'mean = {mean_sediment:.2f}')
        plt.xlabel('Time (timesteps)')
        plt.ylabel('Sediment Load [kg/s]')
        plt.title(f'Sediment Load Time Series for {estuary}')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')

        if savefig:
            save_path_s = os.path.join(output_dir, f'02_Sediment_load_per_estuary')
            os.makedirs(save_path_s, exist_ok=True)
            plt.savefig(os.path.join(save_path_s, f'Sediment_{estuary}.png'), 
                        dpi=200, bbox_inches='tight')
        plt.show()
        
        # --- Combined plot ---
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_discharge = 'tab:blue'
        ax1.set_xlabel('Time (timesteps)')
        ax1.set_ylabel('River Discharge $Q_{river}$ [m³/s]', color=color_discharge)
        l1 = ax1.plot(python_datetimes, discharge_series, label='Discharge', color=color_discharge)
        l2 = ax1.axhline(mean_discharge, linestyle='dashed', color='orange', label=f'Mean Q = {mean_discharge:.2f}')
        ax1.tick_params(axis='y', labelcolor=color_discharge)

        ax2 = ax1.twinx()
        color_sediment = 'tab:green'
        ax2.set_ylabel('Sediment Load [kg/s]', color=color_sediment)
        l3 = ax2.plot(python_datetimes, sed_series, label='Sediment', color=color_sediment)
        l4 = ax2.axhline(mean_sediment, linestyle='dashed', color='red', label=f'Mean S = {mean_sediment:.2f}')
        ax2.tick_params(axis='y', labelcolor=color_sediment)
        
        # Sync y-axis limits based on discharge axis
        ymin, ymax = ax1.get_ylim()
        ax2.set_ylim(ymin, ymax)
        
        plt.title(f'Combined Discharge and Sediment Time Series for {estuary}')
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for top legend

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        fig.legend(lines_1 + lines_2, labels_1 + labels_2,
                loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

        if savefig:
            save_path_combined = os.path.join(output_dir, '03_Combined_Qriver_qs')
            os.makedirs(save_path_combined, exist_ok=True)
            plt.savefig(os.path.join(save_path_combined, f'Combined_{estuary}.png'), dpi=200, bbox_inches='tight')
        plt.show()

    mean_df = pd.DataFrame.from_dict(mean_discharge_values, orient='index', columns = ['Mean River Discharge [m3/s]'])
    mean_df.index.name = 'Estuary'  # Set index name)

    mean_sediment_df = pd.DataFrame.from_dict(mean_sediment_values, orient='index', columns=['Mean Sediment Load [kg/s]'])
    mean_sediment_df.index.name = 'Estuary'

    if savedata:
        mean_df.to_excel(os.path.join(save_path, f'mean_river_discharges.xlsx'))
        mean_sediment_df.to_excel(os.path.join(save_path_s, f'mean_sediment_loads.xlsx'))

    #%% Validate estuary locations

    for estuary, (lat, lon) in estuary_coords.items():
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

        # 1. Original estuary location (from your dictionary)
        ax.plot(lon, lat, 'bo', markersize=8, transform=ccrs.PlateCarree(), label='Original Estuary Location')

        # 2. Nearest grid point used for discharge (from estuary_rm_coords)
        rm_lon_estuary, rm_lat_estuary = estuary_rm_coords[estuary]
        print(rm_lon_estuary, rm_lat_estuary)
        lon_index = (rm_lon_estuary / 10) - 180
        lat_index = (rm_lat_estuary / 10) - 90
        print(lon_index, lat_index)
        ax.plot(lon_index, lat_index, 'ro', markersize=8, transform=ccrs.PlateCarree(), label='Nearest Grid Point (Discharge)')

        ax.set_title(f'Satellite View of {estuary}')
        ax.legend()
        
        if savefig:
            save_path_l = os.path.join(output_dir, f'00_Estuary_location_validation')
            os.makedirs(save_path_l, exist_ok=True)
            plt.savefig(os.path.join(save_path_l, f'Location_check_{estuary}.png'), dpi=200, bbox_inches='tight')
        
        plt.show()

#%% Analysis of river discharge variability, intermittency, flashiness

# Estuary_discharge_data is a dict {estuary: np.array of discharge}
results = []

for estuary, q in estuary_discharge_data.items():
    q = np.array(q)
    mean_q = np.mean(q)
    max_q = np.max(q)
    min_q = np.min(q)
    std_q = np.std(q)
    cv = std_q / mean_q if mean_q != 0 else np.nan

    # Actual zero-flow intermittency
    zero_flow_intermittency = np.sum(q == 0) / len(q)

    # Relative low-flow (below 5th percentile) intermittency
    q5 = np.percentile(q, 5)
    relative_zero_flow_intermittency = np.sum(q < q5) / len(q)

    # Flashiness: P90 / P10
    p90 = np.percentile(q, 90)
    p10 = np.percentile(q, 10)
    flashiness = p90 / p10 if p10 != 0 else np.nan

    results.append({
        'Estuary': estuary,
        'Mean': mean_q,
        'Max': max_q,
        'Min': min_q,
        'Std': std_q,
        'CV': cv,
        'Zero-Flow Intermittency': zero_flow_intermittency,
        'Relative Zero-Flow Intermittency (Q < Q5)': relative_zero_flow_intermittency,
        'P90': p90,
        'P10': p10,
        'Flashiness (P90/P10)': flashiness
    })

# Convert to DataFrame for easy viewing and export
df_metrics = pd.DataFrame(results)

# Save to Excel file
if savedata:
    df_metrics.to_excel(os.path.join(output_dir, 'fluvial_sediment_flux_Qriver_metrics_per_estuary.xlsx'), index=False)
