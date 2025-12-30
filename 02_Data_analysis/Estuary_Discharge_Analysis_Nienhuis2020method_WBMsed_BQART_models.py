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
import seaborn as sns

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


def extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series, sed_series):
    """
    Extracts discharge time series values for given estuary coordinates by
    finding the nearest grid point in rm_lon/rm_lat space.

    Args:
        estuary_coords (dict): Dictionary of estuary coordinates (lat, lon).
        rm_lon (np.ndarray): Flattened rm_lon grid.
        rm_lat (np.ndarray): Flattened rm_lat grid.
        discharge_series (np.ndarray): Discharge time series data (grid x time).
        sed_series (np.ndarray): Sediment time series data (grid x time).

    Returns:
        tuple: Dictionaries of (discharge time series, rm coordinates, sediment time series) for each estuary
    """
    estuary_discharge_timeseries = {}
    estuary_sed_timeseries = {}
    estuary_rm_coords = {}  # Store rm_lon, rm_lat for plotting

    for estuary, (lat, lon) in estuary_coords.items():
        # Transform lat/lon to rm_lon/rm_lat
        estuary_rm_lon, estuary_rm_lat = transform_coordinates(lon, lat)

        # Find the index of the nearest grid point
        distances = np.sqrt((rm_lon - estuary_rm_lon)**2 + (rm_lat - estuary_rm_lat)**2)
        nearest_index = np.argmin(distances)

        # Extract the discharge time series for that grid point
        estuary_discharge_timeseries[estuary] = discharge_series[nearest_index, :]
        estuary_sed_timeseries[estuary] = sed_series[nearest_index, :]
        estuary_rm_coords[estuary] = (rm_lon[nearest_index], rm_lat[nearest_index])

    return estuary_discharge_timeseries, estuary_rm_coords, estuary_sed_timeseries


def plot_global_estuary_distribution(rm_lon, rm_lat, mean_discharge=None, savefig=False, output_dir=None):
    """
    Plot global distribution of estuaries using different visualization methods.
    
    Parameters:
        rm_lon (np.ndarray): Array of longitude coordinates
        rm_lat (np.ndarray): Array of latitude coordinates
        mean_discharge (np.ndarray, optional): Mean discharge values for color coding
        savefig (bool): Whether to save figures
        output_dir (str, optional): Directory to save figures
    """
    # Simple scatterplot
    plt.figure(figsize=(12, 8))
    plt.scatter(rm_lon, rm_lat, alpha=0.5, s=5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Global Distribution of Deltas/Estuaries based on Nienhuis et al. 2020')
    plt.grid(True, alpha=0.3)
    if savefig and output_dir:
        plt.savefig(os.path.join(output_dir, 'global_estuary_distribution.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # Density visualization using hexbin
    plt.figure(figsize=(12, 8))
    hb = plt.hexbin(rm_lon, rm_lat, gridsize=50, cmap='viridis')
    cb = plt.colorbar(hb, label='Number of Estuaries')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Density of Deltas/Estuaries Worldwide based on Nienhuis et al. 2020')
    if savefig and output_dir:
        plt.savefig(os.path.join(output_dir, 'estuary_density_hexbin.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # If mean discharge is provided, create a color-coded scatter plot
    if mean_discharge is not None:
        vmin = np.percentile(mean_discharge, 5)  # 5th percentile
        vmax = np.percentile(mean_discharge, 95)  # 95th percentile

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(rm_lon, rm_lat, c=mean_discharge, cmap='viridis',
                            alpha=0.7, s=10, edgecolors='none', vmin=vmin, vmax=vmax)
        cb = plt.colorbar(scatter, label='Average Discharge')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Global Distribution of Estuaries with Average Discharge Values based on Nienhuis et al. 2020')
        plt.grid(True, alpha=0.3)
        if savefig and output_dir:
            plt.savefig(os.path.join(output_dir, 'estuary_discharge_distribution.png'), dpi=200, bbox_inches='tight')
        plt.show()


def plot_estuary_timeseries(estuary_name, discharge_series, sed_series, datetimes, savefig=False, output_dir=None):
    """
    Generate time series plots for a specific estuary's discharge and sediment data.
    
    Parameters:
        estuary_name (str): Name of the estuary
        discharge_series (np.ndarray): Discharge time series data
        sed_series (np.ndarray): Sediment time series data
        datetimes (list): List of datetime objects
        savefig (bool): Whether to save figures
        output_dir (str, optional): Base directory to save figures
    """
    # Calculate mean values
    mean_discharge = np.mean(discharge_series)
    mean_sediment = np.mean(sed_series)
    
    # --- Discharge plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, discharge_series, label=estuary_name)
    plt.axhline(mean_discharge, linestyle='dashed', color='orange', label=f'mean = {mean_discharge:.2f}')
    plt.xlabel('Time')
    plt.ylabel('River Discharge $Q_{river}$ [m3/s]')
    plt.title(f'$Q_{{river}}$ Time Series for {estuary_name} based on WBMsed')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    if savefig and output_dir:
        save_path = os.path.join(output_dir, '01_River_discharge_Qriver_per_estuary')
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f'Q_{estuary_name}.png'), dpi=200, bbox_inches='tight')
    plt.show()

    # --- Sediment plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, sed_series, label='Sediment', color='green')
    plt.axhline(mean_sediment, linestyle='dashed', color='red', label=f'mean = {mean_sediment:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Sediment Load [kg/s]')
    plt.title(f'Sediment Discharge Time Series for {estuary_name} based on WBMsed + BQART')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    if savefig and output_dir:
        save_path_s = os.path.join(output_dir, '02_Sediment_load_per_estuary')
        os.makedirs(save_path_s, exist_ok=True)
        plt.savefig(os.path.join(save_path_s, f'Sediment_{estuary_name}.png'), dpi=200, bbox_inches='tight')
    plt.show()
    
    # --- Combined plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_discharge = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('River Discharge $Q_{river}$ [m³/s]', color=color_discharge)
    l1 = ax1.plot(datetimes, discharge_series, label='Discharge', color=color_discharge)
    l2 = ax1.axhline(mean_discharge, linestyle='dashed', color='orange', label=f'Mean Q = {mean_discharge:.2f}')
    ax1.tick_params(axis='y', labelcolor=color_discharge)

    ax2 = ax1.twinx()
    color_sediment = 'tab:green'
    ax2.set_ylabel('Sediment Load [kg/s]', color=color_sediment)
    l3 = ax2.plot(datetimes, sed_series, label='Sediment', color=color_sediment)
    l4 = ax2.axhline(mean_sediment, linestyle='dashed', color='red', label=f'Mean S = {mean_sediment:.2f}')
    ax2.tick_params(axis='y', labelcolor=color_sediment)
    
    # Sync y-axis limits based on discharge axis
    ymin, ymax = ax1.get_ylim()
    ax2.set_ylim(ymin, ymax)
    
    plt.title(f'Combined Discharge and Sediment Time Series for {estuary_name} based on WBMsed + BQART')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2,
              loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

    if savefig and output_dir:
        save_path_combined = os.path.join(output_dir, '03_Combined_Qriver_qs')
        os.makedirs(save_path_combined, exist_ok=True)
        plt.savefig(os.path.join(save_path_combined, f'Combined_{estuary_name}.png'), dpi=200, bbox_inches='tight')
    plt.show()
    
    return mean_discharge, mean_sediment


def validate_estuary_location(estuary_name, original_coords, grid_coords, savefig=False, output_dir=None):
    """
    Validate estuary locations by plotting both original coordinates and nearest grid point.
    
    Parameters:
        estuary_name (str): Name of the estuary
        original_coords (tuple): Original (lat, lon) coordinates
        grid_coords (tuple): Grid coordinates (rm_lon, rm_lat)
        savefig (bool): Whether to save figures
        output_dir (str, optional): Base directory to save figures
    """
    lat, lon = original_coords
    rm_lon, rm_lat = grid_coords
    
    # Convert rm coordinates back to lat/lon
    lon_grid = (rm_lon / 10) - 180
    lat_grid = (rm_lat / 10) - 90
    
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

    # Plot original location and nearest grid point
    ax.plot(lon, lat, 'bo', markersize=8, transform=ccrs.PlateCarree(), label='Original Estuary Location (Nienhuis 2017)')
    ax.plot(lon_grid, lat_grid, 'ro', markersize=8, transform=ccrs.PlateCarree(), label='Nearest Grid Point (Discharge WBMsed, Nienhuis 2020)')

    ax.set_title(f'Satellite View of {estuary_name}')
    ax.legend()
    
    if savefig and output_dir:
        save_path_l = os.path.join(output_dir, '00_Estuary_location_validation')
        os.makedirs(save_path_l, exist_ok=True)
        plt.savefig(os.path.join(save_path_l, f'Location_check_{estuary_name}.png'), dpi=200, bbox_inches='tight')
    
    plt.show()


def analyze_discharge_metrics(estuary_discharge_data):
    """
    Calculate discharge variability, intermittency, and flashiness metrics for estuaries.
    
    Parameters:
        estuary_discharge_data (dict): Dictionary of estuary discharge time series
        
    Returns:
        pd.DataFrame: DataFrame containing calculated metrics
    """
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
    
    return pd.DataFrame(results)

def visualize_discharge_metrics(df, output_dir="04_Metrics_per_estuary"):
    """
    Visualizes discharge metrics using bar and scatter plots with improved formatting.
    Ensures all axes start at zero, clearly identifies estuaries, and has proper spacing
    to prevent text from being cut off.

    Args:
        df (pd.DataFrame): DataFrame containing estuary discharge metrics.
        output_dir (str): Directory to save the plots.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a custom color palette for consistent estuary colors across plots
    num_estuaries = len(df)
    colors = plt.cm.tab10(np.linspace(0, 1, num_estuaries))
    estuary_colors = dict(zip(df['Estuary'], colors))
    
    # 1. Bar Chart for Mean Discharge
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Estuary'], df['Mean'], color=[estuary_colors[e] for e in df['Estuary']])
    plt.xlabel('Estuary', fontsize=12)
    plt.ylabel('Mean Discharge (m³/s)', fontsize=12)
    plt.title('Mean Discharge by Estuary', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Add more space at the bottom and top of the plot
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(os.path.join(output_dir, 'mean_discharge_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Scatter Plot: Mean vs. CV with estuary labels
    plt.figure(figsize=(12, 10))
    for estuary in df['Estuary']:
        estuary_data = df[df['Estuary'] == estuary]
        plt.scatter(estuary_data['Mean'], estuary_data['CV'], 
                   color=estuary_colors[estuary], s=100, label=estuary)
        
        # Adjust text positions to avoid overlap
        x_pos = estuary_data['Mean'].values[0]
        y_pos = estuary_data['CV'].values[0]
        # Add some offset for text to avoid overlapping with points
        plt.annotate(estuary, (x_pos, y_pos),
                    xytext=(7, 7), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.xlabel('Mean Discharge (m³/s)', fontsize=12)
    plt.ylabel('Coefficient of Variation (CV)', fontsize=12)
    plt.title('Mean Discharge vs. Coefficient of Variation', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)  # Ensure x-axis starts at zero
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add a legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.75)  # Make room for the legend
    
    plt.savefig(os.path.join(output_dir, 'mean_vs_cv_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Bar Chart for CV
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Estuary'], df['CV'], color=[estuary_colors[e] for e in df['Estuary']])
    plt.xlabel('Estuary', fontsize=12)
    plt.ylabel('Coefficient of Variation (CV)', fontsize=12)
    plt.title('Coefficient of Variation by Estuary', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(os.path.join(output_dir, 'cv_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Bar Chart for Flashiness (P90/P10)
    plt.figure(figsize=(14, 8))
    bars = plt.bar(df['Estuary'], df['Flashiness (P90/P10)'], 
                  color=[estuary_colors[e] for e in df['Estuary']])
    plt.xlabel('Estuary', fontsize=12)
    plt.ylabel('Flashiness (P90/P10)', fontsize=12)
    plt.title('Flashiness by Estuary', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.subplots_adjust(bottom=0.2, top=0.9)
    plt.savefig(os.path.join(output_dir, 'flashiness_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter Plot: Mean vs. Flashiness with estuary labels
    plt.figure(figsize=(12, 10))
    for estuary in df['Estuary']:
        estuary_data = df[df['Estuary'] == estuary]
        plt.scatter(estuary_data['Mean'], estuary_data['Flashiness (P90/P10)'], 
                   color=estuary_colors[estuary], s=100, label=estuary)
        
        # Adjust text positions to avoid overlap
        x_pos = estuary_data['Mean'].values[0]
        y_pos = estuary_data['Flashiness (P90/P10)'].values[0]
        plt.annotate(estuary, (x_pos, y_pos),
                    xytext=(7, 7), textcoords='offset points', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    plt.xlabel('Mean Discharge (m³/s)', fontsize=12)
    plt.ylabel('Flashiness (P90/P10)', fontsize=12)
    plt.title('Mean Discharge vs. Flashiness', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(left=0)  # Ensure x-axis starts at zero
    plt.ylim(bottom=0)  # Ensure y-axis starts at zero
    
    # Add a legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.75)
    
    plt.savefig(os.path.join(output_dir, 'mean_vs_flashiness_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation heatmap between metrics
    metrics_for_corr = ['Mean', 'Max', 'Min', 'Std', 'CV', 
                        'Zero-Flow Intermittency', 
                        'Relative Zero-Flow Intermittency (Q < Q5)', 
                        'Flashiness (P90/P10)']
    
    corr_matrix = df[metrics_for_corr].corr()
    
    plt.figure(figsize=(14, 12))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                         linewidths=0.5, fmt='.2f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Between Discharge Metrics', fontsize=14)
    
    # Ensure heatmap labels are visible
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Adjust layout for heatmap
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

#%% Main execution function
def main():
    # Configuration variables - adjust these for each run
    INPUT_DIR = r"C:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Data\01_Discharge_var_int_flash"
    OUTPUT_DIR = r"C:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Data\01_Discharge_var_int_flash\01_Analysis_smallselection_estuaries"
    MAT_FILE = "qs_timeseries_Nienhuis2020.mat"     # Path relative to INPUT_DIR
    TIF_FILE = None         # Set to path if basin area TIF file is needed, relative to INPUT_DIR
    
    # If you want to save the plotted figures/dataframes
    SAVE_FIG = True
    SAVE_DATA = True
    
    # Define estuary coordinates (lat, lon)
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
    
    # Create directories if they don't exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    mat_file_path = os.path.join(INPUT_DIR, MAT_FILE)
    tif_path = os.path.join(INPUT_DIR, TIF_FILE) if TIF_FILE else None
    data = load_data(mat_file_path, tif_path)
    
    # Extract variables from data
    rm_lat = data['rm_lat']
    rm_lon = data['rm_lon']
    discharge_series = data['discharge_series']
    sed_series = data['sed_series']
    datetimes = data['datetimes']
    
    # Plot global estuary distribution
    mean_discharge = np.nanmean(discharge_series, axis=1)  # Average discharge per estuary
    plot_global_estuary_distribution(rm_lon, rm_lat, mean_discharge, SAVE_FIG, OUTPUT_DIR)
    
    # Extract data for selected estuaries
    estuary_discharge_data, estuary_rm_coords, estuary_sed_data = extract_discharge_timeseries(
        estuary_coords, rm_lon, rm_lat, discharge_series, sed_series)
    
    # Dictionaries to store mean values
    mean_discharge_values = {}
    mean_sediment_values = {}
    
    # Process each estuary
    for estuary in estuary_coords:
        # Plot time series
        mean_discharge, mean_sediment = plot_estuary_timeseries(
            estuary, 
            estuary_discharge_data[estuary], 
            estuary_sed_data[estuary], 
            datetimes, 
            SAVE_FIG, 
            OUTPUT_DIR
        )
        
        # Store mean values
        mean_discharge_values[estuary] = mean_discharge
        mean_sediment_values[estuary] = mean_sediment
        
        # Validate location
        validate_estuary_location(
            estuary, 
            estuary_coords[estuary], 
            estuary_rm_coords[estuary], 
            SAVE_FIG, 
            OUTPUT_DIR
        )
    
    # Create DataFrames with mean values
    mean_df = pd.DataFrame.from_dict(mean_discharge_values, orient='index', columns=['Mean River Discharge [m3/s]'])
    mean_df.index.name = 'Estuary'
    
    mean_sediment_df = pd.DataFrame.from_dict(mean_sediment_values, orient='index', columns=['Mean Sediment Load [kg/s]'])
    mean_sediment_df.index.name = 'Estuary'
    
    # Save mean values to Excel
    if SAVE_DATA:
        discharge_path = os.path.join(OUTPUT_DIR, '01_River_discharge_Qriver_per_estuary')
        sediment_path = os.path.join(OUTPUT_DIR, '02_Sediment_load_per_estuary')
        os.makedirs(discharge_path, exist_ok=True)
        os.makedirs(sediment_path, exist_ok=True)
        
        mean_df.to_excel(os.path.join(discharge_path, 'mean_river_discharges.xlsx'))
        mean_sediment_df.to_excel(os.path.join(sediment_path, 'mean_sediment_loads.xlsx'))
    
    # Analyze discharge metrics
    df_metrics = analyze_discharge_metrics(estuary_discharge_data)
    
    # Visualize discharge metrics and save to a dedicated directory
    metrics_output_dir = os.path.join(OUTPUT_DIR, "04_Metrics_per_estuary")
    visualize_discharge_metrics(df_metrics, metrics_output_dir)

    # Save metrics to Excel
    if SAVE_DATA:
        df_metrics.to_excel(os.path.join(OUTPUT_DIR, 'fluvial_sediment_flux_Qriver_metrics_per_estuary.xlsx'), index=False)
        
    return df_metrics, mean_df, mean_sediment_df


#%% Execute script if run directly
if __name__ == "__main__":
    df_metrics, mean_discharge_df, mean_sediment_df = main()
    
    # Display summary
    print("\nAnalysis complete. Summary of metrics:")
    print(df_metrics[['Estuary', 'Mean', 'CV', 'Flashiness (P90/P10)']].to_string(index=False))