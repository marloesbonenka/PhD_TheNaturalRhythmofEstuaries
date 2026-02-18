
# -*- coding: utf-8 -*-
"""Functions for plotting discharge time series for estuaries based on WBMsed data."""
#%% --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys

# Add the working directory here FUNCTIONS is located
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\02_Data_analysis")

from FUNCTIONS.FUNCS_utils import transform_coordinates

#%% 
def plot_global_delta_distribution(rm_lon, rm_lat, mean_discharge=None, savefig=False, output_dir=None):
    """
    Plot global distribution of deltas using different visualization methods.
    
    Parameters:
        rm_lon (np.ndarray): Array of longitude coordinates
        rm_lat (np.ndarray): Array of latitude coordinates
        mean_discharge (np.ndarray, optional): Mean discharge values for color coding
        savefig (bool): Whether to save figures
        output_dir (str, optional): Directory to save figures
    """
    outdir = Path(output_dir) if output_dir else None
    # Simple scatterplot
    plt.figure(figsize=(12, 8))
    plt.scatter(rm_lon, rm_lat, alpha=0.5, s=5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Global Distribution of Deltas based on Nienhuis et al. 2020')
    plt.grid(True, alpha=0.3)
    if savefig and outdir:
        plt.savefig(outdir / 'global_delta_distribution.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Density visualization using hexbin
    plt.figure(figsize=(12, 8))
    hb = plt.hexbin(rm_lon, rm_lat, gridsize=50, cmap='viridis')
    cb = plt.colorbar(hb, label='Number of Deltas')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Density of Deltas Worldwide based on Nienhuis et al. 2020')
    if savefig and outdir:
        plt.savefig(outdir / 'delta_density_hexbin.png', dpi=200, bbox_inches='tight')
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
        plt.title('Global Distribution of Deltas with Average Discharge Values based on Nienhuis et al. 2020')
        plt.grid(True, alpha=0.3)
        if savefig and outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / 'delta_discharge_distribution.png', dpi=200, bbox_inches='tight')
        plt.show()

def extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series, sed_series, search_radius=0.2):
    """
    Extracts data by finding the MAX discharge pixel within a small radius.
    This prevents 'missing' the river channel by a few kilometers.
    """
    estuary_discharge_timeseries = {}
    estuary_sed_timeseries = {}
    estuary_rm_coords = {}
    
    for estuary, (lat, lon) in estuary_coords.items():
        # 1. Target rm coordinates
        target_rm_lon, target_rm_lat = transform_coordinates(lon, lat)

        # 2. Calculate distances to ALL 18,680 deltas
        distances = np.sqrt((rm_lon - target_rm_lon)**2 + (rm_lat - target_rm_lat)**2)
        
        # 3. USE THE SEARCH RADIUS: Filter for points within the radius (e.g., 0.2 units)
        # Note: 1 unit in 'rm' space = 0.1 degrees
        within_radius_indices = np.where(distances <= (search_radius * 10))[0]
        
        # Fallback: If nothing is in radius, just take the single nearest point
        if len(within_radius_indices) == 0:
            best_index = np.argmin(distances)
        else:
            # 4. Distance-Weighted Selection among candidates within radius
            # We want high discharge, but we penalize points further from our target
            candidate_means = np.mean(discharge_series[within_radius_indices, :], axis=1)
            candidate_dists = distances[within_radius_indices]
            
            # Score = Discharge / (Distance + small_offset)
            # Higher score is better. 0.01 prevents division by zero.
            scores = candidate_means / (candidate_dists + 0.01)
            best_index = within_radius_indices[np.argmax(scores)]

        # 5. Extract and Clean
        q_data = np.maximum(np.nan_to_num(discharge_series[best_index, :]), 0)
        s_data = np.maximum(np.nan_to_num(sed_series[best_index, :]), 0)

        estuary_discharge_timeseries[estuary] = q_data
        estuary_sed_timeseries[estuary] = s_data
        estuary_rm_coords[estuary] = (rm_lon[best_index], rm_lat[best_index])
        
    return estuary_discharge_timeseries, estuary_rm_coords, estuary_sed_timeseries

def validate_estuary_location(estuary_name, original_coords, grid_coords, savefig=False, output_dir=None):
    """
    Validate estuary locations by plotting both original coordinates and extracted grid point.
    
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

    # Plot original location and extracted grid point
    ax.plot(lon, lat, 'bo', markersize=8, transform=ccrs.PlateCarree(), label='Input coordinates')
    ax.plot(lon_grid, lat_grid, 'ro', markersize=8, transform=ccrs.PlateCarree(), label='Extracted WBMsed grid point')

    ax.set_title(f'{estuary_name}')
    ax.legend()
    
    outdir = Path(output_dir) if output_dir else None
    if savefig and outdir:
        save_path_l = outdir / '00_Estuary_location_validation'
        save_path_l.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_l / f'Location_check_{estuary_name}.png', dpi=200, bbox_inches='tight')
    
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
    plt.title(f'$Q_{{river}}$ for {estuary_name} based on WBMsed')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    outdir = Path(output_dir) if output_dir else None
    if savefig and outdir:
        save_path = outdir / '01_River_discharge_Qriver_per_estuary'
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / f'Q_{estuary_name}.png', dpi=200, bbox_inches='tight')
    plt.show()

    # --- Sediment plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, sed_series, label='Sediment', color='green')
    plt.axhline(mean_sediment, linestyle='dashed', color='red', label=f'mean = {mean_sediment:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Sediment Load [kg/s]')
    plt.title(f'Sediment load for {estuary_name} based on WBMsed + BQART')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    if savefig and outdir:
        save_path_s = outdir / '02_Sediment_load_per_estuary'
        save_path_s.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_s / f'Sediment_{estuary_name}.png', dpi=200, bbox_inches='tight')
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
    
    # Sync y-axis limits based on largest axis
    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()

    global_ymin = min(ymin1, ymin2)
    global_ymax = max(ymax1, ymax2)

    ax1.set_ylim(global_ymin, global_ymax)
    ax2.set_ylim(global_ymin, global_ymax)

    plt.title(f'Combined discharge and sediment load for {estuary_name} based on WBMsed + BQART')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2,
              loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

    if savefig and outdir:
        save_path_combined = outdir / '03_Combined_Qriver_qs'
        save_path_combined.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_combined / f'Combined_{estuary_name}.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # --- 4. Moving Average Plot ---
    # 365-day window for the annual trend
    q_df = pd.Series(discharge_series, index=datetimes)
    moving_avg = q_df.rolling(window=365, center=True).mean()

    plt.figure(figsize=(10, 6))
    # Background: actual values in light grey
    plt.plot(datetimes, discharge_series, color='lightgrey', alpha=0.5, label='Daily Discharge')
    # Foreground: moving average in bold blue
    plt.plot(datetimes, moving_avg, color='tab:blue', linewidth=2, label='365-day Moving Average')
    plt.axhline(mean_discharge, linestyle='dashed', color='orange', label=f'mean = {mean_discharge:.2f}')
    
    plt.xlabel('Time')
    plt.ylabel('River Discharge $Q_{river}$ [m3/s]')
    plt.title(f'Long-term Trend ($Q_{{river}}$) for {estuary_name}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    if savefig and outdir:
        save_path_ma = outdir / '05_Moving_Average_Discharge'
        save_path_ma.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_ma / f'Trend_{estuary_name}.png', dpi=200, bbox_inches='tight')
    plt.show()

    return mean_discharge, mean_sediment, moving_avg
# %%
