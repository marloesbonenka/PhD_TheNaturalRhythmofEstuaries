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

    plt.figure(figsize=(12, 8))
    plt.scatter(rm_lon, rm_lat, alpha=0.5, s=5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Global Distribution of Deltas based on Nienhuis et al. 2020')
    plt.grid(True, alpha=0.3)
    if savefig and outdir:
        plt.savefig(outdir / 'global_delta_distribution.png', dpi=200, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(12, 8))
    hb = plt.hexbin(rm_lon, rm_lat, gridsize=50, cmap='viridis')
    plt.colorbar(hb, label='Number of Deltas')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Density of Deltas Worldwide based on Nienhuis et al. 2020')
    if savefig and outdir:
        plt.savefig(outdir / 'delta_density_hexbin.png', dpi=200, bbox_inches='tight')
    plt.show()

    if mean_discharge is not None:
        vmin = np.percentile(mean_discharge, 5)
        vmax = np.percentile(mean_discharge, 95)
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(rm_lon, rm_lat, c=mean_discharge, cmap='viridis',
                              alpha=0.7, s=10, edgecolors='none', vmin=vmin, vmax=vmax)
        plt.colorbar(scatter, label='Average Discharge')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Global Distribution of Deltas with Average Discharge Values')
        plt.grid(True, alpha=0.3)
        if savefig and outdir:
            outdir.mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir / 'delta_discharge_distribution.png', dpi=200, bbox_inches='tight')
        plt.show()


def extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series, sed_series,
                                  gd_rm_lon=None, gd_rm_lat=None, gd_basin_area=None,
                                  search_radius=0.2):
    """
    Extracts discharge and sediment time series for each estuary using the same
    logic as the original MATLAB get_Qriver_timeseries function:

      1. Find the best-matching WBMsed grid point using a cost function that
         combines geographic distance AND basin area similarity (if available).
      2. Compute a scaling factor: fac = WBMsed_basin_area / estuary_basin_area
      3. Scale the extracted discharge and sediment series by fac.

    Parameters:
        estuary_coords (dict): {name: (lat, lon, basin_area_km2)} 
                               basin_area_km2 is optional; if not provided, no scaling is applied.
                               Accepts both (lat, lon) and (lat, lon, basin_area) tuples.
        rm_lon (np.ndarray): WBMsed grid longitude coordinates (rm space)
        rm_lat (np.ndarray): WBMsed grid latitude coordinates (rm space)
        discharge_series (np.ndarray): Discharge array (n_points x n_timesteps)
        sed_series (np.ndarray): Sediment array (n_points x n_timesteps)
        gd_rm_lon (np.ndarray, optional): GlobalDeltaData mouth longitudes in rm space
        gd_rm_lat (np.ndarray, optional): GlobalDeltaData mouth latitudes in rm space
        gd_basin_area (np.ndarray, optional): GlobalDeltaData basin areas in km²
        search_radius (float): Search radius in rm units (1 unit = 0.1 degree)

    Returns:
        tuple: (estuary_discharge_timeseries, estuary_rm_coords, estuary_sed_timeseries, estuary_scaling_factors)
    """
    estuary_discharge_timeseries = {}
    estuary_sed_timeseries       = {}
    estuary_rm_coords            = {}
    estuary_scaling_factors      = {}

    has_global_delta = (gd_rm_lon is not None and 
                        gd_rm_lat is not None and 
                        gd_basin_area is not None)

    for estuary, coords in estuary_coords.items():
        # Unpack coords — support both (lat, lon) and (lat, lon, basin_area)
        if len(coords) == 3:
            lat, lon, estuary_basin_area = coords
        else:
            lat, lon = coords
            estuary_basin_area = None

        # --- Step 1: Convert estuary coords to rm space ---
        target_rm_lon, target_rm_lat = transform_coordinates(lon, lat)

        # --- Step 2: Find basin area for this estuary from GlobalDeltaData ---
        # Look up the nearest GlobalDeltaData entry to get its basin area,
        # which we use to find the best-matching WBMsed grid point.
        if estuary_basin_area is None and has_global_delta:
            gd_dist = np.sqrt((gd_rm_lon - target_rm_lon)**2 + 
                              (gd_rm_lat - target_rm_lat)**2)
            nearest_gd = np.argmin(gd_dist)
            if gd_dist[nearest_gd] < 10:  # within 1 degree
                estuary_basin_area = gd_basin_area[nearest_gd]
                print(f"  {estuary}: basin area looked up from GlobalDeltaData = "
                      f"{estuary_basin_area:.1f} km² (dist={gd_dist[nearest_gd]:.2f} rm units)")
            else:
                print(f"  {estuary}: WARNING - no GlobalDeltaData match within 1 degree. "
                      f"No basin area scaling applied.")

        # --- Step 3: Find best WBMsed grid point ---
        # Geographic distance in rm space
        geo_dist = np.sqrt((rm_lon - target_rm_lon)**2 + (rm_lat - target_rm_lat)**2)

        if estuary_basin_area is not None and estuary_basin_area > 0:
            # Replicate MATLAB cost function:
            # cost = geo_dist 
            #      + |( wbm_basin_area - target_basin_area ) / target_basin_area| * log10(target_basin_area)
            #      + 10 * (geo_dist > 8)   <- hard penalty for points > 0.8 degrees away

            # Get mean discharge as proxy for WBMsed basin area at each point
            # (We use GlobalDeltaData to find WBMsed basin area at candidate points)
            if has_global_delta:
                # For each WBMsed candidate, find its basin area from GlobalDeltaData
                wbm_basin_areas = np.full(len(rm_lon), np.nan)
                within_radius = np.where(geo_dist <= (search_radius * 10))[0]

                if len(within_radius) == 0:
                    within_radius = np.array([np.argmin(geo_dist)])

                for idx in within_radius:
                    gd_dist_to_wbm = np.sqrt((gd_rm_lon - rm_lon[idx])**2 + 
                                             (gd_rm_lat - rm_lat[idx])**2)
                    nearest = np.argmin(gd_dist_to_wbm)
                    if gd_dist_to_wbm[nearest] < 5:
                        wbm_basin_areas[idx] = gd_basin_area[nearest]

                # Compute MATLAB-style cost for candidates within radius
                candidate_costs = np.full(len(rm_lon), np.inf)
                for idx in within_radius:
                    wba = wbm_basin_areas[idx]
                    if np.isnan(wba) or wba <= 0:
                        basin_term = 0  # can't compute, ignore basin area term
                    else:
                        basin_term = (abs(wba - estuary_basin_area) / 
                                      estuary_basin_area * 
                                      np.log10(estuary_basin_area))
                    hard_penalty = 10 if geo_dist[idx] > 8 else 0
                    candidate_costs[idx] = geo_dist[idx] + basin_term + hard_penalty

                best_index = np.argmin(candidate_costs)

            else:
                # No GlobalDeltaData: fall back to distance-weighted discharge score
                within_radius = np.where(geo_dist <= (search_radius * 10))[0]
                if len(within_radius) == 0:
                    best_index = np.argmin(geo_dist)
                else:
                    candidate_means = np.mean(discharge_series[within_radius, :], axis=1)
                    candidate_dists = geo_dist[within_radius]
                    scores = candidate_means / (candidate_dists + 0.01)
                    best_index = within_radius[np.argmax(scores)]

        else:
            # No basin area available: distance + discharge score only
            within_radius = np.where(geo_dist <= (search_radius * 10))[0]
            if len(within_radius) == 0:
                best_index = np.argmin(geo_dist)
            else:
                candidate_means = np.mean(discharge_series[within_radius, :], axis=1)
                candidate_dists = geo_dist[within_radius]
                scores = candidate_means / (candidate_dists + 0.01)
                best_index = within_radius[np.argmax(scores)]

        # --- Step 4: Compute scaling factor (fac = WBMsed_basin / estuary_basin) ---
        fac = 1.0  # default: no scaling
        if estuary_basin_area is not None and estuary_basin_area > 0 and has_global_delta:
            wbm_basin_area_best = wbm_basin_areas[best_index]
            if not np.isnan(wbm_basin_area_best) and wbm_basin_area_best > 0:
                fac = wbm_basin_area_best / estuary_basin_area
                print(f"  {estuary}: WBMsed basin={wbm_basin_area_best:.1f} km², "
                      f"estuary basin={estuary_basin_area:.1f} km², fac={fac:.4f}")
            else:
                print(f"  {estuary}: Could not determine WBMsed basin area for best point. "
                      f"fac=1.0 (no scaling).")

        # --- Step 5: Extract, clean, and scale ---
        q_data = np.maximum(np.nan_to_num(discharge_series[best_index, :]), 0) * fac
        s_data = np.maximum(np.nan_to_num(sed_series[best_index, :]),       0) * fac

        estuary_discharge_timeseries[estuary] = q_data
        estuary_sed_timeseries[estuary]       = s_data
        estuary_rm_coords[estuary]            = (rm_lon[best_index], rm_lat[best_index])
        estuary_scaling_factors[estuary]      = fac

    return estuary_discharge_timeseries, estuary_rm_coords, estuary_sed_timeseries, estuary_scaling_factors


def validate_estuary_location(estuary_name, original_coords, grid_coords, savefig=False, output_dir=None):
    """
    Validate estuary locations by plotting both original coordinates and extracted grid point.
    
    Parameters:
        estuary_name (str): Name of the estuary
        original_coords (tuple): Original (lat, lon) or (lat, lon, basin_area) coordinates
        grid_coords (tuple): Grid coordinates (rm_lon, rm_lat)
        savefig (bool): Whether to save figures
        output_dir (str, optional): Base directory to save figures
    """
    # Support both (lat, lon) and (lat, lon, basin_area) tuples
    lat, lon = original_coords[0], original_coords[1]
    rm_lon, rm_lat = grid_coords
    
    lon_grid = (rm_lon / 10) - 180
    lat_grid = (rm_lat / 10) - 90
    
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    extent = [lon - 5, lon + 5, lat - 5, lat + 5]
    ax.set_extent(extent)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.RIVERS)
    ax.stock_img()
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


def plot_estuary_timeseries(estuary_name, discharge_series, sed_series, datetimes,
                             savefig=False, output_dir=None, scaling_factor=1.0, 
                             window_ma = 7):
    """
    Generate time series plots for a specific estuary's discharge and sediment data.
    
    Parameters:
        estuary_name (str): Name of the estuary
        discharge_series (np.ndarray): Discharge time series data
        sed_series (np.ndarray): Sediment time series data
        datetimes (list): List of datetime objects
        savefig (bool): Whether to save figures
        output_dir (str, optional): Base directory to save figures
        scaling_factor (float): fac value applied during extraction (shown in title)
        window_ma (int): Window size for moving average calculation
    """
    mean_discharge = np.mean(discharge_series)
    mean_sediment  = np.mean(sed_series)
    fac_label = f"  [fac={scaling_factor:.3f}]" if scaling_factor != 1.0 else ""

    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, discharge_series, label=estuary_name, color='tab:blue')
    plt.axhline(mean_discharge, linestyle='dashed', color='orange',
                label=f'mean = {mean_discharge:.2f}')
    plt.xlabel('Time')
    plt.ylabel('River Discharge $Q_{river}$ [m³/s]')
    plt.title(f'$Q_{{river}}$ for {estuary_name} based on WBMsed{fac_label}')
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
    plt.axhline(mean_sediment, linestyle='dashed', color='red',
                label=f'mean = {mean_sediment:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Sediment Load [kg/s]')
    plt.title(f'Sediment load for {estuary_name} based on WBMsed + BQART{fac_label}')
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
    ax1.plot(datetimes, discharge_series, label='Discharge', color=color_discharge)
    ax1.axhline(mean_discharge, linestyle='dashed', color='orange',
                label=f'Mean Q = {mean_discharge:.2f}')
    ax1.tick_params(axis='y', labelcolor=color_discharge)

    ax2 = ax1.twinx()
    color_sediment = 'tab:green'
    ax2.set_ylabel('Sediment Load [kg/s]', color=color_sediment)
    ax2.plot(datetimes, sed_series, label='Sediment', color=color_sediment)
    ax2.axhline(mean_sediment, linestyle='dashed', color='red',
                label=f'Mean S = {mean_sediment:.2f}')
    ax2.tick_params(axis='y', labelcolor=color_sediment)

    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
    ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))

    plt.title(f'Combined discharge and sediment for {estuary_name}{fac_label}')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    fig.legend(lines_1 + lines_2, labels_1 + labels_2,
               loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)

    if savefig and outdir:
        save_path_combined = outdir / '03_Combined_Qriver_qs'
        save_path_combined.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_combined / f'Combined_{estuary_name}.png', dpi=200, bbox_inches='tight')
    plt.show()

    # --- Moving Average plot ---
    q_df = pd.Series(discharge_series, index=datetimes)
    moving_avg = q_df.rolling(window=window_ma, center=True).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, discharge_series, color='lightgrey', alpha=0.5, label='Daily Discharge')
    plt.plot(datetimes, moving_avg, color='tab:blue', linewidth=2, label=f'{window_ma}-day Moving Average')
    plt.axhline(mean_discharge, linestyle='dashed', color='orange',
                label=f'mean = {mean_discharge:.2f}')
    plt.xlabel('Time')
    plt.ylabel('River Discharge $Q_{river}$ [m³/s]')
    plt.title(f'Long-term Trend ($Q_{{river}}$) for {estuary_name}{fac_label}')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    if savefig and outdir:
        save_path_ma = outdir / '05_Moving_Average_Discharge'
        save_path_ma.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_ma / f'Trend_{estuary_name}_MA{window_ma}.png', dpi=200, bbox_inches='tight')
    
    plt.show()

    return mean_discharge, mean_sediment, moving_avg