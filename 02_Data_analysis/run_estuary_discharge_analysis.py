#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title: Estuary Discharge Analysis and Visualization
Description: This script loads estuary data from a .mat file, plots discharge time series,
             and visualizes the global distribution of estuaries using scatterplots and density maps.
             It is based on the WBMsed BQART models and follows the method of Nienhuis et al. (2020) for discharge analysis.
Author: Marloes Bonenkamp
Date: February 18, 2026
"""
#%% --- IMPORTS ---
from pathlib import Path
import pandas as pd
import sys

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\02_Data_analysis")

from FUNCTIONS.FUNCS_utils import transform_coordinates, load_data
from FUNCTIONS.FUNCS_variability_metrics import analyze_discharge_metrics, visualize_discharge_metrics
from FUNCTIONS.FUNCS_discharge_timeseries_WBMsed import plot_estuary_timeseries, plot_global_delta_distribution, extract_discharge_timeseries, validate_estuary_location

#%%
# Configuration variables - adjust these for each run
INPUT_DIR = r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash"
OUTPUT_DIR = r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash\01_Analysis_smallselection_estuaries"
MAT_FILE = "qs_timeseries_Nienhuis2020.mat"     # Path relative to INPUT_DIR
TIF_FILE = None         # Set to path if basin area TIF file is needed, relative to INPUT_DIR

# If you want to save the plotted figures/dataframes
SAVE_FIG = True
SAVE_DATA = True

# Create directories if they don't exist
Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load data
mat_file_path = Path(INPUT_DIR) / MAT_FILE
tif_path = Path(INPUT_DIR) / TIF_FILE if TIF_FILE else None
data = load_data(mat_file_path, tif_path)

# Extract variables from data
rm_lat = data['rm_lat']
rm_lon = data['rm_lon']
discharge_series = data['discharge_series']
sed_series = data['sed_series']
datetimes = data['datetimes']

#%%
# Define estuary coordinates (lat, lon)
estuary_coords = {
    'Mississippi': (29.15, -89.25),  # validation of method
    'Amazon': (-0.1, -50.4),         # validation of method
    'Eel': (40.63, -124.31),
    'Klamath': (41.54, -124.08),
    'Cacipore': (3.6, -51.2),
    'Suriname': (5.84, -55.11),
    'Demerara': (6.79, -58.18),
    'Yangon': (16.52, 96.29),
    'Sokyosen': (36.9, 126.9),
    'Wai Bian': (-8.10, 139.97),
    'Thames': (51.5, 0.6),
    'Columbia': (46.25, -124.05),
    'Gironde': (45.58, -1.05),
    'Chao Phraya': (13.55, 100.59),
    'Fly': (-8.62, 143.70),
    'Rio de la Plata': (-35.00, -56.00)
}

#%% Plot map of global distribution of deltas, based on mean discharge
# mean_discharge = np.nanmean(discharge_series, axis=1)  # Average discharge per estuary
# plot_global_delta_distribution(rm_lon, rm_lat, mean_discharge, SAVE_FIG, OUTPUT_DIR)

#%%

estuary_discharge_data, estuary_rm_coords, estuary_sed_data = extract_discharge_timeseries(estuary_coords, rm_lon, rm_lat, discharge_series, sed_series)

# Dictionaries to store mean values
mean_discharge_values = {}
mean_sediment_values = {}
estuary_discharge_moving_average = {}

# Process each estuary
for estuary in estuary_coords:
    # Plot time series
    mean_discharge, mean_sediment, discharge_moving_average = plot_estuary_timeseries(
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
    estuary_discharge_moving_average[estuary] = discharge_moving_average

    # Validate location
    validate_estuary_location(
        estuary, 
        estuary_coords[estuary], 
        estuary_rm_coords[estuary], 
        SAVE_FIG, 
        OUTPUT_DIR
    )
#%%
# Create DataFrames with mean values
mean_df = pd.DataFrame.from_dict(mean_discharge_values, orient='index', columns=['Mean River Discharge [m3/s]'])
mean_df.index.name = 'Estuary'

mean_sediment_df = pd.DataFrame.from_dict(mean_sediment_values, orient='index', columns=['Mean Sediment Load [kg/s]'])
mean_sediment_df.index.name = 'Estuary'

# Save mean values to Excel
if SAVE_DATA:
    discharge_path = Path(OUTPUT_DIR) / '01_River_discharge_Qriver_per_estuary'
    sediment_path = Path(OUTPUT_DIR) / '02_Sediment_load_per_estuary'
    moving_avg_path = Path(OUTPUT_DIR) / '03_Moving_Average_Discharge_per_estuary'
    discharge_path.mkdir(parents=True, exist_ok=True)
    sediment_path.mkdir(parents=True, exist_ok=True)
    moving_avg_path.mkdir(parents=True, exist_ok=True)
    mean_df.to_excel(discharge_path / 'mean_river_discharges.xlsx')
    mean_sediment_df.to_excel(sediment_path / 'mean_sediment_loads.xlsx')

    # Save moving average data to Excel
    for estuary, moving_avg in estuary_discharge_moving_average.items():
        moving_avg.to_excel(moving_avg_path / f'moving_average_{estuary}.xlsx')     

# Analyze discharge metrics
df_metrics = analyze_discharge_metrics(estuary_discharge_data)
df_metrics_moving_avg = analyze_discharge_metrics(estuary_discharge_moving_average)

# Visualize discharge metrics and save to a dedicated directory
metrics_output_dir = Path(OUTPUT_DIR) / "04_Metrics_per_estuary"
metrics_output_dir.mkdir(parents=True, exist_ok=True)

visualize_discharge_metrics(df_metrics, metrics_output_dir)
visualize_discharge_metrics(df_metrics_moving_avg, metrics_output_dir / "moving_average")

# Save metrics to Excel
if SAVE_DATA:
    df_metrics.to_excel(Path(OUTPUT_DIR) / 'fluvial_sediment_flux_Qriver_metrics_per_estuary.xlsx', index=False)
    df_metrics_moving_avg.to_excel(Path(OUTPUT_DIR) / 'fluvial_sediment_flux_Qriver_metrics_per_estuary_moving_average.xlsx', index=False)
# %%
