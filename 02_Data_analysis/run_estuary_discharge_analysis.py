#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title: Estuary Discharge Analysis and Visualization
Description: This script loads estuary data from a .mat file, plots discharge time series,
             and visualizes the global distribution of estuaries using scatterplots and density maps.
             It is based on the WBMsed BQART models and follows the method of Nienhuis et al. (2020).
             Basin area scaling is applied following the original MATLAB get_Qriver_timeseries logic.
Author: Marloes Bonenkamp
Date: February 2026
"""

#%% --- IMPORTS ---
from pathlib import Path
import numpy as np
import pandas as pd
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\02_Data_analysis")

from FUNCTIONS.FUNCS_utils import transform_coordinates, load_data
from FUNCTIONS.FUNCS_variability_metrics import (
    analyze_discharge_metrics,
    visualize_discharge_metrics,
    visualize_discharge_metrics_comparison
)
from FUNCTIONS.FUNCS_discharge_timeseries_WBMsed import (
    plot_estuary_timeseries,
    plot_global_delta_distribution,
    extract_discharge_timeseries,
    validate_estuary_location
)

#%% --- CONFIGURATION ---

INPUT_DIR  = r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash"
OUTPUT_DIR = r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash\01_Analysis_smallselection_estuaries"

MAT_FILE          = "qs_timeseries_Nienhuis2020.mat"
TIF_FILE          = r"Nienhuis2020_Scripts\bqart_a.tif"          # relative to INPUT_DIR; set None to skip
GLOBAL_DELTA_FILE = r"Nienhuis2020_Scripts\GlobalDeltaData.mat"  # relative to INPUT_DIR; set None to skip

SAVE_FIG  = True
SAVE_DATA = True

Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

mat_file_path      = Path(INPUT_DIR) / MAT_FILE
tif_path           = Path(INPUT_DIR) / TIF_FILE          if TIF_FILE          else None
global_delta_path  = Path(INPUT_DIR) / GLOBAL_DELTA_FILE if GLOBAL_DELTA_FILE else None
CACHE_FILE         = Path(OUTPUT_DIR) / "estuary_timeseries_cache.xlsx"

window_ma = 7  # Window size for moving average (in days)

#%% --- ESTUARY COORDINATES ---
# Format: 'Name': (lat, lon)
# Basin area will be looked up automatically from GlobalDeltaData.mat.
# If you want to override basin area manually for a specific estuary,
# use: 'Name': (lat, lon, basin_area_km2)
estuary_coords = {
    # 'Mississippi': (29.15, -89.25),  # USA, for validation
    # 'Amazon': (-0.1, -50.4),         # Brazil, for validation

    'Chao Phraya': (13.55, 100.59),         #Thailand
    'Colorado (MX)': (31.83, -114.82),      #Mexico
    'Cacipore': (3.6, -51.2),               #French Guiana
    # 'Cromary Firth': (57.69, -4.02),        #UK
    # 'Demerara': (6.79, -58.18),             #Guyana
    # 'Firth of Tay': (56.45, -2.83),         #UK
    # 'Fly': (-8.62, 143.70),                 #Papua New Guinea
    # 'Gironde': (45.58, -1.05),              #France
    # 'Guayas': (-2.55, -79.88),              #Ecuador
    # 'Humber': (53.62, -0.11),               #UK
    # 'Kumbe': (-8.36, 140.23),               #Indonesia
    # 'Ord': (-15.5, 128.35),                 #Australia
    # 'Purna': (20.91, 72.78),                #India
    # 'Sokyosen': (36.9, 126.9),              #South Korea 
    # 'Sungai Merauke': (-8.47, 140.35),      #Indonesia
    # 'Suriname': (5.84, -55.11),             #Suriname
    # 'Taeryong': (39.63, 125.48),            #North Korea
    # 'Tapi': (21.15, 72.75),                 #India
    # 'Yangon': (16.52, 96.29),               #Myanmar
    # 'Wai Bian': (-8.10, 139.97),            #Indonesia   
    # 'Western Scheldt': (51.42, 3.57),       #Netherlands

    # #Excluded                                                         because:
    # 'Eel': (40.63, -124.31),                #USA                      - River-dominated
    # 'Klamath': (41.54, -124.08),            #USA                      - River-dominated
    # 'Thames': (51.5, 0.6),                  #UK                       - Too tide-dominated
    # 'Columbia': (46.25, -124.05),           #USA                      - Reason unclear
    # 'Rio de la Plata': (-35.00, -56.00),    #Argentina/Uruguay        - Reason unclear
    # 'Mengha (Ganges-Brahmaputra)': (21.47, 91.06), #Bangladesh        - Reason unclear
}

#%% --- LOAD DATA (with caching) ---

if CACHE_FILE.exists():
    print("Cache found. Loading extracted estuary time series from cache...")
    cache_xl = pd.read_excel(CACHE_FILE, sheet_name=None, index_col=0, parse_dates=True)

    estuary_discharge_data = {name: cache_xl[f'Q_{name[:28]}'].squeeze().values
                               for name in estuary_coords if f'Q_{name[:28]}' in cache_xl}
    estuary_sed_data       = {name: cache_xl[f'S_{name[:28]}'].squeeze().values
                               for name in estuary_coords if f'S_{name[:28]}' in cache_xl}

    first_sheet    = next(iter(cache_xl.values()))
    datetimes      = first_sheet.index.tolist()

    rm_df = cache_xl.get('rm_coords')
    estuary_rm_coords = ({row.Index: (row.rm_lon, row.rm_lat) for row in rm_df.itertuples()}
                         if rm_df is not None else {})

    fac_df = cache_xl.get('scaling_factors')
    estuary_scaling_factors = ({row.Index: row.fac for row in fac_df.itertuples()}
                                if fac_df is not None else {name: 1.0 for name in estuary_coords})

    print(f"Loaded {len(estuary_discharge_data)} estuaries from cache.")

else:
    print("No cache found. Loading full .mat file...")
    data = load_data(mat_file_path, tif_path=tif_path, global_delta_path=global_delta_path)

    rm_lat           = data['rm_lat']
    rm_lon           = data['rm_lon']
    discharge_series = data['discharge_series']
    sed_series       = data['sed_series']
    datetimes        = data['datetimes']

    gd_rm_lon    = data.get('gd_rm_lon')
    gd_rm_lat    = data.get('gd_rm_lat')
    gd_basin_area = data.get('gd_basin_area')

    print("\nExtracting discharge time series with basin area correction...")
    estuary_discharge_data, estuary_rm_coords, estuary_sed_data, estuary_scaling_factors = \
        extract_discharge_timeseries(
            estuary_coords, rm_lon, rm_lat, discharge_series, sed_series,
            gd_rm_lon=gd_rm_lon, gd_rm_lat=gd_rm_lat, gd_basin_area=gd_basin_area
        )

    # --- Save cache ---
    print("\nSaving extracted time series to cache...")
    with pd.ExcelWriter(CACHE_FILE, engine='openpyxl') as writer:
        for estuary in estuary_coords:
            pd.DataFrame(
                estuary_discharge_data[estuary],
                index=datetimes,
                columns=['discharge_m3s']
            ).to_excel(writer, sheet_name=f'Q_{estuary[:28]}')

            pd.DataFrame(
                estuary_sed_data[estuary],
                index=datetimes,
                columns=['sediment_kgs']
            ).to_excel(writer, sheet_name=f'S_{estuary[:28]}')

        pd.DataFrame.from_dict(
            estuary_rm_coords, orient='index', columns=['rm_lon', 'rm_lat']
        ).to_excel(writer, sheet_name='rm_coords')

        pd.DataFrame.from_dict(
            estuary_scaling_factors, orient='index', columns=['fac']
        ).to_excel(writer, sheet_name='scaling_factors')

    print(f"Cache saved to {CACHE_FILE}")

#%% --- PROCESS EACH ESTUARY ---

mean_discharge_values         = {}
mean_sediment_values          = {}
estuary_discharge_moving_average = {}

for estuary in estuary_coords:
    print(f"\nProcessing: {estuary}")

    fac = estuary_scaling_factors.get(estuary, 1.0)

    mean_discharge, mean_sediment, discharge_moving_average = plot_estuary_timeseries(
        estuary,
        estuary_discharge_data[estuary],
        estuary_sed_data[estuary],
        datetimes,
        SAVE_FIG,
        OUTPUT_DIR,
        scaling_factor=fac,
        window_ma=window_ma
    )
    mean_discharge_values[estuary]            = mean_discharge
    mean_sediment_values[estuary]             = mean_sediment
    estuary_discharge_moving_average[estuary] = discharge_moving_average

    validate_estuary_location(
        estuary,
        estuary_coords[estuary],
        estuary_rm_coords[estuary],
        SAVE_FIG,
        OUTPUT_DIR
    )
#%%
print("\n--- SCALING FACTOR DIAGNOSTIC ---")
for estuary, fac in estuary_scaling_factors.items():
    mean_raw = np.mean(estuary_discharge_data[estuary])
    print(f"  {estuary}: fac={fac:.6f}, mean_discharge={mean_raw:.4f} m³/s")

#%% --- SAVE MEAN VALUES ---

mean_df = pd.DataFrame.from_dict(
    mean_discharge_values, orient='index', columns=['Mean River Discharge [m3/s]'])
mean_df.index.name = 'Estuary'

mean_sediment_df = pd.DataFrame.from_dict(
    mean_sediment_values, orient='index', columns=['Mean Sediment Load [kg/s]'])
mean_sediment_df.index.name = 'Estuary'

if SAVE_DATA:
    discharge_path  = Path(OUTPUT_DIR) / '01_River_discharge_Qriver_per_estuary'
    sediment_path   = Path(OUTPUT_DIR) / '02_Sediment_load_per_estuary'
    moving_avg_path = Path(OUTPUT_DIR) / '05_Moving_Average_Discharge'
    for p in [discharge_path, sediment_path, moving_avg_path]:
        p.mkdir(parents=True, exist_ok=True)

    mean_df.to_excel(discharge_path / 'mean_river_discharges.xlsx')
    mean_sediment_df.to_excel(sediment_path / 'mean_sediment_loads.xlsx')

    for estuary, moving_avg in estuary_discharge_moving_average.items():
        moving_avg.to_excel(moving_avg_path / f'moving_average_{estuary}.xlsx')

#%% --- METRICS ---

# Diagnostic check on moving average data
print("\n--- Moving Average Dictionary Check ---")
for name, series in estuary_discharge_moving_average.items():
    vals  = np.array(series, dtype=float)
    valid = vals[~np.isnan(vals)]
    print(f"  {name}: {len(valid)} valid values, mean={np.nanmean(vals):.2f}")

# Compute metrics
df_metrics            = analyze_discharge_metrics(estuary_discharge_data)
df_metrics_moving_avg = analyze_discharge_metrics(estuary_discharge_moving_average)

# Visualize
metrics_output_dir = Path(OUTPUT_DIR) / "04_Metrics_per_estuary"
metrics_output_dir.mkdir(parents=True, exist_ok=True)

visualize_discharge_metrics(df_metrics, metrics_output_dir)
visualize_discharge_metrics(df_metrics_moving_avg, moving_avg_path)
visualize_discharge_metrics_comparison(
    df_metrics, 
    df_metrics_moving_avg, 
    moving_avg_path
)
# Save metrics
if SAVE_DATA:
    df_metrics.to_excel(
        Path(OUTPUT_DIR) / 'fluvial_sediment_flux_Qriver_metrics_per_estuary.xlsx', index=False)
    df_metrics_moving_avg.to_excel(
        Path(OUTPUT_DIR) / 'fluvial_sediment_flux_Qriver_metrics_per_estuary_moving_average.xlsx', index=False)

# %%
