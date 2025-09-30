#%% 
"""Enhanced Delft3D-4 Flow NetCDF Analysis Script with Seaborn
Last edit: June 2025
Author: Marloes Bonenkamp
Enhanced with seaborn visualizations for estuary analysis"""

#%%
import seaborn as sns
import datetime
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import sys

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Python\03_Model_postprocessing")

#%%
from FUNCTIONS.FUNCS_postprocessing_general import *
from FUNCTIONS.FUNCS_postprocessing_braiding_index import *
from FUNCTIONS.FUNCS_postprocessing_map_output import *
from FUNCTIONS.FUNCS_postprocessing_his_output import *

#%% CONFIGURATION
# Which DISCHARGE SCENARIO do you want to post-process?
discharge = 250 
scenario = f'01_baserun{discharge}' #'01_run{discharge}_seasonal'

# Which range of TIMESTEPS do you want to post-process? 
# note: 100 timesteps = ~30 seconds loading time
slice_start = 0     #4200
slice_end = 120     #4300
amount_to_plot = 3

small_estuary_model = False

if small_estuary_model:
    # PATHS for small estuary model
    model_location = r"fill in small estuary path here"
    
    trim_file = os.path.join(model_location, 'trim-estuary.nc') 
    trih_file = os.path.join(model_location, 'trim-estuary.nc')
    save_dir = os.path.join(model_location, 'postprocessing_plots')

    # OUTPUT TIME SETTINGS (based on .mdf file)
    map_output_interval = 4500      # minutes (Flmap interval)
    his_output_interval = 4500      # minutes (Flhis interval)
    total_duration = 132480         # minutes (exact duration from Tstop)
    reference_date = datetime.datetime(2015, 2, 16)  # Itdate = #2015-02-16#

else:
    #PATHS for my model
    model_location = r"U:\PhDNaturalRhythmEstuaries\Models\04_RiverDischargeVariability_domain45x15"
    runname = f's1_{discharge}_Wup_300m' 
    scenario_name = f'{scenario}'

    trim_file = os.path.join(model_location, runname, scenario_name, 'trim-varriver_tidewest.nc') 
    trih_file = os.path.join(model_location, runname, scenario_name, 'trih-varriver_tidewest.nc')
    save_dir = os.path.join(model_location, runname, scenario_name, 'postprocessing_plots')

    # OUTPUT TIME SETTINGS (based on .mdf file)
    map_output_interval = 1300    # minutes (Flmap interval)
    his_output_interval = 720     # minutes (Flhis interval)
    total_duration = 524160/2      # minutes (exact duration from Tstop)
    reference_date = datetime.datetime(2024, 1, 1)  # Itdate = #2024-01-01#

os.makedirs(save_dir, exist_ok=True)
save_figure = True

#%% LOAD DATASETS
print('Loading trim_file...')
start_time = time.time()
dataset_trim = nc.Dataset(trim_file, mode='r')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Data loading 'trim_file with nc.Dataset' took: {elapsed_time:.4f} seconds")

print('Loading trih_file...')
start_time = time.time()
dataset_trih = nc.Dataset(trih_file, mode='r')
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Data loading 'trih_file with nc.Dataset' took: {elapsed_time:.4f} seconds")

# %% Check which variables are stored in the .nc file
# print('trim variable analysis:')
# ds_trim_vars = check_available_variables(dataset_trim)
# print(' ')
# print('trih variable analysis:')
# ds_trih_vars = check_available_variables(dataset_trih)

#%% LOAD COORDINATES
print('Loading coordinates...')
x = load_variable(dataset_trim, "XCOR")
y = load_variable(dataset_trim, "YCOR")
print('Coordinates loaded')

#%% Get cross-section coordinates for braiding index and plotting (FOR MAP POST-PROCESSING ONLY)
col_indices, N_coords, x_targets = get_cross_section_coordinates(x, y)
print(f"Cross-sections defined at x-coordinates: {x_targets}")

#%% Load full dataset (this takes way too long, find another way)
# start_time = time.time()
# bed_level = load_variable(dataset_trim, "DPS", range=slice(0,None))
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Data loading took: {elapsed_time:.4f} seconds")

#%% 1a. SPATIAL MAP PLOTS (BED LEVEL, WATER LEVEL, ...)
print("\n=== SPATIAL PLOTS ===")

run_spatial_plots = True  # Set to False to skip this section
# timesteps = np.arange(slice_start, slice_start + amount_to_plot + 1, 1)
timesteps = np.arange(slice_start, slice_end+int((slice_end-slice_start)/amount_to_plot), int((slice_end-slice_start)/amount_to_plot))

for timestep in timesteps: 
    # 1a. BED LEVEL AND WATER LEVEL MAP PLOTS
    print("\n=== BED LEVEL AND WATER LEVEL PLOTS ===")
    if run_spatial_plots:
        # Load data 
        print(f"Loading data for timestep {timestep}...")
        
        bed_level = load_single_timestep_variable(dataset_trim, "DPS", timestep=timestep)
        water_level = load_single_timestep_variable(dataset_trim, "S1", timestep=timestep)
        water_depth = water_level+bed_level
        # Create plots
        print("Creating spatial plots...")

        plot_map(x, y, bed_level, 'bed_level', col_indices, N_coords, timestep, scenario, save_dir, save_figure)
        plot_map(x, y, water_level, 'water_level', col_indices, N_coords, timestep, scenario, save_dir, save_figure)
        plot_map(x, y, water_depth, 'water_depth', col_indices, N_coords, timestep, scenario, save_dir, save_figure)
    
    # 1b. VELOCITY MAP PLOTS 
    print("\n=== VELOCITY PLOTS ===")
    run_velocity_plots = False  # Set to False to skip this section

    if run_velocity_plots:
        # Example usage for U1 velocity at timestep=1, surface layer
        velocity = load_single_timestep_variable(dataset_trim, "U1", timestep=timestep, remove=1, layer=0)
        
        plot_velocity(x, y, velocity, 'U1', col_indices, N_coords, timestep, scenario, save_dir, save_figure)

print("Map plots completed.")

#%% 2. CROSS-SECTION ANALYSIS (HIS FILE)
print("\n=== DISCHARGE ANALYSIS ===")
run_discharge_analysis = True  # Set to False to skip this section

if run_discharge_analysis:
    variable = 'CTR'  # Current transport rate
    print(f"Analyzing {variable} data...")
    
    if small_estuary_model:
        #STATION_NAMES for small estuary model
        x_values = [2, 18, 44, 50, 60, 65, 70, 75, 80, 90, 100, 125, 150, 180, 225, 287]
        
        station_names = [f"({x},161)..({x},2)" for x in x_values]
    else:
        #STATION_NAMES for my model 
        station_names = [f'river_km_{i}' for i in range(27)]
    
    # Extract discharge data
    results, all_stations = extract_his_data(dataset_trih, variable, station_names)

    # for name in station_names:
    #     data = results.get(name)
    #     if data is None:
    #         print(f"[MISSING] No data for {name}")
    #     elif np.all(np.isnan(data)):
    #         print(f"[EMPTY] All NaN values for {name}")
    #     else:
    #         print(f"[OK] Data found for {name}: min={np.nanmin(data)}, max={np.nanmax(data)}")
        
    #         print(f'full data = {data}')
    for name in station_names:
        time, discharge = results[name]
        print(f"{name} time range: {time.min()} to {time.max()}")
        print(f"{name} discharge range: min={np.nanmin(discharge)}, max={np.nanmax(discharge)}")
        print(f"{name}: Number of time points = {len(time)}")

    # Plot discharge time series
    his_plot_discharge_timeseries(
        results, station_names, reference_date, save_dir, save_figure,
        time_range=(int(slice_start*map_output_interval/his_output_interval), 
                   int(slice_end*map_output_interval/his_output_interval))
    )
    
    print("Discharge analysis completed.")

#%% 3. BRAIDING INDEX ANALYSIS
print("\n=== BRAIDING INDEX ANALYSIS ===")
run_braiding_analysis = False  # Set to False to skip this section

if run_braiding_analysis:
    print("Loading shear stress data...")
    tau_max = load_variable(dataset_trim, "TAUMAX", range=slice(slice_start, slice_end))
    
    print("Computing braiding index...")
    BI_per_cross_section, datetimes, _, _, _ = compute_BI_per_cross_section(
        x, y, tau_max, slice_start, map_output_interval, reference_date, theta=0.5
    )
    
    # Plot braiding index results
    df_BI = plot_braiding_index_timeseries(BI_per_cross_section, x_targets, datetimes)
    plot_mean_braiding_index(df_BI, x_targets)
    
    print("Braiding index analysis completed.")
#%%
#%% """TRIM-file, load a couple variables for a couple timesteps:""" 
vars = {
        'shear_stress': "TAUMAX" }#, 
        # 'bed_level':    "DPS", 
        # 'water_level':  "S1"
        
        # }

vars_results = {}

for key, var_name in vars.items():
    start_time = time.time()
    data = load_variable(dataset_trim, var_name, range=slice(slice_start, slice_end))
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Data loading for {key} {var_name} took: {elapsed_time:.4f} seconds")

    vars_results[key] = data

# Compute braiding index: load shear stress data
tau_max = vars_results['shear_stress']

BI_per_cross_section, datetimes, x_targets, col_indices, N_coords = compute_BI_per_cross_section(x, y, tau_max, slice_start, map_output_interval, reference_date, theta=0.5)

df_BI = plot_braiding_index_timeseries(BI_per_cross_section, x_targets, datetimes)

plot_mean_braiding_index(df_BI, x_targets)

#%%    
# SEDIMENT BUDGET
# sed_budget = calculate_sediment_budget(
#     dataset_trih, 'river_km_0', 'river_km_26'
# )
# print(f"Total sediment in: {sed_budget['total_in']:.2e}")
# print(f"Total sediment out: {sed_budget['total_out']:.2e}")
# print(f"Net balance (entire simulation): {sed_budget['net_balance']:.2e}")

# #%%
# #total length: 52417

# # print_steady_state = np.arange(52096, 52400, 6)
# print_steady_state = np.arange(int(slice_start*map_output_interval/his_output_interval), int(slice_end*map_output_interval/his_output_interval), 6)

# for i in print_steady_state:
#     result_lastsed = get_last_sediment_transport(dataset_trih, 'river_km_1', 'river_km_26', i, reference_date)
#     # print(f"At t={result_lastsed['last_time']}, upstream: {result_lastsed['last_in']}, downstream: {result_lastsed['last_out']}, difference: {result_lastsed['difference']}")
    
#     upstream = result_lastsed['last_in']
#     downstream = result_lastsed['last_out']

#     # Downstream as a percentage of upstream
#     if upstream != 0:
#         downstream_pct = (downstream / upstream) 
#         print(f"For {scenario}{discharge} at {result_lastsed['real_date']}, {result_lastsed['last_time']:.2f}: Downstream sediment transport is {downstream_pct:.2f} times upstream.")
#     else:
#         print("Upstream sediment transport is zero; cannot compute percentage.")

#     # (Optional) Relative loss as a percentage
#     if upstream != 0:
#         loss_pct = ((upstream - downstream) / upstream)
#         print(f"For {scenario}{discharge} at {result_lastsed['real_date']}, {result_lastsed['last_time']:.2f}: Relative loss: {loss_pct:.2f} times upstream sediment is not transported downstream.")
# # %%
# import matplotlib.dates as mdates
# # Collect data for plotting
# time_points = []
# sediment_ratios = []


# for i in print_steady_state:
#     # Get sediment data
#     result_lastsed = get_last_sediment_transport(dataset_trih, 'river_km_0', 'river_km_26', i, reference_date)
    
#     # Calculate ratios
#     upstream_sed = result_lastsed['last_in']
#     downstream_sed = result_lastsed['last_out']
#     sediment_ratio = downstream_sed / upstream_sed if upstream_sed != 0 else np.nan

#     # Store values
#     time_points.append(result_lastsed['real_date'])
#     sediment_ratios.append(sediment_ratio)

# # Create plot
# fig, ax1 = plt.subplots(figsize=(12, 6))

# # Plot sediment transport ratio
# color = 'tab:red'
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Sediment Transport Ratio (Downstream/Upstream)', color=color)
# ax1.plot(time_points, sediment_ratios, color=color, label='Sediment Ratio')
# ax1.tick_params(axis='y', labelcolor=color)

# # # Create second y-axis for discharge
# # ax2 = ax1.twinx()  
# # color = 'tab:blue'
# # ax2.set_ylabel('Discharge Ratio (Downstream/Upstream)', color=color)
# # ax2.plot(time_points, discharge_values, color=color, linestyle='--', label='Discharge Ratio')
# # ax2.tick_params(axis='y', labelcolor=color)

# # Formatting
# plt.title('Sediment Transport Ratios Over Time')
# fig.tight_layout()
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
# plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
# plt.gcf().autofmt_xdate()
# plt.show()
# # %%
