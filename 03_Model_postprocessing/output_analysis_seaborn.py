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

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

from FUNCTIONS.NEWFUNCS_output_analysis_seaborn import *
from FUNCTIONS.FUNCS_postprocessing_braiding_index import *
from FUNCTIONS.FUNCS_postprocessing_map_output import *
from FUNCTIONS.FUNCS_postprocessing_his_output import *

#%% WHICH DISCHARGE SCENARIO DO YOU WANT TO POST-PROCESS?
discharge=500 
scenario='00_baserun'

#%% WHICH RANGE OF TIMESTEPS DO YOU WANT TO POST-PROCESS? 
# note: 100 timesteps = ~30 seconds loading time

slice_start = 0
slice_end = 100

# slice_start = 4250
# slice_end = 4300

#%% PATHS

model_location = r'U:\PhDNaturalRhythmEstuaries\Models\0_GUI_model' #r'U:\PhDNaturalRhythmEstuaries\Models\04_RiverDischarge_Intermittency_Flashiness_domain35x15'
runname = f'{discharge}_ConstantRiver_NoTide_3000m_400m' #_VariableRiver_TideWest_3000m_300m'ConstantRiver_NoTide_3000m_400m'
scenario_name = f'{scenario}{discharge}_oldbathy_ini'

trim_file = os.path.join(model_location, 'trim-constantriver_tidewest.nc') #os.path.join(model_location, runname, scenario_name, 'trim-constantriver_tidewest.nc')
trih_file = os.path.join(model_location, 'trih-constantriver_tidewest.nc')

# OUTPUT TIME SETTINGS (based on your .mdf file)
map_output_interval = 120    # minutes (Flmap interval)
his_output_interval = 10     # minutes (Flhis interval)
total_duration = 524160      # minutes (exact duration from Tstop)

# REFERENCE DATE AND TIME CONVERSION
reference_date = datetime.datetime(2024, 1, 1)  # Itdate = #2024-01-01#
#%%
# Open file directly
start_time = time.time()
dataset_trim = nc.Dataset(trim_file, mode='r')
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Data loading 'trim_file with nc.Dataset' took: {elapsed_time:.4f} seconds")

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

#%%
x = load_variable(dataset_trim, "XCOR")
y = load_variable(dataset_trim, "YCOR")

#%% Load full dataset (this takes way too long, find another way)

# start_time = time.time()
# bed_level = load_variable(dataset_trim, "DPS", range=slice(0,None))
# end_time = time.time()
# elapsed_time = end_time - start_time

# print(f"Data loading took: {elapsed_time:.4f} seconds")

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

#%% Get single timestep bed-level and water-level to plot
timestep = slice_start

bed_level0 = load_variable(dataset_trim, "DPS", range = slice(0, 1), remove=1)
bed_level0 = np.squeeze(bed_level0)

bed_level = load_variable(dataset_trim, "DPS", range = slice(timestep, timestep+1), remove=1)
bed_level = np.squeeze(bed_level)

water_level0 = load_variable(dataset_trim, "S1", range = slice(0, 1), remove=1)
water_level0 = np.squeeze(water_level0)

water_level = load_variable(dataset_trim, "S1", range = slice(timestep, timestep+1), remove=1)
water_level = np.squeeze(water_level)

# Check bed level
plot_map(x, y, bed_level0, 'bed_level', col_indices, N_coords)
plot_map(x, y, bed_level, 'bed_level', col_indices, N_coords)

# Check water level 
plot_map(x, y, bed_level0+water_level0, 'water_level', col_indices, N_coords)
plot_map(x, y, bed_level+water_level, 'water_level', col_indices, N_coords)
#%%
# Check velocity instead of momentary discharge
velocity0 = load_variable(dataset_trim, "U1", range = slice(0, 1), remove=1)
velocity0 = np.squeeze(velocity0)

velocity = load_variable(dataset_trim, "U1", range = slice(timestep, timestep+1), remove=1)
velocity = np.squeeze(velocity)

plot_map(x, y, velocity0, 'water_level', col_indices, N_coords)
plot_map(x, y, velocity, 'water_level', col_indices, N_coords)

# %% Make this plot as well for the ZCURU (horizontal velocity)
variable = 'CTR'
print(f"Analyzing {variable} data")

station_names = [f'river_km_{i}' for i in range(27)]
results, all_stations = extract_his_data(
    dataset_trih, variable, station_names)

# Plot discharge time series
his_plot_discharge_timeseries(results, station_names, reference_date, 
                        time_range=(int(slice_start*map_output_interval/his_output_interval), int(slice_end*map_output_interval/his_output_interval)))

#%%    
# SEDIMENT BUDGET
sed_budget = calculate_sediment_budget(
    dataset_trih, 'river_km_0', 'river_km_26'
)
print(f"Total sediment in: {sed_budget['total_in']:.2e}")
print(f"Total sediment out: {sed_budget['total_out']:.2e}")
print(f"Net balance (entire simulation): {sed_budget['net_balance']:.2e}")

#%%
#total length: 52417

# print_steady_state = np.arange(52096, 52400, 6)
print_steady_state = np.arange(int(slice_start*map_output_interval/his_output_interval), int(slice_end*map_output_interval/his_output_interval), 6)

for i in print_steady_state:
    result_lastsed = get_last_sediment_transport(dataset_trih, 'river_km_1', 'river_km_26', i, reference_date)
    # print(f"At t={result_lastsed['last_time']}, upstream: {result_lastsed['last_in']}, downstream: {result_lastsed['last_out']}, difference: {result_lastsed['difference']}")
    
    upstream = result_lastsed['last_in']
    downstream = result_lastsed['last_out']

    # Downstream as a percentage of upstream
    if upstream != 0:
        downstream_pct = (downstream / upstream) 
        print(f"For {scenario}{discharge} at {result_lastsed['real_date']}, {result_lastsed['last_time']:.2f}: Downstream sediment transport is {downstream_pct:.2f} times upstream.")
    else:
        print("Upstream sediment transport is zero; cannot compute percentage.")

    # (Optional) Relative loss as a percentage
    if upstream != 0:
        loss_pct = ((upstream - downstream) / upstream)
        print(f"For {scenario}{discharge} at {result_lastsed['real_date']}, {result_lastsed['last_time']:.2f}: Relative loss: {loss_pct:.2f} times upstream sediment is not transported downstream.")
# %%
import matplotlib.dates as mdates
# Collect data for plotting
time_points = []
sediment_ratios = []


for i in print_steady_state:
    # Get sediment data
    result_lastsed = get_last_sediment_transport(dataset_trih, 'river_km_0', 'river_km_26', i, reference_date)
    
    # Calculate ratios
    upstream_sed = result_lastsed['last_in']
    downstream_sed = result_lastsed['last_out']
    sediment_ratio = downstream_sed / upstream_sed if upstream_sed != 0 else np.nan

    # Store values
    time_points.append(result_lastsed['real_date'])
    sediment_ratios.append(sediment_ratio)

# Create plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot sediment transport ratio
color = 'tab:red'
ax1.set_xlabel('Time')
ax1.set_ylabel('Sediment Transport Ratio (Downstream/Upstream)', color=color)
ax1.plot(time_points, sediment_ratios, color=color, label='Sediment Ratio')
ax1.tick_params(axis='y', labelcolor=color)

# # Create second y-axis for discharge
# ax2 = ax1.twinx()  
# color = 'tab:blue'
# ax2.set_ylabel('Discharge Ratio (Downstream/Upstream)', color=color)
# ax2.plot(time_points, discharge_values, color=color, linestyle='--', label='Discharge Ratio')
# ax2.tick_params(axis='y', labelcolor=color)

# Formatting
plt.title('Sediment Transport Ratios Over Time')
fig.tight_layout()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.gcf().autofmt_xdate()
plt.show()
# %%
