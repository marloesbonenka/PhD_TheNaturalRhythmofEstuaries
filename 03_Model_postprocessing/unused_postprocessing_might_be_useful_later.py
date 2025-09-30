#%% 
"""Delft3D-4 Flow NetCDF Analysis: Morphological Estuary Analysis.
Last edit: June 2025
Author: Marloes Bonenkamp
"""

#%% IMPORTS AND SETUP
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import os
import sys

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Python\03_Model_postprocessing")

from FUNCTIONS.FUNCS_postprocessing_general import *
from FUNCTIONS.FUNCS_postprocessing_braiding_index import *
from FUNCTIONS.FUNCS_postprocessing_map_output import *
from FUNCTIONS.FUNCS_postprocessing_his_output import *

#%% CONFIGURATION
# Which DISCHARGE SCENARIO do you want to post-process?
discharge = 250 

# Scenario options: 
#f'01_baserun{discharge}'
#f'02_run{discharge}_seasonal'
#f'03_run{discharge}_flashy'

scenario = f'02_run{discharge}_seasonal' 

# Run options: 
runname = f's1_{discharge}_Wup_300m' 

# Which range of TIMESTEPS do you want to post-process? 
slice_start = 0     
slice_end = 120     # maximum is 120 (for full map output)

# How many map plots to make?
amount_to_plot = 3

# Model run time settings (based on .mdf file)
reference_date = datetime.datetime(2024, 1, 1)  # Itdate = #2024-01-01#
Tstart = 2.628e6
Tstop = 2.8908e6
total_duration = Tstop - Tstart
total_duration_days = total_duration / (60 * 24)

map_output_interval = 1300    # minutes (Flmap interval)
his_output_interval = 720     # minutes (Flhis interval)    

map_output_interval_hours = map_output_interval / 60
his_output_interval_hours = his_output_interval / 60

#%% Define masking settings

# Define estuary bounds
x_min, x_max = 20000, 45000
y_min, y_max = 5000, 10000

# Define land threshold
bed_threshold = 6  # exclude land (higher than 6 m)

#%% File locations
model_location = r"U:\PhDNaturalRhythmEstuaries\Models\04_RiverDischargeVariability_domain45x15"
scenario_name = f'{scenario}'

trim_file = os.path.join(model_location, runname, scenario_name, 'trim-varriver_tidewest.nc') 
trih_file = os.path.join(model_location, runname, scenario_name, 'trih-varriver_tidewest.nc')
save_dir = os.path.join(model_location, runname, scenario_name, 'postprocessing_plots')

os.makedirs(save_dir, exist_ok=True)
save_figure = True

#%% LOAD FULL DATASETS
print("Loading trim_file...")
start_time = time.time()
dataset_trim = nc.Dataset(trim_file, mode='r')
print(f"Trim file loaded in {time.time() - start_time:.2f} seconds")

print("Loading trih_file...")
start_time = time.time()
dataset_trih = nc.Dataset(trih_file, mode='r')
print(f"Trih file loaded in {time.time() - start_time:.2f} seconds")

# Get the shape of the a variable -- to figure out the length of the t, x, y arrays
print('trim shape (t, x, y) =', dataset_trim.variables['DPS'].shape)

print("Loading coordinates...")
x = load_variable(dataset_trim, "XCOR")
y = load_variable(dataset_trim, "YCOR")

nx, ny = x.shape
print(f"x/y shape: {x.shape}")

# #Sanity check for grid layout (check x-direction and y-direction)
# plt.figure()
# plt.pcolormesh(x, y, x, shading='auto')  # coloring by x values just to visualize
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('XCOR grid coloring by X values')
# plt.colorbar(label='x value')
# plt.axis('equal')
# plt.show()


#%% LOAD VARIABLES OF CHOICE (if you want to see all available variables, use: dataset_trim.variables)
# Use load_variable if the variable is spaced on the grid (M-N) == True for almost every variable
# Don't use load_variable for e.g. (morphological) time
variable = 'DPS'
long_name = 'bed level'

print(f"Loading full {variable} ({long_name}) variable...")
start_time = time.time()
bedlev = -1 * load_variable(dataset_trim, variable, range=slice(slice_start, slice_end)) # multiply by -1 because Delft3D output notes down as positive
time_steps = bedlev.shape[0]
print(f'{variable} variable shape is: {bedlev.shape}')
print(f"Full {variable} loaded in {time.time() - start_time:.2f} seconds")

variable = 'MORFT'
long_name = 'morphological time'

print(f"Loading full {variable} ({long_name}) variable...")
start_time = time.time()
morphtime = dataset_trim.variables[variable][:]
print(f'{variable} variable shape is: {morphtime.shape}')
print(f"Full {variable} loaded in {time.time() - start_time:.2f} seconds")

#%% Convert morphological time in days since Tstart to actual dates
morph_days = np.array(morphtime[:])

morph_datetimes = np.array([reference_date + datetime.timedelta(days=float(day)) for day in morph_days])

#%% CALCULATE ESTUARY-AVERAGED BED LEVEL 
print("Calculating domain-averaged bed level...")
domain_averaged_bedlevel = np.nanmean(bedlev, axis=(1, 2))  # For each timestep, the mean over all grid points is taken

print("Masking and calculating estuary-averaged bed level...")

# Wet mask (keep only esturaine areas, exclude land areas)
wet_mask = bedlev < bed_threshold  

# Spatial mask for estuary
spatial_mask = (
    (x >= x_min) & (x <= x_max) &
    (y >= y_min) & (y <= y_max)
)

# Combine masks
full_mask = spatial_mask[None, :, :] & wet_mask  # shape (t, x, y)

# Apply mask
masked_bedlev = np.where(full_mask, bedlev, np.nan)

# Compute estuary-averaged bed level per timestep
estuary_averaged_bedlevel = np.nanmean(masked_bedlev, axis=(1, 2))  # shape: (t,)

#%% PLOT DOMAIN-AVERAGED BED LEVEL
plt.figure(figsize=(10, 4))
plt.plot(np.arange(time_steps), domain_averaged_bedlevel)
plt.title("Domain-Averaged Bed Level Over Time")
plt.xlabel("Timestep")
plt.ylabel("Bed level (m DPS)\n(more negative = higher)")
plt.grid(True)
plt.tight_layout()

if save_figure:
    plt.savefig(os.path.join(save_dir, "DOMAIN_averaged_bedlevel_over_time.png"))

plt.show()


#%% CALCULATE ACTIVE ESTUARY WIDTH
# DPS interpretation settings
depth_cutoff = 0.0  # m below datum for defining "active" width

print("Computing active estuary width along estuary axis...")
active_width = np.zeros((time_steps, nx))  # (time, x)

for t in range(time_steps):
    for i in range(nx):  # loop over x
        bed_row = bedlev[t, i, :]  # shape: (140,)
        y_row = y[i, :]            # shape: (140,)
        mask = bed_row >= -depth_cutoff
        if np.any(mask):
            y_active = y_row[mask]
            active_width[t, i] = y_active.max() - y_active.min()
        else:
            active_width[t, i] = 0.0
#%% POSTPROCESS ONE SCENARIO AT A TIME

# CALCULATE WIDTH-AVERAGED BED LEVEL AT 5 KM AND 15 KM FROM ESTUARY MOUTH
# 1. Get the x and y slices at t = 0 (since x and y don't change over time (same grid for each timestep))
x0 = x[:, 0]  # shape: (ny,)
y0 = y[0, :]  # shape: (nx,)

# 2. Find i (x-direction) indices closest to 25000 (5km upstream from mouth) and 35000 (15km upstream from mouth)
x_5km_idx = np.argmin(np.abs(x0 - 25000))
x_15km_idx = np.argmin(np.abs(x0 - 35000))

# 3. Find j (y-direction) indices within 5000–10000
y_indices = np.where((y0 >= 5000) & (y0 <= 10000))[0]

# 4. Extract cross-sections
bedlev_5km = bedlev[:, x_5km_idx, y_indices]      # shape: (time, len(y_indices))
bedlev_15km = bedlev[:, x_15km_idx, y_indices]    # shape: (time, len(y_indices))

# 5. Mask land (bedlev >= 6 m)
bedlev_5km_masked = np.where(bedlev_5km < 6, bedlev_5km, np.nan)
bedlev_15km_masked = np.where(bedlev_15km < 6, bedlev_15km, np.nan)

# 6. Compute width-averaged bed level at 5 km and 15 km upstream from the estuary mouth
averaged_bedlev_5km = np.nanmean(bedlev_5km_masked, axis=(1))       # shape: (t,)
averaged_bedlev_15km = np.nanmean(bedlev_15km_masked, axis=(1))     # shape: (t,)

# 7. Visualize
plt.figure(figsize=(10, 5))
plt.plot(averaged_bedlev_5km, label='5 km upstream', lw=2)
plt.plot(averaged_bedlev_15km, label='15 km upstream', lw=2)

plt.xlabel('time')
plt.ylabel('Width-averaged bed level [m]')
plt.title(f'Morphological evolution of bed level at two estuary cross-sections for {scenario}')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% PLOT ACTIVE WIDTH AT MID-ESTUARY OVER TIME
mid_idx = nx // 2
plt.figure(figsize=(10, 4))
plt.plot(np.arange(time_steps), active_width[:, mid_idx])
plt.title(f"Active Estuary Width at Mid-Estuary (x = {mid_idx})")
plt.xlabel("Timestep")
plt.ylabel("Active width (m)")
plt.grid(True)
plt.tight_layout()

if save_figure:
    plt.savefig(os.path.join(save_dir, "active_width_mid_estuary_over_time.png"))

plt.show()

#%% OPTIONAL: HEATMAP OF ACTIVE WIDTH OVER TIME AND SPACE
plt.figure(figsize=(12, 5))
sns.heatmap(active_width, cmap="viridis", cbar_kws={"label": "Active width (m)"})
plt.title("Active Estuary Width Over Time and Space")
plt.xlabel("Estuary axis (x)")
plt.ylabel("Timestep")
plt.tight_layout()

if save_figure:
    plt.savefig(os.path.join(save_dir, "active_width_heatmap.png"))
    
plt.show()
 #%% CALCULATE WIDTH-AVERAGED BED LEVEL PROFILE FOR ENTIRE ESTUARY AT FINAL TIMESTEP

print("Computing width-averaged bed level profile for entire estuary...")

# 1. Get the x and y coordinates (same for all timesteps)
x0 = x[:, 0]  # x-coordinates along the estuary (shape: ny)
y0 = y[0, :]  # y-coordinates across the estuary (shape: nx)

# 2. Find indices within the estuary bounds
x_indices = np.where((x0 >= x_min) & (x0 <= x_max))[0]
y_indices = np.where((y0 >= y_min) & (y0 <= y_max))[0]

print(f"Found {len(x_indices)} x-indices and {len(y_indices)} y-indices within estuary bounds")

# 3. Get the final timestep bed level data
final_timestep = -1  # Last timestep
bedlev_final = bedlev[final_timestep, :, :]  # shape: (ny, nx)

# 4. Extract estuary region
bedlev_estuary = bedlev_final[np.ix_(x_indices, y_indices)]  # shape: (len(x_indices), len(y_indices))

# 5. Mask land areas (bed level >= 6 m)
bedlev_estuary_masked = np.where(bedlev_estuary < bed_threshold, bedlev_estuary, np.nan)

# 6. Compute width-averaged bed level for each x-location
width_averaged_bedlev = np.nanmean(bedlev_estuary_masked, axis=1)  # Average across y-direction

# 7. Get corresponding x-coordinates
x_coords_estuary = x0[x_indices]

# 8. Remove any NaN values (locations where all y-values were land)
valid_indices = ~np.isnan(width_averaged_bedlev)
x_coords_clean = x_coords_estuary[valid_indices]
width_averaged_bedlev_clean = width_averaged_bedlev[valid_indices]

print(f"Successfully computed width-averaged bed level for {len(x_coords_clean)} x-locations")

#%% VISUALIZATION
plt.figure(figsize=(12, 6))

# Convert x-coordinates to km for better readability
x_coords_km = x_coords_clean / 1000

plt.plot(x_coords_km, width_averaged_bedlev_clean, 'b-', linewidth=2, marker='o', markersize=3)
plt.xlabel('Distance along estuary [km]')
plt.ylabel('Width-averaged bed level [m]')
plt.title(f'Width-averaged bed level profile at final timestep\n{scenario} - {runname}')
plt.grid(True, alpha=0.3)

# Add some styling
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='MSL')
plt.legend()

# Invert y-axis if you want deeper areas to appear lower
# plt.gca().invert_yaxis()

plt.tight_layout()

if save_figure:
    plt.savefig(os.path.join(save_dir, f'width_averaged_bedlevel_profile_final_timestep_{scenario}.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_dir}")

plt.show()

#%% ADDITIONAL ANALYSIS (OPTIONAL)

# # Print some statistics
# print("\n=== Width-averaged bed level statistics ===")
# print(f"Minimum bed level: {np.min(width_averaged_bedlev_clean):.2f} m")
# print(f"Maximum bed level: {np.max(width_averaged_bedlev_clean):.2f} m")
# print(f"Mean bed level: {np.mean(width_averaged_bedlev_clean):.2f} m")
# print(f"Standard deviation: {np.std(width_averaged_bedlev_clean):.2f} m")

# # Find deepest and shallowest locations
# min_idx = np.argmin(width_averaged_bedlev_clean)
# max_idx = np.argmax(width_averaged_bedlev_clean)

# print(f"Deepest location: x = {x_coords_clean[min_idx]/1000:.1f} km, depth = {width_averaged_bedlev_clean[min_idx]:.2f} m")
# print(f"Shallowest location: x = {x_coords_clean[max_idx]/1000:.1f} km, depth = {width_averaged_bedlev_clean[max_idx]:.2f} m")

# # SAVE DATA (OPTIONAL) - MULTIPLE TIMESTEPS INCLUDING STD
# # Save the profile data for future use
# output_data = {
#     'selected_timesteps': selected_timesteps,
#     'timestep_labels': timestep_labels,
#     'width_averaged_profiles': width_averaged_profiles,
#     'width_std_profiles': width_std_profiles,  # NEW: Include std profiles
#     'valid_x_coordinates': valid_x_coords,  # List of arrays
#     'scenario': scenario,
#     'runname': runname,
#     'morphological_times_days': [morph_days[ts] if ts < len(morph_days) else None for ts in selected_timesteps],
#     'num_timesteps_analyzed': len(selected_timesteps),
#     'estuary_bounds': {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max},
#     'bed_threshold': bed_threshold
# }

# # Save as numpy file
# np.save(os.path.join(save_dir, f'width_averaged_profiles_and_std_evolution_{scenario}.npy'), output_data)
# print(f"\nProfile evolution data (including std) saved to {save_dir}")

# # Also save as text files for easy import into other software
# for i, (mean_profile, std_profile, x_coords, timestep) in enumerate(zip(width_averaged_profiles, width_std_profiles, valid_x_coords, selected_timesteps)):
#     filename = f'width_mean_std_profile_timestep_{timestep:03d}_{scenario}.txt'
    
#     # Calculate coefficient of variation
#     mean_abs = np.abs(mean_profile)
#     cv = np.where(mean_abs > 0.1, std_profile / mean_abs, np.nan)
    
#     # Save combined data
#     np.savetxt(os.path.join(save_dir, filename), 
#             np.column_stack((x_coords/1000, mean_profile, std_profile, cv)),
#             header='x_coordinate_km\tmean_bedlevel_m\tstd_bedlevel_m\tcoeff_variation',
#             delimiter='\t',
#             fmt='%.4f')

# print(f"Individual timestep profiles (mean + std) saved as text files")