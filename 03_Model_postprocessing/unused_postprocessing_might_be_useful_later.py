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

#%% 
"""Delft3D-4 Flow NetCDF Analysis: Morphological Estuary Analysis, for multiple scenarios in one plot.
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
discharge = 500 

# Scenario options: 
scenarios = [
    f'01_baserun{discharge}',
    f'02_run{discharge}_seasonal'#,
    #f'03_run{discharge}_flashy'
    ]

# Run options: 
runname = f's2_{discharge}_Wup_300m' 

# Which range of TIMESTEPS do you want to post-process? 
slice_start = 1     
slice_end = 361

# How many map plots to make?
amount_to_plot = 2

# Model run time settings (based on .mdf file)
reference_date = datetime.datetime(2024, 1, 1)  # Itdate = #2024-01-01#
Tstart = 2.628e6
Tstop = 3.1464e6
total_duration = Tstop - Tstart
total_duration_days = total_duration / (60 * 24)

map_output_interval = 1440    # minutes (Flmap interval)
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
model_location = r"U:\PhDNaturalRhythmEstuaries\Models\05_RiverDischargeVariability_domain45x15"

save_figure = True

#%% POST-PROCESSING SETTINGS: TICK WHAT YOU WANT TO RUN
run_spatial_plots = True                # Set to True to run, set to False to skip this section
run_width_averaged_bedlevel = True     # Set to True to run, set to False to skip this section
run_braiding_analysis = True           # Set to True to run, set to False to skip this section

# Dictionary to store results for all scenarios
results = {}

# Define colors and markers for visualization
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
scenario_labels = ['Baserun', 'Seasonal', 'Flashy']
markers = ['o', 's', '^']   # Circle, Square, Triangle
linestyles = ['-', '--', '-.']

#%% PROCESS ALL SCENARIOS
for i, scenario in enumerate(scenarios):
    print(f"\n{'='*60}")
    print(f"PROCESSING SCENARIO {i+1}/{len(scenarios)}: {scenario}")
    print(f"{'='*60}")
    
    # Define file paths for current scenario
    trim_file = os.path.join(model_location, runname, scenario, 'trim-varriver_tidewest.nc') 
    trih_file = os.path.join(model_location, runname, scenario, 'trih-varriver_tidewest.nc')
    save_dir = os.path.join(model_location, runname, scenario, 'postprocessing_plots')
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(trim_file):
        print(f"WARNING: Trim file not found for {scenario}. Skipping...")
        continue
        
    # LOAD FULL DATASETS FOR CURRENT SCENARIO
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

    # LOAD VARIABLES FOR CURRENT SCENARIO
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

    variable = 'TAUMAX'
    long_name = 'shear stress'

    print(f"Loading full {variable} ({long_name}) variable...")
    start_time = time.time()
    tau_max = load_variable(dataset_trim, variable, range=slice(slice_start, slice_end))
    print(f'{variable} variable shape is: {tau_max.shape}')
    print(f"Full {variable} loaded in {time.time() - start_time:.2f} seconds")

    # Convert morphological time in days since Tstart to actual dates
    morph_days = np.array(morphtime[:])
    morph_datetimes = np.array([reference_date + datetime.timedelta(days=float(day)) for day in morph_days])

    # Get cross-section coordinates for braiding index and plotting
    col_indices, N_coords, x_targets = get_cross_section_coordinates(x, y, x_targets = np.arange(x_min, x_max, 100))
    print(f"Cross-sections defined at x-coordinates: {x_targets}")

    # 1a. SPATIAL MAP PLOTS (BED LEVEL, WATER LEVEL, ...) 
    print("\n=== SPATIAL PLOTS ===")
    timesteps = np.linspace(slice_start, slice_end-1, amount_to_plot).astype(int)

    if run_spatial_plots:
        for timestep in timesteps: 
            print(f"Loading data for timestep {timestep}...")
            
            bed_level = load_single_timestep_variable(dataset_trim, "DPS", timestep=timestep)
            water_level = load_single_timestep_variable(dataset_trim, "S1", timestep=timestep)
            water_depth = water_level + bed_level
            
            print("Creating spatial plots...")
            plot_map(x, y, bed_level, 'bed_level', col_indices, N_coords, timestep, scenario, save_dir, save_figure)
            plot_map(x, y, water_level, 'water_level', col_indices, N_coords, timestep, scenario, save_dir, save_figure)
            plot_map(x, y, water_depth, 'water_depth', col_indices, N_coords, timestep, scenario, save_dir, save_figure)
        
        # Store spatial data for difference maps (last timestep only)
        last_timestep_idx = slice_end

        if last_timestep_idx < bedlev.shape[0]:
            bed_level_last = load_single_timestep_variable(dataset_trim, "DPS", timestep=last_timestep_idx)
            
            if scenario not in results:
                results[scenario] = {}
            
            results[scenario]['spatial_data'] = {
                'bed_level_last': bed_level_last,
                'x': x,
                'y': y,
                'col_indices': col_indices,
                'N_coords': N_coords,
                'last_timestep': slice_end - 1
            }
            print(f"Stored spatial data for difference maps (timestep {slice_end - 1})")

        print("SPATIAL PLOTS completed.")
    else:
        print('SPATIAL PLOTS skipped.')

    # 2a. WIDTH-AVERAGED BED LEVEL ALONG ESTUARY FOR MULTIPLE TIMESTEPS
    print("\n=== WIDTH-AVERAGED BED LEVEL PROFILE ===")

    if run_width_averaged_bedlevel: 
        # Configuration for multiple timesteps
        num_timesteps_to_plot = 2  # Number of timesteps you want to analyze

        print(f"Computing width-averaged bed level profile for {num_timesteps_to_plot} timesteps...")

        # 1. Get the x and y coordinates (same for all timesteps)
        x0 = x[:, 0]  # x-coordinates along the estuary (shape: ny)
        y0 = y[0, :]  # y-coordinates across the estuary (shape: nx)

        # 2. Find indices within the estuary bounds
        x_indices = np.where((x0 >= x_min) & (x0 <= x_max))[0]
        y_indices = np.where((y0 >= y_min) & (y0 <= y_max))[0]

        print(f"Found {len(x_indices)} x-indices and {len(y_indices)} y-indices within estuary bounds")

        # 3. Select timesteps - evenly spaced including first and last
        total_timesteps = bedlev.shape[0]
        if num_timesteps_to_plot >= total_timesteps:
            selected_timesteps = np.arange(total_timesteps)
        else:
            selected_timesteps = np.linspace(0, total_timesteps-1, num_timesteps_to_plot, dtype=int)

        print(f"Selected timesteps: {selected_timesteps} out of {total_timesteps} total timesteps")

        # 4. Get corresponding x-coordinates (will be same for all timesteps)
        x_coords_estuary = x0[x_indices]

        # 5. Initialize arrays to store results
        width_averaged_profiles = []
        width_std_profiles = []
        valid_x_coords = []
        timestep_labels = []

        # 6. Process each selected timestep
        for j, timestep in enumerate(selected_timesteps):
            print(f"Processing timestep {timestep} ({j+1}/{len(selected_timesteps)})...")
            
            # Get bed level data for this timestep
            bedlev_current = bedlev[timestep, :, :]  # shape: (ny, nx)
            
            # Extract estuary region
            bedlev_estuary = bedlev_current[np.ix_(x_indices, y_indices)]  # shape: (len(x_indices), len(y_indices))
            
            # Mask land areas (bed level >= 6 m)
            bedlev_estuary_masked = np.where(bedlev_estuary < bed_threshold, bedlev_estuary, np.nan)
            
            # Compute width-averaged bed level and standard deviation for each x-location
            width_averaged_bedlev = np.nanmean(bedlev_estuary_masked, axis=1)  # Average across y-direction
            width_std_bedlev = np.nanstd(bedlev_estuary_masked, axis=1)        # Standard deviation across y-direction

            # Remove any NaN values (locations where all y-values were land)
            valid_indices = ~np.isnan(width_averaged_bedlev)
            
            x_coords_clean = x_coords_estuary[valid_indices]
            width_averaged_bedlev_clean = width_averaged_bedlev[valid_indices]
            width_std_bedlev_clean = width_std_bedlev[valid_indices]
            
            # Store results
            width_averaged_profiles.append(width_averaged_bedlev_clean)
            width_std_profiles.append(width_std_bedlev_clean)
            valid_x_coords.append(x_coords_clean)
                
            # Create label with morphological time if available
            if timestep < len(morph_days):
                morph_time_years = morph_days[timestep] / 365.25
                timestep_labels.append(f't = {timestep} ({morph_time_years:.0f} years)')
            else:
                timestep_labels.append(f't = {timestep}')

        print(f"Successfully computed width-averaged bed level profiles for {len(selected_timesteps)} timesteps")

        # Store results for current scenario
        results[scenario] = {
            'width_averaged_profiles': width_averaged_profiles,
            'width_std_profiles': width_std_profiles,
            'valid_x_coords': valid_x_coords,
            'timestep_labels': timestep_labels,
            'selected_timesteps': selected_timesteps,
            'morph_datetimes': morph_datetimes[:len(selected_timesteps)] if len(morph_datetimes) >= len(selected_timesteps) else morph_datetimes,
            'scenario_label': scenario_labels[i]
        }
        
        print("WIDTH-AVERAGED BED LEVEL PROFILE completed.")
    else: 
        print('WIDTH-AVERAGED BED LEVEL PROFILE skipped.')

    # 2b. BRAIDING INDEX ALONG ESTUARY FOR MULTIPLE TIMESTEPS
    print("\n=== BRAIDING INDEX ANALYSIS ===")

    if run_braiding_analysis:
        print("Computing braiding index...")
        BI_per_cross_section, datetimes, _, _, _ = compute_BI_per_cross_section(
            x, y, tau_max, slice_start, map_output_interval, reference_date, theta=0.25, x_targets=x_targets
        )
        
        # Store braiding index results for current scenario
        if 'braiding_index' not in results.get(scenario, {}):
            results[scenario]['braiding_index'] = {}
        
        results[scenario]['braiding_index'] = {
            'BI_per_cross_section': BI_per_cross_section,
            'datetimes': datetimes,
            'x_targets': x_targets,
            'BI_first_timestep': BI_per_cross_section[0] if len(BI_per_cross_section) > 0 else None,  # t=0
            'BI_last_timestep': BI_per_cross_section[-1] if len(BI_per_cross_section) > 0 else None   # final t
        }
        
        print("BRAIDING INDEX ANALYSIS completed.")
    else:
        print('BRAIDING INDEX ANALYSIS skipped.')
    
    # Close datasets to free memory
    dataset_trim.close()
    dataset_trih.close()
    
    print(f"Scenario {scenario} processed successfully!")

#%% CREATE COMPREHENSIVE VISUALIZATIONS
print(f"\n{'='*60}")
print("CREATING COMPREHENSIVE VISUALIZATIONS")
print(f"{'='*60}")

# Create combined save directory
save_dir_combined = os.path.join(model_location, runname, 'combined_scenarios')
os.makedirs(save_dir_combined, exist_ok=True)

#%% PLOT 1: WIDTH-AVERAGED BED LEVEL COMPARISON ACROSS SCENARIOS
if run_width_averaged_bedlevel and results:
    print("Creating width-averaged bed level comparison plots...")
    
    # Plot for each timestep, comparing all scenarios
    for ts_idx in range(len(list(results.values())[0]['selected_timesteps'])):
        plt.figure(figsize=(14, 8))
        
        for i, (scenario, data) in enumerate(results.items()):
            if ts_idx < len(data['width_averaged_profiles']):
                profile = data['width_averaged_profiles'][ts_idx]
                x_coords = data['valid_x_coords'][ts_idx]
                
                # Convert x-coordinates to km for better readability
                x_coords_km = x_coords / 1000
                
                plt.plot(x_coords_km, profile, 
                        color=colors[i], 
                        linestyle=linestyles[i],
                        marker=markers[i],
                        markersize=3,
                        markevery=max(1, len(profile)//15),
                        linewidth=2, 
                        label=f'{data["scenario_label"]}',
                        alpha=0.8)

        timestep_label = list(results.values())[0]['timestep_labels'][ts_idx]

        plt.xlabel('Distance along estuary [km]', fontsize=12)
        plt.ylabel('Width-averaged bed level [m]', fontsize=12)
        plt.title(f'Width-averaged bed level comparison across scenarios\n{timestep_label} - Q = {discharge} m³/s', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(os.path.join(save_dir_combined, f'width_averaged_comparison_timestep_{ts_idx}_Q{discharge}.png'), 
                        dpi=300, bbox_inches='tight')
        plt.show()

    # Plot evolution over time for each scenario (all timesteps on one plot)
    plt.figure(figsize=(16, 10))
    
    for i, (scenario, data) in enumerate(results.items()):
        for ts_idx, profile in enumerate(data['width_averaged_profiles']):
            x_coords = data['valid_x_coords'][ts_idx]
            x_coords_km = x_coords / 1000
            
            alpha = 0.4 + 0.6 * (ts_idx / (len(data['width_averaged_profiles']) - 1))  # Increasing alpha over time
            line_width = 1.5 + 1 * (ts_idx / (len(data['width_averaged_profiles']) - 1))  # Increasing line width
            
            if ts_idx == 0:  # Only label the first line for each scenario
                label = f'{data["scenario_label"]}'
            else:
                label = None
                
            plt.plot(x_coords_km, profile, 
                    color=colors[i], 
                    linestyle=linestyles[i],
                    linewidth=line_width,
                    alpha=alpha,
                    label=label)

    plt.xlabel('Distance along estuary [km]', fontsize=12)
    plt.ylabel('Width-averaged bed level [m]', fontsize=12)
    plt.title(f'Width-averaged bed level evolution comparison\nAll scenarios and timesteps - Q = {discharge} m³/s', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_figure:
        plt.savefig(os.path.join(save_dir_combined, f'width_averaged_evolution_all_scenarios_Q{discharge}.png'), 
                    dpi=300, bbox_inches='tight')
    plt.show()

#%% PLOT 2: STANDARD DEVIATION COMPARISON
if run_width_averaged_bedlevel and results:
    print("Creating standard deviation comparison plots...")
    
    # Plot for each timestep, comparing all scenarios
    for ts_idx in range(len(list(results.values())[0]['selected_timesteps'])):
        plt.figure(figsize=(14, 8))
        
        for i, (scenario, data) in enumerate(results.items()):
            if ts_idx < len(data['width_std_profiles']):
                std_profile = data['width_std_profiles'][ts_idx]
                x_coords = data['valid_x_coords'][ts_idx]
                
                # Convert x-coordinates to km for better readability
                x_coords_km = x_coords / 1000
                
                plt.plot(x_coords_km, std_profile, 
                        color=colors[i], 
                        linestyle=linestyles[i],
                        marker=markers[i],
                        markersize=3,
                        markevery=max(1, len(std_profile)//15),
                        linewidth=2, 
                        label=f'{data["scenario_label"]}',
                        alpha=0.8)

        timestep_label = list(results.values())[0]['timestep_labels'][ts_idx]
        plt.xlabel('Distance along estuary [km]', fontsize=12)
        plt.ylabel('Standard deviation of bed level [m]', fontsize=12)
        plt.title(f'Bed level variability comparison across scenarios\n{timestep_label} - Q = {discharge} m³/s', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(os.path.join(save_dir_combined, f'width_std_comparison_timestep_{ts_idx}_Q{discharge}.png'), 
                        dpi=300, bbox_inches='tight')
        plt.show()

#%% PLOT 3: BRAIDING INDEX COMPARISON
if run_braiding_analysis and results:
    print("Creating braiding index evolution comparison...")
    
    # Find baserun scenario
    baserun_scenario = None
    baserun_data = None
    
    for scenario, data in results.items():
        if 'baserun' in scenario.lower() and 'braiding_index' in data:
            baserun_scenario = scenario
            baserun_data = data
            break
    
    if baserun_scenario and baserun_data:
        print(f"Using {baserun_scenario} for initial and final baserun conditions")
        
        # Get baserun data
        baserun_BI_initial = baserun_data['braiding_index']['BI_first_timestep']  # t=0
        baserun_BI_final = baserun_data['braiding_index']['BI_last_timestep']    # t=end
        x_targets = baserun_data['braiding_index']['x_targets']
        x_targets_km = x_targets / 1000
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Plot baserun t=0 (initial condition)
        plt.plot(x_targets_km, baserun_BI_initial, 
                color='black', 
                linestyle='-',
                marker='o',
                markersize=4,
                markevery=max(1, len(baserun_BI_initial)//20),
                linewidth=3, 
                label='Baserun t=0 (initial)',
                alpha=0.9)
        
        # Plot baserun t=end 
        plt.plot(x_targets_km, baserun_BI_final, 
                color='gray', 
                linestyle='-',
                marker='s',
                markersize=4,
                markevery=max(1, len(baserun_BI_final)//20),
                linewidth=3, 
                label='Baserun t=end',
                alpha=0.9)
        
        # Plot final condition for each other scenario (t=end)
        for i, (scenario, data) in enumerate(results.items()):
            if scenario != baserun_scenario and 'braiding_index' in data and data['braiding_index']['BI_last_timestep'] is not None:
                BI_final = data['braiding_index']['BI_last_timestep']
                
                plt.plot(x_targets_km, BI_final, 
                        color=colors[i], 
                        linestyle=linestyles[i],
                        marker=markers[i],
                        markersize=4,
                        markevery=max(1, len(BI_final)//20),
                        linewidth=2, 
                        label=f'{data["scenario_label"]} t=end',
                        alpha=0.8)

        plt.xlabel('Distance along estuary [km]', fontsize=12)
        plt.ylabel('Braiding Index', fontsize=12)
        plt.title(f'Braiding Index: Initial vs Final Conditions\nQ = {discharge} m³/s', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_figure:
            plt.savefig(os.path.join(save_dir_combined, f'braiding_index_evolution_Q{discharge}.png'), 
                        dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"\nBraiding Index Statistics:")
        print(f"Baserun t=0: Mean={np.nanmean(baserun_BI_initial):.3f}, Std={np.nanstd(baserun_BI_initial):.3f}")
        print(f"Baserun t=end: Mean={np.nanmean(baserun_BI_final):.3f}, Std={np.nanstd(baserun_BI_final):.3f}")
        
        for scenario, data in results.items():
            if scenario != baserun_scenario and 'braiding_index' in data and data['braiding_index']['BI_last_timestep'] is not None:
                BI_final = data['braiding_index']['BI_last_timestep']
                print(f"{data['scenario_label']} t=end: Mean={np.nanmean(BI_final):.3f}, Std={np.nanstd(BI_final):.3f}")
    
    else:
        print("WARNING: No baserun scenario found or braiding index data missing. Skipping braiding index evolution plot.")

# if run_braiding_analysis and results:
#     print("Creating braiding index comparison plots...")
    
#     plt.figure(figsize=(16, 10))
    
#     for i, (scenario, data) in enumerate(results.items()):
#         if 'braiding_index' in data:
#             BI_data = data['braiding_index']['BI_per_cross_section']
#             x_targets = data['braiding_index']['x_targets']
            
#             # Calculate mean braiding index across all timesteps for each cross-section
#             mean_BI = np.nanmean(BI_data, axis=0)
#             std_BI = np.nanstd(BI_data, axis=0)
            
#             # Convert x-coordinates to km
#             x_targets_km = x_targets / 1000
            
#             plt.plot(x_targets_km, mean_BI, 
#                     color=colors[i], 
#                     linestyle=linestyles[i],
#                     marker=markers[i],
#                     markersize=4,
#                     markevery=max(1, len(mean_BI)//20),
#                     linewidth=2, 
#                     label=f'{data["scenario_label"]} (mean)',
#                     alpha=0.8)
            
#             # Add error bars for standard deviation
#             plt.fill_between(x_targets_km, mean_BI - std_BI, mean_BI + std_BI, 
#                            color=colors[i], alpha=0.2)

#     plt.xlabel('Distance along estuary [km]', fontsize=12)
#     plt.ylabel('Mean Braiding Index', fontsize=12)
#     plt.title(f'Mean braiding index comparison across scenarios\nQ = {discharge} m³/s', fontsize=14)
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
    
#     if save_figure:
#         plt.savefig(os.path.join(save_dir_combined, f'braiding_index_comparison_Q{discharge}.png'), 
#                     dpi=300, bbox_inches='tight')
#     plt.show()

#%% PLOT 4: SPATIAL DIFFERENCE MAPS (BED LEVEL) - LAST TIMESTEP ONLY
print("Creating spatial difference maps...")

# Find baserun scenario
baserun_scenario = None
baserun_data = None
for scenario, data in results.items():
    if 'baserun' in scenario.lower():
        baserun_scenario = scenario
        baserun_data = data
        break

if baserun_scenario and 'spatial_data' in baserun_data:
    print(f"Using {baserun_scenario} as reference for difference maps")
    baserun_bed = baserun_data['spatial_data']['bed_level_last']
    x = baserun_data['spatial_data']['x']
    y = baserun_data['spatial_data']['y']
    col_indices = baserun_data['spatial_data']['col_indices']
    N_coords = baserun_data['spatial_data']['N_coords']
    last_timestep = baserun_data['spatial_data']['last_timestep']
    
    # Create difference maps for non-baserun scenarios
    for scenario, data in results.items():
        if scenario != baserun_scenario and 'spatial_data' in data:
            scenario_bed = data['spatial_data']['bed_level_last']
            
            # Calculate difference (scenario - baserun)
            bed_difference = scenario_bed - baserun_bed
            
            # Create difference plot
            plt.figure(figsize=(14, 10))
            
            # Create the plot
            vmin, vmax = np.nanpercentile(bed_difference, [2, 98])  # Use 2-98 percentile for better contrast
            im = plt.pcolormesh(x, y, bed_difference, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('Bed level difference [m]\n(Scenario - Baserun)', fontsize=12)
            
            # Add cross-section lines if available
            if col_indices is not None and N_coords is not None:
                for i, (col_idx, N_coord) in enumerate(zip(col_indices, N_coords)):
                    if i % 50 == 0:  # Show every 50th cross-section to avoid clutter
                        plt.plot(x[col_idx, :N_coord], y[col_idx, :N_coord], 'k-', alpha=0.3, linewidth=0.5)
            
            # Set labels and title
            plt.xlabel('X coordinate [m]', fontsize=12)
            plt.ylabel('Y coordinate [m]', fontsize=12)
            plt.title(f'Bed level difference at final timestep (t={last_timestep})\n{data["scenario_label"]} - Baserun (Q = {discharge} m³/s)', fontsize=14)
            
            # Set aspect ratio and limits
            plt.axis('equal')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            
            plt.tight_layout()
            
            # Save the plot
            if save_figure:
                difference_filename = f'bed_difference_{scenario}_vs_baserun_timestep_{last_timestep}_Q{discharge}.png'
                plt.savefig(os.path.join(save_dir_combined, difference_filename), 
                           dpi=300, bbox_inches='tight')
                print(f"Difference map saved: {difference_filename}")
            
            plt.show()
            
            # Print some statistics about the differences
            print(f"\nBed level difference statistics for {data['scenario_label']}:")
            print(f"  Mean difference: {np.nanmean(bed_difference):.4f} m")
            print(f"  Std difference: {np.nanstd(bed_difference):.4f} m")
            print(f"  Max erosion (negative): {np.nanmin(bed_difference):.4f} m")
            print(f"  Max deposition (positive): {np.nanmax(bed_difference):.4f} m")
            print(f"  RMS difference: {np.sqrt(np.nanmean(bed_difference**2)):.4f} m")

    # Create a combined difference plot showing all scenarios
    n_non_baserun = sum(1 for s in results.keys() if s != baserun_scenario and 'spatial_data' in results[s])
    
    if n_non_baserun > 1:
        fig, axes = plt.subplots(1, n_non_baserun, figsize=(7*n_non_baserun, 6))
        if n_non_baserun == 1:
            axes = [axes]
        
        # Calculate global vmin, vmax for consistent color scaling
        all_differences = []
        for scenario, data in results.items():
            if scenario != baserun_scenario and 'spatial_data' in data:
                scenario_bed = data['spatial_data']['bed_level_last']
                bed_difference = scenario_bed - baserun_bed
                all_differences.append(bed_difference)
        
        if all_differences:
            global_vmin, global_vmax = np.nanpercentile(np.concatenate([diff.flatten() for diff in all_differences]), [2, 98])
        
        plot_idx = 0
        for scenario, data in results.items():
            if scenario != baserun_scenario and 'spatial_data' in data:
                scenario_bed = data['spatial_data']['bed_level_last']
                bed_difference = scenario_bed - baserun_bed
                
                ax = axes[plot_idx]
                
                # Create the plot
                im = ax.pcolormesh(x, y, bed_difference, cmap='RdBu_r', 
                                  vmin=global_vmin, vmax=global_vmax, shading='auto')
                
                # Add cross-section lines
                if col_indices is not None and N_coords is not None:
                    for i, (col_idx, N_coord) in enumerate(zip(col_indices, N_coords)):
                        if i % 50 == 0:
                            ax.plot(x[col_idx, :N_coord], y[col_idx, :N_coord], 'k-', alpha=0.3, linewidth=0.5)
                
                # Set labels and title
                ax.set_xlabel('X coordinate [m]', fontsize=10)
                ax.set_ylabel('Y coordinate [m]', fontsize=10)
                ax.set_title(f'{data["scenario_label"]} - Baserun', fontsize=12)
                
                # Set aspect ratio and limits
                ax.set_aspect('equal')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                plot_idx += 1
        
        # Add a single colorbar for all subplots
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Bed level difference [m]\n(Scenario - Baserun)', fontsize=12)
        
        fig.suptitle(f'Bed level differences at final timestep (t={last_timestep})\nQ = {discharge} m³/s', fontsize=14)
        
        # Save the combined plot
        if save_figure:
            combined_filename = f'bed_differences_combined_timestep_{last_timestep}_Q{discharge}.png'
            plt.savefig(os.path.join(save_dir_combined, combined_filename), 
                       dpi=300, bbox_inches='tight')
            print(f"Combined difference map saved: {combined_filename}")
        
        plt.show()

else:
    print("WARNING: No baserun scenario found or spatial data missing. Skipping difference maps.")

# print("Creating spatial difference maps...")

# # Find baserun scenario
# baserun_scenario = None
# baserun_data = None
# for scenario, data in results.items():
#     if 'baserun' in scenario.lower():
#         baserun_scenario = scenario
#         baserun_data = data
#         break

# if baserun_scenario and 'spatial_data' in baserun_data:
#     print(f"Using {baserun_scenario} as reference for difference maps")
#     baserun_bed = baserun_data['spatial_data']['bed_level_last']
#     x = baserun_data['spatial_data']['x']
#     y = baserun_data['spatial_data']['y']
#     col_indices = baserun_data['spatial_data']['col_indices']
#     N_coords = baserun_data['spatial_data']['N_coords']
#     last_timestep = baserun_data['spatial_data']['last_timestep']
    
#     # Create difference maps for non-baserun scenarios
#     for scenario, data in results.items():
#         if scenario != baserun_scenario and 'spatial_data' in data:
#             scenario_bed = data['spatial_data']['bed_level_last']
            
#             # Calculate difference (scenario - baserun)
#             bed_difference = scenario_bed - baserun_bed
            
#             # Create difference plot
#             plt.figure(figsize=(14, 10))
            
#             # Create the plot
#             vmin, vmax = np.nanpercentile(bed_difference, [2, 98])  # Use 2-98 percentile for better contrast
#             im = plt.pcolormesh(x, y, bed_difference, cmap='RdBu_r', vmin=vmin, vmax=vmax, shading='auto')
            
#             # Add colorbar
#             cbar = plt.colorbar(im, shrink=0.8)
#             cbar.set_label('Bed level difference [m]\n(Scenario - Baserun)', fontsize=12)
            
#             # Add cross-section lines if available
#             if col_indices is not None and N_coords is not None:
#                 for i, (col_idx, N_coord) in enumerate(zip(col_indices, N_coords)):
#                     if i % 50 == 0:  # Show every 50th cross-section to avoid clutter
#                         plt.plot(x[col_idx, :N_coord], y[col_idx, :N_coord], 'k-', alpha=0.3, linewidth=0.5)
            
#             # Set labels and title
#             plt.xlabel('X coordinate [m]', fontsize=12)
#             plt.ylabel('Y coordinate [m]', fontsize=12)
#             plt.title(f'Bed level difference at final timestep (t={last_timestep})\n{data["scenario_label"]} - Baserun (Q = {discharge} m³/s)', fontsize=14)
            
#             # Set aspect ratio and limits
#             plt.axis('equal')
#             plt.xlim(x_min, x_max)
#             plt.ylim(y_min, y_max)
            
#             plt.tight_layout()
            
#             # Save the plot
#             if save_figure:
#                 difference_filename = f'bed_difference_{scenario}_vs_baserun_timestep_{last_timestep}_Q{discharge}.png'
#                 plt.savefig(os.path.join(save_dir_combined, difference_filename), 
#                            dpi=300, bbox_inches='tight')
#                 print(f"Difference map saved: {difference_filename}")
            
#             plt.show()
            
#             # Print some statistics about the differences
#             print(f"\nBed level difference statistics for {data['scenario_label']}:")
#             print(f"  Mean difference: {np.nanmean(bed_difference):.4f} m")
#             print(f"  Std difference: {np.nanstd(bed_difference):.4f} m")
#             print(f"  Max erosion (negative): {np.nanmin(bed_difference):.4f} m")
#             print(f"  Max deposition (positive): {np.nanmax(bed_difference):.4f} m")
#             print(f"  RMS difference: {np.sqrt(np.nanmean(bed_difference**2)):.4f} m")

#     # Create a combined difference plot showing all scenarios
#     n_non_baserun = sum(1 for s in results.keys() if s != baserun_scenario and 'spatial_data' in results[s])
    
#     if n_non_baserun > 1:
#         fig, axes = plt.subplots(1, n_non_baserun, figsize=(7*n_non_baserun, 6))
#         if n_non_baserun == 1:
#             axes = [axes]
        
#         # Calculate global vmin, vmax for consistent color scaling
#         all_differences = []
#         for scenario, data in results.items():
#             if scenario != baserun_scenario and 'spatial_data' in data:
#                 scenario_bed = data['spatial_data']['bed_level_last']
#                 bed_difference = scenario_bed - baserun_bed
#                 all_differences.append(bed_difference)
        
#         if all_differences:
#             global_vmin, global_vmax = np.nanpercentile(np.concatenate([diff.flatten() for diff in all_differences]), [2, 98])
        
#         plot_idx = 0
#         for scenario, data in results.items():
#             if scenario != baserun_scenario and 'spatial_data' in data:
#                 scenario_bed = data['spatial_data']['bed_level_last']
#                 bed_difference = scenario_bed - baserun_bed
                
#                 ax = axes[plot_idx]
                
#                 # Create the plot
#                 im = ax.pcolormesh(x, y, bed_difference, cmap='RdBu_r', 
#                                   vmin=global_vmin, vmax=global_vmax, shading='auto')
                
#                 # Add cross-section lines
#                 if col_indices is not None and N_coords is not None:
#                     for i, (col_idx, N_coord) in enumerate(zip(col_indices, N_coords)):
#                         if i % 50 == 0:
#                             ax.plot(x[col_idx, :N_coord], y[col_idx, :N_coord], 'k-', alpha=0.3, linewidth=0.5)
                
#                 # Set labels and title
#                 ax.set_xlabel('X coordinate [m]', fontsize=10)
#                 ax.set_ylabel('Y coordinate [m]', fontsize=10)
#                 ax.set_title(f'{data["scenario_label"]} - Baserun', fontsize=12)
                
#                 # Set aspect ratio and limits
#                 ax.set_aspect('equal')
#                 ax.set_xlim(x_min, x_max)
#                 ax.set_ylim(y_min, y_max)
                
#                 plot_idx += 1
        
#         # Add a single colorbar for all subplots
#         fig.subplots_adjust(right=0.85)
#         cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
#         cbar = fig.colorbar(im, cax=cbar_ax)
#         cbar.set_label('Bed level difference [m]\n(Scenario - Baserun)', fontsize=12)
        
#         fig.suptitle(f'Bed level differences at final timestep (t={last_timestep})\nQ = {discharge} m³/s', fontsize=14)
        
#         # Save the combined plot
#         if save_figure:
#             combined_filename = f'bed_differences_combined_timestep_{last_timestep}_Q{discharge}.png'
#             plt.savefig(os.path.join(save_dir_combined, combined_filename), 
#                        dpi=300, bbox_inches='tight')
#             print(f"Combined difference map saved: {combined_filename}")
        
#         plt.show()

# else:
#     print("WARNING: No baserun scenario found or spatial data missing. Skipping difference maps.")

#%% SUMMARY STATISTICS
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")

for scenario, data in results.items():
    print(f"\n{data['scenario_label']} ({scenario}):")
    
    if 'width_averaged_profiles' in data:
        # Calculate statistics for width-averaged profiles
        for ts_idx, profile in enumerate(data['width_averaged_profiles']):
            timestep_label = data['timestep_labels'][ts_idx]
            print(f"  {timestep_label}:")
            print(f"    Mean bed level: {np.mean(profile):.3f} m")
            print(f"    Min bed level: {np.min(profile):.3f} m")
            print(f"    Max bed level: {np.max(profile):.3f} m")
            print(f"    Std bed level: {np.std(profile):.3f} m")
    
    if 'braiding_index' in data:
        BI_data = data['braiding_index']['BI_per_cross_section']
        mean_BI_overall = np.nanmean(BI_data)
        std_BI_overall = np.nanstd(BI_data)
        print(f"  Braiding Index:")
        print(f"    Overall mean: {mean_BI_overall:.3f}")
        print(f"    Overall std: {std_BI_overall:.3f}")

print(f"\nAnalysis complete!")

if save_figure:
    print(f"Results saved in: {save_dir_combined}")
else: 
    print(f"Resuls are not saved: save_figure = False")

print(f"Processed {len(results)} scenarios successfully.")

#%%