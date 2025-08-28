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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Python\03_Model_postprocessing")

from FUNCTIONS.FUNCS_postprocessing_general import *
from FUNCTIONS.FUNCS_postprocessing_braiding_index import *
from FUNCTIONS.FUNCS_postprocessing_map_output import *
from FUNCTIONS.FUNCS_postprocessing_his_output import *
from FUNCTIONS.FUNCS_postprocessing_hypsometry import *

#%% CONFIGURATION
fontsize_labels = 16 - 4
fontsize_titles = fontsize_labels + 4
fontsize_axes = fontsize_labels - 2
fontcolor = 'black'

config = '04_RiverDischargeVariability_domain45x15'

# Which model do you want to post-process?
model_location = os.path.join(r"U:\PhDNaturalRhythmEstuaries\Models", config)

# All discharge scenarios to process
discharges = [500]  #[250, 500, 1000]

# Which DISCHARGE SCENARIO do you want to post-process?
discharge = discharges[0]
runname = get_runname(discharge) # returns runname in f-string format


# Scenario options (template - will be filled for each discharge)
scenario_templates = [#'00_testvalidity_velocity_1boundary']
        '01_baserun{discharge}', 
        '02_run{discharge}_seasonal', 
        '03_run{discharge}_flashy'
]

number = 2
scenario = scenario_templates[number].format(discharge=discharge)
scenario_name = scenario_templates[number].format(discharge=discharge)

# Which range of TIMESTEPS [MAP] do you want to post-process? 
slice_start = 1     
slice_end = 361

# How many SPATIAL MAP plots to make?
amount_to_plot = 4

# Do you want to save the figures?
save_figure = True

#%%  Model settings (based on .mdf file)

reference_date = datetime.datetime(2024, 1, 1)  # Itdate = #2024-01-01#
Tstart = 2.628e6

print(f'config = {config}')

if config == '04_RiverDischargeVariability_domain45x15':
    # 04_RiverDischargeVariability = 0.5 hydrodynamicyear runs 
    Tstop = 2.8908e6
    map_output_interval = 1300    # minutes (Flmap interval)
    his_output_interval = 720     # minutes (Flhis interval) 

elif config == '05_RiverDischargeVariability_domain45x15':
    # 05_RiverDischargeVariability = 1 hydrodynamic year runs
    map_output_interval = 1440    # minutes (Flmap interval)
    his_output_interval = 720     # minutes (Flhis interval)   
    Tstop = 3.1464e6

else:
    # Test velocity output runs
    map_output_interval = 30
    his_output_interval = 30
    Tstop = 2.628e6 + (24*60)

total_duration = Tstop - Tstart
total_duration_days = total_duration / (60 * 24)

map_output_interval_hours = map_output_interval / 60
his_output_interval_hours = his_output_interval / 60

#%% Estuary characteristics

# Define estuary bounds
x_min, x_max = 20000, 45000
y_min, y_max = 5000, 10000

# Define land threshold
bed_threshold = 6  # exclude land (higher than 6 m)

#%% File locations + loading

trim_file = os.path.join(model_location, runname, scenario_name, 'trim-varriver_tidewest.nc') 
trih_file = os.path.join(model_location, runname, scenario_name, 'trih-varriver_tidewest.nc')
save_dir = os.path.join(model_location, runname, scenario_name, 'postprocessing_plots')

os.makedirs(save_dir, exist_ok=True)
save_figure = True

# #Load full datasets
print("Loading trim_file...")
start_time = time.time()
dataset_trim = nc.Dataset(trim_file, mode='r')
print(f"Trim file loaded in {time.time() - start_time:.2f} seconds")

print("Loading trih_file...")
start_time = time.time()
dataset_trih = nc.Dataset(trih_file, mode='r')
print(f"Trih file loaded in {time.time() - start_time:.2f} seconds")

#%% ====================  HIS ANALYSIS ==================== 
# note: HIS output is always in seconds since ref-date, regardless of Tunit

# CROSS-SECTION ANALYSIS FOR ONE SCENARIO IN SEPERATE PLOTS 
print("\n=== DISCHARGE ANALYSIS ===")
run_single_discharge_analysis = False  # Set to False to skip this section

if run_single_discharge_analysis:
    variable = 'ZWL'  
    variable_label = 'Water Level [m]'

    # variable = 'CTR'  # Current transport rate
    # variable_label = 'Q [m³/s]'

    print(f"Analyzing {variable} data...")
    
    #STATION_NAMES for my model 
    station_names = [f'river_km_{i}' for i in range(3)] #use 27 for all cross-sections
    
    # Extract discharge data
    results, all_stations = extract_his_data(dataset_trih, variable, station_names)

    for name in station_names:
        time, var = results[name]
        print(f"{name} time range: {time.min()} to {time.max()}")
        print(f"{name} {variable} range: min={np.nanmin(var)}, max={np.nanmax(var)}")
        print(f"{name}: Number of time points = {len(time)}")

    # Plot discharge time series
    his_plot_timeseries(
        results, station_names, reference_date, variable, variable_label, save_dir, save_figure,
        time_range=(int(slice_start*map_output_interval/his_output_interval), 
                   int(slice_end*map_output_interval/his_output_interval))
    )
    
    print(f"{variable} analysis completed.")

#%%  CROSS-SECTION ANALYSIS FOR MULTIPLE SCENARIOS IN ONE (SUB)PLOT
print("\n=== MULTI-SCENARIO-VARIABLE ANALYSIS ===")
run_multi_variable_analysis = False

if run_multi_variable_analysis:
    # DEFINE WHICH VARIABLES TO ANALYZE
    analysis_configs = [
        {#! CTR is a transect/cross-section variable (NTRUV)
            'variable': 'q1',   # Use consistent naming
            'variable_label': 'Q [m³/s]',
            'netcdf_variable': 'CTR'
          }  # Variable name in NetCDF file
        #  },
        # { #! ZWL is a station variable (NOSTAT) <-- doesn't work for this approach.
        #     'variable': 'zwl',   # Water level
        #     'variable_label': 'Water Level [m]',
        #     'netcdf_variable': 'ZWL'  # adjust if different in your NetCDF
        # }
    ]
    
    # Time range settings
    time_start = 0
    time_end = 360
    
    # STATION_NAMES for your model 
    station_names = [f'river_km_{i}' for i in np.arange(0,27,4)] # [f'river_km_{i}' for i in range(27)] for all cross-sections
    
    # Loop through each variable to analyze
    for config in analysis_configs:
        print(f"\n--- Analyzing {config['variable']} ({config['variable_label']}) ---")
        
        # Store all results for this variable
        all_results = {}
        
        # Process each discharge scenario
        for discharge in discharges:
            print(f"\nProcessing scenario Q = {discharge}...")
            
            runname = get_runname(discharge)
            all_results[discharge] = {}
            
            # Process each scenario for this discharge
            for scenario_template in scenario_templates:
                scenario_name = scenario_template.format(discharge=discharge)
                print(f"  Processing scenario: {scenario_name}")
                
                # File paths
                trih_file = os.path.join(model_location, runname, scenario_name, 'trih-varriver_tidewest.nc')
                
                try:
                    # Load dataset
                    dataset_trih = nc.Dataset(trih_file, mode='r')
                    
                    # Extract data for this variable
                    results, all_stations = extract_his_data(dataset_trih, config['netcdf_variable'], station_names)
                    all_results[discharge][scenario_name] = results
                    
                    # Close dataset to free memory
                    dataset_trih.close()
                    
                except Exception as e:
                    print(f"    Error processing {scenario_name}: {e}")
                    all_results[discharge][scenario_name] = None
        
        # Create the plot for this variable using the generic function
        plot_detailed_multi_scenarios(
            all_results, discharges, scenario_templates, 
            station_names, model_location, 
            save_figure, time_start, time_end,
            get_runname,
            variable=config['variable'],
            variable_label=config['variable_label'],
            reference_date=reference_date
        )
        
        print(f"{config['variable']} analysis completed.")
    
    print("Multi-variable analysis completed.")

#%% ====================  MAP ANALYSIS ==================== 

# Get the shape of the a variable -- to figure out the length of the t, x, y arrays
print('trim shape (t, x, y) =', dataset_trim.variables['DPS'].shape)

print("Loading coordinates...")
x = load_variable(dataset_trim, "XCOR")
y = load_variable(dataset_trim, "YCOR")

nx, ny = x.shape
print(f"x/y shape: {x.shape}")
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
tau_max = load_variable(dataset_trim, variable, range=slice(slice_start, slice_end)) # multiply by -1 because Delft3D output notes down as positive
time_steps = bedlev.shape[0]
print(f'{variable} variable shape is: {bedlev.shape}')
print(f"Full {variable} loaded in {time.time() - start_time:.2f} seconds")

#%% Convert morphological time in days since Tstart to actual dates
morph_days = np.array(morphtime[:])

morph_datetimes = np.array([reference_date + datetime.timedelta(days=float(day)) for day in morph_days])

# Get cross-section coordinates for braiding index and plotting (FOR POST-PROCESSING ONLY)
col_indices, N_coords, x_targets = get_cross_section_coordinates(x, y, x_targets = np.arange(x_min, x_max, 100))
print(f"Cross-sections defined at x-coordinates: {x_targets}")

#%% POST-PROCESSING SETTINGS: Set to True to run, set to False to skip the section
run_spatial_plots = False
run_width_averaged_bedlevel = True
run_cumulative_width_averaged_bedlevel = False
run_braiding_analysis = False
run_hypsometric_analysis = False
run_multi_scenario_hypsometric = False

#%% HYPSOMETRIC CURVE ANALYSIS
print("\n=== HYPSOMETRIC CURVE ANALYSIS ===")

# single run for 1 scenario

if run_hypsometric_analysis:
    print("Starting hypsometric curve analysis...")
    
    # Define timesteps to analyze
    reference_t = 0  # Grey reference line
    analysis_timesteps = np.arange(1, 202, 20)  # [1, 3, 5, 7, 9] - modify as needed
    
    print(f"Analyzing timesteps: {analysis_timesteps}")
    print(f"Reference timestep: {reference_t}")
    
    # Create hypsometric curves
    elevations_ref, areas_ref = plot_hypsometric_curves(
        bedlev, x, y, x_min, x_max, y_min, y_max, 
        bed_threshold=bed_threshold,
        timesteps=analysis_timesteps,
        reference_timestep=reference_t,
        scenario=scenario,
        save_dir=save_dir,
        save_figure=save_figure
    )
    
    # Print some statistics
    if len(elevations_ref) > 0:
        total_area = areas_ref[-1]  # Maximum cumulative area
        min_elevation = elevations_ref[0]
        max_elevation = elevations_ref[-1]
        
        print(f"\n=== Hypsometric Analysis Results ===")
        print(f"Total estuary area (excluding land): {total_area:.2f} km²")
        print(f"Elevation range: {min_elevation:.2f} to {max_elevation:.2f} m")
        print(f"Area below {bed_threshold}m threshold: {total_area:.2f} km²")
        
        # Calculate area at specific elevation percentiles
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            idx = int(len(areas_ref) * p / 100)
            if idx < len(areas_ref):
                print(f"Area below {elevations_ref[idx]:.2f}m elevation ({p}th percentile): {areas_ref[idx]:.2f} km²")
    
    print("Hypsometric curve analysis completed.")
else:
    print('HYPSOMETRIC CURVE ANALYSIS skipped.')
#%%
if run_multi_scenario_hypsometric:
    print("Starting multi-scenario hypsometric curve analysis...")
    
    # Define timesteps for individual scenario plots
    reference_t = 0
    analysis_timesteps = np.arange(1, 202, 20)  # Modify as needed
    
    # Store all scenario data for comparison plot
    all_scenario_data = {}
    
    # Process each discharge scenario
    for discharge in discharges:
        print(f"\n=== Processing discharge Q = {discharge} m³/s ===")
        runname = get_runname(discharge)
        
        # Process each scenario for this discharge
        for scenario_template in scenario_templates:
            scenario_name = scenario_template.format(discharge=discharge)
            print(f"\nProcessing scenario: {scenario_name}")
            
            # File paths
            trim_file_scenario = os.path.join(model_location, runname, scenario_name, 'trim-varriver_tidewest.nc')
            
            try:
                # Load dataset
                print(f"Loading trim file: {trim_file_scenario}")
                dataset_trim_scenario = nc.Dataset(trim_file_scenario, mode='r')
                
                # Load coordinates (should be same for all scenarios)
                x_scenario = load_variable(dataset_trim_scenario, "XCOR")
                y_scenario = load_variable(dataset_trim_scenario, "YCOR")
                
                # Load bed level data
                print(f"Loading bed level data...")
                bedlev_scenario = -1 * load_variable(dataset_trim_scenario, 'DPS', range=slice(slice_start, slice_end))
                
                print(f"Bed level shape: {bedlev_scenario.shape}")
                
                # Store data for scenario comparison plot
                all_scenario_data[scenario_name] = (x_scenario, y_scenario, bedlev_scenario)
                
                # Create individual hypsometric plot for this scenario
                print(f"Creating individual hypsometric plot for {scenario_name}...")
                elevations_ref, areas_ref = plot_hypsometric_curves(
                    bedlev_scenario, x_scenario, y_scenario, x_min, x_max, y_min, y_max,
                    bed_threshold=bed_threshold,
                    timesteps=analysis_timesteps,
                    reference_timestep=reference_t,
                    scenario=scenario_name,
                    save_dir=save_dir,
                    save_figure=save_figure
                )
                
                # Close dataset to free memory
                dataset_trim_scenario.close()
                
            except Exception as e:
                print(f"Error processing {scenario_name}: {e}")
                continue
    
    # Create scenario comparison plot
    if len(all_scenario_data) > 1:
        print(f"\n=== Creating scenario comparison plot ===")
        print(f"Available scenarios: {list(all_scenario_data.keys())}")
        
        for discharge in discharges:  # Create comparison plot for each discharge
            # Filter scenarios for this discharge
            discharge_scenarios = {k: v for k, v in all_scenario_data.items() 
                                 if f'{discharge}' in k}
            
            if len(discharge_scenarios) > 0:
                print(f"Creating comparison plot for Q = {discharge} m³/s")
                plot_scenario_comparison_hypsometric(
                    discharge_scenarios,
                    x_min, x_max, y_min, y_max, bed_threshold=6, 
                    reference_timestep=0,
                    final_timestep=-1,  # Use last timestep
                    scenario_colors={
                        'baserun': 'tab:blue',
                        'seasonal': 'tab:orange',
                        'flashy': 'tab:green'
                    },
                    save_dir=save_dir,
                    save_figure=save_figure,
                    discharge=discharge
                )
    else:
        print("Not enough scenarios loaded for comparison plot")
    
    print("Multi-scenario hypsometric curve analysis completed.")
else:
    print('MULTI-SCENARIO HYPSOMETRIC CURVE ANALYSIS skipped.')

#%% 1a. SPATIAL MAP PLOTS (BED LEVEL, WATER LEVEL, ...) 
# ========================================================
print("\n=== SPATIAL PLOTS ===")
# timesteps = np.linspace(slice_start, slice_end-1, amount_to_plot).astype(int)
timesteps = np.arange(0,10,1)

if run_spatial_plots:
    for timestep in timesteps: 
        # 1a. BED LEVEL AND WATER LEVEL MAP PLOTS
        print("\n=== BED LEVEL AND WATER LEVEL PLOTS ===") 
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
        run_velocity_plots = True  # Set to False to skip this section

        if run_velocity_plots:
            # Example usage for U1 velocity at timestep=1, surface layer
            velocity = load_single_timestep_variable(dataset_trim, "U1", timestep=timestep, remove=1, layer=0)
            
            plot_velocity(x, y, velocity, 'U1', col_indices, N_coords, timestep, scenario, save_dir, save_figure)

        print("SPATIAL PLOTS completed.")
else:
    print('SPATIAL PLOTS skipped.')

#%% 2a. WIDTH-AVERAGED BED LEVEL ALONG ESTUARY FOR MULTIPLE TIMESTEPS
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
    for i, timestep in enumerate(selected_timesteps):
        print(f"Processing timestep {timestep} ({i+1}/{len(selected_timesteps)})...")
        
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
        width_std_profiles.append(width_std_bedlev_clean)  # NEW: Store std profiles
        valid_x_coords.append(x_coords_clean)
            
        # Create label with morphological time if available
        if timestep < len(morph_days):
            morph_time_years = morph_days[timestep] / 365.25
            timestep_labels.append(f't = {timestep} ({morph_time_years:.0f} years)')
        else:
            timestep_labels.append(f't = {timestep}')

    print(f"Successfully computed width-averaged bed level profiles for {len(selected_timesteps)} timesteps")

    # VISUALIZATION WIDTH-AVERAGED BED LEVEL
    plt.figure(figsize=(14, 8))

    # Define colors for different timesteps (Blues; YlOrBr; OrRd)
    colors = plt.cm.YlOrBr(np.linspace(0.3, 1, len(selected_timesteps)))

    # Plot each timestep
    for i, (profile, x_coords, label) in enumerate(zip(width_averaged_profiles, valid_x_coords, timestep_labels)):
        # Convert x-coordinates to km for better readability
        x_coords_km = x_coords / 1000
        
        plt.plot(x_coords_km, profile, 
                color=colors[i], linewidth=2, marker='o', markersize=2, 
                label=label, alpha=0.8)

    plt.xlabel('Distance along estuary [km]')
    plt.ylabel('Width-averaged bed level [m]')
    plt.title(f'Width-averaged bed level evolution along estuary\n{scenario} - {runname}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Invert y-axis if you want deeper areas to appear lower
    # plt.gca().invert_yaxis()

    plt.tight_layout()

    if save_figure:
        plt.savefig(os.path.join(save_dir, f'width_averaged_bedlevel_evolution_{scenario}.png'), 
                    dpi=300, bbox_inches='tight', transparent=True)
        print(f"Evolution figure saved to {save_dir}")

    plt.show()

    # VISUALIZATION - WIDTH-AVERAGED STANDARD DEVIATION
    plt.figure(figsize=(14, 8))

    # Define colors for different timesteps
    colors = plt.cm.plasma(np.linspace(0, 1, len(selected_timesteps)))  # Different colormap for distinction

    # Plot standard deviation for each timestep
    for i, (std_profile, x_coords, label) in enumerate(zip(width_std_profiles, valid_x_coords, timestep_labels)):
        # Convert x-coordinates to km for better readability
        x_coords_km = x_coords / 1000
        
        plt.plot(x_coords_km, std_profile, 
                color=colors[i], linewidth=2, marker='s', markersize=2, 
                label=label, alpha=0.8)

    plt.xlabel('Distance along estuary [km]')
    plt.ylabel('Standard deviation of bed level [m]')
    plt.title(f'Width-averaged bed level variability (std dev) along estuary\n{scenario} - {runname}')
    plt.grid(True, alpha=0.3)

    # Add reference line at zero
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_figure:
        plt.savefig(os.path.join(save_dir, f'width_std_bedlevel_evolution_{scenario}.png'), 
                    dpi=300, bbox_inches='tight', transparent=True)
        print(f"Standard deviation evolution figure saved to {save_dir}")

    plt.show()

    # ADDITIONAL ANALYSIS (OPTIONAL) - MULTIPLE TIMESTEPS INCLUDING STD

    additional_optional_analysis = True

    if additional_optional_analysis:
        # Print some statistics for each timestep
        print("\n=== Width-averaged bed level statistics for all timesteps ===")
        for i, (mean_profile, std_profile, timestep, label) in enumerate(zip(width_averaged_profiles, width_std_profiles, selected_timesteps, timestep_labels)):
            print(f"\n{label}:")
            print(f"  Mean bed level - Min: {np.min(mean_profile):.2f} m, Max: {np.max(mean_profile):.2f} m, Avg: {np.mean(mean_profile):.2f} m")
            print(f"  Std deviation - Min: {np.min(std_profile):.2f} m, Max: {np.max(std_profile):.2f} m, Avg: {np.mean(std_profile):.2f} m")
            
            # Find locations with highest and lowest variability
            min_std_idx = np.argmin(std_profile)
            max_std_idx = np.argmax(std_profile)
            x_coords = valid_x_coords[i]
            
            print(f"  Most uniform location: x = {x_coords[min_std_idx]/1000:.1f} km, std = {std_profile[min_std_idx]:.3f} m")
            print(f"  Most variable location: x = {x_coords[max_std_idx]/1000:.1f} km, std = {std_profile[max_std_idx]:.3f} m")
            
            # Calculate coefficient of variation statistics
            mean_abs = np.abs(mean_profile)
            cv = np.where(mean_abs > 0.1, std_profile / mean_abs, np.nan)
            valid_cv = cv[~np.isnan(cv)]
            if len(valid_cv) > 0:
                print(f"  Coefficient of variation - Min: {np.min(valid_cv):.3f}, Max: {np.max(valid_cv):.3f}, Avg: {np.mean(valid_cv):.3f}")

        # Compare variability between first and last timesteps if available
        if len(selected_timesteps) > 1:
            print(f"\n=== Variability comparison between first and last timesteps ===")
            first_std = width_std_profiles[0]
            last_std = width_std_profiles[-1]
            
            if np.array_equal(valid_x_coords[0], valid_x_coords[-1]):
                mean_std_change = np.mean(last_std - first_std)
                print(f"  Mean change in standard deviation: {mean_std_change:.3f} m")
                
                # Find locations with biggest changes in variability
                std_change = last_std - first_std
                max_increase_idx = np.argmax(std_change)
                max_decrease_idx = np.argmin(std_change)
                x_coords = valid_x_coords[0]
                
                print(f"  Largest variability increase: x = {x_coords[max_increase_idx]/1000:.1f} km, change = +{std_change[max_increase_idx]:.3f} m")
                print(f"  Largest variability decrease: x = {x_coords[max_decrease_idx]/1000:.1f} km, change = {std_change[max_decrease_idx]:.3f} m")
            else:
                print("  (Cannot compute direct comparison - different valid x-coordinates)")
else: 
    print('WIDTH-AVERAGED BED LEVEL PROFILE skipped.')

#%% 2a-i. CUMULATIVE WIDTH-AVERAGED BED LEVEL CHANGE ALONG ESTUARY OVER TIME (morphological activity)

if run_cumulative_width_averaged_bedlevel: 
    print("\n=== CUMULATIVE WIDTH-AVERAGED BED LEVEL CHANGE ALONG ESTUARY OVER TIME ===")

    # 1. Get the x coordinates along estuary (nx,)
    along_x = x[:, 0]

    x_inds = np.where((along_x >= x_min) & (along_x <= x_max))[0]
    along_x_estuary = along_x[x_inds]

    # 2. Mask land locations (bed level >= 6 is land)
    bedlev_masked = np.where(bedlev < 6, bedlev, np.nan)  # shape: (time, nx, ny)

    # 3. Compute width-averaged bed level at each time and x-location (shape: time x nx)
    width_avg_bedlev = np.nanmean(bedlev_masked, axis=2)

    # 3b. Only keep estuary part along x
    width_avg_bedlev_estuary = width_avg_bedlev[:, x_inds]

    # 4. Calculate differences along time axis (axis=0)
    differences = np.diff(width_avg_bedlev_estuary, axis=0)

    # 5. Prepend zeros for the first timestep to make the shape consistent (shape will be time x nx)
    zeros_row = np.zeros((1, width_avg_bedlev_estuary.shape[1]))
    abs_differences = np.abs(differences)
    abs_differences_with_prepended = np.vstack([zeros_row, abs_differences])

    fig, ax = plt.subplots(figsize=(12, 8))

    # 6. Calculate cumulative sum of absolute changes through time for each x-location
    cumulative_activity = np.cumsum(abs_differences_with_prepended, axis=0)

    # Define extent for imshow: [x_min, x_max, y_min, y_max] -> x in km, y in time steps
    extent = [along_x_estuary.min() / 1000, along_x_estuary.max() / 1000, 0, cumulative_activity.shape[0]]

    # Heatmap
    im = ax.imshow(
        cumulative_activity,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis',
        vmin=0,
        vmax=5  # np.percentile(cumulative_activity, 98) if you want adaptive scaling
    )

    # --- Consistent colorbar placement using axes divider ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.tick_params(labelsize=fontsize_axes, colors=fontcolor)
    cbar.set_label(r'$\Sigma |\Delta h|$ [m]', fontsize=fontsize_labels, color=fontcolor)
    cbar.outline.set_edgecolor(fontcolor)

    # Labels / title
    ax.set_xlabel('Along-estuary distance [km]', fontsize=fontsize_labels, color=fontcolor)
    ax.set_ylabel('timestep', fontsize=fontsize_labels, color=fontcolor)
    ax.set_title(f'{scenario}: cumulative bed level change along estuary',
                fontsize=fontsize_titles, color=fontcolor)

    # --- Info box ---
    final_cumulative = cumulative_activity[-1, :]  # last timestep values (shape = nx)
    x_maxvalue_estuary = along_x_estuary[np.argmax(final_cumulative)] / 1000
    x_minvalue_estuary = along_x_estuary[np.argmin(final_cumulative)] / 1000
    max_val, min_val = np.max(final_cumulative), np.min(final_cumulative)

    textstr = (
        rf"$(\Sigma |\Delta h|)_{{\mathrm{{max}}}}$ = {max_val:.2f} m at {x_maxvalue_estuary:.1f} km"
        "\n"
        rf"$(\Sigma |\Delta h|)_{{\mathrm{{min}}}}$ = {min_val:.2f} m at {x_minvalue_estuary:.1f} km"
    )

    ax.text(
        0.02, 0.03, textstr,
        transform=ax.transAxes,
        fontsize=fontsize_labels,
        verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.8)
    )

    # --- Ticks ---
    ax.tick_params(axis='both', which='major', labelsize=fontsize_labels, colors=fontcolor)
    ax.set_xticks([20, 25, 30, 35, 40, 45])  # add more if needed
    
    # Remove or recolor main plot outline
    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)  # or 'white' for white outline

    # Save before show
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
                f'cumulative_bed_change_heatmap_{scenario}_{fontcolor}.png'),
                dpi=300, bbox_inches='tight', transparent=True)

    plt.show()

# #Visual check of content
# for location_index in [0, cumulative_activity.shape[1]//4, cumulative_activity.shape[1]//2, cumulative_activity.shape[1]-1]:
#     plt.plot(cumulative_activity[:, location_index], label=f'Location {along_x[location_index]/1000:.1f} km')
# plt.xlabel('Timestep')
# plt.ylabel('Cumulative absolute bed level change [m]')
# plt.legend()
# plt.show()

# %% # 2b. BRAIDING INDEX ALONG ESTUARY FOR MULTIPLE TIMESTEPS
print("\n=== BRAIDING INDEX ANALYSIS ===")

if run_braiding_analysis:
    print("Computing braiding index...")
    BI_per_cross_section, datetimes, _, _, _ = compute_BI_per_cross_section(
        x, y, tau_max, slice_start, map_output_interval, reference_date, theta=0.5, x_targets=x_targets
    )
    
    # Plot braiding index results
    df_BI = plot_braiding_index_timeseries(BI_per_cross_section, x_targets, datetimes, n_timesteps=2)
    plot_mean_braiding_index(df_BI, x_targets)
    
    print("BRAIDING INDEX ANALYSIS completed.")
else:
    print('BRAIDING INDEX ANALYSIS skipped.')

# %%
