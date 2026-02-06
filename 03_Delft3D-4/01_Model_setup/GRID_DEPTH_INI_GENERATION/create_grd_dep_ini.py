"""
Delft3D-FLOW Grid and Bathymetry Generation Script
Author: Marloes Bonenkamp
Date: March 12, 2025
Description: Generates bathymetry for an estuary domain, allows for visual inspection, and then creates .grd and .dep files upon request
"""

#%% Import modules

import sys
import numpy as np
from pathlib import Path

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"C:\Users\marloesbonenka\surfdrive\Python\01_Model_setup\GRID_DEPTH_INI_GENERATION")

#%% Import functions

from FUNCTIONS.FUNCS_create_dep import *
from FUNCTIONS.FUNCS_create_grd import *
from FUNCTIONS.FUNCS_create_ini import *

#%% Functions to generate the output files 

def generate_files(grid_data, resolution, output_dir, ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM, ini_value):
    """
    Generates .grd and .dep files for a specific resolution.
    Now supports both uniform and variable grids.
    """
    if resolution not in grid_data:
        print(f"Error: No data for resolution {resolution}")
        return

    data = grid_data[resolution]

    # Write and save grd and dep files
    write_grd_file(output_dir / f'grid_slopingsea_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m.grd', data['x'], data['y'])
    write_dep_file(output_dir / f'dep_slopingsea_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m.dep', data['dep_data'])
    
    # Use the 2D meshgrid data for the X and Y coordinates
    if ADD_NOISE:
        xyz_filename = output_dir / f'noisybathy_seed{NOISE_SEED}.xyz'
    else:
        xyz_filename = output_dir / f'bathymetry_slopingsea_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m.xyz'
    write_xyz_file(xyz_filename, data['X'], data['Y'], data['dep_data'], crs_epsg=3857)
    
    # # Generate the base INI arrays for Delft3D-4
    # zero_ini = generate_ini(data, 0)['ini_data']
    # regular_ini = generate_ini(data, ini_value)['ini_data']

    # # Repeat the zero_ini vertically 20 times
    # repeated_zero = np.tile(zero_ini, (20, 1))  # Stack 20 copies vertically

    # # Combine regular_ini on top of the repeated zero_ini
    # full_ini_data = np.vstack([regular_ini, repeated_zero])

    # # Write and save ini file
    # write_ini_file(os.path.join(output_dir, f'ini_slopingsea_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m_inivalue{ini_value}m.ini'), full_ini_data)
    
    # Generate the initial water level data
    ini_result = generate_ini_depth(data, SEA_EXTENT)
    
    # Write the .xyz file for the Initial Water Level
    ini_filename = output_dir / f'ini_depth_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m.xyz'
    write_ini_xyz_file(ini_filename, data['X'], data['Y'], ini_result['ini_data'])

    print(f"Generated .grd, .dep and .ini files for {resolution} resolution.")

def generate_all_files(grid_data, base_resolutions, output_dir, ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM, ini_value):
    """
    Generates .grd and .dep files for all specified resolutions.
    """
    for res in base_resolutions:
        if res in grid_data:
            generate_files(grid_data, res, output_dir, ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM, ini_value)
        else:
            print(f"Error: No data for resolution {res}")
    
    print("All specified .grd and .dep files have been generated.")

#%%

# Print current working directory
print("Current working directory:", Path.cwd())

# Noise Configuration
ADD_NOISE = True                # Set to True to add noise to estuary bathymetry
NOISE_AMPLITUDE_CM = 20.0        # Noise amplitude in centimeters (e.g., 5 cm = ±5 cm random variation)
NOISE_SEED = 2                # Random seed for reproducibility (set to None for random noise each run)

# Configuration Settings
path_loc = Path(r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
model_name = Path("Noisy_Bathy") / f"Seed_{NOISE_SEED}" if ADD_NOISE else Path("Base_Bath")

# Grid Configuration
USE_VARIABLE_GRID = True  # Set to True for variable grid, False for uniform grid

if USE_VARIABLE_GRID:
    # Variable Grid Configuration - Different resolutions for x and y directions
    FINE_RES_X = 85      # Fine resolution in x-direction (meters)
    MEDIUM_RES_X = 150   # Medium resolution in x-direction (meters)  
    # FINE_RES_X = 250
    # MEDIUM_RES_X = 250
    COARSE_RES_X = 250   # Coarse resolution in x-direction (meters)
    
    FINE_RES_Y = 50      # Fine resolution in y-direction (meters)
    MEDIUM_RES_Y = 100   # Medium resolution in y-direction (meters)
    # FINE_RES_Y = 200
    # MEDIUM_RES_Y = 200  
    COARSE_RES_Y = 200   # Coarse resolution in y-direction (meters)
    
    # note: ratio between grid cells should not exceed 1.3

    # Spatial Grid Configuration - Fine grid in specific region with U-shaped buffer
    FINE_BOUNDS = (18000, 45000, 5500, 9500)  # (x_min, x_max, y_min, y_max) for fine region
    BUFFER_WIDTH = 1000  # Width of medium resolution buffer around fine region
    
    BASE_RESOLUTIONS = [f"spatial_x{FINE_RES_X}_{MEDIUM_RES_X}_{COARSE_RES_X}_y{FINE_RES_Y}_{MEDIUM_RES_Y}_{COARSE_RES_Y}m"]
else:
    # Uniform Grid Configuration
    BASE_RESOLUTIONS = [100]  # Grid cell sizes in meters

DOMAIN_EXTENT = (45000, 15000)  # Domain size (x_length, y_length) in meters
SEA_EXTENT = 20000             # Sea basin extent in x-direction
SLOPE_EXTENT = 5000            # Extent of the sloping part of the sea basin, set to zero if no slope is required
ESTUARYWIDTH_UPSTREAM = 500
ESTUARYWIDTH_DOWNSTREAM = 3000
SEABASIN_DEPTH = 15
SAVE_FIGURES = 'yes'            # Set to 'yes' to save figures, 'no' to just show them
INI_VALUE = 0.5

# Create the output directory if it doesn't exist
output_dir = path_loc / model_name / ('D3D_base_input' if USE_VARIABLE_GRID else f'D3D_base_input_{ESTUARYWIDTH_DOWNSTREAM}_{ESTUARYWIDTH_UPSTREAM}')
output_dir.mkdir(parents=True, exist_ok=True)

# Dictionary to store grid and depth data
grid_data = {}

if USE_VARIABLE_GRID:
    # Generate spatial-based variable grid
    grid_name = f"spatial_x{FINE_RES_X}_{MEDIUM_RES_X}_{COARSE_RES_X}_y{FINE_RES_Y}_{MEDIUM_RES_Y}_{COARSE_RES_Y}m"
    
    # Generate variable grid coordinates using spatial method
    grid_coords = generate_variable_grid_spatial(
        DOMAIN_EXTENT, 
        FINE_RES_X, MEDIUM_RES_X, COARSE_RES_X,
        FINE_RES_Y, MEDIUM_RES_Y, COARSE_RES_Y,
        FINE_BOUNDS, BUFFER_WIDTH
    )

    
    # Generate bathymetry on variable grid
    data = generate_bathymetry_variable_grid(
        grid_coords['x'], grid_coords['y'],
        DOMAIN_EXTENT, SEA_EXTENT, SLOPE_EXTENT, SEABASIN_DEPTH, 
        ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM
    )
    
    # Add noise to estuary bathymetry if enabled
    if ADD_NOISE:
        print(f"Adding noise to estuary bathymetry: ±{NOISE_AMPLITUDE_CM} cm")
        # Create estuary mask for variable grid
        X, Y = data['X'], data['Y']
        ny, nx = X.shape
        river_length = DOMAIN_EXTENT[0] - SEA_EXTENT
        y_center = DOMAIN_EXTENT[1] / 2 + 1
        estuary_mask = np.zeros((ny, nx), dtype=bool)
        
        for i in range(ny):
            for j in range(nx):
                if X[i, j] > SEA_EXTENT:
                    x_pos = X[i, j] - SEA_EXTENT
                    river_width = ESTUARYWIDTH_DOWNSTREAM * np.exp(np.log(ESTUARYWIDTH_UPSTREAM / ESTUARYWIDTH_DOWNSTREAM) * x_pos / river_length)
                    if abs(Y[i, j] - y_center) < river_width / 2:
                        estuary_mask[i, j] = True
        
        # Apply noise
        if NOISE_SEED is not None:
            np.random.seed(NOISE_SEED)
        noise_amplitude_m = NOISE_AMPLITUDE_CM / 100.0
        noise = np.random.uniform(-noise_amplitude_m, noise_amplitude_m, (ny, nx))
        data['dep_data'][estuary_mask] += noise[estuary_mask]
        data['estuary_mask'] = estuary_mask
    
    # Store data
    grid_data[grid_name] = data
    
    ini = generate_ini_depth(data, SEA_EXTENT)

    # Plot bathymetry and grid
    plot_bathymetry(data['X'], data['Y'], data['dep_data'], grid_name, output_dir, SAVE_FIGURES)
    plot_bathymetry(data['X'], data['Y'], ini['ini_data'], f"{grid_name}_ini", output_dir, SAVE_FIGURES)
    plot_variable_grid(data['x'], data['y'], grid_name, output_dir, SAVE_FIGURES)
    
else:
    # Generate uniform grids (your existing code)
    for res in BASE_RESOLUTIONS:
        # Generate bathymetry - with or without noise
        if ADD_NOISE:
            print(f"Generating bathymetry with noise: ±{NOISE_AMPLITUDE_CM} cm")
            data = generate_bathymetry_with_noise(
                res, DOMAIN_EXTENT, SEA_EXTENT, SLOPE_EXTENT, SEABASIN_DEPTH, 
                ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM,
                noise_amplitude_cm=NOISE_AMPLITUDE_CM, random_seed=NOISE_SEED
            )
        else:
            data = generate_bathymetry(res, DOMAIN_EXTENT, SEA_EXTENT, SLOPE_EXTENT, SEABASIN_DEPTH, ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM)
        
        # Store data
        grid_data[res] = {'x': data['x'], 
                            'y': data['y'], 
                            'dep_data': data['dep_data'],
                            'X': data['X'],
                            'Y': data['Y']}
        ini = generate_ini(data, INI_VALUE)

        # Plot bathymetry and grid
        plot_bathymetry(data['X'], data['Y'], data['dep_data'], res, output_dir, SAVE_FIGURES)
        plot_bathymetry(data['X'], data['Y'], ini['ini_data'], res, output_dir)
        plot_grid(data['x'], data['y'], len(data['x']), len(data['y']), res, output_dir, SAVE_FIGURES)

if SAVE_FIGURES == 'yes':
    print("All plots generated and saved.")
else:
    print("All plots displayed.")

#%% Generate .grd and .dep files using your existing function structure
generate_all_files(grid_data, BASE_RESOLUTIONS, output_dir, ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM, INI_VALUE)

# %%
