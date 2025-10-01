"""
Delft3D-FLOW Grid and Bathymetry Generation Script
Author: Marloes Bonenkamp
Date: March 12, 2025
Description: Generates bathymetry for an estuary domain, allows for visual inspection, and then creates .grd and .dep files upon request
"""

#%% Import modules

import os
import sys
import numpy as np

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
    write_grd_file(os.path.join(output_dir, f'grid_slopingsea_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m.grd'), data['x'], data['y'])
    write_dep_file(os.path.join(output_dir, f'dep_slopingsea_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m.dep'), data['dep_data'])
    
    # Generate the base INI arrays
    zero_ini = generate_ini(data, 0)['ini_data']
    regular_ini = generate_ini(data, ini_value)['ini_data']

    # Repeat the zero_ini vertically 20 times
    repeated_zero = np.tile(zero_ini, (20, 1))  # Stack 20 copies vertically

    # Combine regular_ini on top of the repeated zero_ini
    full_ini_data = np.vstack([regular_ini, repeated_zero])

    # Write and save ini file
    write_ini_file(os.path.join(output_dir, f'ini_slopingsea_{resolution}_Wup{ESTUARYWIDTH_UPSTREAM}m_Wdown{ESTUARYWIDTH_DOWNSTREAM}m_inivalue{ini_value}m.ini'), full_ini_data)
    
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
print("Current working directory:", os.getcwd())

# Configuration Settings
path_loc = r"u:\PhDNaturalRhythmEstuaries\Models"
model_name = "04_RiverDischargeVariability_domain45x15"

# Grid Configuration
USE_VARIABLE_GRID = True  # Set to True for variable grid, False for uniform grid

if USE_VARIABLE_GRID:
    # Variable Grid Configuration - Different resolutions for x and y directions
    FINE_RES_X = 85      # Fine resolution in x-direction (meters)
    MEDIUM_RES_X = 150   # Medium resolution in x-direction (meters)  
    COARSE_RES_X = 250   # Coarse resolution in x-direction (meters)
    
    FINE_RES_Y = 50      # Fine resolution in y-direction (meters)
    MEDIUM_RES_Y = 100   # Medium resolution in y-direction (meters)  
    COARSE_RES_Y = 225   # Coarse resolution in y-direction (meters)
    
    # Spatial Grid Configuration - Fine grid in specific region with U-shaped buffer
    FINE_BOUNDS = (18000, 45000, 5500, 9500)  # (x_min, x_max, y_min, y_max) for fine region
    BUFFER_WIDTH = 1000  # Width of medium resolution buffer around fine region
    
    BASE_RESOLUTIONS = [f"spatial_x{FINE_RES_X}_{MEDIUM_RES_X}_{COARSE_RES_X}_y{FINE_RES_Y}_{MEDIUM_RES_Y}_{COARSE_RES_Y}m"]
else:
    # Uniform Grid Configuration
    BASE_RESOLUTIONS = [25, 50, 100]  # Grid cell sizes in meters

DOMAIN_EXTENT = (45000, 15000)  # Domain size (x_length, y_length) in meters
SEA_EXTENT = 20000             # Sea basin extent in x-direction
SLOPE_EXTENT = 5000            # Extent of the sloping part of the sea basin, set to zero if no slope is required
ESTUARYWIDTH_UPSTREAM = 500
ESTUARYWIDTH_DOWNSTREAM = 3000
SEABASIN_DEPTH = 15
SAVE_FIGURES = 'no'            # Set to 'yes' to save figures, 'no' to just show them
INI_VALUE = 0.5

# Create the output directory if it doesn't exist
output_dir = os.path.join(path_loc, model_name, 'D3D_base_input' if USE_VARIABLE_GRID else f'D3D_base_input_{ESTUARYWIDTH_DOWNSTREAM}_{ESTUARYWIDTH_UPSTREAM}')
os.makedirs(output_dir, exist_ok=True)

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
    
    # Store data
    grid_data[grid_name] = data
    
    ini = generate_ini(data, INI_VALUE)

    # Plot bathymetry and grid
    plot_bathymetry(data['X'], data['Y'], data['dep_data'], grid_name, output_dir, SAVE_FIGURES)
    plot_bathymetry(data['X'], data['Y'], ini['ini_data'], f"{grid_name}_ini", output_dir, SAVE_FIGURES)
    plot_variable_grid(data['x'], data['y'], grid_name, output_dir, SAVE_FIGURES)
    
else:
    # Generate uniform grids (your existing code)
    for res in BASE_RESOLUTIONS:
        # Generate bathymetry
        data = generate_bathymetry(res, DOMAIN_EXTENT, SEA_EXTENT, SLOPE_EXTENT, SEABASIN_DEPTH, ESTUARYWIDTH_UPSTREAM, ESTUARYWIDTH_DOWNSTREAM)
        
        # Store data
        grid_data[res] = {'x': data['x'], 
                            'y': data['y'], 
                            'dep_data': data['dep_data'],
                            'X': data['X'],-
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
