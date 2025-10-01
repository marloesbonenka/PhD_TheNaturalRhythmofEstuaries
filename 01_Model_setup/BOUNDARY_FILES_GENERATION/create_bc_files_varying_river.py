"""
Delft3D-FLOW River Boundary Condition Generation Script
Author: Marloes Bonenkamp
Date: April 25, 2025
Description: Generates river boundary conditions for an estuary domain, incorporating
             variability, flashiness, climate change scenarios, and a sinusoidal distribution
             of discharge among four river cells. Includes checks for cell-to-cell
             difference limit, preservation of the sinusoidal pattern, and total mean discharge.

USER GUIDELINES:

1.  Ensure the 'grid_info' dictionary contains:
    -   'nx': Number of sea basin cells in x-direction (m-direction)
    -   'ny': Number of sea basin cells in y-direction (n-direction)
    -   'river_cells': List of (x, y) coordinates for river boundary cells

2.  Set the 'params' dictionary with:
    -   'total_discharge': Total river discharge in m³/s
    -   'duration_min': Total duration of boundary conditions in minutes
    -   'time_step': Time step for variations over consecutive river cells in minutes
    -   'cv': Coefficient of variation for discharge (how much deviation from mean)
    -   'flashiness': P90/P10 ratio for discharge   (difference high flows/low flows; 
                                                    flashiness literally means:    
                                                    high flows x times higher than low flows)
    -   'pattern_type': "constant", "seasonal", "flashy", or "intermittent"
    -   'climate_scenario': "EE", "ED", or "PI-Med" 
        corresponding to EE: Increased Extreme Events, ED: Extended Droughts, PI-Med: Progressive Intensification (Medium Rate)

3.  Adjust the 'output_dir' path to your desired location for the boundary files.

4.  Run this script after your grid and depth file generation.

5.  Verify the generated files in the output directory:
    -   boundary.bnd
    -   river_boundary.bct
    -   tide.bch
    -   transport.bcc

6.  Always check the generated files for consistency with your model setup.  Pay special
    attention to the console output for warnings about cell-to-cell difference,
    sinusoidal pattern deviation, and total mean discharge preservation.

Note: This script uses tabs for formatting as per Delft3D-FLOW requirements.
"""

#%% 
# Load packages
import os
import sys
import numpy as np
import matplotlib as mpl

#%%
#Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"C:\Users\marloesbonenka\surfdrive\Python\01_Model_setup\BOUNDARY_FILES_GENERATION")

#%%
import importlib.util
print(importlib.util.find_spec("FUNCTIONS"))
print(importlib.util.find_spec("FUNCTIONS.FUNCS_create_bc_files_varying_river"))
#%%
# Load functions to create Delft3D boundary files
from FUNCTIONS.FUNCS_create_bc_files_varying_river import *

# Load functions to visualize/validate the created Delft3D boundary files
from FUNCTIONS.FUNCS_validate_created_bc_files_varying_river import *

#%% PLOTTING SETTINGS   
defaultcolour = 'black'
defaultfont = 20

mpl.rcParams['text.color'] = 'black'          # Default text color
mpl.rcParams['font.size'] = defaultfont             # Default font size

mpl.rcParams['axes.titlesize'] = defaultfont+4      # Title font size
mpl.rcParams['axes.titlecolor'] = defaultcolour     # Title color
mpl.rcParams['axes.labelsize'] = defaultfont        # Axis label size
mpl.rcParams['axes.labelcolor'] = defaultcolour     # Axis label color
mpl.rcParams['axes.facecolor'] = defaultcolour      # Background color of the axes (plot area)

mpl.rcParams['xtick.labelsize'] = defaultfont       # X tick labels size
mpl.rcParams['xtick.color'] = defaultcolour         # tick color matches text color
mpl.rcParams['xtick.labelcolor'] = defaultcolour

mpl.rcParams['ytick.labelsize'] = defaultfont       # Y tick labels size
mpl.rcParams['ytick.color'] = defaultcolour
mpl.rcParams['ytick.labelcolor'] = defaultcolour

mpl.rcParams['axes.grid'] = True                    # Default enable grid
mpl.rcParams['grid.alpha'] = 0.3                    # Grid transparency

mpl.rcParams['figure.figsize'] = (11, 8)            # Default figure size (width, height) in inches
mpl.rcParams['legend.fontsize'] = defaultfont       # Legend font size

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['savefig.dpi'] = 600


 #%% Configuration settings
discharge = 500
factor_fix = 1.0                                        # factor to compensate for discrepancy between input through boundary and inflow in model (has to do with the four adjacent boundary cells)
total_discharge = int(discharge * factor_fix)           # Total river discharge in m³/s
nyears = 12
duration_min    = 365.25 * 24 * 60 * nyears               # Total simulation duration in minutes;  525600 = 1 year;     2629440 = 5 years
time_step_rcel  = 1440                                  # Time step for variations over consecutive river cells in minutes, to force bar formation      

# IMPORTANT: Update these values based on your grid script
grid_info = {
    'nx': 99,#100,                                          # Number of sea basin cells in x-direction (m-direction)
    'ny': 141,#153,                                          # Number of sea basin cells in y-direction (n-direction)
    'river_cells': [
        (395, 72),
        (395, 71),
        (395, 70),
        (395, 69)
    ]
}

# Specify the directory of model runs
specific_scenario_location = f'RCEM_bc_plots_nobackground'
output_dir = r"u:\PhDNaturalRhythmEstuaries\Models\05_RiverDischargeVariability_domain45x15"

output_dir = os.path.join(output_dir, specific_scenario_location)
os.makedirs(output_dir, exist_ok=True)

#%%
# Define scenarios  
scenarios = [
    {
        "name":                 f"01_baserun{discharge}",               # Name for the single scenario
        "total_discharge":      total_discharge,                        # Total river discharge in m³/s
        "duration_min":         duration_min,                           # Total simulation duration in minutes
        "time_step":            time_step_rcel,                         # Time step for variations over river cells in minutes
        #"cv":                   0,                                      # Long-term/overall variability: standard deviation of the mean
        "pattern_type":         "constant"                              # or "seasonal"
    }
    ,    
    {
        "name":                 f"02_run{discharge}_seasonal",          # Name for the single scenario
        "total_discharge":      total_discharge,                        # Total river discharge in m³/s
        "duration_min":         duration_min,                           # 525600 = 1year #5 years = 2629440, total simulation duration in minutes
        "time_step":            time_step_rcel,                         # Time step for variations over river cells in minutes
        #"cv":                   0.5,                                    # Long-term/overall variability: standard deviation of the mean
        "pattern_type":         "seasonal"                              # or "constant"
    }
    ,    
    {
        "name":                 f"03_run{discharge}_flashy",            # Name for the single scenario
        "total_discharge":      total_discharge,                        # Total river discharge in m³/s
        "duration_min":         duration_min,                           # 525600 = 1year #5 years = 2629440,  # Total simulation duration in minutes
        "time_step":            time_step_rcel,                         # Time step for variations over river cells in minutes
        # "cv":                   1.0,                                    # Long-term/overall variability: standard deviation of the mean
        "pattern_type":         "flashy"                                # or "seasonal" or "constant" (flashiness refers to daily fluctuations, where high values indicate frequent abrupt changes)
    }
]

for scenario in scenarios:
    scenario_dir = os.path.join(output_dir, scenario["name"])
    boundary_dir = os.path.join(scenario_dir, "boundaryfiles")
    os.makedirs(boundary_dir, exist_ok=True)

    generate_boundary_files(grid_info, scenario, boundary_dir)

# Visualize and analyze can use output_dir and scenario names accordingly
visualize_discharge_scenarios(scenarios, output_dir, grid_info)

#%%