"""
Delft3D-FM River Boundary Condition Generation Script
Author: Marloes Bonenkamp
Date: Janaury 15, 2026
Description: Generates river boundary conditions for an estuary domain, incorporating
             variability, flashiness, climate change scenarios, and a sinusoidal distribution
             of discharge among four river cells. Includes checks for cell-to-cell
             difference limit, preservation of the sinusoidal pattern, and total mean discharge.
"""

#%% 
# Load packages
import os
import sys

#%%
#Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\01_Model_setup\BOUNDARY_FILES_GENERATION")

# Load functions to create Delft3D-FM boundary files
from FUNCTIONS.FUNCS_create_bc_varyingriver_csv_FM import *

 #%% Configuration settings
total_discharge = 250                   # Total river discharge in m³/s
nyears = 52
duration_min    = 365.25 * 24 * 60 * nyears               # Total simulation duration in minutes;  525600 = 1 year;     2629440 = 5 years
time_step_rcel  = 1440                                  # Time step for variations over consecutive river cells in minutes, to force bar formation      

# IMPORTANT: Update these values based on your grid script
grid_info = {
    'nx': 99,#100,                                           # Number of sea basin cells in x-direction (m-direction)
    'ny': 141,#153,                                          # Number of sea basin cells in y-direction (n-direction)
    'river_cells': [
        (395, 72),
        (395, 71),
        (395, 70),
        (395, 69)
    ]
}

# Specify the directory of model runs
specific_scenario_location = f'CSVfiles_boundaries_50hydroyears'
output_dir = r"u:\PhDNaturalRhythmEstuaries\Models"

output_dir = os.path.join(output_dir, specific_scenario_location)
os.makedirs(output_dir, exist_ok=True)

#%%
# Define scenarios  
scenarios = [
    {
        "name":                 f"01_baserun{total_discharge}",               # Name for the single scenario
        "total_discharge":      total_discharge,                        # Total river discharge in m³/s
        "duration_min":         duration_min,                           # Total simulation duration in minutes
        "time_step":            time_step_rcel,                         # Time step for variations over river cells in minutes
        #"cv":                   0,                                      # Long-term/overall variability: standard deviation of the mean
        "pattern_type":         "constant"                              # or "seasonal"
    }
    ,    
    {
        "name":                 f"02_run{total_discharge}_seasonal",          # Name for the single scenario
        "total_discharge":      total_discharge,                        # Total river discharge in m³/s
        "duration_min":         duration_min,                           # 525600 = 1year #5 years = 2629440, total simulation duration in minutes
        "time_step":            time_step_rcel,                         # Time step for variations over river cells in minutes
        #"cv":                   0.5,                                    # Long-term/overall variability: standard deviation of the mean
        "pattern_type":         "seasonal"                              # or "constant"
    }
    ,    
    {
        "name":                 f"03_run{total_discharge}_flashy",            # Name for the single scenario
        "total_discharge":      total_discharge,                        # Total river discharge in m³/s
        "duration_min":         duration_min,                           # 525600 = 1year #5 years = 2629440,  # Total simulation duration in minutes
        "time_step":            time_step_rcel,                         # Time step for variations over river cells in minutes
        # "cv":                   1.0,                                    # Long-term/overall variability: standard deviation of the mean
        "pattern_type":         "flashy"                                # or "seasonal" or "constant" (flashiness refers to daily fluctuations, where high values indicate frequent abrupt changes)
    }
    ,    
    {
        "name":                 f"04_run{total_discharge}_singlepeak",        # Name for the single peak scenario
        "total_discharge":      total_discharge,                        # Total river discharge in m³/s
        "duration_min":         duration_min,                           # Total simulation duration in minutes
        "time_step":            time_step_rcel,                         # Time step for variations over river cells in minutes
        "pattern_type":         "singlepeak"                            # One peak per year, no droughts, same magnitude/duration as flashy
    }
]

for scenario in scenarios:
    # Create the specific directory for this scenario's boundary files
    scenario_name = scenario["name"]
    boundary_dir = os.path.join(output_dir, scenario_name, "boundaryfiles_csv")
    os.makedirs(boundary_dir, exist_ok=True)

    # Call the updated function
    generate_river_discharges_fm(
        grid_info=grid_info, 
        params=scenario, 
        output_dir=boundary_dir, 
        start_date_str='2024-01-01 00:00:00'
    )

print("finished --> load CSVs into GUI and create. bc files")
#%%