import numpy as np
import pandas as pd
import os
import re
from matplotlib.colors import LinearSegmentedColormap

#%%
# --- CHECK VARIABLES IN DELFT3D OUTPUT ---
def check_available_variables_xarray(ds):
    """Updated for xarray/dfm_tools datasets"""
    print("Available variables in dataset:\n")
    # xarray uses ds.data_vars for the main variables
    for var_name in sorted(ds.data_vars):
        var = ds[var_name]
        print(f"  {var_name}:")
        print(f"    shape         = {var.shape}")
        print(f"    dimensions    = {var.dims}")
        
        # xarray stores metadata in the .attrs dictionary
        for attr in ['units', 'long_name', 'standard_name', 'description']:
            if attr in var.attrs:
                print(f"    {attr:13} = {var.attrs[attr]}")
        
        print("") 

    return {'all_vars': list(ds.data_vars)}

# --- EXTRACT MORFAC FROM FOLDER NAME ---
def get_mf_number(folder_name):
    match = re.search(r'MF_?(\d+)', folder_name)
    return int(match.group(1)) if match else 999

# --- EXTRACT COMPUTATION TIME FROM .dia FILE ---
def extract_computation_time(dia_file_path):
    """
    Extract computation time in days and hours from FlowFM_0000.dia file.
    Returns tuple: (days, hours) or (None, None) if not found.
    """
    try:
        with open(dia_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        comp_time_days = None
        comp_time_hours = None
        
        for line in lines:
            if 'total computation time (d)' in line:
                # Extract the number from the line
                # Example: "** INFO   : total computation time (d)  :             4.6385087087"
                parts = line.split(':')
                if len(parts) >= 3:  # There are multiple colons
                    # Take the last part after the last colon
                    try:
                        comp_time_days = float(parts[-1].strip())
                    except ValueError:
                        pass
                        
            elif 'total computation time (h)' in line:
                parts = line.split(':')
                if len(parts) >= 3:
                    try:
                        comp_time_hours = float(parts[-1].strip())
                    except ValueError:
                        pass
        
        return comp_time_days, comp_time_hours
    
    except Exception as e:
        print(f"    Warning: Could not read .dia file: {e}")
        return None, None
    

# --- CUSTOM COLORMAP ---
def create_terrain_colormap():
    colors = [
        (0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
        (0.40, "#00ffff"), (0.50, "#ffffcc"), (0.60, "#ffcc00"),
        (0.75, "#cc6600"), (0.90, "#228B22"), (1.00, "#006400"),
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

terrain_cmap = create_terrain_colormap()

# --- MORPHOLOGICAL TIME SELECTION ---
def find_timestep_for_target_morphtime(ds, target_morph_years, start_date):
    """
    Find the timestep where morphological time reaches the target.
    Calculates: morph_time = hydro_time_elapsed * morfac
    """
    start_timestamp = pd.Timestamp(start_date)
    times = pd.to_datetime(ds['time'].values)
    
    # Calculate elapsed hydrodynamic time in years for each timestep
    hydro_elapsed_years = np.array([(t - start_timestamp).days / 365.25 for t in times])
    
    # Get MORFAC values at each timestep
    if 'morfac' in ds:
        morfac_values = ds['morfac'].values
    else:
        raise ValueError("MORFAC variable not found in dataset")
    
    # Calculate morphological time at each timestep: morph_time = hydro_time * morfac
    morph_time_years = hydro_elapsed_years * morfac_values
    
    # Find closest timestep to target morphological time
    time_diffs = np.abs(morph_time_years - target_morph_years)
    closest_idx = int(np.argmin(time_diffs))
    
    actual_morph_years = morph_time_years[closest_idx]
    actual_hydro_years = hydro_elapsed_years[closest_idx]
    actual_morfac = morfac_values[closest_idx]
    actual_time = times[closest_idx]
    
    return closest_idx, actual_time, actual_hydro_years, actual_morph_years, actual_morfac
