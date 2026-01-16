""""Mass balance sensitivity plot: Total Volume Change vs. Morphological Time
This script processes multiple Delft3D-FM model runs with different MORFAC settings"""
#%%
from pathlib import Path
import matplotlib.pyplot as plt
import dfm_tools as dfmt
import numpy as np
import pandas as pd
import xarray as xr
import sys
import re

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\Delft3D-FM\Postprocessing")

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *

#%%
# --- SETTINGS ---
# Choose discharge variability type: '02_seasonal' or '03_flashy'
discharge_type = '02_seasonal'  # Change to '03_flashy' as needed

# Target morphological time to extract from data
morfyears = 50  
tmorph_period = f'Tmorph_{morfyears}years' 

base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC") / discharge_type / tmorph_period
timed_out_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC") / discharge_type / tmorph_period / 'timed-out'
all_folders = [f.name for f in base_directory.iterdir() if f.name.startswith('MF') and f.is_dir()]
run_startdate = '2025-01-01'

var_name = 'mesh2d_mor_bl' # Bed level
area_name = 'mesh2d_flowelem_ba' # Area of cells

# Optional: Check available variables in the first dataset (set to False to skip)
check_vars = False

plt.figure(figsize=(10, 6))
variables_checked = False

# Dictionary to store data for each MORFAC run
data_by_morfac = {}

for folder in all_folders:
    model_location = base_directory / folder
    file_pattern = str(model_location / 'output' / '*_map.nc')
    
    # 1. Get Morfac for this run (extract MF number)
    mf_match = re.search(r'MF(\d+(?:\.\d+)?)', folder)
    current_mf = float(mf_match.group(1)) if mf_match else None
    if current_mf is None:
        print(f"Warning: Could not extract MORFAC value from {folder}, skipping.")
        continue
    
    # Match timed-out folder by MF number only (ignoring run ID after underscore)
    mf_prefix = f"MF{int(current_mf)}"  # e.g., "MF1", "MF10"
    timed_out_pattern = None
    if timed_out_directory.exists():
        # Find folder in timed-out that starts with the same MF number
        matching_folders = [f for f in timed_out_directory.iterdir() 
                           if f.is_dir() and f.name.startswith(mf_prefix + '_')]
        if matching_folders:
            timed_out_location = matching_folders[0]
            timed_out_pattern = str(timed_out_location / 'output' / '*_map.nc')
    
    try:
        # Initialize arrays to store temporal data
        all_morph_years = []
        all_volume_change = []
        
        # Process timed-out dataset first (if it exists)
        if timed_out_pattern:
            try:
                print(f"\n  Loading timed-out data for {mf_prefix}...")
                ds_timed_out = dfmt.open_partitioned_dataset(timed_out_pattern)
                
                # Check available variables on first dataset (timed-out)
                if check_vars and not variables_checked:
                    print(f"\n=== Available variables in {folder} (timed-out) ===")
                    check_available_variables_xarray(ds_timed_out)
                    variables_checked = True
                
                # Calculate morphological time for timed-out dataset
                start_timestamp = pd.Timestamp(run_startdate)
                times_to = pd.to_datetime(ds_timed_out['time'].values)
                hydro_elapsed_years_to = np.array([(t - start_timestamp).days / 365.25 for t in times_to])
                
                if 'morfac' in ds_timed_out:
                    morfac_values_to = ds_timed_out['morfac'].values
                else:
                    morfac_values_to = np.full_like(hydro_elapsed_years_to, current_mf)
                
                morph_years_to = hydro_elapsed_years_to * morfac_values_to
                
                # Calculate volume for timed-out dataset
                areas_to = ds_timed_out[area_name]
                bed_levels_to = ds_timed_out[var_name]
                total_volume_to = (bed_levels_to * areas_to).sum(dim=ds_timed_out[var_name].dims[-1])
                volume_change_to = total_volume_to - total_volume_to.isel(time=0)
                
                # Store timed-out results
                all_morph_years.extend(morph_years_to)
                all_volume_change.extend(volume_change_to.values)
                
                initial_volume = total_volume_to.isel(time=0)
                print(f"  Timed-out: {len(morph_years_to)} timesteps, ends at Tmorph={morph_years_to[-1]:.1f}y")
                
                ds_timed_out.close()
                
            except Exception as e:
                print(f"  Warning: Could not load timed-out data for {folder}: {e}")
                initial_volume = None
        else:
            initial_volume = None
        
        # Process main dataset
        print(f"  Loading main data for {mf_prefix}...")
        ds = dfmt.open_partitioned_dataset(file_pattern)
        
        # Check available variables on first dataset (main, if not checked from timed-out)
        if check_vars and not variables_checked:
            print(f"\n=== Available variables in {folder} ===")
            check_available_variables_xarray(ds)
            variables_checked = True
        
        # Calculate morphological time for main dataset
        start_timestamp = pd.Timestamp(run_startdate)
        times = pd.to_datetime(ds['time'].values)
        hydro_elapsed_years = np.array([(t - start_timestamp).days / 365.25 for t in times])
        
        if 'morfac' in ds:
            morfac_values = ds['morfac'].values
        else:
            morfac_values = np.full_like(hydro_elapsed_years, current_mf)
        
        morph_years = hydro_elapsed_years * morfac_values
        
        # If we have timed-out data, use the continuity in real time
        # The main dataset time continues from where timed-out ended
        if all_morph_years:  # Check if timed-out data was loaded
            # Get the last real time from timed-out dataset
            times_to_end = pd.to_datetime(ds_timed_out['time'].values[-1])
            # Use that as reference for main dataset to avoid double-adding offset
            hydro_elapsed_years_adjusted = np.array([(t - times_to_end).days / 365.25 for t in times])
            # Calculate morphological time relative to end of timed-out
            morph_years_offset = hydro_elapsed_years_adjusted * morfac_values
            # Add the last timed-out morph time to get absolute timeline
            morph_years = morph_years_offset + all_morph_years[-1]
            print(f"  DEBUG: Timed-out ended at Tmorph={all_morph_years[-1]:.1f}y, real time={times_to_end}")
            print(f"  DEBUG: Main first morph_year={morph_years[0]:.1f}y, last={morph_years[-1]:.1f}y")
        
        # Calculate volume for main dataset
        areas = ds[area_name]
        bed_levels = ds[var_name]
        total_volume = (bed_levels * areas).sum(dim=ds[var_name].dims[-1])
        
        # Use initial volume from timed-out if available, otherwise from main dataset
        if initial_volume is not None:
            # Calculate volume change relative to first main timestep
            volume_change = total_volume - total_volume.isel(time=0)
            # Shift it up by the final volume change from timed-out
            final_timed_out_volume_change = all_volume_change[-1] if all_volume_change else 0
            volume_change = volume_change + final_timed_out_volume_change
        else:
            volume_change = total_volume - total_volume.isel(time=0)
        
        # Store main results
        all_morph_years.extend(morph_years)
        all_volume_change.extend(volume_change.values)
        
        # Print continuity info
        if initial_volume is not None:
            print(f"  Main: {len(morph_years)} timesteps, continues from Tmorph={morph_years[0]:.1f}y to {morph_years[-1]:.1f}y")
        else:
            print(f"  Main: {len(morph_years)} timesteps, Tmorph range: {morph_years[0]:.1f}-{morph_years[-1]:.1f}y")
        
        # 5. Store data for later plotting
        data_by_morfac[int(current_mf)] = {
            'morph_years': all_morph_years,
            'volume_change': all_volume_change
        }
        
        print(f"Processed {mf_prefix}: {len(all_morph_years)} total timesteps, Tmorph={all_morph_years[0]:.1f}y to {all_morph_years[-1]:.1f}y")
        ds.close()

    except Exception as e:
        print(f"Error processing {folder}: {e}")
        
#%%
# ============================================================
# PLOTTING 
# ============================================================

print("\n" + "="*60)
print("PLOTTING PHASE")
print("="*60 + "\n")

plt.figure(figsize=(10, 6))

for morfac, data in sorted(data_by_morfac.items()):
    plt.plot(data['morph_years'], data['volume_change'], 
             label=f'MORFAC = {morfac}', marker='o', markersize=4, linewidth=2)

plt.xlabel('Morphological Time [years]')
plt.xlim(0, morfyears)
plt.ylabel('Change in Sediment Volume [m³]')
plt.title('Mass Balance: cumulative volume change since t = 0')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save figure with dynamic name
figure_name = f'1_{discharge_type}_MFsensitivity_massbalance_{tmorph_period}.png'
figure_path = base_directory.parent / figure_name
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {figure_path}")

plt.show()
# %%
