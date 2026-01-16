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

# --- TOGGLE: Which morphological time periods to load and plot ---
load_tmorph_periods = [50, 400]  # Load both 50 and 400-year data
# Alternative: load_tmorph_periods = [50]  # Only 50-year data
# Alternative: load_tmorph_periods = [400]  # Only 400-year data

# Global storage for all loaded data (persists across runs)
if 'all_loaded_data' not in globals():
    all_loaded_data = {}  # Structure: {(tmorph_period, morfac): {'morph_years': [...], 'volume_change': [...]}}

var_name = 'mesh2d_mor_bl' # Bed level
area_name = 'mesh2d_flowelem_ba' # Area of cells

# Optional: Check available variables in the first dataset (set to False to skip)
check_vars = False

variables_checked = False

# ============================================================
# DATA LOADING SECTION
# ============================================================

for morfyears in load_tmorph_periods:
    tmorph_period = f'Tmorph_{morfyears}years'
    print(f"\n{'='*60}")
    print(f"Loading data for {tmorph_period}...")
    print(f"{'='*60}")
    
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC") / discharge_type / tmorph_period
    timed_out_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC") / discharge_type / tmorph_period / 'timed-out'
    all_folders = [f.name for f in base_directory.iterdir() if f.name.startswith('MF') and f.is_dir()]
    run_startdate = '2025-01-01'

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
            
            # 5. Store data for later plotting (using global storage)
            data_key = (tmorph_period, int(current_mf))
            all_loaded_data[data_key] = {
                'morph_years': all_morph_years,
                'volume_change': all_volume_change
            }
            
            print(f"Processed {mf_prefix}: {len(all_morph_years)} total timesteps, Tmorph={all_morph_years[0]:.1f}y to {all_morph_years[-1]:.1f}y")
            ds.close()

        except Exception as e:
            print(f"Error processing {folder}: {e}")

#%%
# ============================================================
# PLOTTING SECTION - Automatically uses loaded data
# ============================================================

print("\n" + "="*60)
print("PLOTTING PHASE")
print("="*60)
print(f"Available data keys: {list(all_loaded_data.keys())}\n")

# --- PLOTTING CONTROL ---
# Automatically use all loaded tmorph_periods
tmorph_periods_to_plot = [f'Tmorph_{y}years' for y in load_tmorph_periods]

# ============================================================
# PLOT 1: Combined plot (all loaded tmorph_periods)
# ============================================================
plt.figure(figsize=(12, 7))

# Sort by morfac values (low to high) and get colormap
relevant_keys = [k for k in all_loaded_data.keys() if k[0] in tmorph_periods_to_plot]
morfac_values = sorted(set([k[1] for k in relevant_keys]))
n_morfacs = len(morfac_values)

# Create colormap with better distinction (low to high)
cmap = plt.cm.get_cmap('viridis')
colors = [cmap(i / max(1, n_morfacs - 1)) for i in range(n_morfacs)]

# Map morfac to color
morfac_to_color = {morfac: colors[i] for i, morfac in enumerate(morfac_values)}

# Plot in sorted morfac order (low to high)
for morfac in morfac_values:
    for data_key in sorted(all_loaded_data.keys()):
        tmorph_period_key, key_morfac = data_key
        
        # Filter by selected tmorph_periods and matching morfac
        if tmorph_period_key not in tmorph_periods_to_plot or key_morfac != morfac:
            continue
        
        data = all_loaded_data[data_key]
        label = f'{morfac}'
        color = morfac_to_color[morfac]
        plt.plot(data['morph_years'], data['volume_change'], 
                 label=label, marker='o', markersize=4, linewidth=2, color=color)

plt.xlabel('Morphological Time [years]')
plt.xlim(0,50)
plt.ylim(0,1e7)
plt.ylabel('Change in Sediment Volume [m³]')
plt.title('Mass Balance: cumulative volume change since t = 0')
plt.legend(title='${MORFAC}$ =', loc='upper left', title_fontsize='large', fontsize='medium')
plt.grid(True, linestyle='--', alpha=0.6)

# Save figure with dynamic name
last_tmorph_period = f'Tmorph_{load_tmorph_periods[-1]}years'
figure_name = f'{discharge_type}_MFsensitivity_massbalance_combined.png'
figure_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC") / discharge_type / figure_name
plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"\nFigure 1 (Combined) saved to: {figure_path}")
plt.show()

# ============================================================
# PLOT 2: Only Tmorph_50years
# ============================================================
if 'Tmorph_50years' in [f'Tmorph_{y}years' for y in load_tmorph_periods]:
    plt.figure(figsize=(12, 7))
    
    tmorph_periods_to_plot_50 = ['Tmorph_50years']
    relevant_keys_50 = [k for k in all_loaded_data.keys() if k[0] in tmorph_periods_to_plot_50]
    morfac_values_50 = sorted(set([k[1] for k in relevant_keys_50]))
    n_morfacs_50 = len(morfac_values_50)
    
    cmap_50 = plt.cm.get_cmap('viridis')
    colors_50 = [cmap_50(i / max(1, n_morfacs_50 - 1)) for i in range(n_morfacs_50)]
    morfac_to_color_50 = {morfac: colors_50[i] for i, morfac in enumerate(morfac_values_50)}
    
    for morfac in morfac_values_50:
        for data_key in sorted(all_loaded_data.keys()):
            tmorph_period_key, key_morfac = data_key
            
            if tmorph_period_key not in tmorph_periods_to_plot_50 or key_morfac != morfac:
                continue
            
            data = all_loaded_data[data_key]
            label = f'{morfac}'
            color = morfac_to_color_50[morfac]
            plt.plot(data['morph_years'], data['volume_change'], 
                     label=label, marker='o', markersize=4, linewidth=2, color=color)
    
    plt.xlabel('Morphological Time [years]')
    plt.xlim(0, 50)
    plt.ylim(0, 1e7)
    plt.ylabel('Change in Sediment Volume [m³]')
    plt.title('Mass Balance (Tmorph = 50 years): cumulative volume change since t = 0')
    plt.legend(title='${MORFAC}$ =', loc='upper left', title_fontsize='large', fontsize='medium')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    figure_name_50 = f'{discharge_type}_MFsensitivity_massbalance_Tmorph_50years.png'
    figure_path_50 = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC") / discharge_type / figure_name_50
    plt.savefig(figure_path_50, dpi=300, bbox_inches='tight')
    print(f"Figure 2 (Tmorph_50years) saved to: {figure_path_50}")
    plt.show()

# ============================================================
# PLOT 3: Only Tmorph_400years
# ============================================================
if 'Tmorph_400years' in [f'Tmorph_{y}years' for y in load_tmorph_periods]:
    plt.figure(figsize=(12, 7))
    
    tmorph_periods_to_plot_400 = ['Tmorph_400years']
    relevant_keys_400 = [k for k in all_loaded_data.keys() if k[0] in tmorph_periods_to_plot_400]
    morfac_values_400 = sorted(set([k[1] for k in relevant_keys_400]))
    n_morfacs_400 = len(morfac_values_400)
    
    cmap_400 = plt.cm.get_cmap('viridis')
    colors_400 = [cmap_400(i / max(1, n_morfacs_400 - 1)) for i in range(n_morfacs_400)]
    morfac_to_color_400 = {morfac: colors_400[i] for i, morfac in enumerate(morfac_values_400)}
    
    for morfac in morfac_values_400:
        for data_key in sorted(all_loaded_data.keys()):
            tmorph_period_key, key_morfac = data_key
            
            if tmorph_period_key not in tmorph_periods_to_plot_400 or key_morfac != morfac:
                continue
            
            data = all_loaded_data[data_key]
            label = f'{morfac}'
            color = morfac_to_color_400[morfac]
            plt.plot(data['morph_years'], data['volume_change'], 
                     label=label, marker='o', markersize=4, linewidth=2, color=color)
    
    plt.xlabel('Morphological Time [years]')
    plt.xlim(0, 400)
    plt.ylabel('Change in Sediment Volume [m³]')
    plt.title('Mass Balance (Tmorph = 400 years): cumulative volume change since t = 0')
    plt.legend(title='${MORFAC}$ =', loc='upper left', title_fontsize='large', fontsize='medium')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    figure_name_400 = f'{discharge_type}_MFsensitivity_massbalance_Tmorph_400years.png'
    figure_path_400 = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC") / discharge_type / figure_name_400
    plt.savefig(figure_path_400, dpi=300, bbox_inches='tight')
    print(f"Figure 3 (Tmorph_400years) saved to: {figure_path_400}")
    plt.show()

# %%
