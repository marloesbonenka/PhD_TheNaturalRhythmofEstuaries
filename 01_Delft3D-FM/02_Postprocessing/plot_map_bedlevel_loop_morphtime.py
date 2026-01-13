"""Batch plot final bed levels for all MF folders"""
#%% 
import os
import matplotlib.pyplot as plt
import dfm_tools as dfmt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import sys

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\Delft3D-FM\Postprocessing")

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *

#%%
# --- 1. SETTINGS & PATHS ---
scenarios_morfac = True
scenarios_discharge = False 
scenarios_variability = False 
apply_detrending = True
reference_time_idx = 0

var_name = 'mesh2d_mor_bl'

# Special-case MF50 reference (same logic as in first script)
special_base = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
special_config = r"Test_MORFAC\Tmorph_50years"
use_mf50_reference = False       # will be set after base_directory is defined

# For MORFAC: Get all subdirectories starting with 'MF'
if scenarios_morfac: 
    base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\Tmorph_400years"
    all_folders = [f for f in os.listdir(base_directory) if f.startswith('MF') and os.path.isdir(os.path.join(base_directory, f))]
    run_startdate = '2025-01-01'
    morfyears = 400

    # --- COMPARISON SETTINGS ---
    compare_against_baseline = False
    baseline_prefix = 'MF100' 
    comparison_vmin = -2.0  
    comparison_vmax = 2.0
    
    # Identify the baseline folder from the full list for pre-loading later
    baseline_search = [f for f in all_folders if f.startswith(baseline_prefix)]
    
    # Logic to handle the folder list based on mode
    if compare_against_baseline:
        # Only process other folders for difference maps
        model_folders = [f for f in all_folders if not f.startswith(baseline_prefix)]
    else:
        # Process everything for standard plots
        model_folders = all_folders

# For discharge: 
if scenarios_discharge:
    base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_OneRiverBoundary"
    model_folders = [f for f in os.listdir(base_directory) 
                 if os.path.isdir(os.path.join(base_directory, f)) 
                 and (f.startswith('01_'))] #or f.startswith('02_') or f.startswith('03_')
    run_startdate = '2001-01-01'
    morfyears = 2000

    # For variability: 
if scenarios_variability:
    base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_FourRiverBoundaries"
    model_folders = [f for f in os.listdir(base_directory) 
                 if os.path.isdir(os.path.join(base_directory, f)) 
                 and (f.startswith('02_') or f.startswith('03_'))] #or f.startswith('02_') or f.startswith('03_')
    run_startdate = '2024-01-01'
    morfyears = 2000

print(f"Found {len(model_folders)} folders to process.")

# Determine if reference t = 0 from a run with no restart (only for the MORFAC scenario)
if scenarios_morfac:
    # Match logic of first script
    base_root = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
    config_rel = os.path.relpath(base_directory, base_root)
    use_mf50_reference = (
        base_root == special_base and
        config_rel.replace('\\', '/') == special_config.replace('\\', '/')
    )
else:
    use_mf50_reference = False

# --- For detrending: reference t=0 from a run with no restart ---
reference_bed_MF50 = None
if apply_detrending and use_mf50_reference:
    mf50_folder = [f for f in model_folders if get_mf_number(f) == 50]
    if len(mf50_folder) == 1:
        mf50_folder = mf50_folder[0]
        mf50_location = os.path.join(base_directory, mf50_folder)
        mf50_pattern = os.path.join(mf50_location, 'output', '*_map.nc')
        ds_mf50 = dfmt.open_partitioned_dataset(mf50_pattern)
        if 'mesh2d_mor_bl' in ds_mf50:
            reference_bed_MF50 = ds_mf50['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()
            print("Using MF50 reference bed for detrending of all runs.")
        else:
            print("MF50 dataset does not contain mesh2d_mor_bl; falling back to per‑run reference.")
            use_mf50_reference = False
        ds_mf50.close()
    else:
        print("No unique MF50 folder found; falling back to per‑run reference.")
        use_mf50_reference = False

# --- preload baseline scenario for comparison ---
baseline_bed_data = None
if scenarios_morfac and compare_against_baseline:
    if baseline_search:
        baseline_folder_name = baseline_search[0]
        base_path = os.path.join(base_directory, baseline_folder_name, 'output', '*_map.nc')
        print(f"\n--- Loading Baseline Scenario for Comparison: {baseline_folder_name} ---")
        
        with dfmt.open_partitioned_dataset(base_path) as ds_base:
            # Match the target morph time used for all other plots
            idx_b, _, _, _, _ = find_timestep_for_target_morphtime(ds_base, morfyears, run_startdate)
            # We use .values.copy() to keep the data in memory after closing the file
            baseline_bed_data = ds_base[var_name].isel(time=idx_b).values.copy()
            print(f"Baseline loaded successfully from index {idx_b}.")
    else:
        print(f"Warning: Baseline prefix '{baseline_prefix}' not found in model_folders.")

# --- loop through directories ---
for folder in model_folders:
    model_location = os.path.join(base_directory, folder)
    output_plots_dir = os.path.join(model_location, 'output_plots')
    os.makedirs(output_plots_dir, exist_ok=True)

    file_pattern = os.path.join(model_location, 'output', '*_map.nc')
    print(f"\nProcessing: {folder}")

    try:
        # 1. load data
        ds = dfmt.open_partitioned_dataset(file_pattern)
        if var_name not in ds:
            print(f"Skipping {folder}: Variable {var_name} not found.")
            ds.close(); continue

        # 2. define reference bed (for detrending scenario)
        reference_bed = None
        if apply_detrending:
            if use_mf50_reference and (reference_bed_MF50 is not None):
                reference_bed = reference_bed_MF50
            elif 'time' in ds[var_name].dims:
                reference_bed = ds[var_name].isel(time=reference_time_idx).values.copy()

        # 3. extract target data
        if 'time' in ds[var_name].dims:
            closest_idx, actual_time, actual_hydro_years, actual_morph_years, current_morfac = \
                find_timestep_for_target_morphtime(ds, morfyears, run_startdate)
            data_to_plot = ds[var_name].isel(time=closest_idx)
            
            # optional detrending
            if apply_detrending and (reference_bed is not None):
                data_to_plot = data_to_plot - reference_bed
            
            timestamp_str = str(actual_time).split('.')[0].replace('T', ' ')
            title_info = f"{timestamp_str} | MF={current_morfac:.0f} | Tmorph={actual_morph_years:.1f}y"
        else:
            data_to_plot = ds[var_name]
            if apply_detrending and (reference_bed is not None):
                data_to_plot = data_to_plot - reference_bed
            title_info = "Static Map"

        # 4. choose plots
        if not compare_against_baseline:
            # --- standard bed level plots ---
            fig, ax = plt.subplots(figsize=(12, 8))
            pc = data_to_plot.ugrid.plot(ax=ax, cmap=terrain_cmap, add_colorbar=False, 
                                        edgecolors='none', vmin=-15, vmax=15)
            ax.set_aspect('equal')
            det_suf = " (detrended)" if apply_detrending else ""
            ax.set_title(f"Bed level{det_suf}: {folder}\n{title_info}")
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(pc, cax=cax).set_label('Bed Level [m]')
            
            save_name = f"terrain_map_final{'_detrended' if apply_detrending else ''}_{folder}.png"
            plt.savefig(os.path.join(output_plots_dir, save_name), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)

        elif compare_against_baseline and baseline_bed_data is not None:
            # --- difference maps ---
            print(f"Generating difference map: {folder} - {baseline_prefix}")
            diff_values = data_to_plot - baseline_bed_data

            fig_diff, ax_diff = plt.subplots(figsize=(12, 8))
            pc_diff = diff_values.ugrid.plot(ax=ax_diff, cmap='RdBu_r', add_colorbar=False, 
                                            edgecolors='none', vmin=comparison_vmin, vmax=comparison_vmax)
            ax_diff.set_aspect('equal')
            ax_diff.set_title(f"Bed Level Difference: {folder} vs {baseline_prefix}\n(Red=Accretion, Blue=Erosion)")

            divider_diff = make_axes_locatable(ax_diff)
            cax_diff = divider_diff.append_axes("right", size="3%", pad=0.1)
            plt.colorbar(pc_diff, cax=cax_diff).set_label('Difference [m]')

            save_name_diff = f"difference_map_{folder}_vs_{baseline_prefix}.png"
            plt.savefig(os.path.join(output_plots_dir, save_name_diff), dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_diff)

        ds.close()

    except Exception as e:
        print(f"Error processing {folder}: {e}")