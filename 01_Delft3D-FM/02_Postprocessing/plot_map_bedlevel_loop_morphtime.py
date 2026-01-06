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

# Special-case MF50 reference (same logic as in first script)
special_base = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
special_config = r"Test_MORFAC\Tmorph_50years"
use_mf50_reference = False       # will be set after base_directory is defined

# For MORFAC: Get all subdirectories starting with 'MF'
if scenarios_morfac: 
    base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\Tmorph_50years"
    model_folders = [f for f in os.listdir(base_directory) if f.startswith('MF') and os.path.isdir(os.path.join(base_directory, f))]
    run_startdate = '2025-01-01'
    morfyears = 50
    
# For discharge: 
if scenarios_discharge:
    base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_OneRiverBoundary"
    model_folders = [f for f in os.listdir(base_directory) 
                 if os.path.isdir(os.path.join(base_directory, f)) 
                 and (f.startswith('01_'))] #or f.startswith('02_') or f.startswith('03_')
    run_startdate = '2001-01-01'

    # For variability: 
if scenarios_variability:
    base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_OneRiverBoundary"
    model_folders = [f for f in os.listdir(base_directory) 
                 if os.path.isdir(os.path.join(base_directory, f)) 
                 and (f.startswith('02_') or f.startswith('03_'))] #or f.startswith('02_') or f.startswith('03_')
    run_startdate = '2024-01-01'

print(f"Found {len(model_folders)} folders to process.")

# Determine if MF50 global reference should be used (only for the MORFAC scenario)
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

# --- OPTIONAL: GLOBAL REFERENCE FROM MF50 ---
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

# --- LOOP THROUGH DIRECTORIES ---
for folder in model_folders:
    model_location = os.path.join(base_directory, folder)
    
    # Create output_plots directory inside each model folder
    output_plots_dir = os.path.join(model_location, 'output_plots')
    os.makedirs(output_plots_dir, exist_ok=True)

    file_pattern = os.path.join(model_location, 'output', '*_map.nc')
    
    print(f"\nProcessing: {folder}")
    
    # Extract computation time from .dia file
    dia_file = os.path.join(model_location, 'output', 'FlowFM_0000.dia')
    comp_days, comp_hours = extract_computation_time(dia_file)
    
    if comp_days is not None and comp_hours is not None:
        print(f"  ** INFO   : total computation time (d)  : {comp_days:20.10f}")
        print(f"  ** INFO   : total computation time (h)  : {comp_hours:20.10f}")
    
    try:
        # Load the partitioned dataset
        ds = dfmt.open_partitioned_dataset(file_pattern)
        var_name = 'mesh2d_mor_bl'
        
        if var_name not in ds:
            print(f"Skipping {folder}: Variable {var_name} not found.")
            ds.close()
            continue
        
        # --- CHOOSE REFERENCE BED FOR DETRENDING ---
        reference_bed = None
        if apply_detrending:
            if use_mf50_reference and (reference_bed_MF50 is not None):
                # Use MF50 t=0 for all runs (same as first script)
                print(f"Using MF50 reference bed (time index {reference_time_idx}) for detrending of {folder}...")
                reference_bed = reference_bed_MF50
            else:
                # Per‑run reference at reference_time_idx
                if 'time' in ds[var_name].dims:
                    print(f"Storing per‑run reference bed (time index {reference_time_idx}) for {folder}...")
                    reference_bed = ds[var_name].isel(time=reference_time_idx).values.copy()
                else:
                    # If no time dimension, skip detrending
                    print(f"{folder}: no time dimension in {var_name}; detrending disabled for this run.")
                    reference_bed = None

        # Find the correct timestep based on morphological time
        if 'time' in ds[var_name].dims:
            closest_idx, actual_time, actual_hydro_years, actual_morph_years, current_morfac = \
                find_timestep_for_target_morphtime(ds, morfyears, run_startdate)

            print(f"  MORFAC: {current_morfac:.1f}")
            print(f"  Hydrodynamic time elapsed: {actual_hydro_years:.2f} years")
            print(f"  Morphological time: {actual_morph_years:.2f} years")
            print(f"  Hydrodynamic date: {actual_time}")
            print(f"  Time index: {closest_idx}")

            data_to_plot = ds[var_name].isel(time=closest_idx)

            # Apply detrending if possible
            if apply_detrending and (reference_bed is not None):
                print(f"Applying detrending at time index {closest_idx} for {folder}...")
                data_to_plot = data_to_plot - reference_bed

            timestamp_str = str(actual_time).split('.')[0].replace('T', ' ')
            title_info = f"{timestamp_str} | MORFAC = {current_morfac:.0f} | Tmorph = {actual_morph_years:.1f} years"
        else:
            data_to_plot = ds[var_name]
            if apply_detrending and (reference_bed is not None):
                print(f"Applying detrending (static map) for {folder}...")
                data_to_plot = data_to_plot - reference_bed
            title_info = "Static Map"

        # --- PLOTTING ---
        fig, ax = plt.subplots(figsize=(12, 8))

        # For detrended maps the range may need adjusting; here keep original vmin/vmax
        pc = data_to_plot.ugrid.plot(
            ax=ax,
            cmap=terrain_cmap,
            add_colorbar=False,
            edgecolors='none',
            vmin=-15,
            vmax=13
        )

        ax.set_aspect('equal')

        detrend_label = " (detrended)" if apply_detrending and (reference_bed is not None) else ""
        ax.set_title(f"Bed level{detrend_label} on {title_info}", color='black')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(pc, cax=cax)
        cbar.set_label('Bed Level [m]')

        plt.tight_layout()

        # Save in the specific model folder
        suffix = "_detrended" if apply_detrending and (reference_bed is not None) else ""
        save_name = f"terrain_map_final{suffix}_{folder}.png"
        save_path = os.path.join(output_plots_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Successfully saved: {save_name}")

        # Close plot and dataset to save memory
        plt.close(fig)
        ds.close()

    except Exception as e:
        print(f"Error processing {folder}: {e}")

print("\nBatch processing complete.")
# %%
#         # Find the correct timestep based on morphological time
#         if 'time' in ds[var_name].dims:
#             closest_idx, actual_time, actual_hydro_years, actual_morph_years, current_morfac = \
#                 find_timestep_for_target_morphtime(ds, morfyears, run_startdate)
            
#             print(f"  MORFAC: {current_morfac:.1f}")
#             print(f"  Hydrodynamic time elapsed: {actual_hydro_years:.2f} years")
#             print(f"  Morphological time: {actual_morph_years:.2f} years")
#             print(f"  Hydrodynamic date: {actual_time}")
#             print(f"  Time index: {closest_idx}")
            
#             data_to_plot = ds[var_name].isel(time=closest_idx)
            
#             timestamp_str = str(actual_time).split('.')[0].replace('T', ' ')
#             title_info = f"{timestamp_str} | MORFAC = {current_morfac:.0f} | Tmorph = {actual_morph_years:.1f} years"
#         else:
#             data_to_plot = ds[var_name]
#             title_info = "Static Map"

#         # --- PLOTTING ---
#         fig, ax = plt.subplots(figsize=(12, 8))
        
#         pc = data_to_plot.ugrid.plot(
#             ax=ax, 
#             cmap=terrain_cmap, 
#             add_colorbar=False, 
#             edgecolors='none',
#             vmin=-15,
#             vmax=13
#         )

#         ax.set_aspect('equal')
#         ax.set_title(f"Bed level on {title_info}", color='black')

#         # Add colorbar
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="3%", pad=0.1)
#         cbar = plt.colorbar(pc, cax=cax)
#         cbar.set_label('Bed Level [m]')

#         plt.tight_layout()

#         # Save in the specific model folder
#         save_name = f"terrain_map_final_{folder}.png"
#         save_path = os.path.join(output_plots_dir, save_name)
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.show()
#         print(f"Successfully saved: {save_name}")
        
#         # Close plot and dataset to save memory
#         plt.close(fig)
#         ds.close()

#     except Exception as e:
#         print(f"Error processing {folder}: {e}")

# print("\nBatch processing complete.")
# # %%
