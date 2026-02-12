"""Batch plot final bed levels for all MF folders"""
#%% 
from pathlib import Path
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
from FUNCTIONS.F_cache import DatasetCache, load_results_cache, save_results_cache

#%%
# --- 1. SETTINGS & PATHS ---
scenarios_morfac = False
scenarios_discharge = False 
scenarios_variability = False 
apply_detrending = True
reference_time_idx = 0

var_name = 'mesh2d_mor_bl' #, 'mesh2d_taus', 'mesh2d_s1'

# Special-case MF50 reference (same logic as in first script)
special_base = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
special_config = r"Test_MORFAC\03_flashy\Tmorph_50years"
use_mf50_reference = False       # will be set after base_directory is defined

# For MORFAC: Get all subdirectories starting with 'MF'
if scenarios_morfac: 
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\03_flashy\Tmorph_400years")
    all_folders = [p.name for p in base_directory.iterdir() if p.is_dir() and p.name.startswith('MF')]
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
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_OneRiverBoundary")
    model_folders = [p.name for p in base_directory.iterdir()
                 if p.is_dir()
                 and (p.name.startswith('01_'))] #or p.name.startswith('02_') or p.name.startswith('03_')
    run_startdate = '2001-01-01'
    morfyears = 2000

# For variability: 
if scenarios_variability:
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output")
    model_folders = ['1_Q500_rst.9093769']
                # [p.name for p in base_directory.iterdir()
                #  if p.is_dir()
                #  and (p.name.startswith('02_') or p.name.startswith('03_'))]
                 
    run_startdate = '2024-01-01'
    morfyears = 3000

print(f"Found {len(model_folders)} folders to process.")

dataset_cache = DatasetCache()
cache_dir = base_directory / "output_plots" / "_cache"
cache_path = cache_dir / f"bedlevel_cache_{var_name}.pkl"
try:
    # Determine if reference t = 0 from a run with no restart (only for the MORFAC scenario)
    if scenarios_morfac:
        # Match logic of first script
        base_root = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
        config_rel = base_directory.relative_to(base_root).as_posix()
        use_mf50_reference = (
            base_root == special_base and
            config_rel == Path(special_config).as_posix()
        )
    else:
        use_mf50_reference = False

    # --- load cached reference data if available ---
    reference_bed_MF50 = None
    baseline_bed_data = None
    cache_dirty = False
    cache_settings = {
        'base_directory': str(base_directory),
        'model_folders': model_folders,
        'var_name': var_name,
        'reference_time_idx': reference_time_idx,
        'morfyears': morfyears,
        'run_startdate': run_startdate,
        'apply_detrending': apply_detrending,
        'compare_against_baseline': compare_against_baseline if scenarios_morfac else False,
        'baseline_prefix': baseline_prefix if (scenarios_morfac and compare_against_baseline) else None,
        'use_mf50_reference': use_mf50_reference,
    }

    loaded_results, loaded_meta = load_results_cache(cache_path, cache_settings)
    if loaded_results is not None:
        reference_bed_MF50 = loaded_results.get('reference_bed_MF50')
        baseline_bed_data = loaded_results.get('baseline_bed_data')
        print(f"Loaded cached reference data from: {cache_path}")
    else:
        print(f"Cache not found or settings differ, computing reference data...")
        cache_dirty = True

    # --- For detrending: reference t=0 from a run with no restart ---
    if apply_detrending and use_mf50_reference and reference_bed_MF50 is None:
        mf50_folder = [f for f in model_folders if get_mf_number(f) == 50]
        if len(mf50_folder) == 1:
            mf50_folder = mf50_folder[0]
            mf50_location = base_directory / mf50_folder
            mf50_pattern = str(mf50_location / 'output' / '*_map.nc')
            ds_mf50 = dataset_cache.get_partitioned(mf50_pattern)
            if 'mesh2d_mor_bl' in ds_mf50:
                reference_bed_MF50 = ds_mf50['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()
                print("Using MF50 reference bed for detrending of all runs.")
            else:
                print("MF50 dataset does not contain mesh2d_mor_bl; falling back to per‑run reference.")
                use_mf50_reference = False
        else:
            print("No unique MF50 folder found; falling back to per‑run reference.")
            use_mf50_reference = False

    # --- preload baseline scenario for comparison ---
    if scenarios_morfac and compare_against_baseline and baseline_bed_data is None:
        if baseline_search:
            baseline_folder_name = baseline_search[0]
            base_path = str(base_directory / baseline_folder_name / 'output' / '*_map.nc')
            print(f"\n--- Loading Baseline Scenario for Comparison: {baseline_folder_name} ---")

            ds_base = dataset_cache.get_partitioned(base_path)
            # Match the target morph time used for all other plots
            idx_b, _, _, _, _ = find_timestep_for_target_morphtime(ds_base, morfyears, run_startdate)
            # We use .values.copy() to keep the data in memory after closing the file
            baseline_bed_data = ds_base[var_name].isel(time=idx_b).values.copy()
            print(f"Baseline loaded successfully from index {idx_b}.")
        else:
            print(f"Warning: Baseline prefix '{baseline_prefix}' not found in model_folders.")

    # --- loop through directories ---
    for folder in model_folders:
        model_location = base_directory / folder
        output_plots_dir = model_location / 'output_plots'
        output_plots_dir.mkdir(parents=True, exist_ok=True)

        file_pattern = str(model_location / 'output' / '*_map.nc')
        print(f"\nProcessing: {folder}")

        try:
            # 1. load data
            ds = dataset_cache.get_partitioned(file_pattern)
            if var_name not in ds:
                print(f"Skipping {folder}: Variable {var_name} not found.")
                continue

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
                plt.savefig(output_plots_dir / save_name, dpi=300, bbox_inches='tight')
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
                plt.savefig(output_plots_dir / save_name_diff, dpi=300, bbox_inches='tight')
                plt.show()
                plt.close(fig_diff)

        except Exception as e:
            print(f"Error processing {folder}: {e}")

    if cache_dirty:
        save_results_cache(
            cache_path,
            results={
                'reference_bed_MF50': reference_bed_MF50,
                'baseline_bed_data': baseline_bed_data,
            },
            settings=cache_settings,
        )
        print(f"Saved reference cache to: {cache_path}")
finally:
    dataset_cache.close_all()