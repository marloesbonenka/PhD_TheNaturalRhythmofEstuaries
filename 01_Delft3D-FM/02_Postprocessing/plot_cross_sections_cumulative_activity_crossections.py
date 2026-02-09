"""Cumulative activity on cross-section bed profiles from MAP files.

This script:
- Extracts bed profiles directly from MAP files using x-coordinates
- Computes cumulative activity: Σ|Δz| over time at each transect point
- Plots:
  (top) heatmap of cumulative activity (time vs cross-section distance)
  (bottom) first bedlevel profile (t=0)

Notes
-----
- Uses restart stitching pattern for complete time series
- Supports both MORFAC sensitivity analysis and discharge variability analysis
- Includes disk caching for fast reruns
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from tqdm import tqdm

# Add path for FUNCTIONS
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import get_mf_number
from FUNCTIONS.F_cache import (
    DatasetCache,
    get_profile_cache_path, load_profile_cache, save_profile_cache
)
from FUNCTIONS.F_braiding_index import get_bed_profile
from FUNCTIONS.F_morphological_activity import (
    cumulative_activity,
    morph_years_from_datetimes,
    plot_activity_and_first_profile,
)


# =============================================================================
# Configuration
# =============================================================================

# ANALYSIS_MODE: "variability" for river discharge variability scenarios
#                "morfac" for MORFAC sensitivity analysis
ANALYSIS_MODE = "variability"

if ANALYSIS_MODE == "variability":
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
    config = 'Model_Output'
    # Mapping: restart folder prefix -> timed-out folder prefix
    # 1 = constant (baserun), 2 = seasonal, 3 = flashy, 4 = singlepeak
    VARIABILITY_MAP = {
        '1': '01_baserun500',
        '2': '02_run500_seasonal',
        '3': '03_run500_flashy',
        '4': '04_run500_singlepeak',
    }
    # Which scenarios to process (set to None or empty list for all)
    SCENARIOS_TO_PROCESS = ['1']  # e.g., ['1'] for baserun only, None for all
    use_folder_morfac = False
    default_morfac = 100  # MORFAC used in variability runs

elif ANALYSIS_MODE == "morfac":
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
    config = r'TestingBoundaries_and_SensitivityAnalyses\Test_MORFAC\02_seasonal\Tmorph_50years'
    use_folder_morfac = True  # Extract MORFAC from folder name (MF1, MF2, etc.)
    default_morfac = 1.0

timed_out_dir = base_directory / config / "timed-out"

# --- CROSS-SECTION SETTINGS ---
# X-coordinates in meters (km value * 1000)
selected_x_coords = [20000, 30000, 40000]

# Y-range of the estuary (min, max) and sampling resolution
# Based on MAP file: mesh2d_face_y ranges from 337.5 to 15000.5 m
y_range = (300, 15000)
n_y_samples = 300

# Bedlevel variable in MAP
map_bedlevel_var = "mesh2d_mor_bl"

# Masking threshold for land
bedlevel_land_threshold = 8.0  # set None to disable

# Time control (stride > 1 to reduce runtime)
time_stride = 1

# Morph-time conversion
run_startdate = None  # e.g. "2025-01-01", or None to use first timestamp

# --- CACHE SETTINGS ---
compute = False  # Set True to force recompute, False to use cache if available

# Output
output_dirname = "output_plots_crosssections_cumactivity"


# =============================================================================
# Search & Sort Folders
# =============================================================================

base_path = base_directory / config
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")

if ANALYSIS_MODE == "variability":
    # Find restart folders: start with digit and contain "_rst"
    model_folders = [f.name for f in base_path.iterdir() 
                     if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
    # Filter by SCENARIOS_TO_PROCESS if specified
    if SCENARIOS_TO_PROCESS:
        model_folders = [f for f in model_folders if f.split('_')[0] in SCENARIOS_TO_PROCESS]
    # Sort by leading number
    model_folders.sort(key=lambda x: int(x.split('_')[0]))
elif ANALYSIS_MODE == "morfac":
    model_folders = [f.name for f in base_path.iterdir() 
                     if f.is_dir() and f.name.startswith('MF')]
    model_folders.sort(key=get_mf_number)

print(f"Found {len(model_folders)} run folders in: {base_path}")


# =============================================================================
# Main Processing Loop
# =============================================================================

dataset_cache = DatasetCache()
output_dir = base_path / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)

for folder in model_folders:
    model_location = base_path / folder
    
    # --- UNIFIED CACHE ---
    cache_path = get_profile_cache_path(model_location, folder)
    cache_settings_profiles = {
        'y_range': y_range,
        'n_y_samples': n_y_samples,
        'bedlevel_land_threshold': bedlevel_land_threshold,
        'time_stride': time_stride,
        'map_bedlevel_var': map_bedlevel_var,
    }
    
    folder_results = {}
    missing_x_coords = selected_x_coords  # by default, compute all
    
    # Try to load from unified cache
    if not compute:
        loaded_results, missing_x_coords = load_profile_cache(
            cache_path, selected_x_coords, cache_settings_profiles
        )
        if loaded_results:
            # Convert loaded format to our working format (handle key name compatibility)
            for cs_name, data in loaded_results.items():
                # Handle times vs time_series key name
                times_data = data.get('times') or data.get('time_series')
                folder_results[cs_name] = {
                    'profiles': data['profiles'],
                    'times': times_data,
                    'dist': data['dist'],
                    'morfac': data.get('morfac', default_morfac),
                }
            print(f"\n" + "="*60)
            print(f"LOADED FROM CACHE: {folder} ({len(loaded_results)} cross-sections)")
            if missing_x_coords:
                print(f"  Still need to compute: {missing_x_coords}")
            else:
                # Nothing to compute - skip to analysis
                pass

    # --- 1. RESTART LOGIC (Find all parts) ---
    all_run_paths = []
    
    if ANALYSIS_MODE == "variability":
        scenario_num = folder.split('_')[0]
        if scenario_num in VARIABILITY_MAP and timed_out_dir.exists():
            timed_out_folder = VARIABILITY_MAP[scenario_num]
            timed_out_path = timed_out_dir / timed_out_folder
            if timed_out_path.exists():
                all_run_paths.append(timed_out_path)
                
    elif ANALYSIS_MODE == "morfac":
        if 'restart' in folder.lower() and timed_out_dir.exists():
            mf_prefix = folder.split('_')[0]
            matches = [f.name for f in timed_out_dir.iterdir() if f.name.startswith(mf_prefix)]
            if matches:
                all_run_paths.append(timed_out_dir / matches[0])
    
    all_run_paths.append(model_location)

    # Determine MORFAC
    if use_folder_morfac:
        morfac = float(get_mf_number(folder))
    else:
        morfac = default_morfac

    # --- 2. COMPUTE MISSING CROSS-SECTIONS ---
    if missing_x_coords:
        print(f"\n" + "="*60)
        print(f"PROCESSING: {folder}")
        print(f"Computing {len(missing_x_coords)} cross-sections, MORFAC={morfac}")
        print(f"Stitching {len(all_run_paths)} parts")

        loaded_datasets = []
        loaded_trees = []
        new_results = {}

        try:
            # Load MAP parts + KDTree
            for p_path in all_run_paths:
                print(f"   -> Opening Map: {p_path.name}")
                map_pattern = str(p_path / 'output' / '*_map.nc')
                ds_map = dataset_cache.get_partitioned(map_pattern, chunks={'time': 1})
                if map_bedlevel_var not in ds_map:
                    raise KeyError(f"{map_bedlevel_var} not found in MAP for {p_path}")

                face_x = ds_map['mesh2d_face_x'].values
                face_y = ds_map['mesh2d_face_y'].values
                tree = cKDTree(np.vstack([face_x, face_y]).T)

                loaded_datasets.append(ds_map)
                loaded_trees.append(tree)

            # Create y-coordinates for sampling
            y_samples = np.linspace(y_range[0], y_range[1], n_y_samples)

            # Process only missing x-coordinates
            for x_coord in missing_x_coords:
                cs_name = f"km{int(x_coord / 1000)}"
                print(f"   Analyzing x = {x_coord}m ({cs_name})...")

                cs_x = np.full(n_y_samples, x_coord)
                cs_y = y_samples
                dist = y_samples - y_samples[0]

                all_times = []
                all_profiles = []

                # Extract profiles sequentially over stitched parts
                for ds_map, tree in zip(loaded_datasets, loaded_trees):
                    time_vals = pd.to_datetime(ds_map.time.values)
                    idxs = range(0, len(time_vals), int(time_stride))
                    for t in tqdm(idxs, desc=f"      Timesteps", leave=False):
                        profile = get_bed_profile(ds_map, tree, cs_x, cs_y, t)
                        if bedlevel_land_threshold is not None:
                            profile = profile.copy()
                            profile[profile > float(bedlevel_land_threshold)] = np.nan
                        all_times.append(time_vals[t])
                        all_profiles.append(profile)

                # Clean up overlaps between restart parts
                df_idx = pd.DataFrame({'time': all_times, 'p_idx': np.arange(len(all_profiles))})
                df_idx = df_idx.drop_duplicates('time').sort_values('time')

                profiles_clean = [all_profiles[int(i)] for i in df_idx['p_idx'].values]
                times_clean = pd.to_datetime(df_idx['time'].values)

                new_results[cs_name] = {
                    'profiles': profiles_clean,
                    'times': times_clean,
                    'dist': dist,
                    'morfac': morfac,
                }
                folder_results[cs_name] = new_results[cs_name]

            # Save newly computed results to unified cache
            save_profile_cache(cache_path, new_results, cache_settings_profiles)
            print(f"   Saved {len(new_results)} cross-sections to cache: {cache_path}")

        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue
        finally:
            for ds in loaded_datasets:
                ds.close()

    # --- 3. PLOTTING (from cache or freshly computed results) ---
    try:
        for x_coord in selected_x_coords:
            cs_name = f"km{int(x_coord / 1000)}"
            if cs_name not in folder_results:
                continue

            data = folder_results[cs_name]
            profiles_clean = data['profiles']
            times_clean = data['times']
            dist = data['dist']
            morfac = data.get('morfac', default_morfac)

            Z = np.vstack([p[None, :] for p in profiles_clean])
            cum = cumulative_activity(Z)

            morph_years = morph_years_from_datetimes(
                pd.to_datetime(times_clean), 
                startdate=run_startdate, 
                morfac=morfac
            )
            first_profile = Z[0, :]

            outpath = output_dir / f"{folder}_{cs_name}_cumactivity.png"
            plot_activity_and_first_profile(
                dist_m=dist,
                first_profile=first_profile,
                cumact=cum,
                morph_years=morph_years,
                title=f"{folder}: {cs_name}",
                outpath=outpath,
                show=False,
            )
            print(f"  Saved: {outpath}")

    except Exception as e:
        print(f"Error plotting {folder}: {e}")
    finally:
        plt.close('all')

dataset_cache.close_all()

print("\n" + "="*60)
print("ALL FOLDERS COMPLETED.")
