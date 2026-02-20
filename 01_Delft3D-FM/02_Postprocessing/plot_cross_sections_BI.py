"""
Analyze bed level and braiding index at cross-sections from Delft3D-FM .his/map output
Robust version: Handles restarts with mesh changes and optimizes U-drive access.
"""

#%% IMPORTS
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys

#%% Add path for succesful loading of functions
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *
from FUNCTIONS.F_cache import *
from FUNCTIONS.F_loaddata import read_discharge_from_bc_files, extract_discharge_at_x

#%% --- CONFIGURATION ---
# ANALYSIS_MODE: "variability" for river discharge variability scenarios
#                "morfac" for MORFAC sensitivity analysis
ANALYSIS_MODE = "variability"


if ANALYSIS_MODE == "variability":
    DISCHARGE = 500  # or 1000, etc.
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
    config = f'Model_Output/Q{DISCHARGE}'
    default_morfac = 100  # All variability scenarios use MORFAC=100
    # Mapping: restart folder prefix -> timed-out folder prefix
    # 1 = constant (baserun), 2 = seasonal, 3 = flashy, 4 = singlepeak
    VARIABILITY_MAP = {
        '1': f'01_baserun{DISCHARGE}',
        '2': f'02_run{DISCHARGE}_seasonal',
        '3': f'03_run{DISCHARGE}_flashy',
        '4': f'04_run{DISCHARGE}_singlepeak',
    }
    # Which scenarios to process (set to None or empty list for all)
    SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']  # e.g., ['1'] for baserun only, ['1', '2'] for multiple, None for all

elif ANALYSIS_MODE == "morfac":
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
    config = r'TestingBoundaries_and_SensitivityAnalyses\Test_MORFAC\02_seasonal\Tmorph_50years'
    default_morfac = None  # Will extract from folder name

timed_out_dir = base_directory / config / "timed-out"

# --- SETTINGS ---
n_slices = 5  
safety_buffer = 0.20 

# Cross-section x-coordinates in meters (km value * 1000)
# e.g., km20 -> 20000m, km26 -> 26000m
selected_x_coords = [20000, 30000, 40000]

# Y-range of the estuary (min, max) and sampling resolution
# Based on MAP file: mesh2d_face_y ranges from 337.5 to 15000.5 m
y_range = (5000, 10000)  # Estuary width in meters
n_y_samples = 150  # Number of points to sample across the width

# X-coordinate for discharge extraction (km 40 = 40000 m)
DISCHARGE_X_COORD = 40000

# --- CACHE SETTINGS ---
compute = False  # Set True to force recompute, False to use cache if available

# Settings dict for cache validation (if any setting changes, cache is invalidated)
cache_settings = {
    'selected_x_coords': selected_x_coords,
    'y_range': y_range,
    'n_y_samples': n_y_samples,
    'safety_buffer': safety_buffer,
    'n_slices': n_slices,
}

#%% --- SEARCH & SORT FOLDERS ---
# For variability: restart folders start with digit (e.g., "1_Q500_rst...")
# For morfac: restart folders start with "MF" (e.g., "MF1_restart...")
if ANALYSIS_MODE == "variability":
    # Find restart folders: start with digit and contain "_rst"
    model_folders = [f.name for f in (base_directory / config).iterdir() 
                     if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
    # Filter by SCENARIOS_TO_PROCESS if specified
    if SCENARIOS_TO_PROCESS:
        model_folders = [f for f in model_folders if f.split('_')[0] in SCENARIOS_TO_PROCESS]
    # Sort by leading number (e.g., "1_Q500..." -> 1)
    model_folders.sort(key=lambda x: int(x.split('_')[0]))
elif ANALYSIS_MODE == "morfac":
    model_folders = [f.name for f in (base_directory / config).iterdir() 
                     if f.is_dir() and f.name.startswith('MF')]
    model_folders.sort(key=get_mf_number)

#%% --- MAIN PROCESSING LOOP ---

dataset_cache = DatasetCache()
for folder in model_folders:
    model_location = base_directory / config / folder
    
    # --- UNIFIED CACHE: shared between all profile-based scripts ---
    cache_path = get_profile_cache_path(model_location, folder)
    
    # Try to load from unified cache (only loads x-coords we need)
    folder_results, missing_x_coords = load_profile_cache(cache_path, selected_x_coords)
    
    # If loaded from cache, ensure bi_series exists (compute from profiles if needed)
    for cs_name, data in folder_results.items():
        # Handle key name compatibility (times vs time_series)
        if 'time_series' not in data and 'times' in data:
            data['time_series'] = data['times']
        # Compute bi_series if not cached (e.g., loaded from cumactivity cache)
        if 'bi_series' not in data and 'profiles' in data:
            print(f"   Computing braiding index for {cs_name} from cached profiles...")
            data['bi_series'] = []
            for profile in data['profiles']:
                plot_profile = profile.copy()
                plot_profile[plot_profile > 8.0] = np.nan
                bi = compute_braiding_index_with_threshold(plot_profile, safety_buffer=safety_buffer)
                data['bi_series'].append(bi)
    
    if not compute and not missing_x_coords:
        print(f"\n" + "="*60)
        print(f"LOADED FROM CACHE: {folder} ({len(folder_results)} cross-sections)")
        use_cache = True
    elif not compute and folder_results:
        print(f"\n" + "="*60)
        print(f"PARTIAL CACHE: {folder}")
        print(f"   Found: {list(folder_results.keys())}")
        print(f"   Missing: {[f'km{int(x/1000)}' for x in missing_x_coords]}")
        use_cache = False  # Need to compute missing ones
    else:
        if not compute:
            print(f"\nNo cache found for {folder}, computing...")
        use_cache = False
        missing_x_coords = selected_x_coords  # Compute all
    
    # --- 1. GET DISCHARGE (Priority: BC files > Cache > Map extraction) ---
    discharge_cache_key = "discharge_bc"
    all_discharge_times = []
    all_discharge_values = []
    discharge_source = None

    print("Attempting to read discharge from BC input files...")
    bc_times, bc_values = read_discharge_from_bc_files(model_location)

    if bc_times is not None:
        all_discharge_times = bc_times
        all_discharge_values = bc_values
        discharge_source = "BC files"
    else:
        for cs_name, data in folder_results.items():
            if discharge_cache_key in data:
                all_discharge_times = data[discharge_cache_key]['times']
                all_discharge_values = data[discharge_cache_key]['values']
                discharge_source = "cache"
                print(f"Loaded discharge from cache ({len(all_discharge_times)} timesteps).")
                break

    need_discharge_from_map = not bool(all_discharge_times)

    # --- 2. RESTART LOGIC (Find all parts) ---
    all_run_paths = []
    
    if ANALYSIS_MODE == "variability":
        # Variability mode: match by leading digit (e.g., "1_Q500_rst..." -> "01_baserun500")
        scenario_num = folder.split('_')[0]  # e.g., "1", "2", "3", "4"
        if scenario_num in VARIABILITY_MAP and timed_out_dir.exists():
            timed_out_folder = VARIABILITY_MAP[scenario_num]
            timed_out_path = timed_out_dir / timed_out_folder
            if timed_out_path.exists():
                all_run_paths.append(timed_out_path)
                
    elif ANALYSIS_MODE == "morfac":
        # Morfac mode: match by MF prefix (e.g., "MF2_restart..." -> "MF2...")
        if 'restart' in folder:
            mf_prefix = folder.split('_')[0]  # e.g., "MF1", "MF2"
            if timed_out_dir.exists():
                match = [f.name for f in timed_out_dir.iterdir() if f.name.startswith(mf_prefix)]
                if match:
                    all_run_paths.append(timed_out_dir / match[0])
    
    all_run_paths.append(model_location)

    # --- 3. COMPUTE MISSING CROSS-SECTIONS / DISCHARGE ---
    need_profiles_from_map = (not folder_results) or bool(missing_x_coords)
    need_map_open = need_profiles_from_map or need_discharge_from_map

    if need_map_open:
        print(f"\n" + "="*60)
        print(f"PROCESSING FOLDER: {folder}")
        print(f"Stitching {len(all_run_paths)} parts.")
        if need_profiles_from_map:
            print(f"Computing {len(missing_x_coords)} cross-sections...")

        loaded_datasets = []
        loaded_trees = []
        
        try:
            variables_to_open = ['mesh2d_mor_bl', 'mesh2d_face_x', 'mesh2d_face_y']
            if need_discharge_from_map:
                variables_to_open += ['mesh2d_q1', 'mesh2d_edge_x', 'mesh2d_edge_y']
            variables_to_open = list(dict.fromkeys(variables_to_open))

            for p_path in all_run_paths:
                print(f"   -> Opening Map: {p_path.name}")
                ds = dataset_cache.get_partitioned(
                    str(p_path / 'output' / '*_map.nc'),
                    variables=variables_to_open,
                    chunks={'time': 200},
                )
                
                if need_profiles_from_map:
                    face_x = ds['mesh2d_face_x'].values
                    face_y = ds['mesh2d_face_y'].values
                    tree = cKDTree(np.vstack([face_x, face_y]).T)
                    loaded_trees.append(tree)

                loaded_datasets.append(ds)

            if need_profiles_from_map:
                # Create y-coordinates for sampling
                y_samples = np.linspace(y_range[0], y_range[1], n_y_samples)
                
                new_results = {}
                for x_coord in missing_x_coords:
                    cs_name = f"km{int(x_coord / 1000)}"
                    print(f"   Analyzing x = {x_coord}m ({cs_name})...")
                    
                    cs_x = np.full(n_y_samples, x_coord)
                    cs_y = y_samples
                    dist = y_samples - y_samples[0]
                    
                    full_bi_series = []
                    full_times = []
                    all_profiles_raw = []

                    for ds_map, tree in zip(loaded_datasets, loaded_trees):
                        nearest_indices = get_nearest_face_indices(tree, cs_x, cs_y)

                        # Read only sampled faces for all timesteps (time, points)
                        xr_ds = getattr(ds_map, 'obj', ds_map)
                        bl = xr_ds['mesh2d_mor_bl'].isel(mesh2d_nFaces=nearest_indices).values
                        time_vals = pd.to_datetime(xr_ds.time.values)

                        for t in tqdm(range(bl.shape[0]), desc=f"      Timesteps", leave=False):
                            plot_profile = bl[t, :].copy()
                            plot_profile[plot_profile > 8.0] = np.nan
                            bi = compute_braiding_index_with_threshold(plot_profile, safety_buffer=safety_buffer)

                            full_bi_series.append(bi)
                            full_times.append(time_vals[t])
                            all_profiles_raw.append(plot_profile)

                    new_results[cs_name] = {
                        'bi_series': full_bi_series,
                        'times': full_times,  # standardized key name
                        'time_series': full_times,  # backward compatibility
                        'profiles': all_profiles_raw,
                        'dist': dist,
                    }

                # Add new results to folder_results
                folder_results.update(new_results)
                
                # Save to unified cache (merges with existing)
                save_profile_cache(cache_path, new_results, cache_settings)
                print(f"   Saved to unified cache: {cache_path}")

            if need_discharge_from_map:
                print("Extracting discharge from map files (fallback)...")
                for ds in loaded_datasets:
                    q_times, q_values = extract_discharge_at_x(ds, DISCHARGE_X_COORD, y_range)
                    if q_times is not None:
                        all_discharge_times.extend(q_times)
                        all_discharge_values.extend(q_values)

                if all_discharge_times:
                    discharge_source = "map files"
                    if folder_results:
                        first_cs = list(folder_results.keys())[0]
                        discharge_data = {
                            discharge_cache_key: {
                                'times': all_discharge_times,
                                'values': all_discharge_values,
                            }
                        }
                        save_profile_cache(cache_path, {first_cs: discharge_data}, cache_settings)
                        print(f"  Saved discharge to cache: {discharge_cache_key}")

        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue
        finally:
            for ds in loaded_datasets:
                ds.close()

    # --- 3. PLOTTING (from cache or freshly computed results) ---
    try:
        if discharge_source == "BC files":
            discharge_label = "River Discharge Input [m³/s]"
        else:
            discharge_label = f"Discharge at km{int(DISCHARGE_X_COORD/1000)} [m³/s]"

        n_cs = len(selected_x_coords)
        fig, axes = plt.subplots(n_cs, 2, figsize=(18, 5 * n_cs))
        if n_cs == 1: axes = axes.reshape(1, -1)

        # Get morfac: use default if set, otherwise extract from folder name
        morfac = default_morfac if default_morfac else float(get_mf_number(folder))

        for i, x_coord in enumerate(selected_x_coords):
            cs_name = f"km{int(x_coord / 1000)}"
            if cs_name not in folder_results:
                continue
                
            data = folder_results[cs_name]
            ax_spatial, ax_bi = axes[i, 0], axes[i, 1]
            
            df_results = pd.DataFrame({
                'time': data['time_series'], 
                'bi': data['bi_series'], 
                'p_idx': range(len(data['profiles']))
            })
            df_results = df_results.drop_duplicates('time').sort_values('time')
            
            # Convert to morphological time in years (starting from 0)
            times = pd.to_datetime(df_results['time'])
            t0 = times.iloc[0]
            hydro_days = (times - t0).dt.total_seconds() / 86400
            morph_years = (hydro_days * morfac) / 365.25  # morphological years
            df_results['morph_years'] = morph_years
            
            slice_indices = np.linspace(0, len(df_results) - 1, n_slices, dtype=int)
            colors = plt.cm.plasma(np.linspace(0, 0.8, n_slices))
            
            dist = data['dist']
            all_profiles_raw = data['profiles']
            
            for c_idx, row_idx in enumerate(slice_indices):
                data_row = df_results.iloc[row_idx]
                prof = all_profiles_raw[int(data_row['p_idx'])]
                lbl = f"t={data_row['morph_years']:.1f}yr"
                
                ax_spatial.plot(dist, prof, color=colors[c_idx], label=lbl, linewidth=1.2)
                ax_spatial.axhline(np.nanmean(prof) - safety_buffer, color=colors[c_idx], 
                                   linestyle='--', alpha=0.3)

            ax_spatial.set_title(f"Profile: {cs_name} (Mean - {int(safety_buffer*100)}cm)")
            ax_spatial.set_xlabel("Width [m]")
            ax_spatial.set_ylabel("Bed Level [m]")
            ax_spatial.grid(True, alpha=0.2)
            ax_spatial.legend(loc='best', fontsize='x-small')

            ax_bi.plot(df_results['morph_years'], df_results['bi'], color='black', alpha=0.7)
            ax_bi.set_title(f"Braiding Index: {cs_name}")
            ax_bi.set_xlabel("Morphological time [years]")
            ax_bi.set_ylabel("No. of Channels")
            ax_bi.set_ylim(0, 8)
            ax_bi.grid(True, alpha=0.2)

            if all_discharge_times:
                discharge_times = pd.to_datetime(all_discharge_times)
                discharge_morph_years = (discharge_times - t0).total_seconds() / 86400
                discharge_morph_years = (discharge_morph_years * morfac) / 365.25

                bi_min = df_results['morph_years'].min()
                bi_max = df_results['morph_years'].max()
                mask = (discharge_morph_years >= bi_min) & (discharge_morph_years <= bi_max)
                discharge_morph_years = discharge_morph_years[mask]
                discharge_values = np.asarray(all_discharge_values)[mask]

                ax_q = ax_bi.twinx()
                ax_q.plot(
                    discharge_morph_years,
                    discharge_values,
                    color='lightgrey',
                    alpha=0.9,
                    linewidth=1.5,
                    label='Discharge'
                )
                ax_q.set_ylabel(discharge_label, color='dimgrey')
                ax_q.tick_params(axis='y', labelcolor='dimgrey')

        plt.tight_layout()
        save_name = f"braiding_index_fulltime_{folder}.png"
        plt.savefig(base_directory / config / save_name, dpi=300)
        plt.show()
        print(f"Finished {folder}. Figure saved.")

    except Exception as e:
        print(f"Error plotting {folder}: {e}")
    finally:
        plt.close('all')

dataset_cache.close_all()

print("\n" + "="*60)
print("ALL FOLDERS COMPLETED.")