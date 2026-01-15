"""
Analyze bed level and braiding index at cross-sections from Delft3D-FM .his/map output
Robust version: Handles restarts with mesh changes and optimizes U-drive access.
"""

#%% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import dfm_tools as dfmt
import time
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm
import sys

#%% Add path for functions
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *

#%% --- CONFIGURATION ---
base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
config = 'Test_OneRiverBoundary'
#'Test_MORFAC/Tmorph_400years'
timed_out_dir = os.path.join(base_directory, config, "timed-out")

# --- SETTINGS ---
n_slices = 5  
safety_buffer = 0.20 
selected_cross_sections = ["ObservationCrossSection_Estuary_km20",
                           "ObservationCrossSection_Estuary_km26",
                           "ObservationCrossSection_Estuary_km32",
                           "ObservationCrossSection_Estuary_km38",
                           "ObservationCrossSection_Estuary_km42",
                           "ObservationCrossSection_Estuary_km43",
                           "ObservationCrossSection_Estuary_km44"]

#%% --- SEARCH & SORT FOLDERS ---
model_folders = [f for f in os.listdir(os.path.join(base_directory, config)) if f.startswith('0')]
model_folders.sort(key=get_mf_number)

#%% --- MAIN PROCESSING LOOP ---

for folder in model_folders:
    model_location = os.path.join(base_directory, config, folder)
    
    # --- 1. RESTART LOGIC (Find all parts) ---
    all_run_paths = []
    if 'restart' in folder:
        mf_prefix = folder.split('_')[0] # e.g., "MF1"
        if os.path.exists(timed_out_dir):
            match = [f for f in os.listdir(timed_out_dir) if f.startswith(mf_prefix)]
            if match:
                all_run_paths.append(os.path.join(timed_out_dir, match[0]))
    all_run_paths.append(model_location)

    print(f"\n" + "="*60)
    print(f"PROCESSING FOLDER: {folder}")
    print(f"Stitching {len(all_run_paths)} parts.")

    # --- 2. LOAD DATASETS ONCE PER FOLDER ---
    loaded_datasets = []
    loaded_trees = []
    
    try:
        for p_path in all_run_paths:
            print(f"   -> Opening Map: {os.path.basename(p_path)}")
            # Chunks=1 keeps metadata loading fast and memory low
            ds = dfmt.open_partitioned_dataset(os.path.join(p_path, 'output', '*_map.nc'), chunks={'time': 1})
            
            # Build Tree for this part's specific mesh (handles mesh size changes)
            face_x = ds['mesh2d_face_x'].values
            face_y = ds['mesh2d_face_y'].values
            tree = cKDTree(np.vstack([face_x, face_y]).T)
            
            loaded_datasets.append(ds)
            loaded_trees.append(tree)

        # Load HIS once (from the last part) for cross-section metadata
        his_file = os.path.join(all_run_paths[-1], 'output', 'FlowFM_0000_his.nc')
        ds_his = xr.open_dataset(his_file)

        # --- 3. ANALYZE CROSS SECTIONS ---
        n_cs = len(selected_cross_sections)
        fig, axes = plt.subplots(n_cs, 2, figsize=(18, 5 * n_cs))
        if n_cs == 1: axes = axes.reshape(1, -1)

        for i, cs_name in enumerate(selected_cross_sections):
            print(f"   Analyzing {cs_name}...")
            ax_spatial, ax_bi = axes[i, 0], axes[i, 1]
            
            # Get Geometry from HIS
            cs_names = [n.decode() if isinstance(n, bytes) else n for n in ds_his.cross_section_name.values]
            if cs_name not in cs_names: continue
            
            idx = cs_names.index(cs_name)
            start = int(ds_his['cross_section_geom_node_count'].values[:idx].sum())
            end = start + int(ds_his['cross_section_geom_node_count'].values[idx])
            cs_x = ds_his['cross_section_geom_node_coordx'].values[start:end]
            cs_y = ds_his['cross_section_geom_node_coordy'].values[start:end]
            dist = np.sqrt((cs_x - cs_x[0])**2 + (cs_y - cs_y[0])**2)
            
            full_bi_series = []
            full_time_series = []
            all_profiles_raw = []

            # Extract data sequentially from the loaded parts
            for ds_map, tree in zip(loaded_datasets, loaded_trees):
                for t in tqdm(range(len(ds_map.time)), desc=f"      Timesteps", leave=False):
                    profile = get_bed_profile(ds_map, tree, cs_x, cs_y, t)
                    
                    # BI Calculation
                    plot_profile = profile.copy()
                    plot_profile[plot_profile > 8.0] = np.nan
                    bi = compute_braiding_index_with_threshold(plot_profile, safety_buffer=safety_buffer)
                    
                    full_bi_series.append(bi)
                    full_time_series.append(pd.to_datetime(ds_map.time.values[t]))
                    all_profiles_raw.append(plot_profile)

            # --- Handle Time-Stitching & Plotting ---
            df_results = pd.DataFrame({
                'time': full_time_series, 
                'bi': full_bi_series, 
                'p_idx': range(len(all_profiles_raw))
            })
            # Clean up restart overlaps
            df_results = df_results.drop_duplicates('time').sort_values('time')
            
            slice_indices = np.linspace(0, len(df_results) - 1, n_slices, dtype=int)
            colors = plt.cm.plasma(np.linspace(0, 0.8, n_slices))
            
            # Profile Plotting
            for c_idx, row_idx in enumerate(slice_indices):
                data_row = df_results.iloc[row_idx]
                prof = all_profiles_raw[int(data_row['p_idx'])]
                lbl = data_row['time'].strftime('%Y-%m-%d')
                
                ax_spatial.plot(dist, prof, color=colors[c_idx], label=lbl, linewidth=1.2)
                ax_spatial.axhline(np.nanmean(prof) - safety_buffer, color=colors[c_idx], 
                                   linestyle='--', alpha=0.3)

            ax_spatial.set_title(f"Profile: {cs_name} (Mean - {int(safety_buffer*100)}cm)")
            ax_spatial.set_xlabel("Width [m]")
            ax_spatial.set_ylabel("Bed Level [m]")
            ax_spatial.grid(True, alpha=0.2)
            ax_spatial.legend(loc='best', fontsize='x-small')

            # Braiding Index Plotting
            ax_bi.plot(df_results['time'], df_results['bi'], color='black', alpha=0.7)
            ax_bi.set_title(f"Braiding Index: {cs_name}")
            ax_bi.set_xlabel("Date")
            ax_bi.set_ylabel("No. of Channels")
            ax_bi.set_ylim(0, 8)
            ax_bi.grid(True, alpha=0.2)
            plt.setp(ax_bi.get_xticklabels(), rotation=30)

        # --- SAVE FIGURE ---
        plt.tight_layout()
        save_name = f"cross_section_analysis_{folder}.png"
        plt.savefig(os.path.join(base_directory, config, save_name), dpi=300)
        plt.close(fig) 
        print(f"Finished {folder}. Figure saved.")

    except Exception as e:
        print(f"Error processing {folder}: {e}")

    finally:
        # --- 4. CLEAN UP FOR NEXT FOLDER ---
        for ds in loaded_datasets:
            ds.close()
        if 'ds_his' in locals():
            ds_his.close()
        plt.close('all')

print("\n" + "="*60)
print("ALL FOLDERS COMPLETED.")