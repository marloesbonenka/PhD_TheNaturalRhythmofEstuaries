"""Analyze bed level and braiding index at cross-sections from Delft3D-FM .his output"""

#%% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
import dfm_tools as dfmt
import time

#%% Add path for functions
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\Delft3D-FM\Postprocessing")

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *

#%% --- CONFIGURATION ---
base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
config = 'Test_MORFAC/Tmorph_50years'

start_date = np.datetime64('2025-01-01')

# --- SETTINGS ---
n_selected_times = 5  # Number of slices to show
selected_cross_sections = ["ObservationCrossSection_Estuary_km20",
                           "ObservationCrossSection_Estuary_km26",
                           "ObservationCrossSection_Estuary_km32",
                           "ObservationCrossSection_Estuary_km38",
                           "ObservationCrossSection_Estuary_km44"]

#%% --- SEARCH & SORT FOLDERS ---
model_folders = [f for f in os.listdir(os.path.join(base_directory, config)) if f.startswith('MF')]
model_folders.sort(key=get_mf_number)

#%% --- MAIN PROCESSING LOOP ---

# Configuration 
n_slices = 5
safety_buffer = 0.20 # A channel must be 20 cm deeper than mean bed level in that cross-section

folder = model_folders[1]

#If you want to plot every output, uncomment the below line and tab all below that
#for folder in model_folders:
#Don't forget to tab here if you uncomment the line above, and uncomment 'continue'
model_location = os.path.join(base_directory, config, folder)

his_file = os.path.join(model_location, 'output', 'FlowFM_0000_his.nc')
map_file_pattern = os.path.join(model_location, 'output', '*_map.nc')

if not os.path.exists(his_file):
    print(f"Skipping {folder}, his file not found.")
    #continue

print(f"\nProcessing: {folder}")

# Load datasets
print("Loading his file...")
start_time = time.time()
ds_his = xr.open_dataset(his_file)
print(f"His file loaded in {time.time() - start_time:.2f} seconds")

print("Loading map file...")
start_time = time.time()
ds_map = dfmt.open_partitioned_dataset(map_file_pattern)
print(f"Map file loaded in {time.time() - start_time:.2f} seconds")

map_datetimes = pd.to_datetime(ds_map['time'].values)

# Build KDTree for spatial lookups on map grid
face_x = ds_map['mesh2d_face_x'].values
face_y = ds_map['mesh2d_face_y'].values
tree = cKDTree(np.vstack([face_x, face_y]).T)

n_times_map = len(ds_map.time)
slice_indices = np.linspace(0, n_times_map - 1, n_slices, dtype=int)

#%%
# Setup Figure
n_cs = len(selected_cross_sections)
fig, axes = plt.subplots(n_cs, 2, figsize=(18, 5 * n_cs))
if n_cs == 1: axes = axes.reshape(1, -1)

for i, cs_name in enumerate(selected_cross_sections):
    print(f"  Analyzing {cs_name}...")
    ax_spatial = axes[i, 0]
    ax_bi = axes[i, 1]
    
    # --- Get Geometry from HIS file ---
    cs_names = [n.decode() if isinstance(n, bytes) else n for n in ds_his.cross_section_name.values]
    if cs_name not in cs_names: continue
    
    idx = cs_names.index(cs_name)
    start = int(ds_his['cross_section_geom_node_count'].values[:idx].sum())
    end = start + int(ds_his['cross_section_geom_node_count'].values[idx])
    
    cs_x = ds_his['cross_section_geom_node_coordx'].values[start:end]
    cs_y = ds_his['cross_section_geom_node_coordy'].values[start:end]
    dist = np.sqrt((cs_x - cs_x[0])**2 + (cs_y - cs_y[0])**2)
    
    # --- Loop through time to compute BI and slices ---
    bi_over_time = []
    colors = plt.cm.hsv(np.linspace(0, 0.8, n_slices))
    color_idx = 0
    
    for t in range(n_times_map):
        profile = get_bed_profile(ds_map, tree, cs_x, cs_y, t)
        
        #Ignore land values, i.e. bed level > x m
        plot_profile = profile.copy()
        plot_profile[plot_profile > 8.0] = np.nan

        # BI Calculation
        bi = compute_braiding_index_with_threshold(plot_profile, safety_buffer=safety_buffer)
        bi_over_time.append(bi)
        
        # Spatial Plotting (only for selected slices)
        if t in slice_indices:
            mbl = np.nanmean(plot_profile)

            # Format the datetime for the legend (e.g., '2025-06-15')
            lbl = map_datetimes[t].strftime('%Y-%m-%d')
            
            # Plot the profile
            ax_spatial.plot(dist, plot_profile, color=colors[color_idx], label=lbl, linewidth=1.2)
            # Plot the Mean Bed Level - 20cm (The threshold line)
            ax_spatial.axhline(mbl - safety_buffer, color=colors[color_idx], 
                                linestyle='--', alpha=0.5, linewidth=1)
            color_idx += 1

    # --- Formatting Left Plot ---
    ax_spatial.set_title(f"Profile: {cs_name} (Dashed = Mean - {int(safety_buffer*100)}cm)")
    ax_spatial.set_xlabel("Width [m]")
    ax_spatial.set_ylabel("Bed Level [m]")
    ax_spatial.grid(True, alpha=0.2)
    ax_spatial.legend(loc='best')
    # ax_spatial.invert_yaxis() # Uncomment if you want depths increasing downwards

    # --- Formatting Right Plot ---
    ax_bi.plot(range(n_times_map), bi_over_time, color='black', alpha=0.7)
    ax_bi.set_title(f"Braiding Index (Count of Channels > {int(safety_buffer*100)}cm below mean)")
    ax_bi.set_xlabel("Date")
    ax_bi.set_ylabel("No. of Channels")
    ax_bi.set_ylim(0, max(bi_over_time) + 2 if bi_over_time else 5)
    ax_bi.grid(True, alpha=0.2)

plt.tight_layout()
save_name = f"Estuary_Analysis_{folder}.png"
plt.savefig(os.path.join(base_directory, config, save_name), dpi=300)
plt.show()

ds_his.close()
ds_map.close()
# %%
