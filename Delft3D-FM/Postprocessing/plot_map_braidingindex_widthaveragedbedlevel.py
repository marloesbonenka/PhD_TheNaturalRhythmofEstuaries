""""Post-process multiple """

#%% IMPORTS
import os
import numpy as np
import matplotlib.pyplot as plt
import dfm_tools as dfmt
import xarray as xr
import sys

#%%
# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\Delft3D-FM\Postprocessing")

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *

#%% --- CONFIGURATION ---
base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
config = 'Test_MORFAC/Tmorph_400years'
target_year = 400 

start_date = np.datetime64('2025-01-01') 
x_targets = np.arange(20000, 44001, 1000)
y_range = (5000, 10000)

tau_threshold = 0.05
var_tau = 'mesh2d_tausmax'
depth_threshold = 0.2 # A depth_threshold of 0.2 means a channel must be 20% deeper than the average depth of that specific cross-section to be counted (ignores thin water over bars).
var_depth = 'mesh2d_waterdepth'

bed_threshold = 6

#%% --- PLOT SETTINGS ---
check_variables = True

compare_braiding_index = True
plot_braiding_index_individual = False

compare_width_averaged_bedlevel = True
plot_width_averaged_bedlevel_individual = True

#%% 
# --- SEARCH & SORT FOLDERS ---
model_folders = [f for f in os.listdir(os.path.join(base_directory, config)) if f.startswith('MF')]
model_folders.sort(key=get_mf_number)

# --- STORE FINAL YEAR RESULTS ---
comparison_results = {}

# --- COMPUTE MAP RESULTS FOR EACH RUN ---
for i, folder in enumerate(model_folders):
    model_location = os.path.join(base_directory, config, folder)
    file_pattern = os.path.join(model_location, 'output', '*_map.nc')

    save_dir = os.path.join(model_location, 'output_plots')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nProcessing: {folder}")
    
    # 1. LOAD FM DATA
    ds = dfmt.open_partitioned_dataset(file_pattern)

    # 2. MORPHOLOGICAL TIME LOGIC (Robust to restarts)
    mf_val = get_mf_number(folder)
    delta_time = ds.time.values - start_date
    hydro_years = delta_time / np.timedelta64(365, 'D')
    morph_years = hydro_years * mf_val

    # Find the index closest to Year 50
    ts_final = np.argmin(np.abs(morph_years - target_year))
    
    print(f"Folder: {folder:10} | Found Year: {morph_years[ts_final]:.2f} at Index: {ts_final}")
    
    # Initialize dictionary for this MF
    comparison_results[mf_val] = {}

    # --- CHECK VARIABLES ---
    if check_variables and i == 0:
        check_available_variables_xarray(ds)
        break

    # 3. BRAIDING ANALYSIS
    if compare_braiding_index:
        # 3.1 shear stress method (fixed threshold over entire estuary)
        if var_tau in ds:
            print(f"Computing BI for {folder}...")
            bi_tau, _ = compute_BI_FM(ds, var_tau, x_targets, y_range, threshold=tau_threshold, time_idx=ts_final)
            
            # Store Year 50 specifically
            comparison_results[mf_val]['BI_tau'] = bi_tau[0, :]
    
            if plot_braiding_index_individual: 
                plt.figure(figsize=(10, 6))
                plt.plot(x_targets/1000, bi_tau[ts_final, :], 'o-', label=f'Morph Year {morph_years[ts_final]:.1f}')
                plt.xlabel('Distance [km]')
                plt.ylabel('Braiding Index')
                plt.title(f'BI: {folder}')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(save_dir, f'braiding_index_{folder}.png'))
                plt.close()

        # 2. NEW METHOD: Water Depth (Relative Threshold)
        if var_depth in ds:
            bi_depth, _ = compute_BI_FM(ds, var_depth, x_targets, y_range, 
                                        threshold=depth_threshold, time_idx=ts_final, method='relative')
            comparison_results[mf_val]['BI_depth'] = bi_depth[0, :]

    # 4. WIDTH-AVERAGED BED LEVEL
    if compare_width_averaged_bedlevel:
        print(f"Computing Bed Level for {folder}...")
        var_name = "mesh2d_mor_bl"
        dx = 1000 
        x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        
        face_x = ds['mesh2d_face_x'].values
        face_y = ds['mesh2d_face_y'].values
        width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])

        bedlev_data = ds[var_name].isel(time=ts_final).values
        valid_mask = (width_mask) & (bedlev_data < bed_threshold)
        
        temp_means = []
        for k in range(len(x_bins)-1):
            bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k+1])
            temp_means.append(np.mean(bedlev_data[bin_mask]) if np.any(bin_mask) else np.nan)
        
        comparison_results[mf_val]['BL'] = np.array(temp_means)
        comparison_results[mf_val]['x_centers'] = x_centers

    ds.close()

# %% --- 5. FINAL COMPARISON PLOT ---
print("\nGenerating Comparison Plot...")
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True)

sorted_mfs = sorted(comparison_results.keys())
colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_mfs)))

for idx, mf in enumerate(sorted_mfs):
    data = comparison_results[mf]
    
    # Plot Shear Stress BI
    if 'BI_tau' in data:
        ax1.plot(x_targets/1000, data['BI_tau'], label=f'MF {mf}', color=colors[idx], marker='o', ms=4)
    
    # Plot Water Depth BI (Normalized for bars)
    if 'BI_depth' in data:
        ax2.plot(x_targets/1000, data['BI_depth'], label=f'MF {mf}', color=colors[idx], marker='s', ms=4, linestyle='--')
    
    # Plot Bed Level
    if 'BL' in data:
        ax3.plot(data['x_centers']/1000, data['BL'], color=colors[idx], linewidth=2)

ax1.set_title(f'BI ({var_tau}), fixed threshold: tau > {tau_threshold} N/m2')
ax1.set_ylabel('braiding index')
ax2.set_title(f'BI ({var_depth}), relative threshold: {int(depth_threshold*100)}% above mean water depth')
ax2.set_ylabel('braiding index')
ax3.set_title('Width-averaged bed level')
ax3.set_xlabel('x-coordinate along estuary [km]')
ax3.set_ylabel('width-averaged bed level [m]')
ax1.legend(loc='best')
for ax in [ax1, ax2, ax3]: ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(base_directory, config, f'comparison_Tmorph_50years.png'))
plt.show()

print(f'Saved comparison plot at {os.path.join(base_directory, config)}')
print("\nAll FM processing complete.")

#%%