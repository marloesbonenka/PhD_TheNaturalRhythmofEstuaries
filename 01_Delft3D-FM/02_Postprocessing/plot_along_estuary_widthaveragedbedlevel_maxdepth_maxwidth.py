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
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *
from FUNCTIONS.F_channelwidth import *
from FUNCTIONS.F_cache import DatasetCache

#%% --- CONFIGURATION ---
# Model output
base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
target_year = 400 
config = f'Test_MORFAC/03_flashy/Tmorph_{target_year}years'

# Braiding index
tau_threshold = 0.05
depth_threshold = 0.2 # A depth_threshold of 0.2 means a channel must be 20% deeper than the average depth of that specific cross-section to be counted (ignores thin water over bars).

# Land threshold
bed_threshold = 6

# Channel depth + width analysis
depth_percentile = 95  # For maximum depth analysis (95th percentile)
safety_buffer = 0.20  # For channel width analysis (20 cm below mean)

#%% -- special configuration | do not change ---
special_base = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
special_config = 'Test_MORFAC/03_flashy/Tmorph_50years'
use_mf50_reference = (base_directory == special_base) and (config == special_config)

var_tau = 'mesh2d_tausmax'
var_depth = 'mesh2d_waterdepth'

start_date = np.datetime64('2025-01-01') 
x_targets = np.arange(20000, 44001, 1000)
y_range = (5000, 10000)

#%% --- SETTINGS ---
apply_detrending = False  # Subtract initial bed level to see changes
reference_time_idx = 0   # Time index to use as reference (0 = first timestep)
use_absolute_depth = True  # Use absolute depth values (positive = deep)

check_variables = False

compare_braiding_index = False
plot_braiding_index_individual = False

compare_width_averaged_bedlevel = True
plot_width_averaged_bedlevel_individual = False

compare_max_depth = True  
plot_max_depth_individual = False

compare_channel_width = True 
plot_channel_width_individual = False

#%% --- SEARCH & SORT FOLDERS ---
model_folders = [f for f in os.listdir(os.path.join(base_directory, config)) if f.startswith('MF')]
model_folders.sort(key=get_mf_number)

dataset_cache = DatasetCache()
# --- OPTIONAL: GLOBAL REFERENCE FROM MF50 ---
reference_bed_MF50 = None
if apply_detrending and use_mf50_reference:
    mf50_folder = [f for f in model_folders if get_mf_number(f) == 50]
    if len(mf50_folder) == 1:
        mf50_folder = mf50_folder[0]
        mf50_location = os.path.join(base_directory, config, mf50_folder)
        mf50_pattern = os.path.join(mf50_location, 'output', '*_map.nc')
        ds_mf50 = dataset_cache.get_partitioned(mf50_pattern)
        reference_bed_MF50 = ds_mf50['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()
    else:
        # Fallback: no MF50 found, keep run-specific behavior
        use_mf50_reference = False

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
    ds = dataset_cache.get_partitioned(file_pattern)

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

    # --- DETRENDING: Store reference bed level if needed ---
    if apply_detrending:
        if use_mf50_reference and (reference_bed_MF50 is not None):
            # Use MF50 time index 0 for all runs
            print(f"Using MF50 reference bed (time index {reference_time_idx}) for detrending of {folder}...")
            reference_bed = reference_bed_MF50
        else:
            # Default: per‑run reference at reference_time_idx
            print(f"Storing reference bed level at time index {reference_time_idx} for {folder}...")
            reference_bed = ds['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()
            
        # --- CHECK VARIABLES ---
        if check_variables and i == 0:
            check_available_variables_xarray(ds)
            break

    # Build KDTree for spatial queries (needed for new analyses)
    if compare_max_depth or compare_channel_width:
        face_x = ds['mesh2d_face_x'].values
        face_y = ds['mesh2d_face_y'].values
        from scipy.spatial import cKDTree
        tree = cKDTree(np.vstack([face_x, face_y]).T)

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

        bedlev_data = ds[var_name].isel(time=ts_final).values.copy()

        # Apply detrending if enabled
        if apply_detrending:
            bedlev_data = bedlev_data - reference_bed
            # For detrended data, don't use bed_threshold filter (data is centered around 0)
            # Only filter based on spatial domain
            valid_mask = width_mask
        else:
            # For non-detrended data, use bed_threshold to exclude high land values
            valid_mask = (width_mask) & (bedlev_data < bed_threshold)
        
        temp_means = []
        for k in range(len(x_bins)-1):
            bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k+1])
            temp_means.append(np.mean(bedlev_data[bin_mask]) if np.any(bin_mask) else np.nan)
        
        comparison_results[mf_val]['BL'] = np.array(temp_means)
        comparison_results[mf_val]['x_centers'] = x_centers

    # 5. MAXIMUM DEPTH ANALYSIS (95th percentile)
    if compare_max_depth:
        print(f"Computing Maximum Depth for {folder}...")
        var_name = "mesh2d_mor_bl"
        dx = 1000
        x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        
        face_x = ds['mesh2d_face_x'].values
        face_y = ds['mesh2d_face_y'].values
        width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
        
        bedlev_data = ds[var_name].isel(time=ts_final).values.copy()
        
        # Apply detrending if enabled
        if apply_detrending:
            bedlev_data = bedlev_data - reference_bed
        
        # For depth calculation: convert bed level to depth
        # Depth is positive downward (negative bed level = deep channel)
        if use_absolute_depth:
            # Use absolute value to make all depths positive
            depths_field = np.abs(bedlev_data)
        else:
            # Traditional: depth = -bed_level (negative values become positive)
            depths_field = -bedlev_data
        
        # Apply thresholds
        if apply_detrending:
            valid_mask = width_mask  # No bed_threshold when detrended
        else:
            valid_mask = (width_mask) & (bedlev_data < bed_threshold)
        
        max_depths = []
        for k in range(len(x_bins)-1):
            bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k+1])
            if np.any(bin_mask):
                bin_depths = depths_field[bin_mask]
                valid_depths = bin_depths[~np.isnan(bin_depths)]
                if len(valid_depths) > 0:
                    max_depth = np.percentile(valid_depths, depth_percentile)
                    max_depths.append(max_depth)
                else:
                    max_depths.append(np.nan)
            else:
                max_depths.append(np.nan)
        
        comparison_results[mf_val]['MaxDepth'] = np.array(max_depths)
        
        if plot_max_depth_individual:
            plt.figure(figsize=(10, 6))
            plt.plot(x_centers/1000, max_depths, 'o-', color='steelblue')
            plt.xlabel('Distance [km]')
            depth_label = 'Absolute Depth' if use_absolute_depth else 'Depth'
            detrend_label = ' (Detrended)' if apply_detrending else ''
            plt.ylabel(f'{depth_percentile}th Percentile {depth_label} [m]{detrend_label}')
            plt.title(f'Maximum Channel Depth: {folder}')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'max_depth_{folder}.png'))
            plt.close()

    # 6. CHANNEL WIDTH ANALYSIS
    if compare_channel_width:
        print(f"Computing Channel Widths for {folder}...")
        
        max_widths = []
        for x_coord in x_targets:
            distances, bed_profile = get_bed_profile_at_x(
                ds, tree, x_coord, y_range, ts_final, 
                reference_bed=reference_bed if apply_detrending else None,
                detrend=apply_detrending
            )
            
            # Filter out land values
            if apply_detrending:
                # For detrended data, use different threshold logic
                bed_profile[np.abs(bed_profile) > bed_threshold] = np.nan
            else:
                bed_profile[bed_profile > bed_threshold] = np.nan
            
            max_width = compute_max_channel_width(bed_profile, distances, safety_buffer)
            max_widths.append(max_width)
        
        comparison_results[mf_val]['ChannelWidth'] = np.array(max_widths)
        
        if plot_channel_width_individual:
            plt.figure(figsize=(10, 6))
            plt.plot(x_targets/1000, max_widths, 'o-', color='coral')
            plt.xlabel('Distance [km]')
            plt.ylabel('Max Channel Width [m]')
            detrend_label = ' (Detrended)' if apply_detrending else ''
            plt.title(f'Maximum Channel Width: {folder}{detrend_label}')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, f'channel_width_{folder}.png'))
            plt.close()

    ds.close()

# %% --- 7. FINAL COMPARISON PLOT ---
print("\nGenerating Comparison Plot...")

# Count active plots
n_plots = sum([
    compare_braiding_index and 'BI_tau' in comparison_results[list(comparison_results.keys())[0]],
    compare_braiding_index and 'BI_depth' in comparison_results[list(comparison_results.keys())[0]],
    compare_width_averaged_bedlevel,
    compare_max_depth,
    compare_channel_width
])

if n_plots == 0:
    print("No plots to generate!")
else:
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]
    
    sorted_mfs = sorted(comparison_results.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_mfs)))
    
    plot_idx = 0
    
    # Plot 1: Shear Stress BI
    if compare_braiding_index and 'BI_tau' in comparison_results[sorted_mfs[0]]:
        for idx, mf in enumerate(sorted_mfs):
            data = comparison_results[mf]
            if 'BI_tau' in data:
                axes[plot_idx].plot(x_targets/1000, data['BI_tau'], 
                                   label=f'MF {mf}', color=colors[idx], marker='o', ms=4)
        axes[plot_idx].set_title(f'BI ({var_tau}), fixed threshold: tau > {tau_threshold} N/m²')
        axes[plot_idx].set_ylabel('braiding index')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1
    
    # Plot 2: Water Depth BI
    if compare_braiding_index and 'BI_depth' in comparison_results[sorted_mfs[0]]:
        for idx, mf in enumerate(sorted_mfs):
            data = comparison_results[mf]
            if 'BI_depth' in data:
                axes[plot_idx].plot(x_targets/1000, data['BI_depth'], 
                                   label=f'MF {mf}', color=colors[idx], marker='s', ms=4, linestyle='--')
        axes[plot_idx].set_title(f'BI ({var_depth}), relative threshold: {int(depth_threshold*100)}% above mean water depth')
        axes[plot_idx].set_ylabel('braiding index')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1
    
    # Plot 3: Bed Level
    if compare_width_averaged_bedlevel:
        for idx, mf in enumerate(sorted_mfs):
            data = comparison_results[mf]
            if 'BL' in data:
                axes[plot_idx].plot(data['x_centers']/1000, data['BL'], 
                                   color=colors[idx], linewidth=2, label=f'MF {mf}')
        axes[plot_idx].set_title('width-averaged bed level')
        axes[plot_idx].set_ylabel('bed level [m]')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1
    
    # Plot 4: Maximum Depth
    if compare_max_depth:
        for idx, mf in enumerate(sorted_mfs):
            data = comparison_results[mf]
            if 'MaxDepth' in data:
                axes[plot_idx].plot(data['x_centers']/1000, data['MaxDepth'], 
                                   color=colors[idx], linewidth=2, label=f'MF {mf}', marker='o', ms=3)
        axes[plot_idx].set_title(f'p{depth_percentile} channel depth')
        axes[plot_idx].set_ylabel('depth [m]')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1
    
    # Plot 5: Channel Width
    if compare_channel_width:
        for idx, mf in enumerate(sorted_mfs):
            data = comparison_results[mf]
            if 'ChannelWidth' in data:
                axes[plot_idx].plot(x_targets/1000, data['ChannelWidth'], 
                                   color=colors[idx], linewidth=2, label=f'MF {mf}', marker='s', ms=3)
        axes[plot_idx].set_title(f'maximum channel width (threshold: mean depth - {int(safety_buffer*100)} cm)')
        axes[plot_idx].set_ylabel('width [m]')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
    
    axes[-1].set_xlabel('x-coordinate along estuary [km]')
    
    plt.tight_layout()

    if apply_detrending:
        plt.savefig(os.path.join(base_directory, config, f'sensitivity_MF_detrended_along_estuary_Tmorph_{target_year}years.png'), dpi=300)
    else:
        plt.savefig(os.path.join(base_directory, config, f'sensitivity_MF_along_estuary_Tmorph_{target_year}years.png'), dpi=300)
    plt.show()
    
    print(f'Saved comparison plot at {os.path.join(base_directory, config)}')

dataset_cache.close_all()

print("\nAll FM processing complete.")