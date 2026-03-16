""""Post-process multiple scenarios map output, 
plot overall morphological characteristics along the estuary (bed level, max depth, channel width)"""

#%% IMPORTS
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

#%%
# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import (
    _date_to_filename_tag,
    _date_to_label,
    _scenario_label,
    _scenario_color,
    get_variability_map,
    find_variability_model_folders,
    get_target_snapshot_dates,
    get_snapshot_matches_by_target_dates,
    get_mf_number,
    check_available_variables_xarray,
    sort_scenario_keys,
    group_snapshot_by_scenario,
    stack_metric_arrays,
    draw_metric_with_optional_envelope,
)
from FUNCTIONS.F_braiding_index import *
from FUNCTIONS.F_channelwidth import *
from FUNCTIONS.F_hypsometry import compute_hypsometric_curve
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths


#%% --- CONFIGURATION ---
# Model output
DISCHARGE = 1000  # or 1000, etc.
NOISY = False
ADD_NON_NOISY_BASELINE_Q500 = False
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f'Model_Output/Q{DISCHARGE}'

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

#%% --- SETTINGS ---
var_tau = 'mesh2d_tausmax'
var_depth = 'mesh2d_waterdepth'

start_date = np.datetime64('2025-01-01') 
x_targets = np.arange(20000, 44001, 1000)
y_range = (5000, 10000)

CACHE_BBOX = [1, 1, 45000, 15000]  # must match the bbox used in plot_map_at_timestep.py
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

apply_detrending = False  # Subtract initial bed level to see changes
reference_time_idx = 0   # Time index to use as reference (0 = first timestep)
use_absolute_depth = True  # Use absolute depth values (positive = deep)

# Snapshot settings (hydrodynamic dates)
SNAPSHOT_TARGET_DATES = None  # e.g. ['2027-01-01', '2035-01-01']; None -> equally spaced in SNAPSHOT_DATE_RANGE
SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2055-12-31'))
SNAPSHOT_COUNT = 6

check_variables = False

compare_braiding_index = False
plot_braiding_index_individual = False

compare_width_averaged_bedlevel = True
plot_width_averaged_bedlevel_individual = True

compare_max_depth = True  
plot_max_depth_individual = True

compare_channel_width = True 
plot_channel_width_individual = True

compare_hypsometric = True
plot_hypsometric_individual = True

# Human-readable labels per scenario number (used in combined plots)
SCENARIO_LABELS = {
    '1': 'Constant',
    '2': 'Seasonal',
    '3': 'Flashy',
    '4': 'Single peak',
}

# Colours: one per scenario (used in combined plots)
SCENARIO_COLORS = {
    '1': '#1f77b4',   # blue   - Constant
    '2': '#ff7f0e',   # orange - Seasonal
    '3': '#2ca02c',   # green  - Flashy
    '4': '#d62728',   # red    - Single peak
}
BASELINE_COLOR = next(iter(SCENARIO_COLORS.values()))


#%% --- SEARCH & SORT FOLDERS ---
base_path = base_directory / config

VARIABILITY_MAP = get_variability_map(DISCHARGE)
if DISCHARGE == 500 and NOISY:
    noisy_base_path = base_path / f'0_Noise_Q{DISCHARGE}'
    if noisy_base_path.exists() and noisy_base_path.is_dir():
        base_path = noisy_base_path

model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=None,
    analyze_noisy=NOISY,
)


# Directories
assessment_dir = base_path / 'cached_data'
assessment_dir.mkdir(parents=True, exist_ok=True)
timed_out_dir = base_path / 'timed-out'
summary_output_dir = base_path / 'output_plots'
summary_output_dir.mkdir(parents=True, exist_ok=True)

# --- OPTIONAL: GLOBAL REFERENCE FROM MF50 ---
reference_bed_MF50 = None
if apply_detrending and use_mf50_reference:
    mf50_folder = [f for f in model_folders if get_mf_number(f) == 50]
    if len(mf50_folder) == 1:
        mf50_folder = mf50_folder[0]
        mf50_run_paths = get_stitched_map_run_paths(
            base_path=base_path,
            folder_name=mf50_folder.name,
            timed_out_dir=timed_out_dir,
            variability_map=VARIABILITY_MAP,
            analyze_noisy=False,
        )
        if not mf50_run_paths:
            mf50_run_paths = [base_path / mf50_folder]
        cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
        ds_mf50 = load_or_update_map_cache_multi(
            cache_dir=assessment_dir,
            folder_name=mf50_folder.name,
            run_paths=mf50_run_paths,
            var_names=['mesh2d_mor_bl'],
            bbox=CACHE_BBOX,
            append_time=APPEND_TIMESTEPS,
            append_vars=APPEND_VARIABLES,
            cache_tag=cache_tag,
        )
        if ds_mf50 is not None:
            reference_bed_MF50 = ds_mf50['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()
            ds_mf50.close()
    else:
        # Fallback: no MF50 found, keep run-specific behavior
        use_mf50_reference = False

# --- STORE SNAPSHOT RESULTS ---
comparison_results = {}
baseline_comparison_results = {}
comparison_labels = {}

target_snapshot_dates = get_target_snapshot_dates(
    count=SNAPSHOT_COUNT,
    explicit_dates=SNAPSHOT_TARGET_DATES,
    date_range=SNAPSHOT_DATE_RANGE,
)

print("\nTarget hydrodynamic snapshot dates:")
for dt in target_snapshot_dates:
    print(f"  - {_date_to_label(dt)}")

# --- COMPUTE MAP RESULTS FOR EACH RUN ---
for i, folder in enumerate(model_folders):
    model_location = base_path / folder
    folder_str = folder.name
    save_dir = summary_output_dir / f'mapplots_{folder_str}'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {folder_str}")
    scenario_color = _scenario_color(folder_str, SCENARIO_COLORS)
    scenario_label = _scenario_label(folder_str, SCENARIO_LABELS)
    
    # 1. LOAD FM DATA (cached)
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder.name,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=NOISY,
    )
    if not run_paths:
        run_paths = [model_location]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=['mesh2d_mor_bl'],
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )
    if ds is None:
        print(f"  No cached data for {folder}, skipping.")
        continue

    # 2. Time logic per scenario (no MORFAC sensitivity here)
    delta_time = ds.time.values - start_date
    hydro_years = delta_time / np.timedelta64(365, 'D')

    snapshot_matches = get_snapshot_matches_by_target_dates(ds.time.values, target_snapshot_dates)
    if not snapshot_matches:
        print(f"  No timesteps found for {folder}, skipping.")
        ds.close()
        continue
    last_snapshot_idx = snapshot_matches[-1][1]

    # --- DETRENDING: Store reference bed level if needed ---
    if apply_detrending:
        if use_mf50_reference and (reference_bed_MF50 is not None):
            # Use MF50 time index 0 for all runs
            print(f"Using MF50 reference bed (time index {reference_time_idx}) for detrending of {folder}...")
            reference_bed = reference_bed_MF50
        else:
            # Default: per-run reference at reference_time_idx
            print(f"Storing reference bed level at time index {reference_time_idx} for {folder}...")
            reference_bed = ds['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()

    # --- CHECK VARIABLES (moved outside apply_detrending block) ---
    if check_variables and i == 0:
        check_available_variables_xarray(ds)
        break

    # Extract face coordinates using robust helper (works across xugrid versions)
    face_x, face_y = _get_face_coords(ds)

    # Build KDTree for spatial queries (needed for channel width analysis)
    if compare_channel_width:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.vstack([face_x, face_y]).T)

    for target_dt, ts_idx, actual_dt in snapshot_matches:
        target_label = _date_to_label(target_dt)
        actual_label = _date_to_label(actual_dt)
        snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
        snapshot_label = f"target={target_label} | actual={actual_label}"
        print(f"Scenario: {folder_str:25} | {snapshot_label} | HydroYear={hydro_years[ts_idx]:.2f}")

        comparison_results.setdefault(snapshot_key, {})
        comparison_results[snapshot_key][folder_str] = {}
        comparison_labels[snapshot_key] = target_label

        width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
        bedlev_data_snapshot = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

        # 4. WIDTH-AVERAGED BED LEVEL
        if compare_width_averaged_bedlevel:
            print(f"Computing Bed Level for {folder} ({snapshot_label})...")
            dx = 1000
            x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2

            bedlev_data = bedlev_data_snapshot.copy()

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

            comparison_results[snapshot_key][folder_str]['BL'] = np.array(temp_means)
            comparison_results[snapshot_key][folder_str]['x_centers'] = x_centers

            if plot_width_averaged_bedlevel_individual:
                plt.figure(figsize=(10, 6))
                plt.plot(x_centers/1000, temp_means, 'o-', color=scenario_color)
                plt.xlabel('Distance [km]')
                detrend_label = ' (Detrended)' if apply_detrending else ''
                plt.ylabel(f'Width-averaged Bed Level [m]{detrend_label}')
                plt.title(f'Width-averaged Bed Level: {scenario_label} ({snapshot_label})')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / f'width_averaged_bedlevel_map_{actual_label}_{folder_str}_Q{DISCHARGE}.png')
                if ts_idx == last_snapshot_idx:
                    plt.savefig(save_dir / f'width_averaged_bedlevel_map_final_{folder_str}_Q{DISCHARGE}.png')
                plt.close()

        # 5. MAXIMUM DEPTH ANALYSIS (95th percentile)
        if compare_max_depth:
            print(f"Computing Maximum Depth for {folder} ({snapshot_label})...")
            dx = 1000
            x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2

            bedlev_data = bedlev_data_snapshot.copy()

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

            comparison_results[snapshot_key][folder_str]['MaxDepth'] = np.array(max_depths)

            if plot_max_depth_individual:
                plt.figure(figsize=(10, 6))
                plt.plot(x_centers/1000, max_depths, 'o-', color=scenario_color)
                plt.xlabel('Distance [km]')
                depth_label = 'Absolute Depth' if use_absolute_depth else 'Depth'
                detrend_label = ' (Detrended)' if apply_detrending else ''
                plt.ylabel(f'{depth_percentile}th Percentile {depth_label} [m]{detrend_label}')
                plt.title(f'Maximum Channel Depth: {scenario_label} ({snapshot_label})')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / f'max_depth_map_{actual_label}_{folder_str}_Q{DISCHARGE}.png')
                if ts_idx == last_snapshot_idx:
                    plt.savefig(save_dir / f'max_depth_map_final_{folder_str}_Q{DISCHARGE}.png')
                plt.close()

        # 6. CHANNEL WIDTH ANALYSIS
        if compare_channel_width:
            print(f"Computing Channel Widths for {folder_str} ({snapshot_label})...")

            max_widths = []
            for x_coord in x_targets:
                distances, bed_profile = get_bed_profile_at_x(
                    ds, tree, x_coord, y_range, ts_idx,
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

            comparison_results[snapshot_key][folder_str]['ChannelWidth'] = np.array(max_widths)

            if plot_channel_width_individual:
                plt.figure(figsize=(10, 6))
                plt.plot(x_targets/1000, max_widths, 'o-', color=scenario_color)
                plt.xlabel('Distance [km]')
                plt.ylabel('Max Channel Width [m]')
                detrend_label = ' (Detrended)' if apply_detrending else ''
                plt.title(f'Maximum Channel Width: {scenario_label}{detrend_label} ({snapshot_label})')
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / f'channel_width_map_{actual_label}_{folder_str}_Q{DISCHARGE}.png')
                if ts_idx == last_snapshot_idx:
                    plt.savefig(save_dir / f'channel_width_map_final_{folder_str}_Q{DISCHARGE}.png')
                plt.close()

        # 7. HYPSOMETRIC CURVE
        if compare_hypsometric:
            print(f"Computing Hypsometric Curve for {folder_str} ({snapshot_label})...")
            bedlev_data = bedlev_data_snapshot.copy()
            x_mask = (face_x >= x_targets[0]) & (face_x <= x_targets[-1])
            domain_mask = width_mask & x_mask

            if apply_detrending:
                bedlev_data = bedlev_data - reference_bed
                valid_mask = domain_mask
            else:
                valid_mask = domain_mask & (bedlev_data < bed_threshold)

            elev_curve, area_curve, area_label = compute_hypsometric_curve(
                bedlev_data=bedlev_data,
                valid_mask=valid_mask,
                face_area=None,
            )

            comparison_results[snapshot_key][folder_str]['HypsoElevation'] = elev_curve
            comparison_results[snapshot_key][folder_str]['HypsoArea'] = area_curve
            comparison_results[snapshot_key][folder_str]['HypsoAreaLabel'] = area_label

            if plot_hypsometric_individual and elev_curve.size > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(area_curve, elev_curve, '-', color=scenario_color, linewidth=2)
                plt.xlabel(area_label)
                detrend_label = ' (Detrended)' if apply_detrending else ''
                plt.ylabel(f'Bed elevation [m]{detrend_label}')
                plt.title(f'Hypsometric Curve: {scenario_label} ({snapshot_label})')
                if not apply_detrending:
                    plt.axhline(y=bed_threshold, color='red', linestyle='--', alpha=0.7)
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / f'hypsometric_curve_{actual_label}_{folder_str}_Q{DISCHARGE}.png')
                if ts_idx == last_snapshot_idx:
                    plt.savefig(save_dir / f'hypsometric_curve_final_{folder_str}_Q{DISCHARGE}.png')
                plt.close()

    ds.close()

# --- OPTIONAL: ADD NON-NOISY Q500 BASELINE FOR COMPARISON ---
if NOISY and DISCHARGE == 500 and ADD_NON_NOISY_BASELINE_Q500:
    baseline_base_path = base_directory / config
    baseline_timed_out_dir = baseline_base_path / 'timed-out'
    baseline_assessment_dir = baseline_base_path / 'cached_data'
    baseline_assessment_dir.mkdir(parents=True, exist_ok=True)

    baseline_model_folders = find_variability_model_folders(
        base_path=baseline_base_path,
        discharge=DISCHARGE,
        scenarios_to_process=None,
        analyze_noisy=False,
    )

    print(f"\nLoading non-noisy Q500 baseline runs: {len(baseline_model_folders)} found")

    for i, folder in enumerate(baseline_model_folders):
        model_location = baseline_base_path / folder
        folder_str = folder.name

        print(f"\nProcessing baseline: {folder_str}")

        run_paths = get_stitched_map_run_paths(
            base_path=baseline_base_path,
            folder_name=folder.name,
            timed_out_dir=baseline_timed_out_dir,
            variability_map=VARIABILITY_MAP,
            analyze_noisy=False,
        )
        if not run_paths:
            run_paths = [model_location]

        cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
        ds = load_or_update_map_cache_multi(
            cache_dir=baseline_assessment_dir,
            folder_name=folder.name,
            run_paths=run_paths,
            var_names=['mesh2d_mor_bl'],
            bbox=CACHE_BBOX,
            append_time=APPEND_TIMESTEPS,
            append_vars=APPEND_VARIABLES,
            cache_tag=cache_tag,
        )
        if ds is None:
            print(f"  No cached baseline data for {folder}, skipping.")
            continue

        delta_time = ds.time.values - start_date
        hydro_years = delta_time / np.timedelta64(365, 'D')

        snapshot_matches = get_snapshot_matches_by_target_dates(ds.time.values, target_snapshot_dates)
        if not snapshot_matches:
            print(f"  No baseline timesteps found for {folder}, skipping.")
            ds.close()
            continue

        if apply_detrending:
            if use_mf50_reference and (reference_bed_MF50 is not None):
                reference_bed = reference_bed_MF50
            else:
                reference_bed = ds['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()

        if check_variables and i == 0:
            check_available_variables_xarray(ds)
            break

        face_x, face_y = _get_face_coords(ds)

        if compare_channel_width:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.vstack([face_x, face_y]).T)

        for target_dt, ts_idx, actual_dt in snapshot_matches:
            target_label = _date_to_label(target_dt)
            actual_label = _date_to_label(actual_dt)
            snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
            snapshot_label = f"target={target_label} | actual={actual_label}"
            print(f"Baseline: {folder_str:25} | {snapshot_label} | HydroYear={hydro_years[ts_idx]:.2f}")

            baseline_comparison_results.setdefault(snapshot_key, {})
            baseline_comparison_results[snapshot_key][folder_str] = {}
            comparison_labels[snapshot_key] = target_label

            width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
            bedlev_data_snapshot = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

            if compare_width_averaged_bedlevel:
                dx = 1000
                x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
                x_centers = (x_bins[:-1] + x_bins[1:]) / 2

                bedlev_data = bedlev_data_snapshot.copy()
                if apply_detrending:
                    bedlev_data = bedlev_data - reference_bed
                    valid_mask = width_mask
                else:
                    valid_mask = (width_mask) & (bedlev_data < bed_threshold)

                temp_means = []
                for k in range(len(x_bins)-1):
                    bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k+1])
                    temp_means.append(np.mean(bedlev_data[bin_mask]) if np.any(bin_mask) else np.nan)

                baseline_comparison_results[snapshot_key][folder_str]['BL'] = np.array(temp_means)
                baseline_comparison_results[snapshot_key][folder_str]['x_centers'] = x_centers

            if compare_max_depth:
                dx = 1000
                x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
                x_centers = (x_bins[:-1] + x_bins[1:]) / 2

                bedlev_data = bedlev_data_snapshot.copy()
                if apply_detrending:
                    bedlev_data = bedlev_data - reference_bed

                if use_absolute_depth:
                    depths_field = np.abs(bedlev_data)
                else:
                    depths_field = -bedlev_data

                if apply_detrending:
                    valid_mask = width_mask
                else:
                    valid_mask = (width_mask) & (bedlev_data < bed_threshold)

                max_depths = []
                for k in range(len(x_bins)-1):
                    bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k+1])
                    if np.any(bin_mask):
                        bin_depths = depths_field[bin_mask]
                        valid_depths = bin_depths[~np.isnan(bin_depths)]
                        if len(valid_depths) > 0:
                            max_depths.append(np.percentile(valid_depths, depth_percentile))
                        else:
                            max_depths.append(np.nan)
                    else:
                        max_depths.append(np.nan)

                baseline_comparison_results[snapshot_key][folder_str]['MaxDepth'] = np.array(max_depths)
                baseline_comparison_results[snapshot_key][folder_str]['x_centers'] = x_centers

            if compare_channel_width:
                max_widths = []
                for x_coord in x_targets:
                    distances, bed_profile = get_bed_profile_at_x(
                        ds, tree, x_coord, y_range, ts_idx,
                        reference_bed=reference_bed if apply_detrending else None,
                        detrend=apply_detrending
                    )

                    if apply_detrending:
                        bed_profile[np.abs(bed_profile) > bed_threshold] = np.nan
                    else:
                        bed_profile[bed_profile > bed_threshold] = np.nan

                    max_width = compute_max_channel_width(bed_profile, distances, safety_buffer)
                    max_widths.append(max_width)

                baseline_comparison_results[snapshot_key][folder_str]['ChannelWidth'] = np.array(max_widths)

            if compare_hypsometric:
                bedlev_data = bedlev_data_snapshot.copy()
                x_mask = (face_x >= x_targets[0]) & (face_x <= x_targets[-1])
                domain_mask = width_mask & x_mask

                if apply_detrending:
                    bedlev_data = bedlev_data - reference_bed
                    valid_mask = domain_mask
                else:
                    valid_mask = domain_mask & (bedlev_data < bed_threshold)

                elev_curve, area_curve, area_label = compute_hypsometric_curve(
                    bedlev_data=bedlev_data,
                    valid_mask=valid_mask,
                    face_area=None,
                )

                baseline_comparison_results[snapshot_key][folder_str]['HypsoElevation'] = elev_curve
                baseline_comparison_results[snapshot_key][folder_str]['HypsoArea'] = area_curve
                baseline_comparison_results[snapshot_key][folder_str]['HypsoAreaLabel'] = area_label

        ds.close()


# %% --- 7. FINAL COMPARISON PLOT ---
print("\nGenerating Comparison Plot...")

for snapshot_key, snapshot_results in comparison_results.items():
    if not snapshot_results:
        continue

    # Count active plots
    first_key = list(snapshot_results.keys())[0]
    n_plots = sum([
        compare_braiding_index and 'BI_tau' in snapshot_results[first_key],
        compare_braiding_index and 'BI_depth' in snapshot_results[first_key],
        compare_width_averaged_bedlevel,
        compare_max_depth,
        compare_channel_width
    ])

    if n_plots == 0:
        print("No plots to generate!")
        continue

    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    scenario_groups = group_snapshot_by_scenario(snapshot_results)
    baseline_groups = group_snapshot_by_scenario(baseline_comparison_results.get(snapshot_key, {}))
    sorted_scenarios = sort_scenario_keys(scenario_groups.keys())
    plot_idx = 0

    def _plot_noisy_context(ax, x, y_noisy_stack, y_base_for_envelope=None, add_labels=False):
        """Plot thin noisy trajectories and noisy envelope, optionally including baserun in bounds."""
        if not NOISY or y_noisy_stack is None or y_noisy_stack.shape[0] < 1:
            return

        # Thin grey trajectories for all noisy runs.
        for i in range(y_noisy_stack.shape[0]):
            ax.plot(
                x,
                y_noisy_stack[i],
                color='grey',
                alpha=0.35,
                linewidth=0.7,
                label='Noisy runs' if (add_labels and i == 0) else None,
                zorder=1,
            )

        y_env_stack = y_noisy_stack
        if y_base_for_envelope is not None:
            yb = np.asarray(y_base_for_envelope)
            if yb.ndim == 1:
                yb = yb[np.newaxis, :]
            if yb.size > 0:
                y_env_stack = np.vstack([y_env_stack, yb])

        if y_env_stack.shape[0] > 1:
            y_min = np.nanmin(y_env_stack, axis=0)
            y_max = np.nanmax(y_env_stack, axis=0)
            ax.fill_between(
                x,
                y_min,
                y_max,
                color='grey',
                alpha=0.18,
                label='Noisy envelope (incl. baserun)' if add_labels else None,
                zorder=1,
            )

    # Plot 1: Shear Stress BI
    if compare_braiding_index and 'BI_tau' in snapshot_results[first_key]:
        noisy_legend_added = False
        for scenario in sorted_scenarios:
            run_items = scenario_groups[scenario]
            y_stack = stack_metric_arrays(run_items, 'BI_tau')
            y_base = stack_metric_arrays(baseline_groups[scenario], 'BI_tau') if scenario in baseline_groups else None

            _plot_noisy_context(
                axes[plot_idx],
                x_targets / 1000,
                y_stack,
                y_base_for_envelope=y_base,
                add_labels=not noisy_legend_added,
            )
            if NOISY and y_stack is not None and y_stack.shape[0] > 0:
                noisy_legend_added = True

            if not NOISY:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_targets / 1000,
                    y_stack=y_stack,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='o',
                )
            if scenario in baseline_groups:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_targets / 1000,
                    y_stack=y_base,
                    color=BASELINE_COLOR,
                    label=f"{_scenario_label(scenario, SCENARIO_LABELS)} baseline (solid)",
                    add_envelope=False,
                    marker=None,
                    linestyle='-',
                )
        axes[plot_idx].set_title(f'BI ({var_tau}), fixed threshold: tau > {tau_threshold} N/m²')
        axes[plot_idx].set_ylabel('braiding index')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1

    # Plot 2: Water Depth BI
    if compare_braiding_index and 'BI_depth' in snapshot_results[first_key]:
        noisy_legend_added = False
        for scenario in sorted_scenarios:
            run_items = scenario_groups[scenario]
            y_stack = stack_metric_arrays(run_items, 'BI_depth')
            y_base = stack_metric_arrays(baseline_groups[scenario], 'BI_depth') if scenario in baseline_groups else None

            _plot_noisy_context(
                axes[plot_idx],
                x_targets / 1000,
                y_stack,
                y_base_for_envelope=y_base,
                add_labels=not noisy_legend_added,
            )
            if NOISY and y_stack is not None and y_stack.shape[0] > 0:
                noisy_legend_added = True

            if not NOISY:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_targets / 1000,
                    y_stack=y_stack,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='s',
                )
            if scenario in baseline_groups:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_targets / 1000,
                    y_stack=y_base,
                    color=BASELINE_COLOR,
                    label=f"{_scenario_label(scenario, SCENARIO_LABELS)} baseline (solid)",
                    add_envelope=False,
                    marker=None,
                    linestyle='-',
                )
        axes[plot_idx].set_title(f'BI ({var_depth}), relative threshold: {int(depth_threshold*100)}% above mean water depth')
        axes[plot_idx].set_ylabel('braiding index')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1

    # Plot 3: Bed Level
    if compare_width_averaged_bedlevel:
        noisy_legend_added = False
        for scenario in sorted_scenarios:
            run_items = scenario_groups[scenario]
            y_stack = stack_metric_arrays(run_items, 'BL')
            if y_stack is None:
                continue
            first_with_x = next((d for _, d in run_items if 'x_centers' in d), None)
            if first_with_x is None:
                continue
            x_vals = first_with_x['x_centers'] / 1000
            if not NOISY:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_vals,
                    y_stack=y_stack,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker=None,
                )
            if scenario in baseline_groups:
                y_base = stack_metric_arrays(baseline_groups[scenario], 'BL')
                base_x_data = next((d for _, d in baseline_groups[scenario] if 'x_centers' in d), None)
                _plot_noisy_context(
                    axes[plot_idx],
                    x_vals,
                    y_stack,
                    y_base_for_envelope=y_base,
                    add_labels=not noisy_legend_added,
                )
                if NOISY and y_stack is not None and y_stack.shape[0] > 0:
                    noisy_legend_added = True
                if base_x_data is not None:
                    draw_metric_with_optional_envelope(
                        ax=axes[plot_idx],
                        x=base_x_data['x_centers'] / 1000,
                        y_stack=y_base,
                        color=BASELINE_COLOR,
                        label=f"{_scenario_label(scenario, SCENARIO_LABELS)} baseline (solid)",
                        add_envelope=False,
                        marker=None,
                        linestyle='-',
                    )
            else:
                _plot_noisy_context(
                    axes[plot_idx],
                    x_vals,
                    y_stack,
                    y_base_for_envelope=None,
                    add_labels=not noisy_legend_added,
                )
                if NOISY and y_stack is not None and y_stack.shape[0] > 0:
                    noisy_legend_added = True
        axes[plot_idx].set_title('width-averaged bed level')
        axes[plot_idx].set_ylabel('bed level [m]')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1

    # Plot 4: Maximum Depth
    if compare_max_depth:
        noisy_legend_added = False
        for scenario in sorted_scenarios:
            run_items = scenario_groups[scenario]
            y_stack = stack_metric_arrays(run_items, 'MaxDepth')
            if y_stack is None:
                continue
            first_with_x = next((d for _, d in run_items if 'x_centers' in d), None)
            if first_with_x is None:
                continue
            x_vals = first_with_x['x_centers'] / 1000
            if not NOISY:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_vals,
                    y_stack=y_stack,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='o',
                )
            if scenario in baseline_groups:
                y_base = stack_metric_arrays(baseline_groups[scenario], 'MaxDepth')
                base_x_data = next((d for _, d in baseline_groups[scenario] if 'x_centers' in d), None)
                _plot_noisy_context(
                    axes[plot_idx],
                    x_vals,
                    y_stack,
                    y_base_for_envelope=y_base,
                    add_labels=not noisy_legend_added,
                )
                if NOISY and y_stack is not None and y_stack.shape[0] > 0:
                    noisy_legend_added = True
                if base_x_data is not None:
                    draw_metric_with_optional_envelope(
                        ax=axes[plot_idx],
                        x=base_x_data['x_centers'] / 1000,
                        y_stack=y_base,
                        color=BASELINE_COLOR,
                        label=f"{_scenario_label(scenario, SCENARIO_LABELS)} baseline (solid)",
                        add_envelope=False,
                        marker=None,
                        linestyle='-',
                    )
            else:
                _plot_noisy_context(
                    axes[plot_idx],
                    x_vals,
                    y_stack,
                    y_base_for_envelope=None,
                    add_labels=not noisy_legend_added,
                )
                if NOISY and y_stack is not None and y_stack.shape[0] > 0:
                    noisy_legend_added = True
        axes[plot_idx].set_title(f'p{depth_percentile} channel depth')
        axes[plot_idx].set_ylabel('depth [m]')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)
        plot_idx += 1

    # Plot 5: Channel Width
    if compare_channel_width:
        noisy_legend_added = False
        for scenario in sorted_scenarios:
            run_items = scenario_groups[scenario]
            y_stack = stack_metric_arrays(run_items, 'ChannelWidth')
            y_base = stack_metric_arrays(baseline_groups[scenario], 'ChannelWidth') if scenario in baseline_groups else None

            _plot_noisy_context(
                axes[plot_idx],
                x_targets / 1000,
                y_stack,
                y_base_for_envelope=y_base,
                add_labels=not noisy_legend_added,
            )
            if NOISY and y_stack is not None and y_stack.shape[0] > 0:
                noisy_legend_added = True

            if not NOISY:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_targets / 1000,
                    y_stack=y_stack,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='s',
                )
            if scenario in baseline_groups:
                draw_metric_with_optional_envelope(
                    ax=axes[plot_idx],
                    x=x_targets / 1000,
                    y_stack=y_base,
                    color=BASELINE_COLOR,
                    label=f"{_scenario_label(scenario, SCENARIO_LABELS)} baseline (solid)",
                    add_envelope=False,
                    marker=None,
                    linestyle='-',
                )
        axes[plot_idx].set_title(f'maximum channel width (threshold: mean depth - {int(safety_buffer*100)} cm)')
        axes[plot_idx].set_ylabel('width [m]')
        axes[plot_idx].legend(loc='best')
        axes[plot_idx].grid(True, alpha=0.2)

    axes[-1].set_xlabel('x-coordinate along estuary [km]')
    fig.suptitle(f"Hydrodynamic snapshot around {comparison_labels.get(snapshot_key, snapshot_key)} for $Q_{{mean}}$ = {DISCHARGE} m³/s", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    snapshot_date = comparison_labels.get(snapshot_key, snapshot_key)

    if apply_detrending:
        plt.savefig(summary_output_dir / f'overall_morphology_variability_comparison_detrended_{snapshot_date}_Q{DISCHARGE}.png', dpi=300)
    else:
        plt.savefig(summary_output_dir / f'overall_morphology_variability_comparison_{snapshot_date}_Q{DISCHARGE}.png', dpi=300)
    plt.show()

    print(f'Saved comparison plot at {summary_output_dir} for {snapshot_key}')

    # Optional second set: noisy envelope context + deterministic variability overlays
    # (baserun + seasonal + flashy + singlepeak).
    if NOISY and baseline_groups:
        n_ref_plots = n_plots
        fig_ref, axes_ref = plt.subplots(n_ref_plots, 1, figsize=(12, 4 * n_ref_plots), sharex=True)
        if n_ref_plots == 1:
            axes_ref = [axes_ref]

        ref_scenarios = sort_scenario_keys(baseline_groups.keys())
        noisy_scenarios = sort_scenario_keys(scenario_groups.keys())

        def _stack_noisy_all(metric_key):
            stacks = []
            for scn in noisy_scenarios:
                y_scn = stack_metric_arrays(scenario_groups[scn], metric_key)
                if y_scn is not None and y_scn.size > 0:
                    stacks.append(y_scn)
            if not stacks:
                return None
            return np.vstack(stacks)

        ref_idx = 0

        if compare_braiding_index and 'BI_tau' in snapshot_results[first_key]:
            y_noisy = _stack_noisy_all('BI_tau')
            _plot_noisy_context(
                ax=axes_ref[ref_idx],
                x=x_targets / 1000,
                y_noisy_stack=y_noisy,
                y_base_for_envelope=None,
                add_labels=True,
            )
            for scenario in ref_scenarios:
                y_ref = stack_metric_arrays(baseline_groups[scenario], 'BI_tau')
                draw_metric_with_optional_envelope(
                    ax=axes_ref[ref_idx],
                    x=x_targets / 1000,
                    y_stack=y_ref,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='o',
                    linestyle='-',
                )
            axes_ref[ref_idx].set_title(f'BI ({var_tau}), fixed threshold: tau > {tau_threshold} N/m²')
            axes_ref[ref_idx].set_ylabel('braiding index')
            axes_ref[ref_idx].legend(loc='best')
            axes_ref[ref_idx].grid(True, alpha=0.2)
            ref_idx += 1

        if compare_braiding_index and 'BI_depth' in snapshot_results[first_key]:
            y_noisy = _stack_noisy_all('BI_depth')
            _plot_noisy_context(
                ax=axes_ref[ref_idx],
                x=x_targets / 1000,
                y_noisy_stack=y_noisy,
                y_base_for_envelope=None,
                add_labels=True,
            )
            for scenario in ref_scenarios:
                y_ref = stack_metric_arrays(baseline_groups[scenario], 'BI_depth')
                draw_metric_with_optional_envelope(
                    ax=axes_ref[ref_idx],
                    x=x_targets / 1000,
                    y_stack=y_ref,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='s',
                    linestyle='-',
                )
            axes_ref[ref_idx].set_title(f'BI ({var_depth}), relative threshold: {int(depth_threshold*100)}% above mean water depth')
            axes_ref[ref_idx].set_ylabel('braiding index')
            axes_ref[ref_idx].legend(loc='best')
            axes_ref[ref_idx].grid(True, alpha=0.2)
            ref_idx += 1

        if compare_width_averaged_bedlevel:
            noisy_items_bl = []
            for scn in noisy_scenarios:
                noisy_items_bl.extend(scenario_groups[scn])
            y_noisy_bl = _stack_noisy_all('BL')
            x_noisy_bl_data = next((d for _, d in noisy_items_bl if 'x_centers' in d), None)
            if x_noisy_bl_data is not None:
                _plot_noisy_context(
                    ax=axes_ref[ref_idx],
                    x=x_noisy_bl_data['x_centers'] / 1000,
                    y_noisy_stack=y_noisy_bl,
                    y_base_for_envelope=None,
                    add_labels=True,
                )
            for scenario in ref_scenarios:
                run_items = baseline_groups[scenario]
                y_ref = stack_metric_arrays(run_items, 'BL')
                if y_ref is None:
                    continue
                x_ref_data = next((d for _, d in run_items if 'x_centers' in d), None)
                if x_ref_data is None:
                    continue
                draw_metric_with_optional_envelope(
                    ax=axes_ref[ref_idx],
                    x=x_ref_data['x_centers'] / 1000,
                    y_stack=y_ref,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker=None,
                    linestyle='-',
                )
            axes_ref[ref_idx].set_title('width-averaged bed level')
            axes_ref[ref_idx].set_ylabel('bed level [m]')
            axes_ref[ref_idx].legend(loc='best')
            axes_ref[ref_idx].grid(True, alpha=0.2)
            ref_idx += 1

        if compare_max_depth:
            noisy_items_md = []
            for scn in noisy_scenarios:
                noisy_items_md.extend(scenario_groups[scn])
            y_noisy_md = _stack_noisy_all('MaxDepth')
            x_noisy_md_data = next((d for _, d in noisy_items_md if 'x_centers' in d), None)
            if x_noisy_md_data is not None:
                _plot_noisy_context(
                    ax=axes_ref[ref_idx],
                    x=x_noisy_md_data['x_centers'] / 1000,
                    y_noisy_stack=y_noisy_md,
                    y_base_for_envelope=None,
                    add_labels=True,
                )
            for scenario in ref_scenarios:
                run_items = baseline_groups[scenario]
                y_ref = stack_metric_arrays(run_items, 'MaxDepth')
                if y_ref is None:
                    continue
                x_ref_data = next((d for _, d in run_items if 'x_centers' in d), None)
                if x_ref_data is None:
                    continue
                draw_metric_with_optional_envelope(
                    ax=axes_ref[ref_idx],
                    x=x_ref_data['x_centers'] / 1000,
                    y_stack=y_ref,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='o',
                    linestyle='-',
                )
            axes_ref[ref_idx].set_title(f'p{depth_percentile} channel depth')
            axes_ref[ref_idx].set_ylabel('depth [m]')
            axes_ref[ref_idx].legend(loc='best')
            axes_ref[ref_idx].grid(True, alpha=0.2)
            ref_idx += 1

        if compare_channel_width:
            y_noisy = _stack_noisy_all('ChannelWidth')
            _plot_noisy_context(
                ax=axes_ref[ref_idx],
                x=x_targets / 1000,
                y_noisy_stack=y_noisy,
                y_base_for_envelope=None,
                add_labels=True,
            )
            for scenario in ref_scenarios:
                y_ref = stack_metric_arrays(baseline_groups[scenario], 'ChannelWidth')
                draw_metric_with_optional_envelope(
                    ax=axes_ref[ref_idx],
                    x=x_targets / 1000,
                    y_stack=y_ref,
                    color=_scenario_color(scenario, SCENARIO_COLORS),
                    label=_scenario_label(scenario, SCENARIO_LABELS),
                    add_envelope=False,
                    marker='s',
                    linestyle='-',
                )
            axes_ref[ref_idx].set_title(f'maximum channel width (threshold: mean depth - {int(safety_buffer*100)} cm)')
            axes_ref[ref_idx].set_ylabel('width [m]')
            axes_ref[ref_idx].legend(loc='best')
            axes_ref[ref_idx].grid(True, alpha=0.2)

        axes_ref[-1].set_xlabel('x-coordinate along estuary [km]')
        fig_ref.suptitle(
            f"Noisy envelope + variability overlays around {comparison_labels.get(snapshot_key, snapshot_key)}",
            fontsize=12,
        )
        fig_ref.tight_layout(rect=[0, 0.03, 1, 0.97])

        if apply_detrending:
            fig_ref.savefig(summary_output_dir / f'overall_morphology_variability_overlay_detrended_{snapshot_date}_Q{DISCHARGE}.png', dpi=300)
        else:
            fig_ref.savefig(summary_output_dir / f'overall_morphology_variability_overlay_{snapshot_date}_Q{DISCHARGE}.png', dpi=300)
        plt.show()
        print(f'Saved noisy-envelope + variability overlay comparison at {summary_output_dir} for {snapshot_key}')

    # Separate hypsometric comparison plot for this snapshot.
    if compare_hypsometric:
        fig_h, ax_h = plt.subplots(figsize=(10, 6))
        has_hypso = False
        area_labels = []

        for scenario in sorted_scenarios:
            run_items = scenario_groups[scenario]
            curves = []
            for _, data in run_items:
                if 'HypsoElevation' not in data or 'HypsoArea' not in data:
                    continue
                if data['HypsoElevation'].size == 0:
                    continue
                curves.append((np.asarray(data['HypsoArea']), np.asarray(data['HypsoElevation']), data.get('HypsoAreaLabel', 'Cumulative area')))

            if not curves:
                continue

            has_hypso = True
            area_labels.extend([lbl for _, _, lbl in curves])
            color = _scenario_color(scenario, SCENARIO_COLORS)
            label = _scenario_label(scenario, SCENARIO_LABELS)

            if NOISY and len(curves) > 1:
                min_common = max(np.nanmin(a) for a, _, _ in curves)
                max_common = min(np.nanmax(a) for a, _, _ in curves)
                if np.isfinite(min_common) and np.isfinite(max_common) and max_common > min_common:
                    area_grid = np.linspace(min_common, max_common, 300)
                    elev_stack = []
                    for area_vals, elev_vals, _ in curves:
                        order = np.argsort(area_vals)
                        area_sorted = area_vals[order]
                        elev_sorted = elev_vals[order]
                        elev_stack.append(np.interp(area_grid, area_sorted, elev_sorted))
                    elev_stack = np.vstack(elev_stack)
                    ax_h.fill_between(
                        area_grid,
                        np.nanmin(elev_stack, axis=0),
                        np.nanmax(elev_stack, axis=0),
                        color=color,
                        alpha=0.18,
                    )
                    ax_h.plot(area_grid, np.nanmean(elev_stack, axis=0), linewidth=2, color=color, label=label)
                else:
                    area_vals, elev_vals, _ = curves[0]
                    ax_h.plot(area_vals, elev_vals, linewidth=2, color=color, label=label)
            else:
                area_vals, elev_vals, _ = curves[0]
                ax_h.plot(area_vals, elev_vals, linewidth=2, color=color, label=label)

            if scenario in baseline_groups:
                base_curves = []
                for _, base_data in baseline_groups[scenario]:
                    if 'HypsoElevation' not in base_data or 'HypsoArea' not in base_data:
                        continue
                    if base_data['HypsoElevation'].size == 0:
                        continue
                    base_curves.append((
                        np.asarray(base_data['HypsoArea']),
                        np.asarray(base_data['HypsoElevation']),
                    ))
                if base_curves:
                    base_area, base_elev = base_curves[0]
                    ax_h.plot(
                        base_area,
                        base_elev,
                        linewidth=2,
                        color=BASELINE_COLOR,
                        linestyle='-',
                        label=f"{label} baseline",
                    )

        if has_hypso:
            x_label = area_labels[0] if len(set(area_labels)) == 1 else 'Cumulative area'
            ax_h.set_xlabel(x_label)
            ax_h.set_ylabel('Bed elevation [m]')
            ax_h.set_title(f'Hypsometric curves around {comparison_labels.get(snapshot_key, snapshot_key)}  for $Q_{{mean}}$ = {DISCHARGE} m³/s')
            if not apply_detrending:
                ax_h.axhline(y=bed_threshold, color='red', linestyle='--', alpha=0.7)
            ax_h.grid(True, alpha=0.2)
            ax_h.legend(loc='best')
            fig_h.tight_layout()
            snapshot_date = comparison_labels.get(snapshot_key, snapshot_key)
            if apply_detrending:
                fig_h.savefig(summary_output_dir / f'hypsometric_comparison_detrended_{snapshot_date}_Q{DISCHARGE}.png', dpi=300)
            else:
                fig_h.savefig(summary_output_dir / f'hypsometric_comparison_{snapshot_date}_Q{DISCHARGE}.png', dpi=300)
            plt.show()
            print(f'Saved hypsometric comparison plot at {summary_output_dir} for {snapshot_key}')
        else:
            plt.close(fig_h)

print("\nAll processing complete.")

#%% """"Post-process multiple scenarios map output, 
# plot overall morphological characteristics along the estuary (bed level, max depth, channel width)"""

# #%% IMPORTS
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# import sys

# #%%
# # Add the current working directory (where FUNCTIONS is located)
# sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

# from FUNCTIONS.F_general import *
# from FUNCTIONS.F_braiding_index import *
# from FUNCTIONS.F_channelwidth import *
# from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
# from FUNCTIONS.F_loaddata import get_stitched_map_run_paths


# #%% --- CONFIGURATION ---
# # Model output
# DISCHARGE = 1000  # or 1000, etc.
# base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
# config = f'Model_Output/Q{DISCHARGE}'

# # Braiding index
# tau_threshold = 0.05
# depth_threshold = 0.2 # A depth_threshold of 0.2 means a channel must be 20% deeper than the average depth of that specific cross-section to be counted (ignores thin water over bars).

# # Land threshold
# bed_threshold = 6

# # Channel depth + width analysis
# depth_percentile = 95  # For maximum depth analysis (95th percentile)
# safety_buffer = 0.20  # For channel width analysis (20 cm below mean)

# #%% -- special configuration | do not change ---
# special_base = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
# special_config = 'Test_MORFAC/03_flashy/Tmorph_50years'
# use_mf50_reference = (base_directory == special_base) and (config == special_config)

# #%% --- SETTINGS ---
# var_tau = 'mesh2d_tausmax'
# var_depth = 'mesh2d_waterdepth'

# start_date = np.datetime64('2025-01-01') 
# x_targets = np.arange(20000, 44001, 1000)
# y_range = (5000, 10000)

# CACHE_BBOX = [x_targets[0], y_range[0], x_targets[-1], y_range[1]]
# CACHE_TAG = None
# APPEND_TIMESTEPS = True
# APPEND_VARIABLES = True

# apply_detrending = False  # Subtract initial bed level to see changes
# reference_time_idx = 0   # Time index to use as reference (0 = first timestep)
# use_absolute_depth = True  # Use absolute depth values (positive = deep)

# # Snapshot settings (hydrodynamic dates)
# SNAPSHOT_TARGET_DATES = None  # e.g. ['2027-01-01', '2035-01-01']; None -> equally spaced in SNAPSHOT_DATE_RANGE
# SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2055-12-31'))
# SNAPSHOT_COUNT = 4

# check_variables = False

# compare_braiding_index = False
# plot_braiding_index_individual = False

# compare_width_averaged_bedlevel = True
# plot_width_averaged_bedlevel_individual = True

# compare_max_depth = True  
# plot_max_depth_individual = True

# compare_channel_width = True 
# plot_channel_width_individual = True

# compare_hypsometric = True
# plot_hypsometric_individual = True

# # Human-readable labels per scenario number (used in combined plots)
# SCENARIO_LABELS = {
#     '1': 'Constant',
#     '2': 'Seasonal',
#     '3': 'Flashy',
#     '4': 'Single peak',
# }

# # Colours: one per scenario (used in combined plots)
# SCENARIO_COLORS = {
#     '1': '#1f77b4',   # blue   - Constant
#     '2': '#ff7f0e',   # orange - Seasonal
#     '3': '#2ca02c',   # green  - Flashy
#     '4': '#d62728',   # red    - Single peak
# }


# def _date_to_filename_tag(dt64):
#     return str(np.datetime_as_string(dt64, unit='D')).replace('-', '')


# def _date_to_label(dt64):
#     return str(np.datetime_as_string(dt64, unit='D'))


# def _scenario_key_from_folder(folder_name):
#     try:
#         return str(int(str(folder_name).split('_')[0]))
#     except Exception:
#         return str(folder_name).split('_')[0]


# def _scenario_label(folder_name):
#     key = _scenario_key_from_folder(folder_name)
#     return SCENARIO_LABELS.get(key, str(folder_name))


# def _scenario_color(folder_name):
#     key = _scenario_key_from_folder(folder_name)
#     return SCENARIO_COLORS.get(key, 'grey')


# def _scenario_legend_label(folder_name):
#     key = _scenario_key_from_folder(folder_name)
#     base = SCENARIO_LABELS.get(key, key)
#     return f"{base} ({folder_name})"


# def get_target_snapshot_dates(count=4, explicit_dates=None, date_range=None):
#     if explicit_dates:
#         return [np.datetime64(d).astype('datetime64[ns]') for d in explicit_dates]

#     count = max(2, int(count))
#     if date_range is None:
#         start_dt = np.datetime64('2025-01-01').astype('datetime64[ns]')
#         end_dt = np.datetime64('2055-12-31').astype('datetime64[ns]')
#     else:
#         start_dt = np.datetime64(date_range[0]).astype('datetime64[ns]')
#         end_dt = np.datetime64(date_range[1]).astype('datetime64[ns]')

#     # Build an even spacing in nanoseconds to avoid index-based alignment.
#     ns_grid = np.linspace(start_dt.astype('int64'), end_dt.astype('int64'), count)
#     return [np.datetime64(int(ns), 'ns') for ns in ns_grid]


# def get_snapshot_matches_by_target_dates(time_values, target_dates):
#     if len(time_values) == 0:
#         return []

#     time_dt = np.array(time_values, dtype='datetime64[ns]')
#     time_ns = time_dt.astype('int64')
#     matches = []
#     for target_dt in target_dates:
#         target_ns = np.datetime64(target_dt, 'ns').astype('int64')
#         ts_idx = int(np.argmin(np.abs(time_ns - target_ns)))
#         actual_dt = time_dt[ts_idx]
#         matches.append((target_dt, ts_idx, actual_dt))
#     return matches


# def compute_hypsometric_curve(bedlev_data, valid_mask, face_area=None):
#     vals = bedlev_data[valid_mask]
#     vals = vals[np.isfinite(vals)]
#     if vals.size == 0:
#         return np.array([]), np.array([]), 'Cumulative area'

#     if face_area is not None:
#         area_vals = face_area[valid_mask]
#         area_vals = area_vals[np.isfinite(vals)] if area_vals.shape != vals.shape else area_vals
#         area_vals = np.asarray(area_vals, dtype=float)
#         if area_vals.size != vals.size or np.all(~np.isfinite(area_vals)):
#             area_vals = np.ones_like(vals, dtype=float)
#             area_label = 'Cumulative area fraction [-]'
#             to_plot_area = False
#         else:
#             area_vals = np.where(np.isfinite(area_vals), area_vals, 0.0)
#             area_label = 'Cumulative area [km²]'
#             to_plot_area = True
#     else:
#         area_vals = np.ones_like(vals, dtype=float)
#         area_label = 'Cumulative area fraction [-]'
#         to_plot_area = False

#     order = np.argsort(vals)
#     elev_sorted = vals[order]
#     area_sorted = area_vals[order]
#     cum_area = np.cumsum(area_sorted)

#     if to_plot_area:
#         cum_area = cum_area / 1e6  # m² -> km²
#     else:
#         cum_area = cum_area / cum_area[-1]

#     return elev_sorted, cum_area, area_label

# #%% --- SEARCH & SORT FOLDERS ---
# base_path = base_directory / config

# if DISCHARGE == 500:
#     VARIABILITY_MAP = {
#         '1': f'01_baserun{DISCHARGE}',
#         '2': f'02_run{DISCHARGE}_seasonal',
#         '3': f'03_run{DISCHARGE}_flashy',
#         '4': f'04_run{DISCHARGE}_singlepeak'
#     }
#     # Find run folders starting with a digit (e.g. 1_rst, 2_rst)
#     model_folders = [f for f in base_path.iterdir() 
#                     if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
#     model_folders.sort(key=lambda x: int(x.name.split('_')[0]))

# if DISCHARGE == 1000:
#     VARIABILITY_MAP = {
#         '01': f'01_baserun{DISCHARGE}',
#         '02': f'02_run{DISCHARGE}_seasonal',
#         '03': f'03_run{DISCHARGE}_flashy',
#         '04': f'04_run{DISCHARGE}_singlepeak'
#     }
#     # Find run folders starting with a digit (e.g. 1_rst, 2_rst)
#     model_folders = [f for f in base_path.iterdir() 
#                     if f.is_dir() and f.name[0].isdigit()]
#     model_folders.sort(key=lambda x: int(x.name.split('_')[0]))


# # Directories
# assessment_dir = base_path / 'cached_data'
# assessment_dir.mkdir(parents=True, exist_ok=True)
# timed_out_dir = base_path / 'timed-out'
# summary_output_dir = base_path / 'output_plots'
# summary_output_dir.mkdir(parents=True, exist_ok=True)

# # --- OPTIONAL: GLOBAL REFERENCE FROM MF50 ---
# reference_bed_MF50 = None
# if apply_detrending and use_mf50_reference:
#     mf50_folder = [f for f in model_folders if get_mf_number(f) == 50]
#     if len(mf50_folder) == 1:
#         mf50_folder = mf50_folder[0]
#         mf50_run_paths = get_stitched_map_run_paths(
#             base_path=base_path,
#             folder_name=mf50_folder,
#             timed_out_dir=timed_out_dir,
#             variability_map=VARIABILITY_MAP,
#             analyze_noisy=False,
#         )
#         if not mf50_run_paths:
#             mf50_run_paths = [base_path / mf50_folder]
#         cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
#         ds_mf50 = load_or_update_map_cache_multi(
#             cache_dir=assessment_dir,
#             folder_name=mf50_folder,
#             run_paths=mf50_run_paths,
#             var_names=['mesh2d_mor_bl'],
#             bbox=CACHE_BBOX,
#             append_time=APPEND_TIMESTEPS,
#             append_vars=APPEND_VARIABLES,
#             cache_tag=cache_tag,
#         )
#         if ds_mf50 is not None:
#             reference_bed_MF50 = ds_mf50['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()
#             ds_mf50.close()
#     else:
#         # Fallback: no MF50 found, keep run-specific behavior
#         use_mf50_reference = False

# # --- STORE SNAPSHOT RESULTS ---
# comparison_results = {}
# comparison_labels = {}

# target_snapshot_dates = get_target_snapshot_dates(
#     count=SNAPSHOT_COUNT,
#     explicit_dates=SNAPSHOT_TARGET_DATES,
#     date_range=SNAPSHOT_DATE_RANGE,
# )

# print("\nTarget hydrodynamic snapshot dates:")
# for dt in target_snapshot_dates:
#     print(f"  - {_date_to_label(dt)}")

# # --- COMPUTE MAP RESULTS FOR EACH RUN ---
# for i, folder in enumerate(model_folders):
#     model_location = base_path / folder
#     folder_str = folder.name
#     save_dir = summary_output_dir / f'mapplots_{folder_str}'
#     save_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\nProcessing: {folder_str}")
#     scenario_color = _scenario_color(folder_str)
#     scenario_label = _scenario_label(folder_str)
    
#     # 1. LOAD FM DATA (cached)
#     run_paths = get_stitched_map_run_paths(
#         base_path=base_path,
#         folder_name=folder,
#         timed_out_dir=timed_out_dir,
#         variability_map=VARIABILITY_MAP,
#         analyze_noisy=False,
#     )
#     if not run_paths:
#         run_paths = [model_location]

#     cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
#     ds = load_or_update_map_cache_multi(
#         cache_dir=assessment_dir,
#         folder_name=folder,
#         run_paths=run_paths,
#         var_names=['mesh2d_mor_bl'],#, 'mesh2d_face_x', 'mesh2d_face_y'],
#         bbox=CACHE_BBOX,
#         append_time=APPEND_TIMESTEPS,
#         append_vars=APPEND_VARIABLES,
#         cache_tag=cache_tag,
#     )
#     if ds is None:
#         print(f"  No cached data for {folder}, skipping.")
#         continue

#     # 2. Time logic per scenario (no MORFAC sensitivity here)
#     delta_time = ds.time.values - start_date
#     hydro_years = delta_time / np.timedelta64(365, 'D')

#     snapshot_matches = get_snapshot_matches_by_target_dates(ds.time.values, target_snapshot_dates)
#     if not snapshot_matches:
#         print(f"  No timesteps found for {folder}, skipping.")
#         ds.close()
#         continue
#     last_snapshot_idx = snapshot_matches[-1][1]

#     # --- DETRENDING: Store reference bed level if needed ---
#     if apply_detrending:
#         if use_mf50_reference and (reference_bed_MF50 is not None):
#             # Use MF50 time index 0 for all runs
#             print(f"Using MF50 reference bed (time index {reference_time_idx}) for detrending of {folder}...")
#             reference_bed = reference_bed_MF50
#         else:
#             # Default: per‑run reference at reference_time_idx
#             print(f"Storing reference bed level at time index {reference_time_idx} for {folder}...")
#             reference_bed = ds['mesh2d_mor_bl'].isel(time=reference_time_idx).values.copy()
            
#         # --- CHECK VARIABLES ---
#         if check_variables and i == 0:
#             check_available_variables_xarray(ds)
#             break

#     # Build KDTree for spatial queries (needed for new analyses)
#     if compare_max_depth or compare_channel_width:
#         face_x = ds.grid.face_coordinates[:, 0]
#         face_y = ds.grid.face_coordinates[:, 1]
#         from scipy.spatial import cKDTree
#         tree = cKDTree(np.vstack([face_x, face_y]).T)

#     for target_dt, ts_idx, actual_dt in snapshot_matches:
#         target_label = _date_to_label(target_dt)
#         actual_label = _date_to_label(actual_dt)
#         snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
#         snapshot_label = f"target={target_label} | actual={actual_label}"
#         print(f"Scenario: {folder_str:25} | {snapshot_label} | HydroYear={hydro_years[ts_idx]:.2f}")

#         comparison_results.setdefault(snapshot_key, {})
#         comparison_results[snapshot_key][folder_str] = {}
#         comparison_labels[snapshot_key] = target_label

#         face_x = ds.grid.face_coordinates[:, 0]
#         face_y = ds.grid.face_coordinates[:, 1]
#         width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])

#         # 4. WIDTH-AVERAGED BED LEVEL
#         if compare_width_averaged_bedlevel:
#             print(f"Computing Bed Level for {folder} ({snapshot_label})...")
#             var_name = "mesh2d_mor_bl"
#             dx = 1000
#             x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
#             x_centers = (x_bins[:-1] + x_bins[1:]) / 2

#             bedlev_data = ds[var_name].isel(time=ts_idx).values.copy()

#             # Apply detrending if enabled
#             if apply_detrending:
#                 bedlev_data = bedlev_data - reference_bed
#                 # For detrended data, don't use bed_threshold filter (data is centered around 0)
#                 # Only filter based on spatial domain
#                 valid_mask = width_mask
#             else:
#                 # For non-detrended data, use bed_threshold to exclude high land values
#                 valid_mask = (width_mask) & (bedlev_data < bed_threshold)

#             temp_means = []
#             for k in range(len(x_bins)-1):
#                 bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k+1])
#                 temp_means.append(np.mean(bedlev_data[bin_mask]) if np.any(bin_mask) else np.nan)

#             comparison_results[snapshot_key][folder_str]['BL'] = np.array(temp_means)
#             comparison_results[snapshot_key][folder_str]['x_centers'] = x_centers

#             if plot_width_averaged_bedlevel_individual:
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(x_centers/1000, temp_means, 'o-', color=scenario_color)
#                 plt.xlabel('Distance [km]')
#                 detrend_label = ' (Detrended)' if apply_detrending else ''
#                 plt.ylabel(f'Width-averaged Bed Level [m]{detrend_label}')
#                 plt.title(f'Width-averaged Bed Level: {scenario_label} ({snapshot_label})')
#                 plt.grid(True, alpha=0.3)
#                 plt.savefig(save_dir / f'width_averaged_bedlevel_map_{actual_label}_{folder_str}.png')
#                 if ts_idx == last_snapshot_idx:
#                     plt.savefig(save_dir / f'width_averaged_bedlevel_map_final_{folder_str}.png')
#                 plt.close()

#         # 5. MAXIMUM DEPTH ANALYSIS (95th percentile)
#         if compare_max_depth:
#             print(f"Computing Maximum Depth for {folder} ({snapshot_label})...")
#             var_name = "mesh2d_mor_bl"
#             dx = 1000
#             x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
#             x_centers = (x_bins[:-1] + x_bins[1:]) / 2

#             bedlev_data = ds[var_name].isel(time=ts_idx).values.copy()

#             # Apply detrending if enabled
#             if apply_detrending:
#                 bedlev_data = bedlev_data - reference_bed

#             # For depth calculation: convert bed level to depth
#             # Depth is positive downward (negative bed level = deep channel)
#             if use_absolute_depth:
#                 # Use absolute value to make all depths positive
#                 depths_field = np.abs(bedlev_data)
#             else:
#                 # Traditional: depth = -bed_level (negative values become positive)
#                 depths_field = -bedlev_data

#             # Apply thresholds
#             if apply_detrending:
#                 valid_mask = width_mask  # No bed_threshold when detrended
#             else:
#                 valid_mask = (width_mask) & (bedlev_data < bed_threshold)

#             max_depths = []
#             for k in range(len(x_bins)-1):
#                 bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k+1])
#                 if np.any(bin_mask):
#                     bin_depths = depths_field[bin_mask]
#                     valid_depths = bin_depths[~np.isnan(bin_depths)]
#                     if len(valid_depths) > 0:
#                         max_depth = np.percentile(valid_depths, depth_percentile)
#                         max_depths.append(max_depth)
#                     else:
#                         max_depths.append(np.nan)
#                 else:
#                     max_depths.append(np.nan)

#             comparison_results[snapshot_key][folder_str]['MaxDepth'] = np.array(max_depths)

#             if plot_max_depth_individual:
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(x_centers/1000, max_depths, 'o-', color=scenario_color)
#                 plt.xlabel('Distance [km]')
#                 depth_label = 'Absolute Depth' if use_absolute_depth else 'Depth'
#                 detrend_label = ' (Detrended)' if apply_detrending else ''
#                 plt.ylabel(f'{depth_percentile}th Percentile {depth_label} [m]{detrend_label}')
#                 plt.title(f'Maximum Channel Depth: {scenario_label} ({snapshot_label})')
#                 plt.grid(True, alpha=0.3)
#                 plt.savefig(save_dir / f'max_depth_map_{actual_label}_{folder_str}.png')
#                 if ts_idx == last_snapshot_idx:
#                     plt.savefig(save_dir / f'max_depth_map_final_{folder_str}.png')
#                 plt.close()

#         # 6. CHANNEL WIDTH ANALYSIS
#         if compare_channel_width:
#             print(f"Computing Channel Widths for {folder_str} ({snapshot_label})...")

#             max_widths = []
#             for x_coord in x_targets:
#                 distances, bed_profile = get_bed_profile_at_x(
#                     ds, tree, x_coord, y_range, ts_idx,
#                     reference_bed=reference_bed if apply_detrending else None,
#                     detrend=apply_detrending
#                 )

#                 # Filter out land values
#                 if apply_detrending:
#                     # For detrended data, use different threshold logic
#                     bed_profile[np.abs(bed_profile) > bed_threshold] = np.nan
#                 else:
#                     bed_profile[bed_profile > bed_threshold] = np.nan

#                 max_width = compute_max_channel_width(bed_profile, distances, safety_buffer)
#                 max_widths.append(max_width)

#             comparison_results[snapshot_key][folder_str]['ChannelWidth'] = np.array(max_widths)

#             if plot_channel_width_individual:
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(x_targets/1000, max_widths, 'o-', color=scenario_color)
#                 plt.xlabel('Distance [km]')
#                 plt.ylabel('Max Channel Width [m]')
#                 detrend_label = ' (Detrended)' if apply_detrending else ''
#                 plt.title(f'Maximum Channel Width: {scenario_label}{detrend_label} ({snapshot_label})')
#                 plt.grid(True, alpha=0.3)
#                 plt.savefig(save_dir / f'channel_width_map_{actual_label}_{folder_str}.png')
#                 if ts_idx == last_snapshot_idx:
#                     plt.savefig(save_dir / f'channel_width_map_final_{folder_str}.png')
#                 plt.close()

#         # 7. HYPSOMETRIC CURVE
#         if compare_hypsometric:
#             print(f"Computing Hypsometric Curve for {folder_str} ({snapshot_label})...")
#             bedlev_data = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

#             if apply_detrending:
#                 bedlev_data = bedlev_data - reference_bed
#                 valid_mask = width_mask
#             else:
#                 valid_mask = (width_mask) & (bedlev_data < bed_threshold)

#             elev_curve, area_curve, area_label = compute_hypsometric_curve(
#                 bedlev_data=bedlev_data,
#                 valid_mask=valid_mask,
#                 face_area=None,
#             )

#             comparison_results[snapshot_key][folder_str]['HypsoElevation'] = elev_curve
#             comparison_results[snapshot_key][folder_str]['HypsoArea'] = area_curve
#             comparison_results[snapshot_key][folder_str]['HypsoAreaLabel'] = area_label

#             if plot_hypsometric_individual and elev_curve.size > 0:
#                 plt.figure(figsize=(10, 6))
#                 plt.plot(area_curve, elev_curve, '-', color=scenario_color, linewidth=2)
#                 plt.xlabel(area_label)
#                 detrend_label = ' (Detrended)' if apply_detrending else ''
#                 plt.ylabel(f'Bed elevation [m]{detrend_label}')
#                 plt.title(f'Hypsometric Curve: {scenario_label} ({snapshot_label})')
#                 if not apply_detrending:
#                     plt.axhline(y=bed_threshold, color='red', linestyle='--', alpha=0.7)
#                 plt.grid(True, alpha=0.3)
#                 plt.savefig(save_dir / f'hypsometric_curve_{actual_label}_{folder_str}.png')
#                 if ts_idx == last_snapshot_idx:
#                     plt.savefig(save_dir / f'hypsometric_curve_final_{folder_str}.png')
#                 plt.close()

#     ds.close()

# # %% --- 7. FINAL COMPARISON PLOT ---
# print("\nGenerating Comparison Plot...")

# for snapshot_key, snapshot_results in comparison_results.items():
#     if not snapshot_results:
#         continue

#     # Count active plots
#     first_key = list(snapshot_results.keys())[0]
#     n_plots = sum([
#         compare_braiding_index and 'BI_tau' in snapshot_results[first_key],
#         compare_braiding_index and 'BI_depth' in snapshot_results[first_key],
#         compare_width_averaged_bedlevel,
#         compare_max_depth,
#         compare_channel_width
#     ])

#     if n_plots == 0:
#         print("No plots to generate!")
#         continue

#     fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
#     if n_plots == 1:
#         axes = [axes]

#     sorted_scenarios = sorted(snapshot_results.keys())
#     plot_idx = 0

#     # Plot 1: Shear Stress BI
#     if compare_braiding_index and 'BI_tau' in snapshot_results[first_key]:
#         for idx, scenario in enumerate(sorted_scenarios):
#             data = snapshot_results[scenario]
#             if 'BI_tau' in data:
#                 axes[plot_idx].plot(x_targets/1000, data['BI_tau'],
#                                    label=_scenario_legend_label(scenario), color=_scenario_color(scenario), marker='o', ms=4)
#         axes[plot_idx].set_title(f'BI ({var_tau}), fixed threshold: tau > {tau_threshold} N/m²')
#         axes[plot_idx].set_ylabel('braiding index')
#         axes[plot_idx].legend(loc='best')
#         axes[plot_idx].grid(True, alpha=0.2)
#         plot_idx += 1

#     # Plot 2: Water Depth BI
#     if compare_braiding_index and 'BI_depth' in snapshot_results[first_key]:
#         for idx, scenario in enumerate(sorted_scenarios):
#             data = snapshot_results[scenario]
#             if 'BI_depth' in data:
#                 axes[plot_idx].plot(x_targets/1000, data['BI_depth'],
#                                    label=_scenario_legend_label(scenario), color=_scenario_color(scenario), marker='s', ms=4, linestyle='--')
#         axes[plot_idx].set_title(f'BI ({var_depth}), relative threshold: {int(depth_threshold*100)}% above mean water depth')
#         axes[plot_idx].set_ylabel('braiding index')
#         axes[plot_idx].legend(loc='best')
#         axes[plot_idx].grid(True, alpha=0.2)
#         plot_idx += 1

#     # Plot 3: Bed Level
#     if compare_width_averaged_bedlevel:
#         for idx, scenario in enumerate(sorted_scenarios):
#             data = snapshot_results[scenario]
#             if 'BL' in data:
#                 axes[plot_idx].plot(data['x_centers']/1000, data['BL'],
#                                    color=_scenario_color(scenario), linewidth=2, label=_scenario_legend_label(scenario))
#         axes[plot_idx].set_title('width-averaged bed level')
#         axes[plot_idx].set_ylabel('bed level [m]')
#         axes[plot_idx].legend(loc='best')
#         axes[plot_idx].grid(True, alpha=0.2)
#         plot_idx += 1

#     # Plot 4: Maximum Depth
#     if compare_max_depth:
#         for idx, scenario in enumerate(sorted_scenarios):
#             data = snapshot_results[scenario]
#             if 'MaxDepth' in data:
#                 axes[plot_idx].plot(data['x_centers']/1000, data['MaxDepth'],
#                                    color=_scenario_color(scenario), linewidth=2, label=_scenario_legend_label(scenario), marker='o', ms=3)
#         axes[plot_idx].set_title(f'p{depth_percentile} channel depth')
#         axes[plot_idx].set_ylabel('depth [m]')
#         axes[plot_idx].legend(loc='best')
#         axes[plot_idx].grid(True, alpha=0.2)
#         plot_idx += 1

#     # Plot 5: Channel Width
#     if compare_channel_width:
#         for idx, scenario in enumerate(sorted_scenarios):
#             data = snapshot_results[scenario]
#             if 'ChannelWidth' in data:
#                 axes[plot_idx].plot(x_targets/1000, data['ChannelWidth'],
#                                    color=_scenario_color(scenario), linewidth=2, label=_scenario_legend_label(scenario), marker='s', ms=3)
#         axes[plot_idx].set_title(f'maximum channel width (threshold: mean depth - {int(safety_buffer*100)} cm)')
#         axes[plot_idx].set_ylabel('width [m]')
#         axes[plot_idx].legend(loc='best')
#         axes[plot_idx].grid(True, alpha=0.2)

#     axes[-1].set_xlabel('x-coordinate along estuary [km]')
#     fig.suptitle(f"Hydrodynamic snapshot around {comparison_labels.get(snapshot_key, snapshot_key)}", fontsize=12)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.97])

#     snapshot_date = comparison_labels.get(snapshot_key, snapshot_key)

#     if apply_detrending:
#         plt.savefig(summary_output_dir / f'overall_morphology_variability_comparison_detrended_{snapshot_date}.png', dpi=300)
#     else:
#         plt.savefig(summary_output_dir / f'overall_morphology_variability_comparison_{snapshot_date}.png', dpi=300)
#     plt.show()

#     print(f'Saved comparison plot at {summary_output_dir} for {snapshot_key}')

#     # Separate hypsometric comparison plot for this snapshot.
#     if compare_hypsometric:
#         fig_h, ax_h = plt.subplots(figsize=(10, 6))
#         has_hypso = False
#         area_labels = []

#         for scenario in sorted_scenarios:
#             data = snapshot_results[scenario]
#             if 'HypsoElevation' not in data or 'HypsoArea' not in data:
#                 continue
#             if data['HypsoElevation'].size == 0:
#                 continue

#             has_hypso = True
#             area_labels.append(data.get('HypsoAreaLabel', 'Cumulative area'))
#             ax_h.plot(
#                 data['HypsoArea'],
#                 data['HypsoElevation'],
#                 linewidth=2,
#                 color=_scenario_color(scenario),
#                 label=_scenario_legend_label(scenario),
#             )

#         if has_hypso:
#             x_label = area_labels[0] if len(set(area_labels)) == 1 else 'Cumulative area'
#             ax_h.set_xlabel(x_label)
#             ax_h.set_ylabel('Bed elevation [m]')
#             ax_h.set_title(f'Hypsometric curves around {comparison_labels.get(snapshot_key, snapshot_key)}')
#             if not apply_detrending:
#                 ax_h.axhline(y=bed_threshold, color='red', linestyle='--', alpha=0.7)
#             ax_h.grid(True, alpha=0.2)
#             ax_h.legend(loc='best')
#             fig_h.tight_layout()
#             snapshot_date = comparison_labels.get(snapshot_key, snapshot_key)
#             if apply_detrending:
#                 fig_h.savefig(summary_output_dir / f'hypsometric_comparison_detrended_{snapshot_date}.png', dpi=300)
#             else:
#                 fig_h.savefig(summary_output_dir / f'hypsometric_comparison_{snapshot_date}.png', dpi=300)
#             plt.show()
#             print(f'Saved hypsometric comparison plot at {summary_output_dir} for {snapshot_key}')
#         else:
#             plt.close(fig_h)

# print("\nAll processing complete.")