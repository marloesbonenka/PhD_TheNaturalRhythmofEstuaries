"""Cumulative activity plot for Delft3D-FM output."""
#%%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import get_mf_number
from FUNCTIONS.F_cache import DatasetCache, load_results_cache, save_results_cache
from FUNCTIONS.F_morphological_activity import cumulative_activity, morph_years_from_datetimes

#%% --- SETTINGS & PATHS ---

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
    SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']  # e.g., ['1'] for baserun only, None for all
    use_folder_morfac = False
    default_morfac = 100  # MORFAC used in variability runs

elif ANALYSIS_MODE == "morfac":
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
    config = r'TestingBoundaries_and_SensitivityAnalyses\Test_MORFAC\02_seasonal\Tmorph_50years'
    use_folder_morfac = True  # Extract MORFAC from folder name (MF1, MF2, etc.)
    default_morfac = 1.0

timed_out_dir = base_directory / config / "timed-out"

var_name = "mesh2d_mor_bl"

# Estuary bounds (same as other FM scripts)
x_targets = np.arange(20000, 44001, 1000)
y_range = (5000, 10000)
dx = int(np.diff(x_targets).min())

# Spatial binning for width-averaged profiles
x_bin_size = dx  # meters
bed_threshold = 8  # exclude land (higher than 6 m)

# Time slicing
slice_start = 0
slice_end = None  # None -> use full length

# Plot settings
save_figure = True
fontcolor = "black"
fontsize_labels = 12
fontsize_titles = 14
fontsize_axes = 10

# Detrending (matches width-averaged bed level logic)
apply_detrending = False
reference_time_idx = 0

# Cache + plotting controls
force_recompute = False
show_plots = True
run_startdate = None  # e.g. "2025-01-01", or None to use first timestamp

# Output
output_dirname = "output_plots_cumulative_activity"

# =============================================================================
# Search & Sort Folders
# =============================================================================

base_path = base_directory / config
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")

if ANALYSIS_MODE == "variability":
    model_folders = [f.name for f in base_path.iterdir()
                     if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
    if SCENARIOS_TO_PROCESS:
        model_folders = [f for f in model_folders if f.split('_')[0] in SCENARIOS_TO_PROCESS]
    model_folders.sort(key=lambda x: int(x.split('_')[0]))
elif ANALYSIS_MODE == "morfac":
    model_folders = [f.name for f in base_path.iterdir()
                     if f.is_dir() and f.name.startswith('MF')]
    model_folders.sort(key=get_mf_number)

print(f"Found {len(model_folders)} run folders in: {base_path}")

output_dir = base_path / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)
cache_path = output_dir / "cached_results_widthavg.pkl"

results = {}
run_names = {}

cache_settings = {
    'analysis_mode': ANALYSIS_MODE,
    'base_directory': str(base_directory),
    'config': config,
    'model_folders': list(model_folders),
    'x_targets_start': float(x_targets[0]),
    'x_targets_end': float(x_targets[-1]),
    'x_targets_step': float(x_targets[1] - x_targets[0]) if len(x_targets) > 1 else float(x_bin_size),
    'y_range': tuple(y_range),
    'x_bin_size': float(x_bin_size),
    'bed_threshold': float(bed_threshold),
    'apply_detrending': bool(apply_detrending),
    'reference_time_idx': int(reference_time_idx),
    'slice_start': int(slice_start),
    'slice_end': None if slice_end is None else int(slice_end),
    'var_name': var_name,
    'run_startdate': None if run_startdate is None else str(run_startdate),
}

if not force_recompute:
    loaded_results, loaded_meta = load_results_cache(cache_path, cache_settings)
    if loaded_results is not None:
        results = loaded_results
        run_names = loaded_meta.get('run_names', {})
        print(f"Loaded cached results from: {cache_path}")
        print(f"  Cached folders: {list(results.keys())}")
        print(f"  Current folders: {model_folders}")
        # Warn if keys don't match
        matched = [f for f in model_folders if f in results]
        if not matched:
            print(f"  WARNING: No folder names match between cache and current model_folders!")
            print(f"  Cache will be recomputed.")
            results = {}
            run_names = {}
    else:
        print(f"Cache not found or settings differ, computing results...")

#%% --- LOOP THROUGH RUNS ---
if force_recompute or not results:
    dataset_cache = DatasetCache()


    for folder in model_folders:
        model_location = base_path / folder
        print(f"\nProcessing: {folder}")

        # --- RESTART STITCHING (timed-out + restart, same as cross-section scripts) ---
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

        # --- LOAD & STITCH all run parts ---
        all_bedlev = []   # list of 2-D arrays  (n_time_part, n_faces)
        all_times = []    # list of datetime arrays

        for run_path in all_run_paths:
            file_pattern = str(run_path / "output" / "*_map.nc")
            
            ds = dataset_cache.get_partitioned(
                file_pattern,
                variables=['mesh2d_mor_bl', 'mesh2d_face_x', 'mesh2d_face_y'],
                chunks={'time': 200},
            )
            
            if var_name not in ds:
                print(f"  Skipping {run_path.name}: Variable {var_name} not found.")
                continue

            part_times = pd.to_datetime(ds["time"].values)
            # Load bedlevel lazily per chunk to avoid huge memory spike
            bedlev_da = ds[var_name]
            print(f"  Loading {run_path.name}: {bedlev_da.sizes['time']} timesteps...")
            all_times.append(part_times)
            all_bedlev.append(bedlev_da.values)  # (n_time, n_faces)
            # Face coords are the same for all parts
            face_x = ds["mesh2d_face_x"].values
            face_y = ds["mesh2d_face_y"].values

        if not all_bedlev:
            print(f"  No valid data for {folder}, skipping.")
            continue

        # Concatenate parts along time
        if len(all_bedlev) > 1:
            bedlev_full = np.concatenate(all_bedlev, axis=0)
            times_full = np.concatenate(all_times)
        else:
            bedlev_full = all_bedlev[0]
            times_full = all_times[0]

        # Optional detrending (reference bed)
        reference_bed = None
        if apply_detrending:
            reference_bed = bedlev_full[reference_time_idx].copy()

        # Masks and bins (match width-averaged bed level logic)
        width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
        x_bins = np.arange(x_targets[0], x_targets[-1] + x_bin_size, x_bin_size)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        bin_index = np.digitize(face_x, x_bins) - 1
        in_range = (bin_index >= 0) & (bin_index < len(x_centers))
        base_mask = width_mask & in_range

        # Time selection
        n_time = bedlev_full.shape[0]
        start_idx = max(0, slice_start)
        end_idx = n_time if slice_end is None else min(slice_end, n_time)
        bedlev_slice = bedlev_full[start_idx:end_idx]  # (n_sel, n_faces)
        n_sel = bedlev_slice.shape[0]

        # Vectorized width-averaged bed level using np.bincount
        n_bins = len(x_centers)
        width_avg_bedlev = np.full((n_sel, n_bins), np.nan)

        # Pre-compute bin labels for valid faces
        valid_face_idx = np.where(base_mask)[0]
        valid_bins = bin_index[valid_face_idx]

        print(f"  Computing width-averaged bed level ({n_sel} timesteps, {n_bins} bins)...")
        for t_i in range(n_sel):
            bedlev_t = bedlev_slice[t_i]
            if apply_detrending and (reference_bed is not None):
                bedlev_t = bedlev_t - reference_bed
                face_vals = bedlev_t[valid_face_idx]
                face_bins = valid_bins
            else:
                # Additional threshold mask
                threshold_ok = bedlev_t[valid_face_idx] < bed_threshold
                face_vals = bedlev_t[valid_face_idx][threshold_ok]
                face_bins = valid_bins[threshold_ok]

            if len(face_vals) == 0:
                continue

            # Vectorised bin mean via bincount (no inner Python loop)
            bin_sums = np.bincount(face_bins, weights=face_vals, minlength=n_bins)
            bin_counts = np.bincount(face_bins, minlength=n_bins)
            valid = bin_counts > 0
            width_avg_bedlev[t_i, valid] = bin_sums[valid] / bin_counts[valid]

            if (t_i + 1) % 200 == 0 or t_i == n_sel - 1:
                print(f"    timestep {t_i + 1}/{n_sel}")

        # Cumulative activity (shared function)
        cum_act = cumulative_activity(width_avg_bedlev)

        # Morphological time (shared function)
        times = pd.to_datetime(times_full[start_idx:end_idx])
        morph_years = morph_years_from_datetimes(times, startdate=run_startdate, morfac=morfac)

        results[folder] = {
            'cumulative_activity': cum_act,
            'x_centers': x_centers,
            'morph_years': morph_years,
        }
        run_names[folder] = folder
        print(f"  Done: {bedlev_full.shape[0]} timesteps, MORFAC={morfac}")

        # Save cache after each run
        save_results_cache(
            cache_path,
            results,
            settings=cache_settings,
            metadata={'run_names': run_names},
        )
        print(f"Saved cache after {folder}: {cache_path}")

    dataset_cache.close_all()

#--- plotting from cache/results ---
for folder in model_folders:
    if folder not in results:
        continue

    data = results[folder]
    cum_act = data['cumulative_activity']
    x_centers = data['x_centers']
    morph_years = data['morph_years']

    y_min, y_max = morph_years[0], morph_years[-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    extent = [x_centers.min() / 1000, x_centers.max() / 1000, y_min, y_max]
    vmax = np.nanpercentile(cum_act, 98) if np.any(np.isfinite(cum_act)) else 1

    im = ax.imshow(
        cum_act,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cbar.ax.tick_params(labelsize=fontsize_axes, colors=fontcolor)
    cbar.set_label(r"$\Sigma |\Delta h|$ [m]", fontsize=fontsize_labels, color=fontcolor)
    cbar.outline.set_edgecolor(fontcolor)

    ax.set_xlabel("Along-estuary distance [km]", fontsize=fontsize_labels, color=fontcolor)
    ax.set_ylabel("Morphological time [years]", fontsize=fontsize_labels, color=fontcolor)
    ax.set_title(f"{folder}: cumulative bed level change along estuary", fontsize=fontsize_titles, color=fontcolor)

    final_cumulative = cum_act[-1, :]
    x_maxvalue_estuary = x_centers[np.nanargmax(final_cumulative)] / 1000
    x_minvalue_estuary = x_centers[np.nanargmin(final_cumulative)] / 1000
    max_val, min_val = np.nanmax(final_cumulative), np.nanmin(final_cumulative)

    textstr = (
        rf"$(\Sigma |\Delta h|)_{{\mathrm{{max}}}}$ = {max_val:.2f} m at {x_maxvalue_estuary:.1f} km"
        "\n"
        rf"$(\Sigma |\Delta h|)_{{\mathrm{{min}}}}$ = {min_val:.2f} m at {x_minvalue_estuary:.1f} km"
    )

    ax.text(
        0.02, 0.03, textstr,
        transform=ax.transAxes,
        fontsize=fontsize_labels,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.8),
    )

    ax.tick_params(axis="both", which="major", labelsize=fontsize_axes, colors=fontcolor)
    for spine in ax.spines.values():
        spine.set_edgecolor(fontcolor)

    plt.tight_layout()
    if save_figure:
        save_name = f"cumulative_bed_change_heatmap_{folder}.png"
        plt.savefig(output_dir / save_name, dpi=300, bbox_inches="tight", transparent=True)
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

# %%
