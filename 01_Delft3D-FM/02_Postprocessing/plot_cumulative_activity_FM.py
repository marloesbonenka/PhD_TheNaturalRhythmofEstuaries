"""Cumulative activity plot for Delft3D-FM output."""
#%%
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dfm_tools as dfmt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import get_mf_number
from FUNCTIONS.F_cache import *

#%% --- SETTINGS & PATHS ---
scenarios_morfac = True
scenarios_discharge = False
scenarios_variability = False

var_name = "mesh2d_mor_bl"

# Estuary bounds (same as other FM scripts)
x_targets = np.arange(20000, 44001, 1000)
y_range = (5000, 10000)
dx = int(np.diff(x_targets).min())

# Spatial binning for width-averaged profiles
x_bin_size = dx  # meters
bed_threshold = 6  # exclude land (higher than 6 m)

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
compute = False
show_plots = True
base_directory = r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_FourRiverBoundaries" #r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\02_seasonal\Tmorph_50years"
model_folders = ['02_seasonal_Qr500_T2m_FourRiverBoundaries']
run_startdate = None  # None -> use first timestamp in dataset

# For MORFAC: Get all subdirectories starting with 'MF'

# if scenarios_morfac:
#     base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\02_seasonal\Tmorph_50years"
#     model_folders = [
#         f for f in os.listdir(base_directory)
#         if f.startswith("MF") and os.path.isdir(os.path.join(base_directory, f))
#     ]
#     run_startdate = "2025-01-01"

# # For discharge:
# if scenarios_discharge:
#     base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_OneRiverBoundary"
#     model_folders = [
#         f for f in os.listdir(base_directory)
#         if os.path.isdir(os.path.join(base_directory, f)) and f.startswith("01_")
#     ]
#     run_startdate = "2001-01-01"

# # For variability:
# if scenarios_variability:
#     base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_FourRiverBoundaries"
#     model_folders = [
#         f for f in os.listdir(base_directory)
#         if os.path.isdir(os.path.join(base_directory, f)) and (f.startswith("02_") or f.startswith("03_"))
#     ]
#     run_startdate = "2024-01-01"

# print(f"Found {len(model_folders)} folders to process.")

base_dir = Path(base_directory)
output_dir = base_dir / "output_plots_cumulative_activity"
output_dir.mkdir(parents=True, exist_ok=True)
cache_path = output_dir / "cached_results.pkl"

results = {}
run_names = {}

cache_settings = {
    'base_directory': str(base_directory),
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

if (not compute) and cache_path.exists():
    cached = load_cache(cache_path)
    meta = cached.get('metadata', {})
    cached_settings = meta.get('settings', {})
    if cached_settings == cache_settings:
        results = cached.get('results', {})
        run_names = meta.get('run_names', {})
        print(f"Loaded cached results from: {cache_path}")
    else:
        print("Cache settings differ from current settings; recomputing.")
        compute = True
else:
    if not compute:
        print(f"Cache not found, computing results: {cache_path}")

#%% --- LOOP THROUGH RUNS ---
if compute:
    dataset_cache = DatasetCache()
    try:
        for folder in model_folders:
            model_location = os.path.join(base_directory, folder)
            file_pattern = os.path.join(model_location, "output", "*_map.nc")
            print(f"\nProcessing: {folder}")

            try:
                ds = dataset_cache.get_partitioned(file_pattern)
                if var_name not in ds:
                    print(f"Skipping {folder}: Variable {var_name} not found.")
                    continue

            # Coordinates (faces)
            face_x = ds["mesh2d_face_x"].values
            face_y = ds["mesh2d_face_y"].values

            # Optional detrending (reference bed)
            reference_bed = None
            if apply_detrending and 'time' in ds[var_name].dims:
                reference_bed = ds[var_name].isel(time=reference_time_idx).values.copy()

            # Masks and bins (match width-averaged bed level logic)
            width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])
            x_bins = np.arange(x_targets[0], x_targets[-1] + x_bin_size, x_bin_size)
            x_centers = (x_bins[:-1] + x_bins[1:]) / 2
            bin_index = np.digitize(face_x, x_bins) - 1
            in_range = (bin_index >= 0) & (bin_index < len(x_centers))
            base_mask = width_mask & in_range

            # Time selection
            n_time = ds[var_name].sizes.get("time", 1)
            start_idx = max(0, slice_start)
            end_idx = n_time if slice_end is None else min(slice_end, n_time)
            time_indices = np.arange(start_idx, end_idx)

            # Pre-allocate width-averaged bed level array
            width_avg_bedlev = np.full((len(time_indices), len(x_centers)), np.nan)

            for t_i, time_idx in enumerate(time_indices):
                bedlev_t = ds[var_name].isel(time=time_idx).values
                if apply_detrending and (reference_bed is not None):
                    bedlev_t = bedlev_t - reference_bed
                    valid_mask = base_mask
                else:
                    valid_mask = base_mask & (bedlev_t < bed_threshold)

                # Bin-wise mean
                for bi in range(len(x_centers)):
                    bin_mask = valid_mask & (bin_index == bi)
                    if np.any(bin_mask):
                        width_avg_bedlev[t_i, bi] = np.nanmean(bedlev_t[bin_mask])

            # Differences and cumulative activity
            differences = np.diff(width_avg_bedlev, axis=0)
            zeros_row = np.zeros((1, width_avg_bedlev.shape[1]))
            abs_differences = np.abs(differences)
            abs_differences_with_prepended = np.vstack([zeros_row, abs_differences])
            cumulative_activity = np.cumsum(abs_differences_with_prepended, axis=0)

            # Morphological time (years)
            if "time" in ds:
                times = pd.to_datetime(ds["time"].values)
                if run_startdate is None:
                    start_timestamp = pd.to_datetime(ds["time"].values[0])
                else:
                    start_timestamp = pd.Timestamp(run_startdate)
                hydro_years = np.array([(t - start_timestamp).days / 365.25 for t in times])
                hydro_years = hydro_years[start_idx:end_idx]

                if "morfac" in ds:
                    morfac_values = ds["morfac"].values[start_idx:end_idx]
                elif scenarios_morfac:
                    morfac_values = np.full_like(hydro_years, get_mf_number(folder), dtype=float)
                else:
                    morfac_values = np.ones_like(hydro_years, dtype=float)

                morph_years = hydro_years * morfac_values
            else:
                morph_years = np.arange(cumulative_activity.shape[0])

            results[folder] = {
                'cumulative_activity': cumulative_activity,
                'x_centers': x_centers,
                'morph_years': morph_years,
            }
            run_names[folder] = folder

            except Exception as e:
                print(f"Error processing {folder}: {e}")
    finally:
        dataset_cache.close_all()

    save_cache(cache_path, {
        'results': results,
        'metadata': {
            'run_names': run_names,
            'settings': cache_settings,
        },
    })
    print(f"Saved cache: {cache_path}")

# --- plotting from cache/results ---
for folder in model_folders:
    if folder not in results:
        continue

    data = results[folder]
    cumulative_activity = data['cumulative_activity']
    x_centers = data['x_centers']
    morph_years = data['morph_years']

    y_min, y_max = morph_years[0], morph_years[-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    extent = [x_centers.min() / 1000, x_centers.max() / 1000, y_min, y_max]
    vmax = np.nanpercentile(cumulative_activity, 98) if np.any(np.isfinite(cumulative_activity)) else 1

    im = ax.imshow(
        cumulative_activity,
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

    final_cumulative = cumulative_activity[-1, :]
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
