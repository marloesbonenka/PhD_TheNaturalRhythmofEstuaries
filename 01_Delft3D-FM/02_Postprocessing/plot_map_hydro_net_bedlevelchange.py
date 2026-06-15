"""Plot net erosion/deposition over each tidal cycle, with cycle boundaries
defined by low-water slack (minimum spatially-averaged water depth),
removing phase-aliasing from fixed 12h windows.
"""

# %% Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- CONFIGURATION ---

SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'peak_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
}

BED_LEVEL_VAR  = 'mesh2d_mor_bl'           # adjust if needed
WATER_DEPTH_VAR = 'mesh2d_waterdepth'

# Region to average water depth over for tidal phase detection
# Use tidal reach (lower estuary) where tidal signal is strongest
TIDE_DETECT_XLIM = (19000, 30000)
TIDE_DETECT_YLIM = (5000, 10000)

# Spatial binning for erosion/deposition profiles
X_BIN_WIDTH_M = 1000
ZOOM_XLIM     = (19000, 45000)
ZOOM_YLIM     = (5000, 10000)

# Expected tidal period in hours — used to set minimum distance between peaks
TIDAL_PERIOD_H = 12

# Cache settings
CACHE_BBOX       = [1, 1, 45000, 15000]
CACHE_TAG        = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

SCENARIO_COLORS = {
    'constant':  'dimgrey',
    'low_flow':  'steelblue',
    'peak_flow': 'firebrick',
    'mean_flow': 'darkorange',
}
SCENARIO_LABELS = {
    'constant':  'Constant flow',
    'low_flow':  'Low flow (R$_{peak}$=5, n=4)',
    'peak_flow': 'Peak flow (R$_{peak}$=5, n=4)',
    'mean_flow': 'Mean flow (R$_{peak}$=5, n=4)',
}

# %% --- HELPERS ---

def bin_along_x(face_x, face_y, values, x_bins, zoom_xlim, zoom_ylim):
    """Spatially average values into x-bins within the zoom window."""
    in_zoom = (
        (face_x >= zoom_xlim[0]) & (face_x <= zoom_xlim[1]) &
        (face_y >= zoom_ylim[0]) & (face_y <= zoom_ylim[1])
    )
    fx = face_x[in_zoom]
    v  = values[in_zoom]

    bin_centers, bin_means = [], []
    for i in range(len(x_bins) - 1):
        mask = (fx >= x_bins[i]) & (fx < x_bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append(0.5 * (x_bins[i] + x_bins[i + 1]))
            bin_means.append(np.nanmean(v[mask]))

    return np.array(bin_centers) / 1000, np.array(bin_means)   # x in km


def detect_low_water_indices(ds, face_x, face_y, water_depth_var,
                              tide_xlim, tide_ylim, tidal_period_h):
    """
    Find timestep indices of low-water slack by detecting minima in
    spatially-averaged water depth over the tidal detection region.

    Returns sorted array of timestep indices.
    """
    in_tide_region = (
        (face_x >= tide_xlim[0]) & (face_x <= tide_xlim[1]) &
        (face_y >= tide_ylim[0]) & (face_y <= tide_ylim[1])
    )

    # Mean water depth over tidal region at each timestep
    depth_da = ds[water_depth_var]   # (time, nFaces)
    depth_mean = depth_da.values[:, in_tide_region].mean(axis=1)   # (nTime,)

    # Minimum distance between minima: slightly less than one tidal period
    time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
    dt_h = (time_values[1] - time_values[0]) / np.timedelta64(1, 'h')
    min_distance = int(0.8 * tidal_period_h / dt_h)

    # Find minima (invert signal for find_peaks)
    minima_idx, props = find_peaks(-depth_mean, distance=min_distance, prominence=0.05)

    print(f"  Detected {len(minima_idx)} low-water moments at timesteps: {minima_idx.tolist()}")
    print(f"  Corresponding times: {[str(np.datetime_as_string(time_values[i], unit='s')) for i in minima_idx]}")

    return minima_idx, depth_mean


# %% --- MAIN ---

x_bins = np.arange(ZOOM_XLIM[0], ZOOM_XLIM[1] + X_BIN_WIDTH_M, X_BIN_WIDTH_M)

# Collect results per scenario: list of (x_km, dz_binned) per cycle
all_results  = {}   # label → list of (x_km, dz_mm)
all_lw_times = {}   # label → low-water time indices

# --- Diagnostic figure: water depth timeseries + detected minima ---
fig_diag, axes_diag = plt.subplots(
    len(SCENARIOS), 1,
    figsize=(10, 2.5 * len(SCENARIOS)),
    sharex=True,
    constrained_layout=True,
)
fig_diag.suptitle('Low-water detection — spatially averaged water depth (tidal region)', fontsize=10)

for ax_d, (label, folder_path) in zip(axes_diag, SCENARIOS.items()):

    print(f"\n{'='*60}")
    print(f"Scenario: {label}")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=None,
        variability_map=None,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=[BED_LEVEL_VAR, WATER_DEPTH_VAR],
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds is None:
        print(f"  [SKIP] No data for {folder_name}")
        continue

    try:
        missing = [v for v in [BED_LEVEL_VAR, WATER_DEPTH_VAR] if v not in ds]
        if missing:
            print(f"  [SKIP] Missing variables: {missing}")
            print(f"  Available: {list(ds.data_vars)}")
            continue

        time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
        time_hours  = (time_values - time_values[0]) / np.timedelta64(1, 'h')
        face_x = ds.grid.face_coordinates[:, 0]
        face_y = ds.grid.face_coordinates[:, 1]

        # --- Detect low-water moments ---
        lw_indices, depth_mean = detect_low_water_indices(
            ds, face_x, face_y, WATER_DEPTH_VAR,
            TIDE_DETECT_XLIM, TIDE_DETECT_YLIM, TIDAL_PERIOD_H,
        )

        # --- Diagnostic plot ---
        ax_d.plot(time_hours, depth_mean, color=SCENARIO_COLORS[label], linewidth=1.2)
        ax_d.scatter(
            time_hours[lw_indices], depth_mean[lw_indices],
            color='black', zorder=5, s=30, label='Low water',
        )
        ax_d.set_ylabel('Mean depth [m]', fontsize=8)
        ax_d.set_title(SCENARIO_LABELS[label], fontsize=9)
        ax_d.grid(True, alpha=0.3)
        ax_d.legend(fontsize=7)

        if len(lw_indices) < 2:
            print(f"  [SKIP] Need at least 2 low-water moments to define a cycle.")
            continue

        # --- Compute net bed level change between consecutive low-water moments ---
        cycles = list(zip(lw_indices[:-1], lw_indices[1:]))
        all_results[label]  = []
        all_lw_times[label] = lw_indices

        for i_start, i_end in cycles:
            bed_start = ds[BED_LEVEL_VAR].isel(time=i_start).values
            bed_end   = ds[BED_LEVEL_VAR].isel(time=i_end).values

            if bed_start.ndim > 1:
                bed_start = bed_start.sum(axis=0)
                bed_end   = bed_end.sum(axis=0)

            delta_bed = bed_end - bed_start
            x_km, dz_binned = bin_along_x(face_x, face_y, delta_bed, x_bins, ZOOM_XLIM, ZOOM_YLIM)
            all_results[label].append((x_km, dz_binned))

    finally:
        ds.close()

axes_diag[-1].set_xlabel('Time [h]', fontsize=9)
diag_path = list(SCENARIOS.values())[0].parent / 'output_plots' / 'net_erosion_deposition' / 'low_water_detection.png'
diag_path.parent.mkdir(parents=True, exist_ok=True)
fig_diag.savefig(diag_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nDiagnostic saved: {diag_path}")

# %% --- RESULTS FIGURE ---
# Determine max number of cycles detected across scenarios
max_cycles = max((len(v) for v in all_results.values()), default=0)
if max_cycles == 0:
    print("No cycles detected — check BED_LEVEL_VAR and WATER_DEPTH_VAR names.")
else:
    fig, axes = plt.subplots(
        1, max_cycles,
        figsize=(4 * max_cycles, 4),
        sharey=True,
        constrained_layout=True,
    )
    if max_cycles == 1:
        axes = [axes]

    fig.suptitle(
        'Net erosion/deposition per tidal cycle (low water → low water)\nDetailed-hydro runs — Q500',
        fontsize=11,
    )

    for label, cycles_data in all_results.items():
        lw_idx = all_lw_times[label]
        for c_idx, (x_km, dz_mm) in enumerate(cycles_data):
            ax = axes[c_idx]
            ax.plot(
                x_km, dz_mm * 1000,
                color=SCENARIO_COLORS[label],
                label=SCENARIO_LABELS[label],
                linewidth=1.5,
                alpha=0.85,
            )
            ax.set_title(f'Cycle {c_idx + 1}\n(LW{c_idx+1} → LW{c_idx+2})', fontsize=9)
            ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
            ax.set_xlabel('Distance along estuary [km]')
            if c_idx == 0:
                ax.set_ylabel('Net bed level change [mm/cycle]')
            ax.grid(True, alpha=0.3)

    # Deduplicated legend
    handles, labels_leg = axes[-1].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_leg):
        if l not in seen:
            seen[l] = h
    fig.legend(
        seen.values(), seen.keys(),
        loc='lower center', ncol=len(SCENARIOS),
        fontsize=8, bbox_to_anchor=(0.5, -0.12), frameon=False,
    )

    out_path = diag_path.parent / 'net_erosion_deposition_low_water_cycles.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Results saved: {out_path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
#%%