"""Estuary slope sensitivity: scatter of R_peak vs n_peaks, coloured by slope.

For each scenario the mean bed level profile (km 20–45) is averaged across
runs and the overall along-estuary slope is extracted from a linear fit.
The result is plotted as a 2-D scatter where

  x-axis  = n_peaks  (peak frequency)
  y-axis  = R_peak   (peak amplitude)
  colour  = slope [m km⁻¹]  (negative = deepening seaward)

One plot is made per snapshot date.
"""

#%% IMPORTS
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import (
    _date_to_filename_tag,
    _date_to_label,
    _scenario_label,
    _parse_pm_n,
    get_variability_map,
    find_variability_model_folders,
    get_target_snapshot_dates,
    get_snapshot_matches_by_target_dates,
    sort_scenario_keys,
    group_snapshot_by_scenario,
    stack_metric_arrays,
)

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths


#%% --- CONFIGURATION ---
DISCHARGE = 500
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config = f'Model_Output/Q{DISCHARGE}'

COMPUTE_MEAN = True           # True → width-averaged mean bed level
CHANNEL_INIT_THRESHOLD = 2.2  # defines frozen t=0 channel footprint

start_date = np.datetime64('2025-01-01')
x_targets  = np.arange(20000, 44001, 1000)
y_range    = (5000, 10000)

CACHE_BBOX       = [1, 1, 45000, 15000]
CACHE_TAG        = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

SNAPSHOT_TARGET_DATES = None
SNAPSHOT_DATE_RANGE   = (np.datetime64('2025-01-01'), np.datetime64('2031-12-31'))
SNAPSHOT_COUNT        = 6

# km range used for the linear slope fit
SLOPE_KM_MIN = 20
SLOPE_KM_MAX = 45


#%% --- SCENARIO LABELS ---
SCENARIO_LABELS = {
    '1':  'pm1_n0 (constant)',
    '2':  'pm2_n1',
    '3':  'pm3_n5',
    '4':  'pm3_n1',
    '5':  'pm5_n1',
    '6':  'pm4_n3',
    '7':  'pm3_n4',
    '8':  'pm2_n6',
    '9':  'pm5_n3',
    '10': 'pm3_n3',
    '11': 'pm2_n3',
    '12': 'pm5_n4',
    '13': 'pm4_n4',
    '14': 'pm2_n4',
}

# --- Font sizes ---
FONTSIZE_TITLE  = 18
FONTSIZE_LABELS = FONTSIZE_TITLE - 4
FONTSIZE_TICKS  = FONTSIZE_LABELS - 2

plt.rcParams.update(plt.rcParamsDefault)


#%% --- SEARCH FOLDERS ---
base_path = base_directory / config
VARIABILITY_MAP = get_variability_map(DISCHARGE)

model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=None,
    analyze_noisy=False,
)

assessment_dir = base_path / 'cached_data'
assessment_dir.mkdir(parents=True, exist_ok=True)
timed_out_dir  = base_path / 'timed-out'

output_dir = base_path / 'output_plots' / 'plots_pm_n_sensitivity'
output_dir.mkdir(parents=True, exist_ok=True)


#%% --- LOAD DATA ---
comparison_results = {}
comparison_labels  = {}
channel_masks      = {}  # {folder_str: {bin_idx: boolean array}}

target_snapshot_dates = get_target_snapshot_dates(
    count=SNAPSHOT_COUNT,
    explicit_dates=SNAPSHOT_TARGET_DATES,
    date_range=SNAPSHOT_DATE_RANGE,
)

print("\nTarget snapshot dates:")
for dt in target_snapshot_dates:
    print(f"  - {_date_to_label(dt)}")

for folder in model_folders:
    folder_str = folder.name
    print(f"\nProcessing: {folder_str}")

    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder.name,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [base_path / folder]

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
        print(f"  No cached data for {folder_str}, skipping.")
        continue

    snapshot_matches = get_snapshot_matches_by_target_dates(ds.time.values, target_snapshot_dates)
    if not snapshot_matches:
        print(f"  No timesteps found for {folder_str}, skipping.")
        ds.close()
        continue

    face_x, face_y = _get_face_coords(ds)
    width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])

    dx    = 1000
    x_bins    = np.arange(x_targets[0], x_targets[-1] + dx, dx)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2

    # Build frozen t=0 channel mask once per folder
    if folder_str not in channel_masks:
        _init_bl   = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
        _valid_init = width_mask & (_init_bl < CHANNEL_INIT_THRESHOLD)
        channel_masks[folder_str] = {}
        for _k in range(len(x_bins) - 1):
            _bm = _valid_init & (face_x >= x_bins[_k]) & (face_x < x_bins[_k + 1])
            channel_masks[folder_str][_k] = _bm
        print(f"  Channel mask computed from t=0.")

    for target_dt, ts_idx, actual_dt in snapshot_matches:
        snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
        comparison_results.setdefault(snapshot_key, {})
        comparison_labels[snapshot_key] = _date_to_label(target_dt)

        bedlev_data = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

        bin_metrics = []
        for k in range(len(x_bins) - 1):
            bin_mask  = channel_masks[folder_str][k]
            if np.any(bin_mask):
                vals = bedlev_data[bin_mask]
                vals = vals[~np.isnan(vals)]
                val  = np.mean(vals) if len(vals) > 0 else np.nan
            else:
                val = np.nan
            bin_metrics.append(val)

        comparison_results[snapshot_key][folder_str] = {
            'BL':       np.array(bin_metrics),
            'x_centers': x_centers,
        }
        print(f"  Snapshot {_date_to_label(target_dt)}: OK")

    ds.close()


#%% --- COMPUTE SLOPES & PLOT ---
_metric_desc = 'mean'

for snapshot_key, snapshot_results in comparison_results.items():
    if not snapshot_results:
        continue

    snap_label    = comparison_labels.get(snapshot_key, snapshot_key)
    scenario_groups = group_snapshot_by_scenario(snapshot_results)
    all_scen_keys   = sort_scenario_keys(scenario_groups.keys())

    # --- collect (pm, n, slope) per scenario ---
    records = []   # list of (pm, n, slope)

    for scen_key in all_scen_keys:
        label = _scenario_label(scen_key, SCENARIO_LABELS)
        pm, n = _parse_pm_n(label)
        if pm is None:
            continue

        # Mean bed-level profile averaged across runs in this scenario
        y_stack = stack_metric_arrays(scenario_groups[scen_key], 'BL')
        if y_stack is None:
            continue
        y_mean = np.nanmean(y_stack, axis=0)

        x_km_data = next(
            (d['x_centers'] / 1000 for _, d in scenario_groups[scen_key] if 'x_centers' in d),
            x_targets / 1000,
        )

        # Restrict to slope km range
        mask_slope = (x_km_data >= SLOPE_KM_MIN) & (x_km_data <= SLOPE_KM_MAX)
        x_fit = x_km_data[mask_slope]
        y_fit = y_mean[mask_slope]

        valid = ~np.isnan(y_fit)
        if valid.sum() < 2:
            continue

        # Linear fit: slope in m km⁻¹
        coeffs = np.polyfit(x_fit[valid], y_fit[valid], 1)
        slope  = coeffs[0]   # m km⁻¹

        records.append((pm, n, slope))

    if not records:
        print(f"  No records for snapshot {snap_label}, skipping.")
        continue

    pm_arr    = np.array([r[0] for r in records])
    n_arr     = np.array([r[1] for r in records])
    slope_arr = np.array([r[2] for r in records])

    # Separate constant scenario (n == 0) for annotation
    const_mask  = (n_arr == 0)
    var_mask    = ~const_mask

    # --- colour mapping for slope ---
    # Sequential colormap spanning the actual slope range so small differences
    # between scenarios are visible even when all slopes have the same sign.
    slope_var = slope_arr[var_mask]
    vmin_c    = np.nanmin(slope_var) if slope_var.size > 0 else 0.0
    vmax_c    = np.nanmax(slope_var) if slope_var.size > 0 else 1.0
    # Add a small symmetric margin so extreme points are not clipped
    _margin   = max((vmax_c - vmin_c) * 0.05, 1e-6)
    norm      = mcolors.Normalize(vmin=vmin_c - _margin, vmax=vmax_c + _margin)
    cmap      = cm.plasma

    # --- figure ---
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sc = ax.scatter(
        n_arr[var_mask],
        pm_arr[var_mask],
        c=slope_var,
        cmap=cmap,
        norm=norm,
        s=120,
        edgecolors='k',
        linewidths=0.6,
        zorder=3,
    )

    # Annotate each point with its slope value
    for pm_v, n_v, sl in zip(pm_arr[var_mask], n_arr[var_mask], slope_var):
        ax.annotate(
            f'{sl:+.3f}',
            xy=(n_v, pm_v),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=FONTSIZE_TICKS - 2,
            color='0.2',
        )

    # Constant scenario marker (n=0 not on the grid, annotate separately)
    if const_mask.any():
        ax.axhline(
            pm_arr[const_mask][0],
            color='0.55', linestyle='--', linewidth=1.0, alpha=0.5, zorder=1,
        )
        ax.annotate(
            f"constant\nslope = {slope_arr[const_mask][0]:+.3f} m km⁻¹",
            xy=(0.02, 0.96), xycoords='axes fraction',
            fontsize=FONTSIZE_TICKS - 2, color='0.4',
            va='top',
        )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label('estuary slope [m km⁻¹]  (km 20–45)', fontsize=FONTSIZE_TICKS)
    cbar.ax.tick_params(labelsize=FONTSIZE_TICKS - 2)
    # Add "flatter" / "steeper" labels at the bottom and top of the colorbar
    cbar.ax.text(
        0.5, -0.02, 'flatter',
        ha='center', va='top', transform=cbar.ax.transAxes,
        fontsize=FONTSIZE_TICKS - 2, color='0.3', style='italic',
    )
    cbar.ax.text(
        0.5, 1.02, 'steeper',
        ha='center', va='bottom', transform=cbar.ax.transAxes,
        fontsize=FONTSIZE_TICKS - 2, color='0.3', style='italic',
    )

    ax.set_xlabel(r'$n_{\mathrm{peaks}}$  (peak frequency)', fontsize=FONTSIZE_LABELS)
    ax.set_ylabel(r'$R_{\mathrm{peak}}$  (peak amplitude)', fontsize=FONTSIZE_LABELS)
    ax.tick_params(labelsize=FONTSIZE_TICKS)

    # Integer ticks matching the scenario grid
    unique_n  = sorted(set(n_arr[var_mask].astype(int)))
    unique_pm = sorted(set(pm_arr[var_mask].astype(int)))
    ax.set_xticks(unique_n)
    ax.set_yticks(unique_pm)

    ax.grid(True, alpha=0.25, linewidth=0.5)

    ax.set_title(
        f'Estuary slope vs discharge variability parameters\n'
        f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
        fontsize=FONTSIZE_TITLE - 2, fontweight='bold', pad=8,
    )

    fig.tight_layout()

    fname = f'slope_scatter_{snap_label}_Q{DISCHARGE}_{_metric_desc}.png'
    fig.savefig(output_dir / fname, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'  Saved: {fname}')

print("\nDone.")
