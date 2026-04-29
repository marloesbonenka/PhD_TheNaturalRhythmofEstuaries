"""Location of minimum p95 channel depth along the estuary

For a single snapshot (2031-12-31), computes the p95 maximum channel depth
profile along the estuary for every scenario run across Q = 250 / 500 / 1000,
then finds the x-location of the shallowest point (minimum value in that
profile).

Scenario folders are discovered automatically from the config directory.
Expected folder name pattern:  \d{2}_Qr{Q}_pm{pm}_n{n}.{runid}
e.g.  01_Qr500_pm1_n0.9600302

Result: scatter plot of min-depth / location vs. discharge amplitude (pm),
colour-coded by discharge.
"""

#%% IMPORTS
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import (
    _date_to_filename_tag,
    _date_to_label,
    get_variability_map,
    get_target_snapshot_dates,
    get_snapshot_matches_by_target_dates,
)
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths


#%% --- CONFIGURATION ---
DISCHARGES = [250, 500, 1000]
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")

depth_percentile = 95
bed_threshold = 6
use_absolute_depth = True

x_targets = np.arange(20000, 44001, 100)
y_range = (5000, 10000)

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Time range over which to compute the trend (one snapshot per year)
SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2031-12-31'))
SNAPSHOT_COUNT      = 7

# Colour per discharge value
DISCHARGE_COLORS = {250: '#3B6064', 500: '#87BBA2', 1000: '#C9E4CA'}

# --- Line width ---
LINE_WIDTH = 1.8   # width of scatter marker edges / annotation lines

# --- Font sizes ---
FONTSIZE_TITLE  = 11   # figure suptitle and panel titles
FONTSIZE_LABELS = 9    # axis labels and legend title
FONTSIZE_TICKS  = 8    # tick labels and legend text


#%% --- FIGURE STYLE ---
STYLE = 'default'   # 'default'   →  white background, black text
                    # 'whitefig'  →  transparent figure, white axes background, white text

STYLES = {
    'default': {},
    'whitefig': {
        'figure.facecolor':    'none',
        'axes.facecolor':      'white',
        'axes.edgecolor':      'white',
        'axes.labelcolor':     'white',
        'xtick.color':         'white',
        'ytick.color':         'white',
        'text.color':          'white',
        'grid.color':          '#cccccc',
        'legend.facecolor':    'none',
        'legend.edgecolor':    'white',
        'savefig.transparent': False,
    },
}

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])
_tc = plt.rcParams['text.color']                        # convenience: text/title color
_tr = plt.rcParams.get('savefig.transparent', False)    # convenience: transparent flag for savefig

dx = 100
x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
x_centers = (x_bins[:-1] + x_bins[1:]) / 2


#%% --- HELPERS ---
# Pattern:  \d{2}_Qr{Q}_pm{pm}_n{n}.{anything}
_FOLDER_RE = re.compile(r'^\d{2}_Qr(\d+)_pm(\d+)_n(\d+)\.')

def _discover_scenario_folders(base_path, discharge):
    """Return list of (folder, pm_val, n_val) for all matching run folders."""
    results = []
    if not base_path.exists():
        return results
    for folder in sorted(base_path.iterdir()):
        if not folder.is_dir():
            continue
        m = _FOLDER_RE.match(folder.name)
        if m and int(m.group(1)) == discharge:
            results.append((folder, int(m.group(2)), int(m.group(3))))
    return results


#%% --- LOAD DATA ---
# datadict: {folder_name: {'discharge': int, 'pm': int, 'n': int,
#                          'times_yr':         decimal-year array per snapshot,
#                          'x_min_km_series':  x-location of min depth per snapshot,
#                          'min_depth_series': min depth value per snapshot,
#                          'slope_km_yr':      linear trend slope [km/yr]}}
datadict = {}

target_snapshot_dates = get_target_snapshot_dates(
    count=SNAPSHOT_COUNT,
    date_range=SNAPSHOT_DATE_RANGE,
)

print(f"Target snapshot dates: {[_date_to_label(d) for d in target_snapshot_dates]}\n")

for discharge in DISCHARGES:
    config   = f'Model_Output/Q{discharge}'
    base_path = base_directory / config

    if not base_path.exists():
        print(f"[SKIP] Folder does not exist: {base_path}")
        continue

    scenario_folders = _discover_scenario_folders(base_path, discharge)
    if not scenario_folders:
        print(f"[SKIP] No matching simulation folders found in: {base_path}")
        continue

    print(f"Q={discharge}: found {len(scenario_folders)} scenario folder(s).")

    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)
    timed_out_dir = base_path / 'timed-out'

    VARIABILITY_MAP = get_variability_map(discharge)
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)

    for folder, pm_val, n_val in scenario_folders:
        folder_str = folder.name
        print(f"  Processing: {folder_str}")

        run_paths = get_stitched_map_run_paths(
            base_path=base_path,
            folder_name=folder_str,
            timed_out_dir=timed_out_dir,
            variability_map=VARIABILITY_MAP,
            analyze_noisy=False,
        )
        if not run_paths:
            run_paths = [folder]

        ds = load_or_update_map_cache_multi(
            cache_dir=assessment_dir,
            folder_name=folder_str,
            run_paths=run_paths,
            var_names=['mesh2d_mor_bl'],
            bbox=CACHE_BBOX,
            append_time=APPEND_TIMESTEPS,
            append_vars=APPEND_VARIABLES,
            cache_tag=cache_tag,
        )
        if ds is None:
            print(f"    No cached data, skipping.")
            continue

        snapshot_matches = get_snapshot_matches_by_target_dates(
            ds.time.values, target_snapshot_dates
        )
        if not snapshot_matches:
            print(f"    No matching timesteps, skipping.")
            ds.close()
            continue

        face_x, face_y = _get_face_coords(ds)
        width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])

        times_yr_list    = []
        x_min_km_list    = []
        min_depth_list   = []

        for target_dt, ts_idx, actual_dt in snapshot_matches:
            bedlev_data  = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
            depths_field = np.abs(bedlev_data) if use_absolute_depth else -bedlev_data
            valid_mask   = width_mask & (bedlev_data < bed_threshold)

            profile = []
            for k in range(len(x_bins) - 1):
                bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
                if np.any(bin_mask):
                    vd = depths_field[bin_mask]
                    vd = vd[~np.isnan(vd)]
                    profile.append(
                        np.percentile(vd, depth_percentile) if len(vd) > 0 else np.nan
                    )
                else:
                    profile.append(np.nan)

            profile   = np.array(profile)
            valid_idx = np.where(~np.isnan(profile))[0]
            if len(valid_idx) == 0:
                continue

            min_idx = valid_idx[np.argmin(profile[valid_idx])]
            # Convert target_dt (numpy datetime64) to decimal year
            t_yr = (target_dt.astype('datetime64[D]') - np.datetime64('2000-01-01', 'D')).astype(float) / 365.25 + 2000.0
            times_yr_list.append(t_yr)
            x_min_km_list.append(x_centers[min_idx] / 1000)
            min_depth_list.append(float(profile[min_idx]))

        ds.close()

        if len(times_yr_list) < 2:
            print(f"    Too few valid snapshots ({len(times_yr_list)}), skipping.")
            continue

        times_yr         = np.array(times_yr_list)
        x_min_km_series  = np.array(x_min_km_list)
        min_depth_series = np.array(min_depth_list)
        slope_km_yr, _   = np.polyfit(times_yr, x_min_km_series, 1)

        datadict[folder_str] = {
            'discharge':        discharge,
            'pm':               pm_val,
            'n':                n_val,
            'times_yr':         times_yr,
            'x_min_km_series':  x_min_km_series,
            'min_depth_series': min_depth_series,
            'slope_km_yr':      float(slope_km_yr),
        }
        print(f"    pm={pm_val}, n={n_val}  →  slope = {slope_km_yr:+.3f} km/yr  "
              f"(final x = {x_min_km_series[-1]:.1f} km, depth = {min_depth_series[-1]:.2f} m)")

print(f"\nCollected data for {len(datadict)} runs.")


#%% --- COMPUTE DELTA SLOPE (relative to constant run per discharge) ---
# The constant run is pm=1, n=0.  delta_slope = slope_scenario - slope_constant.
# Negative delta_slope → extra seaward movement relative to the constant run.
for discharge in DISCHARGES:
    const_entry = next(
        (v for v in datadict.values()
         if v['discharge'] == discharge and v['pm'] == 1 and v['n'] == 0),
        None,
    )
    if const_entry is None:
        print(f"[WARNING] No constant run (pm=1, n=0) found for Q={discharge}; "
              f"delta_slope will equal the absolute slope.")
    slope_const = const_entry['slope_km_yr'] if const_entry is not None else 0.0
    for v in datadict.values():
        if v['discharge'] == discharge:
            v['delta_slope_km_yr'] = v['slope_km_yr'] - slope_const


#%% --- PLOT ---
if not datadict:
    print("No data to plot.")
else:
    present_discharges = sorted({v['discharge'] for v in datadict.values()})
    _style_tag = f'_{STYLE}' if STYLE != 'default' else ''
    q_str = '_'.join(f'Q{q}' for q in present_discharges)
    ref_q = present_discharges[0]
    output_dir = (
        base_directory / f'Model_Output/Q{ref_q}' / 'output_plots' / 'plots_location_mindepth'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _date_end = _date_to_label(target_snapshot_dates[-1])
    _date_start = _date_to_label(target_snapshot_dates[0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for discharge in present_discharges:
        color  = DISCHARGE_COLORS.get(discharge, 'tab:gray')
        subset = {k: v for k, v in datadict.items() if v['discharge'] == discharge}

        pm_arr     = np.array([v['pm']                   for v in subset.values()], dtype=float)
        depths_arr = np.array([v['min_depth_series'][-1] for v in subset.values()], dtype=float)
        x_locs_arr = np.array([v['x_min_km_series'][-1]  for v in subset.values()], dtype=float)
        n_arr      = [v['n'] for v in subset.values()]
        label_q    = f'Q = {discharge} m³/s'

        # Panel A: min depth at final snapshot vs pm
        axes[0].scatter(pm_arr, depths_arr, color=color, zorder=3, label=label_q)
        for x_, y_, n_ in zip(pm_arr, depths_arr, n_arr):
            axes[0].annotate(f'n{n_}', (x_, y_), textcoords='offset points',
                             xytext=(4, 4), fontsize=FONTSIZE_TICKS, color=color)

        # Panel B: x-location at final snapshot vs pm
        axes[1].scatter(pm_arr, x_locs_arr, color=color, zorder=3, label=label_q)
        for x_, y_, n_ in zip(pm_arr, x_locs_arr, n_arr):
            axes[1].annotate(f'n{n_}', (x_, y_), textcoords='offset points',
                             xytext=(4, 4), fontsize=FONTSIZE_TICKS, color=color)

    axes[0].set_xlabel('Discharge amplitude $R_{\\mathrm{peak}}$ (pm)', fontsize=FONTSIZE_LABELS)
    axes[0].set_ylabel(f'Min p{depth_percentile} depth [m]', fontsize=FONTSIZE_LABELS)
    axes[0].set_title(f'Shallowest depth at {_date_end}', fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=FONTSIZE_TICKS)
    axes[0].legend(fontsize=FONTSIZE_TICKS)

    axes[1].set_xlabel('Discharge amplitude $R_{\\mathrm{peak}}$ (pm)', fontsize=FONTSIZE_LABELS)
    axes[1].set_ylabel('x-location of min depth [km]', fontsize=FONTSIZE_LABELS)
    axes[1].set_title(f'Location of shallowest point at {_date_end}', fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=FONTSIZE_TICKS)
    axes[1].legend(fontsize=FONTSIZE_TICKS)

    fig.suptitle(
        f'p{depth_percentile} channel depth  |  dx = {dx} m bins  |  Q = {q_str}',
        fontsize=FONTSIZE_TITLE, fontweight='bold', color=_tc,
    )
    fig.tight_layout()

    fname = f'location_mindepth_{q_str}{_style_tag}.png'
    fig.savefig(output_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
    plt.show()
    plt.close(fig)
    print(f'Saved: {output_dir / fname}')

    # ---- Figure 2: pm × n scatter, colour = metric ----
    METRICS = [
        ('min_depth_series', -1, f'Min p{depth_percentile} depth [m]',  'viridis',  f'depth_{_date_end}'),
        ('x_min_km_series',  -1, 'x-location of min depth [km]',        'plasma',   f'xloc_{_date_end}'),
    ]

    for discharge in present_discharges:
        subset = {k: v for k, v in datadict.items() if v['discharge'] == discharge}
        if not subset:
            continue

        pm_arr  = np.array([v['pm'] for v in subset.values()], dtype=float)
        n_arr   = np.array([v['n']  for v in subset.values()], dtype=float)

        fig2, axes2 = plt.subplots(1, len(METRICS), figsize=(5.5 * len(METRICS), 4.5))
        if len(METRICS) == 1:
            axes2 = [axes2]

        for ax2, (key, idx, cbar_label, cmap, _) in zip(axes2, METRICS):
            vals = np.array([v[key][idx] for v in subset.values()], dtype=float)
            sc = ax2.scatter(pm_arr, n_arr, c=vals, cmap=cmap, s=120, zorder=3,
                             edgecolors='0.3', linewidths=0.5)
            cb = fig2.colorbar(sc, ax=ax2, pad=0.02)
            cb.set_label(cbar_label, fontsize=FONTSIZE_LABELS)
            cb.ax.tick_params(labelsize=FONTSIZE_TICKS)
            for x_, y_, v_ in zip(pm_arr, n_arr, vals):
                ax2.annotate(f'{v_:.2f}', (x_, y_), textcoords='offset points',
                             xytext=(6, 4), fontsize=FONTSIZE_TICKS - 1)
            ax2.set_xlabel('Discharge amplitude $R_{\\mathrm{peak}}$ (pm)', fontsize=FONTSIZE_LABELS)
            ax2.set_ylabel('Number of peaks $n$', fontsize=FONTSIZE_LABELS)
            ax2.set_title(cbar_label, fontsize=FONTSIZE_TITLE, fontweight='bold')
            ax2.tick_params(labelsize=FONTSIZE_TICKS)
            ax2.grid(True, alpha=0.25)

        fig2.suptitle(
            f'pm × n parameter space  |  Q = {discharge} m³/s  |  {_date_end}',
            fontsize=FONTSIZE_TITLE, fontweight='bold', color=_tc,
        )
        fig2.tight_layout()

        fname2 = f'pmn_scatter_Q{discharge}{_style_tag}.png'
        fig2.savefig(output_dir / fname2, dpi=200, bbox_inches='tight', transparent=_tr)
        plt.show()
        plt.close(fig2)
        print(f'Saved: {output_dir / fname2}')

print("\nDone.")
