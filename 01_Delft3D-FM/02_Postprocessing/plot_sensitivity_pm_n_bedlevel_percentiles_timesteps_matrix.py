"""pm/n sensitivity analysis – p95 maximum channel depth  (time-evolution matrix)

Layout: matrix of subplots — rows = n_peaks values, columns = R_peak values.
Each panel shows the channel-depth profile for all snapshots (lines coloured by time).
Also produces a normalised version (difference from constant scenario).
Data loading is identical to plot_sensitivity_pm_n_max_depth.py.
"""

#%% IMPORTS
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import (
    _date_to_filename_tag,
    _date_to_label,
    _scenario_label,
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

percentiles = [5, 50, 95]
bed_threshold = 6
CHANNEL_INIT_THRESHOLD = 2.2  # defines the channel footprint from t=0
channel_masks = {}  # {folder_str: {bin_idx: boolean array}}

start_date = np.datetime64('2025-01-01')
x_targets = np.arange(20000, 44001, 1000)
y_range = (5000, 10000)

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

SNAPSHOT_TARGET_DATES = None
SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2031-12-31'))
SNAPSHOT_COUNT = 6

# Natural variability envelope — noisy repeats of the constant scenario.
# Shown once per panel (last snapshot) as a grey band for reference.
SHOW_NOISY_ENVELOPE = True
NOISY_BASE_PATH = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\1_RiverDischargeVariability_domain45x15"
    r"\Model_Output\Q500\0_Noise_Q500"
)
NOISY_SUBFOLDERS = [
    '1_Q500_noisy0.9095347',
    '1_Q500_noisy1_rst.9160657',
    '1_Q500_noisy2_rst.9160663',
]

SHOW_DIFFERENCE = True   # show difference-from-constant plot
SHOW_DETRENDED  = True   # show detrended plot (change relative to initial bed level)


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

# Constant scenario colour
GREY_CONST = "#7f7f7f"

# Fixed axes dimensions (same as plot_sensitivity_pm_n_max_depth.py)
AX_W, AX_H = 3.5, 3.0   # axes width / height in inches
# Margins in inches
_LEFT   = 0.95
_RIGHT  = 0.20
_TOP    = 1.30
_BOT    = 0.65
_WSPACE = 0.10   # horizontal gap between panels in inches
_HSPACE = 0.80   # vertical gap between rows in inches

# --- Line width ---
LINE_WIDTH       = 1.8
LINE_WIDTH_CONST = 1.5

# --- Font sizes ---
FONTSIZE_TITLE  = 18
FONTSIZE_LABELS = FONTSIZE_TITLE - 4
FONTSIZE_TICKS  = FONTSIZE_LABELS - 2


#%% --- FIGURE STYLE ---
STYLE = 'default'   # 'default' or 'whitefig'

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
_tc = plt.rcParams['text.color']
_tr = plt.rcParams.get('savefig.transparent', False)


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
timed_out_dir = base_path / 'timed-out'
matrix_output_dir = base_path / 'output_plots' / 'plots_pm_n_timesteps_matrix'
matrix_output_dir.mkdir(parents=True, exist_ok=True)


#%% --- LOAD DATA ---
comparison_results = {}
comparison_labels  = {}
initial_profiles   = {}  # {folder_str: {pct: 1-D array at t=0}}

target_snapshot_dates = get_target_snapshot_dates(
    count=SNAPSHOT_COUNT,
    explicit_dates=SNAPSHOT_TARGET_DATES,
    date_range=SNAPSHOT_DATE_RANGE,
)

print("\nTarget hydrodynamic snapshot dates:")
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
    dx = 1000
    x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2

    # Initial (t=0) profile — builds frozen channel mask for all plot modes
    if folder_str not in initial_profiles:
        _init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
        _valid_init = width_mask & (_init_bl < CHANNEL_INIT_THRESHOLD)
        channel_masks[folder_str] = {}
        initial_profiles[folder_str] = {}

        for _k in range(len(x_bins) - 1):
            _bm = _valid_init & (face_x >= x_bins[_k]) & (face_x < x_bins[_k + 1])
            channel_masks[folder_str][_k] = _bm  # frozen for all timesteps

        for pct in percentiles:
            _init_percs = []
            for _k in range(len(x_bins) - 1):
                _bm = channel_masks[folder_str][_k]
                if np.any(_bm):
                    _vd = _init_bl[_bm]
                    _vd = _vd[~np.isnan(_vd)]
                    _init_percs.append(np.percentile(_vd, pct) if len(_vd) > 0 else np.nan)
                else:
                    _init_percs.append(np.nan)
            initial_profiles[folder_str][pct] = np.array(_init_percs)
        print(f"  Initial profile (t=0) and channel mask computed.")

    for target_dt, ts_idx, actual_dt in snapshot_matches:
        snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
        comparison_results.setdefault(snapshot_key, {})
        comparison_labels[snapshot_key] = _date_to_label(target_dt)

        bedlev_data = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

        result = {'x_centers': x_centers}
        for pct in percentiles:
            profile = []
            for k in range(len(x_bins) - 1):
                bin_mask = channel_masks[folder_str][k]  # <-- frozen t=0 channel mask
                if np.any(bin_mask):
                    valid_depths = bedlev_data[bin_mask]
                    valid_depths = valid_depths[~np.isnan(valid_depths)]
                    profile.append(np.percentile(valid_depths, pct) if len(valid_depths) > 0 else np.nan)
                else:
                    profile.append(np.nan)
            result[f'Depth_p{pct}'] = np.array(profile)

        comparison_results[snapshot_key][folder_str] = result
        print(f"  Snapshot {_date_to_label(target_dt)}: computed bed levels at p{percentiles}.")

    ds.close()


#%% --- LOAD NOISY ENVELOPE DATA ---
# Builds {snapshot_key: {'env_min', 'env_max', 'x_km'}} from the noisy repeats.
noisy_envelope_data = {}

if SHOW_NOISY_ENVELOPE:
    if not NOISY_BASE_PATH.exists():
        print(f"[WARNING] Noisy base path not found: {NOISY_BASE_PATH}")
    else:
        _dx = 1000
        _x_bins = np.arange(x_targets[0], x_targets[-1] + _dx, _dx)
        _x_centers = (_x_bins[:-1] + _x_bins[1:]) / 2

        _noisy_cache_dir = NOISY_BASE_PATH / 'cached_data'
        _noisy_cache_dir.mkdir(parents=True, exist_ok=True)

        _noisy_profiles      = {pct: {} for pct in percentiles}
        _noisy_init_profiles = {pct: [] for pct in percentiles}

        for _subfolder in NOISY_SUBFOLDERS:
            _noisy_folder = NOISY_BASE_PATH / _subfolder
            if not _noisy_folder.exists():
                print(f"[WARNING] Noisy subfolder not found: {_noisy_folder}")
                continue
            print(f"Loading noisy run: {_subfolder}")

            _ds_n = load_or_update_map_cache_multi(
                cache_dir=_noisy_cache_dir,
                folder_name=_subfolder,
                run_paths=[_noisy_folder],
                var_names=['mesh2d_mor_bl'],
                bbox=CACHE_BBOX,
                append_time=APPEND_TIMESTEPS,
                append_vars=APPEND_VARIABLES,
                cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
            )
            if _ds_n is None:
                print(f"  No cached data for {_subfolder}, skipping.")
                continue

            _snaps_n = get_snapshot_matches_by_target_dates(
                _ds_n.time.values, target_snapshot_dates
            )
            _fx_n, _fy_n = _get_face_coords(_ds_n)
            _wmask_n = (_fy_n >= y_range[0]) & (_fy_n <= y_range[1])

            # Build frozen t=0 channel mask using CHANNEL_INIT_THRESHOLD
            _init_bl_n = _ds_n['mesh2d_mor_bl'].isel(time=0).values.copy()
            _valid_init_n = _wmask_n & (_init_bl_n < CHANNEL_INIT_THRESHOLD)
            _noisy_channel_masks = {}
            for _ki in range(len(_x_bins) - 1):
                _bmi = _valid_init_n & (_fx_n >= _x_bins[_ki]) & (_fx_n < _x_bins[_ki + 1])
                _noisy_channel_masks[_ki] = _bmi

            # Initial (t=0) profile per percentile for detrended normalization
            if SHOW_DETRENDED:
                for pct in percentiles:
                    _init_percs_n = []
                    for _ki in range(len(_x_bins) - 1):
                        _bmi = _noisy_channel_masks[_ki]
                        if np.any(_bmi):
                            _vdi = _init_bl_n[_bmi]
                            _vdi = _vdi[~np.isnan(_vdi)]
                            _init_percs_n.append(np.percentile(_vdi, pct) if len(_vdi) > 0 else np.nan)
                        else:
                            _init_percs_n.append(np.nan)
                    _noisy_init_profiles[pct].append(np.array(_init_percs_n))

            for _tdt, _ts_idx, _adt in _snaps_n:
                _snap_key = f"d{_date_to_filename_tag(_tdt)}"
                _bl = _ds_n['mesh2d_mor_bl'].isel(time=_ts_idx).values.copy()

                for pct in percentiles:
                    _mdepths = []
                    for _k in range(len(_x_bins) - 1):
                        _bm = _noisy_channel_masks[_k]  # <-- frozen t=0 channel mask
                        if np.any(_bm):
                            _vd = _bl[_bm]
                            _vd = _vd[~np.isnan(_vd)]
                            _mdepths.append(np.percentile(_vd, pct) if len(_vd) > 0 else np.nan)
                        else:
                            _mdepths.append(np.nan)
                    _noisy_profiles[pct].setdefault(_snap_key, []).append(np.array(_mdepths))
                print(f"  {_subfolder}: snapshot {_date_to_label(_tdt)} OK")

            _ds_n.close()

        for pct in percentiles:
            for _snap_key, _profs in _noisy_profiles[pct].items():
                if _profs:
                    noisy_envelope_data.setdefault(_snap_key, {})[pct] = {
                        'profiles': list(_profs),
                        'x_km':     _x_centers / 1000,
                    }
            if SHOW_DETRENDED and _noisy_init_profiles[pct]:
                for _snap_key in noisy_envelope_data:
                    noisy_envelope_data[_snap_key].setdefault(pct, {})['initial_profile'] = np.nanmean(
                        np.vstack(_noisy_init_profiles[pct]), axis=0
                    )
        print(f"Noisy envelope ready for {len(noisy_envelope_data)} snapshots.")


#%% --- HELPERS ---
def _parse_pm_n(label_str):
    """Extract (pm, n) ints from a label like 'pm3_n5' or 'pm1_n0 (constant)'."""
    m = re.match(r'pm(\d+)_n(\d+)', label_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


#%% --- BUILD SCENARIO STRUCTURE ---
# Group data per snapshot into scenario groups
scenario_groups_all = {}
for snap_key, snap_results in comparison_results.items():
    scenario_groups_all[snap_key] = group_snapshot_by_scenario(snap_results)

# Use first snapshot to discover all available scenarios
first_snap_key = next(iter(comparison_results))
scen_pm_n = {}
for scen_key in sort_scenario_keys(scenario_groups_all[first_snap_key].keys()):
    label = _scenario_label(scen_key, SCENARIO_LABELS)
    pm, n = _parse_pm_n(label)
    if pm is not None:
        scen_pm_n[scen_key] = (pm, n)

baseline_scen = next((k for k, (pm, n) in scen_pm_n.items() if n == 0), None)

all_pm_vals = sorted({pm for pm, n in scen_pm_n.values() if n > 0})
all_n_vals  = sorted({n  for pm, n in scen_pm_n.values() if n > 0})

# Snapshot ordering (chronological)
all_snap_keys   = list(comparison_results.keys())
all_snap_labels = [comparison_labels[k] for k in all_snap_keys]
n_snaps = len(all_snap_keys)

# Colormap: sequential (plasma), one colour per snapshot (early = light, late = dark)
SNAP_CMAP = plt.cm.plasma
SNAP_COLORS = {
    k: SNAP_CMAP(0.15 + 0.70 * i / max(n_snaps - 1, 1))
    for i, k in enumerate(all_snap_keys)
}


def _get_y_snap(scen_key, snap_key, pct=95):
    """Mean Depth_p{pct} across runs for (scenario, snapshot) as raw bed elevation (negative = deeper)."""
    grp = scenario_groups_all.get(snap_key, {})
    if scen_key not in grp:
        return None
    y_stack = stack_metric_arrays(grp[scen_key], f'Depth_p{pct}')
    if y_stack is None:
        return None
    return np.nanmean(y_stack, axis=0)


def _get_x_snap(scen_key, snap_key):
    grp = scenario_groups_all.get(snap_key, {})
    if scen_key not in grp:
        return x_targets / 1000
    x_data = next((d for _, d in grp[scen_key] if 'x_centers' in d), None)
    return x_data['x_centers'] / 1000 if x_data else x_targets / 1000


def _get_initial_profile_snap(scen_key, pct):
    """Mean initial (t=0) p{pct} profile across runs in a scenario."""
    grp = scenario_groups_all.get(first_snap_key, {})
    if scen_key not in grp:
        return None
    profs = [initial_profiles[fn][pct] for fn, _ in grp[scen_key]
             if fn in initial_profiles and pct in initial_profiles.get(fn, {})]
    if not profs:
        return None
    return np.nanmean(np.vstack(profs), axis=0)


# Constant scenario profile per snapshot, stored per percentile
y_const_by_snap = {}
x_const_by_snap = {}
for snap_key in all_snap_keys:
    if baseline_scen:
        y_const_by_snap[snap_key] = {pct: _get_y_snap(baseline_scen, snap_key, pct) for pct in percentiles}
        x_const_by_snap[snap_key] = _get_x_snap(baseline_scen, snap_key)

# Finalise noisy envelope (include constant run alongside the noisy repeats)
if SHOW_NOISY_ENVELOPE:
    for snap_key in all_snap_keys:
        if snap_key in noisy_envelope_data:
            for pct in percentiles:
                _pct_env = noisy_envelope_data[snap_key].get(pct, {})
                if 'profiles' in _pct_env:
                    _all_profs = list(_pct_env['profiles'])
                    y_c = y_const_by_snap.get(snap_key, {}).get(pct)
                    if y_c is not None:
                        _all_profs.append(y_c)
                    _stk = np.vstack(_all_profs)
                    _m   = np.nanmean(_stk, axis=0)
                    _s   = np.nanstd(_stk, axis=0)
                    _pct_env['env_min'] = _m - 2 * _s
                    _pct_env['env_max'] = _m + 2 * _s


#%% --- MATRIX PLOTS ---
# Rows = n_peaks (ascending top→bottom), Columns = R_peak / pm (ascending left→right)
# Lines per panel = snapshots, coloured by time.
# One figure per percentile × normalise (False / True).

pm_n_to_scen = {
    (pm, n): scen_key
    for scen_key, (pm, n) in scen_pm_n.items()
    if n > 0
}

n_rows = len(all_n_vals)
n_cols = len(all_pm_vals)

last_snap_key = all_snap_keys[-1]

for pct in percentiles:
    # Detrended references for this percentile
    y_init_const_pct = _get_initial_profile_snap(baseline_scen, pct) if baseline_scen else None
    y_const_det_pct = {
        snap_key: (
            y_const_by_snap[snap_key][pct] - y_init_const_pct
            if y_const_by_snap[snap_key][pct] is not None and y_init_const_pct is not None
            else None
        )
        for snap_key in all_snap_keys
    }
    _noisy_init_pct = (
        {snap_key: noisy_envelope_data.get(snap_key, {}).get(pct, {}).get('initial_profile')
         for snap_key in all_snap_keys}
        if SHOW_NOISY_ENVELOPE else {}
    )

    _plot_modes = ['absolute']
    if SHOW_DIFFERENCE:
        _plot_modes.append('difference')
    if SHOW_DETRENDED:
        _plot_modes.append('detrended')

    for plot_mode in _plot_modes:
        normalise = (plot_mode == 'difference')
        detrended = (plot_mode == 'detrended')

        if plot_mode == 'absolute':
            norm_tag   = ''
            norm_title = ''
            ylabel_phys = f'bed level [m]  (p{pct})'
        elif plot_mode == 'difference':
            norm_tag   = '_difference'
            norm_title = '  (difference from constant)'
            ylabel_phys = f'p{pct} bed level\n(difference from constant)  [m]'
        else:  # detrended
            norm_tag   = '_detrended'
            norm_title = '  (change from initial bed)'
            ylabel_phys = f'p{pct} bed level\n(change from initial bed)  [m]'

        _fig_w = _LEFT + n_cols * AX_W + (n_cols - 1) * _WSPACE + _RIGHT
        _fig_h = _BOT + n_rows * AX_H + (n_rows - 1) * _HSPACE + _TOP

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(_fig_w, _fig_h),
            sharey=True, sharex=False,
            squeeze=False,
        )

        for ri, n_val in enumerate(all_n_vals):        # rows: ascending n_peaks top→bottom
            for ci, pm_val in enumerate(all_pm_vals):  # cols: ascending R_peak left→right
                ax = axes[ri][ci]
                scen_key = pm_n_to_scen.get((pm_val, n_val))

                # Grey dashed constant reference
                if not normalise and not detrended and baseline_scen:
                    x_c = x_const_by_snap.get(first_snap_key)
                    y_c = y_const_by_snap.get(first_snap_key, {}).get(pct)
                    if y_c is not None and x_c is not None:
                        ax.plot(x_c, y_c, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                                linestyle='--', zorder=2)
                if normalise:
                    ax.axhline(0.0, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                               linestyle='--', zorder=2)
                if detrended:
                    x_c = x_const_by_snap.get(first_snap_key)
                    y_cd = y_const_det_pct.get(first_snap_key)
                    if y_cd is not None and x_c is not None:
                        ax.plot(x_c, y_cd, color=GREY_CONST, linewidth=LINE_WIDTH_CONST,
                                linestyle='--', zorder=2)

                # Natural variability envelope (last snapshot only, to avoid clutter)
                if SHOW_NOISY_ENVELOPE and last_snap_key in noisy_envelope_data:
                    _env = noisy_envelope_data[last_snap_key].get(pct, {})
                    if 'env_min' in _env and 'env_max' in _env:
                        _emin = _env['env_min'].copy()
                        _emax = _env['env_max'].copy()
                        if normalise:
                            y_c = y_const_by_snap.get(last_snap_key, {}).get(pct)
                            if y_c is not None:
                                _emin = _emin - y_c
                                _emax = _emax - y_c
                        elif detrended:
                            _ni = _noisy_init_pct.get(last_snap_key)
                            if _ni is not None:
                                _emin = _emin - _ni
                                _emax = _emax - _ni
                        ax.fill_between(
                            _env['x_km'], _emin, _emax,
                            alpha=0.20, color='0.55', zorder=1,
                        )

                if scen_key is not None:
                    for snap_key in all_snap_keys:
                        y = _get_y_snap(scen_key, snap_key, pct)
                        if y is None:
                            continue
                        if normalise:
                            y_c = y_const_by_snap.get(snap_key, {}).get(pct)
                            if y_c is not None:
                                y = y - y_c
                        elif detrended:
                            _y_init = _get_initial_profile_snap(scen_key, pct)
                            if _y_init is not None:
                                y = y - _y_init
                        x = _get_x_snap(scen_key, snap_key)
                        ax.plot(x, y, color=SNAP_COLORS[snap_key], linewidth=LINE_WIDTH, zorder=3)
                else:
                    # No scenario for this (pm, n) combination — grey out the panel
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=FONTSIZE_TICKS, color='0.5')

                # Column header: R_peak value (top row only)
                if ri == 0:
                    pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                    ax.set_title(
                        f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}',
                        fontsize=FONTSIZE_TITLE, fontweight='bold', pad=5,
                    )

                # Row label + physical ylabel (left column only)
                if ci == 0:
                    ax.set_ylabel(
                        f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}\n{ylabel_phys}',
                        fontsize=FONTSIZE_LABELS,
                    )
                    ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)

                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.set_xlabel('distance along estuary [km]', fontsize=FONTSIZE_TICKS)
                ax.tick_params(labelsize=FONTSIZE_TICKS)

        # --- Shared legend ---
        legend_handles = []
        if SHOW_NOISY_ENVELOPE and last_snap_key in noisy_envelope_data and \
                pct in noisy_envelope_data.get(last_snap_key, {}):
            legend_handles.append(
                mpatches.Patch(
                    facecolor='0.55', alpha=0.4,
                    label=r'$\pm 2\sigma$ natural variability (last snapshot)',
                )
            )
        legend_handles.append(
            mlines.Line2D([], [], color=GREY_CONST, linewidth=LINE_WIDTH_CONST, linestyle='--',
                          label='constant (pm1_n0)')
        )
        for snap_key, snap_label in zip(all_snap_keys, all_snap_labels):
            legend_handles.append(
                mlines.Line2D([], [], color=SNAP_COLORS[snap_key], linewidth=LINE_WIDTH,
                              linestyle='-', label=snap_label)
            )

        fig.legend(
            handles=legend_handles,
            title='Snapshot (hydrodynamic year)',
            title_fontsize=FONTSIZE_LABELS, fontsize=FONTSIZE_TICKS,
            loc='lower center', ncol=len(legend_handles),
            bbox_to_anchor=(0.5, -0.06), frameon=True,
        )
        fig.suptitle(
            f'p{pct} channel depth — time evolution per (pm, n) scenario{norm_title}\n'
            f'Q = {DISCHARGE} m³/s  |  rows = $n_{{\\mathrm{{peaks}}}}$,  columns = $R_{{\\mathrm{{peak}}}}$',
            fontsize=FONTSIZE_TITLE, fontweight='bold', y=0.995, color=_tc,
        )
        fig.subplots_adjust(
            left=_LEFT / _fig_w,
            right=1 - _RIGHT / _fig_w,
            bottom=_BOT / _fig_h,
            top=1 - _TOP / _fig_h,
            wspace=_WSPACE / AX_W,
            hspace=_HSPACE / AX_H,
        )

        _noisy_tag = '_noisy' if SHOW_NOISY_ENVELOPE else ''
        fname = f'sensitivity_matrix_timesteps_maxdepth{norm_tag}{_noisy_tag}_{STYLE}_Q{DISCHARGE}_p{pct}.png'
        fig.savefig(matrix_output_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
        fig.savefig(matrix_output_dir / fname.replace('.png', '.pdf'), bbox_inches='tight', transparent=_tr)
        plt.show()
        plt.close(fig)
        print(f'Saved: {fname}')

print("\nDone.")
