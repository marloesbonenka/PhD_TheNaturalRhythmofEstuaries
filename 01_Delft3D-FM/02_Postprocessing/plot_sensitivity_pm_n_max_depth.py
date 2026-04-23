"""pm/n sensitivity analysis – p95 maximum channel depth

Layout: one subplot per fixed parameter (n or pm), lines per varying parameter.
Colors follow the same PALETTE as plot_scenario_lines.py.
Two figure sets per snapshot:
  A) Effect of R_peak  – one panel per n_peaks  (colours = R_peak values)
  B) Effect of n_peaks – one panel per R_peak   (colours = n_peaks values)
Both sets are also saved as a normalised version (ratio to constant scenario).
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

depth_percentile = 95
bed_threshold = 6
use_absolute_depth = True

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

# Natural variability envelope — noisy repeats of the constant scenario
# from the 1_RiverDischargeVariability_domain45x15 model folder.
# Set SHOW_NOISY_ENVELOPE = True to overlay the grey band on every panel.
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

# Fixed axes dimensions — same across all figure variants so subplots align
AX_W, AX_H = 3.5, 3.0   # axes width / height in inches (not panel/figure size)
# Margins in inches (space outside the axes area):
_LEFT   = 0.95  # left:   y-label (up to 2 lines, 9pt) + ticks (8pt)
_RIGHT  = 0.20  # right:  small buffer
_TOP    = 0.80  # top:    subplot title (fontsize 10) + gap + suptitle (2 lines, fontsize 11)
_BOT    = 0.65  # bottom: x-label + ticks at fontsize 8
_WSPACE = 0.10  # gap between panels in inches (small; sharey=True)


#%% --- FIGURE STYLE ---
STYLE = 'whitefig'   # 'default'   →  white background, black text
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
sensitivity_output_dir = base_path / 'output_plots' / 'plots_pm_n_sensitivity'
sensitivity_output_dir.mkdir(parents=True, exist_ok=True)


#%% --- LOAD DATA ---
comparison_results = {}
comparison_labels = {}

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

    for target_dt, ts_idx, actual_dt in snapshot_matches:
        snapshot_key = f"d{_date_to_filename_tag(target_dt)}"
        comparison_results.setdefault(snapshot_key, {})
        comparison_labels[snapshot_key] = _date_to_label(target_dt)

        bedlev_data = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
        depths_field = np.abs(bedlev_data) if use_absolute_depth else -bedlev_data
        valid_mask = width_mask & (bedlev_data < bed_threshold)

        max_depths = []
        for k in range(len(x_bins) - 1):
            bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
            if np.any(bin_mask):
                valid_depths = depths_field[bin_mask]
                valid_depths = valid_depths[~np.isnan(valid_depths)]
                max_depths.append(
                    np.percentile(valid_depths, depth_percentile) if len(valid_depths) > 0 else np.nan
                )
            else:
                max_depths.append(np.nan)

        comparison_results[snapshot_key][folder_str] = {
            'MaxDepth': np.array(max_depths),
            'x_centers': x_centers,
        }
        print(f"  Snapshot {_date_to_label(target_dt)}: computed p{depth_percentile} max depth.")

    ds.close()


#%% --- LOAD NOISY ENVELOPE DATA ---
# Builds {snapshot_key: {'env_min', 'env_max', 'x_km'}} from the noisy repeats.
# env_min / env_max are expressed as BED ELEVATION [m] (negative = channel depth),
# matching the sign convention used by _get_y().

noisy_envelope_data = {}  # populated only when SHOW_NOISY_ENVELOPE is True

if SHOW_NOISY_ENVELOPE:
    if not NOISY_BASE_PATH.exists():
        print(f"[WARNING] Noisy base path not found: {NOISY_BASE_PATH}")
    else:
        _dx = 1000
        _x_bins = np.arange(x_targets[0], x_targets[-1] + _dx, _dx)
        _x_centers = (_x_bins[:-1] + _x_bins[1:]) / 2

        _noisy_cache_dir = NOISY_BASE_PATH / 'cached_data'
        _noisy_cache_dir.mkdir(parents=True, exist_ok=True)

        _noisy_profiles = {}  # {snapshot_key: [1-D array per run]}

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

            for _tdt, _ts_idx, _adt in _snaps_n:
                _snap_key = f"d{_date_to_filename_tag(_tdt)}"
                _bl = _ds_n['mesh2d_mor_bl'].isel(time=_ts_idx).values.copy()
                _dep = np.abs(_bl) if use_absolute_depth else -_bl
                _valid = _wmask_n & (_bl < bed_threshold)

                _mdepths = []
                for _k in range(len(_x_bins) - 1):
                    _bm = _valid & (_fx_n >= _x_bins[_k]) & (_fx_n < _x_bins[_k + 1])
                    if np.any(_bm):
                        _vd = _dep[_bm]
                        _vd = _vd[~np.isnan(_vd)]
                        _mdepths.append(
                            np.percentile(_vd, depth_percentile)
                            if len(_vd) > 0 else np.nan
                        )
                    else:
                        _mdepths.append(np.nan)

                _noisy_profiles.setdefault(_snap_key, []).append(np.array(_mdepths))
                print(f"  {_subfolder}: snapshot {_date_to_label(_tdt)} OK")

            _ds_n.close()

        # Store negated profiles (bed elevation, matching _get_y sign convention).
        # The ±2σ envelope is finalised in the plotting loop once y_const is known,
        # so the constant run can be included alongside the noisy repeats.
        for _snap_key, _profs in _noisy_profiles.items():
            if _profs:
                noisy_envelope_data[_snap_key] = {
                    'profiles': [-_p for _p in _profs],  # negated to match _get_y()
                    'x_km':     _x_centers / 1000,
                }
        print(f"Noisy envelope ready for {len(noisy_envelope_data)} snapshots.")


#%% --- HELPERS ---
def _parse_pm_n(label_str):
    """Extract (pm, n) ints from a label like 'pm3_n5' or 'pm1_n0 (constant)'."""
    m = re.match(r'pm(\d+)_n(\d+)', label_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


#%% --- SENSITIVITY PLOTS ---
for snapshot_key, snapshot_results in comparison_results.items():
    if not snapshot_results:
        continue

    is_last_snapshot = (snapshot_key == list(comparison_results.keys())[-1])

    scenario_groups = group_snapshot_by_scenario(snapshot_results)
    all_scen_keys = sort_scenario_keys(scenario_groups.keys())
    snap_label = comparison_labels.get(snapshot_key, snapshot_key)

    # Parse pm, n for each scenario
    scen_pm_n = {}
    for scen_key in all_scen_keys:
        label = _scenario_label(scen_key, SCENARIO_LABELS)
        pm, n = _parse_pm_n(label)
        if pm is not None:
            scen_pm_n[scen_key] = (pm, n)

    baseline_scen = next((k for k, (pm, n) in scen_pm_n.items() if n == 0), None)

    # Build groupings
    pm_by_n = {}   # {n_val: [(pm_val, scen_key), ...]}
    n_by_pm = {}   # {pm_val: [(n_val, scen_key), ...]}
    for scen_key, (pm, n) in scen_pm_n.items():
        if n == 0:
            continue
        pm_by_n.setdefault(n, []).append((pm, scen_key))
        n_by_pm.setdefault(pm, []).append((n, scen_key))
    for n in pm_by_n:
        pm_by_n[n].sort()
    for pm in n_by_pm:
        n_by_pm[pm].sort()

    all_pm_vals = sorted({pm for pm, n in scen_pm_n.values() if n > 0})
    all_n_vals  = sorted({n  for pm, n in scen_pm_n.values() if n > 0})

    # Colormaps: Blues for pm (light→dark), Greens for n (light→dark)
    _n_pm = max(len(all_pm_vals) - 1, 1)
    PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}
    _n_n = max(len(all_n_vals) - 1, 1)
    N_COLOR  = {n:  plt.cm.Greens(0.35 + 0.55 * i / _n_n) for i, n  in enumerate(all_n_vals)}

    def _get_y(scen_key):
        """Mean MaxDepth across runs, negated to show as bed elevation (negative = deeper)."""
        y_stack = stack_metric_arrays(scenario_groups[scen_key], 'MaxDepth')
        if y_stack is None:
            return None
        return -np.nanmean(y_stack, axis=0)

    def _get_x(scen_key):
        x_data = next((d for _, d in scenario_groups[scen_key] if 'x_centers' in d), None)
        return x_data['x_centers'] / 1000 if x_data else x_targets / 1000

    y_const = _get_y(baseline_scen) if baseline_scen else None
    x_const = _get_x(baseline_scen) if baseline_scen else None

    # Finalise ±2σ envelope: noisy repeats + constant run
    if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_envelope_data:
        _nd = noisy_envelope_data[snapshot_key]
        if 'profiles' in _nd:
            _all_profs = list(_nd['profiles'])
            if y_const is not None:
                _all_profs.append(y_const)
            _stk = np.vstack(_all_profs)
            _m   = np.nanmean(_stk, axis=0)
            _s   = np.nanstd(_stk, axis=0)
            _nd['env_min'] = _m - 2 * _s
            _nd['env_max'] = _m + 2 * _s

    for normalise in (False, True):
        norm_tag   = '_difference' if normalise else ''
        norm_title = '  (difference from constant)' if normalise else ''
        ylabel = (
            f'p{depth_percentile} depth\n(difference from constant)  [m]'
            if normalise
            else f'bed level [m]  (p{depth_percentile} depth)'
        )

        # ---- Figure A: pm-effect, one panel per n ----
        sorted_n_vals = sorted(pm_by_n.keys())
        if sorted_n_vals:
            n_panels = len(sorted_n_vals)
            _fig_w = _LEFT + n_panels * AX_W + (n_panels - 1) * _WSPACE + _RIGHT
            _fig_h = _BOT + AX_H + _TOP
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(_fig_w, _fig_h),
                sharey=True, sharex=False,
            )
            if n_panels == 1:
                axes = [axes]

            for ci, n_val in enumerate(sorted_n_vals):
                ax = axes[ci]

                # Grey dashed constant reference
                if not normalise and y_const is not None:
                    ax.plot(x_const, y_const, color=GREY_CONST, linewidth=1.5,
                            linestyle='--', label='constant (pm1_n0)', zorder=2)
                if normalise:
                    ax.axhline(0.0, color=GREY_CONST, linewidth=1.5, linestyle='--',
                               label='constant (pm1_n0)', zorder=2)

                # Natural variability envelope
                if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_envelope_data:
                    _env = noisy_envelope_data[snapshot_key]
                    _emin = _env['env_min'].copy()
                    _emax = _env['env_max'].copy()
                    if normalise and y_const is not None:
                        _emin = _emin - y_const
                        _emax = _emax - y_const
                    ax.fill_between(
                        _env['x_km'], _emin, _emax,
                        alpha=0.25, color='0.55', zorder=1,
                        label=r'$\pm 2\sigma$ natural variability',
                    )

                for pm_val, scen_key in pm_by_n[n_val]:
                    y = _get_y(scen_key)
                    if y is None:
                        continue
                    if normalise and y_const is not None:
                        y = y - y_const
                    x = _get_x(scen_key)
                    pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                    ax.plot(x, y, color=PM_COLOR[pm_val], linewidth=1.8,
                            label=f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}', zorder=3)

                ax.set_title(
                    f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}',
                    fontsize=10, fontweight='bold', pad=5,
                )
                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.set_xlabel('distance along estuary [km]', fontsize=8)
                ax.tick_params(labelsize=8)
                if ci == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                    ax.tick_params(axis='y', labelsize=8)

            # Shared legend – constant first, then pm values sorted small→large
            legend_handles = []
            if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_envelope_data:
                legend_handles.append(
                    mpatches.Patch(
                        facecolor='0.55', alpha=0.4,
                        label=r'$\pm 2\sigma$ natural variability',
                    )
                )
            legend_handles.append(
                mlines.Line2D([], [], color=GREY_CONST, linewidth=1.5, linestyle='--',
                              label='constant (pm1_n0)')
            )
            for pm_val in sorted(all_pm_vals):
                pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                legend_handles.append(
                    mlines.Line2D([], [], color=PM_COLOR[pm_val], linewidth=1.8,
                                  linestyle='-', label=f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}')
                )
            fig.legend(
                handles=legend_handles,
                title_fontsize=9, fontsize=8, loc='lower center',
                ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.1), frameon=True,
            )
            fig.suptitle(
                f'Effect of $R_{{\\mathrm{{peak}}}}$ on p{depth_percentile} channel depth{norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=11, fontweight='bold', y=0.99, color=_tc,
            )
            fig.subplots_adjust(
                left=_LEFT / _fig_w,
                right=1 - _RIGHT / _fig_w,
                bottom=_BOT / _fig_h,
                top=1 - _TOP / _fig_h,
                wspace=_WSPACE / AX_W,
            )
            _noisy_tag = '_noisy' if SHOW_NOISY_ENVELOPE else ''
            fname = f'sensitivity_pm_effect_maxdepth{norm_tag}{_noisy_tag}_{snap_label}_Q{DISCHARGE}.png'
            fig.savefig(sensitivity_output_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
            if is_last_snapshot:
                fig.savefig(sensitivity_output_dir / fname.replace('.png', '.pdf'), bbox_inches='tight', transparent=_tr)
            plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')

        # ---- Figure B: n-effect, one panel per pm ----
        sorted_pm_vals = sorted(n_by_pm.keys())
        if sorted_pm_vals:
            pm_panels = len(sorted_pm_vals)
            _fig_w = _LEFT + pm_panels * AX_W + (pm_panels - 1) * _WSPACE + _RIGHT
            _fig_h = _BOT + AX_H + _TOP
            fig, axes = plt.subplots(
                1, pm_panels,
                figsize=(_fig_w, _fig_h),
                sharey=True, sharex=False,
            )
            if pm_panels == 1:
                axes = [axes]

            for ci, pm_val in enumerate(sorted_pm_vals):
                ax = axes[ci]

                if not normalise and y_const is not None:
                    ax.plot(x_const, y_const, color=GREY_CONST, linewidth=1.5,
                            linestyle='--', label='constant (pm1_n0)', zorder=2)
                if normalise:
                    ax.axhline(0.0, color=GREY_CONST, linewidth=1.5, linestyle='--',
                               label='constant (pm1_n0)', zorder=2)

                # Natural variability envelope
                if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_envelope_data:
                    _env = noisy_envelope_data[snapshot_key]
                    _emin = _env['env_min'].copy()
                    _emax = _env['env_max'].copy()
                    if normalise and y_const is not None:
                        _emin = _emin - y_const
                        _emax = _emax - y_const
                    ax.fill_between(
                        _env['x_km'], _emin, _emax,
                        alpha=0.25, color='0.55', zorder=1,
                        label=r'$\pm 2\sigma$ natural variability',
                    )

                for n_val, scen_key in n_by_pm[pm_val]:
                    y = _get_y(scen_key)
                    if y is None:
                        continue
                    if normalise and y_const is not None:
                        y = y - y_const
                    x = _get_x(scen_key)
                    ax.plot(x, y, color=N_COLOR[n_val], linewidth=1.8,
                            label=f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}', zorder=3)

                pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                ax.set_title(
                    f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}',
                    fontsize=10, fontweight='bold', pad=5,
                )
                ax.grid(True, alpha=0.22, linewidth=0.5)
                ax.set_xlabel('distance along estuary [km]', fontsize=8)
                ax.tick_params(labelsize=8)
                if ci == 0:
                    ax.set_ylabel(ylabel, fontsize=9)
                    ax.tick_params(axis='y', labelsize=8)

            # Shared legend – constant first, then n values sorted small→large
            legend_handles = []
            if SHOW_NOISY_ENVELOPE and snapshot_key in noisy_envelope_data:
                legend_handles.append(
                    mpatches.Patch(
                        facecolor='0.55', alpha=0.4,
                        label=r'$\pm 2\sigma$ natural variability',
                    )
                )
            legend_handles.append(
                mlines.Line2D([], [], color=GREY_CONST, linewidth=1.5, linestyle='--',
                              label='constant (pm1_n0)')
            )
            for n_val in sorted(all_n_vals):
                legend_handles.append(
                    mlines.Line2D([], [], color=N_COLOR[n_val], linewidth=1.8,
                                  linestyle='-', label=f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}')
                )
            fig.legend(
                handles=legend_handles, title='Number of peaks',
                title_fontsize=9, fontsize=8, loc='lower center',
                ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.18), frameon=True,
            )
            fig.suptitle(
                f'Effect of $n_{{\\mathrm{{peaks}}}}$ on p{depth_percentile} channel depth{norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=11, fontweight='bold', y=0.99, color=_tc,
            )
            fig.subplots_adjust(
                left=_LEFT / _fig_w,
                right=1 - _RIGHT / _fig_w,
                bottom=_BOT / _fig_h,
                top=1 - _TOP / _fig_h,
                wspace=_WSPACE / AX_W,
            )
            _noisy_tag = '_noisy' if SHOW_NOISY_ENVELOPE else ''
            fname = f'sensitivity_n_effect_maxdepth{norm_tag}{_noisy_tag}_{snap_label}_Q{DISCHARGE}.png'
            fig.savefig(sensitivity_output_dir / fname, dpi=200, bbox_inches='tight', transparent=_tr)
            if is_last_snapshot:
                fig.savefig(sensitivity_output_dir / fname.replace('.png', '.pdf'), bbox_inches='tight', transparent=_tr)
            plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')

print("\nDone.")
