"""pm/n sensitivity analysis: width-averaged bed level

Layout: one subplot per fixed parameter (n or pm), lines per varying parameter.
Colors follow the same PALETTE as plot_scenario_lines.py.
Two figure sets per snapshot:
  A) Effect of R_peak:  one panel per n_peaks  (colours = R_peak values)
  B) Effect of n_peaks: one panel per R_peak   (colours = n_peaks values)
Both sets are also saved as a normalised version (ratio to constant scenario).
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
NOISY = False
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config = f'Model_Output/Q{DISCHARGE}'

bed_threshold = 6

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

PANEL_W, PANEL_H = 4.0, 3.5


#%% --- SEARCH FOLDERS ---
base_path = base_directory / config
VARIABILITY_MAP = get_variability_map(DISCHARGE)

model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=None,
    analyze_noisy=NOISY,
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
        analyze_noisy=NOISY,
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
        valid_mask = width_mask & (bedlev_data < bed_threshold)

        bin_means = []
        for k in range(len(x_bins) - 1):
            bin_mask = valid_mask & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
            bin_means.append(np.mean(bedlev_data[bin_mask]) if np.any(bin_mask) else np.nan)

        comparison_results[snapshot_key][folder_str] = {
            'BL': np.array(bin_means),
            'x_centers': x_centers,
        }
        print(f"  Snapshot {_date_to_label(target_dt)}: computed width-averaged bed level.")

    ds.close()


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
        """Mean width-averaged bed level across runs."""
        y_stack = stack_metric_arrays(scenario_groups[scen_key], 'BL')
        if y_stack is None:
            return None
        return np.nanmean(y_stack, axis=0)

    def _get_x(scen_key):
        x_data = next((d for _, d in scenario_groups[scen_key] if 'x_centers' in d), None)
        return x_data['x_centers'] / 1000 if x_data else x_targets / 1000

    y_const = _get_y(baseline_scen) if baseline_scen else None
    x_const = _get_x(baseline_scen) if baseline_scen else None
    const_denom = np.where(np.abs(y_const) < 1e-6, np.nan, y_const) if y_const is not None else None

    for normalise in (False, True):
        norm_tag   = '_normalised' if normalise else ''
        norm_title = '  (normalised by constant)' if normalise else ''
        ylabel = (
            'width-averaged bed level\n(ratio to constant)'
            if normalise
            else 'bed level [m]'
        )

        # ---- Figure A: pm-effect, one panel per n ----
        sorted_n_vals = sorted(pm_by_n.keys())
        if sorted_n_vals:
            n_panels = len(sorted_n_vals)
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(PANEL_W * n_panels, PANEL_H),
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
                    ax.axhline(1.0, color=GREY_CONST, linewidth=1.5, linestyle='--',
                               label='constant (pm1_n0)', zorder=2)

                for pm_val, scen_key in pm_by_n[n_val]:
                    y = _get_y(scen_key)
                    if y is None:
                        continue
                    if normalise and const_denom is not None:
                        y = y / const_denom
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
            legend_handles = [
                mlines.Line2D([], [], color=GREY_CONST, linewidth=1.5, linestyle='--',
                              label='constant (pm1_n0)')
            ]
            for pm_val in sorted(all_pm_vals):
                pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
                legend_handles.append(
                    mlines.Line2D([], [], color=PM_COLOR[pm_val], linewidth=1.8,
                                  linestyle='-', label=f'$R_{{\\mathrm{{peak}}}}$ = {pr_label}')
                )
            fig.legend(
                handles=legend_handles, title='Peak / mean ratio',
                title_fontsize=9, fontsize=8, loc='lower center',
                ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.18), frameon=True,
            )
            fig.suptitle(
                f'Effect of $R_{{\\mathrm{{peak}}}}$ on width-averaged bed level{norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=11, fontweight='bold', y=1.02,
            )
            fig.tight_layout()
            fname = f'sensitivity_pm_effect_bedlevel{norm_tag}_{snap_label}_Q{DISCHARGE}.png'
            fig.savefig(sensitivity_output_dir / fname, dpi=200, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')

        # ---- Figure B: n-effect, one panel per pm ----
        sorted_pm_vals = sorted(n_by_pm.keys())
        if sorted_pm_vals:
            pm_panels = len(sorted_pm_vals)
            fig, axes = plt.subplots(
                1, pm_panels,
                figsize=(PANEL_W * pm_panels, PANEL_H),
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
                    ax.axhline(1.0, color=GREY_CONST, linewidth=1.5, linestyle='--',
                               label='constant (pm1_n0)', zorder=2)

                for n_val, scen_key in n_by_pm[pm_val]:
                    y = _get_y(scen_key)
                    if y is None:
                        continue
                    if normalise and const_denom is not None:
                        y = y / const_denom
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
            legend_handles = [
                mlines.Line2D([], [], color=GREY_CONST, linewidth=1.5, linestyle='--',
                              label='constant (pm1_n0)')
            ]
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
                f'Effect of $n_{{\\mathrm{{peaks}}}}$ on width-averaged bed level{norm_title}\n'
                f'Snapshot: {snap_label},  Q = {DISCHARGE} m³/s',
                fontsize=11, fontweight='bold', y=1.02,
            )
            fig.tight_layout()
            fname = f'sensitivity_n_effect_bedlevel{norm_tag}_{snap_label}_Q{DISCHARGE}.png'
            fig.savefig(sensitivity_output_dir / fname, dpi=200, bbox_inches='tight', transparent=True)
            plt.show()
            plt.close(fig)
            print(f'  Saved: {fname}')

print("\nDone.")
