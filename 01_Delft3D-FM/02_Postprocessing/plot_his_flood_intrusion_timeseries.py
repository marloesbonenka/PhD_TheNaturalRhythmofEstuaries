"""
Plot the maximum flood intrusion point (estuary x-coordinate) over simulation time,
one line per discharge variability scenario.

Uses the same loading approach as plot_his_along_estuary_tidalriverdominance:
load_cross_section_data_from_cache with select_max_flood_per_cycle=True, which
selects one timestep per tidal day (the moment of deepest flood penetration) and
auto-detects the flood sign convention.
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib as mpl
import sys
from pathlib import Path
import pandas as pd

# ─── MOVING AVERAGE WINDOW ───────────────────────────────────────────────────
MA_PERIOD = 'yearly'   # <-- change this: 'daily' | 'weekly' | 'monthly' | 'yearly'
# ─────────────────────────────────────────────────────────────────────────────

PERIOD_YEARS = {'daily': 1/365, 'weekly': 1/52, 'monthly': 1/12, 'yearly': 1.0}
period_years = PERIOD_YEARS[MA_PERIOD]

# ── FORCE WHITE STYLE (overrides any dark_background set globally) ─────────────
plt.style.use('default')
mpl.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'savefig.facecolor': 'white',
    'text.color':        'black',
    'axes.labelcolor':   'black',
    'xtick.color':       'black',
    'ytick.color':       'black',
})

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import (
    load_cross_section_data, load_cross_section_data_from_cache,
)
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
dis_var        = 'cross_section_discharge'
output_dirname = "plots_his_max_floodintrusion"

SCENARIOS_TO_PROCESS = None
DISCHARGE            = 500

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

# Constant scenario colour (same as bedlevel script)
GREY_CONST = '#7f7f7f'

# ── PATHS ──────────────────────────────────────────────────────────────────────
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config         = f"Model_Output/Q{DISCHARGE}"
base_path      = base_directory / config

output_dir = base_path / 'output_plots' / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / "timed-out"
if not timed_out_dir.exists():
    timed_out_dir = None

VARIABILITY_MAP = get_variability_map(DISCHARGE)

# ── FIND RUN FOLDERS ───────────────────────────────────────────────────────────
folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=False,
)
model_folders = [f.name for f in folders]

# ── BUILD HIS FILE PATH MAP ────────────────────────────────────────────────────
run_his_paths = {}
for folder in model_folders:
    model_location = base_path / folder
    his_paths      = []
    scenario_key   = str(int(folder.split('_')[0]))

    if timed_out_dir is not None:
        timed_out_folder = VARIABILITY_MAP.get(scenario_key, folder)
        timed_out_path   = timed_out_dir / timed_out_folder / "output" / "FlowFM_0000_his.nc"
        if timed_out_path.exists():
            his_paths.append(timed_out_path)

    main_his_path = model_location / "output" / "FlowFM_0000_his.nc"
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

# ── CACHE DIR ─────────────────────────────────────────────────────────────────
cache_dir = base_path / "cached_data"
cache_dir.mkdir(exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
# Load one timestep per tidal day (moment of deepest flood penetration).
# The flood sign is auto-detected inside load_cross_section_data_from_cache,
# matching exactly what plot_his_along_estuary_tidalriverdominance does.

def _load(cache_file, his_file_paths):
    kwargs = dict(q_var=dis_var, select_max_flood_per_cycle=True, exclude_last_timestep=True)
    if cache_file is not None and cache_file.exists():
        print(f"  [cache] {cache_file.name}")
        return load_cross_section_data_from_cache(cache_file, **kwargs)
    else:
        return load_cross_section_data(his_file_paths, **kwargs)


scenario_data = {}
for folder in model_folders:
    scenario_key  = str(int(folder.split('_')[0]))
    scenario_name = folder
    run_id        = '_'.join(folder.split('_')[1:])
    cache_file    = cache_dir / f"hisoutput_{int(scenario_key)}_{run_id}.nc"
    his_file_paths = run_his_paths.get(folder)
    if his_file_paths is None:
        continue

    data = _load(cache_file, his_file_paths)
    scenario_data[scenario_key] = data
    print(f"Scenario {scenario_key} ({SCENARIO_LABELS.get(scenario_key, '')}): "
          f"{data['n_timesteps']} tidal cycles loaded, flood_sign={data['flood_sign_used']}")

# ── COLORMAPS  (Blues = R_peak effect, Greens = n_peaks effect) ───────────────
def _parse_pm_n(label_str):
    """Extract (pm, n) ints from a label like 'pm3_n5' or 'pm1_n0 (constant)'."""
    m = re.match(r'pm(\d+)_n(\d+)', label_str.strip())
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

scen_pm_n = {}
for _sk in scenario_data:
    _label = SCENARIO_LABELS.get(_sk, _sk)
    _pm, _n = _parse_pm_n(_label)
    if _pm is not None:
        scen_pm_n[_sk] = (_pm, _n)

all_pm_vals = sorted({pm for pm, n in scen_pm_n.values() if n > 0})
all_n_vals  = sorted({n  for pm, n in scen_pm_n.values() if n > 0})
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}
_n_n = max(len(all_n_vals) - 1, 1)
N_COLOR  = {n:  plt.cm.Greens(0.35 + 0.55 * i / _n_n) for i, n  in enumerate(all_n_vals)}

# Group scenarios for panel plots (same logic as bedlevel script)
baseline_scen = next((k for k, (pm, n) in scen_pm_n.items() if n == 0), None)

pm_by_n = {}   # {n_val: [(pm_val, scen_key), ...]}
n_by_pm = {}   # {pm_val: [(n_val, scen_key), ...]}
for _sk, (_pm, _n) in scen_pm_n.items():
    if _n == 0:
        continue
    pm_by_n.setdefault(_n, []).append((_pm, _sk))
    n_by_pm.setdefault(_pm, []).append((_n, _sk))
for _n in pm_by_n:
    pm_by_n[_n].sort()
for _pm in n_by_pm:
    n_by_pm[_pm].sort()

# ── HELPER ────────────────────────────────────────────────────────────────────
def compute_max_flood_km(q, km_positions, flood_sign):
    """
    For each timestep, return the km of the most landward cross-section in
    flood conditions. Returns (n_time,) array; NaN where no flooding occurs.
    """
    flood_mask = q > 0 if flood_sign > 0 else q < 0
    km_grid    = np.broadcast_to(km_positions, q.shape)
    flood_km   = np.where(flood_mask, km_grid, np.nan)
    return np.nanmax(flood_km, axis=1)

# ── PRE-PROCESS ───────────────────────────────────────────────────────────────
processed = {}

for scenario_key, data in scenario_data.items():
    km_positions   = data['km_positions']
    flood_sign     = data['flood_sign_used']
    q              = data[dis_var].values    # DataArray → (n_cycles, n_km)
    t              = data['times']           # datetime64, one per tidal cycle

    max_flood_km = compute_max_flood_km(q, km_positions, flood_sign=flood_sign)

    nan_frac = np.isnan(max_flood_km).mean()
    if nan_frac > 0.05:
        print(f"  [WARNING] Scenario {scenario_key}: {nan_frac*100:.1f}% cycles have no flood conditions")

    # Simulation time in years from start of series
    t_years = (t - t[0]) / np.timedelta64(1, 'D') / 365.25

    _pm_val, _n_val = scen_pm_n.get(scenario_key, (None, None))
    _is_const = (_n_val == 0)
    processed[scenario_key] = dict(
        label        = SCENARIO_LABELS.get(scenario_key, scenario_key),
        pm_color     = GREY_CONST if _is_const else PM_COLOR.get(_pm_val, 'grey'),
        n_color      = GREY_CONST if _is_const else N_COLOR.get(_n_val, 'grey'),
        t_years      = t_years,
        max_flood_km = max_flood_km,
    )
#%%
# ── PLOT ──────────────────────────────────────────────────────────────────────
for color_mode in ['pm', 'n']:
    color_key   = f'{color_mode}_color'
    color_title = 'R_peak' if color_mode == 'pm' else 'n_peaks'
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario_key in sorted(processed.keys()):
        d = processed[scenario_key]
        ax.plot(d['max_flood_km'], d['t_years'],
                color=d[color_key], linewidth=1.2,
                label=d['label'])

    ax.set_xlim(20, 45)
    ax.set_xlabel('flood intrusion (x-coordinate along estuary)', fontsize=11)
    ax.set_ylabel('years', fontsize=11)
    ax.set_title(
        f'Maximum flood intrusion point over time  (Q{DISCHARGE}, daily max-flood per tidal cycle)\n'
        f'coloured by {color_title}',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = f"flood_intrusion_km_over_time_Q{DISCHARGE}_{color_mode}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {output_dir / fname}")

#%%
# ── PER-SCENARIO PLOTS ────────────────────────────────────────────────────────
scenario_keys = sorted(processed.keys())
for scenario_key in scenario_keys:
    d = processed[scenario_key]

    t = np.array(d['t_years'])
    flood = np.array(d['max_flood_km'])
    dt = np.median(np.diff(t))
    window = max(1, round(period_years / dt))
    flood_ma = pd.Series(flood).rolling(window=window, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(flood, t, color=d['pm_color'], linewidth=0.5, alpha=0.3, label='Raw')
    ax.plot(flood_ma, t, color=d['pm_color'], linewidth=1.8, label=f'{MA_PERIOD.capitalize()} MA')

    ax.set_xlim(20, 45)
    ax.set_xlabel('flood intrusion (x-coordinate along estuary)', fontsize=11)
    ax.set_ylabel('years', fontsize=11)
    ax.set_title(
        f'{d["label"]}  —  Q{DISCHARGE}  ({MA_PERIOD} moving average)',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fname_s = f"flood_intrusion_km_over_time_Q{DISCHARGE}_{scenario_key}.png"
    fig.savefig(output_dir / fname_s, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {output_dir / fname_s}")

# %%
# ── PANEL FIGURE A: Effect of R_peak  (one panel per n_peaks, coloured by pm) ──
_d_const = processed.get(baseline_scen)
sorted_n_vals = sorted(pm_by_n.keys())
if sorted_n_vals:
    n_panels = len(sorted_n_vals)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 8),
                             sharey=True, sharex=True)
    if n_panels == 1:
        axes = [axes]

    for ci, n_val in enumerate(sorted_n_vals):
        ax = axes[ci]

        # grey dashed constant reference
        if _d_const is not None:
            _t = np.array(_d_const['t_years'])
            _f = np.array(_d_const['max_flood_km'])
            _dt = np.median(np.diff(_t))
            _win = max(1, round(period_years / _dt))
            _ma = pd.Series(_f).rolling(window=_win, center=True, min_periods=1).mean()
            ax.plot(_ma, _t, color=GREY_CONST, linewidth=1.5, linestyle='--',
                    label='constant (pm1_n0)', zorder=2)

        for pm_val, scen_key in pm_by_n[n_val]:
            d = processed.get(scen_key)
            if d is None:
                continue
            t = np.array(d['t_years'])
            flood = np.array(d['max_flood_km'])
            dt = np.median(np.diff(t))
            window = max(1, round(period_years / dt))
            flood_ma = pd.Series(flood).rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(flood, t, color=PM_COLOR[pm_val], linewidth=0.5, alpha=0.3, zorder=1)
            ax.plot(flood_ma, t, color=PM_COLOR[pm_val], linewidth=1.8,
                    label=f'$R_{{\\mathrm{{peak}}}}$ = {int(pm_val)}', zorder=3)

        ax.set_title(f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}', fontsize=13, fontweight='bold')
        ax.set_xlim(20, 45)
        ax.set_xlabel('flood intrusion [km]', fontsize=11)
        ax.grid(alpha=0.3)
        if ci == 0:
            ax.set_ylabel('years', fontsize=11)

    legend_handles = [
        mlines.Line2D([], [], color=GREY_CONST, linewidth=1.5, linestyle='--',
                      label='constant (pm1_n0)')
    ]
    for pm_val in sorted(all_pm_vals):
        legend_handles.append(
            mlines.Line2D([], [], color=PM_COLOR[pm_val], linewidth=1.8,
                          label=f'$R_{{\\mathrm{{peak}}}}$ = {int(pm_val)}')
        )
    fig.legend(handles=legend_handles, fontsize=10, loc='lower center',
               ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.suptitle(
        f'Effect of $R_{{\\mathrm{{peak}}}}$ on flood intrusion  (Q{DISCHARGE}, {MA_PERIOD} MA)',
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    fname = f"flood_intrusion_panels_pm_effect_{MA_PERIOD}_Q{DISCHARGE}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {output_dir / fname}")

# ── PANEL FIGURE B: Effect of n_peaks  (one panel per R_peak, coloured by n) ──
sorted_pm_vals = sorted(n_by_pm.keys())
if sorted_pm_vals:
    pm_panels = len(sorted_pm_vals)
    fig, axes = plt.subplots(1, pm_panels, figsize=(5 * pm_panels, 8),
                             sharey=True, sharex=True)
    if pm_panels == 1:
        axes = [axes]

    for ci, pm_val in enumerate(sorted_pm_vals):
        ax = axes[ci]

        if _d_const is not None:
            _t = np.array(_d_const['t_years'])
            _f = np.array(_d_const['max_flood_km'])
            _dt = np.median(np.diff(_t))
            _win = max(1, round(period_years / _dt))
            _ma = pd.Series(_f).rolling(window=_win, center=True, min_periods=1).mean()
            ax.plot(_ma, _t, color=GREY_CONST, linewidth=1.5, linestyle='--',
                    label='constant (pm1_n0)', zorder=2)

        for n_val, scen_key in n_by_pm[pm_val]:
            d = processed.get(scen_key)
            if d is None:
                continue
            t = np.array(d['t_years'])
            flood = np.array(d['max_flood_km'])
            dt = np.median(np.diff(t))
            window = max(1, round(period_years / dt))
            flood_ma = pd.Series(flood).rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(flood, t, color=N_COLOR[n_val], linewidth=0.5, alpha=0.3, zorder=1)
            ax.plot(flood_ma, t, color=N_COLOR[n_val], linewidth=1.8,
                    label=f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}', zorder=3)

        ax.set_title(f'$R_{{\\mathrm{{peak}}}}$ = {int(pm_val)}', fontsize=13, fontweight='bold')
        ax.set_xlim(20, 45)
        ax.set_xlabel('flood intrusion [km]', fontsize=11)
        ax.grid(alpha=0.3)
        if ci == 0:
            ax.set_ylabel('years', fontsize=11)

    legend_handles = [
        mlines.Line2D([], [], color=GREY_CONST, linewidth=1.5, linestyle='--',
                      label='constant (pm1_n0)')
    ]
    for n_val in sorted(all_n_vals):
        legend_handles.append(
            mlines.Line2D([], [], color=N_COLOR[n_val], linewidth=1.8,
                          label=f'$n_{{\\mathrm{{peaks}}}}$ = {n_val}')
        )
    fig.legend(handles=legend_handles, fontsize=10, loc='lower center',
               ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.suptitle(
        f'Effect of $n_{{\\mathrm{{peaks}}}}$ on flood intrusion  (Q{DISCHARGE}, {MA_PERIOD} MA)',
        fontsize=13, fontweight='bold', y=1.01,
    )
    fig.tight_layout()
    fname = f"flood_intrusion_panels_n_effect_{MA_PERIOD}_Q{DISCHARGE}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {output_dir / fname}")

# %%
# ── SEPARATE P50 PLOT ────────────────────────────────────────────────────────
for color_mode in ['pm', 'n']:
    color_key   = f'{color_mode}_color'
    color_title = 'R_peak' if color_mode == 'pm' else 'n_peaks'
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario_key in sorted(processed.keys()):
        d = processed[scenario_key]

        t = np.array(d['t_years'])
        flood = np.array(d['max_flood_km'])

        dt = np.median(np.diff(t))
        window = max(1, round(period_years / dt))

        flood_p50 = pd.Series(flood).rolling(window=window, center=True, min_periods=1).quantile(0.5)

        ax.plot(flood, 100 * t, color=d[color_key], linewidth=0.5, alpha=0.3)
        ax.plot(flood_p50, 100 * t, color=d[color_key], linewidth=2.0, linestyle='--', label=d['label'])

    ax.set_xlim(19.5, 45)
    ax.set_ylim(0, 3100)
    ax.set_xlabel('flood intrusion (x-coordinate along estuary)', fontsize=11)
    ax.set_ylabel('years', fontsize=11)
    ax.set_title(
        f'Maximum flood intrusion point over time  (Q{DISCHARGE}, {MA_PERIOD} rolling p50)\n'
        f'coloured by {color_title}',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = f"moving_p50_{MA_PERIOD}_flood_intrusion_km_over_time_Q{DISCHARGE}_{color_mode}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {output_dir / fname}")
# %%
# ── SEPARATE P40 PLOT ────────────────────────────────────────────────────────
for color_mode in ['pm', 'n']:
    color_key   = f'{color_mode}_color'
    color_title = 'R_peak' if color_mode == 'pm' else 'n_peaks'
    fig, ax = plt.subplots(figsize=(10, 10))

    for scenario_key in sorted(processed.keys()):
        d = processed[scenario_key]

        t = np.array(d['t_years'])
        flood = np.array(d['max_flood_km'])

        dt = np.median(np.diff(t))
        window = max(1, round(period_years / dt))

        flood_p40 = pd.Series(flood).rolling(window=window, center=True, min_periods=1).quantile(0.4)

        ax.plot(flood, t, color=d[color_key], linewidth=0.5, alpha=0.3)
        ax.plot(flood_p40, t, color=d[color_key], linewidth=2.0, linestyle='--', label=d['label'])

    ax.set_xlim(20, 45)
    ax.set_xlabel('flood intrusion (x-coordinate along estuary)', fontsize=11)
    ax.set_ylabel('years', fontsize=11)
    ax.set_title(
        f'Maximum flood intrusion point over time  (Q{DISCHARGE}, {MA_PERIOD} rolling p40)\n'
        f'coloured by {color_title}',
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fname = f"moving_p40_{MA_PERIOD}_flood_intrusion_km_over_time_Q{DISCHARGE}_{color_mode}.png"
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {output_dir / fname}")
# %%
