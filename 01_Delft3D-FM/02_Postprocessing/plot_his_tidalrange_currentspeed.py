"""
Assess along-estuary tidal range from HIS output under high-flow and
low-flow conditions.
"""
#%%
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')
mpl.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor':'white',
    'text.color':       'black',
    'axes.labelcolor':  'black',
    'xtick.color':      'black',
    'ytick.color':      'black',
})

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_loaddata import read_discharge_from_bc_files
from FUNCTIONS.F_tidalrange_currentspeed import (
    cycle_metric,
    load_station_waterlevels_from_cache_or_his,
)


# =============================================================================
# CONFIG
# =============================================================================
DISCHARGE = 500
SCENARIOS_TO_PROCESS = None
OUTPUT_DIRNAME = 'plots_his_tidalrange_currentspeed'

WATERLEVEL_VAR = 'waterlevel'

# Station filter — longitudinal estuary stations only
STATION_PATTERN = r'^Observation(?:Point|CrossSection)_Estuary_km(\d+)$'

# Tidal-cycle settings
TIDAL_CYCLE_HOURS = 12
EXCLUDE_LAST_TIMESTEP = True

# Number of tidal cycles centred on each high/low flow event per year
N_WINDOW_CYCLES = 10

SHOW_FIGURES = False

SCENARIO_LABELS = {}
SCENARIO_COLORS = {}


# =============================================================================
# PATHS + RUN DISCOVERY
# =============================================================================
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config    = f"Model_Output/Q{DISCHARGE}"
base_path = base_directory / config

output_dir = base_path / 'output_plots' / OUTPUT_DIRNAME
output_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / 'timed-out'
if not timed_out_dir.exists():
    timed_out_dir = None

VARIABILITY_MAP = get_variability_map(DISCHARGE)
folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=False,
)
model_folders = [f.name for f in folders]

run_his_paths = {}
for folder in model_folders:
    model_location = base_path / folder
    his_paths      = []
    scenario_key   = str(int(folder.split('_')[0]))

    if timed_out_dir is not None:
        timed_out_folder = VARIABILITY_MAP.get(scenario_key, folder)
        timed_out_path   = timed_out_dir / timed_out_folder / 'output' / 'FlowFM_0000_his.nc'
        if timed_out_path.exists():
            his_paths.append(timed_out_path)

    main_his_path = model_location / 'output' / 'FlowFM_0000_his.nc'
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

cache_dir = base_path / 'cached_data'
cache_dir.mkdir(exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================
_PM_N_RE = re.compile(r"_pm(\d+(?:\.\d+)?)_n(\d+(?:\.\d+)?)", re.IGNORECASE)


def _parse_scenario_params(folder_name):
    """Return (peak_ratio, n_peaks) parsed from folder name, or (None, None)."""
    m = _PM_N_RE.search(folder_name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def _find_flow_windows(bc_times, bc_q, model_times, n_window_cycles, cycle_hours):
    """Return (high_windows, low_windows) as index-pair lists into model_times.

    One window per simulated year, centred on the annual discharge peak/minimum.
    """
    bc_t64   = np.array([np.datetime64(t, 'ns') for t in bc_times])
    bc_q_arr = np.array(bc_q, dtype=float)

    bc_t_s          = (bc_t64 - bc_t64[0]).astype('timedelta64[s]').astype(float)
    mod_t_s         = (model_times.astype('datetime64[ns]') - bc_t64[0]).astype('timedelta64[s]').astype(float)
    mod_t_s_clamped = np.clip(mod_t_s, bc_t_s[0], bc_t_s[-1])
    q_model = np.interp(mod_t_s_clamped, bc_t_s, bc_q_arr)

    dt_hours        = float((model_times[1] - model_times[0]) / np.timedelta64(1, 'h'))
    steps_per_cycle = max(1, int(np.round(cycle_hours / dt_hours)))
    window_size     = n_window_cycles * steps_per_cycle
    half_window     = window_size // 2
    n_total         = len(model_times)

    t_days   = (model_times - model_times[0]) / np.timedelta64(1, 'D')
    year_idx = (t_days / 365.25).astype(int)

    high_windows, low_windows = [], []
    for yr in np.unique(year_idx):
        yr_pos = np.where(year_idx == yr)[0]
        if len(yr_pos) < steps_per_cycle:
            continue
        yr_q        = q_model[yr_pos]
        centre_high = int(yr_pos[np.argmax(yr_q)])
        centre_low  = int(yr_pos[np.argmin(yr_q)])
        for centre, store in [(centre_high, high_windows), (centre_low, low_windows)]:
            start = max(0, centre - half_window)
            end   = min(n_total, start + window_size)
            if (end - start) >= steps_per_cycle:
                store.append((start, end))

    return high_windows, low_windows


def _yearly_profiles(times, matrix, windows, cycle_hours, reducer):
    """(n_years, n_x) array — one mean profile per window. None if empty."""
    profiles = []
    for start, end in windows:
        seg_t = times[start:end]
        seg_m = matrix[start:end, :]
        if len(seg_t) < 2:
            continue
        _, vals = cycle_metric(seg_t, seg_m, cycle_hours=cycle_hours, reducer=reducer)
        if vals.size > 0:
            profiles.append(np.nanmean(vals, axis=0))
    if not profiles:
        return None
    return np.vstack(profiles)


# =============================================================================
# LOAD + PROCESS
# =============================================================================
scenario_results = {}
tr_reducer = lambda seg: np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)

for folder in model_folders:
    scenario_key       = str(int(folder.split('_')[0]))
    run_id             = '_'.join(folder.split('_')[1:])
    station_cache_file = cache_dir / f"hisoutput_stations_{int(scenario_key)}_{run_id}.nc"
    his_file_paths     = run_his_paths.get(folder)

    if his_file_paths is None:
        continue

    print(f"\n[SCENARIO {scenario_key}] {folder}")

    # Prescribed discharge from BC files
    model_location = base_path / folder
    bc_times, bc_q = read_discharge_from_bc_files(
        model_location,
        bc_pattern="*_inflow_sinuous_Gaussian.bc",
    )
    if bc_times is None:
        print(f"  [WARNING] No BC discharge — skipping")
        continue

    # Water levels
    wl_data  = load_station_waterlevels_from_cache_or_his(
        station_cache_file,
        his_file_paths,
        waterlevel_var=WATERLEVEL_VAR,
        station_pattern=STATION_PATTERN,
        exclude_last_timestep=EXCLUDE_LAST_TIMESTEP,
    )
    wl_times = wl_data['times']
    wl       = wl_data['waterlevel']
    wl_km    = wl_data['station_km']

    # High/low flow windows
    high_windows, low_windows = _find_flow_windows(
        bc_times, bc_q, wl_times, N_WINDOW_CYCLES, TIDAL_CYCLE_HOURS,
    )
    print(f"  High-flow windows: {len(high_windows)} | Low-flow: {len(low_windows)}")

    # Tidal range profiles
    tr_high_yearly = _yearly_profiles(wl_times, wl, high_windows, TIDAL_CYCLE_HOURS, tr_reducer)
    tr_low_yearly  = _yearly_profiles(wl_times, wl, low_windows,  TIDAL_CYCLE_HOURS, tr_reducer)
    tr_high = None if tr_high_yearly is None else np.nanmean(tr_high_yearly, axis=0)
    tr_low  = None if tr_low_yearly  is None else np.nanmean(tr_low_yearly,  axis=0)

    peak_ratio, n_peaks = _parse_scenario_params(folder)

    scenario_results[scenario_key] = {
        'folder':         folder,
        'label':          SCENARIO_LABELS.get(scenario_key, scenario_key),
        'color':          SCENARIO_COLORS.get(scenario_key, 'grey'),
        'peak_ratio':     peak_ratio,
        'n_peaks':        n_peaks,
        'tr_km':          wl_km,
        'tr_high':        tr_high,
        'tr_low':         tr_low,
        'tr_high_yearly': tr_high_yearly,
        'tr_low_yearly':  tr_low_yearly,
    }


# =============================================================================
# PLOTS
# =============================================================================
if not scenario_results:
    raise RuntimeError('No scenarios processed. Check paths and filters.')


# ---- 1. Per-scenario: year x km heatmaps ------------------------------------
for scenario_key in sorted(scenario_results.keys()):
    d = scenario_results[scenario_key]

    for flow_label, tr_yearly in [
        ('high_flow', d['tr_high_yearly']),
        ('low_flow',  d['tr_low_yearly']),
    ]:
        if tr_yearly is None:
            continue

        n_years    = tr_yearly.shape[0]
        year_ticks = np.arange(1, n_years + 1)

        fig, ax = plt.subplots(figsize=(10, max(3, n_years * 0.4 + 2)))
        h = ax.pcolormesh(d['tr_km'], year_ticks, tr_yearly, shading='auto', cmap='viridis')
        plt.colorbar(h, ax=ax).set_label('Tidal range [m]')
        ax.set_xlabel('Estuary distance [km from sea]')
        ax.set_ylabel('Year')
        ax.set_title(
            f"{d['label']} — tidal range ({flow_label.replace('_', '-')}) (Q{DISCHARGE})"
        )
        fig.tight_layout()
        fig.savefig(
            output_dir / f"heatmap_yr_km_{flow_label}_Q{DISCHARGE}_{scenario_key}.png",
            dpi=300, bbox_inches='tight',
        )
        if SHOW_FIGURES:
            plt.show()
        else:
            plt.close(fig)


# ---- 2. Per-scenario: mean line profiles, high-flow vs low-flow -------------
for scenario_key in sorted(scenario_results.keys()):
    d = scenario_results[scenario_key]

    fig, ax = plt.subplots(figsize=(10, 5))
    if d['tr_high'] is not None:
        ax.plot(d['tr_km'], d['tr_high'], color='tab:red',  linewidth=1.8, label='High flow')
    if d['tr_low'] is not None:
        ax.plot(d['tr_km'], d['tr_low'],  color='tab:blue', linewidth=1.8, label='Low flow')
    ax.set_xlabel('Estuary distance [km from sea]')
    ax.set_ylabel('Tidal range [m]')
    ax.set_title(f"{d['label']} — mean tidal range, high-flow vs low-flow (Q{DISCHARGE})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"profiles_highlow_Q{DISCHARGE}_{scenario_key}.png",
        dpi=300, bbox_inches='tight',
    )
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ---- 3. Cross-scenario imshow: scenario x km --------------------------------
# Sort rows by (peak_ratio, n_peaks) — same convention as mass-balance heatmap.
def _sort_key(sk):
    d  = scenario_results[sk]
    pr = d['peak_ratio'] if d['peak_ratio'] is not None else 0.0
    np_ = d['n_peaks']   if d['n_peaks']   is not None else 0.0
    return (pr, np_)

sorted_keys = sorted(scenario_results.keys(), key=_sort_key)

# Common km grid: union of all station positions; interpolate outliers.
all_km = np.unique(np.concatenate([scenario_results[sk]['tr_km'] for sk in sorted_keys]))

def _interp_to_common(profile, src_km, dst_km):
    if profile is None:
        return np.full(len(dst_km), np.nan)
    return np.interp(dst_km, src_km, profile, left=np.nan, right=np.nan)

n_scen    = len(sorted_keys)
grid_low  = np.full((n_scen, len(all_km)), np.nan)
grid_high = np.full((n_scen, len(all_km)), np.nan)

for i, sk in enumerate(sorted_keys):
    d = scenario_results[sk]
    grid_low[i, :]  = _interp_to_common(d['tr_low'],  d['tr_km'], all_km)
    grid_high[i, :] = _interp_to_common(d['tr_high'], d['tr_km'], all_km)

grid_diff  = grid_low - grid_high   # positive = tidal prism expansion during low flow

def _row_label(sk):
    d  = scenario_results[sk]
    pr = d['peak_ratio']
    np_ = d['n_peaks']
    base = d['label']
    if pr is not None and np_ is not None:
        return f"{base}  pm{pr:.1f}  n{np_:.2f}"
    return base

row_labels = [_row_label(sk) for sk in sorted_keys]

fig_w = max(8, len(all_km) * 0.15 + 2)
fig_h = max(4, n_scen * 0.55 + 1.5)

for grid, title_suffix, fname_suffix, cmap, sym in [
    (grid_low,  'Low-flow tidal range',            'lowflow',  'viridis', False),
    (grid_high, 'High-flow tidal range',           'highflow', 'viridis', False),
    (grid_diff, 'Low-flow minus high-flow (delta TR)', 'contrast', 'RdBu_r', True),
]:
    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    if sym:
        _abs  = max(abs(vmin), abs(vmax))
        vmin, vmax = -_abs, _abs

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(
        grid,
        aspect='auto',
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        origin='upper',
        extent=[all_km[0], all_km[-1], n_scen - 0.5, -0.5],
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Tidal range [m]' if not sym else 'Delta tidal range [m]', fontsize=9)
    ax.set_yticks(range(n_scen))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel('Estuary distance [km from sea]', fontsize=10)
    ax.set_title(f"{title_suffix} — all scenarios (Q{DISCHARGE})", fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(
        output_dir / f"scenario_km_heatmap_{fname_suffix}_Q{DISCHARGE}.png",
        dpi=300, bbox_inches='tight',
    )
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


# ---- 4. Cross-scenario: low-flow profiles overlaid --------------------------
fig, ax = plt.subplots(figsize=(10, 5))
for scenario_key in sorted_keys:
    d = scenario_results[scenario_key]
    if d['tr_low'] is not None:
        ax.plot(d['tr_km'], d['tr_low'], color=d['color'], linewidth=1.5, label=d['label'])
ax.set_xlabel('Estuary distance [km from sea]')
ax.set_ylabel('Tidal range [m]')
ax.set_title(f"Low-flow tidal range — all scenarios (Q{DISCHARGE})")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(
    output_dir / f"lowflow_profiles_all_scenarios_Q{DISCHARGE}.png",
    dpi=300, bbox_inches='tight',
)
if SHOW_FIGURES:
    plt.show()
else:
    plt.close(fig)


print(f"\nSaved outputs in: {output_dir}")
#%%
