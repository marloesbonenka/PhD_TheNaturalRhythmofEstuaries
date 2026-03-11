"""
Plot the maximum flood intrusion point (estuary x-coordinate) over simulation time,
one line per discharge variability scenario.

Uses the same loading approach as plot_his_along_estuary_tidalriverdominance:
load_cross_section_data_from_cache with select_max_flood_per_cycle=True, which
selects one timestep per tidal day (the moment of deepest flood penetration) and
auto-detects the flood sign convention.
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path

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

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
dis_var        = 'cross_section_discharge'
output_dirname = "plots_his_max_floodintrusion"

SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']
DISCHARGE            = 500

SCENARIO_LABELS = {
    '1': 'Constant',
    '2': 'Seasonal',
    '3': 'Flashy',
    '4': 'Single peak',
}
SCENARIO_COLORS = {
    '1': '#1f77b4',
    '2': '#ff7f0e',
    '3': '#2ca02c',
    '4': '#d62728',
}

# ── PATHS ──────────────────────────────────────────────────────────────────────
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config         = f"Model_Output/Q{DISCHARGE}"
base_path      = base_directory / config

output_dir = base_path / 'output_plots' / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / "timed-out"
if not timed_out_dir.exists():
    timed_out_dir = None

VARIABILITY_MAP = {
    '1': f'01_baserun{DISCHARGE}',
    '2': f'02_run{DISCHARGE}_seasonal',
    '3': f'03_run{DISCHARGE}_flashy',
    '4': f'04_run{DISCHARGE}_singlepeak',
}

# ── FIND RUN FOLDERS ───────────────────────────────────────────────────────────
model_folders = [f.name for f in base_path.iterdir()
                 if f.is_dir() and f.name[0].isdigit()]
if SCENARIOS_TO_PROCESS:
    scenario_filter = set(int(s) for s in SCENARIOS_TO_PROCESS)
    model_folders = [f for f in model_folders if int(f.split('_')[0]) in scenario_filter]
model_folders.sort(key=lambda x: int(x.split('_')[0]))

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

    processed[scenario_key] = dict(
        label        = SCENARIO_LABELS.get(scenario_key, scenario_key),
        color        = SCENARIO_COLORS.get(scenario_key, 'grey'),
        t_years      = t_years,
        max_flood_km = max_flood_km,
    )
#%%
# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))

for scenario_key in sorted(processed.keys()):
    d = processed[scenario_key]
    ax.plot(d['max_flood_km'], d['t_years'], 
            color=d['color'], linewidth=1.2,
            label=d['label'])
    
ax.set_xlim(20,45)
ax.set_xlabel('Flood intrusion (km from sea)', fontsize=11)
ax.set_ylabel('Simulation time (years)', fontsize=11)
ax.set_title(
    f'Maximum flood intrusion point over time  (Q{DISCHARGE}, daily max-flood per tidal cycle)',
    fontsize=12
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

fig.tight_layout()
fname = f"flood_intrusion_km_over_time_Q{DISCHARGE}.png"
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

    ax.plot(flood, t, color=d['color'], linewidth=0.5, alpha=0.3, label='Raw')
    ax.plot(flood_ma, t, color=d['color'], linewidth=1.8, label=f'{MA_PERIOD.capitalize()} MA')

    ax.set_xlim(20, 45)
    ax.set_xlabel('Flood intrusion (km from sea)', fontsize=11)
    ax.set_ylabel('Simulation time (years)', fontsize=11)
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
import pandas as pd

# ─── MOVING AVERAGE WINDOW ───────────────────────────────────────────────────
MA_PERIOD = 'yearly'   # <-- change this: 'daily' | 'weekly' | 'monthly' | 'yearly'
# ─────────────────────────────────────────────────────────────────────────────

PERIOD_YEARS = {'daily': 1/365, 'weekly': 1/52, 'monthly': 1/12, 'yearly': 1.0}
period_years = PERIOD_YEARS[MA_PERIOD]

fig, ax = plt.subplots(figsize=(10, 10))

for scenario_key in sorted(processed.keys()):
    d = processed[scenario_key]

    t = np.array(d['t_years'])
    flood = np.array(d['max_flood_km'])

    dt = np.median(np.diff(t))
    window = max(1, round(period_years / dt))

    flood_ma = pd.Series(flood).rolling(window=window, center=True, min_periods=1).mean()

    ax.plot(flood, t, color=d['color'], linewidth=0.5, alpha=0.3)
    ax.plot(flood_ma, t, color=d['color'], linewidth=1.8, label=d['label'])

ax.set_xlim(20, 45)
ax.set_xlabel('Flood intrusion (km from sea)', fontsize=11)
ax.set_ylabel('Simulation time (years)', fontsize=11)
ax.set_title(
    f'Maximum flood intrusion point over time  (Q{DISCHARGE}, {MA_PERIOD} moving average)',
    fontsize=12
)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

fig.tight_layout()
fname = f"moving_avg_{MA_PERIOD}_flood_intrusion_km_over_time_Q{DISCHARGE}.png"
fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
plt.show()
print(f"\nSaved: {output_dir / fname}")
# %%
