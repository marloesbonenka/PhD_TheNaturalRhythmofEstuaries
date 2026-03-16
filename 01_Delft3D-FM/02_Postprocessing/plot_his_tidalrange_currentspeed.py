"""
Assess along-estuary hydrodynamics from HIS output:
1) Tidal range from station water levels (per tidal cycle).
2) Current speed from cross-section velocity (per tidal cycle).
3) Water-surface slope from longitudinal station water levels.

This script follows the same scenario stitching logic as other HIS postprocessing
scripts in this folder.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Force white style for consistency with other scripts
plt.style.use('default')
mpl.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_tidalrange_currentspeed import (
    compute_cycle_windows,
    compute_slope_cm_per_km,
    cycle_metric,
    load_station_waterlevels,
    load_velocity_from_his_or_cache,
)


# =============================================================================
# CONFIG
# =============================================================================
DISCHARGE = 500
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']
OUTPUT_DIRNAME = 'plots_his_tidalrange_currentspeed'

# Variable names in HIS
VELOCITY_VAR = 'cross_section_velocity'
WATERLEVEL_VAR = 'waterlevel'

# Station filter (only longitudinal estuary stations used for slope/range)
STATION_PATTERN = r'^Observation(?:Point|CrossSection)_Estuary_km(\d+)$'

# Tidal-cycle settings
TIDAL_CYCLE_HOURS = 12
EXCLUDE_LAST_TIMESTEP = True

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


# =============================================================================
# PATHS + RUN DISCOVERY
# =============================================================================
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"
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
    his_paths = []
    scenario_key = str(int(folder.split('_')[0]))

    if timed_out_dir is not None:
        timed_out_folder = VARIABILITY_MAP.get(scenario_key, folder)
        timed_out_path = timed_out_dir / timed_out_folder / 'output' / 'FlowFM_0000_his.nc'
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
# LOAD + PROCESS
# =============================================================================
scenario_results = {}

for folder in model_folders:
    scenario_key = str(int(folder.split('_')[0]))
    run_id = '_'.join(folder.split('_')[1:])
    cache_file = cache_dir / f"hisoutput_{int(scenario_key)}_{run_id}.nc"
    his_file_paths = run_his_paths.get(folder)

    if his_file_paths is None:
        continue

    print(f"\n[SCENARIO {scenario_key}] {folder}")

    # 1) Velocity (cross-section)
    v_data = load_velocity_from_his_or_cache(
        cache_file,
        his_file_paths,
        velocity_var=VELOCITY_VAR,
        exclude_last_timestep=EXCLUDE_LAST_TIMESTEP,
    )
    v_times = v_data['times']
    v_km = np.array(v_data['km_positions'])
    v = v_data[VELOCITY_VAR].values

    # 2) Water level (stations)
    wl_data = load_station_waterlevels(
        his_file_paths,
        waterlevel_var=WATERLEVEL_VAR,
        station_pattern=STATION_PATTERN,
        exclude_last_timestep=EXCLUDE_LAST_TIMESTEP,
    )
    wl_times = wl_data['times']
    wl = wl_data['waterlevel']
    wl_km = wl_data['station_km']

    # 3) Per-tidal-cycle metrics
    tr_times, tidal_range = cycle_metric(
        wl_times,
        wl,
        cycle_hours=TIDAL_CYCLE_HOURS,
        reducer=lambda seg: np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0),
    )

    cs_times, current_speed = cycle_metric(
        v_times,
        np.abs(v),
        cycle_hours=TIDAL_CYCLE_HOURS,
        reducer=lambda seg: np.nanmax(seg, axis=0),
    )

    slope_windows = compute_cycle_windows(wl_times, cycle_hours=TIDAL_CYCLE_HOURS)
    slope_times = []
    slope_vals = []
    for start, end in slope_windows:
        slope_times.append(wl_times[start])
        slope_vals.append(compute_slope_cm_per_km(wl[start:end, :], wl_km))

    slope_times = np.array(slope_times)
    slope_vals = np.array(slope_vals)

    scenario_results[scenario_key] = {
        'label': SCENARIO_LABELS.get(scenario_key, scenario_key),
        'color': SCENARIO_COLORS.get(scenario_key, 'grey'),
        'tr_times': tr_times,
        'tidal_range': tidal_range,
        'tr_km': wl_km,
        'cs_times': cs_times,
        'current_speed': current_speed,
        'cs_km': v_km,
        'slope_times': slope_times,
        'slope_cm_per_km': slope_vals,
    }

    print(f"  Tidal range cycles: {len(tr_times)} | stations: {len(wl_km)}")
    print(f"  Speed cycles:       {len(cs_times)} | cross-sections: {len(v_km)}")


# =============================================================================
# PLOTS
# =============================================================================
if not scenario_results:
    raise RuntimeError('No scenarios processed. Check paths and filters.')

for scenario_key in sorted(scenario_results.keys()):
    d = scenario_results[scenario_key]

    # Convert time to years from first cycle in each metric
    tr_years = (d['tr_times'] - d['tr_times'][0]) / np.timedelta64(1, 'D') / 365.25
    cs_years = (d['cs_times'] - d['cs_times'][0]) / np.timedelta64(1, 'D') / 365.25
    slope_years = (d['slope_times'] - d['slope_times'][0]) / np.timedelta64(1, 'D') / 365.25

    # --- Heatmap: tidal range ---
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    h1 = ax1.pcolormesh(
        d['tr_km'],
        tr_years,
        d['tidal_range'],
        shading='auto',
        cmap='viridis',
    )
    cbar1 = plt.colorbar(h1, ax=ax1)
    cbar1.set_label('Tidal range [m]')
    ax1.set_xlabel('Estuary distance [km from sea]')
    ax1.set_ylabel('Simulation time [years]')
    ax1.set_title(f"{d['label']} - tidal range per tidal cycle")
    fig1.tight_layout()
    fig1.savefig(output_dir / f"tidal_range_heatmap_Q{DISCHARGE}_{scenario_key}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Heatmap: current speed ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    h2 = ax2.pcolormesh(
        d['cs_km'],
        cs_years,
        d['current_speed'],
        shading='auto',
        cmap='plasma',
    )
    cbar2 = plt.colorbar(h2, ax=ax2)
    cbar2.set_label('Max |cross-section velocity| per cycle [m/s]')
    ax2.set_xlabel('Estuary distance [km from sea]')
    ax2.set_ylabel('Simulation time [years]')
    ax2.set_title(f"{d['label']} - current speed per tidal cycle")
    fig2.tight_layout()
    fig2.savefig(output_dir / f"current_speed_heatmap_Q{DISCHARGE}_{scenario_key}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Time series: water-surface slope ---
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(slope_years, d['slope_cm_per_km'], color=d['color'], linewidth=1.2)
    ax3.axhline(0.0, color='k', linewidth=0.8, alpha=0.4)
    ax3.set_xlabel('Simulation time [years]')
    ax3.set_ylabel('Water-surface slope [cm/km]')
    ax3.set_title(f"{d['label']} - cycle-mean water-surface slope")
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(output_dir / f"water_surface_slope_Q{DISCHARGE}_{scenario_key}.png", dpi=300, bbox_inches='tight')
    plt.show()


# --- Cross-scenario comparison: slope ---
fig, ax = plt.subplots(figsize=(10, 5))
for scenario_key in sorted(scenario_results.keys()):
    d = scenario_results[scenario_key]
    y = d['slope_cm_per_km']
    x = (d['slope_times'] - d['slope_times'][0]) / np.timedelta64(1, 'D') / 365.25
    ax.plot(x, y, color=d['color'], linewidth=1.2, label=d['label'])

ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.4)
ax.set_xlabel('Simulation time [years]')
ax.set_ylabel('Water-surface slope [cm/km]')
ax.set_title(f"Cycle-mean water-surface slope comparison (Q{DISCHARGE})")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / f"water_surface_slope_comparison_Q{DISCHARGE}.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSaved outputs in: {output_dir}")
#%%