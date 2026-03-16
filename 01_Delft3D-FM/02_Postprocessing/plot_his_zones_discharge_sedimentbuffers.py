"""
Plot sediment buffer volumes along estuary.
Divide estuary in sections based on distance from sea to river,
extract sediment transport through cross sections at the boundaries of these sections,
plot the cumulative sediment transport (buffer volume) over time for each section,
the instantaneous rate of change, tidal-averaged residual, trapping efficiency,
system sensitivity (dV/dt vs Q_river), and recovery time (cross-correlation).

Sensitivity and recovery plots: one figure per km section, all scenarios overlaid.
"""

# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path
import re

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import load_cross_section_data, load_and_cache_scenario
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_sedimentbuffers import (
    tidal_avg,
    load_sedimentbuffer_runs,
    align_runs_to_common_time,
    trim_to_end,
)

# %% --- CONFIGURATION ---
# Workflow:
# - "full": original detailed per-scenario and sensitivity analysis
# - "envelope": noisy-envelope + variability comparison plots
WORKFLOW_MODE = "full"

sed_var   = 'cross_section_bedload_sediment_transport'
dis_var   = 'cross_section_discharge'
output_dirname = "plots_his_sedimentbuffer"

mpl.rcParams['figure.figsize'] = (8, 6)

# Cross-section km boundaries for buffer boxes
box_edges = np.arange(20, 50, 5)   # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i + 1]) for i in range(len(box_edges) - 1)]

# River boundary cross-section (most upstream, used as forcing signal)
RIVER_KM = 44

# Tidal averaging window (hours) — one tidal cycle
TIDAL_WINDOW_HOURS = 24

# Maximum lag for cross-correlation recovery plot (days)
MAX_LAG_DAYS = 365

# Output timestep of the HIS file (hours) — adjust if not 1h
DT_HOURS = 1                        #3600 seconds = 1 hour

# Number of timesteps to skip at the start (model spin-up)
SPINUP_STEPS = 24 * 3  # 3 days, adjust based on your output interval

# Scenario filters
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']
DISCHARGE   = 1000
ANALYZE_NOISY = False

# Envelope mode settings
BASE_SCENARIO = '1'
ENVELOPE_PLOT_MODE = 'all'  # 'noise_only' or 'all'

# Human-readable labels per scenario number (used in combined plots)
SCENARIO_LABELS = {
    '1': 'Constant',
    '2': 'Seasonal',
    '3': 'Flashy',
    '4': 'Single peak',
}

# Colours: one per scenario (used in combined sensitivity / recovery plots)
SCENARIO_COLORS = {
    '1': '#1f77b4',   # blue   – Constant
    '2': '#ff7f0e',   # orange – Seasonal
    '3': '#2ca02c',   # green  – Flashy
    '4': '#d62728',   # red    – Single peak
}

# %% --- PATHS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"
if ANALYZE_NOISY:
    base_path = base_directory / config / f"0_Noise_Q{DISCHARGE}"
else:
    base_path = base_directory / config

output_dir = base_path / 'output_plots' / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / "timed-out"
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")
if not timed_out_dir.exists():
    timed_out_dir = None
    print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')

VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=ANALYZE_NOISY,
)

print(f"Found {len(model_folders)} run folders in: {base_path}")


def run_envelope_workflow():
    """Run noisy-envelope plots (previously in plot_his_sedimentbalance_variability_envelope.py)."""
    scenario_config = {
        "1": {"color": SCENARIO_COLORS["1"], "label": SCENARIO_LABELS["1"]},
        "2": {"color": SCENARIO_COLORS["2"], "label": SCENARIO_LABELS["2"]},
        "3": {"color": SCENARIO_COLORS["3"], "label": SCENARIO_LABELS["3"]},
        "4": {"color": SCENARIO_COLORS["4"], "label": SCENARIO_LABELS["4"]},
    }

    variability_base_path = base_directory / config
    variability_cache_dir = variability_base_path / "cached_data"
    noisy_base_path = variability_base_path / f"0_Noise_Q{DISCHARGE}"
    noisy_cache_dir = noisy_base_path / "cached_data"

    if ENVELOPE_PLOT_MODE == 'noise_only':
        envelope_output_dir = noisy_base_path / output_dirname
    else:
        envelope_output_dir = variability_base_path / output_dirname
    envelope_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Envelope output dir: {envelope_output_dir}")

    base_runs = load_sedimentbuffer_runs(
        base_path=variability_base_path,
        cache_dir=variability_cache_dir,
        discharge=DISCHARGE,
        variability_map=VARIABILITY_MAP,
        boxes=boxes,
        var_name=sed_var,
        analyze_noisy=False,
        scenario_filter={int(BASE_SCENARIO)},
    )
    if not base_runs:
        raise FileNotFoundError(
            f"No run found for base scenario {BASE_SCENARIO} in {variability_base_path}."
        )

    base_data = list(base_runs.values())[0]
    base_time = base_data['time']
    base_buffers = base_data['buffers']
    base_cfg = scenario_config[BASE_SCENARIO]

    variability_runs = {}
    if ENVELOPE_PLOT_MODE == 'all':
        other_nums = {int(k) for k in scenario_config if k != BASE_SCENARIO}
        all_var_runs = load_sedimentbuffer_runs(
            base_path=variability_base_path,
            cache_dir=variability_cache_dir,
            discharge=DISCHARGE,
            variability_map=VARIABILITY_MAP,
            boxes=boxes,
            var_name=sed_var,
            analyze_noisy=False,
            scenario_filter=other_nums,
        )
        for folder_name, run_data in all_var_runs.items():
            scenario_num = str(int(folder_name.split('_')[0]))
            if scenario_num not in scenario_config:
                continue
            variability_runs[scenario_num] = {
                'time': run_data['time'],
                'buffers': run_data['buffers'],
                'label': scenario_config[scenario_num]['label'],
                'color': scenario_config[scenario_num]['color'],
            }

    if noisy_base_path.exists():
        noisy_runs = load_sedimentbuffer_runs(
            base_path=noisy_base_path,
            cache_dir=noisy_cache_dir,
            discharge=DISCHARGE,
            variability_map=VARIABILITY_MAP,
            boxes=boxes,
            var_name=sed_var,
            analyze_noisy=True,
            scenario_filter=None,
        )
    else:
        noisy_runs = {}
        print(f"[INFO] No noisy base path found: {noisy_base_path}")

    print(f"Found {len(noisy_runs)} noisy runs")

    if ENVELOPE_PLOT_MODE == 'noise_only' and not noisy_runs:
        raise FileNotFoundError(
            f"No noisy runs found in {noisy_base_path}. "
            f"Set ENVELOPE_PLOT_MODE='all' or add noisy runs."
        )
    if ENVELOPE_PLOT_MODE == 'all' and not noisy_runs:
        print("[INFO] No noisy runs found; plotting base + variability scenarios without noisy envelope.")

    all_times = [base_time]
    all_times += [r['time'] for r in noisy_runs.values()]
    all_times += [r['time'] for r in variability_runs.values()]
    t_end_min = min(t[-1] for t in all_times)
    print(f"Shortest simulation end time across all runs: {t_end_min}")

    for box_key in boxes:
        box_start, box_end = box_key
        fig, ax = plt.subplots()

        if noisy_runs:
            base_time_trimmed = base_time[trim_to_end(base_time, t_end_min)]
            noisy_stack = align_runs_to_common_time(base_time_trimmed, noisy_runs, box_key)

            base_buf_for_env = base_buffers.get(box_key)
            if base_buf_for_env is not None:
                base_row = base_buf_for_env[trim_to_end(base_time, t_end_min)][np.newaxis, :]
                if noisy_stack.size > 0:
                    noisy_stack = np.vstack([noisy_stack, base_row])
                else:
                    noisy_stack = base_row

            if noisy_stack.size > 0:
                env_min = np.nanmin(noisy_stack, axis=0)
                env_max = np.nanmax(noisy_stack, axis=0)

                for i, run_data in enumerate(noisy_runs.values()):
                    buf = run_data['buffers'].get(box_key)
                    if buf is None:
                        continue
                    mask = trim_to_end(run_data['time'], t_end_min)
                    ax.plot(
                        run_data['time'][mask],
                        buf[mask],
                        color='grey',
                        alpha=0.35,
                        linewidth=0.7,
                        label='Noisy runs' if i == 0 else None,
                    )

                ax.fill_between(
                    base_time_trimmed,
                    env_min,
                    env_max,
                    color='grey',
                    alpha=0.2,
                    label='Noisy envelope',
                )

        if ENVELOPE_PLOT_MODE == 'all':
            for run_data in variability_runs.values():
                buf = run_data['buffers'].get(box_key)
                if buf is None:
                    continue
                mask = trim_to_end(run_data['time'], t_end_min)
                ax.plot(
                    run_data['time'][mask],
                    buf[mask],
                    color=run_data['color'],
                    linewidth=1.2,
                    label=run_data['label'],
                )

        base_buf = base_buffers.get(box_key)
        if base_buf is not None:
            mask = trim_to_end(base_time, t_end_min)
            ax.plot(
                base_time[mask],
                base_buf[mask],
                color=base_cfg['color'],
                linewidth=1.8,
                label=base_cfg['label'],
            )

        ax.set_xlabel('Time')
        ax.set_ylabel('Buffer Volume (m³)')
        ax.set_title(f'Sediment buffer volume — {box_start}–{box_end} km')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = envelope_output_dir / f"Q{DISCHARGE}_sedimentbuffer_box_{box_start}_{box_end}km.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.show()


if WORKFLOW_MODE == "envelope":
    run_envelope_workflow()
    raise SystemExit(0)

# %% --- BUILD HIS FILE PATH MAP ---
run_his_paths = {}
for folder in model_folders:
    model_location = base_path / folder
    his_paths = []
    scenario_num = folder.name.split('_')[0]
    try:
        scenario_key = str(int(scenario_num))
    except Exception:
        scenario_key = scenario_num

    if ANALYZE_NOISY:
        match = re.search(r'noisy(\d+)', folder.name)
        timed_out_folder = None
        if timed_out_dir is None:
            print('[WARNING] Timed-out directory not available; skipping timed-out noisy runs.')
        elif match:
            noisy_id = match.group(0)
            for f in timed_out_dir.iterdir():
                if f.is_dir() and noisy_id in f.name:
                    timed_out_folder = f.name
                    break
        if timed_out_folder:
            timed_out_path = timed_out_dir / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)
    else:
        if timed_out_dir is not None:
            timed_out_folder = VARIABILITY_MAP.get(scenario_key, folder.name)
            timed_out_path = timed_out_dir / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)

    main_his_path = model_location / "output" / "FlowFM_0000_his.nc"
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

# %% --- CACHE DIR SETUP ---
cache_dir = base_path / "cached_data"
if cache_dir.exists():
    if not cache_dir.is_dir():
        raise RuntimeError(f"[ERROR] {cache_dir} exists but is not a directory.")
    try:
        _ = list(cache_dir.iterdir())
    except Exception as e:
        raise RuntimeError(f"[ERROR] {cache_dir} is not accessible: {e}")
else:
    cache_dir.mkdir(exist_ok=True)

# %% --- LOAD ALL DATA (sediment transport + discharge) ---
scenario_data = {}

for scenario_dir, his_file_paths in run_his_paths.items():
    scenario_name = Path(scenario_dir).name
    scenario_num  = scenario_name.split('_')[0]
    run_id        = '_'.join(scenario_name.split('_')[1:])
    cache_file    = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

    # Load sediment transport (with buffer volumes)
    _, result = load_and_cache_scenario(
        scenario_dir=scenario_dir,
        his_file_paths=his_file_paths,
        cache_file=cache_file,
        boxes=boxes,
        var_name=sed_var,
    )

    # Load discharge (no buffer volumes needed)
    _, result_dis = load_and_cache_scenario(
        scenario_dir=scenario_dir,
        his_file_paths=his_file_paths,
        cache_file=cache_file,
        boxes=boxes,
        var_name=dis_var,
    )

    # Merge discharge into the main result dict
    result[dis_var] = result_dis[dis_var]

    scenario_data[scenario_dir] = result

# %% ============================================================
#    PER-SCENARIO PLOTS  — one figure per scenario, all sections in each
# ===============================================================
for scenario_dir, data in scenario_data.items():
    scenario_name = Path(scenario_dir).name
    scenario_num  = scenario_name.split('_')[0]
    scenario_label = SCENARIO_LABELS.get(str(int(scenario_num)), scenario_name)

    km_positions   = data['km_positions']
    transport      = data[sed_var]          # (time, km)  cumulative kg
    Q_all          = data[dis_var]          # (time, km)  m³/s
    time           = data['t']
    buffer_volumes = data['buffer_volumes']

    window = int(TIDAL_WINDOW_HOURS / DT_HOURS)

    # River discharge at upstream boundary
    idx_river  = np.argmin(np.abs(km_positions - RIVER_KM))

    print(f"{scenario_name}: closest cross-section to {RIVER_KM} km is at {km_positions[idx_river]:.2f} km")

    Q_river = data[dis_var][:, idx_river]
    time    = data['t']

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, Q_river, lw=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title(f'Discharge at {km_positions[idx_river]:.2f} km — {scenario_name}')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()

    # ── 1. CUMULATIVE BUFFER VOLUME ──────────────────────────
    fig, ax = plt.subplots()
    for (box_start, box_end), buf in buffer_volumes.items():
        ax.plot(time, buf, label=f'{box_start}–{box_end} km')
    ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
    ax.set_ylabel('Buffer volume (m³)')
    ax.set_ylim(-0.1, 1.15e11)
    ax.legend()
    ax.set_title(f'Sediment buffer volumes — {scenario_label}')
    ax.grid()
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario_name}_sediment_buffer_volume_cumulative.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # ── 2. INSTANTANEOUS dV/dt ───────────────────────────────
    fig, ax = plt.subplots()
    for (box_start, box_end), buf in buffer_volumes.items():
        ax.plot(time[1:], np.diff(buf), label=f'{box_start}–{box_end} km')
    ax.set_ylim(-0.25e8, 0.25e8)
    ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
    ax.set_ylabel('dV/dt (m³/timestep)')
    ax.legend()
    ax.set_title(f'Instantaneous buffer change — {scenario_label}')
    ax.grid()
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario_name}_sediment_buffer_volume_instantaneous.png",
                dpi=300, bbox_inches='tight')
    plt.show()

    # ── 3. TIDAL-AVERAGED RESIDUAL dV/dt ────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for (box_start, box_end), buf in buffer_volumes.items():
        d_buf = np.diff(buf)
        if len(d_buf) >= window:
            ax.plot(time[1:], tidal_avg(d_buf, window),
                    label=f'Residual: {box_start}–{box_end} km', linewidth=2)
        else:
            print(f"Warning: too short for tidal window at {box_start} km")
    ax.axhline(0, color='black', lw=1.5, ls='--')
    ax.set_ylim(-0.25e7, 0.25e7)
    ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
    ax.set_ylabel('Tidal avg dV/dt (m³/timestep)')
    ax.set_title(f'Residual sediment storage rate — {scenario_label}')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario_name}_residual_buffer_change.png", dpi=300)
    plt.show()

    # ── 4. TRAPPING EFFICIENCY ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for (box_start, box_end), buf in buffer_volumes.items():
        idx_up = np.argmin(np.abs(km_positions - box_start))
        dV         = np.diff(buf)
        dIn_gross  = np.abs(np.diff(transport[:, idx_up]))
        efficiency = (np.convolve(dV,        np.ones(window), mode='same') /
                     (np.convolve(dIn_gross, np.ones(window), mode='same') + 1e-6))
        ax.plot(time[1:], efficiency, label=f'Efficiency: {box_start}–{box_end} km')
    ax.axhline(0, color='black', lw=1,   ls='--')
    ax.axhline(1, color='red',   lw=1,   ls=':', label='Total trap')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
    ax.set_ylabel('Tidal-averaged efficiency (–)')
    ax.set_title(f'Sediment trapping efficiency — {scenario_label}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario_name}_trapping_efficiency.png", dpi=300)
    plt.show()


# %% ============================================================
#    PRE-PROCESS: collect trimmed arrays per scenario
#    (shared by both combined plots below)
# ===============================================================
window        = int(TIDAL_WINDOW_HOURS / DT_HOURS)
max_lag_steps = int(MAX_LAG_DAYS * 24 / DT_HOURS)

t_end_min = min(
    data['t'][SPINUP_STEPS:].max()
    for data in scenario_data.values()
)
print(f"Shortest simulation end time across scenarios: {t_end_min}")

processed = {}   # scenario_key → dict of trimmed arrays

for scenario_dir, data in scenario_data.items():
    scenario_num  = Path(scenario_dir).name.split('_')[0]
    scenario_key  = str(int(scenario_num))

    km_positions   = data['km_positions']
    Q_all          = data[dis_var]
    buffer_volumes = data['buffer_volumes']
    time           = data['t']

    idx_river = np.argmin(np.abs(km_positions - RIVER_KM))
    
    time_trimmed = time[SPINUP_STEPS:]
    t_end_idx = np.searchsorted(time_trimmed, t_end_min, side='right')

    processed[scenario_key] = dict(
        label                  = SCENARIO_LABELS.get(scenario_key, scenario_dir),
        color                  = SCENARIO_COLORS.get(scenario_key, 'grey'),
        Q_river                = Q_all[SPINUP_STEPS : SPINUP_STEPS + t_end_idx, idx_river],
        time                   = time_trimmed[:t_end_idx],
        buffer_volumes_trimmed = {k: v[SPINUP_STEPS : SPINUP_STEPS + t_end_idx]
                                  for k, v in buffer_volumes.items()},
        km_positions           = km_positions,
        transport              = data[sed_var][:SPINUP_STEPS + t_end_idx],
    )


# %% ============================================================
#    PLOT 5a — SENSITIVITY  (dV/dt vs Q_river)
#    One figure per km section, all scenarios overlaid
# ===============================================================
for (box_start, box_end) in boxes:
    fig, ax = plt.subplots(figsize=(8, 6))

    for scenario_key, d in processed.items():
        buf    = d['buffer_volumes_trimmed'][(box_start, box_end)]
        Q_plot = d['Q_river'][:-1]          # align with np.diff output
        dV_dt  = np.diff(buf)

        # Scatter (low alpha to avoid overplotting)
        ax.scatter(Q_plot, dV_dt, alpha=0.15, s=6, color=d['color'])

        # Quadratic trend line (labelled — this is the visual signal)
        z  = np.polyfit(Q_plot, dV_dt, 2)
        xp = np.linspace(Q_plot.min(), Q_plot.max(), 300)
        ax.plot(xp, np.poly1d(z)(xp), '-', linewidth=2.5,
                color=d['color'], label=d['label'])

    ax.axhline(0, color='black', lw=1,   ls='--', zorder=3)
    ax.axvline(0, color='black', lw=0.8, ls='--', zorder=3)
    ax.set_xlabel(f'River discharge at {RIVER_KM} km (m³/s)', fontsize=11)
    ax.set_ylabel('Section buffer change  dV/dt  (m³/timestep)', fontsize=11)
    ax.set_title(f'System sensitivity — section {box_start}–{box_end} km', fontsize=12)
    ax.legend(title='Scenario', fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output_dir / f"section_{box_start}-{box_end}km_sensitivity_allscenarios.png",
        dpi=300, bbox_inches='tight'
    )
    plt.show()
# %%
