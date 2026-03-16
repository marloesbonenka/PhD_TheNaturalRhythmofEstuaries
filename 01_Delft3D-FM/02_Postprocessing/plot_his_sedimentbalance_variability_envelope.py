"""
Plot sediment buffer volume envelope for noisy runs vs base scenario.
Uses the same folder-discovery and caching logic as extract_cache_his.py.

PLOT_MODE = 'noise_only' : base run + noisy envelope only
                           → saved in 0_Noise_Q{DISCHARGE}/plots_his_sedimentbuffer_envelope
PLOT_MODE = 'all'        : base run + noisy envelope + all variability scenarios
                           → saved in Q{DISCHARGE}/plots_his_sedimentbuffer_envelope
"""
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import load_and_cache_scenario, get_stitched_his_paths
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders

#%% --- CONFIGURATION ---
var_name = 'cross_section_bedload_sediment_transport' # cumulative bed load sediment transport [kg]
output_dirname = Path("output_plots") / "plots_his_sedimentbuffer_envelope"

mpl.rcParams['figure.figsize'] = (10, 5)

# Define cross-section km boundaries
box_edges = np.arange(20, 50, 5)  # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i+1]) for i in range(len(box_edges)-1)]

DISCHARGE = 500
BASE_SCENARIO = '1'

# --- PLOT MODE ---
# 'noise_only' : base run + noisy envelope only  → saved in 0_Noise_Q{DISCHARGE}
# 'all'        : base run + noisy envelope (if existing) + all variability scenarios → saved in Q{DISCHARGE}
PLOT_MODE = 'noise_only'  # <-- change here

SCENARIO_CONFIG = {
    "baserun":    {"color": "tab:blue",   "label": "Constant discharge"},
    "seasonal":   {"color": "tab:orange", "label": "Seasonal discharge"},
    "flashy":     {"color": "tab:green",  "label": "Flashy discharge"},
    "singlepeak": {"color": "tab:red",    "label": "Single peak discharge"},
}

VARIABILITY_SCENARIOS = {
    '1': 'baserun',
    '2': 'seasonal',
    '3': 'flashy',
    '4': 'singlepeak',
}

VARIABILITY_MAP = get_variability_map(DISCHARGE)

#%% --- PATHS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"

variability_base_path = base_directory / config
variability_cache_dir = variability_base_path / "cached_data"

noisy_base_path = variability_base_path / f"0_Noise_Q{DISCHARGE}"
noisy_cache_dir = noisy_base_path / "cached_data"

# Output dir depends on PLOT_MODE
if PLOT_MODE == 'noise_only':
    output_dir = noisy_base_path / output_dirname
else:
    output_dir = variability_base_path / output_dirname

output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {output_dir}")

#%% --- HELPER FUNCTIONS ---

def load_runs(base_path, cache_dir, analyze_noisy, scenario_filter=None):
    """
    Find run folders and load data using the same logic as extract_cache_his.py.
    Returns dict: folder_name -> {'time': array, 'buffers': {(start, end): array}}
    """
    timed_out_dir = base_path / "timed-out"
    if not timed_out_dir.exists():
        timed_out_dir = None

    folders = find_variability_model_folders(
        base_path=base_path,
        discharge=DISCHARGE,
        scenarios_to_process=scenario_filter,
        analyze_noisy=analyze_noisy,
    )

    cache_dir.mkdir(exist_ok=True)
    runs = {}
    for folder in folders:
        his_paths = get_stitched_his_paths(
            base_path=base_path,
            folder_name=folder,
            timed_out_dir=timed_out_dir,
            variability_map=VARIABILITY_MAP,
            analyze_noisy=analyze_noisy,
        )
        if not his_paths:
            print(f"[WARNING] No HIS files found for {folder.name}, skipping.")
            continue

        scenario_num = folder.name.split('_')[0]
        run_id = '_'.join(folder.name.split('_')[1:])
        cache_file = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

        _, data = load_and_cache_scenario(
            scenario_dir=folder,
            his_file_paths=his_paths,
            cache_file=cache_file,
            boxes=boxes,
            var_name=var_name,
        )
        runs[folder.name] = {
            'time': data['t'],
            'buffers': data['buffer_volumes'],
        }
    return runs


def align_to_common_time(base_time, runs_dict, box_key):
    """
    Interpolate all runs in runs_dict onto base_time for a given box_key.
    Returns array of shape (n_runs, len(base_time)).
    """
    aligned = []
    for run_name, run_data in runs_dict.items():
        t = run_data['time']
        buf = run_data['buffers'].get(box_key)
        if buf is None:
            continue
        if np.issubdtype(t.dtype, np.datetime64):
            t_f      = t.astype('datetime64[ns]').astype(np.float64)
            t_base_f = base_time.astype('datetime64[ns]').astype(np.float64)
        else:
            t_f      = t.astype(np.float64)
            t_base_f = base_time.astype(np.float64)
        aligned.append(np.interp(t_base_f, t_f, buf))
    return np.array(aligned)


#%% --- 1. LOAD BASE RUN (always needed) ---
base_runs = load_runs(
    variability_base_path, variability_cache_dir,
    analyze_noisy=False, scenario_filter={int(BASE_SCENARIO)},
)
if not base_runs:
    raise FileNotFoundError(
        f"No run found for base scenario {BASE_SCENARIO} in {variability_base_path}."
    )
base_data = list(base_runs.values())[0]
base_time    = base_data['time']
base_buffers = base_data['buffers']
base_cfg = SCENARIO_CONFIG[VARIABILITY_SCENARIOS[BASE_SCENARIO]]


#%% --- 2. LOAD OTHER VARIABILITY SCENARIOS (only in 'all' mode) ---
variability_runs = {}  # scenario_num -> {'time', 'buffers', 'label', 'color'}

if PLOT_MODE == 'all':
    other_nums = {int(k) for k in VARIABILITY_SCENARIOS if k != BASE_SCENARIO}
    all_var_runs = load_runs(
        variability_base_path, variability_cache_dir,
        analyze_noisy=False, scenario_filter=other_nums,
    )
    for folder_name, run_data in all_var_runs.items():
        scenario_num = str(int(folder_name.split('_')[0]))
        if scenario_num not in VARIABILITY_SCENARIOS:
            continue
        cfg = SCENARIO_CONFIG[VARIABILITY_SCENARIOS[scenario_num]]
        variability_runs[scenario_num] = {
            'time': run_data['time'], 'buffers': run_data['buffers'],
            'label': cfg['label'], 'color': cfg['color'],
        }


#%% --- 3. LOAD NOISY RUNS ---
noisy_runs = load_runs(noisy_base_path, noisy_cache_dir, analyze_noisy=True)
print(f"\nFound {len(noisy_runs)} noisy runs:")
for name in noisy_runs:
    print(f"  {name}")

if not noisy_runs:
    raise FileNotFoundError(
        f"No noisy runs found in {noisy_base_path}. "
        f"Make sure HIS output exists or run extract_cache_his.py first."
    )


#%% --- 4. PLOT: one figure per box ---

# ── Cap all plots at the shortest simulation time across all loaded runs ──
all_times = [base_time] + \
            [r['time'] for r in noisy_runs.values()] + \
            [r['time'] for r in variability_runs.values()]
t_end_min = min(t[-1] for t in all_times)
print(f"Shortest simulation end time across all runs: {t_end_min}")

# Trim helper: boolean mask for a given time array
def trim(t):
    return t <= t_end_min

for box_key in boxes:
    box_start, box_end = box_key
    fig, ax = plt.subplots()

    # --- Noisy envelope (plotted first so scenario lines appear on top) ---
    noisy_stack = align_to_common_time(base_time[trim(base_time)], noisy_runs, box_key)

    # Always include the base scenario in the envelope bounds
    base_buf_for_env = base_buffers.get(box_key)
    if base_buf_for_env is not None:
        base_row = base_buf_for_env[trim(base_time)][np.newaxis, :]
        if noisy_stack.size > 0:
            noisy_stack = np.vstack([noisy_stack, base_row])
        else:
            noisy_stack = base_row

    if noisy_stack.size > 0:
        env_min = np.nanmin(noisy_stack, axis=0)
        env_max = np.nanmax(noisy_stack, axis=0)

        for i, (run_name, run_data) in enumerate(noisy_runs.items()):
            buf = run_data['buffers'].get(box_key)
            if buf is None:
                continue
            mask = trim(run_data['time'])
            ax.plot(run_data['time'][mask], buf[mask], color='grey', alpha=0.35, linewidth=0.7,
                    label='Noisy runs' if i == 0 else None)

        ax.fill_between(base_time[trim(base_time)], env_min, env_max,
                        color='grey', alpha=0.2, label='Noisy envelope')
    else:
        print(f"[WARNING] No noisy data for box {box_start}-{box_end} km.")

    # --- Variability scenario lines (only in 'all' mode) ---
    if PLOT_MODE == 'all':
        for scenario_num, run_data in variability_runs.items():
            buf = run_data['buffers'].get(box_key)
            if buf is None:
                continue
            mask = trim(run_data['time'])
            ax.plot(run_data['time'][mask], buf[mask],
                    color=run_data['color'], linewidth=1.2,
                    label=run_data['label'])

    # --- Base run (always on top) ---
    base_buf = base_buffers.get(box_key)
    if base_buf is not None:
        mask = trim(base_time)
        ax.plot(base_time[mask], base_buf[mask],
                color=base_cfg['color'], linewidth=1.8,
                label=base_cfg['label'])

    ax.set_xlabel('Time')
    ax.set_ylabel('Buffer Volume (m³)')
    ax.set_title(f'Sediment buffer volume — {box_start}–{box_end} km')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / f"envelope_box_{box_start}_{box_end}km.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.show()
