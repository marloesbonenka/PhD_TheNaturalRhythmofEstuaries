"""
Plot sediment buffer volume envelope for noisy runs vs base scenario.

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
import xarray as xr
from pathlib import Path

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import load_cross_section_data, load_and_cache_scenario

#%% --- CONFIGURATION ---
var_name = 'cross_section_bedload_sediment_transport'
output_dirname = "plots_his_sedimentbuffer_envelope"

mpl.rcParams['figure.figsize'] = (10, 5)

# Define cross-section km boundaries
box_edges = np.arange(20, 50, 5)  # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i+1]) for i in range(len(box_edges)-1)]

DISCHARGE = 500
BASE_SCENARIO = '1'

# --- PLOT MODE ---
# 'noise_only' : base run + noisy envelope only  → saved in 0_Noise_Q{DISCHARGE}
# 'all'        : base run + noisy envelope + all variability scenarios → saved in Q{DISCHARGE}
PLOT_MODE = 'all'  # <-- change here

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

def load_buffer_from_cache(cache_file, boxes):
    """Load time and buffer volumes from a cached NetCDF file."""
    ds = xr.open_dataset(cache_file)
    time = ds['t'].values
    try:
        time = time.astype('datetime64[ns]')
    except Exception:
        pass
    buffer_volumes = {}
    for box_start, box_end in boxes:
        key = f'buffer_{int(box_start)}_{int(box_end)}'
        if key in ds:
            buffer_volumes[(box_start, box_end)] = ds[key].values
    ds.close()
    return time, buffer_volumes


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
base_cache_files = list(variability_cache_dir.glob(
    f"sedtransport_timeseries_{BASE_SCENARIO}_Q{DISCHARGE}_*.nc"
))
base_cache_files = [f for f in base_cache_files if 'noisy' not in f.name]
if not base_cache_files:
    raise FileNotFoundError(
        f"No cache file found for base scenario {BASE_SCENARIO} in {variability_cache_dir}."
    )
base_cfg = SCENARIO_CONFIG[VARIABILITY_SCENARIOS[BASE_SCENARIO]]
print(f"Loading base run: {base_cache_files[0].name}")
base_time, base_buffers = load_buffer_from_cache(base_cache_files[0], boxes)


#%% --- 2. LOAD OTHER VARIABILITY SCENARIOS (only in 'all' mode) ---
variability_runs = {}  # scenario_num -> {'time', 'buffers', 'label', 'color'}

if PLOT_MODE == 'all':
    for scenario_num, config_key in VARIABILITY_SCENARIOS.items():
        if scenario_num == BASE_SCENARIO:
            continue  # base run loaded separately above
        cfg = SCENARIO_CONFIG[config_key]
        cache_files = list(variability_cache_dir.glob(
            f"sedtransport_timeseries_{scenario_num}_Q{DISCHARGE}_*.nc"
        ))
        cache_files = [f for f in cache_files if 'noisy' not in f.name]
        if not cache_files:
            print(f"[WARNING] No cache file for scenario {scenario_num} ({cfg['label']}), skipping.")
            continue
        print(f"Loading scenario {scenario_num} ({cfg['label']}): {cache_files[0].name}")
        time, buffers = load_buffer_from_cache(cache_files[0], boxes)
        variability_runs[scenario_num] = {
            'time': time, 'buffers': buffers,
            'label': cfg['label'], 'color': cfg['color']
        }


#%% --- 3. LOAD NOISY RUNS ---
noisy_cache_files = list(noisy_cache_dir.glob(
    f"sedtransport_timeseries_{BASE_SCENARIO}_Q{DISCHARGE}_noisy*.nc"
))
print(f"\nFound {len(noisy_cache_files)} noisy cache files:")
for f in noisy_cache_files:
    print(f"  {f.name}")

if not noisy_cache_files:
    raise FileNotFoundError(
        f"No noisy cache files found in {noisy_cache_dir}. "
        f"Run the main script with ANALYZE_NOISY=True first."
    )

noisy_runs = {}
for cache_file in noisy_cache_files:
    time, buffers = load_buffer_from_cache(cache_file, boxes)
    noisy_runs[cache_file.stem] = {'time': time, 'buffers': buffers}


#%% --- 4. PLOT: one figure per box ---
for box_key in boxes:
    box_start, box_end = box_key
    fig, ax = plt.subplots()

    # --- Noisy envelope (plotted first so scenario lines appear on top) ---
    noisy_stack = align_to_common_time(base_time, noisy_runs, box_key)
    if noisy_stack.size > 0:
        env_min = np.nanmin(noisy_stack, axis=0)
        env_max = np.nanmax(noisy_stack, axis=0)

        for i, (run_name, run_data) in enumerate(noisy_runs.items()):
            buf = run_data['buffers'].get(box_key)
            if buf is None:
                continue
            ax.plot(run_data['time'], buf, color='grey', alpha=0.35, linewidth=0.7,
                    label='Noisy runs' if i == 0 else None)

        ax.fill_between(base_time, env_min, env_max,
                        color='grey', alpha=0.2, label='Noisy envelope')
    else:
        print(f"[WARNING] No noisy data for box {box_start}-{box_end} km.")

    # --- Variability scenario lines (only in 'all' mode) ---
    if PLOT_MODE == 'all':
        for scenario_num, run_data in variability_runs.items():
            buf = run_data['buffers'].get(box_key)
            if buf is None:
                continue
            ax.plot(run_data['time'], buf,
                    color=run_data['color'], linewidth=1.2,
                    label=run_data['label'])

    # --- Base run (always on top) ---
    base_buf = base_buffers.get(box_key)
    if base_buf is not None:
        ax.plot(base_time, base_buf,
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

print("Done.")
#%% working code below, without adjustments for variability scenario plots
# """
# Plot sediment buffer volume envelope for noisy runs vs base scenario.
# For each box section, plot the base run (1_Q500_rst) as a line,
# and shade the envelope spanned by the noisy runs.
# """
# #%% IMPORTS
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import sys
# import xarray as xr
# from pathlib import Path
# import re

# sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
# from FUNCTIONS.F_loaddata import load_cross_section_data, load_and_cache_scenario

# #%% --- CONFIGURATION ---
# var_name = 'cross_section_bedload_sediment_transport'
# output_dirname = "plots_his_sedimentbuffer_envelope"

# mpl.rcParams['figure.figsize'] = (10, 5)

# # Define cross-section km boundaries
# box_edges = np.arange(20, 50, 5)  # [20, 25, 30, 35, 40, 45]
# boxes = [(box_edges[i], box_edges[i+1]) for i in range(len(box_edges)-1)]

# DISCHARGE = 500
# BASE_SCENARIO = '1'  # scenario number of the base/deterministic run

# #%% --- PATHS ---
# base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
# config = f"Model_Output/Q{DISCHARGE}"

# # Noisy runs path
# noisy_base_path = base_directory / config / f"0_Noise_Q{DISCHARGE}"
# noisy_cache_dir = noisy_base_path / "cached_data"
# noisy_output_dir = noisy_base_path / output_dirname
# noisy_output_dir.mkdir(parents=True, exist_ok=True)

# # Variability runs path (for the base/deterministic run)
# variability_base_path = base_directory / config
# variability_cache_dir = variability_base_path / "cached_data"

# timed_out_dir = variability_base_path / "timed-out"

# #%% --- HELPER: load buffer volumes from a cache file ---
# def load_buffer_from_cache(cache_file):
#     """Load buffer volumes and time from a cached NetCDF file."""
#     ds = xr.open_dataset(cache_file)
#     time = ds['t'].values
#     # Try to parse as datetime
#     try:
#         time = time.astype('datetime64[ns]')
#     except Exception:
#         pass
#     buffer_volumes = {}
#     for box_start, box_end in boxes:
#         key = f'buffer_{int(box_start)}_{int(box_end)}'
#         if key in ds:
#             buffer_volumes[(box_start, box_end)] = ds[key].values
#     ds.close()
#     return time, buffer_volumes


# def load_or_compute_buffers(scenario_dir_name, his_file_paths, cache_dir):
#     """Load from cache if available, otherwise compute and cache."""
#     scenario_num = scenario_dir_name.split('_')[0]
#     run_id = '_'.join(scenario_dir_name.split('_')[1:])
#     cache_file = cache_dir / f"sedtransport_timeseries_{scenario_num}_{run_id}.nc"

#     if cache_file.exists():
#         print(f"Loading from cache: {cache_file}")
#         time, buffer_volumes = load_buffer_from_cache(cache_file)
#         return time, buffer_volumes

#     print(f"Loading from HIS files: {scenario_dir_name}")
#     _, result = load_and_cache_scenario(
#         scenario_dir_name, his_file_paths, cache_dir, boxes, var_name, DISCHARGE
#     )

#     if 'buffer_volumes' in result:
#         return result['t'], result['buffer_volumes']

#     # Compute buffer volumes
#     km_positions = np.array(result['km_positions'])
#     transport = result['discharge']
#     time = result['t']
#     buffer_volumes = {}
#     for box_start, box_end in boxes:
#         idx_up = np.argmin(np.abs(km_positions - box_start))
#         idx_down = np.argmin(np.abs(km_positions - box_end))
#         buffer_volumes[(box_start, box_end)] = transport[:, idx_up] - transport[:, idx_down]

#     # Save cache
#     t_vals = time.astype(str) if np.issubdtype(time.dtype, np.datetime64) else time.astype(float)
#     buffer_dict = {
#         f'buffer_{int(bs)}_{int(be)}': (['time'], np.array(buf))
#         for (bs, be), buf in buffer_volumes.items()
#     }
#     ds = xr.Dataset({
#         'km_positions': (['km'], km_positions),
#         'discharge': (['time', 'km'], np.array(transport)),
#         't': (['time'], t_vals),
#         **buffer_dict
#     })
#     comp = dict(zlib=True, complevel=4)
#     ds.to_netcdf(cache_file, encoding={v: comp for v in ds.data_vars})
#     print(f"  Saved cache: {cache_file}")

#     return time, buffer_volumes


# #%% --- 1. FIND AND LOAD BASE RUN (deterministic, no noise) ---
# # Look in variability cache dir for scenario 1 cache files
# base_cache_files = list(variability_cache_dir.glob(f"sedtransport_timeseries_{BASE_SCENARIO}_Q{DISCHARGE}_*.nc"))
# if not base_cache_files:
#     raise FileNotFoundError(
#         f"No cache file found for base scenario {BASE_SCENARIO} in {variability_cache_dir}. "
#         f"Run the main script first."
#     )
# # Pick the deterministic (non-noisy, non-rst) one — adjust pattern if needed
# # Prefer files NOT containing 'noisy' and NOT containing 'rst' in the name
# base_cache_files_clean = [f for f in base_cache_files if 'noisy' not in f.name and 'rst' not in f.name]
# if not base_cache_files_clean:
#     # Fall back: just take the first one
#     base_cache_files_clean = base_cache_files
# base_cache_file = base_cache_files_clean[0]
# print(f"Base run cache file: {base_cache_file}")

# base_time, base_buffers = load_buffer_from_cache(base_cache_file)


# #%% --- 2. FIND AND LOAD ALL NOISY RUNS FOR SCENARIO 1 ---
# # Find all noisy cache files for scenario 1
# noisy_cache_files = list(noisy_cache_dir.glob(f"sedtransport_timeseries_{BASE_SCENARIO}_Q{DISCHARGE}_noisy*.nc"))
# print(f"Found {len(noisy_cache_files)} noisy cache files:")
# for f in noisy_cache_files:
#     print(f"  {f.name}")

# if not noisy_cache_files:
#     raise FileNotFoundError(
#         f"No noisy cache files found in {noisy_cache_dir}. "
#         f"Run the main script with ANALYZE_NOISY=True first."
#     )

# noisy_runs = {}
# for cache_file in noisy_cache_files:
#     time, buffer_volumes = load_buffer_from_cache(cache_file)
#     noisy_runs[cache_file.stem] = {'time': time, 'buffers': buffer_volumes}


# #%% --- 3. PLOT: one figure per box, envelope of noisy runs + base ---
# # Helper: interpolate all noisy runs to a common time axis for envelope
# def align_to_common_time(base_time, noisy_runs, box_key):
#     """
#     Align all noisy run timeseries to the base time axis.
#     Returns array of shape (n_noisy_runs, len(base_time)).
#     Assumes time is datetime64 or numeric float.
#     """
#     aligned = []
#     for run_name, run_data in noisy_runs.items():
#         t_noisy = run_data['time']
#         buf_noisy = run_data['buffers'].get(box_key)
#         if buf_noisy is None:
#             continue
#         # Convert datetime64 to float (ns) for interpolation if needed
#         if np.issubdtype(t_noisy.dtype, np.datetime64):
#             t_noisy_f = t_noisy.astype('datetime64[ns]').astype(np.float64)
#             t_base_f = base_time.astype('datetime64[ns]').astype(np.float64)
#         else:
#             t_noisy_f = t_noisy.astype(np.float64)
#             t_base_f = base_time.astype(np.float64)
#         # Interpolate noisy run to base time axis
#         buf_interp = np.interp(t_base_f, t_noisy_f, buf_noisy)
#         aligned.append(buf_interp)
#     return np.array(aligned)  # shape: (n_runs, n_time)


# for box_key in boxes:
#     box_start, box_end = box_key
#     fig, ax = plt.subplots()

#     # Stack noisy runs and compute envelope
#     noisy_stack = align_to_common_time(base_time, noisy_runs, box_key)

#     if noisy_stack.size == 0:
#         print(f"[WARNING] No noisy data for box {box_start}-{box_end} km, skipping.")
#         continue

#     env_min = np.nanmin(noisy_stack, axis=0)
#     env_max = np.nanmax(noisy_stack, axis=0)

#     # Plot each individual noisy run as thin lines
#     for i, (run_name, run_data) in enumerate(noisy_runs.items()):
#         buf = run_data['buffers'].get(box_key)
#         if buf is None:
#             continue
#         ax.plot(run_data['time'], buf, color='steelblue', alpha=0.4, linewidth=0.8,
#                 label='Noisy runs' if i == 0 else None)

#     # Shade envelope
#     ax.fill_between(base_time, env_min, env_max,
#                     color='steelblue', alpha=0.2, label='Noisy envelope')

#     # Plot base run on top
#     base_buf = base_buffers.get(box_key)
#     if base_buf is not None:
#         ax.plot(base_time, base_buf, color='black', linewidth=1.5,
#                 label=f'Base run ({BASE_SCENARIO}_Q{DISCHARGE})')

#     ax.set_xlabel('Time')
#     ax.set_ylabel('Buffer Volume (m³)')
#     ax.set_title(f'Sediment buffer volume — {box_start}–{box_end} km')
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()

#     fig_path = noisy_output_dir / f"envelope_box_{box_start}_{box_end}km.png"
#     plt.savefig(fig_path, dpi=300, bbox_inches='tight')
#     print(f"Saved: {fig_path}")
#     plt.show()

# print("Done.")