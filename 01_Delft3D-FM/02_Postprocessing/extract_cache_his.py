"""
Add one or more variables from HIS output files to the scenario cache.

Run this script to pre-populate (or extend) the hisoutput_*.nc cache files
with any variable available in the Delft3D-FM HIS output, without re-running
the full analysis script. Buffer volumes are computed automatically for
cumulative transport variables (see BUFFER_VOLUME_VARS in F_loaddata).

Supported variable types:
- cross-section variables (time x cross_section)
- station/point variables (time x station)

Usage: set VARIABLES_TO_CACHE and the scenario filters below, then run.
"""

# %% IMPORTS
import sys
import numpy as np
import xarray as xr
from pathlib import Path

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import load_and_cache_scenario, get_stitched_his_paths

# %% --- CONFIGURATION ---

# Variables to add to the cache. Add as many as needed.
VARIABLES_TO_CACHE = [
    'cross_section_discharge',
    'cross_section_bedload_sediment_transport',
    'cross_section_velocity',
    # 'waterlevel',
    # 'cross_section_suspended_sediment_transport',
]

# Box edges for buffer volume computation (only used for transport variables)
box_edges = np.arange(20, 50, 5)  # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i + 1]) for i in range(len(box_edges) - 1)]

# Scenario filters — match settings from your analysis script
SCENARIOS_TO_PROCESS = ['0', '1', '2', '3', '4']
DISCHARGE = 500
ANALYZE_NOISY = True

# %% --- PATHS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"
if ANALYZE_NOISY:
    base_path = base_directory / config / f"0_Noise_Q{DISCHARGE}"
else:
    base_path = base_directory / config

timed_out_dir = base_path / "timed-out"
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")
if not timed_out_dir.exists():
    timed_out_dir = None
    print('[WARNING] Timed-out directory not found.')

# %% --- FIND RUN FOLDERS ---
# Mapping: restart folder prefix -> timed-out folder prefix
if DISCHARGE == 500:
    VARIABILITY_MAP = {
        '1': f'01_baserun{DISCHARGE}',
        '2': f'02_run{DISCHARGE}_seasonal',
        '3': f'03_run{DISCHARGE}_flashy',
        '4': f'04_run{DISCHARGE}_singlepeak'
    }
    # Find run folders starting with a digit
    # Noisy runs are identified by 'noisy' in the name (they may not have '_rst')
    if ANALYZE_NOISY:
        model_folders = [f for f in base_path.iterdir()
                        if f.is_dir() and f.name[0].isdigit() and 'noisy' in f.name.lower()]
    else:
        model_folders = [f for f in base_path.iterdir()
                        if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
    model_folders.sort(key=lambda x: int(x.name.split('_')[0]))

if DISCHARGE == 1000:
    VARIABILITY_MAP = {
        '01': f'01_baserun{DISCHARGE}',
        '02': f'02_run{DISCHARGE}_seasonal',
        '03': f'03_run{DISCHARGE}_flashy',
        '04': f'04_run{DISCHARGE}_singlepeak'
    }
    # Find run folders starting with a digit (e.g. 1_rst, 2_rst)
    model_folders = [f for f in base_path.iterdir() 
                    if f.is_dir() and f.name[0].isdigit()]
    model_folders.sort(key=lambda x: int(x.name.split('_')[0]))

if SCENARIOS_TO_PROCESS:
    try:
        scenario_filter = set(int(s) for s in SCENARIOS_TO_PROCESS)
    except Exception:
        scenario_filter = set()
    model_folders = [f for f in model_folders if int(f.name.split('_')[0]) in scenario_filter]

print(f"Found {len(model_folders)} run folders in: {base_path}")

# %% --- BUILD HIS FILE PATHS (same logic as analysis script) ---
run_his_paths = {}
for folder in model_folders:
    his_paths = get_stitched_his_paths(
        base_path=base_path,
        folder_name=folder,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=ANALYZE_NOISY,
    )

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

# %% --- CACHE DIR ---
cache_dir = base_path / "cached_data"
cache_dir.mkdir(exist_ok=True)

# %% --- POPULATE CACHE ---
def _cache_file_for_variable(cache_dir, scenario_num, run_id, his_file_paths, var_name):
    """
    Keep legacy cross-section cache filenames unchanged.
    Write station variables to dedicated files to avoid mixing schemas.
    """
    with xr.open_dataset(his_file_paths[0]) as ds0:
        if var_name not in ds0:
            raise KeyError(f"Variable '{var_name}' not found in {his_file_paths[0]}")
        dims = ds0[var_name].dims

    if 'station' in dims:
        return cache_dir / f"hisoutput_stations_{int(scenario_num)}_{run_id}.nc"

    # Default and backward-compatible path for cross-section and other legacy vars.
    return cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"


for var_name in VARIABLES_TO_CACHE:
    print(f"\n{'='*60}")
    print(f"Variable: {var_name}")
    print(f"{'='*60}")

    for scenario_dir, his_file_paths in run_his_paths.items():
        scenario_name = Path(scenario_dir).name
        scenario_num = scenario_name.split('_')[0]
        run_id = '_'.join(scenario_name.split('_')[1:])
        cache_file = _cache_file_for_variable(
            cache_dir=cache_dir,
            scenario_num=scenario_num,
            run_id=run_id,
            his_file_paths=his_file_paths,
            var_name=var_name,
        )

        # Check upfront so the summary is clear
        already_cached = False
        if cache_file.exists():
            with xr.open_dataset(cache_file) as ds_check:
                already_cached = var_name in ds_check

        if already_cached:
            print(f"  [SKIP] '{var_name}' already in cache for {scenario_dir}")
            continue

        print(f"  [LOAD] {scenario_dir}")
        print(f"         -> {cache_file.name}")
        load_and_cache_scenario(
            scenario_dir=scenario_dir,
            his_file_paths=his_file_paths,
            cache_file=cache_file,
            boxes=boxes,
            var_name=var_name,
        )

print(f"\nDone. Cache files are in: {cache_dir}")