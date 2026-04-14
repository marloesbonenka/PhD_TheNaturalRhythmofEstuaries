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
from FUNCTIONS.F_loaddata import load_and_cache_scenario, get_stitched_his_paths, _cache_file_for_variable
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders

# %% --- CONFIGURATION ---

# Variables to add to the cache. Add as many as needed.
VARIABLES_TO_CACHE = [
    'cross_section_discharge',
    'cross_section_bedload_sediment_transport',
    'cross_section_velocity',
    'waterlevel',
    'bedlevel',
    'cross_section_suspended_sediment_transport',
]

# Box edges for buffer volume computation (only used for transport variables)
box_edges = np.arange(20, 50, 5)  # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i + 1]) for i in range(len(box_edges) - 1)]

# Scenario filters — match settings from your analysis script
SCENARIOS_TO_PROCESS = None  # None = all scenarios found in base_path
DISCHARGE = 500
ANALYZE_NOISY = False

# %% --- PATHS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
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
VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=ANALYZE_NOISY,
)

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
for scenario_dir, his_file_paths in run_his_paths.items():
    scenario_name = Path(scenario_dir).name
    scenario_num = scenario_name.split('_')[0]
    run_id = '_'.join(scenario_name.split('_')[1:])

    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")

    # Group requested variables by destination cache file (stations vs cross-sections).
    vars_by_cache_file = {}
    for var_name in VARIABLES_TO_CACHE:
        cache_file = _cache_file_for_variable(
            cache_dir=cache_dir,
            scenario_num=scenario_num,
            run_id=run_id,
            his_file_paths=his_file_paths,
            var_name=var_name,
        )
        vars_by_cache_file.setdefault(cache_file, []).append(var_name)

    for cache_file, var_names in vars_by_cache_file.items():
        existing_vars = set()
        if cache_file.exists():
            print(f"  Cache file exists: {cache_file.name}, opening to check existing variables...  ")
            with xr.open_dataset(cache_file) as ds_check:
                existing_vars = set(ds_check.data_vars)

        for var_name in var_names:
            if var_name in existing_vars:
                print(f"  [SKIP] '{var_name}' already in {cache_file.name}")
                continue

            print(f"  [LOAD] {var_name}")
            print(f"         -> {cache_file.name}")
            load_and_cache_scenario(
                scenario_dir=scenario_dir,
                his_file_paths=his_file_paths,
                cache_file=cache_file,
                boxes=boxes,
                var_name=var_name,
            )
            existing_vars.add(var_name)

print(f"\nDone. Cache files are in: {cache_dir}")