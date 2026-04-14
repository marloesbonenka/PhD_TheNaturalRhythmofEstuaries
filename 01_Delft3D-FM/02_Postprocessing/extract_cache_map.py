"" "This script loads the mesh2d_mor_bl variable from the model output NetCDF files, "
"applies a spatial mask to select only the estuary area, "
"and saves the resulting datasets as new NetCDF files for each run. "
"It handles both the restart and timed-out parts of the runs, "
"stitching them together along the time dimension before applying the mask. "
"The output files are saved in a specified directory for further analysis." ""

#%%
import sys
import numpy as np
from pathlib import Path
from FUNCTIONS.F_map_cache import (
    cache_tag_from_bbox,
    load_or_update_map_cache_multi,
)
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
#%%
# =============================================================================
# 1. SETUP & PATHS
# =============================================================================

# Add the directory where the FUNCTIONS folder is located
functions_root = Path(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
if str(functions_root) not in sys.path:
    sys.path.append(str(functions_root))

# --- SETTINGS ---

ANALYSIS_MODE = "variability"
DISCHARGE = 500 # Adjust this to match your specific discharge scenario (e.g., 500, 1000, etc.)
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config = f'Model_Output/Q{DISCHARGE}'
base_path = base_directory / config
timed_out_dir = base_path / "timed-out"
# List all variables you want to extract and save
var_names = ["mesh2d_mor_bl", "mesh2d_s1", "mesh2d_u1","mesh2d_taus", "mesh2d_flowelem_ba"]

# Spatial subset bounds [xmin, ymin, xmax, ymax]
BBOX = [1, 1, 45000, 15000]

# Cache behavior
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True
CACHE_TAG = None  # Set to "full" for full domain or custom tag

# Snapshot settings: evenly spaced timesteps over the simulation period
# Set SNAPSHOT_COUNT = None to save ALL timesteps (original behaviour)
SNAPSHOT_COUNT = 6
SNAPSHOT_DATE_RANGE = (np.datetime64('2025-01-01'), np.datetime64('2031-12-31'))

# Compute evenly spaced target dates
if SNAPSHOT_COUNT is not None:
    _start_ns = np.datetime64(SNAPSHOT_DATE_RANGE[0]).astype('datetime64[ns]').astype('int64')
    _end_ns   = np.datetime64(SNAPSHOT_DATE_RANGE[1]).astype('datetime64[ns]').astype('int64')
    TARGET_DATES = [np.datetime64(int(ns), 'ns')
                    for ns in np.linspace(_start_ns, _end_ns, SNAPSHOT_COUNT)]
else:
    TARGET_DATES = None

VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(base_path, DISCHARGE)

# Directories
output_dir = base_path / 'cached_data'
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. PROCESSING LOOP
# =============================================================================


print(f"Starting extraction for {len(model_folders)} runs...")


for folder in model_folders:
    print(f"\n--- Processing Run: {folder.name} ---")

    all_run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder.name,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=False,
    )
    if not all_run_paths:
        all_run_paths = [folder]

    for p in all_run_paths:
        if p != folder:
            print(f"  Found timed-out part: {p.name}")

    cache_tag = cache_tag_from_bbox(BBOX, CACHE_TAG)
    print(f"  Updating cache(s) for {folder.name}")
    ds_out = load_or_update_map_cache_multi(
        cache_dir=output_dir,
        folder_name=folder.name,
        run_paths=all_run_paths,
        var_names=var_names,
        bbox=BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
        target_dates=TARGET_DATES,
    )

    if ds_out is None:
        print(f"  No data to cache for {folder.name}")
    else:
        print(f"  Successfully cached mapoutput_* for {folder.name}")

print("\n" + "="*30)
print("PROCESSING COMPLETE")
print(f"Files saved in: {output_dir}")
print("="*30)
# %%
