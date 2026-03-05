"" "This script loads the mesh2d_mor_bl variable from the model output NetCDF files, "
"applies a spatial mask to select only the estuary area, "
"and saves the resulting datasets as new NetCDF files for each run. "
"It handles both the restart and timed-out parts of the runs, "
"stitching them together along the time dimension before applying the mask. "
"The output files are saved in a specified directory for further analysis." ""

#%%
import sys
from pathlib import Path
from FUNCTIONS.F_map_cache import (
    cache_tag_from_bbox,
    load_or_update_map_cache_multi,
)

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
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f'Model_Output/Q{DISCHARGE}'
base_path = base_directory / config
timed_out_dir = base_path / "timed-out"
# List all variables you want to extract and save
var_names = ["mesh2d_mor_bl", "mesh2d_s1", "mesh2d_taus"]

# Spatial subset bounds [xmin, ymin, xmax, ymax]
BBOX = [20000, 5000, 45000, 10000]

# Cache behavior
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True
CACHE_TAG = None  # Set to "full" for full domain or custom tag

# Mapping: restart folder prefix -> timed-out folder prefix
VARIABILITY_MAP = {
    '1': '01_baserun{DISCHARGE}',
    '2': '02_run{DISCHARGE}_seasonal',
    '3': '03_run{DISCHARGE}_flashy',
    '4': '04_run{DISCHARGE}_singlepeak'
}


# Directories
output_dir = base_path / 'cached_data'
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. PROCESSING LOOP
# =============================================================================

# Find run folders starting with a digit (e.g. 1_rst, 2_rst)
model_folders = [f for f in base_path.iterdir() 
                 if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
model_folders.sort(key=lambda x: int(x.name.split('_')[0]))

print(f"Starting extraction for {len(model_folders)} runs...")


for folder in model_folders:
    print(f"\n--- Processing Run: {folder.name} ---")

    # --- STITCHING LOGIC ---
    all_run_paths = []
    scenario_num = folder.name.split('_')[0]

    # 1. Check for timed-out part
    if scenario_num in VARIABILITY_MAP:
        t_out_path = timed_out_dir / VARIABILITY_MAP[scenario_num]
        if t_out_path.exists():
            all_run_paths.append(t_out_path)
            print(f"  Found timed-out part: {t_out_path.name}")

    # 2. Add the restart part
    all_run_paths.append(folder)

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
