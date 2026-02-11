"" "This script loads the mesh2d_mor_bl variable from the model output NetCDF files, "
"applies a spatial mask to select only the estuary area, "
"and saves the resulting datasets as new NetCDF files for each run. "
"It handles both the restart and timed-out parts of the runs, "
"stitching them together along the time dimension before applying the mask. "
"The output files are saved in a specified directory for further analysis." ""

#%%
import sys
import xarray as xr
import xugrid as xu
from pathlib import Path

# =============================================================================
# 1. SETUP & PATHS
# =============================================================================

# Add the directory where the FUNCTIONS folder is located
functions_root = Path(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
if str(functions_root) not in sys.path:
    sys.path.append(str(functions_root))

from FUNCTIONS.F_cache import DatasetCache

# --- Settings ---
ANALYSIS_MODE = "variability"
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = 'Model_Output'
var_name = "mesh2d_mor_bl"

# Spatial subset bounds [xmin, ymin, xmax, ymax]
# Matches your x_targets (20000-44001) and y_range (5000-10000)
BBOX = [20000, 5000, 45000, 10000]

# Mapping: restart folder prefix -> timed-out folder prefix
VARIABILITY_MAP = {
    # '1': '01_baserun500',
    # '2': '02_run500_seasonal',
    # '3': '03_run500_flashy',
    '4': '04_run500_singlepeak'
}

# Directories
base_path = base_directory / config
timed_out_dir = base_path / "timed-out"
output_dir = base_path / "assessment_netcdfs"
output_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. PROCESSING LOOP
# =============================================================================

# Find run folders starting with a digit (e.g. 1_rst, 2_rst)
model_folders = [f for f in base_path.iterdir() 
                 if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
model_folders.sort(key=lambda x: int(x.name.split('_')[0]))

# Initialize the imported cache
ds_cache = DatasetCache()

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
    
    # --- LOAD & CONCATENATE ---
    datasets = []
    for run_path in all_run_paths:
        file_pattern = str(run_path / "output" / "*_map.nc")
        
        try:
            # get_partitioned preserves topology variables automatically 
            # as long as mesh2d_mor_bl is in variables list
            part_ds = ds_cache.get_partitioned(
                file_pattern, 
                variables=[var_name],
                chunks={'time': 100}
            )
            datasets.append(part_ds)
            print(f"  Stitched {run_path.name}")
        except Exception as e:
            print(f"  Warning: Could not load {run_path.name}. Error: {e}")

    if not datasets:
        continue

    # Combine parts along the time dimension
    print("  Combining parts into full timeline...")
    full_ds = xu.concat(datasets, dim="time")

    # --- MASKING (XUGRID) ---
    print(f"  Applying spatial mask: {BBOX}...")
    # Wrap xarray dataset into xugrid
    # uds = xu.UgridDataset(full_ds)
    
    # Select the estuary area
    uds_masked = full_ds.ugrid.sel(x=slice(BBOX[0], BBOX[2]), y=slice(BBOX[1], BBOX[3]))

    # --- SAVE OUTPUT ---
    save_filename = f"assessment_{var_name}_{folder.name}.nc"
    save_path = output_dir / save_filename
    
    print(f"  Saving to {save_filename}...")
    uds_masked.to_netcdf(save_path)
    print(f"  Successfully exported.")

# Close all opened NetCDFs
ds_cache.close_all()
print("\n" + "="*30)
print("PROCESSING COMPLETE")
print(f"Files saved in: {output_dir}")
print("="*30)