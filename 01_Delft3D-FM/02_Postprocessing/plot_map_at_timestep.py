"""Plot map output at a certain timestep"""
#%% 
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from FUNCTIONS.F_general import create_bedlevel_colormap, create_water_colormap, create_shear_stress_colormap
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

#%% --- 1. SETTINGS ---
# Which scenarios to process (set to None or empty list for all)
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']  # Use all scenarios
DISCHARGE = 500
# --- Variable selection ---
var_names = ['mesh2d_mor_bl', 'mesh2d_s1', 'mesh2d_taus']  # e.g. ['mesh2d_mor_bl'] or all three
time_to_extract = None
target_hydrodynamic_date = None #'2055-12-31' # e.g. '2055-12-31'; when set, nearest timestep is used per run

# Cache settings
CACHE_BBOX = [1, 1, 45000, 15000] # xmin, ymin, xmax, ymax
CACHE_TAG = None
APPEND_TIMESTEPS = False
APPEND_VARIABLES = False

#%%
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"
base_path = base_directory / config

assessment_dir = base_path / 'cached_data'

timed_out_dir = base_path / "timed-out"
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")
if not timed_out_dir.exists():
    timed_out_dir = None
    print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')
    #raise FileNotFoundError(f"Timed-out directory not found: {timed_out_dir}")

VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=False,
)

configs = {
    'mesh2d_mor_bl': {
        'cmap': create_bedlevel_colormap(),
        'vmin': -15,
        'vmax': 15,
        'label': 'Bed Level [m]',
        'file_tag': 'bedlevel_map'
    },
    'mesh2d_s1': {
        'cmap': create_water_colormap(),
        'vmin': -1,   # Adjust based on your tide/datum
        'vmax': 3,
        'label': 'Water Level [m]',
        'file_tag': 'water_level_map'
    },
    'mesh2d_taus': {
        'cmap': create_shear_stress_colormap(),
        'vmin': 0,
        'vmax': 5,    # Adjust based on flow intensity
        'label': 'Bed Shear Stress [N/m²]',
        'file_tag': 'shear_stress_map'
    }
}
#%%
# =============================================================================
# 2. PROCESSING LOOP
# =============================================================================

for folder in model_folders:
    model_location = base_path / folder
    output_plots_dir = base_path / 'output_plots' / 'map_plots'
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing: {folder.name}")

    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder.name,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [model_location]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder.name,
        run_paths=run_paths,
        var_names=var_names,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds is None:
        print(f"Skipping {folder.name}: no data cached.")
        continue

    try:
        if 'time' not in ds.dims or len(ds.time) == 0:
            print(f"Skipping {folder.name}: no time dimension found.")
            continue

        time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
        print(f"  Found {len(time_values)} timestep(s): {time_values[0]} -> {time_values[-1]}")

        # --- Loop over all timesteps ---
        for idx in range(len(time_values)):
            actual_dt = np.datetime64(time_values[idx], 'ns')
            actual_label = str(np.datetime_as_string(actual_dt, unit='s')).replace('T', ' ')
            actual_tag = str(np.datetime_as_string(actual_dt, unit='D'))
            print(f"  Plotting timestep {idx+1}/{len(time_values)}: {actual_label}")

            ds_t = ds.isel(time=idx)

            # --- Loop over all variables ---
            for var_name in var_names:
                if var_name not in ds_t:
                    print(f"    Skipping variable {var_name}: not found in dataset.")
                    continue

                current_cfg = configs[var_name]

                fig, ax = plt.subplots(figsize=(12, 8))
                pc = ds_t[var_name].ugrid.plot(
                    ax=ax,
                    cmap=current_cfg['cmap'],
                    add_colorbar=False,
                    edgecolors='none',
                    vmin=current_cfg['vmin'],
                    vmax=current_cfg['vmax']
                )
                ax.set_aspect('equal')
                ax.set_title(f"{current_cfg['label']} | {folder.name} | {actual_label}", color='black')

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.1)
                cbar = plt.colorbar(pc, cax=cax)
                cbar.set_label(current_cfg['label'])

                plt.tight_layout()
                save_name = f"{current_cfg['file_tag']}_{actual_tag}_{folder.name}.png"
                save_path = output_plots_dir / save_name
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)  # prevents memory issues over many timesteps
                print(f"    Saved: {save_name}")
    finally:
        ds.close()

print("\n" + "="*30)
print("BATCH PLOTTING COMPLETE")
print("="*30)

