"""Plot map output at a certain timestep"""
#%% 
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from FUNCTIONS.F_general import create_bedlevel_colormap, create_water_colormap, create_shear_stress_colormap
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi

#%% --- 1. SETTINGS ---
# Which scenarios to process (set to None or empty list for all)
SCENARIOS_TO_PROCESS = ['1', '2']#, '2', '3', '4']  # Use all scenarios
DISCHARGE = 1000
# --- Variable selection ---
var_names = ['mesh2d_mor_bl', 'mesh2d_s1', 'mesh2d_taus']  # e.g. ['mesh2d_mor_bl'] or all three
time_to_extract = -1

# Cache settings
CACHE_BBOX = None  # None = full domain
CACHE_TAG = "full"
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

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

# Mapping: restart folder prefix -> timed-out folder prefix
# 1 = constant (baserun), 2 = seasonal, 3 = flashy, 4 = singlepeak
VARIABILITY_MAP = {
    '1': f'01_baserun{DISCHARGE}',
    '2': f'02_run{DISCHARGE}_seasonal',
    '3': f'03_run{DISCHARGE}_flashy',
    '4': f'04_run{DISCHARGE}_singlepeak',
}

# Find all run folders: start with digit (with or without '_rst')
model_folders = [f.name for f in base_path.iterdir() 
                    if f.is_dir() and f.name[0].isdigit()]

if SCENARIOS_TO_PROCESS:
    try:
        scenario_filter = set(int(s) for s in SCENARIOS_TO_PROCESS)
    except Exception:
        scenario_filter = set()
    model_folders = [f for f in model_folders if int(f.split('_')[0]) in scenario_filter]
# Sort by leading number
model_folders.sort(key=lambda x: int(x.split('_')[0]))

print(f"Found {len(model_folders)} run folders in: {base_path}")

# model_folders = [
#     f'1_Q{DISCHARGE}_rst.9093769',
#     f'2_Q{DISCHARGE}_rst_seasonal.9093860',
#     f'3_Q{DISCHARGE}_rst_flashy.9094053']

configs = {
    'mesh2d_mor_bl': {
        'cmap': create_bedlevel_colormap(),
        'vmin': -15,
        'vmax': 15,
        'label': 'Bed Level [m]',
        'file_tag': 'bedlevel_map_final'
    },
    'mesh2d_s1': {
        'cmap': create_water_colormap(),
        'vmin': -1,   # Adjust based on your tide/datum
        'vmax': 3,
        'label': 'Water Level [m]',
        'file_tag': 'water_level_map_final'
    },
    'mesh2d_taus': {
        'cmap': create_shear_stress_colormap(),
        'vmin': 0,
        'vmax': 5,    # Adjust based on flow intensity
        'label': 'Bed Shear Stress [N/m²]',
        'file_tag': 'shear_stress_map_final'
    }
}

for folder in model_folders:
    model_location = base_path / folder
    output_plots_dir = model_location / 'output_plots'
    output_plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing: {folder}")

    run_paths = []
    scenario_num = folder.split('_')[0]
    if timed_out_dir is not None and scenario_num in VARIABILITY_MAP:
        timed_out_path = timed_out_dir / VARIABILITY_MAP[scenario_num]
        if timed_out_path.exists():
            run_paths.append(timed_out_path)
    run_paths.append(model_location)

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder,
        run_paths=run_paths,
        var_names=var_names,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds is None:
        print(f"Skipping {folder}: no data cached.")
        continue

    if time_to_extract is not None and 'time' in ds.dims:
        ds = ds.isel(time=time_to_extract)

    for var_name in var_names:
        if var_name not in ds:
            print(f"Skipping {folder}: Variable {var_name} not found.")
            continue
        data_source = ds
        current_cfg = configs[var_name]
        cmap = current_cfg['cmap']

        # Extract timing and data
        if 'time' in data_source[var_name].dims:
            data_to_plot = data_source[var_name]
            try:
                raw_time = data_source['time'].values
                if hasattr(raw_time, '__len__') and len(raw_time) == 1:
                    raw_time = raw_time[0]
                timestamp_str = str(raw_time).split('.')[0].replace('T', ' ')
            except Exception:
                timestamp_str = "Unknown time"
        else:
            data_to_plot = data_source[var_name]
            timestamp_str = f"at timestep {time_to_extract}"

        # --- 4. PLOTTING ---        
        fig, ax = plt.subplots(figsize=(12, 8))

        pc = data_to_plot.ugrid.plot(
            ax=ax,
            cmap=cmap,
            add_colorbar=False,
            edgecolors='none',
            vmin=current_cfg['vmin'],
            vmax=current_cfg['vmax']
        )
        ax.set_aspect('equal')
        ax.set_title(f"{current_cfg['label']} on {timestamp_str}", color='black')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(pc, cax=cax)
        cbar.set_label(current_cfg['label'])
        plt.tight_layout()
        save_name = f"{current_cfg['file_tag']}_{folder}.png"
        save_path = output_plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Successfully saved: {save_name}")

print("\nBatch processing complete.")
# %%
