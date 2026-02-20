"""Plot map output at a certain timestep"""
#%% 
import os
import matplotlib.pyplot as plt
import xugrid as xu
from mpl_toolkits.axes_grid1 import make_axes_locatable
from FUNCTIONS.F_cache import DatasetCache
from FUNCTIONS.F_general import create_bedlevel_colormap, create_water_colormap, create_shear_stress_colormap

#%%
def select_time(ds):
    if 'time' in ds.dims and time_to_extract is not None:
        return ds.isel(time=time_to_extract)
    return ds


#%% --- 1. SETTINGS & PATHS ---
DISCHARGE = 500  # or 1000, etc.
base_directory = f"U:\\PhDNaturalRhythmEstuaries\\Models\\1_RiverDischargeVariability_domain45x15\\Model_Output\\Q{DISCHARGE}"
assessment_dir = os.path.join(base_directory, 'cached_data', 'assessment_netcdfs')
model_folders = [
    f'1_Q{DISCHARGE}_rst.9093769',
    f'2_Q{DISCHARGE}_rst_seasonal.9093860',
    f'3_Q{DISCHARGE}_rst_flashy.9094053']

# --- Variable selection ---
var_names = ['mesh2d_mor_bl']#, 'mesh2d_s1', 'mesh2d_taus']  # e.g. ['mesh2d_mor_bl'] or all three
time_to_extract = -1

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

dataset_cache = DatasetCache()

for folder in model_folders:
    model_location = os.path.join(base_directory, folder)
    output_plots_dir = os.path.join(model_location, 'output_plots')
    os.makedirs(output_plots_dir, exist_ok=True)
    file_pattern = os.path.join(model_location, 'output', '*_map.nc')
    print(f"\nProcessing: {folder}")


    # Check for assessment file for mesh2d_mor_bl
    assessment_file = os.path.join(
        assessment_dir, f"assessment_mesh2d_mor_bl_{folder}.nc"
    )
    ds_assessment = None
    assessment_used = False
    if os.path.exists(assessment_file):
        print(f"Assessment file found for {folder}: {assessment_file}")
        import xarray as xr
        ds_assessment = xr.open_dataset(assessment_file)
        if time_to_extract is not None and 'time' in ds_assessment.dims:
            ds_assessment = ds_assessment.isel(time=time_to_extract)
        # If only mesh2d_mor_bl is requested, we can skip loading the partitioned dataset
        if var_names == ['mesh2d_mor_bl']:
            assessment_used = True

    # Only load the partitioned dataset if needed
    ds = None
    if not assessment_used:
        if time_to_extract is None:
            ds = dataset_cache.get_partitioned(file_pattern, variables=var_names)
        else:
            ds = dataset_cache.get_partitioned(file_pattern, variables=var_names, preprocess=select_time)

    for var_name in var_names:
        # If mesh2d_mor_bl and assessment file exists, use that
        if var_name == 'mesh2d_mor_bl' and ds_assessment is not None and var_name in ds_assessment:
            data_source = ds_assessment
            print(f"Using assessment file for {var_name} in {folder}")
        else:
            if ds is None or var_name not in ds:
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
        save_path = os.path.join(output_plots_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Successfully saved: {save_name}")

dataset_cache.close_all()

print("\nBatch processing complete.")
# %%
