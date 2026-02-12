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
base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output"
model_folders = ['1_Q500_rst.9093769',
                 '2_Q500_rst_seasonal.9093860',
                 '3_Q500_rst_flashy.9094053',
                 '4_Q500_rst_singlepeak.9093985']  # Add more folders as needed

# --- Variable selection ---
# Specify one or more variables to load and plot
var_names = ['mesh2d_mor_bl', 'mesh2d_s1', 'mesh2d_taus']  # e.g. ['mesh2d_mor_bl'] or all three
time_to_extract = -1

# Configs for each variable
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

    # Load all requested variables at once, memory-efficient for time
    if time_to_extract is None:
        ds = dataset_cache.get_partitioned(file_pattern, variables=var_names)
    else:
        ds = dataset_cache.get_partitioned(file_pattern, variables=var_names, preprocess=select_time)

    # Now plot each variable present in ds and in var_names
    for var_name in var_names:
        if var_name not in ds:
            print(f"Skipping {folder}: Variable {var_name} not found.")
            continue
        current_cfg = configs[var_name]
        cmap = current_cfg['cmap']

        # Extract timing and data
        if 'time' in ds[var_name].dims:
            data_to_plot = ds[var_name]
            try:
                raw_time = ds['time'].values
                if hasattr(raw_time, '__len__') and len(raw_time) == 1:
                    raw_time = raw_time[0]
                timestamp_str = str(raw_time).split('.')[0].replace('T', ' ')
            except Exception:
                timestamp_str = "Unknown time"
        else:
            data_to_plot = ds[var_name]
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

    save_path = os.path.join(output_plots_dir, save_name)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Successfully saved: {save_name}")
    
dataset_cache.close_all()

print("\nBatch processing complete.")
# %%
