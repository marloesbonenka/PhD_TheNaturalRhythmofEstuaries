"""Very simple map plot"""
#%% 
from pathlib import Path
import matplotlib.pyplot as plt
import dfm_tools as dfmt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from FUNCTIONS.F_general import terrain_cmap
from FUNCTIONS.F_cache import *
#%%

# --- 1. SETTINGS & PATHS ---
# Update these to match your actual folders
base_location = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
model_location = base_location / 'Test_MORFAC' / '02_seasonal' / 'Tmorph_50years' / 'MF50_sens.8778435'
# discharge = 500
# scenario = f"01_baserun{discharge}"
# # This pattern finds all partitioned map files
# file_pattern = model_location / f"Q{discharge}" / scenario / 'DFM_OUTPUT_*' / "*_map.nc"
file_pattern = model_location / 'output' / '*_map.nc'

# --- 2. LOADING DATA ---
print(f"Searching for files: {file_pattern}")
dataset_cache = DatasetCache()
ds = dataset_cache.get_partitioned(str(file_pattern))

#%%
# --- DIAGNOSTIC: WHAT IS IN MY DATASET? ---
print(f"{'Variable Name':<25} | {'Dimensions':<20} | {'Description'}")
print("-" * 80)

for var in ds.data_vars:
    # Get dimensions as a string
    dims = str(ds[var].dims)
    # Try to get the long_name attribute if it exists
    long_name = ds[var].attrs.get('long_name', 'No description')
    
    # Highlight variables that have 'time' in them
    indicator = " [TIME-DEPENDENT]" if 'time' in ds[var].dims else ""
    
    print(f"{var:<25} | {dims:<20} | {long_name}{indicator}")
#%%
var_name = 'mesh2d_mor_bl'

def build_title_info(time_index, label):
    raw_time = ds['time'].isel(time=time_index).values
    timestamp_str = str(raw_time).split('.')[0].replace('T', ' ')
    if 'morfac' in ds:
        current_morfac = ds['morfac'].isel(time=time_index).values
        return f"{label}: {timestamp_str} | MORFAC = {current_morfac:.0f}"
    return f"{label}: {timestamp_str}"

def plot_timestep(data, title_info, save_name):
    fig, ax = plt.subplots(figsize=(12, 8))
    pc = data.ugrid.plot(
        ax=ax,
        cmap=terrain_cmap,
        add_colorbar=False,
        edgecolors='none',
        vmin=-15,
        vmax=15
    )

    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = plt.colorbar(pc, cax=cax)
    cbar.set_label('Bed Level [m]')

    plt.tight_layout()

    save_path = model_location / 'output_plots' / save_name 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()

if var_name not in ds:
    print(f"Variable {var_name} not found.")
else:
    if 'time' in ds[var_name].dims:
        first_index = 0
        last_index = -1

        first_data = ds[var_name].isel(time=first_index)
        # first_title = build_title_info(first_index, "first timestep")
        first_title = 'at t = 0'
        print(f"Selected first timestep: {first_title}")
        plot_timestep(first_data, first_title, "terrain_map_first.png")

        last_data = ds[var_name].isel(time=last_index)
        last_title = build_title_info(last_index, "Last timestep")
        print(f"Selected last timestep: {last_title}")
        plot_timestep(last_data, last_title, "terrain_map_final.png")
    else:
        data_to_plot = ds[var_name]
        title_info = "Static Map"
        plot_timestep(data_to_plot, title_info, "terrain_map_static.png")

# Clean up
dataset_cache.close_all()
# %%
