"""Very simple map plot"""
#%% 
import os
import matplotlib.pyplot as plt
import dfm_tools as dfmt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%%

# --- 1. SETTINGS & PATHS ---
# Update these to match your actual folders
model_location = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\MF50_sens.8778435"
# discharge = 500
# scenario = f"01_baserun{discharge}"
# # This pattern finds all partitioned map files
# file_pattern = os.path.join(model_location, f"Q{discharge}", scenario, 'DFM_OUTPUT_*', "*_map.nc")
file_pattern = os.path.join(model_location, 'output', '*_map.nc')

# --- 2. CUSTOM COLORMAP ---
def create_terrain_colormap():
    colors = [
        (0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
        (0.40, "#00ffff"), (0.50, "#ffffcc"), (0.60, "#ffcc00"),
        (0.75, "#cc6600"), (0.90, "#228B22"), (1.00, "#006400"),
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

terrain_cmap = create_terrain_colormap()

# --- 3. LOADING DATA ---
print(f"Searching for files: {file_pattern}")
ds = dfmt.open_partitioned_dataset(file_pattern)

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

# #Check if the variable exists and has a time dimension

# if var_name not in ds:
#     print(f"Variable {var_name} not found. Available variables: {list(ds.data_vars)}")
# else:
#     # If the variable has 'time', take the last one. Otherwise, take it as is.
#     if 'time' in ds[var_name].dims:
#         data_to_plot = ds[var_name].isel(time=-1)
#         print("Selected the last timestep.")
#     else:
#         data_to_plot = ds[var_name]
#         print("No time dimension found, plotting the static map.")

if var_name not in ds:
    print(f"Variable {var_name} not found.")
else:
    if 'time' in ds[var_name].dims:
        # Select the last timestep
        data_to_plot = ds[var_name].isel(time=-1)
        
        # 1. Get the actual timestamp
        # Converting to string and taking the first 16 characters for YYYY-MM-DD HH:MM
        raw_time = ds['time'].isel(time=-1).values
        timestamp_str = str(raw_time).split('.')[0].replace('T', ' ') 
        
        # 2. Get the MORFAC value at the last timestep
        # We use .values to get the number from the array
        current_morfac = ds['morfac'].isel(time=-1).values
        
        title_info = f"{timestamp_str} | MORFAC = {current_morfac:.0f}"
        print(f"Selected last timestep: {title_info}")
    else:
        data_to_plot = ds[var_name]
        title_info = "Static Map"
        
# --- 4. PLOTTING ---
fig, ax = plt.subplots(figsize=(12, 8))

# Use the ugrid plotting engine
pc = data_to_plot.ugrid.plot(
    ax=ax, 
    cmap=terrain_cmap, 
    add_colorbar=False, 
    edgecolors='none', # 'none' makes it look smoother/terrain-like
    vmin = -15,
    vmax = 13
)

# Formatting
ax.set_aspect('equal')
ax.set_title(f"Bed level on {title_info}", color='black')

# Add colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1)
cbar = plt.colorbar(pc, cax=cax)
cbar.set_label('Bed Level [m]')

plt.tight_layout()

# Save and Show
save_path = os.path.join(model_location, f"terrain_map_final.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {save_path}")
plt.show()

# Clean up
ds.close()
# %%
