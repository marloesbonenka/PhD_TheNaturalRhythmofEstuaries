"""Batch plot final bed levels for all MF folders"""
#%% 
import os
import matplotlib.pyplot as plt
import dfm_tools as dfmt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from FUNCTIONS.F_cache import DatasetCache

# --- 1. SETTINGS & PATHS ---
base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\Tmorph_50years"

# --- 2. CUSTOM COLORMAP ---
def create_terrain_colormap():
    colors = [
        (0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
        (0.40, "#00ffff"), (0.50, "#ffffcc"), (0.60, "#ffcc00"),
        (0.75, "#cc6600"), (0.90, "#228B22"), (1.00, "#006400"),
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

terrain_cmap = create_terrain_colormap()
var_name = 'mesh2d_mor_bl'

# --- 3. LOOP THROUGH DIRECTORIES ---
# Get all subdirectories starting with 'MF'
model_folders = [f for f in os.listdir(base_directory) if f.startswith('MF') and os.path.isdir(os.path.join(base_directory, f))]

print(f"Found {len(model_folders)} folders to process.")

dataset_cache = DatasetCache()
try:
    for folder in model_folders:
        model_location = os.path.join(base_directory, folder)
        
        # Create output_plots directory inside each model folder
        output_plots_dir = os.path.join(model_location, 'output_plots')
        os.makedirs(output_plots_dir, exist_ok=True)

        file_pattern = os.path.join(model_location, 'output', '*_map.nc')
        
        print(f"\nProcessing: {folder}")
        
        try:
            # Load the partitioned dataset
            ds = dataset_cache.get_partitioned(file_pattern)
            
            if var_name not in ds:
                print(f"Skipping {folder}: Variable {var_name} not found.")
                continue

        # Extract timing and data
        if 'time' in ds[var_name].dims:
            data_to_plot = ds[var_name].isel(time=-1)
            raw_time = ds['time'].isel(time=-1).values
            timestamp_str = str(raw_time).split('.')[0].replace('T', ' ') 
            
            # Extract MORFAC if available
            if 'morfac' in ds:
                current_morfac = ds['morfac'].isel(time=-1).values
                title_info = f"{timestamp_str} | MORFAC = {current_morfac:.0f}"
            else:
                title_info = f"{timestamp_str}"
        else:
            data_to_plot = ds[var_name]
            title_info = "Static Map"

        # --- 4. PLOTTING ---
        fig, ax = plt.subplots(figsize=(12, 8))
        
        pc = data_to_plot.ugrid.plot(
            ax=ax, 
            cmap=terrain_cmap, 
            add_colorbar=False, 
            edgecolors='none',
            vmin=-15,
            vmax=13
        )

        ax.set_aspect('equal')
        ax.set_title(f"Bed level on {title_info}", color='black')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(pc, cax=cax)
        cbar.set_label('Bed Level [m]')

        plt.tight_layout()

        # Save in the specific model folder
        save_name = f"terrain_map_final_{folder}.png"
        save_path = os.path.join(output_plots_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Successfully saved: {save_name}")
        
            # Close plot to save memory
            plt.close(fig)

        except Exception as e:
            print(f"Error processing {folder}: {e}")
finally:
    dataset_cache.close_all()

print("\nBatch processing complete.")