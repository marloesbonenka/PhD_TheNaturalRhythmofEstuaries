"""Batch plot final bed levels for all MF folders"""
#%% 
import os
import matplotlib.pyplot as plt
import dfm_tools as dfmt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd

# --- 1. SETTINGS & PATHS ---
base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\Tmorph_400years"
morfyears = 400

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

# --- 3. MORPHOLOGICAL TIME SELECTION ---
def find_timestep_for_target_morphtime(ds, target_morph_years=morfyears, start_date='2025-01-01'):
    """
    Find the timestep where morphological time reaches the target.
    Calculates: morph_time = hydro_time_elapsed * morfac
    """
    start_timestamp = pd.Timestamp(start_date)
    times = pd.to_datetime(ds['time'].values)
    
    # Calculate elapsed hydrodynamic time in years for each timestep
    hydro_elapsed_years = np.array([(t - start_timestamp).days / 365.25 for t in times])
    
    # Get MORFAC values at each timestep
    if 'morfac' in ds:
        morfac_values = ds['morfac'].values
    else:
        raise ValueError("MORFAC variable not found in dataset")
    
    # Calculate morphological time at each timestep: morph_time = hydro_time * morfac
    morph_time_years = hydro_elapsed_years * morfac_values
    
    # Find closest timestep to target morphological time
    time_diffs = np.abs(morph_time_years - target_morph_years)
    closest_idx = int(np.argmin(time_diffs))
    
    actual_morph_years = morph_time_years[closest_idx]
    actual_hydro_years = hydro_elapsed_years[closest_idx]
    actual_morfac = morfac_values[closest_idx]
    actual_time = times[closest_idx]
    
    return closest_idx, actual_time, actual_hydro_years, actual_morph_years, actual_morfac

# --- 4. EXTRACT COMPUTATION TIME FROM .dia FILE ---
def extract_computation_time(dia_file_path):
    """
    Extract computation time in days and hours from FlowFM_0000.dia file.
    Returns tuple: (days, hours) or (None, None) if not found.
    """
    try:
        with open(dia_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        comp_time_days = None
        comp_time_hours = None
        
        for line in lines:
            if 'total computation time (d)' in line:
                # Extract the number from the line
                # Example: "** INFO   : total computation time (d)  :             4.6385087087"
                parts = line.split(':')
                if len(parts) >= 3:  # There are multiple colons
                    # Take the last part after the last colon
                    try:
                        comp_time_days = float(parts[-1].strip())
                    except ValueError:
                        pass
                        
            elif 'total computation time (h)' in line:
                parts = line.split(':')
                if len(parts) >= 3:
                    try:
                        comp_time_hours = float(parts[-1].strip())
                    except ValueError:
                        pass
        
        return comp_time_days, comp_time_hours
    
    except Exception as e:
        print(f"    Warning: Could not read .dia file: {e}")
        return None, None

# --- 5. LOOP THROUGH DIRECTORIES ---
# Get all subdirectories starting with 'MF'
model_folders = [f for f in os.listdir(base_directory) if f.startswith('MF') and os.path.isdir(os.path.join(base_directory, f))]

print(f"Found {len(model_folders)} folders to process.")

for folder in model_folders:
    model_location = os.path.join(base_directory, folder)
    
    # Create output_plots directory inside each model folder
    output_plots_dir = os.path.join(model_location, 'output_plots')
    os.makedirs(output_plots_dir, exist_ok=True)

    file_pattern = os.path.join(model_location, 'output', '*_map.nc')
    
    print(f"\nProcessing: {folder}")
    
    # Extract computation time from .dia file
    dia_file = os.path.join(model_location, 'output', 'FlowFM_0000.dia')
    comp_days, comp_hours = extract_computation_time(dia_file)
    
    if comp_days is not None and comp_hours is not None:
        print(f"  ** INFO   : total computation time (d)  : {comp_days:20.10f}")
        print(f"  ** INFO   : total computation time (h)  : {comp_hours:20.10f}")
    
    try:
        # Load the partitioned dataset
        ds = dfmt.open_partitioned_dataset(file_pattern)
        
        if var_name not in ds:
            print(f"Skipping {folder}: Variable {var_name} not found.")
            ds.close()
            continue
        
        # Find the correct timestep based on morphological time
        if 'time' in ds[var_name].dims:
            closest_idx, actual_time, actual_hydro_years, actual_morph_years, current_morfac = \
                find_timestep_for_target_morphtime(ds, target_morph_years=morfyears)
            
            print(f"  MORFAC: {current_morfac:.1f}")
            print(f"  Hydrodynamic time elapsed: {actual_hydro_years:.2f} years")
            print(f"  Morphological time: {actual_morph_years:.2f} years")
            print(f"  Hydrodynamic date: {actual_time}")
            print(f"  Time index: {closest_idx}")
            
            data_to_plot = ds[var_name].isel(time=closest_idx)
            
            timestamp_str = str(actual_time).split('.')[0].replace('T', ' ')
            title_info = f"{timestamp_str} | MORFAC = {current_morfac:.0f} | Tmorph = {actual_morph_years:.1f} years"
        else:
            data_to_plot = ds[var_name]
            title_info = "Static Map"

        # --- 6. PLOTTING ---
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
        
        # Close plot and dataset to save memory
        plt.close(fig)
        ds.close()

    except Exception as e:
        print(f"Error processing {folder}: {e}")

print("\nBatch processing complete.")
# %%
