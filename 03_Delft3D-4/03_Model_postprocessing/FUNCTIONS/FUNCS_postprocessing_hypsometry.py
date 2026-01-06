#%% 
"""Delft3D-4 Flow NetCDF Analysis: Morphological Estuary Analysis: compute hypsometric curves.
Last edit: August 2025
Author: Marloes Bonenkamp
"""

#%% IMPORTS AND SETUP
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import os
import sys

def calculate_hypsometric_curve(bed_level, x, y, x_min, x_max, y_min, y_max, 
                               bed_threshold=6, n_bins=50):
    """
    Calculate hypsometric curve for a given bed level array within specified bounds.
    
    Parameters:
    - bed_level: 2D array of bed levels (ny, nx)
    - x, y: 2D coordinate arrays
    - x_min, x_max, y_min, y_max: Domain bounds
    - bed_threshold: Threshold to exclude land areas
    - n_bins: Number of elevation bins
    
    Returns:
    - elevations: Array of elevation bin centers
    - cumulative_area: Array of cumulative areas (km²)
    """
    
    print(f"Input shapes - bed_level: {bed_level.shape}, x: {x.shape}, y: {y.shape}")
    
    # Create meshgrid of all coordinate points
    x_flat = x.flatten()
    y_flat = y.flatten()
    bed_flat = bed_level.flatten()
    
    # Find points within estuary bounds
    estuary_mask = ((x_flat >= x_min) & (x_flat <= x_max) & 
                   (y_flat >= y_min) & (y_flat <= y_max))
    
    # Extract estuary data
    x_estuary = x_flat[estuary_mask]
    y_estuary = y_flat[estuary_mask]
    bed_estuary = bed_flat[estuary_mask]
    
    # Mask out land areas (bed level >= threshold)
    water_mask = bed_estuary < bed_threshold
    
    # Get valid bed levels and corresponding coordinates
    valid_bed_levels = bed_estuary[water_mask]
    valid_x = x_estuary[water_mask]
    valid_y = y_estuary[water_mask]
    
    print(f"Found {len(valid_bed_levels)} valid water points in estuary region")
    
    if len(valid_bed_levels) == 0:
        print("Warning: No valid water points found in estuary region")
        return np.array([]), np.array([])
    
    # For variable grid, we need to estimate cell areas more carefully
    # Use a simplified approach: estimate local grid spacing
    cell_areas = []
    
    # Create coordinate reference arrays for spacing calculation
    unique_x = np.unique(x_flat)
    unique_y = np.unique(y_flat)
    unique_x = np.sort(unique_x)
    unique_y = np.sort(unique_y)
    
    for i in range(len(valid_x)):
        x_coord = valid_x[i]
        y_coord = valid_y[i]
        
        # Find nearest x and y grid lines
        x_idx = np.argmin(np.abs(unique_x - x_coord))
        y_idx = np.argmin(np.abs(unique_y - y_coord))
        
        # Calculate local grid spacing
        if x_idx > 0 and x_idx < len(unique_x) - 1:
            dx = (unique_x[x_idx + 1] - unique_x[x_idx - 1]) / 2
        elif x_idx == 0:
            dx = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 100  # fallback
        else:
            dx = unique_x[-1] - unique_x[-2] if len(unique_x) > 1 else 100  # fallback
            
        if y_idx > 0 and y_idx < len(unique_y) - 1:
            dy = (unique_y[y_idx + 1] - unique_y[y_idx - 1]) / 2
        elif y_idx == 0:
            dy = unique_y[1] - unique_y[0] if len(unique_y) > 1 else 100  # fallback
        else:
            dy = unique_y[-1] - unique_y[-2] if len(unique_y) > 1 else 100  # fallback
        
        cell_area = dx * dy  # in m²
        cell_areas.append(cell_area)
    
    cell_areas = np.array(cell_areas)
    
    # Create elevation bins
    min_elevation = np.min(valid_bed_levels)
    max_elevation = np.max(valid_bed_levels)
    
    print(f"Elevation range: {min_elevation:.2f} to {max_elevation:.2f} m")
    
    bin_edges = np.linspace(min_elevation, max_elevation, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate cumulative area for each elevation
    cumulative_areas = []
    
    for elevation in bin_centers:
        # Find all areas at or below this elevation
        below_mask = valid_bed_levels <= elevation
        total_area = np.sum(cell_areas[below_mask])  # in m²
        total_area_km2 = total_area / 1e6  # convert to km²
        cumulative_areas.append(total_area_km2)
    
    return bin_centers, np.array(cumulative_areas)

def plot_hypsometric_curves(bedlev, x, y, x_min, x_max, y_min, y_max, 
                           bed_threshold=6, timesteps=None, reference_timestep=0,
                           scenario='', save_dir='', save_figure=True, scenario_colormaps=None):
    """
    Plot hypsometric curves for multiple timesteps.
    
    Parameters:
    - bedlev: 3D array of bed levels (time, ny, nx)
    - x, y: 2D coordinate arrays
    - x_min, x_max, y_min, y_max: Estuary bounds
    - bed_threshold: Threshold to exclude land
    - timesteps: Array of timesteps to plot (default: first 10 with step 2)
    - reference_timestep: Timestep to plot as grey reference line
    - scenario: Scenario name for plot title
    - save_dir: Directory to save figure
    - save_figure: Whether to save the figure
    """
    if scenario_colormaps is None:
        # Map scenario names to matplotlib colormaps
        scenario_colormaps = {
            'baserun': plt.cm.Blues,
            'seasonal': plt.cm.Oranges,
            'flashy': plt.cm.Greens
        }
    
    
    if timesteps is None:
        timesteps = np.arange(1, min(10, bedlev.shape[0]), 2)
    
    plt.figure()
    
    # Plot reference timestep (t=0) as grey line
    print(f"Calculating hypsometric curve for reference timestep {reference_timestep}...")
    elevations_ref, areas_ref = calculate_hypsometric_curve(
        bedlev[reference_timestep], x, y, x_min, x_max, y_min, y_max, bed_threshold
    )
    
    if len(elevations_ref) > 0:
        plt.plot(areas_ref, elevations_ref, color='grey', 
                label=f't = {reference_timestep} (reference)', alpha=0.8)
    
    # Choose colormap based on scenario, default to Blues if unknown
    colormap = scenario_colormaps.get(scenario, plt.cm.Blues)
    
    # Plot other timesteps with Blues colormap
    if len(timesteps) > 0:
        colors = colormap(np.linspace(0.4, 1.0, len(timesteps)))
        
        for i, timestep in enumerate(timesteps):
            if timestep >= bedlev.shape[0]:
                print(f"Warning: Timestep {timestep} exceeds available data range (max: {bedlev.shape[0]-1})")
                continue
                
            print(f"Calculating hypsometric curve for timestep {timestep}...")
            elevations, areas = calculate_hypsometric_curve(
                bedlev[timestep], x, y, x_min, x_max, y_min, y_max, bed_threshold
            )
            
            if len(elevations) > 0:
                plt.plot(areas, elevations, color=colors[i], 
                        label=f't = {timestep}', alpha=0.9)
    
    # Formatting
    plt.xlabel('Cumulative area [km²]')
    plt.ylabel('Elevation [m]')
    plt.title(f'Hypsometric curves - estuary evolution\n{scenario}')
    plt.grid(True)
    
    # Add horizontal line at bed threshold
    plt.axhline(y=bed_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'land threshold ({bed_threshold} m)')
    
    plt.legend(loc='lower right', labelcolor='linecolor')
    
    plt.tight_layout()
    
    if save_figure and save_dir:
        filename = f'hypsometric_curves_{scenario}_t{reference_timestep}_{timesteps[0]}to{timesteps[-1]}.png'
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', transparent=True)
        print(f"Hypsometric curves saved to {save_dir}")
    
    plt.show()
    
    return elevations_ref, areas_ref

def plot_scenario_comparison_hypsometric(all_scenario_data, x_min, x_max, y_min, y_max, 
                           bed_threshold=6, reference_timestep=0, final_timestep=-1, 
                           scenario_colors=None, save_dir='', save_figure=True, discharge=500):
    """
    Plot hypsometric curves comparing different scenarios.
    Shows t=0 from baserun in grey, and final timestep for all scenarios in different colors.
    
    Parameters:
    - all_scenario_data: Dict with scenario names as keys, each containing (elevations, areas, x, y, bedlev)
    - reference_timestep: Which timestep to use as grey reference (default: 0)
    - final_timestep: Which timestep to use for scenario comparison (default: -1, last timestep)
    - scenario_colors: Dict mapping scenario names to colors
    - save_dir: Directory to save figure
    - save_figure: Whether to save the figure
    - discharge: Discharge value for labeling
    """
    
    if scenario_colors is None:
        scenario_colors = {
            'baserun': 'tab:blue',
            'seasonal': 'tab:orange', 
            'flashy': 'tab:green'
        }
    
    # Map scenario names to descriptive labels
    label_map = {
        'baserun': 'Constant discharge',
        'seasonal': 'Seasonal discharge',
        'flashy': 'Flashy discharge'
    }

    plt.figure()
    
    # Plot reference timestep (t=0) from baserun in grey
    baserun_key = None
    for key in all_scenario_data.keys():
        if 'baserun' in key.lower():
            baserun_key = key
            break
    
    if baserun_key and baserun_key in all_scenario_data:
        x, y, bedlev = all_scenario_data[baserun_key]
        
        if reference_timestep < bedlev.shape[0]:
            elevations_ref, areas_ref = calculate_hypsometric_curve(
                bedlev[reference_timestep], x, y, x_min, x_max, y_min, y_max, bed_threshold
            )
            
            if len(elevations_ref) > 0:
                plt.plot(areas_ref, elevations_ref, color='grey', 
                        label=f't = {reference_timestep} (reference)', alpha=0.8, linestyle='--')
    
    # Plot final timestep for each scenario
    for scenario_name, (x, y, bedlev) in all_scenario_data.items():
        # Determine final timestep index
        final_idx = final_timestep if final_timestep >= 0 else bedlev.shape[0] - 1
        
        if final_idx >= bedlev.shape[0]:
            final_idx = bedlev.shape[0] - 1
            
        print(f"Plotting {scenario_name} at timestep {final_idx}")
        
        elevations, areas = calculate_hypsometric_curve(
            bedlev[final_idx], x, y, x_min, x_max, y_min, y_max, bed_threshold
        )
        
        if len(elevations) > 0:
            # Determine color and label
            color = 'black'  # default
            clean_name = scenario_name.replace(f'{discharge}', '').replace('_', ' ')
            
            for scenario_key, scenario_color in scenario_colors.items():
                if scenario_key.lower() in scenario_name.lower():
                    color = scenario_color
                    break
            
            # Determine label using label_map
            label_name = None
            for scenario_key, descriptive_label in label_map.items():
                if scenario_key.lower() in scenario_name.lower():
                    label_name = descriptive_label
                    break
            if label_name is None:
                label_name = scenario_name.replace(f'{discharge}', '').replace('_', ' ')
                
            plt.plot(areas, elevations, color=color, 
                    label=f'{label_name}', alpha=0.9)
    
    # Formatting
    plt.xlabel('Cumulative area [km²]')
    plt.ylabel('Elevation [m]')
    plt.title(f'Hypsometric curves - scenario comparison\nQ = {discharge} [m³/s]')
    plt.grid(True)
    plt.legend(loc='lower right', labelcolor='linecolor')
    
    # # Add horizontal line at bed threshold
    # plt.axhline(y=bed_threshold, color='red', linestyle=':', alpha=0.7, 
    #             label=f'land threshold ({bed_threshold} m)')
    
    plt.tight_layout()
    
    if save_figure and save_dir:
        filename = f'hypsometric_comparison_Q{discharge}_scenarios.png'
        plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', transparent=True)
        print(f"Scenario comparison plot saved to {save_dir}")
    
    plt.show()

# %%
