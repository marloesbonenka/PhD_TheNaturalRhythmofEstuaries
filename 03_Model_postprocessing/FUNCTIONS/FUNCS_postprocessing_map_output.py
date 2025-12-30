import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap# MAP OUTPUT PLOTTING
import plotly.express as px
from mpl_toolkits.axes_grid1 import make_axes_locatable

edgecolor = 'white'

def create_terrain_colormap():
    """Create a custom terrain-like colormap for bed level visualization."""
    colors = [
        (0.00, "#000066"),   # deep water
        (0.10, "#0000ff"),   # blue
        (0.30, "#00ffff"),   # cyan
        (0.40, "#00ffff"),   # water edge
        (0.50, "#ffffcc"),   # land edge
        (0.60, "#ffcc00"),   # orange
        (0.75, "#cc6600"),   # brown
        (0.90, "#228B22"),   # green
        (1.00, "#006400"),   # dark green
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

def create_wlev_colormap():
    return cm.get_cmap('RdBu_r')  # reversed Red-Blue with white at 0

# def create_wlev_colormap():
#     colors = [
#             (0.00,  "#a50026"),   
#             (0.35,  "#a50026"),  
#             (0.50,  "#ffffff"),  
#             (0.65,  "#74add1"), 
#             (1.00,  "#313695") 
#     ]

#     return LinearSegmentedColormap.from_list("custom_terrain", colors)

def create_velocity_colormap():
    return cm.get_cmap('PiYG')  # perceptually clear diverging

# def create_velocity_colormap():
#     """
#     Create a custom sequential colormap for velocity visualization.
#     Colors from dark blue (low velocity) to bright yellow (high velocity).
#     """
#     colors = [
#         (0.00, "#a50026"),   # very dark blue (almost black)
#         (0.35, "#a50026"),   # purple
#         (0.50, "#ffffff"),   # magenta
#         (0.65, "#74add1"),   # coral
#         (1.00, "#313695"),   # bright yellow-orange
#     ]
#     return LinearSegmentedColormap.from_list("custom_velocity", colors)

def create_depth_colormap():
    """
    Sequential colormap for water depth.
    Shallow (light blue) to deep (dark blue).
    """
    colors = [
        (0.0, "#d0f0f9"),  # very light blue (shallow ~2.5 m)
        (0.3, "#74add1"),  # light-medium blue
        (0.6, "#4575b4"),  # medium blue
        (1.0, "#08306b"),  # dark navy blue (deep ~15 m)
    ]
    return LinearSegmentedColormap.from_list("depth_seq", colors)

def plot_map(x, y, data, param_type, col_indices, N_coords, timestep, scenario, save_dir, save_figure=False):
    """
    Plot bed level with cross-section lines overlaid.
    
    Parameters:
    -----------
    x, y : np.ndarray
        Coordinate arrays
    data : np.ndarray
        Data (2D)
    col_indices : np.ndarray
        Column indices for cross-sections
    N_coords : list
        Row indices for cross-sections
    title : str
        Plot title
    vmin, vmax, vcenter : float
        Colormap limits and center
    """
    plot_cross_sections = False

    # Ensure arrays are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    data = np.asarray(data)
    
    if param_type == 'bed_level': 
        title = f"bed level at t = {timestep} for {scenario}"
        # Create colormap and normalization
        terrain_like = create_terrain_colormap()
        label = 'bed level [m]'
        vmin=-15
        vmax=10
        vcenter=0
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        plot_data = -data

    elif param_type == 'water_level': 
        title = f"water level at t = {timestep} for {scenario}"
        # Create colormap and normalization
        terrain_like = create_wlev_colormap()
        label = 'water level [m]'
        vmin=-3
        vmax=3
        vcenter=0
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        plot_data = data
    
    elif param_type == 'water_depth': 
        title = f"water depth at t = {timestep} for {scenario}"
        # Create colormap and normalization
        terrain_like = create_depth_colormap()
        label = 'water depth [m]'
        vmin=0
        vmax=15
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        plot_data = data
    
    else:
        print('error: no parameter type specified')

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bed level
    mesh = ax.pcolormesh(x/1000, y/1000, plot_data, shading='auto', cmap=terrain_like, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', label=label)      
    cbar.set_label(label)
    cbar.outline.set_edgecolor(edgecolor)
    
    # Plot cross-section lines
    if plot_cross_sections:
        for i, N in enumerate(N_coords):
            x_cross = x[N, col_indices]
            y_cross = y[N, col_indices]
            ax.plot(x_cross, y_cross, color='darkred', linestyle='dashed', linewidth=1)
    
    # Formatting
    ax.set_title(title)
    ax.set_xlabel('x-coordinate [km]')
    ax.set_ylabel('y-coordinate [km]')
    ax.tick_params(axis='both', which='major')  
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
        
    # Remove or recolor main plot outline
    for spine in ax.spines.values():
        spine.set_edgecolor(edgecolor)  # or 'white' for white outline

    sns.despine(ax=ax)
    plt.tight_layout()
    
    if save_figure:
        figname = os.path.join(save_dir, f'map_{param_type}_at_timestep_{timestep}_{edgecolor}.png')
        plt.savefig(figname, bbox_inches='tight', transparent=True)

    plt.show()
    

def plot_velocity(x, y, data, velocity_type, col_indices, N_coords, timestep, scenario, save_dir, save_figure=False):
    """
    Plot velocity map with cross-section lines overlaid.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate arrays
    data : np.ndarray
        Velocity data (2D)
    velocity_type : str
        Type of velocity ('U1' or 'V1')
    col_indices : np.ndarray
        Column indices for cross-sections
    N_coords : list
        Row indices for cross-sections
    """
    plot_cross_sections = False

    # Ensure arrays are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    data = np.asarray(data)

    # Set title and label based on velocity type
    if velocity_type == 'U1':
        title = f'velocity (U1, x-direction) at t = {timestep} for {scenario}'
        label = 'velocity U1 [m/s]'
    elif velocity_type == 'V1':
        title = f'velocity (V1, y-direction at t = {timestep} for {scenario}'
        label = 'velocity V1 [m/s]'
    else:
        print('error: unknown velocity type')
        return

    # Create colormap and normalization
    terrain_like = create_velocity_colormap()
    vmin = np.nanmin(-2)
    vmax = np.nanmax(2)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    mesh = ax.pcolormesh(x, y, data, shading='auto', cmap=terrain_like, norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cbar = fig.colorbar(mesh, cax=cax, orientation='vertical', label=label)    
    cbar.set_label(label)  

    # Plot cross-section lines
    if plot_cross_sections:
        for i, N in enumerate(N_coords):
            x_cross = x[N, col_indices]
            y_cross = y[N, col_indices]
            ax.plot(x_cross, y_cross, color='darkred', linestyle='dashed', linewidth=1)
    
    # Formatting
    ax.set_title(title)
    ax.set_xlabel('x-coordinate [m]')
    ax.set_ylabel('y-coordinate [m]')
    ax.tick_params(axis='both', which='major')  
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_figure:
        figname = os.path.join(save_dir, f'map_{velocity_type}_at_timestep_{timestep}.png')
        plt.savefig(figname, bbox_inches='tight', transparent=True)
    
    plt.show()

    fig = px.imshow(
        data.T,                   # Transpose the data to correct orientation
        origin='lower',
        color_continuous_scale='RdBu',
        zmin=-2,
        zmax=2,
        labels={'color': 'velocity [m/s]'},
        aspect='equal'
    )

    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='y'
    )

    fig.show()