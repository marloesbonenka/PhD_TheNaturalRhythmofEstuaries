import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors

# BRAIDING INDEX FUNCTIONS 
def count_channels(line1, line2):
    """
    Count channels for computing braiding index
    """
    crossings = 0
    for i in range(1, len(line1)):
        if (line1[i-1] > line2[i-1] and line1[i] < line2[i]) or (line1[i-1] < line2[i-1] and line1[i] > line2[i]):
            crossings += 1
    channels = crossings / 2 
    return channels

def get_cross_section_coordinates(x, y, x_targets=None, y_start=5000, y_end=10000):
    """
    Calculate cross-section coordinates for braiding index computation.
    
    Parameters:
    -----------
    x : np.ndarray
        X-coordinate array
    y : np.ndarray
        Y-coordinate array  
    x_targets : list, optional
        Target x-coordinates for cross-sections. Default: np.arange(20000, 44001, 1000)
    y_start : float
        Start of y-range for cross-sections
    y_end : float
        End of y-range for cross-sections
        
    Returns:
    --------
    tuple: (col_indices, N_coords, x_targets)
        col_indices: Column indices within y-range
        N_coords: Row indices closest to target x-coordinates
        x_targets: Target x-coordinates used
    """
    if x_targets is None:
        x_targets = np.arange(20000, 44001, 1000)
    
    # Get the x value for each row (all columns in a row are the same)
    x_row_values = x[:, 0]  # shape: (nrows,)
    
    # Get the y values for each column (all rows in a column are the same)
    y_col_values = y[0, :]  # shape: (ncols,)
    
    # Find column indices where y is within specified range
    col_mask = (y_col_values >= y_start) & (y_col_values <= y_end)
    col_indices = np.where(col_mask)[0]
    
    # For each target, find the row index with the closest x value
    N_coords = [np.argmin(np.abs(x_row_values - target)) for target in x_targets]
    
    return col_indices, N_coords, x_targets

def compute_BI_per_cross_section(x, y, tau_max, slice_start, map_output_interval, 
                                reference_date, theta=0.5, x_targets=None, 
                                y_start=5000, y_end=10000):
    """
    Compute braiding index per cross-section over time.
    
    Parameters:
    -----------
    x, y : np.ndarray
        Coordinate arrays
    tau_max : np.ndarray
        Maximum shear stress data (timesteps, rows, cols)
    slice_start : int
        Starting timestep index
    map_output_interval : int
        Map output interval in minutes
    reference_date : datetime
        Reference date for time conversion
    theta : float
        Threshold shear for channel counting
    x_targets : list, optional
        Target x-coordinates for cross-sections
    y_start, y_end : float
        Y-range for cross-sections
        
    Returns:
    --------
    tuple: (BI_per_cross_section, datetimes, x_targets, col_indices, N_coords)
    """

    n_timesteps = tau_max.shape[0]
    model_times = np.arange(slice_start, slice_start + n_timesteps) * map_output_interval
    datetimes = reference_date + pd.to_timedelta(model_times, unit='m')
    
    # Get cross-section coordinates
    col_indices, N_coords, x_targets = get_cross_section_coordinates(
        x, y, x_targets, y_start, y_end
    )
    
    BI_total = []
    BI_per_cross_section = []
    
    for i in range(n_timesteps):
        data = tau_max[i]
        No_channels = 0
        BI_xs = []
        
        for N in N_coords:
            z_cross = abs(data[N, col_indices])
            channels = count_channels(z_cross, [theta] * len(z_cross))
            No_channels += channels
            BI_xs.append(channels)
        
        BI = No_channels / len(N_coords)
        BI_total.append(BI)
        BI_per_cross_section.append(BI_xs)
    
    return BI_per_cross_section, datetimes, x_targets, col_indices, N_coords

def plot_braiding_index_timeseries(BI_per_cross_section, x_targets, datetimes, 
                                  n_timesteps=6):
    """
    Plot braiding index evolution over time for multiple cross-sections.
    
    Parameters:
    -----------
    BI_per_cross_section : list
        Braiding index data per cross-section
    x_targets : np.ndarray
        X-coordinates of cross-sections
    datetimes : pd.DatetimeIndex
        Datetime array
    n_timesteps : int
        Number of timesteps to plot
    """
    # Convert to DataFrame
    df_BI = pd.DataFrame(BI_per_cross_section, columns=x_targets)
    timesteps_to_plot = np.linspace(0, len(df_BI)-1, num=n_timesteps, dtype=int)
    
    # Create colormap
    colors = ['#add8e6', '#00008b']  # light blue to dark blue
    cmap = mcolors.LinearSegmentedColormap.from_list('seq_cmap', colors, N=n_timesteps)
    
    plt.figure(figsize=(10, 6))
    for i, t_idx in enumerate(timesteps_to_plot):
        plt.plot(
            x_targets,
            df_BI.iloc[t_idx, :],
            color=cmap(i / (n_timesteps - 1)),
            label=datetimes[t_idx].strftime('%Y-%m-%d %H:%M')
        )
    
    plt.xlabel('x-coordinate (m)')
    plt.ylabel('Braiding Index')
    plt.title('Braiding Index at Multiple Timesteps')
    plt.grid(True, alpha=0.3)
    plt.legend(title="Timestep (datetime)", loc='best')
    plt.tight_layout()
    plt.show()
    
    return df_BI

def plot_mean_braiding_index(df_BI, x_targets):
    """
    Plot time-averaged braiding index across cross-sections.
    
    Parameters:
    -----------
    df_BI : pd.DataFrame
        Braiding index DataFrame
    x_targets : np.ndarray
        X-coordinates of cross-sections
    """
    mean_BI = df_BI.mean(axis=0)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_targets, mean_BI, 'o-', color='crimson')
    plt.xlabel('x-coordinate (m)')
    plt.ylabel('Braiding Index')
    plt.title('Time-averaged Braiding Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()