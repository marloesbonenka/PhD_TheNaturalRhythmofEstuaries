import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import datetime

#%% HIS OUTPUT - ANALYZING AND PLOTTING
def extract_his_data(dataset_trih, variable, station_names):
    """
    Extract data from history file for specified stations.
    
    Parameters:
    -----------
    dataset_trih : netCDF4.Dataset
        History file dataset
    variable     : string 
        Variable from history file in which you are interested
    station_names : list
        List of station names to extract
    reference_date : datetime
        Reference date for time conversion
        
    Returns:
    --------
    dict: Dictionary with station names as keys and (time, variable) tuples as values
    """
    # Get all available station names FOR MY MODEL
    names_bytes = dataset_trih['NAMTRA'][:]
    all_station_names = [
        b''.join(names_bytes[i]).decode('utf-8').strip()
        for i in range(names_bytes.shape[0])
    ]
    
    # Get the variable data
    var_data = dataset_trih[variable][:]
    
    # Check dimensionality and handle accordingly
    if var_data.ndim == 3:
        # 3D array: (time, LSEDTOT, NTRUV)
        # Extract the first sediment fraction (index 0 in middle dimension)
        print(f"    Detected 3D variable '{variable}' with shape {var_data.shape}, extracting first sediment fraction")
        var_data = var_data[:, 0, :]  # Now shape is (time, NTRUV)

    elif var_data.ndim == 2:
        # 2D array: (time, NTRUV) - already in correct format
        print(f"    Detected 2D variable '{variable}' with shape {var_data.shape}")

    else:
        raise ValueError(f"Unexpected variable dimensions for '{variable}': {var_data.ndim}D array with shape {var_data.shape}")
    
    results = {}
    for name in station_names:
        if name not in all_station_names:
            raise ValueError(f"Station '{name}' not found. Available: {all_station_names}")
        
        idx = all_station_names.index(name)
        time = dataset_trih['time'][:]
        var = var_data[:, idx]  # Now this works for both CTR and SBTR
        results[name] = (time, var)
    
    return results, all_station_names

def his_plot_discharge_timeseries(discharge_results, station_names, reference_date, save_dir, save_figure=False, time_range=None):
    """
    Plot discharge time series for multiple stations.
    
    Parameters:
    -----------
    discharge_results : dict
        Results from extract_discharge_data
    station_names : list
        Station names to plot
    reference_date : datetime
        Reference date for time conversion
    time_range : tuple, optional
        (start_idx, end_idx) for time range to plot
    figsize : tuple
        Figure size
    """
    colors = ['green', 'pink', 'orange', 'blue', 'red', 'purple']
    
    for i, station in enumerate(station_names):
        time, discharge = discharge_results[station]
        datetimes_his = reference_date + pd.to_timedelta(time, unit='s')
        
        if time_range:
            start_idx, end_idx = time_range
            time_slice = slice(start_idx, end_idx)
            datetimes_plot = datetimes_his[time_slice]
            discharge_plot = discharge[time_slice]
        else:
            datetimes_plot = datetimes_his
            discharge_plot = discharge
        
        plt.figure(figsize=(12,4))
        plt.plot(datetimes_plot, discharge_plot, 
                label=f'{station} discharge', 
                color=colors[i % len(colors)])
        plt.xlabel('Time')
        plt.ylabel('Discharge [m³/s]')
        plt.title(f'Discharge at {station}')
        plt.legend()
        plt.tight_layout()

        if save_figure:
            figname = os.path.join(save_dir, f'his_CTR_at_cross_section_{station}_t_{start_idx}_{end_idx}.pdf')
            plt.savefig(figname, dpi=300, bbox_inches='tight')

        plt.show()

def his_plot_timeseries(var_results, station_names, reference_date, variable, variable_label, save_dir, save_figure=False, time_range=None):
    """
    Plot discharge time series for multiple stations.
    
    Parameters:
    -----------
    var_results     : dict
        Results from extract_discharge_data
    station_names   : list
        Station names to plot
    reference_date  : datetime
        Reference date for time conversion
    variable        : str
        Variable to look at (CTR for discharge, ZWL for water level etc.)
    variable_label  : str
        Label for variable to look at (CTR for discharge, ZWL for water level etc.)
    time_range      : tuple, optional
        (start_idx, end_idx) for time range to plot
    figsize         : tuple
        Figure size
    """
    colors = ['green', 'pink', 'orange', 'blue', 'red', 'purple']
    
    for i, station in enumerate(station_names):
        time, var = var_results[station]
        datetimes_his = reference_date + pd.to_timedelta(time, unit='s')
        
        if time_range:
            start_idx, end_idx = time_range
            time_slice = slice(start_idx, end_idx)
            datetimes_plot = datetimes_his[time_slice]
            discharge_plot = var[time_slice]
        else:
            datetimes_plot = datetimes_his
            discharge_plot = var
        
        plt.figure(figsize=(12,4))
        plt.plot(datetimes_plot, discharge_plot, 
                label=f'{station}', 
                color=colors[i % len(colors)])
        plt.xlabel('Time')
        plt.ylabel(f'{variable_label}')
        plt.title(f'{variable} at {station}')
        plt.legend()
        plt.tight_layout()

        if save_figure:
            figname = os.path.join(save_dir, f'his_{variable}_at_cross_section_{station}_t_{start_idx}_{end_idx}.pdf')
            plt.savefig(figname, dpi=300, bbox_inches='tight')

        plt.show()


def plot_detailed_multi_scenarios(all_results, control_values, scenario_templates, 
                                 station_names, model_location, 
                                 save_figure, time_start, time_end,
                                 get_runname_func, 
                                 variable='q1', variable_label='Q [m³/s]', 
                                 reference_date=datetime.datetime(2024, 1, 1)):
    """
    Create a detailed subplot arrangement with separate plots for each station and scenario
    
    Parameters:
    -----------
    all_results : dict
        Nested dictionary containing results for all scenarios
    control_values : list
        List of control parameter values (e.g., discharges, water levels)
    scenario_templates : list
        Template strings for scenario names
    station_names : list
        Names of monitoring stations
    model_location : str
        Path for saving figures
    save_figure : bool
        Whether to save the figure
    time_start : int
        Start index for time slicing
    time_end : int
        End index for time slicing
    get_runname_func : function
        Function to get runname from control value
    variable : str
        Variable to plot (e.g., 'q1', 's1', 'u1')
    variable_label : str
        Y-axis label for the variable (e.g., 'Q [m³/s]', 'Water Level [m]', 'Velocity [m/s]')
    control_param : str
        Name of control parameter (e.g., 'discharge', 'water_level')
    control_unit : str
        Unit of control parameter for titles
    reference_date : datetime
        Reference date for time conversion
    """
    
    n_stations = len(station_names)
    n_control = len(control_values)
    
    # Create figure with subplots (stations as rows, control values as columns)
    fig, axes = plt.subplots(n_stations, n_control, 
                            figsize=(6*n_control, 4*n_stations), 
                            sharey=False, sharex=True)
    
    if n_stations == 1:
        axes = axes.reshape(1, -1)
    if n_control == 1:
        axes = axes.reshape(-1, 1)
    
    # Colors and labels
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    scenario_labels = ['Baserun', 'Seasonal', 'Flashy']
    
    for i, station_name in enumerate(station_names):
        for j, control_value in enumerate(control_values):
            ax = axes[i, j]
            
            # Set titles
            if i == 0:
                runname = get_runname_func(control_value)
                scenario_id = runname.split('_')[0]  # Extract s1, s2, s3, etc.
                ax.set_title(f'{scenario_id}_{control_value}', 
                           fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{station_name}\n\n{variable_label}', fontsize=10)
            
            # Plot scenarios
            for k, scenario_template in enumerate(scenario_templates):
                # Create scenario name - handle different parameter names
                if '{discharge}' in scenario_template:
                    scenario_name = scenario_template.format(discharge=control_value)
                elif '{water_level}' in scenario_template:
                    scenario_name = scenario_template.format(water_level=control_value)
                else:
                    # Generic formatting - assumes single parameter
                    scenario_name = scenario_template.format(control_value)
                
                if (all_results[control_value][scenario_name] is not None and 
                    station_name in all_results[control_value][scenario_name]):
                    
                    results = all_results[control_value][scenario_name]
                    time, variable_data = results[station_name]
                    
                    # Convert time to datetime
                    real_time = reference_date + pd.to_timedelta(time, unit='s')   # or 'm' if needed

                    # Apply slice
                    real_time_slice = real_time[time_start:time_end]
                    variable_slice = variable_data[time_start:time_end]
                    
                    # Plot
                    ax.plot(real_time_slice, variable_slice, 
                           color=colors[k], alpha=0.8, linewidth=1,
                           label=scenario_labels[k] if i == 0 and j == 0 else "")
            
            ax.grid(True, alpha=0.3)
            if i == n_stations - 1:
                ax.set_xlabel('Time', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
    
    # Add legend
    if n_stations > 0 and n_control > 0:
        axes[0, 0].legend(fontsize=9)
    
    plt.tight_layout()
    
    # Save
    if save_figure:
        first_runname = get_runname_func(control_values[0])
        save_dir = os.path.join(model_location, first_runname, f'multi_{variable}_comparison')
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f'detailed_{variable}_comparison_all_scenarios.pdf'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Detailed figure saved to: {save_path}")
    
    plt.show()


# # Convenience wrapper functions for specific variables
# def plot_detailed_multi_discharge_scenarios(all_results, discharges, scenario_templates, 
#                                            station_names, model_location, 
#                                            save_figure, time_start, time_end, 
#                                            reference_date=datetime.datetime(2024, 1, 1)):
#     """Wrapper for discharge plotting (backwards compatibility)"""
#     return plot_detailed_multi_scenarios(
#         all_results, discharges, scenario_templates, station_names, 
#         model_location, save_figure, time_start, time_end,
#         variable='q1', variable_label='Q [m³/s]', 
#         control_param='discharge', control_unit='m³/s',
#         reference_date=reference_date
#     )


# def plot_detailed_multi_water_level_scenarios(all_results, water_levels, scenario_templates, 
#                                              station_names, model_location, 
#                                              save_figure, time_start, time_end, 
#                                              reference_date=datetime.datetime(2024, 1, 1)):
#     """Plot water level scenarios"""
#     return plot_detailed_multi_scenarios(
#         all_results, water_levels, scenario_templates, station_names, 
#         model_location, save_figure, time_start, time_end,
#         variable='ZWL', variable_label='Water Level [m]', 
#         control_param='water_level', control_unit='m',
#         reference_date=reference_date
#     )
