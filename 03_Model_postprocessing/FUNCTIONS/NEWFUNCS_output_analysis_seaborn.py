import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

#%% HELPER FUNCTIONS
def check_available_variables(ds):
    """Helper function to check what variables are available in dataset"""
    print("Available variables in dataset:")
    for var in sorted(ds.variables.keys()):
        print(f"  {var}: {ds[var].shape if hasattr(ds[var], 'shape') else 'N/A'}")

    return {
        'all_vars': list(ds.variables.keys())
    }

def load_variable(dataset, variable, range = None, remove = 1):

    if range:
        return dataset.variables[variable][range, remove:-remove, remove:-remove]
    else:
        return dataset.variables[variable][remove:-remove, remove:-remove]


#%% SEDIMENT FUNCTIONS

def calculate_sediment_budget(dataset_trih, upstream_station, downstream_station, 
                            sed_var='SBTR'):
    """
    Calculate sediment budget between upstream and downstream stations.
    
    Parameters:
    -----------
    dataset_trih : netCDF4.Dataset
        History file dataset
    upstream_station : str
        Name of upstream station
    downstream_station : str
        Name of downstream station
    sed_var : str
        Sediment variable name (default: 'SBTR')
        
    Returns:
    --------
    dict: Dictionary with sediment budget information
    """
    # Get station names
    names_bytes = dataset_trih['NAMTRA'][:]
    all_station_names = [
        b''.join(names_bytes[i]).decode('utf-8').strip()
        for i in range(names_bytes.shape[0])
    ]
    
    # Extract sediment data
    results_sed = {}
    for name in [upstream_station, downstream_station]:
        if name not in all_station_names:
            raise ValueError(f"Station '{name}' not found. Available: {all_station_names}")
        
        idx = all_station_names.index(name)
        time = dataset_trih['time'][:]
        
        # Handle different dimensions of sediment data
        sed_data = dataset_trih[sed_var]
        if sed_data.ndim == 3:  # (time, sediment_fraction, station)
            bedload = sed_data[:, 0, idx]  # First sediment fraction
        else:  # (time, station)
            bedload = sed_data[:, idx]
            
        results_sed[name] = (time, bedload)
    
    # Calculate sediment budget
    time = results_sed[upstream_station][0]
    bedload_in = results_sed[upstream_station][1]
    bedload_out = results_sed[downstream_station][1]
    
    # Time interval (assumes regular time steps)
    dt = np.median(np.diff(time))
    
    # Total sediment
    total_in = np.sum(bedload_in * dt)
    total_out = np.sum(bedload_out * dt)
    net_balance = total_in - total_out
    
    return {
        'total_in': total_in,
        'total_out': total_out,
        'net_balance': net_balance,
        'time_interval': dt,
        'upstream_station': upstream_station,
        'downstream_station': downstream_station
    }

def get_last_sediment_transport(dataset_trih, upstream_station, downstream_station, index, reference_date, sed_var='SBTR'):
    """
    Get sediment transport at the last timestep for upstream and downstream stations.
    """
    # Get station names
    names_bytes = dataset_trih['NAMTRA'][:]
    all_station_names = [
        b''.join(names_bytes[i]).decode('utf-8').strip()
        for i in range(names_bytes.shape[0])
    ]
    
    results_sed = {}
    for name in [upstream_station, downstream_station]:
        if name not in all_station_names:
            raise ValueError(f"Station '{name}' not found. Available: {all_station_names}")
        
        idx = all_station_names.index(name)
        time = dataset_trih['time'][:]
        sed_data = dataset_trih[sed_var]
        if sed_data.ndim == 3:
            bedload = sed_data[:, 0, idx]
        else:
            bedload = sed_data[:, idx]
        results_sed[name] = (time, bedload)
    
    # Get the index value of choice
    last_time = results_sed[upstream_station][0][index]
    last_in = results_sed[upstream_station][1][index]
    last_out = results_sed[downstream_station][1][index]

    real_datetime = reference_date + pd.to_timedelta(last_time, unit='s')
    

    return {
        'real_date': real_datetime,
        'last_time': last_time,
        'upstream_station': upstream_station,
        'downstream_station': downstream_station,
        'last_in': last_in,
        'last_out': last_out,
        'difference': last_in - last_out
    }
# %%