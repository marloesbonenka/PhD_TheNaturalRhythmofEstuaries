import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

#%% HELPER FUNCTIONS
# def check_available_variables(ds):
#     """Helper function to check what variables are available in dataset"""
#     print("Available variables in dataset:")
#     for var in sorted(ds.variables.keys()):
#         print(f"  {var}: {ds[var].shape if hasattr(ds[var], 'shape') else 'N/A'}")

#     return {
#         'all_vars': list(ds.variables.keys())
#     }

def get_runname(discharge):
    """Function to get runname based on discharge"""
    if discharge == 250:
        return f's1_{discharge}_Wup_300m'
    elif discharge == 500:
        return f's2_{discharge}_Wup_300m' 
    elif discharge == 1000:
        return f's3_{discharge}_Wup_300m' 
    elif discharge == 2000:
        return f's4_{discharge}_Wup_bnd_500m' 
    elif discharge == 4000:
        return f's5_{discharge}_Wup_bnd_500m'
    else:
        raise ValueError(f'discharge invalid: {discharge}')
    
def check_available_variables(ds):
    """Helper function to check variables and their metadata in a netCDF4 dataset"""
    print("Available variables in dataset:\n")
    for var_name in sorted(ds.variables.keys()):
        var = ds.variables[var_name]
        print(f"  {var_name}:")
        print(f"    shape        = {var.shape}")
        print(f"    dimensions   = {var.dimensions}")
        
        # List common attributes if available
        for attr in ['units', 'long_name', 'standard_name', 'description']:
            if attr in var.ncattrs():
                print(f"    {attr:13} = {getattr(var, attr)}")
        
        print("")  # Blank line for spacing

    return {
        'all_vars': list(ds.variables.keys())
    }

def load_variable(dataset, variable, range = None, remove = 1):
    """Load variable from NetCDF dataset with optional spatial and temporal slicing"""
    if range:
        return dataset.variables[variable][range, remove:-remove, remove:-remove]
    else:
        return dataset.variables[variable][remove:-remove, remove:-remove]

def load_single_timestep_variable(dataset, variable, timestep=0, remove=1, layer=0):
    """
    Load a single timestep of any variable and select a specific layer if present.
    Returns a 2D array.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        NetCDF dataset
    variable : str
        Variable name (e.g., "U1", "V1", "DPS", "S1")
    timestep : int
        Timestep to load (default: 0)
    remove : int
        Number of boundary cells to remove
    layer : int
        Layer index to select (default: 0, surface layer)

    Returns
    -------
    np.ndarray
        2D array of values for the selected layer/timestep
    """
    # Get the variable object to inspect dimensions
    var = dataset.variables[variable]

    # Load data for the specified timestep and remove boundaries
    # We need to handle the number of dimensions and the layer dimension
    if 'time' in var.dimensions:
        # Find the time dimension index
        time_idx = var.dimensions.index('time')
        # Construct the slice for time
        time_slice = slice(timestep, timestep+1)
        # Construct the slice for spatial dimensions (last two)
        spatial_slice = (slice(remove, -remove), slice(remove, -remove))
        
        # Check if there is a layer dimension (e.g., 'KMAXOUT_RESTR')
        if any(dim.startswith('KMAX') or dim.startswith('layers') or dim.startswith('k') for dim in var.dimensions):
            # There is a layer dimension
            layer_idx = [i for i, dim in enumerate(var.dimensions) if dim.startswith('KMAX') or dim.startswith('layers') or dim.startswith('k')][0]
            # Construct the slice for the layer
            layer_slice = layer
            # Build the slice for all dimensions
            slices = [slice(None)] * len(var.dimensions)
            slices[time_idx] = time_slice
            slices[layer_idx] = layer_slice
            slices[-2:] = spatial_slice
            # Load the data
            data = var[tuple(slices)]
       
        else:
            # No layer dimension; just time and spatial
            slices = [slice(None)] * len(var.dimensions)
            slices[time_idx] = time_slice
            slices[-2:] = spatial_slice
            data = var[tuple(slices)]
    
    else:
        # No time dimension; just spatial
        spatial_slice = (slice(remove, -remove), slice(remove, -remove))
        data = var[spatial_slice]
    
    # Squeeze out singleton dimensions (like time or layer if present)
    return np.squeeze(data)

def load_velocity(dataset, variable, timestep=0, remove=1, layer=0):
    """
    Load velocity variable and select a specific layer.
    Returns a 2D array for the selected layer.

    Parameters
    ----------
    dataset : netCDF4.Dataset
        NetCDF dataset
    variable : str
        Variable name (e.g., "U1" or "V1")
    timestep : int
        Timestep to load (default: 0)
    remove : int
        Number of boundary cells to remove
    layer : int
        Layer index to select (default: 0, surface layer)

    Returns
    -------
    np.ndarray
        2D array of velocity values for the selected layer
    """
    # Load the data for the specified timestep and remove boundaries
    data = load_variable(dataset, variable, range=slice(timestep, timestep+1), remove=remove)
    
    # Check if there is data in the layer dimension
    if data.shape[1] == 0:
        # No data available for any layer; return NaN-filled array of the correct spatial shape
        return np.full(data.shape[2:], np.nan)
    else:
        # Select the specified layer
        return data[0, layer]
    
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