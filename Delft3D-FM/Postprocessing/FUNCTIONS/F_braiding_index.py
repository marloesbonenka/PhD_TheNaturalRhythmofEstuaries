import numpy as np
from scipy.spatial import cKDTree

import numpy as np
from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial import cKDTree

import numpy as np
from scipy.spatial import cKDTree

def compute_BI_FM(ds, var_name, x_targets, y_range, threshold=0.5, time_idx=None, method='fixed'):
    """
    Computes Braiding Index for Delft3D-FM with a 0-crossing safety buffer.
    """
    y_coords = np.linspace(y_range[0], y_range[1], 200) 
    face_x, face_y = ds['mesh2d_face_x'].values, ds['mesh2d_face_y'].values
    tree = cKDTree(np.vstack([face_x, face_y]).T)
    
    indices = [time_idx] if time_idx is not None else range(len(ds.time))
    BI_results = []

    for t in indices:
        BI_xs = []
        data_t = ds[var_name].isel(time=t).values
        
        for x_target in x_targets:
            query_points = np.vstack([np.full_like(y_coords, x_target), y_coords]).T
            _, nearest_indices = tree.query(query_points)
            
            # Use absolute values to prevent sign-flip issues at the datum
            z_cross = np.abs(data_t[nearest_indices])
            z_cross = np.nan_to_num(z_cross, nan=0.0)
            
            # --- UPDATED CHANNEL DETECTION ---
            if method == 'relative':
                # mean * (1 + 20%) + 10cm safety buffer
                # The 0.1 buffer ensures we don't get '0' threshold at the 0-datum
                current_threshold = np.nanmean(z_cross) * (1 + threshold) + 0.1
            else:
                current_threshold = threshold
            
            crossings = 0
            for i in range(1, len(z_cross)):
                if (z_cross[i-1] <= current_threshold and z_cross[i] > current_threshold) or \
                   (z_cross[i-1] > current_threshold and z_cross[i] <= current_threshold):
                    crossings += 1
            
            BI_xs.append(crossings / 2)
        BI_results.append(BI_xs)
        
    return np.array(BI_results), None

def compute_BI_FM1(ds, var_name, x_targets, y_range, threshold=0.5, time_idx=None, method='fixed'):
    """
    Computes Braiding Index for Delft3D-FM.
    'fixed': threshold is an absolute value.
    'relative': threshold is (mean_depth * factor) + buffer to handle 0-crossings.
    """
    y_start, y_end = y_range
    y_coords = np.linspace(y_start, y_end, 200) 
    
    face_x = ds['mesh2d_face_x'].values
    face_y = ds['mesh2d_face_y'].values
    tree = cKDTree(np.vstack([face_x, face_y]).T)
    
    indices = [time_idx] if time_idx is not None else range(len(ds.time))
    BI_results = []
    times = ds.time.values

    for t in indices:
        BI_xs = []
        # Use .values to avoid xarray overhead in the loop
        data_t = ds[var_name].isel(time=t).values
        
        for x_target in x_targets:
            query_points = np.vstack([np.full_like(y_coords, x_target), y_coords]).T
            _, nearest_indices = tree.query(query_points)
            
            # We use absolute depth to ensure we don't have sign issues
            z_cross = np.abs(data_t[nearest_indices])
            z_cross = np.nan_to_num(z_cross, nan=0.0)
            
            # --- IMPROVED CHANNEL DETECTION ---
            if method == 'relative':
                local_mean = np.nanmean(z_cross)
                # We add a buffer (e.g., 0.1m) to prevent the threshold from being 0
                # 'threshold' here acts as the factor (e.g., 0.2)
                current_threshold = local_mean * (1 + threshold) + 0.1
            else:
                current_threshold = threshold
            
            crossings = 0
            for i in range(1, len(z_cross)):
                # Logic: Is the water deeper than our (mean + 20% + 10cm) limit?
                if (z_cross[i-1] <= current_threshold and z_cross[i] > current_threshold) or \
                   (z_cross[i-1] > current_threshold and z_cross[i] <= current_threshold):
                    crossings += 1
            
            BI_xs.append(crossings / 2)
        BI_results.append(BI_xs)
        
    return np.array(BI_results), times

def compute_BI_FM_old(ds, var_name, x_targets, y_range, threshold=0.5, time_idx=None):
    """
    Computes Braiding Index for Delft3D-FM using a fast KDTree 
    to find the nearest cell centers for points along a transect.
    """
    y_start, y_end = y_range
    y_coords = np.linspace(y_start, y_end, 200) 
    
    # Extract FM mesh coordinates
    face_x = ds['mesh2d_face_x'].values
    face_y = ds['mesh2d_face_y'].values
    
    # Build the spatial tree once per model run
    tree = cKDTree(np.vstack([face_x, face_y]).T)
    
    # Determine which indices to process
    if time_idx is not None:
        indices = [time_idx]
    else:
        indices = range(len(ds.time))

    BI_results = []
    times = ds.time.values

    for t in indices:
        BI_xs = []
        data_t = ds[var_name].isel(time=t).values
        
        for x_target in x_targets:
            # Create the transect points
            query_points = np.vstack([np.full_like(y_coords, x_target), y_coords]).T
            
            # Find nearest neighbor indices for all 200 points at once
            _, nearest_indices = tree.query(query_points)
            z_cross = np.abs(data_t[nearest_indices])
            z_cross = np.nan_to_num(z_cross, nan=0.0)
            
            # Crossing logic
            crossings = 0
            for i in range(1, len(z_cross)):
                if (z_cross[i-1] <= threshold and z_cross[i] > threshold) or \
                   (z_cross[i-1] > threshold and z_cross[i] <= threshold):
                    crossings += 1
            
            BI_xs.append(crossings / 2)
            
        BI_results.append(BI_xs)
        # Note: 't' is index, but the time is checked in the main script
        
    return np.array(BI_results), times