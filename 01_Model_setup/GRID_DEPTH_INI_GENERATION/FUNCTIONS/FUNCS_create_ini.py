import numpy as np
#%%
def water_level_along_estuary(x, desired_level, estuary_length=25001, estuary_down = 2, estuary_up = -2, x_estuary_begin = 20000): 
    return ((estuary_down - estuary_up) / estuary_length) * (x - x_estuary_begin) + desired_level

def generate_ini_depth(grid_data, sea_extent, desired_river_depth=2.0):
    """
    Creates water depth for D-Flow FM.
    - Sea: Depth = dep (results in Level 0.0)
    - River: Depth = constant (results in sloping surface)
    - Land: Depth = 0.0
    """
    X = grid_data['X']
    Y = grid_data['Y']
    dep = grid_data['dep_data'] # Positive is deep

    # Initialize everything as dry (0.0)
    ini_depth = np.zeros_like(dep, dtype=float)

    # 1. Sea Basin Logic (Horizontal level at 0.0)
    # This works for both your deep basin and your slope mask
    sea_mask = (X <= sea_extent) & (dep > 0)
    ini_depth[sea_mask] = dep[sea_mask]

    # 2. Estuary/River Logic
    # We identify the river based on your bathymetry parameters
    # Any cell with X > sea_extent AND dep > land levels is part of the river
    # Or simply use the same width/bed logic to identify the channel
    river_mask = (X > sea_extent) & (dep > -2.1) # Anything deeper than land_downstream
    ini_depth[river_mask] = desired_river_depth

    return {'X': X, 'Y': Y, 'ini_data': ini_depth}

#%%
def generate_ini(grid_data, desired_level, estuary_depth = 2):
    # Bed elevation in m, negative below datum.
    x = grid_data['x']
    y = grid_data['y']

    X = grid_data['X']
    Y = grid_data['Y']

    dep = grid_data['dep_data']

    buffer_include_estuary_only = (estuary_depth + 0.2)

    # Mask wet cells, i.e. the places that are not land and thus are deeper than - land_depth
    wet_mask = dep >= -buffer_include_estuary_only

    ## Mark all cells as dry (Delft3D-4, use -999)
    #ini_shape = np.full_like(dep, -999, dtype=float)
    
    # Mark all cells as dry (Delft3D-FM, use np.nan and filter before writing)
    ini_shape = np.full_like(dep, np.nan, dtype=float)
    
    # For the ini values, you need to stack more than just the water level, 
    # so for this, I built in the tic for desired_level = 0, 
    # to copy a lot of zeros below the water level ini values.
    
    if desired_level != 0:
        # For the sea basin: initial water level = desired depth
        sea_basin_mask = (dep > estuary_depth) & wet_mask 
        ini_shape[sea_basin_mask] = desired_level

        # For the estuary
        estuary_mask = (dep < estuary_depth) * (dep >= -buffer_include_estuary_only) & wet_mask
        
        estuary_water_level = water_level_along_estuary(X, desired_level)
        ini_shape[estuary_mask] = estuary_water_level[estuary_mask]
    else:
        ini_shape = np.full_like(dep, 0, dtype=float)

    return {'x': x, 'y': y, 'ini_data': ini_shape, 'X': X, 'Y': Y}

def write_ini_xyz_file(filename, X_coords, Y_coords, ini_data, crs_epsg=3857):
    """
    Writes a point-based initial water level file (.xyz) for use in Delft3D-FM.
    """
    
    # 1. Flatten the arrays: X, Y, and the initial water level (Z/zeta)
    X_flat = X_coords.flatten()
    Y_flat = Y_coords.flatten()
    Z_flat = ini_data.flatten() # ini_data is the initial water level (zeta)
    
    # # 2. Combine into a single array (X, Y, Z/zeta)
    # xyz_data = np.stack((X_flat, Y_flat, Z_flat), axis=1)
    
    # *** CORRECTION: Filter out NaN values (the 'dry' cells) ***
    # Create a mask for all non-NaN values in the initial water level data
    valid_mask = ~np.isnan(Z_flat)
    
    # Apply the mask to all flattened arrays
    X_filtered = X_flat[valid_mask]
    Y_filtered = Y_flat[valid_mask]
    Z_filtered = Z_flat[valid_mask]
    
    # 2. Combine into a single array (X, Y, Z/zeta)
    xyz_data = np.stack((X_filtered, Y_filtered, Z_filtered), axis=1)

    # 3. Write to file with the CRS header
    with open(filename, 'w', encoding='ascii') as f:
        # # Header (Optional, but good for tracking)
        # f.write(f'Initial Water Level (CRS: EPSG:{crs_epsg})\n')
        # f.write(f'EPSG:{crs_epsg}\n') 
        
        # Write the data columns: X Y Zeta (space separated)
        np.savetxt(f, xyz_data, fmt='%.8f', delimiter=' ')
        
    print(f"INI (.xyz) file saved: {filename}")
    
def write_ini_file(filename, ini_data):
    """
    Writes an .ini file (or any grid data file) in Delft3D format with correct dimensions,
    including an extra row and column.
    """
    ny, nx = ini_data.shape
    nodata_value = -999.0  # Default for missing data

    # Add an extra row and column to match Delft3D requirements
    ini_nx = nx + 1
    ini_ny = ny + 1

    with open(filename, 'w') as f:
        for j in range(ini_ny):
            for i in range(ini_nx):
                if (i < nx) and (j < ny):
                    value = ini_data[j, i]
                else:
                    value = nodata_value
                f.write(f"{value:.8f} ")
            f.write("\n")
    print(f"INI file saved: {filename}")