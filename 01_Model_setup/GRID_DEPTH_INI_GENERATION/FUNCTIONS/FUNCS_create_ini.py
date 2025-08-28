import numpy as np

def water_level_along_estuary(x, desired_level, estuary_length=25001, estuary_down = 2, estuary_up = -2, x_estuary_begin = 20000): 
    return ((estuary_down - estuary_up) / estuary_length) * (x - x_estuary_begin) + desired_level

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

    # Mark all cells as dry
    ini_shape = np.full_like(dep, -999, dtype=float)

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