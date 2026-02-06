import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

#%% 

def write_xyz_file(filename, X_coords, Y_coords, depth_data, crs_epsg=3857):
    """
    Writes a point-based bathymetry file (.xyz) for use in Delft3D-FM.
    
    The file will contain a header defining the CRS, followed by X, Y, Z data.
    
    Parameters:
    - filename: The path to save the .xyz file.
    - X_coords: The 2D array of X-coordinates (e.g., data['X']).
    - Y_coords: The 2D array of Y-coordinates (e.g., data['Y']).
    - depth_data: The 2D array of depth values (e.g., data['dep_data']).
    - crs_epsg: The EPSG code to write in the header (defaulting to your bathymetry's CRS).
    """
    # 1. Flatten the arrays for column writing
    X_flat = X_coords.flatten()
    Y_flat = Y_coords.flatten()
    
    # 2. CONVERSION: Flip the sign.
    # Delft3D-4: Positive = Depth (down)
    # D-Flow FM: Positive = Elevation (up)
    # Multiplying by -1 turns a 15m depth into a -15m bed level.
    Z_flat = -1.0 * depth_data.flatten() 
    
    # 3. Combine into a single array (X, Y, Z)
    xyz_data = np.stack((X_flat, Y_flat, Z_flat), axis=1)

    # 4. Write to file with the CRS header
    with open(filename, 'w', encoding='ascii') as f:
        # # Write the header line specifying the CRS for the Delft3D-FM GUI
        # f.write(f'Generated bathymetry for Delft3D-FM (CRS: EPSG:{crs_epsg})\n')
        # f.write(f'EPSG:{crs_epsg}\n') # The line Delft3D-FM uses to read the CRS
        
        # Write the data columns: X Y Z (space separated)
        # Using numpy's savetxt is typically more efficient and handles formatting
        np.savetxt(f, xyz_data, fmt='%.8f', delimiter=' ')
        
    print(f"XYZ file saved: {filename}")

def write_dep_file(filename, depth_data):
    """
    Writes a .dep file in Delft3D format with correct dimensions, including an extra row and column.
    """
    ny, nx = depth_data.shape
    nodata_value = -999.0  # Delft3D default for missing data

    # Add an extra row and column to match Delft3D requirements
    dep_nx = nx + 1  # +1 in X direction
    dep_ny = ny + 1  # +1 in Y direction

    with open(filename, 'w', encoding='ascii') as f:
        # # Write the header
        # f.write("Generated bathymetry for Delft3D\n")
        # f.write(f"{dep_nx} {dep_ny}\n")

        # Write data values row by row
        for j in range(dep_ny):
            for i in range(dep_nx):
                # If within defined depth data indices
                if (i < nx) and (j < ny):
                    value = depth_data[j, i]
                else:
                    value = nodata_value  # Fill missing cells with nodata_value (-999)
                f.write(f"{value:.5f} ")  # Write value with 3 decimal places
            f.write("\n")  # Newline after each row

    print(f"Depth file saved: {filename}")

def generate_bathymetry(res, domain_extent, sea_extent, slope_extent, seabasin_depth, estuarywidth_upstream, estuarywidth_downstream):
    """
    Generates bathymetry grid and depth data for the specified resolution.
    """
    land_downstream = 2.1
    land_upstream = 4.1

    nx = int(domain_extent[0] / res) + 2   # +2 to include both ends starting at (1)
    ny = int(domain_extent[1] / res) + 2

    x = np.linspace(1, domain_extent[0] + res + 1, nx)   # Starts at x=1 (1-based indexing)
    y = np.linspace(1, domain_extent[1] + res + 1, ny)   # Starts at y=1 (1-based indexing)
    X, Y = np.meshgrid(x, y)

    # Initialize depth data array
    dep_data = np.zeros((ny, nx))

    if slope_extent <= 0:
        # If no slope is required, apply a uniform depth 
        sea_mask = (X <= sea_extent)
        dep_data[sea_mask] = seabasin_depth

    else:
        # Deep sea basin (constant depth)
        deep_sea_mask = (X <= (sea_extent - slope_extent))
        dep_data[deep_sea_mask] = seabasin_depth

        # Sloping sea basin (from beach level to deep sea level)
        slope_mask = ((X > sea_extent - slope_extent) & (X <= sea_extent))
        normalized_dist = (sea_extent - X[slope_mask]) / slope_extent
        dep_data[slope_mask] = 2 + normalized_dist * (seabasin_depth - 2)


    # Land (sloping from land_downstream m to land_upstream m, corresponding to -land_downstream to -land_upstream in dep_data)
    land_mask = (X > sea_extent)
    x_land = X[land_mask]
    dep_data[land_mask] = -land_downstream + (x_land - sea_extent) * ((-land_upstream + land_downstream) / (domain_extent[0] - sea_extent))
    
    # River / estuary
    river_length = domain_extent[0] - sea_extent  # Length of river

    for i in range(ny):
        for j in range(nx):
            if X[i, j] > sea_extent:
                # Calculate river width at current x-coordinate
                x_pos = X[i, j] - sea_extent
                river_width = estuarywidth_downstream * np.exp(np.log(estuarywidth_upstream / estuarywidth_downstream) * x_pos / river_length)
                    
                # Check if point is within the river
                if abs(Y[i, j] - 7500) < river_width/2:
                    # Calculate river bed level at current x-coordinate
                    river_bed = 2 - (x_pos / river_length) * 4  # Linear slope from -2 to 2
                    dep_data[i, j] = river_bed
    
    return {'x': x, 'y': y, 'dep_data': dep_data, 'X': X, 'Y': Y}


def generate_bathymetry_with_noise(res, domain_extent, sea_extent, slope_extent, seabasin_depth, 
                                   estuarywidth_upstream, estuarywidth_downstream, 
                                   noise_amplitude_cm=5.0, random_seed=None):
    """
    Generates bathymetry grid and depth data with noise added to the estuary bed.
    
    The sea basin and land areas remain unchanged. Only the estuary/river bed 
    receives pixel-level random noise.
    
    Parameters:
    - res: Grid resolution in meters
    - domain_extent: (x_length, y_length) tuple
    - sea_extent: Sea basin extent in x-direction
    - slope_extent: Sloping region extent in sea basin
    - seabasin_depth: Maximum depth of sea basin
    - estuarywidth_upstream, estuarywidth_downstream: Estuary widths
    - noise_amplitude_cm: Maximum noise amplitude in centimeters (default: 5 cm)
    - random_seed: Optional seed for reproducibility (default: None)
    
    Returns:
    - Dictionary with grid and bathymetry data including 'estuary_mask'
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    land_downstream = 2.1
    land_upstream = 4.1

    nx = int(domain_extent[0] / res) + 2   # +2 to include both ends starting at (1)
    ny = int(domain_extent[1] / res) + 2

    x = np.linspace(1, domain_extent[0] + res + 1, nx)   # Starts at x=1 (1-based indexing)
    y = np.linspace(1, domain_extent[1] + res + 1, ny)   # Starts at y=1 (1-based indexing)
    X, Y = np.meshgrid(x, y)

    # Initialize depth data array
    dep_data = np.zeros((ny, nx))
    
    # Initialize estuary mask to track which cells are in the estuary
    estuary_mask = np.zeros((ny, nx), dtype=bool)

    if slope_extent <= 0:
        # If no slope is required, apply a uniform depth 
        sea_mask = (X <= sea_extent)
        dep_data[sea_mask] = seabasin_depth
    else:
        # Deep sea basin (constant depth)
        deep_sea_mask = (X <= (sea_extent - slope_extent))
        dep_data[deep_sea_mask] = seabasin_depth

        # Sloping sea basin (from beach level to deep sea level)
        slope_mask = ((X > sea_extent - slope_extent) & (X <= sea_extent))
        normalized_dist = (sea_extent - X[slope_mask]) / slope_extent
        dep_data[slope_mask] = 2 + normalized_dist * (seabasin_depth - 2)

    # Land (sloping from land_downstream m to land_upstream m)
    land_mask = (X > sea_extent)
    x_land = X[land_mask]
    dep_data[land_mask] = -land_downstream + (x_land - sea_extent) * ((-land_upstream + land_downstream) / (domain_extent[0] - sea_extent))
    
    # River / estuary
    river_length = domain_extent[0] - sea_extent  # Length of river

    for i in range(ny):
        for j in range(nx):
            if X[i, j] > sea_extent:
                # Calculate river width at current x-coordinate
                x_pos = X[i, j] - sea_extent
                river_width = estuarywidth_downstream * np.exp(np.log(estuarywidth_upstream / estuarywidth_downstream) * x_pos / river_length)
                    
                # Check if point is within the river
                if abs(Y[i, j] - 7500) < river_width/2:
                    # Calculate river bed level at current x-coordinate
                    river_bed = 2 - (x_pos / river_length) * 4  # Linear slope from -2 to 2
                    dep_data[i, j] = river_bed
                    estuary_mask[i, j] = True
    
    # Add noise only to the estuary cells
    # Convert noise amplitude from cm to meters
    noise_amplitude_m = noise_amplitude_cm / 100.0
    
    # Generate random noise for the entire grid
    noise = np.random.uniform(-noise_amplitude_m, noise_amplitude_m, (ny, nx))
    
    # Apply noise only to estuary cells
    dep_data[estuary_mask] += noise[estuary_mask]
    
    return {'x': x, 'y': y, 'dep_data': dep_data, 'X': X, 'Y': Y, 'estuary_mask': estuary_mask}


def plot_bathymetry(X, Y, dep_data, res, output_dir, save_figures=False):
    """
    Creates a plot of the bathymetry data.
    """
    # Create custom colormap
    colors = ['#000080',   # Deep blue for sea depths (-15m)
            '#1E90FF',   # Light blue for shallow waters
            '#ADD8E6',   # Lighter blue for shallower waters
            '#F0F8FF',   # Very light blue (almost white)
            '#FFFFFF',   # White at -1m
            '#FFFFCC',   # Light yellow for 0m
            '#FFFF00',   # Yellow at 1m
            '#FFD700',   # Darker yellow
            '#8B4513',   # Brown transition
            '#3CB371', #MediumSeaGreen
            '#008000',   # Green for land.
            '#006400']   # Dark Green for land.

    # Define the color boundaries
    boundaries = [-15, -2, -1, -0.5, 0, 0.5, 1, 2, 6, 8, 10]

    # Normalize boundaries to range [0, 1] for colormap definition
    norm_positions = [(b - min(boundaries)) / (max(boundaries) - min(boundaries)) for b in boundaries]

    # Create a LinearSegmentedColormap with explicit color distribution
    custom_cmap = LinearSegmentedColormap.from_list('custom_bathymetry', list(zip(norm_positions, colors)))
    
    fig1, ax1 = plt.subplots(figsize=(15, 8))
    im = ax1.pcolormesh(X, Y, -dep_data, cmap=custom_cmap)

    ax1.set_aspect('equal')
    ax1.set_title(f'Bathymetry (Resolution: {res}m)')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Distance (m)')

    cbar = fig1.colorbar(im, ax=ax1, label='Elevation (m)', ticks=boundaries)

    if save_figures:
        plotname = f'bathymetry_{res}m.png'
        savedir = os.path.join(output_dir, 'plots')
        os.makedirs(savedir, exist_ok=True)
        
        plt.savefig(os.path.join(savedir, plotname), dpi=300, bbox_inches='tight')
        print(f"Bathymetry plot for {res}m resolution saved.")
    else:
        plt.show()

    plt.close(fig1)  # Close bathymetry figure


def generate_bathymetry_variable_grid(x, y, domain_extent, sea_extent, slope_extent, 
                                    seabasin_depth, estuarywidth_upstream, estuarywidth_downstream, land_up=10, land_down=8, estuary_up=2, estuary_down=2):
    """
    Generates bathymetry for variable grid spacing.
    
    Parameters:
    - x, y: Variable spacing coordinate arrays
    - domain_extent: (x_length, y_length) tuple
    - sea_extent: Sea basin extent in x-direction  
    - slope_extent: Sloping region extent
    - seabasin_depth: Maximum depth of sea basin
    - estuarywidth_upstream, estuarywidth_downstream: Estuary widths
    - land_up: Bed level for land upstream [specify the absolute value: it must be a positive value in real-life, corresponding to a negative value in the depth file]
    - land_down: Bed level for land downstream [specify the absolute value: it must be a positive value in real-life, corresponding to a negative value in the depth file]
    - estuary_down: Bed level where estuary flows into sea [specify the absolute value: it must be a negative value in real-life, corresponding to a positive value in the depth file]
    - estuary_up: Bed level estuary upstream [specify the absolute value: it must be a positive value in real-life, corresponding to a negative value in the depth file]
    
    Returns:
    - Dictionary with grid and bathymetry data
    """
    
    # Create meshgrid from variable spacing arrays
    X, Y = np.meshgrid(x, y)
    ny, nx = X.shape
    
    # Initialize depth data array
    dep_data = np.zeros((ny, nx))

    if slope_extent <= 0:
        # If no slope is required, apply a uniform depth 
        sea_mask = (X <= sea_extent)
        dep_data[sea_mask] = seabasin_depth
    else:
        # Deep sea basin (constant depth)
        deep_sea_mask = (X <= (sea_extent - slope_extent))
        dep_data[deep_sea_mask] = seabasin_depth

        # Sloping sea basin (from beach level to deep sea level)
        slope_mask = ((X > sea_extent - slope_extent) & (X <= sea_extent))
        normalized_dist = (sea_extent - X[slope_mask]) / slope_extent
        dep_data[slope_mask] = estuary_down + normalized_dist * (seabasin_depth - estuary_down)
        
    # Land (sloping from land_down m to land_up m, corresponding to -land_down m to -land_up in dep_data, 
    # since they are positive values in real life, but downwards is the positive direction in the depth file)

    land_mask = (X > sea_extent)
    x_land = X[land_mask]
    dep_data[land_mask] = -land_down + (x_land - sea_extent) * ((-land_up + land_down) / (domain_extent[0] - sea_extent))
    
    # River / estuary
    river_length = domain_extent[0] - sea_extent  # Length of river
    y_center_actual = domain_extent[1] / 2 + 1

    for i in range(ny):
        for j in range(nx):
            if X[i, j] > sea_extent:
                # Calculate river width at current x-coordinate
                x_pos = X[i, j] - sea_extent
                river_width = estuarywidth_downstream * np.exp(np.log(estuarywidth_upstream / estuarywidth_downstream) * x_pos / river_length)
                    
                # Check if point is within the river
                if abs(Y[i, j] - y_center_actual) < river_width / 2:
                    # Calculate river bed level at current x-coordinate
                    river_bed = estuary_down - (x_pos / river_length) * (estuary_up + estuary_down)  # Linear slope from -2 to 2
                    dep_data[i, j] = river_bed
    
    # A check to see how the bathymetry generation file sizes correspond:
    print(f"Grid dimensions: x={len(x)}, y={len(y)}")
    print(f"Meshgrid shape: X.shape={X.shape}, Y.shape={Y.shape}")
    print(f"Depth data shape: {dep_data.shape}")

    return {'x': x, 'y': y, 'dep_data': dep_data, 'X': X, 'Y': Y}