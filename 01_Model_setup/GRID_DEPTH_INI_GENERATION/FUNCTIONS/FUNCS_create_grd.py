import numpy as np
import matplotlib.pyplot as plt
import os 

#%% Uniform grid generation and plotting

def write_grd_file(filename, x, y):
    """
    Writes a Delft3D-FLOW grid file (.grd) in the correct format with 1-based indexing.
    """
    nx, ny = len(x), len(y)

    with open(filename, 'w') as f:
        # Write header
        f.write("Coordinate System = Cartesian\n")
        f.write(f"{nx} {ny}\n")
        f.write("0 0 0\n")

        # Write X coordinates (ETA lines)
        for j in range(ny):
            f.write(f"ETA= {j+1}\t")  # Start ETA line with tab
            for i, val in enumerate(x):
                if i > 0 and i % 5 == 0:
                    f.write("\n\t")  # Add tab for continuation lines
                f.write(f"{val:.6e}\t")  # Use scientific notation with tab separation
            f.write("\n")  # End of this ETA line

        # Write Y coordinates (ETA lines)
        for j in range(ny):
            f.write(f"ETA= {j+1}\t")  # Start ETA line with tab
            for i in range(nx):
                if i > 0 and i % 5 == 0:
                    f.write("\n\t")  # Add tab for continuation lines
                f.write(f"{y[j]:.6e}\t")  # Use scientific notation with tab separation
            f.write("\n")  # End of this ETA line

    print(f"Grid file saved: {filename}")

def plot_grid(x, y, nx, ny, res, output_dir, save_figures=False):
    """
    Creates a plot of the grid.
    """
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    ax2.set_aspect('equal')
    ax2.set_title(f'Grid (Resolution: {res}m)')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Distance (m)')

    # Plot grid lines
    for i in range(nx):
        ax2.plot(x[i] * np.ones(ny), y, 'k-', linewidth=0.5)
    for j in range(ny):
        ax2.plot(x, y[j] * np.ones(nx), 'k-', linewidth=0.5)

    if save_figures:
        plotname = f'grid_{res}m.png'
        savedir = os.path.join(output_dir, 'plots')
        os.makedirs(savedir, exist_ok=True)
        
        plt.savefig(os.path.join(savedir, plotname), dpi=300, bbox_inches='tight')
        print(f"Grid plot for {res}m resolution saved.")
    else:
        plt.show()

    plt.close(fig2)  # Close grid figure

#%% Variable grid sizes generation and plotting
def generate_variable_grid_spatial(domain_extent, 
                                 fine_res_x, medium_res_x, coarse_res_x,
                                 fine_res_y, medium_res_y, coarse_res_y,
                                 fine_bounds, buffer_width=1000, adjust_spacing=True):
    """
    Generates variable grid spacing with spatial control over fine/medium/coarse regions.
    
    Parameters:
    - domain_extent: (x_length, y_length) tuple
    - fine_res_x, medium_res_x, coarse_res_x: Grid resolutions for x-direction zones
    - fine_res_y, medium_res_y, coarse_res_y: Grid resolutions for y-direction zones
    - fine_bounds: (x_min, x_max, y_min, y_max) tuple defining fine grid region
    - buffer_width: Width of medium resolution buffer around fine region
    - adjust_spacing: Whether to adjust spacing to avoid squeezed cells
    - sea_boundary_x: X-coordinate boundary between sea (coarse) and estuary regions
    
    Returns:
    - Dictionary with 'x' and 'y' coordinate arrays
    """
    
    # Unpack fine region bounds
    fine_x_min, fine_x_max, fine_y_min, fine_y_max = fine_bounds
    
    # Define buffer region bounds (U-shaped around fine region)
    buffer_x_min = fine_x_min - buffer_width
    buffer_x_max = fine_x_max  # No buffer on right side since fine goes to domain edge
    buffer_y_min = fine_y_min - buffer_width
    buffer_y_max = fine_y_max + buffer_width
    
    # Generate x-coordinates with variable spacing - work backwards from domain end
    x_coords = []
    
    is_uniform_x = (fine_res_x == medium_res_x == coarse_res_x)
    is_uniform_y = (fine_res_y == medium_res_y == coarse_res_y)
    
    # Trigger the simple grid generation if resolution is constant in both directions
    if is_uniform_x and is_uniform_y:
        print(f"Resolutions are constant across the domain (X={coarse_res_x}m, Y={coarse_res_y}m). \n Generating a uniform rectilinear grid.")
        
        uniform_res_x = coarse_res_x
        uniform_res_y = coarse_res_y

        # --- Uniform X-Coordinates ---
        x_coords = np.arange(1, domain_extent[0] + 1 + uniform_res_x, uniform_res_x)
        if x_coords[-1] < domain_extent[0] + 1:
            x_coords = np.append(x_coords, domain_extent[0] + 1)
        if x_coords[0] > 1:
             x_coords = np.insert(x_coords, 0, 1)

        # --- Uniform Y-Coordinates ---
        y_coords = np.arange(1, domain_extent[1] + 1 + uniform_res_y, uniform_res_y)
        if y_coords[-1] < domain_extent[1] + 1:
            y_coords = np.append(y_coords, domain_extent[1] + 1)
        if y_coords[0] > 1:
             y_coords = np.insert(y_coords, 0, 1)

        # Final cleanup and return
        x_coords = np.unique(x_coords)
        y_coords = np.unique(y_coords)
        
        return {'x': x_coords, 'y': y_coords}

    if adjust_spacing:
        # Work backwards from the domain end to ensure fine cells are exactly the right size
        temp_coords = []
        current_x = domain_extent[0] + 1  # Start from domain end
        
        # Work backwards through fine region
        while current_x > fine_x_min:
            temp_coords.append(current_x)
            current_x -= fine_res_x
            if current_x <= fine_x_min:
                temp_coords.append(fine_x_min)
                break
        
        # Work backwards through buffer region
        current_x = fine_x_min
        while current_x > buffer_x_min:
            temp_coords.append(current_x)
            current_x -= medium_res_x
            if current_x <= buffer_x_min:
                temp_coords.append(buffer_x_min)
                break
        
        # Work backwards through coarse region, but adjust the first cells to fit
        current_x = buffer_x_min
        coarse_coords = []
        while current_x > 1:
            coarse_coords.append(current_x)
            current_x -= coarse_res_x
        
        # Add the starting point
        coarse_coords.append(1)
        
        # If there's a mismatch, distribute it among the first few coarse cells
        if len(coarse_coords) > 1:
            actual_distance = coarse_coords[0] - coarse_coords[-1]  # Distance covered by coarse cells
            target_distance = buffer_x_min - 1  # Distance that should be covered
            adjustment = target_distance - actual_distance
            
            if abs(adjustment) > 0.1:  # Only adjust if significant difference
                # Distribute the adjustment among the first few coarse cells
                n_cells_to_adjust = min(3, len(coarse_coords) - 1)  # Adjust up to 3 cells
                adjustment_per_cell = adjustment / n_cells_to_adjust
                
                for i in range(1, n_cells_to_adjust + 1):
                    coarse_coords[i] += adjustment_per_cell * i
        
        # Combine all coordinates and reverse (since we worked backwards)
        all_coords = coarse_coords[::-1] + temp_coords[::-1]
        x_coords = sorted(list(set(all_coords)))  # Remove duplicates and sort
        
    else:
        # Original forward method
        current_x = 1
        while current_x <= domain_extent[0] + 1:
            x_coords.append(current_x)
            
            # Determine resolution based on x-position
            if buffer_x_min <= current_x <= fine_x_max:
                if fine_x_min <= current_x <= fine_x_max:
                    step = fine_res_x
                else:
                    step = medium_res_x
            else:
                step = coarse_res_x
                
            current_x += step
        
        if x_coords[-1] < domain_extent[0] + 1:
            x_coords.append(domain_extent[0] + 1)
    
    # Generate y-coordinates with similar approach
    y_coords = []

    if adjust_spacing:
        # Work from center outward to ensure fine cells are correct
        y_center = (fine_y_min + fine_y_max) / 2
        
        # Generate coordinates for upper half
        temp_coords_upper = []
        current_y = y_center
        
        # Fine region (upper half)
        while current_y < fine_y_max:
            current_y += fine_res_y 
            if current_y >= fine_y_max:
                temp_coords_upper.append(fine_y_max)
                break
            temp_coords_upper.append(current_y)
        
        # Buffer region (upper)
        current_y = fine_y_max
        while current_y < buffer_y_max:
            current_y += medium_res_y
            if current_y >= buffer_y_max:
                temp_coords_upper.append(buffer_y_max)
                break
            temp_coords_upper.append(current_y)
        
        # Coarse region (upper) - adjust to fit
        current_y = buffer_y_max
        coarse_coords_upper = []
        while current_y < domain_extent[1] + 1:
            current_y += coarse_res_y
            coarse_coords_upper.append(min(current_y, domain_extent[1] + 1))
            if current_y >= domain_extent[1] + 1:
                break
        
        # Mirror for lower half
        temp_coords_lower = []
        for coord in temp_coords_upper:
            mirror_coord = 2 * y_center - coord
            if mirror_coord >= 1:
                temp_coords_lower.append(mirror_coord)
        
        coarse_coords_lower = []
        for coord in coarse_coords_upper:
            mirror_coord = 2 * y_center - coord
            if mirror_coord >= 1:
                coarse_coords_lower.append(mirror_coord)
        
        # Combine all y-coordinates
        all_y_coords = sorted(coarse_coords_lower + temp_coords_lower + [y_center] + temp_coords_upper + coarse_coords_upper)
        y_coords = [coord for coord in all_y_coords if 1 <= coord <= domain_extent[1] + 1]
        
    else:
        # Original method for y-coordinates
        current_y = 1
        while current_y <= domain_extent[1] + 1:
            y_coords.append(current_y)
            
            if buffer_y_min <= current_y <= buffer_y_max:
                if fine_y_min <= current_y <= fine_y_max:
                    step = fine_res_y
                else:
                    step = medium_res_y
            else:
                step = coarse_res_y
                
            current_y += step
        
        if y_coords[-1] < domain_extent[1] + 1:
            y_coords.append(domain_extent[1] + 1)
    
    return {'x': np.array(x_coords), 'y': np.array(y_coords)}


def plot_variable_grid(x, y, grid_name, output_dir, save_figures=False):
    """
    Creates a plot of the variable grid.
    """
    fig2, ax2 = plt.subplots(figsize=(15, 8))
    ax2.set_aspect('equal')
    ax2.set_title(f'Variable Grid ({grid_name})')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Distance (m)')

    nx, ny = len(x), len(y)
    
    # Plot grid lines - subsample for visualization if grid is very dense
    step_x = max(1, nx // 50)  # Show max 50 lines in x
    step_y = max(1, ny // 50)  # Show max 50 lines in y
    
    for i in range(0, nx, step_x):
        ax2.plot(x[i] * np.ones(ny), y, 'k-', linewidth=0.5)
    for j in range(0, ny, step_y):
        ax2.plot(x, y[j] * np.ones(nx), 'k-', linewidth=0.5)

    if save_figures:
        plotname = f'variable_grid_{grid_name}.png'
        savedir = os.path.join(output_dir, 'plots')
        os.makedirs(savedir, exist_ok=True)
        
        plt.savefig(os.path.join(savedir, plotname), dpi=300, bbox_inches='tight')
        print(f"Variable grid plot saved.")
    else:
        plt.show()

    plt.close(fig2)  # Close grid figure