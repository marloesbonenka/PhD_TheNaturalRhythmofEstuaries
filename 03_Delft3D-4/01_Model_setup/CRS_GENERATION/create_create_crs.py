#%%
import os

def generate_north_south_cross_sections_variable_grid(
    nx, ny, x_upstream, y_center, river_length_km, 
    upstream_width, downstream_width, cell_size_x, cell_size_y, 
    buffer_cells, output_file
):
    """
    Generate cross-sections for variable grid, placing them every km along the river.
    River flows from EAST to WEST into a sea basin.
    
    Parameters:
    - nx, ny: Sea basin grid dimensions
    - x_upstream: Eastern (upstream) x-coordinate where river enters grid
    - y_center: Central y-coordinate of the river in grid (70)
    - river_length_km: Total physical river length (26 km)
    - upstream_width: Width at upstream end in meters (300)
    - downstream_width: Width at downstream end in meters (3000)
    - cell_size_x: Cell size in x-direction in meters (85)
    - cell_size_y: Cell size in y-direction in meters (50)
    - buffer_cells: Extra cells on each side for cross-sections
    - output_file: Output file path
    - river_end_x: X-coordinate where river ends (enters sea), if None auto-estimate
    """
    
    # Calculate cells per km based on river channel span in grid
    river_channel_cells = river_length_km * 1000  / cell_size_x
    cells_per_km_x = river_channel_cells / river_length_km
    
    cross_sections = []
    
    print(f"River channel spans from x={x_upstream} to x={int(x_upstream-river_channel_cells)} ({int(river_channel_cells)} cells)")
    print(f"Physical river length: {river_length_km} km")
    print(f"Cells per km: {cells_per_km_x:.3f}")
    print(f"Each grid cell represents ~{river_length_km * 1000 / river_channel_cells:.2f} m of river")
    print(f"Starting from x={x_upstream} (upstream/east) moving west")
    
    # Generate cross-sections every km
    for i in range(0, river_length_km + 1):  # Start from km 0, go to km 27
        if i == 0:
            x = x_upstream - 1
        else:
            # Calculate x-coordinate (moving from east to west, upstream to downstream)
            x = int(round(x_upstream - i * cells_per_km_x))
            
            # Ensure x is within bounds (don't go past the river end)
            x = max(int(x_upstream-river_channel_cells), min(x_upstream, x))
        
        # Calculate distance from upstream for width interpolation
        distance_from_upstream = i * 1000  # meters
        river_length_m = river_length_km * 1000
        
        # Interpolate width at this location
        width = upstream_width + (downstream_width - upstream_width) * (distance_from_upstream / river_length_m)
        
        # Convert width to cells and add buffer
        half_width_cells = int((width / 2) / cell_size_y) + buffer_cells
        
        # Calculate y-coordinates
        y1 = max(1, y_center - half_width_cells)
        y2 = min(ny, y_center + half_width_cells)
        
        cross_sections.append((
            f"river_km_{i}", x, y1, x, y2
        ))
        
        if i <= 5 or i % 5 == 0:  # Print first 5 and every 5th for brevity
            print(f"km {i}: x={x}, y1={y1}, y2={y2}, width={width:.0f}m")

    # Add boundary cross-sections
    cross_sections.append(("west_sea_bnd", 3, 2, 3, ny-2))
    cross_sections.append(("south_sea_bnd", nx-2, 2, 2, 2))
    cross_sections.append(("north_sea_bnd", 2, ny-2, nx-2, ny-2))

    # Add a cross section over the length of the estuary
    cross_sections.append(("river_parallel", x_upstream - 1, y_center, int(x_upstream-river_channel_cells), y_center))

    # Write to file
    with open(output_file, "w") as f:
        for name, x1, y1, x2, y2 in cross_sections:
            f.write(f"{name:<22}{x1:>6}{y1:>8}{x2:>8}{y2:>7}\n")

    print(f"\nCross section file generated: {output_file}")
    print(f"Generated {len(cross_sections)-3} river cross-sections")

    return cross_sections
#%%
if __name__ == "__main__":
    # Grid parameters
    # Sea basin
    nx = 101         # Grid cells in x-direction
    ny = 141        # Grid cells in y-direction
    
    # River parameters
    x_upstream = 394    # This should be the upstream (eastern) boundary
    y_center = 70       # Central y-coordinate
    river_length_km = 26
    upstream_width = 400    # meters
    downstream_width = 3000 # meters
    
    # Cell sizes (from your specification)
    cell_size_x = 85    # meters per cell in x-direction
    cell_size_y = 50    # meters per cell in y-direction
    
    buffer_cells = 2
    
    output_dir = r"u:\PhDNaturalRhythmEstuaries\Models\02_RiverDischargeVariability_domain45x15\D3D_base_input"
    output_file = "cross_sections_variable_grid.crs"
    
    print("=== Cross-section generation for river flowing into sea basin ===")
    print(f"Sea basin grid: {nx} × {ny} cells")
    print(f"Cell size: {cell_size_x}m × {cell_size_y}m")
    print(f"River length: {river_length_km} km (physical river length)")
    print(f"River enters at x={x_upstream} (east) and flows west into sea basin")
    
    # Estimate how many cells the river channel spans in the grid
    river_channel_cells = river_length_km * 1000  / cell_size_x
    print(f"Estimated river channel in grid: ~{river_channel_cells} cells")
    print(f"Physical distance per grid cell along river: ~{river_length_km * 1000 / river_channel_cells:.0f}m/cell")
    print("")
    
    cross_sections1 = generate_north_south_cross_sections_variable_grid(
        nx, ny, x_upstream, y_center, river_length_km,
        upstream_width, downstream_width, cell_size_x, cell_size_y,
        buffer_cells, os.path.join(output_dir, output_file)
    )
    
    print(f"\n" + "="*50)
    
#%%
