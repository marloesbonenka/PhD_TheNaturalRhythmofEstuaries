"""
Delft3D-FLOW River Boundary Condition Generation Script
Author: Marloes Bonenkamp
Date: March 25, 2025
Description: Generates river boundary conditions for an estuary domain, varying sinusoidally over four cells.

To integrate this with grid and dep file generation script:
- After generating grid, identify the river boundary cells and store their coordinates in a grid_info dictionary.
- Pass this grid_info to the generate_bct function along with other parameters.
- Adjust the num_cells parameter based on how many river boundaries you need.
- Call the generate_bct function after your grid and dep file generation.

USER GUIDELINES:

1. Ensure your grid generation script populates the 'grid_info' dictionary with:
   - 'nx': Number of cells in x-direction (m-direction)
   - 'ny': Number of cells in y-direction (n-direction)
   - 'river_cells': List of (x, y) coordinates for river boundary cells

2. Set the 'params' dictionary with:
   - 'total_discharge': Total river discharge in m³/s
   - 'duration_min': Total simulation duration in minutes
   - 'time_step': Time step for output in minutes

3. Adjust the 'output_dir' path to your desired location for the boundary files

4. Run this script after your grid and depth file generation

5. Verify the generated files in the output directory:
   - boundary.bnd
   - river_boundary.bct
   - tide.bch
   - transport.bcc

6. Always check the generated files for consistency with your model setup

Note: This script uses tabs for formatting as per Delft3D-FLOW requirements.
"""
#%%

import numpy as np
import os

def generate_boundary_files(grid_info, params, output_dir):
    """
    Generate all boundary condition files for Delft3D-FLOW.

    Args:
    grid_info (dict): Contains grid specifications (nx, ny, river_cells).
    params (dict): Simulation parameters (total_discharge, duration_min, time_step).
    output_dir (str): Directory to save the generated files.
    """
    def generate_bnd(grid_info):
        """
        Generates .bnd file content with strict formatting.

        Parameters:
        grid_info (dict): Dictionary containing grid size information.

        Returns:
        str: Formatted boundary condition content.
        """
        bnd_lines = []

        # neu_south and neu_north
        nx = grid_info['nx']
        ny = grid_info['ny']

        bnd_lines.append(f"{'neu_south':<21}Z H     2{1:>6}{nx:>6}{1:>6}  0.0000000e+000")
        bnd_lines.append(f"{'neu_north':<21}Z H     2{ny:>6}{nx:>6}{ny:>6}  0.0000000e+000")

        # River boundaries
        for i, (m, n) in enumerate(grid_info['river_cells']):
            bnd_lines.append(
                f"{f'river{i+1}':<21}T T   {m:>3}{n:>6}{m:>6}{n+1:>6}  0.0000000e+000 Uniform             "
            )

        return "\n".join(bnd_lines)
    
    def generate_bct(total_q, num_cells, duration_min, time_step):
        amplitude = total_q * 0.25
        omega = 2 * np.pi / (duration_min * 60)
        time_points = np.arange(0, duration_min + time_step, time_step)

        bct_content = []
        all_q_values = []  # Store q_values for all cells

        for cell_idx in range(num_cells):
            phase = cell_idx * np.pi / 2
            q_values = -(total_q / num_cells + amplitude * np.sin(omega * time_points * 60 + phase))
            all_q_values.append(q_values)

        # Adjust discharge values to ensure maximum difference of 10 m³/s between cells
        for i in range(len(time_points)):
            for cell_idx in range(1, num_cells):
                diff = all_q_values[cell_idx][i] - all_q_values[cell_idx - 1][i]
                if abs(diff) > 10:
                    adjustment = (abs(diff) - 10) * np.sign(diff)
                    all_q_values[cell_idx][i] -= adjustment / 2
                    all_q_values[cell_idx - 1][i] += adjustment / 2

        for cell_idx in range(num_cells):
            section_number = cell_idx + 2
            boundary_name = f'river{cell_idx + 1}'.ljust(20)

            header = [
                f"table-name           'Boundary Section : {section_number}'",
                f"contents             'Uniform             '",
                f"location             '{boundary_name}'",
                "time-function        'non-equidistant'",
                "reference-time       20150216",
                "time-unit            'minutes'",
                "interpolation        'linear'",
                "parameter            'time                '                     unit '[min]'",
                "parameter            'total discharge (t)  end A'               unit '[m3/s]'",
                "parameter            'total discharge (t)  end B'               unit '[m3/s]'",
                f"records-in-table     {len(time_points)}"
            ]

            bct_content.extend(header)

            for t, q in zip(time_points, all_q_values[cell_idx]):
                bct_content.append(f"{t:<12.0f}\t{q:7.2f}\t{q:7.2f}")

            bct_content.append('')

        return '\n'.join(bct_content)

    def generate_bch():
        """Generate .bch file content for tidal conditions."""
        bch_content = [
            "  0.0000000e+000\t3.0000000e+001",
            "",
            "  0.0000000e+000\t2.0000000e+000",
            "  0.0000000e+000\t2.0000000e+000",
            "  0.0000000e+000\t2.0000000e+000",
            "  0.0000000e+000\t2.0000000e+000",
            "",
            "                  3.0000000e+000",
            "                  0.0000000e+000",
            "                  3.0000000e+000",
            "                  0.0000000e+000"
        ]
        return '\n'.join(bch_content)

    def generate_bcc(num_cells, duration_min):
        """Generate .bcc file content for sediment transport conditions."""
        bcc_content = []
        for i in range(num_cells + 2):  # +2 for neu_south and neu_north
            location = f"neu_south" if i == 0 else f"neu_north" if i == 1 else f"river{i - 1}"
            bcc_content.extend([
                f"table-name           'Boundary Section : {i+1}'",
                f"contents             'Uniform   '",
                f"location             '{location.ljust(20)}'",
                "time-function        'non-equidistant'",
                "reference-time       20150216",
                "time-unit            'minutes'",
                "interpolation        'linear'",
                "parameter            'time                '  unit '[min]'",
                "parameter            'Sediment1            end A uniform'       unit '[kg/m3]'",
                "parameter            'Sediment1            end B uniform'       unit '[kg/m3]'",
                "records-in-table     2",
                " 0.0000000e+000\t0.0000000e+000\t0.0000000e+000",
                f" {duration_min:.7e}\t0.0000000e+000\t0.0000000e+000",
                ""
            ])
        return '\n'.join(bcc_content)

    # Generate and save .bnd file
    bnd_content = generate_bnd(grid_info)
    with open(os.path.join(output_dir, 'boundary.bnd'), 'w') as f:
        f.write(bnd_content)

    # Generate and save .bct file
    bct_content = generate_bct(params['total_discharge'], len(grid_info['river_cells']), params['duration_min'], params['time_step'])
    with open(os.path.join(output_dir, 'river_boundary.bct'), 'w') as f:
        f.write(bct_content)

    # Generate and save .bch file
    bch_content = generate_bch()
    with open(os.path.join(output_dir, 'tide.bch'), 'w') as f:
        f.write(bch_content)

    # Generate and save .bcc file
    bcc_content = generate_bcc(len(grid_info['river_cells']), params['duration_min'])
    with open(os.path.join(output_dir, 'transport.bcc'), 'w') as f:
        f.write(bcc_content)

    print("All boundary files generated successfully.")

# Example usage
output_dir = r"C:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Models\Estuary_funnelshape_testmodel\River_boundary_conditions"
os.makedirs(output_dir, exist_ok=True)

# IMPORTANT: Update these values based on your grid generation script
grid_info = {
    'nx': 100,  # Number of sea basin cells in x-direction (m-direction)
    'ny': 153,  # Number of sea basin cells in y-direction (n-direction)
    'river_cells': [
        (303, 78),
        (303, 77),
        (303, 76),
        (303, 75)
    ]
}

# IMPORTANT: Adjust these parameters as needed for your simulation
params = {
    'total_discharge': 80,  # Total river discharge in m³/s
    'duration_min': 2629440,  # Total simulation duration in minutes
    'time_step': 109500  # Time step for variations over river cells in minutes
}

# Generate all boundary files
generate_boundary_files(grid_info, params, output_dir)
# %%
