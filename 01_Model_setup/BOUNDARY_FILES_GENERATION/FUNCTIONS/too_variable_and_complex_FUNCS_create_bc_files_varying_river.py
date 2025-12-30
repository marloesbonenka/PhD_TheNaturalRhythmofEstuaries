import numpy as np
import os
from scipy.optimize import curve_fit

#%% All functions to generate D3D-4 boundary files for an estuary model

def generate_boundary_files(grid_info, params, boundary_dir, tide_west=True, tide_north_south=False):

    def generate_bnd(grid_info):
        bnd_lines = []
        nx = grid_info['nx']
        ny = grid_info['ny']

        #Tide boundaries
        if tide_north_south:
            # South boundary: from (i=2, j=1) to (i=nx, j=1)
            bnd_lines.append(f"{'neu_south':<21}Z H     2{1:>6}{nx:>6}{1:>6}  0.0000000e+000")

            # North boundary: from (i=2, j=ny) to (i=nx, j=ny)
            bnd_lines.append(f"{'neu_north':<21}Z H     2{ny:>6}{nx:>6}{ny:>6}  0.0000000e+000")
        
        if tide_west:
            # West boundary along i=1, from j=2 to j=(ny-1)
            bnd_lines.append(f"{'neu_west':<21}Z H     1{2:>6}{1:>6}{ny-1:>6}  0.0000000e+000")

        #River boundaries
        for i, (m, n) in enumerate(grid_info['river_cells']):
            bnd_lines.append(
                f"{f'river{i+1}':<21}T T   {m:>3}{n:>6}{m:>6}{n+1:>6}  0.0000000e+000 Uniform             "
            )
        return "\n".join(bnd_lines)
    
    def generate_bct(total_q, num_cells, duration_min, time_step, cv, flashiness, pattern_type, climate_scenario):
        """
        Generate boundary condition time series with improved handling of flashiness and constraints.
        Parameters:
        - total_q: Total discharge (m³/s)
        - num_cells: Number of river cells
        - duration_min: Simulation duration in minutes
        - time_step: Time step of boundary condition variation in minutes
        - cv: Coefficient of variation for initial noise
        - flashiness: Target flashiness (P90/P10 ratio)
        - pattern_type: Flow pattern - "seasonal", "flashy", "intermittent", or "constant"
        - climate_scenario: Climate scenario - "EE", "ED", or "PI-Med"
        """
        
        # 1. Setup time steps and base cell distribution
        time_points = np.arange(0, duration_min, time_step)
        num_steps = len(time_points)
        cell_base_distribution = np.linspace(0.8, 1.2, num_cells)
        cell_base_distribution = cell_base_distribution / np.sum(cell_base_distribution) * num_cells

        # 2. Generate total discharge timeseries
        if pattern_type == "constant":
            # Constant discharge, no variation over time, except base distribution over cells 
            discharge_total = np.ones(num_steps) * total_q
            
        else:
            # Base discharge time series (start with Gaussian noise)
            np.random.seed(0)  # For reproducibility

            discharge_total = np.random.normal(total_q, cv * total_q, num_steps)
            
            # Ensure positivity
            discharge_total[discharge_total < 0] = 0
            
            # Apply pattern and climate scenario
            discharge_total = __apply_pattern_and_climate(discharge_total, total_q, time_points, pattern_type, climate_scenario)
            
            # Ensure positivity
            discharge_total[discharge_total < 0] = 0
            
            # Achieve target flashiness
            discharge_total = __achieve_flashiness(discharge_total, total_q, flashiness)

            # Test actual flashiness (P90/P10 ratio)
            final_p90 = np.percentile(discharge_total, 90)
            final_p10 = np.percentile(discharge_total, 10)
            actual_flashiness = final_p90 / final_p10
            
            print(f"Target flashiness: {flashiness}")
            print(f"Actual flashiness: {actual_flashiness:.2f}")
            print(f"P90 value: {final_p90:.2f}, P10 value: {final_p10:.2f}")
            
            flashiness_error = abs(actual_flashiness - flashiness) / flashiness * 100
            print(f"Flashiness error: {flashiness_error:.1f}%")
            
            if flashiness_error > 10:
                print(f"WARNING: Actual flashiness deviates significantly from target")

        # 3. Allocate discharge to adjacent river boundary cells
        all_q_values = np.zeros((num_cells, num_steps))

        amplitude_perturb = 0.05
        cycles_per_set = 12
        freq = 2 * np.pi / (duration_min / cycles_per_set)

        if pattern_type != 'constant':
            # Parameters for sinusoidal perturbation on distribution weights

            for t_idx, t in enumerate (time_points):
                phase = freq * t
                perturb = amplitude_perturb * np.array([np.sin(phase + i * np.pi / (num_cells + 1)) for i in range(num_cells)])
                
                weights = cell_base_distribution + perturb

                weights = np.clip(weights, 0.8, 1.2)
                weights = weights / np.sum(weights) * num_cells

                # Allocate discharge for this time step
                for cell_idx in range(num_cells):
                    base_fraction = weights[cell_idx] / num_cells
                    all_q_values[cell_idx, t_idx] = -discharge_total[t_idx] * base_fraction

            # 4. First pass of cell-to-cell difference constraint on base flow
            max_diff_limit = 5.0  # Slightly lower to leave room for sinusoidal additions
            
            for i in range(num_steps):
                for cell_idx in range(1, num_cells):
                    diff = all_q_values[cell_idx, i] - all_q_values[cell_idx - 1, i]
                    if abs(diff) > max_diff_limit:
                        adjustment = diff - np.sign(diff) * max_diff_limit
                        all_q_values[cell_idx, i] -= adjustment * 0.5
                        all_q_values[cell_idx - 1, i] += adjustment * 0.5
                        
            # 5. Now add smaller sinusoidal components that won't break constraints
            # Using smaller amplitude to avoid breaking constraints
            amplitude = 0.1 * total_q  # Reduced amplitude
            
            for cell_idx in range(num_cells):
                phase = cell_idx * np.pi / (num_cells + 1)  # Better phase spacing
                sinusoidal_component = amplitude * np.sin(2 * np.pi * time_points / duration_min + phase)
                all_q_values[cell_idx, :] += sinusoidal_component
            
            # 5. Second pass of cell-to-cell difference constraint after adding sinusoidal components
            for i in range(num_steps):
                # Forward pass
                for cell_idx in range(1, num_cells):
                    diff = all_q_values[cell_idx, i] - all_q_values[cell_idx - 1, i]
                    if abs(diff) > 10.0:  # Strict 10.0 limit
                        adjustment = diff - np.sign(diff) * 10.0
                        all_q_values[cell_idx, i] -= adjustment * 0.6  # Bias adjustment towards the current cell
                        all_q_values[cell_idx - 1, i] += adjustment * 0.4  # Less adjustment to previous cell
                
                # Backward pass for better smoothing
                for cell_idx in range(num_cells - 2, -1, -1):
                    diff = all_q_values[cell_idx, i] - all_q_values[cell_idx + 1, i]
                    if abs(diff) > 10.0:
                        adjustment = diff - np.sign(diff) * 10.0
                        all_q_values[cell_idx, i] -= adjustment * 0.6
                        all_q_values[cell_idx + 1, i] += adjustment * 0.4
            
            # 6. Apply absolute bounds to all values
            max_discharge = 0.0   # Maximum discharge (least negative, as values are negative)
            min_discharge = -2.0 * total_q / num_cells  # Minimum discharge (most negative)
            
            for cell_idx in range(num_cells):
                all_q_values[cell_idx, :] = np.clip(all_q_values[cell_idx, :], min_discharge, max_discharge)
            
            # 7. Crucial step: Scale the final values to match the target total discharge
            current_total = np.sum(np.mean(all_q_values, axis=1))
            target_total = -total_q  # Target is negative total_q since we're using negative values
            
            if abs(current_total) > 0.001:  # Avoid division by zero
                scaling_factor = target_total / current_total
                all_q_values = all_q_values * scaling_factor
        else: 
            # # For the constant scenario, make sure there is no sinusoidal variation over the cumulative discharge
            # for cell_idx in range(num_cells):
            #     base_fraction = cell_base_distribution[cell_idx] / num_cells
            #     all_q_values[cell_idx, :] = -discharge_total * base_fraction  # Negative for outflow, respecting sign convention

            for t_idx, t in enumerate(time_points):
                phase = freq * t
                perturb = amplitude_perturb * np.array([
                    np.sin(phase + i * np.pi / (num_cells + 1)) for i in range(num_cells)
                ])
                weights = cell_base_distribution + perturb
                weights = np.clip(weights, 0.01, None)
                weights = weights / np.sum(weights)  # Normalize so sum(weights) == 1

                for cell_idx in range(num_cells):
                    all_q_values[cell_idx, t_idx] = -discharge_total[t_idx] * weights[cell_idx]


        # Only run tests for non-constant patterns
        if pattern_type != "constant":
            # TESTING SECTION
            # Test cell-to-cell difference limit
            max_diff = 0
            for i in range(num_steps):
                for cell_idx in range(1, num_cells):
                    diff = abs(all_q_values[cell_idx, i] - all_q_values[cell_idx - 1, i])
                    max_diff = max(max_diff, diff)
            
            if max_diff > 10.001:  # Allow tiny numerical error
                print(f"WARNING: Cell-to-cell difference limit exceeded: {max_diff:.2f}")
            else:
                print(f"Cell-to-cell difference check passed: Max difference is {max_diff:.2f}")
            
            # Test sinusoidal pattern - with adjusted expectations
            amplitudes = np.zeros(num_cells)
            expected_amplitude = amplitude * abs(scaling_factor)  # Adjust for final scaling
            
            for cell_idx in range(num_cells):
                def sine_func(t, amp, phase):
                    return amp * np.sin(2 * np.pi * t / duration_min + phase)
                
                try:
                    # Extract sinusoidal component by detrending
                    cell_mean = np.mean(all_q_values[cell_idx])
                    detrended = all_q_values[cell_idx] - cell_mean
                    
                    popt, pcov = curve_fit(sine_func, time_points, detrended, 
                                        p0=[expected_amplitude, cell_idx * np.pi / (num_cells + 1)])
                    amplitudes[cell_idx] = abs(popt[0])
                except RuntimeError:
                    print(f"WARNING: Could not fit sine wave to cell {cell_idx + 1}. Check data.")
                    amplitudes[cell_idx] = np.nan
            
            amplitude_tolerance = 0.25 * expected_amplitude  # More permissive tolerance
            
            for cell_idx in range(num_cells):
                if not np.isnan(amplitudes[cell_idx]):
                    amplitude_difference = abs(amplitudes[cell_idx] - expected_amplitude)
                    if amplitude_difference > amplitude_tolerance:
                        print(f"WARNING: Cell {cell_idx + 1} amplitude deviates: {amplitude_difference:.2f}")
                else:
                    print(f"WARNING: Amplitude check skipped for cell {cell_idx + 1} due to fitting error.")
            
            # After all perturbations, constraints, and clipping:
            total_discharge_timeseries = np.sum(all_q_values, axis=0)
            actual_mean = np.mean(total_discharge_timeseries)
            scaling_factor = -total_q / actual_mean
            all_q_values *= scaling_factor

            # Optional: Check the new mean
            new_mean = np.mean(np.sum(all_q_values, axis=0))
            print(f"Adjusted mean discharge: {new_mean:.2f} (should be {-total_q:.2f})")

            # # Test that total mean discharge is preserved
            # total_discharge_actual = np.sum(np.mean(all_q_values, axis=1))
            # discharge_tolerance = 0.01 * abs(total_q)  # 1% tolerance
            
            # if abs(total_discharge_actual - (-total_q)) > discharge_tolerance:
            #     print(f"WARNING: Total mean discharge not preserved: Expected {-total_q:.2f}, Actual {total_discharge_actual:.2f}")
            # else:
            #     print(f"Mean discharge check passed: Expected {-total_q:.2f}, Actual {total_discharge_actual:.2f}")
        else:
            print("Constant discharge mode: No variability tests performed")
            print(f"Total discharge: {total_q:.2f} m³/s")
            total_discharge_actual = np.sum(np.mean(all_q_values, axis=1))
            print(f"Actual discharge: {-total_discharge_actual:.2f} m³/s")
        
        # 8. Create .bct File Content
        spinup_duration = 2880  # minutes, e.g. 2 days
        spinup_discharge = - total_q / num_cells # constant across all cells for constant pattern

        bct_content = []

        for cell_idx in range(num_cells):
            section_number = cell_idx + 2
            boundary_name = f'river{cell_idx + 1}'.ljust(20)
            
            header = [
                f"table-name           'Boundary Section : {section_number}'",
                f"contents             'Uniform             '",
                f"location             '{boundary_name}'",
                "time-function        'non-equidistant'",
                "reference-time       20240101",
                "time-unit            'minutes'",
                "interpolation        'linear'",
                "parameter            'time                '                     unit '[min]'",
                "parameter            'total discharge (t)  end A'               unit '[m3/s]'",
                "parameter            'total discharge (t)  end B'               unit '[m3/s]'",
                f"records-in-table     {len(time_points)}"
            ]
            
            bct_content.extend(header)
        
            for t, q in zip(time_points, all_q_values[cell_idx]):
                # Add spin-up constant discharge rows
                if t <= spinup_duration:
                    bct_content.append(f"{t:<12.0f}\t{spinup_discharge:7.2f}\t{spinup_discharge:7.2f}")
                # Add varying discharge values after spin-up period
                if t > spinup_duration: 
                    bct_content.append(f"{t:<12.0f}\t{q:7.2f}\t{q:7.2f}")

                # # Optional: add blank line between sections for loading in GUI
                # bct_content.append('') 
        
        return '\n'.join(bct_content)
    
    def __apply_pattern_and_climate(discharge_total, total_q, time_points, pattern_type, climate_scenario):
        """Apply flow pattern and climate scenario adjustments."""
        num_steps = len(discharge_total)
        time_step = time_points[1] - time_points[0]  # Assuming uniform spacing

        # Apply pattern type (example implementation)
        if pattern_type == "seasonal":
            # One cycle per year (annual pattern)
            seasonal_amplitude = 0.25 * total_q
            discharge_total += seasonal_amplitude * np.sin(2 * np.pi * time_points / (365 * 24 * 60))
       
        elif pattern_type == "flashy":
            event_frequency = 10 #times during entire boundary condition timeseries 
            event_duration_days = 7 #days
            minutes_per_day = 24 * 60
            event_duration = int(event_duration_days * minutes_per_day / time_step)

            for _ in range(event_frequency):
                if num_steps - event_duration <= 0:
                    continue  # skip if event can't fit
                event_start = np.random.randint(0, num_steps - event_duration)
                discharge_total = add_ramped_event(discharge_total, event_start, event_duration, 0.8 * total_q)
        
        elif pattern_type == "intermittent":
            low_flow_prob = 0.8
            low_flow_discharge = 0.1 * total_q
            discharge_total[np.random.rand(num_steps) < low_flow_prob] = low_flow_discharge
        
        # Climate change adjustments (simplified)
        if climate_scenario == "EE":
            discharge_total[discharge_total > total_q] *= 1.2
        elif climate_scenario == "ED":
            discharge_total[discharge_total < 0.2 * total_q] *= 0.5
            discharge_total[discharge_total > total_q] *= 1.1
        elif climate_scenario == "PI-Med":
            discharge_total *= np.linspace(1, 2, len(discharge_total))
        
        return discharge_total
    
    def add_ramped_event(series, start, duration, peak_height):

        ramp_up = np.linspace(0, peak_height, duration//2)
        ramp_down = np.linspace(peak_height, 0, duration - duration//2)
        event = np.concatenate([ramp_up, ramp_down])
        end = min(start + duration, len(series))
        event = event[:end-start]
        series[start:end] += event

        return series
    
    def __achieve_flashiness(discharge_total, total_q, flashiness):
        np.random.seed(0)  # For reproducibility

        # Iterative approach to achieve target flashiness
        max_iterations = 10
        for iteration in range(max_iterations):
            # Calculate current flashiness
            current_p90 = np.percentile(discharge_total, 90)
            current_p10 = np.percentile(discharge_total, 10)
            
            # Avoid division by zero
            if current_p10 <= 0:
                current_p10 = 0.01 * total_q
                
            current_flashiness = current_p90 / current_p10
            
            # Break if we're within 5% of target
            if abs(current_flashiness - flashiness) / flashiness < 0.05:
                break
                
            # Adjust values to increase flashiness
            if current_flashiness < flashiness:
                # To increase flashiness: increase high flows and decrease low flows
                scale_high = np.min([1.5, (flashiness / current_flashiness)**0.5])
                scale_low = np.max([0.5, (current_flashiness / flashiness)**0.5])
                
                high_mask = discharge_total > np.percentile(discharge_total, 75)
                low_mask = discharge_total < np.percentile(discharge_total, 25)
                
                discharge_total[high_mask] *= scale_high
                discharge_total[low_mask] *= scale_low
            else:
                # To decrease flashiness: decrease high flows and increase low flows
                scale_high = np.max([0.7, (flashiness / current_flashiness)**0.5])
                scale_low = np.min([1.5, (current_flashiness / flashiness)**0.5])
                
                high_mask = discharge_total > np.percentile(discharge_total, 75)
                low_mask = discharge_total < np.percentile(discharge_total, 25)
                
                discharge_total[high_mask] *= scale_high
                discharge_total[low_mask] *= scale_low
            
            # Enforce mean after each iteration
            discharge_total = discharge_total * (total_q / np.mean(discharge_total))
        
        return discharge_total
                                       
    def generate_bch():
        """Generate .bch file content for tidal conditions."""
        if tide_north_south:
            bch_content = [
                "  0.0000000e+000\t3.0000000e+001",
                "",
                "  0.0000000e+000\t3.0000000e+000",
                "  0.0000000e+000\t3.0000000e+000",
                "  0.0000000e+000\t3.0000000e+000",
                "  0.0000000e+000\t3.0000000e+000",
                "",
                "                  3.0000000e+000",
                "                  0.0000000e+000",
                "                  3.0000000e+000",
                "                  0.0000000e+000"
            ]

        if tide_west:
            bch_content = [
                "  0.0000000e+000\t3.0000000e+001",
                "",
                "  0.0000000e+000\t3.0000000e+000",
                "  0.0000000e+000\t3.0000000e+000",
                "",
                "                  0.0000000e+000",
                "                  0.0000000e+000"

            ]
        return '\n'.join(bch_content)

    def generate_bcc(num_cells, duration_min):
        """Generate .bcc file content for sediment transport conditions."""
        bcc_content = []
        section = 1  # Section counter

        # Tidal boundaries
        if tide_north_south:
            bcc_content.extend([
                f"table-name           'Boundary Section : {section}'",
                f"contents             'Uniform   '",
                f"location             '{'neu_south'.ljust(20)}'",
                "time-function        'non-equidistant'",
                "reference-time       20240101",
                "time-unit            'minutes'",
                "interpolation        'linear'",
                "parameter            'time                '  unit '[min]'",
                "parameter            'Sediment1            end A uniform'       unit '[kg/m3]'",
                "parameter            'Sediment1            end B uniform'       unit '[kg/m3]'",
                "records-in-table     2",
                " 0.0000000e+000\t0.0000000e+000\t0.0000000e+000",
                f" {duration_min:.7e}\t0.0000000e+000\t0.0000000e+000"
            ])
            section += 1

            bcc_content.extend([
                f"table-name           'Boundary Section : {section}'",
                f"contents             'Uniform   '",
                f"location             '{'neu_north'.ljust(20)}'",
                "time-function        'non-equidistant'",
                "reference-time       20240101",
                "time-unit            'minutes'",
                "interpolation        'linear'",
                "parameter            'time                '  unit '[min]'",
                "parameter            'Sediment1            end A uniform'       unit '[kg/m3]'",
                "parameter            'Sediment1            end B uniform'       unit '[kg/m3]'",
                "records-in-table     2",
                " 0.0000000e+000\t0.0000000e+000\t0.0000000e+000",
                f" {duration_min:.7e}\t0.0000000e+000\t0.0000000e+000"
            ])
            section += 1

        if tide_west:
            bcc_content.extend([
                f"table-name           'Boundary Section : {section}'",
                f"contents             'Uniform   '",
                f"location             '{'neu_west'.ljust(20)}'",
                "time-function        'non-equidistant'",
                "reference-time       20240101",
                "time-unit            'minutes'",
                "interpolation        'linear'",
                "parameter            'time                '  unit '[min]'",
                "parameter            'Sediment1            end A uniform'       unit '[kg/m3]'",
                "parameter            'Sediment1            end B uniform'       unit '[kg/m3]'",
                "records-in-table     2",
                " 0.0000000e+000\t0.0000000e+000\t0.0000000e+000",
                f" {duration_min:.7e}\t0.0000000e+000\t0.0000000e+000"
            ])
            section += 1

        # River boundaries
        for i in range(num_cells):
            bcc_content.extend([
                f"table-name           'Boundary Section : {section}'",
                f"contents             'Uniform   '",
                f"location             '{f'river{i+1}'.ljust(20)}'",
                "time-function        'non-equidistant'",
                "reference-time       20240101",
                "time-unit            'minutes'",
                "interpolation        'linear'",
                "parameter            'time                '  unit '[min]'",
                "parameter            'Sediment1            end A uniform'       unit '[kg/m3]'",
                "parameter            'Sediment1            end B uniform'       unit '[kg/m3]'",
                "records-in-table     2",
                " 0.0000000e+000\t0.0000000e+000\t0.0000000e+000",
                f" {duration_min:.7e}\t0.0000000e+000\t0.0000000e+000"
            ])
            section += 1    

        return '\n'.join(bcc_content)

    # Generate and save .bnd file
    bnd_content = generate_bnd(grid_info)
    with open(os.path.join(boundary_dir, 'boundary.bnd'), 'w') as f:
        f.write(bnd_content)

    # Generate and save .bct file
    bct_content = generate_bct(params['total_discharge'], len(grid_info['river_cells']), params['duration_min'], params['time_step'],params['cv'], params['flashiness'], params['pattern_type'], params['climate_scenario'])
    with open(os.path.join(boundary_dir, 'river_boundary.bct'), 'w') as f:
        f.write(bct_content)

    # Generate and save .bch file
    bch_content = generate_bch()
    with open(os.path.join(boundary_dir, 'tide.bch'), 'w') as f:
        f.write(bch_content)

    # Generate and save .bcc file
    bcc_content = generate_bcc(len(grid_info['river_cells']), params['duration_min'])
    with open(os.path.join(boundary_dir, 'transport.bcc'), 'w') as f:
        f.write(bcc_content)

    print("All boundary files generated successfully.")


# %%
