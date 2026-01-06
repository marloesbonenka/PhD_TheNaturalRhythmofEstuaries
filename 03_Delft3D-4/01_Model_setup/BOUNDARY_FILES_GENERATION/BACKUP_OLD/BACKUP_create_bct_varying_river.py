

def generate_bct_0505(total_q, num_cells, duration_min, time_step, cv, flashiness, pattern_type, climate_scenario):
    """
    Generate boundary condition time series with efficient handling of flashiness and constraints.
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
    
    # 1. Setup time steps and base cell distribution (common for all patterns)
    time_points = np.arange(0, duration_min + time_step, time_step)
    num_steps = len(time_points)
    cell_base_distribution = np.linspace(0.8, 1.2, num_cells)
    cell_base_distribution = cell_base_distribution / np.sum(cell_base_distribution) * num_cells
    
    # 2. Generate total discharge timeseries
    if pattern_type == 'constant':
        discharge_total = np.ones(num_steps) * total_q
    else:
        # Start with Gaussian noise
        np.random.seed(0)  # For reproducibility
        discharge_total = np.random.normal(total_q, cv * total_q, num_steps)
        discharge_total[discharge_total < 0] = 0
        
        # Apply pattern and climate scenario in one step
        _apply_pattern_and_climate(discharge_total, total_q, time_points, pattern_type, climate_scenario)
        
        # Ensure no negative values after pattern application
        discharge_total[discharge_total < 0] = 0
        
        # Adjust for target flashiness
        _adjust_flashiness(discharge_total, total_q, flashiness)
        
        # Ensure we have correct mean discharge after flashiness adjustment
        discharge_total = discharge_total * (total_q / np.mean(discharge_total))
    
    # 3. Allocate discharge to cells with smoothing
    all_q_values = _allocate_discharge(discharge_total, cell_base_distribution, num_cells, time_points, duration_min)
    
    # 4. Apply constraints and scaling
    _apply_constraints(all_q_values, num_cells, num_steps, total_q)
    
    # 5. Testing (minimal but essential)
    _run_tests(all_q_values, num_cells, num_steps, total_q, flashiness, pattern_type)
    
    # 6. Create BCT file
    return _create_bct_content(all_q_values, time_points, total_q, num_cells)

def _apply_pattern_and_climate(discharge_total, total_q, time_points, pattern_type, climate_scenario):
    """Apply flow pattern and climate scenario adjustments."""
    # Apply pattern type
    if pattern_type == "seasonal":
        seasonal_amplitude = 0.25 * total_q
        discharge_total += seasonal_amplitude * np.sin(2 * 2 * np.pi * time_points / (365 * 24 * 60))
    elif pattern_type == "flashy":
        event_frequency = 10
        event_duration = 24 * 60
        for _ in range(event_frequency):
            event_start = np.random.randint(0, len(discharge_total))
            event_end = min(event_start + int(event_duration / np.mean(np.diff(time_points))), len(discharge_total))
            discharge_total[event_start:event_end] += 0.8 * total_q
    elif pattern_type == "intermittent":
        low_flow_prob = 0.8
        discharge_total[np.random.rand(len(discharge_total)) < low_flow_prob] = 0.1 * total_q
    
    # Apply climate scenario
    if climate_scenario == "EE":
        discharge_total[discharge_total > total_q] *= 1.2
    elif climate_scenario == "ED":
        discharge_total[discharge_total < 0.2 * total_q] *= 0.5
        discharge_total[discharge_total > total_q] *= 1.1
    elif climate_scenario == "PI-Med":
        discharge_total *= np.linspace(1, 2, len(discharge_total))
    
    # Ensure no negative values after pattern and climate application
    discharge_total[discharge_total < 0] = 0

def _adjust_flashiness(discharge_total, total_q, flashiness):
    """Iteratively adjust the discharge series to achieve target flashiness."""
    for _ in range(10):  # Max iterations
        current_p90 = np.percentile(discharge_total, 90)
        current_p10 = max(np.percentile(discharge_total, 10), 0.01 * total_q)  # Avoid division by zero
        current_flashiness = current_p90 / current_p10
        
        # Break if we're within 5% of target
        if abs(current_flashiness - flashiness) / flashiness < 0.05:
            break
            
        # Adjust values to increase/decrease flashiness
        if current_flashiness < flashiness:
            scale_high = min(1.5, (flashiness / current_flashiness)**0.5)
            scale_low = max(0.5, (current_flashiness / flashiness)**0.5)
        else:
            scale_high = max(0.7, (flashiness / current_flashiness)**0.5)
            scale_low = min(1.5, (current_flashiness / flashiness)**0.5)
        
        # Apply scaling
        high_mask = discharge_total > np.percentile(discharge_total, 75)
        low_mask = discharge_total < np.percentile(discharge_total, 25)
        discharge_total[high_mask] *= scale_high
        discharge_total[low_mask] *= scale_low
        
        # Maintain original mean
        discharge_total = discharge_total * (total_q / np.mean(discharge_total))
    
    # Report final flashiness metrics
    final_p90 = np.percentile(discharge_total, 90)
    final_p10 = np.percentile(discharge_total, 10)
    actual_flashiness = final_p90 / final_p10
    print(f"Target flashiness: {flashiness}")
    print(f"Actual flashiness: {actual_flashiness:.2f}")
    print(f"P90: {final_p90:.2f}, P10: {final_p10:.2f}")
    
    flashiness_error = abs(actual_flashiness - flashiness) / flashiness * 100
    if flashiness_error > 10:
        print(f"WARNING: Flashiness error: {flashiness_error:.1f}%")

def _allocate_discharge(discharge_total, cell_base_distribution, num_cells, time_points, duration_min):
    """Allocate discharge to cells with sinusoidal perturbation."""
    num_steps = len(time_points)
    all_q_values = np.zeros((num_cells, num_steps))
    
    # Parameters for sinusoidal perturbation
    amplitude_perturb = 0.03
    cycles_per_set = 12
    freq = 2 * np.pi / (duration_min / cycles_per_set)
    
    for t_idx, t in enumerate(time_points):
        phase = freq * t
        perturb = amplitude_perturb * np.array([np.sin(phase + i * np.pi / (num_cells + 1)) for i in range(num_cells)])
        
        weights = cell_base_distribution + perturb
        weights = np.clip(weights, 0.8, 1.2)
        weights = weights / np.sum(weights) * num_cells
        
        # Allocate discharge for this time step with negative sign for outflow
        for cell_idx in range(num_cells):
            all_q_values[cell_idx, t_idx] = -discharge_total[t_idx] * weights[cell_idx] / num_cells
    
    return all_q_values

def _apply_constraints(all_q_values, num_cells, num_steps, total_q):
    """Apply constraints and scaling to discharge values."""
    # Cell-to-cell difference constraint
    max_diff_limit = 9.0
    for t_idx in range(num_steps):
        for cell_idx in range(1, num_cells):
            diff = all_q_values[cell_idx, t_idx] - all_q_values[cell_idx - 1, t_idx]
            if abs(diff) > max_diff_limit:
                adjustment = diff - np.sign(diff) * max_diff_limit
                all_q_values[cell_idx, t_idx] -= adjustment * 0.5
                all_q_values[cell_idx - 1, t_idx] += adjustment * 0.5
    
    # Apply absolute bounds - CRITICAL STEP
    max_discharge = 0.0  # No values should be above 0 (since we're using negative outflow)
    min_discharge = -2.0 * total_q / num_cells
    
    # Apply clipping across the entire array
    all_q_values[:] = np.clip(all_q_values, min_discharge, max_discharge)
    
    # Scale to match target total discharge
    current_total = np.sum(np.mean(all_q_values, axis=1))
    target_total = -total_q
    
    if abs(current_total) > 0.001:  # Avoid division by zero
        scaling_factor = target_total / current_total
        all_q_values *= scaling_factor
    
    # Double-check after scaling to ensure all values remain negative
    all_q_values[all_q_values > 0] = 0
    
    return all_q_values

def _run_tests(all_q_values, num_cells, num_steps, total_q, flashiness, pattern_type='variable'):
    """Run tests to verify the generated discharge values (global, not per cell)."""
    import numpy as np

    # Check if ANY values are positive (should all be negative)
    if np.any(all_q_values > 0):
        print(f"ERROR: Found {np.sum(all_q_values > 0)} positive discharge values!")
        # Fix them again (redundant safety)
        all_q_values[all_q_values > 0] = 0
    else:
        print("All discharge values are correctly negative ✓")

    if pattern_type != "constant":
        # Flatten all values and use positive values for percentiles
        q_all = -all_q_values.flatten()  # Discharge is negative, so take -1

        mean_q = np.mean(q_all)
        std_q = np.std(q_all)
        p90 = np.percentile(q_all, 90)
        p10 = np.percentile(q_all, 10)

        # Avoid division by zero
        flashiness_value = p90 / p10 if p10 != 0 else np.nan
        cv_value = std_q / mean_q if mean_q != 0 else np.nan

        flashiness_error = abs(flashiness_value - flashiness) / max(flashiness, 1e-10) * 100

        print(f"\nGlobal P90: {p90:.2f}, P10: {p10:.2f}")
        print(f"Global flashiness (P90/P10): target={flashiness:.2f}, actual={flashiness_value:.2f} "
            f"{'- PASSED' if flashiness_error <= 5.0 else f'- ERROR: {flashiness_error:.2f}%'}")
        print(f"Global coefficient of variation (std/mean): {cv_value:.2f}")

        # Test cell-to-cell difference limit
        max_diff = 0
        for t_idx in range(num_steps):
            for cell_idx in range(1, num_cells):
                diff = abs(all_q_values[cell_idx, t_idx] - all_q_values[cell_idx - 1, t_idx])
                max_diff = max(max_diff, diff)

        print(f"Cell-to-cell max difference: {max_diff:.2f}" +
            (" - PASSED" if max_diff <= 10.001 else " - EXCEEDED LIMIT"))

        # Test mean discharge preservation
        total_discharge_actual = np.sum(np.mean(all_q_values, axis=1))
        discharge_error = abs(total_discharge_actual - (-total_q)) / max(abs(total_q), 1e-10) * 100

        print(f"Mean discharge: target={-total_q:.2f}, actual={total_discharge_actual:.2f}" +
            (f" - PASSED ({discharge_error:.2f}%)" if discharge_error <= 1.0 else f" - ERROR: {discharge_error:.2f}%"))

    else:
        print("Constant discharge mode: No variability tests performed")
        print(f"Total discharge: {total_q:.2f} m³/s")
        total_discharge_actual = np.sum(np.mean(all_q_values, axis=1))
        print(f"Actual discharge: {-total_discharge_actual:.2f} m³/s")

def _create_bct_content(all_q_values, time_points, total_q, num_cells):
    """Create BCT file content."""
    spinup_duration = 2880  # minutes
    spinup_discharge = -total_q / num_cells
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
            if t <= spinup_duration:
                bct_content.append(f"{t:<12.0f}\t{spinup_discharge:7.2f}\t{spinup_discharge:7.2f}")
            if t > spinup_duration:
                bct_content.append(f"{t:<12.0f}\t{q:7.2f}\t{q:7.2f}")
    
    return '\n'.join(bct_content)
    

def generate_bct_0205(total_q, num_cells, duration_min, time_step, cv, flashiness, pattern_type, climate_scenario):
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
        
        # 1. Generate Total Discharge Time Series
        time_points = np.arange(0, duration_min + time_step, time_step)
        num_steps = len(time_points)
        
        # Special case for constant discharge
        if pattern_type == "constant":
            # For constant pattern, set a completely uniform discharge with no variation
            discharge_total = np.ones(num_steps) * total_q
            all_q_values = np.zeros((num_cells, num_steps))
            
            for cell_idx in range(num_cells):
                # Divide total discharge evenly among all cells
                all_q_values[cell_idx, :] = -total_q / num_cells  # Negative for outflow
            
        else:
            # Original code for variable patterns
            std_dev = cv * total_q
            
            # Base discharge time series (start with Gaussian noise)
            np.random.seed(42)  # For reproducibility
            discharge_total = np.random.normal(total_q, std_dev, num_steps)
            
            # Ensure positivity
            discharge_total[discharge_total < 0] = 0
            
            # Apply pattern type (example implementation)
            if pattern_type == "seasonal":
                # Two cycles per year (semi-annual pattern)
                seasonal_amplitude = 0.25 * total_q
                discharge_total += seasonal_amplitude * np.sin(2 * 2 * np.pi * time_points / (365 * 24 * 60))
            elif pattern_type == "flashy":
                event_frequency = 10
                event_duration = 24 * 60
                for _ in range(event_frequency):
                    event_start = np.random.randint(0, num_steps)
                    event_end = min(event_start + int(event_duration / time_step), num_steps)
                    discharge_total[event_start:event_end] += 0.8 * total_q
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
                variability_increase = np.linspace(1, 2, num_steps)
                discharge_total *= variability_increase
            
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
                
            # 2. Allocate Discharge to Rivers
            # First, create basic distribution without sinusoidal components
            all_q_values = np.zeros((num_cells, num_steps))
            
            # Distribute base flow unevenly but with proper total
            cell_base_distribution = np.linspace(0.8, 1.2, num_cells)  # Less extreme to help with constraints
            cell_base_distribution = cell_base_distribution / np.sum(cell_base_distribution) * num_cells
            
            for cell_idx in range(num_cells):
                base_fraction = cell_base_distribution[cell_idx] / num_cells
                all_q_values[cell_idx, :] = -discharge_total * base_fraction  # Negative for outflow, respecting sign convention
            
            # 3. First pass of cell-to-cell difference constraint on base flow
            max_diff_limit = 9.0  # Slightly lower to leave room for sinusoidal additions
            
            for i in range(num_steps):
                for cell_idx in range(1, num_cells):
                    diff = all_q_values[cell_idx, i] - all_q_values[cell_idx - 1, i]
                    if abs(diff) > max_diff_limit:
                        adjustment = diff - np.sign(diff) * max_diff_limit
                        all_q_values[cell_idx, i] -= adjustment * 0.5
                        all_q_values[cell_idx - 1, i] += adjustment * 0.5
                        
            # 4. Now add smaller sinusoidal components that won't break constraints
            # Using smaller amplitude to avoid breaking constraints
            amplitude = 0.1 * total_q  # Reduced amplitude
            
            for cell_idx in range(num_cells):
                phase = cell_idx * np.pi / (num_cells + 1)  # Better phase spacing
                sinusoidal_component = amplitude * np.sin(2 * np.pi * time_points / (duration_min * 0.5) + phase)
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
            min_discharge = -20.0  # Minimum discharge (most negative)
            
            for cell_idx in range(num_cells):
                all_q_values[cell_idx, :] = np.clip(all_q_values[cell_idx, :], min_discharge, max_discharge)
            
            # 7. Crucial step: Scale the final values to match the target total discharge
            current_total = np.sum(np.mean(all_q_values, axis=1))
            target_total = -total_q  # Target is negative total_q since we're using negative values
            
            if abs(current_total) > 0.001:  # Avoid division by zero
                scaling_factor = target_total / current_total
                all_q_values = all_q_values * scaling_factor
        
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
                    return amp * np.sin(2 * np.pi * t / (duration_min * 0.5) + phase)
                
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
            
            # Test that total mean discharge is preserved
            total_discharge_actual = np.sum(np.mean(all_q_values, axis=1))
            discharge_tolerance = 0.01 * abs(total_q)  # 1% tolerance
            
            if abs(total_discharge_actual - (-total_q)) > discharge_tolerance:
                print(f"WARNING: Total mean discharge not preserved: Expected {-total_q:.2f}, Actual {total_discharge_actual:.2f}")
            else:
                print(f"Mean discharge check passed: Expected {-total_q:.2f}, Actual {total_discharge_actual:.2f}")
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

def generate_bct(total_q, num_cells, duration_min, time_step, cv, flashiness, pattern_type, climate_scenario):
    """
    Generate boundary condition time series with improved handling of flashiness and constraints.
    Parameters:
        - total_q: Total discharge (m³/s)
        - num_cells: Number of river cells
        - duration_min: Simulation duration in minutes
        - time_step: Time step in minutes
        - cv: Coefficient of variation for initial noise
        - flashiness: Target flashiness (P90/P10 ratio)
        - pattern_type: Flow pattern - "seasonal", "flashy", or "intermittent"
        - climate_scenario: Climate scenario - "EE", "ED", or "PI-Med"
        """
    
    # 1. Generate Total Discharge Time Series
    std_dev = cv * total_q
    time_points = np.arange(0, duration_min + time_step, time_step)
    num_steps = len(time_points)
    
    # Base discharge time series (start with Gaussian noise)
    np.random.seed(42)  # For reproducibility
    discharge_total = np.random.normal(total_q, std_dev, num_steps)
    
    # Ensure positivity
    discharge_total[discharge_total < 0] = 0
    
    # Apply pattern type (example implementation)
    if pattern_type == "seasonal":
        # Two cycles per year (semi-annual pattern)
        seasonal_amplitude = 0.25 * total_q
        discharge_total += seasonal_amplitude * np.sin(2 * 2 * np.pi * time_points / (365 * 24 * 60))
    elif pattern_type == "flashy":
        event_frequency = 10
        event_duration = 24 * 60
        for _ in range(event_frequency):
            event_start = np.random.randint(0, num_steps)
            event_end = min(event_start + int(event_duration / time_step), num_steps)
            discharge_total[event_start:event_end] += 0.8 * total_q
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
        variability_increase = np.linspace(1, 2, num_steps)
        discharge_total *= variability_increase
    
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
    
    # 2. Allocate Discharge to Rivers
    # First, create basic distribution without sinusoidal components
    all_q_values = np.zeros((num_cells, num_steps))
    
    # Distribute base flow unevenly but with proper total
    cell_base_distribution = np.linspace(0.8, 1.2, num_cells)  # Less extreme to help with constraints
    cell_base_distribution = cell_base_distribution / np.sum(cell_base_distribution) * num_cells
    
    for cell_idx in range(num_cells):
        base_fraction = cell_base_distribution[cell_idx] / num_cells
        all_q_values[cell_idx, :] = -discharge_total * base_fraction  # Negative for outflow, respecting sign convention
    
    # 3. First pass of cell-to-cell difference constraint on base flow
    max_diff_limit = 9.0  # Slightly lower to leave room for sinusoidal additions
    
    for i in range(num_steps):
        for cell_idx in range(1, num_cells):
            diff = all_q_values[cell_idx, i] - all_q_values[cell_idx - 1, i]
            if abs(diff) > max_diff_limit:
                adjustment = diff - np.sign(diff) * max_diff_limit
                all_q_values[cell_idx, i] -= adjustment * 0.5
                all_q_values[cell_idx - 1, i] += adjustment * 0.5
                
    # 4. Now add smaller sinusoidal components that won't break constraints
    # Using smaller amplitude to avoid breaking constraints
    amplitude = 0.1 * total_q  # Reduced amplitude
    
    for cell_idx in range(num_cells):
        phase = cell_idx * np.pi / (num_cells + 1)  # Better phase spacing
        sinusoidal_component = amplitude * np.sin(2 * np.pi * time_points / (duration_min * 0.5) + phase)
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
    min_discharge = -20.0  # Minimum discharge (most negative)
    
    for cell_idx in range(num_cells):
        all_q_values[cell_idx, :] = np.clip(all_q_values[cell_idx, :], min_discharge, max_discharge)
    
    # 7. Crucial step: Scale the final values to match the target total discharge
    current_total = np.sum(np.mean(all_q_values, axis=1))
    target_total = -total_q  # Target is negative total_q since we're using negative values
    
    if abs(current_total) > 0.001:  # Avoid division by zero
        scaling_factor = target_total / current_total
        all_q_values = all_q_values * scaling_factor
    
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
            return amp * np.sin(2 * np.pi * t / (duration_min * 0.5) + phase)
        
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
    
    # Test that total mean discharge is preserved
    total_discharge_actual = np.sum(np.mean(all_q_values, axis=1))
    discharge_tolerance = 0.01 * abs(total_q)  # 1% tolerance
    
    if abs(total_discharge_actual - (-total_q)) > discharge_tolerance:
        print(f"WARNING: Total mean discharge not preserved: Expected {-total_q:.2f}, Actual {total_discharge_actual:.2f}")
    else:
        print(f"Mean discharge check passed: Expected {-total_q:.2f}, Actual {total_discharge_actual:.2f}")
    
    # 8. Create .bct File Content
    spinup_duration = 2880  # minutes, e.g. 2 days
    spinup_discharge = - total_q / 4 # negative due to flow from right to left

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

def generate_bcc(num_cells, duration_min):
    bcc_content = []

    for i in range(num_cells + 2):
        location = f"neu_south" if i == 0 else f"neu_north" if i == 1 else f"river{i - 1}"
        bcc_content.extend([
            f"table-name           'Boundary Section : {i+1}'",
            f"contents             'Uniform   '",
            f"location             '{location.ljust(20)}'",
            "time-function        'non-equidistant'",
            "reference-time       20240101",
            "time-unit            'minutes'",
            "interpolation        'linear'",
            "parameter            'time                '  unit '[min]'",
            "parameter            'Sediment1            end A uniform'       unit '[kg/m3]'",
            "parameter            'Sediment1            end B uniform'       unit '[kg/m3]'",
            "records-in-table     2",
            " 0.0000000e+000\t0.0000000e+000\t0.0000000e+000",
            f" {duration_min:.7e}\t0.0000000e+000\t0.0000000e+000"#,
            # f"" #If loading in GUI, this line is necessary
        ])
    return '\n'.join(bcc_content)