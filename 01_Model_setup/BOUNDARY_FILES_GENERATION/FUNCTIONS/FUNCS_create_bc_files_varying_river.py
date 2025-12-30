#%%
import numpy as np
import os

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
    
    # def generate_seasonal_ar1_discharge(
    #     total_q,
    #     num_steps,
    #     time_points,
    #     cv,
    #     ar1_rho=0.95,
    #     spring_amplitude=0.25,
    #     phase_shift_spring=80,   # Peak around day 80 (spring)
    #     autumn_amplitude=0.15,   # Relative amplitude for autumn peak
    #     phase_shift_autumn=260,  # Peak around day 260 (autumn)
    #     summer_event_amplitude=0.35,  # Amplitude for summer rainfall event
    #     summer_event_day=200,    # Day of summer rainfall event
    #     summer_event_width=5     # Width (std dev) of summer event in days
    # ):
    #     """
    #     Generate a synthetic seasonal discharge time series with AR(1) noise.
    #     - total_q: mean discharge (m³/s)
    #     - num_steps: number of time steps
    #     - time_points: array of time points (minutes)
    #     - cv: coefficient of variation for AR(1) noise
    #     - ar1_rho: AR(1) autocorrelation coefficient
    #     - spring_amplitude: relative amplitude of spring peak (fraction of mean)
    #     - phase_shift_spring: day of spring peak
    #     - autumn_amplitude: relative amplitude of autumn peak (fraction of mean)
    #     - phase_shift_autumn: day of autumn peak
    #     - summer_event_amplitude: amplitude of summer rainfall event (fraction of mean)
    #     - summer_event_day: day of summer rainfall event
    #     - summer_event_width: width (std dev) of summer event in days
    #     """
    #     # 1. Create seasonal mean with two peaks (spring & autumn)
    #     t_days = time_points / (60 * 24)
    #     period = 365.25

    #     # Main seasonal signals
    #     spring_peak = spring_amplitude * np.sin(2 * np.pi * (t_days - phase_shift_spring) / period)
    #     autumn_peak = autumn_amplitude * np.sin(2 * np.pi * (t_days - phase_shift_autumn) / period)
    #     seasonal_mean = total_q * (1 + spring_peak + autumn_peak)

    #     # 2. Add summer rainfall event (Gaussian pulse)
    #     summer_event = summer_event_amplitude * np.exp(-0.5 * ((t_days - summer_event_day) / summer_event_width) ** 2)
    #     seasonal_mean += total_q * summer_event

    #     # 3. Generate AR(1) noise (zero mean, std = cv * total_q)
    #     ar1_std = cv * total_q
    #     noise = np.zeros(num_steps)
    #     rng = np.random.default_rng(0)
    #     noise[0] = rng.normal(0, ar1_std)
    #     for i in range(1, num_steps):
    #         noise[i] = ar1_rho * noise[i-1] + rng.normal(0, ar1_std * np.sqrt(1 - ar1_rho**2))
        
    #     # 3. Combine
    #     discharge = seasonal_mean + noise

    #     # Ensure positivity
    #     discharge[discharge < 0] = 0

    #     return discharge
    
    def generate_bct(total_q, num_cells, duration_min, time_step, pattern_type, noise=False):
        # 1. Compute the timeseries for the variable river discharge boundary and determine a base for the redistribution over the different boundary cells (to force bar formation)
        time_points = np.arange(0, duration_min, time_step)
        num_steps = len(time_points)
        t_days = time_points / (60 * 24)  # Convert to days

        # 2. Generate discharge patterns based on scenario type
        if pattern_type == "constant":
            discharge_total = np.ones(num_steps) * total_q
            
        elif pattern_type == "seasonal":
            # Continuous seasonal signal: low in summer, high in winter
            # Using cosine with phase shift so minimum occurs around day 200 (summer)
            seasonal_amplitude = 0.4  # 30% variation around mean
            phase_shift = 200  # Summer minimum around day 200
            
            # Cosine function: peaks around day 20 (winter), minimum around day 200 (summer)
            seasonal_factor = 1 - seasonal_amplitude * np.cos(2 * np.pi * (t_days - phase_shift) / 365.25)
            discharge_total = total_q * seasonal_factor
            
            if noise:
                # Add small random variability (much smaller than your original cv)
                rng = np.random.default_rng(42)
                noise_std = 0.05 * total_q  # 5% noise
                ar1_rho = 0.8
                noise = np.zeros(num_steps)
                noise[0] = rng.normal(0, noise_std)
                for i in range(1, num_steps):
                    noise[i] = ar1_rho * noise[i-1] + rng.normal(0, noise_std * np.sqrt(1 - ar1_rho**2))
                
                discharge_total += noise
            
        elif pattern_type == "flashy":
            # Start with base discharge slightly above minimum to allow for low events
            base_discharge = 0.8 * total_q  # Base at 80% of mean
            discharge_total = np.full(num_steps, base_discharge)
            
            rng = np.random.default_rng(42)
            
            # Generate flood events (high discharge, max 3 days)
            flood_magnitude = 2.5 * total_q  # 3x mean discharge
            flood_duration_days = 3
            flood_duration_steps = int(flood_duration_days * 24 * 60 / time_step)  # Convert days to time steps
            
            # Generate low-flow events (40% of mean discharge)
            low_flow_magnitude = 0.4 * total_q
            low_flow_duration_days = 2
            low_flow_duration_steps = int(low_flow_duration_days * 24 * 60 / time_step)  # Convert days to time steps
            
            # Calculate how many events we need to maintain the mean
            simulation_days = duration_min / (60 * 24)  # Convert minutes to days
            num_years = int(np.floor(simulation_days) / 365.25)

            # Define wet and dry seasons
            wetseason_start, wetseason_end = 91, 183   # May to August
            dryseason1_start, dryseason1_end = 0, 90
            dryseason2_start, dryseason2_end = 184, 365

            floods_per_year = 5  # fixed 5 flood events per year
            flood_relative_starts = np.linspace(wetseason_start, wetseason_end - flood_duration_days, floods_per_year, dtype=int)

            flood_starts = []
            for year in range(num_years):
                for flood_day in flood_relative_starts:
                    abs_start_day = int(year * 365.25 + flood_day)
                    start_step = int(abs_start_day * 24 * 60 / time_step)
                    if start_step <= num_steps - flood_duration_steps:
                        flood_starts.append(start_step)

            # Low-flow events: explicitly fix 5 low flow events per year total (2 in first dry season, 3 in second dry season)
            low_events_per_year = 5
            low_events_dry1 = 2
            low_events_dry2 = low_events_per_year - low_events_dry1

            margin_days = 10

            low_flow_starts = []
            if low_events_dry1 > 0:
                low_dry1_relative = np.linspace(dryseason1_start + margin_days, dryseason1_end - margin_days - low_flow_duration_days, low_events_dry1, dtype=int)
                for year in range(num_years):
                    for low_day in low_dry1_relative:
                        abs_start_day = int(year * 365.25 + low_day)
                        start_step = int(abs_start_day * 24 * 60 / time_step)
                        if start_step <= num_steps - low_flow_duration_steps:
                            low_flow_starts.append(start_step)

            if low_events_dry2 > 0:
                low_dry2_relative = np.linspace(dryseason2_start + margin_days, dryseason2_end - margin_days - low_flow_duration_days, low_events_dry2, dtype=int)
                for year in range(num_years):
                    for low_day in low_dry2_relative:
                        abs_start_day = int(year * 365.25 + low_day)
                        start_step = int(abs_start_day * 24 * 60 / time_step)
                        if start_step <= num_steps - low_flow_duration_steps:
                            low_flow_starts.append(start_step)

            for start in flood_starts:
                end = min(start + flood_duration_steps, num_steps)
                discharge_total[start:end] = flood_magnitude

            for start in low_flow_starts:
                end = min(start + low_flow_duration_steps, num_steps)
                discharge_total[start:end] = low_flow_magnitude

        else:
            raise ValueError(f"Unknown pattern_type: {pattern_type}")

        # Ensure positivity and check mean preservation
        discharge_total[discharge_total < 0] = 0.1 * total_q  # Minimum 10% of mean
        
        # Check and correct mean if needed
        mean_total_discharge = np.mean(discharge_total)
        if abs(mean_total_discharge - total_q) > 0.01 * total_q:
            print(f'Deviation is larger than 1%: discharge_total = {mean_total_discharge:.2f}, target = {total_q}')
            print(f'Absolute difference = {abs(mean_total_discharge - total_q):.2f}')
            
            scaling_factor = total_q / mean_total_discharge
            discharge_total = discharge_total * scaling_factor
            new_mean = np.mean(discharge_total)
            
            print(f'Discharge corrected with scaling factor: {scaling_factor:.4f}')
            print(f'After correction, discharge_total = {new_mean:.2f}')
        
        # Ensure minimum discharge after scaling
        discharge_total[discharge_total < 0.1 * total_q] = 0.1 * total_q

        # Test actual flashiness (P90/P10 ratio)
        final_p90 = np.percentile(discharge_total, 90)
        final_p10 = np.percentile(discharge_total, 10)
        actual_flashiness = final_p90 / final_p10

        print(f"Pattern type: {pattern_type}")
        print(f"Final mean discharge: {np.mean(discharge_total):.2f} m³/s")
        print(f"Actual flashiness: {actual_flashiness:.2f}")
        print(f"P90 value: {final_p90:.2f}, P10 value: {final_p10:.2f}")
        
        # 3. Allocate discharge to adjacent river boundary cells
        num_steps = len(discharge_total)
        all_q_values = np.zeros((num_cells, num_steps))
        
        # Create time-varying base distribution instead of fixed one
        rng = np.random.default_rng(0)  # Fixed seed for reproducibility
        
        # Generate slowly varying weights for each cell over time
        weights_stack = np.zeros((num_cells, num_steps))
        
        # Create several sine waves with different frequencies and phases for each cell
        for cell_idx in range(num_cells):
            # Generate a unique set of sine waves for each cell
            # Use different periods (30 days, 60 days, 90 days, etc.) to create non-repeating patterns
            base = np.zeros(num_steps)
            
            # Add 3-4 sine waves with different periods and random phase shifts
            for i in range(4):
                period = rng.integers(30, 120) * 24 * 60 / time_step  # Convert days to steps
                amplitude = rng.uniform(0.7, 1.0)
                phase = rng.uniform(0, 2 * np.pi)
                base += amplitude * np.sin(2 * np.pi * np.arange(num_steps) / period + phase)
            
            # Add small random noise to create more natural variations
            noise = rng.normal(0, 0.5, num_steps)
            # Smooth the noise to avoid abrupt changes
            smooth_noise = np.convolve(noise, np.ones(24)/24, mode='same')
            
            # Combine base pattern with noise
            cell_weights = base + smooth_noise
            
            # Store in weights stack
            weights_stack[cell_idx, :] = cell_weights
        
        # Normalize to ensure positivity
        weights_min = np.min(weights_stack)
        if weights_min < 0:
            weights_stack -= weights_min  # Shift all values to be non-negative
        
        # Add a small positive value to avoid zeros
        weights_stack += 0.5
        
        # Normalize across cells at each time step
        # This ensures the sum equals 1 at each time step
        for t in range(num_steps):
            weights_stack[:, t] = weights_stack[:, t] / np.sum(weights_stack[:, t])
        
        # Check that no cell consistently dominates
        mean_weights = np.mean(weights_stack, axis=1)
        weight_range = np.max(mean_weights) - np.min(mean_weights)
        print(f"Mean cell weights: {mean_weights}")
        print(f"Weight range: {weight_range:.4f} (should be small)")
        
        # If weights are still too biased, apply additional balancing
        if weight_range > 0.05:  # If range > 5%, apply correction
            print("Applying additional balancing to cell distributions")
            # Apply a correction to reduce the range between cells
            for t in range(num_steps):
                # Mix with equal weights to reduce bias
                equal_weights = np.ones(num_cells) / num_cells
                # Mix 70% of original dynamic pattern with 30% equal distribution
                weights_stack[:, t] = 0.7 * weights_stack[:, t] + 0.3 * equal_weights
                # Renormalize
                weights_stack[:, t] = weights_stack[:, t] / np.sum(weights_stack[:, t])
        
        # Allocate discharge
        for cell_idx in range(num_cells):
            all_q_values[cell_idx, :] = -discharge_total * weights_stack[cell_idx, :]
            
        # 4. Create .bct File Content
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
                "  0.0000000e+000\t2.0000000e+000",
                "  0.0000000e+000\t2.0000000e+000",
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
    bct_content = generate_bct(params['total_discharge'], 
                               len(grid_info['river_cells']), 
                               params['duration_min'], 
                               params['time_step'],  
                               params['pattern_type'])
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

        # Old - more complex method for the seasonal and flashy patterns:
        # # 2. Compute river discharge variability signal over the timeseries

        # if pattern_type == "constant":
        #     discharge_total = np.ones(num_steps) * total_q
            
        # elif pattern_type == "seasonal":
        #     discharge_total = generate_seasonal_ar1_discharge(
        #                 total_q,
        #                 num_steps,
        #                 time_points,
        #                 cv,
        #                 ar1_rho=0.95,
        #                 spring_amplitude=0.25,
        #                 phase_shift_spring=80,          # Peak around day 80 (spring)
        #                 autumn_amplitude=0.25,          # Relative amplitude for autumn peak
        #                 phase_shift_autumn=260,         # Peak around day 260 (autumn)
        #                 summer_event_amplitude=0.3,     # Amplitude for summer rainfall event
        #                 summer_event_day=200,           # Day of summer rainfall event
        #                 summer_event_width=5            # Width (std dev) of summer event in days
        #             )

        # elif pattern_type == "flashy":
        #     discharge_total = generate_seasonal_ar1_discharge(
        #                 total_q,
        #                 num_steps,
        #                 time_points,
        #                 cv,
        #                 ar1_rho=0.9,
        #                 spring_amplitude=0.45,
        #                 phase_shift_spring=80,          # Peak around day 80 (spring)
        #                 autumn_amplitude=0.35,          # Relative amplitude for autumn peak
        #                 phase_shift_autumn=260,         # Peak around day 260 (autumn)
        #                 summer_event_amplitude=0.55,    # Amplitude for summer rainfall event
        #                 summer_event_day=200,           # Day of summer rainfall event
        #                 summer_event_width=5            # Width (std dev) of summer event in days
        #             )

        # # Check if total_q is preserved and correct if deviation is larger than 1% of total_q
        # mean_total_discharge = np.mean(discharge_total)

        # if abs(mean_total_discharge - total_q) > 0.01 * total_q:
        #     print('Deviation is larger than 1% discharge_total =', f'{mean_total_discharge:.2f}', 'Q0 =', total_q)
        #     print('Absolute difference =', f'{abs(mean_total_discharge - total_q):.2f}')
            
        #     scaling_factor = total_q / mean_total_discharge
        #     discharge_total = discharge_total * scaling_factor
        #     new_mean = np.mean(discharge_total)

        #     print('discharge_total is corrected with a scaling factor of:', f'{scaling_factor:.2f}')
        #     print('After correction, discharge_total =', f'{new_mean:.2f}')

        # discharge_total[discharge_total < 0] = 0  # ensure positivity