#%%
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#%%
def generate_river_discharges_fm(grid_info, params, output_dir, start_date_str='2024-01-01 00:00:00'):
    """
    Generates 5 separate CSV files with NEGATIVE discharge values for DFM.
    1 Cumulative + 4 Individual Section Files.
    """
    total_q = params['total_discharge']
    duration_min = params['duration_min']
    time_step = params['time_step']
    pattern_type = params['pattern_type']
    num_cells = len(grid_info['river_cells'])
    
    # 1. Setup Time and Dates
    time_minutes = np.arange(0, duration_min + time_step, time_step)
    num_steps = len(time_minutes)
    t_days = time_minutes / (60 * 24)
    
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    timestamps = [start_dt + timedelta(minutes=int(m)) for m in time_minutes]

    # 2. Pattern Generation (Calculating absolute magnitudes first)
    if pattern_type == "constant":
        discharge_abs = np.ones(num_steps) * total_q
        
    elif pattern_type == "seasonal":
        seasonal_amplitude = 0.4 
        phase_shift = 200  
        seasonal_factor = 1 - seasonal_amplitude * np.cos(2 * np.pi * (t_days - phase_shift) / 365.25)
        discharge_abs = total_q * seasonal_factor
            
    elif pattern_type == "flashy":
        base_discharge = 0.8 * total_q
        discharge_abs = np.full(num_steps, base_discharge)
        rng = np.random.default_rng(0)
        
        flood_magnitude = 2.5 * total_q
        flood_duration_steps = int(3 * 24 * 60 / time_step)
        low_flow_magnitude = 0.4 * total_q
        low_flow_duration_steps = int(2 * 24 * 60 / time_step)
        
        num_years = int(np.floor(duration_min / (365.25 * 24 * 60)))
        
        for year in range(max(1, num_years)):
            # Floods (Wet Season)
            flood_starts = np.linspace(91, 180, 5, dtype=int)
            for d in flood_starts:
                s = int((year * 365.25 + d) * 24 * 60 / time_step)
                if s < num_steps:
                    discharge_abs[s : s + flood_duration_steps] = flood_magnitude

            # Low flows (Dry Seasons)
            low_days = np.linspace(10, 80, 2, dtype=int).tolist() + np.linspace(200, 350, 3, dtype=int).tolist()
            for d in low_days:
                s = int((year * 365.25 + d) * 24 * 60 / time_step)
                if s < num_steps:
                    discharge_abs[s : s + low_flow_duration_steps] = low_flow_magnitude
    
    elif pattern_type == "singlepeak":
        # Single peak per year, no droughts - same magnitude and duration as flashy
        base_discharge = 0.8 * total_q
        discharge_abs = np.full(num_steps, base_discharge)
        
        flood_magnitude = 2.5 * total_q
        flood_duration_steps = int(3 * 24 * 60 / time_step)  # 3 days
        
        num_years = int(np.floor(duration_min / (365.25 * 24 * 60)))
        
        for year in range(max(1, num_years)):
            # Single flood peak around mid-year (day 135, midpoint of original wet season)
            flood_day = 135
            s = int((year * 365.25 + flood_day) * 24 * 60 / time_step)
            if s < num_steps:
                discharge_abs[s : s + flood_duration_steps] = flood_magnitude
        # No low flow/drought periods
    
    # Scaling & Positivity Check
    discharge_abs[discharge_abs < 0.1 * total_q] = 0.1 * total_q
    scaling_factor = total_q / np.mean(discharge_abs)
    discharge_abs *= scaling_factor

    # 3. Quasi-steady, FM-safe partitioning (Schuurman-style)
    num_cells = len(grid_info['river_cells'])
    num_steps = len(time_minutes)

    # Perturbation parameters
    A = 0.25                       # ±25% amplitude of partitioned discharge
    T_years = 500                   # period (very slow, centennial)
    T_minutes = T_years * 365.25 * 24 * 60

    # Normalized boundary coordinates (0–1 along the boundary)
    x = np.linspace(0, 1, num_cells, endpoint=False)

    weights_stack = np.zeros((num_cells, num_steps))

    for t_idx, t_min in enumerate(time_minutes):
        # Temporal factor is extremely slow; quasi-steady over 10 years
        temporal_factor = np.sin(2 * np.pi * t_min / T_minutes)
        
        # Spatial sinusoid along boundary
        spatial_factor = np.sin(2 * np.pi * x)
        
        # Compute fraction weights
        w = (1 / num_cells) * (1 + A * temporal_factor * spatial_factor)
        
        # Safety: enforce positivity and normalize
        w[w < 0] = 0.0
        w /= w.sum()
        
        weights_stack[:, t_idx] = w

    # 4. Save Files
    spinup_duration = 2880 # 2 days
    
    # File 1: Cumulative (Sum of all cells)
    pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': discharge_abs}).to_csv(
        os.path.join(output_dir, 'discharge_cumulative.csv'), index=False)

    # Files 2-5: Individual Sections
    for i in range(num_cells):
        section_q = discharge_abs * weights_stack[i, :]
        
        # Apply Spin-up (constant mean inflow)
        mask = time_minutes <= spinup_duration
        section_q[mask] = total_q / num_cells
        
        # Save
        pd.DataFrame({'timestamp': timestamps, 'discharge_m3s': section_q}).to_csv(
            os.path.join(output_dir, f'river_section_{i+1}.csv'), index=False)

    print(f"Scenario {params.get('name', '')}: Successfully saved 5 CSVs.")