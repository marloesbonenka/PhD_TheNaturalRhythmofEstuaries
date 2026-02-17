"""Utility functions for loading HIS data and resolving run folders."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from FUNCTIONS.F_tidalriverdominance import *
from FUNCTIONS.F_cache import *


def select_representative_days(times, n_periods=3):
	"""Select one hydrodynamic day from each period of the simulation."""
	n_total = len(times)
	period_size = n_total / n_periods

	day_duration_seconds = 24 * 3600
	dt = times[1] - times[0]
	dt_seconds = dt / np.timedelta64(1, 's')
	timesteps_per_day = int(np.round(day_duration_seconds / dt_seconds))

	selected_indices = []

	for period in range(n_periods):
		period_start = int(period * period_size)
		day_start = period_start + int(period_size / 2) - timesteps_per_day // 2
		day_start = max(0, min(day_start, n_total - timesteps_per_day))

		day_indices = np.arange(day_start, min(day_start + timesteps_per_day, n_total))
		selected_indices.extend(day_indices)

	return np.array(sorted(set(selected_indices)))


def find_mf_run_folder(base_dir, mf_number):
	"""Find the run folder matching the MF number (e.g., MF50_sens.8778435)."""
	base_dir = Path(base_dir)
	pattern = f"MF{mf_number}*"
	candidates = [p for p in base_dir.glob(pattern) if p.is_dir()]
	mf_regex = re.compile(rf"^MF{re.escape(str(mf_number))}(?!\d)")
	matches = sorted([p for p in candidates if mf_regex.match(p.name)])
	if not matches:
		raise FileNotFoundError(f"No run folder found for pattern '{pattern}' in {base_dir}")
	return matches[0], matches[0].name


def get_his_paths_for_run(base_dir, run_folder):

	"""
	Return list of HIS files to load. If run folder contains 'restart',
	include the timed-out part(s) (if available) before the main run.
	Supports both MF and scenario number-based folder naming.
	"""
	base_dir = Path(base_dir)
	run_folder = Path(run_folder)
	paths = [run_folder / "output" / "FlowFM_0000_his.nc"]
	debug_msgs = []
	
	timed_out_dir = base_dir / "timed-out"
	
	# Try MF logic first
	mf_match = re.search(r"MF(\d+(?:\.\d+)?)", run_folder.name)
	scenario_match = re.match(r"(\d+)[^\d]?", run_folder.name)
	timed_out_candidates = []
	if timed_out_dir.exists():
		if mf_match:
			mf_prefix = f"MF{int(float(mf_match.group(1)))}"
			timed_out_candidates = [p for p in timed_out_dir.rglob("*")
									if p.is_dir() and p.name.startswith(mf_prefix + "_")]
		elif scenario_match:
			# e.g. 3_Q500_rst_flashy.9094053 → 03_run500_flashy
			scenario_num = scenario_match.group(1)
			scenario_num_padded = scenario_num.zfill(2)
			# Find folders like 03_run500_flashy
			timed_out_candidates = [p for p in timed_out_dir.iterdir()
									if p.is_dir() and p.name.startswith(f"{scenario_num_padded}_")]
		matching = sorted(timed_out_candidates, key=lambda p: (p.name, str(p)))
		if matching:
			timed_out_paths = []
			for timed_out_folder in matching:
				timed_out_path = timed_out_folder / "output" / "FlowFM_0000_his.nc"
				if timed_out_path.exists():
					timed_out_paths.append(timed_out_path)
					debug_msgs.append(f"[DEBUG] Timed-out HIS found: {timed_out_path}")
				else:
					debug_msgs.append(f"[DEBUG] Timed-out HIS missing: {timed_out_path}")
			paths = timed_out_paths + paths
		else:
			if "restart" in run_folder.name.lower():
				debug_msgs.append("[DEBUG] Timed-out folder not found for restart run.")
	else:
		if "restart" in run_folder.name.lower():
			debug_msgs.append("[DEBUG] Timed-out dir missing or MF/scenario not found; skipping timed-out search.")

	for msg in debug_msgs:
		print(msg)
	# Return unique existing paths while preserving order
	seen = set()
	unique_paths = []
	for p in paths:
		if p.exists() and p not in seen:
			unique_paths.append(p)
			seen.add(p)
	return unique_paths


def open_his_dataset(his_paths):
	"""Open a single HIS file as dataset."""
	if isinstance(his_paths, (list, tuple)):
		if len(his_paths) == 1:
			return xr.open_dataset(his_paths[0])
		raise ValueError("Use manual append in load_cross_section_data for multiple HIS files")
	return xr.open_dataset(his_paths)


def load_cross_section_data(his_file_path, q_var='cross_section_discharge',
							estuary_only=True, km_range=(20, 45),
							select_cycles_hydrodynamic=True, n_periods=3,
							select_max_flood=False, flood_sign=-1,
							select_max_flood_per_cycle=False,
							exclude_last_timestep=False,
							exclude_last_n_days=0,
							selected_time_indices=None,
							dataset_cache=None):
	"""
	Load discharge data from HIS file(s) and extract cross-section information.
	
	Parameters
	----------
	dataset_cache : DatasetCache, optional
		A DatasetCache instance for caching datasets. If None, datasets are opened
		without caching and caller is responsible for closing them.
	"""
	use_cache = dataset_cache is not None
	
	if isinstance(his_file_path, (list, tuple)) and len(his_file_path) > 1:
		if use_cache:
			datasets = [dataset_cache.get_xr(p) for p in his_file_path]
			ds_first = datasets[0]
			ds_for_coords = ds_first
		else:
			ds_first = xr.open_dataset(his_file_path[0])
			ds_for_coords = ds_first
			datasets = None
	else:
		ds_first = None
		if use_cache:
			ds_for_coords = dataset_cache.get_xr(his_file_path if not isinstance(his_file_path, (list, tuple)) else his_file_path[0])
		else:
			ds_for_coords = open_his_dataset(his_file_path)

	cs_coords = ds_for_coords['cross_section_geom_node_coordx'].values
	cs_count = ds_for_coords['cross_section_geom_node_count'].values

	km_list = []
	idx_list = []
	x_start = 0

	for cs_idx, count in enumerate(cs_count):
		x_coords = cs_coords[x_start:x_start + int(count)]
		if len(x_coords) > 0:
			mean_x = np.mean(x_coords)
			km_pos = mean_x / 1000.0

			if estuary_only:
				if km_range[0] <= km_pos <= km_range[1]:
					km_list.append(km_pos)
					idx_list.append(cs_idx)
			else:
				km_list.append(km_pos)
				idx_list.append(cs_idx)
		x_start += int(count)

	if len(km_list) > 0:
		sorted_order = np.argsort(km_list)
		plot_km = np.array(km_list)[sorted_order]
		plot_indices = np.array(idx_list)[sorted_order]
	else:
		raise ValueError("No cross-sections found matching the specified criteria")

	if isinstance(his_file_path, (list, tuple)) and len(his_file_path) > 1:
		q_list = []
		t_list = []
		if datasets is None:
			datasets = [ds_first] + [xr.open_dataset(p) for p in his_file_path[1:]]
		last_time = None
		last_q_end = None
		for i, ds_part in enumerate(datasets):
			q_part = ds_part[q_var].isel(cross_section=plot_indices)
			t_part = ds_part['time'].values
			# Offset cumulative variables for seamless stitching
			if i > 0 and last_q_end is not None:
				# Only apply for cumulative variables (assume if 'cumulative' in q_var or user sets flag)
				if 'cumulative' in q_var or 'bedload_sediment_transport' in q_var or 'suspended_sediment_transport' in q_var:
					# Add last value of previous part to all of this part (per cross-section)
					q_part = q_part + last_q_end
			if last_time is not None and len(t_part) > 1:
				dt = t_part[1] - t_part[0]
				offset = (last_time - t_part[0]) + dt
				t_part = t_part + offset
			q_list.append(q_part)
			t_list.append(t_part)
			last_time = t_part[-1] if len(t_part) else last_time
			# Store last value for offsetting next part
			if q_part.shape[0] > 0:
				last_q_end = q_part[-1].values
		q_data = xr.concat(q_list, dim='time')
		times = np.concatenate(t_list)
		if not use_cache:
			for ds_part in datasets:
				ds_part.close()
		ds = ds_for_coords
	else:
		ds = ds_for_coords
		q_data = ds[q_var].isel(cross_section=plot_indices)
		times = ds['time'].values

	if exclude_last_timestep and len(times) > 1:
		q_data = q_data.isel(time=slice(0, -1))
		times = times[:-1]

	if exclude_last_n_days and len(times) > 1:
		dt = times[1] - times[0]
		dt_seconds = dt / np.timedelta64(1, 's')
		timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
		drop_steps = int(exclude_last_n_days) * timesteps_per_day
		if drop_steps > 0 and len(times) > drop_steps:
			q_data = q_data.isel(time=slice(0, -drop_steps))
			times = times[:-drop_steps]

	max_flood_km = None
	flood_sign_used = flood_sign
	def _flip_sign(sign):
		return 1 if sign == -1 else -1
	if selected_time_indices is not None:
		selected_time_indices = np.asarray(selected_time_indices, dtype=int)
		q_data = q_data.isel(time=selected_time_indices)
		times_selected = times[selected_time_indices]
		n_timesteps_original = len(times)
		selection_mode = 'external'
	elif select_max_flood_per_cycle:
		print("  Selecting max flood timestep for each cycle...")
		selected_time_indices = select_max_flood_indices_per_cycle(times, q_data, plot_km, flood_sign=flood_sign)
		if len(selected_time_indices) == 0:
			alt_sign = _flip_sign(flood_sign)
			alt_indices = select_max_flood_indices_per_cycle(times, q_data, plot_km, flood_sign=alt_sign)
			if len(alt_indices) > 0:
				selected_time_indices = alt_indices
				flood_sign_used = alt_sign
		q_data = q_data.isel(time=selected_time_indices)
		times_selected = times[selected_time_indices]
		n_timesteps_original = len(times)
		selection_mode = 'max_flood_per_cycle'
	elif select_max_flood:
		print("  Selecting maximum flood penetration timestep...")
		try:
			t_idx, max_flood_km = select_max_flood_timestep(q_data, plot_km, flood_sign=flood_sign)
		except ValueError:
			alt_sign = _flip_sign(flood_sign)
			t_idx, max_flood_km = select_max_flood_timestep(q_data, plot_km, flood_sign=alt_sign)
			flood_sign_used = alt_sign
		q_data = q_data.isel(time=[t_idx])
		times_selected = np.array([times[t_idx]])
		selected_time_indices = np.array([t_idx])
		n_timesteps_original = len(times)
		selection_mode = 'max_flood'
	elif select_cycles_hydrodynamic:
		print("  Selecting one hydrodynamic day from each period...")
		selected_time_indices = select_representative_days(times, n_periods=n_periods)
		dt = times[1] - times[0]
		dt_seconds = dt / np.timedelta64(1, 's')
		timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
		print(f"  Timesteps per day: {timesteps_per_day}")
		print(f"  Selecting {len(selected_time_indices)} total timesteps (~{len(selected_time_indices) // timesteps_per_day} complete days)")
		q_data = q_data.isel(time=selected_time_indices)
		times_selected = times[selected_time_indices]
		n_timesteps_original = len(times)
		selection_mode = 'representative_days'
	else:
		times_selected = times
		selected_time_indices = np.arange(len(times))
		n_timesteps_original = len(times)
		selection_mode = 'all'

	time_hours = (times_selected - times[0]) / np.timedelta64(1, 'h')
	times_datetime = pd.to_datetime(times_selected)

	return {
		'ds': ds,
		'discharge': q_data,
		'km_positions': plot_km,
		't': times, # raw time > timesteps
		'times': times_selected,
		'times_datetime': times_datetime,
		'time_hours': time_hours,
		'n_timesteps': len(times_selected),
		'n_timesteps_original': n_timesteps_original,
		'selected_indices': selected_time_indices,
		'cross_section_indices': plot_indices,
		'selection_mode': selection_mode,
		'max_flood_km': max_flood_km,
		'flood_sign_used': flood_sign_used,
	}


def read_discharge_from_bc_files(model_path, bc_pattern="*_Qr*_inflow_sinuous.bc"):
    """
    Read and sum discharge from boundary condition (.bc) files.
    
    The BC files contain discharge for individual river cells (01, 02, 03, 04).
    This function sums them to get the total river discharge.
    
    Parameters
    ----------
    model_path : Path
        Path to the model folder containing .bc files
    bc_pattern : str
        Glob pattern to find BC files
        
    Returns
    -------
    times : list of datetime
        Timestamps
    discharges : list of float
        Total discharge values (sum of all cells)
    """
    import re
    from datetime import datetime, timedelta
    
    # Find BC files
    bc_files = list(model_path.glob(bc_pattern))
    if not bc_files:
        # Also check in parent folder
        bc_files = list(model_path.parent.glob(bc_pattern))
    
    if not bc_files:
        print(f"  No BC files found matching {bc_pattern} in {model_path}")
        return None, None
    
    print(f"  Found {len(bc_files)} BC files: {[f.name for f in bc_files]}")
    
    # Parse all BC files
    all_data = {}
    reference_date = None
    
    for bc_file in bc_files:
        with open(bc_file, 'r') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        # Parse header to get reference date
        for line in lines:
            if 'seconds since' in line.lower():
                # Extract reference date: "unit = seconds since 2001-01-01 00:00:00"
                match = re.search(r'seconds since\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if match:
                    reference_date = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                break
        
        # Parse data lines (after header)
        data_started = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('[') or '=' in line:
                continue
            
            # Data lines are: "seconds   discharge"
            parts = line.split()
            if len(parts) >= 2:
                try:
                    seconds = float(parts[0])
                    discharge = float(parts[1])
                    
                    if seconds not in all_data:
                        all_data[seconds] = 0.0
                    all_data[seconds] += discharge
                except ValueError:
                    continue
    
    if not all_data or reference_date is None:
        print("  Could not parse BC files.")
        return None, None
    
    # Convert to timestamps and lists
    sorted_seconds = sorted(all_data.keys())
    times = [reference_date + timedelta(seconds=s) for s in sorted_seconds]
    discharges = [all_data[s] for s in sorted_seconds]
    
    print(f"  Read {len(times)} timesteps from BC files (ref: {reference_date})")
    
    return times, discharges


def extract_discharge_at_x(ds, x_target, y_range, time_indices=None):
    """
    Extract total discharge through edges near a given x-coordinate.
    
    Samples mesh2d_q1 (discharge through flow links) at a cross-section.
    Returns the integrated/summed discharge across the width.
    """
    # Get edge coordinates
    if 'mesh2d_edge_x' in ds:
        edge_x = ds['mesh2d_edge_x'].values
        edge_y = ds['mesh2d_edge_y'].values
    else:
        # Fallback: use face coordinates if edge coords not available
        print("  Warning: mesh2d_edge_x not found, using face coordinates.")
        edge_x = ds['mesh2d_face_x'].values
        edge_y = ds['mesh2d_face_y'].values
    
    # Find edges near the target x-coordinate and within y_range
    x_tolerance = 500  # meters tolerance for x-coordinate
    mask = (np.abs(edge_x - x_target) < x_tolerance) & \
           (edge_y >= y_range[0]) & (edge_y <= y_range[1])
    
    edge_indices = np.where(mask)[0]
    
    if len(edge_indices) == 0:
        print(f"  Warning: No edges found near x={x_target}m.")
        return None, None
    
    print(f"  Found {len(edge_indices)} edges near x={x_target/1000:.0f}km for discharge.")
    
    # Extract discharge time series
    if time_indices is None:
        time_indices = range(len(ds.time))
    
    times = []
    discharges = []
    
    for t in tqdm(time_indices, desc="  Extracting discharge", leave=False):
        q_vals = ds['mesh2d_q1'].isel(time=t).values[edge_indices]
        # Sum absolute discharge across the cross-section (accounts for different flow directions)
        total_q = np.nansum(np.abs(q_vals))
        
        times.append(pd.to_datetime(ds.time.values[t]))
        discharges.append(total_q)
    
    return times, discharges


def split_by_hydrodynamic_cycle(times, values, cycle_days=365.25):
    """
    Split a time series into individual hydrodynamic cycles.
    
    Parameters
    ----------
    times : array-like of datetime
        Time stamps
    values : array-like
        Values corresponding to times
    cycle_days : float
        Duration of one hydrodynamic cycle in days
        
    Returns
    -------
    cycles : list of dict
        Each dict contains: {'hydro_day': array, 'values': array, 'cycle_num': int}
    """
    times = pd.to_datetime(times)
    t0 = times[0]
    
    # Calculate hydro days from start
    hydro_days = (times - t0).total_seconds() / 86400
    
    # Determine cycle number for each point
    cycle_nums = (hydro_days // cycle_days).astype(int)
    
    # Split into cycles
    unique_cycles = np.unique(cycle_nums)
    cycles = []
    
    for cycle_num in unique_cycles:
        mask = cycle_nums == cycle_num
        cycle_hydro_days = hydro_days[mask] - (cycle_num * cycle_days)  # Reset to start of cycle
        cycle_values = np.array(values)[mask]
        
        cycles.append({
            'hydro_day': cycle_hydro_days.values,
            'values': cycle_values,
            'cycle_num': int(cycle_num),
        })
    
    return cycles