"""
Plot longitudinal discharge profiles from sea to river over morphological time,
to visualize the transition from tidal to river dominance.
"""

import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def select_representative_days(times, n_periods=3):
    """
    Select one hydrodynamic day from each period of the simulation.
    
    A hydrodynamic day is one complete semi-diurnal tidal cycle (24 hours).
    Divides the simulation into N periods and selects one day from each.
    
    Parameters
    ----------
    times : np.ndarray
        Array of all timesteps (in seconds)
    n_periods : int
        Number of periods to divide simulation into (default: 3 = start/middle/end)
    
    Returns
    -------
    np.ndarray : Indices of selected timesteps (one complete day from each period)
    """
    n_total = len(times)
    period_size = n_total / n_periods
    
    # Duration of one hydrodynamic day in seconds (24 hours)
    day_duration_seconds = 24 * 3600
    
    # Calculate timestep interval from the data
    dt = times[1] - times[0]  # timedelta between timesteps
    # Convert timedelta to seconds
    dt_seconds = dt / np.timedelta64(1, 's')
    timesteps_per_day = int(np.round(day_duration_seconds / dt_seconds))
    
    selected_indices = []
    
    for period in range(n_periods):
        # Start of this period
        period_start = int(period * period_size)
        
        # Select one complete day starting from the middle of this period
        day_start = period_start + int(period_size / 2) - timesteps_per_day // 2
        day_start = max(0, min(day_start, n_total - timesteps_per_day))
        
        day_indices = np.arange(day_start, min(day_start + timesteps_per_day, n_total))
        selected_indices.extend(day_indices)
    
    return np.array(sorted(set(selected_indices)))


def select_max_flood_timestep(discharge, km_positions, flood_sign=-1):
    """
    Select the timestep with maximum flood penetration (sea to river).
    
    Parameters
    ----------
    discharge : xarray.DataArray
        Discharge data with dims (time, cross_section)
    km_positions : np.ndarray
        Cross-section positions (km, sea to river)
    flood_sign : int
        Sign convention for flood (default: -1 means negative discharge is flood)
    
    Returns
    -------
    tuple : (t_idx, max_flood_km)
    """
    q = discharge.values
    if flood_sign < 0:
        flood_mask = q < 0
    else:
        flood_mask = q > 0
    
    km_grid = np.broadcast_to(km_positions, q.shape)
    flood_km = np.where(flood_mask, km_grid, np.nan)
    max_flood_km_per_time = np.nanmax(flood_km, axis=1)
    
    if np.all(np.isnan(max_flood_km_per_time)):
        raise ValueError("No flood conditions found in the selected discharge data")
    
    t_idx = int(np.nanargmax(max_flood_km_per_time))
    return t_idx, max_flood_km_per_time[t_idx]


def select_max_flood_indices_per_cycle(times, discharge, km_positions, flood_sign=-1):
    """
    Select the max-flood timestep within each tidal day (24h) cycle.
    
    Parameters
    ----------
    times : np.ndarray
        Array of all timesteps (datetime64)
    discharge : xarray.DataArray
        Discharge data with dims (time, cross_section)
    km_positions : np.ndarray
        Cross-section positions (km, sea to river)
    flood_sign : int
        Sign convention for flood (default: -1 means negative discharge is flood)
    
    Returns
    -------
    np.ndarray : Indices of max-flood timestep per cycle
    """
    dt = times[1] - times[0]
    dt_seconds = dt / np.timedelta64(1, 's')
    timesteps_per_day = int(np.round(24 * 3600 / dt_seconds))
    n_total = len(times)
    indices = []
    
    for start in range(0, n_total, timesteps_per_day):
        end = start + timesteps_per_day
        if end > n_total:
            break
        q_slice = discharge.isel(time=slice(start, end))
        try:
            t_idx_local, _ = select_max_flood_timestep(q_slice, km_positions, flood_sign=flood_sign)
        except ValueError:
            continue
        indices.append(start + t_idx_local)
    
    return np.array(indices, dtype=int)


def select_max_flood_indices_by_period(times, discharge, km_positions, n_periods=3, flood_sign=-1):
    """
    Select the max-flood timestep within each of N simulation periods.
    
    Parameters
    ----------
    times : np.ndarray
        Array of all timesteps (datetime64)
    discharge : xarray.DataArray
        Discharge data with dims (time, cross_section)
    km_positions : np.ndarray
        Cross-section positions (km, sea to river)
    n_periods : int
        Number of periods (default: 3)
    flood_sign : int
        Sign convention for flood (default: -1 means negative discharge is flood)
    
    Returns
    -------
    np.ndarray : Indices of max-flood timestep per period
    """
    n_total = len(times)
    period_size = n_total / n_periods
    indices = []
    
    for period in range(n_periods):
        start = int(period * period_size)
        end = int(min((period + 1) * period_size, n_total))
        if end - start < 2:
            continue
        q_slice = discharge.isel(time=slice(start, end))
        try:
            t_idx_local, _ = select_max_flood_timestep(q_slice, km_positions, flood_sign=flood_sign)
        except ValueError:
            continue
        indices.append(start + t_idx_local)
    
    return np.array(indices, dtype=int)


def load_cross_section_data(his_file_path, q_var='cross_section_discharge', 
                            estuary_only=True, km_range=(20, 45),
                            select_cycles_hydrodynamic=True, n_periods=3,
                            select_max_flood=False, flood_sign=-1,
                            select_max_flood_per_cycle=False,
                            exclude_last_timestep=False):
    """
    Load discharge data from HIS file and extract cross-section information.
    
    Parameters
    ----------
    his_file_path : Path or str
        Path to the HIS netCDF file
    q_var : str
        Variable name for discharge data
    estuary_only : bool
        If True, filter to estuary cross-sections only (exclude SeaBnd)
    km_range : tuple
        Min and max km values to include (default: 18-45 for estuary)
    select_cycles_hydrodynamic : bool
        If True, automatically detect and select tidal cycles from hydrodynamics
    n_periods : int
        Number of periods to divide simulation into (default: 3 = start/middle/end)
    select_max_flood : bool
        If True, select the single timestep with maximum flood penetration
    flood_sign : int
        Sign convention for flood (default: -1 means negative discharge is flood)
    select_max_flood_per_cycle : bool
        If True, select the max-flood timestep within each tidal cycle
    exclude_last_timestep : bool
        If True, exclude the last timestep from analysis
    
    Returns
    -------
    dict : Contains discharge data, coordinates, times, and cross-section info
    """
    ds = xr.open_dataset(his_file_path)
    
    # Extract cross-section coordinates
    # Each cross-section has multiple nodes; we'll use the mean x-coordinate
    cs_coords = ds['cross_section_geom_node_coordx'].values
    cs_count = ds['cross_section_geom_node_count'].values
    
    km_list = []
    idx_list = []
    x_start = 0
    
    for cs_idx, count in enumerate(cs_count):
        x_coords = cs_coords[x_start:x_start + int(count)]
        if len(x_coords) > 0:
            mean_x = np.mean(x_coords)
            # Normalize to km (assuming coordinates in meters, divide by 1000)
            km_pos = mean_x / 1000.0
            
            if estuary_only:
                # Filter to specified range
                if km_range[0] <= km_pos <= km_range[1]:
                    km_list.append(km_pos)
                    idx_list.append(cs_idx)
            else:
                km_list.append(km_pos)
                idx_list.append(cs_idx)
        x_start += int(count)
    
    # Sort by km position (sea to river)
    if len(km_list) > 0:
        sorted_order = np.argsort(km_list)
        plot_km = np.array(km_list)[sorted_order]
        plot_indices = np.array(idx_list)[sorted_order]
    else:
        raise ValueError("No cross-sections found matching the specified criteria")
    
    # Extract discharge data
    q_data = ds[q_var].isel(cross_section=plot_indices)
    times = ds['time'].values
    
    if exclude_last_timestep and len(times) > 1:
        q_data = q_data.isel(time=slice(0, -1))
        times = times[:-1]
    
    # Select representative days if requested
    max_flood_km = None
    if select_max_flood_per_cycle:
        print("  Selecting max flood timestep for each cycle...")
        selected_time_indices = select_max_flood_indices_per_cycle(times, q_data, plot_km, flood_sign=flood_sign)
        q_data = q_data.isel(time=selected_time_indices)
        times_selected = times[selected_time_indices]
        n_timesteps_original = len(times)
        selection_mode = 'max_flood_per_cycle'
    elif select_max_flood:
        print("  Selecting maximum flood penetration timestep...")
        t_idx, max_flood_km = select_max_flood_timestep(q_data, plot_km, flood_sign=flood_sign)
        q_data = q_data.isel(time=[t_idx])
        times_selected = np.array([times[t_idx]])
        selected_time_indices = np.array([t_idx])
        n_timesteps_original = len(times)
        selection_mode = 'max_flood'
    elif select_cycles_hydrodynamic:
        print(f"  Selecting one hydrodynamic day from each period...")
        
        # Collect all timestep indices for the 3 days
        selected_time_indices = select_representative_days(times, n_periods=n_periods)
        
        # Calculate timesteps per day for info
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
    
    # Convert times to hours (datetime64 to float)
    time_hours = (times_selected - times[0]) / np.timedelta64(1, 'h')
    times_datetime = pd.to_datetime(times_selected)
    
    return {
        'ds': ds,
        'discharge': q_data,
        'km_positions': plot_km,
        'times': times_selected,
        'times_datetime': times_datetime,
        'time_hours': time_hours,
        'n_timesteps': len(times_selected),
        'n_timesteps_original': n_timesteps_original,
        'selected_indices': selected_time_indices,
        'cross_section_indices': plot_indices,
        'selection_mode': selection_mode,
        'max_flood_km': max_flood_km,
    }


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_max_flood_profile(data, figsize=(12, 4)):
    """
    Plot discharge profile at maximum flood penetration timestep.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_cross_section_data()
    figsize : tuple
        Figure size
    
    Returns
    -------
    tuple : (fig, ax)
    """
    discharge = data['discharge']
    km_positions = data['km_positions']
    times = data['times_datetime']
    max_flood_km = data.get('max_flood_km')
    
    if 'time' in discharge.dims:
        q_profile = discharge.isel(time=0).values
    else:
        q_profile = discharge.values
    time_label = pd.to_datetime(times[0]).strftime('%Y-%m-%d %H:%M') if len(times) else ''
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(km_positions, q_profile, color='tab:blue', linewidth=2)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Discharge [m³/s]', fontsize=11, fontweight='bold')
    if max_flood_km is not None:
        ax.set_title(f'Max flood profile at {time_label} (penetration to km {max_flood_km:.1f})',
                     fontsize=12, fontweight='bold')
    else:
        ax.set_title(f'Max flood profile at {time_label}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle=':')
    
    return fig, ax


def plot_multiple_max_flood_profiles(data, indices, figsize=(12, 5)):
    """
    Plot multiple max-flood profiles for selected timesteps.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_cross_section_data()
    indices : np.ndarray
        Indices of timesteps to plot
    figsize : tuple
        Figure size
    
    Returns
    -------
    tuple : (fig, ax)
    """
    discharge = data['discharge']
    km_positions = data['km_positions']
    times = pd.to_datetime(data['times_datetime'])
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(indices)))
    
    for i, t_idx in enumerate(indices):
        q_profile = discharge.isel(time=t_idx).values
        label = times[t_idx].strftime('%Y-%m-%d')
        ax.plot(km_positions, q_profile, color=colors[i], linewidth=2, label=label)
    
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Discharge [m³/s]', fontsize=11, fontweight='bold')
    ax.set_title('Max-flood profiles (representative periods)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle=':')
    ax.legend(loc='best', fontsize=9)
    
    return fig, ax

def plot_discharge_statistics(data, figsize=(14, 8)):
    """
    Create subplot showing mean, min, max discharge profiles.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_cross_section_data()
    figsize : tuple
        Figure size
    
    Returns
    -------
    tuple : (fig, axes)
    """
    discharge = data['discharge']
    km_positions = data['km_positions']
    
    # Calculate statistics
    mean_q = discharge.mean(dim='time').values
    std_q = discharge.std(dim='time').values
    max_q = discharge.max(dim='time').values
    min_q = discharge.min(dim='time').values
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot 1: Mean with standard deviation band
    ax = axes[0]
    ax.plot(km_positions, mean_q, 'b-', linewidth=2.5, label='Mean Discharge')
    ax.fill_between(km_positions, mean_q - std_q, mean_q + std_q,
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_ylabel('Discharge [m³/s]', fontsize=11, fontweight='bold')
    ax.set_title('Mean Longitudinal Discharge Profile with Variability', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle=':')
    ax.legend(loc='upper left')
    
    # Plot 2: Range (min/max)
    ax = axes[1]
    ax.fill_between(km_positions, min_q, max_q, alpha=0.4, color='red', label='Min-Max Range')
    ax.plot(km_positions, mean_q, 'b-', linewidth=2, label='Mean', alpha=0.7)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Discharge [m³/s]', fontsize=11, fontweight='bold')
    ax.set_title('Discharge Range Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle=':')
    ax.legend(loc='upper left')
    
    return fig, axes


def plot_upstream_inflow_timeseries(data, figsize=(12, 4)):
    """
    Plot discharge time series at the most upstream cross-section.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_cross_section_data()
    figsize : tuple
        Figure size
    
    Returns
    -------
    tuple : (fig, ax)
    """
    discharge = data['discharge']
    km_positions = data['km_positions']
    times = data['times_datetime']
    
    upstream_idx = int(np.argmax(km_positions))
    q_upstream = discharge.isel(cross_section=upstream_idx).values
    
    times_dt = pd.to_datetime(times)
    q_vals = np.asarray(q_upstream)
    
    # Break lines at large gaps (e.g., between selected days)
    if len(times_dt) > 1:
        dt_seconds = np.diff(times_dt.values) / np.timedelta64(1, 's')
        median_dt = np.nanmedian(dt_seconds)
        gap_threshold = median_dt * 1.5
        gap_indices = np.where(dt_seconds > gap_threshold)[0]
        
        times_plot = times_dt.to_numpy()
        q_plot = q_vals.copy()
        for offset, idx in enumerate(gap_indices, start=1):
            insert_at = idx + offset
            times_plot = np.insert(times_plot, insert_at, np.datetime64('NaT'))
            q_plot = np.insert(q_plot, insert_at, np.nan)
    else:
        times_plot = times_dt.to_numpy()
        q_plot = q_vals
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times_plot, q_plot, color='tab:red', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
    ax.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Discharge [m³/s]', fontsize=11, fontweight='bold')
    ax.set_title(f'Upstream Inflow (most upstream cross-section, km {km_positions[upstream_idx]:.1f})',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.5, linestyle=':')
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.autofmt_xdate()
    
    return fig, ax


def plot_discharge_heatmap(data, figsize=(14, 6), flood_sign=-1, show_flood_limit=True,
                           percentile_low=2, percentile_high=95, symmetric_scale=False):
    """
    Create a 2D heatmap showing discharge evolution in space and time.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_cross_section_data()
    figsize : tuple
        Figure size
    flood_sign : int
        Sign convention for flood (default: -1 means negative discharge is flood)
    show_flood_limit : bool
        If True, overlay the flood-penetration limit line
    percentile_low : float
        Lower percentile for color scaling
    percentile_high : float
        Upper percentile for color scaling
    symmetric_scale : bool
        If True, enforce symmetric limits around zero
    
    Returns
    -------
    tuple : (fig, ax)
    """
    discharge = data['discharge']
    km_positions = data['km_positions']
    times = data['times_datetime']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap (avoid smoothing by using pcolormesh)
    times_num = mdates.date2num(times)
    q_vals = discharge.values
    finite_vals = q_vals[np.isfinite(q_vals)]
    if finite_vals.size > 0:
        p_low, p_high = np.percentile(finite_vals, [percentile_low, percentile_high])
        if symmetric_scale:
            abs_lim = max(abs(p_low), abs(p_high))
            vmin, vmax = -abs_lim, abs_lim
        else:
            vmin, vmax = p_low, p_high
        norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = None
    im = ax.pcolormesh(km_positions, times_num, q_vals, cmap='RdBu_r', shading='auto', norm=norm)
    cbar = plt.colorbar(im, ax=ax, label='Discharge [m³/s]')
    
    ax.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time', fontsize=11, fontweight='bold')
    ax.set_title('Discharge Evolution: Space-Time Heatmap', fontsize=12, fontweight='bold')
    ax.yaxis_date()
    locator = mdates.MonthLocator(interval=1)
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    
    if show_flood_limit:
        if flood_sign < 0:
            flood_mask = q_vals < 0
        else:
            flood_mask = q_vals > 0
        km_grid = np.broadcast_to(km_positions, q_vals.shape)
        flood_km = np.where(flood_mask, km_grid, np.nan)
        max_flood_km_per_time = np.nanmax(flood_km, axis=1)
        ax.plot(max_flood_km_per_time, times_num, color='black', linewidth=1.2, label='Flood limit')
        ax.legend(loc='upper right', fontsize=9)
    
    return fig, ax


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # --- SETTINGS ---
    his_file_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\02_seasonal\Tmorph_50years\MF50_sens.8778435\output\FlowFM_0000_his.nc")
    
    try:
        # ===== DATA LOADING =====
        print("Loading cross-section data...")
        data = load_cross_section_data(
            his_file_path, 
            estuary_only=True, 
            km_range=(20, 45),
            select_cycles_hydrodynamic=False,  # Use full time series (all cycles)
            n_periods=3,                       # Get cycles from start/middle/end
            select_max_flood=True,             # Select max flood penetration
            flood_sign=-1,                     # Negative discharge = flood
            exclude_last_timestep=True
        )
        
        heatmap_data = None
        full_data = None
        if data.get('selection_mode') == 'max_flood':
            heatmap_data = load_cross_section_data(
                his_file_path,
                estuary_only=True,
                km_range=(20, 45),
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=-1,
                select_max_flood_per_cycle=True,
                exclude_last_timestep=True
            )
            full_data = load_cross_section_data(
                his_file_path,
                estuary_only=True,
                km_range=(20, 45),
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=-1,
                select_max_flood_per_cycle=False,
                exclude_last_timestep=True
            )
        print(f"✓ Selected {data['n_timesteps']} timesteps from {data['n_timesteps_original']} total")
        print(f"✓ Found {len(data['km_positions'])} cross-sections")
        print(f"✓ KM range: {data['km_positions'].min():.1f} to {data['km_positions'].max():.1f} km")
        print()
        
        # ===== PLOTTING =====
        print("Creating visualizations...")
        
        if data.get('selection_mode') == 'max_flood':
            # Plot: Max flood profile
            print("  - Max flood longitudinal profile...")
            fig1, ax1 = plot_max_flood_profile(data)
            plt.tight_layout()
            plt.show()

            if full_data is not None:
                print("  - Max flood profiles (representative periods)...")
                rep_indices = select_max_flood_indices_by_period(
                    full_data['times'],
                    full_data['discharge'],
                    full_data['km_positions'],
                    n_periods=3,
                    flood_sign=-1
                )
                if len(rep_indices) > 0:
                    fig2, ax2 = plot_multiple_max_flood_profiles(full_data, rep_indices)
                    plt.tight_layout()
                    plt.show()

            if heatmap_data is not None:
                # Plot: Heatmap using max-flood moment per cycle
                print("  - Space-time heatmap (max flood per cycle)...")
                fig3, ax3 = plot_discharge_heatmap(heatmap_data)
                plt.tight_layout()
                plt.show()
        else:
            # Plot 1: Statistics
            print("  - Discharge statistics...")
            fig1, axes1 = plot_discharge_statistics(data)
            plt.tight_layout()
            plt.show()

            # Plot 2: Upstream inflow (most upstream cross-section)
            print("  - Upstream inflow time series...")
            fig2, ax2 = plot_upstream_inflow_timeseries(data)
            plt.tight_layout()
            plt.show()
            
            # Plot 3: Heatmap
            print("  - Space-time heatmap...")
            fig3, ax3 = plot_discharge_heatmap(data)
            plt.tight_layout()
            plt.show()
        
        # Close dataset
        data['ds'].close()
        print("\n✓ Done!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()