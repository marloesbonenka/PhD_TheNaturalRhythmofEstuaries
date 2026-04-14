""" """
#%%
import xarray as xr
import xugrid as xu
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# 1. SETTINGS
# =============================================================================
cache_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\...\cached_data")
run_name = "Your_Run_Name_Here"
T_tidal = 12 # Tidal period in hours

# =============================================================================
# 2. LOAD & PREPARE DATA
# =============================================================================
# Load cached velocity and shear stress
# Note: Ensure 'mesh2d_u1' was included in your extraction var_names
ds_u = xu.open_dataset(cache_dir / f"mapoutput_mesh2d_u1_{run_name}.nc")
ds_tau = xu.open_dataset(cache_dir / f"mapoutput_mesh2d_taus_{run_name}.nc")

# Merge them into one working dataset
ds = xu.merge([ds_u, ds_tau])

# Calculate the number of timesteps in one tidal cycle
dt_minutes = (ds.time.diff('time').median().values / np.timedelta64(1, 'm'))
window_size = int((T_tidal * 60) / dt_minutes)

print(f"Detected dt: {dt_minutes} min. Window size for {T_tidal}h: {window_size} steps.")

# =============================================================================
# 3. HYDRODYNAMIC DECOMPOSITION
# =============================================================================

# A. Extract U_river (The Residual/Mean Flow)
# We apply a rolling mean over exactly one tidal cycle
ds['u_river'] = ds['mesh2d_u1'].rolling(time=window_size, center=True).mean()

# B. Extract U_tide (The Tidal Amplitude)
# 1. Get the oscillating part
u_prime = ds['mesh2d_u1'] - ds['u_river']
# 2. Get the amplitude (peak velocity of the tide)
ds['u_tide_amp'] = u_prime.rolling(time=window_size, center=True).max()

# C. Extract Max Shear Stress (The Geomorphic Driver)
ds['tau_max'] = ds['mesh2d_taus'].rolling(time=window_size, center=True).max()

# =============================================================================
# 4. SPATIAL ANALYSIS (Width-Averaging)
# =============================================================================
# To plot a longitudinal profile, we average across the width (Y-axis)
# This assumes your X-axis is the main estuary axis
profile = ds.groupby(ds.mesh2d_face_x).mean() 

# =============================================================================
# 5. PLOTTING
# =============================================================================
# Select a specific time (e.g., during a peak discharge pulse)
t_idx = 100 # Adjust to your period of interest

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Panel 1: Fluvial Component
profile['u_river'].isel(time=t_idx).plot(ax=axes[0], color='blue', label='$U_{river}$')
axes[0].set_ylabel('Residual Velocity (m/s)')
axes[0].set_title(f'Hydrodynamic Decomposition - {run_name}')

# Panel 2: Tidal Component
profile['u_tide_amp'].isel(time=t_idx).plot(ax=axes[1], color='red', label='$U_{tide}$ Amp')
axes[1].set_ylabel('Tidal Amplitude (m/s)')

# Panel 3: Geomorphic Work
profile['tau_max'].isel(time=t_idx).plot(ax=axes[2], color='black')
axes[2].axhline(0.1, color='gray', linestyle='--', label='$\\tau_{cr}$') # Example threshold
axes[2].set_ylabel('Peak Shear Stress (Pa)')
axes[2].set_xlabel('Distance from Mouth (m)')

for ax in axes:
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()