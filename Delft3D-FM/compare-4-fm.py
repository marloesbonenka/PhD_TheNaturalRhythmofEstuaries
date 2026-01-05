import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import dfm_tools as dfmt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

# --- 1. HARDCODED PATHS ---
# Delft3D-FM Path (using your folder structure)
path_fm_dir = r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_FourRiverBoundaries\01_constant_Qr500_T2m_FourRiverBoundaries"
file_pattern_fm = os.path.join(path_fm_dir, 'output', '*_map.nc')
path_d3d4 = r"U:\PhDNaturalRhythmEstuaries\Models\0_RiverDischargeVariability_domain45x15_D3D4\05_RiverDischargeVariability_domain45x15\s2_500_Wup_300m\01_baserun500\VM_output\trim-varriver_tidewest.nc" # Standard NetCDF export from D3D4

# --- 2. CUSTOM COLORMAPS ---
def create_terrain_colormap():
    colors = [(0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
              (0.40, "#00ffff"), (0.50, "#ffffcc"), (0.60, "#ffcc00"),
              (0.75, "#cc6600"), (0.90, "#228B22"), (1.00, "#006400")]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

terrain_cmap = create_terrain_colormap()
diff_cmap = plt.get_cmap('RdBu_r')

# --- 3. LOADING DATA ---
print("Loading Delft3D-FM dataset...")
# Load exactly how you did in your batch script
ds_fm = dfmt.open_partitioned_dataset(file_pattern_fm)

# --- FIX: Handle duplicate times manually if they exist ---
if 'time' in ds_fm.dims:
    _, index = np.unique(ds_fm['time'], return_index=True)
    ds_fm = ds_fm.isel(time=index)

print("Loading Delft3D-4 dataset...")
ds_4 = xr.open_dataset(path_d3d4)

# --- 4. DATA EXTRACTION ---
var_name_fm = 'mesh2d_mor_bl' # Back to your original variable
bl_fm_raw = ds_fm[var_name_fm].isel(time=-1)
x_fm = ds_fm['mesh2d_face_x'].values
y_fm = ds_fm['mesh2d_face_y'].values

# D3D4: Extract coordinates and Bed Level (Elevation = -Depth)
x_4 = ds_4['XCOR'].values 
y_4 = ds_4['YCOR'].values
bl_4 = -ds_4['DPS'].isel(time=-1).values

# --- 5. INTERPOLATION FOR DIFFERENCE ---
print("Interpolating FM to D3D4 grid...")
points_fm = np.vstack((x_fm, y_fm)).T
values_fm = bl_fm_raw.values
# Interpolate FM onto D3D4 structured grid
bl_fm_interp = griddata(points_fm, values_fm, (x_4, y_4), method='linear')

diff = bl_4 - bl_fm_interp

# --- 6. PLOTTING ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True, sharey=True)

# Shared color limits
norm_terrain = plt.Normalize(vmin=-15, vmax=13)
norm_diff = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)

# Plot 1: D3D4
im1 = ax1.pcolormesh(x_4/1000, y_4/1000, bl_4, cmap=terrain_cmap, norm=norm_terrain, shading='auto')
ax1.set_title("Final Bed Level: Delft3D-4")

# Plot 2: FM (Using your ugrid plot logic)
pc = bl_fm_raw.ugrid.plot(ax=ax2, cmap=terrain_cmap, norm=norm_terrain, add_colorbar=False)
ax2.set_title("Final Bed Level: Delft3D-FM")

# Plot 3: Difference
im3 = ax3.pcolormesh(x_4/1000, y_4/1000, diff, cmap=diff_cmap, norm=norm_diff, shading='auto')
ax3.set_title("Difference (D3D4 - FM)")

# Add colorbars
for im, ax, lbl in zip([im1, pc, im3], [ax1, ax2, ax3], ['Bed Level [m]', 'Bed Level [m]', 'Diff [m]']):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax, label=lbl)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()

ds_fm.close()
ds_4.close()