#%% IMPORTS
import matplotlib.pyplot as plt
import dfm_tools as dfmt
import numpy as np
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

#%% --- CONFIGURATION ---
model_folder = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783\output")

# Assuming standard map file naming pattern
map_file_pattern = str(model_folder / "*_map.nc")
TIMESTEP_INDEX = -1  # -1 for the last, 0 for the first

#%%
# --- CUSTOM COLORMAP ---
def create_terrain_colormap():
    colors = [
        (0.00, "#000066"), (0.10, "#0000ff"), (0.30, "#00ffff"),
        (0.40, "#00ffff"), (0.50, "#fcfcfc"), (0.60, "#f3df91"),
        (0.75, "#ffd000"), (0.90, "#228B22"), (1.00, "#006400"),
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

# --- EXECUTION ---
# 1. Load data
ds = dfmt.open_partitioned_dataset(map_file_pattern)

# 2. Select variable and timestep
data_to_plot = ds['mesh2d_mor_bl'].isel(time=TIMESTEP_INDEX)

# 3. Plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot using xugrid
pc = data_to_plot.ugrid.plot(
    ax=ax,
    cmap=create_terrain_colormap(),
    vmin=-10, vmax=10,
    add_colorbar=True,
    cbar_kwargs={'label': 'Bed Level [m]'}
)

ax.set_aspect('equal')
ax.set_title(f"Bed Level at {np.datetime_as_string(data_to_plot.time.values, unit='D')}")

plt.show()

# Clean up
ds.close()