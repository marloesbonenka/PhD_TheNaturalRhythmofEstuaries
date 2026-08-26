#%% IMPORTS
import matplotlib.pyplot as plt
import dfm_tools as dfmt
import numpy as np
from pathlib import Path
from matplotlib.ticker import FuncFormatter
import cmocean

#%% --- CONFIGURATION ---
model_folder = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783\output")

# Assuming standard map file naming pattern
map_file_pattern = str(model_folder / "*_map.nc")
TIMESTEP_INDEX = 0  # -1 for the last, 0 for the first

# --- EXECUTION ---
# 1. Load data
ds = dfmt.open_partitioned_dataset(map_file_pattern)

# 2. Select variable and timestep
data_to_plot = ds['mesh2d_mor_bl'].isel(time=TIMESTEP_INDEX)

# 3. Plotting
fig, ax = plt.subplots(figsize=(170 / 25.4, 6))

# Plot using xugrid
pc = data_to_plot.ugrid.plot(
    ax=ax,
    cmap=cmocean.cm.delta,
    vmin=-10, vmax=10,
    add_colorbar=True,
    edgecolors='none',
    cbar_kwargs={'label': 'bed level [m]'}
)

ax.set_aspect('equal')
_km_formatter = FuncFormatter(lambda val, pos: f'{val / 1000:.0f}')
ax.xaxis.set_major_formatter(_km_formatter)
ax.yaxis.set_major_formatter(_km_formatter)
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_title(
    f"bed level [m] snapshot \u2014 "
    f"{np.datetime_as_string(data_to_plot.time.values, unit='D')}",
    fontsize=8,
)

plt.show()

# Clean up
ds.close()