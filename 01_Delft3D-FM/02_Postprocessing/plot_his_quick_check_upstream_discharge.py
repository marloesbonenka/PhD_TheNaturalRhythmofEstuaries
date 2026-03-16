"""Quick isolated check for upstream discharge from one HIS file.

This script does not use or write cache files.
"""
#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from FUNCTIONS.F_loaddata import load_cross_section_data


HIS_FILE = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q1000\01_baserun1000\output\FlowFM_0000_his.nc"
    #r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q1000\02_run1000_seasonal\output\FlowFM_0000_his.nc"
    #r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\TestingBoundaries_and_SensitivityAnalyses\Test_OneRiverBoundary\01_constant_Qr1000_T2m_D3D4comparison\output\FlowFM_0000_his.nc"
)
# List of river kms to plot
RIVER_KMS = [30, 32, 33, 35, 40, 42, 44, 45]
DISCHARGE_VAR = "cross_section_discharge"
#%%
if not HIS_FILE.exists():
    raise FileNotFoundError(f"HIS file not found: {HIS_FILE}")

data = load_cross_section_data(
    his_file_path=HIS_FILE,
    q_var=DISCHARGE_VAR,
    estuary_only=True,
    km_range=(20, 45),
    select_cycles_hydrodynamic=False,
)
#%%
km_positions = np.asarray(data["km_positions"])
discharge = data[DISCHARGE_VAR].values
time = data["t"]

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(RIVER_KMS)))

for i, river_km in enumerate(RIVER_KMS):
    idx_river = int(np.argmin(np.abs(km_positions - river_km)))
    km_river = float(km_positions[idx_river])
    q_river = discharge[:, idx_river]
    # Plot full time series for each river km
    ax.plot(time, q_river, lw=1.0, color=colors[i], label=f"{km_river:.2f} km")
    print(f"Closest cross-section to {river_km} km: {km_river:.2f} km")
    print(
        f"Discharge stats at {km_river:.2f} km (min/max/mean): "
        f"{np.nanmin(q_river):.3f}, {np.nanmax(q_river):.3f}, {np.nanmean(q_river):.3f} m3/s"
    )

ax.set_xlabel("Time")
ax.set_ylabel("Discharge (m3/s)")
ax.set_title("Upstream discharge at selected river kms")
ax.grid(alpha=0.3)
ax.legend(title="River km")
fig.tight_layout()
plt.show()

if data.get("ds") is not None:
    data["ds"].close()

# %%
