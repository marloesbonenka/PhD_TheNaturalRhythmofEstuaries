"""
Compute the along-estuary tidal prism through the mouth (km 20) for the 
using cross-sectional discharge integration method.

Diagnostic plot shows raw tidal discharge (flood/ebb) over two cycles.
"""

# %% Imports
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_loaddata import get_stitched_his_paths
# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'mean flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
    'peak flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
}

# --- HIS Settings ---
MOUTH_CROSS_SECTION_KEYWORDS = ['20', 'mouth']
UPSTREAM_CROSS_SECTION_KEYWORDS = ['44', 'upstream']
PLOT_WINDOW_DAYS = 1
PARAMETER_TO_ANALYZE = 'cross_section_discharge'

SCENARIO_COLORS = {
    'constant':  '#888888',
    'low flow':  '#92C5DE',
    'mean flow': '#4393C3',
    'peak flow': '#084594',
}
#%%
# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_his_tidal_discharge(his_paths, t_start, t_end, xs_keywords, upstream_xs_keywords):
    """
    Open HIS dataset, slice to [t_start, t_end], locate the mouth cross-section,
    and return time_seconds (relative to t_start) and q_tidal (mean-removed discharge).
    """
    ds_his       = xr.open_mfdataset(his_paths, coords="minimal", compat="override")
    ds_his_slice = ds_his.sel(time=slice(t_start, t_end))

    xs_names = [
        name.decode('utf-8').strip() if isinstance(name, bytes) else str(name).strip()
        for name in ds_his_slice['cross_section_name'].values
    ]
    # --- Mouth cross-section (km 20) ---
    try:
        xs_idx_mouth = next(
            i for i, name in enumerate(xs_names)
            if any(k in name for k in xs_keywords)
        )
    except StopIteration:
        print(f"  [WARNING] Mouth keywords {xs_keywords} not found. Defaulting to index 0.")
        xs_idx_mouth = 0

    # --- Upstream cross-section (km 44) ---
    try:
        xs_idx_river = next(
            i for i, name in enumerate(xs_names)
            if any(k in name for k in upstream_xs_keywords)
        )
    except StopIteration:
        print(f"  [WARNING] Upstream keywords {upstream_xs_keywords} not found. Falling back to mean subtraction.")
        xs_idx_river = None

    print(f"  [XS] Mouth   : '{xs_names[xs_idx_mouth]}'")
    if xs_idx_river is not None:
        print(f"  [XS] Upstream: '{xs_names[xs_idx_river]}'")
    else:
        print(f"  [XS] Upstream: not found — using mean subtraction as fallback.")

    # --- Extract discharge arrays ---
    time_values  = ds_his_slice.time.values
    time_seconds = (time_values - time_values[0]) / np.timedelta64(1, 's')

    q_mouth = ds_his_slice['cross_section_discharge'].values[:, xs_idx_mouth]

    if xs_idx_river is not None:
        q_river = ds_his_slice['cross_section_discharge'].values[:, xs_idx_river]
    else:
        q_river = np.mean(q_mouth)

    q_tidal = -(q_mouth - q_river)

    ds_his.close()
    return time_seconds, q_tidal, time_values

def get_last_n_days_window(his_paths, n_days):
    """Return (t_start, t_end) covering the last n_days of the HIS dataset."""
    ds = xr.open_mfdataset(his_paths, coords="minimal", compat="override")
    t_end   = ds.time.values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    ds.close()
    return t_start, t_end
#%%
prism_results = {}
fig, ax = plt.subplots(figsize=(9, 4.5))

for label, folder_path in SCENARIOS.items():
    print(f"\nProcessing Scenario: {label}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    his_paths = get_stitched_his_paths(
        base_path=base_path, folder_name=folder_name,
        timed_out_dir=None, variability_map=None, analyze_noisy=False
    )
    if not his_paths:
        his_paths = list(folder_path.glob("*_his.nc"))
    if not his_paths:
        print("  [SKIP] No HIS data found.")
        continue
    
    # Tidal prism 
    t_prism_start, t_prism_end = get_last_n_days_window(his_paths, PLOT_WINDOW_DAYS)
    print(f"  [INFO] Prism cycle window:  {t_prism_start} → {t_prism_end}")

    time_s_prism, q_tidal_prism, time_values_prism = load_his_tidal_discharge(
        his_paths, t_prism_start, t_prism_end,
        MOUTH_CROSS_SECTION_KEYWORDS, UPSTREAM_CROSS_SECTION_KEYWORDS
    )
    dt = np.median(np.diff(time_s_prism))

    total_volume_m3 = np.sum(np.abs(q_tidal_prism)) * dt
    tidal_prism_m3  = total_volume_m3 / 2
    tidal_prism_Mm3 = tidal_prism_m3 / 1e6

    prism_results[label] = tidal_prism_Mm3
    print(f"  [RESULT] Tidal prism: {tidal_prism_Mm3:.3f} Mm³")

    color       = SCENARIO_COLORS.get(label, 'black')

    ax.fill_between(time_values_prism, q_tidal_prism, 0,
                    where=(q_tidal_prism >= 0),
                    color=color, alpha=0.25, linewidth=0)
    ax.fill_between(time_values_prism, q_tidal_prism, 0,
                    where=(q_tidal_prism < 0),
                    color=color, alpha=0.25, linewidth=0)
    ax.plot(time_values_prism, q_tidal_prism,
            color=color, lw=1.8,
            label=f"{label}  (prism: {tidal_prism_Mm3:.2f} Mm³)")

# ---------------------------------------------------------------------------
# PRINT SUMMARY & FINALISE PLOT
# ---------------------------------------------------------------------------
print(f"\n{'='*45}")
print(" TIDAL PRISM RESULTS (FINAL CYCLE)")
print(f"{'='*45}")
for scenario, prism in prism_results.items():
    print(f"  {scenario.ljust(12)} : {prism:.3f} Mm³")
print('='*45)

ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('time since start of two-cycle window [hours]')
ax.set_ylabel('tidal discharge [m³/s]')
ax.set_title('tidal discharge at estuary mouth — flood (+) / ebb (-)', fontweight='bold')
ax.legend(title='scenario', frameon=True, fontsize=9)
ax.grid(True, lw=0.4, alpha=0.5)
fig.tight_layout()
plt.show()
# %%