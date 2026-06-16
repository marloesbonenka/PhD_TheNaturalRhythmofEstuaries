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
import matplotlib.dates as mdates

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

PARAMETER_META = {
    'cross_section_discharge': {
        'ylabel':      'tidal discharge [m³/s]',
        'title':       'tidal discharge at estuary mouth — flood (+) / ebb (-)',
        'compute_prism': True,
        'prism_unit':  'Mm³',
        'prism_scale': 1e6,
        'subtract_river': True,
    },
    'cross_section_velocity': {
        'ylabel':      'cross-sectional velocity [m/s]',
        'title':       'tidal velocity at estuary mouth — flood (+) / ebb (-)',
        'compute_prism': False,
        'subtract_river': False,
    },
    'cross_section_sand': {
        'ylabel':      'sand transport [kg/s]',
        'title':       'sand transport at estuary mouth',
        'compute_prism': False,
        'subtract_river': False,
    },
}
#%%
# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_his_tidal_signal(his_paths, t_start, t_end, xs_keywords, upstream_xs_keywords,
                           parameter, subtract_river=True):
    ds_his       = xr.open_mfdataset(his_paths, coords="minimal", compat="override")
    ds_his_slice = ds_his.sel(time=slice(t_start, t_end))

    xs_names = [
        name.decode('utf-8').strip() if isinstance(name, bytes) else str(name).strip()
        for name in ds_his_slice['cross_section_name'].values
    ]

    try:
        xs_idx_mouth = next(i for i, name in enumerate(xs_names)
                            if any(k in name for k in xs_keywords))
    except StopIteration:
        print(f"  [WARNING] Mouth keywords not found. Defaulting to index 0.")
        xs_idx_mouth = 0

    xs_idx_river = None
    if subtract_river:
        try:
            xs_idx_river = next(i for i, name in enumerate(xs_names)
                                if any(k in name for k in upstream_xs_keywords))
        except StopIteration:
            print(f"  [WARNING] Upstream keywords not found. Falling back to mean subtraction.")

    print(f"  [XS] Mouth   : '{xs_names[xs_idx_mouth]}'")
    if xs_idx_river is not None:
        print(f"  [XS] Upstream: '{xs_names[xs_idx_river]}'")

    time_values  = ds_his_slice.time.values
    time_seconds = (time_values - time_values[0]) / np.timedelta64(1, 's')

    q_mouth = ds_his_slice[parameter].values[:, xs_idx_mouth]

    if subtract_river:
        q_river = (ds_his_slice[parameter].values[:, xs_idx_river]
                   if xs_idx_river is not None else np.mean(q_mouth))
        signal = -(q_mouth - q_river)
    else:
        signal = q_mouth

    ds_his.close()
    return time_seconds, signal, time_values

def get_last_n_days_window(his_paths, n_days):
    """Return (t_start, t_end) covering the last n_days of the HIS dataset."""
    ds = xr.open_mfdataset(his_paths, coords="minimal", compat="override")
    t_end   = ds.time.values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    ds.close()
    return t_start, t_end
#%%
meta = PARAMETER_META[PARAMETER_TO_ANALYZE]
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

    t_start, t_end = get_last_n_days_window(his_paths, PLOT_WINDOW_DAYS)
    print(f"  [INFO] Window: {t_start} → {t_end}")

    time_s, signal, time_dt = load_his_tidal_signal(
        his_paths, t_start, t_end,
        MOUTH_CROSS_SECTION_KEYWORDS, UPSTREAM_CROSS_SECTION_KEYWORDS,
        parameter=PARAMETER_TO_ANALYZE,
        subtract_river=meta['subtract_river'],
    )

    # --- Tidal prism (discharge only) ---
    legend_suffix = ''
    if meta['compute_prism']:
        dt              = np.median(np.diff(time_s))
        prism           = np.sum(np.abs(signal)) * dt / 2 / meta['prism_scale']
        prism_results[label] = prism
        legend_suffix   = f"  (prism: {prism:.2f} {meta['prism_unit']})"
        print(f"  [RESULT] Tidal prism: {prism:.3f} {meta['prism_unit']}")

    color = SCENARIO_COLORS.get(label, 'black')
    ax.fill_between(time_dt, signal, 0, where=(signal >= 0), color=color, alpha=0.25, linewidth=0)
    ax.fill_between(time_dt, signal, 0, where=(signal < 0),  color=color, alpha=0.25, linewidth=0)
    ax.plot(time_dt, signal, color=color, lw=1.8, label=f"{label}{legend_suffix}")

# ---------------------------------------------------------------------------
# PRINT SUMMARY & FINALISE PLOT
# ---------------------------------------------------------------------------
if meta['compute_prism']:
    print(f"\n{'='*45}")
    print(f" TIDAL PRISM RESULTS  [{meta['prism_unit']}]")
    print(f"{'='*45}")
    for scenario, prism in prism_results.items():
        print(f"  {scenario.ljust(12)} : {prism:.3f} {meta['prism_unit']}")
    print('='*45)


ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel('time [UTC]')
ax.set_ylabel(meta['ylabel'])
ax.set_title(meta['title'], fontweight='bold')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b %H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
# fig.autofmt_xdate(rotation=90)
ax.legend(title='scenario', frameon=True, fontsize=9)
ax.grid(True, lw=0.4, alpha=0.5)
fig.tight_layout()
plt.show()
# %%