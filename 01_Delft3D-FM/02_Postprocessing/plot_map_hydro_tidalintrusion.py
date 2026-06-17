"""Extract the x-location of the limit of tidal intrusion (LTI) over time,
for each scenario, using cross-sectionally averaged SIGNED velocity and a
rolling time window.

The signal that actually separates tidal-influenced from purely-fluvial
reaches is the TEMPORAL VARIATION at each x over a tidal cycle:
  - Where tidal influence reaches: cross-sectional mean velocity Ubar(x,t)
    oscillates between flood (+, toward river) and ebb (-, toward sea) as
    the tide goes in and out -> Ubar swings positive at some point in the
    cycle.
  - Where tidal influence does NOT reach: flow is dominated by river
    discharge, which is one-directional (-x, toward the sea) and roughly
    steady -> Ubar stays negative (or near zero) throughout the cycle,
    never swinging positive.

So: for each x, take the windowed (e.g. 12-hour) MAXIMUM of Ubar(x,t).
Where that windowed max is still clearly positive (flood reaches there),
you're inside the tidally-influenced zone. Where it drops to ~0 or stays
negative, the flood signal has died out -- that x is (an estimate of) the
limit of tidal intrusion for that window.

Rolling-window extreme of the cross-sectionally averaged velocity vs x, 
then read off where it crosses zero (or a small threshold) moving landward.
"""

# %% Imports
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- CONFIGURATION ---

SCENARIOS = {
    'constant':   Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low_flow':   Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'peak_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    'mean_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
}

VAR_NAME  = 'mesh2d_ucx'
LOAD_VARS = [VAR_NAME]

FLOOD_VEL_THRESHOLD = 0.01   # [m/s]

N_DAYS = 1

# Spatial binning along x (independent of grid resolution / channel narrowing)
CACHE_BBOX   = [1, 1, 45000, 15000]
CACHE_TAG    = None
X_BIN_WIDTH  = 100      # [m]
X_MIN, X_MAX = 19000, 45000

# Output
OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\max_tidal_intrusion")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% --- HELPERS ---

def get_last_n_days_window(ds, n_days):
    """Return (t_start, t_end) as numpy datetime64 covering the last n_days
    of the dataset's time coordinate."""
    t_end   = ds.time.values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    return t_start, t_end

def compute_xprofile_mean(data_t, face_x, var_name, x_edges):
    """Cross-sectional MEAN of the signed variable at each x-bin, for one
    timestep. Mean (not max) is the right collapse here because we want
    the net cross-sectional tendency (flood vs ebb)."""

    vals = data_t[var_name].values   # signed, do NOT take abs()
    bin_idx = np.digitize(face_x, x_edges) - 1
    n_bins = len(x_edges) - 1
    profile = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            profile[b] = np.nanmean(vals[mask])
    return profile


def build_ubar_matrix(ds, face_x, var_name, x_edges):
    """Build the full Ubar(x, t) matrix: cross-sectional mean signed
    velocity at every x-bin, for every timestep. Shape: (n_time, n_xbins)."""
    n_time = len(ds.time)
    n_bins = len(x_edges) - 1
    Ubar = np.full((n_time, n_bins), np.nan)
    for idx in range(n_time):
        data_t = ds.isel(time=idx)
        Ubar[idx, :] = compute_xprofile_mean(data_t, face_x, var_name, x_edges)
    return Ubar

def front_from_profile(profile_1d, x_centers, threshold):
    """Given a 1D profile of cross-sectionally averaged velocity, 
    return the x-location of the limit of tidal intrusion
    as the first x where the profile drops below the threshold moving
    landward (increasing x)."""

    active = profile_1d > threshold
    if not active[0]:
        return np.nan   # no flood signal even at the mouth -- shouldn't happen
    for i in range(len(profile_1d)):
        if not active[i]:
            return x_centers[i]
    return x_centers[-1]   # flood signal persists across the whole window (edge case)



# %% --- MAIN LOOP ---

x_edges   = np.arange(X_MIN, X_MAX + X_BIN_WIDTH, X_BIN_WIDTH)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

results = {}   # label -> dict with 'time', 'front_x' (per-window LTI estimate)

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}\nScenario: {label}\n{'='*60}")

    base_path   = folder_path.parent
    folder_name = folder_path.name
    cache_dir   = base_path / 'cached_data'
    cache_dir.mkdir(parents=True, exist_ok=True)

    run_paths = get_stitched_map_run_paths(
        base_path=base_path, folder_name=folder_name,
        timed_out_dir=None, variability_map=None, analyze_noisy=False,
    ) or [folder_path]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=cache_dir, folder_name=folder_name, run_paths=run_paths,
        var_names=LOAD_VARS, bbox=CACHE_BBOX,
        append_time=True, append_vars=True, cache_tag=cache_tag,
    )

    if ds is None or 'time' not in ds.dims or VAR_NAME not in ds:
        print(f"  [SKIP] missing data for {label}")
        continue

    # --- Restrict to the last N_DAYS only ---
    t_start, t_end = get_last_n_days_window(ds, N_DAYS)
    ds = ds.sel(time=slice(t_start, t_end))
    print(f"  Using last {N_DAYS} day(s): {t_start} -> {t_end}  ({len(ds.time)} timesteps)")

    if len(ds.time) == 0:
        print(f"  [SKIP] no timesteps remain after slicing to last {N_DAYS} days")
        continue

    face_x = ds.grid.face_coordinates[:, 0]
    time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
    # --- Build Ubar(x, t) ---
    Ubar = build_ubar_matrix(ds, face_x, VAR_NAME, x_edges)   # (n_time, n_xbins)

    fronts_instant = np.full(len(time_values), np.nan)
    for idx in range(len(time_values)):
        fronts_instant[idx] = front_from_profile(Ubar[idx, :], x_centers, FLOOD_VEL_THRESHOLD)

    df = pd.DataFrame({'time': time_values, 'front_x_instant': fronts_instant})
    results[label] = df
    df.to_csv(OUTPUT_DIR / f"front_x_{label}.csv", index=False)

    ds.close()
    valid = fronts_instant[~np.isnan(fronts_instant)]
    if len(valid):
        print(f"  Done. front_x range: {valid.min():.0f} - {valid.max():.0f} m")
    else:
        print("  Done. No valid front_x found -- check FLOOD_VEL_THRESHOLD / WINDOW_TIMESTEPS.")

# %% --- SUMMARY: limit of tidal intrusion per scenario ---

summary_rows = []

for label, df in results.items():

    lti_instant_max = df['front_x_instant'].max()

    folder_path = SCENARIOS[label]

    match = re.search(r'pm(\d+(?:\.\d+)?)', folder_path.name)

    if match:
        peak_amplitude = float(match.group(1))
    else:
        peak_amplitude = np.nan

    summary_rows.append({
        'scenario': label,
        'peak_amplitude': peak_amplitude,
        'LTI_instant_max': lti_instant_max,
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "LTI_summary.csv", index=False)
print("\n" + "="*60)
print(summary.to_string(index=False))
print("="*60)

# %% --- PLOT: front_x(t) per scenario, windowed vs instantaneous ---
fig, ax = plt.subplots(figsize=(6, 5))

ax.scatter(
    summary['LTI_instant_max'],
    summary['peak_amplitude'],
    s=100
)

for _, row in summary.iterrows():
    ax.annotate(
        row['scenario'],
        ( row['LTI_instant_max'], row['peak_amplitude']),
        xytext=(5, 5),
        textcoords='offset points'
    )

ax.set_ylabel('peak amplitude (pm)')
ax.set_xlabel('maximum tidal intrusion [m]')
ax.set_title('maximum tidal intrusion vs peak amplitude')

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "LTI_vs_peak_amplitude.png",
    dpi=300
)
plt.close()

# #%%
# fig, ax = plt.subplots(figsize=(10, 5))
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# for i, (label, df) in enumerate(results.items()):
#     c = colors[i % len(colors)]
#     ax.plot(peak_amplitude, summary_rows['LTI_instant_max'], label=f"{label} (instantaneous)",
#             marker='', alpha=0.5, color=c)
# ax.set_xlabel('time')
# ax.set_ylabel('limit of tidal intrusion [m]')
# ax.set_title(f'Tidal intrusion front over time, per scenario')
# ax.legend(fontsize=8)
# plt.tight_layout()
# plt.savefig(OUTPUT_DIR / "tidal_intrusion_front_timeseries.png", dpi=300)
# plt.close(fig)

print(f"\nOutputs written to: {OUTPUT_DIR.resolve()}")
# %%
