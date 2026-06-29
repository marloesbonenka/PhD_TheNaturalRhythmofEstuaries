"""Extract the x-location of the limit of tidal intrusion (LTI) over time,
for each scenario, using cross-sectionally averaged SIGNED velocity and a
rolling time window -- across all three discharge magnitudes
(Q = 250 / 500 / 1000 m3/s), restricted to runs 01, 06, 09, 10, 11.

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

Scenario folders are discovered automatically per discharge. Expected
folder name pattern: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_mean].{runid}
e.g.  dhr_01_Qr500_pm1_n0.9724783
      dhr_06_Qr250_pm4_n3_mean.10280150
Only run_id in {01, 06, 09, 10, 11} are included -- this deliberately
skips other one-off runs sitting in the same detailed-hydro-run folder
(e.g. the Q500 dhr_12_..._lowflow / _peakflow / _meanflow set).
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
NORMALIZE = True

BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

DISCHARGES = [250, 500, 1000]

# Only these run-IDs belong to the pm/n sensitivity matrix we want here.
RUN_IDS_TO_INCLUDE = {1, 6, 9, 10, 11}

# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}.{runid}  or  ..._n{n}_mean.{runid}
_FOLDER_RE = re.compile(r'^dhr_(\d{2})_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\d+$')

VAR_NAME  = 'mesh2d_ucx'
LOAD_VARS = [VAR_NAME]

FLOOD_VEL_THRESHOLD = 0.00001   # [m/s]

N_DAYS = 1

# Spatial binning along x (independent of grid resolution / channel narrowing)
CACHE_BBOX   = [1, 1, 45000, 15000]
CACHE_TAG    = None
X_BIN_WIDTH  = 100      # [m]
X_MIN, X_MAX = 19000, 45000

# Colour per discharge value (same palette as the min-depth plot)
DISCHARGE_COLORS = {250: '#3B6064', 500: '#87BBA2', 1000: '#C9E4CA'}

# Output (combined across discharges, so not nested under a single Q folder)
OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'max_tidal_intrusion'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Figure style ---
STYLE = 'default'   # 'default'   -> white background, black text
                     # 'bluefig'   -> transparent figure, dark-teal text/axes
                     # 'whitefig'  -> transparent figure, white axes/text

STYLES = {
    'default': {},
    'whitefig': {
        'figure.facecolor':    'none',
        'axes.facecolor':      'white',
        'axes.edgecolor':      'white',
        'axes.labelcolor':     'white',
        'xtick.color':         'white',
        'ytick.color':         'white',
        'text.color':          'white',
        'grid.color':          '#cccccc',
        'legend.facecolor':    'none',
        'legend.edgecolor':    'white',
        'savefig.transparent': False,
    },
}

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])
_tc = plt.rcParams['text.color']
_tr = plt.rcParams.get('savefig.transparent', False)

FONTSIZE_TITLE  = 18
FONTSIZE_LABELS = FONTSIZE_TITLE - 4
FONTSIZE_TICKS  = FONTSIZE_LABELS - 2

# %% --- HELPERS ---

def discover_scenario_folders(dhr_base, discharge):
    """Find dhr_XX_Qr{discharge}_pm{pm}_n{n}[_mean].{runid} folders inside
    a detailed-hydro-run directory, restricted to RUN_IDS_TO_INCLUDE.
    Returns list of (folder_path, run_id, pm_val, n_val)."""
    results = []
    if not dhr_base.exists():
        return results
    for folder in sorted(dhr_base.iterdir()):
        if not folder.is_dir():
            continue
        m = _FOLDER_RE.match(folder.name)
        if not m:
            continue
        run_id, q_val, pm_val, n_val = (int(m.group(1)), int(m.group(2)),
                                         int(m.group(3)), int(m.group(4)))
        if run_id not in RUN_IDS_TO_INCLUDE or q_val != discharge:
            continue
        results.append((folder, run_id, pm_val, n_val))
    return results


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
    """Given a 1D profile of cross-sectionally averaged velocity, return
    the x-location of the limit of tidal intrusion as the first x where
    the profile drops below the threshold moving landward (increasing x)."""
    active = profile_1d > threshold
    if not active[0]:
        return np.nan   # no flood signal even at the mouth -- shouldn't happen
    for i in range(len(profile_1d)):
        if not active[i]:
            return x_centers[i]
    return x_centers[-1]   # flood signal persists across the whole window (edge case)


# %% --- MAIN LOOP: compute LTI per scenario, across all discharges ---

x_edges   = np.arange(X_MIN, X_MAX + X_BIN_WIDTH, X_BIN_WIDTH)
x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

results  = {}   # label -> DataFrame with 'time', 'front_x_instant'
datadict = {}   # label -> dict with discharge, run_id, pm, n, LTI_instant_max

for discharge in DISCHARGES:
    dhr_base = BASE_DIR / f"Q{discharge}" / "detailed-hydro-run"
    scenario_folders = discover_scenario_folders(dhr_base, discharge)

    if not scenario_folders:
        print(f"[SKIP] No matching scenario folders found in: {dhr_base}")
        continue

    print(f"\n{'='*60}\nQ = {discharge} m3/s: found {len(scenario_folders)} scenario(s)\n{'='*60}")

    cache_dir = dhr_base / 'cached_data'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)

    for folder_path, run_id, pm_val, n_val in scenario_folders:
        label = f"Qr{discharge}_dhr{run_id:02d}_pm{pm_val}_n{n_val}"
        print(f"  Processing: {folder_path.name}  ->  {label}")

        run_paths = get_stitched_map_run_paths(
            base_path=dhr_base, folder_name=folder_path.name,
            timed_out_dir=None, variability_map=None, analyze_noisy=False,
        ) or [folder_path]

        ds = load_or_update_map_cache_multi(
            cache_dir=cache_dir, folder_name=folder_path.name, run_paths=run_paths,
            var_names=LOAD_VARS, bbox=CACHE_BBOX,
            append_time=True, append_vars=True, cache_tag=cache_tag,
        )

        if ds is None or 'time' not in ds.dims or VAR_NAME not in ds:
            print(f"    [SKIP] missing data for {label}")
            continue

        # --- Restrict to the last N_DAYS only ---
        t_start, t_end = get_last_n_days_window(ds, N_DAYS)
        ds = ds.sel(time=slice(t_start, t_end))

        if len(ds.time) == 0:
            print(f"    [SKIP] no timesteps remain after slicing to last {N_DAYS} days")
            ds.close()
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
        if len(valid) == 0:
            print("    Done. No valid front_x found -- check FLOOD_VEL_THRESHOLD.")
            continue

        lti_instant_max = float(valid.max())
        print(f"    Done. LTI_instant_max = {lti_instant_max:.0f} m  ({lti_instant_max/1000:.2f} km)")

        datadict[label] = {
            'label':              label,
            'discharge':          discharge,
            'run_id':             run_id,
            'pm':                 pm_val,
            'n':                  n_val,
            'LTI_instant_max_m':  lti_instant_max,
            'LTI_instant_max_km': lti_instant_max / 1000,
        }

print(f"\nCollected LTI results for {len(datadict)} runs.")

# %% --- SUMMARY ---

summary = pd.DataFrame(datadict.values())
summary.to_csv(OUTPUT_DIR / "LTI_summary.csv", index=False)
print("\n" + "="*60)
print(summary.to_string(index=False))
print("="*60)

# %% --- PLOT: max tidal intrusion location vs peak amplitude, by discharge ---
if not datadict:
    print("No data to plot.")
else:
    present_discharges = sorted({v['discharge'] for v in datadict.values()})

    fig, ax = plt.subplots(figsize=(10, 6))

    for discharge in present_discharges:
        color  = DISCHARGE_COLORS.get(discharge, 'tab:gray')
        subset = {k: v for k, v in datadict.items() if v['discharge'] == discharge}

        pm_arr  = np.array([v['pm']                  for v in subset.values()], dtype=float)
        lti_km  = np.array([v['LTI_instant_max_km']  for v in subset.values()], dtype=float)
        if NORMALIZE:
            lti_km_n0 = np.array([v['LTI_instant_max_km']  for v in subset.values() if v['n'] == 0], dtype=float)
            lti_km_normalized = lti_km / lti_km_n0 if len(lti_km_n0) > 0 else lti_km
        else:
            lti_km_normalized = lti_km
        n_arr   = [v['n'] for v in subset.values()]
        label_q = f'Q = {discharge} m\u00b3/s'

        ax.scatter(lti_km_normalized, pm_arr, color=color, zorder=3, s=80, label=label_q)
        
        for x_, y_, n_ in zip(lti_km_normalized, pm_arr, n_arr):
            ax.annotate(f'n{n_}', (x_, y_), textcoords='offset points',
                        xytext=(4, 4), fontsize=FONTSIZE_TICKS, color=color)

    ax.set_ylabel('discharge amplitude $R_{\\mathrm{peak}}$', fontsize=FONTSIZE_LABELS)
    # ax.set_title('limit of tidal intrusion vs peak amplitude', fontsize=FONTSIZE_TITLE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICKS)
    ax.legend(fontsize=FONTSIZE_TICKS, loc='center left', bbox_to_anchor=(1.0, 0.5))

    # fig.suptitle(
    #     f'limit of tidal intrusion  |  last {N_DAYS} day(s)  |  threshold = {FLOOD_VEL_THRESHOLD} m/s',
    #     fontsize=FONTSIZE_TITLE, color=_tc,
    # )
    fig.tight_layout()

    if NORMALIZE:
        ax.set_xlabel('normalized x-location of maximum tidal intrusion [km]', fontsize=FONTSIZE_LABELS)
        fname = 'LTI_vs_peak_amplitude_allQ_normalized'
    else:
        ax.set_xlabel('x-location of maximum tidal intrusion [km]', fontsize=FONTSIZE_LABELS)
        fname = 'LTI_vs_peak_amplitude_allQ'
        
    fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=200, bbox_inches='tight', transparent=_tr) if NORMALIZE else None
    fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight', transparent=_tr)
    plt.show()
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / f"{fname}.png"}')

print(f"\nOutputs written to: {OUTPUT_DIR.resolve()}")


# %%
