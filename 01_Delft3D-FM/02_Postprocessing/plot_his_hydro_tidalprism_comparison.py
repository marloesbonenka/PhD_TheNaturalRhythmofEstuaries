"""Compute the tidal prism (Mm^3) through the estuary mouth for each (Q, pm, n)
scenario in the discharge-variability sensitivity matrix, across all three
discharge magnitudes (Q = 250 / 500 / 1000 m3/s), restricted to runs
01, 06, 09, 10, 11 -- then plot tidal prism as a function of discharge peak
amplitude (R_peak / pm), coloured by mean discharge.

This is the tidal-prism analogue of plot_intertidal_area_allQ.py: same
scenario discovery (regex-based folder matching, run_id filter, combined-
across-Q plotting convention) -- but the per-run metric is the tidal prism
computed from HIS cross-section discharge (compute_tidal_prism.py's method)
instead of the intertidal area from MAP wet/dry classification.

Tidal prism definition (unchanged from compute_tidal_prism.py):
  prism = sum(|Q_tidal|) * dt / 2,  scaled to Mm^3
where Q_tidal is the river-discharge-corrected cross-sectional discharge at
the estuary mouth (km 20: mouth signal minus km 44/upstream signal), taken
over the last TIDAL_PRISM_WINDOW_DAYS of each run.

Scenario folders are discovered automatically per discharge. Expected
folder name pattern: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}[_mean].{runid}
e.g.  dhr_01_Qr500_pm1_n0.9724783
      dhr_06_Qr250_pm4_n3_mean.10280150
Only run_id in {01, 06, 09, 10, 11} are included -- this deliberately
skips other one-off runs sitting in the same detailed-hydro-run folder
(e.g. the Q500 dhr_12_..._lowflow / _peakflow / _meanflow set).

Note: unlike compute_tidal_prism.py, this script does NOT render a per-
scenario flood/ebb diagnostic time series -- it only needs the scalar tidal
prism per scenario. If you want that diagnostic for a specific scenario,
run compute_tidal_prism.py directly on the relevant SCENARIOS dict.
"""

# %% Imports
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_loaddata import get_stitched_his_paths

# %% --- CONFIGURATION ---
NORMALIZE = False   # if True, also produce a version normalized to each discharge's n=0 (constant) run

BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

DISCHARGES = [250, 500, 1000]

# Only these run-IDs belong to the pm/n sensitivity matrix we want here.
RUN_IDS_TO_INCLUDE = {1, 6, 9, 10, 11}

# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}.{runid}  or  ..._n{n}_mean.{runid}
_FOLDER_RE = re.compile(r'^dhr_(\d{2})_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\d+$')

# --- HIS settings (tidal prism calc, unchanged from compute_tidal_prism.py) ---
MOUTH_CROSS_SECTION_KEYWORDS    = ['km20', 'mouth']
UPSTREAM_CROSS_SECTION_KEYWORDS = ['km44', 'upstream']
VARIABLE_TO_ANALYZE             = 'cross_section_discharge'
SOURCE                          = 'cross_section'
SUBTRACT_RIVER                  = True
NEGATE                          = True       # positive flood / negative ebb
TIDAL_PRISM_WINDOW_DAYS         = 1          # ~2 tidal cycles, taken from end of run
PRISM_UNIT                      = 'Mm\u00b3'
PRISM_SCALE                     = 1e6

# Colour per discharge value (same palette as the LTI / intertidal-area plots)
DISCHARGE_COLORS = {250: '#3B6064', 500: '#87BBA2', 1000: '#C9E4CA'}

# Output (combined across discharges, so not nested under a single Q folder)
OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'tidal_prism'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Figure style ---
plt.rcParams.update(plt.rcParamsDefault)
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


def get_last_n_days_window(his_paths, n_days):
    """Return (t_start, t_end) covering the last n_days of the HIS dataset."""
    ds = xr.open_mfdataset(his_paths, coords="minimal", compat="override")
    t_end   = ds.time.values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    ds.close()
    return t_start, t_end


def load_his_signal(his_paths, t_start, t_end, location_keywords, upstream_keywords,
                     parameter, source='cross_section', subtract_river=True, negate=False):
    """Load + sign-correct the HIS signal at the mouth, river-corrected if
    requested. Mirrors load_his_data() in compute_tidal_prism.py."""

    SOURCE_DIM = {'cross_section': 'cross_section_name', 'station': 'station_name'}
    dim = SOURCE_DIM[source]

    ds_his       = xr.open_mfdataset(his_paths, coords="minimal", compat="override")
    ds_his_slice = ds_his.sel(time=slice(t_start, t_end))

    loc_names = [
        name.decode('utf-8').strip() if isinstance(name, bytes) else str(name).strip()
        for name in ds_his_slice[dim].values
    ]

    try:
        idx_mouth = next(i for i, name in enumerate(loc_names)
                         if any(k in name for k in location_keywords))
    except StopIteration:
        print("    [WARNING] Mouth keywords not found. Defaulting to index 0.")
        idx_mouth = 0

    idx_river = None
    if subtract_river and source == 'cross_section':
        try:
            idx_river = next(i for i, name in enumerate(loc_names)
                             if any(k in name for k in upstream_keywords))
        except StopIteration:
            print("    [WARNING] Upstream keywords not found. Falling back to no river correction.")

    time_values  = ds_his_slice.time.values
    time_seconds = (time_values - time_values[0]) / np.timedelta64(1, 's')

    if source == 'station':
        data_mouth = ds_his_slice[parameter].sel(stations=idx_mouth).values
    else:
        data_mouth = ds_his_slice[parameter].values[:, idx_mouth]

    sign = -1 if negate else 1

    if subtract_river and idx_river is not None:
        data_river = ds_his_slice[parameter].values[:, idx_river]
        signal = sign * (data_mouth - data_river)
    else:
        signal = sign * data_mouth

    ds_his.close()
    return time_seconds, signal


def compute_tidal_prism(time_seconds, signal):
    """prism = sum(|signal|) * dt / 2, scaled to PRISM_UNIT.
    Formula unchanged from compute_tidal_prism.py."""
    dt    = np.median(np.diff(time_seconds))
    prism = np.sum(np.abs(signal)) * dt / 2 / PRISM_SCALE
    return prism


# %% --- MAIN LOOP: compute tidal prism per scenario, across all discharges ---

datadict = {}   # label -> dict with discharge, run_id, pm, n, prism_Mm3

for discharge in DISCHARGES:
    dhr_base = BASE_DIR / f"Q{discharge}" / "detailed-hydro-run"
    scenario_folders = discover_scenario_folders(dhr_base, discharge)

    if not scenario_folders:
        print(f"[SKIP] No matching scenario folders found in: {dhr_base}")
        continue

    print(f"\n{'='*60}\nQ = {discharge} m3/s: found {len(scenario_folders)} scenario(s)\n{'='*60}")

    for folder_path, run_id, pm_val, n_val in scenario_folders:
        label = f"Qr{discharge}_dhr{run_id:02d}_pm{pm_val}_n{n_val}"
        print(f"  Processing: {folder_path.name}  ->  {label}")

        his_paths = get_stitched_his_paths(
            base_path=dhr_base, folder_name=folder_path.name,
            timed_out_dir=None, variability_map=None, analyze_noisy=False,
        )
        if not his_paths:
            his_paths = list(folder_path.glob("*_his.nc"))
        if not his_paths:
            print(f"    [SKIP] No HIS data found.")
            continue

        t_start, t_end = get_last_n_days_window(his_paths, TIDAL_PRISM_WINDOW_DAYS)

        time_s, signal = load_his_signal(
            his_paths, t_start, t_end,
            MOUTH_CROSS_SECTION_KEYWORDS, UPSTREAM_CROSS_SECTION_KEYWORDS,
            parameter=VARIABLE_TO_ANALYZE, source=SOURCE,
            subtract_river=SUBTRACT_RIVER, negate=NEGATE,
        )

        prism = compute_tidal_prism(time_s, signal)
        print(f"    Done. Tidal prism = {prism:.3f} {PRISM_UNIT}")

        datadict[label] = {
            'label':     label,
            'discharge': discharge,
            'run_id':    run_id,
            'pm':        pm_val,
            'n':         n_val,
            'prism_Mm3': prism,
        }

print(f"\nCollected tidal-prism results for {len(datadict)} runs.")

# %% --- SUMMARY ---

summary = pd.DataFrame(datadict.values())
summary.to_csv(OUTPUT_DIR / "tidal_prism_summary.csv", index=False)
print("\n" + "="*60)
print(summary.to_string(index=False))
print("="*60)

# %% --- PLOT: tidal prism vs peak amplitude, by discharge ---

if not datadict:
    print("No data to plot.")
else:
    present_discharges = sorted({v['discharge'] for v in datadict.values()})

    fig, ax = plt.subplots(figsize=(10, 6))

    for discharge in present_discharges:
        color  = DISCHARGE_COLORS.get(discharge, 'tab:gray')
        subset = {k: v for k, v in datadict.items() if v['discharge'] == discharge}

        pm_arr    = np.array([v['pm']        for v in subset.values()], dtype=float)
        prism_arr = np.array([v['prism_Mm3'] for v in subset.values()], dtype=float)
        if NORMALIZE:
            prism_n0 = np.array([v['prism_Mm3'] for v in subset.values() if v['n'] == 0], dtype=float)
            prism_plot = prism_arr / prism_n0 if len(prism_n0) > 0 else prism_arr
        else:
            prism_plot = prism_arr
        n_arr   = [v['n'] for v in subset.values()]
        label_q = f'Q = {discharge} m\u00b3/s'

        ax.scatter(prism_plot, pm_arr, color=color, zorder=3, s=80, label=label_q)

        for x_, y_, n_ in zip(prism_plot, pm_arr, n_arr):
            ax.annotate(f'n{n_}', (x_, y_), textcoords='offset points',
                        xytext=(4, 4), fontsize=FONTSIZE_TICKS, color=color)

    ax.set_ylabel('discharge amplitude $R_{\\mathrm{peak}}$', fontsize=FONTSIZE_LABELS)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICKS)
    ax.legend(fontsize=FONTSIZE_TICKS, loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()

    if NORMALIZE:
        ax.set_xlabel('normalized tidal prism [-]', fontsize=FONTSIZE_LABELS)
        fname = 'tidal_prism_vs_peak_amplitude_allQ_normalized'
    else:
        ax.set_xlabel(f'tidal prism [{PRISM_UNIT}]', fontsize=FONTSIZE_LABELS)
        fname = 'tidal_prism_vs_peak_amplitude_allQ'

    fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=200, bbox_inches='tight', transparent=_tr)
    fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight', transparent=_tr)
    plt.show()
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / f"{fname}.png"}')

print(f"\nOutputs written to: {OUTPUT_DIR.resolve()}")

# %%