"""Extract the intertidal area (area that falls wet AND dry at least once during
one tidal cycle) for each scenario, across all three discharge magnitudes
(Q = 250 / 500 / 1000 m3/s), then plot intertidal area as a function of discharge peak amplitude

Definitions (same as plot_intertidal_area_tidal_cycle.py):
  wet         : mesh2d_waterdepth > WET_THRESHOLD            (per timestep)
  always wet  : wet at every timestep in the window           -> SUBTIDAL  (excluded)
  always dry  : dry at every timestep in the window            -> SUPRATIDAL/land (excluded)
  intertidal  : wet at >=1 timestep AND dry at >=1 timestep    -> INTERTIDAL (included)

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
NORMALIZE = False   # if True, also produce a version normalized to each discharge's n=0 (constant) run

BASE_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output")

DISCHARGES = [250, 500, 1000]

# Only these run-IDs belong to the pm/n sensitivity matrix we want here.
RUN_IDS_TO_INCLUDE = {1, 6, 9, 10, 11}

# Matches: dhr_{run_id}_Qr{Q}_pm{pm}_n{n}.{runid}  or  ..._n{n}_mean.{runid}
_FOLDER_RE = re.compile(r'^dhr_(\d{2})_Qr(\d+)_pm(\d+)_n(\d+)(?:_mean)?\.\d+$')

LOAD_VARS = ['mesh2d_waterdepth', 'mesh2d_flowelem_ba']

FACE_X_VAR = 'mesh2d_face_x'   # mesh coordinate, already present in ds -- not cached separately

WET_THRESHOLD  = 0.0001   # [m] -- depths above this are "wet", same as Epshu in Delft3D-FM
WINDOW_HOURS   = 12.0    # fixed window length (~1 tidal cycle), taken from the end of each run

# Cache settings
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG  = None

# --- tidal-zone restriction in x (same window as the bed-level comparison script) ---
X_MIN = 20000.0   # [m]
X_MAX = 45000.0   # [m]  (all y included)

# Colour per discharge value (same palette as the LTI / min-depth plots)
DISCHARGE_COLORS = {250: '#3B6064', 500: '#87BBA2', 1000: '#C9E4CA'}

# Output (combined across discharges, so not nested under a single Q folder)
OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'intertidal_area'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Figure style ---
STYLE = 'default'

plt.rcParams.update(plt.rcParamsDefault)
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


def get_last_n_hours_window(time_values, n_hours):
    """Return boolean mask selecting the last n_hours of a datetime64 array."""
    t_end   = time_values[-1]
    t_start = t_end - np.timedelta64(int(n_hours * 3600), 's')
    return time_values >= t_start


def compute_intertidal_area(ds_window):
    """Wet/dry classification over the time window, restricted to the tidal
    zone in x, -> intertidal area [m^2]. Mirrors the logic in
    plot_intertidal_area_tidal_cycle.py, plus the x-restriction also used in
    plot_intertidal_area_bedlevel_allQ.py."""
    depth_vals = ds_window['mesh2d_waterdepth'].values   # (n_window, nFaces)
    wet_mask_t = depth_vals > WET_THRESHOLD                # (n_window, nFaces), bool

    always_wet = wet_mask_t.all(axis=0)
    always_dry = (~wet_mask_t).all(axis=0)
    intertidal = ~always_wet & ~always_dry

    x_vals = (ds_window.coords[FACE_X_VAR].values if FACE_X_VAR in ds_window.coords
              else ds_window[FACE_X_VAR].values)
    in_tidal_zone_x = (x_vals >= X_MIN) & (x_vals <= X_MAX)
    intertidal = intertidal & in_tidal_zone_x

    ba_da = ds_window['mesh2d_flowelem_ba']
    ba_vals = ba_da.isel(time=0).values if 'time' in ba_da.dims else ba_da.values

    area_m2 = float(np.nansum(ba_vals[intertidal]))
    return area_m2, intertidal.sum(), wet_mask_t.shape[1]


# %% --- MAIN LOOP: compute intertidal area per scenario, across all discharges ---

datadict = {}   # label -> dict with discharge, run_id, pm, n, intertidal_area_m2/km2

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

        if ds is None or 'time' not in ds.dims or 'mesh2d_waterdepth' not in ds or 'mesh2d_flowelem_ba' not in ds:
            print(f"    [SKIP] missing data for {label}")
            if ds is not None:
                ds.close()
            continue

        if FACE_X_VAR not in ds.coords and FACE_X_VAR not in ds:
            print(f"    [SKIP] missing {FACE_X_VAR} for {label}")
            ds.close()
            continue

        time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
        window_mask = get_last_n_hours_window(time_values, WINDOW_HOURS)
        n_window = int(window_mask.sum())

        if n_window < 2:
            print(f"    [SKIP] not enough timesteps in last {WINDOW_HOURS}h window (found {n_window})")
            ds.close()
            continue

        actual_window_hours = (time_values[window_mask][-1] - time_values[window_mask][0]) / np.timedelta64(1, 'h')
        if actual_window_hours < 11.0:
            print(f"    [WARNING] actual window is only {actual_window_hours:.1f}h -- "
                  f"shorter than one tidal cycle, interpret with caution.")

        ds_window = ds.isel(time=np.where(window_mask)[0])

        area_m2, n_intertidal_faces, n_faces = compute_intertidal_area(ds_window)
        ds.close()

        area_km2 = area_m2 / 1e6
        print(f"    Done. Intertidal area = {area_m2:,.0f} m^2  ({area_km2:.4f} km^2)  "
              f"[{n_intertidal_faces}/{n_faces} faces]")

        datadict[label] = {
            'label':                 label,
            'discharge':             discharge,
            'run_id':                run_id,
            'pm':                    pm_val,
            'n':                     n_val,
            'intertidal_area_m2':    area_m2,
            'intertidal_area_km2':   area_km2,
        }

print(f"\nCollected intertidal-area results for {len(datadict)} runs.")

# %% --- SUMMARY ---

summary = pd.DataFrame(datadict.values())
summary.to_csv(OUTPUT_DIR / "intertidal_area_summary.csv", index=False)
print("\n" + "="*60)
print(summary.to_string(index=False))
print("="*60)

# %% --- PLOT: intertidal area vs peak amplitude, by discharge ---
if not datadict:
    print("No data to plot.")
else:
    present_discharges = sorted({v['discharge'] for v in datadict.values()})

    fig, ax = plt.subplots(figsize=(10, 6))

    for discharge in present_discharges:
        color  = DISCHARGE_COLORS.get(discharge, 'tab:gray')
        subset = {k: v for k, v in datadict.items() if v['discharge'] == discharge}

        pm_arr    = np.array([v['pm']                  for v in subset.values()], dtype=float)
        area_km2  = np.array([v['intertidal_area_km2'] for v in subset.values()], dtype=float)
        if NORMALIZE:
            area_km2_n0 = np.array([v['intertidal_area_km2'] for v in subset.values() if v['n'] == 0], dtype=float)
            area_km2_normalized = area_km2 / area_km2_n0 if len(area_km2_n0) > 0 else area_km2
        else:
            area_km2_normalized = area_km2
        n_arr   = [v['n'] for v in subset.values()]
        label_q = f'Q = {discharge} m\u00b3/s'

        ax.scatter(area_km2_normalized, pm_arr, color=color, zorder=3, s=80, label=label_q)

        for x_, y_, n_ in zip(area_km2_normalized, pm_arr, n_arr):
            ax.annotate(f'n{n_}', (x_, y_), textcoords='offset points',
                        xytext=(4, 4), fontsize=FONTSIZE_TICKS, color=color)

    ax.set_ylabel('discharge amplitude $R_{\\mathrm{peak}}$', fontsize=FONTSIZE_LABELS)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONTSIZE_TICKS)
    ax.legend(fontsize=FONTSIZE_TICKS, loc='center left', bbox_to_anchor=(1.0, 0.5))

    fig.tight_layout()

    if NORMALIZE:
        ax.set_xlabel('normalized intertidal area [-]', fontsize=FONTSIZE_LABELS)
        fname = 'intertidal_area_vs_peak_amplitude_allQ_normalized'
    else:
        ax.set_xlabel('intertidal area [km$^2$]', fontsize=FONTSIZE_LABELS)
        fname = 'intertidal_area_vs_peak_amplitude_allQ'

    fig.savefig(OUTPUT_DIR / f'{fname}.png', dpi=200, bbox_inches='tight', transparent=_tr)
    fig.savefig(OUTPUT_DIR / f'{fname}.pdf', bbox_inches='tight', transparent=_tr)
    plt.show()
    plt.close(fig)
    print(f'Saved: {OUTPUT_DIR / f"{fname}.png"}')

print(f"\nOutputs written to: {OUTPUT_DIR.resolve()}")

# %%