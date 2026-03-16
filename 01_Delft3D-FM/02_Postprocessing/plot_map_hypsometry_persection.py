"""
Plot hypsometric curves per estuary section, over time and across scenarios.

For each section (defined by box_edges along x):
  - Per-scenario plot: one line per timestep (legend = date labels), y = bed level, x = cumulative area fraction
  - Comparison plot:   all scenarios overlaid at each snapshot date

Data source: map cache via load_or_update_map_cache_multi (same cache as map plotting script).
"""

# %% IMPORTS
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- CONFIGURATION ---
DISCHARGE = 1000
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"

# Sections (in metres, matching map face_x coordinates)
box_edges = np.arange(20, 50, 5) * 1000   # [20000, 25000, 30000, 35000, 40000, 45000] m
boxes = [(box_edges[i], box_edges[i + 1]) for i in range(len(box_edges) - 1)]

# Y-range to restrict to channel strip
y_range = (5000, 10000)

# Land threshold: faces with bed level >= this are excluded (intertidal/land)
bed_threshold = 6   # m

# Apply detrending (subtract first timestep bed level)
apply_detrending = False

# Snapshot target dates for comparison plots (nearest available timestep is used per scenario)
# Set to None for equally spaced snapshots within SNAPSHOT_DATE_RANGE
SNAPSHOT_TARGET_DATES = None   # e.g. ['2027-01-01', '2035-01-01', '2045-01-01', '2055-01-01']
SNAPSHOT_DATE_RANGE   = (np.datetime64('2025-01-01'), np.datetime64('2055-12-31'))
SNAPSHOT_COUNT        = 4

# Which scenarios to include (None = all found)
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']

# Cache settings (must match the map script)
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG  = None
APPEND_TIMESTEPS = False   # Cache already built by map plotting script — do not reload raw map files
APPEND_VARIABLES = False

# Output
output_dirname = "plots_map_hypsometric_sections"

# Human-readable labels and colours per scenario number
SCENARIO_LABELS = {
    '1': 'Constant',
    '2': 'Seasonal',
    '3': 'Flashy',
    '4': 'Single peak',
}
SCENARIO_COLORS = {
    '1': '#1f77b4',
    '2': '#ff7f0e',
    '3': '#2ca02c',
    '4': '#d62728',
}

mpl.rcParams['figure.figsize'] = (9, 6)

# %% --- PATHS ---
base_path      = base_directory / config
assessment_dir = base_path / 'cached_data'
assessment_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / 'timed-out'
if not timed_out_dir.exists():
    timed_out_dir = None
    print('[WARNING] Timed-out directory not found.')

summary_output_dir = base_path / 'output_plots' / output_dirname
summary_output_dir.mkdir(parents=True, exist_ok=True)

# %% --- FIND RUN FOLDERS (discharge-dependent logic) ---
VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=False,
)

print(f"Found {len(model_folders)} run folders in: {base_path}")

# %% --- HELPERS ---

def _scenario_key(folder_name):
    try:
        return str(int(str(folder_name).split('_')[0]))
    except Exception:
        return str(folder_name).split('_')[0]


def _scenario_label(folder_name):
    return SCENARIO_LABELS.get(_scenario_key(folder_name), str(folder_name))


def _scenario_color(folder_name):
    return SCENARIO_COLORS.get(_scenario_key(folder_name), 'grey')


def _date_label(dt64_ns):
    return str(np.datetime_as_string(np.datetime64(dt64_ns, 'ns'), unit='D'))


def get_snapshot_indices(time_values, target_dates):
    """Return list of (target_dt, ts_idx, actual_dt) for each target date."""
    time_ns = np.array(time_values, dtype='datetime64[ns]').astype('int64')
    results = []
    for tgt in target_dates:
        tgt_ns = np.datetime64(tgt, 'ns').astype('int64')
        idx = int(np.argmin(np.abs(time_ns - tgt_ns)))
        results.append((tgt, idx, time_values[idx]))
    return results


def build_snapshot_dates(count, explicit_dates, date_range):
    if explicit_dates:
        return [np.datetime64(d, 'ns') for d in explicit_dates]
    count = max(2, int(count))
    start_ns = np.datetime64(date_range[0], 'ns').astype('int64')
    end_ns   = np.datetime64(date_range[1], 'ns').astype('int64')
    return [np.datetime64(int(ns), 'ns')
            for ns in np.linspace(start_ns, end_ns, count)]


def _get_face_coords(ds):
    """Robustly extract face_x and face_y from a xugrid UgridDataset.
    Tries multiple access patterns to handle differences across xugrid versions."""
    try:
        return np.asarray(ds.grids[0].face_x), np.asarray(ds.grids[0].face_y)
    except Exception:
        pass
    try:
        return np.asarray(ds.grid.face_x), np.asarray(ds.grid.face_y)
    except Exception:
        pass
    try:
        return np.asarray(ds.coords['mesh2d_face_x']), np.asarray(ds.coords['mesh2d_face_y'])
    except Exception:
        pass
    raise RuntimeError(
        "Could not extract face_x / face_y from the xugrid dataset. "
        "Check that face_coordinates are preserved in the cache topology."
    )


def compute_hypsometric_curve(bedlev_data, valid_mask):
    """
    Returns (elev_sorted, cum_area_fraction) for faces within valid_mask.
    cum_area_fraction runs 0 → 1 (no face-area weighting; each face counts equally).
    """
    vals = bedlev_data[valid_mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([]), np.array([])
    order    = np.argsort(vals)
    elev     = vals[order]
    cum_area = np.arange(1, len(elev) + 1, dtype=float) / len(elev)
    return elev, cum_area


# %% --- BUILD SNAPSHOT DATE LIST ---
target_snapshot_dates = build_snapshot_dates(
    count          = SNAPSHOT_COUNT,
    explicit_dates = SNAPSHOT_TARGET_DATES,
    date_range     = SNAPSHOT_DATE_RANGE,
)
print("Target snapshot dates:")
for d in target_snapshot_dates:
    print(f"  {_date_label(d)}")

# %% --- MAIN PROCESSING LOOP ---
# comparison_store[snapshot_key][folder_str][section_key] = {'elev', 'area'}
comparison_store  = {}
snapshot_labels   = {}   # snapshot_key -> human-readable date string

for folder in model_folders:
    folder_str     = folder.name
    scenario_color = _scenario_color(folder_str)
    scenario_lbl   = _scenario_label(folder_str)
    save_dir       = summary_output_dir / f'individual_{folder_str}'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing: {folder_str} ---")

    # Load from cache
    run_paths = get_stitched_map_run_paths(
        base_path       = base_path,
        folder_name     = folder.name,
        timed_out_dir   = timed_out_dir,
        variability_map = VARIABILITY_MAP,
        analyze_noisy   = False,
    )
    if not run_paths:
        run_paths = [folder]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir    = assessment_dir,
        folder_name  = folder.name,
        run_paths    = run_paths,
        var_names    = ['mesh2d_mor_bl'],
        bbox         = CACHE_BBOX,
        append_time  = APPEND_TIMESTEPS,
        append_vars  = APPEND_VARIABLES,
        cache_tag    = cache_tag,
    )

    if ds is None:
        print(f"  No cached data for {folder_str}, skipping.")
        continue

    time_values = ds.time.values
    face_x, face_y = _get_face_coords(ds)
    n_times     = len(time_values)
    print(f"  {n_times} timestep(s): {_date_label(time_values[0])} → {_date_label(time_values[-1])}")

    # Reference bed for detrending (first timestep)
    reference_bed = ds['mesh2d_mor_bl'].isel(time=0).values.copy() if apply_detrending else None

    # Find snapshot indices
    snapshot_matches = get_snapshot_indices(time_values, target_snapshot_dates)

    # --- Per-section individual plots (all timesteps, one scenario) ---
    for box_start_m, box_end_m in boxes:
        box_start_km = int(box_start_m / 1000)
        box_end_km   = int(box_end_m   / 1000)
        section_key  = f'{box_start_km}_{box_end_km}'

        section_mask = (
            (face_x >= box_start_m) & (face_x < box_end_m) &
            (face_y >= y_range[0])  & (face_y <= y_range[1])
        )
        if not np.any(section_mask):
            print(f"  [WARNING] No faces in section {box_start_km}–{box_end_km} km, skipping.")
            continue

        # Colour palette: one colour per timestep
        t_cmap   = plt.cm.plasma
        t_colors = [t_cmap(i / max(n_times - 1, 1)) for i in range(n_times)]

        fig, ax = plt.subplots()

        for t_idx in range(n_times):
            bedlev = ds['mesh2d_mor_bl'].isel(time=t_idx).values.copy()
            if apply_detrending:
                bedlev = bedlev - reference_bed
                valid_mask = section_mask
            else:
                valid_mask = section_mask & (bedlev < bed_threshold)

            elev, cum_area = compute_hypsometric_curve(bedlev, valid_mask)
            if elev.size == 0:
                continue

            ax.plot(cum_area, elev,
                    color=t_colors[t_idx],
                    linewidth=1.1,
                    alpha=0.85,
                    label=_date_label(time_values[t_idx]))

        # Legend: show all dates (user requested legend, not colorbar)
        handles, labels = ax.get_legend_handles_labels()
        # If many timesteps, thin the legend to avoid clutter (show ~8 evenly spaced)
        MAX_LEGEND_ENTRIES = 8
        if len(handles) > MAX_LEGEND_ENTRIES:
            step     = max(1, len(handles) // MAX_LEGEND_ENTRIES)
            handles  = handles[::step]
            labels   = labels[::step]
        ax.legend(handles, labels, title='Date', fontsize=7,
                  loc='best', ncol=1, framealpha=0.7)

        ax.set_xlabel('Cumulative area fraction [–]')
        detrend_sfx = ' (Detrended)' if apply_detrending else ''
        ax.set_ylabel(f'Bed elevation [m]{detrend_sfx}')
        ax.set_title(
            f'Hypsometric curves over time\n'
            f'{scenario_lbl} — section {box_start_km}–{box_end_km} km'
        )
        if not apply_detrending:
            ax.axhline(y=bed_threshold, color='red', ls='--',
                       lw=1, alpha=0.6, label=f'Land threshold ({bed_threshold} m)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f'hypso_section_{box_start_km}_{box_end_km}km_{folder_str}.png'
        fig.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fname}")
        plt.show()
        plt.close(fig)

    # --- Store snapshot results for comparison plots ---
    for tgt_dt, ts_idx, actual_dt in snapshot_matches:
        snap_key = f"snap_{_date_label(tgt_dt).replace('-', '')}"
        snapshot_labels[snap_key] = _date_label(tgt_dt)
        comparison_store.setdefault(snap_key, {})
        comparison_store[snap_key].setdefault(folder_str, {})

        bedlev = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()
        if apply_detrending:
            bedlev = bedlev - reference_bed

        for box_start_m, box_end_m in boxes:
            box_start_km = int(box_start_m / 1000)
            box_end_km   = int(box_end_m   / 1000)
            section_key  = f'{box_start_km}_{box_end_km}'

            section_mask = (
                (face_x >= box_start_m) & (face_x < box_end_m) &
                (face_y >= y_range[0])  & (face_y <= y_range[1])
            )
            if not np.any(section_mask):
                continue

            valid_mask = section_mask if apply_detrending else (section_mask & (bedlev < bed_threshold))
            elev, cum_area = compute_hypsometric_curve(bedlev, valid_mask)

            if elev.size > 0:
                comparison_store[snap_key][folder_str][section_key] = {
                    'elev': elev,
                    'area': cum_area,
                    'actual_date': _date_label(actual_dt),
                }

    ds.close()

# %% --- COMPARISON PLOTS: all scenarios per section, one figure per snapshot date ---
print("\n--- Generating comparison plots ---")

for snap_key, snap_scenarios in comparison_store.items():
    if not snap_scenarios:
        continue
    tgt_date_label = snapshot_labels.get(snap_key, snap_key)

    for box_start_m, box_end_m in boxes:
        box_start_km = int(box_start_m / 1000)
        box_end_km   = int(box_end_m   / 1000)
        section_key  = f'{box_start_km}_{box_end_km}'

        fig, ax = plt.subplots()
        has_data = False

        for folder_str, section_dict in snap_scenarios.items():
            if section_key not in section_dict:
                continue
            d = section_dict[section_key]
            actual_lbl = d['actual_date']
            ax.plot(
                d['area'], d['elev'],
                color     = _scenario_color(folder_str),
                linewidth = 2,
                label     = f"{_scenario_label(folder_str)} ({actual_lbl})",
            )
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.set_xlabel('Cumulative area fraction [–]')
        detrend_sfx = ' (Detrended)' if apply_detrending else ''
        ax.set_ylabel(f'Bed elevation [m]{detrend_sfx}')
        ax.set_title(
            f'Hypsometric curves — section {box_start_km}–{box_end_km} km\n'
            f'Target snapshot: {tgt_date_label} — all scenarios'
        )
        if not apply_detrending:
            ax.axhline(y=bed_threshold, color='red', ls='--',
                       lw=1, alpha=0.6, label=f'Land threshold ({bed_threshold} m)')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        detrend_tag = '_detrended' if apply_detrending else ''
        fname = (f'hypso_comparison_section_{box_start_km}_{box_end_km}km'
                 f'_{tgt_date_label.replace("-", "")}{detrend_tag}.png')
        fig.savefig(summary_output_dir / fname, dpi=300, bbox_inches='tight')
        print(f"Saved: {fname}")
        plt.show()
        plt.close(fig)

print("\nAll hypsometric section plots complete.")