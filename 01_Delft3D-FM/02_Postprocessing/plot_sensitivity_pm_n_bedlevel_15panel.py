"""15-panel bed level percentile sensitivity plot (Q = 500 only).

Rows (top -> bottom): p95, mean, p5 bed level, each computed within the
frozen t=0 channel mask (same convention as
plot_sensitivity_pm_n_bedlevel_percentiles.py).
Columns (left -> right): n_peaks = 1, 3, 4, 5, 6 (excluding the constant
pm1_n0 baseline).
Colors: R_peak (amplitude) value, Blues palette (light -> dark).

Style mirrors plot_map_hydro_netsedtransport.py: AGU rcParams (Calibri,
8pt), gridspec-based spacing with inch-based margins, and axis labels kept
at the minimum necessary — a single shared x-label at the figure level, and
per-row y-labels (since each row is a different metric) on the leftmost
column only. Column headers (n_peaks values) sit on the top row only.

A single legend (constant reference + natural-variability envelope + one
line per R_peak value) is shown at the bottom of the figure.
"""

# %% --- Imports ---
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

from FUNCTIONS.F_general import (
    _parse_pm_n,
    _scenario_label,
    get_variability_map,
    find_variability_model_folders,
    get_snapshot_matches_by_target_dates,
    sort_scenario_keys,
    group_snapshot_by_scenario,
    stack_metric_arrays,
)
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- 1. SETTINGS ---
DISCHARGE = 500
BASE_MODEL_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
BASE_DIR = BASE_MODEL_DIR / 'Model_Output'
base_path = BASE_DIR / f"Q{DISCHARGE}"

TARGET_DATE = np.datetime64('2031-01-01')

CHANNEL_INIT_THRESHOLD = 2.2   # defines the channel footprint from t=0
x_targets = np.arange(20000, 44001, 1000)
y_range = (5000, 10000)
dx = 1000

XTICK_STEP_KM = 5   # x-axis tick spacing (distance along estuary), in km
YTICK_STEP_M = 2    # y-axis tick spacing (bed level), in m

CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Natural variability envelope — noisy repeats of the constant scenario
SHOW_NOISY_ENVELOPE = True
NOISY_BASE_PATH = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models"
    r"\1_RiverDischargeVariability_domain45x15"
    r"\Model_Output\Q500\0_Noise_Q500"
)
NOISY_SUBFOLDERS = [
    '1_Q500_noisy0.9095347',
    '1_Q500_noisy1_rst.9160657',
    '1_Q500_noisy2_rst.9160663',
]

SCENARIO_LABELS = {
    '1':  'pm1_n0 (constant)',
    '2':  'pm2_n1',
    '3':  'pm3_n5',
    '4':  'pm3_n1',
    '5':  'pm5_n1',
    '6':  'pm4_n3',
    '7':  'pm3_n4',
    '8':  'pm2_n6',
    '9':  'pm5_n3',
    '10': 'pm3_n3',
    '11': 'pm2_n3',
    '12': 'pm5_n4',
    '13': 'pm4_n4',
    '14': 'pm2_n4',
}

GREY_CONST = "#7f7f7f"

# Row definitions: (row y-label, metric key, percentile or None for mean)
ROW_SPECS = [
    ('p95 bed level [m]', 'p95', 95),
    # ('mean bed level [m]', 'mean', None),
    ('p5 bed level [m]', 'p5', 5),
]

# --- AGU figure sizing (figures must be 50-170 mm wide) ---
MM_TO_IN = 1 / 25.4
FIGURE_WIDTH_MM = 170          # AGU full-page width

AGU_RC = {
    'font.size': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Calibri', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 9,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Calibri',
    'mathtext.it': 'Calibri:italic',
    'mathtext.bf': 'Calibri:bold',

    # --- Line weights: avoid hairlines (AGU rejects anything under 0.5pt) ---
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.75,
    'grid.linewidth': 0.4,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35,
    'ytick.minor.width': 0.35,

    # --- Keep text as editable text in vector exports (not outlined paths) ---
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',

    # --- Resolution / export ---
    'figure.dpi': 150,          # screen preview only
    'savefig.dpi': 300,         # within AGU's 300-600 ppi raster range
}
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(AGU_RC)

OUTPUT_DIR = BASE_DIR / 'output_plots_combined' / 'bedlevel_percentiles_pm_n_15panel'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assessment_dir = base_path / 'cached_data'
assessment_dir.mkdir(parents=True, exist_ok=True)
timed_out_dir = base_path / 'timed-out'


# %% --- 2. HELPER ---
def _bin_metrics_for_bedlevel(bed_1d, bin_mask):
    """Return (p95, p5) bed level for the faces selected by bin_mask."""
    if not np.any(bin_mask):
        return np.nan, np.nan
    vals = bed_1d[bin_mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    return np.percentile(vals, 95), np.percentile(vals, 5)


x_bins = np.arange(x_targets[0], x_targets[-1] + dx, dx)
x_centers = (x_bins[:-1] + x_bins[1:]) / 2
x_km_fallback = x_centers / 1000


# %% --- 3. LOAD MODEL DATA (single snapshot, all runs) ---
VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path, discharge=DISCHARGE, scenarios_to_process=None, analyze_noisy=False,
)

scenario_results = {}  # {folder_str: {'p95': arr, 'mean': arr, 'p5': arr, 'x_centers': arr}}

for folder in model_folders:
    folder_str = folder.name
    print(f"Processing: {folder_str}")

    run_paths = get_stitched_map_run_paths(
        base_path=base_path, folder_name=folder.name,
        timed_out_dir=timed_out_dir, variability_map=VARIABILITY_MAP, analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [base_path / folder]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir, folder_name=folder.name, run_paths=run_paths,
        var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS, append_vars=APPEND_VARIABLES, cache_tag=cache_tag,
    )
    if ds is None:
        print(f"  No cached data for {folder_str}, skipping.")
        continue

    matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
    if not matches:
        print(f"  No timesteps found for {folder_str}, skipping.")
        ds.close()
        continue
    _, ts_idx, actual_dt = matches[0]

    face_x, face_y = _get_face_coords(ds)
    width_mask = (face_y >= y_range[0]) & (face_y <= y_range[1])

    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values.copy()
    valid_init = width_mask & (init_bl < CHANNEL_INIT_THRESHOLD)   # frozen t=0 channel mask

    bed_at_snapshot = ds['mesh2d_mor_bl'].isel(time=ts_idx).values.copy()

    p95_arr, p5_arr = [], []
    for k in range(len(x_bins) - 1):
        bin_mask = valid_init & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
        p95_v, p5_v = _bin_metrics_for_bedlevel(bed_at_snapshot, bin_mask)
        p95_arr.append(p95_v)
        p5_arr.append(p5_v)

    scenario_results[folder_str] = {
        'p95': np.array(p95_arr), 'p5': np.array(p5_arr),
        'x_centers': x_centers,
    }
    print(f"  Snapshot {actual_dt}: computed p95/p5.")
    ds.close()


# %% --- 4. GROUP BY SCENARIO / PARSE pm,n ---
scenario_groups = group_snapshot_by_scenario(scenario_results)
all_scen_keys = sort_scenario_keys(scenario_groups.keys())

scen_pm_n = {}
for scen_key in all_scen_keys:
    label = _scenario_label(scen_key, SCENARIO_LABELS)
    pm, n = _parse_pm_n(label)
    if pm is not None:
        scen_pm_n[scen_key] = (pm, n)

baseline_scen = next((k for k, (pm, n) in scen_pm_n.items() if n == 0), None)

pm_by_n = {}   # {n_val: [(pm_val, scen_key), ...]}
for scen_key, (pm, n) in scen_pm_n.items():
    if n == 0:
        continue
    pm_by_n.setdefault(n, []).append((pm, scen_key))
for n in pm_by_n:
    pm_by_n[n].sort()

all_pm_vals = sorted({pm for pm, n in scen_pm_n.values() if n > 0})
sorted_n_vals = sorted(pm_by_n.keys())

_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}


def _get_y(scen_key, metric_key):
    y_stack = stack_metric_arrays(scenario_groups[scen_key], metric_key)
    if y_stack is None:
        return None
    return np.nanmean(y_stack, axis=0)


def _get_x(scen_key):
    x_data = next((d for _, d in scenario_groups[scen_key] if 'x_centers' in d), None)
    return x_data['x_centers'] / 1000 if x_data is not None else x_km_fallback


x_const = _get_x(baseline_scen) if baseline_scen else x_km_fallback
y_const_by_metric = {
    metric_key: (_get_y(baseline_scen, metric_key) if baseline_scen else None)
    for _, metric_key, _ in ROW_SPECS
}


# %% --- 5. NATURAL VARIABILITY ENVELOPE (±2σ, per metric) ---
noisy_metric_profiles = {metric_key: [] for _, metric_key, _ in ROW_SPECS}

if SHOW_NOISY_ENVELOPE:
    if not NOISY_BASE_PATH.exists():
        print(f"[WARNING] Noisy base path not found: {NOISY_BASE_PATH}")
    else:
        noisy_cache_dir = NOISY_BASE_PATH / 'cached_data'
        noisy_cache_dir.mkdir(parents=True, exist_ok=True)

        for subfolder in NOISY_SUBFOLDERS:
            noisy_folder = NOISY_BASE_PATH / subfolder
            if not noisy_folder.exists():
                print(f"[WARNING] Noisy subfolder not found: {noisy_folder}")
                continue
            print(f"Loading noisy run: {subfolder}")

            ds_n = load_or_update_map_cache_multi(
                cache_dir=noisy_cache_dir, folder_name=subfolder, run_paths=[noisy_folder],
                var_names=['mesh2d_mor_bl'], bbox=CACHE_BBOX,
                append_time=APPEND_TIMESTEPS, append_vars=APPEND_VARIABLES,
                cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
            )
            if ds_n is None:
                print(f"  No cached data for {subfolder}, skipping.")
                continue

            matches_n = get_snapshot_matches_by_target_dates(ds_n.time.values, [TARGET_DATE])
            if not matches_n:
                print(f"  No timesteps found for {subfolder}, skipping.")
                ds_n.close()
                continue
            _, ts_idx_n, _ = matches_n[0]

            fx_n, fy_n = _get_face_coords(ds_n)
            wmask_n = (fy_n >= y_range[0]) & (fy_n <= y_range[1])
            init_bl_n = ds_n['mesh2d_mor_bl'].isel(time=0).values.copy()
            valid_init_n = wmask_n & (init_bl_n < CHANNEL_INIT_THRESHOLD)
            bed_n = ds_n['mesh2d_mor_bl'].isel(time=ts_idx_n).values.copy()

            p95_n, p5_n = [], []
            for k in range(len(x_bins) - 1):
                bin_mask = valid_init_n & (fx_n >= x_bins[k]) & (fx_n < x_bins[k + 1])
                p95_v, p5_v = _bin_metrics_for_bedlevel(bed_n, bin_mask)
                p95_n.append(p95_v)
                p5_n.append(p5_v)

            noisy_metric_profiles['p95'].append(np.array(p95_n))
            noisy_metric_profiles['p5'].append(np.array(p5_n))
            print(f"  {subfolder}: OK")
            ds_n.close()

envelope_by_metric = {}
for metric_key, profs in noisy_metric_profiles.items():
    all_profs = list(profs)
    if y_const_by_metric.get(metric_key) is not None:
        all_profs.append(y_const_by_metric[metric_key])
    if not all_profs:
        continue
    stk = np.vstack(all_profs)
    m = np.nanmean(stk, axis=0)
    s = np.nanstd(stk, axis=0)
    envelope_by_metric[metric_key] = (m - 2 * s, m + 2 * s)


# %% --- 6. FIGURE / GRIDSPEC LAYOUT ---
n_cols = len(sorted_n_vals)
n_rows = len(ROW_SPECS)

fig_width_in = FIGURE_WIDTH_MM * MM_TO_IN
panel_width_in = fig_width_in / n_cols
panel_height_in = 0.8 * panel_width_in  

ROW_TITLE_GAP_IN = 0.15   # gap between stacked rows, for each row's subplot title
TOP_MARGIN_IN = 0.55      # space above the top row for the figure suptitle + its own title
BOTTOM_MARGIN_IN = 0.8   # space below the bottom row for x tick labels + x-axis label

fig_height_in = panel_height_in * n_rows + (n_rows - 1) * ROW_TITLE_GAP_IN + TOP_MARGIN_IN + BOTTOM_MARGIN_IN

fig = plt.figure(figsize=(fig_width_in, fig_height_in))
gs = gridspec.GridSpec(
    n_rows, n_cols, figure=fig,
    wspace=0.08, hspace=ROW_TITLE_GAP_IN / panel_height_in,
    top=1 - TOP_MARGIN_IN / fig_height_in,
    bottom=BOTTOM_MARGIN_IN / fig_height_in,
)

axes = np.empty((n_rows, n_cols), dtype=object)
ax_ref_x = None
for r in range(n_rows):
    ax_ref_y = None
    for c in range(n_cols):
        ax = fig.add_subplot(gs[r, c], sharex=ax_ref_x, sharey=ax_ref_y)
        if ax_ref_x is None:
            ax_ref_x = ax
        if ax_ref_y is None:
            ax_ref_y = ax
        axes[r, c] = ax


# %% --- 7. PLOT EACH PANEL ---
for r, (row_label, metric_key, _percentile) in enumerate(ROW_SPECS):
    y_const = y_const_by_metric.get(metric_key)
    env = envelope_by_metric.get(metric_key)

    for c, n_val in enumerate(sorted_n_vals):
        ax = axes[r, c]

        if SHOW_NOISY_ENVELOPE and env is not None:
            ax.fill_between(x_const, env[0], env[1], alpha=0.25, color='0.55', zorder=1)

        if y_const is not None:
            ax.plot(x_const, y_const, color=GREY_CONST, linewidth=0.9, linestyle='--', zorder=2)

        for pm_val, scen_key in pm_by_n[n_val]:
            y = _get_y(scen_key, metric_key)
            if y is None:
                continue
            x = _get_x(scen_key)
            ax.plot(x, y, color=PM_COLOR[pm_val], linewidth=1.0, zorder=3)

        ax.xaxis.set_major_locator(MultipleLocator(XTICK_STEP_KM))
        ax.yaxis.set_major_locator(MultipleLocator(YTICK_STEP_M))
        ax.grid(True, alpha=0.2, linewidth=0.3)
        ax.set_xlabel('')
        ax.set_ylabel('')

        if r == 0:
            ax.set_title(f'$n_{{peaks}}$ = {n_val}', fontsize=8)
        if r != n_rows - 1:
            ax.tick_params(labelbottom=False)
        if c == 0:
            ax.set_ylabel(row_label, fontsize=8)
        else:
            ax.tick_params(labelleft=False)

# %% --- 8. SHARED LEGEND + SAVE ---
legend_handles = []
if SHOW_NOISY_ENVELOPE:
    legend_handles.append(
        mpatches.Patch(facecolor='0.55', alpha=0.4, label=r'$\pm 2\sigma$ natural variability')
    )
legend_handles.append(
    mlines.Line2D([], [], color=GREY_CONST, linewidth=0.9, linestyle='--', label='constant')
)
for pm_val in all_pm_vals:
    pr_label = str(int(pm_val)) if pm_val == int(pm_val) else str(pm_val)
    legend_handles.append(
        mlines.Line2D([], [], color=PM_COLOR[pm_val], linewidth=1.2, label=f'$R_{{peak}}$ = {pr_label}')
    )

fig.legend(
    handles=legend_handles, fontsize=8, loc='lower center',
    ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.02), frameon=True,
)

fig.supxlabel('distance along estuary [km]', y=0.10)
fig.suptitle(
    f"Bed level percentiles vs. $R_{{peak}}$ / $n_{{peaks}}$ \u2014 {TARGET_DATE} snapshot, Q = {DISCHARGE} m\u00b3/s",
    fontsize=8, y=0.99,
)

save_name = f"AGU_bedlevel_percentiles_15panel_Q{DISCHARGE}_{TARGET_DATE}"
save_path = OUTPUT_DIR / f"{save_name}.png"
fig.savefig(save_path, dpi=300, bbox_inches='tight', transparent=True)
fig.savefig(OUTPUT_DIR / f"{save_name}.pdf", bbox_inches='tight', transparent=True)
plt.show()
print(f"Saved: {save_path}")

# %%
