"""
Plot sediment buffer volumes along estuary.
Divide estuary in sections based on distance from sea to river,
extract sediment transport through cross sections at the boundaries of these sections,
plot the cumulative sediment transport (buffer volume) over time for each section,
the instantaneous rate of change, tidal-averaged residual, trapping efficiency,
system sensitivity (dV/dt vs Q_river), and recovery time (cross-correlation).

Sensitivity and recovery plots: one figure per km section, all scenarios overlaid.
"""

# %% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import sys
from pathlib import Path
import re

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import load_cross_section_data, load_and_cache_scenario
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_sedimentbuffers import (
    tidal_avg,
    run_envelope_workflow,
)

# %% --- CONFIGURATION ---
# Workflow:
# - "full": original detailed per-scenario and sensitivity analysis
# - "envelope": noisy-envelope + variability comparison plots
# - "matrix": per-zone matrix plots only (skip all other plots)
WORKFLOW_MODE = "matrix"

sed_var   = 'cross_section_bedload_sediment_transport'
dis_var   = 'cross_section_discharge'
output_dirname = "plots_his_sedimentbuffer"

mpl.rcParams['figure.figsize'] = (8, 6)

# Cross-section km boundaries for buffer boxes
box_edges = np.arange(20, 50, 5)   # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i + 1]) for i in range(len(box_edges) - 1)]

# River boundary cross-section (most upstream, used as forcing signal)
RIVER_KM = 44

# Tidal averaging window (hours) — one tidal cycle
TIDAL_WINDOW_HOURS = 24

# Maximum lag for cross-correlation recovery plot (days)
MAX_LAG_DAYS = 365

# Output timestep of the HIS file (hours) — adjust if not 1h
DT_HOURS = 1                        #3600 seconds = 1 hour

# Number of timesteps to skip at the start (model spin-up)
SPINUP_STEPS = 24 * 3  # 3 days, adjust based on your output interval

# Scenario filters
SCENARIOS_TO_PROCESS = None
DISCHARGE   = 500
ANALYZE_NOISY = False

# Envelope mode settings
BASE_SCENARIO = '1'
ENVELOPE_PLOT_MODE = 'scenarios_only'  # 'noise_only', 'all', or 'scenarios_only'
SHOW_NOISY_ENVELOPE = False  # If False, keep scenario comparison lines only (no grey noisy runs/envelope).

# Toggle: set True to produce per-zone matrix plots (columns = n_peaks, rows = peak_ratio)
#         set False to skip matrix plots
PLOT_AS_MATRIX = True

# Human-readable labels per scenario number (used in combined plots)
SCENARIO_LABELS = {}

# colorblind friendly
SCENARIO_COLORS = {}

# %% --- PATHS ---
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian")
config = f"Model_Output/Q{DISCHARGE}"
if ANALYZE_NOISY:
    base_path = base_directory / config / f"0_Noise_Q{DISCHARGE}"
else:
    base_path = base_directory / config

output_dir = base_path / 'output_plots' / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / "timed-out"
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")
if not timed_out_dir.exists():
    timed_out_dir = None
    print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')

VARIABILITY_MAP = get_variability_map(DISCHARGE)
model_folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=ANALYZE_NOISY,
)

print(f"Found {len(model_folders)} run folders in: {base_path}")

if WORKFLOW_MODE == "envelope":
    run_envelope_workflow(
        base_directory=base_directory,
        config=config,
        discharge=DISCHARGE,
        output_dirname=output_dirname,
        boxes=boxes,
        sed_var=sed_var,
        variability_map=VARIABILITY_MAP,
        scenario_labels=SCENARIO_LABELS,
        scenario_colors=SCENARIO_COLORS,
        base_scenario=BASE_SCENARIO,
        envelope_plot_mode=ENVELOPE_PLOT_MODE,
        show_noisy_envelope=SHOW_NOISY_ENVELOPE,
    )
    raise SystemExit(0)

# %% --- BUILD HIS FILE PATH MAP ---
run_his_paths = {}
for folder in model_folders:
    model_location = base_path / folder
    his_paths = []
    scenario_num = folder.name.split('_')[0]
    try:
        scenario_key = str(int(scenario_num))
    except Exception:
        scenario_key = scenario_num

    if ANALYZE_NOISY:
        match = re.search(r'noisy(\d+)', folder.name)
        timed_out_folder = None
        if timed_out_dir is None:
            print('[WARNING] Timed-out directory not available; skipping timed-out noisy runs.')
        elif match:
            noisy_id = match.group(0)
            for f in timed_out_dir.iterdir():
                if f.is_dir() and noisy_id in f.name:
                    timed_out_folder = f.name
                    break
        if timed_out_folder:
            timed_out_path = timed_out_dir / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)
    else:
        if timed_out_dir is not None:
            timed_out_folder = VARIABILITY_MAP.get(scenario_key, folder.name)
            timed_out_path = timed_out_dir / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)

    main_his_path = model_location / "output" / "FlowFM_0000_his.nc"
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

# %% --- CACHE DIR SETUP ---
cache_dir = base_path / "cached_data"
if cache_dir.exists():
    if not cache_dir.is_dir():
        raise RuntimeError(f"[ERROR] {cache_dir} exists but is not a directory.")
    try:
        _ = list(cache_dir.iterdir())
    except Exception as e:
        raise RuntimeError(f"[ERROR] {cache_dir} is not accessible: {e}")
else:
    cache_dir.mkdir(exist_ok=True)

# %% --- LOAD ALL DATA (sediment transport + discharge) ---
scenario_data = {}

for scenario_dir, his_file_paths in run_his_paths.items():
    scenario_name = Path(scenario_dir).name
    scenario_num  = scenario_name.split('_')[0]
    run_id        = '_'.join(scenario_name.split('_')[1:])
    cache_file    = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

    # Load sediment transport (with buffer volumes)
    _, result = load_and_cache_scenario(
        scenario_dir=scenario_dir,
        his_file_paths=his_file_paths,
        cache_file=cache_file,
        boxes=boxes,
        var_name=sed_var,
    )

    # Load discharge (no buffer volumes needed)
    _, result_dis = load_and_cache_scenario(
        scenario_dir=scenario_dir,
        his_file_paths=his_file_paths,
        cache_file=cache_file,
        boxes=boxes,
        var_name=dis_var,
    )

    # Merge discharge into the main result dict
    result[dis_var] = result_dis[dis_var]

    scenario_data[scenario_dir] = result

# %% ============================================================
#    PER-SCENARIO PLOTS  — one figure per scenario, all sections in each
#    (skipped when WORKFLOW_MODE == "matrix")
# ===============================================================
if WORKFLOW_MODE != "matrix":
    for scenario_dir, data in scenario_data.items():
        scenario_name  = Path(scenario_dir).name
        scenario_num   = scenario_name.split('_')[0]
        scenario_label = SCENARIO_LABELS.get(str(int(scenario_num)), scenario_name)

        km_positions   = data['km_positions']
        transport      = data[sed_var]          # (time, km)  cumulative kg
        Q_all          = data[dis_var]          # (time, km)  m³/s
        time           = data['t']
        buffer_volumes = data['buffer_volumes']

        window = int(TIDAL_WINDOW_HOURS / DT_HOURS)

        # River discharge at upstream boundary
        idx_river  = np.argmin(np.abs(km_positions - RIVER_KM))

        print(f"{scenario_name}: closest cross-section to {RIVER_KM} km is at {km_positions[idx_river]:.2f} km")

        Q_river = data[dis_var][:, idx_river]
        time    = data['t']

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, Q_river, lw=0.8)
        ax.set_xlabel('Time')
        ax.set_ylabel('Discharge (m³/s)')
        ax.set_title(f'Discharge at {km_positions[idx_river]:.2f} km — {scenario_name}')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()

        # ── 1. CUMULATIVE BUFFER VOLUME ──────────────────────────
        fig, ax = plt.subplots()
        for (box_start, box_end), buf in buffer_volumes.items():
            ax.plot(time, buf, label=f'{box_start}–{box_end} km')
        ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
        ax.set_ylabel('Buffer volume (m³)')
        ax.set_ylim(-0.1, 1.15e11)
        ax.legend()
        ax.set_title(f'Sediment buffer volumes — {scenario_label}')
        ax.grid()
        fig.tight_layout()
        fig.savefig(output_dir / f"{scenario_name}_sediment_buffer_volume_cumulative.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    # # ── 2. INSTANTANEOUS dV/dt ───────────────────────────────
    # fig, ax = plt.subplots()
    # for (box_start, box_end), buf in buffer_volumes.items():
    #     ax.plot(time[1:], np.diff(buf), label=f'{box_start}–{box_end} km')
    # ax.set_ylim(-0.25e8, 0.25e8)
    # ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
    # ax.set_ylabel('dV/dt (m³/timestep)')
    # ax.legend()
    # ax.set_title(f'Instantaneous buffer change — {scenario_label}')
    # ax.grid()
    # fig.tight_layout()
    # fig.savefig(output_dir / f"{scenario_name}_sediment_buffer_volume_instantaneous.png",
    #             dpi=300, bbox_inches='tight')
    # plt.show()

    # # ── 3. TIDAL-AVERAGED RESIDUAL dV/dt ────────────────────
    # fig, ax = plt.subplots(figsize=(12, 6))
    # for (box_start, box_end), buf in buffer_volumes.items():
    #     d_buf = np.diff(buf)
    #     if len(d_buf) >= window:
    #         ax.plot(time[1:], tidal_avg(d_buf, window),
    #                 label=f'Residual: {box_start}–{box_end} km', linewidth=2)
    #     else:
    #         print(f"Warning: too short for tidal window at {box_start} km")
    # ax.axhline(0, color='black', lw=1.5, ls='--')
    # ax.set_ylim(-0.25e7, 0.25e7)
    # ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
    # ax.set_ylabel('Tidal avg dV/dt (m³/timestep)')
    # ax.set_title(f'Residual sediment storage rate — {scenario_label}')
    # ax.legend()
    # ax.grid(True, which='both', alpha=0.3)
    # fig.tight_layout()
    # fig.savefig(output_dir / f"{scenario_name}_residual_buffer_change.png", dpi=300)
    # plt.show()

    # # ── 4. TRAPPING EFFICIENCY ───────────────────────────────
    # fig, ax = plt.subplots(figsize=(12, 6))
    # for (box_start, box_end), buf in buffer_volumes.items():
    #     idx_up = np.argmin(np.abs(km_positions - box_start))
    #     dV         = np.diff(buf)
    #     dIn_gross  = np.abs(np.diff(transport[:, idx_up]))
    #     efficiency = (np.convolve(dV,        np.ones(window), mode='same') /
    #                  (np.convolve(dIn_gross, np.ones(window), mode='same') + 1e-6))
    #     ax.plot(time[1:], efficiency, label=f'Efficiency: {box_start}–{box_end} km')
    # ax.axhline(0, color='black', lw=1,   ls='--')
    # ax.axhline(1, color='red',   lw=1,   ls=':', label='Total trap')
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_xlabel('Hydrodynamic time (×100 = morphological time)')
    # ax.set_ylabel('Tidal-averaged efficiency (–)')
    # ax.set_title(f'Sediment trapping efficiency — {scenario_label}')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # fig.tight_layout()
    # fig.savefig(output_dir / f"{scenario_name}_trapping_efficiency.png", dpi=300)
    # plt.show()


# %% ============================================================
#    PRE-PROCESS + SENSITIVITY  (skipped in matrix mode)
# ===============================================================
if WORKFLOW_MODE != "matrix":
    window        = int(TIDAL_WINDOW_HOURS / DT_HOURS)
    max_lag_steps = int(MAX_LAG_DAYS * 24 / DT_HOURS)

    t_end_min = min(
        data['t'][SPINUP_STEPS:].max()
        for data in scenario_data.values()
    )
    print(f"Shortest simulation end time across scenarios: {t_end_min}")

    processed = {}   # scenario_key → dict of trimmed arrays

    for scenario_dir, data in scenario_data.items():
        scenario_num  = Path(scenario_dir).name.split('_')[0]
        scenario_key  = str(int(scenario_num))

        km_positions   = data['km_positions']
        Q_all          = data[dis_var]
        buffer_volumes = data['buffer_volumes']
        time           = data['t']

        idx_river = np.argmin(np.abs(km_positions - RIVER_KM))
        
        time_trimmed = time[SPINUP_STEPS:]
        t_end_idx = np.searchsorted(time_trimmed, t_end_min, side='right')

        processed[scenario_key] = dict(
            label                  = SCENARIO_LABELS.get(scenario_key, scenario_dir),
            color                  = SCENARIO_COLORS.get(scenario_key, 'grey'),
            Q_river                = Q_all[SPINUP_STEPS : SPINUP_STEPS + t_end_idx, idx_river],
            time                   = time_trimmed[:t_end_idx],
            buffer_volumes_trimmed = {k: v[SPINUP_STEPS : SPINUP_STEPS + t_end_idx]
                                      for k, v in buffer_volumes.items()},
            km_positions           = km_positions,
            transport              = data[sed_var][:SPINUP_STEPS + t_end_idx],
        )

    # SENSITIVITY  (dV/dt vs Q_river)
    for (box_start, box_end) in boxes:
        fig, ax = plt.subplots(figsize=(8, 6))

        for scenario_key, d in processed.items():
            buf    = d['buffer_volumes_trimmed'][(box_start, box_end)]
            Q_plot = d['Q_river'][:-1]          # align with np.diff output
            dV_dt  = np.diff(buf)

            # Scatter (low alpha to avoid overplotting)
            ax.scatter(Q_plot, dV_dt, alpha=0.15, s=6, color=d['color'])

            # Quadratic trend line (labelled — this is the visual signal)
            z  = np.polyfit(Q_plot, dV_dt, 2)
            xp = np.linspace(Q_plot.min(), Q_plot.max(), 300)
            ax.plot(xp, np.poly1d(z)(xp), '-', linewidth=2.5,
                    color=d['color'], label=d['label'])

        ax.axhline(0, color='black', lw=1,   ls='--', zorder=3)
        ax.axvline(0, color='black', lw=0.8, ls='--', zorder=3)
        ax.set_xlabel(f'River discharge at {RIVER_KM} km (m³/s)', fontsize=11)
        ax.set_ylabel('Section buffer change  dV/dt  (m³/timestep)', fontsize=11)
        ax.set_title(f'System sensitivity — section {box_start}–{box_end} km', fontsize=12)
        ax.legend(title='Scenario', fontsize=9)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"section_{box_start}-{box_end}km_sensitivity_allscenarios.png",
            dpi=300, bbox_inches='tight'
        )
        plt.show()

# %% --- MATRIX PLOTS (one per zone) ---
if WORKFLOW_MODE == "matrix" or PLOT_AS_MATRIX:
    _FOLDER_RE = re.compile(r"_pm(\d+(?:\.\d+)?)_n(\d+)", re.IGNORECASE)

    # Annotate each loaded scenario with peak_ratio / n_peaks
    parseable_scenarios = []
    for scenario_dir, data in scenario_data.items():
        folder_name = Path(scenario_dir).name
        m = _FOLDER_RE.search(folder_name)
        if m:
            scenario_num = folder_name.split('_')[0]
            scenario_key = str(int(scenario_num))
            parseable_scenarios.append({
                'peak_ratio':   float(m.group(1)),
                'n_peaks':      int(m.group(2)),
                'scenario_key': scenario_key,
                'data':         data,
            })

    if not parseable_scenarios:
        print("[WARNING] No _pm<r>_n<n> folder names found; skipping matrix plots.")
    else:
        all_n_peaks     = sorted({s['n_peaks']    for s in parseable_scenarios})
        all_peak_ratios = sorted({s['peak_ratio'] for s in parseable_scenarios})
        n_cols    = len(all_n_peaks)
        n_rows    = len(all_peak_ratios)
        row_order = list(reversed(all_peak_ratios))   # highest ratio at top

        PANEL_W, PANEL_H = 3.4, 2.4
        LINE_COLOR = "#1f77b4"

        for (box_start, box_end) in boxes:
            # Global y-limits for this zone
            all_buf_vals = np.concatenate([
                s['data']['buffer_volumes'][(box_start, box_end)]
                for s in parseable_scenarios
                if (box_start, box_end) in s['data']['buffer_volumes']
            ])
            g_ymin, g_ymax = float(np.nanmin(all_buf_vals)), float(np.nanmax(all_buf_vals))
            ypad = max((g_ymax - g_ymin) * 0.08, 1.0)
            global_ylim = (g_ymin - ypad, g_ymax + ypad)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(PANEL_W * n_cols, PANEL_H * n_rows),
                sharex=False, sharey=True,
            )
            axes = np.atleast_2d(axes)

            for ri, peak_ratio in enumerate(row_order):
                for ci, n_peaks in enumerate(all_n_peaks):
                    ax = axes[ri, ci]
                    matches = [
                        s for s in parseable_scenarios
                        if s['peak_ratio'] == peak_ratio
                        and s['n_peaks'] == n_peaks
                        and (box_start, box_end) in s['data']['buffer_volumes']
                    ]
                    if not matches:
                        ax.set_visible(False)
                        continue

                    for s in matches:
                        buf  = s['data']['buffer_volumes'][(box_start, box_end)]
                        time = s['data']['t']
                        color = (
                            SCENARIO_COLORS.get(s['scenario_key'], LINE_COLOR)
                            if SCENARIO_COLORS else LINE_COLOR
                        )
                        ax.plot(time, buf, color=color, linewidth=0.9)

                    ax.set_ylim(global_ylim)
                    ax.grid(True, alpha=0.22, linewidth=0.5)

                    # x-axis: date labels on bottom row only
                    if ri == n_rows - 1:
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.tick_params(axis="x", labelsize=7, rotation=40)
                    else:
                        ax.set_xticklabels([])
                        ax.tick_params(axis="x", length=0)

                    # y-axis: tick labels on left column only
                    if ci == 0:
                        ax.tick_params(axis="y", labelsize=7)
                        ax.set_ylabel("Buffer volume (m\u00b3)", fontsize=7, labelpad=3)
                    else:
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", length=0)

            # Column headers (n_peaks)
            for ci, n_peaks in enumerate(all_n_peaks):
                for ri in range(n_rows):
                    if axes[ri, ci].get_visible():
                        axes[ri, ci].set_title(
                            f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}",
                            fontsize=9, fontweight="bold", pad=5,
                        )
                        break

            # Row labels (peak_ratio)
            for ri, peak_ratio in enumerate(row_order):
                pr_str = (
                    f"{int(peak_ratio)}"
                    if peak_ratio == int(peak_ratio)
                    else f"{peak_ratio}"
                )
                for ci in range(n_cols - 1, -1, -1):
                    if axes[ri, ci].get_visible():
                        axes[ri, ci].yaxis.set_label_position("right")
                        axes[ri, ci].set_ylabel(
                            f"$R_{{\\mathrm{{peak}}}}$ = {pr_str}",
                            rotation=270, labelpad=15,
                            fontsize=9, fontweight="bold",
                        )
                        break

            fig.text(0.5, 0.005, "Number of peaks per year   \u2192",
                     ha="center", fontsize=10, style="italic", color="0.35")
            fig.text(0.005, 0.5, "Peak / mean ratio   \u2192",
                     va="center", fontsize=10, style="italic", color="0.35",
                     rotation="vertical")
            fig.suptitle(
                f"Sediment buffer volume \u2014 zone {box_start}\u2013{box_end} km  "
                f"($Q_{{\\mathrm{{mean}}}}$ = {DISCHARGE} m\u00b3/s)",
                fontsize=13, fontweight="bold", y=1.01,
            )
            fig.tight_layout(rect=[0.03, 0.03, 1.0, 1.0])

            fig_path = output_dir / (
                f"matrix_zone_{box_start}-{box_end}km_buffer_volume_Q{DISCHARGE}.png"
            )
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.show()
            print(f"Saved matrix plot: {fig_path}")
# %%
