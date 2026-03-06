"""
CCF-based lag analysis: response time of estuarine sections to river discharge variability.

Following Wang et al. (2025, Nat. Commun.) but applied to hourly model output.

METHOD — Normalised cross-correlation (static CCF)
───────────────────────────────────────────────────
For tidal-averaged Q_river(t) and dV/dt(t) of length N, the normalised CCF is:

    rho(tau) = sum_t [ Q'(t) * (dV/dt)'(t+tau) ]
               ─────────────────────────────────────────────────────
               sqrt( sum_t [Q'(t)]^2  *  sum_t [(dV/dt)'(t)]^2 )

where Q'(t) = Q(t) - mean(Q)  and  (dV/dt)'(t) = dV/dt(t) - mean(dV/dt)
are the anomaly series (mean removed), and tau is the lag in timesteps.

The primary lag Delta_t is the lag of the first significant local maximum
in |rho(tau)| for tau > 0, where significance is |rho| > 2/sqrt(N_eff)
and N_eff = N / smooth_steps (conservative for autocorrelated series).

METHOD — Sliding-window CCF
────────────────────────────
The same CCF is computed in non-overlapping windows of one hydrodynamic year
(8760 hourly timesteps) across the full simulation. Each window yields one
lag estimate, producing a time series of Delta_t over ~40 morphological years.
This reveals whether the estuary's response time evolves as morphology adjusts.

Sign convention:
    rho(Delta_t) > 0  →  deposition follows Q peak  (dep)
    rho(Delta_t) < 0  →  erosion follows Q peak      (ero)

Output: one storyboard per scenario for each method.
"""

# ── IMPORTS ────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path
from scipy.signal import find_peaks
import re

# ── FORCE WHITE STYLE (overrides any dark_background set globally) ─────────────
plt.style.use('default')
mpl.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'savefig.facecolor': 'white',
    'text.color':       'black',
    'axes.labelcolor':  'black',
    'xtick.color':      'black',
    'ytick.color':      'black',
})

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import load_cross_section_data, load_and_cache_scenario

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
sed_var        = 'cross_section_bedload_sediment_transport'
dis_var        = 'cross_section_discharge'
output_dirname = "plots_his_ccf_response"

mpl.rcParams['figure.figsize'] = (8, 6)

# Section boundaries
box_edges = np.arange(20, 50, 5)
boxes     = [(box_edges[i], box_edges[i + 1]) for i in range(len(box_edges) - 1)]

RIVER_KM          = 44          # upstream Q boundary cross-section
TIDAL_WINDOW_HOURS = 24         # hours for tidal averaging
DT_HOURS          = 1           # model output timestep (hours)
SPINUP_STEPS      = 24 * 3      # timesteps to skip at start

# ── SCENARIO SETTINGS ─────────────────────────────────────────────────────────
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']
DISCHARGE   = 500
ANALYZE_NOISY = False

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

# ── PATHS ──────────────────────────────────────────────────────────────────────
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config   = f"Model_Output/Q{DISCHARGE}"
base_path = base_directory / config

output_dir = base_path / 'output_plots' / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / "timed-out"
if not timed_out_dir.exists():
    timed_out_dir = None

VARIABILITY_MAP = {
    '1': f'01_baserun{DISCHARGE}',
    '2': f'02_run{DISCHARGE}_seasonal',
    '3': f'03_run{DISCHARGE}_flashy',
    '4': f'04_run{DISCHARGE}_singlepeak',
}

# ── FIND RUN FOLDERS ───────────────────────────────────────────────────────────
model_folders = [f.name for f in base_path.iterdir()
                 if f.is_dir() and f.name[0].isdigit()]
if SCENARIOS_TO_PROCESS:
    scenario_filter = set(int(s) for s in SCENARIOS_TO_PROCESS)
    model_folders = [f for f in model_folders if int(f.split('_')[0]) in scenario_filter]
model_folders.sort(key=lambda x: int(x.split('_')[0]))

# ── BUILD HIS FILE PATH MAP ────────────────────────────────────────────────────
run_his_paths = {}
for folder in model_folders:
    model_location = base_path / folder
    his_paths      = []
    scenario_key   = str(int(folder.split('_')[0]))

    if timed_out_dir is not None:
        timed_out_folder = VARIABILITY_MAP.get(scenario_key, folder)
        timed_out_path   = timed_out_dir / timed_out_folder / "output" / "FlowFM_0000_his.nc"
        if timed_out_path.exists():
            his_paths.append(timed_out_path)

    main_his_path = model_location / "output" / "FlowFM_0000_his.nc"
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

# ── CACHE DIR ─────────────────────────────────────────────────────────────────
cache_dir = base_path / "cached_data"
cache_dir.mkdir(exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
scenario_data = {}

for scenario_dir, his_file_paths in run_his_paths.items():
    scenario_name = Path(scenario_dir).name
    scenario_num  = scenario_dir.split('_')[0]
    run_id        = '_'.join(scenario_name.split('_')[1:])
    cache_file    = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

    _, result = load_and_cache_scenario(
        scenario_dir=scenario_dir, his_file_paths=his_file_paths,
        cache_file=cache_file, boxes=boxes, var_name=sed_var,
    )
    _, result_dis = load_and_cache_scenario(
        scenario_dir=scenario_dir, his_file_paths=his_file_paths,
        cache_file=cache_file, boxes=boxes, var_name=dis_var,
    )
    result[dis_var] = result_dis[dis_var]
    scenario_data[scenario_dir] = result

# ── HELPERS ───────────────────────────────────────────────────────────────────
def tidal_avg(arr, window):
    """Centred moving average over `window` timesteps."""
    return np.convolve(arr, np.ones(window) / window, mode='same')


# ── PRE-PROCESS: trim spinup, smooth, extract Q and dV/dt per scenario ────────
window       = int(TIDAL_WINDOW_HOURS / DT_HOURS)

processed = {}

for scenario_dir, data in scenario_data.items():
    scenario_key = str(int(scenario_dir.split('_')[0]))

    km_positions   = data['km_positions']
    buffer_volumes = data['buffer_volumes']
    Q_all          = data[dis_var]

    idx_river = np.argmin(np.abs(km_positions - RIVER_KM))

    # Trim spinup
    Q_raw = Q_all[SPINUP_STEPS:, idx_river]
    bv    = {k: v[SPINUP_STEPS:] for k, v in buffer_volumes.items()}

    # Tidal-smooth Q; keep raw dV/dt (diff of buffer)
    Q_smooth = tidal_avg(Q_raw, window)


    print(f"Scenario {scenario_key} ({SCENARIO_LABELS.get(scenario_key, '')}): "
          f"{len(peaks)} peaks detected")

    # Compute dV/dt per section (tidal-smoothed)
    dVdt = {}
    for box_key, buf in bv.items():
        raw_dvdt      = np.diff(buf)
        dVdt[box_key] = tidal_avg(raw_dvdt, window)

    processed[scenario_key] = dict(
        label        = SCENARIO_LABELS.get(scenario_key, scenario_dir),
        color        = SCENARIO_COLORS.get(scenario_key, 'grey'),
        Q_smooth     = Q_smooth,
        dVdt         = dVdt,
        km_positions = km_positions,
    )


# ── PLOT C: CCF validation — Flashy and Single peak only ─────────────────────
# Cross-correlation of tidal-averaged Q anomaly vs tidal-averaged dV/dt anomaly,
# following Wang et al. (2025) Eq. 3 but on hourly data after heavy smoothing.
# Restricted to event-driven scenarios where CCF is interpretable.
# The first significant positive peak after lag=0 is the reported CCF lag.
#
# Significance threshold: |ρ| > 2/sqrt(N_eff)
# where N_eff = N / smooth_steps (conservative for autocorrelated series)

from scipy.signal import correlate as _correlate, correlation_lags as _corr_lags
from scipy.signal import find_peaks as _find_peaks2

CCF_SCENARIOS         = {'1': 'Constant', '2': 'Seasonal', '3': 'Flashy', '4': 'Single peak'}
CCF_SMOOTH_DAYS       = 5      # additional smoothing beyond tidal avg (days)
CCF_MAX_LAG_DAYS      = 90     # max lag to display (days)
CCF_MIN_PEAK_LAG_DAYS = 1      # ignore peaks within first day (tidal artefact)

ccf_smooth_steps   = int(CCF_SMOOTH_DAYS       * 24 / DT_HOURS)
ccf_max_lag_steps  = int(CCF_MAX_LAG_DAYS      * 24 / DT_HOURS)
ccf_min_peak_steps = int(CCF_MIN_PEAK_LAG_DAYS * 24 / DT_HOURS)


def compute_ccf(Q_river, dV_dt, smooth_steps, max_lag_steps, min_peak_steps):
    """
    Normalised CCF (Wang et al. 2025, Eq. 3) on additionally smoothed series.
    Returns lags_days, ccf_vals, ccf_lag (days or None), sig_thresh.
    """
    Q_s  = tidal_avg(Q_river, smooth_steps)
    dV_s = tidal_avg(dV_dt,   smooth_steps)

    Q_anom  = Q_s  - Q_s.mean()
    dV_anom = dV_s - dV_s.mean()

    n = min(len(Q_anom), len(dV_anom))
    Q_anom  = Q_anom[:n]
    dV_anom = dV_anom[:n]

    denom = np.sqrt(np.sum(Q_anom**2) * np.sum(dV_anom**2))
    if denom == 0:
        return None, None, None, None

    corr  = _correlate(dV_anom, Q_anom, mode='full') / denom
    lags  = _corr_lags(n, n, mode='full')

    mask      = (lags >= 0) & (lags <= max_lag_steps)
    lags_days = lags[mask] * DT_HOURS / 24.0
    ccf_vals  = corr[mask]

    # Conservative significance: N_eff = N / smooth_steps
    N_eff      = max(n // smooth_steps, 3)
    sig_thresh = 2.0 / np.sqrt(N_eff)

    # First significant local maximum in |ρ| beyond min_peak_steps
    # Using local maxima avoids small wiggles; 'first' avoids picking a
    # late large peak when an earlier real signal exists.
    valid_mask = (lags[mask] >= min_peak_steps)
    valid_idx  = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return lags_days, ccf_vals, None, None, sig_thresh

    abs_ccf_valid = np.abs(ccf_vals[valid_idx])

    # Find local maxima in |ρ| among valid lags
    local_max_idx, _ = _find_peaks2(abs_ccf_valid)

    if len(local_max_idx) == 0:
        return lags_days, ccf_vals, None, None, sig_thresh

    # Keep only those exceeding significance threshold
    sig_local = local_max_idx[abs_ccf_valid[local_max_idx] > sig_thresh]

    if len(sig_local) == 0:
        ccf_lag  = None
        ccf_sign = None
    else:
        # First significant local maximum (smallest lag)
        best_local = valid_idx[sig_local[0]]
        ccf_lag    = lags_days[best_local]
        ccf_sign   = 'dep' if ccf_vals[best_local] > 0 else 'ero'

    return lags_days, ccf_vals, ccf_lag, ccf_sign, sig_thresh



# ── PRINT WORKFLOW SUMMARY ────────────────────────────────────────────────────
print()
print("=" * 70)
print("  CCF LAG ANALYSIS — MATHEMATICAL WORKFLOW")
print("=" * 70)
print()
print("  Signals used:")
print(f"    Q_river(t)  : tidal-averaged discharge at km {RIVER_KM} [m³/s]")
print( "    dV/dt(t)    : tidal-averaged sediment buffer change rate [m³/timestep]")
print()
print("  Step 1 — Smoothing")
print(f"    Both signals smoothed with {CCF_SMOOTH_DAYS}-day centred moving average")
print( "    (on top of 24 h tidal average) to suppress residual tidal variance.")
print()
print("  Step 2 — Anomaly (mean removal)")
print( "    Q'(t)      = Q(t)      - mean(Q)")
print( "    (dV/dt)'(t) = dV/dt(t) - mean(dV/dt)")
print( "    Required by the Pearson correlation definition (Wang et al. Eq. 3).")
print()
print("  Step 3 — Normalised cross-correlation")
print()
print("             sum_t [ Q'(t) * (dV/dt)'(t + tau) ]")
print("  rho(tau) = ──────────────────────────────────────────────────────")
print("             sqrt( sum_t [Q'(t)]²  *  sum_t [(dV/dt)'(t)]² )")
print()
print("    tau = lag in hourly timesteps (0, 1, 2, ...)")
print("    rho in [-1, 1]; positive = Q and dV/dt co-vary at this lag")
print()
print("  Step 4 — Lag detection")
print( "    Delta_t = lag of first significant local maximum in |rho(tau)|")
print(f"    Significance threshold: |rho| > 2 / sqrt(N_eff)")
print(f"    N_eff = N / smooth_steps  (conservative for autocorrelated series)")
print()
print("  Step 5 — Sign interpretation")
print( "    rho(Delta_t) > 0  →  deposition follows Q peak   (dep)")
print( "    rho(Delta_t) < 0  →  erosion follows Q peak       (ero)")
print()
print("  Static CCF  : computed over full simulation → one Delta_t per section")
print(f"  Sliding CCF : non-overlapping {SW_WINDOW_DAYS}-day windows → Delta_t(year)")
print("=" * 70)
print()

# ── one figure per scenario, sections 20-25 to 40-45 km across columns ──────
for scenario_key, scenario_name in CCF_SCENARIOS.items():
    d = processed.get(scenario_key)
    num_boxes = len(boxes)

    fig, axes = plt.subplots(1, num_boxes,
                             figsize=(4.5 * num_boxes, 4.5),
                             sharey=True)
    fig.suptitle(
        f'CCF validation — Scenario: {scenario_name}\n'
        f'(tidal-avg + {CCF_SMOOTH_DAYS}-day smooth,  max lag {CCF_MAX_LAG_DAYS} d)',
        fontsize=13
    )

    for i, (box_start, box_end) in enumerate(boxes):
        ax = axes[i]

        if d is None:
            ax.set_title(f'{box_start}–{box_end} km')
            ax.text(0.5, 0.5, 'Scenario not found', transform=ax.transAxes,
                    ha='center', va='center', color='grey')
            continue

        dV_dt = d['dVdt'].get((box_start, box_end))
        if dV_dt is None:
            ax.set_title(f'{box_start}–{box_end} km')
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='grey')
            continue

        lags_days, ccf_vals, ccf_lag, ccf_sign, sig_thresh = compute_ccf(
            Q_river        = d['Q_smooth'],
            dV_dt          = dV_dt,
            smooth_steps   = ccf_smooth_steps,
            max_lag_steps  = ccf_max_lag_steps,
            min_peak_steps = ccf_min_peak_steps,
        )

        if lags_days is None:
            ax.set_title(f'{box_start}–{box_end} km')
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', color='grey')
            continue

        ax.plot(lags_days, ccf_vals, color=d['color'], linewidth=2)
        ax.axhline(0,           color='black', lw=0.8, ls=':')
        ax.axhline( sig_thresh, color='grey',  lw=1.0, ls='--',
                    label=f'sig. (±{sig_thresh:.3f})')
        ax.axhline(-sig_thresh, color='grey',  lw=1.0, ls='--')
        ax.axvline(0,           color='black', lw=1.2, ls='--')



        ax.set_title(f'{box_start}–{box_end} km', fontsize=11)
        ax.set_xlabel('Lag (days)', fontsize=9)
        if i == 0:
            ax.set_ylabel('Normalised ρ (–)', fontsize=9)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        output_dir / f"CCF_storyboard_{scenario_key}_{scenario_name.replace(' ', '_')}.png",
        dpi=300, bbox_inches='tight'
    )
    plt.show()
    print(f"  Saved: CCF_storyboard_{scenario_key}_{scenario_name}.png")

print("\nAll plots (composite + CCF) saved to:", output_dir)


# ── PLOT D: SLIDING WINDOW CCF ────────────────────────────────────────────────
# For each section and scenario, compute the CCF in a moving window across the
# full simulation. At each window position, extract the dominant lag (global
# argmax |rho|). Plot lag vs simulation time to reveal whether response time
# changes over the morphological evolution of the estuary.
#
# Output: one figure per scenario, sections across columns (same layout as B/C)
# Filename: his_alongestuary_sliding_CCF_dVdt_Qriver_{scenario}.png

SW_WINDOW_DAYS   = 365          # one full hydrodynamic year per window
SW_STEP_DAYS     = 365          # step = window width → non-overlapping yearly windows
SW_MAX_LAG_DAYS  = 90           # max lag searched inside each window
SW_MIN_LAG_DAYS  = 1            # exclude sub-daily tidal artefact
SW_SMOOTH_DAYS   = 5            # smoothing before CCF inside each window

sw_window_steps  = int(SW_WINDOW_DAYS  * 24 / DT_HOURS)
sw_step_steps    = int(SW_STEP_DAYS    * 24 / DT_HOURS)
sw_max_lag_steps = int(SW_MAX_LAG_DAYS * 24 / DT_HOURS)
sw_min_lag_steps = int(SW_MIN_LAG_DAYS * 24 / DT_HOURS)
sw_smooth_steps  = int(SW_SMOOTH_DAYS  * 24 / DT_HOURS)


def sliding_ccf_lag(Q, dV_dt, window_steps, step_steps,
                    max_lag_steps, min_lag_steps, smooth_steps):
    """
    Slide a window across the full timeseries. In each window:
      1. Smooth both signals
      2. Remove mean (anomaly)
      3. Compute normalised CCF (Wang et al. Eq. 3)
      4. Find dominant lag = global argmax(|rho|) beyond min_lag_steps
      5. Record lag, sign, and peak |rho| as quality indicator

    Returns:
        win_centers : centre time of each window (timestep index)
        lags_days   : dominant lag in each window (days), NaN if not significant
        peak_rho    : |rho| at dominant lag (quality indicator)
        signs       : +1 deposition, -1 erosion, 0 not significant
    """
    n = min(len(Q), len(dV_dt))
    starts = np.arange(0, n - window_steps, step_steps)

    win_centers = []
    lags_days   = []
    peak_rho    = []
    signs       = []

    for s in starts:
        e = s + window_steps
        Q_win  = tidal_avg(Q[s:e],     smooth_steps)
        dV_win = tidal_avg(dV_dt[s:e], smooth_steps)

        Q_anom  = Q_win  - Q_win.mean()
        dV_anom = dV_win - dV_win.mean()

        denom = np.sqrt(np.sum(Q_anom**2) * np.sum(dV_anom**2))
        if denom < 1e-12:
            win_centers.append(s + window_steps // 2)
            lags_days.append(np.nan)
            peak_rho.append(np.nan)
            signs.append(0)
            continue

        corr = _correlate(dV_anom, Q_anom, mode='full') / denom
        lags = _corr_lags(len(Q_anom), len(dV_anom), mode='full')

        # Positive lags only, within max_lag, beyond min_lag
        valid = (lags >= min_lag_steps) & (lags <= max_lag_steps)
        if not np.any(valid):
            win_centers.append(s + window_steps // 2)
            lags_days.append(np.nan)
            peak_rho.append(np.nan)
            signs.append(0)
            continue

        # Significance threshold for this window
        N_eff      = max(window_steps // smooth_steps, 3)
        sig_thresh = 2.0 / np.sqrt(N_eff)

        # First significant local maximum in |rho| — same logic as compute_ccf
        abs_corr_valid = np.abs(corr[valid])
        local_max_idx, _ = find_peaks(abs_corr_valid)

        best_lag  = None
        best_rho  = None
        if len(local_max_idx) > 0:
            sig_local = local_max_idx[abs_corr_valid[local_max_idx] > sig_thresh]
            if len(sig_local) > 0:
                first = sig_local[0]          # first significant local max
                best_lag = lags[valid][first]
                best_rho = corr[valid][first]

        win_centers.append(s + window_steps // 2)
        if best_lag is not None:
            lags_days.append(best_lag * DT_HOURS / 24.0)
            peak_rho.append(np.abs(best_rho))
            signs.append(1 if best_rho > 0 else -1)
        else:
            lags_days.append(np.nan)
            peak_rho.append(np.nanmax(abs_corr_valid) if len(abs_corr_valid) else np.nan)
            signs.append(0)

    return (np.array(win_centers),
            np.array(lags_days),
            np.array(peak_rho),
            np.array(signs))


# ── one figure per scenario, sections across columns ─────────────────────────
for scenario_key, d in processed.items():
    num_boxes = len(boxes)

    fig, axes = plt.subplots(2, num_boxes,
                             figsize=(4.5 * num_boxes, 7),
                             sharex=True)
    fig.suptitle(
        f"Scenario: {d['label']} — sliding-window CCF lag  Q\u2192dV/dt\n"
        f"(window={SW_WINDOW_DAYS}d, step={SW_STEP_DAYS}d, "
        f"max lag={SW_MAX_LAG_DAYS}d, smooth={SW_SMOOTH_DAYS}d)",
        fontsize=13
    )

    for i, (box_start, box_end) in enumerate(boxes):
        ax_lag = axes[0, i]   # top row: lag over time
        ax_rho = axes[1, i]   # bottom row: |rho| quality

        dV_dt = d['dVdt'].get((box_start, box_end))
        if dV_dt is None:
            ax_lag.set_title(f'{box_start}\u2013{box_end} km')
            continue

        win_centers, lags, rho, signs = sliding_ccf_lag(
            Q            = d['Q_smooth'],
            dV_dt        = dV_dt,
            window_steps = sw_window_steps,
            step_steps   = sw_step_steps,
            max_lag_steps= sw_max_lag_steps,
            min_lag_steps= sw_min_lag_steps,
            smooth_steps = sw_smooth_steps,
        )

        # Convert window centres to years (relative to start)
        time_years = win_centers * DT_HOURS / 24.0 / 365.25

        # Colour points by dep/ero/ns
        dep_mask = signs ==  1
        ero_mask = signs == -1
        ns_mask  = signs ==  0

        if np.any(dep_mask):
            ax_lag.scatter(time_years[dep_mask], lags[dep_mask],
                           s=18, color=d['color'], alpha=0.8,
                           label='dep', zorder=3)
        if np.any(ero_mask):
            ax_lag.scatter(time_years[ero_mask], lags[ero_mask],
                           s=18, color=d['color'], alpha=0.8,
                           marker='v', label='ero', zorder=3)
        if np.any(ns_mask):
            ax_lag.scatter(time_years[ns_mask],
                           np.zeros(ns_mask.sum()),   # plot at y=0 as placeholder
                           s=8, color='lightgrey', alpha=0.5,
                           label='n.s.', zorder=2)

        ax_rho.plot(time_years, rho, color=d['color'], lw=1.2, alpha=0.7)

        # Significance threshold line on rho panel
        N_eff      = max(sw_window_steps // sw_smooth_steps, 3)
        sig_thresh = 2.0 / np.sqrt(N_eff)
        ax_rho.axhline(sig_thresh, color='grey', lw=1, ls='--',
                       label=f'sig. ({sig_thresh:.3f})')

        ax_lag.set_title(f'{box_start}\u2013{box_end} km', fontsize=11)
        ax_lag.set_ylim(0, SW_MAX_LAG_DAYS)
        ax_lag.set_ylabel('Lag (days)', fontsize=8)
        ax_lag.legend(fontsize=7, loc='upper right')
        ax_lag.grid(alpha=0.2)

        ax_rho.set_xlabel('Simulation time (years)', fontsize=8)
        ax_rho.set_ylabel('Peak |\u03c1|', fontsize=8)
        ax_rho.legend(fontsize=7, loc='upper right')
        ax_rho.grid(alpha=0.2)

    fig.tight_layout()
    fname = (f"his_alongestuary_sliding_CCF_dVdt_Qriver_"
             f"{scenario_key}_{d['label'].replace(' ', '_')}.png")
    fig.savefig(output_dir / fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {fname}")

print("\nAll sliding-window CCF plots saved to:", output_dir)