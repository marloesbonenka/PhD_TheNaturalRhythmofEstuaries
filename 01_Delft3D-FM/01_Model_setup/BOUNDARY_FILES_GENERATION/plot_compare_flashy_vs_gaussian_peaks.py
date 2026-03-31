#%%
"""
Compare individual peaks from the original 'flashy' discharge scenarios (CSV)
with a Gaussian-generated scenario (peak/mean = 3, n_peaks = 5).

For each of the N_PEAKS peaks a 10-day zoomed window is shown, allowing direct
side-by-side comparison of peak shape, width, and magnitude.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# ── 1. Configuration ───────────────────────────────────────────────────────────
BASE_DIR = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models"
    r"\1_RiverDischargeVariability_domain45x15\Model_Input"
)
OUTPUT_DIR = BASE_DIR / "plots_river_bct"

# Discharge folder to compare (sets Q_mean for the Gaussian to the same value)
TARGET_DISCHARGE = "1000"       # e.g. "250", "500", "1000"

# Gaussian scenario parameters
GAUSSIAN_PEAK_MEAN_RATIO = 3.0  # peak / mean  (P/M)
N_PEAKS = 5                     # number of Gaussian pulses

ZOOM_HALF_WINDOW = 5.0          # ± days around each peak  → 10-day total window

SAVE_FIG = True
FIG_DPI = 300

# Colours consistent with FUNCS_plot_discharge_scenarios.py
COLOR_FLASHY   = '#009E73'   # teal-green
COLOR_GAUSSIAN = '#D55E00'   # red-orange

# ── 2. Generate Gaussian discharge series ──────────────────────────────────────
days          = 365
Q_mean_target = float(TARGET_DISCHARGE)
Q_base        = Q_mean_target * 0.8          # 80 % base-flow ratio (same as script)
V_total_excess = (Q_mean_target - Q_base) * days

Q_peak_gauss = Q_mean_target * GAUSSIAN_PEAK_MEAN_RATIO
A = Q_peak_gauss - Q_base
if A <= 0:
    raise ValueError("Gaussian peak discharge must exceed base discharge (A > 0).")

V_event = V_total_excess / N_PEAKS
sigma   = V_event / (A * np.sqrt(2 * np.pi))

t_gauss = np.arange(days, dtype=float)          # integer days 0 … 364
segment = days / N_PEAKS
gauss_event_centers = np.linspace(segment / 2, days - segment / 2, N_PEAKS)

q_gauss = np.full(days, Q_base, dtype=float)
for t0 in gauss_event_centers:
    q_gauss += A * np.exp(-(t_gauss - t0) ** 2 / (2 * sigma ** 2))

# ── 3. Load flashy CSV(s) for the target discharge ────────────────────────────
def _find_flashy_csvs(base_dir: Path, discharge_val: str) -> dict:
    """Return {scenario_name: Path} for all flashy scenarios under Q<discharge_val>."""
    discharge_folder = base_dir / f"Q{discharge_val}"
    if not discharge_folder.is_dir():
        raise FileNotFoundError(f"Discharge folder not found: {discharge_folder}")

    csvs = {}
    # Check timed_out first (preferred; same priority logic as main script)
    timed_out = discharge_folder / "timed_out"
    for root in [timed_out, discharge_folder]:
        if not root.is_dir():
            continue
        for scenario_folder in sorted(root.iterdir()):
            if scenario_folder.is_dir() and "flashy" in scenario_folder.name.lower():
                csv_path = scenario_folder / "boundaryfiles_csv" / "discharge_cumulative.csv"
                if csv_path.exists() and scenario_folder.name not in csvs:
                    csvs[scenario_folder.name] = csv_path
    return csvs


flashy_csvs = _find_flashy_csvs(BASE_DIR, TARGET_DISCHARGE)
if not flashy_csvs:
    raise FileNotFoundError(
        f"No flashy scenario CSV found for Q={TARGET_DISCHARGE} under {BASE_DIR}"
    )

# Parse each flashy CSV – keep only first simulation year, resample to daily
flashy_series_daily = {}
for name, csv_path in flashy_csvs.items():
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    first_year = df["timestamp"].min().year
    df = df[df["timestamp"].dt.year == first_year].set_index("timestamp").sort_index()
    # Resample to daily mean for consistent comparison with the Gaussian (daily)
    df_daily = df["discharge_m3s"].resample("D").mean().dropna()
    df_daily.index = (df_daily.index - df_daily.index[0]).days  # integer day 0..364
    flashy_series_daily[name] = df_daily

# Use the first (or only) flashy series
flash_name, flash_daily = next(iter(flashy_series_daily.items()))
t_vals = flash_daily.index.to_numpy(dtype=float)
q_vals = flash_daily.to_numpy(dtype=float)

# ── 4. Detect the top N_PEAKS peaks in the flashy series ──────────────────────
# minimum separation: half of expected inter-peak spacing
min_sep = max(1, int(days / (N_PEAKS * 2)))
flashy_peak_indices, _ = find_peaks(q_vals, distance=min_sep)

if len(flashy_peak_indices) == 0:
    raise RuntimeError(
        "No peaks detected in the flashy discharge series. "
        "Check that the CSV contains the expected flood pulses."
    )

# Keep top N_PEAKS by amplitude, then re-sort chronologically
if len(flashy_peak_indices) > N_PEAKS:
    ranked = np.argsort(q_vals[flashy_peak_indices])[::-1][:N_PEAKS]
    flashy_peak_indices = np.sort(flashy_peak_indices[ranked])

n_panels = min(len(flashy_peak_indices), N_PEAKS)

# ── 5. Build comparison figure ────────────────────────────────────────────────
fig, axes = plt.subplots(
    1, n_panels,
    figsize=(3.5 * n_panels, 4.5),
    sharey=True,
)
if n_panels == 1:
    axes = [axes]

for i, ax in enumerate(axes):

    # ── Original flashy peak ──
    fp_center = float(t_vals[flashy_peak_indices[i]])
    mask_f    = np.abs(t_vals - fp_center) <= ZOOM_HALF_WINDOW
    t_rel_f   = t_vals[mask_f] - fp_center
    q_rel_f   = q_vals[mask_f]

    ax.plot(t_rel_f, q_rel_f,
            color=COLOR_FLASHY, lw=2.5, label="Flashy (original)")
    ax.scatter([0.0], [q_vals[flashy_peak_indices[i]]],
               color=COLOR_FLASHY, s=50, zorder=6)

    # ── Gaussian peak ──
    gp_center  = gauss_event_centers[i]
    mask_g     = np.abs(t_gauss - gp_center) <= ZOOM_HALF_WINDOW
    t_rel_g    = t_gauss[mask_g] - gp_center
    q_rel_g    = q_gauss[mask_g]
    gauss_peak = np.max(q_rel_g)          # max within the window = peak value

    ax.plot(t_rel_g, q_rel_g,
            color=COLOR_GAUSSIAN, lw=2.5,
            label=f"Gaussian (P/M={GAUSSIAN_PEAK_MEAN_RATIO:.0f}, n={N_PEAKS})")
    ax.scatter([t_rel_g[np.argmax(q_rel_g)]], [gauss_peak],
               color=COLOR_GAUSSIAN, s=50, zorder=6)

    # Reference mean line
    ax.axhline(Q_mean_target, color="gray", lw=0.9, ls=":", alpha=0.7, label=f"Q_mean = {Q_mean_target:.0f} m³/s")

    # Vertical marker at peak reference
    ax.axvline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)

    # ── Decoration ──
    ax.set_xlim(-ZOOM_HALF_WINDOW, ZOOM_HALF_WINDOW)
    ax.set_xticks(np.arange(-int(ZOOM_HALF_WINDOW), int(ZOOM_HALF_WINDOW) + 1))
    ax.set_xlabel("days relative to peak")
    if i == 0:
        ax.set_ylabel("discharge [m³/s]")
    peak_day_label = int(np.round(fp_center)) + 1     # 1-based day-of-year
    ax.set_title(
        f"Peak {i + 1}  (day {peak_day_label})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    if i == n_panels - 1:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

fig.suptitle(
    f"Peak comparison — Flashy (original) vs. Gaussian "
    f"(P/M = {GAUSSIAN_PEAK_MEAN_RATIO:.0f}, n = {N_PEAKS})  |  "
    f"Q = {TARGET_DISCHARGE} m³/s",
    fontsize=12,
    y=1.01,
)
fig.tight_layout()

if SAVE_FIG:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"compare_flashy_vs_gaussian_peaks_Q{TARGET_DISCHARGE}.png"
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")

plt.show()
#%%
