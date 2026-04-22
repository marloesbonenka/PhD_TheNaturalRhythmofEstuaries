"""
Visualization: Discharge scenarios – one panel per n_peaks, amplitudes as colors
Author : Marloes Bonenkamp
Date   : April 2026

Creates one subplot per number-of-peaks value.  Within each panel all
peak / mean ratios are overlaid, each with a consistent color.

Input   : u:/…/Q500/  (scans sub-folders automatically)
Output  : u:/…/Q500/plots_river_bct/scenario_lines_Q500.png
"""
#%%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines

# ============================================================================
# Configuration
# ============================================================================
# --- Figure style ---
STYLE = 'whitefig'   # 'default'   →  white background, black text
                    # 'whitefig'  →  transparent figure, white axes background, white text
                    # 'transparent_white' →  transparent figure, white axes background, black text


TOTAL_Q    = 500          # m³/s  – used only for labels and output filename
BASE_DIR   = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
    r"\Model_Input\Q500"
)
OUTPUT_DIR  = BASE_DIR / "plots_river_bct"

OUTPUT_FILE = OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_{STYLE}.png"  # base; overridden per plot

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
        'grid.color':          'white',
        'legend.facecolor':    'none',
        'legend.edgecolor':    'white',
        'savefig.transparent': True,
    },
}

# --- Font sizes ---
FONTSIZE_TITLE  = 12
FONTSIZE_LABELS = 9    # axis labels
FONTSIZE_TICKS  = 8    # tick numbers

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])
plt.rcParams.update({
    'axes.titlesize':  FONTSIZE_TITLE,
    'axes.labelsize':  FONTSIZE_LABELS,
    'xtick.labelsize': FONTSIZE_TICKS,
    'ytick.labelsize': FONTSIZE_TICKS,
})
_tc = plt.rcParams['text.color']   # convenience shorthand

# ============================================================================
# Discover scenario folders and parse peak_ratio / n_peaks
# ============================================================================
FOLDER_RE = re.compile(r"_pm(\d+(?:\.\d+)?)_n(\d+)$", re.IGNORECASE)

scenarios: dict[tuple, dict] = {}

for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir():
        continue
    m = FOLDER_RE.search(folder.name)
    if not m:
        continue
    peak_ratio = float(m.group(1))
    n_peaks    = int(m.group(2))
    csv_path   = folder / "boundaryfiles_csv" / "discharge_cumulative.csv"
    if not csv_path.exists():
        print(f"  WARNING – CSV not found for '{folder.name}', skipping.")
        continue
    scenarios[(peak_ratio, n_peaks)] = {
        "name": folder.name,
        "csv":  csv_path,
    }

if not scenarios:
    raise FileNotFoundError(f"No valid scenario folders found under:\n  {BASE_DIR}")

print(f"Found {len(scenarios)} scenario(s).")

# ============================================================================
# Build sorted axis values
# ============================================================================
all_n_peaks     = sorted({k[1] for k in scenarios})   # one panel each
all_peak_ratios = sorted({k[0] for k in scenarios})   # one color each

print(f"  n_peaks     axis : {all_n_peaks}")
print(f"  peak_ratio  axis : {all_peak_ratios}")

# ============================================================================
# Consistent color map: one fixed color per peak_ratio value
# ============================================================================
PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # yellow-green
    "#17becf",  # cyan
]

RATIO_COLOR: dict[float, str] = {
    ratio: PALETTE[i % len(PALETTE)]
    for i, ratio in enumerate(all_peak_ratios)
}

# ============================================================================
# Pre-compute global y-axis limits
# ============================================================================
global_ymin = np.inf
global_ymax = -np.inf

for key, info in scenarios.items():
    df_tmp = pd.read_csv(info["csv"])
    df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"])
    first_yr = df_tmp["timestamp"].dt.year.min()
    q = df_tmp[df_tmp["timestamp"].dt.year == first_yr]["discharge_m3s"]
    global_ymin = min(global_ymin, q.min())
    global_ymax = max(global_ymax, q.max())

ypad = max((global_ymax - global_ymin) * 0.08, 1.0)
global_ylim = (global_ymin - ypad, global_ymax + ypad)

# ============================================================================
# Plot 1 – one panel per n_peaks, lines colored by peak_ratio
# ============================================================================
NPEAK_COLOR: dict[int, str] = {
    n: PALETTE[i % len(PALETTE)]
    for i, n in enumerate(all_n_peaks)
}

fig1, axes1 = plt.subplots(1, len(all_n_peaks), figsize=(4 * len(all_n_peaks), 3),
                            sharey=True, sharex=False)
if len(all_n_peaks) == 1:
    axes1 = [axes1]

for ci, n_peaks in enumerate(all_n_peaks):
    ax = axes1[ci]
    for peak_ratio in all_peak_ratios:
        if (peak_ratio, n_peaks) not in scenarios:
            continue
        df = pd.read_csv(scenarios[(peak_ratio, n_peaks)]["csv"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df_yr = df[df["timestamp"].dt.year == df["timestamp"].dt.year.min()].copy()
        pr_label = f"{int(peak_ratio)}" if peak_ratio == int(peak_ratio) else f"{peak_ratio}"
        ax.plot(df_yr["timestamp"], df_yr["discharge_m3s"],
                color=RATIO_COLOR[peak_ratio], linewidth=1.2,
                label=f"$R_{{\\mathrm{{peak}}}}$ = {pr_label}")
    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICKS, rotation=40)
    ax.set_title(f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=5, color=_tc)
    if ci == 0:
        ax.set_ylabel("Q [m³/s]", fontsize=FONTSIZE_LABELS, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICKS)

fig1.legend(
    handles=[mlines.Line2D([], [], color=RATIO_COLOR[r], linewidth=1.8,
                           label=(f"$R_{{\\mathrm{{peak}}}}$ = {int(r)}"
                                  if r == int(r) else f"$R_{{\\mathrm{{peak}}}}$ = {r}"))
             for r in all_peak_ratios],
    title="Peak / mean ratio", title_fontsize=FONTSIZE_TICKS, fontsize=FONTSIZE_TICKS,
    loc="lower center", ncol=len(all_peak_ratios), bbox_to_anchor=(0.5, -0.18), frameon=True,
)
fig1.suptitle(f"River discharge scenarios  (Q = {TOTAL_Q} m³/s)",
              fontsize=FONTSIZE_TITLE, fontweight="bold", y=1.02, color=_tc)
fig1.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
_tr = plt.rcParams.get('savefig.transparent', False)
fig1.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_{STYLE}_by_frequency.png", dpi=200, bbox_inches="tight", transparent=_tr)
fig1.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_{STYLE}_by_frequency.pdf", bbox_inches="tight", transparent=_tr)
plt.show(fig1)
print(f"Saved: scenario_lines_Q{TOTAL_Q}_{STYLE}_by_frequency (.png/.pdf)")

# ============================================================================
# Plot 2 – one panel per peak_ratio, lines colored by n_peaks
# ============================================================================
NPEAK_COLOR: dict[int, str] = {
    n: PALETTE[i % len(PALETTE)]
    for i, n in enumerate(all_n_peaks)
}

fig2, axes2 = plt.subplots(1, len(all_peak_ratios), figsize=(4 * len(all_peak_ratios), 3),
                            sharey=True, sharex=False)
if len(all_peak_ratios) == 1:
    axes2 = [axes2]

for ci, peak_ratio in enumerate(all_peak_ratios):
    ax = axes2[ci]
    for n_peaks in all_n_peaks:
        if (peak_ratio, n_peaks) not in scenarios:
            continue
        df = pd.read_csv(scenarios[(peak_ratio, n_peaks)]["csv"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df_yr = df[df["timestamp"].dt.year == df["timestamp"].dt.year.min()].copy()
        ax.plot(df_yr["timestamp"], df_yr["discharge_m3s"],
                color=NPEAK_COLOR[n_peaks], linewidth=1.2,
                label=f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}")
    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.22, linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICKS, rotation=40)
    pr_label = f"{int(peak_ratio)}" if peak_ratio == int(peak_ratio) else f"{peak_ratio}"
    ax.set_title(f"$R_{{\\mathrm{{peak}}}}$ = {pr_label}",
                 fontsize=FONTSIZE_TITLE, fontweight="bold", pad=5, color=_tc)
    if ci == 0:
        ax.set_ylabel("Q [m³/s]", fontsize=FONTSIZE_LABELS, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICKS)

fig2.legend(
    handles=[mlines.Line2D([], [], color=NPEAK_COLOR[n], linewidth=1.8,
                           label=f"$n_{{\\mathrm{{peaks}}}}$ = {n}")
             for n in all_n_peaks],
    title="Number of peaks", title_fontsize=FONTSIZE_TICKS, fontsize=FONTSIZE_TICKS,
    loc="lower center", ncol=len(all_n_peaks), bbox_to_anchor=(0.5, -0.18), frameon=True,
)
fig2.suptitle(f"River discharge scenarios  (Q = {TOTAL_Q} m³/s)",
              fontsize=FONTSIZE_TITLE, fontweight="bold", y=1.02, color=_tc)
fig2.tight_layout()

fig2.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_{STYLE}_by_amplitude.png", dpi=200, bbox_inches="tight", transparent=_tr)
fig2.savefig(OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_{STYLE}_by_amplitude.pdf", bbox_inches="tight", transparent=_tr)
plt.show(fig2)
print(f"Saved: scenario_lines_Q{TOTAL_Q}_{STYLE}_by_amplitude (.png/.pdf)")
#%%
