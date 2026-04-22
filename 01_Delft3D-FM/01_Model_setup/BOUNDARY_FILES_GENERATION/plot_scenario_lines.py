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
TOTAL_Q    = 500          # m³/s  – used only for labels and output filename
BASE_DIR   = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian"
    r"\Model_Input\Q500"
)
OUTPUT_DIR  = BASE_DIR / "plots_river_bct"
OUTPUT_FILE = OUTPUT_DIR / f"scenario_lines_Q{TOTAL_Q}_{STYLE}.png"

# --- Figure style ---
STYLE = 'default'   # 'default'   →  white background, black text
                    # 'whitefig'  →  transparent figure, white axes background, white text
                    # 'transparent_white' →  transparent figure, white axes background, black text
STYLES = {
    'default': {},
    'transparent_white': {
        'figure.facecolor':    'none',
        'axes.facecolor':      'white',
        'savefig.transparent': True,
    },
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
# Create figure – one panel per n_peaks value
# ============================================================================
n_panels = len(all_n_peaks)
PANEL_W  = 4.0
PANEL_H  = 3.0

fig, axes = plt.subplots(
    1, n_panels,
    figsize=(PANEL_W * n_panels, PANEL_H),
    sharey=True,
    sharex=False,
)

if n_panels == 1:
    axes = [axes]

# ============================================================================
# Fill each panel
# ============================================================================
for ci, n_peaks in enumerate(all_n_peaks):
    ax = axes[ci]

    for peak_ratio in all_peak_ratios:
        key = (peak_ratio, n_peaks)
        if key not in scenarios:
            continue

        info = scenarios[key]
        df = pd.read_csv(info["csv"])
        if "timestamp" not in df.columns or "discharge_m3s" not in df.columns:
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        first_year = df["timestamp"].dt.year.min()
        df_yr = df[df["timestamp"].dt.year == first_year].copy()

        pr_label = (f"{int(peak_ratio)}"
                    if peak_ratio == int(peak_ratio)
                    else f"{peak_ratio}")

        ax.plot(
            df_yr["timestamp"],
            df_yr["discharge_m3s"],
            color=RATIO_COLOR[peak_ratio],
            linewidth=1.2,
            label=f"$R_{{\\mathrm{{peak}}}}$ = {pr_label}",
        )

    ax.set_ylim(global_ylim)
    ax.grid(True, alpha=0.22, linewidth=0.5)

    # x-axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.tick_params(axis="x", labelsize=FONTSIZE_TICKS, rotation=40)

    # Panel title
    ax.set_title(
        f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}",
        fontsize=FONTSIZE_TITLE, fontweight="bold", pad=5, color=_tc,
    )

    # y-axis label only on leftmost panel
    if ci == 0:
        ax.set_ylabel("Q [m³/s]", fontsize=FONTSIZE_LABELS, labelpad=4)
        ax.tick_params(axis="y", labelsize=FONTSIZE_TICKS)

# ============================================================================
# Shared legend – build from the consistent color mapping
# ============================================================================
legend_handles = []
for peak_ratio in all_peak_ratios:
    pr_label = (f"{int(peak_ratio)}"
                if peak_ratio == int(peak_ratio)
                else f"{peak_ratio}")
    legend_handles.append(
        mlines.Line2D(
            [], [],
            color=RATIO_COLOR[peak_ratio],
            linewidth=1.8,
            label=f"$R_{{\\mathrm{{peak}}}}$ = {pr_label}",
        )
    )

fig.legend(
    handles=legend_handles,
    title="Peak / mean ratio",
    title_fontsize=FONTSIZE_TICKS,
    fontsize=FONTSIZE_TICKS,
    loc="lower center",
    ncol=len(all_peak_ratios),
    bbox_to_anchor=(0.5, -0.18),
    frameon=True,
)

fig.suptitle(
    f"River discharge scenarios  (Q = {TOTAL_Q} m³/s)",
    fontsize=FONTSIZE_TITLE, fontweight="bold", y=1.02, color=_tc,
)

fig.tight_layout()

# ============================================================================
# Save
# ============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight",
            transparent=plt.rcParams.get('savefig.transparent', False))
plt.show(fig)
print(f"\nSaved to:\n  {OUTPUT_FILE}")
#%%
