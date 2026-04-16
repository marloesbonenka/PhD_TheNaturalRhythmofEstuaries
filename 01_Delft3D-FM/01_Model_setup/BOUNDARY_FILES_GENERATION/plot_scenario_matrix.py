"""
Visualization: Scenario matrix plot
Author : Marloes Bonenkamp
Date   : April 2026

Creates a matrix of subplots organised by:
  - columns  : number of peaks per year  (n_peaks),  increasing left → right
  - rows     : peak / mean ratio         (peak_ratio), increasing bottom → top (stacked)

Each panel shows the first simulation year of the river discharge time series.

Input   : u:/…/Q500/  (scans sub-folders automatically)
Output  : u:/…/Q500/plots_river_bct/scenario_matrix_Q500.png
"""
#%%
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
OUTPUT_FILE = OUTPUT_DIR / f"scenario_matrix_Q{TOTAL_Q}.png"

# ============================================================================
# Discover scenario folders and parse peak_ratio / n_peaks
# ============================================================================
# Matches folder endings like  _pm3_n4  or  _pm1.5_n2
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
all_n_peaks     = sorted({k[1] for k in scenarios})   # columns: left → right
all_peak_ratios = sorted({k[0] for k in scenarios})   # rows:    bottom → top

n_cols = len(all_n_peaks)
n_rows = len(all_peak_ratios)

# matplotlib row 0 = top → reverse so highest ratio ends up at the top
row_order = list(reversed(all_peak_ratios))

print(f"  n_peaks     axis : {all_n_peaks}")
print(f"  peak_ratio  axis : {all_peak_ratios}")

# ============================================================================
# Pre-compute global y-axis limits across all valid scenarios
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

ypad_global = max((global_ymax - global_ymin) * 0.08, 1.0)
global_ylim = (global_ymin - ypad_global, global_ymax + ypad_global)

# ============================================================================
# Create figure
# ============================================================================
PANEL_W = 3.4
PANEL_H = 2.4

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(PANEL_W * n_cols, PANEL_H * n_rows),
    sharex=False,
    sharey=False,
)

# Normalise to always be a 2-D numpy array of axes
axes = np.atleast_2d(axes)
if n_rows == 1 and n_cols > 1:
    axes = axes                          # already (1, n_cols)
elif n_cols == 1 and n_rows > 1:
    axes = axes                          # already (n_rows, 1)

# ============================================================================
# Consistent color map: one fixed color per peak_ratio value
# (same palette as plot_scenario_lines.py)
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
# Fill each panel
# ============================================================================
for ri, peak_ratio in enumerate(row_order):
    for ci, n_peaks in enumerate(all_n_peaks):
        ax  = axes[ri, ci]
        key = (peak_ratio, n_peaks)

        if key not in scenarios:
            # No scenario for this (peak_ratio, n_peaks) combination
            ax.set_visible(False)
            continue

        info = scenarios[key]

        # -- Load CSV and extract first year --------------------------------
        df = pd.read_csv(info["csv"])
        if "timestamp" not in df.columns or "discharge_m3s" not in df.columns:
            ax.text(0.5, 0.5, "missing columns",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=7)
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        first_year = df["timestamp"].dt.year.min()
        df_yr = df[df["timestamp"].dt.year == first_year].copy()

        # -- Plot ------------------------------------------------------------
        ax.plot(df_yr["timestamp"], df_yr["discharge_m3s"],
                color=RATIO_COLOR[peak_ratio], linewidth=0.9)
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.set_ylim(global_ylim)

        # Scenario number label (top-left)
        scenario_num = info["name"].split("_")[0]
        ax.text(0.03, 0.93, f"S{int(scenario_num)}",
                transform=ax.transAxes, fontsize=7,
                ha="left", va="top", color="0.45")

        # -- x-axis: show month labels only on bottom row -------------------
        if ri == n_rows - 1:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.tick_params(axis="x", labelsize=7, rotation=40)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

        # -- y-axis: show tick labels only on left column -------------------
        if ci == 0:
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel("Q [m³/s]", fontsize=7, labelpad=3)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

# ============================================================================
# Column headers: n_peaks value at the very top of each column
# ============================================================================
for ci, n_peaks in enumerate(all_n_peaks):
    # Use the top-most visible panel in this column to host the title
    for ri in range(n_rows):
        if axes[ri, ci].get_visible():
            axes[ri, ci].set_title(
                f"$n_{{\\mathrm{{peaks}}}}$ = {n_peaks}",
                fontsize=9, fontweight="bold", pad=5,
            )
            break

# ============================================================================
# Row labels: peak/mean ratio on the right-hand side of each row
# ============================================================================
for ri, peak_ratio in enumerate(row_order):
    # Use the right-most visible panel in this row
    for ci in range(n_cols - 1, -1, -1):
        if axes[ri, ci].get_visible():
            pr_str = (f"{int(peak_ratio)}"
                      if peak_ratio == int(peak_ratio)
                      else f"{peak_ratio}")
            axes[ri, ci].yaxis.set_label_position("right")
            axes[ri, ci].set_ylabel(
                f"$R_{{\\mathrm{{peak}}}}$ = {pr_str}",
                rotation=270, labelpad=15,
                fontsize=9, fontweight="bold",
            )
            break

# ============================================================================
# Figure-level axis descriptions
# ============================================================================
fig.text(
    0.5, 0.005,
    "Number of peaks per year   →",
    ha="center", fontsize=10, style="italic", color="0.35",
)
fig.text(
    0.005, 0.5,
    "Peak / mean ratio   →",
    va="center", fontsize=10, style="italic", color="0.35",
    rotation="vertical",
)

fig.suptitle(
    f"River discharge scenarios  (Q = {TOTAL_Q} m³/s)",
    fontsize=13, fontweight="bold", y=1.01,
)

fig.tight_layout(rect=[0.03, 0.03, 1.0, 1.0])

# ============================================================================
# Save
# ============================================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight", transparent=True)
plt.show(fig)
print(f"\nSaved to:\n  {OUTPUT_FILE}")
#%%