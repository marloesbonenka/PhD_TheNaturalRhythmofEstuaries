"""Script to make a conceptual plot of the "threshold space" for 
classifying persistent morphological change based on different hydrological regimes
by assessing the number of peaks and peak/mean ratio"""

#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
#%%
# ── INPUT DATA ────────────────────────────────────────────────────────────────
total_discharge = 500  # m³/s

scenarios = [
    {"name": f"01_Qr{total_discharge}_pm1_n0",  "peak_ratio": 1.0, "n_peaks": 0, "status": "no_change"},
    {"name": f"02_Qr{total_discharge}_pm2_n1",  "peak_ratio": 2,   "n_peaks": 1, "status": "no_change"},
    {"name": f"03_Qr{total_discharge}_pm3_n5",  "peak_ratio": 3,   "n_peaks": 5, "status": "change"},
    {"name": f"04_Qr{total_discharge}_pm3_n1",  "peak_ratio": 3,   "n_peaks": 1, "status": "no_change"},
    {"name": f"05_Qr{total_discharge}_pm5_n1",  "peak_ratio": 5,   "n_peaks": 1, "status": "change"},
    {"name": f"06_Qr{total_discharge}_pm4_n3",  "peak_ratio": 4,   "n_peaks": 3, "status": "change"},
    {"name": f"07_Qr{total_discharge}_pm3_n4",  "peak_ratio": 3,   "n_peaks": 4, "status": "change"},
    {"name": f"08_Qr{total_discharge}_pm2_n6",  "peak_ratio": 2,   "n_peaks": 6, "status": "change"},
    {"name": f"09_Qr{total_discharge}_pm5_n3",  "peak_ratio": 5,   "n_peaks": 3, "status": "change"},
    {"name": f"10_Qr{total_discharge}_pm3_n3",  "peak_ratio": 3,   "n_peaks": 3, "status": "change"},
    {"name": f"11_Qr{total_discharge}_pm2_n3",  "peak_ratio": 2,   "n_peaks": 3, "status": "no_change"},
    {"name": f"12_Qr{total_discharge}_pm5_n4",  "peak_ratio": 5,   "n_peaks": 4, "status": "change"},
    {"name": f"13_Qr{total_discharge}_pm4_n4",  "peak_ratio": 4,   "n_peaks": 4, "status": "change"},
    {"name": f"14_Qr{total_discharge}_pm2_n4",  "peak_ratio": 2,   "n_peaks": 4, "status": "no_change"},
]

# Hypothetical threshold curve (x = n_peaks, y = peak/mean)
threshold_x = np.array([0.5, 1.1, 1.6, 2.5, 3.5, 4.5, 5.5, 6.5])
threshold_y = np.array([5, 4.2, 3.6, 3.0, 2.5, 2.0, 1.65, 1.4])

#%%
# ── PLOT STYLE ────────────────────────────────────────────────────────────────
COLORS = {
    "change":    "#F3480F",
    "no_change": "#0a94f0",
    "unknown":   "#639922",
    "old":       "#aaaaaa",
}

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.set_facecolor("white")
fig.patch.set_facecolor("white")

# Threshold curve + shading
ax.plot(threshold_x, threshold_y, color="#73726c", linewidth=1.5,
        linestyle="--", zorder=2, label="Threshold (hypothetical)")
ax.fill_between(threshold_x, threshold_y, 5.2,
                color="#D85A30", alpha=0.07, zorder=1)

# Label offsets (tweak per point to avoid overlap)
offsets = {
    f"01_Qr{total_discharge}_pm1_n0":  ( 0.12, -0.18),
    f"02_Qr{total_discharge}_pm2_n1":  ( 0.12,  0.12),
    f"03_Qr{total_discharge}_pm3_n5":  (-0.20,  0.14),
    f"04_Qr{total_discharge}_pm3_n1":  ( 0.12, -0.18),
    f"05_Qr{total_discharge}_pm5_n1":  (-0.22,  0.14),
    f"06_Qr{total_discharge}_pm4_n3":  ( 0.12,  0.12),
    f"07_Qr{total_discharge}_pm3_n4":  ( 0.12,  0.12),
    f"08_Qr{total_discharge}_pm2_n6":  ( 0.12, -0.18),
    f"09_Qr{total_discharge}_pm5_n3":  ( 0.12,  0.12),
    f"10_Qr{total_discharge}_pm3_n3":  ( 0.12,  0.12),
    f"11_Qr{total_discharge}_pm2_n3":  (-0.22, -0.18),
    f"12_Qr{total_discharge}_pm5_n4":  ( 0.12,  0.12),
    f"13_Qr{total_discharge}_pm4_n4":  (-0.22,  0.14),
    f"14_Qr{total_discharge}_pm2_n4":  ( 0.12, -0.18),
}

# Plot each scenario
for s in scenarios:
    name = s["name"]
    x, y = s["n_peaks"], s["peak_ratio"]
    dx, dy = offsets.get(name, (0.12, 0.08))
    color = COLORS[s["status"]]
    ax.scatter(x, y, marker="D", s=90,
               facecolors=color + "33", edgecolors=color,
               linewidths=1.5, zorder=3)
    # ax.text(x + dx, y + dy, name, fontsize=8.5,
    #         color="#3d3d3a", va="center", zorder=4)

# ── AXES & LABELS ─────────────────────────────────────────────────────────────
ax.set_xlabel("Number of peaks  (#)", fontsize=12)
ax.set_ylabel("Peak / mean discharge ratio  (–)", fontsize=12)
ax.set_xlim(-0.8, 7.0)
ax.set_ylim(-0.3, 5.4)
ax.set_xticks(range(0, 7))
ax.tick_params(labelsize=10)
ax.grid(color="#e0e0e0", linewidth=0.5, zorder=0)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("#cccccc")

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=COLORS["no_change"] + "33",
                   edgecolor=COLORS["no_change"], linewidth=1.5,
                   label="No persistent change"),
    mpatches.Patch(facecolor=COLORS["change"] + "33",
                   edgecolor=COLORS["change"], linewidth=1.5,
                   label="Persistent change"),
    plt.Line2D([0], [0], color="#73726c", linewidth=1.5,
               linestyle="--", label="Threshold"),
]
ax.legend(handles=legend_elements, fontsize=8.5, framealpha=0.9,
          edgecolor="#cccccc", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
plt.savefig("threshold_space.png", dpi=150, bbox_inches="tight")
plt.show()
# %%
