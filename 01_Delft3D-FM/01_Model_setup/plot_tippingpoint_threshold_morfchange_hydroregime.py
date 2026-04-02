"""Script to make a conceptual plot of the "threshold space" for 
classifying persistent morphological change based on different hydrological regimes
by assessing the number of peaks and peak/mean ratio"""

#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
#%%
# ── INPUT DATA ────────────────────────────────────────────────────────────────
scenarios = {
    # Old scenarios (S2–S4) – shown for reference, to be dismissed
    "S1 (constant)":     {"n_peaks": 0, "peak_mean": 1.0,  "status": "no_change",  "old": False},
    "S2 (seasonal)":     {"n_peaks": 1, "peak_mean": 1.5,  "status": "no_change",  "old": True},
    "S3 (flashy)":       {"n_peaks": 5, "peak_mean": 3.0,  "status": "change",     "old": True},
    "S4 (single peak)":  {"n_peaks": 1, "peak_mean": 3.0,  "status": "no_change",  "old": True},
    # New scenarios (S5–S10) – cover full peak/mean parameter space (1–5)
    "S5":                {"n_peaks": 2, "peak_mean": 1.5,  "status": "unknown",    "old": False},
    "S6":                {"n_peaks": 3, "peak_mean": 3.0,  "status": "unknown",    "old": False},
    "S7":                {"n_peaks": 5, "peak_mean": 4.0,  "status": "unknown",    "old": False},
    "S8":                {"n_peaks": 5, "peak_mean": 1.5,  "status": "unknown",    "old": False},
    "S9":                {"n_peaks": 1, "peak_mean": 5.0,  "status": "unknown",    "old": False},
    "S10":               {"n_peaks": 3, "peak_mean": 5.0,  "status": "unknown",    "old": False},
}

# Hypothetical threshold curve (x = n_peaks, y = peak/mean)
threshold_x = np.array([0.0, 0.8, 1.6, 2.5, 3.5, 4.5, 5.5, 6.5])
threshold_y = np.array([4.8, 4.2, 3.6, 3.0, 2.5, 2.0, 1.65, 1.4])

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
    "S1 (constant)":    ( 0.12, -0.18),
    "S2 (seasonal)":    (-0.20, -0.18),
    "S3 (flashy)":      (-0.20,  0.14),
    "S4 (single peak)": ( 0.12, -0.18),
    "S5":               ( 0.12,  0.12),
    "S6":               ( 0.12,  0.12),
    "S7":               ( 0.12,  0.12),
    "S8":               ( 0.12, -0.18),
    "S9":               (-0.22,  0.14),
    "S10":              ( 0.12,  0.12),
}

# Plot each scenario
for name, s in scenarios.items():
    x, y = s["n_peaks"], s["peak_mean"]
    dx, dy = offsets.get(name, (0.12, 0.08))

    if s["old"]:
        # Old scenarios: hollow circles in their status color (dismissed but still visible)
        color = COLORS[s["status"]]
        ax.scatter(x, y, marker="o", s=70, facecolors="none",
                   edgecolors=color, linewidths=1.2, zorder=3, alpha=0.6)
        ax.text(x + dx, y + dy, name, fontsize=7.5, color=color,
                va="center", zorder=4, alpha=0.7, style="italic")
    else:
        # New scenarios (S5–S10): colored diamond markers
        color = COLORS["unknown"]
        ax.scatter(x, y, marker="D", s=90,
                   facecolors=color + "33", edgecolors=color,
                   linewidths=1.5, zorder=3)
        ax.text(x + dx, y + dy, name, fontsize=8.5,
                color="#3d3d3a", va="center", zorder=4)

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
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=COLORS["no_change"], markeredgewidth=1.2,
               markersize=7, label="No persistent change — old (dismissed)"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=COLORS["change"], markeredgewidth=1.2,
               markersize=7, label="Persistent change — old (dismissed)"),
    mpatches.Patch(facecolor=COLORS["unknown"] + "33",
                   edgecolor=COLORS["unknown"], linewidth=1.5,
                   label="New scenarios S1, S5–S10 (proposed)"),
    plt.Line2D([0], [0], color="#73726c", linewidth=1.5,
               linestyle="--", label="Threshold (hypothetical)"),
]
ax.legend(handles=legend_elements, fontsize=8.5, framealpha=0.9,
          edgecolor="#cccccc", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
plt.savefig("threshold_space.png", dpi=150, bbox_inches="tight")
plt.show()
# %%
