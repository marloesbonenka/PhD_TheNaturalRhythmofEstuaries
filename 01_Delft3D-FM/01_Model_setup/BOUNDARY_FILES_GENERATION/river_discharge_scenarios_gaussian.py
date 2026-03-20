"""Create river discharge variability scenarios based on a Gaussian distribution of events,
ensuring the same total volume over the year but varying in magnitude and frequency.
This script generates both hydrodynamic (continuous) and morphodynamic (stepped) plots for each scenario."""
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#%%--- 1. Setup ---
days = 365
mf = 100
t_hd = np.arange(days)
t_md = np.arange(0, (days + 1) * mf, mf)

Q_mean_target = 1000.0
Q_base = 700.0
V_total_excess = (Q_mean_target - Q_base) * days

# Optional output settings
SAVE_HYDRO_DISCHARGE_FIG = True
SAVE_MORPHO_DISCHARGE_FIG = True
SAVE_HYDRO_ZOOM_FIG = True
SAVE_MORPHO_ZOOM_FIG = True
FIG_DPI = 300

OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Input\Gaussian_scenarios_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HYDRO_DISCHARGE_FIG_NAME = "river_discharge_scenarios_hydro.png"
MORPHO_DISCHARGE_FIG_NAME = "river_discharge_scenarios_morpho.png"
HYDRO_ZOOM_FIG_NAME = "river_discharge_scenarios_hydro_zoom50d_with_tide.png"
MORPHO_ZOOM_FIG_NAME = "river_discharge_scenarios_morpho_zoom50d_with_tide.png"

ZOOM_WINDOW_DAYS = 50
ZOOM_CENTER_DAY = days / 2

#%% --- 2. Define Combinations ---
# Format: Name: (Peak_Ratio, Number_of_Events)
scenarios = {
    "Constant": (1.0, 0),
    "Seasonal": (2.0, 2),
    "Wet/dry":  (3.0, 5),
    "Flashy":   (3.5, 1),
    "Episodic": (5.0, 1) # Single massive event
}

SCENARIO_COLORS = {
    "Constant": '#1f77b4',
    "Seasonal": '#ff7f0e',
    "Wet/dry":  '#2ca02c',
    "Flashy":   '#d62728',
    "Episodic": '#9467bd'
}

#%% --- 3. Plotting ---
# Simplified semi-diurnal tide: amplitude 2 m, period 12 hours (= 0.5 day)
tide_amp = 2.0
tide_period_days = 0.5
t_tide_hd = np.linspace(0, days, days * 24, endpoint=False)
tide_hd = tide_amp * np.sin(2 * np.pi * t_tide_hd / tide_period_days)

t_tide_md = t_tide_hd * mf
tide_md = tide_amp * np.sin(2 * np.pi * t_tide_md / (tide_period_days * mf))

zoom_start_hd = max(0.0, ZOOM_CENTER_DAY - ZOOM_WINDOW_DAYS / 2)
zoom_end_hd = min(float(days), ZOOM_CENTER_DAY + ZOOM_WINDOW_DAYS / 2)
zoom_start_md = zoom_start_hd * mf
zoom_end_md = zoom_end_hd * mf

# Full-period discharge figure (hydrodynamic)
fig_h_full, ax_h_full = plt.subplots(1, 1, figsize=(11, 5))

# Full-period discharge figure (morphodynamic)
fig_m_full, ax_m_full = plt.subplots(1, 1, figsize=(11, 5))

# Zoomed figure with subplots (hydrodynamic: discharge + tide)
fig_h_zoom, (ax_h_zoom_q, ax_h_zoom_t) = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    gridspec_kw={'height_ratios': [2.5, 1]},
    sharex=False,
)

# Zoomed figure with subplots (morphodynamic: discharge + tide)
fig_m_zoom, (ax_m_zoom_q, ax_m_zoom_t) = plt.subplots(
    2,
    1,
    figsize=(11, 8),
    gridspec_kw={'height_ratios': [2.5, 1]},
    sharex=False,
)

for name, (ratio, n_events) in scenarios.items():
    color = SCENARIO_COLORS[name]

    if name == "Constant":
        q_vals = np.full(days, Q_mean_target)
    else:
        Q_peak = Q_mean_target * ratio
        A = Q_peak - Q_base
        V_event = V_total_excess / n_events
        sigma = V_event / (A * np.sqrt(2 * np.pi))

        # Place events at the center of n equal segments, avoiding the edges
        segment = days / n_events
        event_centers = np.linspace(segment / 2, days - segment / 2, n_events)
        q_vals = np.full(days, Q_base)
        for t0 in event_centers:
            q_vals += A * np.exp(-(t_hd - t0)**2 / (2 * sigma**2))

    # Hydrodynamic discharge: full-period and zoomed
    ax_h_full.plot(t_hd, q_vals, lw=2, color=color, label=f'{name} (n={n_events}, P/M={ratio})')
    ax_h_zoom_q.plot(t_hd, q_vals, lw=2, color=color, label=f'{name} (n={n_events}, P/M={ratio})')

    # Morphodynamic discharge: full-period and zoomed
    q_stepped = np.append(q_vals, q_vals[-1])
    ax_m_full.step(t_md, q_stepped, where='post', color=color, alpha=0.7, lw=1.5,
                   label=f'{name} (n={n_events}, P/M={ratio})')
    ax_m_zoom_q.step(t_md, q_stepped, where='post', color=color, alpha=0.7, lw=1.5,
                     label=f'{name} (n={n_events}, P/M={ratio})')

# Plot tidal signals in zoomed figures
ax_h_zoom_t.plot(t_tide_hd, tide_hd, color='k', lw=1.0)
ax_m_zoom_t.step(t_tide_md, tide_md, where='post', color='k', lw=1.0)

# --- 4. Formatting full-period discharge figures ---
ax_h_full.set_ylim(0, Q_mean_target * 5.5)
ax_h_full.set_xlim(0, days)
ax_h_full.set_title("Hydrodynamic time: magnitude & frequency combinations", fontsize=14)
ax_h_full.set_xlabel("Hydrodynamic time [days]")
ax_h_full.set_ylabel("Discharge [m³/s]")
ax_h_full.legend(loc='best')
ax_h_full.grid(True, alpha=0.2)

ax_m_full.set_ylim(0, Q_mean_target * 5.5)
ax_m_full.set_xlim(0, days * mf)
ax_m_full.set_title(f"Morphodynamic time (morfac = {mf})", fontsize=14)
ax_m_full.set_xlabel("Morphodynamic time [days]")
ax_m_full.set_ylabel("Discharge [m³/s]")
ax_m_full.legend(loc='best')
ax_m_full.grid(True, alpha=0.2)

# --- 5. Formatting zoomed hydrodynamic figure ---
ax_h_zoom_q.set_ylim(0, Q_mean_target * 5.5)
ax_h_zoom_q.set_xlim(zoom_start_hd, zoom_end_hd)
ax_h_zoom_q.set_title(
    f"Hydrodynamic zoom ({ZOOM_WINDOW_DAYS} days, centered near mid-year)",
    fontsize=14,
)
ax_h_zoom_q.set_ylabel("Discharge [m³/s]")
ax_h_zoom_q.legend(loc='best')
ax_h_zoom_q.grid(True, alpha=0.2)

ax_h_zoom_t.set_xlim(zoom_start_hd, zoom_end_hd)
ax_h_zoom_t.set_ylim(-tide_amp * 1.2, tide_amp * 1.2)
ax_h_zoom_t.set_xlabel("Hydrodynamic time [days]")
ax_h_zoom_t.set_ylabel("Tide [m]")
ax_h_zoom_t.grid(True, alpha=0.2)

# --- 6. Formatting zoomed morphodynamic figure ---
ax_m_zoom_q.set_ylim(0, Q_mean_target * 5.5)
ax_m_zoom_q.set_xlim(zoom_start_md, zoom_end_md)
ax_m_zoom_q.set_title(
    f"Morphodynamic zoom ({ZOOM_WINDOW_DAYS} hydro-days, centered near mid-year, morfac = {mf})",
    fontsize=14,
)
ax_m_zoom_q.set_ylabel("Discharge [m³/s]")
ax_m_zoom_q.legend(loc='best')
ax_m_zoom_q.grid(True, alpha=0.2)

ax_m_zoom_t.set_xlim(zoom_start_md, zoom_end_md)
ax_m_zoom_t.set_ylim(-tide_amp * 1.2, tide_amp * 1.2)
ax_m_zoom_t.set_xlabel("Morphodynamic time [days]")
ax_m_zoom_t.set_ylabel("Tide [m]")
ax_m_zoom_t.grid(True, alpha=0.2)

fig_h_full.tight_layout()
fig_m_full.tight_layout()
fig_h_zoom.tight_layout()
fig_m_zoom.tight_layout()

if SAVE_HYDRO_DISCHARGE_FIG:
    fig_h_full.savefig(OUTPUT_DIR / HYDRO_DISCHARGE_FIG_NAME, dpi=FIG_DPI, bbox_inches='tight')

if SAVE_MORPHO_DISCHARGE_FIG:
    fig_m_full.savefig(OUTPUT_DIR / MORPHO_DISCHARGE_FIG_NAME, dpi=FIG_DPI, bbox_inches='tight')

if SAVE_HYDRO_ZOOM_FIG:
    fig_h_zoom.savefig(OUTPUT_DIR / HYDRO_ZOOM_FIG_NAME, dpi=FIG_DPI, bbox_inches='tight')

if SAVE_MORPHO_ZOOM_FIG:
    fig_m_zoom.savefig(OUTPUT_DIR / MORPHO_ZOOM_FIG_NAME, dpi=FIG_DPI, bbox_inches='tight')

plt.show()
#%%
