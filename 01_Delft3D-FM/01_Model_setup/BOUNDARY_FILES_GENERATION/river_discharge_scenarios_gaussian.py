"""Create river discharge variability scenarios based on a Gaussian distribution of events,
ensuring the same total volume over the year but varying in magnitude and frequency.
This script generates both hydrodynamic (continuous) and morphodynamic (stepped) plots for each scenario."""
#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt

#%%--- 1. Setup ---
days = 365
mf = 100
t_hd = np.arange(days)
t_md = np.arange(0, (days + 1) * mf, mf)

Q_mean_target = 1000.0
Q_base = 700.0
V_total_excess = (Q_mean_target - Q_base) * days

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
# Create a single figure with 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

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

    # Plot on the first axis (Hydrodynamic)
    ax1.plot(t_hd, q_vals, lw=2, color=color, label=f'{name} (n={n_events}, P/M={ratio})')

    # Plot on the second axis (Morphodynamic)
    q_stepped = np.append(q_vals, q_vals[-1])
    ax2.step(t_md, q_stepped, where='post', color=color, alpha=0.7, lw=1.5)

# --- 4. Formatting Axis 1 (Hydro) ---
ax1.set_ylim(0, Q_mean_target * 5.5)
ax1.set_title("Hydrodynamic time: magnitude & frequency combinations", fontsize=14)
ax1.set_xlabel("Hydrodynamic time [days]")
ax1.set_ylabel("Discharge [m³/s]")
ax1.legend(loc='best')
ax1.grid(True, alpha=0.2)

# --- 5. Formatting Axis 2 (Morpho) ---
ax2.set_ylim(0, Q_mean_target * 5.5)
ax2.set_title(f"Morphodynamic time (morfac = {mf})", fontsize=14)
ax2.set_ylabel("Discharge [m³/s]")
ax2.set_xlabel("Morphodynamic time [days]")
ax2.legend(loc='best')
ax2.grid(True, alpha=0.2)

fig.tight_layout()
plt.show()
#%%
