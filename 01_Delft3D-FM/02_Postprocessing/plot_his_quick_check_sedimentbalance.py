"""Compare sediment transport (inflow at the river boundary vs outflow at
the estuary mouth) over the final two tidal cycles, from a single HIS file.

Inflow cross-section  : x ~ 45 km (river boundary)
Outflow cross-section : x ~ 20 km (estuary mouth)

This script does not use or write cache files.

NOTE ON VARIABLE NAME
----------------------
SEDIMENT_VAR below is a best guess ("cross_section_sediment_transport").
If load_cross_section_data raises a KeyError, the except-block will open
the HIS file directly and print every cross_section_* variable containing
"sed" or "transport" so you can read off the correct name and update
SEDIMENT_VAR.

NOTE ON SIGN CONVENTION
------------------------
Cross-section transport sign in D-Flow FM follows each cross-section's own
defined orientation, not simply the domain +x/-x axis. INFLOW_SIGN /
OUTFLOW_SIGN below assume a positive value at km45 means "into the domain"
(river -> estuary) and a positive value at km20 means "out of the domain"
(estuary -> sea). Check the printed net values against a known
one-directional, constant-discharge run and flip a sign to -1 if a
direction comes out backwards.

NOTE ON TIDAL CYCLE
--------------------
The final N_CYCLES (default 2) full tidal periods of the run are selected.
Two cycles is the default because the first cycles in a run can still carry
a spin-up transient (larger, decaying oscillations); the later ones settle
into a repeating pattern -- check your own discharge/water-level time
series to confirm the run has settled by the window you select.
TIDAL_PERIOD_HOURS defaults to the M2 period (12.42 h) -- adjust if your
forcing uses a different constituent/period. A per-cycle breakdown is
printed in addition to the combined total so the two cycles can be checked
against each other for consistency.
"""
#%%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from FUNCTIONS.F_loaddata import load_cross_section_data


HIS_FILE = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783\output\FlowFM_0000_his.nc"
    # r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_mean.10280080\output\FlowFM_0000_his.nc"
)

SEDIMENT_VAR = "cross_section_bedload_sediment_transport"  # <-- confirm/adjust, see fallback below
KM_INFLOW = 45.0    # river boundary
KM_OUTFLOW = 20.0   # estuary mouth

INFLOW_SIGN = +1.0
OUTFLOW_SIGN = +1.0

TIDAL_PERIOD_HOURS = 12.42  # M2; adjust if needed
N_CYCLES = 2  # final two cycles (later ones are the settled/repeating pair, per the discharge check)

#%%
if not HIS_FILE.exists():
    raise FileNotFoundError(f"HIS file not found: {HIS_FILE}")

try:
    data = load_cross_section_data(
        his_file_path=HIS_FILE,
        q_var=SEDIMENT_VAR,
        estuary_only=True,
        km_range=(20, 45),
        select_cycles_hydrodynamic=False,
    )
except KeyError:
    with xr.open_dataset(HIS_FILE) as ds:
        candidates = [
            v for v in ds.data_vars
            if "cross_section" in v and ("sed" in v.lower() or "transport" in v.lower())
        ]
    raise KeyError(
        f"'{SEDIMENT_VAR}' not found in HIS file. Candidate cross-section variables:\n"
        + "\n".join(candidates)
        + "\n\nUpdate SEDIMENT_VAR above and re-run."
    )

#%%
km_positions = np.asarray(data["km_positions"])
transport = data[SEDIMENT_VAR].values  # shape (ntime, ncross)
time = np.asarray(data["t"])

idx_in = int(np.argmin(np.abs(km_positions - KM_INFLOW)))
idx_out = int(np.argmin(np.abs(km_positions - KM_OUTFLOW)))
km_in = float(km_positions[idx_in])
km_out = float(km_positions[idx_out])

q_in = INFLOW_SIGN * transport[:, idx_in]
q_out = OUTFLOW_SIGN * transport[:, idx_out]

print(f"Inflow cross-section:  requested {KM_INFLOW} km -> using {km_in:.2f} km")
print(f"Outflow cross-section: requested {KM_OUTFLOW} km -> using {km_out:.2f} km")

#%%
# --- select the final full tidal cycle(s) from the time series ---
if np.issubdtype(time.dtype, np.datetime64):
    time_seconds = (time - time[0]) / np.timedelta64(1, "s")
else:
    time_seconds = np.asarray(time, dtype=float)

period_seconds = TIDAL_PERIOD_HOURS * 3600.0
window_seconds = N_CYCLES * period_seconds

t_end = time_seconds[-1]
t_start = t_end - window_seconds
mask = time_seconds >= t_start

if mask.sum() < 2:
    raise ValueError(
        "Not enough timesteps in the requested tidal-cycle window; "
        "check TIDAL_PERIOD_HOURS / N_CYCLES against the HIS output interval."
    )

t_cycle = time[mask]
t_cycle_s = time_seconds[mask] - time_seconds[mask][0]
q_in_cycle = q_in[mask]
q_out_cycle = q_out[mask]

#%%
# --- cumulative transport over the full selected window (trapezoidal integration) ---
cum_in = np.concatenate(
    [[0.0], np.cumsum(0.5 * (q_in_cycle[1:] + q_in_cycle[:-1]) * np.diff(t_cycle_s))]
)
cum_out = np.concatenate(
    [[0.0], np.cumsum(0.5 * (q_out_cycle[1:] + q_out_cycle[:-1]) * np.diff(t_cycle_s))]
)

net_in = cum_in[-1]
net_out = cum_out[-1]
net_budget = net_in - net_out  # positive = net deposition within the domain over this window

print(f"Net sediment in  at {km_in:.2f} km over {N_CYCLES} tidal cycle(s): {net_in:.3f} m3")
print(f"Net sediment out at {km_out:.2f} km over {N_CYCLES} tidal cycle(s): {net_out:.3f} m3")
print(
    f"Net budget (in - out), combined: {net_budget:.3f} m3  "
    f"({'accumulation' if net_budget > 0 else 'export'} within domain)"
)

#%%
# --- per-cycle breakdown, so the two final cycles can be compared for consistency ---
cycle_bounds_hr = np.arange(N_CYCLES + 1) * TIDAL_PERIOD_HOURS
t_cycle_hr = t_cycle_s / 3600.0

print("\nPer-cycle breakdown:")
for i in range(N_CYCLES):
    c_mask = (t_cycle_hr >= cycle_bounds_hr[i]) & (t_cycle_hr <= cycle_bounds_hr[i + 1])
    if c_mask.sum() < 2:
        continue
    t_c = t_cycle_s[c_mask]
    q_in_c = q_in_cycle[c_mask]
    q_out_c = q_out_cycle[c_mask]
    net_in_c = np.trapezoid(q_in_c, t_c)
    net_out_c = np.trapezoid(q_out_c, t_c)
    print(
        f"  Cycle {i + 1}: in = {net_in_c:.3f} m3, out = {net_out_c:.3f} m3, "
        f"budget = {net_in_c - net_out_c:.3f} m3"
    )

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(t_cycle_s / 3600, q_in_cycle, color="tab:blue", lw=1.2, label=f"Inflow, {km_in:.2f} km")
ax1.plot(t_cycle_s / 3600, q_out_cycle, color="tab:red", lw=1.2, label=f"Outflow, {km_out:.2f} km")
ax1.axhline(0, color="k", lw=0.5)
for b in cycle_bounds_hr[1:-1]:
    ax1.axvline(b, color="k", lw=0.5, ls="--")
ax1.set_ylabel("Sediment transport (m3/s)")
ax1.set_title(f"Sediment transport over the final {N_CYCLES} tidal cycles")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(t_cycle_s / 3600, cum_in, color="tab:blue", lw=1.5, label="Cumulative in")
ax2.plot(t_cycle_s / 3600, cum_out, color="tab:red", lw=1.5, label="Cumulative out")
ax2.axhline(0, color="k", lw=0.5)
for b in cycle_bounds_hr[1:-1]:
    ax2.axvline(b, color="k", lw=0.5, ls="--")
ax2.set_xlabel("Time within window (hours)")
ax2.set_ylabel("Cumulative sediment (m3)")
ax2.set_title(f"Combined net budget: {net_budget:.2f} m3")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
plt.show()

if data.get("ds") is not None:
    data["ds"].close()

# %%