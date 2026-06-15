"""
Along-estuary p5 / p95 water level profiles for four detailed-hydro-run scenarios,
plus an accommodation-space figure (Figure 2) overlaying p95 water level and p95 bed
level on the same axes to visualise the vertical gap available for bar aggradation.

For each scenario, the script:
  1. Loads the cached water level (mesh2d_s1) and bed level (mesh2d_mor_bl).
  2. Bins face cells by x-coordinate into equal-width intervals.
  3. Computes p5 / p95 across (time × faces) within each bin.
  4. Produces two figures:
       Fig 1 — p95 (high water) and p5 (low water) water level, all scenarios.
       Fig 2 — p95 water level vs p95 bed level on the same axes; one panel per
               scenario pair (constant vs. peak flow) to make the accommodation
               space argument explicit.
"""

# %% Imports
import sys
from pathlib import Path

import cmocean                          # noqa: F401  (kept for colourmap consistency)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'mean flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
    'peak flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
}

VAR_WL    = 'mesh2d_s1'
VAR_BL    = 'mesh2d_mor_bl'
LOAD_VARS = [VAR_WL, VAR_BL]

# Spatial extent used when building / reading the cache
CACHE_BBOX       = [1, 1, 45000, 15000]
CACHE_TAG        = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Along-estuary binning (x-coordinate)
X_BIN_WIDTH = 500          # metres — adjust to match your morpho script's resolution
X_MIN       = 19000        # trim sea-side margin (model mouth)
X_MAX       = 45000        # upstream limit

# Wet-cell filter: exclude cells that are essentially always dry
# (water level == fill value or NaN).  Set to None to skip.
WET_FRACTION_THRESHOLD = 0.1   # cell must be wet in ≥10 % of timesteps

# Colours for the four scenarios — order matches SCENARIOS dict
SCENARIO_COLORS = {
    'constant':  '#888888',
    'low flow':  '#92C5DE',
    'mean flow': '#4393C3',
    'peak flow': '#084594',
}
SCENARIO_LINESTYLE = {
    'constant':  '--',
    'low flow':  '-',
    'mean flow': '-',
    'peak flow': '-',
}

# Output
OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\water level")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILENAME_WL   = "waterlevel_p5_p95_along_estuary.png"
OUTPUT_FILENAME_ACCOM = "accommodation_space_along_estuary.png"

FIGURE_DPI = 200

# ---------------------------------------------------------------------------
# BIN EDGES
# ---------------------------------------------------------------------------

bin_edges  = np.arange(X_MIN, X_MAX + X_BIN_WIDTH, X_BIN_WIDTH)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / 1000   # → km for x-axis
n_bins = len(bin_centres)

# ---------------------------------------------------------------------------
# COLLECT PERCENTILE PROFILES PER SCENARIO
# ---------------------------------------------------------------------------

results = {}   # label → {'p5': array, 'p95': array, 'p95_bl': array}

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=None,
        variability_map=None,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]

    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=LOAD_VARS,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds is None:
        print(f"  [SKIP] No cached data found.")
        continue

    if VAR_WL not in ds:
        print(f"  [SKIP] '{VAR_WL}' not in dataset.")
        ds.close()
        continue

    # --- Face x-coordinates ---
    face_x = ds.grid.face_coordinates[:, 0]   # shape (nFaces,)

    # --- Water level array: (time, nFaces) ---
    wl = ds[VAR_WL].values   # shape (time, nFaces)

    # --- Bed level array: (time, nFaces) or (nFaces,) ---
    bl_raw = ds[VAR_BL].values if VAR_BL in ds else None
    if bl_raw is not None and bl_raw.ndim == 1:
        # static bed level — broadcast to (1, nFaces) so binning is uniform
        bl_raw = bl_raw[np.newaxis, :]

    # --- Optional: mask cells that are almost always dry (fill / NaN) ---
    if WET_FRACTION_THRESHOLD is not None:
        wet_frac = np.isfinite(wl).mean(axis=0)
        active   = wet_frac >= WET_FRACTION_THRESHOLD
        face_x_a = face_x[active]
        wl_a     = wl[:, active]
        bl_a     = bl_raw[:, active] if bl_raw is not None else None
        print(f"  Active cells after wet-fraction filter: {active.sum()} / {len(active)}")
    else:
        face_x_a, wl_a, bl_a = face_x, wl, bl_raw

    # --- Bin by x-coordinate and compute percentiles ---
    p5_profile   = np.full(n_bins, np.nan)
    p95_profile  = np.full(n_bins, np.nan)
    p95_bl_profile = np.full(n_bins, np.nan)

    bin_indices = np.digitize(face_x_a, bin_edges) - 1   # 0-based bin index

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue

        # Water level
        vals_wl = wl_a[:, mask].ravel()
        vals_wl = vals_wl[np.isfinite(vals_wl)]
        if len(vals_wl) > 0:
            p5_profile[b]  = np.percentile(vals_wl, 5)
            p95_profile[b] = np.percentile(vals_wl, 95)

        # Bed level p95 (bar top)
        if bl_a is not None:
            vals_bl = bl_a[:, mask].ravel()
            vals_bl = vals_bl[np.isfinite(vals_bl)]
            if len(vals_bl) > 0:
                p95_bl_profile[b] = np.percentile(vals_bl, 95)

    results[label] = {'p5': p5_profile, 'p95': p95_profile, 'p95_bl': p95_bl_profile}
    ds.close()
    print(f"  Profile computed. p5  WL: [{np.nanmin(p5_profile):.2f}, {np.nanmax(p5_profile):.2f}] m")
    print(f"                    p95 WL: [{np.nanmin(p95_profile):.2f}, {np.nanmax(p95_profile):.2f}] m")
    print(f"                    p95 BL: [{np.nanmin(p95_bl_profile):.2f}, {np.nanmax(p95_bl_profile):.2f}] m")

# ---------------------------------------------------------------------------
# PLOT - FIGURE 1: WATER LEVEL ENVELOPE
# ---------------------------------------------------------------------------

fig1, axes1 = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    constrained_layout=True,
)

ax_p95, ax_p5 = axes1   # top = high water, bottom = low water

for label, prof in results.items():
    color = SCENARIO_COLORS.get(label, 'black')
    ls    = SCENARIO_LINESTYLE.get(label, '-')
    lw    = 1.5 if label == 'constant' else 2.0

    ax_p95.plot(bin_centres, prof['p95'], color=color, ls=ls, lw=lw, label=label)
    ax_p5.plot( bin_centres, prof['p5'],  color=color, ls=ls, lw=lw, label=label)

# --- Formatting Fig 1 ---
for ax, title in zip([ax_p95, ax_p5], ['p95 water level (high water)', 'p5 water level (low water)']):
    ax.set_ylabel('water level [m]')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axhline(0, color='grey', lw=0.7, ls=':')   # MSL reference
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

ax_p5.set_xlabel('distance along estuary [km]')
ax_p5.set_xlim(X_MIN / 1000, X_MAX / 1000)

# Shared legend below figure 1
handles, labels = ax_p95.get_legend_handles_labels()
fig1.legend(
    handles, labels,
    loc='lower center',
    ncol=len(results),
    bbox_to_anchor=(0.5, -0.06),
    frameon=False,
    fontsize=9,
)

fig1.suptitle(
    'Along-estuary water level envelope\n'
    r'$Q_r = 500\ \mathrm{m^3/s}$,  all timesteps (2 tidal cycles)',
    fontsize=11,
)

# --- Save Fig 1 ---
out_path_wl = OUTPUT_DIR / OUTPUT_FILENAME_WL
fig1.savefig(out_path_wl, dpi=FIGURE_DPI, bbox_inches='tight')
print(f"\nSaved Figure 1: {out_path_wl}")


# ---------------------------------------------------------------------------
# PLOT - FIGURE 2: ACCOMMODATION SPACE
# ---------------------------------------------------------------------------

fig2, axes2 = plt.subplots(
    2, 1,
    figsize=(8, 6),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

ax_top, ax_bot = axes2

# --- Top Panel: Constant Flow ---
if 'constant' in results:
    prof_c = results['constant']
    color_c = SCENARIO_COLORS.get('constant', '#888888')
    
    # Plot Water Level & Bed Level
    ax_top.plot(bin_centres, prof_c['p95'], color=color_c, lw=1.5, ls='--', label='p95 Water Level (constant)')
    ax_top.plot(bin_centres, prof_c['p95_bl'], color='saddlebrown', lw=2.0, label='p95 Bed Level (Bar Top)')
    
    # Shade Accommodation Space
    ax_top.fill_between(bin_centres, prof_c['p95_bl'], prof_c['p95'], color=color_c, alpha=0.15, label='Accommodation Space')

    # Formatting
    ax_top.set_ylabel('elevation [m]')
    ax_top.set_title('Accommodation Space: Constant Flow', fontsize=10, fontweight='bold')
    ax_top.axhline(0, color='grey', lw=0.7, ls=':')
    ax_top.grid(True, lw=0.4, alpha=0.5)
    ax_top.yaxis.set_minor_locator(mticker.AutoMinorLocator())


# --- Bottom Panel: Hydro-variability (Low, Mean, Peak) ---
if 'peak flow' in results:
    prof_p = results['peak flow']
    
    # 1. Plot Bed Level first so it sits cleanly in the background/fill
    ax_bot.plot(bin_centres, prof_p['p95_bl'], color='saddlebrown', lw=2.0, label='p95 Bed Level (Bar Top)')
    
    # 2. Plot Low Flow Water Level
    if 'low flow' in results:
        ax_bot.plot(bin_centres, results['low flow']['p95'], color=SCENARIO_COLORS['low flow'], lw=1.5, label='p95 Water Level (low flow)')
        
    # 3. Plot Mean Flow Water Level
    if 'mean flow' in results:
        ax_bot.plot(bin_centres, results['mean flow']['p95'], color=SCENARIO_COLORS['mean flow'], lw=1.5, label='p95 Water Level (mean flow)')
        
    # 4. Plot Peak Flow Water Level
    ax_bot.plot(bin_centres, prof_p['p95'], color=SCENARIO_COLORS['peak flow'], lw=2.0, label='p95 Water Level (peak flow)')
    
    # 5. Shade total maximum accommodation space (bounded by the Peak Flow high water mark)
    ax_bot.fill_between(bin_centres, prof_p['p95_bl'], prof_p['p95'], color=SCENARIO_COLORS['peak flow'], alpha=0.15, label='Accommodation Space')

    # Formatting
    ax_bot.set_ylabel('elevation [m]')
    ax_bot.set_title('Accommodation Space: Variable Flow Scenarios', fontsize=10, fontweight='bold')
    ax_bot.axhline(0, color='grey', lw=0.7, ls=':')
    ax_bot.grid(True, lw=0.4, alpha=0.5)
    ax_bot.yaxis.set_minor_locator(mticker.AutoMinorLocator())

# Global X-axis formatting
ax_bot.set_xlabel('distance along estuary [km]')
ax_bot.set_xlim(X_MIN / 1000, X_MAX / 1000)

# --- Dynamic & Deduplicated Legend ---
handles2 = []
labels2 = []
for ax in axes2:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels2:
            handles2.append(handle)
            labels2.append(label)

fig2.legend(
    handles2, labels2,
    loc='lower center',
    ncol=3,  # 3 columns perfectly fits the 6 unique entries into a 2x3 grid
    bbox_to_anchor=(0.5, -0.13),
    frameon=False,
    fontsize=9,
)

fig2.suptitle(
    'Accommodation Space: High Water vs Bar Top\n'
    r'$Q_r = 500\ \mathrm{m^3/s}$',
    fontsize=11,
)

# --- Save Fig 2 ---
out_path_accom = OUTPUT_DIR / OUTPUT_FILENAME_ACCOM
fig2.savefig(out_path_accom, dpi=FIGURE_DPI, bbox_inches='tight')
plt.show()
print(f"Saved Figure 2: {out_path_accom}")
#%%
# """
# Along-estuary p5 / p95 water level profiles for four detailed-hydro-run scenarios.

# For each scenario, the script:
#   1. Loads the cached water level (mesh2d_s1) across all timesteps.
#   2. Bins face cells by x-coordinate into equal-width intervals.
#   3. Computes p5 and p95 across (time × faces) within each bin.
#   4. Plots all scenarios as coloured lines on two shared axes (p5 top, p95 bottom).

# Output: one PNG saved next to the script's output folder.
# """

# # %% Imports
# import sys
# from pathlib import Path

# import cmocean                          # noqa: F401  (kept for colourmap consistency)
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import numpy as np

# sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

# from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
# from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# # ---------------------------------------------------------------------------
# # CONFIGURATION
# # ---------------------------------------------------------------------------

# SCENARIOS = {
#     'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
#     'low flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
#     'mean flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
#     'peak flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
# }

# VAR_NAME  = 'mesh2d_s1'
# LOAD_VARS = [VAR_NAME]

# # Spatial extent used when building / reading the cache
# CACHE_BBOX       = [1, 1, 45000, 15000]
# CACHE_TAG        = None
# APPEND_TIMESTEPS = True
# APPEND_VARIABLES = True

# # Along-estuary binning (x-coordinate)
# X_BIN_WIDTH = 500          # metres — adjust to match your morpho script's resolution
# X_MIN       = 19000        # trim sea-side margin (model mouth)
# X_MAX       = 45000        # upstream limit

# # Wet-cell filter: exclude cells that are essentially always dry
# # (water level == fill value or NaN).  Set to None to skip.
# WET_FRACTION_THRESHOLD = 0.1   # cell must be wet in ≥10 % of timesteps

# # Colours for the four scenarios — order matches SCENARIOS dict
# SCENARIO_COLORS = {
#     'constant':  '#888888',
#     'low flow':  "#E6B4F5",
#     'mean flow': "#D240FF",
#     'peak flow': "#730694",
# }
# SCENARIO_LINESTYLE = {
#     'constant':  '--',
#     'low flow':  '-',
#     'mean flow': '-',
#     'peak flow': '-',
# }

# # Output
# OUTPUT_DIR = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\water level")
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_FILENAME = "waterlevel_p5_p95_along_estuary.png"

# FIGURE_DPI = 200

# # ---------------------------------------------------------------------------
# # BIN EDGES
# # ---------------------------------------------------------------------------

# bin_edges  = np.arange(X_MIN, X_MAX + X_BIN_WIDTH, X_BIN_WIDTH)
# bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:]) / 1000   # → km for x-axis
# n_bins = len(bin_centres)

# # ---------------------------------------------------------------------------
# # COLLECT PERCENTILE PROFILES PER SCENARIO
# # ---------------------------------------------------------------------------

# results = {}   # label → {'p5': array(n_bins,), 'p95': array(n_bins,)}

# for label, folder_path in SCENARIOS.items():
#     print(f"\n{'='*60}")
#     print(f"Scenario: {label}  ({folder_path.name})")
#     print(f"{'='*60}")

#     base_path      = folder_path.parent
#     folder_name    = folder_path.name
#     assessment_dir = base_path / 'cached_data'
#     assessment_dir.mkdir(parents=True, exist_ok=True)

#     run_paths = get_stitched_map_run_paths(
#         base_path=base_path,
#         folder_name=folder_name,
#         timed_out_dir=None,
#         variability_map=None,
#         analyze_noisy=False,
#     )
#     if not run_paths:
#         run_paths = [folder_path]

#     cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
#     ds = load_or_update_map_cache_multi(
#         cache_dir=assessment_dir,
#         folder_name=folder_name,
#         run_paths=run_paths,
#         var_names=LOAD_VARS,
#         bbox=CACHE_BBOX,
#         append_time=APPEND_TIMESTEPS,
#         append_vars=APPEND_VARIABLES,
#         cache_tag=cache_tag,
#     )

#     if ds is None:
#         print(f"  [SKIP] No cached data found.")
#         continue

#     if VAR_NAME not in ds:
#         print(f"  [SKIP] '{VAR_NAME}' not in dataset.")
#         ds.close()
#         continue

#     # --- Face x-coordinates ---
#     face_x = ds.grid.face_coordinates[:, 0]   # shape (nFaces,)

#     # --- Water level array: (time, nFaces) ---
#     wl = ds[VAR_NAME].values   # shape (time, nFaces)

#     # --- Optional: mask cells that are almost always dry (fill / NaN) ---
#     if WET_FRACTION_THRESHOLD is not None:
#         wet_frac = np.isfinite(wl).mean(axis=0)
#         active   = wet_frac >= WET_FRACTION_THRESHOLD
#         face_x   = face_x[active]
#         wl       = wl[:, active]
#         print(f"  Active cells after wet-fraction filter: {active.sum()} / {len(active)}")

#     # --- Bin by x-coordinate and compute percentiles ---
#     p5_profile  = np.full(n_bins, np.nan)
#     p95_profile = np.full(n_bins, np.nan)

#     bin_indices = np.digitize(face_x, bin_edges) - 1   # 0-based bin index

#     for b in range(n_bins):
#         mask = bin_indices == b
#         if mask.sum() == 0:
#             continue
#         vals = wl[:, mask].ravel()           # flatten time × cells in bin
#         vals = vals[np.isfinite(vals)]       # drop NaNs / fill values
#         if len(vals) == 0:
#             continue
#         p5_profile[b]  = np.percentile(vals, 5)
#         p95_profile[b] = np.percentile(vals, 95)

#     results[label] = {'p5': p5_profile, 'p95': p95_profile}
#     ds.close()
#     print(f"  Profile computed. p5 range: [{np.nanmin(p5_profile):.2f}, {np.nanmax(p5_profile):.2f}] m")
#     print(f"                    p95 range: [{np.nanmin(p95_profile):.2f}, {np.nanmax(p95_profile):.2f}] m")

# # ---------------------------------------------------------------------------
# # PLOT
# # ---------------------------------------------------------------------------

# fig, axes = plt.subplots(
#     2, 1,
#     figsize=(8, 6),
#     sharex=True,
#     constrained_layout=True,
# )

# ax_p95, ax_p5 = axes   # top = high water, bottom = low water

# for label, prof in results.items():
#     color = SCENARIO_COLORS.get(label, 'black')
#     ls    = SCENARIO_LINESTYLE.get(label, '-')
#     lw    = 1.5 if label == 'constant' else 2.0

#     ax_p95.plot(bin_centres, prof['p95'], color=color, ls=ls, lw=lw, label=label)
#     ax_p5.plot( bin_centres, prof['p5'],  color=color, ls=ls, lw=lw, label=label)

# # --- Formatting ---
# for ax, title in zip([ax_p95, ax_p5], ['p95 water level (high water)', 'p5 water level (low water)']):
#     ax.set_ylabel('water level [m]')
#     ax.set_title(title, fontsize=10, fontweight='bold')
#     ax.axhline(0, color='grey', lw=0.7, ls=':')   # MSL reference
#     ax.grid(True, lw=0.4, alpha=0.5)
#     ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

# ax_p5.set_xlabel('distance along estuary [km]')
# ax_p5.set_xlim(X_MIN / 1000, X_MAX / 1000)

# # Shared legend below figure
# handles, labels = ax_p95.get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     loc='lower center',
#     ncol=len(results),
#     bbox_to_anchor=(0.5, -0.06),
#     frameon=False,
#     fontsize=9,
# )

# fig.suptitle(
#     'Along-estuary water level envelope\n'
#     r'$Q_r = 500\ \mathrm{m^3/s}$,  all timesteps (2 tidal cycles)',
#     fontsize=11,
# )

# # --- Save ---
# out_path = OUTPUT_DIR / OUTPUT_FILENAME
# fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches='tight')
# plt.show()
# print(f"\nSaved: {out_path}")
# #%%