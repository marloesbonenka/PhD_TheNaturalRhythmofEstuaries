"""Plot net (flood- vs ebb-dominant) sediment transport over one tidal cycle,
taken from the last 24h of the simulation (≈ 2 M2 cycles, averaged to 1 cycle),
for the detailed-hydro-run scenarios:
  - constant flow
  - low flow
  - peak flow
  - mean flow

Sign convention:
  Transport aligned with the LOCAL flow direction when that flow is landward
  (ucx > 0 component dominating) = FLOOD-directed transport = BLUE (+)
  Transport aligned with locally seaward flow                = EBB-directed  = RED (-)

Unlike a simple sign(sx) approach, this projects the full transport vector
(sx, sy) onto the LOCAL depth-averaged velocity direction (ucx, ucy) at each
cell and timestep:

    s_along = (sx*ucx + sy*ucy) / |u|

This captures the full magnitude of transport in meandering/angled channel
reaches (where sx alone would truncate the signal to its x-component and
under-represent transport that's locally flood-directed but not x-aligned),
while still giving a physically meaningful flood(+)/ebb(-) sign based on
which way the water is actually moving at that location and instant.

This mirrors Figure 8 of the reference (net sediment transport, blue = import/
flood, red = export/ebb).
"""

# %% Imports
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- CONFIGURATION ---

# Scenarios: label → full path to the run folder
SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'peak_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503")
}

LOAD_VARS = ['mesh2d_sxtot', 'mesh2d_sytot', 'mesh2d_ucx', 'mesh2d_ucy']
# ucx/ucy (depth-averaged velocity components) are needed to determine the LOCAL
# flood/ebb direction at each cell — see s_along projection below. sytot is now
# used directly in that projection (no longer just an optional QC overlay).

# Small velocity magnitude [m/s] below which the local flow direction is considered
# too noisy/undefined to trust for sign assignment (e.g. near flow reversal or
# near-dry cells). Cells below this threshold at a given timestep contribute zero
# to s_along rather than an unstable/blown-up projection.
MIN_VELOCITY_FOR_PROJECTION = 0.01

# Window definition: last 24h of the run (~2 tidal cycles). The result plotted is the
# TIME-AVERAGED signed along-flow transport over this window — i.e. integral over time
# divided by elapsed time, giving units of [m^2/s], same as the instantaneous fields.
WINDOW_HOURS = 24.0

# Spatial zoom (model coordinates [m]) — same as your GIF script
ZOOM      = True
ZOOM_XLIM = (19000, 45000)
ZOOM_YLIM = (5000, 10000)

# Cache settings (must match what's already cached, or it will be extended/rebuilt)
CACHE_BBOX        = [1, 1, 45000, 15000]
CACHE_TAG         = None
APPEND_TIMESTEPS  = True
APPEND_VARIABLES  = True

# Colour scale for net transport (signed, diverging, centered at 0).
# NOTE: this is a time-AVERAGED net flux, so flood/ebb partially cancel — expect
# magnitudes well below the instantaneous VMAX you use for raw sed. transport snapshots
# (e.g. your snapshot script uses up to 1e-5; net flux here will typically be smaller).
# Start with a smaller VMAX and adjust per scenario/inspection.
VMAX = 2e-6
VMIN = -VMAX
CMAP = plt.cm.RdBu   # red = negative (ebb/seaward), blue = positive (flood/landward)
VAR_LABEL = 'net along-flow sediment transport [m²/s]\n(+landward/flood, -seaward/ebb)'

# Optional: overlay a light quiver of the time-averaged net (sx, sy) vector for QC,
# so you can visually compare the raw vector direction against the along-flow-
# projected colour field. Off by default since the reference figure is colour-only.
SHOW_QUIVER       = False
QUIVER_STRIDE     = 10
QUIVER_SCALE      = 60

# Output
FRAME_DPI = 150
STYLE = 'default'
plt.rcParams.update(plt.rcParamsDefault)
_tc = plt.rcParams['text.color']

# %% --- PROCESSING ---

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    output_dir = base_path / 'output_plots' / 'net_sediment_transport_tidal_cycle'
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve run paths ---
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=None,
        variability_map=None,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]

    # --- Load / update cache ---
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
        print(f"  [SKIP] No data cached for {folder_name}")
        continue

    try:
        if 'time' not in ds.dims or len(ds.time) == 0:
            print(f"  [SKIP] No time dimension in dataset.")
            continue

        if 'mesh2d_sxtot' not in ds or 'mesh2d_sytot' not in ds:
            print(f"  [SKIP] Sediment transport components not found in dataset.")
            continue

        if 'mesh2d_ucx' not in ds or 'mesh2d_ucy' not in ds:
            print(f"  [SKIP] Velocity components (mesh2d_ucx/ucy) not found in dataset — "
                  f"required for local flood/ebb direction projection.")
            continue

        time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
        n_t = len(time_values)
        print(f"  Total timesteps: {n_t}  ({time_values[0]} → {time_values[-1]})")

        # --- Select last WINDOW_HOURS of data ---
        t_end   = time_values[-1]
        t_start = t_end - np.timedelta64(int(WINDOW_HOURS * 3600), 's')
        window_mask = time_values >= t_start
        n_window = int(window_mask.sum())

        if n_window < 2:
            print(f"  [SKIP] Not enough timesteps in last {WINDOW_HOURS}h window "
                  f"(found {n_window}).")
            continue

        print(f"  Using last {WINDOW_HOURS}h: {n_window} timesteps "
              f"({time_values[window_mask][0]} → {time_values[window_mask][-1]})")

        actual_window_hours = (time_values[window_mask][-1] - time_values[window_mask][0]) / np.timedelta64(1, 'h')
        if actual_window_hours < 11.0:   # less than ~1 M2 cycle (12h25m)
            print(f"  [WARNING] Actual available window is only {actual_window_hours:.1f}h — "
                  f"shorter than one tidal cycle. Result will reflect a partial cycle, "
                  f"not a flood/ebb-averaged net transport. Interpret with caution.")

        ds_window = ds.isel(time=np.where(window_mask)[0])

        # --- Project transport onto local flow direction, then time-integrate ---
        # sxtot/sytot have shape (time, nSedTot, nFaces) → sum over sediment fractions first
        sx = ds_window['mesh2d_sxtot'].sum(dim='nSedTot').values   # (n_window, nFaces)
        sy = ds_window['mesh2d_sytot'].sum(dim='nSedTot').values   # (n_window, nFaces)
        ucx = ds_window['mesh2d_ucx'].values                        # (n_window, nFaces)
        ucy = ds_window['mesh2d_ucy'].values                        # (n_window, nFaces)

        u_mag = np.sqrt(ucx**2 + ucy**2)

        # s_along: signed transport magnitude aligned with the LOCAL flow direction at
        # each cell and timestep. Positive where transport moves the same way the water
        # is currently flowing AND that flow is landward-dominant; this is handled
        # implicitly since the projection (sx*ucx + sy*ucy) is positive when transport
        # and velocity point the same way, and ucx/ucy itself carries the landward(+x)/
        # seaward(-x) sense of the local flow (curved channels included, since ucy
        # contributes too).
        with np.errstate(invalid='ignore', divide='ignore'):
            s_along = np.where(
                u_mag > MIN_VELOCITY_FOR_PROJECTION,
                (sx * ucx + sy * ucy) / np.where(u_mag > 0, u_mag, 1),
                0.0,
            )

        # Time coordinate in seconds, for proper (non-uniform-safe) trapezoidal integration
        t_seconds = (ds_window['time'].values - ds_window['time'].values[0]) / np.timedelta64(1, 's')
        t_seconds = t_seconds.astype(float)

        # Integrate the signed along-flow transport over time → [m^2] per face over the window
        net_transport_integrated = np.trapz(s_along, x=t_seconds, axis=0)

        # Time-averaged signed along-flow flux over the window [m^2/s], same units as
        # instantaneous sxtot/sytot.
        elapsed_seconds = t_seconds[-1] - t_seconds[0]
        net_transport_per_cycle = net_transport_integrated / elapsed_seconds

        net_da = ds_window['mesh2d_sxtot'].isel(time=0, nSedTot=0).copy(data=net_transport_per_cycle)

        # --- Optional QC quiver: net (sx, sy) vector, oriented/colored by s_along sign ---
        if SHOW_QUIVER:
            net_sx_integrated = np.trapz(sx, x=t_seconds, axis=0) / elapsed_seconds
            net_sy_integrated = np.trapz(sy, x=t_seconds, axis=0) / elapsed_seconds
            face_x = ds.grid.face_coordinates[:, 0]
            face_y = ds.grid.face_coordinates[:, 1]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 4))

        pc = net_da.ugrid.plot(
            ax=ax,
            cmap=CMAP,
            add_colorbar=False,
            edgecolors='none',
            vmin=VMIN,
            vmax=VMAX,
        )

        if SHOW_QUIVER:
            mag = np.sqrt(net_sx_integrated**2 + net_sy_integrated**2)
            wet      = mag > (VMAX * 1e-2)
            idx_plot = np.where(wet)[0][::QUIVER_STRIDE]
            norm_t   = np.where(mag[idx_plot] > 1e-12, mag[idx_plot], 1)
            ax.quiver(
                face_x[idx_plot],                    face_y[idx_plot],
                net_sx_integrated[idx_plot] / norm_t, net_sy_integrated[idx_plot] / norm_t,
                scale=QUIVER_SCALE,
                color='grey',
                width=0.0015,
                headwidth=4,
                zorder=5,
                alpha=0.6,
            )

        ax.set_aspect('equal')
        if ZOOM:
            ax.set_xlim(ZOOM_XLIM)
            ax.set_ylim(ZOOM_YLIM)

        window_str = (f"{np.datetime_as_string(time_values[window_mask][0], unit='h')} → "
                      f"{np.datetime_as_string(time_values[window_mask][-1], unit='h')}")
        ax.set_title(
            f"net along-flow sediment transport (time-avg.) | {label} | {window_str}",
            color=_tc,
        )
        ax.set_xlabel('x [m]', color=_tc)
        ax.set_ylabel('y [m]', color=_tc)

        divider = make_axes_locatable(ax)
        cax     = divider.append_axes("right", size="3%", pad=0.1)
        cbar    = plt.colorbar(pc, cax=cax)
        cbar.set_label(VAR_LABEL, color=_tc)
        cbar.ax.yaxis.set_tick_params(color=_tc)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_tc)

        plt.tight_layout()

        fig_name = f"net_sed_transport_{label}_{folder_name}.png"
        fig_path = output_dir / fig_name
        plt.savefig(fig_path, dpi=FRAME_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    finally:
        ds.close()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
# %%

# """Plot net (flood- vs ebb-dominant) sediment transport over one tidal cycle,
# taken from the last 24h of the simulation (≈ 2 M2 cycles, averaged to 1 cycle),
# for the detailed-hydro-run scenarios:
#   - constant flow
#   - low flow
#   - peak flow
#   - mean flow

# Sign convention:
#   +x (seaward → landward, i.e. towards the river) = FLOOD-directed transport = BLUE
#   -x (landward → seaward, i.e. towards the sea)   = EBB-directed transport   = RED

# This mirrors Figure 8 of the reference (net sediment transport, blue = import/flood,
# red = export/ebb), but computed as a signed x-component (mesh2d_sxtot) rather than
# a direction-projected magnitude, since the estuary domain is approximately
# x-aligned (sea at low x, river at high x).
# """

# # %% Imports
# import sys
# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

# from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
# from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# # %% --- CONFIGURATION ---

# # Scenarios: label → full path to the run folder
# SCENARIOS = {
#     'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
#     'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
#     'peak_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
#     'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503")
# }

# LOAD_VARS = ['mesh2d_sxtot', 'mesh2d_sytot']   # sytot loaded too, for an optional QC quiver overlay

# # Window definition: last 24h of the run (~2 tidal cycles), result divided by 2
# # to express as a "per cycle" net transport, matching the reference figure's framing.
# WINDOW_HOURS      = 24.0
# CYCLES_IN_WINDOW  = 2.0   # 24h ≈ 2x M2 (12h25m); used purely to normalize to "per cycle"

# # Spatial zoom (model coordinates [m]) — same as your GIF script
# ZOOM      = True
# ZOOM_XLIM = (19000, 45000)
# ZOOM_YLIM = (5000, 10000)

# # Cache settings (must match what's already cached, or it will be extended/rebuilt)
# CACHE_BBOX        = [1, 1, 45000, 15000]
# CACHE_TAG         = None
# APPEND_TIMESTEPS  = True
# APPEND_VARIABLES  = True

# # Colour scale for net transport (signed, diverging, centered at 0).
# # NOTE: this is a time-AVERAGED net flux, so flood/ebb partially cancel — expect
# # magnitudes well below the instantaneous VMAX you use for raw sed. transport snapshots
# # (e.g. your snapshot script uses up to 1e-5; net flux here will typically be smaller).
# # Start with a smaller VMAX and adjust per scenario/inspection.
# VMAX = 2e-6
# VMIN = -VMAX
# CMAP = plt.cm.RdBu   # red = negative (ebb/seaward), blue = positive (flood/landward)
# VAR_LABEL = 'net sediment transport [m²/s]\n(+landward/flood, -seaward/ebb)'

# # Optional: overlay a light quiver of the (signed sx, raw sy) net vector for QC.
# # Off by default since the reference figure is colour-only; flip to True if useful.
# SHOW_QUIVER       = False
# QUIVER_STRIDE     = 10
# QUIVER_SCALE      = 60

# # Output
# FRAME_DPI = 150
# STYLE = 'default'
# plt.rcParams.update(plt.rcParamsDefault)
# _tc = plt.rcParams['text.color']

# # %% --- PROCESSING ---

# for label, folder_path in SCENARIOS.items():
#     print(f"\n{'='*60}")
#     print(f"Scenario: {label}  ({folder_path.name})")
#     print(f"{'='*60}")

#     base_path      = folder_path.parent
#     folder_name    = folder_path.name
#     assessment_dir = base_path / 'cached_data'
#     assessment_dir.mkdir(parents=True, exist_ok=True)

#     output_dir = base_path / 'output_plots' / 'net_sediment_transport_tidal_cycle'
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # --- Resolve run paths ---
#     run_paths = get_stitched_map_run_paths(
#         base_path=base_path,
#         folder_name=folder_name,
#         timed_out_dir=None,
#         variability_map=None,
#         analyze_noisy=False,
#     )
#     if not run_paths:
#         run_paths = [folder_path]

#     # --- Load / update cache ---
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
#         print(f"  [SKIP] No data cached for {folder_name}")
#         continue

#     try:
#         if 'time' not in ds.dims or len(ds.time) == 0:
#             print(f"  [SKIP] No time dimension in dataset.")
#             continue

#         if 'mesh2d_sxtot' not in ds:
#             print(f"  [SKIP] Variable 'mesh2d_sxtot' not found in dataset.")
#             continue

#         time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
#         n_t = len(time_values)
#         print(f"  Total timesteps: {n_t}  ({time_values[0]} → {time_values[-1]})")

#         # --- Select last WINDOW_HOURS of data ---
#         t_end   = time_values[-1]
#         t_start = t_end - np.timedelta64(int(WINDOW_HOURS * 3600), 's')
#         window_mask = time_values >= t_start
#         n_window = int(window_mask.sum())

#         if n_window < 2:
#             print(f"  [SKIP] Not enough timesteps in last {WINDOW_HOURS}h window "
#                   f"(found {n_window}).")
#             continue

#         print(f"  Using last {WINDOW_HOURS}h: {n_window} timesteps "
#               f"({time_values[window_mask][0]} → {time_values[window_mask][-1]})")

#         actual_window_hours = (time_values[window_mask][-1] - time_values[window_mask][0]) / np.timedelta64(1, 'h')
#         if actual_window_hours < 11.0:   # less than ~1 M2 cycle (12h25m)
#             print(f"  [WARNING] Actual available window is only {actual_window_hours:.1f}h — "
#                   f"shorter than one tidal cycle. Result will reflect a partial cycle, "
#                   f"not a flood/ebb-averaged net transport. Interpret with caution.")

#         ds_window = ds.isel(time=np.where(window_mask)[0])

#         # --- Time-integrate the signed x-component over the window ---
#         # sxtot has shape (time, nSedTot, nFaces) → sum over sediment fractions first
#         sx = ds_window['mesh2d_sxtot'].sum(dim='nSedTot')   # (time, nFaces)

#         # Time coordinate in seconds, for proper (non-uniform-safe) trapezoidal integration
#         t_seconds = (ds_window['time'].values - ds_window['time'].values[0]) / np.timedelta64(1, 's')
#         t_seconds = t_seconds.astype(float)

#         sx_vals = sx.values   # (n_window, nFaces)

#         # Integrate over time → cumulative transport [m^2] per face over the window
#         net_transport_integrated = np.trapz(sx_vals, x=t_seconds, axis=0)

#         # Time-averaged signed flux over the window [m^2/s], same units as instantaneous
#         # sxtot. Averaging over ~2 cycles (rather than reporting a raw 1-cycle sum) is what
#         # CYCLES_IN_WINDOW is for conceptually — with WINDOW_HOURS=24 ≈ 2x M2, this is
#         # equivalent to averaging the two cycles' net flux together.
#         elapsed_seconds = t_seconds[-1] - t_seconds[0]
#         net_transport_per_cycle = net_transport_integrated / elapsed_seconds

#         net_da = ds_window['mesh2d_sxtot'].isel(time=0, nSedTot=0).copy(data=net_transport_per_cycle)

#         # --- Optional: net y-component too, for QC quiver ---
#         if SHOW_QUIVER and 'mesh2d_sytot' in ds_window:
#             sy_vals = ds_window['mesh2d_sytot'].sum(dim='nSedTot').values
#             net_sy_integrated = np.trapz(sy_vals, x=t_seconds, axis=0)
#             net_sy_per_cycle = net_sy_integrated / elapsed_seconds
#             face_x = ds.grid.face_coordinates[:, 0]
#             face_y = ds.grid.face_coordinates[:, 1]

#         # --- Plot ---
#         fig, ax = plt.subplots(figsize=(12, 4))

#         pc = net_da.ugrid.plot(
#             ax=ax,
#             cmap=CMAP,
#             add_colorbar=False,
#             edgecolors='none',
#             vmin=VMIN,
#             vmax=VMAX,
#         )

#         if SHOW_QUIVER:
#             mag = np.sqrt(net_transport_per_cycle**2 + net_sy_per_cycle**2)
#             wet      = mag > (VMAX * 1e-2)
#             idx_plot = np.where(wet)[0][::QUIVER_STRIDE]
#             norm_t   = np.where(mag[idx_plot] > 1e-12, mag[idx_plot], 1)
#             ax.quiver(
#                 face_x[idx_plot],                       face_y[idx_plot],
#                 net_transport_per_cycle[idx_plot] / norm_t, net_sy_per_cycle[idx_plot] / norm_t,
#                 scale=QUIVER_SCALE,
#                 color='grey',
#                 width=0.0015,
#                 headwidth=4,
#                 zorder=5,
#                 alpha=0.6,
#             )

#         ax.set_aspect('equal')
#         if ZOOM:
#             ax.set_xlim(ZOOM_XLIM)
#             ax.set_ylim(ZOOM_YLIM)

#         window_str = (f"{np.datetime_as_string(time_values[window_mask][0], unit='h')} → "
#                       f"{np.datetime_as_string(time_values[window_mask][-1], unit='h')}")
#         ax.set_title(
#             f"net sediment transport (per tidal cycle) | {label} | {window_str}",
#             color=_tc,
#         )
#         ax.set_xlabel('x [m]', color=_tc)
#         ax.set_ylabel('y [m]', color=_tc)

#         divider = make_axes_locatable(ax)
#         cax     = divider.append_axes("right", size="3%", pad=0.1)
#         cbar    = plt.colorbar(pc, cax=cax)
#         cbar.set_label(VAR_LABEL, color=_tc)
#         cbar.ax.yaxis.set_tick_params(color=_tc)
#         plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_tc)

#         plt.tight_layout()

#         fig_name = f"net_sed_transport_{label}_{folder_name}.png"
#         fig_path = output_dir / fig_name
#         plt.savefig(fig_path, dpi=FRAME_DPI, bbox_inches='tight')
#         plt.close(fig)
#         print(f"  Saved: {fig_path}")

#     finally:
#         ds.close()

# print("\n" + "=" * 60)
# print("DONE")
# print("=" * 60)
# # %%