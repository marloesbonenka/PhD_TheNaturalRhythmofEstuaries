"""Plot net (flood- vs ebb-dominant) sediment transport over one tidal cycle,
taken from the last 24h of the simulation (≈ 2 M2 cycles, averaged to 1 cycle),
for the detailed-hydro-run scenarios

Sign convention (landward = +x, towards the river; seaward = -x, towards the
mouth — see SCENARIOS paths, mouth at low x, river at high x):
  FLOOD-directed transport = BLUE (+)   — occurring while local flow is landward
  EBB-directed transport   = RED  (-)   — occurring while local flow is seaward

Computed in two steps, at each cell and timestep:
  1. transport_mag_along_flow = (sx*ucx + sy*ucy) / |u|, clipped to >= 0.
     This is the magnitude of sediment transport aligned with the local flow
     direction
  2. flood_ebb_sign = sign(ucx) — whether the flow itself is landward (+) or
     seaward (-) at the snapshot moment
  s_along = transport_mag_along_flow * flood_ebb_sign

"""

# %% Imports
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths
from FUNCTIONS.F_general import _parse_pm_n

# %% --- CONFIGURATION ---

# Scenarios: label → full path to the run folder
SCENARIOS = {
    # Q = 250 m3/s
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q250\detailed-hydro-run\dhr_01_Qr250_pm1_n0_mean.10280149"),
    'mean_flow1': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q250\detailed-hydro-run\dhr_06_Qr250_pm4_n3_mean.10280150"),
    'mean_flow2': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q250\detailed-hydro-run\dhr_09_Qr250_pm5_n3_mean.10280151"),
    'mean_flow3': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q250\detailed-hydro-run\dhr_10_Qr250_pm3_n3_mean.10280152"),
    'mean_flow4': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q250\detailed-hydro-run\dhr_11_Qr250_pm2_n3_mean.10280153"),

    # Q = 500 m3/s
    # 'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    # 'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    # 'peak_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    # 'mean_flow1': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_06_Qr500_pm4_n3_mean.10280084"),
    # 'mean_flow2': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_09_Qr500_pm5_n3_mean.10280083"),
    # 'mean_flow3': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_10_Qr500_pm3_n3_mean.10280082"),
    # 'mean_flow4': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_11_Qr500_pm2_n3_mean.10280081"),
    # 'mean_flow5': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503")
}

LOAD_VARS = ['mesh2d_sxtot', 'mesh2d_sytot', 'mesh2d_ucx', 'mesh2d_ucy'] # load sediment transport and velocity components

MIN_VELOCITY_FOR_PROJECTION = 0.0001

# time window for calculation
WINDOW_HOURS = 24.0

# spatial zoom settings
ZOOM      = True
ZOOM_XLIM = (20000, 30000)
ZOOM_YLIM = (5000, 10000)

# cache settings 
CACHE_BBOX        = [1, 1, 45000, 15000]
CACHE_TAG         = None
APPEND_TIMESTEPS  = True
APPEND_VARIABLES  = True

# colour scale for net transport (signed, diverging, centered at 0).
VMAX = 5e-7
VMIN = -VMAX
CMAP = plt.cm.RdBu   # red = negative (ebb/seaward), blue = positive (flood/landward)
VAR_LABEL = r'net sediment transport [$m^3\,s^{-1}\,m^{-1}$]'

# optional: overlay a light quiver of the time-averaged net (sx, sy) vector for QC,
SHOW_QUIVER       = False
QUIVER_STRIDE     = 10
QUIVER_SCALE      = 60

# output
FRAME_DPI = 300
STYLE = 'default'
plt.rcParams.update(plt.rcParamsDefault)
_tc = plt.rcParams['text.color']

# %% --- PROCESSING ---
scen_pm_n = {}

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name

    pm_n_match = re.search(r'pm\d+_n\d+', folder_name)
    pm, n = _parse_pm_n(pm_n_match.group(0)) if pm_n_match else (None, None)
    pm_n_str = f"peak amplitude {pm} | peak frequency {n}" if pm is not None else ""
    display_label = re.sub(r'\d+$', '', label)   # 'mean_flow3' -> 'mean_flow'

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

        # --- compute flood/ebb transport magnitude, then time-integrate ---
        # sxtot/sytot have shape (time, nSedTot, nFaces) → sum over sediment fractions first
        sx = ds_window['mesh2d_sxtot'].sum(dim='nSedTot').values   # (n_window, nFaces)
        sy = ds_window['mesh2d_sytot'].sum(dim='nSedTot').values   # (n_window, nFaces)
        ucx = ds_window['mesh2d_ucx'].values                        # (n_window, nFaces)
        ucy = ds_window['mesh2d_ucy'].values                        # (n_window, nFaces)

        u_mag = np.sqrt(ucx**2 + ucy**2) #compute velocity magnitude and use to define the local flow direction

        # STEP 1 — transport magnitude aligned with the local flow direction.
        with np.errstate(invalid='ignore', divide='ignore'):
            transport_mag_along_flow = np.where(
                u_mag > MIN_VELOCITY_FOR_PROJECTION,
                (sx * ucx + sy * ucy) / np.where(u_mag > 0, u_mag, 1),
                0.0,
            )
        
        transport_mag_along_flow = np.clip(transport_mag_along_flow, 0.0, None)

        # STEP 2 — assign flood(+)/ebb(-) sign based on whether the flow is
        # landward or seaward at that instant. 
        flood_ebb_sign = np.sign(ucx)

        s_along = transport_mag_along_flow * flood_ebb_sign

        # time coordinate in seconds, for trapezoidal integration
        t_seconds = (ds_window['time'].values - ds_window['time'].values[0]) / np.timedelta64(1, 's')
        t_seconds = t_seconds.astype(float)

        # Integrate the signed along-flow transport over time → [m^2] per face over the window
        net_transport_integrated = np.trapezoid(s_along, x=t_seconds, axis=0)

        # Time-averaged signed along-flow flux over the window [m^2/s], same units as instantaneous sxtot/sytot.
        elapsed_seconds = t_seconds[-1] - t_seconds[0]
        net_transport_per_cycle = net_transport_integrated / elapsed_seconds

        net_da = ds_window['mesh2d_sxtot'].isel(time=0, nSedTot=0).copy(data=net_transport_per_cycle)

        # --- Optional QC quiver: net (sx, sy) vector, oriented/colored by s_along sign ---
        if SHOW_QUIVER:
            net_sx_integrated = np.trapezoid(sx, x=t_seconds, axis=0) / elapsed_seconds
            net_sy_integrated = np.trapezoid(sy, x=t_seconds, axis=0) / elapsed_seconds
            face_x = ds.grid.face_coordinates[:, 0]
            face_y = ds.grid.face_coordinates[:, 1]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(14, 4))

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
        ax.set_title(
            f"{display_label.replace('_', ' ')} | {pm_n_str}",
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
        if ZOOM:
            ax.set_xlim(ZOOM_XLIM)
            ax.set_ylim(ZOOM_YLIM)
            fig_name = f"zoom_net_sed_transport_{label}_{folder_name}.png"        
        else:
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