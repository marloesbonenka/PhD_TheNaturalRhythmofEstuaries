"""Plot a variable (e.g., water depth (mesh2d_waterdepth)) for every map timestep and create a GIF
for three specific detailed-hydro-run scenarios:
  - constant flow
  - low flow snapshot
  - peak flow snapshot
"""

# %% Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import create_shear_stress_colormap, create_water_colormap
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

VARIABLE_TO_ANALYZE = 'sediment transport'    
                    # options are 
                    # 'water depth'            (mesh2d_waterdepth)
                    # 'shear stress'           (mesh2d_taus)
                    # 'velocity'               (mesh2d_ucmag + quiver of mesh2d_ucx/ucy)
                    # 'sediment transport'     (mesh2d_sxtot/sytot combined magnitude + quiver)


# Spatial zoom (model coordinates [m])
ZOOM     = True
ZOOM_XLIM = (19000, 45000)
ZOOM_YLIM = (5000, 10000)

# Cache settings
CACHE_BBOX         = [1, 1, 45000, 15000]
CACHE_TAG          = None
APPEND_TIMESTEPS   = True
APPEND_VARIABLES   = True

# GIF settings
GIF_FPS   = 2          # frames per second
FRAME_DPI = 150        # lower than PNG saves to keep file sizes manageable

# Output
STYLE  = 'default'     # 'default' → white bg;  'whitefig' → transparent
STYLES = {
    'default': {},
    'whitefig': {
        'figure.facecolor': 'none',
        'axes.facecolor':   'none',
        'axes.edgecolor':   'white',
        'axes.labelcolor':  'white',
        'xtick.color':      'white',
        'ytick.color':      'white',
        'text.color':       'white',
        'savefig.transparent': True,
    },
}
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])
_tc = plt.rcParams['text.color']

# Vector display settings (only used when VARIABLE_TO_ANALYZE == 'velocity')
QUIVER_STRIDE = 10       # plot every Nth wet cell — increase to reduce clutter
QUIVER_SCALE  = 60       # higher = shorter arrows, lower = longer
QUIVER_COLOR  = 'white'
QUIVER_WET_THRESHOLD = 0.01   # m/s — cells below this are treated as dry

# --- VARIABLE-SPECIFIC SETTINGS ---

if VARIABLE_TO_ANALYZE == 'water depth':
    VAR_NAME   = 'mesh2d_waterdepth'
    VAR_LABEL  = 'water depth [m]'
    VMIN, VMAX = 0, 10
    cmap       = create_water_colormap()
    LOAD_VARS  = [VAR_NAME]

elif VARIABLE_TO_ANALYZE == 'shear stress':
    VAR_NAME   = 'mesh2d_taus'
    VAR_LABEL  = 'shear stress [Pa]'
    VMIN, VMAX = 0, 5
    cmap       = create_shear_stress_colormap()
    LOAD_VARS  = [VAR_NAME]

elif VARIABLE_TO_ANALYZE == 'velocity':
    VAR_NAME   = 'mesh2d_ucmag'
    VAR_LABEL  = 'velocity [m/s]'
    VMIN, VMAX = 0, 2
    cmap       = plt.cm.plasma
    LOAD_VARS  = ['mesh2d_ucmag', 'mesh2d_ucx', 'mesh2d_ucy']   # all three needed

elif VARIABLE_TO_ANALYZE == 'sediment transport':
    VAR_NAME   = 'mesh2d_sxtot'   # exists in ds → passes the guard; colour fill uses smag_da instead
    VAR_LABEL  = 'sed. transport [m²/s]'
    VMIN, VMAX = 0, 1e-05
    cmap       = plt.cm.hot_r
    LOAD_VARS  = ['mesh2d_sxtot', 'mesh2d_sytot']

# %% --- PROCESSING ---

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    _zoom_tag  = f"_zoom_x{ZOOM_XLIM[0]}-{ZOOM_XLIM[1]}_y{ZOOM_YLIM[0]}-{ZOOM_YLIM[1]}" if ZOOM else ""
    output_dir = base_path / 'output_plots' / VARIABLE_TO_ANALYZE / label
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

        if VAR_NAME not in ds:
            print(f"  [SKIP] Variable '{VAR_NAME}' not found in dataset.")
            continue

        time_values = np.asarray(ds.time.values).astype('datetime64[ns]')
        print(f"  Timesteps: {len(time_values)}  ({time_values[0]} → {time_values[-1]})")

        # --- Pre-extract face coordinates once (needed for quiver) ---
        if VARIABLE_TO_ANALYZE in ('velocity', 'sediment transport'):
            face_x = ds.grid.face_coordinates[:, 0]
            face_y = ds.grid.face_coordinates[:, 1]

        frame_paths = []

        # --- Plot each timestep ---
        for idx in range(len(time_values)):
            dt_str = str(np.datetime_as_string(np.datetime64(time_values[idx], 'ns'), unit='s')).replace('T', ' ')
            dt_tag = str(np.datetime_as_string(np.datetime64(time_values[idx], 'ns'), unit='D'))
            print(f"  [{idx+1}/{len(time_values)}] {dt_str}")

            data_t = ds.isel(time=idx)

            fig, ax = plt.subplots(figsize=(12, 4))

            # --- Colour fill + vectors ---
            if VARIABLE_TO_ANALYZE == 'sediment transport':
                sx_t   = data_t['mesh2d_sxtot'].values.sum(axis=0)   # (nFaces,)
                sy_t   = data_t['mesh2d_sytot'].values.sum(axis=0)
                smag_t = np.sqrt(sx_t**2 + sy_t**2)

                smag_da = data_t['mesh2d_sxtot'].isel(nSedTot=0).copy(data=smag_t)
                pc = smag_da.ugrid.plot(
                    ax=ax,
                    cmap=cmap,
                    add_colorbar=False,
                    edgecolors='none',
                    vmin=VMIN,
                    vmax=VMAX,
                )

                wet      = smag_t > 1e-6
                idx_plot = np.where(wet)[0][::QUIVER_STRIDE]
                norm_t = np.where(smag_t[idx_plot] > 1e-10, smag_t[idx_plot], 1)   # avoid div/0
                ax.quiver(
                    face_x[idx_plot],          face_y[idx_plot],
                    sx_t[idx_plot] / norm_t,   sy_t[idx_plot] / norm_t,
                    scale=QUIVER_SCALE,       
                    color='lightgrey',
                    width=0.002,
                    headwidth=4,
                    zorder=5,
                )

            else:
                # Generic colour fill for water depth, shear stress, velocity
                pc = data_t[VAR_NAME].ugrid.plot(
                    ax=ax,
                    cmap=cmap,
                    add_colorbar=False,
                    edgecolors='none',
                    vmin=VMIN,
                    vmax=VMAX,
                )

                if VARIABLE_TO_ANALYZE == 'velocity':
                    ucx = data_t['mesh2d_ucx'].values
                    ucy = data_t['mesh2d_ucy'].values
                    mag = data_t['mesh2d_ucmag'].values

                    wet      = mag > QUIVER_WET_THRESHOLD
                    idx_plot = np.where(wet)[0][::QUIVER_STRIDE]
                    ax.quiver(
                        face_x[idx_plot], face_y[idx_plot],
                        ucx[idx_plot],    ucy[idx_plot],
                        scale=QUIVER_SCALE,
                        color=QUIVER_COLOR,
                        width=0.002,
                        headwidth=4,
                        zorder=5,
                    )

            ax.set_aspect('equal')
            if ZOOM:
                ax.set_xlim(ZOOM_XLIM)
                ax.set_ylim(ZOOM_YLIM)

            ax.set_title(f"{VAR_LABEL} | {label} | {dt_str}", color=_tc)

            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right", size="3%", pad=0.1)
            cbar    = plt.colorbar(pc, cax=cax)
            cbar.set_label(VAR_LABEL, color=_tc)
            cbar.ax.yaxis.set_tick_params(color=_tc)
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_tc)

            plt.tight_layout()

            frame_name = f"frame_{idx:04d}_{dt_tag}_{folder_name}.png"
            frame_path = output_dir / frame_name
            plt.savefig(frame_path, dpi=FRAME_DPI, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_path)
            print(f"    Saved: {frame_name}")

        # --- Assemble GIF ---
        if frame_paths:
            gif_path = output_dir / f"{VARIABLE_TO_ANALYZE}_{label}_{folder_name}.gif"
            frames = [Image.open(fp) for fp in frame_paths]
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / GIF_FPS),
                loop=0,
            )
            for f in frames:
                f.close()
            print(f"\n  GIF saved: {gif_path}")

    finally:
        ds.close()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
#%%