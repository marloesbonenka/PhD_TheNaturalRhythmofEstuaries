"""Plot water depth (mesh2d_waterdepth) for every map timestep and create a GIF
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

from FUNCTIONS.F_general import create_water_colormap
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- CONFIGURATION ---

# Scenarios: label → full path to the run folder
SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'peak_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow.9728503"),
}

VAR_NAME   = 'mesh2d_waterdepth'
VAR_LABEL  = 'Water Depth [m]'
VMIN, VMAX = 0, 10          # colour scale — adjust to your data range

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

# %% --- PROCESSING ---

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path     = folder_path.parent   # detailed-hydro-run dir
    folder_name   = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    _zoom_tag = f"_zoom_x{ZOOM_XLIM[0]}-{ZOOM_XLIM[1]}_y{ZOOM_YLIM[0]}-{ZOOM_YLIM[1]}" if ZOOM else ""
    output_dir = base_path / 'output_plots' / 'waterdepth_gif' / label
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve run paths (handles potential timed-out stitching) ---
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_name,
        timed_out_dir=None,
        variability_map=None,
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]   # fall back to the folder itself

    # --- Load / update cache ---
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=[VAR_NAME],
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

        cmap = create_water_colormap()
        frame_paths = []

        # --- Plot each timestep ---
        for idx in range(len(time_values)):
            dt_str  = str(np.datetime_as_string(np.datetime64(time_values[idx], 'ns'), unit='s')).replace('T', ' ')
            dt_tag  = str(np.datetime_as_string(np.datetime64(time_values[idx], 'ns'), unit='D'))
            print(f"  [{idx+1}/{len(time_values)}] {dt_str}")

            data_t = ds.isel(time=idx)[VAR_NAME]

            fig, ax = plt.subplots(figsize=(12, 4))
            pc = data_t.ugrid.plot(
                ax=ax,
                cmap=cmap,
                add_colorbar=False,
                edgecolors='none',
                vmin=VMIN,
                vmax=VMAX,
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
            gif_path = output_dir / f"waterdepth_{label}_{folder_name}.gif"
            frames = [Image.open(fp) for fp in frame_paths]
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=int(1000 / GIF_FPS),  # ms per frame
                loop=0,
            )
            # Close images to release file handles
            for f in frames:
                f.close()
            print(f"\n  GIF saved: {gif_path}")

    finally:
        ds.close()

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
