"""Calculate the intertidal area (area that falls wet AND dry at least once during
one tidal cycle) for each detailed-hydro-run scenario, and produce an animated GIF
showing the wet/dry state evolving over that cycle, with the classified intertidal
zone overlaid as a static outline/hatch on every frame. The computed intertidal
area is shown in the legend/title of every frame.

Definitions:
  wet         : mesh2d_waterdepth > WET_THRESHOLD            (per timestep)
  always wet  : wet at every timestep in the window           → SUBTIDAL  (excluded)
  always dry  : dry at every timestep in the window            → SUPRATIDAL/land (excluded)
  intertidal  : wet at >=1 timestep AND dry at >=1 timestep    → INTERTIDAL (included)

Area is computed from mesh2d_flowelem_ba (per-face surface area, m²), summed only
over cells classified as intertidal.

Window: last 12h of the dataset (fixed duration, per scenario, taken from each
run's own final timestep).
"""

# %% Imports
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- CONFIGURATION ---
SCENARIOS = {
    # 'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q1000\detailed-hydro-run\dhr_01_Qr1000_pm1_n0_mean.10302507"),
    # 'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q1000\detailed-hydro-run\dhr_09_Qr1000_pm5_n3_mean.10302529")
    
    # 'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q250\detailed-hydro-run\dhr_01_Qr250_pm1_n0.10280149"),
    # 'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q250\detailed-hydro-run\dhr_09_Qr250_pm5_n3_mean.10280151")
    
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_09_Qr500_pm5_n3_mean.10280083")

    # 'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    # 'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    # 'peak_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    # 'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503")
}

LOAD_VARS = ['mesh2d_waterdepth', 'mesh2d_flowelem_ba']

WET_THRESHOLD  = 0.001    # [m] — depths above this are "wet"
WINDOW_HOURS   = 12.0    # fixed window length, taken from the end of each run

# Spatial zoom (model coordinates [m]) — same as your other scripts
ZOOM      = True
ZOOM_XLIM = (20000, 45000)
ZOOM_YLIM = (5000, 10000)

# Cache settings
CACHE_BBOX        = [1, 1, 45000, 15000]
CACHE_TAG         = 'intertidals'
APPEND_TIMESTEPS  = True
APPEND_VARIABLES  = True

# GIF settings
GIF_FPS   = 2
FRAME_DPI = 150

# Wet/dry animation colours
COLOR_WET = '#3b6fa0'
COLOR_DRY = '#d9c9a5'

# Static intertidal-zone overlay style (drawn the same on every frame)
# Static intertidal-zone overlay style (drawn the same on every frame)
INTERTIDAL_OUTLINE_COLOR = 'red'
INTERTIDAL_OUTLINE_WIDTH = 0.3      # was 0.8 — thinner edges so they don't visually merge

STYLE = 'default'
plt.rcParams.update(plt.rcParamsDefault)
_tc = plt.rcParams['text.color']

# %% --- PROCESSING ---

results_summary = {}   # label -> intertidal area [m^2], for a combined summary print at the end

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    output_dir = base_path / 'output_plots' / 'intertidal_area' / label
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

        if 'mesh2d_waterdepth' not in ds:
            print(f"  [SKIP] Variable 'mesh2d_waterdepth' not found in dataset.")
            continue

        if 'mesh2d_flowelem_ba' not in ds:
            print(f"  [SKIP] Variable 'mesh2d_flowelem_ba' not found in dataset.")
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

        actual_window_hours = (time_values[window_mask][-1] - time_values[window_mask][0]) / np.timedelta64(1, 'h')
        print(f"  Using last {WINDOW_HOURS}h: {n_window} timesteps "
              f"({time_values[window_mask][0]} → {time_values[window_mask][-1]}, "
              f"actual span {actual_window_hours:.2f}h)")
        if actual_window_hours < 11.0:
            print(f"  [WARNING] Actual available window is only {actual_window_hours:.1f}h — "
                  f"shorter than one tidal cycle. Wet/dry classification may not capture a "
                  f"full cycle. Interpret with caution.")

        ds_window = ds.isel(time=np.where(window_mask)[0])

        # --- Wet/dry classification over the window ---
        depth_vals = ds_window['mesh2d_waterdepth'].values   # (n_window, nFaces)
        wet_mask_t = depth_vals > WET_THRESHOLD               # (n_window, nFaces), bool

        always_wet = wet_mask_t.all(axis=0)
        always_dry = (~wet_mask_t).all(axis=0)
        intertidal = ~always_wet & ~always_dry

        n_faces = wet_mask_t.shape[1]
        print(f"  Faces: {n_faces}  | always wet: {always_wet.sum()}  "
              f"| always dry: {always_dry.sum()}  | intertidal: {intertidal.sum()}")
        
        print(f"  Always wet: {always_wet.sum()} ({100*always_wet.sum()/n_faces:.1f}%)")
        print(f"  Always dry: {always_dry.sum()} ({100*always_dry.sum()/n_faces:.1f}%)")
        print(f"  Intertidal: {intertidal.sum()} ({100*intertidal.sum()/n_faces:.1f}%)")

        # Check noise floor on cells that are dry almost always
        near_threshold = depth_vals[:, always_dry]
        if near_threshold.size:
            print(f"  Max depth among 'always dry' cells: {np.nanmax(near_threshold):.4f} m")

        # Distribution of max depth reached by 'intertidal' cells
        max_depth_intertidal = depth_vals[:, intertidal].max(axis=0)
        print(f"  Intertidal cells max-depth distribution: "
            f"min={max_depth_intertidal.min():.3f}, "
            f"p50={np.percentile(max_depth_intertidal,50):.3f}, "
            f"p95={np.percentile(max_depth_intertidal,95):.3f}, "
            f"max={max_depth_intertidal.max():.3f}")

        # Distribution of min depth reached by 'intertidal' cells (should dip below threshold)
        min_depth_intertidal = depth_vals[:, intertidal].min(axis=0)
        print(f"  Intertidal cells min-depth distribution: "
            f"min={min_depth_intertidal.min():.3f}, "
            f"p50={np.percentile(min_depth_intertidal,50):.3f}, "
            f"p95={np.percentile(min_depth_intertidal,95):.3f}, "
            f"max={min_depth_intertidal.max():.3f}")

        # --- Area calculation ---
        # mesh2d_flowelem_ba should be static (no time dim) — take first timestep's values
        # to be safe in case it carries a time dimension in the cached dataset.
        ba_da = ds_window['mesh2d_flowelem_ba']
        if 'time' in ba_da.dims:
            ba_vals = ba_da.isel(time=0).values
        else:
            ba_vals = ba_da.values

        intertidal_area_m2 = float(np.nansum(ba_vals[intertidal]))
        intertidal_area_km2 = intertidal_area_m2 / 1e6
        results_summary[label] = intertidal_area_m2
        print(f"  Intertidal area: {intertidal_area_m2:,.0f} m²  ({intertidal_area_km2:.4f} km²)")

        # --- Prepare static intertidal-zone overlay (boolean DataArray for plotting) ---
        intertidal_da = ds_window['mesh2d_waterdepth'].isel(time=0).copy(
            data=intertidal.astype(float)
        )

        # --- Build wet/dry colormap (binary) ---
        wetdry_cmap = mcolors.ListedColormap([COLOR_DRY, COLOR_WET])
        wetdry_norm = mcolors.BoundaryNorm([0, 0.5, 1], wetdry_cmap.N)

        frame_paths = []

        for idx in range(n_window):
            dt_str = str(np.datetime_as_string(
                np.datetime64(time_values[window_mask][idx], 'ns'), unit='s'
            )).replace('T', ' ')
            dt_tag = str(np.datetime_as_string(
                np.datetime64(time_values[window_mask][idx], 'ns'), unit='m'
            )).replace(':', '')
            print(f"  [{idx+1}/{n_window}] {dt_str}")

            wet_da_t = ds_window['mesh2d_waterdepth'].isel(time=idx).copy(
                data=wet_mask_t[idx].astype(float)
            )

            fig, ax = plt.subplots(figsize=(12, 4))

            # Animated wet/dry state
            wet_da_t.ugrid.plot(
                ax=ax,
                cmap=wetdry_cmap,
                norm=wetdry_norm,
                add_colorbar=False,
                edgecolors='none',
            )

            # Static intertidal zone overlay — outline via a contour-like boundary plot.
            # Drawing the boolean intertidal field with a transparent fill and hatched/edged
            # style on top, so it reads as an outline over the animated wet/dry base layer.
            try:
                intertidal_da.ugrid.plot(
                    ax=ax,
                    cmap=mcolors.ListedColormap(['none', 'red']),
                    norm=wetdry_norm,
                    add_colorbar=False,
                    edgecolors='none',
                    alpha=0.35,   # transparent fill so wet/dry base layer still shows through
                )
            except TypeError:
                # Some xugrid/matplotlib versions don't accept hatch via ugrid.plot kwargs —
                # fall back to outline-only without hatching.
                intertidal_da.ugrid.plot(
                    ax=ax,
                    cmap=mcolors.ListedColormap(['none', 'none']),
                    norm=wetdry_norm,
                    add_colorbar=False,
                    edgecolors=INTERTIDAL_OUTLINE_COLOR,
                    linewidths=INTERTIDAL_OUTLINE_WIDTH,
                )

            ax.set_aspect('equal')
            if ZOOM:
                ax.set_xlim(ZOOM_XLIM)
                ax.set_ylim(ZOOM_YLIM)

            ax.set_title(
                f"wet/dry | {label} | {dt_str}\n"
                f"intertidal area (this cycle): {intertidal_area_km2:.3f} km² "
                f"({intertidal_area_m2:,.0f} m²)",
                color=_tc,
                fontsize=10,
            )
            ax.set_xlabel('x [m]', color=_tc)
            ax.set_ylabel('y [m]', color=_tc)

            # Manual legend: wet, dry, intertidal zone outline
            legend_handles = [
                plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=COLOR_WET,
                           markeredgecolor='none', markersize=10, label='wet'),
                plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=COLOR_DRY,
                           markeredgecolor='none', markersize=10, label='dry'),
                plt.Line2D([0], [0], color=INTERTIDAL_OUTLINE_COLOR,
                           linewidth=INTERTIDAL_OUTLINE_WIDTH * 2,
                           label=f'intertidal zone\n({intertidal_area_km2:.3f} km²)'),
            ]
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.85)

            plt.tight_layout()

            frame_name = f"frame_{idx:04d}_{dt_tag}_{folder_name}.png"
            frame_path = output_dir / frame_name
            plt.savefig(frame_path, dpi=FRAME_DPI, bbox_inches='tight')
            plt.close(fig)
            frame_paths.append(frame_path)

        # --- Assemble GIF ---
        if frame_paths:
            gif_path = output_dir / f"intertidal_wetdry_{label}_{folder_name}.gif"
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
print("SUMMARY — intertidal area per scenario")
print("=" * 60)
for label, area_m2 in results_summary.items():
    print(f"  {label:12s}: {area_m2:>14,.0f} m²   ({area_m2/1e6:.4f} km²)")
print("=" * 60)
print("DONE")
# %%