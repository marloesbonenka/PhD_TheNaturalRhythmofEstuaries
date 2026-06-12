"""Compute and plot intertidal / subtidal / supratidal masks
from mesh2d_waterdepth for detailed-hydro-run scenarios.

Classification is based on tidal-cycle alternation (no fixed thresholds):
  Intertidal  : cell goes wet AND dry in at least one tidal cycle
  Subtidal    : always wet — never dries in any cycle
  Supratidal  : never wet — always dry in every cycle
"""

# %% Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# %% --- CONFIGURATION ---

SCENARIOS = {
    'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'peak_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
}

# Wet/dry threshold [m] — exclude numerical noise
WET_THRESHOLD = 0.05

# Tidal cycle length in timesteps (= hours, given hourly map output)
# M2 semi-diurnal period ≈ 12 h  →  2 high + 2 low waters per day
TIDAL_CYCLE_HOURS = 12

# Spin-up: skip this many timesteps before computing wet fraction
SPINUP_STEPS = 0   # set > 0 if your run has a ramp-up period

# Spatial zoom
ZOOM      = True
ZOOM_XLIM = (19000, 45000)
ZOOM_YLIM = (5000, 10000)

# Cache settings
CACHE_BBOX       = [1, 1, 45000, 15000]
CACHE_TAG        = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

FRAME_DPI = 150

# %% --- ZONE COLORMAP ---
# 0 = supratidal (white), 1 = intertidal (green), 2 = subtidal (blue)
ZONE_CMAP   = mcolors.ListedColormap(["#0E912A", "#75461A", '#2166ac'])
ZONE_BOUNDS = [0, 1, 2, 3]
ZONE_NORM   = mcolors.BoundaryNorm(ZONE_BOUNDS, ZONE_CMAP.N)

plt.rcParams.update(plt.rcParamsDefault)

# %% --- PROCESSING ---

summary = {}   # label → zone areas

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    output_dir = base_path / 'output_plots' / 'intertidal' / label
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

    # --- Load cache ---
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=['mesh2d_waterdepth', 'mesh2d_flowelem_ba', 'mesh2d_flowelem_bl'],
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
            print(f"  [SKIP] No time dimension.")
            continue
        if 'mesh2d_waterdepth' not in ds:
            print(f"  [SKIP] mesh2d_waterdepth not found.")
            continue

        wd = ds['mesh2d_waterdepth']   # (time, face)

        # Skip spin-up timesteps if configured
        if SPINUP_STEPS > 0:
            wd = wd.isel(time=slice(SPINUP_STEPS, None))

        n_times = len(wd.time)
        print(f"  Timesteps used: {n_times}  (skipped {SPINUP_STEPS} spin-up)")

        # ------------------------------------------------------------------
        # 1. WET FRACTION per cell across all timesteps
        # ------------------------------------------------------------------
        wet_fraction = (wd > WET_THRESHOLD).mean(dim='time')   # [0..1] per face

        # ------------------------------------------------------------------
        # 2. ZONE CLASSIFICATION from tidal-cycle alternation
        #    0 = supratidal  (never wet in any cycle)
        #    1 = intertidal  (goes wet AND dry in ≥1 tidal cycle)
        #    2 = subtidal    (always wet, never dries in any cycle)
        # ------------------------------------------------------------------
        wd_np = wd.values   # (time, nFaces)
        n_steps = wd_np.shape[0]
        n_cycles = n_steps // TIDAL_CYCLE_HOURS
        if n_cycles == 0:
            raise ValueError(
                f"Not enough timesteps ({n_steps}) for even one "
                f"tidal cycle of {TIDAL_CYCLE_HOURS} h."
            )
        # Trim to complete cycles and reshape → (n_cycles, cycle_len, nFaces)
        wd_cycles = wd_np[: n_cycles * TIDAL_CYCLE_HOURS].reshape(
            n_cycles, TIDAL_CYCLE_HOURS, wd_np.shape[1]
        )
        print(f"  Tidal cycles used: {n_cycles}  ({TIDAL_CYCLE_HOURS} h each)")

        wet_in_cycle = (wd_cycles > WET_THRESHOLD).any(axis=1)   # (n_cycles, nFaces)
        dry_in_cycle = (wd_cycles <= WET_THRESHOLD).any(axis=1)  # (n_cycles, nFaces)

        # Intertidal: wet AND dry in at least one cycle
        intertidal_mask  = (wet_in_cycle & dry_in_cycle).any(axis=0)
        # Subtidal: wet in every cycle and never dries
        subtidal_mask    = wet_in_cycle.all(axis=0) & (~dry_in_cycle.any(axis=0))
        # Supratidal: never wet in any cycle (always dry)
        supratidal_mask  = ~wet_in_cycle.any(axis=0)

        # Priority: intertidal > subtidal > supratidal
        zone_values = np.where(intertidal_mask, 1,
                      np.where(subtidal_mask,   2, 0))

        # Wrap into xarray DataArray on the same grid for ugrid plotting
        zone = xr.DataArray(
            zone_values,
            dims=wet_fraction.dims,
            coords=wet_fraction.coords,
        )

        # ------------------------------------------------------------------
        # 3. INTERTIDAL AREA [m²] using mesh2d_flowelem_ba
        # ------------------------------------------------------------------
        # Align face_areas to the same face dimension as wet_fraction,
        # in case the cached variables have different spatial extents.
        ba = ds['mesh2d_flowelem_ba']
        face_dim = wet_fraction.dims[0]  # e.g. 'mesh2d_nFaces'
        if face_dim in ba.dims:
            ba = ba.sel({face_dim: wet_fraction[face_dim]})
        face_areas = ba.values   # [m²], shape (nFaces,)

        intertidal_area = float(face_areas[zone_values == 1].sum())
        subtidal_area   = float(face_areas[zone_values == 2].sum())
        supratidal_area = float(face_areas[zone_values == 0].sum())
        total_area      = intertidal_area + subtidal_area + supratidal_area

        summary[label] = {
            'intertidal': intertidal_area,
            'subtidal':   subtidal_area,
            'supratidal': supratidal_area,
        }

        print(f"  Supratidal : {supratidal_area/1e6:,.3f} km²  ({100*supratidal_area/total_area:.1f}%)")
        print(f"  Intertidal : {intertidal_area/1e6:,.3f} km²  ({100*intertidal_area/total_area:.1f}%)")
        print(f"  Subtidal   : {subtidal_area/1e6:,.3f} km²  ({100*subtidal_area/total_area:.1f}%)")
        print(f"  Total      : {total_area/1e6:,.3f} km²")

        # ------------------------------------------------------------------
        # 4. PLOT A — Categorical zone map
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 4))

        pc = zone.ugrid.plot(
            ax=ax,
            cmap=ZONE_CMAP,
            norm=ZONE_NORM,
            add_colorbar=False,
            edgecolors='none',
        )
        ax.set_aspect('equal')
        if ZOOM:
            ax.set_xlim(ZOOM_XLIM)
            ax.set_ylim(ZOOM_YLIM)

        ax.set_title(f"Tidal Zone Classification | {label}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(norm=ZONE_NORM, cmap=ZONE_CMAP),
            cax=cax, ticks=[0.5, 1.5, 2.5],
        )
        cbar.ax.set_yticklabels(['Supratidal', 'Intertidal', 'Subtidal'])

        plt.tight_layout()
        plt.savefig(output_dir / f"zone_map_{label}.png", dpi=FRAME_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: zone_map_{label}.png")

        # ------------------------------------------------------------------
        # 5. PLOT B — Continuous wet fraction map
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 4))

        pc = wet_fraction.ugrid.plot(
            ax=ax,
            cmap='RdYlBu',
            vmin=0, vmax=1,
            add_colorbar=False,
            edgecolors='none',
        )
        ax.set_aspect('equal')
        if ZOOM:
            ax.set_xlim(ZOOM_XLIM)
            ax.set_ylim(ZOOM_YLIM)

        ax.set_title(f"Wet Fraction (0 = always dry, 1 = always wet) | {label}")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(pc, cax=cax)
        cbar.set_label("Wet fraction [-]")

        plt.tight_layout()
        plt.savefig(output_dir / f"wet_fraction_{label}.png", dpi=FRAME_DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: wet_fraction_{label}.png")

        # ------------------------------------------------------------------
        # 6. PLOT C — Bed level distribution per zone (validation)
        # ------------------------------------------------------------------
        if 'mesh2d_flowelem_bl' in ds:
            zb = ds['mesh2d_flowelem_bl'].values   # static bed level (nFaces,)

            fig, ax = plt.subplots(figsize=(7, 4))
            for zone_id, zone_name, color in zip(
                [0, 1, 2],
                ['Supratidal', 'Intertidal', 'Subtidal'],
                ['#d4b483', '#5aab61', '#2166ac'],
            ):
                mask = zone_values == zone_id
                if mask.sum() > 0:
                    ax.hist(zb[mask], bins=50, alpha=0.6,
                            label=f"{zone_name} (n={mask.sum():,})",
                            color=color, density=True)

            ax.set_xlabel("Bed level [m]")
            ax.set_ylabel("Density [-]")
            ax.set_title(f"Bed level distribution per tidal zone | {label}")
            ax.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"bedlevel_per_zone_{label}.png",
                        dpi=FRAME_DPI, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: bedlevel_per_zone_{label}.png")

    finally:
        ds.close()

# %% --- CROSS-SCENARIO SUMMARY BAR CHART ---

if summary:
    fig, ax = plt.subplots(figsize=(7, 4))

    labels     = list(summary.keys())
    supra_vals = [summary[l]['supratidal'] / 1e6 for l in labels]
    inter_vals = [summary[l]['intertidal'] / 1e6 for l in labels]
    sub_vals   = [summary[l]['subtidal']   / 1e6 for l in labels]

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, supra_vals, w, label='Supratidal', color='#d4b483')
    ax.bar(x,     inter_vals, w, label='Intertidal', color='#5aab61')
    ax.bar(x + w, sub_vals,   w, label='Subtidal',   color='#2166ac')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Area [km²]")
    ax.set_title("Tidal zone areas by scenario")
    ax.legend()
    plt.tight_layout()

    summary_path = (
        list(SCENARIOS.values())[0].parent
        / 'output_plots' / 'intertidal' / 'zone_area_summary.png'
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(summary_path, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSummary bar chart saved: {summary_path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)