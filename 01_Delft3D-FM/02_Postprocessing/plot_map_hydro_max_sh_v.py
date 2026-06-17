"""
Along-estuary diagram of p95 velocity / shear stress
+ location of maximum flood-directed sediment transport
for the last N days of each detailed-hydro-run scenario.

Output:
  1. PNG per variable  (x = along-estuary, y = time, colour = p95)
  2. Flood-transport intrusion PNG  (time series of x-location of max flood transport)

Convention in model output:
  - Positive transport  → ebb-directed
  - Negative transport  → flood-directed
"""

# %% Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import pandas as pd

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'peak_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
}

# Variables to produce max diagrams for
VARIABLES = [
    'velocity',        # uses mesh2d_ucmag
    'shear stress',    # uses mesh2d_taus
]

# Along-estuary binning — restricted to 20–45 km
X_MIN = 20000              # metres
X_MAX = 45000              # metres
X_BIN_WIDTH = 150          # metres — width of each along-estuary bin

# Percentile for cross-section aggregation (p95 removes outliers)
PERCENTILE = 70

# Last N days to analyse
N_DAYS = 1

# Spatial domain
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG  = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Colour limits
CLIM = {
    'velocity':    (0, 1.5),   # m/s
    'shear stress': (0, 5.0),  # Pa
}
CMAP = {
    'velocity':    'plasma',
    'shear stress': 'hot_r',
}

# Output
OUTPUT_BASE = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\output_plots\max_shear_velocity")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

SCENARIO_COLORS = {
    'constant':  '#4477AA',
    'low_flow':  '#66CCEE',
    'peak_flow': '#EE6677',
    'mean_flow': '#CCBB44',
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_last_n_days_window(ds, n_days):
    """Return (t_start, t_end) as numpy datetime64 covering the last n_days."""
    t_end   = ds.time.values[-1]
    t_start = t_end - np.timedelta64(int(n_days * 24 * 3600), 's')
    return t_start, t_end


def build_x_bins(x_min, x_max, bin_width):
    edges = np.arange(x_min, x_max + bin_width, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return edges, centres


def face_x_coords(ds):
    """Extract face x-coordinates from the unstructured grid."""
    return ds.grid.face_coordinates[:, 0]


def assign_bin_indices(face_x, bin_edges):
    """For each face, return the bin index (0-based). Faces outside range → -1."""
    idx = np.searchsorted(bin_edges[1:], face_x, side='left')
    idx[face_x < bin_edges[0]]  = -1
    idx[face_x >= bin_edges[-1]] = -1
    return idx


def compute_p95(ds, var_name, face_x, bin_edges, t_start, t_end):
    """
    For each timestep and each x-bin compute the p95 of var_name across
    all faces in that bin.  Returns (times, bin_centres, max_array).
    """
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_bins = len(bin_centres)

    # Time slice
    ds_sel = ds.sel(time=slice(t_start, t_end))
    times  = ds_sel.time.values
    n_t    = len(times)

    bin_idx = assign_bin_indices(face_x, bin_edges)

    result = np.full((n_t, n_bins), np.nan)

    for ti in range(n_t):
        data_t = ds_sel[var_name].isel(time=ti).values   # (nFaces,)
        for bi in range(n_bins):
            mask = bin_idx == bi
            vals = data_t[mask]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                result[ti, bi] = np.nanpercentile(vals, PERCENTILE)

    return times, bin_centres, result

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

# Storage for data
max_data = {v: {} for v in VARIABLES}   # var → label → (times, bin_centres, matrix)

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

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

    # --- Determine which variables to load ---
    load_vars = set()
    for var_label in VARIABLES:
        if var_label == 'velocity':
            load_vars.update(['mesh2d_ucmag', 'mesh2d_ucx', 'mesh2d_ucy'])

        elif var_label == 'shear stress':
            load_vars.add('mesh2d_taus')
    load_vars.update(['mesh2d_sxtot', 'mesh2d_sytot'])

    # --- Load cache ---
    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=list(load_vars),
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds is None:
        print(f"  [SKIP] No data for {folder_name}")
        continue

    try:
        if 'time' not in ds.dims or len(ds.time) == 0:
            print(f"  [SKIP] No time dimension.")
            continue

        # --- Time window ---
        t_start, t_end = get_last_n_days_window(ds, N_DAYS)
        print(f"  Time window: {t_start} → {t_end}")

        # --- Face coordinates and x-bins (20–45 km only) ---
        fx = face_x_coords(ds)
        bin_edges, bin_centres = build_x_bins(
            x_min=X_MIN, x_max=X_MAX, bin_width=X_BIN_WIDTH
        )

        # --- Max for each variable ---
        for var_label in VARIABLES:
            if var_label == 'velocity':
                var_name = 'mesh2d_ucmag'
            elif var_label == 'shear stress':
                var_name = 'mesh2d_taus'

            if var_name not in ds:
                print(f"  [SKIP] {var_name} not in dataset")
                continue

            print(f"  Computing p{PERCENTILE} max for {var_label} ...")
            times, bcs, matrix = compute_p95(
                ds, var_name, fx, bin_edges, t_start, t_end
            )
            max_data[var_label][label] = (times, bcs, matrix)
            print(f"    → shape {matrix.shape}")

    finally:
        ds.close()

# ---------------------------------------------------------------------------
# PLOT — Max diagrams (one figure per variable, one subplot per scenario)
# ---------------------------------------------------------------------------

CRITICAL_SHEAR_THRESHOLD = 0.15 #Pa

for var_label in VARIABLES:
    scenarios_available = [lbl for lbl in SCENARIOS if lbl in max_data[var_label]]
    if not scenarios_available:
        print(f"No data for {var_label}, skipping plot.")
        continue


    n_scen = len(scenarios_available)
    fig, axes = plt.subplots(
        n_scen, 1, figsize=(14, 3.5 * (n_scen)),
        sharex=True, sharey=False
    )
    if (n_scen) == 1:
        axes = [axes]

    vmin, vmax = CLIM[var_label]
    cmap_name  = CMAP[var_label]

    for ax, label in zip(axes, scenarios_available):
        times, bcs, matrix = max_data[var_label][label]

        # x = along-estuary distance [km], y = time
        
        # pcolormesh: rows=y (time), cols=x (distance) → matrix already (n_t, n_bins)
        pcm = ax.pcolormesh(
            bcs / 1000,    # x: along-estuary distance [km]
            times,     # y: time (matplotlib date numbers)
            matrix,        # (n_t, n_bins) — rows=time, cols=distance ✓
            cmap=cmap_name, vmin=vmin, vmax=vmax,
            shading='auto'
        )
        
        ax.set_ylabel('Time')
        ax.set_title(f'{var_label} — {label}  (p{PERCENTILE})')
        ax.yaxis_date()
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))

        cb = fig.colorbar(pcm, ax=ax, pad=0.02)
        cb.set_label(
            'velocity [m/s]' if var_label == 'velocity' else 'shear stress [Pa]'
        )

    

    axes[-1].set_xlabel('Along-estuary distance [km]')
    fig.suptitle(
        f'Along-estuary p{PERCENTILE} {var_label} — last {N_DAYS} days',
        fontsize=13, y=1.01
    )
    fig.tight_layout()

    out_path = OUTPUT_BASE / f"{var_label.replace(' ', '_')}_p{PERCENTILE}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved: {out_path}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)