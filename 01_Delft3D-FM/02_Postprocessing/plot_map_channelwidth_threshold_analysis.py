# %% Channel-width threshold analysis from final MAP bed levels
"""
For each cross-section x and bed-elevation threshold z_threshold, calculate:

    W(z_threshold) = integral[ bed_level(y) < z_threshold ] dy

This is a purely geometric bed-level metric. It does NOT use water level,
wet/dry state, or the HIS-file cross_section_area variable.

Outputs
-------
1. Final bed-level profiles at selected cross-sections.
2. Width-below-threshold curves for selected cross-sections.
3. Along-estuary heatmap of:
       Delta W = W(pm5_n1) - W(pm3_n5)
4. Along-estuary width profiles for selected elevation thresholds.

Interpretation
--------------
Positive Delta W:
    pm5_n1 has more bed width below that fixed bed-elevation threshold.

Negative Delta W:
    pm3_n5 has more bed width below that threshold.

Important:
- Thresholds are fixed absolute bed elevations and are identical between
  scenarios.
- "Total width below threshold" can include multiple channels or isolated
  depressions. It is not automatically equivalent to the width of one
  connected, hydraulically active channel.
"""

# %%
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Add this only if your own helper functions are needed elsewhere.
# This script is standalone and does not require your FUNCTIONS modules.
sys.path.append(
    r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing"
)

SCENARIOS = {
    "pm3_n5": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\detailed-hydro-run"
        r"\dhr_03_Qr500_pm3_n5_mean.964903"
    ),
    "pm5_n1": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian"
        r"\Model_Output\Q500\detailed-hydro-run"
        r"\dhr_05_Qr500_pm5_n1_mean.964908"
    ),
}

# Comparison order matters: Delta W = second scenario minus first scenario.
COMPARE_LIST = ["pm3_n5", "pm5_n1"]

# Uses the same overall folder convention as your HIS-file script.
OUTPUT_DIR = (
    SCENARIOS[COMPARE_LIST[0]].parent
    / "output_plots"
    / "channel_width_thresholds"
    / f"comparison_{COMPARE_LIST[0]}_vs_{COMPARE_LIST[1]}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_SUBDIR = "output"
MAP_GLOB = "*_map.nc"

MAP_BEDLEVEL_VAR = "mesh2d_mor_bl"
MAP_FACE_X_VAR = "mesh2d_face_x"
MAP_FACE_Y_VAR = "mesh2d_face_y"

# Cross-sections: same nominal along-estuary locations as in the HIS analysis.
X_COORDS_M = np.arange(20_000, 45_000, 1_000)
CROSS_SECTION_NAMES = [f"km{int(x / 1000)}" for x in X_COORDS_M]

# Sampling strip across the idealised estuary.
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y_SAMPLES = 500

# Set False if you want to preserve all sampled bed elevations.
# True is recommended if cells representing elevated dry land occur in this strip.
APPLY_LAND_MASK = True
LAND_BEDLEVEL_THRESHOLD_M = 8.0

# Fixed absolute bed-level thresholds [m] used in both scenarios.
# Adjust after inspecting your profile plots. Finer spacing gives a smoother heatmap.
BED_THRESHOLDS_M = np.arange(-15.0, 8.01, 0.25)

# These cross-sections receive individual profile and W(z) plots.
SELECTED_PROFILE_KM = [20, 24, 28, 32, 36, 40, 44]

# These thresholds receive W(x) along-estuary comparison plots.
# Choose levels that represent deep channels, intermediate channel flanks,
# and shallow bars in your model.
PLOT_THRESHOLDS_M = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5]

# Plot appearance.
DPI = 180
SCENARIO_COLORS = {
    "pm3_n5": plt.cm.Blues(0.60),
    "pm5_n1": plt.cm.Blues(0.92),
}
DIFF_CMAP = "RdBu_r"


# %%
# =============================================================================
# FUNCTIONS
# =============================================================================

def open_map_dataset(model_folder):
    """
    Open MAP output from one scenario.

    If your MAP output is partitioned into multiple files, open_mfdataset
    combines them. If it is a single file, this works as well.
    """
    map_files = sorted((model_folder / MAP_SUBDIR).glob(MAP_GLOB))

    if not map_files:
        raise FileNotFoundError(
            f"No MAP files matching '{MAP_GLOB}' found in:\n"
            f"  {model_folder / MAP_SUBDIR}"
        )

    print(f"  Found {len(map_files)} MAP file(s) in {model_folder.name}")

    if len(map_files) == 1:
        ds = xr.open_dataset(map_files[0])
    else:
        ds = xr.open_mfdataset(
            [str(f) for f in map_files],
            combine="by_coords",
            parallel=False,
        )

    required = [MAP_BEDLEVEL_VAR, MAP_FACE_X_VAR, MAP_FACE_Y_VAR]
    missing = [name for name in required if name not in ds]
    if missing:
        ds.close()
        raise KeyError(
            f"Missing required MAP variable(s): {missing}\n"
            f"Available variables include:\n{list(ds.variables)[:30]}"
        )

    return ds


def get_face_dimension(ds):
    """
    Identify the face dimension from mesh2d_mor_bl.

    Usually this is 'mesh2d_nFaces', but it is determined automatically
    rather than hard-coded.
    """
    dims = ds[MAP_BEDLEVEL_VAR].dims

    if "time" not in dims:
        raise ValueError(
            f"Expected a time dimension in {MAP_BEDLEVEL_VAR}; found dims={dims}"
        )

    non_time_dims = [dim for dim in dims if dim != "time"]

    if len(non_time_dims) != 1:
        raise ValueError(
            f"Could not uniquely identify face dimension for "
            f"{MAP_BEDLEVEL_VAR}. Found dims={dims}"
        )

    return non_time_dims[0]


def get_final_bedlevel(ds):
    """
    Return final bed-level values as a 1-D NumPy array, plus final time.
    """
    final_da = ds[MAP_BEDLEVEL_VAR].isel(time=-1)

    if final_da.ndim != 1:
        raise ValueError(
            f"Expected final {MAP_BEDLEVEL_VAR} to be 1-D over faces, "
            f"but found dimensions {final_da.dims}."
        )

    final_time = ds["time"].isel(time=-1).values
    final_bedlevel = np.asarray(final_da.values, dtype=float)

    return final_bedlevel, final_time


def sample_final_profiles(ds, x_coords_m, y_range_m, n_y_samples):
    """
    Sample final bed level along cross-sections using nearest MAP faces.

    Returns
    -------
    y_samples : ndarray, shape (n_y_samples,)
        Cross-section y-coordinates [m].
    profiles : dict
        {cross_section_name: final bed-level profile}.
    final_time : numpy datetime-like
        Timestamp of final MAP output.
    """
    face_x = np.asarray(ds[MAP_FACE_X_VAR].values, dtype=float)
    face_y = np.asarray(ds[MAP_FACE_Y_VAR].values, dtype=float)
    final_bedlevel, final_time = get_final_bedlevel(ds)

    tree = cKDTree(np.column_stack([face_x, face_y]))

    y_samples = np.linspace(y_range_m[0], y_range_m[1], n_y_samples)
    profiles = {}

    for x_m in x_coords_m:
        cross_section_name = f"km{int(x_m / 1000)}"

        sample_points = np.column_stack(
            [
                np.full(y_samples.size, x_m, dtype=float),
                y_samples,
            ]
        )

        _, nearest_face_indices = tree.query(sample_points)
        profile = final_bedlevel[nearest_face_indices].astype(float)

        if APPLY_LAND_MASK:
            profile[profile > LAND_BEDLEVEL_THRESHOLD_M] = np.nan

        profiles[cross_section_name] = profile

    return y_samples, profiles, final_time


def width_below_threshold(profile, y_m, threshold_m):
    """
    Calculate total cross-sectional width where bed level is below threshold.

    Uses sampled y-spacing as the representative width of each sample point.
    NaN values do not contribute to width.
    """
    dy = np.nanmedian(np.diff(y_m))
    below_threshold = np.isfinite(profile) & (profile < threshold_m)
    return np.sum(below_threshold) * dy


def calculate_width_matrix(profiles, y_m, thresholds_m):
    """
    Compute W(x, z) for all cross-sections and thresholds.

    Returns
    -------
    width_matrix : ndarray, shape (n_thresholds, n_cross_sections)
        width_matrix[i, j] is width below thresholds_m[i] at cross-section j.
    """
    section_names = list(profiles.keys())

    width_matrix = np.full(
        (len(thresholds_m), len(section_names)),
        np.nan,
        dtype=float,
    )

    for j, section_name in enumerate(section_names):
        profile = profiles[section_name]

        for i, threshold_m in enumerate(thresholds_m):
            width_matrix[i, j] = width_below_threshold(
                profile=profile,
                y_m=y_m,
                threshold_m=threshold_m,
            )

    return width_matrix, section_names


def threshold_label(threshold_m):
    """Create a safe compact label for filenames."""
    return f"{threshold_m:+.2f}".replace("+", "pos").replace("-", "neg").replace(".", "p")


def get_symmetric_limit(array):
    """Return a robust symmetric colourbar limit for a difference field."""
    finite_values = np.abs(array[np.isfinite(array)])

    if finite_values.size == 0:
        return 1.0

    value = np.nanpercentile(finite_values, 98)

    if value == 0:
        value = 1.0

    return value


# %%
# =============================================================================
# LOAD FINAL BED PROFILES
# =============================================================================

scenario_profiles = {}
scenario_final_times = {}
y_samples_m = None

for scenario_name in COMPARE_LIST:
    model_folder = SCENARIOS[scenario_name]

    print("\n" + "=" * 80)
    print(f"Loading final MAP bathymetry: {scenario_name}")
    print(f"Folder: {model_folder}")
    print("=" * 80)

    ds_map = None

    try:
        ds_map = open_map_dataset(model_folder)

        y_samples_m_this, profiles, final_time = sample_final_profiles(
            ds=ds_map,
            x_coords_m=X_COORDS_M,
            y_range_m=Y_RANGE_M,
            n_y_samples=N_Y_SAMPLES,
        )

        if y_samples_m is None:
            y_samples_m = y_samples_m_this
        elif not np.allclose(y_samples_m, y_samples_m_this):
            raise ValueError("Cross-section sampling coordinates differ between scenarios.")

        scenario_profiles[scenario_name] = profiles
        scenario_final_times[scenario_name] = final_time

        print(f"  Final MAP time: {final_time}")
        print(f"  Extracted {len(profiles)} bed profiles.")

    finally:
        if ds_map is not None:
            ds_map.close()


# %%
# =============================================================================
# CALCULATE WIDTH BELOW BED-ELEVATION THRESHOLD
# =============================================================================

width_data = {}
section_names = None

for scenario_name in COMPARE_LIST:
    width_matrix, section_names_this = calculate_width_matrix(
        profiles=scenario_profiles[scenario_name],
        y_m=y_samples_m,
        thresholds_m=BED_THRESHOLDS_M,
    )

    if section_names is None:
        section_names = section_names_this
    elif section_names != section_names_this:
        raise ValueError("Cross-section order differs between scenarios.")

    width_data[scenario_name] = width_matrix

x_km = X_COORDS_M / 1000.0

reference_scenario = COMPARE_LIST[0]
comparison_scenario = COMPARE_LIST[1]

width_difference_m = (
    width_data[comparison_scenario]
    - width_data[reference_scenario]
)

print("\nWidth calculation completed.")
print(f"Difference convention: {comparison_scenario} - {reference_scenario}")


# %%
# =============================================================================
# PLOT 1: FINAL BED-LEVEL PROFILES
# =============================================================================

bed_profiles_dir = OUTPUT_DIR / "bed_profiles"
bed_profiles_dir.mkdir(parents=True, exist_ok=True)

for km in SELECTED_PROFILE_KM:
    section_name = f"km{km}"

    if section_name not in scenario_profiles[reference_scenario]:
        print(f"[SKIP] {section_name} is not available.")
        continue

    fig, ax = plt.subplots(figsize=(10, 4.8))

    for scenario_name in COMPARE_LIST:
        profile = scenario_profiles[scenario_name][section_name]

        ax.plot(
            y_samples_m / 1000.0,
            profile,
            color=SCENARIO_COLORS.get(scenario_name, "tab:blue"),
            linewidth=2.0,
            label=f"{scenario_name} | final MAP",
        )

    for threshold_m in PLOT_THRESHOLDS_M:
        ax.axhline(
            threshold_m,
            color="0.65",
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

    ax.set_title(f"Final cross-section bed profile — {section_name}")
    ax.set_xlabel("Cross-estuary distance [km]")
    ax.set_ylabel("Bed level [m]")
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()

    outpath = bed_profiles_dir / f"final_bed_profiles_{section_name}.png"
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outpath.name}")


# %%
# =============================================================================
# PLOT 2: WIDTH VERSUS BED-ELEVATION THRESHOLD AT SELECTED CROSS-SECTIONS
# =============================================================================

threshold_curves_dir = OUTPUT_DIR / "threshold_width_curves"
threshold_curves_dir.mkdir(parents=True, exist_ok=True)

for km in SELECTED_PROFILE_KM:
    section_name = f"km{km}"

    if section_name not in section_names:
        print(f"[SKIP] {section_name} is not available.")
        continue

    section_index = section_names.index(section_name)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for scenario_name in COMPARE_LIST:
        width_curve_m = width_data[scenario_name][:, section_index]

        ax.plot(
            BED_THRESHOLDS_M,
            width_curve_m,
            color=SCENARIO_COLORS.get(scenario_name, "tab:blue"),
            linewidth=2.0,
            label=scenario_name,
        )

    ax.set_title(f"Width below bed-elevation threshold — {section_name}")
    ax.set_xlabel("Bed-elevation threshold [m]")
    ax.set_ylabel("Total width below threshold [m]")
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best")

    fig.tight_layout()

    outpath = threshold_curves_dir / f"width_below_threshold_{section_name}.png"
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outpath.name}")


# %%
# =============================================================================
# PLOT 3: ALONG-ESTUARY / THRESHOLD HEATMAP OF DELTA WIDTH
# =============================================================================

fig, ax = plt.subplots(figsize=(11, 6.5))

vmax = get_symmetric_limit(width_difference_m)

mesh = ax.pcolormesh(
    x_km,
    BED_THRESHOLDS_M,
    width_difference_m,
    shading="auto",
    cmap=DIFF_CMAP,
    vmin=-vmax,
    vmax=vmax,
)

contours = ax.contour(
    x_km,
    BED_THRESHOLDS_M,
    width_difference_m,
    levels=[0.0],
    colors="k",
    linewidths=1.0,
)
ax.clabel(contours, fmt={0.0: "ΔW = 0"}, fontsize=8)

colourbar = fig.colorbar(mesh, ax=ax)
colourbar.set_label(
    f"Δ width below threshold [m]\n"
    f"({comparison_scenario} − {reference_scenario})"
)

ax.set_title(
    "Difference in total cross-sectional width below fixed bed-elevation threshold"
)
ax.set_xlabel("Along-estuary distance [km]")
ax.set_ylabel("Bed-elevation threshold [m]")

fig.tight_layout()

outpath = OUTPUT_DIR / (
    f"heatmap_delta_width_{comparison_scenario}_minus_{reference_scenario}.png"
)
fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {outpath.name}")


# %%
# =============================================================================
# PLOT 4: ALONG-ESTUARY WIDTH PROFILES AT SELECTED THRESHOLDS
# =============================================================================

along_estuary_dir = OUTPUT_DIR / "along_estuary_width_profiles"
along_estuary_dir.mkdir(parents=True, exist_ok=True)

for threshold_m in PLOT_THRESHOLDS_M:
    threshold_index = np.argmin(np.abs(BED_THRESHOLDS_M - threshold_m))
    used_threshold_m = BED_THRESHOLDS_M[threshold_index]

    fig, ax = plt.subplots(figsize=(10, 4.8))

    for scenario_name in COMPARE_LIST:
        width_profile_m = width_data[scenario_name][threshold_index, :]

        ax.plot(
            x_km,
            width_profile_m,
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            color=SCENARIO_COLORS.get(scenario_name, "tab:blue"),
            label=scenario_name,
        )

    ax.set_title(
        f"Total width below fixed bed elevation {used_threshold_m:.2f} m"
    )
    ax.set_xlabel("Along-estuary distance [km]")
    ax.set_ylabel("Total width below threshold [m]")
    ax.grid(True, alpha=0.30)
    ax.legend(loc="best")

    fig.tight_layout()

    outpath = along_estuary_dir / (
        f"along_estuary_width_threshold_{threshold_label(used_threshold_m)}m.png"
    )
    fig.savefig(outpath, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outpath.name}")


# %%
# =============================================================================
# OPTIONAL: SAVE NUMERICAL RESULTS AS CSV
# =============================================================================

csv_dir = OUTPUT_DIR / "csv"
csv_dir.mkdir(parents=True, exist_ok=True)

for scenario_name in COMPARE_LIST:
    csv_path = csv_dir / f"width_below_threshold_{scenario_name}.csv"

    header = "bed_threshold_m," + ",".join(section_names)
    table = np.column_stack(
        [
            BED_THRESHOLDS_M,
            width_data[scenario_name],
        ]
    )

    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6f",
    )

    print(f"Saved: {csv_path.name}")

difference_csv_path = csv_dir / (
    f"delta_width_{comparison_scenario}_minus_{reference_scenario}.csv"
)

header = "bed_threshold_m," + ",".join(section_names)
difference_table = np.column_stack(
    [
        BED_THRESHOLDS_M,
        width_difference_m,
    ]
)

np.savetxt(
    difference_csv_path,
    difference_table,
    delimiter=",",
    header=header,
    comments="",
    fmt="%.6f",
)

print(f"Saved: {difference_csv_path.name}")


# %%
print("\n" + "=" * 80)
print("DONE")
print(f"Output folder:\n{OUTPUT_DIR}")
print(f"Final MAP times used:")
for scenario_name in COMPARE_LIST:
    print(f"  {scenario_name}: {scenario_final_times[scenario_name]}")
print("=" * 80)