# %% Compare initial bed profiles and final-2-hour water-level envelopes
"""
Compare detailed-hydrodynamic-run cross-sections for mean-flow and peak-flow
simulations.

For every cross-section:
- Plot the bed level at the first available output timestep.
- Plot the maximum water level during the final two hours.
- Plot the minimum water level during the final two hours.

The script searches each detailed-hydro-run directory for a suitable D-Flow FM
map-output NetCDF file. It does not require the cached morphological MAP files.
"""

# %%
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

# %%
# =============================================================================
# PROJECT IMPORTS
# =============================================================================

# If this script is in postprocessing/, keep this import block.
# Otherwise, remove or adjust it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions.F_general import apply_plot_style

# %%
# =============================================================================
# INPUT: DETAILED-HYDRODYNAMIC RUN DIRECTORIES
# =============================================================================

SCENARIOS = {
    "pm3_n5_mean_flow": Path(
        r"u:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500"
        r"\detailed-hydro-run\dhr_03_Qr500_pm3_n5_mean.964903"
    ),
    "pm5_n1_mean_flow": Path(
        r"u:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500"
        r"\detailed-hydro-run\dhr_05_Qr500_pm5_n1_mean.964908"
    ),
    "pm3_n5_peak_flow": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500"
        r"\detailed-hydro-run\dhr_03_Qr500_pm3_n5_peak.964907"
    ),
    "pm5_n1_peak_flow": Path(
        r"U:\PhDNaturalRhythmEstuaries\Models"
        r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500"
        r"\detailed-hydro-run\dhr_05_Qr500_pm5_n1_peak.965221"
    ),
}

COMPARISON_PAIRS = {
    "mean_flow": ["pm3_n5_mean_flow", "pm5_n1_mean_flow"],
    "peak_flow": ["pm3_n5_peak_flow", "pm5_n1_peak_flow"],
}

# %%
# =============================================================================
# CROSS-SECTION SETTINGS
# =============================================================================

CROSS_SECTION_X_KM = np.arange(25.0, 46.0, 1.0)
X_COORDS_M = CROSS_SECTION_X_KM * 1_000.0

# Same central strip as your morphological cross-section script.
Y_RANGE_M = (5_000.0, 10_000.0)
N_Y_SAMPLES = 500

# Optional mask for dry/high bed points in profile plots.
# Set to None to retain all sampled values.
PROFILE_LAND_THRESHOLD_M = 8.0

# %%
# =============================================================================
# OUTPUT SETTINGS
# =============================================================================

OUTPUT_DIR = Path(
    r"u:\PhDNaturalRhythmEstuaries\Models"
    r"\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500"
    r"\detailed-hydro-run\output_plots"
    r"\cross_sections_initial_bed_water_level_envelope"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

apply_plot_style(
    "AGU",
    font_size=10,
    title_delta=0,
    tick_delta=0,
    **{"grid.linewidth": 0.5},
)

# %%
# =============================================================================
# SCENARIO COLOURS
# =============================================================================

all_pm_vals = sorted([0, 2, 3, 4, 5])
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {
    pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm)
    for i, pm in enumerate(all_pm_vals)
}

all_n_vals = sorted([1, 3, 5])
_n_n = max(len(all_n_vals) - 1, 1)
N_COLOR = {
    n: plt.cm.Greens(0.35 + 0.55 * i / _n_n)
    for i, n in enumerate(all_n_vals)
}


def get_scenario_color(label, color_by="pm"):
    """Return a scenario colour based on its pm or n value."""
    pm_match = re.search(r"pm(\d+)", label)
    n_match = re.search(r"n(\d+)", label)

    if color_by == "pm" and pm_match:
        pm_val = int(pm_match.group(1))
        if pm_val in PM_COLOR:
            return PM_COLOR[pm_val]

    elif color_by == "n" and n_match:
        n_val = int(n_match.group(1))
        if n_val in N_COLOR:
            return N_COLOR[n_val]

    return "tab:blue"


def scenario_pm_value(label):
    match = re.search(r"pm(\d+)", label)
    return int(match.group(1)) if match else None


def scenario_n_value(label):
    match = re.search(r"n(\d+)", label)
    return int(match.group(1)) if match else None


def scenario_legend_label(label):
    pm_value = scenario_pm_value(label)
    n_value = scenario_n_value(label)
    return (
        rf"$R_{{peak}} = {pm_value}$, "
        rf"$n_{{peaks}} = {n_value}$"
    )

# %%
# =============================================================================
# FUNCTIONS: FIND AND OPEN MAP OUTPUT
# =============================================================================

BEDLEVEL_CANDIDATES = [
    "mesh2d_flowelem_bl",
    "FlowElem_bl",
    "mesh2d_mor_bl",
]

WATERLEVEL_CANDIDATES = [
    "mesh2d_s1",
    "s1",
]

FACE_X_CANDIDATES = [
    "mesh2d_face_x",
    "FlowElem_xcc",
]

FACE_Y_CANDIDATES = [
    "mesh2d_face_y",
    "FlowElem_ycc",
]


def find_first_variable(ds, candidates, description):
    """Return the first available variable name from a candidate list."""
    available = list(ds.variables)

    for name in candidates:
        if name in ds:
            return name

    raise KeyError(
        f"Could not find {description}. "
        f"Tried: {candidates}\n"
        f"Available variables:\n{available}"
    )


def find_map_file(run_dir):
    """
    Find the most plausible D-Flow FM map-output NetCDF in a run directory.

    The function prefers files whose name contains 'map', but falls back to
    any NetCDF file containing both a water-level and bed-level variable.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist:\n{run_dir}")

    nc_files = sorted(run_dir.rglob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found under:\n{run_dir}")

    map_like_files = [
        path for path in nc_files
        if "map" in path.name.lower()
    ]

    search_order = map_like_files + [
        path for path in nc_files if path not in map_like_files
    ]

    for nc_file in search_order:
        try:
            with xr.open_dataset(nc_file) as ds:
                bed_var = find_first_variable(ds, BEDLEVEL_CANDIDATES, "bed level")
                water_var = find_first_variable(
                    ds, WATERLEVEL_CANDIDATES, "water level"
                )

            return nc_file, bed_var, water_var

        except (KeyError, OSError, ValueError):
            continue

    raise FileNotFoundError(
        "No NetCDF file containing both bed level and water level was found "
        f"under:\n{run_dir}"
    )


def get_face_dimension(ds, variable_name):
    """Return the unique non-time dimension of a face-based variable."""
    dims = ds[variable_name].dims

    if "time" in dims:
        non_time_dims = [dim for dim in dims if dim != "time"]
    else:
        non_time_dims = list(dims)

    if len(non_time_dims) != 1:
        raise ValueError(
            f"Expected one spatial dimension for '{variable_name}', "
            f"but found dimensions {dims}."
        )

    return non_time_dims[0]

# %%
# =============================================================================
# FUNCTIONS: SAMPLE CROSS-SECTIONS
# =============================================================================

def sample_profiles_at_x(
    ds,
    variable_name,
    time_indices=None,
    reduction=None,
):
    """
    Sample a face-based variable on regular y-lines at each target x.

    Parameters
    ----------
    ds : xr.Dataset
        D-Flow FM map-output dataset.
    variable_name : str
        Variable to sample, e.g. bed level or water level.
    time_indices : array-like or None
        Time indices to use. If None, use all times.
    reduction : str or None
        One of 'first', 'max', 'min', or None.
        - 'first': first selected time.
        - 'max': maximum over selected times.
        - 'min': minimum over selected times.
        - None: return every selected time separately.

    Returns
    -------
    y_samples_m : ndarray
    profiles : dict[str, ndarray]
        Cross-section profiles, keyed by section name.
    """
    face_x_var = find_first_variable(ds, FACE_X_CANDIDATES, "face x coordinate")
    face_y_var = find_first_variable(ds, FACE_Y_CANDIDATES, "face y coordinate")

    face_dimension = get_face_dimension(ds, variable_name)

    face_x = np.asarray(ds[face_x_var].values, dtype=float)
    face_y = np.asarray(ds[face_y_var].values, dtype=float)

    da = ds[variable_name]

    if "time" in da.dims:
        if time_indices is None:
            da_selected = da
        else:
            da_selected = da.isel(time=time_indices)
    else:
        da_selected = da

    if reduction == "first":
        if "time" in da_selected.dims:
            da_selected = da_selected.isel(time=0)
    elif reduction == "max":
        if "time" in da_selected.dims:
            da_selected = da_selected.max(dim="time", skipna=True)
    elif reduction == "min":
        if "time" in da_selected.dims:
            da_selected = da_selected.min(dim="time", skipna=True)
    elif reduction is not None:
        raise ValueError(
            "reduction must be one of 'first', 'max', 'min', or None."
        )

    values = np.asarray(da_selected.transpose(face_dimension).values, dtype=float)

    if values.ndim == 0:
        values = values.reshape(1)

    y_samples_m = np.linspace(
        Y_RANGE_M[0],
        Y_RANGE_M[1],
        N_Y_SAMPLES,
    )

    face_tree = cKDTree(np.column_stack((face_x, face_y)))

    profiles = {}

    for x_coord_m in X_COORDS_M:
        section_name = f"km{int(round(x_coord_m / 1000.0))}"

        sample_points = np.column_stack(
            (
                np.full(y_samples_m.size, x_coord_m, dtype=float),
                y_samples_m,
            )
        )

        _, face_indices = face_tree.query(sample_points)

        if values.ndim == 1:
            profile = values[face_indices].astype(float)
        else:
            # This branch is not used after a reduction is applied.
            profile = values[:, face_indices].astype(float)

        if PROFILE_LAND_THRESHOLD_M is not None:
            land_mask = profile > PROFILE_LAND_THRESHOLD_M
            profile[land_mask] = np.nan

        profiles[section_name] = profile

    return y_samples_m, profiles


def get_final_two_hour_indices(ds, waterlevel_var):
    """
    Return indices of map-output times in the final two hours of the run.
    """
    if "time" not in ds[waterlevel_var].dims:
        raise ValueError(
            f"'{waterlevel_var}' has no time dimension; "
            f"dimensions are {ds[waterlevel_var].dims}."
        )

    times = np.asarray(ds["time"].values).astype("datetime64[ns]")

    if times.size == 0:
        raise ValueError("The map output contains no time steps.")

    final_time = times[-1]
    cutoff_time = final_time - np.timedelta64(2, "h")

    final_indices = np.where(times >= cutoff_time)[0]

    if final_indices.size == 0:
        raise ValueError(
            "No output times were found within the final two hours.\n"
            f"Final output time: {final_time}\n"
            f"Cutoff time: {cutoff_time}"
        )

    return final_indices, times[final_indices], final_time

# %%
# =============================================================================
# FUNCTIONS: PLOTTING
# =============================================================================

def plot_cross_section_comparison(
    flow_condition,
    scenario_names,
    section_name,
    y_samples_m,
    initial_bed_profiles,
    maximum_water_levels,
    minimum_water_levels,
):
    """Plot initial bed, maximum water level, and minimum water level."""
    km = int(section_name.replace("km", ""))

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    for scenario_name in scenario_names:
        color = get_scenario_color(scenario_name, color_by="pm")
        label = scenario_legend_label(scenario_name)

        ax.plot(
            y_samples_m / 1000.0,
            initial_bed_profiles[scenario_name][section_name],
            color=color,
            linewidth=1.4,
            linestyle="-",
            label=f"{label} — initial bed",
        )

        ax.plot(
            y_samples_m / 1000.0,
            maximum_water_levels[scenario_name][section_name],
            color=color,
            linewidth=1.1,
            linestyle="--",
            label=f"{label} — max water level",
        )

        ax.plot(
            y_samples_m / 1000.0,
            minimum_water_levels[scenario_name][section_name],
            color=color,
            linewidth=1.1,
            linestyle=":",
            label=f"{label} — min water level",
        )

    ax.set_title(
        rf"$Q_{{mean}} = 500$ m$^3$ s$^{{-1}}$, "
        rf"{flow_condition.replace('_', ' ')}, "
        rf"$x = {km}$ km"
    )
    ax.set_xlabel("Cross-estuary distance [km]")
    ax.set_ylabel("Elevation [m]")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR
        / flow_condition
        / f"cross_section_{section_name}_{flow_condition}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")

# %%
# =============================================================================
# PROCESS EACH SCENARIO
# =============================================================================

initial_bed_profiles = {}
maximum_water_levels = {}
minimum_water_levels = {}
y_samples_reference_m = None
map_file_info = {}
selected_time_info = {}


for scenario_name, run_dir in SCENARIOS.items():
    print("\n" + "=" * 80)
    print(f"Processing {scenario_name}")
    print(f"Run directory: {run_dir}")
    print("=" * 80)

    map_file, bedlevel_var, waterlevel_var = find_map_file(run_dir)
    print(f"Selected map file: {map_file}")
    print(f"Bed-level variable: {bedlevel_var}")
    print(f"Water-level variable: {waterlevel_var}")

    ds = xr.open_dataset(map_file)

    try:
        final_indices, final_times, final_time = get_final_two_hour_indices(
            ds=ds,
            waterlevel_var=waterlevel_var,
        )

        print(
            f"Final output time: {final_time}\n"
            f"Final-two-hour output times: {final_times[0]} to {final_times[-1]}\n"
            f"Number of final-two-hour map times: {final_indices.size}"
        )

        # a) Bed level at the starting output timestep.
        y_samples_m, scenario_initial_bed = sample_profiles_at_x(
            ds=ds,
            variable_name=bedlevel_var,
            time_indices=None,
            reduction="first",
        )

        # b) Maximum water level during the final two hours.
        _, scenario_max_water_level = sample_profiles_at_x(
            ds=ds,
            variable_name=waterlevel_var,
            time_indices=final_indices,
            reduction="max",
        )

        # c) Minimum water level during the final two hours.
        _, scenario_min_water_level = sample_profiles_at_x(
            ds=ds,
            variable_name=waterlevel_var,
            time_indices=final_indices,
            reduction="min",
        )

        if y_samples_reference_m is None:
            y_samples_reference_m = y_samples_m
        elif not np.allclose(y_samples_reference_m, y_samples_m):
            raise ValueError(
                "Regular y-sampling positions differ between scenarios."
            )

        initial_bed_profiles[scenario_name] = scenario_initial_bed
        maximum_water_levels[scenario_name] = scenario_max_water_level
        minimum_water_levels[scenario_name] = scenario_min_water_level

        map_file_info[scenario_name] = map_file
        selected_time_info[scenario_name] = {
            "final_time": final_time,
            "first_final_two_hour_time": final_times[0],
            "last_final_two_hour_time": final_times[-1],
            "n_final_two_hour_times": int(final_indices.size),
        }

    finally:
        ds.close()

# %%
# =============================================================================
# PLOT COMPARISON PAIRS
# =============================================================================

section_names = [f"km{int(round(x_km))}" for x_km in CROSS_SECTION_X_KM]

for flow_condition, scenario_names in COMPARISON_PAIRS.items():
    print(f"\nPlotting {flow_condition} comparison:")
    print(f"  {scenario_names[0]} vs {scenario_names[1]}")

    for section_name in section_names:
        plot_cross_section_comparison(
            flow_condition=flow_condition,
            scenario_names=scenario_names,
            section_name=section_name,
            y_samples_m=y_samples_reference_m,
            initial_bed_profiles=initial_bed_profiles,
            maximum_water_levels=maximum_water_levels,
            minimum_water_levels=minimum_water_levels,
        )

# %%
# =============================================================================
# FINISH
# =============================================================================

print("\n" + "=" * 80)
print("DETAILED-HYDRO CROSS-SECTION ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory:\n{OUTPUT_DIR}")

print("\nSelected map files and final-two-hour windows:")
for scenario_name in SCENARIOS:
    info = selected_time_info[scenario_name]
    print(f"\n{scenario_name}")
    print(f"  Map file: {map_file_info[scenario_name]}")
    print(f"  Final time: {info['final_time']}")
    print(
        f"  Final-two-hour window: "
        f"{info['first_final_two_hour_time']} to "
        f"{info['last_final_two_hour_time']}"
    )
    print(
        f"  Number of map times used: "
        f"{info['n_final_two_hour_times']}"
    )

print("\nPlot conventions:")
print("  Solid line: bed level at the first available output timestep.")
print(
    "  Dashed line: maximum water level during the final two hours "
    "of the detailed-hydro run."
)
print(
    "  Dotted line: minimum water level during the final two hours "
    "of the detailed-hydro run."
)
print("=" * 80)