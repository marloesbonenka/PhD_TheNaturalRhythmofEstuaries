"""
Plot a time series at observation cross-sections from the D-Flow FM his-file
and create GIFs showing a moving dot along that series.

For each scenario:
    - cross-sections: km20 to km44 (ObservationCrossSection_Estuary_km{km})

Each GIF shows:
    - x-axis: time
    - y-axis: HIS_VARIABLE at the cross-section (e.g. wetted area or discharge)
    - grey line: the time series
    - red dot: value at the current frame

Time axis is controlled by COMPARE_TO_MAP_TIMES:
    - False (default): one frame per his-file timestep (optionally every
      FRAME_STRIDE-th). The map output / map cache is not touched.
    - True: the his-file is sampled at the map-output timesteps, so the number
      of frames, frame order and GIF speed match the map GIF exactly.
"""


# %% Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image


sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_loaddata import get_stitched_map_run_paths
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi


# %% --- CONFIGURATION ---


SCENARIOS = {
    # 'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    # 'low_flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    # 'peak_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
    #'mean_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
    'pm3_n5_mean_flow': Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_mean.964903"),
    'pm3_n5_peak_flow': Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_peak.964907"),
    'pm5_n1_mean_flow': Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_05_Qr500_pm5_n1_mean.964908"),

}


# --- Time axis toggle ---
# False -> use the his-file timestamps as they are (fast, no map cache needed)
# True  -> sample the his-file at the map-output timesteps (frames match map GIF)
COMPARE_TO_MAP_TIMES = False

# Only used when COMPARE_TO_MAP_TIMES = False: make a frame for every n-th
# his timestep (1 = every timestep). The grey line always uses all timesteps.
FRAME_STRIDE = 1


# His-file settings
HIS_FILENAME = "output/FlowFM_0000_his.nc"
HIS_VARIABLE = "cross_section_area"
HIS_NAME_DIM = "cross_section"
HIS_NAME_VAR = "cross_section_name"

# Automatically generate cross-sections from km20 to km44
CROSS_SECTIONS = {
    f"km{km}": f"ObservationCrossSection_Estuary_km{km}"
    for km in range(20, 45)
}

# Map-cache settings: only used when COMPARE_TO_MAP_TIMES = True
# (keep identical to the map-GIF script)
MAP_VARIABLES = ["mesh2d_s1"]
CACHE_BBOX       = [1, 1, 45000, 15000]
CACHE_TAG        = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# GIF settings — identical to the map GIF
GIF_FPS   = 2
FRAME_DPI = 150

# Plot styling
LINE_COLOR = "grey"
LINE_ALPHA = 0.7
DOT_COLOR  = "red"
DOT_SIZE   = 45

STYLE = "default"
STYLES = {
    "default": {},
    "whitefig": {
        "figure.facecolor": "none",
        "axes.facecolor":   "none",
        "axes.edgecolor":   "white",
        "axes.labelcolor":  "white",
        "xtick.color":      "white",
        "ytick.color":      "white",
        "text.color":       "white",
        "savefig.transparent": True,
    },
}
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])
_tc = plt.rcParams["text.color"]


# %% --- FUNCTIONS ---


def decode_his_names(name_array):
    """
    Decode D-Flow FM his-file character arrays such as cross_section_name
    into a list of clean strings.
    """
    if name_array.dtype.kind in ("U", "S"):
        names = np.asarray(name_array)
    else:
        names = np.asarray(name_array)

    decoded = []
    for item in names:
        if isinstance(item, bytes):
            s = item.decode("utf-8", errors="ignore")
        elif np.ndim(item) == 0:
            s = str(item)
        else:
            chars = [c.decode("utf-8", errors="ignore") if isinstance(c, bytes) else str(c)
                     for c in item]
            s = "".join(chars)

        s = s.replace("\x00", "").strip()
        decoded.append(s)

    return decoded


def get_cross_section_index(ds, section_name):
    """
    Return the index of the requested observation cross-section.
    """
    if HIS_NAME_VAR not in ds:
        raise KeyError(
            f"'{HIS_NAME_VAR}' not found in his-file. "
            f"Available variables: {list(ds.variables)}"
        )

    names = decode_his_names(ds[HIS_NAME_VAR].values)

    if section_name in names:
        return names.index(section_name)

    matches = [i for i, name in enumerate(names)
               if section_name.lower() in name.lower()]

    if len(matches) == 0:
        raise ValueError(
            f"Cross-section '{section_name}' not found.\n"
            f"Available cross-sections:\n" + "\n".join(names)
        )
    if len(matches) > 1:
        matched_names = [names[i] for i in matches]
        raise ValueError(
            f"Multiple possible matches for '{section_name}': {matched_names}"
        )

    return matches[0]


def get_map_times(folder_path):
    """
    Return the map-output times (datetime64[ns]) of one scenario, read from the
    map cache. Returns None if there is no map data / time dimension.
    Only called when COMPARE_TO_MAP_TIMES = True.
    """
    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / "cached_data"
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
    ds_map = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_name,
        run_paths=run_paths,
        var_names=MAP_VARIABLES,
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag,
    )

    if ds_map is None:
        print(f"  [SKIP] No map data cached for {folder_name}")
        return None

    try:
        if "time" not in ds_map.dims or len(ds_map.time) == 0:
            print("  [SKIP] No time dimension in map dataset.")
            return None
        return np.asarray(ds_map.time.values).astype("datetime64[ns]")
    finally:
        ds_map.close()


def get_his_series(ds_his, variable_name, section_index, sample_times=None):
    """
    Extract a 1-D time series of one his-file variable at one cross-section.

    sample_times=None  -> all his-file timesteps
    sample_times=array -> his-file sliced at those times (nearest-neighbour),
                          e.g. the map-output times
    """
    if variable_name not in ds_his:
        raise KeyError(
            f"Variable '{variable_name}' not found in his-file.\n"
            f"Available variables: {list(ds_his.variables)}"
        )

    da = ds_his[variable_name]

    if HIS_NAME_DIM in da.dims:
        da = da.isel({HIS_NAME_DIM: section_index})
    else:
        raise ValueError(
            f"Variable '{variable_name}' does not have dimension "
            f"'{HIS_NAME_DIM}'. Dimensions found: {da.dims}"
        )

    # Remove any remaining non-time dimensions, e.g., layers
    for dim in list(da.dims):
        if dim != "time":
            da = da.isel({dim: 0})

    if sample_times is not None:
        da = da.sel(time=sample_times, method="nearest")

    return np.asarray(da.values)


def make_discharge_gif(
    time_values,
    discharge_values,
    label,
    section_label,
    folder_name,
    output_dir,
    frame_indices=None,
):
    """
    Create one GIF for one cross-section.

    frame_indices: indices into time_values that get a frame (red dot).
                   None -> one frame per entry in time_values.
    """
    frame_paths = []

    if frame_indices is None:
        frame_indices = range(len(time_values))

    q_min = np.nanmin(discharge_values)
    q_max = np.nanmax(discharge_values)

    if np.isclose(q_min, q_max):
        padding = max(abs(q_max) * 0.1, 1.0)
    else:
        padding = 0.05 * (q_max - q_min)

    y_min = q_min - padding
    y_max = q_max + padding

    for idx in frame_indices:
        t = time_values[idx]
        dt_str = str(np.datetime_as_string(np.datetime64(t, "ns"), unit="s")).replace("T", " ")
        dt_tag = str(np.datetime_as_string(np.datetime64(t, "ns"), unit="D"))

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(
            time_values,
            discharge_values,
            color=LINE_COLOR,
            alpha=LINE_ALPHA,
            linewidth=1.2,
            zorder=1,
        )

        ax.scatter(
            time_values[idx],
            discharge_values[idx],
            color=DOT_COLOR,
            s=DOT_SIZE,
            zorder=3,
        )

        ax.set_xlim(time_values[0], time_values[-1])
        ax.set_ylim(y_min, y_max)

        ax.set_xlabel("time", color=_tc)
        ax.set_ylabel(YLABEL, color=_tc)
        ax.set_title(
            f"{TITLE} | {section_label} | {label} | {dt_str}",
            color=_tc,
        )

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        frame_name = f"frame_{idx:04d}_{dt_tag}_{folder_name}_{section_label}.png"
        frame_path = output_dir / frame_name
        plt.savefig(frame_path, dpi=FRAME_DPI, bbox_inches="tight")
        plt.close(fig)

        frame_paths.append(frame_path)

    # Suffix keeps the his-time GIFs from overwriting the map-time GIFs
    time_tag = "" if COMPARE_TO_MAP_TIMES else "_his"
    gif_path = output_dir / f"discharge_{section_label}_{label}_{folder_name}{time_tag}.gif"

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

    print(f"  GIF saved: {gif_path}")

    return gif_path


# %% --- PROCESSING ---
if HIS_VARIABLE == "cross_section_discharge":
    YLABEL = "discharge [m$^3$/s]"  
    TITLE = "Discharge"
elif HIS_VARIABLE == "cross_section_area":
    YLABEL = "wet area of observation cross-section [m$^2$]"
    TITLE = "Wetted area"
else:
    print(f"Warning: Unrecognized HIS_VARIABLE '{HIS_VARIABLE}'. Using generic y-label.")
    YLABEL = "variable [units]"
    TITLE = "Variable"

print(f"Time axis: {'map-output times' if COMPARE_TO_MAP_TIMES else 'his-file times'}")

for label, folder_path in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"Scenario: {label}  ({folder_path.name})")
    print(f"{'='*60}")

    folder_name = folder_path.name

    output_dir = folder_path.parent / "output_plots" / HIS_VARIABLE / label
    output_dir.mkdir(parents=True, exist_ok=True)

    his_path = folder_path / HIS_FILENAME

    if not his_path.exists():
        print(f"  [SKIP] His-file not found: {his_path}")
        continue

    ds_his = xr.open_dataset(his_path)

    try:
        if "time" not in ds_his.dims or len(ds_his.time) == 0:
            print("  [SKIP] No time dimension in his-file.")
            continue

        his_time = np.asarray(ds_his.time.values).astype("datetime64[ns]")
        print(f"  His timesteps: {len(his_time)}  ({his_time[0]} → {his_time[-1]})")

        if COMPARE_TO_MAP_TIMES:
            map_time = get_map_times(folder_path)
            if map_time is None:
                continue
            print(f"  Map timesteps: {len(map_time)}  ({map_time[0]} → {map_time[-1]})")
            sample_times  = map_time
            time_values   = map_time
            frame_indices = None                                  # one frame per map step
        else:
            sample_times  = None
            time_values   = his_time
            frame_indices = range(0, len(his_time), FRAME_STRIDE)  # his steps

        for section_label, section_name in CROSS_SECTIONS.items():
            print(f"\n  Cross-section: {section_label}")
            print(f"  Requested name: {section_name}")

            try:
                section_idx = get_cross_section_index(ds_his, section_name)
            except (KeyError, ValueError) as e:
                print(f"  [SKIP] {e}")
                continue

            print(f"  Matched his-file index: {section_idx}")

            values = get_his_series(
                ds_his=ds_his,
                variable_name=HIS_VARIABLE,
                section_index=section_idx,
                sample_times=sample_times,
            )

            gif_path = make_discharge_gif(
                time_values=time_values,
                discharge_values=values,
                label=label,
                section_label=section_label,
                folder_name=folder_name,
                output_dir=output_dir,
                frame_indices=frame_indices,
            )

    finally:
        ds_his.close()


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

