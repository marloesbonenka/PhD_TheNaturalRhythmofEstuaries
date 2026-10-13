# %% Imports
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")


# %% --- CONFIGURATION ---

SCENARIOS = {
    'pm3_n5_mean_flow': Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_mean.964903"),
    'pm5_n1_mean_flow': Path(r"u:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_05_Qr500_pm5_n1_mean.964908"),
    'pm3_n5_peak_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_03_Qr500_pm3_n5_peak.964907"),
    'pm5_n1_peak_flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_05_Qr500_pm5_n1_peak.965221"),
}

# Define the pairs to compare in separate plots
COMPARISON_PAIRS = {
    "mean_flow": ["pm3_n5_mean_flow", "pm5_n1_mean_flow"],
    "peak_flow": ["pm3_n5_peak_flow", "pm5_n1_peak_flow"],
}

# --- Color Maps ---
all_pm_vals = sorted([0, 2, 3, 4, 5])  # pm values to compare
_n_pm = max(len(all_pm_vals) - 1, 1)
PM_COLOR = {pm: plt.cm.Blues(0.35 + 0.55 * i / _n_pm) for i, pm in enumerate(all_pm_vals)}

all_n_vals = sorted([1, 3, 5])  # n values to compare
_n_n = max(len(all_n_vals) - 1, 1)
N_COLOR = {n: plt.cm.Greens(0.35 + 0.55 * i / _n_n) for i, n in enumerate(all_n_vals)}


def get_scenario_color(label, color_by="pm"):
    """Extract pm or n from scenario label and return the corresponding color."""
    pm_match = re.search(r'pm(\d+)', label)
    n_match = re.search(r'n(\d+)', label)
    
    if color_by == "pm" and pm_match:
        pm_val = int(pm_match.group(1))
        if pm_val in PM_COLOR:
            return PM_COLOR[pm_val]
    elif color_by == "n" and n_match:
        n_val = int(n_match.group(1))
        if n_val in N_COLOR:
            return N_COLOR[n_val]
            
    return "tab:blue"


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

FRAME_DPI = 150
STYLE = "default"
STYLES = {
    "default": {},
    "whitefig": {
        "figure.facecolor": "none",
        "axes.facecolor":    "none",
        "axes.edgecolor":    "white",
        "axes.labelcolor":   "white",
        "xtick.color":       "white",
        "ytick.color":       "white",
        "text.color":        "white",
        "savefig.transparent": True,
    },
}
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update(STYLES[STYLE])
_tc = plt.rcParams["text.color"]


# %% --- FUNCTIONS ---

def decode_his_names(name_array):
    """Decode D-Flow FM his-file character arrays into a list of clean strings."""
    names = np.asarray(name_array)
    decoded = []
    for item in names:
        if isinstance(item, bytes):
            s = item.decode("utf-8", errors="ignore")
        elif np.ndim(item) == 0:
            s = str(item)
        else:
            chars = [c.decode("utf-8", errors="ignore") if isinstance(c, bytes) else str(c) for c in item]
            s = "".join(chars)
        s = s.replace("\x00", "").strip()
        decoded.append(s)
    return decoded


def get_cross_section_index(ds, section_name):
    """Return the index of the requested observation cross-section."""
    if HIS_NAME_VAR not in ds:
        raise KeyError(f"'{HIS_NAME_VAR}' not found in his-file.")
    names = decode_his_names(ds[HIS_NAME_VAR].values)
    if section_name in names:
        return names.index(section_name)
    matches = [i for i, name in enumerate(names) if section_name.lower() in name.lower()]
    if len(matches) == 0:
        raise ValueError(f"Cross-section '{section_name}' not found.")
    return matches[0]


def get_his_series(ds_his, variable_name, section_index):
    """Extract a 1-D time series of one his-file variable at one cross-section."""
    da = ds_his[variable_name]
    if HIS_NAME_DIM in da.dims:
        da = da.isel({HIS_NAME_DIM: section_index})
    else:
        raise ValueError(f"Variable '{variable_name}' does not have dimension '{HIS_NAME_DIM}'.")

    for dim in list(da.dims):
        if dim != "time":
            da = da.isel({dim: 0})

    return np.asarray(da.values)


# %% --- PROCESSING & PLOTTING ---

if HIS_VARIABLE == "cross_section_discharge":
    YLABEL = "discharge [m$^3$/s]"
    TITLE = "discharge"
elif HIS_VARIABLE == "cross_section_area":
    YLABEL = "wet area of observation cross-section [m$^2$]"
    TITLE = "wetted area"
else:
    YLABEL = "variable [units]"
    TITLE = "Variable"

# Pre-load all required datasets once to save time
scenario_data = {}
for label, folder_path in SCENARIOS.items():
    his_path = folder_path / HIS_FILENAME
    if not his_path.exists():
        print(f"[SKIP] His-file not found for {label}: {his_path}")
        continue
    scenario_data[label] = xr.open_dataset(his_path)

# Iterate through each comparison pair with distinct directories and filenames
for group_name, compare_list in COMPARISON_PAIRS.items():
    print(f"\n{'='*60}")
    print(f"Processing Comparison Group: {group_name} ({' vs '.join(compare_list)})")
    print(f"{'='*60}\n")

    # Distinct output directory per comparison pair
    output_dir = SCENARIOS[compare_list[0]].parent / "output_plots" / HIS_VARIABLE / f"comparison_{group_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    along_estuary_data = {label: {"km": [], "vals": []} for label in compare_list}

    for section_label, section_name in CROSS_SECTIONS.items():
        print(f"  Processing Cross-section: {section_label} ({section_name})")
        
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plotted_any = False

        km_num_match = re.search(r'km(\d+)', section_label)
        km_val = int(km_num_match.group(1)) if km_num_match else None

        for label in compare_list:
            if label not in scenario_data:
                continue
            ds_his = scenario_data[label]
            
            try:
                section_idx = get_cross_section_index(ds_his, section_name)
                time_values = np.asarray(ds_his.time.values).astype("datetime64[ns]")
                values = get_his_series(ds_his, HIS_VARIABLE, section_idx)
                
                # --- Plot Time Series ---
                color = get_scenario_color(label, color_by="pm")
                ax.plot(
                    time_values,
                    values,
                    color=color,
                    alpha=0.85,
                    linewidth=1.5,
                    label=label,
                )
                plotted_any = True

                # --- Collect Average for Profile ---
                if len(time_values) > 0 and km_val is not None:
                    last_time = time_values[-1]
                    start_last_day = last_time - np.timedelta64(12, 'h')
                    mask = time_values >= start_last_day
                    if np.any(mask):
                        last_day_avg = np.nanmean(values[mask])
                    else:
                        last_day_avg = values[-1]
                    
                    along_estuary_data[label]["km"].append(km_val)
                    along_estuary_data[label]["vals"].append(last_day_avg)

            except (KeyError, ValueError) as e:
                print(f"    [SKIP] {label} - {e}")

        if not plotted_any:
            print(f"    [SKIP] No data available for {section_label} in group {group_name}.")
            plt.close(fig)
            continue

        ax.set_xlabel("Time", color=_tc)
        ax.set_ylabel(YLABEL, color=_tc)
        ax.set_title(f"{TITLE} ({group_name}) | {section_label}", color=_tc)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize="small")
        plt.tight_layout()

        # Filename includes group_name to prevent cross-group overwriting
        out_path = output_dir / f"static_compare_{group_name}_{section_label}.png"
        plt.savefig(out_path, dpi=FRAME_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved time series: {out_path.name}")

    # --- Generate Along-Estuary Profile Plot for this Group ---
    print(f"\n  Generating Along-Estuary Profile Plot for {group_name}...")
    fig_prof, ax_prof = plt.subplots(figsize=(9, 4.5))
    profile_plotted = False

    for label in compare_list:
        km_list = along_estuary_data[label]["km"]
        val_list = along_estuary_data[label]["vals"]
        
        if not km_list:
            continue
        
        sorted_indices = np.argsort(km_list)
        km_arr = np.array(km_list)[sorted_indices]
        val_arr = np.array(val_list)[sorted_indices]
        
        color = get_scenario_color(label, color_by="pm")
        ax_prof.plot(
            km_arr,
            val_arr,
            marker='o',
            color=color,
            linewidth=1.5,
            markersize=5,
            label=label,
        )
        profile_plotted = True

    if profile_plotted:
        ax_prof.set_xlabel("along-estuary distance [km]", color=_tc)
        ax_prof.set_ylabel(f"{YLABEL}", color=_tc)
        ax_prof.set_title(f"{group_name}: tidal averaged {TITLE}", color=_tc)
        ax_prof.grid(True, alpha=0.3)
        ax_prof.legend(loc="upper right", fontsize="small")
        plt.tight_layout()

        # Filename includes group_name to prevent profile overwriting
        profile_path = output_dir / f"along_estuary_tidal_averaged_{group_name}.png"
        plt.savefig(profile_path, dpi=FRAME_DPI, bbox_inches="tight")
        plt.close(fig_prof)
        print(f"    Saved profile plot: {profile_path.name}")
    else:
        plt.close(fig_prof)
        print("    [SKIP] Profile plot not generated (no valid data found).")

# Close all opened datasets
for ds in scenario_data.values():
    ds.close()

print("\n" + "=" * 60)
print("DONE - All comparison pairs processed successfully without overwriting.")
print("=" * 60)