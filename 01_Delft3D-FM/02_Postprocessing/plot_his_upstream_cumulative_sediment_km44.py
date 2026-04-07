"""
Compute cumulative sediment signal through the upstream cross section (km44)
for each variability scenario.

Outputs:
- One comparison plot with all scenarios overlaid
"""
#%%
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from FUNCTIONS.F_loaddata import load_and_cache_scenario
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
#%%

# =========================
# Configuration
# =========================
SED_VAR = "cross_section_bedload_sediment_transport"
RIVER_KM = 45

SCENARIOS_TO_PROCESS = None  # None = all; e.g. ['1', '2', '3', '4', '5'] for a subset
DISCHARGE = 500
ANALYZE_NOISY = False

SCENARIO_LABELS = {
    "1": "Constant",
    "2": "Seasonal",
    "3": "Flashy",
    "4": "Single peak",
}

# colorblind friendly
SCENARIO_COLORS = {
    '1': '#56B4E9', #'#1f77b4',   # blue   – Constant
    '2': '#E69F00', #'#ff7f0e',   # orange – Seasonal
    '3': '#009E73', # '#2ca02c',   # green  – Flashy
    '4': '#D55E00', #'#d62728',   # red    – Single peak
}

BASE_DIRECTORY = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
CONFIG = f"Model_Output/Q{DISCHARGE}"

if ANALYZE_NOISY:
    BASE_PATH = BASE_DIRECTORY / CONFIG / f"0_Noise_Q{DISCHARGE}"
else:
    BASE_PATH = BASE_DIRECTORY / CONFIG

OUTPUT_DIR = BASE_PATH / "output_plots" / "plots_his_sedimentsupply_km44"
CACHE_DIR = BASE_PATH / "cached_data"
TIMED_OUT_DIR = BASE_PATH / "timed-out"


#%%

if not BASE_PATH.exists():
    raise FileNotFoundError(f"Base path not found: {BASE_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not TIMED_OUT_DIR.exists():
    TIMED_OUT_DIR = None
    print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')

variability_map = get_variability_map(DISCHARGE)

model_folders = find_variability_model_folders(
    base_path=BASE_PATH,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=ANALYZE_NOISY,
)

print(f"Found {len(model_folders)} run folders in {BASE_PATH}")

# Build HIS file path map with the same stitching logic as other scripts.
run_his_paths = {}
for folder in model_folders:
    model_location = BASE_PATH / folder
    his_paths = []
    scenario_num = folder.name.split('_')[0]
    try:
        scenario_key = str(int(scenario_num))
    except Exception:
        scenario_key = scenario_num

    if ANALYZE_NOISY:
        match = re.search(r'noisy(\d+)', folder.name)
        timed_out_folder = None
        if TIMED_OUT_DIR is None:
            print('[WARNING] Timed-out directory not available; skipping timed-out noisy runs.')
        elif match:
            noisy_id = match.group(0)
            for f in TIMED_OUT_DIR.iterdir():
                if f.is_dir() and noisy_id in f.name:
                    timed_out_folder = f.name
                    break
        if timed_out_folder:
            timed_out_path = TIMED_OUT_DIR / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)
    else:
        if TIMED_OUT_DIR is not None:
            timed_out_folder = variability_map.get(scenario_key, folder.name)
            timed_out_path = TIMED_OUT_DIR / timed_out_folder / "output" / "FlowFM_0000_his.nc"
            if timed_out_path.exists():
                his_paths.append(timed_out_path)

    main_his_path = model_location / "output" / "FlowFM_0000_his.nc"
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

if CACHE_DIR.exists():
    if not CACHE_DIR.is_dir():
        raise RuntimeError(f"[ERROR] {CACHE_DIR} exists but is not a directory.")
    try:
        _ = list(CACHE_DIR.iterdir())
    except Exception as e:
        raise RuntimeError(f"[ERROR] {CACHE_DIR} is not accessible: {e}")
else:
    CACHE_DIR.mkdir(exist_ok=True)

comparison_series = []

for folder, his_paths in run_his_paths.items():

    parts = folder.name.split("_")
    scenario_num = str(int(parts[0]))
    run_id = "_".join(parts[1:]) if len(parts) > 1 else folder.name
    cache_file = CACHE_DIR / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

    _, data = load_and_cache_scenario(
        scenario_dir=folder,
        his_file_paths=his_paths,
        cache_file=cache_file,
        boxes=[],
        var_name=SED_VAR,
    )

    km_positions = np.asarray(data["km_positions"])
    idx_upstream = int(np.argmin(np.abs(km_positions - RIVER_KM)))
    km_actual = float(km_positions[idx_upstream])

    time = pd.to_datetime(np.asarray(data["t"]))
    sediment_transport = np.asarray(data[SED_VAR])[:, idx_upstream]

    scenario_label = SCENARIO_LABELS.get(scenario_num, folder.name)

    comparison_series.append(
        {
            "scenario_number": scenario_num,
            "scenario_label": scenario_label,
            "run_folder": folder.name,
            "km_actual": km_actual,
            "time": time,
            "sediment_transport": sediment_transport,
            "final_value": float(sediment_transport[-1]),
        }
    )

    print(
        f"{folder.name}: km target={RIVER_KM}, km actual={km_actual:.2f}, "
        f"final value={sediment_transport[-1]:.3e}"
    )

if comparison_series:
    comparison_series.sort(key=lambda d: (int(d["scenario_number"]), d["run_folder"]))

    fig, ax = plt.subplots(figsize=(11, 5))
    for series in comparison_series:
        color = SCENARIO_COLORS.get(series["scenario_number"], None)
        label = (
            f"{series['scenario_label']}"# ({series['run_folder']}, "
            #f"km={series['km_actual']:.2f})"
        )
        ax.plot(series["time"], series["sediment_transport"], lw=1.8, color=color, label=label)

    ax.set_title(f"km {RIVER_KM} | $Q_{{mean}}$ = {DISCHARGE} m³/s")
    ax.set_xlabel("Time")
    ax.set_ylabel("cumulative sediment transport [kg]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    fig_path = OUTPUT_DIR / f"comparison_upstream_km{RIVER_KM}_bedload_transport_Q{DISCHARGE}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved comparison plot: {fig_path}")
else:
    print("No scenarios processed; comparison plot not written.")
