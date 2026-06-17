#%% IMPORTS
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import (
    get_variability_map,
    get_snapshot_matches_by_target_dates,
)

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, _get_face_coords
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# --- CONFIGURATION ---
DISCHARGE = 500
TARGET_DATE = np.datetime64('2031-01-01')
DEPTH_PERCENTILE = 95
CACHE_BBOX = [1, 1, 45000, 15000]
CACHE_TAG = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True
X_RANGE = (20000, 44000)
Y_RANGE = (5000, 10000)
CHANNEL_INIT_THRESHOLD = 2.2

# Paths defined by you
MODEL_PATHS = {
    'pm3_n5': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\03_Qr500_pm3_n5.9600329"),
    'pm5_n1': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\05_Qr500_pm5_n1.9517572")
}

# Directories for cache/helpers
base_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500")
assessment_dir = base_path / 'cached_data'
timed_out_dir = base_path / 'timed-out'

# --- DATA PROCESSING ---
results = {}

for label, folder_path in MODEL_PATHS.items():
    print(f"\nProcessing: {folder_path.name}")

    # 1. Path resolution logic
    run_paths = get_stitched_map_run_paths(
        base_path=base_path,
        folder_name=folder_path.name,
        timed_out_dir=timed_out_dir,
        variability_map=get_variability_map(DISCHARGE), # Assuming this helper exists
        analyze_noisy=False,
    )
    if not run_paths:
        run_paths = [folder_path]

    # 2. Cache loading
    ds = load_or_update_map_cache_multi(
        cache_dir=assessment_dir,
        folder_name=folder_path.name,
        run_paths=run_paths,
        var_names=['mesh2d_mor_bl'],
        bbox=CACHE_BBOX,
        append_time=APPEND_TIMESTEPS,
        append_vars=APPEND_VARIABLES,
        cache_tag=cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG),
    )

    if ds is None:
        print(f"  No cached data for {folder_path.name}, skipping.")
        continue

    # 3. Extraction
    matches = get_snapshot_matches_by_target_dates(ds.time.values, [TARGET_DATE])
    if not matches:
        print(f"  No snapshot found for {TARGET_DATE}")
        ds.close()
        continue
    
    _, ts_idx, _ = matches[0]
    bedlev = ds['mesh2d_mor_bl'].isel(time=ts_idx).values
    init_bl = ds['mesh2d_mor_bl'].isel(time=0).values
    face_x, face_y = _get_face_coords(ds)
    
    # Masking and Binning
    mask = (face_y >= Y_RANGE[0]) & (face_y <= Y_RANGE[1]) & (init_bl < CHANNEL_INIT_THRESHOLD)
    dx = 1000
    x_bins = np.arange(X_RANGE[0], X_RANGE[1] + dx, dx)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    
    profile = []
    for k in range(len(x_bins) - 1):
        b_mask = mask & (face_x >= x_bins[k]) & (face_x < x_bins[k + 1])
        vals = bedlev[b_mask]
        profile.append(np.percentile(vals[~np.isnan(vals)], DEPTH_PERCENTILE) if len(vals) > 0 else np.nan)
    
    results[label] = (x_centers / 1000, np.array(profile))
    ds.close()

# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(7, 4))
for label, (x, y) in results.items():
    ax.plot(x, y, label=label, linewidth=2)

ax.set_xlabel('Distance along estuary [km]')
ax.set_ylabel(f'p{DEPTH_PERCENTILE} Bed Level [m]')
ax.set_title(f'Snapshot Comparison: {TARGET_DATE}')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()