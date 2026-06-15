"""
Compute the along-estuary tidal prism through the mouth (km 20) for four 
detailed-hydro-run scenarios using the basin storage volume method.
"""

# %% Imports
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SCENARIOS = {
    'constant':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_01_Qr500_pm1_n0.9724783"),
    'low flow':  Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_lowflow.9728497"),
    'mean flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_meanflow.9728503"),
    'peak flow': Path(r"U:\PhDNaturalRhythmEstuaries\Models\2_RiverDischargeVariability_domain45x15_Gaussian\Model_Output\Q500\detailed-hydro-run\dhr_12_Qr500_pm5_n4_peakflow"),
}

VAR_WL   = 'mesh2d_s1'
VAR_BL   = 'mesh2d_mor_bl'
VAR_BA   = 'mesh2d_flowelem_ba'  # Added Flow Element Base Area variable
LOAD_VARS = [VAR_WL, VAR_BL, VAR_BA]

# Spatial extent used when building / reading the cache
CACHE_BBOX       = [1, 1, 45000, 15000]
CACHE_TAG        = "prism_v2"
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# Mouth Location Filter (Everything upstream of km 20)
X_MOUTH = 20000  # metres

# --- Fluvial Correction Parameters ---
# Define your river discharge Q_r (m³/s) per scenario to filter out trapped river water
RIVER_DISCHARGES = {
    'constant':  500,
    'low flow':  100,   # Adjust if your actual model setup values differ!
    'mean flow': 500,
    'peak flow': 2500, 
}
T_FLOOD_HOURS = 6.21   # Semi-diurnal flood duration window (hours)

SCENARIO_COLORS = {
    'constant':  '#888888',
    'low flow':  '#92C5DE',
    'mean flow': '#4393C3',
    'peak flow': '#084594',
}

# ---------------------------------------------------------------------------
# COMPUTE TIDAL PRISM PER SCENARIO
# ---------------------------------------------------------------------------

prism_results = {}
plt.figure(figsize=(8, 4.5))

for label, folder_path in SCENARIOS.items():
    print(f"\nProcessing Scenario: {label}")
    
    base_path      = folder_path.parent
    folder_name    = folder_path.name
    assessment_dir = base_path / 'cached_data'

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
        print(f"  [SKIP] No data found.")
        continue

    # --- Extract Geometry Coordinates ---
    face_x = ds.grid.face_coordinates[:, 0]
    
    # --- Extract Cell Area using your explicit NetCDF variable ---
    if VAR_BA in ds:
        face_area = ds[VAR_BA].values
    elif hasattr(ds.grid, 'face_areas'):
        face_area = ds.grid.face_areas
    else:
        raise KeyError(f"Could not find cell area data via '{VAR_BA}' or geometry attributes.")

    # FIX: If the cache appended a time dimension (making it 2D), grab just the first timestep
    if face_area.ndim == 2:
        face_area = face_area[0, :]

    # --- Extract Water Level ---
    wl = ds[VAR_WL].values  # Shape: (time, nFaces)
    
    # --- Extract Bed Level (with fallback handling) ---
    if VAR_BL in ds:
        bl_raw = ds[VAR_BL].values
    elif 'mesh2d_flowelem_bl' in ds:
        bl_raw = ds['mesh2d_flowelem_bl'].values
    else:
        bl_raw = None
        print("  [WARNING] No explicit bed level variable found; assuming rigid flat baseline (0.0 m).")
    
    if bl_raw is not None and bl_raw.ndim == 1:
        bl_raw = bl_raw[np.newaxis, :]  # Broadcast static 1D array to (1, nFaces)

    # --- Filter Upstream Cells (x >= 20 km) ---
    upstream_mask = face_x >= X_MOUTH
    wl_u = wl[:, upstream_mask]
    bl_u = bl_raw[:, upstream_mask] if bl_raw is not None else 0.0
    area_u = face_area[upstream_mask]

    # --- Compute Water Depth & Storage Volume over Time ---
    depths = wl_u - bl_u
    depths = np.where(np.isfinite(depths) & (depths > 0), depths, 0.0)
    
    # Total volume in the storage basin at each timestep (m³)
    volume_over_time = np.sum(depths * area_u, axis=1)
    
    # --- Calculate Fluvial-Corrected Marine Tidal Prism ---
    v_max = np.max(volume_over_time)
    v_min = np.min(volume_over_time)
    total_storage_swing_m3 = v_max - v_min
    
    # Isolate true marine volume crossing km 20 by subtracting incoming river water
    q_river = RIVER_DISCHARGES.get(label, 500)
    fluvial_flood_volume_m3 = q_river * (T_FLOOD_HOURS * 3600)
    
    pure_tidal_prism_m3 = total_storage_swing_m3 - fluvial_flood_volume_m3
    tidal_prism_m6 = pure_tidal_prism_m3 / 1e6  # Convert to Millions of m³ (Mm³)
    
    prism_results[label] = tidal_prism_m6
    
    # --- Diagnostic Plot Line ---
    time_hours = np.arange(len(volume_over_time)) * (int(ds.time.diff('time').median() / 1e9) / 3600)
    plt.plot(time_hours, volume_over_time / 1e6, label=f"{label} (Prism: {tidal_prism_m6:.2f} Mm³)", 
             color=SCENARIO_COLORS.get(label, 'black'), lw=2)
    
    ds.close()

# ---------------------------------------------------------------------------
# PRINT SUMMARY TABLE & RENDER DIAGNOSTIC PLOT
# ---------------------------------------------------------------------------
print(f"\n{'='*45}\n PURE MARINE TIDAL PRISM RESULTS (at km 20)\n{'='*45}")
for scenario, prism in prism_results.items():
    print(f" {scenario.ljust(12)} : {prism:.3f} Million m³")
print('='*45)

plt.title('Estuary Basin Water Volume Upstream of km 20', fontsize=11, fontweight='bold')
plt.xlabel('Time [hours]')
plt.ylabel('Water Volume [$10^6\ \mathrm{m^3}$]')
plt.grid(True, lw=0.4, alpha=0.5)
plt.legend(title="Scenario (Marine Prism)", frameon=True, fontsize=9)
plt.tight_layout()
plt.show()