"""
Vegetation analysis along estuarine x-axis using x-binning.

Figures produced (each can be switched on/off with a TRUE/FALSE flag below):
  - Fig VEG_XBIN     : per-timestep % vegetation vs x (bars coloured by mean stem density)
  - Fig INTERTIDAL   : intertidal & supertidal zone along the estuary (absolute width in m  +  % of total estuary width per cross-section)
  - Fig WATERDEPTH   : mean water depth along the estuary, all timesteps superposed
  - Fig MEAN_SCORE   : temporal mean score-DI vs x
  - Fig MEAN_VEGPCT  : temporal mean vegetation % vs x  (relative to ALL cells only)
  - Fig MEAN_DENSITY : temporal mean stem density vs x
  - Fig MORTALITY    : mortality along estuary, one curve per available timestep
  - Fig BEDLEVEL     : bed level evolution at key cross-sections
  - Fig MAP_BESTVEG  : map of best score-DI x-position per timestep
  - Fig MAP_KEYCROSS : map of the key cross-section x-positions

Two vegetation scores are used and always named explicitly to avoid any ambiguity:
  * score-DI = mean stem density x intertidal (colonizable) fraction of the cross-section  [MAIN]
               -> drives the key positions, the best-position map and the bed level cross-sections.
  * score-VD = vegetation % (relative to all cells) x mean stem density                    [COMPARISON]
               -> shown only on the key cross-sections map (letter P) to compare with score-DI.
"""
#%%
import gc
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
#%%
# ─────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────

VEG_OUTPUT_DIR = Path(r"u:\PhDNaturalRhythmEstuaries\Models\DYCOVE_vegetation\DYCOVE_starterpack_fromCamille_forMarloes\FlowFM\input\dflowfm\veg_output")
OUTPUT_DIR     = Path(r"u:\PhDNaturalRhythmEstuaries\Models\DYCOVE_vegetation\DYCOVE_starterpack_fromCamille_forMarloes\FlowFM\figures\CrossSections2")
MAP_NC         = Path(r"u:\PhDNaturalRhythmEstuaries\Models\DYCOVE_vegetation\DYCOVE_starterpack_fromCamille_forMarloes\FlowFM\input\dflowfm\output\FlowFM_map.nc")

FRACTION_THRESHOLD = 0.01
DEPTH_THRESHOLD    = 0.0
XBIN_SIZE_M        = 250
Y_CROSS_WIDTH_M    = 175
CMAP_DENSITY       = "YlGn"
DENSITY_VMIN       = 0
DENSITY_VMAX       = 500
N_KEY_POSITIONS    = 3
VEG_XLIM           = 17000   # scalar -> [VEG_XLIM, x_max] (left bound) ; tuple (a,b) -> [a,b] ; None -> full range
VEG_PCT_YMAX       = 100

# ── Which ETS to process ──
# None  = process EVERY available ETS  -> denser mortality / water depth / score curves
# int   = only that ETS   or   list = only those ETS
PLOT_ONLY_ETS      = 4
# Per-timestep Fig VEG_XBIN can generate a LOT of PNGs when all ETS are processed.
# Limit it here (int / list / None = all processed ETS). Data is always collected for every ETS.
VEG_XBIN_ONLY_ETS  = None

# ── Intertidal / supertidal classification (from inundation frequency over the simulation) ──
# A cell is wet when waterdepth > DEPTH_THRESHOLD. freq = fraction of sampled timesteps a cell is wet.
#   freq >= INUNDATION_HIGH        -> subtidal   (almost always wet)
#   INUNDATION_LOW < freq < HIGH   -> intertidal (colonizable zone)
#   0 < freq <= INUNDATION_LOW     -> supertidal (rarely flooded, above the intertidal)
INUNDATION_LOW     = 0.01
INUNDATION_HIGH    = 0.99
N_FREQ_SAMPLES     = 40      # number of map timesteps sampled to estimate inundation frequency
# Depth (m) above which a cell counts as "wet" for the ZONE classification only.
# Raise it above 0 so that thin residual water films on drying cells (a Delft3D-FM artefact)
# are not counted as "always wet" - otherwise almost everything ends up classified as subtidal
# and the intertidal zone looks negligible. Set to 0.0 to reproduce the previous behaviour.
INUNDATION_WET_DEPTH = 0.05

# ── The two vegetation scores, named unambiguously and reused in every title/legend/print ──
#   score-DI  = density x intertidal fraction   (the main score, used for key positions/maps)
#   score-VD  = veg% x density                  (comparison score only)
SCORE_DI_NAME    = "score-DI"
SCORE_DI_FORMULA = "density x intertidal fraction"
SCORE_VD_NAME    = "score-VD"
SCORE_VD_FORMULA = "veg% x density"
SCORE_DI_LABEL   = f"{SCORE_DI_NAME} ({SCORE_DI_FORMULA})"
SCORE_VD_LABEL   = f"{SCORE_VD_NAME} ({SCORE_VD_FORMULA})"

MORT_PCT_YMAX      = None    # None = autoscale ; number = fixed upper y-limit for mortality plots
# Year window for the mortality plots, in displayed veg-years (see the 'X years' labels).
# None = no limit. Example: MORT_YEAR_MIN = 5, MORT_YEAR_MAX = 15  -> only years 5 to 15.
MORT_YEAR_MIN      = None
MORT_YEAR_MAX      = None

CREATE_ANIMATIONS  = False
ANIMATION_FPS      = 3
ANIMATION_FORMAT   = "gif"
TIME_TOLERANCE     = np.timedelta64(2, "h")

# ─────────────────────────────────────────────
# FIGURE TOGGLES  (True = generate, False = skip)
# ─────────────────────────────────────────────
# SHOW_FIG_VEG_XBIN     = False   # Fig 1  : per-timestep veg% vs x
# SHOW_FIG_INTERTIDAL   = False    # NEW    : intertidal + supertidal zone along estuary (abs + %)
# SHOW_FIG_WATERDEPTH   = False    # NEW    : water depth along estuary, all timesteps
# SHOW_FIG_MEAN_SCORE   = False    # temporal mean score vs x
# SHOW_FIG_MEAN_VEGPCT  = False    # temporal mean veg% vs x  (relative to all active cells)
# SHOW_FIG_VEG_INTER    = False   # NEW    : temporal mean veg% vs x  (relative to intertidal cells)
# SHOW_FIG_MEAN_DENSITY = False   # temporal mean density vs x
# SHOW_FIG_MORTALITY    = True    # mortality along estuary, per timestep
# SHOW_FIG_BEDLEVEL     = False   # bed level evolution at key cross-sections
# SHOW_FIG_MAP_BESTVEG  = False    # map: best veg position per timestep
# SHOW_FIG_MAP_KEYCROSS = False    # map: key cross-section positions

SHOW_FIG_VEG_XBIN     = True   # Fig 1  : per-timestep veg% vs x
SHOW_FIG_INTERTIDAL   = True    # NEW    : intertidal + supertidal zone along estuary (abs + %)
SHOW_FIG_WATERDEPTH   = True    # NEW    : water depth along estuary, all timesteps
SHOW_FIG_MEAN_SCORE   = True   # temporal mean score vs x
SHOW_FIG_MEAN_VEGPCT  = True    # temporal mean veg% vs x  (relative to all active cells)
SHOW_FIG_VEG_INTER    = True   # NEW    : temporal mean veg% vs x  (relative to intertidal cells)
SHOW_FIG_MEAN_DENSITY = True   # temporal mean density vs x
SHOW_FIG_MORTALITY    = True    # mortality along estuary, per timestep
SHOW_FIG_BEDLEVEL     = True   # bed level evolution at key cross-sections
SHOW_FIG_MAP_BESTVEG  = True    # map: best veg position per timestep
SHOW_FIG_MAP_KEYCROSS = True    # map: key cross-section positions

# Mortality causes to plot (set to False to skip)
MORT_CAUSES = {
    "applied_mort_flood"  : True,   # flooding mortality
    "applied_mort_uproot" : True,   # uprooting mortality
    "applied_mort_total"  : True,   # total mortality
    "applied_mort_desic"  : True,
    "applied_mort_burial" : True,
    "applied_mort_scour"  : True,
}
MORT_COLORS = {
    "applied_mort_flood"  : "steelblue",
    "applied_mort_uproot" : "darkorange",
    "applied_mort_total"  : "crimson",
    "applied_mort_desic"  : "purple",
    "applied_mort_burial" : "brown",
    "applied_mort_scour"  : "gray",
}

# Resolution for background bathymetry maps
MAP_CELL_SIZE      = 100  # m

# ─────────────────────────────────────────────
# HELPErS
# ─────────────────────────────────────────────

def _as_set(v):
    """None -> None ; int -> {int} ; iterable -> set."""
    if v is None:
        return None
    if np.isscalar(v):
        return {int(v)}
    return set(int(x) for x in v)

_PLOT_ETS_SET  = _as_set(PLOT_ONLY_ETS)
_FIG1_ETS_SET  = _as_set(VEG_XBIN_ONLY_ETS)


def apply_xlim(ax):
    """Apply VEG_XLIM consistently (scalar -> [x_min, VEG_XLIM], tuple -> as given, None -> full)."""
    if VEG_XLIM is None:
        ax.set_xlim(x_min, x_max)
    elif np.isscalar(VEG_XLIM):
        ax.set_xlim(VEG_XLIM, x_max)   # scalar = LEFT bound (start x at VEG_XLIM), as in the original
    else:
        ax.set_xlim(VEG_XLIM[0], VEG_XLIM[1])


def set_time_ticks(cbar, labels, max_ticks=15):
    """Thin the per-timestep colorbar ticks so labels stay readable when there are many timesteps."""
    n = len(labels)
    if n == 0:
        return
    step = max(1, int(np.ceil(n / max_ticks)))
    idx  = list(range(0, n, step))
    cbar.set_ticks([i + 1 for i in idx])
    cbar.set_ticklabels([f"{i + 1}: {labels[i]}" for i in idx], fontsize=7)

# ─────────────────────────────────────────────
# LOAD ECO TIME VARS
# ─────────────────────────────────────────────

with open(VEG_OUTPUT_DIR / "_eco_time_vars.json") as f:
    eco = json.load(f)

n_ets         = eco["n_ets"]
ecofac        = eco["ecofac"]
veg_int_hr    = int(eco["veg_interval"] / 3600)
days_per_year = (ecofac * veg_int_hr * n_ets) / 24.0

print(f"Eco params: n_ets={n_ets}, ecofac={ecofac}, veg_int_hr={veg_int_hr}h")
print(f"days_per_year = {days_per_year:.1f} days")

veg_state_files = sorted(VEG_OUTPUT_DIR.glob("veg_state_ey*_ets*.nc"))
cohort_files    = sorted(VEG_OUTPUT_DIR.glob("cohort*_*.nc"))
print(f"Found {len(veg_state_files)} veg_state files, {len(cohort_files)} cohort files")

if len(veg_state_files) == 0:
    raise FileNotFoundError(f"No veg_state files found in {VEG_OUTPUT_DIR}")

# ─────────────────────────────────────────────
# TIME HELPERS
# ─────────────────────────────────────────────

def eco_label(eco_year, ets):
    sim_hours   = ((eco_year - 1) * n_ets + ets) * veg_int_hr
    veg_years_f = sim_hours * ecofac / 24.0 / days_per_year
    veg_years   = int(veg_years_f)
    veg_days    = int(round((veg_years_f % 1) * days_per_year))
    if veg_years > 0:
        return f"{veg_years} years" if veg_days == 0 else f"{veg_years} years, {veg_days} days"
    return f"{int(round(sim_hours * ecofac / 24.0))} days"

def eco_year_value(eco_year, ets):
    """Fractional veg-year of a timestep (matches the 'X years' shown on the plots)."""
    sim_hours = ((eco_year - 1) * n_ets + ets) * veg_int_hr
    return sim_hours * ecofac / 24.0 / days_per_year

def get_target_time(ds_vs, eco_year, ets, map_start_time):
    if "time" in ds_vs.coords and ds_vs["time"].size > 0:
        return np.datetime64(ds_vs["time"].values.ravel()[0])
    if "time" in ds_vs.attrs:
        return np.datetime64(ds_vs.attrs["time"])
    sim_hours = ((eco_year - 1) * n_ets + ets) * veg_int_hr
    return map_start_time + np.timedelta64(int(sim_hours), "h")

def get_nearest_time_index(target_time, time_values):
    time_diffs = np.abs(time_values - target_time)
    idx = int(np.argmin(time_diffs))
    return idx, time_diffs[idx]

# ─────────────────────────────────────────────
# LOAD MAP / GRID
# ─────────────────────────────────────────────

ds_map = xr.open_dataset(MAP_NC)
x_coords         = ds_map["mesh2d_face_x"].values
y_coords         = ds_map["mesh2d_face_y"].values
bed_level_static = ds_map["mesh2d_flowelem_bl"].values
time_values      = ds_map["time"].values
map_start_time   = np.datetime64(time_values[0])
x_min, x_max     = x_coords.min(), x_coords.max()

print(f"Grid: {len(x_coords)} faces, X: {x_min:.0f}-{x_max:.0f} m")
print(f"Map: {len(time_values)} timesteps, {time_values[0]} -> {time_values[-1]}")

# Cell area (used for zone widths). (Fall back to a nearest-neighbour estimate if not present.
if "mesh2d_flowelem_ba" in ds_map:
    cell_area = ds_map["mesh2d_flowelem_ba"].values.astype(float)
    print("Cell area: from mesh2d_flowelem_ba")
else:
    nn_d, _   = cKDTree(np.column_stack([x_coords, y_coords])).query(
                    np.column_stack([x_coords, y_coords]), k=2)
    dcell     = float(np.median(nn_d[:, 1]))
    cell_area = np.full(len(x_coords), dcell ** 2, dtype=float)
    print(f"Cell area: estimated from grid spacing (~{dcell:.0f} m cells)")

# ─────────────────────────────────────────────
# INUNDATION FREQUENCY
# ─────────────────────────────────────────────

print("Computing inundation frequency from map.nc ...")
freq_idx = np.unique(np.linspace(0, len(time_values) - 1,
                                 min(len(time_values), N_FREQ_SAMPLES)).astype(int))
wd_sample  = ds_map["mesh2d_waterdepth"].isel(time=freq_idx).values      # (n_sample, n_cells)
# Active cells = wet at least once (any positive depth). Avoids a hand-drawn polygon.
in_polygon = np.any(wd_sample > DEPTH_THRESHOLD, axis=0)
# Inundation frequency for the tidal-zone classification uses the (higher) wet-depth threshold.
inund_freq = np.mean(wd_sample > INUNDATION_WET_DEPTH, axis=0)           # per cell, 0..1
del wd_sample; gc.collect()
print(f"Active cells (ever wet): {in_polygon.sum()} / {len(x_coords)}")
print(f"Zone classification wet-depth threshold: {INUNDATION_WET_DEPTH} m")

is_subtidal   = in_polygon & (inund_freq >= INUNDATION_HIGH)
is_intertidal = in_polygon & (inund_freq >  INUNDATION_LOW) & (inund_freq < INUNDATION_HIGH)
is_supertidal = in_polygon & (inund_freq <= INUNDATION_LOW)
print(f"  subtidal={is_subtidal.sum()}, intertidal={is_intertidal.sum()}, "
      f"supertidal={is_supertidal.sum()}")

# ─────────────────────────────────────────────
# BACKGROUND BATHYMETRY GRID (for maps)
# ─────────────────────────────────────────────

def build_bathy_grid():
    xf, yf = x_coords[in_polygon], y_coords[in_polygon]
    bl     = bed_level_static[in_polygon]
    xg     = np.arange(xf.min(), xf.max() + MAP_CELL_SIZE, MAP_CELL_SIZE)
    yg     = np.arange(yf.min(), yf.max() + MAP_CELL_SIZE, MAP_CELL_SIZE)
    X, Y   = np.meshgrid(xg, yg)
    pts    = np.column_stack([X.ravel(), Y.ravel()])
    tree   = cKDTree(np.column_stack([xf, yf]))
    nn_dist, idx = tree.query(pts, k=1, workers=-1)
    Z      = bl[idx].reshape(X.shape).astype(float)
    # Mask grid points far from any active cell
    pmask  = nn_dist < MAP_CELL_SIZE * 2
    Z[~pmask.reshape(X.shape)] = np.nan
    return np.flipud(Z), xg, np.flip(yg)

print("Building bathymetry background grid...")
Z_bathy, xg_map, yg_map = build_bathy_grid()
cmap_bathy = plt.colormaps["Greys_r"].copy()
cmap_bathy.set_bad("white")
bathy_vmin = np.nanpercentile(Z_bathy, 2)
bathy_vmax = np.nanpercentile(Z_bathy, 98)
mid_y      = (y_coords[in_polygon].min() + y_coords[in_polygon].max()) / 2

def draw_bathy(ax, aspect="auto"):
    """
    Draw bathymetry background on ax, return extent.
    aspect defaults to "auto" so the (very elongated) estuary fills the panel and the
    background is clearly visible. Use aspect="equal" for true geographic proportions.
    """
    extent = [xg_map.min(), xg_map.max(), yg_map.min(), yg_map.max()]
    ax.imshow(Z_bathy, origin="upper", aspect=aspect, cmap=cmap_bathy,
              vmin=bathy_vmin, vmax=bathy_vmax, extent=extent, alpha=0.85, zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    return extent

# ─────────────────────────────────────────────
# X BINS  (cross-section)
# ─────────────────────────────────────────────

x_edges      = np.arange(x_min, x_max + XBIN_SIZE_M, XBIN_SIZE_M)
x_centers    = 0.5 * (x_edges[:-1] + x_edges[1:])
n_bins       = len(x_centers)
x_bin_masks  = [(x_coords >= x_edges[i]) & (x_coords < x_edges[i+1]) & in_polygon
                for i in range(len(x_edges)-1)]
x_bin_ncells = np.array([m.sum() for m in x_bin_masks])
print(f"X-bins: {n_bins} bins of {XBIN_SIZE_M}m")

# ─────────────────────────────────────────────
# INTERTIDAL / SUPERTIDAL WIDTH PER CROSS-SECTION
# ─────────────────────────────────────────────
# A 250 m x-bin usually spans several cell columns, and the grid resolution VARIES along the
# estuary (fine near the sea, coarser inland). So we estimate each cell's across-channel size as
# dy ~ sqrt(area); summing dy over the cells of one x-column gives that column's y-length, and
# averaging over the columns in the bin gives the mean cross-section width. This is
# resolution-independent and, by construction, the total width can never be smaller than the
# intertidal width (both are y-lengths measured the same way).
dy_cell = np.sqrt(np.clip(cell_area, 0, None))

intertidal_width = np.full(n_bins, np.nan)
supertidal_width = np.full(n_bins, np.nan)
subtidal_width   = np.full(n_bins, np.nan)
total_width      = np.full(n_bins, np.nan)
intertidal_pct   = np.full(n_bins, np.nan)
supertidal_pct   = np.full(n_bins, np.nan)

for i, mask in enumerate(x_bin_masks):
    if x_bin_ncells[i] == 0:
        continue
    n_col = np.unique(x_coords[mask]).size            # number of cell columns in this bin
    if n_col == 0:
        continue
    w_tot = dy_cell[mask].sum() / n_col               # mean column y-length = cross-section width
    if w_tot <= 0:
        continue
    total_width[i]      = w_tot
    intertidal_width[i] = dy_cell[mask & is_intertidal].sum() / n_col
    supertidal_width[i] = dy_cell[mask & is_supertidal].sum() / n_col
    subtidal_width[i]   = dy_cell[mask & is_subtidal].sum() / n_col
    intertidal_pct[i]   = 100.0 * intertidal_width[i] / w_tot
    supertidal_pct[i]   = 100.0 * supertidal_width[i] / w_tot

# fraction of colonizable (intertidal) zone per cross-section, used by the new score
intertidal_frac = np.where(np.isnan(intertidal_pct), 0.0, intertidal_pct) / 100.0

# ─────────────────────────────────────────────
# STORAGE
# ─────────────────────────────────────────────

all_score_di    = []
all_timestep_data = []
best_x_per_ts     = []
anim_paths        = {"vegetation": []}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

for vs_file in veg_state_files:
    ds_vs = xr.open_dataset(vs_file)
    try:
        eco_year = int(ds_vs.attrs["eco_year"])
        ets      = int(ds_vs.attrs["ets"])
    except KeyError as e:
        print(f"Skipping {vs_file.name}: missing attribute {e}")
        ds_vs.close(); continue

    if _PLOT_ETS_SET is not None and ets not in _PLOT_ETS_SET:
        ds_vs.close(); continue

    label = eco_label(eco_year, ets)
    print(f"\nProcessing ey{eco_year:02d}_ets{ets:02d} - {label}")

    try:
        target_time = get_target_time(ds_vs, eco_year, ets, map_start_time)
    except Exception as e:
        print(f"  WARNING: {e}"); ds_vs.close(); continue

    if target_time < time_values[0] or target_time > time_values[-1]:
        print(f"  WARNING: target time {target_time} outside map range"); ds_vs.close(); continue

    hydro_idx, nearest_dt = get_nearest_time_index(target_time, time_values)
    if nearest_dt > TIME_TOLERANCE:
        print(f"  WARNING: nearest map time too far ({nearest_dt}), skipping"); ds_vs.close(); continue

    mor_bl     = ds_map["mesh2d_mor_bl"].isel(time=hydro_idx).values
    waterdepth = ds_map["mesh2d_waterdepth"].isel(time=hydro_idx).values

    stem_density = ds_vs["stemdensity"].values
    ds_vs.close()

    # ── Read every cohort file for this (eco_year, ets) ONCE: fraction + all mortality causes ──
    active_mort_causes = [c for c, enabled in MORT_CAUSES.items() if enabled]
    cohort_fractions = []
    cohort_mort      = {cause: [] for cause in active_mort_causes}
    for cf in cohort_files:
        try:
            ds_c = xr.open_dataset(cf)
        except OSError:
            print(f"  WARNING: skipping corrupted {cf.name}"); continue
        if (int(ds_c.attrs.get("eco_year", -1)) == eco_year and
                int(ds_c.attrs.get("ets", -1)) == ets):
            cohort_fractions.append(ds_c["fraction"].values)
            for cause in active_mort_causes:
                if cause in ds_c.data_vars:
                    cohort_mort[cause].append(ds_c[cause].values)
        ds_c.close()

    if not cohort_fractions:
        print("  No cohort files, skipping."); continue

    total_fraction = np.clip(np.sum(cohort_fractions, axis=0), 0, 1)

    # ── Per-bin vegetation metrics ──
    veg_pct_total = np.full(n_bins, np.nan)   # % vegetated cells / all active cells in the bin
    veg_pct_inter = np.full(n_bins, np.nan)   # % vegetated cells / intertidal (colonizable) cells
    mean_dens     = np.full(n_bins, np.nan)   # mean stem density over vegetated cells
    score_di     = np.full(n_bins, np.nan)   # score-DI = mean density * intertidal fraction (main score)
    wd_bin        = np.full(n_bins, np.nan)   # mean water depth over wetted cells in the bin

    for i, (mask, ncells) in enumerate(zip(x_bin_masks, x_bin_ncells)):
        if ncells == 0:
            continue
        frac_b   = total_fraction[mask]
        dens_b   = stem_density[mask]
        veg_mask = frac_b >= FRACTION_THRESHOLD
        n_veg    = int(veg_mask.sum())
        veg_pct_total[i] = 100.0 * n_veg / ncells
        # all vegetated cells of the cross-section / intertidal (colonizable) cells (may exceed 100%)
        n_int = int((mask & is_intertidal).sum())
        if n_int > 0:
            veg_pct_inter[i] = 100.0 * n_veg / n_int
        mean_dens[i]     = dens_b[veg_mask].mean() if n_veg > 0 else 0.0
        # NEW score : density weighted by how colonizable (intertidal) the cross-section is
        score_di[i]     = mean_dens[i] * intertidal_frac[i]
        # water depth of the wetted portion of the cross-section
        wd_b = waterdepth[mask]
        wet  = wd_b > DEPTH_THRESHOLD
        if wet.sum() > 0:
            wd_bin[i] = wd_b[wet].mean()

    if np.all(np.isnan(score_di)) or np.nanmax(score_di) <= 0:
        print("  No valid scores, skipping."); continue

    best_i = int(np.nanargmax(score_di))
    best_x = x_centers[best_i]
    print(f"  Best x-bin: x={best_x:.0f}m | veg%={veg_pct_total[best_i]:.1f}% | "
          f"intertidal%={intertidal_pct[best_i]:.1f}% | "
          f"density={mean_dens[best_i]:.0f} m^-2 | {SCORE_DI_NAME}={score_di[best_i]:.1f}")

    # ── Mean mortality per x-bin (mean over vegetated cells) ──
    mort_by_bin = {}
    for cause in active_mort_causes:
        mort_arr = np.full(n_bins, np.nan)
        if cohort_mort[cause]:
            mort_mean_cell = np.mean(cohort_mort[cause], axis=0)  # mean across cohorts per cell
            for i, (mask, ncells) in enumerate(zip(x_bin_masks, x_bin_ncells)):
                if ncells == 0:
                    continue
                veg_m = total_fraction[mask] >= FRACTION_THRESHOLD
                if veg_m.sum() > 0:
                    mort_arr[i] = mort_mean_cell[mask][veg_m].mean()
        mort_by_bin[cause] = mort_arr

    all_score_di.append(score_di.copy())
    best_x_per_ts.append({"label": label, "x": best_x, "score": score_di[best_i]})
    all_timestep_data.append({
        "eco_year": eco_year, "ets": ets, "label": label,
        "veg_year": eco_year_value(eco_year, ets),
        "mor_bl": mor_bl.copy(),
        "total_fraction": total_fraction.copy(),
        "stem_density": stem_density.copy(),
        "veg_pct_total_arr": veg_pct_total.copy(),
        "veg_pct_inter_arr": veg_pct_inter.copy(),
        "mean_dens_arr":     mean_dens.copy(),
        "waterdepth_arr":    wd_bin.copy(),
        "mortality":         mort_by_bin,
    })

    # ── Fig VEG_XBIN : Vegetation % vs x (relative to all active cells only) ──
    make_fig1 = SHOW_FIG_VEG_XBIN and (_FIG1_ETS_SET is None or ets in _FIG1_ETS_SET)
    if make_fig1:
        cmap_dens = plt.colormaps[CMAP_DENSITY]
        norm_dens = mcolors.Normalize(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX)
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        valid = ~np.isnan(veg_pct_total)
        for i in np.where(valid)[0]:
            ax1.bar(x_centers[i], veg_pct_total[i], width=XBIN_SIZE_M * 0.9,
                    color=cmap_dens(norm_dens(mean_dens[i] if not np.isnan(mean_dens[i]) else 0)),
                    edgecolor="none", zorder=2)
        ax1.plot(x_centers, veg_pct_total, color="black", lw=1.5, ls="-",
                 marker="o", ms=3, label="% veg / all cells", zorder=4)
        ax1.axvline(best_x, color="purple", lw=1.2, ls=":", label=f"Best x ({best_x:.0f}m)")
        sm1 = plt.cm.ScalarMappable(cmap=cmap_dens, norm=norm_dens); sm1.set_array([])
        plt.colorbar(sm1, ax=ax1, pad=0.01).set_label("Mean Density [m^-2]", fontsize=9)
        ax1.set_xlabel("X coordinate [m]", fontsize=11)
        ax1.set_ylabel("Vegetated cells [%]", fontsize=11)
        ax1.set_title(f"Vegetation cover - {label}", fontsize=12)
        apply_xlim(ax1)
        ax1.set_ylim(0, VEG_PCT_YMAX)
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(axis="y", alpha=0.3)
        fig1.tight_layout()
        p = str(OUTPUT_DIR / f"VegXBin_ey{eco_year:02d}_ets{ets:02d}.png")
        fig1.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig1); gc.collect()
        anim_paths["vegetation"].append(p)
        print(f"  Saved: VegXBin_ey{eco_year:02d}_ets{ets:02d}.png")

ds_map.close()

# ─────────────────────────────────────────────
# POST-PROCESSING
# ─────────────────────────────────────────────

if len(all_score_di) == 0:
    print("No timesteps processed. Done.")
    raise SystemExit

score_di_mat = np.array(all_score_di)                                   # (n_ts, n_bins)
vegp_mat      = np.array([tsd["veg_pct_total_arr"] for tsd in all_timestep_data])
vegpi_mat     = np.array([tsd["veg_pct_inter_arr"] for tsd in all_timestep_data])
dens_mat      = np.array([tsd["mean_dens_arr"]     for tsd in all_timestep_data])
wd_mat        = np.array([tsd["waterdepth_arr"]    for tsd in all_timestep_data])

max_score_di_per_x = np.nanmax(score_di_mat, axis=0)
max_dens_per_x  = np.nanmax(dens_mat,      axis=0)
max_vegp_per_x  = np.nanmax(vegp_mat,      axis=0)
# Alternative "score" for comparison = veg% (relative to all cells) x mean density, per timestep
score_vd_mat       = vegp_mat * dens_mat
max_score_vd_per_x = np.nanmax(score_vd_mat, axis=0)

# ── Key positions ──
def get_top_positions(metric, label_str, already_selected_x, n=N_KEY_POSITIONS):
    candidates = {i: metric[i] for i in range(n_bins)
                  if not np.isnan(metric[i]) and x_centers[i] not in already_selected_x}
    top = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [{"rank": r + 1, "x": x_centers[i], "max_val": val, "bin_idx": i, "criterion": label_str}
            for r, (i, val) in enumerate(top)]

top_score   = sorted({i: max_score_di_per_x[i] for i in range(n_bins)
                      if not np.isnan(max_score_di_per_x[i])}.items(),
                     key=lambda kv: kv[1], reverse=True)[:N_KEY_POSITIONS]
top_score_x = set(x_centers[i] for i, _ in top_score)
top_indices = np.sort([i for i, _ in top_score])
key_score_di = [{"rank": r + 1, "x": x_centers[i], "max_score": max_score_di_per_x[i],
                  "bin_idx": i, "criterion": "score"}
                 for r, i in enumerate(top_indices)]

key_dens   = get_top_positions(max_dens_per_x, "density", top_score_x)
top_dens_x = top_score_x | set(kp["x"] for kp in key_dens)
key_vegp   = get_top_positions(max_vegp_per_x, "veg%",    top_dens_x)

# Top positions by intertidal (colonizable) area - independent, time-invariant criterion
_it = sorted([(i, intertidal_width[i]) for i in range(n_bins)
              if not np.isnan(intertidal_width[i]) and intertidal_width[i] > 0],
             key=lambda t: t[1], reverse=True)[:N_KEY_POSITIONS]
key_intertidal = [{"rank": r + 1, "x": x_centers[i], "max_val": val,
                   "int_pct": intertidal_pct[i], "bin_idx": i, "criterion": "intertidal"}
                  for r, (i, val) in enumerate(_it)]

# Top positions by the comparison score veg% x density (independent, to compare with the main score)
_s2 = sorted([(i, max_score_vd_per_x[i]) for i in range(n_bins)
              if not np.isnan(max_score_vd_per_x[i]) and max_score_vd_per_x[i] > 0],
             key=lambda t: t[1], reverse=True)[:N_KEY_POSITIONS]
key_score_vd = [{"rank": r + 1, "x": x_centers[i], "max_val": val,
               "bin_idx": i, "criterion": "score-VD"}
              for r, (i, val) in enumerate(_s2)]

print(f"\nScores:  {SCORE_DI_NAME} = {SCORE_DI_FORMULA}  (main)   |   "
      f"{SCORE_VD_NAME} = {SCORE_VD_FORMULA}  (comparison)")
print(f"Top {N_KEY_POSITIONS} by {SCORE_DI_NAME}:")
for kp in key_score_di:
    print(f"  #{kp['rank']}: x={kp['x']:.0f}m | max {SCORE_DI_NAME}={kp['max_score']:.1f}")
print(f"Additional top {N_KEY_POSITIONS} by density:")
for kp in key_dens:
    print(f"  #{kp['rank']}: x={kp['x']:.0f}m | max_density={kp['max_val']:.0f} m^-2")
print(f"Additional top {N_KEY_POSITIONS} by veg%:")
for kp in key_vegp:
    print(f"  #{kp['rank']}: x={kp['x']:.0f}m | max_veg%={kp['max_val']:.1f}%")
print(f"Top {N_KEY_POSITIONS} by intertidal area:")
for kp in key_intertidal:
    print(f"  #{kp['rank']}: x={kp['x']:.0f}m | intertidal width={kp['max_val']:.0f} m "
          f"({kp['int_pct']:.1f}% of section)")
print(f"Top {N_KEY_POSITIONS} by {SCORE_VD_NAME} ({SCORE_VD_FORMULA}) [comparison]:")
for kp in key_score_vd:
    print(f"  #{kp['rank']}: x={kp['x']:.0f}m | max {SCORE_VD_NAME}={kp['max_val']:.0f}")

# ── Shared colormaps ──
cmap_time  = plt.colormaps["viridis"]
norm_time  = mcolors.Normalize(vmin=1, vmax=len(all_timestep_data))
ts_labels  = [tsd["label"] for tsd in all_timestep_data]

def add_top3_lines(ax, positions):
    for kp in positions:
        ax.axvline(kp["x"], color="red", lw=1.2, ls="--", alpha=0.8,
                   label=f"S{kp['rank']} x={kp['x']:.0f}m")

def add_key_lines(ax, positions, color, prefix, ls="--"):
    for kp in positions:
        ax.axvline(kp["x"], color=color, lw=1.3, ls=ls, alpha=0.85,
                   label=f"{prefix}{kp['rank']} x={kp['x']:.0f}m")

# ─────────────────────────────────────────────
# INTERTIDAL / SUPERTIDAL ZONE ALONG THE ESTUARY
# ─────────────────────────────────────────────
if SHOW_FIG_INTERTIDAL:
    print("\nGenerating intertidal / supertidal figure...")
    fig_it, (ax_abs, ax_pct) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    v = ~np.isnan(total_width)
    ax_abs.plot(x_centers[v], total_width[v], color="0.35", lw=1.5,
                label="Total estuary width")
    ax_abs.fill_between(x_centers[v], 0, intertidal_width[v],
                        color="forestgreen", alpha=0.20, zorder=1)
    ax_abs.plot(x_centers[v], intertidal_width[v], color="forestgreen", lw=1.8,
                marker="o", ms=3, label="Intertidal (colonizable)")
    ax_abs.plot(x_centers[v], supertidal_width[v], color="saddlebrown", lw=1.8,
                marker="s", ms=3, label="Supertidal")
    ax_abs.set_ylabel("Width across the estuary [m]", fontsize=11)
    ax_abs.set_ylim(bottom=0)
    ax_abs.legend(fontsize=8, loc="upper right")
    ax_abs.set_title("Intertidal & supertidal zone width along the estuary", fontsize=12)
    ax_abs.grid(alpha=0.3)

    vp = ~np.isnan(intertidal_pct)
    ax_pct.plot(x_centers[vp], intertidal_pct[vp], color="forestgreen", lw=1.8,
                marker="o", ms=3, label="Intertidal % of cross-section")
    ax_pct.plot(x_centers[vp], supertidal_pct[vp], color="saddlebrown", lw=1.8,
                marker="s", ms=3, label="Supertidal % of cross-section")
    ax_pct.set_xlabel("X coordinate [m]", fontsize=11)
    ax_pct.set_ylabel("Fraction of cross-section [%]", fontsize=11)
    ax_pct.set_title("Intertidal & supertidal share of the total estuary width", fontsize=12)
    ax_pct.legend(fontsize=8, loc="upper right"); ax_pct.grid(alpha=0.3)
    pmax = np.nanmax([np.nanmax(intertidal_pct), np.nanmax(supertidal_pct)])
    ax_pct.set_ylim(0, max(pmax * 1.25, 1.0))   # autoscale so the small % are visible
    apply_xlim(ax_pct)

    fig_it.tight_layout()
    fig_it.savefig(str(OUTPUT_DIR / "Intertidal_Supertidal_vs_X.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_it); gc.collect()
    print("  Saved: Intertidal_Supertidal_vs_X.png")

# ─────────────────────────────────────────────
# NEW FIG : WATER DEPTH ALONG THE ESTUARY
# ─────────────────────────────────────────────
if SHOW_FIG_WATERDEPTH:
    print("\nGenerating water depth figure...")
    fig_wd, ax_wd = plt.subplots(figsize=(12, 5))
    for k, tsd in enumerate(all_timestep_data):
        arr   = tsd["waterdepth_arr"]
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            continue
        ax_wd.plot(x_centers[valid], arr[valid], color=cmap_time(norm_time(k + 1)),
                   lw=1.2, alpha=0.8, marker="o", ms=2)
    # time-mean profile on top
    mean_wd = np.nanmean(wd_mat, axis=0)
    vw = ~np.isnan(mean_wd)
    ax_wd.plot(x_centers[vw], mean_wd[vw], color="black", lw=2.2, label="Temporal mean")
    sm_wd = plt.cm.ScalarMappable(cmap=cmap_time, norm=norm_time); sm_wd.set_array([])
    cbar_wd = fig_wd.colorbar(sm_wd, ax=ax_wd, pad=0.02)
    cbar_wd.set_label("Timestep", fontsize=9)
    set_time_ticks(cbar_wd, ts_labels)
    ax_wd.set_xlabel("X coordinate [m]", fontsize=11)
    ax_wd.set_ylabel("Mean water depth (wet cells) [m]", fontsize=11)
    ax_wd.set_title("Water depth along the estuary\n(mean over wetted cells per x-bin, all timesteps)",
                    fontsize=12)
    apply_xlim(ax_wd)
    ax_wd.set_ylim(bottom=0)
    ax_wd.legend(fontsize=9, loc="upper right"); ax_wd.grid(axis="y", alpha=0.3)
    fig_wd.tight_layout()
    fig_wd.savefig(str(OUTPUT_DIR / "WaterDepth_vs_X.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_wd); gc.collect()
    print("  Saved: WaterDepth_vs_X.png")

# ─────────────────────────────────────────────
# TEMPORAL MEAN GRAPHS
# ─────────────────────────────────────────────

def safe_stat(mat):
    return (np.nanmean(mat, axis=0), np.nanstd(mat, axis=0),
            np.nanmin(mat, axis=0),  np.nanmax(mat, axis=0))

mean_score_t, std_score_t, min_score_t, max_score_t = safe_stat(score_di_mat)
mean_vegp_t,  std_vegp_t,  min_vegp_t,  max_vegp_t  = safe_stat(vegp_mat)
mean_vegpi_t, std_vegpi_t, min_vegpi_t, max_vegpi_t = safe_stat(vegpi_mat)
mean_dens_t,  std_dens_t,  min_dens_t,  max_dens_t  = safe_stat(dens_mat)

def add_stat_bands(ax, x, mean, std, vmin, vmax, color, label_mean):
    valid = ~np.isnan(mean)
    xv = x[valid]
    ax.fill_between(xv, vmin[valid], vmax[valid], color=color, alpha=0.15, label="Min-Max range")
    ax.fill_between(xv, np.maximum(mean[valid] - std[valid], 0), mean[valid] + std[valid],
                    color=color, alpha=0.35, label="Mean +/- std")
    ax.plot(xv, mean[valid], color=color, lw=2, marker="o", ms=3, label=label_mean)

if SHOW_FIG_MEAN_SCORE:
    print("\nGenerating temporal mean score graph...")
    fig_ms, ax_ms = plt.subplots(figsize=(12, 5))
    add_stat_bands(ax_ms, x_centers, mean_score_t, std_score_t, min_score_t, max_score_t,
                   "darkorange", f"Mean {SCORE_DI_NAME}")
    ax_ms.set_xlabel("X coordinate [m]", fontsize=11)
    ax_ms.set_ylabel(SCORE_DI_LABEL, fontsize=11)
    ax_ms.set_title(f"Temporal mean {SCORE_DI_LABEL} vs X\n"
                    "(mean +/- std + min/max envelope)", fontsize=12)
    apply_xlim(ax_ms); ax_ms.set_ylim(bottom=0)
    ax_ms.legend(fontsize=8, loc="upper left"); ax_ms.grid(axis="y", alpha=0.3)
    fig_ms.tight_layout()
    fig_ms.savefig(str(OUTPUT_DIR / "MeanScoreDI_vs_X.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_ms); gc.collect()
    print("  Saved: MeanScoreDI_vs_X.png")

if SHOW_FIG_MEAN_VEGPCT:
    print("Generating temporal mean veg% graph...")
    fig_mp, ax_mp = plt.subplots(figsize=(12, 5))
    add_stat_bands(ax_mp, x_centers, mean_vegp_t, std_vegp_t, min_vegp_t, max_vegp_t,
                   "black", "% veg / all cells (mean)")
    ax_mp.set_xlabel("X coordinate [m]", fontsize=11)
    ax_mp.set_ylabel("Vegetated cells [%]", fontsize=11)
    ax_mp.set_title("Temporal mean vegetation % vs X\n(mean +/- std + min/max envelope)", fontsize=12)
    apply_xlim(ax_mp); ax_mp.set_ylim(0, VEG_PCT_YMAX)
    ax_mp.legend(fontsize=8, loc="upper left"); ax_mp.grid(axis="y", alpha=0.3)
    fig_mp.tight_layout()
    fig_mp.savefig(str(OUTPUT_DIR / "MeanVegPct_vs_X.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_mp); gc.collect()
    print("  Saved: MeanVegPct_vs_X.png")

if SHOW_FIG_VEG_INTER:
    print("Generating veg% (relative to intertidal) graph (per timestep + mean)...")
    fig_vi, ax_vi = plt.subplots(figsize=(12, 5))
    # one thin curve per timestep, coloured by time
    for k, tsd in enumerate(all_timestep_data):
        arr   = tsd["veg_pct_inter_arr"]
        valid = ~np.isnan(arr)
        if valid.sum() == 0:
            continue
        ax_vi.plot(x_centers[valid], arr[valid], color=cmap_time(norm_time(k + 1)),
                   lw=1.0, alpha=0.7)
    # temporal mean on top
    vm = ~np.isnan(mean_vegpi_t)
    ax_vi.plot(x_centers[vm], mean_vegpi_t[vm], color="black", lw=2.3, zorder=6,
               label="Temporal mean")
    ax_vi.axhline(100, color="0.5", lw=1, ls=":", zorder=1, label="100% reference")
    sm_vi = plt.cm.ScalarMappable(cmap=cmap_time, norm=norm_time); sm_vi.set_array([])
    cbar_vi = fig_vi.colorbar(sm_vi, ax=ax_vi, pad=0.02)
    cbar_vi.set_label("Timestep", fontsize=9)
    set_time_ticks(cbar_vi, ts_labels)
    ax_vi.set_xlabel("X coordinate [m]", fontsize=11)
    ax_vi.set_ylabel("Vegetated cells / intertidal cells [%]", fontsize=11)
    ax_vi.set_title("Vegetation relative to the intertidal (colonizable) zone vs X\n"
                    "(all vegetated cells / intertidal cells per cross-section - per timestep + mean)",
                    fontsize=12)
    apply_xlim(ax_vi); ax_vi.set_ylim(bottom=0)
    ax_vi.legend(fontsize=8, loc="upper left"); ax_vi.grid(axis="y", alpha=0.3)
    fig_vi.tight_layout()
    fig_vi.savefig(str(OUTPUT_DIR / "MeanVegPct_Intertidal_vs_X.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_vi); gc.collect()
    print("  Saved: MeanVegPct_Intertidal_vs_X.png")

if SHOW_FIG_MEAN_DENSITY:
    print("Generating temporal mean density graph...")
    fig_md, ax_md = plt.subplots(figsize=(12, 5))
    add_stat_bands(ax_md, x_centers, mean_dens_t, std_dens_t, min_dens_t, max_dens_t,
                   "forestgreen", "Mean density")
    ax_md.set_xlabel("X coordinate [m]", fontsize=11)
    ax_md.set_ylabel("Stem density [m^-2]", fontsize=11)
    ax_md.set_title("Temporal mean stem density vs X\n(mean +/- std + min/max envelope)", fontsize=12)
    apply_xlim(ax_md); ax_md.set_ylim(bottom=0)
    ax_md.legend(fontsize=8, loc="upper left"); ax_md.grid(axis="y", alpha=0.3)
    fig_md.tight_layout()
    fig_md.savefig(str(OUTPUT_DIR / "MeanDensity_vs_X.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_md); gc.collect()
    print("  Saved: MeanDensity_vs_X.png")

# ─────────────────────────────────────────────
# MORTALITY ALONG ESTUARY (one curve per available timestep)
# ─────────────────────────────────────────────
if SHOW_FIG_MORTALITY:
    print("\nGenerating mortality graphs...")
    active_mort_causes = [c for c, enabled in MORT_CAUSES.items() if enabled]
    # timesteps kept within the requested veg-year window
    mort_sel = [k for k, tsd in enumerate(all_timestep_data)
                if (MORT_YEAR_MIN is None or tsd["veg_year"] >= MORT_YEAR_MIN)
                and (MORT_YEAR_MAX is None or tsd["veg_year"] <= MORT_YEAR_MAX)]
    if not mort_sel:
        print("  No timesteps in the requested mortality year window "
              f"[{MORT_YEAR_MIN}, {MORT_YEAR_MAX}], skipping mortality graphs.")
        active_mort_causes = []
    else:
        yr_lo = all_timestep_data[mort_sel[0]]["veg_year"]
        yr_hi = all_timestep_data[mort_sel[-1]]["veg_year"]
        # local time normalisation over the shown window (uses the full colour range)
        norm_mort   = mcolors.Normalize(vmin=1, vmax=len(mort_sel))
        mort_labels = [ts_labels[k] for k in mort_sel]
        range_txt   = ("all years" if (MORT_YEAR_MIN is None and MORT_YEAR_MAX is None)
                       else f"years {yr_lo:.1f}-{yr_hi:.1f}")
        print(f"  Mortality year window: {range_txt} ({len(mort_sel)} timesteps)")
    for cause in active_mort_causes:
        fig_m, ax_m = plt.subplots(figsize=(12, 5))
        any_data = False
        for j, k in enumerate(mort_sel):
            mort_arr = all_timestep_data[k]["mortality"].get(cause, np.full(n_bins, np.nan))
            valid    = ~np.isnan(mort_arr)
            if valid.sum() == 0:
                continue
            any_data = True
            ax_m.plot(x_centers[valid], mort_arr[valid] * 100,
                      color=cmap_time(norm_mort(j + 1)), lw=1.2, alpha=0.8, marker="o", ms=2)
        if not any_data:
            plt.close(fig_m); continue
        sm_m = plt.cm.ScalarMappable(cmap=cmap_time, norm=norm_mort); sm_m.set_array([])
        cbar_m = fig_m.colorbar(sm_m, ax=ax_m, pad=0.02)
        cbar_m.set_label("Timestep", fontsize=9)
        set_time_ticks(cbar_m, mort_labels)
        cause_label = cause.replace("applied_mort_", "").replace("_", " ").title()
        ax_m.set_xlabel("X coordinate [m]", fontsize=11)
        ax_m.set_ylabel(f"Mean {cause_label} mortality [%]", fontsize=11)
        ax_m.set_title(f"Mortality along estuary - {cause_label}  ({range_txt})\n"
                       f"(mean over vegetated cells per x-bin, per timestep)", fontsize=12)
        apply_xlim(ax_m)
        if MORT_PCT_YMAX is not None:
            ax_m.set_ylim(0, MORT_PCT_YMAX)
        else:
            ax_m.set_ylim(bottom=0)
        ax_m.grid(axis="y", alpha=0.3)
        fig_m.tight_layout()
        fname_m = OUTPUT_DIR / f"Mortality_{cause_label.replace(' ', '')}_vs_X.png"
        fig_m.savefig(str(fname_m), dpi=150, bbox_inches="tight")
        plt.close(fig_m); gc.collect()
        print(f"  Saved: {fname_m.name}")

# ─────────────────────────────────────────────
# BED LEVEL EVOLUTION AT KEY CROSS-SECTIONS
# ─────────────────────────────────────────────

def get_single_row_cut(x_pos):
    in_poly_idx = np.where(in_polygon)[0]
    x_in        = x_coords[in_poly_idx]
    unique_x    = np.unique(x_in)
    nearest_x   = unique_x[np.argmin(np.abs(unique_x - x_pos))]
    selected    = in_poly_idx[x_in == nearest_x]
    print(f"    Cross-section at x={nearest_x:.1f}m ({len(selected)} cells)")
    return selected[np.argsort(y_coords[selected])]

def plot_bed_level_evol(kp_list, criterion_label, fname_prefix):
    if not kp_list:
        print(f"  No positions for {criterion_label}, skipping.")
        return
    for kp in kp_list:
        x_pos   = kp["x"]
        row_idx = get_single_row_cut(x_pos)
        y_cs_s  = y_coords[row_idx]
        if len(y_cs_s) == 0:
            print(f"  No cells near x={x_pos:.0f}m, skipping."); continue
        fig2, ax2 = plt.subplots(figsize=(13, 5))
        for k, tsd in enumerate(all_timestep_data):
            color   = cmap_time(norm_time(k + 1))
            bl_cs   = tsd["mor_bl"][row_idx]
            ax2.plot(y_cs_s, bl_cs, color=color, lw=1.3, alpha=0.85)
            frac_cs = tsd["total_fraction"][row_idx]
            dens_cs = tsd["stem_density"][row_idx]
            veg_m   = frac_cs >= FRACTION_THRESHOLD
            if veg_m.sum() > 0:
                ax2.scatter(y_cs_s[veg_m], bl_cs[veg_m], c=dens_cs[veg_m],
                            cmap=CMAP_DENSITY, vmin=DENSITY_VMIN, vmax=DENSITY_VMAX,
                            s=20, zorder=5, alpha=0.7)
        ax2.axhline(0, color="steelblue", lw=1, ls="--", label="MSL = 0m")
        ax2.set_xlabel("Y coordinate [m]", fontsize=11)
        ax2.set_ylabel("Bed level [m]", fontsize=11)
        if criterion_label == "score":
            crit_str = SCORE_DI_LABEL
            val_str  = f"max {SCORE_DI_NAME}={kp['max_score']:.1f}"
        else:
            crit_str = criterion_label
            val_str  = f"max {criterion_label}={kp['max_val']:.1f}"
        ax2.set_title(f"Bed level evolution - x={x_pos:.0f}m\n"
                      f"Top-{kp['rank']} by {crit_str} ({val_str})", fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        # Two colorbars, well separated: timestep (long labels) is placed OUTER (rightmost),
        # density (short labels) INNER. Order of the colorbar() calls sets this: first call = outer.
        sm_t = plt.cm.ScalarMappable(cmap=cmap_time, norm=norm_time); sm_t.set_array([])
        cbar_t = fig2.colorbar(sm_t, ax=ax2, pad=0.10, fraction=0.035)
        cbar_t.set_label("Timestep", fontsize=9)
        set_time_ticks(cbar_t, ts_labels)
        sm_d = plt.cm.ScalarMappable(cmap=CMAP_DENSITY,
                                     norm=mcolors.Normalize(vmin=DENSITY_VMIN, vmax=DENSITY_VMAX))
        sm_d.set_array([])
        cbar_d = fig2.colorbar(sm_d, ax=ax2, pad=0.02, fraction=0.035)
        cbar_d.set_label("Density [m^-2]", fontsize=9)
        fname = OUTPUT_DIR / f"{fname_prefix}_rank{kp['rank']}_x{int(x_pos)}.png"
        fig2.savefig(str(fname), dpi=150, bbox_inches="tight")
        plt.close(fig2); gc.collect()
        print(f"  Saved: {fname.name} ({len(y_cs_s)} cells)")

if SHOW_FIG_BEDLEVEL:
    print(f"\nBed level evolution - top {SCORE_DI_NAME} positions:")
    plot_bed_level_evol(key_score_di, "score", "BedLevel_ScoreDI")
    print("Bed level evolution - top density positions (new only):")
    plot_bed_level_evol(key_dens, "density", "BedLevel_Density")
    print("Bed level evolution - top veg% positions (new only):")
    plot_bed_level_evol(key_vegp, "veg%", "BedLevel_VegPct")

# ─────────────────────────────────────────────
# MAP : best veg x-position per timestep
# ─────────────────────────────────────────────
if SHOW_FIG_MAP_BESTVEG:
    print("\nGenerating map: best veg position per timestep...")
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    draw_bathy(ax3)
    norm_ts = mcolors.Normalize(vmin=1, vmax=len(best_x_per_ts))
    for k, rec in enumerate(best_x_per_ts):
        color = cmap_time(norm_ts(k + 1))
        ax3.axvline(rec["x"], color=color, lw=1.2, alpha=0.5, zorder=3)
        ax3.scatter(rec["x"], mid_y, color=color, s=100, zorder=5, edgecolors="black", lw=0.5)
        ax3.text(rec["x"], mid_y + 400, str(k + 1), ha="center", va="bottom",
                 fontsize=7, fontweight="bold", color=color,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8, alpha=0.85))
    sm3 = plt.cm.ScalarMappable(cmap=cmap_time, norm=norm_ts); sm3.set_array([])
    cbar3 = fig3.colorbar(sm3, ax=ax3, pad=0.02)
    cbar3.set_label("Timestep", fontsize=10)
    set_time_ticks(cbar3, [r["label"] for r in best_x_per_ts])
    ax3.set_xlabel("X [m]", fontsize=11); ax3.set_ylabel("Y [m]", fontsize=11)
    ax3.set_title(f"Best {SCORE_DI_NAME} x-position per timestep\n"
                  f"({SCORE_DI_LABEL})", fontsize=12)
    fig3.tight_layout()
    fig3.savefig(str(OUTPUT_DIR / "Map_BestVegPerTimestep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3); gc.collect()
    print("  Saved: Map_BestVegPerTimestep.png")

# ─────────────────────────────────────────────
# MAP : all key cross-section positions
# ─────────────────────────────────────────────
if SHOW_FIG_MAP_KEYCROSS:
    print("Generating map: all key cross-section positions...")
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    draw_bathy(ax4)
    # (positions, letter, marker, colour, legend label)
    groups = [
        (key_score_di,  "S", "o", plt.colormaps["Reds"](0.7),    f"Top by {SCORE_DI_LABEL}"),
        (key_score_vd,  "P", "X", plt.colormaps["Oranges"](0.8), f"Top by {SCORE_VD_LABEL}"),
        (key_dens,       "D", "s", plt.colormaps["Blues"](0.7),   "Top by density"),
        (key_vegp,       "V", "^", plt.colormaps["Greens"](0.7),  "Top by veg%"),
        (key_intertidal, "I", "D", plt.colormaps["Purples"](0.75), "Top by intertidal area"),
    ]
    legend_handles = []
    for group, letter, marker, gcolor, leg in groups:
        for kp in group:
            x_pos = kp["x"]
            ax4.axvline(x_pos, color=gcolor, lw=1.8, alpha=0.7, zorder=3)
            ax4.scatter(x_pos, mid_y, color=gcolor, s=120, marker=marker,
                        zorder=5, edgecolors="black", lw=0.8)
            ax4.text(x_pos, mid_y + 400, f"{letter}{kp['rank']}", ha="center", va="bottom",
                     fontsize=8, fontweight="bold", color=gcolor,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=gcolor, lw=1.0, alpha=0.9))
        if group:
            legend_handles.append(plt.Line2D([0], [0], color=gcolor, lw=2, marker=marker, ms=7,
                                             label=f"{leg} ({len(group)} pos.)"))
    if legend_handles:
        ax4.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax4.set_xlabel("X [m]", fontsize=11); ax4.set_ylabel("Y [m]", fontsize=11)
    ax4.set_title("Key cross-section positions for bed level analysis\n"
                  f"S={SCORE_DI_LABEL}, P={SCORE_VD_LABEL}, D=density, V=veg%, I=intertidal area",
                  fontsize=10)
    fig4.tight_layout()
    fig4.savefig(str(OUTPUT_DIR / "Map_KeyCrossSections.png"), dpi=150, bbox_inches="tight")
    plt.close(fig4); gc.collect()
    print("  Saved: Map_KeyCrossSections.png")

# ─────────────────────────────────────────────
# ANIMATIONS
# ─────────────────────────────────────────────
if CREATE_ANIMATIONS:
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    print("\nCreating animations...")
    for key, paths in anim_paths.items():
        if len(paths) < 2:
            print(f"  Skipping {key}: only {len(paths)} frame(s)"); continue
        try:
            images   = [imageio.imread(p) for p in paths]
            out_path = OUTPUT_DIR / f"Anim_{key}.{ANIMATION_FORMAT}"
            imageio.mimsave(str(out_path), images, fps=ANIMATION_FPS, loop=0)
            print(f"  Saved: {out_path.name} ({len(images)} frames @ {ANIMATION_FPS}fps)")
        except Exception as e:
            print(f"  ERROR for {key}: {e}")

print("\nDone!")