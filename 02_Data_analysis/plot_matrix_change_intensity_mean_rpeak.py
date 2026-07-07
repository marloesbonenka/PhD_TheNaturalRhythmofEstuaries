"""
plot_matrix_change_intensity_mean_rpeak.py

Creates a matrix of "Water Occurrence Change Intensity" maps (JRC GSW change_abs),
arranged so that:
  - rows    (y-axis, top → bottom) = sorted by mean annual discharge (high → low)
  - columns (x-axis, left → right) = sorted by R_peak = Qmax_annual / Mean (low → high)

Each subplot title shows the estuary name, Mean and R_peak values.
A shared colorbar represents change in water occurrence [%].

Data sources
  Discharge metrics  : Excel (columns 'Estuary', 'Mean', 'R_peak (Qmax_annual/Mean)')
  Spatial change maps: JRC Global Surface Water v1.4 – GlobalSurfaceWater (change_abs)
  AOI boundaries     : single shapefile with all estuary polygons (drawn in QGIS, WGS84)
"""
#%%
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#%%
# =============================================================================
# CONFIGURATION
# =============================================================================
GCP_PROJECT = "ee-marloesbonenkamp"
SCALE       = 30    # m, native Landsat/GSW resolution
TILE_SCALE  = 4     # bump up if reduceRegion times out

OUT_DIR = Path(r"u:\PhDNaturalRhythmEstuaries\Data\02_Estuaries_GlobalSurfaceWater")

# Shapefile with all estuary boundary polygons (WGS84, drawn in QGIS).
BOUNDARIES_SHP = Path(
    r"C:\Users\marloesbonenka\Nextcloud\GlobalData_Estuaries"
    r"\Estuaries_GlobalWaterSurface\estuary_polygons.shp"
)
NAME_COLUMN = "Name"   # column in shapefile that holds the estuary name

# CSV with discharge metrics
DISCHARGE_CSV = Path(
    r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_variability_WBMsed"
    r"\01_Analysis_smallselection_estuaries"
    r"\fluvial_sediment_flux_Qriver_metrics_annualmax.csv"
)

# Per-estuary config:
#   'mouth'    : (lat, lon) – used to auto-select UTM zone
#   'xls_name' : name as it appears in the Excel 'Estuary' column (omit if identical to key)
#   'shp_name' : name as it appears in the shapefile NAME_COLUMN (omit if identical to key)
ESTUARIES = {
    'Western Scheldt': dict(mouth=(51.42,   3.57)),
    'Yangon':          dict(mouth=(16.52,  96.29)),
    'Sittuang':        dict(mouth=(17.11,  96.94)),
    'Chao Phraya':     dict(mouth=(13.55, 100.59), xls_name='Chao Phra'),
    'Gironde':         dict(mouth=(45.58,  -1.05)),
    'Colorado (MX)':   dict(mouth=(31.83, -114.82)),
    'Cacipore':        dict(mouth=( 3.6,  -51.2)),
    'Suriname':        dict(mouth=( 5.84, -55.11)),
    'Demerara':        dict(mouth=( 6.79, -58.18)),
    'Guayas':          dict(mouth=(-2.55, -79.88)),
    'Humber':          dict(mouth=(53.62,  -0.11)),
    'Fly':             dict(mouth=(-8.62, 143.70)),
    'Ord':             dict(mouth=(-15.5,  128.35)),
    'Purna':           dict(mouth=(20.91,  72.78)),
    'Tapi':            dict(mouth=(21.15,  72.75)),
    'Kumbe':           dict(mouth=(-8.5,  140.5)),
    'Sokyosen':        dict(mouth=(40.5,  129.5)),
    'Sungai Merbau':   dict(mouth=( 1.5,  103.0)),
    'Taeryong':        dict(mouth=(39.5,  125.5)),
    'Cromary Firth':   dict(mouth=(57.7,   -4.2), xls_name='Cromary F'),
    'Firth of Tay':    dict(mouth=(56.45,  -2.75), xls_name='Firth of Ta'),
    # add more as you digitise them in QGIS
}

ee.Initialize(project=GCP_PROJECT)

# =============================================================================
# HELPERS
# =============================================================================

def utm_epsg_from_lonlat(lat, lon):
    """Return the EPSG code for the UTM zone covering (lat, lon)."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = 326 if lat >= 0 else 327   # 326xx = North, 327xx = South
    return hemisphere * 100 + zone


def estuary_slug(name):
    """Convert an estuary name to a filesystem-safe slug."""
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')


def load_aoi(path, name_col, name_filter):
    """Load an AOI polygon from shapefile, filter by name_filter."""
    gdf = gpd.read_file(path).to_crs(epsg=4326)
    gdf_filt = gdf[gdf[name_col] == name_filter].copy()
    if gdf_filt.empty:
        raise ValueError(
            f"No feature with {name_col}='{name_filter}' in {path}.\n"
            f"Available: {gdf[name_col].tolist()}"
        )
    return gdf_filt, geemap.geopandas_to_ee(gdf_filt).geometry()


def fetch_change_and_maxextent(aoi, map_crs):
    """
    Download 'change_abs' and 'max_extent' from JRC/GSW1_4/GlobalSurfaceWater
    clipped to aoi on a UTM grid. Returns (change_map, max_ext_map, extent).
    """
    aoi_utm = aoi.transform(map_crs, 1)
    coords  = aoi_utm.bounds(1, map_crs).getInfo()["coordinates"][0]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width  = int((xmax - xmin) / SCALE)
    height = int((ymax - ymin) / SCALE)

    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(aoi)

    grid = {
        "dimensions": {"width": width, "height": height},
        "affineTransform": {
            "scaleX": SCALE,  "shearX": 0, "translateX": xmin,
            "shearY": 0, "scaleY": -SCALE, "translateY": ymax,
        },
        "crsCode": map_crs,
    }

    def _fetch(band):
        return ee.data.computePixels(
            {"expression": gsw.select(band), "fileFormat": "NUMPY_NDARRAY", "grid": grid}
        )[band]

    return _fetch("change_abs"), _fetch("max_extent"), [xmin, xmax, ymin, ymax]

# =============================================================================
# LOAD DISCHARGE METRICS FROM CSV
# =============================================================================
df_xls = pd.read_csv(DISCHARGE_CSV, sep=';')
df_xls.columns = df_xls.columns.str.strip()

# Identify columns robustly (tolerant of minor name variations)
estuary_col = next((c for c in df_xls.columns if "estuary" in c.lower()), None)
mean_col    = next((c for c in df_xls.columns if c.lower() == "mean"), None)
rpeak_col   = next((c for c in df_xls.columns if "r_peak" in c.lower()), None)

if None in (estuary_col, mean_col, rpeak_col):
    raise ValueError(
        f"Could not find required columns in CSV. Found: {list(df_xls.columns)}\n"
        "Need: an 'Estuary' column, a 'Mean' column, and an 'R_peak' column."
    )

df_xls = df_xls.rename(columns={estuary_col: "Estuary", mean_col: "Mean", rpeak_col: "R_peak"})[["Estuary", "Mean", "R_peak"]]
df_xls["Estuary"] = df_xls["Estuary"].str.strip()

# Build a lookup dict: estuary_label → row
xls_lookup = {row["Estuary"]: row for _, row in df_xls.iterrows()}

# =============================================================================
# MATCH ESTUARIES → DOWNLOAD GEE DATA
# =============================================================================
Path(OUT_DIR).mkdir(exist_ok=True)

# Pre-load shapefile names so we can skip quickly without a GEE call
gdf_shp = gpd.read_file(BOUNDARIES_SHP).to_crs(epsg=4326)
available_shp_names = set(gdf_shp[NAME_COLUMN].tolist())

records = []   # one dict per successfully loaded estuary

for name, cfg in ESTUARIES.items():
    xls_name = cfg.get('xls_name', name)

    # Find matching row in Excel (exact first, then prefix/substring fallback)
    xls_row = xls_lookup.get(xls_name)
    if xls_row is None:
        candidates = [
            k for k in xls_lookup
            if xls_name.lower()[:6] in k.lower() or k.lower()[:6] in xls_name.lower()
        ]
        if candidates:
            xls_row = xls_lookup[candidates[0]]

    if xls_row is None:
        print(f"  Skipping '{name}': no matching row in Excel (tried '{xls_name}').")
        continue

    map_crs  = f"EPSG:{utm_epsg_from_lonlat(*cfg['mouth'])}"
    shp_name = cfg.get('shp_name', name)

    # Skip early if the estuary polygon is not in the shapefile
    if shp_name not in available_shp_names:
        print(f"  Skipping '{name}': '{shp_name}' not found in shapefile "
              f"(available: {sorted(available_shp_names)}).")
        continue

    def _to_float(v): return float(str(v).replace(',', '.'))
    print(f"Downloading: {name}  |  Mean={_to_float(xls_row['Mean']):.0f} m³/s  |  R_peak={_to_float(xls_row['R_peak']):.2f}")

    try:
        _, aoi = load_aoi(BOUNDARIES_SHP, NAME_COLUMN, shp_name)
        change_map, max_ext_map, extent = fetch_change_and_maxextent(aoi, map_crs)
        records.append(dict(
            name     = name,
            slug     = estuary_slug(name),
            mean_q   = _to_float(xls_row['Mean']),
            r_peak   = _to_float(xls_row['R_peak']),
            change_map  = change_map,
            max_ext_map = max_ext_map,
            extent   = extent,
            map_crs  = map_crs,
        ))
        print(f"  OK")
    except Exception as exc:
        print(f"  FAILED for '{name}': {exc}")

if not records:
    raise RuntimeError("No estuaries loaded. Check shapefile and Excel configuration.")

# =============================================================================
# SORT INTO MATRIX POSITIONS
# =============================================================================
# Rows: sorted by Mean discharge — high Mean at top (row index 0)
records_by_mean  = sorted(records, key=lambda r: r['mean_q'], reverse=True)
# Columns: sorted by R_peak — low R_peak at left (col index 0)
records_by_rpeak = sorted(records, key=lambda r: r['r_peak'])

row_rank = {r['name']: i for i, r in enumerate(records_by_mean)}
col_rank = {r['name']: i for i, r in enumerate(records_by_rpeak)}

n_rows = len(records)
n_cols = len(records)

# Tick labels for outer axes (values in sorted order)
mean_labels  = [f"{r['mean_q']:.0f}" for r in records_by_mean]
rpeak_labels = [f"{r['r_peak']:.2f}" for r in records_by_rpeak]

# =============================================================================
# BUILD FIGURE
# =============================================================================
SUB_SIZE = 5.0   # inches per subplot cell
FIG_W = SUB_SIZE * n_cols + 2.0   # +2.0 for colorbar
FIG_H = SUB_SIZE * n_rows + 1.5

fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.suptitle(
    "Water Occurrence Change Intensity  [JRC GSW, 1984–1999 → 2000–2021]\n"
    "Matrix: rows = mean annual discharge (↑ high), columns = R_peak (→ high variability)",
    fontsize=14, y=1.01,
)

gs = gridspec.GridSpec(
    n_rows, n_cols,
    figure=fig,
    hspace=0.35,
    wspace=0.25,
    left=0.06, right=0.90,
    top=0.96, bottom=0.06,
)

for rec in records:
    row = row_rank[rec['name']]
    col = col_rank[rec['name']]
    ax  = fig.add_subplot(gs[row, col])

    change_masked = np.ma.masked_where(
        rec['max_ext_map'] == 0, rec['change_map'].astype(float)
    )
    ax.imshow(
        change_masked, cmap="RdBu", vmin=-100, vmax=100,
        extent=rec['extent'], origin="upper",
    )

    ax.set_title(
        f"{rec['name']}\nMean = {rec['mean_q']:.0f} m³/s  |  R$_{{peak}}$ = {rec['r_peak']:.2f}",
        fontsize=12, pad=8, fontweight='bold',
    )
    ax.tick_params(labelbottom=False, labelleft=False, length=2)

# ── shared colorbar ──────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.70])
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=-100, vmax=100))
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("Change in occurrence [%]", fontsize=11)
cb.set_ticks([-100, -50, 0, 50, 100])
cb.set_ticklabels(
    ["-100%\n(more land)", "-50%", "0\n(no change)", "+50%", "+100%\n(more water)"],
    fontsize=9,
)

# ── outer axis direction labels ───────────────────────────────────────────────
fig.text(
    0.47, 0.02,
    "← low discharge variability     R$_{peak}$ = Q$_{max,annual}$ / Mean Q     high discharge variability →",
    ha='center', va='bottom', fontsize=10,
)
fig.text(
    0.01, 0.50,
    "← low     Mean annual discharge [m³/s]     high →",
    ha='left', va='center', rotation='vertical', fontsize=10,
)

# =============================================================================
# SAVE
# =============================================================================
out_path = OUT_DIR / "matrix_change_intensity_mean_rpeak.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
#%%
# =============================================================================
# INDIVIDUAL PLOTS
# =============================================================================
ind_dir = OUT_DIR / "individual"
ind_dir.mkdir(exist_ok=True)

for rec in records:
    fig_i, ax_i = plt.subplots(figsize=(8, 6), constrained_layout=True)
    change_masked = np.ma.masked_where(
        rec['max_ext_map'] == 0, rec['change_map'].astype(float)
    )
    im = ax_i.imshow(change_masked, cmap="RdBu", vmin=-100, vmax=100, origin="upper")
    ax_i.set_title(
        f"{rec['name']}\nMean = {rec['mean_q']:.0f} m³/s  |  R_peak = {rec['r_peak']:.2f}",
        fontsize=12,
    )
    ax_i.tick_params(labelbottom=False, labelleft=False, length=2)
    cb_i = fig_i.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
    cb_i.set_label("Change in occurrence [%]", fontsize=10)
    cb_i.set_ticks([-100, -50, 0, 50, 100])
    cb_i.set_ticklabels(["-100%\n(more land)", "-50%", "0\n(no change)", "+50%", "+100%\n(more water)"], fontsize=8)
    ind_path = ind_dir / f"{rec['slug']}_change_intensity.png"
    fig_i.savefig(ind_path, dpi=200, bbox_inches="tight")
    # plt.close(fig_i)
    # print(f"Saved individual: {ind_path}")
# %%
