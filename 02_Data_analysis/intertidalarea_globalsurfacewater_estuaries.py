"""
gee_intertidal_analysis.py

JRC Global Surface Water (Pekel et al., 2016) analysis for multiple estuaries:
  1. Time series of intertidal ("seasonal water") and channel/subtidal
     ("permanent water") area, 1984-2024 (v1.4 + v1.5 merged).
  2. A spatial map of the water classification, seasonality and recurrence,
     clipped to a manually drawn estuary boundary (shapefile from QGIS).

Workflow:
  1. Draw the estuary boundary polygon in QGIS.
  2. Export as a shapefile to  boundaries/<slug>.shp
     (slug = estuary name lowercased, spaces -> underscores).
  3. Set ACTIVE_ESTUARY below and run.

Run as VS Code interactive cells (#%%) or top to bottom as a plain script.
"""
#%%
from pathlib import Path

import ee
import geemap
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# =============================================================================
# CONFIGURATION
# =============================================================================
GCP_PROJECT = "ee-marloesbonenkamp"
SCALE = 30          # m, native Landsat/GSW resolution
TILE_SCALE = 4      # bump up if reduceRegion times out
EXPORT_MODE = "interactive"   # "interactive" or "drive"
CLASSIFICATION_YEARS = [2020, 2021, 2022, 2023, 2024]

# Single shapefile containing all estuary polygons (drawn in QGIS, WGS84).
BOUNDARIES_SHP = Path(r"C:\Users\marloesbonenka\Nextcloud\GlobalData_Estuaries\Estuaries_GlobalWaterSurface\estuary_polygons.shp")

# Column in the shapefile that holds the estuary name.
# Used to select the correct polygon for ACTIVE_ESTUARY.
# Run `gdf.columns.tolist()` on the shapefile to find the right column name.
NAME_COLUMN = "Name"   # <-- adjust if your shapefile uses a different column

# Per-estuary configuration.
# 'mouth':    (lat, lon) of the estuary mouth -- used to auto-select UTM zone.
# 'shp_name': name as it appears in NAME_COLUMN (only needed if it differs
#             from the key used here).
ESTUARIES = {
    'Western Scheldt': dict(mouth=(51.42,   3.57)),
    # 'Yangon':          dict(mouth=(16.52,  96.29)),
    # 'Chao Phraya':     dict(mouth=(13.55, 100.59)),
    'Gironde':         dict(mouth=(45.58,  -1.05)),
    # 'Humber':          dict(mouth=(53.62,  -0.11)),
    # 'Fly':             dict(mouth=(-8.62, 143.70)),
    # 'Ord':             dict(mouth=(-15.5,  128.35)),
    # 'Demerara':        dict(mouth=( 6.79,  -58.18)),
    'Suriname':        dict(mouth=( 5.84,  -55.11)),
    # 'Guayas':          dict(mouth=(-2.55,  -79.88)),
    # 'Purna':           dict(mouth=(20.91,   72.78)),
    # 'Tapi':            dict(mouth=(21.15,   72.75)),
    # add more as you digitise them in QGIS
}

# --- SELECT WHICH ESTUARY TO RUN ---
ACTIVE_ESTUARY = 'Suriname'

ee.Initialize(project=GCP_PROJECT)

#%%
# =============================================================================
# DERIVED SETTINGS  (do not edit below unless needed)
# =============================================================================

def utm_epsg_from_lonlat(lat, lon):
    """Return the EPSG code for the UTM zone covering (lat, lon)."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = 326 if lat >= 0 else 327   # 326xx = North, 327xx = South
    return hemisphere * 100 + zone


def estuary_slug(name):
    """Convert an estuary name to a filesystem-safe slug."""
    return name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')


_cfg        = ESTUARIES[ACTIVE_ESTUARY]
MOUTH_LAT, MOUTH_LON = _cfg['mouth']
SHP_NAME    = _cfg.get('shp_name', ACTIVE_ESTUARY)  # name to match in NAME_COLUMN
MAP_CRS     = f"EPSG:{utm_epsg_from_lonlat(MOUTH_LAT, MOUTH_LON)}"
SLUG        = estuary_slug(ACTIVE_ESTUARY)
OUT_CSV     = Path("cache") / f"{SLUG}_intertidal_timeseries.csv"
Path("cache").mkdir(exist_ok=True)

print(f"Estuary  : {ACTIVE_ESTUARY}")
print(f"Shapefile: {BOUNDARIES_SHP}  (feature: '{SHP_NAME}' in column '{NAME_COLUMN}')")
print(f"Map CRS  : {MAP_CRS}")
print(f"CSV out  : {OUT_CSV}")

#%%
# =============================================================================
# SHARED HELPERS
# =============================================================================

def load_aoi_from_shapefile(path, name_col=None, name_filter=None):
    """
    Load an estuary boundary polygon from a shapefile (WGS84 or any CRS).
    If name_col and name_filter are given, selects only the matching feature.
    Returns (gdf, ee_geometry) so the GeoDataFrame is available for plotting.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {path}"
        )
    gdf = gpd.read_file(path).to_crs(epsg=4326)  # no-op if already WGS84
    if name_col and name_filter:
        gdf = gdf[gdf[name_col] == name_filter].copy()
        if gdf.empty:
            available = gpd.read_file(path)[name_col].tolist()
            raise ValueError(
                f"No feature with {name_col}='{name_filter}' found in {path}.\n"
                f"Available values: {available}"
            )
    return gdf, geemap.geopandas_to_ee(gdf).geometry()


def build_yearly_collection():
    """
    JRC Global Surface Water, YearlyHistory. waterClass: 0 = no data,
    1 = not water, 2 = seasonal water, 3 = permanent water. v1.4 covers
    1984-2021 (the original Pekel et al. 2016 product); v1.5 extends this
    to 2022-2024 using Landsat Collection 2. Merge both for 1984-2024.
    """
    v14 = ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
    v15 = ee.ImageCollection(
        "projects/global-surface-water/assets/GSW1_5/YearlyHistory"
    ).map(
        lambda img: img.select([0], ["waterClass"]).copyProperties(
            img, ["year", "system:time_start"]
        )
    )
    return v14.merge(v15).sort("year")


def compute_areas_for_image(img, aoi):
    """One reduceRegion call returning intertidal + permanent + valid-obs area [km2]."""
    wc = img.select("waterClass")
    pixel_area_km2 = ee.Image.pixelArea().divide(1e6)

    intertidal = wc.eq(2).multiply(pixel_area_km2).rename("intertidal_km2")
    permanent = wc.eq(3).multiply(pixel_area_km2).rename("permanent_km2")
    valid = wc.gt(0).multiply(pixel_area_km2).rename("valid_km2")

    combined = intertidal.addBands(permanent).addBands(valid)

    stats = combined.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=SCALE,
        maxPixels=1e10,
        tileScale=TILE_SCALE,
        bestEffort=True,
    )
    return ee.Feature(None, stats.set("year", img.get("year")))


#%%
# =============================================================================
# PART 1: INTERTIDAL / CHANNEL AREA TIME SERIES
# =============================================================================
gdf, aoi = load_aoi_from_shapefile(BOUNDARIES_SHP, name_col=NAME_COLUMN, name_filter=SHP_NAME)

yearly = build_yearly_collection().filterBounds(aoi)
fc = ee.FeatureCollection(
    yearly.map(lambda img: compute_areas_for_image(img, aoi))
)

if EXPORT_MODE == "interactive":
    result  = fc.getInfo()
    records = [f["properties"] for f in result["features"]]
    df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(df)
else:
    # More robust for a large AOI / long record: runs server-side as a
    # batch task instead of the ~5 min interactive compute limit.
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description=f"{SLUG}_intertidal_timeseries",
        fileFormat="CSV",
    )
    task.start()
    print("Export started -- check Earth Engine Tasks / your Google Drive.")

#%%
fig, ax = plt.subplots(2, 1, figsize=(8, 4.5))
fig.supylabel("Area [km$^2$]")

ax[0].plot(df["year"], df["intertidal_km2"], marker="o", label="Intertidal (seasonal water)")
ax[1].plot(df["year"], df["permanent_km2"],  marker="o", label="Channel (permanent water)")
ax[0].set_title(
    f"{ACTIVE_ESTUARY} water-class area, JRC Global Surface Water\n"
    "(Pekel et al., 2016; v1.4 + v1.5)"
)
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].set_xlabel("Year")
ax[1].legend()
ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"cache/{SLUG}_intertidal_timeseries.png", dpi=200)
plt.show()


# %%
# =============================================================================
# PART 2: WATER SEASONALITY & RECURRENCE MAPS
# =============================================================================
# Two complementary products from the JRC GSW composite (v1.4):
#
#   Seasonality  (INTRA-annual) – separates 'permanent' water (present in all
#     12 months of a year) from 'seasonal' water (present in only some months).
#     The value (0–12) is the number of months per year that water was detected.
#
#   Recurrence   (INTER-annual) – documents how frequently water returns from
#     one year to another, expressed as a percentage (0–100 %).  Low values
#     (orange) mark episodic inundation; high values (light-blue) mark regularly
#     and predictably inundated surfaces.
#
# Seasonality describes variability *within* a year; recurrence describes
# variability *across* years.  A pixel can be seasonal-but-regular (e.g. a
# tidal flat flooded every summer) or permanent-but-episodic (standing water
# only in wet years).

aoi_map = aoi  # already loaded from the QGIS shapefile above

yearly_map = build_yearly_collection().filter(
    ee.Filter.inList("year", CLASSIFICATION_YEARS)
)
mode_class = yearly_map.select("waterClass").mode().clip(aoi_map).rename("waterClass")

aoi_utm = aoi_map.transform(MAP_CRS, 1)
coords = aoi_utm.bounds(1, MAP_CRS).getInfo()["coordinates"][0]
xs = [pt[0] for pt in coords]
ys = [pt[1] for pt in coords]
xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)
width = int((xmax - xmin) / SCALE)
height = int((ymax - ymin) / SCALE)

params = {
    "expression": mode_class,
    "fileFormat": "NUMPY_NDARRAY",
    "grid": {
        "dimensions": {"width": width, "height": height},
        "affineTransform": {
            "scaleX": SCALE,
            "shearX": 0,
            "translateX": xmin,
            "shearY": 0,
            "scaleY": -SCALE,
            "translateY": ymax,
        },
        "crsCode": MAP_CRS,
    },
}
array = ee.data.computePixels(params)
wc_map = array["waterClass"]  # structured array -> plain 2D array for this band

gsw_composite = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(aoi_map)

def _fetch_gsw_band(band_name):
    """Download one GSW composite band on the same grid as the classification map."""
    p = {
        "expression": gsw_composite.select(band_name),
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": width, "height": height},
            "affineTransform": {
                "scaleX": SCALE,
                "shearX": 0,
                "translateX": xmin,
                "shearY": 0,
                "scaleY": -SCALE,
                "translateY": ymax,
            },
            "crsCode": MAP_CRS,
        },
    }
    return ee.data.computePixels(p)[band_name]


seas_map = _fetch_gsw_band("seasonality")   # 0–12 months/year
rec_map  = _fetch_gsw_band("recurrence")    # 0–100 %

# %%
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

epsg_code = int(MAP_CRS.split(":")[1])
gdf_utm = gdf.to_crs(epsg=epsg_code)

fig, axes = plt.subplots(3, 1, figsize=(20, 12))
fig.suptitle(
    f"{ACTIVE_ESTUARY} · JRC Global Surface Water (Pekel et al., 2016)\n"
    "Three complementary surface-water descriptors",
    fontsize=11,
)
extent = [xmin, xmax, ymin, ymax]

# ── panel 1 · YearlyHistory mode classification ────────────────────────────
cmap_cls = ListedColormap(["white", "#e8e4d8", "#8B5A2B", "#1f77b4"])
norm_cls = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_cls.N)
axes[0].imshow(wc_map, cmap=cmap_cls, norm=norm_cls, extent=extent, origin="upper")
gdf_utm.boundary.plot(ax=axes[0], edgecolor="black", linewidth=0.8)
axes[0].legend(
    handles=[
        Patch(facecolor="#8B5A2B", edgecolor="k", label="Seasonal water"),
        Patch(facecolor="#1f77b4", edgecolor="k", label="Permanent water"),
        Patch(facecolor="#e8e4d8", edgecolor="k", label="Land"),
    ],
    loc="lower right", fontsize=8,
)
axes[0].set_title(
    f"Water Classification\n"
    f"(YearlyHistory mode, {min(CLASSIFICATION_YEARS)}–{max(CLASSIFICATION_YEARS)})",
    fontsize=9,
)

# ── panel 2 · Water Seasonality (INTRA-annual) ─────────────────────────────
# Value = number of months per year that water was observed.
# 0 → not water / no data (masked as white); 1–11 → seasonal; 12 → permanent.
seas_masked = np.ma.masked_equal(seas_map, 0)
im_seas = axes[1].imshow(
    seas_masked, cmap="Blues", vmin=1, vmax=12,
    extent=extent, origin="upper",
)
gdf_utm.boundary.plot(ax=axes[1], edgecolor="black", linewidth=0.8)

# Use make_axes_locatable correctly:
divider2 = make_axes_locatable(axes[1])
cax2 = divider2.append_axes("right", size="3%", pad=0.1)
cb2 = fig.colorbar(im_seas, cax=cax2) # Attach to cax2, NOT axes[1]

# Now set your labels on the correct colorbar object
cb2.set_label("Months / year with water present", fontsize=8)
cb2.set_ticks([1, 4, 8, 12])
cb2.set_ticklabels(["1\n(seasonal)", "4", "8", "12\n(permanent)"], fontsize=7)

axes[1].set_title(
    "water seasonality  [intra-annual]: permanent (12 months/yr) vs seasonal (<12 months/yr) within a single year period",
    fontsize=9,
)

# ── panel 3 · Water Recurrence (INTER-annual) ──────────────────────────────
# Value = percentage of years in the record in which water was detected.
# Orange → episodic (rarely returns); light-blue → regular / predictable.
# 0 → not water / no data (masked as white).
cmap_rec = mcolors.LinearSegmentedColormap.from_list(
    "recurrence", ["#f97b22", "#ffffcc", "#5bc8f5"], N=256
)
cmap_rec.set_under("white")
rec_masked = np.ma.masked_equal(rec_map, 0)
im_rec = axes[2].imshow(
    rec_masked, cmap=cmap_rec, vmin=1, vmax=100,
    extent=extent, origin="upper",
)
gdf_utm.boundary.plot(ax=axes[2], edgecolor="black", linewidth=0.8)

divider3 = make_axes_locatable(axes[2])
cax3 = divider3.append_axes("right", size="3%", pad=0.1)
cb3 = fig.colorbar(im_rec, cax=cax3) # Attach to cax3, NOT axes[2]

cb3.set_label("Water recurrence [%]", fontsize=8)
cb3.set_ticks([1, 25, 50, 75, 100])
cb3.set_ticklabels(["1%\n(episodic)", "25%", "50%", "75%", "100%\n(regular)"], fontsize=7)

axes[2].set_title(
    "water recurrence [inter-annual]: how often water returns year-to-year (%)",
    fontsize=9,
)

for ax in axes:
    ax.set_xlabel(f"East [m], {MAP_CRS}", fontsize=8)
axes[0].set_ylabel(f"North [m], {MAP_CRS}", fontsize=8)

fig.tight_layout()
fig.savefig(f"cache/{SLUG}_seasonality_recurrence_map.png", dpi=250, bbox_inches="tight")
plt.show()
# %%
