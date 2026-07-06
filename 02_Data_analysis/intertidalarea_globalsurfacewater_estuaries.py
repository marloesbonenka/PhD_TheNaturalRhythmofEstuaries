"""
gee_westernscheldt_analysis.py

JRC Global Surface Water (Pekel et al., 2016) analysis for the Western
Scheldt:
  1. Time series of intertidal ("seasonal water") and channel/subtidal
     ("permanent water") area, 1984-2024 (v1.4 + v1.5 merged).
  2. A spatial map of the same classification, clipped to an OSM-derived
     estuary boundary.

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
SCALE = 30  # m, native Landsat/GSW resolution

# --- time series ---
AOI_BBOX = [3.35, 51.20, 4.28, 51.50]  # [minLon, minLat, maxLon, maxLat] -- rough, replace with a real polygon
TILE_SCALE = 4                          # bump up if reduceRegion times out
EXPORT_MODE = "interactive"             # "interactive" or "drive"
OUT_CSV = "westernscheldt_intertidal_timeseries.csv"

# --- map ---
BOUNDARY_CACHE = Path("westerschelde_osm.geojson")
CLASSIFICATION_YEARS = [2020, 2021, 2022, 2023, 2024]
MAP_CRS = "EPSG:32631"  # UTM 31N, covers the Western Scheldt

ee.Initialize(project=GCP_PROJECT)

#%%
# =============================================================================
# SHARED HELPERS
# =============================================================================

def load_aoi_from_bbox(bbox):
    return ee.Geometry.Rectangle(bbox)


def load_aoi_from_shapefile(path_to_shp):
    """
    Preferred over the bounding box: use your own estuary polygon (e.g. the
    outline you use to bound the Delft3D-FM domain, or an OSM/Rijkswaterstaat
    water polygon clipped to the estuary).
    """
    gdf = gpd.read_file(path_to_shp).to_crs(epsg=4326)
    return geemap.geopandas_to_ee(gdf).geometry()


def get_estuary_boundary():
    """
    Fetch the Westerschelde water-body polygon from OpenStreetMap (cached
    locally after the first call so you're not hitting Nominatim/Overpass
    every run). Always eyeball the result once -- OSM place geocoding can
    occasionally return a larger or offset feature than you expect.
    """
    if BOUNDARY_CACHE.exists():
        gdf = gpd.read_file(BOUNDARY_CACHE)
    else:
        import osmnx as ox
        gdf = ox.geocode_to_gdf("Westerschelde, Netherlands")
        gdf = gdf.to_crs(epsg=4326)
        gdf.to_file(BOUNDARY_CACHE, driver="GeoJSON")
        print(f"Saved boundary to {BOUNDARY_CACHE} -- check it in QGIS/geojson.io before trusting it.")
    return gdf


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
aoi_ts = load_aoi_from_bbox(AOI_BBOX)
# aoi_ts = load_aoi_from_shapefile("path/to/western_scheldt.shp")   # <- preferred

yearly = build_yearly_collection().filterBounds(aoi_ts)
fc = ee.FeatureCollection(
    yearly.map(lambda img: compute_areas_for_image(img, aoi_ts))
)

if EXPORT_MODE == "interactive":
    result = fc.getInfo()
    records = [f["properties"] for f in result["features"]]
    df = pd.DataFrame(records).sort_values("year").reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(df)
else:
    # More robust for a large AOI / long record: runs server-side as a
    # batch task instead of the ~5 min interactive compute limit.
    task = ee.batch.Export.table.toDrive(
        collection=fc,
        description="westernscheldt_intertidal_timeseries",
        fileFormat="CSV",
    )
    task.start()
    print("Export started -- check Earth Engine Tasks / your Google Drive.")

#%%
fig, ax = plt.subplots(2, 1, figsize=(8, 4.5))
fig.supylabel("Area [km$^2$]")

ax[0].plot(df["year"], df["intertidal_km2"], marker="o", label="Intertidal (seasonal water)")
ax[1].plot(df["year"], df["permanent_km2"], marker="o", label="Channel (permanent water)")
ax[0].set_title(
    "Western Scheldt water-class area, JRC Global Surface Water\n"
    "(Pekel et al., 2016; v1.4 + v1.5)"
)
ax[0].legend()
ax[0].grid(alpha=0.3)
ax[1].set_xlabel("Year")
ax[1].legend()
ax[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("westernscheldt_intertidal_timeseries.png", dpi=200)
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

gdf = get_estuary_boundary()
aoi_map = geemap.geopandas_to_ee(gdf).geometry()

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

fig, axes = plt.subplots(3, 1, figsize=(20, 12))
fig.suptitle(
    "Western Scheldt · JRC Global Surface Water (Pekel et al., 2016)\n"
    "Three complementary surface-water descriptors",
    fontsize=11,
)
extent  = [xmin, xmax, ymin, ymax]
gdf_utm = gdf.to_crs(epsg=32631)

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

im_rec = axes[2].imshow(rec_masked, cmap=cmap_rec, vmin=1, vmax=100, extent=extent, origin="upper")
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
    ax.set_xlabel("east [m], UTM 31N", fontsize=8)
axes[0].set_ylabel("north [m], UTM 31N", fontsize=8)

fig.tight_layout()
fig.savefig("westernscheldt_seasonality_recurrence_map.png", dpi=250, bbox_inches="tight")
plt.show()
# %%
