#%%
import os
from pathlib import Path
import datetime as dt_module
import ee
import geemap
import geemap.common as gc
import geemap.timelapse as tl

gc.datetime = dt_module
tl.datetime = dt_module

GCP_PROJECT = "ee-marloesbonenkamp"
ee.Initialize(project=GCP_PROJECT)

locations = [
    ("Tapi 01", [72.70, 21.03, 72.82, 21.08]),
    ("Tapi 02", [72.64, 21.06, 72.73, 21.16]),
]

network_dir = Path(r"\\tudelft.net\staff-umbrella\PhDNaturalRhythmEstuaries\Data\06_ee_tests")
network_dir.mkdir(parents=True, exist_ok=True)

for name, bbox in locations:
    region = ee.Geometry.BBox(*bbox)
    final_gif_path = str(network_dir / f"landsat_timelapse_{name.replace(' ', '_')}.gif")

    geemap.landsat_timelapse(
        roi=region,
        start_year=1984,
        end_year=2024,
        start_date="01-01",
        end_date="12-31",
        bands=["Red", "Green", "Blue"],
        vis_params={"min": 0.02, "max": 0.25, "gamma": 1.2},
        frames_per_second=2,
        out_gif=final_gif_path,
        title=f"Landsat Annual Timelapse - {name}",
    )

    if os.path.exists(final_gif_path):
        print(f"SUCCESS! Created timelapse at: {final_gif_path}")
    else:
        print(f"GIF not created for: {name}")
#%%