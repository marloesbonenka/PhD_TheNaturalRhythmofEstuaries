""""
Assess Delft3D-FM his file variables for observation points and cross-sections.

"""

#%%
import os
import dfm_tools as dfmt
import xarray as xr
import sys
from pathlib import Path
#%%
# Add your custom functions path
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\Delft3D-FM\Postprocessing")
from FUNCTIONS.F_general import *
from FUNCTIONS.F_cache import DatasetCache

# --- SETTINGS ---
# Point this to your specific MF50 his file example
his_file_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q500\1_Q500_rst.9093769\output\FlowFM_0000_his.nc")
map_file_pattern = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q500\1_Q500_rst.9093769\output\*_map.nc")

print(f"Opening history file: {his_file_path.name}")

dataset_cache = DatasetCache()

# Since it's not partitioned, we use xr.open_dataset or dfmt.open_dataset
ds_his = dataset_cache.get_xr(his_file_path)

print("\n" + "="*50)
print("HISTORY FILE INSPECTION")
print("="*50)

# 1. Check for Observation Cross-Sections (Discharge)
if 'cross_section_name' in ds_his.coords or 'cross_section_name' in ds_his.data_vars:
    print("\n[✔] Found Cross-Sections:")
    # Decode bytes to strings if necessary
    names = ds_his['cross_section_name'].values
    if isinstance(names[0], bytes):
        names = [n.decode('utf-8').strip() for n in names]
    for name in names:
        print(f"  - {name}")
else:
    print("\n[!] No Cross-Sections found in this file.")

# 2. Check for Observation Points (Water level/Velocity)
if 'station_name' in ds_his.coords or 'station_name' in ds_his.data_vars:
    print("\n[✔] Found Observation Points (Stations):")
    st_names = ds_his['station_name'].values
    if isinstance(st_names[0], bytes):
        st_names = [n.decode('utf-8').strip() for n in st_names]
    print(f"  - Count: {len(st_names)}")
    print(f"  - Examples: {st_names[:3]} ... {st_names[-1:]}")

# 3. List all Data Variables (to find 'q1', 'waterlevel', etc.)
print("\n[i] Available Data Variables in HIS file:")
# Using your custom function if available, or standard print
for var in ds_his.data_vars:
    dims = ds_his[var].dims
    attrs = ds_his[var].attrs.get('long_name', 'No description')
    print(f"  - {var:25} dims: {str(dims):20} | {attrs}")

dataset_cache.close_all()

#%% --- MAP FILE INSPECTION ---
print("\n" + "="*50)
print("MAP FILE INSPECTION")
print("="*50)

ds_map = dfmt.open_partitioned_dataset(str(map_file_pattern), chunks={'time': 1})

# Coordinate ranges
face_x = ds_map['mesh2d_face_x'].values
face_y = ds_map['mesh2d_face_y'].values

print(f"\n[i] Mesh coordinate ranges:")
print(f"  - mesh2d_face_x: {face_x.min():.1f} to {face_x.max():.1f} m")
print(f"  - mesh2d_face_y: {face_y.min():.1f} to {face_y.max():.1f} m")
print(f"  - Number of faces: {len(face_x)}")

# Time info
print(f"\n[i] Time dimension:")
print(f"  - Number of timesteps: {len(ds_map.time)}")
print(f"  - First time: {ds_map.time.values[0]}")
print(f"  - Last time: {ds_map.time.values[-1]}")

# Bed level variable
if 'mesh2d_mor_bl' in ds_map.data_vars:
    bl = ds_map['mesh2d_mor_bl'].isel(time=0).values
    print(f"\n[i] Bed level (mesh2d_mor_bl) at t=0:")
    print(f"  - Min: {bl.min():.2f} m")
    print(f"  - Max: {bl.max():.2f} m")
    print(f"  - Mean: {bl.mean():.2f} m")

# List ALL data variables
print(f"\n[i] Available Data Variables in MAP file:")
for var in ds_map.data_vars:
    dims = ds_map[var].dims
    attrs = ds_map[var].attrs.get('long_name', 'No description')
    print(f"  - {var:40} dims: {str(dims):35} | {attrs}")

ds_map.close()