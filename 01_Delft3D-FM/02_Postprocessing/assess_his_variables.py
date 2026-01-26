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
from FUNCTIONS.F_cache import *

# --- SETTINGS ---
# Point this to your specific MF50 his file example
his_file_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\02_seasonal\Tmorph_50years\MF50_sens.8778435\output\FlowFM_0000_his.nc")

print(f"Opening history file: {his_file_path.name}")

dataset_cache = DatasetCache()
try:
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

except Exception as e:
    print(f"Error opening his file: {e}")
finally:
    dataset_cache.close_all()