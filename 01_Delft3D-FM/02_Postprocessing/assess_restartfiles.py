""""Assess restartfiles and corresponding final date in seconds, for adjusting mdu"""

#%%

import xarray as xr
import pandas as pd
import os
from FUNCTIONS.F_cache import DatasetCache

#%%
# Path to just ONE of your restart files
model_location = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC\MF1_sens"
single_file = os.path.join(model_location, 'FlowFM_0000_20541225_000000_rst.nc')

dataset_cache = DatasetCache()
try:
    # Open a single file without decoding anything
    ds = dataset_cache.get_xr(single_file, decode_times=False)
    
    # 1. Get the raw numeric time
    t_start_seconds = ds.time.values[0]
    
    # 2. Get the Reference Date
    # Look for units attribute, usually "seconds since YYYY-MM-DD HH:MM:SS"
    time_units = ds.time.attrs.get('units', '')
    
    print(f"--- Timing Data Found ---")
    print(f"Raw Tstart (for MDU): {t_start_seconds}")
    print(f"Time Units in file: {time_units}")
    
    # 3. Double-check the human-readable date
    if 'since' in time_units:
        ref_date_str = time_units.split('since ')[1]
        readable_time = pd.to_datetime(ref_date_str) + pd.to_timedelta(t_start_seconds, unit='s')
        print(f"Confirmed Date: {readable_time}")
    
except Exception as e:
    print(f"Failed to read file: {e}")
finally:
    dataset_cache.close_all()