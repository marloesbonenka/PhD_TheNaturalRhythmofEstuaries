"""inspect mat file content"""

#%% --- IMPORTS ---
import h5py
import numpy as np

#%%
# --- Inspect GlobalDeltaData.mat ---
with h5py.File(r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash\Nienhuis2020_Scripts\GlobalDeltaData.mat", 'r') as f:
    print("=== GlobalDeltaData.mat variables ===")
    for key in f.keys():
        item = np.array(f[key]).flatten()
        if np.issubdtype(item.dtype, np.number):
            print(f"  {key}: shape={item.shape}, min={np.nanmin(item):.2f}, max={np.nanmax(item):.2f}")
        else:
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}  <-- non-numeric, skipping min/max")
#%%
# # --- Inspect qs_timeseries .mat ---
# with h5py.File(r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash\qs_timeseries_Nienhuis2020.mat", 'r') as f:
#     print("\n=== qs_timeseries_Nienhuis2020.mat variables ===")
#     for key in f.keys():
#         item = np.array(f[key]).flatten()
#         if np.issubdtype(item.dtype, np.number):
#             print(f"  {key}: shape={item.shape}, min={np.nanmin(item):.2f}, max={np.nanmax(item):.2f}")
#         else:
#             print(f"  {key}: shape={item.shape}, dtype={item.dtype}  <-- non-numeric, skipping min/max")