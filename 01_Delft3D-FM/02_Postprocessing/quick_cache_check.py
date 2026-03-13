""""Check the cache file for station names and water level values at the first time step, and compare with the original his file."""
#%%
from pathlib import Path
import re
import xarray as xr
import numpy as np
#%%
his_file_path = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q500\1_Q500_rst.9093769\output\FlowFM_0000_his.nc")
pattern = r'^Observation(?:Point|CrossSection)_Estuary_km(\d+)$'

#%% Check info in his file
with xr.open_dataset(his_file_path) as ds:
    raw_names = ds["station_name"].values
    names = []
    for row in np.asarray(raw_names):
        if np.asarray(raw_names).ndim == 2:
            s = "".join(chr(int(c)) if isinstance(c, (np.integer, int)) and 0 < int(c) < 256
                        else (c.decode("utf-8", "ignore") if isinstance(c, (bytes, np.bytes_)) else str(c))
                        for c in row)
        else:
            s = row.decode("utf-8", "ignore") if isinstance(row, (bytes, np.bytes_)) else str(row)
        s = s.replace("\x00", "").strip()
        names.append(s)

    matches = [(i, n, int(re.search(r'km(\d+)', n).group(1)))
               for i, n in enumerate(names) if re.match(pattern, n)]

    print(f"Total stations: {len(names)}")
    print(f"Matched estuary stations: {len(matches)}")
    print("First 10 matched names:", [m[1] for m in matches[:10]])

#%%
# Your cache file
cache_file = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q500\cached_data\hisoutput_stations_1_Q500_rst.9093769.nc")

# Use same tolerant pattern as discussed
pattern = r"^Observation(?:Point|CrossSection)_Estuary_km(\d+)$"

def decode_station_names(arr):
    arr = np.asarray(arr)
    names = []

    if arr.ndim == 2:
        for row in arr:
            chars = []
            for c in row:
                if isinstance(c, (bytes, np.bytes_)):
                    chars.append(c.decode("utf-8", errors="ignore"))
                elif isinstance(c, (np.integer, int)):
                    v = int(c)
                    if 0 < v < 256:
                        chars.append(chr(v))
                else:
                    chars.append(str(c))
            s = "".join(chars).replace("\x00", "").strip()
            s = "".join(ch for ch in s if ch.isprintable())
            names.append(s)
    else:
        for c in arr:
            if isinstance(c, (bytes, np.bytes_)):
                s = c.decode("utf-8", errors="ignore")
            else:
                s = str(c)
            s = s.replace("\x00", "").strip()
            s = "".join(ch for ch in s if ch.isprintable())
            names.append(s)

    return names


print("Cache path:", cache_file)
print("Exists:", cache_file.exists())

if not cache_file.exists():
    raise FileNotFoundError(cache_file)

with xr.open_dataset(cache_file) as ds:
    print("\n--- DATASET INFO ---")
    print("Data vars:", list(ds.data_vars))
    print("Coords:", list(ds.coords))
    print("Dims:", dict(ds.dims))

    if "station_name" not in ds:
        raise KeyError("station_name not in cache file")
    if "waterlevel" not in ds:
        raise KeyError("waterlevel not in cache file")

    names = decode_station_names(ds["station_name"].values)
    matches = []
    for i, n in enumerate(names):
        m = re.match(pattern, n)
        if m:
            matches.append((i, int(m.group(1)), n))

    print("\n--- STATIONS ---")
    print("Total stations in cache:", len(names))
    print("Matched estuary stations:", len(matches))
    print("First 10 station names:", names[:10])
    print("First 10 matched names:", [m[2] for m in matches[:10]])

    tname = "t" if "t" in ds else "time" if "time" in ds else None
    if tname is None:
        raise KeyError("Neither 't' nor 'time' found in cache")

    print("\n--- FIRST TIMESTEP ---")
    print("First timestamp:", ds[tname].values[0])

    wl0_all = ds["waterlevel"].isel(time=0).values
    print("waterlevel[time=0] shape (all stations):", wl0_all.shape)
    print("waterlevel[time=0] first 10 all stations:", wl0_all[:10])

    if matches:
        idx = np.array([m[0] for m in matches], dtype=int)
        wl0_sel = wl0_all[idx]
        km_sel = np.array([m[1] for m in matches], dtype=int)
        print("\nSelected estuary km range:", km_sel.min(), "to", km_sel.max())
        print("waterlevel[time=0] selected shape:", wl0_sel.shape)
        print("First 10 selected (km, wl):", list(zip(km_sel[:10], wl0_sel[:10])))
# %%
