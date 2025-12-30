#%%
import numpy as np
# import xarray
import netCDF4 #make sure your interpreter is the Appdata one, there netcdf4 is installed.

#%%
file_path = r"U:\PhDNaturalRhythmEstuaries\Models\Test_Models\FM_vs_FLOW\FM\Test4_D3Dgrid_xyzfile_longseabnd\D3D4_grid_domain45x15_net.nc"

# 1. Open the file
with netCDF4.Dataset(file_path, 'r') as nc_file:
    print(f"## 📁 Contents of: {file_path}")

    # 2. Access contents (Dimensions, Variables, Attributes)

    # Print Global Attributes
    print("\n### Global Attributes:")
    for attr_name in nc_file.ncattrs():
        print(f"- {attr_name}: {nc_file.getncattr(attr_name)}")

    # Print Dimensions
    print("\n### Dimensions:")
    for dim_name, dimension in nc_file.dimensions.items():
        print(f"- {dim_name}: {len(dimension)} (unlimited: {dimension.isunlimited()})")

    # Print Variables (Metadata and Data)
    print("\n### Variables:")
    for var_name, variable in nc_file.variables.items():
        print(f"- **{var_name}** ({variable.dtype})")
        print(f"  - Dimensions: {variable.dimensions}")
        print(f"  - Units: {getattr(variable, 'units', 'N/A')}")
        
        # 3. Read variable data (Example: read the first 10 values)
        if variable.size > 0:
            data = variable[:10]  # Read the first 10 elements
            print(f"  - First 10 values: {data}")
# %%
