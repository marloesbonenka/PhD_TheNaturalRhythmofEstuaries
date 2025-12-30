#%%
import os
import netCDF4 #make sure your interpreter is the Appdata one, there netcdf4 is installed.

#%%
def tag_netcdf_with_crs(netcdf_path, epsg_code):
    """
    Manually injects the CRS metadata into the global attributes of a NetCDF file.
    This fixes the issue where Delft3D-FM 'forgets' the bathymetry.
    
    Parameters:
    - netcdf_path: Full path to the *_net.nc file.
    - epsg_code: The CRS string to assign (e.g., 'EPSG:3857').
    """

    if not os.path.exists(netcdf_path):
        print(f"Error: NetCDF file not found at {netcdf_path}")
        return

    try:
        # Open the file in append mode ('a') to modify attributes
        with netCDF4.Dataset(netcdf_path, 'a') as nc_file:
            # Set the crucial global attribute 'projection' (Standard in D-Flow FM)
            nc_file.setncattr('projection', epsg_code)
            
            # Setting a few other standard attributes for robustness
            nc_file.setncattr('Projection_Code', epsg_code)
            nc_file.setncattr('CoordinateSystem', epsg_code)
            
        print(f"✅ Successfully tagged NetCDF file with CRS: {epsg_code}")
        print("You can now reopen your Delft3D-FM project.")
    
    except Exception as e:
        print(f"An error occurred while writing NetCDF attributes: {e}")

# --- Call the function with your file path ---
netcdf_file_path = r"U:\PhDNaturalRhythmEstuaries\Models\Test_Models\FM_vs_FLOW\newgrid\Project2.dsproj_data\FlowFM\input\D3D4_grid_domain45x15_net.nc"
tag_netcdf_with_crs(netcdf_file_path, 'EPSG:3857')
# %%
