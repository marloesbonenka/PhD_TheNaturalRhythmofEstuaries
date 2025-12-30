import os
from netCDF4 import Dataset
import xarray as xr

BASE_PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'processed_data'))

def get_processed_path(dataset_type, scenario, discharge):
    return os.path.join(BASE_PROCESSED_DIR, dataset_type, f"{scenario}_{discharge}", f"{dataset_type}_saved.nc")

def open_dataset_cached(raw_path, dataset_type, scenario, discharge):
    processed_path = get_processed_path(dataset_type, scenario, discharge)
    if os.path.exists(processed_path):
        print(f"Loading cached {dataset_type} dataset from {processed_path}")
        return Dataset(processed_path, 'r')  # or use contextmanager if preferred
    else:
        print(f"No cached {dataset_type} found, opening raw dataset {raw_path}")
        return Dataset(raw_path, 'r')

def save_dataset_copy(src_path, dataset_type, scenario, discharge):
    save_dir = os.path.join(BASE_PROCESSED_DIR, dataset_type, f"{scenario}_{discharge}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_type}_saved.nc")

    with Dataset(src_path, 'r') as src, Dataset(save_path, 'w', format='NETCDF4_CLASSIC') as dst:
        # Copy global attributes
        dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
        # Copy dimensions
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)
        # Copy variables
        for name, variable in src.variables.items():
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            dst[name][:] = src[name][:]
            dst[name].setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})

    print(f"Saved {dataset_type} dataset copy to {save_path}")
    return save_path