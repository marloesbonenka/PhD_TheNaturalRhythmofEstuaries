import numpy as np
import pandas as pd
import os
import re
#%%
def check_available_variables_xarray(ds):
    """Updated for xarray/dfm_tools datasets"""
    print("Available variables in dataset:\n")
    # xarray uses ds.data_vars for the main variables
    for var_name in sorted(ds.data_vars):
        var = ds[var_name]
        print(f"  {var_name}:")
        print(f"    shape         = {var.shape}")
        print(f"    dimensions    = {var.dims}")
        
        # xarray stores metadata in the .attrs dictionary
        for attr in ['units', 'long_name', 'standard_name', 'description']:
            if attr in var.attrs:
                print(f"    {attr:13} = {var.attrs[attr]}")
        
        print("") 

    return {'all_vars': list(ds.data_vars)}

def get_mf_number(folder_name):
    match = re.search(r'MF_?(\d+)', folder_name)
    return int(match.group(1)) if match else 999