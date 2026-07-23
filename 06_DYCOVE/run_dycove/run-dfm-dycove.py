# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:16:38 2026

@author: dreve
"""

#------------------------------------------------------------------------------
# Import necessary modules
#------------------------------------------------------------------------------
#%%
import os
from pathlib import Path

# _dfm_root = Path("C:/Program Files/Deltares/Delft3D FM Suite 2026.01 HM/plugins/DeltaShell.Dimr/kernels/x64")

# for _sub in ("bin", "share\\bin", "lib"):
#     _d = _dfm_root / _sub
#     if _d.is_dir():
#         os.add_dll_directory(str(_d))
#         os.environ["PATH"] = str(_d) + os.pathsep + os.environ["PATH"]
#         print(f"Added DLL dir: {_d}")

from dycove import VegetationSpecies
from dycove import DFM_hydro as hydro


from datetime import datetime

def read_sim_time_from_mdu(mdu_file, ecofac=None):
    """Extrait sim_time et time_unit depuis un fichier .mdu (format StartDateTime/StopDateTime)"""
    mdu = Path(mdu_file)
    params = {}
    with open(mdu) as f:
        for line in f:
            line = line.strip()
            if '=' not in line or line.startswith('#'):
                continue
            key, _, value = line.partition('=')
            key   = key.strip().lower()
            value = value.split('#')[0].strip()
            if key == 'startdatetime':
                params['StartDateTime'] = value
            elif key == 'stopdatetime':
                params['StopDateTime'] = value

    if 'StopDateTime' not in params:
        raise ValueError(f"'StopDateTime' introuvable dans {mdu_file}")

    fmt = "%Y%m%d%H%M%S"
    t_start = datetime.strptime(params.get('StartDateTime', '20000101000000'), fmt)
    t_stop  = datetime.strptime(params['StopDateTime'], fmt)

    duration_days = (t_stop - t_start).total_seconds() / 3600 / 24
    print(f"Start : {t_start}  →  Stop : {t_stop}")
    print(f"Durée : {duration_days:.2f} jours hydro")

    if ecofac:
        duration_eco = (duration_days * 24 * ecofac) / (365 * 24)
        print(f"Durée : {duration_eco:.4f} ans éco-morpho (ecofac={ecofac})")
        return duration_eco, "eco-morphodynamic years"
    else:
        return duration_days, "hydrodynamic days"

    
    
    
dfm_path = Path("C:\Program Files\Deltares\Delft3D FM Suite 2024.03 HMWQ\plugins\DeltaShell.Dimr\kernels\x64")#Path("C:/Program Files/Deltares/Delft3D FM Suite 2026.01 HM/plugins/DeltaShell.Dimr/kernels/x64")
mdu_path = Path(r"c:\Users\marloesbonenka\DYCOVE\run-dfm-dycove\FlowFM\input_coldstart\FlowFM.mdu")
dimr_xml = Path(r"c:\Users\marloesbonenka\DYCOVE\run-dfm-dycove\dimr.xml")


sim_time, time_unit = read_sim_time_from_mdu(mdu_path) 
print(sim_time, time_unit)

veg = VegetationSpecies(r"c:\Users\marloesbonenka\DYCOVE\run-dfm-dycove\FlowFM\input\SpartinaAnglica.json")

os.chdir(mdu_path.parent)
model = hydro.DFM(
    dfm_path,
    dimr_xml,
    mdu_path,
    vegetation=veg
)


model.run_simulation(
    sim_time, 
    time_unit,     
    n_ets=4,
    veg_interval=43200
)
# %%
