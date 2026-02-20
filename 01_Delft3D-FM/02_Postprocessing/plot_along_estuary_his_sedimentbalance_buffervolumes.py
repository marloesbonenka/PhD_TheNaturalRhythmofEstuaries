"""
Plot sediment buffer volumes along estuary.
Divide estuary in sections based on distance from sea to river,
extract sediment transport through cross sections at the boundaries of these sections,
plot the cumulative sediment transport (buffer volume) over time for each section, 
and also the instantaneous rate of change of the buffer volume to identify periods of rapid erosion or deposition.

"""
#%% IMPORTS 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
from pathlib import Path

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_loaddata import load_cross_section_data

#%% --- CONFIGURATION ---
# What to analyze?
var_name = 'cross_section_bedload_sediment_transport'
output_dirname = "plots_his_sedimentbuffer"

mpl.rcParams['figure.figsize'] = (8, 6)     

# Define cross-section km boundaries
box_edges = np.arange(20, 50, 5)  # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i+1]) for i in range(len(box_edges)-1)]

# Which scenarios to process (set to None or empty list for all)
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']  # Use all scenarios


#%% --- PATHS ---  
DISCHARGE = 500  # or 1000, etc.
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f'Model_Output/Q{DISCHARGE}'
base_path = base_directory / config
output_dir = base_path / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)
timed_out_dir = base_path / "timed-out"
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")
if not timed_out_dir.exists():
    raise FileNotFoundError(f"Timed-out directory not found: {timed_out_dir}")

# Mapping: restart folder prefix -> timed-out folder prefix
# 1 = constant (baserun), 2 = seasonal, 3 = flashy, 4 = singlepeak
VARIABILITY_MAP = {
    '1': f'01_baserun{DISCHARGE}',
    '2': f'02_run{DISCHARGE}_seasonal',
    '3': f'03_run{DISCHARGE}_flashy',
    '4': f'04_run{DISCHARGE}_singlepeak',
}

# Find restart folders: start with digit and contain "_rst"
model_folders = [f.name for f in base_path.iterdir() 
                    if f.is_dir() and f.name[0].isdigit() and '_rst' in f.name.lower()]
# Filter by SCENARIOS_TO_PROCESS if specified
if SCENARIOS_TO_PROCESS:
    model_folders = [f for f in model_folders if f.split('_')[0] in SCENARIOS_TO_PROCESS]
# Sort by leading number
model_folders.sort(key=lambda x: int(x.split('_')[0]))

print(f"Found {len(model_folders)} run folders in: {base_path}")

#%%
# --- 1. For each run, find only the correct timed-out and restart part ---
all_run_paths = []
run_his_paths = {}

for folder in model_folders:
    model_location = base_path / folder
    scenario_num = folder.split('_')[0]
    his_paths = []
    if scenario_num in VARIABILITY_MAP and timed_out_dir.exists():
        timed_out_folder = VARIABILITY_MAP[scenario_num]
        timed_out_path = timed_out_dir / timed_out_folder / "output" / "FlowFM_0000_his.nc"
        if timed_out_path.exists():
            his_paths.append(timed_out_path)
    # Always append the main/restart part
    main_his_path = model_location / "output" / "FlowFM_0000_his.nc"
    if main_his_path.exists():
        his_paths.append(main_his_path)
    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

print(f"Found {len(model_folders)} run folders in: {base_path}")

#%% --- LOOP OVER SCENARIOS AND PLOT ---
#%% --- LOAD AND STITCH ALL DATA FIRST ---
scenario_data = {}
for scenario_dir, his_file_paths in run_his_paths.items():
    print(f"\nLoading scenario: {scenario_dir}")
    data = load_cross_section_data(
        his_file_path=his_file_paths,
        q_var=var_name,
        estuary_only=True,
        km_range=(20, 45),
        select_cycles_hydrodynamic=False
    )
    scenario_data[scenario_dir] = data

#%% --- PLOT ALL SCENARIOS ---
for scenario_dir, data in scenario_data.items():

    km_positions = np.array(data['km_positions'])
    transport = data['discharge']
    time = data['t']
    buffer_volumes = {}
    for box_start, box_end in boxes:
        idx_up = np.argmin(np.abs(km_positions - box_start))
        idx_down = np.argmin(np.abs(km_positions - box_end))
        buffer_volumes[(box_start, box_end)] = transport[:, idx_up] - transport[:, idx_down]

    # --- CUMULATIVE BUFFER VOLUME PLOT ---
    plt.figure()
    for (box_start, box_end), buf in buffer_volumes.items():
        plt.plot(time, buf, label=f'{box_start}-{box_end} km')
    plt.xlabel('Time')
    plt.ylabel('Buffer Volume (m3)')
    plt.ylim(-0.1, 1.15e11)
    plt.legend()
    plt.title(f'Sediment buffer volumes per section for {scenario_dir}')
    plt.grid()
    plt.tight_layout()
    scenario_name = Path(scenario_dir).name
    fig1_path = Path(output_dir) / f"{scenario_name}_sediment_buffer_volume_cumulative.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"Saved cumulative buffer plot to {fig1_path}")
    plt.show()

    # --- INSTANTANEOUS RATE OF CHANGE PLOT ---
    plt.figure()
    all_d_buffer = []
    spinup_steps = 10  # Number of timesteps to skip for percentile calculation
    for (box_start, box_end), buf in buffer_volumes.items():
        d_buffer = np.diff(buf)
        all_d_buffer.append(d_buffer[spinup_steps:])  # Exclude spin-up
        plt.plot(time[1:], d_buffer, label=f'{box_start}-{box_end} km')
    all_d_buffer_flat = np.concatenate(all_d_buffer)
    plt.ylim(-0.25e8, 0.25e8)
    plt.xlabel('hydrodynamic time (x 100 = morphological time)')
    plt.ylabel('dV/dt (m3/timestep)')
    plt.legend()
    plt.title(f'Instantaneous buffer change per section for {scenario_dir}')
    plt.grid()
    plt.tight_layout()
    fig2_path = Path(output_dir) / f"{scenario_name}_sediment_buffer_volume_instantaneous.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"Saved instantaneous buffer plot to {fig2_path}")
    plt.show()

    # --- TRAPPING EFFICIENCY PLOT ---
    plt.figure(figsize=(12, 6))
    
    for (box_start, box_end), buf in buffer_volumes.items():
        idx_up = np.argmin(np.abs(km_positions - box_start))
        
        # 1. Change in storage during this timestep (dV)
        dV = np.diff(buf) 
        
        # 2. Amount of sediment entering the section during this timestep (dIn)
        # We take the diff of the cumulative transport at the upstream boundary
        dIn = np.diff(transport[:, idx_up])
        
        # 3. Efficiency = dV / dIn
        # Add a tiny epsilon to dIn to avoid division by zero during stagnant periods
        efficiency = dV / (dIn + 1e-9)
        
        # 4. Cleaning the data:
        # High-frequency tidal fluctuations will make this swing wildly.
        # We use a rolling mean (e.g., 24-hour or tidal cycle window)
        window = 25 # Adjust based on your timestep frequency
        eff_smooth = np.convolve(efficiency, np.ones(window)/window, mode='same')
        
        plt.plot(time[1:], eff_smooth, label=f'{box_start}-{box_end} km')

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.axhline(1, color='red', lw=1, ls=':', label='100% Trapping')
    plt.ylim(-0.5, 1.5) # Focus on the 0 to 1 range
    plt.ylabel('Instantaneous Trapping Efficiency (-)')
    plt.title(f'Sediment Trapping Efficiency (Rolling Mean) - {scenario_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
# %%
