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
import xarray as xr
from pathlib import Path

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_loaddata import load_and_cache_scenario, get_stitched_his_paths

#%% --- CONFIGURATION ---
# What to analyze?
var_name = 'cross_section_sand' # 'cross_section_bedload_sediment_transport' # cumulative bed load sediment transport [kg] #'cross_section_sand' # flux based on upwind cell [kg/s]

output_dirname = "plots_his_sedimentbuffer"

mpl.rcParams['figure.figsize'] = (8, 6)     

# Define cross-section km boundaries
box_edges = np.arange(20, 50, 5)  # [20, 25, 30, 35, 40, 45]
boxes = [(box_edges[i], box_edges[i+1]) for i in range(len(box_edges)-1)]

# Which scenarios to process (set to None or empty list for all)
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']  # Use all scenarios
DISCHARGE = 500                    # or 1000, etc.
ANALYZE_NOISY = False               # Set to True to analyze noisy scenarios

#%% --- PATHS ---  

base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"
if ANALYZE_NOISY:
    base_path = base_directory / config / f"0_Noise_Q{DISCHARGE}"
else:
    base_path = base_directory / config
output_dir = base_path / output_dirname
output_dir.mkdir(parents=True, exist_ok=True)
timed_out_dir = base_path / "timed-out"
if not base_path.exists():
    raise FileNotFoundError(f"Base path not found: {base_path}")
if not timed_out_dir.exists():
    timed_out_dir = None
    print('[WARNING] Timed-out directory not found. No timed-out scenarios will be included.')
    #raise FileNotFoundError(f"Timed-out directory not found: {timed_out_dir}")

# Mapping: restart folder prefix -> timed-out folder prefix
# 1 = constant (baserun), 2 = seasonal, 3 = flashy, 4 = singlepeak
VARIABILITY_MAP = {
    '1': f'01_baserun{DISCHARGE}',
    '2': f'02_run{DISCHARGE}_seasonal',
    '3': f'03_run{DISCHARGE}_flashy',
    '4': f'04_run{DISCHARGE}_singlepeak',
}

# Find all run folders: start with digit (with or without '_rst')
model_folders = [f.name for f in base_path.iterdir() 
                    if f.is_dir() and f.name[0].isdigit()]
# Filter by SCENARIOS_TO_PROCESS if specified. Normalize leading zeros by
# comparing integer scenario numbers so '1' and '01' match the same scenario.
if SCENARIOS_TO_PROCESS:
    try:
        scenario_filter = set(int(s) for s in SCENARIOS_TO_PROCESS)
    except Exception:
        scenario_filter = set()
    model_folders = [f for f in model_folders if int(f.split('_')[0]) in scenario_filter]
# Sort by leading number
model_folders.sort(key=lambda x: int(x.split('_')[0]))

print(f"Found {len(model_folders)} run folders in: {base_path}")

# #%% For testing: process only the specified folder 
# model_folders=[model_folders[0]] 

#%%
# --- 1. For each run, find only the correct timed-out and restart part ---
all_run_paths = {}
run_his_paths = {}

for folder in model_folders:
    his_paths = get_stitched_his_paths(
        base_path=base_path,
        folder_name=folder,
        timed_out_dir=timed_out_dir,
        variability_map=VARIABILITY_MAP,
        analyze_noisy=ANALYZE_NOISY,
    )

    if his_paths:
        print(f"[HIS-stitch] {folder}: {len(his_paths)} part(s)")
        for p in his_paths:
            print(f"[HIS-stitch]   {p}")
    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

print(f"Found {len(model_folders)} run folders in: {base_path}")

#%% --- CACHE DIR SETUP ---
cache_dir = base_path / "cached_data"
if cache_dir.exists():
    if not cache_dir.is_dir():
        raise RuntimeError(f"[ERROR] {cache_dir} exists but is not a directory. Please remove or rename it.")
    else:
        try:
            _ = list(cache_dir.iterdir())
        except Exception as e:
            raise RuntimeError(f"[ERROR] {cache_dir} exists but is not available: {e}")
else:
    cache_dir.mkdir(exist_ok=True)

#%% --- LOAD AND STITCH ALL DATA FIRST ---
scenario_data = {}
for scenario_dir, his_file_paths in run_his_paths.items():
    scenario_name = Path(scenario_dir).name
    scenario_num = scenario_dir.split('_')[0]
    run_id = '_'.join(scenario_name.split('_')[1:])
    
    # Generic cache name — one file per scenario, all variables appended inside
    cache_file = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

    _, result = load_and_cache_scenario(
        scenario_dir=scenario_dir,
        his_file_paths=his_file_paths,
        cache_file=cache_file,
        boxes=boxes,
        var_name=var_name,
    )

    scenario_data[scenario_dir] = result
    
#%% --- PLOT ALL SCENARIOS & CACHE ---
for scenario_dir, data in scenario_data.items():

    km_positions = data['km_positions']
    transport = data[var_name]   # dynamic variable name
    time = data['t']
    buffer_volumes = data['buffer_volumes']

    # # --- DIAGNOSTIC: Print stats for transport variable ---
    # print(f"\n[DIAGNOSTIC] Scenario: {scenario_dir} - Variable: {var_name}")
    # try:
    #     arr = np.array(transport)
    #     print(f"  Type: {type(transport)}")
    #     print(f"  Shape: {arr.shape}")
    #     print(f"  Min: {arr.min()}")
    #     print(f"  Max: {arr.max()}")
    #     print(f"  Mean: {arr.mean()}")
    #     print(f"  Nonzero count: {(arr != 0).sum()}")
    #     print(f"  Sample values: {arr.flatten()[:10]}")
    # except Exception as e:
    #     print(f"  Could not compute stats: {e}")

    # --- CUMULATIVE BUFFER VOLUME PLOT ---
    plt.figure()
    for (box_start, box_end), buf in buffer_volumes.items():
        plt.plot(time, buf, label=f'{box_start}-{box_end} km')

    plt.xlabel('hydrodynamic time (x 100 = morphological time)')
    plt.ylabel('Buffer Volume (m3)')
    plt.ylim(-0.1, 1.15e11)
    plt.legend()
    plt.title(f'Sediment buffer volumes per section for {scenario_dir}')
    plt.grid()
    plt.tight_layout()
    fig1_path = Path(output_dir) / f"{scenario_name}_sediment_buffer_volume_cumulative.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"Saved cumulative buffer plot to {fig1_path}")
    plt.show()

    # --- INSTANTANEOUS RATE OF CHANGE PLOT ---
    plt.figure()
    for (box_start, box_end), buf in buffer_volumes.items():
        d_buffer = np.diff(buf)
        plt.plot(time[1:], d_buffer, label=f'{box_start}-{box_end} km')
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

    # --- TIDAL AVERAGE (RESIDUAL) RATE OF CHANGE ---
    plt.figure(figsize=(12, 6))

    # Define your tidal window (e.g., 25 hours if timestep is 1h)
    # In Delft3D-FM, check your output interval! 
    window = 25 

    for (box_start, box_end), buf in buffer_volumes.items():
        # 1. Calculate Instantaneous dV (m3 per output timestep)
        # Using np.diff gives us the change between samples
        d_buffer = np.diff(buf)
        
        # 2. Apply a centered moving average (Tidal Filter)
        # This extracts the "Residual" transport (the long-term trend)
        if len(d_buffer) >= window:
            d_buffer_smooth = np.convolve(d_buffer, np.ones(window)/window, mode='same')
            
            # We plot the smoothed line
            plt.plot(time[1:], d_buffer_smooth, 
                    label=f'Residual: {box_start}-{box_end} km', 
                    linewidth=2)
            
            # Optional: Plot the raw instantaneous data in the background with low alpha
            # plt.plot(time[1:], d_buffer, alpha=0.15, color='gray', zorder=1)
        else:
            print(f"Warning: Simulation too short for window {window} at {box_start}km")

    plt.axhline(0, color='black', lw=1.5, ls='--') # The "Equilibrium" line
    plt.ylim(-0.25e7, 0.25e7)
    plt.xlabel('hydrodynamic time (x 100 = morphological time)')
    plt.ylabel('Tidal Avg dV/dt (m³/timestep)')
    plt.title(f'Residual (Tidal-Averaged) Sediment Storage Rate\nScenario: {scenario_name}')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    # Save the plot
    fig_residual_path = Path(output_dir) / f"{scenario_name}_residual_buffer_change.png"
    plt.savefig(fig_residual_path, dpi=300)
    plt.show()

    # --- STABLE ESTUARINE TRAPPING EFFICIENCY ---
    plt.figure(figsize=(12, 6))

    window = 25 # 25 hours for a tidal cycle

    for (box_start, box_end), buf in buffer_volumes.items():
        idx_up = np.argmin(np.abs(km_positions - box_start))
        
        # 1. Instantaneous Net Change (The Numerator)
        # This is the same as dV/dt
        dV = np.diff(buf) 
        
        # 2. Instantaneous Gross Inflow (The Denominator)
        # We take the absolute value of the flux at the upstream gate
        # This represents the "Total Activity" at that boundary
        dIn_gross = np.abs(np.diff(transport[:, idx_up]))
        
        # 3. Tidal Averaging (Summing over 25 hours)
        # We sum the numerator and denominator SEPARATELY before dividing
        net_change_tidal = np.convolve(dV, np.ones(window), mode='same')
        gross_flux_tidal = np.convolve(dIn_gross, np.ones(window), mode='same')
        
        # 4. Calculate Efficiency
        # Adding epsilon to avoid 0/0 during totally stagnant scenarios
        efficiency = net_change_tidal / (gross_flux_tidal + 1e-6)
        
        plt.plot(time[1:], efficiency, label=f'Efficiency: {box_start}-{box_end} km')

    plt.axhline(0, color='black', lw=1, ls='--')
    plt.axhline(1, color='red', lw=1, ls=':', label='Total Trap')
    plt.ylim(-1.1, 1.1) 
    plt.xlabel('hydrodynamic time (x 100 = morphological time)')
    plt.ylabel('Tidal-Averaged Efficiency (-)')
    plt.title(f'Sediment Trapping Efficiency (Gross Flux Method)\nScenario: {scenario_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
# %%
