"""
Plot longitudinal discharge profiles from sea to river over morphological time,
to visualize the transition from tidal to river dominance.

ANALYSIS_MODE = 'variability' : loop over discharge variability scenarios (Q500/Q1000)
ANALYSIS_MODE = 'morfac'      : single MORFAC run from Test_MORFAC folder structure
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path

try:
    base_path = Path(__file__).resolve().parent
except NameError:
    base_path = Path.cwd()
sys.path.append(str(base_path))

from FUNCTIONS.F_loaddata import (
    get_stitched_his_paths, load_cross_section_data, load_cross_section_data_from_cache,
    find_mf_run_folder, get_his_paths_for_run,
)
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_tidalriverdominance import *

#%%===========================================================================
# CONFIGURATION
# ============================================================================

ANALYSIS_MODE = 'variability'  # 'variability' | 'morfac'

exclude_last_n_days = 0

# --- Variability mode settings ---
DISCHARGE = 500
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']  # subset to run; None = all

# --- MORFAC mode settings ---
MORFAC_ROOT_DIR  = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC")
MORFAC_SCENARIO  = "03_flashy"       # 01_constant | 02_seasonal | 03_flashy
MORFAC_TMORPH_YEARS = 400            # 50 | 400 | ...
MORFAC_MF_NUMBER = 100

#%% --- Variability mode: paths and folder discovery ---
if ANALYSIS_MODE == 'variability':
    base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
    config = f"Model_Output/Q{DISCHARGE}"
    run_base_path = base_directory / config

    VARIABILITY_MAP = get_variability_map(DISCHARGE)

    VARIABILITY_SCENARIOS = {
        '1': 'baserun',
        '2': 'seasonal',
        '3': 'flashy',
        '4': 'singlepeak',
    }

    timed_out_dir = run_base_path / "timed-out"
    if not timed_out_dir.exists():
        timed_out_dir = None
        print('[WARNING] Timed-out directory not found.')

    model_folders = find_variability_model_folders(
        base_path=run_base_path,
        discharge=DISCHARGE,
        scenarios_to_process=SCENARIOS_TO_PROCESS,
        analyze_noisy=False,
    )

    print(f"Found {len(model_folders)} run folders in: {run_base_path}")

    cache_dir = run_base_path / "cached_data"

    # Build list of (run_name, label, his_file_paths, output_dir, cache_file)
    runs_to_process = []
    for folder in model_folders:
        his_paths = get_stitched_his_paths(
            base_path=run_base_path,
            folder_name=folder,
            timed_out_dir=timed_out_dir,
            variability_map=VARIABILITY_MAP,
            analyze_noisy=False,
        )
        if not his_paths:
            print(f"[WARNING] No HIS files found for {folder.name}, skipping.")
            continue
        scenario_num = str(int(folder.name.split('_')[0]))
        run_id = '_'.join(folder.name.split('_')[1:])
        cache_file = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"
        label = VARIABILITY_SCENARIOS.get(scenario_num, folder.name)
        out_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Output\Q500\output_plots\plots_his_max_floodintrusion") / f"{scenario_num}_{label}"
        runs_to_process.append((folder.name, label, his_paths, out_dir, cache_file))

else:  # morfac
    morfac_base_dir = MORFAC_ROOT_DIR / MORFAC_SCENARIO / f"Tmorph_{MORFAC_TMORPH_YEARS}years"
    run_folder, run_name = find_mf_run_folder(morfac_base_dir, MORFAC_MF_NUMBER)
    his_paths = get_his_paths_for_run(morfac_base_dir, run_folder)
    out_dir = "output_plots" / "plots_his_tidalintrusion" / f"{MORFAC_MF_NUMBER}_{MORFAC_SCENARIO}"
    runs_to_process = [(run_name, MORFAC_SCENARIO, his_paths, out_dir, None)]

#%%===========================================================================
# MAIN EXECUTION
# ============================================================================

def _load(cache_file, his_file_paths, **kwargs):
    """Load using cache when available, fall back to HIS files."""
    q = kwargs.get('q_var', 'cross_section_discharge')
    if cache_file is not None and cache_file.exists():
        print(f"  [cache] {cache_file.name}")
        d = load_cross_section_data_from_cache(cache_file, **kwargs)
    else:
        d = load_cross_section_data(his_file_paths, **kwargs)
    d['discharge'] = d[q]
    return d


for run_name, label, his_file_paths, output_dir, cache_file in runs_to_process:
    print(f"\n{'='*60}")
    print(f"{label}: {run_name}")
    print(f"{'='*60}")

    if not his_file_paths:
        print(f"[WARNING] No HIS files, skipping.")
        continue

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ===== DATA LOADING =====
        print("Loading cross-section data...")
        data = _load(
            cache_file, his_file_paths,
            q_var='cross_section_discharge',
            select_cycles_hydrodynamic=False,
            n_periods=3,
            select_max_flood=True,
            flood_sign=-1,
            exclude_last_timestep=True,
            exclude_last_n_days=exclude_last_n_days,
        )
        flood_sign_used = data.get('flood_sign_used', -1)

        heatmap_data = None
        full_data = None
        if data.get('selection_mode') == 'max_flood':
            heatmap_data = _load(
                cache_file, his_file_paths,
                q_var='cross_section_discharge',
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=flood_sign_used,
                select_max_flood_per_cycle=True,
                exclude_last_timestep=True,
                exclude_last_n_days=exclude_last_n_days,
            )
            full_data = _load(
                cache_file, his_file_paths,
                q_var='cross_section_discharge',
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=flood_sign_used,
                select_max_flood_per_cycle=False,
                exclude_last_timestep=True,
                exclude_last_n_days=exclude_last_n_days,
            )
        print(f"  Selected {data['n_timesteps']} timesteps from {data['n_timesteps_original']} total")
        print(f"  Found {len(data['km_positions'])} cross-sections")
        print(f"  KM range: {data['km_positions'].min():.1f} to {data['km_positions'].max():.1f} km")
        print()

        # ===== PLOTTING =====
        print("Creating visualizations...")

        settings = {
            'quantity_name': 'Discharge',
            'y_label': 'Discharge [m³/s]',
            'cbar_label': 'Discharge [m³/s]',
            'title': 'Discharge Evolution: Space-Time Heatmap',
            'low_label': 'sea',
            'high_label': 'river',
            'symmetric_scale': False
        }
        file_tag = 'discharge'

        if data.get('selection_mode') == 'max_flood':
            print("  - Max flood longitudinal profile...")
            fig1, ax1 = plot_max_flood_profile(data, y_label=settings['y_label'])
            plt.tight_layout()
            fig1.savefig(output_dir / f"{run_name}_max_flood_profile_{file_tag}.png", dpi=300, bbox_inches='tight')
            plt.show()

            if full_data is not None:
                print("  - Max flood profiles (representative periods)...")
                rep_indices = select_max_flood_indices_by_period(
                    full_data['times'],
                    full_data['discharge'],
                    full_data['km_positions'],
                    n_periods=3,
                    flood_sign=flood_sign_used
                )
                if len(rep_indices) > 0:
                    fig2, ax2 = plot_multiple_max_flood_profiles(
                        full_data,
                        rep_indices,
                        y_label=settings['y_label'],
                        title=f"Max-flood profiles (representative periods) - {settings['quantity_name']}"
                    )
                    plt.tight_layout()
                    fig2.savefig(output_dir / f"{run_name}_max_flood_profiles_representative_{file_tag}.png", dpi=300, bbox_inches='tight')
                    plt.show()

            if heatmap_data is not None:
                print("  - Space-time heatmap (max flood per cycle)...")
                fig3, ax3 = plot_discharge_heatmap(
                    heatmap_data,
                    cbar_label=settings['cbar_label'],
                    title=settings['title'],
                    low_label=settings['low_label'],
                    high_label=settings['high_label'],
                    symmetric_scale=settings['symmetric_scale'],
                    flood_sign=flood_sign_used
                )
                plt.tight_layout()
                fig3.savefig(output_dir / f"{run_name}_heatmap_max_flood_per_cycle_{file_tag}.png", dpi=300, bbox_inches='tight')
                plt.show()
        else:
            print(f"  - {settings['quantity_name']} statistics...")
            fig1, axes1 = plot_discharge_statistics(
                data,
                quantity_name=settings['quantity_name'],
                y_label=settings['y_label']
            )
            plt.tight_layout()
            fig1.savefig(output_dir / f"{run_name}_{file_tag}_statistics.png", dpi=300, bbox_inches='tight')
            plt.show()

            print(f"  - Upstream {settings['quantity_name'].lower()} time series...")
            fig2, ax2 = plot_upstream_inflow_timeseries(
                data,
                quantity_name=settings['quantity_name'],
                y_label=settings['y_label']
            )
            plt.tight_layout()
            fig2.savefig(output_dir / f"{run_name}_upstream_{file_tag}_timeseries.png", dpi=300, bbox_inches='tight')
            plt.show()

            print("  - Space-time heatmap...")
            fig3, ax3 = plot_discharge_heatmap(
                data,
                cbar_label=settings['cbar_label'],
                title=settings['title'],
                low_label=settings['low_label'],
                high_label=settings['high_label'],
                symmetric_scale=settings['symmetric_scale'],
                flood_sign=flood_sign_used
            )
            plt.tight_layout()
            fig3.savefig(output_dir / f"{run_name}_heatmap_{file_tag}.png", dpi=300, bbox_inches='tight')
            plt.show()

        print(f"Done: {run_name}")

    except Exception as e:
        print(f"Error for {run_name}: {e}")
        import traceback
        traceback.print_exc()
