"""
Plot longitudinal discharge profiles from sea to river over morphological time,
to visualize the transition from tidal to river dominance.
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from FUNCTIONS.F_loaddata import *
from FUNCTIONS.F_tidalriverdominance import *

#%%===========================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # --- SETTINGS ---
    root_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC")
    scenario_dir = "02_seasonal"  # 01_constant | 02_seasonal | 03_flashy
    tmorph_years = 50             # 50 | 400 | ...
    base_dir = root_dir / scenario_dir / f"Tmorph_{tmorph_years}years"
    mf_number = 1
    exclude_last_n_days = 0
    
    run_folder, run_name = find_mf_run_folder(base_dir, mf_number)
    his_file_paths = get_his_paths_for_run(base_dir, run_folder)
    if not his_file_paths:
        raise FileNotFoundError(f"No HIS files found for run {run_name}")
    output_dir = run_folder / "output_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ===== DATA LOADING =====
        print("Loading cross-section data...")
        data = load_cross_section_data(
            his_file_paths, 
            estuary_only=True, 
            km_range=(20, 45),
            select_cycles_hydrodynamic=False,  # Use full time series (all cycles)
            n_periods=3,                       # Get cycles from start/middle/end
            select_max_flood=True,             # Select max flood penetration
            flood_sign=-1,                     # Negative discharge = flood
            exclude_last_timestep=True,
            exclude_last_n_days=exclude_last_n_days
        )
        
        heatmap_data = None
        full_data = None
        if data.get('selection_mode') == 'max_flood':
            heatmap_data = load_cross_section_data(
                his_file_paths,
                estuary_only=True,
                km_range=(20, 45),
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=-1,
                select_max_flood_per_cycle=True,
                exclude_last_timestep=True,
                exclude_last_n_days=exclude_last_n_days
            )
            full_data = load_cross_section_data(
                his_file_paths,
                estuary_only=True,
                km_range=(20, 45),
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=-1,
                select_max_flood_per_cycle=False,
                exclude_last_timestep=True,
                exclude_last_n_days=exclude_last_n_days
            )
        print(f"✓ Selected {data['n_timesteps']} timesteps from {data['n_timesteps_original']} total")
        print(f"✓ Found {len(data['km_positions'])} cross-sections")
        print(f"✓ KM range: {data['km_positions'].min():.1f} to {data['km_positions'].max():.1f} km")
        print()
        
        # ===== PLOTTING =====
        print("Creating visualizations...")
        
        if data.get('selection_mode') == 'max_flood':
            # Plot: Max flood profile
            print("  - Max flood longitudinal profile...")
            fig1, ax1 = plot_max_flood_profile(data)
            plt.tight_layout()
            fig1.savefig(output_dir / f"{run_name}_max_flood_profile.png", dpi=300, bbox_inches='tight')
            plt.show()

            if full_data is not None:
                print("  - Max flood profiles (representative periods)...")
                rep_indices = select_max_flood_indices_by_period(
                    full_data['times'],
                    full_data['discharge'],
                    full_data['km_positions'],
                    n_periods=3,
                    flood_sign=-1
                )
                if len(rep_indices) > 0:
                    fig2, ax2 = plot_multiple_max_flood_profiles(full_data, rep_indices)
                    plt.tight_layout()
                    fig2.savefig(output_dir / f"{run_name}_max_flood_profiles_representative.png", dpi=300, bbox_inches='tight')
                    plt.show()

            if heatmap_data is not None:
                # Plot: Heatmap using max-flood moment per cycle
                print("  - Space-time heatmap (max flood per cycle)...")
                fig3, ax3 = plot_discharge_heatmap(heatmap_data)
                plt.tight_layout()
                fig3.savefig(output_dir / f"{run_name}_heatmap_max_flood_per_cycle.png", dpi=300, bbox_inches='tight')
                plt.show()
        else:
            # Plot 1: Statistics
            print("  - Discharge statistics...")
            fig1, axes1 = plot_discharge_statistics(data)
            plt.tight_layout()
            fig1.savefig(output_dir / f"{run_name}_discharge_statistics.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Plot 2: Upstream inflow (most upstream cross-section)
            print("  - Upstream inflow time series...")
            fig2, ax2 = plot_upstream_inflow_timeseries(data)
            plt.tight_layout()
            fig2.savefig(output_dir / f"{run_name}_upstream_inflow_timeseries.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Plot 3: Heatmap
            print("  - Space-time heatmap...")
            fig3, ax3 = plot_discharge_heatmap(data)
            plt.tight_layout()
            fig3.savefig(output_dir / f"{run_name}_heatmap.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Close dataset
        data['ds'].close()
        print("\n✓ Done!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
# %%
