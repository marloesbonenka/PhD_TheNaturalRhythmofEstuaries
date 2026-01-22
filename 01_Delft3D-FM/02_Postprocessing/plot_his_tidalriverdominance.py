"""
Plot longitudinal discharge profiles from sea to river over morphological time,
to visualize the transition from tidal to river dominance.
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path

try:
    base_path = Path(__file__).resolve().parent
except NameError:
    base_path = Path.cwd()
sys.path.append(str(base_path))

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
    plot_variable = 'sediment'  # discharge | sediment | both
    use_cache = True
    sediment_var_name = 'cross_section_bedload_sediment_transport'
    
    run_folder, run_name = find_mf_run_folder(base_dir, mf_number)
    his_file_paths = get_his_paths_for_run(base_dir, run_folder)
    if not his_file_paths:
        raise FileNotFoundError(f"No HIS files found for run {run_name}")
    output_dir = run_folder / "output_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ===== DATA LOADING =====
        print("Loading cross-section data...")
        var_settings = {
            'discharge': {
                'q_var': 'cross_section_discharge',
                'quantity_name': 'Discharge',
                'y_label': 'Discharge [m³/s]',
                'cbar_label': 'Discharge [m³/s]',
                'title': 'Discharge Evolution: Space-Time Heatmap',
                'low_label': 'sea',
                'high_label': 'river',
                'symmetric_scale': False
            },
            'sediment': {
                'q_var': sediment_var_name,
                'quantity_name': 'Sediment transport',
                'y_label': 'Bedload sediment transport',
                'cbar_label': 'Bedload sediment transport',
                'title': 'Bedload Sediment Transport: Space-Time Heatmap',
                'low_label': 'landward',
                'high_label': 'seaward',
                'symmetric_scale': True
            }
        }

        def load_var_data(var_key, select_max_flood):
            settings = var_settings[var_key]
            return load_cross_section_data(
                his_file_paths,
                q_var=settings['q_var'],
                estuary_only=True,
                km_range=(20, 45),
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=select_max_flood,
                flood_sign=-1,
                exclude_last_timestep=True,
                exclude_last_n_days=exclude_last_n_days,
                use_cache=use_cache
            )

        data = load_var_data('discharge' if plot_variable != 'sediment' else 'sediment', select_max_flood=True)
        flood_sign_used = data.get('flood_sign_used', -1)
        
        heatmap_data = None
        full_data = None
        secondary_data = None
        sediment_heatmap_data = None
        if data.get('selection_mode') == 'max_flood':
            heatmap_data = load_cross_section_data(
                his_file_paths,
                q_var=var_settings['discharge' if plot_variable != 'sediment' else 'sediment']['q_var'],
                estuary_only=True,
                km_range=(20, 45),
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=flood_sign_used,
                select_max_flood_per_cycle=True,
                exclude_last_timestep=True,
                exclude_last_n_days=exclude_last_n_days,
                use_cache=use_cache
            )
            full_data = load_cross_section_data(
                his_file_paths,
                q_var=var_settings['discharge' if plot_variable != 'sediment' else 'sediment']['q_var'],
                estuary_only=True,
                km_range=(20, 45),
                select_cycles_hydrodynamic=False,
                n_periods=3,
                select_max_flood=False,
                flood_sign=flood_sign_used,
                select_max_flood_per_cycle=False,
                exclude_last_timestep=True,
                exclude_last_n_days=exclude_last_n_days,
                use_cache=use_cache
            )
            if plot_variable == 'both':
                secondary_data = load_var_data('sediment', select_max_flood=True)
                sediment_heatmap_data = load_cross_section_data(
                    his_file_paths,
                    q_var=var_settings['sediment']['q_var'],
                    estuary_only=True,
                    km_range=(20, 45),
                    select_cycles_hydrodynamic=False,
                    n_periods=3,
                    select_max_flood=False,
                    flood_sign=-1,
                    select_max_flood_per_cycle=True,
                    exclude_last_timestep=True,
                    exclude_last_n_days=exclude_last_n_days,
                    use_cache=use_cache
                )
        print(f"✓ Selected {data['n_timesteps']} timesteps from {data['n_timesteps_original']} total")
        print(f"✓ Found {len(data['km_positions'])} cross-sections")
        print(f"✓ KM range: {data['km_positions'].min():.1f} to {data['km_positions'].max():.1f} km")
        print()
        
        # ===== PLOTTING =====
        print("Creating visualizations...")
        
        settings = var_settings['discharge' if plot_variable != 'sediment' else 'sediment']
        file_tag = settings['quantity_name'].lower().replace(' ', '_')
        if data.get('selection_mode') == 'max_flood':
            # Plot: Max flood profile
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
                # Plot: Heatmap using max-flood moment per cycle
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

                if plot_variable == 'both' and sediment_heatmap_data is not None:
                    print("  - Space-time heatmap (sediment transport, max flood per cycle)...")
                    fig4, ax4 = plot_discharge_heatmap(
                        sediment_heatmap_data,
                        show_flood_limit=False,
                        symmetric_scale=var_settings['sediment']['symmetric_scale'],
                        cbar_label=var_settings['sediment']['cbar_label'],
                        title=var_settings['sediment']['title'],
                        low_label=var_settings['sediment']['low_label'],
                        high_label=var_settings['sediment']['high_label']
                    )
                    plt.tight_layout()
                    fig4.savefig(output_dir / f"{run_name}_heatmap_sediment_max_flood_per_cycle.png", dpi=300, bbox_inches='tight')
                    plt.show()
        else:
            # Plot 1: Statistics
            print(f"  - {settings['quantity_name']} statistics...")
            fig1, axes1 = plot_discharge_statistics(
                data,
                quantity_name=settings['quantity_name'],
                y_label=settings['y_label']
            )
            plt.tight_layout()
            fig1.savefig(output_dir / f"{run_name}_{file_tag}_statistics.png", dpi=300, bbox_inches='tight')
            plt.show()

            # Plot 2: Upstream inflow (most upstream cross-section)
            print(f"  - Upstream {settings['quantity_name'].lower()} time series...")
            fig2, ax2 = plot_upstream_inflow_timeseries(
                data,
                quantity_name=settings['quantity_name'],
                y_label=settings['y_label']
            )
            plt.tight_layout()
            fig2.savefig(output_dir / f"{run_name}_upstream_{file_tag}_timeseries.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Plot 3: Heatmap
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
        
        # Close dataset (only when not caching)
        if not use_cache:
            data['ds'].close()
            if secondary_data is not None:
                secondary_data['ds'].close()
            if sediment_heatmap_data is not None:
                sediment_heatmap_data['ds'].close()
        print("\n✓ Done!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
# %%
