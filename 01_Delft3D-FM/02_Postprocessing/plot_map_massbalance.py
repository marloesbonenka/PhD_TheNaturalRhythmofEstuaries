""""Mass balance sensitivity plot: Total Volume Change vs. Morphological Time
This script processes multiple Delft3D-FM model runs — either discharge variability
scenarios or MORFAC sensitivity runs — and plots cumulative sediment volume change."""
#%%
from pathlib import Path
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    _base_path = Path(__file__).resolve().parent
except NameError:
    _base_path = Path.cwd()
if str(_base_path) not in sys.path:
    sys.path.append(str(_base_path))

from FUNCTIONS.F_general import *
from FUNCTIONS.F_braiding_index import *
from FUNCTIONS.F_map_cache import cache_tag_from_bbox, load_or_update_map_cache_multi, select_cache_path
from FUNCTIONS.F_loaddata import get_stitched_map_run_paths

#%%
# =============================================================================
# 1. CONFIGURATION
# =============================================================================

ANALYSIS_MODE = 'variability'  # 'variability' | 'morfac'

var_name  = 'mesh2d_mor_bl'         # Bed level variable
area_name = 'mesh2d_flowelem_ba'    # Cell area variable

check_vars = False

# Cache settings (applied in both modes)
CACHE_BBOX       = [1, 1, 45000, 15000]  # must match extract_cache_map.py; None = full domain
CACHE_TAG        = None
APPEND_TIMESTEPS = True
APPEND_VARIABLES = True

# --- Variability mode settings ---
DISCHARGE            = 500
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']
run_startdate        = '2025-01-01'

# --- MORFAC mode settings ---
discharge_type       = '02_seasonal'  # '01_constant' | '02_seasonal' | '03_flashy'
load_tmorph_periods  = [50]           # e.g. [50], [400], [50, 400]
morfac_run_startdate = '2025-01-01'

# =============================================================================
# 2. PATHS & FOLDER DISCOVERY
# =============================================================================

base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")

if ANALYSIS_MODE == 'variability':
    config        = f"Model_Output/Q{DISCHARGE}"
    run_base_path = base_directory / config
    timed_out_dir = run_base_path / "timed-out"
    if not timed_out_dir.exists():
        timed_out_dir = None
        print('[WARNING] Timed-out directory not found.')

    VARIABILITY_MAP = get_variability_map(DISCHARGE)
    model_folders = find_variability_model_folders(
        base_path=run_base_path,
        discharge=DISCHARGE,
        scenarios_to_process=SCENARIOS_TO_PROCESS,
        analyze_noisy=False,
    )

    assessment_dir = run_base_path / 'cached_data'
    assessment_dir.mkdir(parents=True, exist_ok=True)

    VARIABILITY_SCENARIOS = {
        '1':  'baserun',  '2':  'seasonal',  '3':  'flashy',  '4':  'singlepeak',
        '01': 'baserun',  '02': 'seasonal',  '03': 'flashy',  '04': 'singlepeak',
    }

# Global storage for all loaded data (persists across cell reruns)
if 'all_loaded_data' not in globals():
    all_loaded_data = {}

variables_checked = False

# =============================================================================
# 3. DATA LOADING
# =============================================================================

try:
    if ANALYSIS_MODE == 'variability':
        print(f"\n{'='*60}")
        print(f"Loading variability data for Q{DISCHARGE}...")
        print(f"{'='*60}")

        for folder in model_folders:
            print(f"\n--- Processing: {folder.name} ---")

            run_paths = get_stitched_map_run_paths(
                base_path=run_base_path,
                folder_name=folder.name,
                timed_out_dir=timed_out_dir,
                variability_map=VARIABILITY_MAP,
                analyze_noisy=False,
            )
            if not run_paths:
                run_paths = [folder]

            cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)

            # --- Cache diagnostics: which per-variable files exist? ---
            for vn in [var_name, area_name]:
                cp = select_cache_path(assessment_dir, folder.name, vn, cache_tag)
                status = 'EXISTS' if cp.exists() else 'MISSING'
                print(f"  [cache-check] {vn}: {status} -> {cp.name}")

            ds = load_or_update_map_cache_multi(
                cache_dir=assessment_dir,
                folder_name=folder.name,
                run_paths=run_paths,
                var_names=[var_name, area_name],
                bbox=CACHE_BBOX,
                append_time=APPEND_TIMESTEPS,
                append_vars=APPEND_VARIABLES,
                cache_tag=cache_tag,
            )
            if ds is None:
                print(f"  Warning: No cached data for {folder.name}, skipping.")
                continue

            try:
                # --- Dataset summary ---
                print(f"  [ds] variables : {list(ds.data_vars)}")
                print(f"  [ds] time range: {ds.time.values[0]} -> {ds.time.values[-1]} "
                      f"({len(ds.time)} timesteps)")

                if check_vars and not variables_checked:
                    check_available_variables_xarray(ds)
                    variables_checked = True

                start_timestamp   = pd.Timestamp(run_startdate)
                times             = pd.to_datetime(ds['time'].values)
                hydro_elapsed_years = np.array([(t - start_timestamp).days / 365.25 for t in times])

                areas        = ds[area_name]
                bed_levels   = ds[var_name]
                total_volume  = (bed_levels * areas).sum(dim=ds[var_name].dims[-1])
                volume_change = total_volume - total_volume.isel(time=0)

                scenario_num = str(int(folder.name.split('_')[0]))
                label        = VARIABILITY_SCENARIOS.get(scenario_num, folder.name)
                all_loaded_data[label] = {
                    'x_values':     list(hydro_elapsed_years),
                    'volume_change': list(volume_change.values),
                }
                print(f"  Processed {folder.name} ({label}): {len(hydro_elapsed_years)} timesteps")
            finally:
                ds.close()

    else:  # morfac
        for morfyears in load_tmorph_periods:
            tmorph_period = f'Tmorph_{morfyears}years'
            print(f"\n{'='*60}")
            print(f"Loading data for {tmorph_period}...")
            print(f"{'='*60}")

            morfac_base         = base_directory / 'Test_MORFAC' / discharge_type / tmorph_period
            timed_out_directory = morfac_base / 'timed-out'
            assessment_dir      = morfac_base / 'cached_data'
            assessment_dir.mkdir(parents=True, exist_ok=True)
            all_mf_folders = [f.name for f in morfac_base.iterdir()
                               if f.name.startswith('MF') and f.is_dir()]

            for folder in all_mf_folders:
                model_location = morfac_base / folder

                mf_match   = re.search(r'MF(\d+(?:\.\d+)?)', folder)
                current_mf = float(mf_match.group(1)) if mf_match else None
                if current_mf is None:
                    print(f"Warning: Could not extract MORFAC from {folder}, skipping.")
                    continue

                stitched_paths = get_stitched_map_run_paths(
                    base_path=morfac_base,
                    folder_name=folder,
                    timed_out_dir=timed_out_directory,
                    variability_map=None,
                    analyze_noisy=False,
                )
                timed_out_location = stitched_paths[0] if len(stitched_paths) > 1 else None

                try:
                    all_morph_years   = []
                    all_volume_change = []
                    timed_out_end_timestamp = None
                    ds_timed_out      = None

                    # Process timed-out dataset first (if it exists)
                    if timed_out_location is not None:
                        try:
                            print(f"\n  Loading timed-out data for MF{int(current_mf)}...")
                            cache_tag    = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
                            ds_timed_out = load_or_update_map_cache_multi(
                                cache_dir=assessment_dir,
                                folder_name=timed_out_location.name,
                                run_paths=[timed_out_location],
                                var_names=[var_name, area_name, 'morfac'],
                                bbox=CACHE_BBOX,
                                append_time=APPEND_TIMESTEPS,
                                append_vars=APPEND_VARIABLES,
                                cache_tag=cache_tag,
                            )
                            if ds_timed_out is None:
                                raise RuntimeError("Timed-out cache returned no data")

                            try:
                                if check_vars and not variables_checked:
                                    check_available_variables_xarray(ds_timed_out)
                                    variables_checked = True

                                start_timestamp        = pd.Timestamp(morfac_run_startdate)
                                times_to               = pd.to_datetime(ds_timed_out['time'].values)
                                hydro_elapsed_years_to = np.array(
                                    [(t - start_timestamp).days / 365.25 for t in times_to])
                                morfac_vals_to = (ds_timed_out['morfac'].values
                                                  if 'morfac' in ds_timed_out
                                                  else np.full_like(hydro_elapsed_years_to, current_mf))
                                morph_years_to = hydro_elapsed_years_to * morfac_vals_to

                                areas_to      = ds_timed_out[area_name]
                                bed_levels_to = ds_timed_out[var_name]
                                total_volume_to  = (bed_levels_to * areas_to).sum(
                                    dim=ds_timed_out[var_name].dims[-1])
                                volume_change_to = total_volume_to - total_volume_to.isel(time=0)

                                all_morph_years.extend(morph_years_to)
                                all_volume_change.extend(volume_change_to.values)
                                timed_out_end_timestamp = times_to[-1]

                                initial_volume = total_volume_to.isel(time=0)
                                print(f"  Timed-out: {len(morph_years_to)} timesteps, "
                                      f"ends at Tmorph={morph_years_to[-1]:.1f}y")
                            finally:
                                ds_timed_out.close()
                                ds_timed_out = None

                        except Exception as e:
                            print(f"  Warning: Could not load timed-out data for {folder}: {e}")
                            initial_volume = None
                    else:
                        initial_volume = None

                    # Process main dataset
                    print(f"  Loading main data for MF{int(current_mf)}...")
                    cache_tag = cache_tag_from_bbox(CACHE_BBOX, CACHE_TAG)
                    ds = load_or_update_map_cache_multi(
                        cache_dir=assessment_dir,
                        folder_name=folder,
                        run_paths=[model_location],
                        var_names=[var_name, area_name, 'morfac'],
                        bbox=CACHE_BBOX,
                        append_time=APPEND_TIMESTEPS,
                        append_vars=APPEND_VARIABLES,
                        cache_tag=cache_tag,
                    )
                    if ds is None:
                        print(f"  Warning: No cached data for {folder}, skipping.")
                        continue

                    try:
                        if check_vars and not variables_checked:
                            check_available_variables_xarray(ds)
                            variables_checked = True

                        start_timestamp     = pd.Timestamp(morfac_run_startdate)
                        times               = pd.to_datetime(ds['time'].values)
                        hydro_elapsed_years = np.array(
                            [(t - start_timestamp).days / 365.25 for t in times])
                        morfac_vals = (ds['morfac'].values
                                       if 'morfac' in ds
                                       else np.full_like(hydro_elapsed_years, current_mf))
                        morph_years = hydro_elapsed_years * morfac_vals

                        if all_morph_years and timed_out_end_timestamp is not None:
                            times_to_end = pd.to_datetime(timed_out_end_timestamp)
                            hydro_adj    = np.array([(t - times_to_end).days / 365.25 for t in times])
                            morph_years  = hydro_adj * morfac_vals + all_morph_years[-1]
                            print(f"  DEBUG: Timed-out ended at Tmorph={all_morph_years[-1]:.1f}y, "
                                  f"real time={times_to_end}")
                            print(f"  DEBUG: Main first morph_year={morph_years[0]:.1f}y, "
                                  f"last={morph_years[-1]:.1f}y")

                        areas        = ds[area_name]
                        bed_levels   = ds[var_name]
                        total_volume = (bed_levels * areas).sum(dim=ds[var_name].dims[-1])

                        if initial_volume is not None:
                            final_to_vc   = all_volume_change[-1] if all_volume_change else 0
                            volume_change = (total_volume - total_volume.isel(time=0)) + final_to_vc
                        else:
                            volume_change = total_volume - total_volume.isel(time=0)

                        all_morph_years.extend(morph_years)
                        all_volume_change.extend(volume_change.values)

                        data_key = (tmorph_period, int(current_mf))
                        all_loaded_data[data_key] = {
                            'x_values':     all_morph_years,
                            'volume_change': all_volume_change,
                        }

                        if initial_volume is not None:
                            print(f"  Main: {len(morph_years)} timesteps, "
                                  f"continues Tmorph={morph_years[0]:.1f}y to {morph_years[-1]:.1f}y")
                        else:
                            print(f"  Main: {len(morph_years)} timesteps, "
                                  f"Tmorph={morph_years[0]:.1f}-{morph_years[-1]:.1f}y")
                        print(f"Processed MF{int(current_mf)}: {len(all_morph_years)} total timesteps")
                    finally:
                        ds.close()

                except Exception as e:
                    print(f"Error processing {folder}: {e}")
finally:
    pass

#%%
# =============================================================================
# 4. PLOTTING
# =============================================================================

print("\n" + "="*60)
print("PLOTTING PHASE")
print("="*60)
print(f"Available data keys: {list(all_loaded_data.keys())}\n")

if ANALYSIS_MODE == 'variability':
    # ------------------------------------------------------------------
    # Variability: single combined plot (one line per scenario)
    # ------------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    scenario_colors = {
        'baserun': 'steelblue', 'seasonal': 'darkorange',
        'flashy': 'firebrick', 'singlepeak': 'seagreen',
    }
    for label, data in all_loaded_data.items():
        color = scenario_colors.get(label, None)
        plt.plot(data['x_values'], data['volume_change'],
                 label=label, marker='o', markersize=4, linewidth=2, color=color)

    plt.xlabel('Hydrodynamic Time [years]')
    plt.ylabel('Change in Sediment Volume [m³]')
    plt.title(f'Mass Balance (Q{DISCHARGE}): cumulative volume change since t = 0')
    plt.legend(title='Scenario', loc='upper left', title_fontsize='large', fontsize='medium')
    plt.grid(True, linestyle='--', alpha=0.6)

    figure_name = f'Q{DISCHARGE}_massbalance_variability.png'
    figure_path = run_base_path / 'output_plots' / figure_name
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {figure_path}")
    plt.show()

else:
    # ------------------------------------------------------------------
    # MORFAC: three plots (combined, Tmorph_50years, Tmorph_400years)
    # ------------------------------------------------------------------
    tmorph_periods_to_plot = [f'Tmorph_{y}years' for y in load_tmorph_periods]

    # PLOT 1: Combined (all loaded tmorph_periods)
    plt.figure(figsize=(12, 7))

    relevant_keys = [k for k in all_loaded_data.keys() if k[0] in tmorph_periods_to_plot]
    morfac_values = sorted(set(k[1] for k in relevant_keys))
    n_morfacs     = len(morfac_values)

    cmap             = plt.cm.get_cmap('viridis')
    colors           = [cmap(i / max(1, n_morfacs - 1)) for i in range(n_morfacs)]
    morfac_to_color  = {mf: colors[i] for i, mf in enumerate(morfac_values)}

    for mf in morfac_values:
        for data_key in sorted(all_loaded_data.keys()):
            tp_key, key_mf = data_key
            if tp_key not in tmorph_periods_to_plot or key_mf != mf:
                continue
            data = all_loaded_data[data_key]
            plt.plot(data['x_values'], data['volume_change'],
                     label=str(mf), marker='o', markersize=4,
                     linewidth=2, color=morfac_to_color[mf])

    plt.xlabel('Morphological Time [years]')
    plt.xlim(0, 50)
    plt.ylim(0, 1e7)
    plt.ylabel('Change in Sediment Volume [m³]')
    plt.title('Mass Balance: cumulative volume change since t = 0')
    plt.legend(title='${MORFAC}$ =', loc='upper left', title_fontsize='large', fontsize='medium')
    plt.grid(True, linestyle='--', alpha=0.6)

    figure_name = f'{discharge_type}_MFsensitivity_massbalance_combined.png'
    figure_path = base_directory / 'Test_MORFAC' / discharge_type / figure_name
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure 1 (Combined) saved to: {figure_path}")
    plt.show()

    # PLOT 2: Tmorph_50years only
    if 'Tmorph_50years' in tmorph_periods_to_plot:
        plt.figure(figsize=(12, 7))

        relevant_keys_50  = [k for k in all_loaded_data.keys() if k[0] == 'Tmorph_50years']
        morfac_values_50  = sorted(set(k[1] for k in relevant_keys_50))
        n_morfacs_50      = len(morfac_values_50)
        colors_50         = [cmap(i / max(1, n_morfacs_50 - 1)) for i in range(n_morfacs_50)]
        morfac_to_color_50 = {mf: colors_50[i] for i, mf in enumerate(morfac_values_50)}

        for mf in morfac_values_50:
            for data_key in sorted(all_loaded_data.keys()):
                tp_key, key_mf = data_key
                if tp_key != 'Tmorph_50years' or key_mf != mf:
                    continue
                data = all_loaded_data[data_key]
                plt.plot(data['x_values'], data['volume_change'],
                         label=str(mf), marker='o', markersize=4,
                         linewidth=2, color=morfac_to_color_50[mf])

        plt.xlabel('Morphological Time [years]')
        plt.xlim(0, 50)
        plt.ylim(0, 1e7)
        plt.ylabel('Change in Sediment Volume [m³]')
        plt.title('Mass Balance (Tmorph = 50 years): cumulative volume change since t = 0')
        plt.legend(title='${MORFAC}$ =', loc='upper left', title_fontsize='large', fontsize='medium')
        plt.grid(True, linestyle='--', alpha=0.6)

        figure_name_50 = f'{discharge_type}_MFsensitivity_massbalance_Tmorph_50years.png'
        figure_path_50 = base_directory / 'Test_MORFAC' / discharge_type / figure_name_50
        plt.savefig(figure_path_50, dpi=300, bbox_inches='tight')
        print(f"Figure 2 (Tmorph_50years) saved to: {figure_path_50}")
        plt.show()

    # PLOT 3: Tmorph_400years only
    if 'Tmorph_400years' in tmorph_periods_to_plot:
        plt.figure(figsize=(12, 7))

        relevant_keys_400  = [k for k in all_loaded_data.keys() if k[0] == 'Tmorph_400years']
        morfac_values_400  = sorted(set(k[1] for k in relevant_keys_400))
        n_morfacs_400      = len(morfac_values_400)
        colors_400         = [cmap(i / max(1, n_morfacs_400 - 1)) for i in range(n_morfacs_400)]
        morfac_to_color_400 = {mf: colors_400[i] for i, mf in enumerate(morfac_values_400)}

        for mf in morfac_values_400:
            for data_key in sorted(all_loaded_data.keys()):
                tp_key, key_mf = data_key
                if tp_key != 'Tmorph_400years' or key_mf != mf:
                    continue
                data = all_loaded_data[data_key]
                plt.plot(data['x_values'], data['volume_change'],
                         label=str(mf), marker='o', markersize=4,
                         linewidth=2, color=morfac_to_color_400[mf])

        plt.xlabel('Morphological Time [years]')
        plt.xlim(0, 400)
        plt.ylabel('Change in Sediment Volume [m³]')
        plt.title('Mass Balance (Tmorph = 400 years): cumulative volume change since t = 0')
        plt.legend(title='${MORFAC}$ =', loc='upper left', title_fontsize='large', fontsize='medium')
        plt.grid(True, linestyle='--', alpha=0.6)

        figure_name_400 = f'{discharge_type}_MFsensitivity_massbalance_Tmorph_400years.png'
        figure_path_400 = base_directory / 'Test_MORFAC' / discharge_type / figure_name_400
        plt.savefig(figure_path_400, dpi=300, bbox_inches='tight')
        print(f"Figure 3 (Tmorph_400years) saved to: {figure_path_400}")
        plt.show()

# %%
