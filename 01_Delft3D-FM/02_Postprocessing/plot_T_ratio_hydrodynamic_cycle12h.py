"""
Compute and plot hydrodynamic T ratio on a 12-hour tidal-cycle basis.

Cycle-based definition (per 12h window):
- Qriver(t): mean upstream discharge over the same 12h window (riverward-positive)
- Qtide_hat(t): characteristic tidal discharge amplitude at the mouth within that window
  computed from demeaned mouth discharge using an amplitude estimator.
- T(t) = Qtide_hat(t) / Qriver(t)

This supports both MORFAC sensitivity analysis and discharge variability analysis.
Workflow mirrors the other scripts:
- Loop over MORFACs or variability scenarios
- Cache results
- Plot T(t) per run
- Plot comparison metrics across MORFACs or scenarios
- Plot distributions (PDF + exceedance + fraction T>1) per run
"""
#%%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
# =============================================================================
# Imports: make this work both as a script and inside notebooks
# =============================================================================

try:
	script_dir = Path(__file__).resolve().parent
except NameError:
	script_dir = Path.cwd()

for candidate in (script_dir, script_dir / "01_Delft3D-FM" / "02_Postprocessing"):
	if (candidate / "FUNCTIONS").exists():
		sys.path.append(str(candidate))
		break

from FUNCTIONS.F_loaddata import (
	find_mf_run_folder,
	get_his_paths_for_run,
	get_stitched_his_paths,
)
from FUNCTIONS.F_general import (
	get_variability_map,
	find_variability_model_folders,
)
from FUNCTIONS.F_T_ratio_hydrodynamic_cycle12h import *


#%%
# --- SETTINGS ---
# ANALYSIS_MODE: "morfac" for MORFAC sensitivity analysis
#                "variability" for discharge variability scenarios
ANALYSIS_MODE = 'variability'

if ANALYSIS_MODE == 'morfac':
	DISCHARGE = 500  # only used in labels/metadata
	root_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC")
	scenario_dir = '03_flashy'
	tmorph_years = 400
	base_dir = root_dir / scenario_dir / f"Tmorph_{tmorph_years}years"

	morfac_values = [100, 200, 400]  # e.g. [1, 2, 5, 10, 50]
	run_values = morfac_values
	run_axis_col = 'morfac'
	run_axis_label = 'MORFAC [-]'
	run_label_map = {mf: f"MF{mf}" for mf in morfac_values}
	default_morfac = None

elif ANALYSIS_MODE == 'variability':
	DISCHARGE = 1000  # or 1000
	base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
	config = f"Model_Output/Q{DISCHARGE}"
	base_dir = base_directory / config
	ANALYZE_NOISY = False
	timed_out_dir = base_dir / "timed-out"
	if not timed_out_dir.exists():
		timed_out_dir = None

	SCENARIOS_TO_PROCESS = None  # e.g. ['1', '3'] or None for all
	default_morfac = 100  # MORFAC used in variability runs
	morfac_values = []

	VARIABILITY_MAP = get_variability_map(DISCHARGE)
	folders = find_variability_model_folders(
		base_path=base_dir,
		discharge=DISCHARGE,
		scenarios_to_process=SCENARIOS_TO_PROCESS,
		analyze_noisy=ANALYZE_NOISY,
	)

	run_his_paths = {}
	run_cache_paths = {}
	for folder in folders:
		his_paths = get_stitched_his_paths(
			base_path=base_dir,
			folder_name=folder,
			timed_out_dir=timed_out_dir,
			variability_map=VARIABILITY_MAP,
			analyze_noisy=ANALYZE_NOISY,
		)
		if his_paths:
			run_his_paths[folder] = his_paths
			scenario_name = folder.name
			scenario_num = scenario_name.split('_')[0]
			run_id = '_'.join(scenario_name.split('_')[1:])
			cache_file = base_dir / "cached_data" / f"hisoutput_{int(scenario_num)}_{run_id}.nc"
			run_cache_paths[folder] = cache_file
		else:
			print(f"[WARNING] No HIS files found for {folder.name}")

	folders = [f for f in folders if f in run_his_paths]
	run_values = [int(f.name.split('_')[0]) for f in folders]
	run_values = sorted(set(run_values))
	run_axis_col = 'scenario'
	run_axis_label = 'Scenario [-]'
	run_label_map = {
		sc: VARIABILITY_MAP.get(str(sc), VARIABILITY_MAP.get(str(sc).zfill(2), f"Scenario {sc}"))
		for sc in run_values
	}

else:
	raise ValueError(f"Unsupported ANALYSIS_MODE: {ANALYSIS_MODE}")

exclude_last_n_days = 0

# 12-hour cycle settings
cycle_hours = 12.0
amp_method = 'half_percentile_range'
amp_percentiles = (5, 95)

# Cross-section filtering
estuary_only = True
km_range = (20, 45)

# Plot toggles
plot_time_series = True
plot_T_vs_morfac_year_windows = True
target_morph_years = [10, 30, 50]  # interpreted as mean over [Y, Y+1)
plot_min_mean_max = True
plot_distributions = True
distribution_year_windows = [None, 10, 30, 50]  # None => whole run

# Cache/compute
output_dir = base_dir / "output_plots" / "plots_his_T_ratio_hydro_cycle12h"
output_dir.mkdir(parents=True, exist_ok=True)
show_plots = True

results = {}
run_names = {}

if ANALYSIS_MODE == 'morfac':
	runs_to_process = []
	for mf in morfac_values:
		run_folder, run_name = find_mf_run_folder(base_dir, mf)
		runs_to_process.append((mf, run_folder, run_name, mf, None, None))
else:
	folder_lookup = {int(f.name.split('_')[0]): f for f in folders}
	runs_to_process = []
	for scenario in run_values:
		run_folder = folder_lookup.get(scenario)
		if run_folder is None:
			continue
		runs_to_process.append((
			scenario,
			run_folder,
			run_folder.name,
			default_morfac,
			run_his_paths.get(run_folder, []),
			run_cache_paths.get(run_folder),
		))

for run_id, run_folder, run_name, morfac_for_calc, his_paths_override, cache_file in runs_to_process:
	if his_paths_override is None:
		his_paths = get_his_paths_for_run(base_dir, run_folder)
	else:
		his_paths = his_paths_override
	if not his_paths:
		raise FileNotFoundError(f"No HIS files found for run={run_name}")

	print(f"Processing {run_label_map.get(run_id, run_name)}: 12h-cycle hydrodynamic T...")
	data = load_hydro_cross_section_data(
		his_paths,
		cache_file=cache_file,
		estuary_only=estuary_only,
		km_range=km_range,
		exclude_last_timestep=True,
		exclude_last_n_days=exclude_last_n_days,
	)

	df = compute_T_hydro_cycle_based(
		data=data,
		morfac=morfac_for_calc,
		cycle_hours=cycle_hours,
		amp_method=amp_method,
		amp_percentiles=amp_percentiles,
	)

	results[run_id] = df
	run_names[run_id] = run_name

	csv_path = output_dir / f"{run_name}_Thydro_cycle{int(cycle_hours)}h.csv"
	df.to_csv(csv_path, index=False)

	if plot_time_series:
		title_label = run_label_map.get(run_id, run_name)
		plot_Thydro_timeseries(
			df,
			title=f"Hydrodynamic T ratio (12h-cycle) - {title_label}",
			output_path=output_dir / f"{run_name}_Thydro_cycle{int(cycle_hours)}h_timeseries.png",
			show=show_plots,
		)

	ds_obj = data.get('ds')
	if ds_obj is not None:
		ds_obj.close()

# --- Comparison plots (mean over year windows) ---
if plot_T_vs_morfac_year_windows and results:
	for year_start in target_morph_years:
		rows = []
		for run_id in run_values:
			df = results.get(run_id)
			if df is None:
				continue
			T_mean_year = get_T_mean_over_year(df, year_start, col='T_hydro', year_duration=1.0)
			rows.append({run_axis_col: run_id, 'T_hydro': T_mean_year})
		sens_df = pd.DataFrame(rows).dropna()

		if ANALYSIS_MODE == 'morfac':
			plot_Thydro_vs_morfac(
				sens_df,
				title=f"Thydro (12h-cycle) vs MORFAC (mean over years {year_start}-{year_start + 1})",
				output_path=output_dir / f"Thydro_vs_MORFAC_cycle{int(cycle_hours)}h_mean_year{year_start}.png",
				show=show_plots,
			)
		else:
			plot_Thydro_vs_run(
				sens_df,
				x_col=run_axis_col,
				xlabel=run_axis_label,
				title=f"Thydro (12h-cycle) vs scenario (mean over years {year_start}-{year_start + 1})",
				output_path=output_dir / f"Thydro_vs_scenario_cycle{int(cycle_hours)}h_mean_year{year_start}.png",
				show=show_plots,
			)

# --- Min/mean/max summaries (whole run) ---
if plot_min_mean_max and results:
	rows_min, rows_mean, rows_max = [], [], []
	for run_id in run_values:
		df = results.get(run_id)
		if df is None or df.empty:
			continue
		vals = df['T_hydro'].values
		rows_min.append({run_axis_col: run_id, 'T_hydro': float(np.nanmin(vals))})
		rows_mean.append({run_axis_col: run_id, 'T_hydro': float(np.nanmean(vals))})
		rows_max.append({run_axis_col: run_id, 'T_hydro': float(np.nanmax(vals))})
	min_df = pd.DataFrame(rows_min).dropna()
	mean_df = pd.DataFrame(rows_mean).dropna()
	max_df = pd.DataFrame(rows_max).dropna()
	if ANALYSIS_MODE == 'morfac':
		plot_Thydro_vs_morfac_min_mean_max(
			min_df=min_df,
			mean_df=mean_df,
			max_df=max_df,
			title="Thydro (12h-cycle) vs MORFAC (min/mean/max)",
			output_path=output_dir / f"Thydro_vs_MORFAC_cycle{int(cycle_hours)}h_min_mean_max.png",
			show=show_plots,
		)
	else:
		plot_Thydro_vs_run_min_mean_max(
			min_df=min_df,
			mean_df=mean_df,
			max_df=max_df,
			x_col=run_axis_col,
			xlabel=run_axis_label,
			title="Thydro (12h-cycle) vs scenario (min/mean/max)",
			output_path=output_dir / f"Thydro_vs_scenario_cycle{int(cycle_hours)}h_min_mean_max.png",
			show=show_plots,
		)

# --- Distribution plots (PDF + exceedance + fraction T>1) ---
if plot_distributions and results:
	for win in distribution_year_windows:
		if win is None:
			tag = "all"
			title_suffix = "(all cycles)"
		else:
			tag = f"year{win}"
			title_suffix = f"(years {win}-{win + 1})"

		if ANALYSIS_MODE == 'morfac':
			plot_exceedance(
				results,
				morfac_values=morfac_values,
				title=f"Exceedance of Thydro {title_suffix}",
				output_path=output_dir / f"Thydro_exceedance_cycle{int(cycle_hours)}h_{tag}.png",
				year_start=win,
				year_duration=1.0,
				show=show_plots,
			)
			plot_pdf(
				results,
				morfac_values=morfac_values,
				title=f"PDF of Thydro {title_suffix}",
				output_path=output_dir / f"Thydro_pdf_cycle{int(cycle_hours)}h_{tag}.png",
				year_start=win,
				year_duration=1.0,
				bins=30,
				show=show_plots,
			)
		else:
			plot_exceedance_by_run(
				results,
				run_values=run_values,
				run_labels=run_label_map,
				title=f"Exceedance of Thydro {title_suffix}",
				output_path=output_dir / f"Thydro_exceedance_cycle{int(cycle_hours)}h_{tag}.png",
				year_start=win,
				year_duration=1.0,
				show=show_plots,
			)
			plot_pdf_by_run(
				results,
				run_values=run_values,
				run_labels=run_label_map,
				title=f"PDF of Thydro {title_suffix}",
				output_path=output_dir / f"Thydro_pdf_cycle{int(cycle_hours)}h_{tag}.png",
				year_start=win,
				year_duration=1.0,
				bins=30,
				show=show_plots,
			)

print("✓ Done!")
