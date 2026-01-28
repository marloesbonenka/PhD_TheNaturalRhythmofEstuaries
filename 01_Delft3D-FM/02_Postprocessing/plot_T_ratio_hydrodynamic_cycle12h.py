"""
Compute and plot hydrodynamic T ratio on a 12-hour tidal-cycle basis.

Cycle-based definition (per 12h window):
- Qriver(t): mean upstream discharge over the same 12h window (riverward-positive)
- Qtide_hat(t): characteristic tidal discharge amplitude at the mouth within that window
  computed from demeaned mouth discharge using an amplitude estimator.
- T(t) = Qtide_hat(t) / Qriver(t)

This is intended for MORFAC-sensitivity analysis under variable river forcing.
Workflow mirrors the other scripts:
- Loop over MORFACs
- Cache results
- Plot T(t) per MORFAC
- Plot T vs MORFAC at morph-year windows (10–11, 30–31, 50–51)
- Plot min/mean/max summaries vs MORFAC
- Plot distributions (PDF + exceedance + fraction T>1) per MORFAC
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
)
from FUNCTIONS.F_cache import *
from FUNCTIONS.F_T_ratio_hydrodynamic_cycle12h import *


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
	# --- SETTINGS ---
	root_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC")
	scenario_dir = "03_flashy"
	tmorph_years = 400
	base_dir = root_dir / scenario_dir / f"Tmorph_{tmorph_years}years"

	morfac_values = [100,200,400]#[1, 2, 5, 10, 50]
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
	output_dir = base_dir / "output_plots_T_ratio_hydro_cycle12h"
	output_dir.mkdir(parents=True, exist_ok=True)
	cache_path = output_dir / "cached_results.pkl"

	compute = True
	show_plots = True

	results = {}
	run_names = {}

	if (not compute) and cache_path.exists():
		cached = load_cache(cache_path)
		results = cached.get('results', {})
		meta = cached.get('metadata', {})
		run_names = meta.get('run_names', {})
		print(f"Loaded cached results from: {cache_path}")

	else:
		if not compute:
			print(f"Cache not found, computing results: {cache_path}")

	if compute:
		for mf in morfac_values:
			run_folder, run_name = find_mf_run_folder(base_dir, mf)
			his_paths = get_his_paths_for_run(base_dir, run_folder)
			if not his_paths:
				raise FileNotFoundError(f"No HIS files found for MF={mf}")

			print(f"Processing MF={mf}: 12h-cycle hydrodynamic T...")
			data = load_hydro_cross_section_data(
				his_paths,
				estuary_only=estuary_only,
				km_range=km_range,
				exclude_last_timestep=True,
				exclude_last_n_days=exclude_last_n_days,
			)

			df = compute_T_hydro_cycle_based(
				data=data,
				morfac=mf,
				cycle_hours=cycle_hours,
				amp_method=amp_method,
				amp_percentiles=amp_percentiles,
			)

			results[mf] = df
			run_names[mf] = run_name

			csv_path = output_dir / f"{run_name}_Thydro_cycle{int(cycle_hours)}h.csv"
			df.to_csv(csv_path, index=False)

			if plot_time_series:
				plot_Thydro_timeseries(
					df,
					title=f"Hydrodynamic T ratio (12h-cycle) - MF{mf}",
					output_path=output_dir / f"{run_name}_Thydro_cycle{int(cycle_hours)}h_timeseries.png",
					show=show_plots,
				)

			data['ds'].close()

		save_cache(cache_path, {
			'results': results,
			'metadata': {
				'morfacs': morfac_values,
				'run_names': run_names,
				'cycle_hours': cycle_hours,
				'amp_method': amp_method,
				'amp_percentiles': amp_percentiles,
				'estuary_only': estuary_only,
				'km_range': km_range,
			}
		})
		print(f"Saved cached results to: {cache_path}")

	# --- MORFAC sensitivity plots (mean over year windows) ---
	if plot_T_vs_morfac_year_windows and results:
		for year_start in target_morph_years:
			rows = []
			for mf in morfac_values:
				df = results.get(mf)
				if df is None:
					continue
				T_mean_year = get_T_mean_over_year(df, year_start, col='T_hydro', year_duration=1.0)
				rows.append({'morfac': mf, 'T_hydro': T_mean_year})
			sens_df = pd.DataFrame(rows).dropna()
			plot_Thydro_vs_morfac(
				sens_df,
				title=f"Thydro (12h-cycle) vs MORFAC (mean over years {year_start}-{year_start + 1})",
				output_path=output_dir / f"Thydro_vs_MORFAC_cycle{int(cycle_hours)}h_mean_year{year_start}.png",
				show=show_plots,
			)

	# --- Min/mean/max summaries vs MORFAC (whole run) ---
	if plot_min_mean_max and results:
		rows_min, rows_mean, rows_max = [], [], []
		for mf in morfac_values:
			df = results.get(mf)
			if df is None or df.empty:
				continue
			vals = df['T_hydro'].values
			rows_min.append({'morfac': mf, 'T_hydro': float(np.nanmin(vals))})
			rows_mean.append({'morfac': mf, 'T_hydro': float(np.nanmean(vals))})
			rows_max.append({'morfac': mf, 'T_hydro': float(np.nanmax(vals))})
		min_df = pd.DataFrame(rows_min).dropna()
		mean_df = pd.DataFrame(rows_mean).dropna()
		max_df = pd.DataFrame(rows_max).dropna()
		plot_Thydro_vs_morfac_min_mean_max(
			min_df=min_df,
			mean_df=mean_df,
			max_df=max_df,
			title="Thydro (12h-cycle) vs MORFAC (min/mean/max)",
			output_path=output_dir / f"Thydro_vs_MORFAC_cycle{int(cycle_hours)}h_min_mean_max.png",
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

	print("✓ Done!")
