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
	load_cross_section_data,
)
from FUNCTIONS.F_cache import save_cache, load_cache


# =============================================================================
# Core helpers
# =============================================================================

def compute_cycle_indices(time_hours, cycle_hours=12.0):
	"""Return list of (start, end) indices for fixed-length cycles."""
	time_hours = np.asarray(time_hours)
	if len(time_hours) < 2:
		return [], np.nan

	dt_hours = np.nanmedian(np.diff(time_hours))
	if not np.isfinite(dt_hours) or dt_hours <= 0:
		return [], np.nan

	steps_per_cycle = int(np.round(float(cycle_hours) / dt_hours))
	if steps_per_cycle < 1:
		return [], dt_hours

	indices = []
	for start in range(0, len(time_hours), steps_per_cycle):
		end = start + steps_per_cycle
		if end > len(time_hours):
			break
		indices.append((start, end))
	return indices, dt_hours


def _riverward_sign(values) -> float:
	sign = float(np.sign(np.nanmean(values)))
	return 1.0 if sign == 0 else sign


def discharge_amplitude(values, method='half_percentile_range', p_low=5, p_high=95):
	"""Estimate tidal discharge amplitude from a time series within one cycle."""
	values = np.asarray(values)
	values = values[np.isfinite(values)]
	if values.size == 0:
		return np.nan

	if method == 'half_range':
		return 0.5 * (np.nanmax(values) - np.nanmin(values))
	if method == 'half_percentile_range':
		lo, hi = np.nanpercentile(values, [p_low, p_high])
		return 0.5 * (hi - lo)
	if method == 'std_sqrt2':
		return float(np.nanstd(values) * np.sqrt(2.0))

	raise ValueError(f"Unknown amplitude method: {method}")


def get_T_mean_over_year(df, year_start, *, col='T_hydro', year_duration=1.0, fallback_to_interp=True):
	"""Average a cycle-based T-series over a morphological-year window."""
	if df.empty or col not in df.columns:
		return np.nan

	sub = df[['morph_year', col]].dropna()
	if sub.empty:
		return np.nan

	lo = float(year_start)
	hi = float(year_start + year_duration)
	mask = (sub['morph_year'].values >= lo) & (sub['morph_year'].values < hi)
	vals = sub.loc[mask, col].values
	if vals.size == 0:
		if not fallback_to_interp:
			return np.nan
		# fallback: nearest-by-time interpolation at year_start
		x = sub['morph_year'].values
		y = sub[col].values
		if year_start <= x.min():
			return float(y[np.argmin(x)])
		if year_start >= x.max():
			return float(y[np.argmax(x)])
		return float(np.interp(year_start, x, y))

	return float(np.nanmean(vals))


def _select_window_values(df, *, year_start=None, year_duration=1.0, col='T_hydro'):
	"""Return finite values for a full-run or a year window."""
	if df.empty or col not in df.columns:
		return np.array([])

	sub = df[['morph_year', col]].dropna()
	if sub.empty:
		return np.array([])

	if year_start is None:
		vals = sub[col].values
		return vals[np.isfinite(vals)]

	lo = float(year_start)
	hi = float(year_start + year_duration)
	mask = (sub['morph_year'].values >= lo) & (sub['morph_year'].values < hi)
	vals = sub.loc[mask, col].values
	return vals[np.isfinite(vals)]


def load_hydro_cross_section_data(
	his_paths,
	*,
	estuary_only=True,
	km_range=(20, 45),
	exclude_last_timestep=True,
	exclude_last_n_days=0,
):
	return load_cross_section_data(
		his_paths,
		q_var='cross_section_discharge',
		estuary_only=estuary_only,
		km_range=km_range,
		select_cycles_hydrodynamic=False,
		select_max_flood=False,
		select_max_flood_per_cycle=False,
		exclude_last_timestep=exclude_last_timestep,
		exclude_last_n_days=exclude_last_n_days,
	)


def compute_T_hydro_cycle_based(
	*,
	data,
	morfac,
	cycle_hours=12.0,
	amp_method='half_percentile_range',
	amp_percentiles=(5, 95),
):
	"""Compute hydrodynamic T ratio per fixed-length cycle."""
	discharge = data['discharge']
	km_positions = np.asarray(data['km_positions'])
	time_hours = np.asarray(data['time_hours'])

	upstream_idx = int(np.argmax(km_positions))
	downstream_idx = int(np.argmin(km_positions))

	q_upstream = discharge.isel(cross_section=upstream_idx).values
	q_mouth = discharge.isel(cross_section=downstream_idx).values

	cycles, _ = compute_cycle_indices(time_hours, cycle_hours=cycle_hours)
	if len(cycles) == 0:
		return pd.DataFrame(columns=['morph_year', 'T_hydro', 'Qriver_mean', 'Qtide_hat'])

	river_sign = _riverward_sign(q_upstream)
	p_low, p_high = amp_percentiles

	rows = []
	for start, end in cycles:
		q_up = q_upstream[start:end] * river_sign
		q_m = q_mouth[start:end]

		Qriver_mean = float(np.nanmean(q_up))
		q_m_demeaned = q_m - np.nanmean(q_m)
		Qtide_hat = discharge_amplitude(q_m_demeaned, method=amp_method, p_low=p_low, p_high=p_high)

		T = np.nan
		if np.isfinite(Qriver_mean) and Qriver_mean > 0 and np.isfinite(Qtide_hat):
			T = Qtide_hat / Qriver_mean

		mid_time_hours = 0.5 * (time_hours[start] + time_hours[end - 1])
		morph_year = (mid_time_hours / 24.0 / 365.0) * morfac

		rows.append({
			'morph_year': morph_year,
			'T_hydro': T,
			'Qriver_mean': Qriver_mean,
			'Qtide_hat': Qtide_hat,
		})

	return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def plot_Thydro_timeseries(df, *, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(10, 5))
	if not df.empty:
		ax.plot(df['morph_year'], df['T_hydro'], linewidth=2)
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('Morphological time [years]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ (12h-cycle) [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_Thydro_vs_morfac(df, *, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(6, 4))
	if not df.empty:
		ax.plot(df['morfac'], df['T_hydro'], 'o-', linewidth=2)
	ax.set_xscale('log')
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_Thydro_vs_morfac_min_mean_max(*, min_df, mean_df, max_df, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(6, 4))
	if not min_df.empty:
		ax.plot(min_df['morfac'], min_df['T_hydro'], 'o-', linewidth=2, label='min')
	if not mean_df.empty:
		ax.plot(mean_df['morfac'], mean_df['T_hydro'], 'o-', linewidth=2, label='mean')
	if not max_df.empty:
		ax.plot(max_df['morfac'], max_df['T_hydro'], 'o-', linewidth=2, label='max')
	ax.set_xscale('log')
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{hydro}$ [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_exceedance(results, *, morfac_values, title, output_path: Path, year_start=None, year_duration=1.0, show=False):
	fig, ax = plt.subplots(figsize=(7, 4))
	for mf in morfac_values:
		df = results.get(mf)
		vals = _select_window_values(df, year_start=year_start, year_duration=year_duration, col='T_hydro')
		vals = vals[np.isfinite(vals)]
		if vals.size == 0:
			continue
		vals = np.sort(vals)
		n = vals.size
		exc = 1.0 - (np.arange(1, n + 1) / n)
		ax.plot(vals, exc, linewidth=2, label=f"MF{mf}")
	ax.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('T [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel('Exceedance P(T > x) [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_pdf(results, *, morfac_values, title, output_path: Path, year_start=None, year_duration=1.0, bins=30, show=False):
	fig, ax = plt.subplots(figsize=(7, 4))
	for mf in morfac_values:
		df = results.get(mf)
		vals = _select_window_values(df, year_start=year_start, year_duration=year_duration, col='T_hydro')
		if vals.size == 0:
			continue
		ax.hist(vals, bins=bins, density=True, histtype='step', linewidth=2, label=f"MF{mf}")
	ax.axvline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('T [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel('PDF [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	ax.legend(loc='best', fontsize=9)
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def plot_fraction_T_gt1(results, *, morfac_values, title, output_path: Path, year_start=None, year_duration=1.0, show=False):
	rows = []
	for mf in morfac_values:
		df = results.get(mf)
		vals = _select_window_values(df, year_start=year_start, year_duration=year_duration, col='T_hydro')
		if vals.size == 0:
			continue
		rows.append({'morfac': mf, 'frac_T_gt1': float(np.mean(vals > 1.0))})
	out = pd.DataFrame(rows)

	fig, ax = plt.subplots(figsize=(6, 4))
	if not out.empty:
		ax.plot(out['morfac'], out['frac_T_gt1'], 'o-', linewidth=2)
	ax.set_xscale('log')
	ax.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel('Fraction of cycles with T > 1 [-]', fontsize=11, fontweight='bold')
	ax.set_ylim(0, 1)
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
	# --- SETTINGS ---
	root_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC")
	scenario_dir = "02_seasonal"
	tmorph_years = 50
	base_dir = root_dir / scenario_dir / f"Tmorph_{tmorph_years}years"

	morfac_values = [1, 2, 5, 10, 50]
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
			plot_fraction_T_gt1(
				results,
				morfac_values=morfac_values,
				title=f"Fraction cycles with Thydro > 1 {title_suffix}",
				output_path=output_dir / f"Thydro_frac_Tgt1_cycle{int(cycle_hours)}h_{tag}.png",
				year_start=win,
				year_duration=1.0,
				show=show_plots,
			)

	print("✓ Done!")
