"""
Compute and plot sedimentary T ratio (kg/s) using cross-section sediment fluxes.

T_sed = S_tide_hat / S_river_mean

Designed for quick testing on a single MORFAC (default MF50) with the
bedload cross-section variable.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
	script_dir = Path(__file__).resolve().parent
except NameError:
	script_dir = Path.cwd()

for candidate in (script_dir, script_dir / "01_Delft3D-FM" / "02_Postprocessing"):
	if (candidate / "FUNCTIONS").exists():
		sys.path.append(str(candidate))
		break

from FUNCTIONS.F_loaddata import *
from FUNCTIONS.F_cache import load_results_cache, save_results_cache

# =============================================================================
# Core helpers (loading and computations)
# =============================================================================

def compute_cycle_indices(time_hours, cycle_hours=24):
	"""Return list of (start, end) indices for each cycle."""
	time_hours = np.asarray(time_hours)
	if len(time_hours) < 2:
		return [], np.nan

	dt_hours = np.nanmedian(np.diff(time_hours))
	if not np.isfinite(dt_hours) or dt_hours <= 0:
		return [], np.nan

	steps_per_cycle = int(np.round(cycle_hours / dt_hours))
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


def discharge_amplitude(values, method='half_range', p_low=5, p_high=95):
	"""Estimate tidal amplitude from a time series within one cycle."""
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


def _sediment_flux_rate(values, time_hours, mode='instantaneous'):
	"""Return sediment flux rate time series [volume per time] from raw output."""
	values = np.asarray(values, dtype=float)
	if mode == 'instantaneous':
		return values
	if mode == 'cumulative':
		if values.size == 0 or len(time_hours) < 2:
			return np.full_like(values, np.nan, dtype=float)
		dt_seconds = np.nanmedian(np.diff(time_hours)) * 3600.0
		if not np.isfinite(dt_seconds) or dt_seconds <= 0:
			return np.full_like(values, np.nan, dtype=float)
		return np.gradient(values, dt_seconds, axis=0)
	raise ValueError(f"Unknown sediment flux mode: {mode}")


def _to_mass_flux(volume_flux, units='m3/s', sediment_density=2650.0, porosity=0.4):
	"""Convert volumetric sediment flux to mass flux [kg/s]."""
	units = units.lower()
	if units in ('kg/s', 'kg s-1', 'kgs-1'):
		return volume_flux
	if units in ('m3/s', 'm^3/s', 'm3 s-1', 'm^3 s-1'):
		return volume_flux * sediment_density * (1.0 - porosity)
	raise ValueError(f"Unsupported sediment flux units: {units}")


def load_sediment_cross_section_data(
	his_paths,
	*,
	q_var,
	estuary_only=True,
	km_range=(20, 45),
	exclude_last_timestep=True,
	exclude_last_n_days=0,
	dataset_cache=None,
):
	return load_cross_section_data(
		his_paths,
		q_var=q_var,
		estuary_only=estuary_only,
		km_range=km_range,
		select_cycles_hydrodynamic=False,
		select_max_flood=False,
		select_max_flood_per_cycle=False,
		exclude_last_timestep=exclude_last_timestep,
		exclude_last_n_days=exclude_last_n_days,
		dataset_cache=dataset_cache,
	)


def compute_T_sediment_Qtide_Qriver(
	*,
	data,
	morfac,
	cycle_hours=24,
	amp_method='half_percentile_range',
	amp_percentiles=(5, 95),
	sed_flux_mode='instantaneous',
	sed_flux_units='m3/s',
	sediment_density=2650.0,
	porosity=0.4,
):
	"""
	Compute sedimentary T ratio per cycle (kg/s):
	T_sed = S_tide_hat / S_river_mean
	"""
	sed = data['discharge']
	km_positions = np.asarray(data['km_positions'])
	time_hours = np.asarray(data['time_hours'])

	upstream_idx = int(np.argmax(km_positions))
	downstream_idx = int(np.argmin(km_positions))

	s_upstream = sed.isel(cross_section=upstream_idx).values
	s_mouth = sed.isel(cross_section=downstream_idx).values

	s_up_rate = _sediment_flux_rate(s_upstream, time_hours, mode=sed_flux_mode)
	s_m_rate = _sediment_flux_rate(s_mouth, time_hours, mode=sed_flux_mode)
	S_up = _to_mass_flux(s_up_rate, units=sed_flux_units, sediment_density=sediment_density, porosity=porosity)
	S_m = _to_mass_flux(s_m_rate, units=sed_flux_units, sediment_density=sediment_density, porosity=porosity)

	cycles, _ = compute_cycle_indices(time_hours, cycle_hours=cycle_hours)
	if len(cycles) == 0:
		return pd.DataFrame(columns=['morph_year', 'T_sed', 'Sriver_mean', 'Stide_hat'])

	river_sign = _riverward_sign(S_up)
	rows = []
	p_low, p_high = amp_percentiles
	for start, end in cycles:
		s_up = S_up[start:end] * river_sign
		s_m = S_m[start:end]

		Sriver_mean = float(np.nanmean(s_up))
		s_m_demeaned = s_m - np.nanmean(s_m)
		Stide_hat = discharge_amplitude(
			s_m_demeaned,
			method=amp_method,
			p_low=p_low,
			p_high=p_high,
		)

		T_sed = np.nan
		if np.isfinite(Sriver_mean) and Sriver_mean > 0 and np.isfinite(Stide_hat):
			T_sed = Stide_hat / Sriver_mean

		mid_time_hours = 0.5 * (time_hours[start] + time_hours[end - 1])
		morph_year = (mid_time_hours / 24.0 / 365.0) * morfac

		rows.append({
			'morph_year': morph_year,
			'T_sed': T_sed,
			'Sriver_mean': Sriver_mean,
			'Stide_hat': Stide_hat,
		})

	return pd.DataFrame(rows)


# =============================================================================
# Plotting (separate and easy to adjust)
# =============================================================================

def plot_Tsed_timeseries(df, *, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(10, 5))
	if not df.empty:
		ax.plot(df['morph_year'], df['T_sed'], linewidth=2)
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('Morphological time [years]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{sed}$ (S$_{tide}$ / S$_{river}$) [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


def get_T_at_morph_year(df, target_year):
	"""Interpolate T_sed at a target morphological year."""
	if df.empty:
		return np.nan

	sub = df[['morph_year', 'T_sed']].dropna()
	if sub.empty:
		return np.nan

	x = sub['morph_year'].values
	y = sub['T_sed'].values

	if target_year <= x.min():
		return y[np.argmin(x)]
	if target_year >= x.max():
		return y[np.argmax(x)]

	return np.interp(target_year, x, y)


def plot_Tsed_vs_morfac(df, *, title, output_path: Path, show=False):
	fig, ax = plt.subplots(figsize=(6, 4))
	if not df.empty:
		ax.plot(df['morfac'], df['T_sed'], 'o-', linewidth=2)
	ax.set_xscale('log')
	ax.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
	ax.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
	ax.set_ylabel(r'T$_{sed}$ [-]', fontsize=11, fontweight='bold')
	ax.set_title(title, fontsize=12, fontweight='bold')
	ax.grid(True, alpha=0.4, linestyle=':')
	plt.tight_layout()
	fig.savefig(output_path, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


# =============================================================================
# MAIN (MF50 quick test)
# =============================================================================

if __name__ == '__main__':
	root_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC")
	scenario_dir = "02_seasonal"
	tmorph_years = 50
	base_dir = root_dir / scenario_dir / f"Tmorph_{tmorph_years}years"

	# MORFACs to test
	morfac_values = [1, 2, 5, 10, 50]
	cycle_hours = 24
	exclude_last_n_days = 0

	# Plot toggles
	plot_time_series = True
	plot_T_vs_morfac = True
	target_morph_years = [10, 30, 50]

	# Sediment settings
	sediment_density = 2650.0
	porosity = 0.4
	amp_method = 'half_percentile_range'
	amp_percentiles = (5, 95)

	# Variable to test
	var_name = 'cross_section_bedload_sediment_transport'
	flux_mode = 'cumulative'
	flux_units = 'm3/s'

	estuary_only = True
	km_range = (20, 45)

	output_dir = base_dir / "output_plots_T_ratio_sediment"
	output_dir.mkdir(parents=True, exist_ok=True)
	cache_path = output_dir / "cached_results.pkl"

	compute = False
	show_plots = True

	results = {}
	run_names = {}

	if not compute:
		loaded_results, loaded_meta = load_results_cache(cache_path)
		if loaded_results is not None:
			results = loaded_results
			run_names = loaded_meta.get('run_names', {})
			print(f"Loaded cached results from: {cache_path}")
			if results and plot_time_series:
				for mf in morfac_values:
					df = results.get(mf)
					if df is None:
						continue
					run_name_cached = run_names.get(mf, f"MF{mf}")
					plot_path = output_dir / f"{run_name_cached}_Tsed_{var_name}.png"
					plot_Tsed_timeseries(
						df,
						title=f"Sedimentary T ratio ({var_name})",
						output_path=plot_path,
						show=show_plots,
					)
		else:
			print(f"Cache not found, computing results...")

	if compute or not results:
		for mf in morfac_values:
			run_folder, run_name = find_mf_run_folder(base_dir, mf)
			his_paths = get_his_paths_for_run(base_dir, run_folder)
			if not his_paths:
				raise FileNotFoundError(f"No HIS files found for MF={mf}")

			print(f"Processing MF={mf}: {var_name} (mode={flux_mode}, units={flux_units})...")
			data = load_sediment_cross_section_data(
				his_paths,
				q_var=var_name,
				estuary_only=estuary_only,
				km_range=km_range,
				exclude_last_timestep=True,
				exclude_last_n_days=exclude_last_n_days,
			)
			df = compute_T_sediment_Qtide_Qriver(
				data=data,
				morfac=mf,
				cycle_hours=cycle_hours,
				amp_method=amp_method,
				amp_percentiles=amp_percentiles,
				sed_flux_mode=flux_mode,
				sed_flux_units=flux_units,
				sediment_density=sediment_density,
				porosity=porosity,
			)
			results[mf] = df
			run_names[mf] = run_name

			csv_path = output_dir / f"{run_name}_Tsed_{var_name}.csv"
			df.to_csv(csv_path, index=False)

			if plot_time_series:
				plot_path = output_dir / f"{run_name}_Tsed_{var_name}.png"
				plot_Tsed_timeseries(
					df,
					title=f"Sedimentary T ratio ({var_name})",
					output_path=plot_path,
					show=show_plots,
				)

			data['ds'].close()

		save_results_cache(
			cache_path,
			results,
			metadata={
				'morfacs': morfac_values,
				'run_names': run_names,
				'run_name': run_name,
				'variable': var_name,
				'flux_mode': flux_mode,
				'flux_units': flux_units,
				'amp_method': amp_method,
				'amp_percentiles': amp_percentiles,
			}
		)
		print(f"Saved cached results to: {cache_path}")

	if plot_T_vs_morfac and results:
		for target_morph_year in target_morph_years:
			rows = []
			for mf in morfac_values:
				df = results.get(mf)
				if df is None:
					continue
				T_target = get_T_at_morph_year(df, target_morph_year)
				rows.append({'morfac': mf, 'T_sed': T_target})
			sens_df = pd.DataFrame(rows).dropna()
			plot_path = output_dir / f"Tsed_vs_MORFAC_{var_name}_at_{target_morph_year}yr.png"
			plot_Tsed_vs_morfac(
				sens_df,
				title=f"Tsed vs MORFAC at {target_morph_year} years",
				output_path=plot_path,
				show=show_plots,
			)

		# Also plot min/max across time for each MORFAC
		rows_min = []
		rows_max = []
		for mf in morfac_values:
			df = results.get(mf)
			if df is None or df.empty:
				continue
			rows_min.append({'morfac': mf, 'T_sed': float(np.nanmin(df['T_sed']))})
			rows_max.append({'morfac': mf, 'T_sed': float(np.nanmax(df['T_sed']))})
		min_df = pd.DataFrame(rows_min).dropna()
		max_df = pd.DataFrame(rows_max).dropna()
		plot_Tsed_vs_morfac(
			min_df,
			title="Tsed vs MORFAC (minimum over time)",
			output_path=output_dir / f"Tsed_vs_MORFAC_{var_name}_min.png",
			show=show_plots,
		)
		plot_Tsed_vs_morfac(
			max_df,
			title="Tsed vs MORFAC (maximum over time)",
			output_path=output_dir / f"Tsed_vs_MORFAC_{var_name}_max.png",
			show=show_plots,
		)

	print("✓ Done!")
