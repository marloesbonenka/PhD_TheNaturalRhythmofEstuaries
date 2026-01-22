"""
Compute and plot the T ratio (river volume / tidal prism) as a function of MORFAC.

Outputs:
1) T ratio evolution over morphological time for each MORFAC.
2) T ratio sensitivity vs MORFAC at a target morphological age.
"""

import sys
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent))

from FUNCTIONS.F_loaddata import (
	find_mf_run_folder,
	get_his_paths_for_run,
	load_cross_section_data,
)


def save_cache(cache_path: Path, payload: dict) -> None:
	cache_path.parent.mkdir(parents=True, exist_ok=True)
	with cache_path.open('wb') as f:
		pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(cache_path: Path) -> dict:
	with cache_path.open('rb') as f:
		return pickle.load(f)

# %%==========================================================================
# Helper functions
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


def _load_run_data(
	his_paths,
	*,
	estuary_only,
	km_range,
	exclude_last_timestep,
	exclude_last_n_days,
):
	return load_cross_section_data(
		his_paths,
		estuary_only=estuary_only,
		km_range=km_range,
		select_cycles_hydrodynamic=False,
		select_max_flood=False,
		select_max_flood_per_cycle=False,
		exclude_last_timestep=exclude_last_timestep,
		exclude_last_n_days=exclude_last_n_days,
	)


def _riverward_sign(q_upstream) -> float:
	sign = float(np.sign(np.nanmean(q_upstream)))
	return 1.0 if sign == 0 else sign


def integrate_volume(values, dt_seconds):
	"""Integrate discharge [m3/s] to volume [m3] over time."""
	values = np.asarray(values)
	if values.size == 0 or not np.isfinite(dt_seconds):
		return np.nan
	return np.trapz(values, dx=dt_seconds)


def compute_T_ratio_for_run(
	morfac,
	his_paths=None,
	cycle_hours=24,
	flood_sign=-1,
	estuary_only=True,
	km_range=(20, 45),
	exclude_last_timestep=True,
	exclude_last_n_days=0,
	data=None,
):
	"""
	Compute T ratio time series for one run.

	T = R / P
	R: river volume over one cycle (upstream cross-section)
	P: tidal prism (flood volume at mouth cross-section)
	"""
	loaded_here = False
	if data is None:
		if his_paths is None:
			raise ValueError("Provide either 'data' or 'his_paths'.")
		data = _load_run_data(
			his_paths,
			estuary_only=estuary_only,
			km_range=km_range,
			exclude_last_timestep=exclude_last_timestep,
			exclude_last_n_days=exclude_last_n_days,
		)
		loaded_here = True

	discharge = data['discharge']
	km_positions = data['km_positions']
	time_hours = data['time_hours']
	times = data['times_datetime']

	upstream_idx = int(np.argmax(km_positions))
	downstream_idx = int(np.argmin(km_positions))

	q_upstream = discharge.isel(cross_section=upstream_idx).values
	q_mouth = discharge.isel(cross_section=downstream_idx).values

	cycles, dt_hours = compute_cycle_indices(time_hours, cycle_hours=cycle_hours)
	dt_seconds = dt_hours * 3600.0

	if len(cycles) == 0:
		if loaded_here:
			data['ds'].close()
		return pd.DataFrame(columns=['morph_year', 'T_ratio', 'R_volume', 'P_volume'])

	river_sign = _riverward_sign(q_upstream)

	rows = []
	for start, end in cycles:
		q_up = q_upstream[start:end]
		q_m = q_mouth[start:end]

		R = integrate_volume(q_up * river_sign, dt_seconds)
		if flood_sign < 0:
			flood = -q_m[q_m < 0]
		else:
			flood = q_m[q_m > 0]
		P = integrate_volume(flood, dt_seconds)

		if P is None or not np.isfinite(P) or P <= 0:
			T = np.nan
		else:
			T = R / P

		mid_time_hours = 0.5 * (time_hours[start] + time_hours[end - 1])
		morph_year = (mid_time_hours / 24.0 / 365.0) * morfac

		rows.append({
			'morph_year': morph_year,
			'T_ratio': T,
			'R_volume': R,
			'P_volume': P,
			'time_mid': pd.to_datetime(times[start]) if len(times) else pd.NaT,
		})

	if loaded_here:
		data['ds'].close()
	return pd.DataFrame(rows)


def compute_T_ratio_profile_for_run(
	morfac,
	his_paths=None,
	cycle_hours=24,
	flood_sign=-1,
	estuary_only=True,
	km_range=(20, 45),
	exclude_last_timestep=True,
	exclude_last_n_days=0,
	data=None,
):
	"""
	Compute longitudinal T ratio profile (mean over cycles) for one run.

	For each cross-section:
	R = net river volume over a cycle (signed riverward)
	P = flood volume over a cycle (landward)
	T = R / P
	"""
	loaded_here = False
	if data is None:
		if his_paths is None:
			raise ValueError("Provide either 'data' or 'his_paths'.")
		data = _load_run_data(
			his_paths,
			estuary_only=estuary_only,
			km_range=km_range,
			exclude_last_timestep=exclude_last_timestep,
			exclude_last_n_days=exclude_last_n_days,
		)
		loaded_here = True

	discharge = data['discharge'].values  # shape: (time, cross_section)
	km_positions = data['km_positions']
	time_hours = data['time_hours']

	cycles, dt_hours = compute_cycle_indices(time_hours, cycle_hours=cycle_hours)
	dt_seconds = dt_hours * 3600.0

	if len(cycles) == 0:
		if loaded_here:
			data['ds'].close()
		return pd.DataFrame(columns=['km', 'T_mean', 'T_median', 'R_mean', 'P_mean'])

	# Determine riverward sign based on upstream cross-section
	upstream_idx = int(np.argmax(km_positions))
	river_sign = _riverward_sign(discharge[:, upstream_idx])

	T_cycles = []
	R_cycles = []
	P_cycles = []

	for start, end in cycles:
		q_slice = discharge[start:end, :]
		# Net river volume over cycle (signed riverward)
		R = np.trapz(q_slice * river_sign, dx=dt_seconds, axis=0)
		# Flood prism over cycle (landward only)
		if flood_sign < 0:
			flood = -q_slice
			flood[flood < 0] = 0.0
		else:
			flood = q_slice
			flood[flood < 0] = 0.0
		P = np.trapz(flood, dx=dt_seconds, axis=0)

		T = np.full_like(P, np.nan, dtype=float)
		valid = np.isfinite(P) & (P > 0)
		T[valid] = R[valid] / P[valid]

		T_cycles.append(T)
		R_cycles.append(R)
		P_cycles.append(P)

	T_cycles = np.vstack(T_cycles)
	R_cycles = np.vstack(R_cycles)
	P_cycles = np.vstack(P_cycles)

	profile = pd.DataFrame({
		'km': km_positions,
		'T_mean': np.nanmean(T_cycles, axis=0),
		'T_median': np.nanmedian(T_cycles, axis=0),
		'R_mean': np.nanmean(R_cycles, axis=0),
		'P_mean': np.nanmean(P_cycles, axis=0),
	})

	if loaded_here:
		data['ds'].close()
	return profile


def compute_transition_timeseries_for_run(
	morfac,
	his_paths=None,
	method='max_flood_penetration',
	cycle_hours=24,
	flood_sign=-1,
	estuary_only=True,
	km_range=(20, 45),
	exclude_last_timestep=True,
	exclude_last_n_days=0,
	min_prism_fraction=0.01,
	data=None,
):
	"""
	Compute a time series of the tide–river transition location per cycle.

	Methods
	- 'max_flood_penetration': location (km) of the maximum landward flood reach within each cycle.
	  This matches the logic used in your tidal-river dominance plotting: find where flood (inward) occurs.
	- 'T_equals_1': location (km) where T = R/P first crosses 1 (sea -> river) within each cycle.
	  Uses cycle-integrated net volume R and flood-only prism P at each cross-section.

	Returns DataFrame with columns: morph_year, transition_km
	"""
	loaded_here = False
	if data is None:
		if his_paths is None:
			raise ValueError("Provide either 'data' or 'his_paths'.")
		data = _load_run_data(
			his_paths,
			estuary_only=estuary_only,
			km_range=km_range,
			exclude_last_timestep=exclude_last_timestep,
			exclude_last_n_days=exclude_last_n_days,
		)
		loaded_here = True

	q = data['discharge'].values  # (time, cross_section)
	km_positions = np.asarray(data['km_positions'])
	time_hours = np.asarray(data['time_hours'])

	cycles, dt_hours = compute_cycle_indices(time_hours, cycle_hours=cycle_hours)
	dt_seconds = dt_hours * 3600.0

	if len(cycles) == 0:
		if loaded_here:
			data['ds'].close()
		return pd.DataFrame(columns=['morph_year', 'transition_km'])

	# Determine riverward sign based on upstream cross-section
	upstream_idx = int(np.argmax(km_positions))
	river_sign = _riverward_sign(q[:, upstream_idx])

	rows = []
	for start, end in cycles:
		q_slice = q[start:end, :]

		mid_time_hours = 0.5 * (time_hours[start] + time_hours[end - 1])
		morph_year = (mid_time_hours / 24.0 / 365.0) * morfac

		if method == 'max_flood_penetration':
			if flood_sign < 0:
				flood_mask = q_slice < 0
			else:
				flood_mask = q_slice > 0
			if not np.any(flood_mask):
				transition_km = np.nan
			else:
				km_grid = np.broadcast_to(km_positions, q_slice.shape)
				flood_km = np.where(flood_mask, km_grid, np.nan)
				max_flood_km_per_time = np.nanmax(flood_km, axis=1)
				transition_km = float(np.nanmax(max_flood_km_per_time))

		elif method == 'T_equals_1':
			R = np.trapz(q_slice * river_sign, dx=dt_seconds, axis=0)
			if flood_sign < 0:
				flood = -q_slice
				flood[flood < 0] = 0.0
			else:
				flood = q_slice
				flood[flood < 0] = 0.0
			P = np.trapz(flood, dx=dt_seconds, axis=0)

			# Mask sections with (near) zero prism to avoid division blow-ups
			P_max = np.nanmax(P) if np.any(np.isfinite(P)) else np.nan
			P_min = min_prism_fraction * P_max if np.isfinite(P_max) else np.nan
			valid = np.isfinite(P) & (P > 0)
			if np.isfinite(P_min):
				valid = valid & (P >= P_min)

			T = np.full_like(P, np.nan, dtype=float)
			T[valid] = R[valid] / P[valid]

			# Find first crossing from sea -> river where T >= 1
			transition_km = np.nan
			if np.any(np.isfinite(T)):
				idxs = np.where(np.isfinite(T))[0]
				# ensure ordered by km from sea to river
				order = np.argsort(km_positions[idxs])
				idxs = idxs[order]
				T_ord = T[idxs]
				km_ord = km_positions[idxs]
				cross = np.where(T_ord >= 1.0)[0]
				if cross.size > 0:
					j = int(cross[0])
					if j == 0:
						transition_km = float(km_ord[0])
					else:
						# linear interpolation between points around T=1
						x0, x1 = float(km_ord[j - 1]), float(km_ord[j])
						y0, y1 = float(T_ord[j - 1]), float(T_ord[j])
						if np.isfinite(y0) and np.isfinite(y1) and (y1 != y0):
							transition_km = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
						else:
							transition_km = float(km_ord[j])
		else:
			raise ValueError(f"Unknown method: {method}")

		rows.append({'morph_year': morph_year, 'transition_km': transition_km})

	if loaded_here:
		data['ds'].close()
	return pd.DataFrame(rows)


def discharge_amplitude(values, method='half_range', p_low=5, p_high=95):
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
		# for a perfect sinusoid, amplitude ~ std * sqrt(2)
		return float(np.nanstd(values) * np.sqrt(2.0))

	raise ValueError(f"Unknown amplitude method: {method}")


def compute_T_paper_Qtide_Qriver_for_run(
	morfac,
	his_paths=None,
	cycle_hours=24,
	estuary_only=True,
	km_range=(20, 45),
	exclude_last_timestep=True,
	exclude_last_n_days=0,
	amp_method='half_percentile_range',
	amp_percentiles=(5, 95),
	data=None,
):
	"""
	Compute paper-style T ratio per cycle:
	T = Q_tide_hat / Q_river_mean

	- Q_river_mean: mean discharge at most-upstream cross-section (riverward positive)
	- Q_tide_hat: characteristic tidal discharge amplitude at the mouth (demeaned, amplitude)

	Convention: T > 1 => tide-dominated (tidal discharge amplitude exceeds fluvial discharge).
	"""
	loaded_here = False
	if data is None:
		if his_paths is None:
			raise ValueError("Provide either 'data' or 'his_paths'.")
		data = _load_run_data(
			his_paths,
			estuary_only=estuary_only,
			km_range=km_range,
			exclude_last_timestep=exclude_last_timestep,
			exclude_last_n_days=exclude_last_n_days,
		)
		loaded_here = True

	discharge = data['discharge']
	km_positions = np.asarray(data['km_positions'])
	time_hours = np.asarray(data['time_hours'])

	upstream_idx = int(np.argmax(km_positions))
	downstream_idx = int(np.argmin(km_positions))

	q_upstream = discharge.isel(cross_section=upstream_idx).values
	q_mouth = discharge.isel(cross_section=downstream_idx).values

	cycles, dt_hours = compute_cycle_indices(time_hours, cycle_hours=cycle_hours)
	if len(cycles) == 0:
		if loaded_here:
			data['ds'].close()
		return pd.DataFrame(columns=['morph_year', 'T_ratio', 'Qriver_mean', 'Qtide_hat'])

	# Ensure riverward positive at upstream
	river_sign = _riverward_sign(q_upstream)

	p_low, p_high = amp_percentiles
	rows = []
	for start, end in cycles:
		q_up = q_upstream[start:end] * river_sign
		q_m = q_mouth[start:end]

		Qriver_mean = float(np.nanmean(q_up))

		# Demean mouth signal so amplitude reflects tidal oscillation not mean flow
		q_m_demeaned = q_m - np.nanmean(q_m)
		Qtide_hat = discharge_amplitude(
			q_m_demeaned,
			method=amp_method,
			p_low=p_low,
			p_high=p_high,
		)

		T = np.nan
		if np.isfinite(Qriver_mean) and Qriver_mean > 0 and np.isfinite(Qtide_hat):
			T = Qtide_hat / Qriver_mean

		mid_time_hours = 0.5 * (time_hours[start] + time_hours[end - 1])
		morph_year = (mid_time_hours / 24.0 / 365.0) * morfac

		rows.append({
			'morph_year': morph_year,
			'T_ratio': T,
			'Qriver_mean': Qriver_mean,
			'Qtide_hat': Qtide_hat,
		})

	if loaded_here:
		data['ds'].close()
	return pd.DataFrame(rows)


def get_T_at_morph_year(df, target_year):
	"""Interpolate T ratio at a target morphological year."""
	if df.empty:
		return np.nan

	sub = df[['morph_year', 'T_ratio']].dropna()
	if sub.empty:
		return np.nan

	x = sub['morph_year'].values
	y = sub['T_ratio'].values

	if target_year <= x.min():
		return y[np.argmin(x)]
	if target_year >= x.max():
		return y[np.argmax(x)]

	return np.interp(target_year, x, y)


def plot_all_figures(
	*,
	plot_time_series,
	plot_longitudinal_profile,
	plot_transition,
	plot_paper_T,
	all_results,
	profile_results,
	transition_results,
	paperT_results,
	sensitivity_rows,
	transition_sensitivity_rows,
	paperT_sensitivity_rows,
	output_dir: Path,
	target_morph_year: float,
	transition_method: str,
	show_plots: bool,
):
	"""Create all figures from precomputed result DataFrames and save them."""
	output_dir.mkdir(parents=True, exist_ok=True)

	# --- Plot 1: Evolution of T ratio over morphological time ---
	if plot_time_series and all_results:
		fig1, ax1 = plt.subplots(figsize=(10, 5))
		for mf, df in all_results.items():
			if df.empty:
				continue
			ax1.plot(df['morph_year'], df['T_ratio'], linewidth=2, label=f"MF {mf}")

		ax1.set_xlabel('Morphological time [years]', fontsize=11, fontweight='bold')
		ax1.set_ylabel('T ratio (R/P)', fontsize=11, fontweight='bold')
		ax1.set_title('Evolution of T ratio over morphological time', fontsize=12, fontweight='bold')
		ax1.grid(True, alpha=0.4, linestyle=':')
		ax1.legend(loc='best', fontsize=9)
		plt.tight_layout()
		fig1.savefig(output_dir / "T_ratio_evolution_over_time.png", dpi=300, bbox_inches='tight')
		if show_plots:
			plt.show()
		else:
			plt.close(fig1)

	# --- Plot 2: Sensitivity of T ratio vs MORFAC ---
	if plot_time_series and len(sensitivity_rows) > 1:
		sens_df = pd.DataFrame(sensitivity_rows).dropna()
		fig2, ax2 = plt.subplots(figsize=(6, 4))
		ax2.plot(sens_df['morfac'], sens_df['T_ratio'], 'o-', linewidth=2)
		ax2.set_xscale('log')
		ax2.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
		ax2.set_ylabel(f'T ratio at {target_morph_year} years', fontsize=11, fontweight='bold')
		ax2.set_title('T ratio sensitivity to MORFAC', fontsize=12, fontweight='bold')
		ax2.grid(True, alpha=0.4, linestyle=':')
		plt.tight_layout()
		fig2.savefig(output_dir / f"T_ratio_sensitivity_at_{target_morph_year}yr.png", dpi=300, bbox_inches='tight')
		if show_plots:
			plt.show()
		else:
			plt.close(fig2)

	# --- Plot 3: Longitudinal T ratio profile ---
	if plot_longitudinal_profile and profile_results:
		fig3, ax3 = plt.subplots(figsize=(10, 5))
		for mf, profile in profile_results.items():
			if profile.empty:
				continue
			ax3.plot(profile['km'], profile['T_mean'], linewidth=2, label=f"MF {mf}")

		ax3.set_xlabel('Distance from Sea [km]', fontsize=11, fontweight='bold')
		ax3.set_ylabel('T ratio (R/P)', fontsize=11, fontweight='bold')
		ax3.set_title('Longitudinal T ratio profile (mean over cycles)', fontsize=12, fontweight='bold')
		ax3.grid(True, alpha=0.4, linestyle=':')
		ax3.legend(loc='best', fontsize=9)
		plt.tight_layout()
		fig3.savefig(output_dir / "T_ratio_profile_longitudinal.png", dpi=300, bbox_inches='tight')
		if show_plots:
			plt.show()
		else:
			plt.close(fig3)

	# --- Plot 4: Transition location over morphological time ---
	if plot_transition and transition_results:
		fig4, ax4 = plt.subplots(figsize=(10, 5))
		for mf, df_tr in transition_results.items():
			if df_tr.empty:
				continue
			ax4.plot(df_tr['morph_year'], df_tr['transition_km'], linewidth=2, label=f"MF {mf}")
		ax4.set_xlabel('Morphological time [years]', fontsize=11, fontweight='bold')
		ax4.set_ylabel('Transition location [km from sea]', fontsize=11, fontweight='bold')
		ax4.set_title(f"Transition location over time ({transition_method})", fontsize=12, fontweight='bold')
		ax4.grid(True, alpha=0.4, linestyle=':')
		ax4.legend(loc='best', fontsize=9)
		plt.tight_layout()
		fig4.savefig(output_dir / f"transition_location_over_time_{transition_method}.png", dpi=300, bbox_inches='tight')
		if show_plots:
			plt.show()
		else:
			plt.close(fig4)

	# --- Plot 5: Transition sensitivity vs MORFAC at a target morph year ---
	if plot_transition and len(transition_sensitivity_rows) > 1:
		sens_tr = pd.DataFrame(transition_sensitivity_rows).dropna()
		fig5, ax5 = plt.subplots(figsize=(6, 4))
		ax5.plot(sens_tr['morfac'], sens_tr['transition_km'], 'o-', linewidth=2)
		ax5.set_xscale('log')
		ax5.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
		ax5.set_ylabel(f'Transition location at {target_morph_year} years [km]', fontsize=11, fontweight='bold')
		ax5.set_title(f"Transition sensitivity ({transition_method})", fontsize=12, fontweight='bold')
		ax5.grid(True, alpha=0.4, linestyle=':')
		plt.tight_layout()
		fig5.savefig(output_dir / f"transition_sensitivity_at_{target_morph_year}yr_{transition_method}.png", dpi=300, bbox_inches='tight')
		if show_plots:
			plt.show()
		else:
			plt.close(fig5)

	# --- Plot 6: Paper-style T = Qtide_hat / Qriver_mean over morphological time ---
	if plot_paper_T and paperT_results:
		fig6, ax6 = plt.subplots(figsize=(10, 5))
		for mf, df_p in paperT_results.items():
			if df_p.empty:
				continue
			ax6.plot(df_p['morph_year'], df_p['T_ratio'], linewidth=2, label=f"MF {mf}")
		ax6.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
		ax6.set_xlabel('Morphological time [years]', fontsize=11, fontweight='bold')
		ax6.set_ylabel(r'T ($\hat{Q}_{tide} / \bar{Q}_{river}$) [-]', fontsize=11, fontweight='bold')
		ax6.set_title('Paper-style T ratio over morphological time', fontsize=12, fontweight='bold')
		ax6.grid(True, alpha=0.4, linestyle=':')
		ax6.legend(loc='best', fontsize=9)
		plt.tight_layout()
		fig6.savefig(output_dir / "paper_T_over_time.png", dpi=300, bbox_inches='tight')
		if show_plots:
			plt.show()
		else:
			plt.close(fig6)

	# --- Plot 7: Paper-style T sensitivity vs MORFAC at target morph year ---
	if plot_paper_T and len(paperT_sensitivity_rows) > 1:
		sens_p = pd.DataFrame(paperT_sensitivity_rows).dropna()
		fig7, ax7 = plt.subplots(figsize=(6, 4))
		ax7.plot(sens_p['morfac'], sens_p['T_paper'], 'o-', linewidth=2)
		ax7.set_xscale('log')
		ax7.axhline(1.0, color='black', linewidth=1, linestyle='--', alpha=0.6)
		ax7.set_xlabel('MORFAC [-]', fontsize=11, fontweight='bold')
		ax7.set_ylabel(rf'T ($\hat{{Q}}_{{tide}} / \bar{{Q}}_{{river}}$) at {target_morph_year} years', fontsize=11, fontweight='bold')
		ax7.set_title('Paper-style T sensitivity to MORFAC', fontsize=12, fontweight='bold')
		ax7.grid(True, alpha=0.4, linestyle=':')
		plt.tight_layout()
		fig7.savefig(output_dir / f"paper_T_sensitivity_at_{target_morph_year}yr.png", dpi=300, bbox_inches='tight')
		if show_plots:
			plt.show()
		else:
			plt.close(fig7)


# %%==========================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
	# --- SETTINGS ---
	root_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Test_MORFAC")
	scenario_dir = "02_seasonal"  # 01_constant | 02_seasonal | 03_flashy
	tmorph_years = 50
	base_dir = root_dir / scenario_dir / f"Tmorph_{tmorph_years}years"

	# Start with a single MORFAC for a quick test
	morfac_values = [1, 2, 5, 10, 50]
	cycle_hours = 24
	flood_sign = -1  # negative discharge = flood
	exclude_last_n_days = 0
	target_morph_year = 50

	# Toggle between methods: 'time_series', 'longitudinal', or 'both'
	# For a first test, run everything for this single MORFAC
	analysis_mode = 'all'
	plot_time_series = analysis_mode in ('time_series', 'both', 'all')
	plot_longitudinal_profile = analysis_mode in ('longitudinal', 'both', 'all')
	plot_transition = analysis_mode in ('transition', 'all')
	plot_paper_T = analysis_mode in ('paper_T', 'all')

	# Paper-style discharge-ratio settings
	amp_method = 'half_percentile_range'  # 'half_range' | 'half_percentile_range' | 'std_sqrt2'
	amp_percentiles = (5, 95)

	# Transition settings (this is the plot you described as the goal)
	transition_method = 'max_flood_penetration'  # 'max_flood_penetration' | 'T_equals_1'
	min_prism_fraction = 0.01  # only used for 'T_equals_1' to prevent P~0 spikes

	estuary_only = True
	km_range = (20, 45)

	output_dir = base_dir / "output_plots_T_ratio"
	output_dir.mkdir(parents=True, exist_ok=True)
	cache_path = output_dir / "cached_results.pkl"

	# Rerun fast: set `compute=False` after first run.
	compute = True
	show_plots = True

	all_results = {}
	profile_results = {}
	transition_results = {}
	sensitivity_rows = []
	transition_sensitivity_rows = []
	paperT_results = {}
	paperT_sensitivity_rows = []

	if (not compute) and cache_path.exists():
		cached = load_cache(cache_path)
		all_results = cached.get('all_results', {})
		profile_results = cached.get('profile_results', {})
		transition_results = cached.get('transition_results', {})
		paperT_results = cached.get('paperT_results', {})
		sensitivity_rows = cached.get('sensitivity_rows', [])
		transition_sensitivity_rows = cached.get('transition_sensitivity_rows', [])
		paperT_sensitivity_rows = cached.get('paperT_sensitivity_rows', [])
		print(f"Loaded cached results from: {cache_path}")
	else:
		if not compute:
			print(f"Cache not found, computing results: {cache_path}")

	for mf in (morfac_values if compute else []):
		print(f"Processing MF={mf}...")
		run_folder, run_name = find_mf_run_folder(base_dir, mf)
		his_paths = get_his_paths_for_run(base_dir, run_folder)
		if not his_paths:
			print(f"  ✗ No HIS files found for MF={mf}")
			continue

		# Load once per run to speed up multiple analyses
		data = _load_run_data(
			his_paths,
			estuary_only=estuary_only,
			km_range=km_range,
			exclude_last_timestep=True,
			exclude_last_n_days=exclude_last_n_days,
		)

		if plot_time_series:
			df = compute_T_ratio_for_run(
				his_paths=None,
				morfac=mf,
				cycle_hours=cycle_hours,
				flood_sign=flood_sign,
				data=data,
			)
			all_results[mf] = df

			csv_path = output_dir / f"{run_name}_T_ratio_timeseries.csv"
			df.to_csv(csv_path, index=False)

			T_target = get_T_at_morph_year(df, target_morph_year)
			sensitivity_rows.append({'morfac': mf, 'T_ratio': T_target})

		if plot_longitudinal_profile:
			profile = compute_T_ratio_profile_for_run(
				his_paths=None,
				morfac=mf,
				cycle_hours=cycle_hours,
				flood_sign=flood_sign,
				data=data,
			)
			profile_results[mf] = profile
			profile_csv = output_dir / f"{run_name}_T_ratio_profile.csv"
			profile.to_csv(profile_csv, index=False)

		if plot_transition:
			transition_df = compute_transition_timeseries_for_run(
				his_paths=None,
				morfac=mf,
				method=transition_method,
				cycle_hours=cycle_hours,
				flood_sign=flood_sign,
				min_prism_fraction=min_prism_fraction,
				data=data,
			)
			transition_results[mf] = transition_df
			transition_csv = output_dir / f"{run_name}_transition_{transition_method}.csv"
			transition_df.to_csv(transition_csv, index=False)
			transition_at_target = get_T_at_morph_year(
				transition_df.rename(columns={'transition_km': 'T_ratio'}),
				target_morph_year,
			)
			transition_sensitivity_rows.append({'morfac': mf, 'transition_km': transition_at_target})

		if plot_paper_T:
			paper_df = compute_T_paper_Qtide_Qriver_for_run(
				his_paths=None,
				morfac=mf,
				cycle_hours=cycle_hours,
				amp_method=amp_method,
				amp_percentiles=amp_percentiles,
				data=data,
			)
			paperT_results[mf] = paper_df
			paper_csv = output_dir / f"{run_name}_Tpaper_Qtidehat_over_Qrivermean.csv"
			paper_df.to_csv(paper_csv, index=False)
			Tpaper_target = get_T_at_morph_year(paper_df, target_morph_year)
			paperT_sensitivity_rows.append({'morfac': mf, 'T_paper': Tpaper_target})

		# Close dataset after all computations for this run
		data['ds'].close()

	if compute:
		save_cache(cache_path, {
			'all_results': all_results,
			'profile_results': profile_results,
			'transition_results': transition_results,
			'paperT_results': paperT_results,
			'sensitivity_rows': sensitivity_rows,
			'transition_sensitivity_rows': transition_sensitivity_rows,
			'paperT_sensitivity_rows': paperT_sensitivity_rows,
			'metadata': {
				'target_morph_year': target_morph_year,
				'transition_method': transition_method,
				'morfac_values': morfac_values,
				'analysis_mode': analysis_mode,
			}
		})
		print(f"Saved cached results to: {cache_path}")

	plot_all_figures(
		plot_time_series=plot_time_series,
		plot_longitudinal_profile=plot_longitudinal_profile,
		plot_transition=plot_transition,
		plot_paper_T=plot_paper_T,
		all_results=all_results,
		profile_results=profile_results,
		transition_results=transition_results,
		paperT_results=paperT_results,
		sensitivity_rows=sensitivity_rows,
		transition_sensitivity_rows=transition_sensitivity_rows,
		paperT_sensitivity_rows=paperT_sensitivity_rows,
		output_dir=output_dir,
		target_morph_year=target_morph_year,
		transition_method=transition_method,
		show_plots=show_plots,
	)

	print("✓ Done!")
