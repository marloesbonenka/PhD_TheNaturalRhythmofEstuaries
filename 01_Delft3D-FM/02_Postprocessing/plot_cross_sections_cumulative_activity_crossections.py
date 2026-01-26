"""Cumulative activity on cross-section bed profiles (from MAP), using cross-section geometry (from HIS).

This answers:
- You have cross-sections defined in the HIS (cross_section geom coords)
- You do NOT want point/station data
- You want to extract bed profiles from MAP (like plot_cross_sections_BI.py)
- Then plot only the *first* profile + a cumulative-activity heatmap above it.

For each selected cross-section:
- Sample bedlevel along the transect (mesh2d_mor_bl) for every output time
- Compute cumulative activity: Σ|Δz| over time at each transect point
- Plot:
  (top) heatmap of cumulative activity (time vs cross-section distance)
  (bottom) first bedlevel profile (t=0)

Notes
-----
- This uses the same restart stitching pattern as plot_cross_sections_BI.py
- For long runs, set `time_stride` to reduce runtime/memory.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import dfm_tools as dfmt
from scipy.spatial import cKDTree
from tqdm import tqdm

# Add path for FUNCTIONS
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")

from FUNCTIONS.F_general import get_mf_number
from FUNCTIONS.F_cache import DatasetCache
from FUNCTIONS.F_braiding_index import get_bed_profile


# =============================================================================
# Configuration
# =============================================================================

base_directory = r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15"
config = r"Test_MORFAC\02_seasonal\Tmorph_50years"  # folder containing MF* runs

# Cross-sections to analyze (3)
selected_cross_sections = [
	"ObservationCrossSection_Estuary_km20",
	"ObservationCrossSection_Estuary_km33",
	"ObservationCrossSection_Estuary_km44",
]

# Bedlevel variable in MAP
map_bedlevel_var = "mesh2d_mor_bl"

# Masking similar to BI script
bedlevel_land_threshold = 8.0  # set None to disable

# Time control
# - Use stride > 1 to reduce runtime (e.g., stride=10 keeps every 10th timestep)
time_stride = 1

# Morph-time conversion
# If your run_startdate differs, set it here. If None, uses first timestamp per run.
run_startdate = None  # e.g. "2025-01-01"
use_folder_morfac = True  # if True, uses MF number from folder name

# Output
output_dirname = "output_plots_crosssections_cumactivity"


# =============================================================================
# Helpers
# =============================================================================

def cumulative_activity(profiles_time_space: np.ndarray) -> np.ndarray:
	"""Σ|Δz| along the time axis; returns array same shape (time, space)."""
	z = np.asarray(profiles_time_space)
	if z.ndim != 2:
		raise ValueError("Expected 2D array (time, space)")
	dz = np.diff(z, axis=0)
	abs_dz = np.abs(dz)
	zeros = np.zeros((1, z.shape[1]))
	abs_dz0 = np.vstack([zeros, abs_dz])
	return np.cumsum(abs_dz0, axis=0)


def morph_years_from_datetimes(times: pd.DatetimeIndex, *, startdate=None, morfac=1.0) -> np.ndarray:
	if startdate is None:
		t0 = times[0]
	else:
		t0 = pd.Timestamp(startdate)
	hydro_years = np.array([(t - t0).total_seconds() / (365.25 * 24 * 3600) for t in times])
	return hydro_years * float(morfac)


def get_cross_section_geom(ds_his: xr.Dataset, cs_name: str):
	cs_names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in ds_his.cross_section_name.values]
	if cs_name not in cs_names:
		return None
	idx = cs_names.index(cs_name)
	start = int(ds_his['cross_section_geom_node_count'].values[:idx].sum())
	end = start + int(ds_his['cross_section_geom_node_count'].values[idx])
	x = ds_his['cross_section_geom_node_coordx'].values[start:end]
	y = ds_his['cross_section_geom_node_coordy'].values[start:end]
	dist = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)
	return x, y, dist


def plot_activity_and_first_profile(*, dist_m, first_profile, cumact, morph_years, title, outpath: Path, show=False):
	fig = plt.figure(figsize=(10, 7))
	gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)
	ax0 = fig.add_subplot(gs[0, 0])
	ax1 = fig.add_subplot(gs[1, 0])

	x_km = dist_m / 1000.0
	extent = [x_km.min(), x_km.max(), morph_years.min(), morph_years.max()]
	vmax = np.nanpercentile(cumact, 98) if np.any(np.isfinite(cumact)) else 1.0

	im = ax0.imshow(
		cumact,
		aspect='auto',
		origin='lower',
		extent=extent,
		cmap='viridis',
		vmin=0,
		vmax=vmax,
	)
	cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
	cbar.set_label(r"$\Sigma |\Delta z_b|$ [m]", fontsize=11, fontweight='bold')

	ax0.set_ylabel('Morphological time [years]', fontsize=11, fontweight='bold')
	ax0.set_title(title, fontsize=12, fontweight='bold')

	ax1.plot(x_km, first_profile, color='black', linewidth=2)
	ax1.set_xlabel('Cross-section distance [km]', fontsize=11, fontweight='bold')
	ax1.set_ylabel('Bed level [m]', fontsize=11, fontweight='bold')
	ax1.grid(True, alpha=0.3, linestyle=':')

	plt.tight_layout()
	fig.savefig(outpath, dpi=300, bbox_inches='tight')
	if show:
		plt.show()
	else:
		plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
	base_path = Path(base_directory) / config
	if not base_path.exists():
		raise FileNotFoundError(f"Base path not found: {base_path}")

	dataset_cache = DatasetCache()

	output_dir = base_path / output_dirname
	output_dir.mkdir(parents=True, exist_ok=True)

	timed_out_dir = base_path / 'timed-out'
	model_folders = [f for f in os.listdir(base_path) if f.startswith('MF') and (base_path / f).is_dir()]
	model_folders.sort(key=get_mf_number)

	print(f"Found {len(model_folders)} run folders in: {base_path}")

	for folder in model_folders:
		run_dir = base_path / folder

		# --- restart stitching (same pattern as BI script) ---
		all_run_paths = []
		if 'restart' in folder.lower() and timed_out_dir.exists():
			mf_prefix = folder.split('_')[0]  # e.g., MF50
			matches = [p for p in os.listdir(timed_out_dir) if p.startswith(mf_prefix)]
			if matches:
				all_run_paths.append(timed_out_dir / matches[0])
		all_run_paths.append(run_dir)

		print("\n" + "=" * 60)
		print(f"PROCESSING: {folder}")
		print(f"Stitching {len(all_run_paths)} parts")

		loaded_datasets = []
		loaded_trees = []
		ds_his = None

		try:
			# Load MAP parts + KDTree
			for p_path in all_run_paths:
				map_pattern = str(p_path / 'output' / '*_map.nc')
				ds_map = dataset_cache.get_partitioned(map_pattern, chunks={'time': 1})
				if map_bedlevel_var not in ds_map:
					raise KeyError(f"{map_bedlevel_var} not found in MAP for {p_path}")

				face_x = ds_map['mesh2d_face_x'].values
				face_y = ds_map['mesh2d_face_y'].values
				tree = cKDTree(np.vstack([face_x, face_y]).T)

				loaded_datasets.append(ds_map)
				loaded_trees.append(tree)

			# Load HIS from last part (cross-section geometry)
			his_path = all_run_paths[-1] / 'output' / 'FlowFM_0000_his.nc'
			ds_his = dataset_cache.get_xr(his_path)

			# Determine MORFAC
			if use_folder_morfac:
				morfac = float(get_mf_number(folder))
			else:
				morfac = 1.0

			# Process each selected cross-section
			for cs_name in selected_cross_sections:
				geom = get_cross_section_geom(ds_his, cs_name)
				if geom is None:
					print(f"  [WARN] Cross-section not found in HIS: {cs_name}")
					continue
				cs_x, cs_y, dist = geom

				all_times = []
				all_profiles = []

				# Extract profiles sequentially over stitched parts
				for ds_map, tree in zip(loaded_datasets, loaded_trees):
					time_vals = pd.to_datetime(ds_map.time.values)
					idxs = range(0, len(time_vals), int(time_stride))
					for t in tqdm(idxs, desc=f"  {cs_name} timesteps", leave=False):
						profile = get_bed_profile(ds_map, tree, cs_x, cs_y, t)
						if bedlevel_land_threshold is not None:
							profile = profile.copy()
							profile[profile > float(bedlevel_land_threshold)] = np.nan
						all_times.append(time_vals[t])
						all_profiles.append(profile)

				# Clean up overlaps between restart parts
				df_idx = pd.DataFrame({'time': all_times, 'p_idx': np.arange(len(all_profiles))})
				df_idx = df_idx.drop_duplicates('time').sort_values('time')

				profiles_clean = [all_profiles[int(i)] for i in df_idx['p_idx'].values]
				times_clean = pd.to_datetime(df_idx['time'].values)

				Z = np.vstack([p[None, :] for p in profiles_clean])
				cum = cumulative_activity(Z)

				morph_years = morph_years_from_datetimes(times_clean, startdate=run_startdate, morfac=morfac)
				first_profile = Z[0, :]

				outpath = output_dir / f"{folder}_{cs_name}_cumactivity.png"
				plot_activity_and_first_profile(
					dist_m=dist,
					first_profile=first_profile,
					cumact=cum,
					morph_years=morph_years,
					title=f"{folder}: {cs_name}",
					outpath=outpath,
					show=True,
				)
				print(f"  Saved: {outpath}")

		except Exception as e:
			print(f"Error processing {folder}: {e}")
		finally:
			plt.close('all')

	dataset_cache.close_all()

	print("\n" + "=" * 60)
	print("ALL FOLDERS COMPLETED.")
