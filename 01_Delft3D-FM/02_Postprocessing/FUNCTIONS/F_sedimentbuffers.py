"""Shared helper functions for sediment buffer postprocessing scripts."""

from pathlib import Path

import numpy as np

from FUNCTIONS.F_loaddata import load_and_cache_scenario, get_stitched_his_paths
from FUNCTIONS.F_general import find_variability_model_folders


def tidal_avg(arr, window):
    """Centered moving average over `window` samples."""
    return np.convolve(arr, np.ones(window) / window, mode="same")


def load_sedimentbuffer_runs(
    base_path,
    cache_dir,
    discharge,
    variability_map,
    boxes,
    var_name,
    analyze_noisy,
    scenario_filter=None,
):
    """
    Load sediment-buffer time series for all matching run folders.

    Returns dict:
      folder_name -> {
          'time': array,
          'buffers': {(start, end): array},
      }
    """
    base_path = Path(base_path)
    cache_dir = Path(cache_dir)

    timed_out_dir = base_path / "timed-out"
    if not timed_out_dir.exists():
        timed_out_dir = None

    folders = find_variability_model_folders(
        base_path=base_path,
        discharge=discharge,
        scenarios_to_process=scenario_filter,
        analyze_noisy=analyze_noisy,
    )

    cache_dir.mkdir(exist_ok=True)
    runs = {}
    for folder in folders:
        his_paths = get_stitched_his_paths(
            base_path=base_path,
            folder_name=folder,
            timed_out_dir=timed_out_dir,
            variability_map=variability_map,
            analyze_noisy=analyze_noisy,
        )
        if not his_paths:
            print(f"[WARNING] No HIS files found for {folder.name}, skipping.")
            continue

        scenario_num = folder.name.split("_")[0]
        run_id = "_".join(folder.name.split("_")[1:])
        cache_file = cache_dir / f"hisoutput_{int(scenario_num)}_{run_id}.nc"

        _, data = load_and_cache_scenario(
            scenario_dir=folder,
            his_file_paths=his_paths,
            cache_file=cache_file,
            boxes=boxes,
            var_name=var_name,
        )

        runs[folder.name] = {
            "time": data["t"],
            "buffers": data["buffer_volumes"],
        }

    return runs


def align_runs_to_common_time(base_time, runs_dict, box_key):
    """
    Interpolate all runs in runs_dict onto base_time for one box key.

    Returns array of shape (n_runs, len(base_time)).
    """
    aligned = []
    for run_data in runs_dict.values():
        t = run_data["time"]
        buf = run_data["buffers"].get(box_key)
        if buf is None:
            continue

        if np.issubdtype(t.dtype, np.datetime64):
            t_f = t.astype("datetime64[ns]").astype(np.float64)
            t_base_f = base_time.astype("datetime64[ns]").astype(np.float64)
        else:
            t_f = t.astype(np.float64)
            t_base_f = base_time.astype(np.float64)

        aligned.append(np.interp(t_base_f, t_f, buf))

    return np.array(aligned)


def trim_to_end(time_array, t_end_min):
    """Boolean mask for values at or before the shared simulation end time."""
    return time_array <= t_end_min
