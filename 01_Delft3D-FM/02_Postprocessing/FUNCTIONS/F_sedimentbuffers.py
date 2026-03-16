"""Shared helper functions for sediment buffer postprocessing scripts."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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

def run_envelope_workflow(
    base_directory,
    config,
    discharge,
    output_dirname,
    boxes,
    sed_var,
    variability_map,
    scenario_labels,
    scenario_colors,
    base_scenario='1',
    envelope_plot_mode='all',
):
    """Run noisy-envelope plots for sediment buffer volumes."""
    scenario_config = {
        "1": {"color": scenario_colors["1"], "label": scenario_labels["1"]},
        "2": {"color": scenario_colors["2"], "label": scenario_labels["2"]},
        "3": {"color": scenario_colors["3"], "label": scenario_labels["3"]},
        "4": {"color": scenario_colors["4"], "label": scenario_labels["4"]},
    }

    base_directory = Path(base_directory)
    variability_base_path = base_directory / config
    variability_cache_dir = variability_base_path / "cached_data"
    noisy_base_path = variability_base_path / f"0_Noise_Q{discharge}"
    noisy_cache_dir = noisy_base_path / "cached_data"

    if envelope_plot_mode == 'noise_only':
        envelope_output_dir = noisy_base_path / output_dirname
    else:
        envelope_output_dir = variability_base_path / output_dirname
    envelope_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Envelope output dir: {envelope_output_dir}")

    base_runs = load_sedimentbuffer_runs(
        base_path=variability_base_path,
        cache_dir=variability_cache_dir,
        discharge=discharge,
        variability_map=variability_map,
        boxes=boxes,
        var_name=sed_var,
        analyze_noisy=False,
        scenario_filter={int(base_scenario)},
    )
    if not base_runs:
        raise FileNotFoundError(
            f"No run found for base scenario {base_scenario} in {variability_base_path}."
        )

    base_data = list(base_runs.values())[0]
    base_time = base_data['time']
    base_buffers = base_data['buffers']
    base_cfg = scenario_config[base_scenario]

    variability_runs = {}
    if envelope_plot_mode == 'all':
        other_nums = {int(k) for k in scenario_config if k != base_scenario}
        all_var_runs = load_sedimentbuffer_runs(
            base_path=variability_base_path,
            cache_dir=variability_cache_dir,
            discharge=discharge,
            variability_map=variability_map,
            boxes=boxes,
            var_name=sed_var,
            analyze_noisy=False,
            scenario_filter=other_nums,
        )
        for folder_name, run_data in all_var_runs.items():
            scenario_num = str(int(folder_name.split('_')[0]))
            if scenario_num not in scenario_config:
                continue
            variability_runs[scenario_num] = {
                'time': run_data['time'],
                'buffers': run_data['buffers'],
                'label': scenario_config[scenario_num]['label'],
                'color': scenario_config[scenario_num]['color'],
            }

    if noisy_base_path.exists():
        noisy_runs = load_sedimentbuffer_runs(
            base_path=noisy_base_path,
            cache_dir=noisy_cache_dir,
            discharge=discharge,
            variability_map=variability_map,
            boxes=boxes,
            var_name=sed_var,
            analyze_noisy=True,
            scenario_filter=None,
        )
    else:
        noisy_runs = {}
        print(f"[INFO] No noisy base path found: {noisy_base_path}")

    print(f"Found {len(noisy_runs)} noisy runs")

    if envelope_plot_mode == 'noise_only' and not noisy_runs:
        raise FileNotFoundError(
            f"No noisy runs found in {noisy_base_path}. "
            f"Set envelope_plot_mode='all' or add noisy runs."
        )
    if envelope_plot_mode == 'all' and not noisy_runs:
        print("[INFO] No noisy runs found; plotting base + variability scenarios without noisy envelope.")

    all_times = [base_time]
    all_times += [r['time'] for r in noisy_runs.values()]
    all_times += [r['time'] for r in variability_runs.values()]
    t_end_min = min(t[-1] for t in all_times)
    print(f"Shortest simulation end time across all runs: {t_end_min}")

    for box_key in boxes:
        box_start, box_end = box_key
        fig, ax = plt.subplots()

        if noisy_runs:
            base_time_trimmed = base_time[trim_to_end(base_time, t_end_min)]
            noisy_stack = align_runs_to_common_time(base_time_trimmed, noisy_runs, box_key)

            base_buf_for_env = base_buffers.get(box_key)
            if base_buf_for_env is not None:
                base_row = base_buf_for_env[trim_to_end(base_time, t_end_min)][np.newaxis, :]
                if noisy_stack.size > 0:
                    noisy_stack = np.vstack([noisy_stack, base_row])
                else:
                    noisy_stack = base_row

            if noisy_stack.size > 0:
                env_min = np.nanmin(noisy_stack, axis=0)
                env_max = np.nanmax(noisy_stack, axis=0)

                for i, run_data in enumerate(noisy_runs.values()):
                    buf = run_data['buffers'].get(box_key)
                    if buf is None:
                        continue
                    mask = trim_to_end(run_data['time'], t_end_min)
                    ax.plot(
                        run_data['time'][mask],
                        buf[mask],
                        color='grey',
                        alpha=0.35,
                        linewidth=0.7,
                        label='Noisy runs' if i == 0 else None,
                    )

                ax.fill_between(
                    base_time_trimmed,
                    env_min,
                    env_max,
                    color='grey',
                    alpha=0.2,
                    label='Noisy envelope',
                )

        if envelope_plot_mode == 'all':
            for run_data in variability_runs.values():
                buf = run_data['buffers'].get(box_key)
                if buf is None:
                    continue
                mask = trim_to_end(run_data['time'], t_end_min)
                ax.plot(
                    run_data['time'][mask],
                    buf[mask],
                    color=run_data['color'],
                    linewidth=1.2,
                    label=run_data['label'],
                )

        base_buf = base_buffers.get(box_key)
        if base_buf is not None:
            mask = trim_to_end(base_time, t_end_min)
            ax.plot(
                base_time[mask],
                base_buf[mask],
                color=base_cfg['color'],
                linewidth=1.8,
                label=base_cfg['label'],
            )

        ax.set_xlabel('Time')
        ax.set_ylabel('Buffer Volume (m³)')
        ax.set_title(f'Sediment buffer volume — {box_start}–{box_end} km')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = envelope_output_dir / f"Q{discharge}_sedimentbuffer_box_{box_start}_{box_end}km.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {fig_path}")
        plt.show()