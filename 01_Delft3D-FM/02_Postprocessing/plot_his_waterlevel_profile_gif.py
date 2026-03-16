"""
Create GIF(s) of along-estuary water-level profiles from station-point data.

X-axis: estuary distance [km from sea]
Y-axis: water level [m]
Frames: selected timesteps through the simulation

Uses separated station cache files (hisoutput_stations_*.nc) when available,
with fallback to HIS files.
"""
#%%
import sys
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import xarray as xr

# Force white style for consistency with other scripts
plt.style.use('default')
mpl.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\02_Postprocessing")
from FUNCTIONS.F_general import get_variability_map, find_variability_model_folders
from FUNCTIONS.F_tidalrange_currentspeed import load_station_waterlevels_from_cache_or_his, decode_name_array

#%%
# =============================================================================
# CONFIG
# =============================================================================
DISCHARGE = 500
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']
OUTPUT_DIRNAME = 'plots_his_waterlevel_profiles'

WATERLEVEL_VAR = 'waterlevel'
BEDLEVEL_VAR = 'bedlevel'
STATION_PATTERN = r'^Observation(?:Point|CrossSection)_Estuary_km(\d+)$'
EXCLUDE_LAST_TIMESTEP = True

# Animation settings
FRAME_STRIDE = 1         # base stride within selected 1-day window
FPS = 10
FIGSIZE = (9, 5)
LINEWIDTH = 2.0
MARKERSIZE = 4
MAX_FRAMES_PER_GIF = 240

# Snapshot settings: create one 1-day GIF for each target moment
SNAPSHOT_COUNT = 6
EVENT_YEAR = 2055
GIF_HALF_WINDOW = np.timedelta64(60, 'h')  # 5-day window centered on the selected event

DISCHARGE_CSV_BASE = Path(
    r"U:\PhDNaturalRhythmEstuaries\Models\ModelBoundaries_50hydroyears"
)
DISCHARGE_CSV_RELATIVE = Path('boundaryfiles_csv') / 'discharge_cumulative.csv'

VARIABILITY_MAP = get_variability_map(DISCHARGE)

SCENARIO_LABELS = {
    '1': 'Constant',
    '2': 'Seasonal',
    '3': 'Flashy',
    '4': 'Single peak',
}

SCENARIO_COLORS = {
    '1': '#1f77b4',
    '2': '#ff7f0e',
    '3': '#2ca02c',
    '4': '#d62728',
}

# Keep bed line color consistent across all plots/GIFs.
BEDLEVEL_COLOR = '#b8860b'


# =============================================================================
# PATHS + RUN DISCOVERY
# =============================================================================
base_directory = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15")
config = f"Model_Output/Q{DISCHARGE}"
base_path = base_directory / config

output_dir = base_path / 'output_plots' / OUTPUT_DIRNAME
output_dir.mkdir(parents=True, exist_ok=True)

timed_out_dir = base_path / 'timed-out'
if not timed_out_dir.exists():
    timed_out_dir = None

folders = find_variability_model_folders(
    base_path=base_path,
    discharge=DISCHARGE,
    scenarios_to_process=SCENARIOS_TO_PROCESS,
    analyze_noisy=False,
)
model_folders = [f.name for f in folders]

run_his_paths = {}
for folder in model_folders:
    model_location = base_path / folder
    his_paths = []
    scenario_key = str(int(folder.split('_')[0]))

    if timed_out_dir is not None:
        timed_out_folder = VARIABILITY_MAP.get(scenario_key, folder)
        timed_out_path = timed_out_dir / timed_out_folder / 'output' / 'FlowFM_0000_his.nc'
        if timed_out_path.exists():
            his_paths.append(timed_out_path)

    main_his_path = model_location / 'output' / 'FlowFM_0000_his.nc'
    if main_his_path.exists():
        his_paths.append(main_his_path)

    if his_paths:
        run_his_paths[folder] = his_paths
    else:
        print(f"[WARNING] No HIS files found for {folder}")

cache_dir = base_path / 'cached_data'
cache_dir.mkdir(exist_ok=True)


# =============================================================================
# GIF CREATION
# =============================================================================
def _read_discharge_csv(csv_path):
    """Read discharge CSV with timestamp in col 1 and discharge [m3/s] in col 2."""
    timestamps = []
    discharges = []

    with csv_path.open('r', newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue

            raw_t = row[0].strip()
            raw_q = row[1].strip()
            if not raw_t or not raw_q:
                continue

            try:
                q = float(raw_q)
                t = np.datetime64(raw_t).astype('datetime64[ns]')
            except Exception:
                # Skip header rows and malformed lines.
                continue

            discharges.append(q)
            timestamps.append(t)

    if not timestamps:
        raise ValueError(f'No valid discharge data found in: {csv_path}')

    times_ns = np.asarray(timestamps, dtype='datetime64[ns]')
    q_arr = np.asarray(discharges, dtype=float)

    order = np.argsort(times_ns)
    return times_ns[order], q_arr[order]


def _local_extrema_indices(values):
    """Return local peak and trough indices based on direct neighbours."""
    if len(values) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    prev_v = values[:-2]
    mid_v = values[1:-1]
    next_v = values[2:]

    peak_mask = (mid_v >= prev_v) & (mid_v > next_v)
    trough_mask = (mid_v <= prev_v) & (mid_v < next_v)

    peak_idx = np.where(peak_mask)[0] + 1
    trough_idx = np.where(trough_mask)[0] + 1
    return peak_idx, trough_idx


def _nearest_index(values, target):
    return int(np.argmin(np.abs(values - target)))


def _build_event_targets(times, discharge, count=SNAPSHOT_COUNT):
    """Build exactly `count` representative event dates for one scenario."""
    if len(times) == 0:
        return []

    q = np.asarray(discharge, dtype=float)
    t = np.asarray(times).astype('datetime64[ns]')
    peak_idx, trough_idx = _local_extrema_indices(q)

    idx_max = int(np.argmax(q))
    idx_min = int(np.argmin(q))
    idx_q90 = _nearest_index(q, np.nanquantile(q, 0.90))
    idx_q10 = _nearest_index(q, np.nanquantile(q, 0.10))
    idx_q75 = _nearest_index(q, np.nanquantile(q, 0.75))

    q50 = np.nanquantile(q, 0.50)
    centered_gradient = np.gradient(q)
    normal_score = np.abs(q - q50) + 0.2 * np.abs(centered_gradient)
    normal_ranked_idx = np.argsort(normal_score)

    selected = []
    used_idx = set()

    def _append_first_available(label, idx_candidates):
        for idx in idx_candidates:
            idx = int(idx)
            if idx not in used_idx:
                selected.append((label, idx))
                used_idx.add(idx)
                return True
        return False

    _append_first_available('peak_main', [idx_max])
    _append_first_available('minimum_main', [idx_min])
    _append_first_available('high_q90', [idx_q90])
    _append_first_available('low_q10', [idx_q10])
    _append_first_available('high_q75', [idx_q75])
    _append_first_available('normal', normal_ranked_idx)

    candidates = []

    if len(peak_idx) > 0:
        for i in peak_idx[np.argsort(q[peak_idx])[::-1]]:
            candidates.append((f'peak_local_{len(candidates)}', int(i)))

    if len(trough_idx) > 0:
        for i in trough_idx[np.argsort(q[trough_idx])]:
            candidates.append((f'trough_local_{len(candidates)}', int(i)))

    # Add evenly spread fallback indices so we always end up with `count` unique targets.
    spread_idx = np.linspace(0, len(t) - 1, num=max(count, 2), dtype=int)
    for i in spread_idx:
        candidates.append((f'spread_{i}', int(i)))

    for label, idx in candidates:
        if idx in used_idx:
            continue
        selected.append((label, idx))
        used_idx.add(idx)
        if len(selected) >= count:
            break

    return [{'label': label, 'target_dt': t[idx], 'idx': idx, 'q': float(q[idx])} for label, idx in selected]


def _select_hydrodynamic_window(times, target_dt, half_window=GIF_HALF_WINDOW):
    """Select one contiguous window around the nearest timestep to target date."""
    times_ns = np.asarray(times).astype('datetime64[ns]')
    if len(times_ns) == 0:
        return np.array([], dtype=int), None, None

    target_ns = np.datetime64(target_dt, 'ns')
    nearest_idx = int(np.argmin(np.abs(times_ns - target_ns)))
    center = times_ns[nearest_idx]
    window_start = center - half_window
    window_end = center + half_window

    if window_start < times_ns[0]:
        shift = times_ns[0] - window_start
        window_start = times_ns[0]
        window_end = window_end + shift
    if window_end > times_ns[-1]:
        shift = window_end - times_ns[-1]
        window_end = times_ns[-1]
        window_start = window_start - shift

    mask = (times_ns >= window_start) & (times_ns <= window_end)
    idx = np.where(mask)[0]
    return idx, window_start, window_end


def _normalize_station_name(name):
    cleaned = str(name).replace('\x00', '').strip()
    return ''.join(ch for ch in cleaned if ch.isprintable())


def _load_station_bedlevels_from_his(
    his_file_paths,
    station_labels,
    bedlevel_var=BEDLEVEL_VAR,
    exclude_last_timestep=True,
):
    """Load bedlevel for selected station labels and stitch across HIS parts."""
    with xr.open_dataset(his_file_paths[0]) as ds0:
        if 'station_name' not in ds0:
            raise KeyError('station_name not found in HIS file')
        if bedlevel_var not in ds0:
            raise KeyError(f"{bedlevel_var} not found in HIS file")

        station_names = [_normalize_station_name(s) for s in decode_name_array(ds0['station_name'].values)]

    label_to_idx = {name: i for i, name in enumerate(station_names)}
    station_idx = []
    missing = []
    for label in station_labels:
        key = _normalize_station_name(label)
        if key in label_to_idx:
            station_idx.append(label_to_idx[key])
        else:
            missing.append(label)

    if missing:
        raise KeyError(f"Could not map {len(missing)} station label(s) to HIS station_name for bedlevel")

    station_idx = np.asarray(station_idx, dtype=int)

    bl_parts = []
    time_parts = []
    last_time = None

    for p in his_file_paths:
        with xr.open_dataset(p) as ds:
            bl = ds[bedlevel_var].isel(station=station_idx)
            tt = ds['time'].values

            if last_time is not None and len(tt) > 1:
                dt = tt[1] - tt[0]
                offset = (last_time - tt[0]) + dt
                tt = tt + offset

            bl_parts.append(bl.values)
            time_parts.append(tt)
            if len(tt) > 0:
                last_time = tt[-1]

    bl_all = np.concatenate(bl_parts, axis=0)
    t_all = np.concatenate(time_parts)

    if exclude_last_timestep and len(t_all) > 1:
        bl_all = bl_all[:-1, :]
        t_all = t_all[:-1]

    return {
        'times': np.asarray(t_all).astype('datetime64[ns]'),
        'bedlevel': bl_all,
    }


def _align_to_reference_times(ref_times, series_times, series_values):
    """Align series to reference timestamps using exact match when possible, else nearest."""
    ref_ns = np.asarray(ref_times).astype('datetime64[ns]')
    s_ns = np.asarray(series_times).astype('datetime64[ns]')

    if len(ref_ns) == len(s_ns) and np.all(ref_ns == s_ns):
        return series_values

    idx = np.searchsorted(s_ns, ref_ns)
    idx = np.clip(idx, 0, len(s_ns) - 1)

    left_idx = np.clip(idx - 1, 0, len(s_ns) - 1)
    pick_left = np.abs(ref_ns - s_ns[left_idx]) <= np.abs(s_ns[idx] - ref_ns)
    idx = np.where(pick_left, left_idx, idx)

    return series_values[idx, :]


def create_profile_gif(*, km, wl, bed, times, scenario_key, label, color, gif_path, y_lim):
    """Create a water-level profile GIF for one scenario."""
    if wl.size == 0 or len(times) == 0:
        raise ValueError('No water-level data available for GIF creation')

    # Keep memory bounded by limiting total number of rendered frames.
    adaptive_stride = max(FRAME_STRIDE, int(np.ceil(len(times) / max(1, MAX_FRAMES_PER_GIF))))
    frame_idx = np.arange(0, len(times), adaptive_stride, dtype=int)
    if frame_idx[-1] != len(times) - 1:
        frame_idx = np.append(frame_idx, len(times) - 1)

    wl_frames = wl[frame_idx, :]
    bed_frames = bed[frame_idx, :] if bed is not None else np.full_like(wl_frames, np.nan)
    t_frames = times[frame_idx]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    wl_line, = ax.plot([], [], color=color, linewidth=LINEWIDTH, marker='o', markersize=MARKERSIZE, label='Water level')
    bed_line, = ax.plot([], [], color=BEDLEVEL_COLOR, linewidth=LINEWIDTH, linestyle='--', label='Bed level')

    ax.set_xlim(float(np.nanmin(km)), float(np.nanmax(km)))
    ax.set_ylim(*y_lim)
    ax.set_xlabel('Estuary distance [km from sea]')
    ax.set_ylabel('Level [m]')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')

    title = ax.set_title('')

    def _init():
        wl_line.set_data([], [])
        bed_line.set_data([], [])
        title.set_text(f"{label} - water/bed profile")
        return wl_line, bed_line, title

    def _update(i):
        wl_line.set_data(km, wl_frames[i, :])
        bed_line.set_data(km, bed_frames[i, :])
        t_hours = (t_frames[i] - t_frames[0]) / np.timedelta64(1, 'h')
        title.set_text(f"{label} - water/bed profile | t={t_hours:.1f} hours")
        return wl_line, bed_line, title

    ani = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(frame_idx),
        interval=max(1, int(1000 / FPS)),
        blit=True,
    )

    writer = animation.PillowWriter(fps=FPS)
    ani.save(gif_path, writer=writer)
    plt.close(ani._fig)
    plt.close(fig)


def _extract_year(times, values, year):
    years = times.astype('datetime64[Y]').astype(int) + 1970
    mask = years == int(year)
    return times[mask], values[mask]


scenario_payload = {}
global_wl_min = np.inf
global_wl_max = -np.inf

for folder in model_folders:
    scenario_key = str(int(folder.split('_')[0]))
    run_id = '_'.join(folder.split('_')[1:])

    his_file_paths = run_his_paths.get(folder)
    if his_file_paths is None:
        continue

    station_cache_file = cache_dir / f"hisoutput_stations_{int(scenario_key)}_{run_id}.nc"

    print(f"\n[SCENARIO {scenario_key}] {folder}")
    print(f"  station cache: {station_cache_file.name}")

    wl_data = load_station_waterlevels_from_cache_or_his(
        cache_file=station_cache_file,
        his_file_paths=his_file_paths,
        waterlevel_var=WATERLEVEL_VAR,
        station_pattern=STATION_PATTERN,
        exclude_last_timestep=EXCLUDE_LAST_TIMESTEP,
    )

    km = wl_data['station_km']
    wl = wl_data['waterlevel']
    station_labels = wl_data['station_labels']
    times = wl_data['times']
    times_ns = np.asarray(times).astype('datetime64[ns]')

    try:
        bed_data = _load_station_bedlevels_from_his(
            his_file_paths=his_file_paths,
            station_labels=station_labels,
            bedlevel_var=BEDLEVEL_VAR,
            exclude_last_timestep=EXCLUDE_LAST_TIMESTEP,
        )
        bed = _align_to_reference_times(times_ns, bed_data['times'], bed_data['bedlevel'])
    except Exception as exc:
        print(f"  [WARNING] Could not load bedlevel from HIS ({BEDLEVEL_VAR}): {exc}")
        bed = np.full_like(wl, np.nan)

    scenario_label = SCENARIO_LABELS.get(scenario_key, scenario_key)
    scenario_color = SCENARIO_COLORS.get(scenario_key, 'grey')

    csv_folder = VARIABILITY_MAP.get(scenario_key, folder)
    discharge_csv = DISCHARGE_CSV_BASE / csv_folder / DISCHARGE_CSV_RELATIVE
    if not discharge_csv.exists():
        print(f"  [WARNING] Discharge CSV not found: {discharge_csv}")
        continue

    q_times, q_vals = _read_discharge_csv(discharge_csv)
    q_times_yr, q_vals_yr = _extract_year(q_times, q_vals, EVENT_YEAR)
    if len(q_times_yr) < 2:
        print(f"  [WARNING] Not enough discharge data in {EVENT_YEAR} for {folder}")
        continue

    events = _build_event_targets(q_times_yr, q_vals_yr, count=SNAPSHOT_COUNT)
    print(f"  Selected {len(events)} discharge-based event(s) for {EVENT_YEAR}.")

    windows = []
    for event in events:
        idx_win, t0_win, t1_win = _select_hydrodynamic_window(times_ns, event['target_dt'], half_window=GIF_HALF_WINDOW)
        if idx_win.size < 2:
            print(f"  [SKIP] event {event['label']}: not enough timesteps in selected 5-day window")
            continue

        wl_win = wl[idx_win, :]
        bed_win = bed[idx_win, :]
        finite_vals = np.concatenate([
            wl_win[np.isfinite(wl_win)],
            bed_win[np.isfinite(bed_win)],
        ])
        if finite_vals.size > 0:
            global_wl_min = min(global_wl_min, float(np.min(finite_vals)))
            global_wl_max = max(global_wl_max, float(np.max(finite_vals)))

        windows.append({
            'event': event,
            'idx': idx_win,
            't0': t0_win,
            't1': t1_win,
        })

    if windows:
        scenario_payload[folder] = {
            'scenario_key': scenario_key,
            'scenario_label': scenario_label,
            'scenario_color': scenario_color,
            'km': km,
            'wl': wl,
            'bed': bed,
            'times_ns': times_ns,
            'windows': windows,
        }

if not np.isfinite(global_wl_min) or not np.isfinite(global_wl_max):
    raise RuntimeError('Could not determine global y-axis limits from selected windows.')

pad = 0.05 * max(1e-6, (global_wl_max - global_wl_min))
global_y_lim = (global_wl_min - pad, global_wl_max + pad)

for folder, payload in scenario_payload.items():
    scenario_key = payload['scenario_key']
    scenario_label = payload['scenario_label']
    scenario_color = payload['scenario_color']
    km = payload['km']
    wl = payload['wl']
    bed = payload['bed']
    times_ns = payload['times_ns']
    windows = payload['windows']

    print(f"\n[SCENARIO {scenario_key}] Creating {len(windows)} GIF(s) with shared y-axis {global_y_lim}")
    for win in windows:
        event = win['event']
        idx_win = win['idx']
        t0_win = win['t0']
        t1_win = win['t1']

        wl_win = wl[idx_win, :]
        bed_win = bed[idx_win, :]
        times_win = times_ns[idx_win]

        event_day = str(np.datetime_as_string(event['target_dt'], unit='D'))
        event_tag = f"{event['label']}_{event_day}".replace(':', '-').replace(' ', '_')
        gif_path = output_dir / f"waterlevel_profile_Q{DISCHARGE}_{scenario_key}_{event_tag}.gif"

        print(
            f"  [GIF] event={event['label']} | Q={event['q']:.2f} m3/s | "
            f"target={event_day} | window={t0_win} -> {t1_win} | steps={len(times_win)}"
        )
        create_profile_gif(
            km=km,
            wl=wl_win,
            bed=bed_win,
            times=times_win,
            scenario_key=scenario_key,
            label=f"{scenario_label} ({event['label']})",
            color=scenario_color,
            gif_path=gif_path,
            y_lim=global_y_lim,
        )

        print(f"  Saved GIF: {gif_path}")

print(f"\nDone. GIFs saved in: {output_dir}")

# %%
