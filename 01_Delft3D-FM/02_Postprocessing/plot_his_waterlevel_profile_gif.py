"""
Create GIF(s) of along-estuary water-level profiles from station-point data.

X-axis: estuary distance [km from sea]
Y-axis: water level [m]
Frames: selected timesteps through the simulation

Uses separated station cache files (hisoutput_stations_*.nc) when available,
with fallback to HIS files.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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
from FUNCTIONS.F_tidalrange_currentspeed import load_station_waterlevels_from_cache_or_his


# =============================================================================
# CONFIG
# =============================================================================
DISCHARGE = 500
SCENARIOS_TO_PROCESS = ['1', '2', '3', '4']
OUTPUT_DIRNAME = 'plots_his_waterlevel_profiles'

WATERLEVEL_VAR = 'waterlevel'
STATION_PATTERN = r'^Observation(?:Point|CrossSection)_Estuary_km(\d+)$'
EXCLUDE_LAST_TIMESTEP = True

# Animation settings
FRAME_STRIDE = 24        # use every Nth timestep to keep GIF manageable
FPS = 10
FIGSIZE = (9, 5)
LINEWIDTH = 2.0
MARKERSIZE = 4

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

VARIABILITY_MAP = {
    '1': f'01_baserun{DISCHARGE}',
    '2': f'02_run{DISCHARGE}_seasonal',
    '3': f'03_run{DISCHARGE}_flashy',
    '4': f'04_run{DISCHARGE}_singlepeak',
}

model_folders = [f.name for f in base_path.iterdir() if f.is_dir() and f.name[0].isdigit()]
if SCENARIOS_TO_PROCESS:
    scenario_filter = set(int(s) for s in SCENARIOS_TO_PROCESS)
    model_folders = [f for f in model_folders if int(f.split('_')[0]) in scenario_filter]
model_folders.sort(key=lambda x: int(x.split('_')[0]))

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
def create_profile_gif(*, km, wl, times, scenario_key, label, color, gif_path):
    """Create a water-level profile GIF for one scenario."""
    if wl.size == 0 or len(times) == 0:
        raise ValueError('No water-level data available for GIF creation')

    frame_idx = np.arange(0, len(times), FRAME_STRIDE, dtype=int)
    if frame_idx[-1] != len(times) - 1:
        frame_idx = np.append(frame_idx, len(times) - 1)

    wl_frames = wl[frame_idx, :]
    t_frames = times[frame_idx]

    y_min = float(np.nanmin(wl_frames))
    y_max = float(np.nanmax(wl_frames))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        raise ValueError('Water-level array contains no finite values')

    pad = 0.05 * max(1e-6, (y_max - y_min))
    y_lim = (y_min - pad, y_max + pad)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    line, = ax.plot([], [], color=color, linewidth=LINEWIDTH, marker='o', markersize=MARKERSIZE)

    ax.set_xlim(float(np.nanmin(km)), float(np.nanmax(km)))
    ax.set_ylim(*y_lim)
    ax.set_xlabel('Estuary distance [km from sea]')
    ax.set_ylabel('Water level [m]')
    ax.grid(alpha=0.3)

    title = ax.set_title('')

    def _init():
        line.set_data([], [])
        title.set_text(f"{label} - water-level profile")
        return line, title

    def _update(i):
        line.set_data(km, wl_frames[i, :])
        t_years = (t_frames[i] - t_frames[0]) / np.timedelta64(1, 'D') / 365.25
        title.set_text(f"{label} - water-level profile | t={t_years:.2f} years")
        return line, title

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
    plt.close(fig)


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
    times = wl_data['times']

    scenario_label = SCENARIO_LABELS.get(scenario_key, scenario_key)
    scenario_color = SCENARIO_COLORS.get(scenario_key, 'grey')
    gif_path = output_dir / f"waterlevel_profile_Q{DISCHARGE}_{scenario_key}.gif"

    create_profile_gif(
        km=km,
        wl=wl,
        times=times,
        scenario_key=scenario_key,
        label=scenario_label,
        color=scenario_color,
        gif_path=gif_path,
    )

    print(f"  Saved GIF: {gif_path}")

print(f"\nDone. GIFs saved in: {output_dir}")
