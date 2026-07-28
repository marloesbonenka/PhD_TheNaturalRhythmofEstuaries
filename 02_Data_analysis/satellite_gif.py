#%% Imports
import re
from pathlib import Path
from PIL import Image

#%% Settings
input_dir = Path(r"u:\PhDNaturalRhythmEstuaries\Data\05_PlanetAI\Tapi")       
output_gif = input_dir / "Tapi01_timelapse.gif"
frame_duration_ms = 800                            # slow playback; bump up for even slower

#%% Find and sort images (missing months are simply not on disk -> just skipped)
pattern = re.compile(r"Tapi01_(\d{2})_(\d{4})\.png$", re.IGNORECASE)

frames_info = []
for f in input_dir.glob("Tapi01_*_*.png"):
    m = pattern.match(f.name)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        frames_info.append((year, month, f))

if not frames_info:
    raise FileNotFoundError(f"No files matching 'Tapi01_MM_YYYY.png' found in {input_dir}")

frames_info.sort(key=lambda x: (x[0], x[1]))  # chronological order (year, then month)

#%% Report which months are missing in the range (informational only, never fails)
first_year, first_month = frames_info[0][0], frames_info[0][1]
last_year, last_month = frames_info[-1][0], frames_info[-1][1]

expected = set()
y, m = first_year, first_month
while (y, m) <= (last_year, last_month):
    expected.add((y, m))
    m += 1
    if m > 12:
        m = 1
        y += 1

present = {(yr, mo) for yr, mo, _ in frames_info}
missing = sorted(expected - present)

if missing:
    print(f"Note: {len(missing)} month(s) missing, gif will just skip these:")
    for yr, mo in missing:
        print(f"  - {mo:02d}/{yr}")
else:
    print("No missing months in the range - full sequence.")

#%% Load images (skip any file that fails to open rather than crashing the whole run)
images = []
for year, month, path in frames_info:
    try:
        img = Image.open(path).convert("RGB")
        images.append(img)
    except Exception as e:
        print(f"Warning: could not open {path.name}, skipping ({e})")

if not images:
    raise RuntimeError("No images could be loaded - check the input directory and filenames.")

#%% Make sure all frames share one size (resize to match the first frame if needed)
ref_size = images[0].size
images = [img if img.size == ref_size else img.resize(ref_size) for img in images]

#%% Save as GIF
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=frame_duration_ms,
    loop=0,
)

print(f"Saved gif with {len(images)} frames to {output_gif}")