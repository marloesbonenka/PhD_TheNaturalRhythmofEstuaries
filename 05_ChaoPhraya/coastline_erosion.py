""""Create GIF from monthly satellite images of coastline erosion at Bang Khun Thian, Bangkok, Thailand.

13.5041°N, 100.4964°E — zoom 14.2
Laem Fa Pha, Phra Samut Chedi District, Samut Prakan, Thailand
"""

#%%

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import re
import imageio 

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\05_ChaoPhraya")

#%%

image_dir = Path(r"c:\Users\marloesbonenka\Nextcloud\Conferences\2025_Bangkok\Satellite images")

#Find all files starting with "zoom"

image_files = list(image_dir.glob("zoom*.png"))

# Extract month and year from filenames and sort chronologically
def get_date(file):
    match = re.search(r"zoom_(\d{2})_(\d{4})_", file.name)
    if not match:
        raise ValueError(f"Could not parse date from {file.name}")

    month = int(match.group(1))
    year = int(match.group(2))

    return year, month

image_files = sorted(image_files, key=get_date)

# Create GIF
gif_path = image_dir / "zoom_BangKhunTien_erosion.gif"

frames = [imageio.imread(file) for file in image_files]

imageio.mimsave(
    gif_path,
    frames,
    fps=1,  # seconds per frame
    loop=0         # infinite loop
)

print(f"GIF saved to: {gif_path}")


# %%
