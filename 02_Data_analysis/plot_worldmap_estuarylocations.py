""""Plot a world map with the locations of the estuaries 
    in the dataset, for visualisation purposes."""

#%%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
#%%
estuary_coords = {
    # 'Mississippi': (29.15, -89.25),  # USA, for validation
    # 'Amazon': (-0.1, -50.4),         # Brazil, for validation

    'Chao Phraya': (13.55, 100.59),         #Thailand
    'Colorado (MX)': (31.83, -114.82),      #Mexico
    'Cacipore': (3.6, -51.2),               #French Guiana
    'Cromary Firth': (57.69, -4.02),        #UK
    'Demerara': (6.79, -58.18),             #Guyana
    'Firth of Tay': (56.45, -2.83),         #UK
    'Fly': (-8.62, 143.70),                 #Papua New Guinea
    'Gironde': (45.58, -1.05),              #France
    'Guayas': (-2.55, -79.88),              #Ecuador
    'Humber': (53.62, -0.11),               #UK
    'Kumbe': (-8.36, 140.23),               #Indonesia
    'Ord': (-15.5, 128.35),                 #Australia
    'Purna': (20.91, 72.78),                #India
    'Sokyosen': (36.9, 126.9),              #South Korea 
    'Sungai Merauke': (-8.47, 140.35),      #Indonesia
    'Suriname': (5.84, -55.11),             #Suriname
    'Taeryong': (39.63, 125.48),            #North Korea
    'Tapi': (21.15, 72.75),                 #India
    'Yangon': (16.52, 96.29),               #Myanmar
    'Bian': (-8.10, 139.97),                #Indonesia   
    'Western Scheldt': (51.42, 3.57),       #Netherlands

    # #Excluded                                                         because:
    # 'Eel': (40.63, -124.31),                #USA                      - River-dominated
    # 'Klamath': (41.54, -124.08),            #USA                      - River-dominated
    # 'Thames': (51.5, 0.6),                  #UK                       - Too tide-dominated
    # 'Columbia': (46.25, -124.05),           #USA                      - Reason unclear
    # 'Rio de la Plata': (-35.00, -56.00),    #Argentina/Uruguay        - Reason unclear
    # 'Mengha (Ganges-Brahmaputra)': (21.47, 91.06), #Bangladesh        - Reason unclear
}

# Label priority in crowded regions: higher values are kept first.
# Example below prefers plotting Humber and de-prioritizes Cromary Firth.
label_priority = {
    'Humber': 10,
    'Cromary Firth': -10,
    'Sungai Merauke': -10,
    'Ord': 10,
    'Tapi': -10,
    'Purna': 10,
    'Taeryong': -10,
    'Bian': -10
}
#%%
lats = [v[0] for v in estuary_coords.values()]
lons = [v[1] for v in estuary_coords.values()]

# Figure
fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.Robinson())

# --- CLEAN MAP STYLE ---
ax.set_facecolor("white")  # remove blue ocean

# Land: only outline, no fill
ax.add_feature(
    cfeature.LAND,
    facecolor="none",
    edgecolor="black",
    linewidth=0.8
)

# Coastlines (slightly thicker)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

# Country borders (dashed as you liked)
ax.add_feature(
    cfeature.BORDERS,
    linestyle=':',
    linewidth=0.7
)

# --- POINTS ---
ax.scatter(
    lons, lats,
    transform=ccrs.PlateCarree(),
    color='#044E64', #FFCD00',
    s=75,           # bigger dots
    zorder=2
)

# --- LABELS ---
def _distance_km(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance in km (haversine)."""
    r = 6371.0
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(lat2 - lat1)
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


names = list(estuary_coords.keys())

# Prefer high-priority labels first; these win in local conflicts.
order = sorted(
    names,
    key=lambda n: (-label_priority.get(n, 0), n),
)

# If a lower-priority estuary sits near a higher-priority one, skip it early.
neighbor_priority_radius_km = 700.0
filtered_names = []
for name in order:
    lat, lon = estuary_coords[name]
    pri = label_priority.get(name, 0)

    overshadowed = False
    for other in names:
        if other == name:
            continue
        other_pri = label_priority.get(other, 0)
        if other_pri <= pri:
            continue
        o_lat, o_lon = estuary_coords[other]
        if _distance_km(lat, lon, o_lat, o_lon) < neighbor_priority_radius_km:
            overshadowed = True
            break

    if not overshadowed:
        filtered_names.append(name)

# True overlap avoidance: place annotations and test rendered text bounding boxes.
offset_candidates = [
    (2, 2), (2, -2), (-2, 2), (-2, -2),
    (3, 1), (3, -1), (-3, 1), (-3, -1),
    (1, 3), (1, -3), (-1, 3), (-1, -3),
    (0, 4), (0, -4),
]

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
accepted_bboxes = []
drawn_annotations = []

for name in filtered_names:
    lat, lon = estuary_coords[name]
    placed = False

    for dx, dy in offset_candidates:
        ha = 'left' if dx >= 0 else 'right'
        va = 'bottom' if dy >= 0 else 'top'
        ann = ax.annotate(
            name,
            xy=(lon, lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(dx, dy),
            textcoords='offset points',
            fontsize=15,
            ha=ha,
            va=va,
            zorder=4,
        )

        fig.canvas.draw()
        bbox = ann.get_window_extent(renderer=renderer).expanded(1.02, 1.08)

        if any(bbox.overlaps(b) for b in accepted_bboxes):
            ann.remove()
            continue

        accepted_bboxes.append(bbox)
        drawn_annotations.append(ann)
        placed = True
        break

    if not placed:
        continue

ax.set_global()
# plt.title("Global Distribution of Estuaries", fontsize=14)

plt.savefig(r"u:\PhDNaturalRhythmEstuaries\Data\worldmap_estuary_locations.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
