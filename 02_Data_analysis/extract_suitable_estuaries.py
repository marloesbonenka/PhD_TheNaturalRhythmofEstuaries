"""Extract estuaries from Nienhuis et al 2018: Tide-Influenced Deltas"""
#%%
from pathlib import Path
import matplotlib as mpl
import sys

sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\02_Data_analysis")
from FUNCTIONS.FUNCS_analyze_Nienhuis2018_dataset import load_data, select_estuaries, plot_discharge_bar, save_selection
#%% --- PLOTTING SETTINGS ---
figsize = (10, 6)
mpl.rcParams['figure.figsize'] = figsize

#%% --- CONFIGURATION ---
# Path to the CSV file containing estuary information
csv_file = Path(r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash\Nienhuis2018_TideInfluencedDeltas.xlsx")
output_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Data\01_Discharge_var_int_flash\Estuary_Selection_Nienhuis2018")
output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

# Define T limits for selecting estuaries
T_LOWER = 1
T_UPPER = 100

# Define Q limits for selecting estuaries
Q_UPPER = 1000

#Selection based on:
SELECT_T = True
SELECT_Q = False 

# Whether to save the selected estuaries to a new Excel file
SAVE_FIG = True
SAVE_EXCEL = False
#%%
df = load_data(csv_file)

# Select estuaries based on T and/or Q
df_selected = select_estuaries(
    df,
    select_T=SELECT_T, t_lower=T_LOWER, t_upper=T_UPPER,
    select_Q=SELECT_Q, q_upper=Q_UPPER,
    drop_outliers=['Amazon'] if not SELECT_Q else None  # Only drop when not filtering by Q
)

# Build label and filename suffix based on active filters
label = f"{T_LOWER} < T < {T_UPPER}" + (f", Q < {Q_UPPER}" if SELECT_Q else "")
suffix = f"T_{T_LOWER}_{T_UPPER}" + (f"_Q_{Q_UPPER}" if SELECT_Q else "_all_Q")

# Plot discharge bar chart and PDF after selection
plot_discharge_bar(
    df_selected,
    title=f'River discharge of selected estuaries ({label})',
    save_path=output_dir / f'selected_estuary_discharge_{suffix}.png' if SAVE_FIG else None
)

# Optionally save to Excel
if SAVE_EXCEL:
    filename = "Selected_Estuaries_T_Q_limits.xlsx" if SELECT_Q else "Selected_Estuaries_T_limits.xlsx"
    save_selection(df_selected, output_dir / filename)
# %%