#%%
from pathlib import Path
import sys
#%%
# Add the FUNCTIONS directory to the path
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\01_Model_setup\BOUNDARY_FILES_GENERATION")

from FUNCTIONS.FUNCS_plot_discharge_scenarios import plot_discharge_scenarios_first_year, compute_CV, compute_p90_p10
#%%

if __name__ == "__main__":
    # Base directory containing all scenario subfolders
    base_dir = Path(r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Input")
    output_base_dir = Path(r"u:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15\Model_Input\plots_river_bct")
    
    # Find all discharge_cumulative.csv files for each scenario per discharge, including timed-out folders
    scenario_groups = {}  # Group scenarios by discharge value (e.g., 250, 500, 1000)

    for discharge_folder in sorted(base_dir.iterdir()):
        if discharge_folder.is_dir():
            discharge_val = discharge_folder.name.replace("Q", "")
            scenario_groups[discharge_val] = {}
            timed_out_folder = discharge_folder / "timed_out"
            timed_out_csvs = {}
            # First, check timed_out folder for CSVs
            if timed_out_folder.is_dir():
                for scenario_folder in sorted(timed_out_folder.iterdir()):
                    if scenario_folder.is_dir():
                        csv_path = scenario_folder / "boundaryfiles_csv" / "discharge_cumulative.csv"
                        if csv_path.exists():
                            timed_out_csvs[scenario_folder.name] = str(csv_path)
                # If CSVs found for all scenarios in timed_out, use only those and skip rest
                if timed_out_csvs:
                    scenario_groups[discharge_val] = timed_out_csvs
                    continue
            # Otherwise, search regular scenario folders
            for scenario_folder in sorted(discharge_folder.iterdir()):
                if scenario_folder.is_dir() and scenario_folder.name != "timed_out":
                    csv_path = scenario_folder / "boundaryfiles_csv" / "discharge_cumulative.csv"
                    if csv_path.exists():
                        scenario_groups[discharge_val][scenario_folder.name] = str(csv_path)
                    # Also check for timed-out subfolders inside scenario
                    timed_out_subfolder = scenario_folder / "timed_out"
                    if timed_out_subfolder.is_dir():
                        for subfolder in sorted(timed_out_subfolder.iterdir()):
                            csv_path_timed_out = subfolder / "boundaryfiles_csv" / "discharge_cumulative.csv"
                            if csv_path_timed_out.exists():
                                scenario_groups[discharge_val][subfolder.name] = str(csv_path_timed_out)

    # Create a plot for each discharge group
    for discharge_val, scenario_csv_paths in scenario_groups.items():
        if not scenario_csv_paths:
            print(f"Skipping discharge {discharge_val}: no CSVs found.")
            continue
        print(f"Creating plot for discharge = {discharge_val} m³/s with scenarios: {list(scenario_csv_paths.keys())}")
        output_dir = output_base_dir / f"discharge_{discharge_val}"
        plot_discharge_scenarios_first_year(
            scenario_csv_paths,
            output_dir,
            output_filename=f"discharge_scenarios_{discharge_val}_first_year.png"
        )

    print(f"All plots saved to: {output_base_dir}")

# %%


