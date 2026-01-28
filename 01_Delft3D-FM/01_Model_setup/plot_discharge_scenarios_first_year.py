from pathlib import Path
import sys

# Add the FUNCTIONS directory to the path
sys.path.append(r"c:\Users\marloesbonenka\Nextcloud\Python\01_Delft3D-FM\01_Model_setup\BOUNDARY_FILES_GENERATION")

from FUNCTIONS.FUNCS_plot_discharge_scenarios import *


if __name__ == "__main__":
    # Base directory containing all scenario subfolders
    base_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\CSVfiles_boundaries_50hydroyears")
    output_base_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\CSVfiles_boundaries_50hydroyears\plots_river_bct")
    
    # Find all subfolders that contain boundaryfiles_csv
    scenario_groups = {}  # Group scenarios by discharge value (e.g., 250, 500, 1000)
    
    for subfolder in sorted(base_dir.iterdir()):
        if subfolder.is_dir():
            csv_path = subfolder / "boundaryfiles_csv" / "discharge_cumulative.csv"
            if csv_path.exists():
                # Extract discharge value from folder name
                # Patterns: "01_baserun250", "02_run500_seasonal", etc.
                folder_name = subfolder.name
                discharge_val = "unknown"
                
                for part in folder_name.split("_"):
                    if part.startswith("baserun"):
                        discharge_val = part.replace("baserun", "")
                        break
                    elif part.startswith("run"):
                        discharge_val = part.replace("run", "")
                        break
                
                if discharge_val not in scenario_groups:
                    scenario_groups[discharge_val] = {}
                scenario_groups[discharge_val][folder_name] = str(csv_path)
    
    # Create a plot for each discharge group
    for discharge_val, scenario_csv_paths in scenario_groups.items():
        print(f"Creating plot for discharge = {discharge_val} m³/s with scenarios: {list(scenario_csv_paths.keys())}")
        output_dir = output_base_dir / f"discharge_{discharge_val}"
        plot_discharge_scenarios_first_year(
            scenario_csv_paths, 
            output_dir,
            output_filename=f"discharge_scenarios_{discharge_val}_first_year.png"
        )
    
    print(f"All plots saved to: {output_base_dir}")
