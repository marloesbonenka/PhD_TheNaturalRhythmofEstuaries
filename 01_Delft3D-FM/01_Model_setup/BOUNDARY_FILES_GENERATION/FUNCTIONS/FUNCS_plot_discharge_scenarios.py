"""
Functions for plotting discharge scenarios.
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Color and label mapping based on scenario type
SCENARIO_CONFIG = {
    "baserun": {"color": "tab:blue", "label": "Constant discharge"},
    "seasonal": {"color": "tab:orange", "label": "Seasonal discharge"},
    "flashy": {"color": "tab:green", "label": "Flashy discharge"},
    "singlepeak": {"color": "tab:red", "label": "Single peak discharge"},
}


def get_scenario_type(scenario_name):
    """
    Extract scenario type from folder name.
    
    Handles patterns like:
    - '01_baserun250' -> 'baserun'
    - '02_run250_seasonal' -> 'seasonal'
    - '03_run500_flashy' -> 'flashy'
    - '04_run1000_singlepeak' -> 'singlepeak'
    """
    scenario_lower = scenario_name.lower()
    
    # Check for pattern type at the end (seasonal, flashy, singlepeak)
    if scenario_lower.endswith("_seasonal"):
        return "seasonal"
    elif scenario_lower.endswith("_flashy"):
        return "flashy"
    elif scenario_lower.endswith("_singlepeak"):
        return "singlepeak"
    # Check for baserun (constant discharge)
    elif "baserun" in scenario_lower:
        return "baserun"
    
    return None


def get_scenario_label(scenario_name):
    """Map scenario folder name to display label."""
    scenario_type = get_scenario_type(scenario_name)
    if scenario_type:
        return SCENARIO_CONFIG[scenario_type]["label"]
    return scenario_name


def get_scenario_color(scenario_name):
    """Assign color based on scenario type."""
    scenario_type = get_scenario_type(scenario_name)
    if scenario_type:
        return SCENARIO_CONFIG[scenario_type]["color"]
    return "tab:gray"


def plot_discharge_scenarios_first_year(
    scenario_csv_paths,
    output_dir,
    output_filename="discharge_scenarios_first_year.png",
):
    """
    Plot cumulative discharge for multiple scenarios (first year only).

    Parameters
    ----------
    scenario_csv_paths : dict
        Mapping of scenario name to CSV path.
        Expected CSV columns: 'timestamp', 'discharge_m3s'
    output_dir : str or Path
        Directory to save the output figure.
    output_filename : str
        Output image filename.
    """
    if not scenario_csv_paths:
        raise ValueError("No scenario CSV paths provided.")

    scenario_items = list(scenario_csv_paths.items())

    first_year = None
    series_data = []

    for scenario_name, csv_path in scenario_items:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "timestamp" not in df.columns or "discharge_m3s" not in df.columns:
            raise ValueError(f"CSV missing required columns: {csv_path}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if first_year is None:
            first_year = df["timestamp"].min().year

        df_year = df[df["timestamp"].dt.year == first_year]
        series_data.append((scenario_name, df_year))

    if first_year is None:
        raise ValueError("Could not determine first year from data.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for idx, (scenario_name, df_year) in enumerate(series_data):
        label_name = get_scenario_label(scenario_name)
        color = get_scenario_color(scenario_name)
        plt.plot(
            df_year["timestamp"],
            df_year["discharge_m3s"],
            label=label_name,
            color=color,
            linewidth=2,
        )

    # plt.title(f"Discharge scenarios {first_year}")
    plt.xlabel("date")
    plt.ylabel("discharge [m³/s]")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", labelcolor="linecolor")

    ax = plt.gca()
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, transparent=True)
    plt.close()
