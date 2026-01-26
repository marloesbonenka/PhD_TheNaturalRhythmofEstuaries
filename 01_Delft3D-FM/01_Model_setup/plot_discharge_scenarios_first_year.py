from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


CUSTOM_SCENARIO_COLORS = ["tab:blue", "tab:orange", "tab:green"]
LABEL_MAP = {
    "01_baserun500": "Constant discharge",
    "02_run500_seasonal": "Seasonal discharge",
    "03_run500_flashy": "Flashy discharge",
}


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
    if len(scenario_items) > len(CUSTOM_SCENARIO_COLORS):
        raise ValueError("Not enough colors for the number of scenarios.")

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
        label_name = LABEL_MAP.get(scenario_name, scenario_name)
        plt.plot(
            df_year["timestamp"],
            df_year["discharge_m3s"],
            label=label_name,
            color=CUSTOM_SCENARIO_COLORS[idx],
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


if __name__ == "__main__":
    scenario_csv_paths = {
        "01_baserun500": r"U:\PhDNaturalRhythmEstuaries\Models\0_Model_SetUp_CSVfiles_boundaries_50years\01_baserun500\boundaryfiles_csv\discharge_cumulative.csv",
        "02_run500_seasonal": r"U:\PhDNaturalRhythmEstuaries\Models\0_Model_SetUp_CSVfiles_boundaries_50years\02_run500_seasonal\boundaryfiles_csv\discharge_cumulative.csv",
        "03_run500_flashy": r"U:\PhDNaturalRhythmEstuaries\Models\0_Model_SetUp_CSVfiles_boundaries_50years\03_run500_flashy\boundaryfiles_csv\discharge_cumulative.csv",
    }

    output_dir = Path(r"U:\PhDNaturalRhythmEstuaries\Models\1_RiverDischargeVariability_domain45x15") / "plots_river_bct"
    plot_discharge_scenarios_first_year(scenario_csv_paths, output_dir)
