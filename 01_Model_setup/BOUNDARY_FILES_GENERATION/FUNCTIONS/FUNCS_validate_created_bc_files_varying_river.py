import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib as mpl

#%%
plots_map_name = '000_bc_plots'

def calculate_flashiness(discharge_values, epsilon=0.01):
    """
    Calculate flashiness of discharge using progressively less extreme percentiles
    to avoid divide-by-zero errors.
    
    Parameters:
    -----------
    discharge_values : numpy array
        Array of discharge values
    epsilon : float
        Small value to determine when a percentile is too close to zero
    
    Returns:
    --------
    flashiness : float
        Calculated flashiness value
    flashiness_method : str
        Description of the method used
    """
    # Use absolute values for discharge
    abs_discharge = np.abs(discharge_values)
    
    # Try percentile pairs in order of preference
    percentile_pairs = [
        (90, 10, "P90/P10"),  # First choice
        (85, 15, "P85/P15"),  # Second choice
        (80, 20, "P80/P20"),  # Third choice
        (75, 25, "P75/P25")   # Fourth choice
    ]
    
    # Try each percentile pair until we find one that works
    for high_pctl, low_pctl, method_name in percentile_pairs:
        high_val = np.percentile(abs_discharge, high_pctl)
        low_val = np.percentile(abs_discharge, low_pctl)
        
        # If denominator is not too close to zero, use this pair
        if low_val >= epsilon:
            flashiness = high_val / low_val
            return flashiness, method_name
    
    # If all pairs have issues, use the last one and add epsilon to prevent div by zero
    high_val = np.percentile(abs_discharge, 75)
    low_val = np.percentile(abs_discharge, 25)
    
    if low_val < epsilon:
        # Last resort - add epsilon to denominator
        flashiness = high_val / (low_val + epsilon)
        return flashiness, "P75/(P25+ε)"
    else:
        flashiness = high_val / low_val
        return flashiness, "P75/P25"

def visualize_discharge_scenarios(scenarios, base_output_dir, grid_info):
    """
    Visualize river discharge scenarios with cumulative and individual river cell plots,
    including yearly plots for enhanced temporal resolution.

    Parameters:
    -----------
    scenarios : list of dict
        List of scenario configurations
    base_output_dir : str
        Base directory where scenario output is stored
    grid_info : dict
        Grid information including river cells
    """
    plots_dir = os.path.join(base_output_dir, 'plots_river_bct')
    os.makedirs(plots_dir, exist_ok=True)

    all_scenarios_data = []

    for scenario in scenarios:
        scenario_name = scenario['name']
        scenario_dir = os.path.join(base_output_dir, scenario_name)
        boundary_dir = os.path.join(scenario_dir, "boundaryfiles")
        bct_file = os.path.join(boundary_dir, 'river_boundary.bct')

        if not os.path.exists(bct_file):
            print(f"Warning: BCT file for scenario {scenario_name} not found at {bct_file}")
            continue

        # Parse the BCT file
        time_points, discharge_values = parse_bct_file(bct_file, len(grid_info['river_cells']))
        discharge_values = np.array(discharge_values)  # Ensure NumPy array for indexing

        all_scenarios_data.append({
            'name': scenario_name,
            'time_points': time_points,
            'discharge_values': discharge_values
        })

        start_date = datetime(2024, 1, 1)
        dates = [start_date + timedelta(minutes=int(t)) for t in time_points]

        # Overall cumulative discharge plot
        plt.figure()
        cumulative_discharge = np.sum(discharge_values, axis=0)
        plt.plot(dates, -cumulative_discharge)
        plt.title(f'{scenario_name}')
        plt.xlabel('date')
        plt.ylabel('discharge * -1 [m³/s]')
        plt.grid(True, alpha=0.3)

        ax = plt.gca()
        ax.tick_params(axis='both', which='major')   
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()
        


        mean_discharge = np.mean(cumulative_discharge)
        min_discharge = np.min(cumulative_discharge)
        max_discharge = np.max(cumulative_discharge)
        cv = np.std(np.abs(cumulative_discharge)) / np.mean(np.abs(cumulative_discharge))

        flashiness_value, flashiness_method = calculate_flashiness(cumulative_discharge)

        stats_text = (
            f"Mean: {mean_discharge:.2f} m³/s\n"
            f"Min: {min_discharge:.2f} m³/s\n"
            f"Max: {max_discharge:.2f} m³/s\n"
            f"{flashiness_method}: {flashiness_value:.2f}\n"
            f"CV: {cv:.2f}"
        )

        plt.annotate(stats_text, xy=(0.02, 0.97), xycoords='axes fraction',
                     bbox=dict(boxstyle="square,pad=0.5", fc="white", alpha=0.8),
                     va='top')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{scenario_name}_cumulative_discharge.png'), dpi=300, transparent=True)
        plt.close()

        # Overall individual river cell discharge plot
        plt.figure()
        colors = plt.cm.viridis(np.linspace(0, 1, len(discharge_values)))
        for i, cell_discharge in enumerate(discharge_values):
            plt.plot(dates, -cell_discharge, label=f'River {i+1}', color=colors[i], alpha=0.7)
        plt.title(f'Individual River Cell Discharges - {scenario_name}')
        plt.xlabel('date')
        plt.ylabel('discharge * -1 [m³/s]')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', labelcolor='linecolor')
        
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{scenario_name}_individual_cells.png'), dpi=300, transparent=True)
        plt.close()

        # Yearly plots for individual years
        start_year = dates[0].year
        end_year = dates[-1].year
        for year in range(start_year, end_year + 1):
            indices = [i for i, d in enumerate(dates) if d.year == year]
            if not indices:
                continue

            year_dates = [dates[i] for i in indices]
            year_cumulative_discharge = np.sum(discharge_values[:, indices], axis=0)

            # Yearly cumulative discharge plot
            plt.figure()
            plt.plot(year_dates, -year_cumulative_discharge)
            plt.title(f'Cumulative River Discharge - {scenario_name} - Year {year}')
            plt.xlabel('date')
            plt.ylabel('discharge * -1 [m³/s]')
            plt.grid(True, alpha=0.3)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{scenario_name}_cumulative_discharge_{year}.png'), dpi=300, transparent=True)
            plt.close()

            # Yearly individual cell discharge plot
            plt.figure()
            for i, cell_discharge in enumerate(discharge_values):
                plt.plot(year_dates, -cell_discharge[indices], label=f'River {i+1}', color=colors[i], alpha=0.7)
            plt.title(f'Individual River Cell Discharges - {scenario_name} - Year {year}')
            plt.xlabel('date')
            plt.ylabel('discharge * -1 [m³/s]')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', labelcolor='linecolor')
            ax = plt.gca()
            ax.tick_params(axis='both', which='major')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'{scenario_name}_individual_cells_{year}.png'), dpi=300, transparent=True)
            plt.close()

    # Plot all scenarios cumulative discharge on one figure
    plt.figure()
    custom_colors_list = ['tab:blue', 'tab:orange', 'tab:green', '#d83034', '#ff73b6', '#003a7d']
    num_scenarios = len(all_scenarios_data)
    if num_scenarios > len(custom_colors_list):
        raise ValueError(f"Not enough colors for {num_scenarios} scenarios.")

    scenario_colors = custom_colors_list[:num_scenarios]
    annotations = []

    # Map scenario names to descriptive labels
    label_map = {
        '01_baserun500': 'Constant discharge',
        '02_run500_seasonal': 'Seasonal discharge',
        '03_run500_flashy': 'Flashy discharge'
    }

    for i, scenario_data in enumerate(all_scenarios_data):
        scenario_name = scenario_data['name']
        time_points = scenario_data['time_points']
        discharge_values = scenario_data['discharge_values']
        dates = [datetime(2024, 1, 1) + timedelta(minutes=int(t)) for t in time_points]
        cumulative_discharge = np.sum(discharge_values, axis=0)
        
        label_name = label_map.get(scenario_name, scenario_name)
        
        plt.plot(dates, -cumulative_discharge, label=label_name,
                 color=scenario_colors[i], linewidth=2)

        mean_q = np.mean(cumulative_discharge)
        flashiness_value, flashiness_method = calculate_flashiness(cumulative_discharge)
        cv = np.std(np.abs(cumulative_discharge)) / np.mean(np.abs(cumulative_discharge))
        annotation = (f"{scenario_name}:\n"
                      f"Mean = {mean_q:.0f} m³/s\n"
                      f"{flashiness_method} = {flashiness_value:.2f}\n"
                      f"CV: {cv:.2f}")
        annotations.append(annotation)

    plt.annotate('\n\n'.join(annotations), xy=(0.02, 0.97), xycoords='axes fraction',
                 bbox=dict(boxstyle="square,pad=0.5", fc="white", alpha=0.8, edgecolor='grey'),
                 va='top')

    # plt.title('comparison of river discharge across scenarios')
    plt.xlabel('date')
    plt.ylabel('discharge * -1 [m³/s]')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', labelcolor='linecolor')
    ax = plt.gca()
    ax.tick_params(axis='both', which='major')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()

    discharge_number = scenarios[0]['total_discharge']
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'{discharge_number}_all_scenarios_comparison.png'), dpi=300, transparent=True)
    plt.close()

    start_year = all_scenarios_data[0]['time_points'][0]  # approximate from first scenario start (in minutes)
    start_date = datetime(2024, 1, 1)
    all_dates = [start_date + timedelta(minutes=int(t)) for t in all_scenarios_data[0]['time_points']]
    overall_start_year = all_dates[0].year
    overall_end_year = all_dates[-1].year

    for year in range(overall_start_year, overall_end_year + 1):
        plt.figure()
        annotations = []
        for i, scenario_data in enumerate(all_scenarios_data):
            scenario_name = scenario_data['name']
            time_points = scenario_data['time_points']
            discharge_values = scenario_data['discharge_values']

            # Convert to dates for indexing
            dates = [start_date + timedelta(minutes=int(t)) for t in time_points]
            indices = [idx for idx, d in enumerate(dates) if d.year == year]
            if not indices:
                continue

            cumulative_discharge_year = np.sum(discharge_values[:, indices], axis=0)
            dates_year = [dates[idx] for idx in indices]

            mean_q = np.mean(cumulative_discharge_year)
            label_name = label_map.get(scenario_name, scenario_name)

            # # Legend label now contains mean discharge
            label_text = f"{label_name}\n$Q_{{\\mathrm{{mean}}}} = {mean_q:.0f}$ m³/s"


            plt.plot(dates_year, -cumulative_discharge_year, label=label_text,
                    color=custom_colors_list[i])

        #     flashiness_value, flashiness_method = calculate_flashiness(cumulative_discharge_year)
        #     cv = np.std(np.abs(cumulative_discharge_year)) / np.mean(np.abs(cumulative_discharge_year))

        #     annotation = (f"{scenario_name}:\n"
        #                 f"Mean = {mean_q:.0f} m³/s\n"
        #                 f"{flashiness_method} = {flashiness_value:.2f}\n"
        #                 f"CV: {cv:.2f}")
        #     annotations.append(annotation)

        # plt.annotate('\n\n'.join(annotations), xy=(0.02, 0.97), xycoords='axes fraction',
        #             bbox=dict(boxstyle="square,pad=0.5", fc="white", alpha=0.6),
        #             va='top')

        plt.title(f'{year}')
        plt.xlabel('date')
        plt.ylabel('discharge * -1 [m³/s]')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', labelcolor='linecolor')
        ax = plt.gca()
        ax.tick_params(axis='both', which='major')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()

        plt.savefig(os.path.join(plots_dir, f'all_scenarios_comparison_{year}.png'), transparent=True)
        plt.close()

    print(f"All visualization plots have been saved to {plots_dir}")

def parse_bct_file(bct_file, num_river_cells):
    """Robust BCT file parser that handles actual Delft3D formatting"""
    with open(bct_file, 'r') as f:
        lines = f.readlines()

    time_points = []
    discharge_values = [[] for _ in range(num_river_cells)]
    current_cell = -1
    records_remaining = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect new boundary section
        if "Boundary Section" in line:
            current_cell += 1
            if current_cell >= num_river_cells:
                break
            continue

        # Get record count
        if line.startswith("records-in-table"):
            records_remaining = int(line.split()[-1])
            continue

        # Parse data lines
        if records_remaining > 0 and current_cell < num_river_cells:
            parts = line.split()
            if len(parts) >= 3:
                if current_cell == 0:  # Only store time once
                    time_points.append(float(parts[0]))
                # Use average of both discharge values
                discharge = (float(parts[1]) + float(parts[2])) / 2
                discharge_values[current_cell].append(discharge)
                records_remaining -= 1

    return np.array(time_points), [np.array(d) for d in discharge_values]

#%%

