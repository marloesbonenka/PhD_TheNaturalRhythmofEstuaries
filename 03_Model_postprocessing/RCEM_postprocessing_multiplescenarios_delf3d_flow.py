#%% 
"""Delft3D-4 Flow NetCDF Analysis: Morphological Estuary Analysis - Restructured.
Last edit: August 2025
Author: Marloes Bonenkamp
"""
#%% IMPORTS AND SETUP
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import time
import os
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional, Any
import netCDF4 as nc

#%% Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\Documents\Python\03_Model_postprocessing")

from FUNCTIONS.FUNCS_postprocessing_general import *
from FUNCTIONS.FUNCS_postprocessing_braiding_index import *
from FUNCTIONS.FUNCS_postprocessing_map_output import *
from FUNCTIONS.FUNCS_postprocessing_his_output import *
from FUNCTIONS.FUNCS_postprocessing_hypsometry import *

#%% PLOTTING SETTINGS   
defaultcolour = 'white'
defaultfont = 20
defaultfigsize = (10,5.3) #(10, 5) for cumulative width-averaged (10, 8)

mpl.rcParams['text.color'] = 'black'          # Default text color
mpl.rcParams['font.size'] = defaultfont             # Default font size

mpl.rcParams['axes.titlesize'] = defaultfont+4      # Title font size
mpl.rcParams['axes.titlecolor'] = defaultcolour     # Title color
mpl.rcParams['axes.labelsize'] = defaultfont        # Axis label size
mpl.rcParams['axes.labelcolor'] = defaultcolour     # Axis label color
mpl.rcParams['axes.facecolor'] = defaultcolour      # Background color of the axes (plot area)

mpl.rcParams['xtick.labelsize'] = defaultfont       # X tick labels size
mpl.rcParams['xtick.color'] = defaultcolour         # tick color matches text color
mpl.rcParams['xtick.labelcolor'] = defaultcolour

mpl.rcParams['ytick.labelsize'] = defaultfont       # Y tick labels size
mpl.rcParams['ytick.color'] = defaultcolour
mpl.rcParams['ytick.labelcolor'] = defaultcolour

mpl.rcParams['axes.grid'] = True                    # Default enable grid
mpl.rcParams['grid.alpha'] = 0.3                    # Grid transparency

mpl.rcParams['figure.figsize'] = defaultfigsize     # Default figure size (width, height) in inches
mpl.rcParams['legend.fontsize'] = defaultfont       # Legend font size

mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['savefig.dpi'] = 600

#%% CONFIGURATION CLASS
class AnalysisConfig:
    """Configuration class to hold all analysis parameters"""
    
    def __init__(self):        
        # Model configuration
        self.config = '05_RiverDischargeVariability_domain45x15'
        self.model_location = os.path.join(r"U:\PhDNaturalRhythmEstuaries\Models", self.config)
        
        # Discharge scenarios to process
        self.discharges = [500]  # [250, 500, 1000]
        
        # Scenario templates (all scenarios to process)
        self.scenario_templates = [
            '01_baserun{discharge}', 
            '02_run{discharge}_seasonal', 
            '03_run{discharge}_flashy'
        ]
        
        # Time range settings
        self.slice_start = 1
        self.slice_end = 360
        self.amount_to_plot = 4
        
        # Model settings
        self.reference_date = datetime.datetime(2024, 1, 1)
        self.Tstart = 3.1464e6
        self._set_model_parameters()
        
        # Estuary characteristics
        self.x_min, self.x_max = 20000, 45000
        self.y_min, self.y_max = 5000, 10000
        self.bed_threshold = 6
        
        # Analysis flags - can be modified before running
        self.run_spatial_plots = False
        self.run_width_averaged_bedlevel = False
        self.run_cumulative_width_averaged_bedlevel = False
        self.run_braiding_analysis = False
        self.run_hypsometric_analysis = False
        self.run_multi_scenario_hypsometric = False
        self.run_single_discharge_analysis = False
        self.run_multi_variable_analysis = False
        self.run_combined_width_averaged_bedlevel = False
        
        # Save settings
        self.save_figure = True
    
    def _set_model_parameters(self):
        """Set model parameters based on configuration"""
        print(f'config = {self.config}')
        
        if self.config == '04_RiverDischargeVariability_domain45x15':
            self.Tstop = 2.8908e6
            self.map_output_interval = 1300
            self.his_output_interval = 720
        elif self.config == '05_RiverDischargeVariability_domain45x15':
            self.map_output_interval = 1440
            self.his_output_interval = 720
            self.Tstop = 3.1464e6
        else:
            self.map_output_interval = 30
            self.his_output_interval = 30
            self.Tstop = 2.628e6 + (24*60)
        
        self.total_duration = self.Tstop - self.Tstart
        self.total_duration_days = self.total_duration / (60 * 24)
        self.map_output_interval_hours = self.map_output_interval / 60
        self.his_output_interval_hours = self.his_output_interval / 60

#%% DATASET MANAGER CLASS
class DatasetManager:
    """Manages NetCDF datasets with caching and efficient memory usage"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._datasets = {}  # Cache for opened datasets
        self._file_paths = {}  # Track file paths
        self._loaded_variables = {}  # Cache for loaded variables
    
    @contextmanager
    def get_dataset(self, discharge: int, scenario: str, dataset_type: str = 'trim'):
        """Context manager for getting datasets with automatic cleanup"""
        key = f"{discharge}_{scenario}_{dataset_type}"
        
        if key not in self._datasets:
            file_path = self._get_file_path(discharge, scenario, dataset_type)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            print(f"Loading {dataset_type} file for {scenario} (Q={discharge})...")
            start_time = time.time()
            self._datasets[key] = nc.Dataset(file_path, mode='r')
            print(f"{dataset_type} file loaded in {time.time() - start_time:.2f} seconds")
        
        try:
            yield self._datasets[key]
        except Exception as e:
            print(f"Error accessing dataset {key}: {e}")
            raise
    
    def _get_file_path(self, discharge: int, scenario: str, dataset_type: str) -> str:
        """Get file path for a given discharge, scenario, and dataset type"""
        runname = get_runname(discharge)
        
        if dataset_type == 'trim':
            filename = 'trim-varriver_tidewest.nc'
        elif dataset_type == 'trih':
            filename = 'trih-varriver_tidewest.nc'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return os.path.join(self.config.model_location, runname, scenario, filename)
    
    def preload_coordinates(self, discharge: int, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preload coordinates (same for all scenarios)"""
        coord_key = "coordinates"
        
        if coord_key not in self._loaded_variables:
            with self.get_dataset(discharge, scenario, 'trim') as dataset:
                print("Loading coordinates...")
                x = load_variable(dataset, "XCOR")
                y = load_variable(dataset, "YCOR")
                self._loaded_variables[coord_key] = (x, y)

                dps_shape = dataset.variables['DPS'].shape
                print(f"trim shape (t, x, y) = {dps_shape}")
                print(f"Coordinates shape: {x.shape}")
        
        return self._loaded_variables[coord_key]
    
    def close_all(self):
        """Close all open datasets"""
        for key, dataset in self._datasets.items():
            try:
                dataset.close()
                print(f"Closed dataset: {key}")
            except Exception as e:
                print(f"Error closing dataset {key}: {e}")
        
        self._datasets.clear()
        self._loaded_variables.clear()
    
    def get_save_dir(self, discharge: int, scenario: str) -> str:
        """Get save directory for a scenario"""
        runname = get_runname(discharge)
        save_dir = os.path.join(self.config.model_location, runname, scenario, 'postprocessing_plots_24aug25')
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    def get_combined_save_dir(self, discharge: int) -> str:
        runname = get_runname(discharge)
        combined_dir = os.path.join(self.config.model_location, runname, 'combined_scenario_postprocessing_plots_24aug25')
        os.makedirs(combined_dir, exist_ok=True)
        return combined_dir

#%% ANALYSIS FUNCTIONS
class DelftAnalyzer:
    """Main analyzer class containing all analysis methods"""
    
    def __init__(self, config: AnalysisConfig, dataset_manager: DatasetManager):
        self.config = config
        self.dm = dataset_manager
    
    def run_his_analysis(self, discharge: int, scenario: str, verbose: bool = True):
        """Run HIS analysis for a single scenario"""
        if not self.config.run_single_discharge_analysis:
            return
        
        if verbose:
            print(f"\n=== HIS ANALYSIS for {scenario} (Q={discharge}) ===")
        
        save_dir = self.dm.get_save_dir(discharge, scenario)
        
        with self.dm.get_dataset(discharge, scenario, 'trih') as dataset_trih:
            variable = 'ZWL'
            variable_label = 'Water Level [m]'
            station_names = [f'river_km_{i}' for i in range(3)]
            
            results, all_stations = extract_his_data(dataset_trih, variable, station_names)
            
            if verbose:
                for name in station_names:
                    time, var = results[name]
                    print(f"{name} time range: {time.min()} to {time.max()}")
                    print(f"{name} {variable} range: min={np.nanmin(var)}, max={np.nanmax(var)}")
            
            his_plot_timeseries(
                results, station_names, self.config.reference_date, variable, variable_label, 
                save_dir, self.config.save_figure,
                time_range=(int(self.config.slice_start * self.config.map_output_interval / self.config.his_output_interval),
                           int(self.config.slice_end * self.config.map_output_interval / self.config.his_output_interval))
            )
    
    def run_spatial_analysis(self, discharge: int, scenario: str, verbose: bool = True):
        """Run spatial analysis for a single scenario"""
        if not self.config.run_spatial_plots:
            return
        
        if verbose:
            print(f"\n=== SPATIAL ANALYSIS for {scenario} (Q={discharge}) ===")
        
        save_dir = self.dm.get_save_dir(discharge, scenario)
        x, y = self.dm.preload_coordinates(discharge, scenario)
        
        # Get cross-section coordinates
        col_indices, N_coords, x_targets = get_cross_section_coordinates(
            x, y, x_targets=np.arange(self.config.x_min, self.config.x_max, 100)
        )
        
        timesteps = np.arange(0, 10, 1)  # Adjust as needed
        
        with self.dm.get_dataset(discharge, scenario, 'trim') as dataset_trim:
            for timestep in timesteps:
                if verbose:
                    print(f"Processing timestep {timestep}...")
                
                # Load data for this timestep
                bed_level = load_single_timestep_variable(dataset_trim, "DPS", timestep=timestep)
                water_level = load_single_timestep_variable(dataset_trim, "S1", timestep=timestep)
                water_depth = water_level + bed_level
                
                # Create plots
                plot_map(x, y, bed_level, 'bed_level', col_indices, N_coords, timestep, scenario, save_dir, self.config.save_figure)
                plot_map(x, y, water_level, 'water_level', col_indices, N_coords, timestep, scenario, save_dir, self.config.save_figure)
                plot_map(x, y, water_depth, 'water_depth', col_indices, N_coords, timestep, scenario, save_dir, self.config.save_figure)
                
                # Velocity plots if needed
                if True:  # Set to False to skip velocity plots
                    velocity = load_single_timestep_variable(dataset_trim, "U1", timestep=timestep, remove=1, layer=0)
                    plot_velocity(x, y, velocity, 'U1', col_indices, N_coords, timestep, scenario, save_dir, self.config.save_figure)
    
    def run_width_averaged_analysis(self, discharge: int, scenario: str, verbose: bool = True):
        """Run width-averaged bed level analysis"""
        if not self.config.run_width_averaged_bedlevel:
            return None, None, None
        
        if verbose:
            print(f"\n=== WIDTH-AVERAGED ANALYSIS for {scenario} (Q={discharge}) ===")
        
        save_dir = self.dm.get_save_dir(discharge, scenario)
        x, y = self.dm.preload_coordinates(discharge, scenario)
        
        with self.dm.get_dataset(discharge, scenario, 'trim') as dataset_trim:
            # Load bedlev data
            bedlev = -1 * load_variable(dataset_trim, 'DPS', 
                                        range=slice(self.config.slice_start, 
                                                    self.config.slice_end))
            # Load morphological time
            morphtime = dataset_trim.variables['MORFT'][:]
            morph_days = np.array(morphtime[:])
            
            profiles, x_coords, labels = self._compute_width_averaged_profiles(x, y, 
                                         bedlev, morph_days, scenario, save_dir, verbose)
            
            return profiles, x_coords, labels
    
    def run_cumulative_analysis(self, discharge: int, scenario: str, verbose: bool = True):
        """Run cumulative width-averaged bed level change analysis"""
        if not self.config.run_cumulative_width_averaged_bedlevel:
            return
        
        if verbose:
            print(f"\n=== CUMULATIVE ANALYSIS for {scenario} (Q={discharge}) ===")
        
        save_dir = self.dm.get_save_dir(discharge, scenario)
        x, y = self.dm.preload_coordinates(discharge, scenario)
        
        with self.dm.get_dataset(discharge, scenario, 'trim') as dataset_trim:
            # Load bedlev data
            bedlev = -1 * load_variable(dataset_trim, 'DPS', range=slice(self.config.slice_start, self.config.slice_end))
            
            # Run cumulative analysis (implement the logic from your original code)
            self._compute_cumulative_bed_change(x, y, bedlev, scenario, save_dir, verbose)
    
    def run_hypsometric_analysis(self, discharge: int, scenario: str, verbose: bool = True):
        """Run hypsometric analysis for a single scenario"""
        if not self.config.run_hypsometric_analysis:
            return
        
        if verbose:
            print(f"\n=== HYPSOMETRIC ANALYSIS for {scenario} (Q={discharge}) ===")
        
        save_dir = self.dm.get_save_dir(discharge, scenario)
        x, y = self.dm.preload_coordinates(discharge, scenario)
        
        with self.dm.get_dataset(discharge, scenario, 'trim') as dataset_trim:
            bedlev = -1 * load_variable(dataset_trim, 'DPS', range=slice(self.config.slice_start, self.config.slice_end))
            
            # Define timesteps to analyze
            reference_t = 0
            analysis_timesteps = np.linspace(self.config.slice_start, self.config.slice_end, 5)
            analysis_timesteps = np.round(analysis_timesteps).astype(int)
            analysis_timesteps[-1] = self.config.slice_end  # force last value to exactly 360
            # Create hypsometric curves
            elevations_ref, areas_ref = plot_hypsometric_curves(
                bedlev, x, y, self.config.x_min, self.config.x_max, self.config.y_min, self.config.y_max,
                bed_threshold=self.config.bed_threshold,
                timesteps=analysis_timesteps,
                reference_timestep=reference_t,
                scenario=scenario,
                save_dir=save_dir,
                save_figure=self.config.save_figure
            )
            
            if verbose and len(elevations_ref) > 0:
                total_area = areas_ref[-1]
                min_elevation = elevations_ref[0]
                max_elevation = elevations_ref[-1]
                print(f"Total estuary area: {total_area:.2f} km²")
                print(f"Elevation range: {min_elevation:.2f} to {max_elevation:.2f} m")
    
    def _compute_width_averaged_profiles(self, x, y, bedlev, morph_days, scenario, save_dir, verbose):
        if verbose:
            print("\n=== WIDTH-AVERAGED BED LEVEL PROFILE ===")
            
        num_timesteps_to_plot = 2  # Or set as you need; can be from config or parameter
        
        # Extract x and y coordinates along/across estuary bounds
        x0 = x[:, 0]
        y0 = y[0, :]

        x_indices = np.where((x0 >= self.config.x_min) & (x0 <= self.config.x_max))[0]
        y_indices = np.where((y0 >= self.config.y_min) & (y0 <= self.config.y_max))[0]

        if verbose:
            print(f"Found {len(x_indices)} x-indices and {len(y_indices)} y-indices within estuary bounds")

        total_timesteps = bedlev.shape[0]
        if num_timesteps_to_plot >= total_timesteps:
            selected_timesteps = np.arange(total_timesteps)
        else:
            selected_timesteps = np.linspace(0, total_timesteps - 1, num_timesteps_to_plot, dtype=int)

        if verbose:
            print(f"Selected timesteps: {selected_timesteps} out of {total_timesteps} total timesteps")

        x_coords_estuary = x0[x_indices]

        # Initialize arrays to store results
        width_averaged_profiles = []
        width_std_profiles = []
        valid_x_coords = []
        timestep_labels = []

        for i, timestep in enumerate(selected_timesteps):
            if verbose:
                print(f"Processing timestep {timestep} ({i + 1}/{len(selected_timesteps)})...")

            bedlev_current = bedlev[timestep, :, :]
            bedlev_estuary = bedlev_current[np.ix_(x_indices, y_indices)]
            bedlev_estuary_masked = np.where(bedlev_estuary < self.config.bed_threshold, bedlev_estuary, np.nan)
            width_averaged_bedlev = np.nanmean(bedlev_estuary_masked, axis=1)
            width_std_bedlev = np.nanstd(bedlev_estuary_masked, axis=1)

            valid_indices = ~np.isnan(width_averaged_bedlev)

            x_coords_clean = x_coords_estuary[valid_indices]
            width_averaged_bedlev_clean = width_averaged_bedlev[valid_indices]
            width_std_bedlev_clean = width_std_bedlev[valid_indices]

            width_averaged_profiles.append(width_averaged_bedlev_clean)
            width_std_profiles.append(width_std_bedlev_clean)
            valid_x_coords.append(x_coords_clean)

            if timestep < len(morph_days):
                morph_time_years = morph_days[timestep] / 365.25
                timestep_labels.append(f't = {timestep} ({morph_time_years:.0f} years)')
            else:
                timestep_labels.append(f't = {timestep}')

        if verbose:
            print(f"Successfully computed width-averaged bed level profiles for {len(selected_timesteps)} timesteps")

        # # Now, create the two plots: width-averaged bed level evolution, and standard deviation evolution
        # self._plot_width_averaged_bedlevel_evolution(width_averaged_profiles, valid_x_coords, timestep_labels, scenario, save_dir)
        # self._plot_width_std_bedlevel_evolution(width_std_profiles, valid_x_coords, timestep_labels, scenario, save_dir)

        return width_averaged_profiles, valid_x_coords, timestep_labels
    
    def _plot_width_averaged_bedlevel_evolution(self, width_averaged_profiles, valid_x_coords, timestep_labels, scenario, save_dir):
        fig, ax = plt.subplots()
        colors = plt.cm.YlOrBr(np.linspace(0.3, 1, len(width_averaged_profiles)))

        for i, (profile, x_coords, label) in enumerate(zip(width_averaged_profiles, valid_x_coords, timestep_labels)):
            x_coords_km = x_coords / 1000
            ax.plot(x_coords_km, profile,
                    color=colors[i], marker='o', markersize=2,
                    label=label, alpha=0.8)

        ax.set_xlabel('Distance along estuary [km]')
        ax.set_ylabel('Width-averaged bed level [m]')
        ax.set_title(f'Width-averaged bed level evolution along estuary\n{scenario}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', labelcolor='linecolor')

        plt.tight_layout()

        if self.config.save_figure:
            plt.savefig(os.path.join(save_dir, f'width_averaged_bedlevel_evolution_{scenario}.png'),
                        transparent=True)

        plt.show()

    def _plot_width_std_bedlevel_evolution(self, width_std_profiles, valid_x_coords, timestep_labels, scenario, save_dir):
        fig, ax = plt.subplots()

        colors = plt.cm.plasma(np.linspace(0, 1, len(width_std_profiles)))

        for i, (std_profile, x_coords, label) in enumerate(zip(width_std_profiles, valid_x_coords, timestep_labels)):
            x_coords_km = x_coords / 1000
            ax.plot(x_coords_km, std_profile,
                    color=colors[i], marker='s', markersize=2,
                    label=label, alpha=0.8)

        ax.set_xlabel('Distance along estuary [km]')
        ax.set_ylabel('Standard deviation of bed level [m]')
        ax.set_title(f'Width-averaged bed level variability (std dev) along estuary\n{scenario}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.legend(loc='upper left', labelcolor='linecolor')

        plt.tight_layout()

        if self.config.save_figure:
            plt.savefig(os.path.join(save_dir, f'width_std_bedlevel_evolution_{scenario}_{defaultcolour}.png'),
                        transparent=True)

        plt.show()

    def _plot_combined_width_averaged_profiles_final_timesteps(self, results_per_scenario, save_dir):
        plt.figure(figsize=defaultfigsize)

        # Map scenario names to colors
        color_map = {
            '01_baserun500': 'tab:blue',
            '02_run500_seasonal': 'tab:orange',
            '03_run500_flashy': 'tab:green'
        }

        # Map scenario names to colors
        label_map = {
            '01_baserun500': 'Constant discharge',
            '02_run500_seasonal': 'Seasonal discharge',
            '03_run500_flashy': 'Flashy discharge'
        }

        initial_color = 'grey'  # Color for t=0 profiles

        plotted_initial = False  # To ensure 't=0' label appears only once

        for scenario, data in results_per_scenario.items():
            if data is None:
                continue
            width_averaged_profiles, valid_x_coords, timestep_labels = data

            # Plot initial timestep profile (t=0) once, in grey
            if not plotted_initial:
                if len(width_averaged_profiles) > 0:
                    x_init = valid_x_coords[0] / 1000
                    profile_init = width_averaged_profiles[0]
                    plt.plot(x_init, profile_init, color=initial_color, linestyle='--', label='t = 0')
                    plotted_initial = True

            # Plot final timestep profile with scenario-specific color
            if len(width_averaged_profiles) > 1:
                x_final = valid_x_coords[-1] / 1000
                profile_final = width_averaged_profiles[-1]
                color = color_map.get(scenario, 'black')  # Default to black if not found
                label_name = label_map.get(scenario, scenario) 
                plt.plot(x_final, profile_final, color=color, label=f'{label_name}')

        plt.xlabel('Distance along estuary [km]')
        plt.ylabel('Width-averaged bed level [m]')
        plt.title('Width-averaged bed level profiles at initial and final timestep for all scenarios')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left', labelcolor='linecolor')

        plt.tight_layout()

        if self.config.save_figure:
            plt.savefig(os.path.join(save_dir, f'combined_width_averaged_bedlevel_evolution_allscenarios_{defaultcolour}.png'),
                        transparent=True)
            
        plt.show()

    def _compute_cumulative_bed_change(self, x, y, bedlev, scenario, save_dir, verbose):
        """Helper method for cumulative analysis"""
        # Implement the cumulative analysis logic from your original code
        if verbose:
            print("Computing cumulative bed level changes...")
        
        # Get the x coordinates along estuary
        along_x = x[:, 0]
        x_inds = np.where((along_x >= self.config.x_min) & (along_x <= self.config.x_max))[0]
        along_x_estuary = along_x[x_inds]
        
        # Mask land locations
        bedlev_masked = np.where(bedlev < self.config.bed_threshold, bedlev, np.nan)
        
        # Compute width-averaged bed level
        width_avg_bedlev = np.nanmean(bedlev_masked, axis=2)
        width_avg_bedlev_estuary = width_avg_bedlev[:, x_inds]
        
        # Calculate differences and cumulative activity
        differences = np.diff(width_avg_bedlev_estuary, axis=0)
        zeros_row = np.zeros((1, width_avg_bedlev_estuary.shape[1]))
        abs_differences = np.abs(differences)
        abs_differences_with_prepended = np.vstack([zeros_row, abs_differences])
        cumulative_activity = np.cumsum(abs_differences_with_prepended, axis=0)
        
        # Create plot
        self._plot_cumulative_heatmap(cumulative_activity, along_x_estuary, scenario, save_dir)
    
    def _plot_cumulative_heatmap(self, cumulative_activity, along_x_estuary, scenario, save_dir):
        """Create cumulative activity heatmap"""
        fig, ax = plt.subplots()
        
        ntime = cumulative_activity.shape[0]  # number of daily outputs
        
        morfac = 400
        y_morph_years = np.arange(ntime) * morfac / 365  # time in morphological years

        extent = [along_x_estuary.min() / 1000, along_x_estuary.max() / 1000, 0, y_morph_years[-1]]

        im = ax.imshow(
            cumulative_activity,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap='viridis',
            vmin=0,
            vmax=10
        )
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label(r'$\Sigma |\Delta h|$ [m]')
        
        # Labels and title
        ax.set_xlabel('Distance along estuary [km]')
        ax.set_ylabel('Time [years]')
        # ax.set_title(f'{scenario}: Cumulative bed level change')
        
        # Add statistics text box
        final_cumulative = cumulative_activity[-1, :]  # last timestep values (shape = nx)
        x_maxvalue_estuary = along_x_estuary[np.argmax(final_cumulative)] / 1000
        x_minvalue_estuary = along_x_estuary[np.argmin(final_cumulative)] / 1000
        max_val, min_val = np.max(final_cumulative), np.min(final_cumulative)
        
        textstr = (
        rf"$(\Sigma |\Delta h|)_{{\mathrm{{max}}}}$ = {max_val:.2f} m at {x_maxvalue_estuary:.1f} km"
        "\n"
        rf"$(\Sigma |\Delta h|)_{{\mathrm{{min}}}}$ = {min_val:.2f} m at {x_minvalue_estuary:.1f} km"
        )

        ax.text(0.02, 0.06, textstr, transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white", alpha=0.7))
        
        plt.tight_layout()
        if self.config.save_figure:
            plt.savefig(os.path.join(save_dir, f'cumulative_bed_change_heatmap_{scenario}_{defaultcolour}.png'),
                       transparent=True)
        plt.show()

#%% MAIN EXECUTION CLASS
class DelftAnalysisRunner:
    """Main runner class that orchestrates all analyses"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.dm = DatasetManager(config)
        self.analyzer = DelftAnalyzer(config, self.dm)
    
    def run_all_scenarios(self, verbose: bool = True):
        try:
            total_scenarios = len(self.config.discharges) * len(self.config.scenario_templates)
            current_scenario = 0

            for discharge in self.config.discharges:
                runname = get_runname(discharge)
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"PROCESSING DISCHARGE: {discharge} m³/s ({runname})")
                    print(f"{'='*60}")

                results_per_scenario = {}  # reset here per discharge

                for scenario_template in self.config.scenario_templates:
                    current_scenario += 1
                    scenario = scenario_template.format(discharge=discharge)

                    if verbose:
                        print(f"\n--- Scenario {current_scenario}/{total_scenarios}: {scenario} ---")

                    try:
                        trim_file = self.dm._get_file_path(discharge, scenario, 'trim')
                        trih_file = self.dm._get_file_path(discharge, scenario, 'trih')

                        if not os.path.exists(trim_file):
                            print(f"Warning: Trim file not found, skipping {scenario}")
                            continue
                        if not os.path.exists(trih_file):
                            print(f"Warning: Trih file not found, skipping {scenario}")
                            continue

                        width_averaged_result = self.run_single_scenario(discharge, scenario, verbose=verbose)
                        results_per_scenario[scenario] = width_averaged_result

                    except Exception as e:
                        print(f"Error processing {scenario}: {e}")
                        if verbose:
                            import traceback
                            traceback.print_exc()
                        continue

                if self.config.run_combined_width_averaged_bedlevel:            
                    # Plot combined widths for this discharge only, after all scenarios processed
                    combined_save_dir = self.dm.get_combined_save_dir(discharge)
                    self.analyzer._plot_combined_width_averaged_profiles_final_timesteps(results_per_scenario, combined_save_dir)

            # Remaining multi-scenario analyses if any
            if self.config.run_multi_scenario_hypsometric:
                self.run_multi_scenario_hypsometric(verbose=verbose)
            if self.config.run_multi_variable_analysis:
                self.run_multi_variable_analysis(verbose=verbose)

        finally:
            self.dm.close_all()
    
    def run_single_scenario(self, discharge: int, scenario: str, verbose: bool = True):
        """Run all analyses for a single scenario"""
        # Run individual analyses
        self.analyzer.run_his_analysis(discharge, scenario, verbose)
        self.analyzer.run_spatial_analysis(discharge, scenario, verbose)

        width_averaged_result = self.analyzer.run_width_averaged_analysis(discharge, scenario, verbose)
        
        self.analyzer.run_cumulative_analysis(discharge, scenario, verbose)
        self.analyzer.run_hypsometric_analysis(discharge, scenario, verbose)

        return width_averaged_result
    
    def run_multi_scenario_hypsometric(self, verbose: bool = True):
        """Run multi-scenario hypsometric analysis"""
        if not self.config.run_multi_scenario_hypsometric:
            if verbose:
                print('MULTI-SCENARIO HYPSOMETRIC CURVE ANALYSIS skipped.')
            return
            
        if verbose:
            print(f"\n{'='*60}")
            print("MULTI-SCENARIO HYPSOMETRIC ANALYSIS")
            print(f"{'='*60}")
            print("Starting multi-scenario hypsometric curve analysis...")
        
        # Define timesteps for individual scenario plots
        reference_t = 0
                
        analysis_timesteps = np.linspace(self.config.slice_start, self.config.slice_end, 5)
        analysis_timesteps = np.round(analysis_timesteps).astype(int)
        analysis_timesteps[-1] = self.config.slice_end  # force last value to exactly 360
        
        # Store all scenario data for comparison plot
        all_scenario_data = {}
        
        # Process each discharge scenario
        for discharge in self.config.discharges:
            if verbose:
                print(f"\n=== Processing discharge Q = {discharge} m³/s ===")
            
            # Process each scenario for this discharge
            for scenario_template in self.config.scenario_templates:
                scenario = scenario_template.format(discharge=discharge)
                if verbose:
                    print(f"\nProcessing scenario: {scenario}")
                
                try:
                    # Get save directory and coordinates
                    save_dir = self.dm.get_save_dir(discharge, scenario)
                    x, y = self.dm.preload_coordinates(discharge, scenario)
                    
                    # Load dataset and bed level data
                    with self.dm.get_dataset(discharge, scenario, 'trim') as dataset_trim:
                        if verbose:
                            print(f"Loading bed level data...")
                        bedlev = -1 * load_variable(dataset_trim, 'DPS', 
                                                range=slice(self.config.slice_start, self.config.slice_end))
                        
                        if verbose:
                            print(f"Bed level shape: {bedlev.shape}")
                        
                        # Store data for scenario comparison plot
                        scenario_key = f"{scenario}_{discharge}"
                        all_scenario_data[scenario_key] = (x, y, bedlev)
                        
                        # Create individual hypsometric plot for this scenario
                        if verbose:
                            print(f"Creating individual hypsometric plot for {scenario}...")
                        elevations_ref, areas_ref = plot_hypsometric_curves(
                            bedlev, x, y, self.config.x_min, self.config.x_max, 
                            self.config.y_min, self.config.y_max,
                            bed_threshold=self.config.bed_threshold,
                            timesteps=analysis_timesteps,
                            reference_timestep=reference_t,
                            scenario=scenario,
                            save_dir=save_dir,
                            save_figure=self.config.save_figure
                        )
                        
                        # Print statistics if verbose
                        if verbose and len(elevations_ref) > 0:
                            total_area = areas_ref[-1]
                            min_elevation = elevations_ref[0]
                            max_elevation = elevations_ref[-1]
                            print(f"Total estuary area: {total_area:.2f} km²")
                            print(f"Elevation range: {min_elevation:.2f} to {max_elevation:.2f} m")
                    
                except Exception as e:
                    if verbose:
                        print(f"Error processing {scenario} (Q={discharge}): {e}")
                    continue
        
        # Create scenario comparison plot
        if len(all_scenario_data) > 1:
            if verbose:
                print(f"\n=== Creating scenario comparison plots ===")
                print(f"Available scenarios: {list(all_scenario_data.keys())}")
            
            for discharge in self.config.discharges:  # Create comparison plot for each discharge
                # Filter scenarios for this discharge
                discharge_scenarios = {k: v for k, v in all_scenario_data.items() 
                                    if f'_{discharge}' in k}
                
                if len(discharge_scenarios) > 0:
                    if verbose:
                        print(f"Creating comparison plot for Q = {discharge} m³/s")
                    
                    # Define scenario colors (customize as needed)
                    scenario_colors = {
                        'baserun': 'tab:blue',
                        'seasonal': 'tab:orange',
                        'flashy': 'tab:green'
                    }
                                 
                    plot_scenario_comparison_hypsometric(
                        discharge_scenarios,
                        self.config.x_min, self.config.x_max, self.config.y_min, self.config.y_max, 
                        bed_threshold=self.config.bed_threshold, 
                        reference_timestep=0,
                        final_timestep=-1,  # Use last timestep
                        scenario_colors=scenario_colors,
                        save_dir=self.dm.get_combined_save_dir(discharge),
                        save_figure=self.config.save_figure,
                        discharge=discharge
                    )
                else:
                    if verbose:
                        print(f"No scenarios found for discharge Q = {discharge} m³/s")
        else:
            if verbose:
                print("Not enough scenarios loaded for comparison plot")
        
        if verbose:
            print("Multi-scenario hypsometric curve analysis completed.")
    
    def run_multi_variable_analysis(self, verbose: bool = True):
        """Run multi-variable analysis across scenarios"""
        if verbose:
            print(f"\n{'='*60}")
            print("MULTI-VARIABLE ANALYSIS")
            print(f"{'='*60}")
        
        # Implement multi-variable analysis
        # This would be similar to your existing multi-variable code
        pass
    
    def run_specific_scenarios(self, scenario_indices: List[int], discharge_indices: List[int] = None, verbose: bool = True):
        """Run analysis for specific scenarios only"""
        if discharge_indices is None:
            discharge_indices = list(range(len(self.config.discharges)))
        
        selected_discharges = [self.config.discharges[i] for i in discharge_indices]
        selected_scenarios = [self.config.scenario_templates[i] for i in scenario_indices]
        
        original_discharges = self.config.discharges
        original_scenarios = self.config.scenario_templates
        
        try:
            # Temporarily modify config
            self.config.discharges = selected_discharges
            self.config.scenario_templates = selected_scenarios
            
            # Run analysis
            self.run_all_scenarios(verbose=verbose)
            
        finally:
            # Restore original config
            self.config.discharges = original_discharges
            self.config.scenario_templates = original_scenarios

#%% MAIN EXECUTION
if __name__ == "__main__":
    # Initialize configuration
    config = AnalysisConfig()
    
    # You can modify analysis flags here
    config.run_spatial_plots = False 

    config.run_cumulative_width_averaged_bedlevel = True

    config.run_multi_scenario_hypsometric = False
    config.run_width_averaged_bedlevel = False
    config.run_combined_width_averaged_bedlevel = False     #if True, requirement: run_width_averaged_bedlevel = True

    # Initialize and run analysis
    runner = DelftAnalysisRunner(config)
    
    print("Starting Delft3D analysis for all scenarios...")
    print(f"Discharges to process: {config.discharges}")
    print(f"Scenarios to process: {config.scenario_templates}")
    print(f"Analysis flags: ")
    for attr in dir(config):
        if attr.startswith('run_'):
            print(f"  {attr}: {getattr(config, attr)}")
    
    # # Run all scenarios
    runner.run_all_scenarios(verbose=True)
    
    #Alternative: run specific scenarios only
    # runner.run_specific_scenarios([0], verbose=True)  # Run first two scenario templates
    
    print("\nAnalysis completed!")
# %%
