"""Delft3D-FM NetCDF Analysis: Morphological Estuary Analysis with dfm-tools
Adapted for partitioned Delft3D-FM output
Last edit: December 2025
Author: Adapted from Marloes Bonenkamp's script
"""

#%% IMPORTS AND SETUP
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import time
import os
import sys
import dfm_tools as dfmt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

def create_terrain_colormap():
    """Create a custom terrain-like colormap for bed level visualization."""
    colors = [
        (0.00, "#000066"),   # deep water
        (0.10, "#0000ff"),   # blue
        (0.30, "#00ffff"),   # cyan
        (0.40, "#00ffff"),   # water edge
        (0.50, "#ffffcc"),   # land edge
        (0.60, "#ffcc00"),   # orange
        (0.75, "#cc6600"),   # brown
        (0.90, "#228B22"),   # green
        (1.00, "#006400"),   # dark green
    ]
    return LinearSegmentedColormap.from_list("custom_terrain", colors)

terrain_like = create_terrain_colormap()

#%% PLOTTING SETTINGS   
defaultcolour = 'white'
defaultfont = 20
defaultfigsize = (10, 5.3)

mpl.rcParams['text.color'] = 'black'
mpl.rcParams['font.size'] = defaultfont
mpl.rcParams['axes.titlesize'] = defaultfont + 4
mpl.rcParams['axes.titlecolor'] = defaultcolour
mpl.rcParams['axes.labelsize'] = defaultfont
mpl.rcParams['axes.labelcolor'] = defaultcolour
mpl.rcParams['axes.facecolor'] = defaultcolour
mpl.rcParams['xtick.labelsize'] = defaultfont
mpl.rcParams['xtick.color'] = defaultcolour
mpl.rcParams['ytick.labelsize'] = defaultfont
mpl.rcParams['ytick.color'] = defaultcolour
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['figure.figsize'] = defaultfigsize
mpl.rcParams['legend.fontsize'] = defaultfont
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['savefig.dpi'] = 600

#%% CONFIGURATION CLASS
class FMAnalysisConfig:
    """Configuration class for Delft3D-FM analysis parameters"""
    
    def __init__(self):        
        # Model configuration
        self.config = 'YourFMModel'
        self.model_location = r"U:\Path\To\Your\FM\Models"
        
        # Discharge scenarios to process
        self.discharges = [500]
        
        # Scenario templates
        self.scenario_templates = [
            '01_baserun{discharge}', 
            '02_run{discharge}_seasonal', 
            '03_run{discharge}_flashy'
        ]
        
        # File naming pattern for partitioned output
        self.map_filename_pattern = "*_map.nc"
        self.his_filename_pattern = "*_his.nc"
        
        # Coordinate reference systems
        self.from_crs = 'EPSG:3857'  # Source CRS
        self.to_crs = 'EPSG:3857'   # Target CRS 
        
        # Time settings
        self.reference_date = datetime.datetime(2024, 1, 1)
        self.time_slice_start = 0
        self.time_slice_end = -1  # -1 means last timestep
        self.num_timesteps_to_plot = 4
        
        # Spatial extent (in target CRS coordinates)
        self.x_min, self.x_max = 20000, 45000
        self.y_min, self.y_max = 5000, 10000
        
        # Thresholds
        self.bed_threshold = 6  # Elevation threshold for land/water
        
        # Analysis flags
        self.run_spatial_plots = False
        self.run_width_averaged_bedlevel = True
        self.run_cumulative_width_averaged_bedlevel = False
        self.run_hypsometric_analysis = False
        self.run_combined_analysis = False
        
        # Save settings
        self.save_figure = True

#%% DATASET MANAGER CLASS FOR FM
class FMDatasetManager:
    """Manages Delft3D-FM partitioned datasets using dfm-tools"""
    
    def __init__(self, config: FMAnalysisConfig):
        self.config = config
        self._datasets = {}
        self._coordinates_cache = {}
    
    def load_dataset(self, discharge: int, scenario: str, dataset_type: str = 'map'):
        """Load partitioned FM dataset using dfm-tools"""
        key = f"{discharge}_{scenario}_{dataset_type}"
        
        if key not in self._datasets:
            file_pattern = self._get_file_pattern(discharge, scenario, dataset_type)
            
            if not os.path.exists(os.path.dirname(file_pattern)):
                raise FileNotFoundError(f"Directory not found: {os.path.dirname(file_pattern)}")
            
            print(f"Loading {dataset_type} files for {scenario} (Q={discharge})...")
            start_time = time.time()
            
            # Load partitioned dataset
            dataset = dfmt.open_partitioned_dataset(file_pattern)
            
            # Set and convert CRS if needed
            if self.config.from_crs and self.config.to_crs:
                dataset.ugrid.set_crs(self.config.from_crs)
                dataset = dataset.ugrid.to_crs(self.config.to_crs)
            
            self._datasets[key] = dataset
            print(f"{dataset_type} files loaded in {time.time() - start_time:.2f} seconds")
            
            # Print available variables
            print(f"Available variables: {list(dataset.data_vars)}")
        
        return self._datasets[key]
    
    def _get_file_pattern(self, discharge: int, scenario: str, dataset_type: str) -> str:
        """Get file pattern for partitioned output"""
        runname = f"Q{discharge}"
        
        if dataset_type == 'map':
            pattern = self.config.map_filename_pattern
        elif dataset_type == 'his':
            pattern = self.config.his_filename_pattern
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return os.path.join(self.config.model_location, runname, scenario, 
                           'DFM_OUTPUT_*', pattern)
    
    def get_face_coordinates(self, dataset):
        """Extract face center coordinates from FM dataset"""
        # Try different possible coordinate variable names
        x_vars = ['mesh2d_face_x', 'FlowElem_xcc', 'face_x']
        y_vars = ['mesh2d_face_y', 'FlowElem_ycc', 'face_y']
        
        x_coord = None
        y_coord = None
        
        for x_var in x_vars:
            if x_var in dataset:
                x_coord = dataset[x_var].values
                break
        
        for y_var in y_vars:
            if y_var in dataset:
                y_coord = dataset[y_var].values
                break
        
        if x_coord is None or y_coord is None:
            raise ValueError("Could not find face coordinate variables in dataset")
        
        return x_coord, y_coord
    
    def close_all(self):
        """Close all open datasets"""
        for key, dataset in self._datasets.items():
            try:
                dataset.close()
                print(f"Closed dataset: {key}")
            except Exception as e:
                print(f"Error closing dataset {key}: {e}")
        
        self._datasets.clear()
        self._coordinates_cache.clear()
    
    def get_save_dir(self, discharge: int, scenario: str) -> str:
        """Get save directory for a scenario"""
        runname = f"Q{discharge}"
        save_dir = os.path.join(self.config.model_location, runname, scenario, 
                               'postprocessing_plots_fm')
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
    
    def get_combined_save_dir(self, discharge: int) -> str:
        """Get combined save directory for all scenarios"""
        runname = f"Q{discharge}"
        combined_dir = os.path.join(self.config.model_location, runname, 
                                   'combined_scenario_plots_fm')
        os.makedirs(combined_dir, exist_ok=True)
        return combined_dir

#%% FM ANALYZER CLASS
class FMAnalyzer:
    """Main analyzer class for Delft3D-FM output"""
    
    def __init__(self, config: FMAnalysisConfig, dataset_manager: FMDatasetManager):
        self.config = config
        self.dm = dataset_manager
    
    def plot_spatial_map(self, discharge: int, scenario: str, 
                        variable: str, timestep: int = 0,
                        vmin=None, vmax=None, cmap='viridis'):
        """Create spatial plot for a single variable and timestep"""
        
        dataset = self.dm.load_dataset(discharge, scenario, 'map')
        save_dir = self.dm.get_save_dir(discharge, scenario)
        
        # Select data at timestep
        if variable not in dataset:
            print(f"Warning: Variable {variable} not found in dataset")
            return
        
        data = dataset[variable].isel(time=timestep)
        
        # Get time info
        time_value = dataset['time'].isel(time=timestep).values
        time_str = pd.Timestamp(time_value).strftime('%Y-%m-%d %H:%M')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot using ugrid
        pc = data.ugrid.plot(ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, 
                            add_colorbar=False, edgecolors='face')
        
        # Set spatial extent
        ax.set_xlim(self.config.x_min, self.config.x_max)
        ax.set_ylim(self.config.y_min, self.config.y_max)
        ax.set_aspect('equal')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(pc, cax=cax)
        cbar.set_label(f"{variable}", fontsize=defaultfont)
        
        ax.set_xlabel('X [m]', fontsize=defaultfont)
        ax.set_ylabel('Y [m]', fontsize=defaultfont)
        ax.set_title(f'{scenario} - {variable}\nTime: {time_str}', 
                    fontsize=defaultfont + 2)
        
        plt.tight_layout()
        
        if self.config.save_figure:
            filename = f'{variable}_t{timestep:04d}_{scenario}.png'
            plt.savefig(os.path.join(save_dir, filename), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compute_width_averaged_profile(self, discharge: int, scenario: str, 
                                       variable: str = 'mesh2d_flowelem_bl'):
        """Compute width-averaged profiles along estuary"""
        
        print(f"\n=== Computing width-averaged profiles for {scenario} ===")
        
        dataset = self.dm.load_dataset(discharge, scenario, 'map')
        save_dir = self.dm.get_save_dir(discharge, scenario)
        
        # Get coordinates
        x_coords, y_coords = self.dm.get_face_coordinates(dataset)
        
        # Get the variable data
        if variable not in dataset:
            print(f"Warning: Variable {variable} not found")
            return None
        
        data = dataset[variable]
        
        # Filter by spatial extent
        mask_x = (x_coords >= self.config.x_min) & (x_coords <= self.config.x_max)
        mask_y = (y_coords >= self.config.y_min) & (y_coords <= self.config.y_max)
        spatial_mask = mask_x & mask_y
        
        # Determine timesteps to analyze
        total_timesteps = len(dataset['time'])
        if self.config.num_timesteps_to_plot >= total_timesteps:
            timesteps = np.arange(total_timesteps)
        else:
            timesteps = np.linspace(0, total_timesteps - 1, 
                                   self.config.num_timesteps_to_plot, dtype=int)
        
        # Create bins along x-direction
        x_bins = np.linspace(self.config.x_min, self.config.x_max, 50)
        x_centers = (x_bins[:-1] + x_bins[1:]) / 2
        
        profiles = []
        profile_stds = []
        time_labels = []
        
        for t_idx in timesteps:
            print(f"Processing timestep {t_idx}/{total_timesteps}...")
            
            # Get data at this timestep
            data_t = data.isel(time=t_idx).values
            
            # Apply spatial mask
            data_masked = data_t[spatial_mask]
            x_masked = x_coords[spatial_mask]
            y_masked = y_coords[spatial_mask]
            
            # Apply threshold (e.g., for bed level)
            if variable in ['mesh2d_flowelem_bl', 'mesh2d_waterdepth']:
                valid_mask = data_masked < self.config.bed_threshold
                data_masked = data_masked[valid_mask]
                x_masked = x_masked[valid_mask]
            
            # Compute width-averaged profile
            profile_mean = []
            profile_std = []
            
            for i in range(len(x_bins) - 1):
                bin_mask = (x_masked >= x_bins[i]) & (x_masked < x_bins[i + 1])
                if np.sum(bin_mask) > 0:
                    profile_mean.append(np.mean(data_masked[bin_mask]))
                    profile_std.append(np.std(data_masked[bin_mask]))
                else:
                    profile_mean.append(np.nan)
                    profile_std.append(np.nan)
            
            profiles.append(np.array(profile_mean))
            profile_stds.append(np.array(profile_std))
            
            # Get time label
            time_value = dataset['time'].isel(time=t_idx).values
            time_str = pd.Timestamp(time_value).strftime('%Y-%m-%d')
            time_labels.append(f't={t_idx} ({time_str})')
        
        # Plot results
        self._plot_width_averaged_profiles(x_centers, profiles, time_labels, 
                                          variable, scenario, save_dir)
        
        return x_centers, profiles, time_labels
    
    def _plot_width_averaged_profiles(self, x_coords, profiles, labels, 
                                     variable, scenario, save_dir):
        """Plot width-averaged profiles"""
        
        fig, ax = plt.subplots(figsize=defaultfigsize)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))
        
        for i, (profile, label) in enumerate(zip(profiles, labels)):
            x_km = x_coords / 1000
            ax.plot(x_km, profile, color=colors[i], label=label, 
                   marker='o', markersize=3, alpha=0.8)
        
        ax.set_xlabel('Distance along estuary [km]')
        ax.set_ylabel(f'Width-averaged {variable}')
        ax.set_title(f'Width-averaged {variable} evolution\n{scenario}')
        ax.legend(loc='best', fontsize=defaultfont - 2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_figure:
            filename = f'width_averaged_{variable}_{scenario}.png'
            plt.savefig(os.path.join(save_dir, filename), 
                       dpi=300, bbox_inches='tight', transparent=True)
        
        plt.show()
    
    def create_difference_map(self, discharge: int, base_scenario: str, 
                             comparison_scenario: str, variable: str, 
                             timestep: int = -1):
        """Create difference map between two scenarios"""
        
        print(f"\n=== Creating difference map: {comparison_scenario} - {base_scenario} ===")
        
        # Load datasets
        base_data = self.dm.load_dataset(discharge, base_scenario, 'map')
        comp_data = self.dm.load_dataset(discharge, comparison_scenario, 'map')
        
        # Get variable at timestep
        base_var = base_data[variable].isel(time=timestep)
        comp_var = comp_data[variable].isel(time=timestep)
        
        # Compute difference
        diff = comp_var - base_var
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use diverging colormap
        vmax = np.nanmax(np.abs(diff.values))
        pc = diff.ugrid.plot(ax=ax, cmap=cm.get_cmap('RdBu_r'), 
                            vmin=-vmax, vmax=vmax,
                            add_colorbar=False, edgecolors='face')
        
        # Set spatial extent
        ax.set_xlim(self.config.x_min, self.config.x_max)
        ax.set_ylim(self.config.y_min, self.config.y_max)
        ax.set_aspect('equal')
        
        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(pc, cax=cax)
        cbar.set_label(f'Δ{variable}', fontsize=defaultfont)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Difference: {comparison_scenario} - {base_scenario}\n{variable}')
        
        plt.tight_layout()
        
        if self.config.save_figure:
            save_dir = self.dm.get_save_dir(discharge, comparison_scenario)
            filename = f'diff_{variable}_{comparison_scenario}_vs_{base_scenario}.png'
            plt.savefig(os.path.join(save_dir, filename), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()

#%% MAIN RUNNER CLASS
class FMAnalysisRunner:
    """Main runner orchestrating all FM analyses"""
    
    def __init__(self, config: FMAnalysisConfig):
        self.config = config
        self.dm = FMDatasetManager(config)
        self.analyzer = FMAnalyzer(config, self.dm)
    
    def run_all_scenarios(self, verbose: bool = True):
        """Run analysis for all configured scenarios"""
        
        try:
            for discharge in self.config.discharges:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"PROCESSING DISCHARGE: {discharge} m³/s")
                    print(f"{'='*60}")
                
                for scenario_template in self.config.scenario_templates:
                    scenario = scenario_template.format(discharge=discharge)
                    
                    if verbose:
                        print(f"\n--- Processing scenario: {scenario} ---")
                    
                    try:
                        self.run_single_scenario(discharge, scenario, verbose)
                    except Exception as e:
                        print(f"Error processing {scenario}: {e}")
                        if verbose:
                            import traceback
                            traceback.print_exc()
        
        finally:
            self.dm.close_all()
    
    def run_single_scenario(self, discharge: int, scenario: str, verbose: bool = True):
        """Run analyses for a single scenario"""
        
        # Spatial plots
        if self.config.run_spatial_plots:
            variables_to_plot = ['mesh2d_waterdepth', 'mesh2d_ucmag', 
                               'mesh2d_flowelem_bl']
            for var in variables_to_plot:
                try:
                    self.analyzer.plot_spatial_map(discharge, scenario, var, 
                                                  timestep=0)
                except Exception as e:
                    if verbose:
                        print(f"Could not plot {var}: {e}")
        
        # Width-averaged analysis
        if self.config.run_width_averaged_bedlevel:
            self.analyzer.compute_width_averaged_profile(discharge, scenario,
                                                        'mesh2d_flowelem_bl')

#%% MAIN EXECUTION
if __name__ == "__main__":
    # Initialize configuration
    config = FMAnalysisConfig()
    
    # Customize settings
    config.model_location = r"U:\PhDNaturalRhythmEstuaries\Models\FM_Models"
    config.discharges = [500]
    config.scenario_templates = ['01_baserun{discharge}']
    
    # Set analysis flags
    config.run_spatial_plots = True
    config.run_width_averaged_bedlevel = True
    
    # Run analysis
    runner = FMAnalysisRunner(config)
    
    print("Starting Delft3D-FM analysis...")
    print(f"Model location: {config.model_location}")
    print(f"Discharges: {config.discharges}")
    print(f"Scenarios: {config.scenario_templates}")
    
    runner.run_all_scenarios(verbose=True)
    
    print("\nAnalysis completed!")