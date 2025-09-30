#%% 
"""Delft3D-4 Flow NetCDF Post-Processing Script with dictionary structure
Last edit: June 2025
Author: Marloes Bonenkamp
"""

#%%
import seaborn as sns
import datetime
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"c:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Python\03_Model_postprocessing")
#%% Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

#%%

from FUNCTIONS.FUNCS_postprocessing_general import *
from FUNCTIONS.FUNCS_postprocessing_braiding_index import *
from FUNCTIONS.FUNCS_postprocessing_map_output import *
from FUNCTIONS.FUNCS_postprocessing_his_output import *

#%%
@dataclass
class ModelConfig:
    """Configuration class inspired by Delft3D-FM approach"""
    name: str
    discharge: int
    scenario: str
    model_location: Path
    trim_file: Path
    trih_file: Path
    save_dir: Path
    map_output_interval: int
    his_output_interval: int
    total_duration: int
    reference_date: datetime.datetime
    is_small_estuary: bool = False
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.save_dir.mkdir(parents=True, exist_ok=True)

class DelftDataManager:
    """Data management class inspired by dfm_tools approach"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.datasets = {}
        self.coordinates = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_datasets(self) -> Dict[str, nc.Dataset]:
        """Load datasets with timing - inspired by FM's structured loading"""
        datasets = {}
        
        for file_type, file_path in [('trim', self.config.trim_file), 
                                   ('trih', self.config.trih_file)]:
            self.logger.info(f'Loading {file_type}_file...')
            start_time = time.time()
            
            try:
                datasets[file_type] = nc.Dataset(file_path, mode='r')
                elapsed_time = time.time() - start_time
                self.logger.info(f"Data loading '{file_type}_file' took: {elapsed_time:.4f} seconds")
            except Exception as e:
                self.logger.error(f"Failed to load {file_type}_file: {e}")
                raise
                
        self.datasets = datasets
        return datasets
    
    def load_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and cache coordinates"""
        if not self.coordinates:
            self.logger.info('Loading coordinates...')
            self.coordinates['x'] = load_variable(self.datasets['trim'], "XCOR")
            self.coordinates['y'] = load_variable(self.datasets['trim'], "YCOR")
            self.logger.info('Coordinates loaded')
        
        return self.coordinates['x'], self.coordinates['y']
    
    def get_cross_sections(self) -> Tuple[List, List, List]:
        """Get cross-section coordinates for analysis"""
        x, y = self.load_coordinates()
        col_indices, N_coords, x_targets = get_cross_section_coordinates(x, y)
        self.logger.info(f"Cross-sections defined at x-coordinates: {x_targets}")
        return col_indices, N_coords, x_targets

class PostProcessor:
    """Main processing class inspired by FM's organized approach"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_manager = DelftDataManager(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load datasets and coordinates
        self.datasets = self.data_manager.load_datasets()
        self.x, self.y = self.data_manager.load_coordinates()
        self.col_indices, self.N_coords, self.x_targets = self.data_manager.get_cross_sections()
        
    def process_spatial_maps(self, timesteps: List[int], variables: List[str] = None,
                           save_figure: bool = True) -> None:
        """Process spatial maps with configurable variables - inspired by FM's plotting loops"""
        
        if variables is None:
            variables = ['bed_level', 'water_level', 'water_depth']
            
        variable_mapping = {
            'bed_level': ('DPS', lambda bed, wl=None: bed),
            'water_level': ('S1', lambda wl, bed=None: wl),
            'water_depth': ('S1', lambda wl, bed: wl + bed)  # Requires both
        }
        
        self.logger.info(f"\n=== PROCESSING SPATIAL MAPS FOR {len(timesteps)} TIMESTEPS ===")
        
        for timestep in timesteps:
            self.logger.info(f"Processing timestep {timestep}...")
            
            # Load required base data
            data_cache = {}
            
            for var_name in variables:
                nc_var, transform_func = variable_mapping[var_name]
                
                # Load base variables if not cached
                if 'bed_level' not in data_cache and ('bed' in transform_func.__code__.co_varnames or var_name == 'bed_level'):
                    data_cache['bed_level'] = load_single_timestep_variable(
                        self.datasets['trim'], "DPS", timestep=timestep)
                        
                if 'water_level' not in data_cache and ('wl' in transform_func.__code__.co_varnames or var_name == 'water_level'):
                    data_cache['water_level'] = load_single_timestep_variable(
                        self.datasets['trim'], "S1", timestep=timestep)
                
                # Apply transformation
                if var_name == 'water_depth':
                    plot_data = transform_func(data_cache['water_level'], data_cache['bed_level'])
                else:
                    plot_data = transform_func(data_cache.get('bed_level'), data_cache.get('water_level'))
                
                # Create plot
                plot_map(self.x, self.y, plot_data, var_name, self.col_indices, 
                        self.N_coords, timestep, self.config.scenario, 
                        self.config.save_dir, save_figure)
        
        self.logger.info("Spatial map processing completed.")
    
    def process_velocity_maps(self, timesteps: List[int], velocity_components: List[str] = None,
                            save_figure: bool = True) -> None:
        """Process velocity maps - modular approach"""
        
        if velocity_components is None:
            velocity_components = ['U1']  # Default to U1 component
            
        self.logger.info(f"\n=== PROCESSING VELOCITY MAPS ===")
        
        for timestep in timesteps:
            for component in velocity_components:
                self.logger.info(f"Processing {component} velocity for timestep {timestep}...")
                
                velocity = load_single_timestep_variable(
                    self.datasets['trim'], component, timestep=timestep, remove=1, layer=0)
                
                plot_velocity(self.x, self.y, velocity, component, self.col_indices,
                            self.N_coords, timestep, self.config.scenario,
                            self.config.save_dir, save_figure)
        
        self.logger.info("Velocity map processing completed.")
    
    def process_discharge_analysis(self, variable: str = 'CTR', 
                                 time_slice: Optional[Tuple[int, int]] = None,
                                 save_figure: bool = True) -> Dict:
        """Process discharge analysis with flexible station handling"""
        
        self.logger.info(f"\n=== PROCESSING DISCHARGE ANALYSIS FOR {variable} ===")
        
        # Configure station names based on model type
        if self.config.is_small_estuary:
            x_values = [2, 18, 44, 50, 60, 65, 70, 75, 80, 90, 100, 125, 150, 180, 225, 287]
            station_names = [f"({x},161)..({x},2)" for x in x_values]
        else:
            station_names = [f'river_km_{i}' for i in range(27)]
        
        # Extract data
        results, all_stations = extract_his_data(self.datasets['trih'], variable, station_names)
        
        # Log data quality
        for name in station_names:
            if name in results:
                time_data, discharge = results[name]
                self.logger.info(f"{name}: time range {time_data.min()} to {time_data.max()}, "
                               f"discharge range {np.nanmin(discharge):.3f} to {np.nanmax(discharge):.3f}")
        
        # Plot time series with optional time slicing
        if time_slice:
            time_range = (int(time_slice[0] * self.config.map_output_interval / self.config.his_output_interval),
                         int(time_slice[1] * self.config.map_output_interval / self.config.his_output_interval))
        else:
            time_range = None
            
        his_plot_discharge_timeseries(
            results, station_names, self.config.reference_date, 
            self.config.save_dir, save_figure, time_range=time_range
        )
        
        self.logger.info("Discharge analysis completed.")
        return results

def create_model_configs() -> Dict[str, ModelConfig]:
    """Factory function to create model configurations - inspired by FM's model dictionaries"""
    
    configs = {}
    
    # Small estuary model config
    configs['small_estuary'] = ModelConfig(
        name='Small Estuary Model',
        discharge=0,  # Not applicable
        scenario='control',
        model_location=Path(r"u:\PhDNaturalRhythmEstuaries\Models\000_Small_estuary_model_Anne\Small estuary model - control model"),
        trim_file=Path(r"u:\PhDNaturalRhythmEstuaries\Models\000_Small_estuary_model_Anne\Small estuary model - control model\trim-estuary.nc"),
        trih_file=Path(r"u:\PhDNaturalRhythmEstuaries\Models\000_Small_estuary_model_Anne\Small estuary model - control model\trih-estuary.nc"),
        save_dir=Path(r"u:\PhDNaturalRhythmEstuaries\Models\000_Small_estuary_model_Anne\Small estuary model - control model\postprocessing_plots"),
        map_output_interval=4500,
        his_output_interval=4500,
        total_duration=132480,
        reference_date=datetime.datetime(2015, 2, 16),
        is_small_estuary=True
    )
    
    return configs

def create_discharge_config(discharge: int = 500) -> ModelConfig:
    """Create discharge-based model configuration"""
    
    scenario = f'00_baserun{discharge}'
    model_location = Path(r"U:\PhDNaturalRhythmEstuaries\Models\06_RiverDischargeVariability_domain35x15")
    runname = f'{discharge}_VariableRiver_TideWest_test_2riverbnds'
    
    return ModelConfig(
        name=f'River Discharge Model - {discharge}',
        discharge=discharge,
        scenario=scenario,
        model_location=model_location,
        trim_file=model_location / runname / scenario / 'trim-varriver_tidewest.nc',
        trih_file=model_location / runname / scenario / 'trih-varriver_tidewest.nc',
        save_dir=model_location / runname / scenario / 'postprocessing_plots',
        map_output_interval=120,
        his_output_interval=10,
        total_duration=524160,
        reference_date=datetime.datetime(2024, 1, 1),
        is_small_estuary=False
    )

def main():
    """Main execution function with configuration flexibility"""
    
    # CONFIGURATION - Choose your model
    USE_SMALL_ESTUARY = False  # Set to True for small estuary model
    DISCHARGE = 500  # Only used for discharge model
    
    if USE_SMALL_ESTUARY:
        configs = create_model_configs()
        config = configs['small_estuary']
    else:
        config = create_discharge_config(DISCHARGE)
    
    # PROCESSING PARAMETERS
    slice_start = 0
    slice_end = 100
    amount_to_plot = 3
    timesteps = np.arange(slice_start, slice_start + amount_to_plot + 1, 1)
    
    # Initialize processor
    processor = PostProcessor(config)
    
    # RUN PROCESSING MODULES
    
    # 1. Spatial maps
    RUN_SPATIAL_PLOTS = True
    if RUN_SPATIAL_PLOTS:
        processor.process_spatial_maps(
            timesteps=timesteps,
            variables=['bed_level', 'water_level', 'water_depth'],
            save_figure=True
        )
    
    # 2. Velocity maps
    RUN_VELOCITY_PLOTS = False
    if RUN_VELOCITY_PLOTS:
        processor.process_velocity_maps(
            timesteps=timesteps,
            velocity_components=['U1'],
            save_figure=True
        )
    
    # 3. Discharge analysis
    RUN_DISCHARGE_ANALYSIS = True
    if RUN_DISCHARGE_ANALYSIS:
        results = processor.process_discharge_analysis(
            variable='CTR',
            time_slice=(slice_start, slice_end),
            save_figure=True
        )
    
    logger.info("All processing completed successfully!")

if __name__ == "__main__":
    main()