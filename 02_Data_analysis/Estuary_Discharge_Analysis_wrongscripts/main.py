import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import from custom modules
from coordinate_transformation import (
    transform_coordinates, 
    create_spatial_index, 
    find_nearest_point_kdtree,
    efficient_basin_area_search
)
from estuary_data import (
    matlab_datenum_to_datetime, 
    load_data_once, 
    extract_discharge_timeseries, 
    get_Qriver_timeseries,
    calculate_vectorized_daily_means
)
from metrics import (
    compute_river_metrics, 
    unified_metric_calculation
)
from visualization import (
    plot_timeseries, 
    plot_estuary_locations, 
    create_world_map_of_estuaries
)
from config import load_config, get_estuary_coordinates

def setup_logging(log_level='INFO'):
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format
    )
    return logging.getLogger(__name__)

def process_existing_data(estuary_coords, data_cache, spatial_index, output_dir, savefig, savedata):
    """
    Process and analyze existing data from the data cache.
    
    Parameters:
        estuary_coords (dict): Dictionary of estuary coordinates
        data_cache (dict): Cached data loaded from files
        spatial_index (cKDTree): Spatial index for efficient queries
        output_dir (str): Directory to save outputs
        
    Returns:
        dict: Results of the analysis
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing existing data...")
    
    # Extract key data from cache
    rm_lon = data_cache['rm_lon']
    rm_lat = data_cache['rm_lat']
    discharge_series = data_cache['discharge_series']
    datetimes = data_cache['datetimes']
    
    # Create map of global estuary distribution
    logger.info("Creating global map of estuaries...")
    mean_discharge = np.nanmean(discharge_series, axis=1)
    create_world_map_of_estuaries(
        rm_lon, rm_lat, mean_discharge, 
        os.path.join(output_dir, 'global_maps'),
        savefig
    )
    
    # Extract discharge time series using efficient spatial querying
    logger.info("Extracting discharge time series for estuaries...")
    estuary_discharge_data, estuary_rm_coords = extract_discharge_timeseries(
        estuary_coords, rm_lon, rm_lat, discharge_series
    )
    
    # Plot estuary locations
    logger.info("Plotting estuary locations...")
    plot_estuary_locations(estuary_coords, estuary_rm_coords, output_dir, savefig)
    
    # Calculate unified metrics
    logger.info("Calculating river metrics...")
    df_metrics = unified_metric_calculation(estuary_discharge_data, "sediment_flux")
    
    # Save to Excel
    metrics_path = os.path.join(output_dir, 'sediment_flux_metrics_per_estuary.xlsx')
    logger.info(f"Saving metrics to {metrics_path}")
    df_metrics.to_excel(metrics_path, index=False)
    
    return {
        'metrics': df_metrics,
        'timeseries': estuary_discharge_data,
        'coordinates': estuary_rm_coords
    }

def regenerate_timeseries(estuary_coords, data_cache, output_dir, savedata):
    """
    Regenerate time series data for estuaries.
    
    Parameters:
        estuary_coords (dict): Dictionary of estuary coordinates
        data_cache (dict): Cached data loaded from files
        output_dir (str): Directory to save outputs
        savedata (bool): Flag to save data
        
    Returns:
        dict: Results of the regenerated time series
    """
    logger = logging.getLogger(__name__)
    logger.info("Regenerating time series data...")
    
    # Extract paths from data cache
    mat_file_path = data_cache['mat_file_path']
    tif_path = data_cache['tif_path']
    basin_area_matrix = data_cache['basin_area']
    
    # Containers for results
    sed_series_dict = {}
    discharge_series_dict = {}
    
    # Process each estuary
    for estuary, (lat, lon) in estuary_coords.items():
        logger.info(f"Processing {estuary}...")
        
        try:
            # Get basin area at the location using efficient search
            row, col = transform_coordinates(lon, lat, return_indices=True)
            basinarea, adjusted_row, adjusted_col = efficient_basin_area_search(
                basin_area_matrix, row, col, search_radius=15
            )
            
            if np.isnan(basinarea):
                logger.warning(f"Could not find valid basin area for {estuary}. Skipping.")
                continue
            
            logger.info(f"Original coordinates: lat={lat}, lon={lon}")
            logger.info(f"Adjusted coordinates: lat={adjusted_row:.5f}, lon={adjusted_col:.5f}")
            logger.info(f"Basin area: {basinarea:.2f}")
            
            # Get time series data
            sed_series, discharge_timeseries, time_values = get_Qriver_timeseries(
                adjusted_row, adjusted_col, basinarea, mat_file_path, tif_path
            )
            
            # Store results
            sed_series_dict[estuary] = sed_series
            discharge_series_dict[estuary] = discharge_timeseries
            
        except Exception as e:
            logger.error(f"Error processing {estuary}: {e}", exc_info=True)
    
    # Prepare time values outside the estuary loop
    time_values_python = [matlab_datenum_to_datetime(dn) for dn in time_values]
    
    # If we have generated any data, save and plot it
    if not sed_series_dict:
        logger.warning("No data was successfully generated for any estuary.")
        return {}
    
    logger.info("Generating outputs for regenerated data...")
    
    # Create DataFrames with vectorized operations
    df_sed = pd.DataFrame.from_dict(sed_series_dict, orient='index').transpose()
    df_sed['datetime'] = time_values_python
    df_sed.set_index('datetime', inplace=True)
    
    df_discharge = pd.DataFrame.from_dict(discharge_series_dict, orient='index').transpose()
    df_discharge['datetime'] = time_values_python
    df_discharge.set_index('datetime', inplace=True)
    
    # Calculate daily means using vectorized operations
    df_sed_daily = calculate_vectorized_daily_means(sed_series_dict, time_values_python)
    df_discharge_daily = calculate_vectorized_daily_means(discharge_series_dict, time_values_python)
    
    # Save to Excel
    save_dir = os.path.join(output_dir, 'regenerated_data')
    os.makedirs(save_dir, exist_ok=True)
    df_sed.to_excel(os.path.join(save_dir, 'regenerated_sediment_flux_timeseries.xlsx'))
    df_discharge.to_excel(os.path.join(save_dir, 'regenerated_discharge_timeseries.xlsx'))
    df_sed_daily.to_excel(os.path.join(save_dir, 'regenerated_sediment_flux_daily_means.xlsx'))
    df_discharge_daily.to_excel(os.path.join(save_dir, 'regenerated_discharge_daily_means.xlsx'))
    
    # Calculate metrics
    df_metrics_sed = unified_metric_calculation(sed_series_dict, "sediment_flux")
    df_metrics_discharge = unified_metric_calculation(discharge_series_dict, "discharge")
    
    # Save metrics to Excel
    df_metrics_sed.to_excel(
        os.path.join(save_dir, 'sediment_flux_metrics_per_estuary.xlsx'),
        index=False
    )
    
    df_metrics_discharge.to_excel(
        os.path.join(save_dir, 'discharge_metrics_per_estuary.xlsx'),
        index=False
    )
    
    return {
        'sediment_metrics': df_metrics_sed,
        'discharge_metrics': df_metrics_discharge,
        'sediment_timeseries': sed_series_dict,
        'discharge_timeseries': discharge_series_dict,
        'daily_means_sediment': df_sed_daily,
        'daily_means_discharge': df_discharge_daily
    }

def main():
    """Main execution function."""
    # Load configuration
    config = load_config()
    
    # Set up logging
    logger = setup_logging(config.get('log_level', 'INFO'))
    
    logger.info("Starting Estuary Discharge Analysis...")
    
    # Get estuary coordinates
    estuary_coords = get_estuary_coordinates()
    
    savefig = config.get('savefig')
    savedata = config.get('savedata')

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load all data once
    data_cache = load_data_once(config['mat_file_path'], config['tif_path'])
    data_cache['mat_file_path'] = config['mat_file_path']  # Add paths to cache for reference
    data_cache['tif_path'] = config['tif_path']
    
    # Create spatial index for efficient point queries
    spatial_index = create_spatial_index((data_cache['rm_lon'], data_cache['rm_lat']))
    
    # Process part 1: Analysis of existing data
    logger.info("Starting Part 1: Analyzing existing data")
    part1_results = process_existing_data(
        estuary_coords,
        data_cache, 
        spatial_index,
        config['output_dir'],
        savefig,
        savedata
    )
    
    # Process part 2: Regenerate timeseries
    logger.info("Starting Part 2: Regenerating timeseries data")
    part2_results = regenerate_timeseries(
        estuary_coords,
        data_cache,
        config['output_dir'],
        savedata
    )
    
    logger.info("Analysis complete. Results saved to: %s", config['output_dir'])
    
    return part1_results, part2_results

if __name__ == "__main__":
    main()