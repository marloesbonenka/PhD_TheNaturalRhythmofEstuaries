#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration module for Estuary Discharge Analysis
"""

import os
import json
import logging

def load_config(config_path=None):
    """
    Load configuration from a JSON file or return default configuration.
    
    Parameters:
        config_path (str, optional): Path to config JSON file
        
    Returns:
        dict: Configuration dictionary
    """
    # Default configuration
    default_config = {
        "input_dir": r"C:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Data\01_Discharge_var_int_flash",
        "output_dir": r"C:\Users\marloesbonenka\OneDrive - Delft University of Technology\Documents\Data\01_Discharge_var_int_flash\01_Analysis_estuaries",
        "log_level": "INFO",
        "savefig": True,
        "savedata": True
    }
    
    # If config path is provided, try to load it
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Update default config with user settings
                default_config.update(user_config)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            logging.warning("Using default configuration.")
    
    # Add derived paths
    default_config["mat_file_path"] = os.path.join(default_config["input_dir"], "qs_timeseries_Nienhuis2020.mat")
    default_config["tif_path"] = os.path.join(default_config["input_dir"], "Nienhuis2020_Scripts", "bqart_a.tif")
    
    return default_config

def get_estuary_coordinates():
    """
    Return dictionary of estuary coordinates.
    
    Returns:
        dict: Dictionary of estuary coordinates {name: (lat, lon)}
    """
    return {
        'Eel': (40.63, -124.31),
        'Klamath': (41.54, -124.08),
        'Cacipore': (3.6, -51.2),
        'Suriname': (5.84, -55.11),
        'Demerara': (6.79, -58.18),
        'Yangon': (16.52, 96.29),
        'Sokyosen': (36.9, 126.9),
        'Wai Bian': (-8.10, 139.97),
        'Thames': (51.5, 0.6)
    }