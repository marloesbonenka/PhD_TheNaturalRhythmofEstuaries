#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title: Comprehensive Estuary Discharge Analysis and Visualization
Description: This script contains functions to do a river discharge metric analysis. 

Author: Marloes Bonenkamp
Date: April 22, 2025
"""
#%%
import numpy as np
import pandas as pd

#%%
def compute_river_metrics(timeseries):
    """
    Compute river discharge metrics (variability, intermittency, flashiness).
    
    Parameters:
        timeseries (np.ndarray): Discharge time series data
        
    Returns:
        dict: Dictionary of computed metrics
    """
    q = np.array(timeseries)
    
    # Basic statistics
    mean_q = np.mean(q)
    max_q = np.max(q)
    min_q = np.min(q)
    std_q = np.std(q)
    cv = std_q / mean_q if mean_q != 0 else np.nan
    
    # Intermittency metrics
    zero_flow_intermittency = np.sum(q == 0) / len(q)
    q5 = np.percentile(q, 5)
    relative_zero_flow_intermittency = np.sum(q < q5) / len(q)
    
    # Flashiness metrics
    p90 = np.percentile(q, 90)
    p10 = np.percentile(q, 10)
    flashiness = p90 / p10 if p10 != 0 else np.nan
    
    return {
        'Mean': mean_q,
        'Max': max_q,
        'Min': min_q,
        'Std': std_q,
        'CV': cv,
        'Zero-Flow Intermittency': zero_flow_intermittency,
        'Relative Zero-Flow Intermittency (Q < Q5)': relative_zero_flow_intermittency,
        'P90': p90,
        'P10': p10,
        'Flashiness (P90/P10)': flashiness
    }

def unified_metric_calculation(series_dict, series_type="discharge"):
    """
    Unified function for computing metrics across multiple time series.
    
    Parameters:
        series_dict (dict): Dictionary of time series data {name: values}
        series_type (str): Type of series ("discharge" or "sediment")
        
    Returns:
        pd.DataFrame: DataFrame of computed metrics
    """
    results = []
    
    for name, timeseries in series_dict.items():
        metrics = compute_river_metrics(timeseries)
        metrics['Name'] = name
        metrics['Type'] = series_type
        results.append(metrics)
    
    return pd.DataFrame(results)