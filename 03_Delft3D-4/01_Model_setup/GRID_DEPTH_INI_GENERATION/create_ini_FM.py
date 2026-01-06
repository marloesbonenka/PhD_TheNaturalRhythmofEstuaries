"""
Delft3D-FM Initial File Generation Script
Author: Marloes Bonenkamp
Date: Dec 4, 2025
Description: Generates initial water level data for estuary model, based on bathymetry.xyz file coordinates
"""

#%%
import pandas as pd
import numpy as np
import os
import sys

# Add the current working directory (where FUNCTIONS is located)
sys.path.append(r"C:\Users\marloesbonenka\surfdrive\Python\01_Model_setup\GRID_DEPTH_INI_GENERATION")

#%%
# --- 1. Define Constants ---

ESTUARY_WATER_DEPTH = 5.0      # Required constant water depth (H) in the Estuary Channel (in meters)
SEA_BASIN_ZETA = -3.0          # Required constant water level (ZETA) in the sea basin (in meters)

# Threshold to separate the two main water body types
ESTUARY_DEPTH_THRESHOLD = 2.0  # Bathymetry threshold in meters (B <= 2 is estuary, B > 2 is sea basin)
DRY_LAND_THRESHOLD = -3.5      # Bathymetry below this elevation is considered dry land (0m water depth)

def create_initial_water_level_xyz(input_filename="bathymetry.xyz", output_filename="initial_water_level.xyz"):
    """
    Reads a Delft3D-FM bathymetry .xyz file, calculates the initial water level (ZETA)
    based on the specified tiered masking, and writes the results to a new .xyz file.

    Initial Conditions:
    1. Estuary Channel (-3.6m <= B <= 2.0m): Constant water depth H = 5.0m (ZETA = B - 5.0m).
    2. Deep Sea Basin (B > 2.0m): Constant water level ZETA = -3.0m.
    3. Dry Land (B < -3.6m): Water depth H = 0.0m (ZETA = B).
    """

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(
            input_filename,
            header=None,
            delim_whitespace=True,
            names=['X', 'Y', 'Bathymetry_m']
        )
    except FileNotFoundError:
        print(f"Error: The input file '{input_filename}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return
        
    # --- Check for unit consistency (Degrees vs. Meters) ---
    max_x = df['X'].max()
    if max_x < 181.0:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!! WARNING: Coordinates appear to be in DEGREES (Lat/Lon) as X_max is less than 181.0. !!")
        print("!! Please ensure your X/Y units match the rest of your model's projected system (METERS). !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    # --- 2. Calculate Initial Water Level (ZETA) ---
    
    # Initialize ZETA column to NaN.
    df['Initial_Water_Level_m'] = np.nan
    
    # --- Priority 1: Estuary Channel Mask (Constant Depth) ---
    # Estuary is defined as shallow water (B < ESTUARY_DEPTH_THRESHOLD) 
    # and not dry land (B >= DRY_LAND_THRESHOLD, i.e., B >= -3.6m).
    # Requirement: Water Depth (H) = ESTUARY_WATER_DEPTH (5.0m). ZETA = B - H.
    estuary_mask = (df['Bathymetry_m'] < ESTUARY_DEPTH_THRESHOLD) & \
                   (df['Bathymetry_m'] >= DRY_LAND_THRESHOLD)
    
    df.loc[estuary_mask, 'Initial_Water_Level_m'] = df.loc[estuary_mask, 'Bathymetry_m'] - ESTUARY_WATER_DEPTH

    # --- Priority 2: Deep Sea Basin Mask (Constant Level) ---
    # Deeper than the estuary threshold (B > 2.0m).
    # Requirement: Water Level (ZETA) = SEA_BASIN_ZETA (-3.0m).
    sea_basin_mask = df['Bathymetry_m'] > ESTUARY_DEPTH_THRESHOLD
    
    # Apply the fixed ZETA requirement.
    df.loc[sea_basin_mask, 'Initial_Water_Level_m'] = SEA_BASIN_ZETA

    # --- 3. Land/Dry Areas: Explicit mask removed; now handled by the catch-all. ---
    
    # --- 4. Handle Remaining Unspecified Areas (Set to Dry) ---
    # Any remaining NaN values (points with B < -3.6m, or others not covered)
    # are set to ZETA = Bathymetry (H=0m). This ensures areas below the DRY_LAND_THRESHOLD_BETA 
    # are marked as dry land.
    remaining_mask = df['Initial_Water_Level_m'].isna()
    df.loc[remaining_mask, 'Initial_Water_Level_m'] = df.loc[remaining_mask, 'Bathymetry_m']

    # --- 5. Write Output File ---
    
    # Select only the X, Y, and the new initial water level columns
    output_df = df[['X', 'Y', 'Initial_Water_Level_m']]

    # Write to the .xyz file with high precision scientific notation
    output_df.to_csv(
        output_filename,
        sep='\t',
        header=False,
        index=False,
        float_format='%.17E'
    )

    print(f"✅ Success! Initial water level data has been saved to '{output_filename}'.")
    print(f"This file contains {len(df)} points with X, Y coordinates and the new initial water level (ZETA).")
    print("\nInitial Water Level Calculation Summary:")
    print(f" - Estuary points ({DRY_LAND_THRESHOLD}m <= B <= {ESTUARY_DEPTH_THRESHOLD}m): ZETA = B - {ESTUARY_WATER_DEPTH}m (Fixed {ESTUARY_WATER_DEPTH}m water depth).")
    print(f" - Sea Basin points (Bathymetry B > {ESTUARY_DEPTH_THRESHOLD}m): ZETA = {SEA_BASIN_ZETA}m (Fixed initial water level).")
    print(f" - Continuity Check: At B={ESTUARY_DEPTH_THRESHOLD}m, Estuary ZETA = 2.0 - 5.0 = -3.0m, which matches the Sea Basin ZETA of -3.0m.")
    print(f" - Dry Land points (B < {DRY_LAND_THRESHOLD}m): ZETA = B (0m water depth).")

#%%
create_initial_water_level_xyz()



