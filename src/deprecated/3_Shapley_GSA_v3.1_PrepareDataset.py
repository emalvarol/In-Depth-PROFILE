# 0.1 Import libraries
# Import libraries
# --- Standard Library Imports ---
import os, glob #

# --- Third-Party Libraries: Data Handling ---
import numpy as np #
import pandas as pd #
import geopandas as gpd #

# --- Third-Party Libraries: Progress and Visualization ---
from tqdm import tqdm #

# --- Project-Specific Imports ---
import User_Paths_and_Inputs #

# Basic variables
workspace = User_Paths_and_Inputs.workspace
PATHS = User_Paths_and_Inputs.PATHS
BASE_FILE_NAME = "MC_Part"
RETURN_PERIODS = User_Paths_and_Inputs.RETURN_PERIODS
RPs_unq = list(RETURN_PERIODS.keys())

# Depths (from HEC-RAS)
df_depth_samples = pd.read_pickle(PATHS['depth_samples_pkl'])
BIDs_flooded = df_depth_samples.groupby('BID')['he'].transform('max') > 0 # At least in one RP
df_depth_samples = df_depth_samples[BIDs_flooded].reset_index(drop=True) # from 5722880 to 4689280 rows

# Building
gdf_buildings = gpd.read_file(PATHS['buildings_ua_shp'])
gdf_buildings_sample = gpd.read_file(PATHS['buildings_ua_sample_shp'])
BIDs_flooded_unq = df_depth_samples['BID'].unique()
gdf_buildings = gdf_buildings[gdf_buildings['BID'].isin(BIDs_flooded_unq)].copy()
BIDs_non_characterized = gdf_buildings[gdf_buildings['BT'] == 0]['BID'].unique().tolist()
#endregion

# Helping functions
def load_mc_part(part_path):
    return pd.read_parquet(part_path)

def load_and_concatenate_mc_parts(results_dir, target_cols,  bids_to_exclude = []):
    """
    Loads specific columns from MC1_Part_*.parquet files to minimize memory footprint.
    Only SN, BID, RP, and columns starting with 'c_' are retained.
    """
    file_list = sorted(glob.glob(os.path.join(results_dir, f"{BASE_FILE_NAME}_*.parquet")))
    
    if not file_list:
        return pd.DataFrame()

    # List comprehension for faster initialization
    parts = []
    exclude_set = set(bids_to_exclude)
    for file in tqdm(file_list, desc="Merging MC Parts"):
        # Load only the subset of columns
        temp_df = pd.read_parquet(file, columns=target_cols)
        if exclude_set:
            temp_df = temp_df[~temp_df['BID'].isin(exclude_set)]
        parts.append(temp_df)
    # Single allocation of memory for the final dataframe
    df = pd.concat(parts, ignore_index=True)
    # Final cleanup of the list of parts
    del temp_df
    del parts
    return df

#region C Sensitivity Analysis
# Load a test part
temp_df = load_mc_part(os.path.join(PATHS['dataset_mc_parts'], f"{BASE_FILE_NAME}_1.parquet"))
# Get all inputs-related columns x_cols_a and those starting with n_ ; p_ ; ff_ ; and m_ as x_cols_b and outputs
x_cols_a = ['SN','BID','RP','BT','ed','he','IH','BH','GL','Cga']
x_cols_b = [c for c in temp_df.columns if c.startswith(('n_', 'p_', 'hc_', 'm_'))]
x_cols = x_cols_a + x_cols_b
y_col = 'c_bf'
# Check unique RP in temp_df
print(temp_df['RP'].unique())

# Load df
full_df = load_and_concatenate_mc_parts(PATHS['dataset_mc_parts'], x_cols + [y_col])
print(full_df['RP'].unique())
# feather
output_path = r"C:\Users\outal\Documents\MC_Parts\XGBoost\xgb_dataset.feather"
full_df = full_df.reset_index(drop=True)
full_df.to_feather(output_path, compression='zstd')

