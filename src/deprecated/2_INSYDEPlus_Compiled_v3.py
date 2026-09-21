####################### INSYDE-Content Flood Damage Model #######################
#################################################################################
#################################################################################
##### SECTION 0: LIBRARIES AND MAIN CONFIGURATION ###############################
#################################################################################
#################################################################################

# region 0 LIBRARIES AND MAIN CONFIGURATION
# region 0.1 Import libraries
# Import libraries
# --- Standard Library Imports ---
import os, time, pickle, re, ast, glob, gc

# --- Third-Party Libraries: Data Handling ---
import numpy as np
import pandas as pd
import geopandas as gpd
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.max_rows', 300)
import polars as pl

# --- Third-Party Libraries: Raster and Spatial ---
import rasterio
from rasterio import features
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon
from shapely.geometry import shape

# --- Third-Party Libraries: Statistics and Sampling ---
from scipy import stats
from scipy.stats import f, chi2, pearson3, gaussian_kde
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.interpolate import interp1d, griddata
from math import pi, isclose


# --- Third-Party Libraries: Progress and Visualization ---
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.ticker import ScalarFormatter, NullFormatter
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# --- Multiprocessing ---
import multiprocess
from functools import partial
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

# --- Project-Specific Imports ---
import User_Paths_and_Inputs

#endregion
# region 0.2 Prepare configuration and data
# Prepare configuration and data
workspace = User_Paths_and_Inputs.workspace
PATHS = User_Paths_and_Inputs.PATHS
CODES = User_Paths_and_Inputs.CODES
RETURN_PERIODS = User_Paths_and_Inputs.RETURN_PERIODS
DIST_CATALOG = User_Paths_and_Inputs.DIST_CATALOG
#endregion
# region 0.3 Set paths
# Paths
fm_path, fo_path = PATHS['fm_pkl'], PATHS['fo_pkl']
# endregion
# region 0.4 Load data
# Survey
df_survey = pd.read_excel(PATHS['survey'], sheet_name="Data")
def transform_survey_to_long(df):
    # Rename cold names with codes
    df = df.rename(columns={v: k for k, v in CODES['Survey'].items()})
    df['GRP'] = df['GRP'].map({v: k for k, v in CODES['Content'].items()}).fillna(df['GRP'])
    # Transform survey data to long format
    df_survey_long = pd.melt(
        df,
        id_vars=list(CODES['Survey'].keys()),
        value_vars=[col for col in df.columns if col.startswith('V') and col[1:].isdigit()],
        var_name='SU',
        value_name='VAL'
    )
    df_survey_long['SU'] = df_survey_long['SU'].str[1:].astype(int)
    bt_map = df_survey_long[df_survey_long['ATR'] == 'BT'].set_index('SU')['VAL']
    df_survey_long['BT'] = df_survey_long['SU'].map(bt_map).astype(int)
    cols = ['SU', 'BT'] + [c for c in df_survey_long.columns if c not in ['SU', 'BT', 'VAL']] + ['VAL']
    df_survey_long = df_survey_long[cols]
    return df_survey_long
df_survey_long = transform_survey_to_long(df_survey)

# Prices (from Web Scrapping)
df_prices = pd.read_excel(PATHS['prices_content'], sheet_name="Sheet1")

# Depths (from HEC-RAS)
PATHS['depth_samples_pkl'] = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\intermediate\Depth_Samples.pkl"
df_depth_samples = pd.read_pickle(PATHS['depth_samples_pkl'])
BIDs_flooded = df_depth_samples.groupby('BID')['he'].transform('max') > 0 # At least in one RP
df_depth_samples = df_depth_samples[BIDs_flooded].reset_index(drop=True) # from 5722880 to 4689280 rows

# Building
gdf_buildings = gpd.read_file(PATHS['buildings_ua_shp'])
gdf_buildings_sample = gpd.read_file(PATHS['buildings_ua_sample_shp'])
BIDs_flooded_unq = df_depth_samples['BID'].unique()
gdf_buildings = gdf_buildings[gdf_buildings['BID'].isin(BIDs_flooded_unq)].copy()
# endregion
# region 0.5 Set main codes
SUs_unq = df_survey_long['SU'].unique()
BTs_unq = df_survey_long['BT'].unique()
CTEs_unq = list(CODES['Content'].keys())
CTIs_unq = ['ETP','WND', 'PUM', 'PRW','ITP','SOI', 'CLE','SKT','RDR', 'PLG']
RPs_unq = list(RETURN_PERIODS.keys())
BIDs_flooded_unq_byRP = {
    rp: df_depth_samples[(df_depth_samples['RP'] == rp) & (df_depth_samples['he'] > 0)]['BID'].unique().tolist()
    for rp in RPs_unq
}
BIDs_non_characterized = gdf_buildings[gdf_buildings['BT'] == 0]['BID'].unique().tolist()
#endregion
#endregion

#################################################################################
#################################################################################
##### SECTION A: PRE-TREATMENT: FUNCTION FITTING
#################################################################################
#################################################################################


# region A PRE-TREATMENT
# region A.1 Create files to save fitted functions
# Initialize saving objects (fm - function main, fo - function observed)
fm_col_fid =        ['FID'] # Unique identifier for each fitted function
fm_col_obs =        ['ODC'] # Observed Data Code (Link to fo_df columns if obs exist)
fm_col_dataset =    ['DC'] # Dataset Column (Link between dataset columns and fm_df)
fm_col_filters =    ['BT','BID', 'RP']
fm_col_samples =    ['MAT']  # Special filter column, since in fm_df is defined but in dataset needs to be sampled first
fm_col_data = [
                    'DT', # Distribution Type (d= discrete, c=continuous)
                    'BP', # Bernoulli Probability (if needed)
                    'FN', # Function Name
                    'FP', # Function Parameters (if needed)
]
fm_columns = fm_col_fid + fm_col_obs + fm_col_dataset + fm_col_filters + fm_col_samples + fm_col_data

# Pre-save and load
if not os.path.exists(fm_path) and not os.path.exists(fo_path):
    fm_df = pd.DataFrame(columns=fm_columns)
    fm_df.to_pickle(fm_path)
    fo_data = {}
    with open(fo_path, 'wb') as f:
        pickle.dump(fo_data, f)
fm_df = pd.read_pickle(fm_path)
with open(fo_path, 'rb') as f:
    fo_data = pickle.load(f)
# endregion
# region A.2 Define funtions
## General
def prints_if(p, T = False):
    if T: print(p)

def serialize_params(params):
    return str(tuple(params)) if isinstance(params, (tuple, list)) else str(params)

## Check function
def check_dc_codes_existence(dc_codes_to_check):
    """
    Checks if a list of DC codes exists in the fitted functions pickle file.
    Prints matches, missing codes, and the filtered DataFrame for existing codes.
    """
    # Use global path from your environment
    global fm_path
    
    if not os.path.exists(fm_path):
        print(f"Error: File not found at {fm_path}")
        return

    # 1. Load the metadata DataFrame
    fm_df = pd.read_pickle(fm_path)
    
    # 2. Identify existing and missing codes
    # Ensure DC column is string for comparison
    existing_all = fm_df['DC'].astype(str).unique().tolist()
    
    found = [dc for dc in dc_codes_to_check if str(dc) in existing_all]
    missing = [dc for dc in dc_codes_to_check if str(dc) not in existing_all]

    # 3. Print Summary
    print("="*50)
    print(f"DC CODE CHECK REPORT")
    print("="*50)
    
    print(f"Total searched: {len(dc_codes_to_check)}")
    print(f"Found         : {len(found)}")
    print(f"Missing       : {len(missing)}")
    
    if found:
        print(f"\nEXISTING CODES: {found}")
    if missing:
        print(f"\nMISSING CODES : {missing}")

    # 4. Filter and Display DataFrame
    if found:
        print("\n" + "-"*50)
        print("FILTERED DATAFRAME (Matches Only):")
        print("-"*50)
        # Filter to show rows matching the found codes
        filtered_df = fm_df[fm_df['DC'].astype(str).isin([str(f) for f in found])]
        
        # Displaying main columns (adjust based on your fm_col_data global if needed)
        cols_to_show = ['DC', 'FN', 'FP', 'ODC']
        existing_cols = [c for c in cols_to_show if c in filtered_df.columns]
        
        if not filtered_df.empty:
            print(filtered_df[existing_cols].to_string(index=False))
        else:
            print("No data available for these codes.")
    
    print("="*50)
    
    return filtered_df if found else pd.DataFrame()

def print_dc_summary(df, column_name):
    """
    Prints a formatted table showing stats for a specific column,
    excluding -999 values from calculations.
    Print stats globally and broken down by each BT category (0-6).
    """
    # 1. Prepare data: identify -999 and replace with NaN for stats
    is_neg_999 = df[column_name] == -999
    valid_data = df[column_name].mask(is_neg_999, np.nan)
    
    # 2. Define the BT range
    bt_range = range(0, 7)
    
    # 3. Calculate Global values
    global_nans = df[column_name].isna().sum()
    global_neg_999 = is_neg_999.sum()
    global_total = len(df)
    
    stats_glob = {
        'mean': valid_data.mean(),
        'min':  valid_data.min(),
        'max':  valid_data.max(),
        'q05':  valid_data.quantile(0.05),
        'q95':  valid_data.quantile(0.95)
    }
    
    # 4. Calculate per BT values
    temp_df = pd.DataFrame({
        'original': df[column_name],
        'is_999': is_neg_999,
        'valid': valid_data,
        'BT': df['BT']
    })
    
    bt_stats = temp_df.groupby('BT').agg(
        nans=('original', lambda x: x.isna().sum()),
        neg_999=('is_999', 'sum'),
        total=('original', 'size'),
        avg=('valid', 'mean'),
        min_val=('valid', 'min'),
        max_val=('valid', 'max'),
        q05=('valid', lambda x: x.quantile(0.05)),
        q95=('valid', lambda x: x.quantile(0.95))
    )
    
    # 5. Construct the summary dictionary
    summary = {
        'Global': [
            global_nans,
            global_neg_999,
            global_total, 
            stats_glob['mean'], 
            stats_glob['min'], 
            stats_glob['max'], 
            stats_glob['q05'], 
            stats_glob['q95']
        ]
    }
    
    for bt in bt_range:
        if bt in bt_stats.index:
            row = bt_stats.loc[bt]
            summary[f'BT{bt}'] = [
                int(row['nans']),
                int(row['neg_999']),
                int(row['total']), 
                row['avg'], 
                row['min_val'], 
                row['max_val'], 
                row['q05'], 
                row['q95']
            ]
        else:
            summary[f'BT{bt}'] = [0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    # 6. Create DataFrame for display
    summary_df = pd.DataFrame(
        summary, 
        index=[
            f'nºnans ({column_name})',
            f'nº-999 ({column_name})',
            f'Total ({column_name})', 
            f'Average ({column_name})',
            f'Min ({column_name})',
            f'Max ({column_name})',
            f'Q05 ({column_name})',
            f'Q95 ({column_name})'
        ]
    )
    
    # 7. Format and Print
    print("\n" + "=" * 95)
    print(f"DATA SUMMARY: {column_name} (Excluding -999)")
    print("-" * 95)
    print(summary_df.to_string(index_names=False, justify='center'))
    print("=" * 95 + "\n")

## Data preparation functions
# Prepare filtered data
def expand_df_survey_long_by_weights(df_survey_long, df_filtered, weight_by):
    """
    Expands df_survey_long by replacing a subset (df_filtered) with its expanded version.
    
    Parameters:
    - df_survey_long: The master DataFrame.
    - df_filtered: The pre-filtered subset to be expanded.
    - weight_by: Tuple of (val_attr, count_attr) e.g., ('Mat', 'N')
    """
    # 0. Work on copies
    df_master = df_survey_long.copy()
    df_sub = df_filtered.copy()
    val_attr, count_attr = weight_by

    # 1. Define identifying columns
    idx_cols = [col for col in df_sub.columns if col not in ['ATR', 'VAL']]

    # 2. Extraction
    vals_df = df_sub[df_sub['ATR'] == val_attr].copy().rename(columns={'VAL': val_attr})
    counts_df = df_sub[df_sub['ATR'] == count_attr].copy().rename(columns={'VAL': count_attr})
    
    # 3. Inner Merge
    df_merged = vals_df.drop(columns='ATR').merge(
        counts_df[idx_cols + [count_attr]], 
        on=idx_cols, 
        how='inner'
    )
    
    # 4. Expansion
    df_merged = df_merged.dropna(subset=[val_attr, count_attr])
    # Repeat rows based on the count attribute
    df_expanded = df_merged.loc[df_merged.index.repeat(df_merged[count_attr].astype(int))].copy()
    
    # 5. Reconstruct structure
    df_expanded['ATR'] = val_attr
    df_expanded['VAL'] = df_expanded[val_attr]
    df_expanded = df_expanded[df_survey_long.columns]
    
    # 6. Combine/Replace
    # Remove the original subset rows from master and append the expanded ones
    # This identifies rows by index to ensure precision
    df_final = pd.concat([
        df_master[~df_master.index.isin(df_filtered.index)], 
        df_expanded
    ], ignore_index=True)
    
    return df_final

def expand_df_survey_long_by_mat(df_filtered, mat_codes_dict):
    """
    Decodes binary material combinations in df_filtered and expands the 
    DataFrame by creating a separate row for each individual material.
    Returns the expanded filtered DataFrame directly.
    """
    df_sub = df_filtered.copy()

    # 1. Decoding logic
    def decode_to_list(code):
        try:
            val_int = int(code)
            # Use bitwise AND to identify constituent keys
            return [k for k in mat_codes_dict.keys() if (val_int & k) == k]
        except (ValueError, TypeError):
            return []

    # 2. Expand the DataFrame
    # Create a column of lists containing individual material codes
    df_sub['VAL_LIST'] = df_sub['VAL'].apply(decode_to_list)
    
    # Create a new row for every element in the lists
    df_expanded = df_sub.explode('VAL_LIST')
    
    # 3. Update VAL with individual codes and clean up
    df_expanded['VAL'] = df_expanded['VAL_LIST']
    df_expanded = df_expanded.drop(columns=['VAL_LIST'])
    
    # Handle cases where VAL might be NaN after expansion (empty lists)
    df_expanded = df_expanded.dropna(subset=['VAL'])
    
    return df_expanded

def fullfill_df_survey_long_with_zeros(df_filtered, full_index):
    """
    Appends missing combinations from full_index to df_filtered with VAL=0,
    preserving SU-level metadata and setting default item placeholders.
    """
    df = df_filtered.copy()
    merge_cols = full_index.columns.tolist()
    
    # 1. Identify missing rows
    missing = full_index.merge(
        df[merge_cols + ['VAL']], 
        on=merge_cols, 
        how='left', 
        indicator=True
    ).query("_merge == 'left_only'").drop(columns=['_merge'])
    
    # 2. Assign VAL = 0 and default placeholders
    missing['VAL'] = 0
    missing['ITV'] = 0
    missing['ITM'] = '-'

    # 3. Map SU-level metadata
    metadata_cols = ['BT', 'VTP', 'DOM', 'ATR']
    available_metadata = [c for c in metadata_cols if c in df.columns]
    
    if available_metadata:
        su_map = df[['SU'] + available_metadata].drop_duplicates('SU').set_index('SU')
        
        for col in available_metadata:
            missing[col] = missing['SU'].map(su_map[col])

    # 4. Return combined dataframe
    return pd.concat([df, missing], ignore_index=True)

def sum_df_survey_long_by_su(df_filtered,
                      sum_by=['SU']):
    """
    Aggregates room-level counts to house-level (SU) counts.
    Preserves SU-level metadata like BT.

    Parameters:
        - df_filtered: The pre-filtered DataFrame to be aggregated.
        - sum_by: List of columns to group by (default is ['SU']).
    Returns:
        - A filtered DataFrame with one row per SU, aggregated VAL, and
        preserved metadata.
    """
    # 1. Define metadata to preserve (must be constant per SU).
    # And Filter metadata: remove columns used in sum_by to avoid duplicates
    available_meta = ['SU', 'BT', 'VTP', 'DOM', 'GRP', 'ITM', 'ITV', 'ATR']
    meta_to_keep = [m for m in available_meta if m in df_filtered.columns and m not in sum_by]

    # 2. Explicitly ensure VAL is numeric before summing
    df_filtered = df_filtered.copy()
    df_filtered['VAL'] = pd.to_numeric(df_filtered['VAL'], errors='coerce')

    # 3. Create the aggregation dictionary
    agg_dict = {'VAL': 'sum'}
    for col in meta_to_keep:
        agg_dict[col] = 'first'

    # 4. Group and aggregate
    df_summed = (
        df_filtered.groupby(sum_by)
        .agg(agg_dict)
        .reset_index()
    )
    
    return df_summed

# Prepared data for fitting functions (by groups and with observed values)
def create_dist_from_df(df_filtered,
                        group_cols=['BT'], 
                        val_col='VAL',
                        add_default_dist=None,
                        astype = float):
    '''
    Creates a distribution dataframe from a pre-filtered survey DataFrame.
    Supports creating multiple default (zeroed) aggregates.
    '''
    # Get raw data
    data_raw = df_filtered.copy()
    data_raw[val_col] = data_raw[val_col].astype(astype)
    
    # Create observed tuple
    obs_df = (
        data_raw.groupby(group_cols)[val_col]
        .apply(lambda x: tuple(x.values))
        .reset_index(name='OBS')
    )

    # Add Aggregate (Default zeroed groups)
    if add_default_dist is not None:
        # 1. Normalize add_default_dist to a list
        defaults_to_keep = [add_default_dist] if isinstance(add_default_dist, str) else add_default_dist
        aggregates = []

        # 2. Create aggregates for each specified column (e.g., BT=0, GRP=actual)
        for col_to_keep in defaults_to_keep:
            if col_to_keep in group_cols:
                # Aggregate by the 'keep' column
                marg = (
                    data_raw.groupby(col_to_keep)[val_col]
                    .apply(lambda x: tuple(x.values))
                    .reset_index(name='OBS')
                )
                # Set all other group columns to 0
                for col in group_cols:
                    if col != col_to_keep:
                        marg[col] = 0
                aggregates.append(marg[group_cols + ['OBS']])

        # 3. Create Global-Global aggregate (e.g., BT=0, GRP=0)
        # Only if multiple columns are provided in add_default_dist or requested
        if len(defaults_to_keep) > 1 or (len(group_cols) == 1 and defaults_to_keep[0] in group_cols):
            global_obs = tuple(data_raw[val_col].values)
            global_row = {col: [0] for col in group_cols}
            global_row['OBS'] = [global_obs]
            aggregates.append(pd.DataFrame(global_row))

        # 4. Combine and remove duplicates
        if aggregates:
            obs_df = pd.concat([obs_df] + aggregates, ignore_index=True)
            obs_df = obs_df.drop_duplicates(subset=group_cols).reset_index(drop=True)

    return obs_df

def ensure_default_dist(dist_df):
    # Specific update (OBS assummed considering all items)
    missing_grps = set(CODES["Content"].keys()) - set(dist_df['GRP'].unique())
    default_template = dist_df[dist_df['GRP'] == 0].copy()
    new_rows = []
    for grp in missing_grps:
        # Create a copy of the default rows for the missing category
        temp_df = default_template.copy()
        temp_df['GRP'] = grp
        new_rows.append(temp_df)
    dist_df = pd.concat([dist_df] + new_rows, ignore_index=True)
    return dist_df

## Fitting functions
# Bernouilli
def get_bernoulli_p(obs_tuple: tuple) -> float:
    """
    Calculates the exact ratio of non-zero observations to total observations.
    """
    return sum(1 for x in obs_tuple if x > 0) / len(obs_tuple)

# Discrete
def fit_discrete_distribution(
        obs_tuple,
        hurdle_fix=False,
        fit_only_empirical=False,
        min_sample_size = 5,
        n_mc_samples=999,
        alpha=0.05,
        accepted_max = None):
    """
    Fits discrete distributions to non-zero item counts.
    Uses a shift (data - 1) to handle the hurdle model requirement (intensity >= 1).
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.special import beta

    def serialize_params(params):
        # 1. Handle NumPy arrays (convert to list)
        if isinstance(params, np.ndarray):
            params = params.tolist()
        
        # 2. Handle lists/tuples (convert elements to native Python types)
        if isinstance(params, (list, tuple)):
            # .item() converts a numpy scalar to a native Python scalar
            clean_params = [p.item() if hasattr(p, 'item') else p for p in params]
            return str(tuple(clean_params))
        
        # 3. Handle single NumPy scalars
        if hasattr(params, 'item'):
            return str(params.item())
            
        return str(params)
    
    def to_py(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [to_py(i) for i in x]
        return x.item() if hasattr(x, 'item') else float(x)
    
    #data = data_sample
    def fit_one_distribution(data, dist_name, hurdle_fix=False):
        if dist_name == 'poisson':
            mu = np.mean(data - 1) if hurdle_fix else np.mean(data)
            log_l = np.sum(stats.poisson.logpmf(data - 1 if hurdle_fix else data, mu))
            return (to_py(mu),), 1, to_py(log_l)

        elif dist_name == 'geom':
            p_geom = 1.0 / np.mean(data) if hurdle_fix else 1.0 / (np.mean(data) + 1)
            log_l = np.sum(stats.geom.logpmf(data if hurdle_fix else data + 1, p_geom))
            return (to_py(p_geom),), 1, to_py(log_l)

        elif dist_name == 'nbinom':
            m, v = np.mean(data - 1 if hurdle_fix else data), np.var(data - 1 if hurdle_fix else data)
            if v > m and m > 0:
                p_init = m / v
                n_init = (m**2) / (v - m)
                res = minimize(lambda ps: -np.sum(stats.nbinom.logpmf(data - 1 if hurdle_fix else data, ps[0], ps[1])), 
                            x0=[n_init, p_init], bounds=[(1e-3, None), (1e-5, 0.999)])
                if res.success:
                    return tuple(to_py(x) for x in res.x), 2, to_py(-res.fun)
            return None

        elif dist_name == 'logser':
            if np.all(data >= 1):
                res = minimize(lambda p: -np.sum(stats.logser.logpmf(data, p)), x0=[0.5], bounds=[(1e-5, 0.999)])
                if res.success:
                    return (to_py(res.x[0]),), 1, to_py(-res.fun)
            return None

        elif dist_name == 'randint':
            low, high = int(np.min(data)), int(np.max(data) + 1)
            log_l = np.sum(stats.randint.logpmf(data, low, high))
            return (low, high), 2, to_py(log_l)

        return None
    
    #dist_name, params, k, log_l = candidate_results[0]
    def perform_discrete_chisquare_test(data, dist_name, params, min_sample_size = 5):
        """
        Performs a Chi-Squared Goodness-of-Fit test for discrete
        distributions.
        
        Args:
            data: Array-like, the observed counts (integers).
            dist_name: String, the name of the scipy.stats distribution (e.g., 'poisson').
            params: Tuple/List, the fitted parameters for the distribution.
            
        Returns:
            p_value: The calculated p-value.
        """
        #print(data,dist_name,params)
        n_samples = len(data)

        # 0. Check for minimum sample size (The "Rule of 13" from scipy docs)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
        # Since we de Bootstrap this is lower.
        if min_sample_size <= 5:
            # Sample too small for a rigorous Chi-Squared test
            return 0.0, False

        # 1. Get Observed Frequencies
        obs_values, obs_counts = np.unique(data, return_counts=True)
        obs_dict = dict(zip(obs_values.tolist(), obs_counts.tolist()))
        #print(obs_values,obs_dict)

        # 2. Define the range to test (from min to max observed)
        dist = getattr(stats, dist_name)
        v_min, _ = dist.support(*params)
        if dist_name in ['dlaplace']:
            v_min = int(max(0, v_min))
        v_max = int(max(data))
        
        all_values = np.arange(v_min, v_max + 1)
        #print(all_values)

        # 3. Calculate Expected Frequencies
        dist = getattr(stats, dist_name)
        # PMF for each value in the range
        expected_probs = dist.pmf(all_values, *params)
        
        # Handle the "Tails":
        # Account for the "Left Tail" (values from 0 to v_min-1)
        left_tail_prob = dist.cdf(v_min - 1, *params) if v_min > 0 else 0
        expected_probs[0] += left_tail_prob

        # Account for the "Right Tail" (values > v_max)
        right_tail_prob = 1.0 - dist.cdf(v_max, *params)
        expected_probs[-1] += right_tail_prob
        
        # Verification: np.sum(expected_probs) should be 1.0
        expected_counts = expected_probs * n_samples
        #print(expected_counts)

        # 4. Re-map observations to the full range (fill zeros for missing counts)
        observed_counts = np.array([obs_dict.get(v, 0) for v in all_values])
        #print(observed_counts)

        # 5. ROBUST BINNING (The "Rule of 5" from scipy docs)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html
        # SciPy's chisquare needs expected frequencies >= 5 to be valid
        f_obs_folded = []
        f_exp_folded = []
        
        current_obs = 0
        current_exp = 0
        
        for o, e in zip(observed_counts, expected_counts):
            current_obs += o
            current_exp += e
            if current_exp >= 5:
                f_obs_folded.append(current_obs)
                f_exp_folded.append(current_exp)
                current_obs = 0
                current_exp = 0
                
        # If the last bin is < 5, fold it into the previous bin
        if current_exp > 0:
            if len(f_obs_folded) > 0:
                f_obs_folded[-1] += current_obs
                f_exp_folded[-1] += current_exp
            else:
                # Entire dataset doesn't meet the "5" threshold
                return 0.0, False
        #print(f_obs_folded,f_exp_folded)

        # 6. Calculate Degrees of Freedom
        # df = (number of bins) - 1 - (number of estimated parameters)
        n_params = len(params)
        ddof = len(f_obs_folded) - 1 - n_params
        #print(n_params,ddof)

        if ddof <= 0:
            # Not enough bins to perform a valid test
            return 0.0, False

        # 7. Perform the Test
        chi_stat, p_value = stats.chisquare(
            to_py(f_obs_folded),
            to_py(f_exp_folded),
            ddof=n_params
        )
        #print(to_py(chi_stat),to_py(p_value))

        return to_py(chi_stat), to_py(p_value)

    def sample_from_distribution(dist_name, params, size):
        """
        Draw a bootstrap sample from the fitted model.
        """
        #print("Sample dist: S1")
        if dist_name == "poisson":
            mu = params[0]
            x = stats.poisson.rvs(mu, size=size)
            return x + 1 if hurdle_fix else x

        if dist_name == "geom":
            p = params[0]
            return stats.geom.rvs(p, size=size)

        if dist_name == "nbinom":
            n, p = params
            x = stats.nbinom.rvs(n, p, size=size)
            return x + 1 if hurdle_fix else x

        if dist_name == "logser":
            p = params[0]
            return stats.logser.rvs(p, size=size)

        if dist_name == "randint":
            low, high = params
            return stats.randint.rvs(low, high, size=size,)

        raise ValueError(f"Unsupported distribution: {dist_name}")

    #dist_name, n_mc_samples = dist_names[3], 100
    def perform_discrete_bootstrap_chisquare_test(data, dist_name, n_mc_samples=999, hurdle_fix=False, min_sample_size = 5):
        """
        Parametric bootstrap version of the chi-square GOF test.

        Returns:
            p_value
        """
        print(f"Bootstrap for  {dist_name}")
        #print("Bootstrap Chi: S1")
        data = np.asarray(data, dtype=int)

        #print("Bootstrap Chi: S2")
        fit_res = fit_one_distribution(data, dist_name, hurdle_fix)
        if fit_res is not None:
            params, k, log_l = fit_res
        else:
            print("Fix not possible")
            return None

        #print("Bootstrap Chi: S3")
        chi_stat_obs, p_value_obs = perform_discrete_chisquare_test(data, dist_name, params, min_sample_size=min_sample_size)

        #print("Bootstrap Chi: S4")
        boot_stats = []
        size = len(data)
        for _ in range(n_mc_samples):
            #print("Sample")
            data_sample = sample_from_distribution(dist_name, params, size)
            #print("Fit")
            fit_res_sample = fit_one_distribution(data_sample, dist_name)
            if fit_res_sample is not None:
                params_sample, k_sample, log_l_sample = fit_res
                #print("Chi")
                chi_stat_sample, p_value_sample = perform_discrete_chisquare_test(
                    data_sample,
                    dist_name,
                    params_sample,
                    min_sample_size=min_sample_size
                )
                #print("Append")
                boot_stats.append(chi_stat_sample)
            else:
                continue
        #
        if len(boot_stats) < 20:
            return 0.0, p_value_obs

        #print("Bootstrap Chi: S3")
        boot_stats = np.asarray(boot_stats, dtype=float)
        p_value_bs = (np.sum(boot_stats >= chi_stat_obs) + 1.0) / (len(boot_stats) + 1.0)
        return to_py(p_value_bs), p_value_obs

    def get_aic(log_l, k):
        return 2 * k - 2 * log_l
    
    # --- 0: Data Preparation ---
    data = np.array(obs_tuple, dtype=int)
    if hurdle_fix:
        data = data[data > 0]
    
    n_samples = len(data)
    
    # --- 1: Forced Empirical ---
    if fit_only_empirical:
        if n_samples == 0:
            return pd.Series(["Constant", serialize_params((0,))])
        return pd.Series(["Empirical", None])
    
    # --- 2: Constant Check ---
    if n_samples == 0:
        return pd.Series(["Constant", serialize_params((0,))])
    unique_vals = np.unique(data)
    if len(unique_vals) == 1:
        return pd.Series(["Constant", serialize_params((int(unique_vals[0]),))])
    
    # --- 3: Distribution Fitting & Bootstrap Evaluation ---
    final_candidates_chi = []
    dist_names = ['poisson', 'geom', 'nbinom', 'logser', 'randint',]
    for d_name in dist_names:
        # 3.1 Fit the distribution
        fit_res = fit_one_distribution(data, d_name, hurdle_fix)
        if fit_res is None:
            continue
        params, k, log_l = fit_res
        
        # 3.2 Check for extreme values (Q999) if accepted_max is provided
        if accepted_max is not None:
            dist = getattr(stats, d_name)
            # Calculate the 99.9th percentile
            try:
                q999 = dist.ppf(0.999, *params)
                
                # Adjust q999 if hurdle_fix was used during fitting
                if hurdle_fix:
                    if d_name in ['poisson', 'nbinom']:
                        q999 += 1
                
                if q999 > accepted_max:
                    # If PPF fails to converge, the tail is likely too long
                    # We assume it exceeds accepted_max
                    continue
            except RuntimeError:
                continue

        # 3.3. Perform Bootstrap Chi-Square Test directly
        bootstrap_res = perform_discrete_bootstrap_chisquare_test(
            data, 
            d_name, 
            n_mc_samples=n_mc_samples, 
            hurdle_fix=hurdle_fix,
            min_sample_size=min_sample_size
        )
        
        if bootstrap_res is not None:
            p_val_bs, p_val_obs = bootstrap_res
            
            # Evaluate using the bootstrap p-value against alpha
            if p_val_bs > alpha:
                aic = get_aic(log_l, k)
                final_candidates_chi.append({
                    'FN': d_name, 
                    'FP': params, 
                    'P_BS': p_val_bs, 
                    'P_OBS': p_val_obs, 
                    'AIC': aic
                })

    # --- 4: Selection ---
    if final_candidates_chi:
        # Select best AIC among accepted distributions
        best = min(final_candidates_chi, key=lambda x: x['AIC'])
        return pd.Series([best['FN'], serialize_params(best['FP'])])

    # --- 5: Empirical Frequency (Fallback) ---
    return pd.Series(["Empirical", None])

def define_discrete_distribution(function, params):
    """
    Returns the distribution name and serialized parameters.
    Supports:
    - 'randint': (min_val, max_val)
    - 'triang': (low, mode, high) - Note: scipy 'triang' is continuous; 
                use for custom sampling logic or as metadata.
    - 'binom': (n, p)         
    """
    if function == 'randint':
        min_val, max_val = params
        # scipy.stats.randint params: low (inclusive), high (exclusive)
        new_params = (int(min_val), int(max_val) + 1)
        dist_info = pd.Series(['randint', str(new_params)])
    elif function == 'triang':
        # params: (low, mode, high)
        # Note: Discrete triangular is often handled via custom logic or 
        # as a continuous distribution rounded to integers.
        low, mode, high = params
        new_params = (float(low), float(mode), float(high))
        dist_info = pd.Series(['triang', str(new_params)])
    elif function == 'binom':
        n, p = params
        # scipy.stats.binom params: n, p
        new_params = (int(n), float(p))
        dist_info = pd.Series(['binom', str(new_params)])
    dist_df = pd.DataFrame([dist_info.values], columns=['FN', 'FP'])
    return dist_df

# Continuous
def fit_continuous_distribution(
        obs_tuple,
        hurdle_fix=False,
        dist_names_to_fit=['uniform'],
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=None,
        accepted_min=None,
        avoid_bs_size=500):
    """
    Fits continuous distributions using the Cramer-von Mises
    (CvM) statistic and bootstrap.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats
    import warnings

    def serialize_params(params):
        if isinstance(params, np.ndarray):
            params = params.tolist()
        if isinstance(params, (list, tuple)):
            clean_params = [p.item() if hasattr(p, 'item') else p for p in params]
            return str(tuple(clean_params))
        if hasattr(params, 'item'):
            return str(params.item())
        return str(params)

    def to_py(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [to_py(i) for i in x]
        return x.item() if hasattr(x, 'item') else float(x)
    
    def get_aic(log_l, k):
        return 2 * k - 2 * log_l

    # --- 0: Data Preparation ---
    data = np.array(obs_tuple, dtype=float)
    if hurdle_fix:
        data = data[data > 0]
    n_samples = len(data)

    # --- 1: Forced KDE ---
    if fit_only_KDE:
        if n_samples == 0:
            return pd.Series(["Constant", serialize_params((0.0,))])
        return pd.Series(["KDE", None])

    # --- 2: Constant Check ---
    if n_samples == 0:
        return pd.Series(["Constant", serialize_params((0.0,))])
    unique_vals = np.unique(data)
    if len(unique_vals) == 1:
        return pd.Series(["Constant", serialize_params((float(unique_vals[0]),))])
    if n_samples < min_sample_size:
        return pd.Series(["KDE", None])

    # --- 3: Distribution Fitting & Bootstrap Evaluation ---
    candidates = []
    for d_name in dist_names_to_fit:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') # Silence the noise for this specific fit
            try:
                if not hasattr(stats, d_name):
                    continue
                dist = getattr(stats, d_name)

                # HURDLE check: Force floc=0 for positive-only distributions
                # List of distributions that should start at 0
                positive_only = ['lognorm', 'gamma', 'weibull_min', 'expon', 'genpareto', 'pareto']
                fit_kwargs = {'floc': 0} if d_name in positive_only else {}
                
                # 3.1 Fit the distribution
                try:
                    params = dist.fit(data, **fit_kwargs)
                    k = len(params)
                    log_l = np.sum(dist.logpdf(data, *params))
                    aic = get_aic(log_l, k)
                    res_1s = stats.cramervonmises(data, d_name, args=params)
                    p_val_1s = float(res_1s.pvalue)
                except Exception:
                    continue

                # 3.2 Check for extreme values (Q05 and/or Q95) if accepted_max and/or accepted_min is provided
                q05 = float(dist.ppf(0.05, *params))
                q95 = float(dist.ppf(0.95, *params))
                if accepted_min is not None:
                    if q05 < accepted_min:
                        continue
                if accepted_max is not None:
                    if q95 > accepted_max:
                        continue
                
                # 3.3. Perform Bootstrap CvM Test directly (if small size)
                if n_samples > avoid_bs_size:
                    candidates.append({
                        'FN': d_name,
                        'FP': to_py(params),
                        'P_1S': to_py(p_val_1s),
                        'P_BS': to_py(p_val_1s),
                        'AIC': to_py(aic),
                        'Q05': to_py(q05),
                        'Q95': to_py(q95),
                        'bs': False
                    })
                    continue
                try:
                    # Map parameters to names: (shape1, shape2, ..., loc, scale)
                    # Scipy distribution params are always ordered: shapes..., loc, scale
                    param_names = (dist.shapes.split(', ') if dist.shapes else []) + ['loc', 'scale']
                    fit_params_dict = dict(zip(param_names, params))

                    gof_res = stats.goodness_of_fit(
                        dist, 
                        data, 
                        fit_params=fit_params_dict, 
                        statistic='cvm', 
                        n_mc_samples=n_mc_samples
                    )
                    p_val_bs = float(gof_res.pvalue)
                except Exception:
                    continue
                
                candidates.append({
                    'FN': d_name,
                    'FP': to_py(params),
                    'P_1S': to_py(p_val_1s),
                    'P_BS': to_py(p_val_bs),
                    'AIC': to_py(aic),
                    'Q05': to_py(q05),
                    'Q95': to_py(q95),
                    'bs': True
                })
            except Exception:
                continue
    # --- Print Results Table ---
    #header = f"{'Dist':<15} | {'P_1S':<8} | {'P_BS':<8} | {'AIC':<10} | {'Q05':<8} | {'Q95':<8}"
    #sep = "-" * len(header)
    #print(header); print(sep)
    #for c in candidates:
    #    print(f"{c['FN']:<15} | {c['P_1S']:<8.4f} | {c['P_BS']:<8.4f} | {c['AIC']:<10.2f} | {c['Q05']:<8.2f} | {c['Q95']:<8.2f}")
    #print(sep + "\n")

    # --- 4: Selection ---
    valid_candidates = [
        c for c in candidates 
        if float(c['P_BS']) > float(alpha)
    ]
    if valid_candidates:
        best = min(valid_candidates, key=lambda x: x['AIC'])
        return pd.Series([best['FN'], serialize_params(best['FP'])])

    # --- 5: KDE (Fallback) ---
    return pd.Series(["KDE", None])

def define_continuous_distribution(function, params, dist_df=None):
    """
    Returns the distribution name and serialized parameters for a continuous distribution.
    If dist_df is provided, it applies the distribution to every row of the DataFrame.
    
    Parameters:
    - function: str, distribution name (triang, truncnorm, lognorm, norm)
    - params: tuple/list of raw parameters
    - dist_df: Optional DataFrame to append the results to as columns 'FN' and 'FP'
    """
    
    def serialize_params(p):
        return str(tuple(p)) if isinstance(p, (tuple, list, np.ndarray)) else str(p)

    def to_py(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return [to_py(i) for i in x]
        return x.item() if hasattr(x, 'item') else float(x)
    
    # 1. Calculate Distribution Parameters
    if function == 'triang':
        min_val, mode_val, max_val = params
        scale = max_val - min_val
        c = (mode_val - min_val) / scale if scale != 0 else 0
        final_params = (c, min_val, scale)
        fn, fp = 'triang', serialize_params(to_py(final_params))

    elif function == 'truncnorm':
        t_min, t_mode, t_max = params
        def objective(s):
            if s <= 0: return 1e10
            loc = t_mode
            a_gen, b_gen = (t_min - loc) / s, (t_max - loc) / s
            current_coverage = stats.truncnorm.cdf(t_max, a_gen, b_gen, loc=loc, scale=s) - \
                               stats.truncnorm.cdf(t_min, a_gen, b_gen, loc=loc, scale=s)
            return (current_coverage - 0.99)**2
        res = minimize(objective, x0=[(t_max - t_min) / 4], method='Nelder-Mead')
        s_opt, loc_opt = res.x[0], t_mode
        dist_params = ((t_min - loc_opt) / s_opt, (t_max - loc_opt) / s_opt, loc_opt, s_opt)
        fn, fp = 'truncnorm', serialize_params(to_py(dist_params))

    elif function == 'lognorm':
        t_min, t_mode, t_max = params
        def objective(vars):
            s, scale = vars
            if s <= 0 or scale <= 0: return 1e10
            current_mode = scale * np.exp(-(s**2))
            current_coverage = stats.lognorm.cdf(t_max, s, scale=scale) - \
                               stats.lognorm.cdf(t_min, s, scale=scale)
            return (current_mode - t_mode)**2 + (current_coverage - 0.99)**2
        res = minimize(objective, x0=[0.5, t_mode], method='Nelder-Mead')
        s_opt, scale_opt = res.x
        dist_params = (s_opt, 0, scale_opt)
        fn, fp = 'lognorm', serialize_params(to_py(dist_params))

    elif function == 'norm':
        fn, fp = 'norm', serialize_params(to_py(params))
    
    else:
        raise ValueError(f"Function {function} not supported.")

    # 2. Return Logic
    if dist_df is not None:
        # Assign values to all rows of the provided DataFrame
        result_df = dist_df.copy()
        result_df['FN'] = fn
        result_df['FP'] = fp
        return result_df
    else:
        # Return a single-row DataFrame as per original behavior
        return pd.DataFrame([[fn, fp]], columns=['FN', 'FP'])

## Table and plots
def save_distribution_occurrence(dist_df, sheet_name, group_cols=['BT']):
    """
    Groups distribution fits by variables, calculates occurrences, and saves to Excel.
    Replaces the sheet if it already exists.
    """

    # 1. Define the path
    output_dir = PATHS['fitted_distributions']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, "distributions_occurrence.xlsx")

    # 2. Filter and calculate occurrences
    filtered_df = dist_df.copy()

    # Dynamic Alphabetical Order for FN
    unique_fn = sorted([x for x in filtered_df['FN'].unique() if pd.notnull(x)])
    filtered_df['FN'] = pd.Categorical(filtered_df['FN'], categories=unique_fn, ordered=True)
    
    # Group and Unstack
    occurrence_df = (
        filtered_df.groupby(group_cols + ['FN'], observed=True)
        .size()
        .unstack(level='FN', fill_value=0)
    )
    
    # 3. Create the "TOTAL" row
    total_values = occurrence_df.sum()
    total_df = pd.DataFrame([total_values])
    
    # Handle MultiIndex for TOTAL row
    if len(group_cols) > 1:
        total_df.index = pd.MultiIndex.from_tuples([('TOTAL',) + (None,) * (len(group_cols)-1)], names=group_cols)
    else:
        total_df.index = pd.Index(['TOTAL'], name=group_cols[0])
    
    # Combine TOTAL and Data
    final_df = pd.concat([total_df, occurrence_df])
    
    # 4. Final Formatting
    # Replace 0 with None so they appear blank in Excel
    final_df = final_df.replace(0, np.nan)
    
    # Reset index and ensure names are correct (removes 'level_0' issues)
    final_output = final_df.reset_index()

    # 5. Save to Excel
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            final_output.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        final_output.to_excel(file_path, sheet_name=sheet_name, index=False)
        
    print(f"Success: Occurrences for {group_cols} saved to sheet '{sheet_name}'.")

def save_discrete_distribution_plots(
        dist_df, 
        name_code, 
        group_cols=None, 
        obs_col='OBS',       # Set default to OBS
        bernoulli_col='BP', 
        name_col='FN', 
        params_col='FP'
    ):
    """
    Creates a mosaic of plots for discrete distributions.
    Ensures 'Observed' data is plotted even when no theoretical fit is available.
    """
    
    def parse_input(val):
        if isinstance(val, str):
            try: return ast.literal_eval(val)
            except: return val
        return val

    rows, cols = 3, 5
    plots_per_page = rows * cols
    n_total = len(dist_df)
    n_pages = int(np.ceil(n_total / plots_per_page))
    
    # Ensure PATHS exists or use a default
    output_dir = "Outputs/PT_Fitted_Distributions" 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for page in range(n_pages):
        fig, axes = plt.subplots(rows, cols, figsize=(22, 12), constrained_layout=True)
        axes_flat = axes.flatten()
        
        start_idx = page * plots_per_page
        end_idx = min(start_idx + plots_per_page, n_total)
        chunk = dist_df.iloc[start_idx:end_idx]
        
        for i, (idx, row) in enumerate(chunk.iterrows()):
            ax = axes_flat[i]
            
            # --- 1. Prepare Data & Metadata ---
            obs_data = np.array([])
            if obs_col in row and pd.notnull(row[obs_col]):
                obs_data = np.array(parse_input(row[obs_col]))
            
            group_title_str = " | ".join([f"{col}: {row[col]}" for col in group_cols]) if group_cols else f"Row: {idx}"
            fn_name = row[name_col]
            params = parse_input(row[params_col]) if params_col in row else None
            
            x_min, x_max = None, None
            y_max_local = 0.1 

            # --- 2. Theoretical Logic (Only if not Empirical/Constant) ---
            pmf_values = None
            x_theoretical = None
            
            if pd.notnull(fn_name) and fn_name not in ['Constant', 'Empirical']:
                try:
                    dist = getattr(stats, fn_name)
                    p_tuple = (params,) if np.isscalar(params) else tuple(params)
                    
                    x_min = int(dist.ppf(0.001, *p_tuple))
                    x_max = int(dist.ppf(0.999, *p_tuple))
                    
                    x_theoretical = np.arange(x_min, x_max + 1)
                    pmf_values = dist.pmf(x_theoretical, *p_tuple)
                    y_max_local = np.max(pmf_values)
                except Exception:
                    ax.text(0.5, 0.5, "Fit Error", transform=ax.transAxes, ha='center', color='red')

            # --- 3. Observed Logic (Critical for "Empirical" plots) ---
            obs_vals, obs_freq = np.array([]), np.array([])
            if obs_data.size > 0:
                unique_vals, counts = np.unique(obs_data, return_counts=True)
                obs_vals = unique_vals
                obs_freq = counts / len(obs_data)
                
                y_max_local = max(y_max_local, np.max(obs_freq))
                
                # Robust limit setting: use observed if theoretical fails or is Empirical
                obs_min, obs_max = int(np.min(obs_data)), int(np.max(obs_data))
                if x_min is None:
                    x_min, x_max = obs_min, obs_max
                else:
                    x_min, x_max = min(x_min, obs_min), max(x_max, obs_max)

            # Fallback for empty/constant data
            if x_min is None: x_min, x_max = 0, 1
            if x_min == x_max: x_max += 1 
            
            # --- 4. Plotting ---
            artists_added = False
            
            if obs_data.size > 0:
                ax.bar(obs_vals, obs_freq, alpha=0.3, color='gray', label='Observed', width=0.8)
                artists_added = True
            
            if pmf_values is not None:
                ax.step(x_theoretical, pmf_values, where='mid', color='red', lw=1.5, label=f'Fit: {fn_name}')
                ax.plot(x_theoretical, pmf_values, 'ro', markersize=2, alpha=0.5)
                artists_added = True

            # --- 5. Formatting ---
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
            ax.set_ylim(0, y_max_local * 1.15)
            
            bp_val = row.get(bernoulli_col, None)
            bp_text = f" | BP: {bp_val:.3f}" if pd.notnull(bp_val) and bp_val != "" else ""
            ax.set_title(f"{group_title_str}\nDist: {fn_name}{bp_text}", fontsize=9, fontweight='bold')
            
            if artists_added:
                ax.legend(fontsize=7, loc='upper right')

        # Hide unused axes
        for j in range(len(chunk), plots_per_page):
            axes_flat[j].axis('off')

        file_name = f"{name_code}_{page + 1}.png"
        plt.savefig(os.path.join(output_dir, file_name), dpi=200)
        plt.close(fig)
        
    print(f"Saved {n_pages} local-limit discrete mosaic(s) to {output_dir}")

def save_continuous_distribution_plots(
        dist_df, 
        name_code, 
        group_cols=None, 
        obs_col=None,
        bernoulli_col='BP',
        name_col='FN', 
        params_col='FP',
        hurdle_fix=False
        ):
    """
    Creates a mosaic of plots with LOCAL axis limits prioritized by the 
    Theoretical Distribution (FN/FP) and falling back to OBS data.
    """
    
    def parse_input(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return val
        return val

    # --- 0. Filter Constants ---
    # We remove 'Constant' rows so they don't take up space in the mosaic
    plot_df = dist_df[dist_df[name_col] != 'Constant'].copy()
    
    if plot_df.empty:
        print(f"No non-constant distributions to plot for {name_code}.")
        return
    
    has_fit_cols = name_col in plot_df.columns and params_col in plot_df.columns
    rows, cols = 3, 5
    plots_per_page = rows * cols
    n_total = len(plot_df)
    n_pages = int(np.ceil(n_total / plots_per_page))
    
    output_dir = PATHS['fitted_distributions']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for page in range(n_pages):
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10), constrained_layout=True)
        axes_flat = axes.flatten()
        
        start_idx = page * plots_per_page
        end_idx = min(start_idx + plots_per_page, n_total)
        chunk = plot_df.iloc[start_idx:end_idx]
        
        for i, (idx, row) in enumerate(chunk.iterrows()):
            
            ax = axes_flat[i]
            
            # --- 1. Data Preparation ---
            obs_data = np.array([])
            if obs_col and obs_col in row and pd.notnull(row[obs_col]):
                obs_data = np.array(parse_input(row[obs_col]))
                # Apply Hurdle Fix: Remove zeros
                if hurdle_fix:
                    obs_data = obs_data[obs_data != 0]
            
            # Handle Bernoulli/Hurdle Probability in Title
            fn_name = row[name_col] if has_fit_cols else None
            params = parse_input(row[params_col]) if has_fit_cols else None

            bp_val = row.get(bernoulli_col, None)
            bp_text = f" | BP: {bp_val:.3f}" if pd.notnull(bp_val) and bp_val != "" else ""
            
            if group_cols:
                group_title_str = " | ".join([f"{col}: {row[col]}" for col in group_cols])
            else:
                group_title_str = f"Row: {idx}"
            
            ax.set_title(f"{group_title_str}\nDist: {fn_name}{bp_text}", fontsize=9, fontweight='bold')

            # --- 1. Determine X-Axis Limits ---
            x_min, x_max = None, None

            # Priority 1: Theoretical Fit (PPF)
            if pd.notnull(fn_name) and fn_name not in ['KDE'] and params is not None:
                try:
                    dist = getattr(stats, fn_name)
                    if isinstance(params, (float, int)): params = (params,)
                    x_min = dist.ppf(0.05, *params)
                    x_max = dist.ppf(0.95, *params)
                except Exception:
                    pass 
            elif fn_name == 'KDE' and obs_data.size > 1 and np.var(obs_data) > 0:
                try:
                    kde = stats.stats.gaussian_kde(obs_data)
                    # For KDE, we calculate percentiles directly from the observed data
                    # as a proxy for the distribution's PPF
                    x_min = np.percentile(obs_data, 5)
                    x_max = np.percentile(obs_data, 95)
                except:
                    pass

            # Priority 2: Observed Data
            if (x_min is None or x_max is None) and obs_data.size > 0:
                x_min = np.percentile(obs_data, 5)
                x_max = np.percentile(obs_data, 95)

            # Final Fallback
            if x_min is None or np.isnan(x_min): x_min, x_max = 0, 1
            
            # --- 2. Plotting ---
            # Histogram (Only if obs_data exists)
            if obs_data.size > 0:
                ax.hist(obs_data, bins=20, density=True, alpha=0.3, color='gray', label='Observed')

            # PDF
            if pd.notnull(fn_name) and fn_name not in ['KDE'] and params is not None:
                try:
                    x_pdf = np.linspace(x_min, x_max, 250)
                    y_pdf = dist.pdf(x_pdf, *params)
                    ax.plot(x_pdf, y_pdf, color='red', lw=2, label=f'Fit: {fn_name}')
                except Exception:
                    ax.text(0.5, 0.5, "PDF Error", transform=ax.transAxes, ha='center', color='red')
            
            # KDE Fallback (Only if obs_data exists)
            elif fn_name == 'KDE' and obs_data.size > 1 and np.var(obs_data) > 0:
                try:
                    kde = stats.stats.gaussian_kde(obs_data)
                    x_kde = np.linspace(x_min, x_max, 250)
                    ax.plot(x_kde, kde(x_kde), color='blue', lw=1.5, ls='--', label='KDE')
                except:
                    pass

            # --- 3. Formatting ---
            ax.set_xlim(x_min, x_max)
            ax.legend(fontsize=7, loc='upper right')

        # Hide unused axes
        for j in range(len(chunk), plots_per_page):
            axes_flat[j].axis('off')

        plt.savefig(os.path.join(output_dir, f"{name_code}_{page + 1}.png"), dpi=200)
        plt.close(fig)
        
    print(f"Saved {n_pages} distribution plot(s) to {output_dir}")

# Functions updating
def bulk_update_functions_files(dist_df, mapping_config, prints=True):
    prints_if("=" * 40, prints)
    start_total = time.time()
    
    # Declare globals
    global fm_path, fo_path
    global fm_col_obs, fm_col_dataset, fm_col_filters, fm_col_samples, fm_col_data

    # 1. Load data files
    fm_df = pd.read_pickle(fm_path)
    with open(fo_path, 'rb') as f:
        fo_data = pickle.load(f)

    # Internal process function with synchronized identity logic
    def process_entry(fm_df_mem, fo_data_mem, update_dict, obs_values=None):
        # Identity columns: combine dataset refs, filters, and samples
        identity_cols = fm_col_dataset + fm_col_filters + fm_col_samples
        match_row_data = {col: update_dict.get(col, 0) for col in identity_cols}
        
        existing_odc = np.nan
        mask = pd.Series(False, index=fm_df_mem.index)
        
        if not fm_df_mem.empty:
            mask = fm_df_mem[identity_cols].apply(
                lambda row: all(
                    (pd.isna(row[col]) and pd.isna(match_row_data[col])) or 
                    (str(row[col]) == str(match_row_data[col])) 
                    for col in identity_cols
                ), axis=1
            )
            if mask.any():
                existing_odc = fm_df_mem.loc[mask, 'ODC'].values[0]

        odc_code = existing_odc
        if obs_values is not None:
            if pd.isna(odc_code):
                if not fo_data_mem:
                    next_idx = 1
                else:
                    indices = [int(re.search(r'OBS_(\d+)', str(k)).group(1)) 
                            for k in fo_data_mem.keys() if re.search(r'OBS_(\d+)', str(k))]
                    next_idx = max(indices) + 1 if indices else 1
                odc_code = f"OBS_{next_idx}"
            fo_data_mem[odc_code] = np.array(obs_values)

        new_row_data = {'ODC': odc_code}
        new_row_data.update(match_row_data) # Identity cols (DC, BT, BID, RP, MAT)
        for col in fm_col_data:
            new_row_data[col] = update_dict.get(col, np.nan)
        
        if mask.any():
            fm_df_mem = fm_df_mem[~mask]
        
        fm_df_mem = pd.concat([fm_df_mem, pd.DataFrame([new_row_data])], ignore_index=True)
        return fm_df_mem, fo_data_mem

    # 2. Process Rows
    total_rows = len(dist_df)
    interval = max(1, total_rows // 10)
    prints_if(f"Processing:{total_rows} rows...", prints)

    for i, (idx, row) in enumerate(dist_df.iterrows()):
        if i % interval == 0 and i > 0:
            prints_if(f"Progress: {(i / total_rows) * 100:.0f}% ({i}/{total_rows} rows)", prints)
        
        row_update_dict = {}

        # Resolve basic mapping
        for fm_col, source in mapping_config.items():
            if isinstance(source, tuple) and source[0] == 'col':
                # Identify if filter or data for default value
                is_filter = fm_col in (fm_col_filters + fm_col_dataset + fm_col_samples)
                default = 0 if is_filter else np.nan
                row_update_dict[fm_col] = row.get(source[1], default)
            else:
                row_update_dict[fm_col] = source

        # --- FIX: Apply DCV prefix logic (DCV_DC) ---
        dcv_val = row_update_dict.get('DCV')
        dc_val = row_update_dict.get('DC')
        if dcv_val and dcv_val != 0 and dcv_val != '0':
            row_update_dict['DC'] = f"{dcv_val}_{dc_val}"

        if row_update_dict.get('FN') == 'KDE':
            row_update_dict['FP'] = np.nan

        fm_df, fo_data = process_entry(fm_df, fo_data, row_update_dict, obs_values=row.get('OBS'))

    # 3. Save
    fm_df.to_pickle(fm_path)
    with open(fo_path, 'wb') as f:
        pickle.dump(fo_data, f)

    prints_if(f"BULK COMPLETE: {len(dist_df)} rows in {time.time() - start_total:.4f} s", prints)
#endregion
# region A.3 Prepare distributions for building
RUN_BLOCK = False
if RUN_BLOCK:
    check_dc_codes_existence(['BT', 'NF', 'BF', 'HU', 'IH', 'GL', 'BH'])
    #---------------------------------------
    # BT - Building Type
    df_filtered = gdf_buildings[
        (gdf_buildings['BT'].notna()) & 
        (gdf_buildings['BT'] != 0)
    ].copy()
    general_obs = tuple(df_filtered['BT'].astype(int).values)
    dist_df = pd.DataFrame({
        'BID': [0], 
        'OBS': [general_obs]
    })
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical = True,
    )
    save_distribution_occurrence(dist_df, "BT", group_cols=['BID'])
    save_discrete_distribution_plots(dist_df, "BT", group_cols=['BID'])
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'BT',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
        prints=True
    )

    #---------------------------------------
    # NF - Number of Floors
    df_filtered = gdf_buildings[
        (gdf_buildings['BT'].notna()) & 
        (gdf_buildings['BT'] != 0)
    ].copy()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=["BT"],
        val_col="NF",
        add_default_dist=["BT"],
        astype = int
    )    
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical = True,
    )
    save_distribution_occurrence(dist_df, "NF", group_cols=['BT'])
    save_discrete_distribution_plots(dist_df, "NF", group_cols=['BT'])
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'NF',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
        prints=True
    )

    #---------------------------------------
    # BF - Basement Floors
    df_filtered = gdf_buildings[
        (gdf_buildings['BT'].notna()) & 
        (gdf_buildings['BT'] != 0)
    ].copy()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=["BT"],
        val_col="BF",
        add_default_dist=["BT"],
        astype = int
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical = True,
    )
    save_distribution_occurrence(dist_df, "BF", group_cols=['BT'])
    save_discrete_distribution_plots(dist_df, "BF", group_cols=['BT'])
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'BF',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
        prints=True
    )

    #---------------------------------------
    # HU - Number of Households (shift +1 in sampling)
    dist_df = define_discrete_distribution(
        function='binom',
        params=(3, 0.45)
    )
    save_discrete_distribution_plots(dist_df, "HU", group_cols=None, obs_col=None)
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'HU',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    # IH - Inter-floor Height
    dist_df = define_continuous_distribution(
        function='triang',
        params=(2.40, 2.50, 3.00)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="IH",
        group_cols=None,
        obs_col=None,
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'IH',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    #---------------------------------------
    ### GL - Ground Floor Level (continuous float)
    dist_df = create_dist_from_df(
        df_survey_long[
            (df_survey_long['GRP'] == 'Difcota') & 
            (~df_survey_long['ROM'].isin(['Sotano'])) & 
            (df_survey_long['VAL'].notna())
        ],
        add_default_dist=["BT"],
        astype = float
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_continuous_distribution,
        dist_names_to_fit=DIST_CATALOG['C_SS_i'],
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=5,
        accepted_min=-5
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="GL",
        group_cols=['BT'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'GL',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    ### BH - Basement Height (continuous float, positive)
    dist_df = define_continuous_distribution(
        function='triang',
        params=(2.40, 2.50, 3.00)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="BH",
        group_cols=None,
        obs_col=None,
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'BH',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    # GP - Ground Floor Perimeter (Cga)
    df_filtered = df_survey_long[
            (df_survey_long['ATR'] == 'Cga') & 
            (df_survey_long['VAL'].notna())
        ]
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = float
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_continuous_distribution,
        dist_names_to_fit=DIST_CATALOG['C_SS_i_vff3'],
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=10,
        accepted_min=1
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="Cga",
        group_cols=['BT'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'Cga',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
#endregion 
# region A.4 Prepare distributions for content
RUN_BLOCK = False
if RUN_BLOCK:
    check_dc_codes_existence([f"n_{code}" for code in CTEs_unq])
    check_dc_codes_existence([f"p_{code}" for code in CTEs_unq])
    check_dc_codes_existence([f"ff_{code}" for code in CTEs_unq])

    #---------------------------------------
    ### n Items (discrete int, start at 0, Hurdle model)
    # Prepare filtered data
    df_filtered = sum_df_survey_long_by_su(
        df_survey_long[
            (df_survey_long['DOM'] == 'Contenido') & 
            (df_survey_long['ATR'] == 'N') & 
            (df_survey_long['VAL'].notna())
        ],
        sum_by=['SU','GRP']
    )
    df_filtered = fullfill_df_survey_long_with_zeros(
        df_filtered,
        (
            df_filtered[['SU']]
            .drop_duplicates()
            .merge(
                pd.DataFrame({'GRP': list(CODES['Content'].keys())}), 
                how='cross'
            )
        )
    )
    # Prepare data to fit distributions
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT', 'GRP'],
        add_default_dist='GRP',
        astype=int
    )
    # Fit distributions
    dist_df['BP'] = dist_df['OBS'].apply(get_bernoulli_p)
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        hurdle_fix=True,
        min_sample_size = 5,
        n_mc_samples=5000,
        alpha=0.025,
        accepted_max = 500
    )
    # Save ocurrence
    save_distribution_occurrence(dist_df, "Content_N", group_cols=['BT', 'GRP'])
    # Save plots
    save_discrete_distribution_plots(dist_df, "Content_N", group_cols=['BT', 'GRP'])
    # Update functions file
    bulk_update_functions_files(
        dist_df,
        {'BT': ('col', 'BT'),
            'DC': ('col', 'GRP'),
            'DCV': 'n',
            'DT': 'd',
            'BP': ('col', 'BP'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
        prints=True
    )

    #---------------------------------------
    ### Price (continuous float, start at >0)
    dist_df = create_dist_from_df(
        df_prices,
        group_cols=['Group_Code'],
        val_col='Scraped_Price',
        add_default_dist='Group_Code',
        astype=float
    )
    dist_df_a = dist_df.loc[dist_df['Group_Code'] != 'VEH'].copy()
    dist_df_b = dist_df.loc[dist_df['Group_Code'] == 'VEH'].copy()
    dist_df_a[['FN', 'FP']] = dist_df_a['OBS'].parallel_apply(
        fit_continuous_distribution,
        dist_names_to_fit=DIST_CATALOG['C_BS_p'],
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=2000,
        accepted_min=0,
        avoid_bs_size=300
    )
    dist_df_b[['FN', 'FP']] = dist_df_b['OBS'].apply(
        fit_continuous_distribution,
        dist_names_to_fit=DIST_CATALOG['C_BS_p'],
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=70000,
        accepted_min=0,
        avoid_bs_size=300
    )
    dist_df = pd.concat([dist_df_a, dist_df_b], axis=0).sort_index()
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="Content_P",
        group_cols=['Group_Code'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP',
        new_global_y_max = 0.015,
        new_global_x_max = 500,
        new_global_x_min = 0,
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': ('col', 'Group_Code'),
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        }
    )

    #---------------------------------------
    ### Fragility Functions, Altitude (continuous float)
    df_filtered = df_survey_long[
            (df_survey_long['DOM'] == 'Contenido') & 
            (df_survey_long['ATR'] == 'Alt') & 
            (df_survey_long['VAL'].notna())
        ]
    df_filtered['VAL'].min(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT', 'GRP'],
        val_col='VAL',
        add_default_dist=['BT', 'GRP'],
        astype=float
    )
    dist_df = dist_df[dist_df['OBS'].apply(len) >= 5].reset_index(drop=True).copy() # Since we cannot have constants for the FF
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_continuous_distribution,
        dist_names_to_fit=DIST_CATALOG['C_SS_i_vff3'],
        fit_only_KDE=False,
        min_sample_size=5000,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=3,
        accepted_min=0,
        avoid_bs_size=300
    )
    dist_df = ensure_default_dist(dist_df)
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="Content_FF",
        group_cols=['BT', 'GRP'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP',
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': ('col', 'GRP'),
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
        prints=True
    )
#endregion 
# region A.5 Prepare distributions for continent
RUN_BLOCK = False
if RUN_BLOCK:
    check_dc_codes_existence(['p_PUM', 'p_DHU', 'p_CLE'])
    #---------------------------------------
    ## Pumping:
    #p_PUM
    '''
    Sources:
        https://www.zaask.es/cuanto-cuesta/fontaneros
        https://shop.aesrewinds.co.uk/blog/sump-pumps/guide-to-flood-
    Unit: €/m3
    Assumptions to tranform from €/h to €/m3:
        -Residential pump flow rate: 12.3 m3/h
    Data:
        Value       €/h     €/m3
        -Minimun    15      1.21
        -Mode       35      2.84
        -Maximun    60      4.87
    INSYDE: 2.5 €/m3
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(1.21, 2.84, 4.87)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_PUM",
        name_col='FN',
        params_col='FP',
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'PUM',
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    ## Dehumidification:
    #p_DHU
    '''
    Source: https://www.habitissimo.es/presupuestos/humedades
    Unit: €/m2
    Data:
        Value   €/m2
        -Min    18
        -Mode   nan
        -Max    35
    INSYDE: 15 €/m2 aprox.
    Assumed mean=1/3=23.66
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(18, 23.66, 35)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_DHU",
        name_col='FN',
        params_col='FP',
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'DHU',
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    ## Cleaning:
    #p_CLE
    '''
    Source: https://www.habitissimo.es/presupuestos/limpieza-casa
    Unit: €/m2
    Assumptions to tranform from €/h to €/m2:
        -Productivity factor: 5.5 m2/h
        -Average GA in case study: 68.8 m2
        -Total hour needed in average: 12.5 h
    Data:
        Value       €/h     €/m2
        -Minimun    8       1.45
        -Mode       15      2.73
        -Maximun    20      3.64
    INSYDE: 2.4 €/m2
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(1.45, 2.73, 3.64)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_CLE",
        name_col='FN',
        params_col='FP',
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'CLE',
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    check_dc_codes_existence([
        'p_SOI', 'm_SOI', 'n_FRI', 'ff_FRI', 'm_EXF', 'p_ETP', 'p_PRW',
        'ff_PRW', 'm_PRW', 'pr_PRW', 'ff_SOI','p_SKT','n_SKT','m_SKT','ff_SKT'])
    #---------------------------------------
    ## Soil:
    ## R1/R2 - Screed/pavement removal and
    #  N2 - Screed replacement
    # p_SOI
    '''
    Sources:
        https://www.habitissimo.es/presupuestos/cambiar-suelo
    Unit: €/m2
    Data:
        Value       €/m2
        -Minimun    15
        -Mode       40
        -Maximun    80
    MAT (mat code): mean:
        -Wood (512):27.5
        -Epoxi (4096):35
        -Cement (8):50
        -Ceramic (16):40
    INSYDE: 11.8 (R1) + 18.7 (N2) = 30.5€/m2
    '''
    soi_mat = {
        0: 40,  # None
        512: 27.5,  # Wood
        4096: 35,   # Epoxi
        8: 50,      # Cement
        16: 40      # Ceramic
    }
    dist_list = []
    for mat_code, mean_val in soi_mat.items():
        # Generate single-row distribution
        temp_df = define_continuous_distribution(
            function='triang',
            params=(15, mean_val, 80)
        )
        # Add the identifier column
        temp_df['MAT'] = mat_code
        dist_list.append(temp_df)
    dist_df = pd.concat(dist_list, ignore_index=True)
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_SOI",
        group_cols=['MAT'],
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'SOI',
            'DCV': 'p',
            'DT': 'c',
            'MAT': ('col', 'MAT'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        }
    )

    # m_SOI (material)
    df_filtered = expand_df_survey_long_by_mat(
        df_survey_long[
            (df_survey_long['GRP'] == 'Suelo') & 
            (df_survey_long['ATR'] == 'Mat') & 
            (df_survey_long['VAL'].notna())
        ],
        CODES['MAT']
    )
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype=int
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical=True,
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="m_SOI",
        group_cols=['BT'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {'BT': ('col', 'BT'),
            'DC': 'SOI',
            'DCV': 'm',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    # ff_SOI (same as INSYDE)
    dist_df = define_continuous_distribution(
        function='truncnorm',
        params=(0.2, 0.4, 0.6)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ff_SOI",
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'SOI',
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    ## Partition walls:
    ## R4 - Partition walls removal
    # pr_PRW
    '''
    Sources:
        https://www.habitissimo.es/presupuestos/albaniles
        https://www.scribd.com/document/955581366/productivity-table
    Unit: €/m2
    Assumptions to tranform from €/h to €/m2:
        -General demolition masonry yield: 0.8 h/m2
    Data:
        Value       €/h     €/m2
        -Minimun    18      14.4
        -Mode       25      20.0
        -Maximun    35      28.0
    INSYDE: 14.9 €/m2
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(14.4, 20.0, 28.0)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="pr_PRW",
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'PRW',
            'DCV': 'pr',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    # m_PRW
    df_survey_long_expanded = expand_df_survey_long_by_weights(
        df_survey_long, 
        df_survey_long[
            (df_survey_long['GRP'] == 'Paredes Interiores') & 
            (df_survey_long['ITM'] == 'Pared')
        ], 
        ('Mat', 'N')
    )
    df_survey_long_expanded2 = expand_df_survey_long_by_mat(
        df_survey_long_expanded[
            (df_survey_long_expanded['GRP'] == 'Paredes Interiores') & 
            (df_survey_long_expanded['ITM'] == 'Pared') & 
            (df_survey_long_expanded['ATR'] == 'Mat') & 
            (df_survey_long_expanded['VAL'].notna())
        ],
        CODES['MAT']
    )
    dist_df = create_dist_from_df(
        df_survey_long_expanded2,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical=True,
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="m_SOI",
        group_cols=['BT'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {'BT': ('col', 'BT'),
            'DC': 'PRW',
            'DCV': 'm',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # ff_PRW (same as INSYDE)
    df_filtered = df_survey_long[
            (df_survey_long['GRP'] == 'Paredes Interiores') & 
            (df_survey_long['ITM'] == 'Pared') & 
            (df_survey_long['ATR'] == 'Alt') & 
            (df_survey_long['VAL'].notna())
        ]
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = define_continuous_distribution(
        function='truncnorm',
        params=(1.5, 1.7, 1.9)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ff_PRW",
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'PRW',
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    ## N1 - Partition walls replacement
    # p_PRW
    '''
    Sources:
        https://www.habitissimo.es/presupuestos/construir-muro-hormigon
    Unit: €/m2
    Data:
        Value       €/m2
        -Minimun    50
        -Mode       90
        -Maximun    150
    INSYDE: 67.2 €/m2
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(50, 90, 150)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_PRW",
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'PRW',
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    #---------------------------------------
    ## Painting:
    #p_ETP
    '''
    Source: https://www.habitissimo.es/presupuestos/pintores
    Unit: €/m2
    Data:
        Value       €/m2
        -Minimun    4
        -Mode       12
        -Maximun    50
    INSYDE: 10.3 €/m2 (external) 8.10 (internal)
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(4, 12, 50)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_ETP",
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'ETP',
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    ## F3 - External painting
    # m_EXF
    df_survey_long_expanded = expand_df_survey_long_by_mat(
        df_survey_long[
            (df_survey_long['GRP'] == 'Revestimiento Fachada') & 
            (df_survey_long['ATR'] == 'Mat') & 
            (df_survey_long['VAL'].notna())
        ],
        CODES['MAT']
    )
    dist_df = create_dist_from_df(
        df_survey_long_expanded,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical=True,
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="m_EXF",
        group_cols=['BT'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {'BT': ('col', 'BT'),
            'DC': 'EXF',
            'DCV': 'm',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    #---------------------------------------
    ## F4 - Internal painting
    # ff_FRI (altitude friso)
    df_filtered = df_survey_long[
            (df_survey_long['GRP'] == 'Paredes Interiores') & 
            (df_survey_long['ITM'] == 'Friso') & 
            (df_survey_long['ATR'] == 'Alt') & 
            (df_survey_long['VAL'].notna())
        ]
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = float
    ) # all < 5
    dist_df = define_continuous_distribution(
        function='triang',
        params=(df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()),
        dist_df=dist_df
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ff_FRI",
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {'BT': ('col', 'BT'),
            'DC': 'FRI',
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # n_FRI
    df_filtered = sum_df_survey_long_by_su(
        df_survey_long[
            (df_survey_long['GRP'] == 'Paredes Interiores') & 
            (df_survey_long['ITM'] == 'Friso') & 
            (df_survey_long['ATR'] == 'N')
        ],
        sum_by=['SU']
    )
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df['BP'] = dist_df['OBS'].apply(get_bernoulli_p)
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_discrete_distribution,
        hurdle_fix=True,
        min_sample_size = 5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max = 500
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="n_FRI",
        group_cols=['BT'],
        obs_col='OBS',
        bernoulli_col='BP',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {'BT': ('col', 'BT'),
            'DC': 'FRI',
            'DCV': 'n',
            'DT': 'd',
            'BP': ('col', 'BP'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    ## Skirting (SKT):
    ## R3/F6 - Skirting removal and replacement
    # p_SKT
    '''
    Source: https://www.habitissimo.es/presupuestos/poner-rodapie
    Unit: €/ml
    Data:
        Value   €/ml
        -Min    2.79
        -Mode   5
        -Max    10
    MAT (mat code): mean:
        -Wood (512): 7
        -PVC (4096): 2.8
        -Aluminium (1): 8.4
        -Stone (2048): 4
    INSYDE: NaN
    '''
    soi_mat = {
        0: 5,  # None
        512: 7,  # Wood
        4096: 2.8,   # PVC
        1: 8.4,      # Aluminium
        2048: 4     # Stone
    }
    dist_list = []
    for mat_code, mean_val in soi_mat.items():
        # Generate single-row distribution
        temp_df = define_continuous_distribution(
            function='triang',
            params=(2.79, mean_val, 10)
        )
        # Add the identifier column
        temp_df['MAT'] = mat_code
        dist_list.append(temp_df)
    dist_df = pd.concat(dist_list, ignore_index=True)
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_SKT",
        group_cols=['MAT'],
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'SKT',
            'DCV': 'p',
            'DT': 'c',
            'MAT': ('col', 'MAT'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        }
    )
    
    # n_SKT (number)
    df_filtered = sum_df_survey_long_by_su(
        df_survey_long[
            (df_survey_long['GRP'] == 'Paredes Interiores') & 
            (df_survey_long['ITM'] == 'Rodapies') & 
            (df_survey_long['ATR'] == 'N')
        ],
        sum_by=['SU']
    )
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df['BP'] = dist_df['OBS'].apply(get_bernoulli_p)
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_discrete_distribution,
        hurdle_fix=True,
        min_sample_size = 5,
        n_mc_samples=5000,
        alpha=0.025,
        accepted_max = 500
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="n_SKT",
        group_cols=['BT'],
        obs_col='OBS',
        bernoulli_col='BP',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'SKT',
            'DCV': 'n',
            'DT': 'd',
            'BP': ('col', 'BP'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # m_SKT (material)
    df_survey_long_expanded = expand_df_survey_long_by_weights(
        df_survey_long, 
        df_survey_long[
            (df_survey_long['GRP'] == 'Paredes Interiores') & 
            (df_survey_long['ITM'] == 'Rodapies')
        ], 
        ('Mat', 'N')
    )
    df_survey_long_expanded2 = expand_df_survey_long_by_mat(
        df_survey_long_expanded[
            (df_survey_long_expanded['GRP'] == 'Paredes Interiores') & 
            (df_survey_long_expanded['ITM'] == 'Rodapies') & 
            (df_survey_long_expanded['ATR'] == 'Mat') & 
            (df_survey_long_expanded['VAL'].notna())
        ],
        CODES['MAT']
    )
    dist_df = create_dist_from_df(
        df_survey_long_expanded2,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical=True,
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="m_SKT",
        group_cols=['BT'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {'BT': ('col', 'BT'),
            'DC': 'SKT',
            'DCV': 'm',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    # ff_SKT
    dist_df = define_continuous_distribution(
        function='triang',
        params=(0, 0.1, 0.2)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ff_SKT",
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'SKT',
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    check_dc_codes_existence(['ff_PLG', 'n_PLG', 'p_PLG', 'ff_WND', 'n_WND', 'p_WND',
                              'ff_RDR', 'm_RDR','n_RDR', 'p_RDR',])
    #---------------------------------------
    ## Doors (RDR - Regular Door):
    ## R8/W1 - Doors removal and replacement
    # p_RDR
    '''
    Source: https://www.habitissimo.es/presupuestos/instalar-puertas
    Unit: €/ud
    Data:
        Value   €/ud
        -Min    119.9
        -Mode   400
        -Max    1200
    MAT (mat code): mean:
        -Wood (512): 235
        -PVC (4096): 860
        -Aluminium (1): 120
        -Metal (4): 600
    INSYDE: 195 €/m2
    '''
    soi_mat = {
        0: 400,  # None
        512: 235,  # Wood
        4096: 860,   # PVC
        1: 120,      # Aluminium
        4: 600     # Stone
    }
    dist_list = []
    for mat_code, mean_val in soi_mat.items():
        # Generate single-row distribution
        temp_df = define_continuous_distribution(
            function='triang',
            params=(119.9, mean_val, 1200)
        )
        # Add the identifier column
        temp_df['MAT'] = mat_code
        dist_list.append(temp_df)
    dist_df = pd.concat(dist_list, ignore_index=True)
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_RDR",
        group_cols=['MAT'],
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'RDR',
            'DCV': 'p',
            'DT': 'c',
            'MAT': ('col', 'MAT'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        }
    )
    
    # n_RDR (number)
    df_filtered = sum_df_survey_long_by_su(
        df_survey_long[
            (df_survey_long['GRP'] == 'Aperturas') & 
            (df_survey_long['ITM'] == 'Regular doors') & 
            (df_survey_long['ATR'] == 'N') & 
            (df_survey_long['VAL'].notna())
        ]
    )
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df['BP'] = dist_df['OBS'].apply(get_bernoulli_p)
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_discrete_distribution,
        hurdle_fix=True,
        min_sample_size = 5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max = 500
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="n_RDR",
        group_cols=['BT'],
        obs_col='OBS',
        bernoulli_col='BP',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'RDR',
            'DCV': 'n',
            'DT': 'd',
            'BP': ('col', 'BP'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # m_RDR (material)
    df_survey_long_expanded = expand_df_survey_long_by_weights(
        df_survey_long, 
        df_survey_long[
            (df_survey_long['GRP'] == 'Aperturas') & 
            (df_survey_long['ITM'] == 'Regular doors')
        ], 
        ('Mat', 'N')
    )
    df_survey_long_expanded2 = expand_df_survey_long_by_mat(
        df_survey_long_expanded[
            (df_survey_long_expanded['GRP'] == 'Aperturas') & 
            (df_survey_long_expanded['ITM'] == 'Regular doors') & 
            (df_survey_long_expanded['ATR'] == 'Mat') & 
            (df_survey_long_expanded['VAL'].notna())
        ],
        CODES['MAT']
    )
    dist_df = create_dist_from_df(
        df_survey_long_expanded2,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical=True,
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="m_RDR",
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'RDR',
            'DCV': 'm',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    # ff_RDR (as INSYDE)
    df_filtered = df_survey_long[
            (df_survey_long['GRP'] == 'Aperturas') & 
            (df_survey_long['ITM'] == 'Regular doors') & 
            (df_survey_long['ATR'] == 'Alt') & 
            (df_survey_long['VAL'].notna())
        ]
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = define_continuous_distribution(
        function='triang',
        params=(0.4, 0.6, 0.8)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ff_RDR",
        name_col='FN',
        params_col='FP'
    ) 
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'RDR',
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    ## Windows (WND):
    ## R9/W2 - Windows removal and replacement
    # p_WND
    '''
    Source: https://www.habitissimo.es/presupuestos/ventanas-pvc
    Unit: €/ud
    Data:
        Value   €/ud
        -Min    250
        -Mode   350
        -Max    600
    INSYDE: 268.5 €/m2
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(250, 350, 600)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_WND",
        name_col='FN',
        params_col='FP'
    ) 
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'WND',
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # n_WND (number)
    df_filtered = sum_df_survey_long_by_su(
        df_survey_long[
            (df_survey_long['GRP'] == 'Aperturas') & 
            (df_survey_long['ITM'] == 'Windows') & 
            (df_survey_long['ATR'] == 'N') & 
            (df_survey_long['VAL'].notna())
        ]
    )
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df['BP'] = dist_df['OBS'].apply(get_bernoulli_p)
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_discrete_distribution,
        hurdle_fix=False,
        min_sample_size = 5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max = 500
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="n_WND",
        group_cols=['BT'],
        obs_col='OBS',
        bernoulli_col='BP',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'WND',
            'DCV': 'n',
            'DT': 'd',
            'BP': ('col', 'BP'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # ff_WND (as INSYDE)
    df_filtered = df_survey_long[
            (df_survey_long['GRP'] == 'Aperturas') & 
            (df_survey_long['ITM'] == 'Windows') & 
            (df_survey_long['ATR'] == 'Alt') & 
            (df_survey_long['VAL'].notna())
        ]
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = define_continuous_distribution(
        function='triang',
        params=(1.3, 1.5, 1.7)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ff_WND",
        name_col='FN',
        params_col='FP'
    ) 
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'WND',
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    #---------------------------------------
    ## Plugs (PLG)
    # p_PLG
    '''
    Source: https://www.habitissimo.es/presupuestos/instalar-puertas
    Unit: €/ud
    Data:
        Value   €/ud
        -Min    35
        -Mode   40
        -Max    90
    INSYDE: 195 €/m2
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(35, 40, 90)
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="p_PLG",
        name_col='FN',
        params_col='FP'
    ) 
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'PLG',
            'DCV': 'p',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # n_PLG (number)
    df_filtered = sum_df_survey_long_by_su(
        df_survey_long[
            (df_survey_long['GRP'] == 'Electricidad') & 
            (df_survey_long['ITM'] == 'Enchufes') & 
            (df_survey_long['ATR'] == 'N') & 
            (df_survey_long['VAL'].notna())
        ]
    )
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df['BP'] = dist_df['OBS'].apply(get_bernoulli_p)
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_discrete_distribution,
        hurdle_fix=True,
        min_sample_size = 5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max = 500
    )
    save_discrete_distribution_plots(
        dist_df=dist_df,
        name_code="n_PLG",
        group_cols=['BT'],
        obs_col='OBS',
        bernoulli_col='BP',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'PLG',
            'DCV': 'n',
            'DT': 'd',
            'BP': ('col', 'BP'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    # ff_PLG
    df_filtered = df_survey_long[
            (df_survey_long['GRP'] == 'Electricidad') & 
            (df_survey_long['ITM'] == 'Enchufes') & 
            (df_survey_long['ATR'] == 'Alt') & 
            (df_survey_long['VAL'].notna())
        ]
    df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = float
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_continuous_distribution,
        dist_names_to_fit=dist_names_to_fit_C_MI_B,
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=3,
        accepted_min=0,
        avoid_bs_size=300
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ff_PLG",
        group_cols=['BT'],
        obs_col='OBS',
        name_col='FN',
        params_col='FP'
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'PLG',
            'DCV': 'ff',
            'DT': 'c',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )

    #---------------------------------------
    ## Others:
    ## P4 - Electrical system replacement
    # p_ELS
    '''
    Sources:
        https://www.habitissimo.es/presupuestos/cambiar-instalacion-electrica
    Unit: €/m2
    Data:
        Value       €/m2
        -Minimun    15
        -Mode       25
        -Maximun    55
    INSYDE: 42.9 (N2) €/m2
    '''
    dist_df = define_continuous_distribution(
        function='triang',
        params=(15, 25, 55)
    )
    bulk_update_functions_files(
        dist_df,
        {
            'DC': 'ELS',
            'DCV': 'p',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
    
    # t_ELS
    dist_df = create_dist_from_df(
        df_survey_long[
            (df_survey_long['GRP'] == 'Electricidad') & 
            (df_survey_long['ITM'] == 'Inst. Eléctrica') & 
            (df_survey_long['ATR'] == 'Tipo') & 
            (df_survey_long['VAL'].notna())
        ],
        group_cols=['BT'],
        val_col='VAL',
        add_default_dist=['BT'],
        astype = int
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        fit_discrete_distribution,
        fit_only_empirical=True,
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BT': ('col', 'BT'),
            'DC': 'ELS',
            'DCV': 't',
            'DT': 'd',
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
    )
#endregion 
# region A.6 Prepare distributions for event features
RUN_BLOCK = False
if RUN_BLOCK:
    check_dc_codes_existence(['he'])
    #---------------------------------------
    ### he - Depth (continuous float, start at 0, Hurdle model)
    bid_max_values = df_depth_samples.groupby('BID')['he'].max()
    bids_to_keep = bid_max_values[bid_max_values > 0].index
    df_filtered = df_depth_samples[df_depth_samples['BID'].isin(bids_to_keep)].copy()
    dist_df = create_dist_from_df(
        df_filtered,
        group_cols=['BID', 'RP'],
        val_col='he',
        astype=float
    )
    df_depth_samples['he'].min(), df_depth_samples['he'].mean(), df_depth_samples['he'].max()
    dist_df['LP'] = dist_df['OBS'].apply(lambda obs: sum(1 for x in obs if x != 0))
    dist_df['LP'].min(), dist_df['LP'].mean(), dist_df['LP'].max()
    dist_df['BP'] = dist_df['OBS'].parallel_apply(get_bernoulli_p)
    dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
        fit_continuous_distribution,
        dist_names_to_fit=DIST_CATALOG['C_BS_p_vB'],
        hurdle_fix=True,
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=999,
        alpha=0.05,
        accepted_max=20,
        accepted_min=0,
        avoid_bs_size=30
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="he",
        group_cols=['BID', 'RP'],
        obs_col='OBS',
        bernoulli_col='BP',
        name_col='FN',
        params_col='FP',
        hurdle_fix=True
    )
    bulk_update_functions_files(
        dist_df,
        {
            'BID': ('col', 'BID'),
            'RP': ('col', 'RP'),
            'DC': 'he',
            'DT': 'c',
            'BP': ('col', 'BP'),
            'FN': ('col', 'FN'),
            'FP': ('col', 'FP')
        },
        prints=True
    )
#endregion 
# region A.7 Prepare distributions for DEM error
RUN_BLOCK = False
if RUN_BLOCK:
    #---------------------------------------
    ### DEM (continuous float)
    dem_errors = []
    with rasterio.open(PATHS["dem_error_tif"]) as src:
        for geom in gdf_buildings.geometry:
            try:
                # Mask raster with polygon geometry
                out_image, out_transform = mask(src, [geom], crop=True)
                # Filter out NoData values (using src.nodata if defined, else assume np.nan)
                data = out_image[0]
                valid_data = data[data != src.nodata] if src.nodata is not None else data[~np.isnan(data)]
                
                # Calculate mean if valid pixels exist
                avg_se = np.mean(valid_data) if valid_data.size > 0 else 0
                dem_errors.append(avg_se)
            except (ValueError, Exception):
                dem_errors.append(0)
    dem_error_df = pd.DataFrame({
        'BID': gdf_buildings['BID'],
        'SE': dem_errors
    })
    dist_df = create_dist_from_df(
        dem_error_df, 
        group_cols=['BID'], 
        val_col='SE'
    )
    dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
        lambda obs_tuple: define_continuous_distribution('norm', (0, float(obs_tuple[0]))).iloc[0]
    )
    save_continuous_distribution_plots(
        dist_df=dist_df,
        name_code="ed",
        group_cols=['BID'],
        name_col='FN',
        params_col='FP',
    )
    bulk_update_functions_files(
            dist_df,
            {
                'BID': ('col', 'BID'),
                'DC': 'ed',
                'DT': 'c',
                'FN': ('col', 'FN'),
                'FP': ('col', 'FP')
            },
            prints=True
        )
#endregion
# region A.8 Final analysis
# Open pkl
fm_df = pd.read_pickle(fm_path)
# Sort
sort_hierarchy = ['DC', 'BT', 'BID', 'RP', 'MAT', 'DT']
fm_df = fm_df.sort_values(by=sort_hierarchy)
# Save general ocurrence
save_distribution_occurrence(
    dist_df=fm_df, 
    sheet_name="Global", 
    group_cols=sort_hierarchy
)
#endregion
#endregion 

#################################################################################
#################################################################################
##### SECTION B: MODEL: ECONOMIC MONTE-CARLO
#################################################################################
#################################################################################

# region B MODEL
# region B.1 Create dataset (define know data)
'''
Conventions:
 nan: Reserved to unknow data. They will be sampled during monte carlo
-999: Reserved to cell which are not applicable, for example, for a BID which do not has 4th floor
'''
def create_dataset():
    col_name_dic = {}
    # B.1.1 Codes
    dataset = pd.concat([
        pd.DataFrame({'BID': bids, 'RP': rp}) 
        for rp, bids in BIDs_flooded_unq_byRP.items()
    ], ignore_index=True).copy() # BID - Building ID / RP - Return Period
    dataset['BT'] = dataset['BID'].map(gdf_buildings.set_index('BID')['BT']) # BT - Dwelling Type
    dataset.loc[dataset['BT'] == 0, 'BT'] = np.nan
    
    dataset['DID'] = range(len(dataset)) # DID - Dataset (row) ID
    dataset['FID'] = np.nan # FID - Function (row) ID
    
    codes_order = ['DID','FID','BT','BID','RP'] # Reorder
    dataset = dataset[codes_order]
    col_name_dic["codes"] = codes_order
    
    # B.1.2 Structure
    dataset['NF'] = dataset['BID'].map(gdf_buildings.set_index('BID')['NF']) # NF - Number of Floors
    dataset.loc[dataset['BT'].isna(), 'NF'] = np.nan
    max_nf = int(dataset['NF'].max())

    dataset['BF'] = dataset['BID'].map(gdf_buildings.set_index('BID')['BF']) # BF - Basement Floors
    dataset.loc[dataset['BT'].isna(), 'BF'] = np.nan
    max_bf = int(dataset['BF'].max())

    dataset['HU'] = dataset['BID'].map(gdf_buildings.set_index('BID')['HU']) # HU - Number of Households
    dataset.loc[dataset['BT'].isna(), 'HU'] = np.nan
    dataset.loc[dataset['BT'].isin([5, 6]), 'HU'] = np.nan

    dataset['IH'] = np.nan # IH - Inter-floor Height
    dataset['BH'] = np.nan # BH - Basement Height

    dataset['GL'] = np.nan # GL - Ground Floor Level
    dataset['BL'] = np.nan # BL - Basement Level

    structure_order = ['HU','NF','BF','IH','BH','GL','BL']
    dataset = dataset[codes_order + structure_order]
    col_name_dic["structure"] = structure_order

    # B.1.3 Areas    
    dataset['GA'] = dataset['BID'].map(gdf_buildings.set_index('BID')['GA']) # GA - Ground Floor Area

    dataset['BA'] = np.where(dataset['BF'] > 0, dataset['GA'], 0) # BA - Basement Area (not observed variability)
    dataset.loc[dataset['GA'].isna(), 'BA'] = np.nan
    dataset.loc[dataset['BT'].isna(), 'BA'] = np.nan

    dataset['SA'] = (dataset['GA'] * dataset['NF']) + (dataset['BA'] * dataset['BF']) # SA - Total Surface Area
    
    areas_order = ['GA','BA','SA']
    col_name_dic["areas"] = areas_order
       
    # B.1.4 Perimeters
    dataset['EP'] = dataset['BID'].map(gdf_buildings.set_index('BID')['EP']) # EP - External Perimeter
    dataset['GP'] = np.nan # GP - Ground Floor Perimeter (internal perimeter - unique physical walls)
    dataset['Cga'] = np.nan # coeficient to calc GP from GA (square shape assumed)
    dataset['BP'] = 4 * np.sqrt(dataset['BA'].astype(float)) # BP - Basement Perimeter (observed in survey, only 1 room per bassement, assumed a square shape)
    
    perimeters_order = ['EP','GP','Cga','BP']
    col_name_dic["perimeters"] = perimeters_order
    
    # B.1.5 DEM features
    dataset['ed'] = np.nan # ed - DEM error altitude outside the building
    col_name_dic["dem"] = ['ed']
    # B.1.6 Event features
    dataset['he'] = np.nan # he - depth outside the building
    event_order = ['he']
    nf_is_nan = dataset['NF'].isna()
    bf_is_nan = dataset['BF'].isna()
    for n in range(-max_bf, max_nf): # hi - depth inside the building
        col_name = f'hi_{n}'
        event_order.append(col_name)
        if n < 0:
            is_valid_floor = (abs(n) <= dataset['BF'])
            is_unknown = bf_is_nan
        else:
            is_valid_floor = (n < dataset['NF'])
            is_unknown = nf_is_nan
        # Vectorized assignment: NaN for existing floors, -999 for others
        dataset[col_name] = np.where(
            is_unknown, 
            np.nan, 
            np.where(is_valid_floor, np.nan, -999)
        )
    col_name_dic["event"] = event_order
    # B.1.7 Content
    new_content_dict = {}
    prefixes = ["n", "p", "ff", "b", "hc", "d", "c"]
    nf_is_nan = dataset['NF'].isna()
    bf_is_nan = dataset['BF'].isna()
    for pf in prefixes:
        pf_group_cols = []  # Temporary list for the current prefix
        for code in CTEs_unq:
            for n in range(-max_bf, max_nf):
                if code == "VEH" and n not in [-1, 0]:
                    continue
                col_name = f"{pf}_{code}_{n}"
                pf_group_cols.append(col_name)
                
                # Define floor validity mask
                if n < 0:
                    floor_exists = (abs(n) <= dataset['BF'])
                    unknown_floor = bf_is_nan
                else:
                    floor_exists = (n < dataset['NF'])
                    unknown_floor = nf_is_nan
                
                # Logic: NaN if floor exists/unknown, -999 if floor is physically impossible
                new_content_dict[col_name] = np.where(
                    unknown_floor, 
                    np.nan, 
                    np.where(floor_exists, np.nan, -999)
                )
                
        # Save the ordered list of columns for this prefix to your dictionary
        col_name_dic[f"{pf}_content"] = pf_group_cols

    # Concatenate new columns to the main dataset
    new_cols_df = pd.DataFrame(new_content_dict, index=dataset.index)
    dataset = pd.concat([dataset, new_cols_df], axis=1).copy()
    
    # B.1.8 Continent
    new_continent_dict = {}
    continent_config = {
        'PUM': {'general': ['p', 'e', 'c'], 'by_floor': []}, #
        'CLE': {'general': ['p'], 'by_floor': ['e', 'c']}, #
        #'DHU': {'general': ['p'], 'by_floor': ['e', 'c']},
        'SOI': {'general': ['m', 'p'], 'by_floor': ['e', 'ff', 'b', "hc", 'd', 'c']},
        
        'SKT': {'general': ['m', 'p'], 'by_floor': ['e', 'n', 'ff', 'b', "hc", 'd', 'c']}, #
        'RDR': {'general': ['m', 'p'], 'by_floor': ['n', 'ff', 'b', "hc", 'd', 'c']}, #
        'WND': {'general': ['p'], 'by_floor': ['n', 'ff', 'b', "hc", 'd', 'c']}, #
        'PLG': {'general': ['p'], 'by_floor': ['n', 'ff', 'b', "hc", 'd', 'c']}, #
        
        'PRW': {'general': ['m', 'pr', 'p'], 'by_floor': ['e', 'ff', 'b', "hc", 'd', 'c']},
        'EXF': {'general': ['m', 'e'], 'by_floor': []},
        'ETP': {'general': ['p', 'c'], 'by_floor': []},
        'ITP': {'general': [], 'by_floor': ['c']},
        #'ELS': {'general': ['t', 'ff', 'd', 'p', 'c'], 'by_floor': []},
        'FRI': {'general': [], 'by_floor': ['n', 'ff']}
    }
    all_prefixes = ["n",'e','m', 't', "p", 'pr', "ff", "b", "hc", "d", "c"]
    nf_is_nan = dataset['NF'].isna()
    bf_is_nan = dataset['BF'].isna()
    for pf in all_prefixes:
        pf_group_cols = []  # List to track columns for the current prefix
        for continent, cfg in continent_config.items():
            # General columns (No floor suffix)
            if pf in cfg['general']:
                col_name = f"{pf}_{continent}"
                new_continent_dict[col_name] = np.nan
                pf_group_cols.append(col_name)
                
            # Floor-specific columns
            if pf in cfg['by_floor']:
                for n in range(-max_bf, max_nf):
                    if continent in ['WND', 'FRI'] and n < 0:
                        continue
                    col_name = f"{pf}_{continent}_{n}"
                    pf_group_cols.append(col_name)
                    # Floor existence mask
                    if n < 0:
                        exists = (abs(n) <= dataset['BF'])
                        is_unknown = bf_is_nan
                    else:
                        exists = (n < dataset['NF'])
                        is_unknown = nf_is_nan
                    
                    new_continent_dict[col_name] = np.where(
                        is_unknown, 
                        np.nan, 
                        np.where(exists, np.nan, -999)
                    )
        # Save the ordered list of columns for this prefix
        col_name_dic[f"{pf}_continent"] = pf_group_cols
    continent_df = pd.DataFrame(new_continent_dict, index=dataset.index)
    dataset = pd.concat([dataset, continent_df], axis=1).copy()
    
    return dataset, col_name_dic

dataset, col_name_dic = create_dataset()

RUN_TEST = False
if RUN_TEST:
    bids_per_rp = dataset.groupby('RP')['BID'].apply(set)
    common_bids = list(set.intersection(*bids_per_rp))
    np.random.seed(42) 
    selected_bids = np.random.choice(common_bids, size=5, replace=False)
    dataset = dataset[dataset['BID'].isin(selected_bids)].copy()
    dataset= dataset.sort_values(by=['BID', 'RP']).reset_index(drop=True)

    # Optional: Print to verify
    print(f"Selected Test BIDs: {selected_bids}")
    print(f"Test Dataset Shape: {dataset.shape}")
    print(dataset[['DID', 'BID', 'RP', 'BT', 'NF', 'BF']].head(10))

#endregion 
# region B.2 Monte Carlo (sample/calculate unknow data)
# region B.2.1 Load fm_df and fo_data
fm_df = pd.read_pickle(fm_path)
fm_df['FID'] = range(len(fm_df))
fm_df = fm_df[['FID'] + [c for c in fm_df.columns if c != 'FID']]
with open(fo_path, 'rb') as f:
    fo_data = pickle.load(f)
#endregion 
# region B.2.2 Monte-Carlo Function
'''
The functions is prepared to allow multiprocessing and
vectorized calculous for fastest execution.
'''
# sn = 1
def it_Monte_Carlo(sn, dataset,
                   fm_df, fo_data,
                   col_name_dic):
    # Needed for multiprocessing
    import ast
    import pandas as pd
    import numpy as np
    from scipy import stats
    
    def ensure_numeric_params(p):
        """Converts string representation of tuples/lists back to numeric
        objects. Cleas numpy-specific strings wrappers"""
        if isinstance(p, str):
            # 1. Strip numpy-style wrappers from the string
            # This turns '(np.float64(13.6), ...)' into '(13.6, ...)'
            clean_p = (p.replace('np.float64', '')
                        .replace('np.int64', '')
                        .replace('np.float32', ''))
            
            try:
                p = ast.literal_eval(clean_p)
            except (ValueError, SyntaxError):
                return p

        # 2. Ensure all elements inside are standard floats (not strings/numpy types)
        if isinstance(p, (tuple, list)):
            try:
                return tuple(float(x) for x in p)
            except (TypeError, ValueError):
                return p
                
        # 3. Handle single numeric values
        try:
            return float(p)
        except (TypeError, ValueError):
            return p

    def prints_if(p, prints = False):
        if prints: print(p)

    def sample_batch(fm_row, size, obs_data = None, hi_values = None, default_BP_sample=0, sample_only_positives = False, q_range=(0.05, 0.95), prints=False,
                    bp_dep_sample=None, dep_sample=None, min_limit=None,):
        """
        Generates a batch of samples.
        If BP is not NaN, it performs a Hurdle model with Bernoulli trials.
        If hi_values are sent, evaluates the Fragility Function (CDF) for a batch of water depths
        (hi_values) returning a probability of damage between 0 to 1. Works within q_range and for KDE
        and scipy functions.
        It can sample only possitive values if desired and within a desired quantil range.
        To prevent numerical errors when tails are heavy it force a min-max quantile of Q001-Q999
        It can handles dependent sampling through bp_dep_sample and dep_sample.
        """
        
        if size <= 0:
            return np.array([])
        
        # Get data from dts_group
        DT = fm_row['DT']
        BP = fm_row['BP']
        FN = fm_row['FN']
        FP = ensure_numeric_params(fm_row['FP'])
        
        # Extract desired quantile bounds
        q_min = max(q_range[0], 0.001)
        q_max = min(q_range[1], 0.999)

        # Check if min limit is set
        actual_min_limit = 0 if sample_only_positives else min_limit
        
        # --- 1. Independent/Dependent Hurdle Masking ---
        if pd.notna(BP):
            if bp_dep_sample is not None:
                # Generate dependent trials for all rows together
                hurdle_results = np.full(size, bp_dep_sample <= BP, dtype=bool)
            else:
                # Generate independent trials for every row in the batch
                # 1 = Pass (gets distribution sample), 0 = Fail (gets default)
                hurdle_results = np.random.binomial(1, BP, size).astype(bool) # Equivalent to bernouilli, but faster performance
            pass_count = hurdle_results.sum()
            # Initialize output with the default value (e.g., 0)
            final_samples = np.full(size, default_BP_sample, dtype=float)
            if pass_count == 0:
                return final_samples
        else:
            # No hurdle: everyone passes
            hurdle_results = np.ones(size, dtype=bool)
            pass_count = size
            final_samples = np.full(size, np.nan)

        # --- 2. Sample only for those who passed ---
        if FN == "Constant":
            if isinstance(FP, (tuple, list, np.ndarray)) and len(FP) > 0:
                val = float(FP[0])
            else:
                val = float(FP)
            batch_samples = np.full(pass_count, val)
        elif FN == "KDE":
            obs_array = np.sort(np.array(obs_data))
            kde = stats.gaussian_kde(obs_data)
            lower_q, upper_q = q_min, q_max
            if actual_min_limit is not None:
                p_min = kde.integrate_box_1d(-np.inf, actual_min_limit)
                lower_q = max(q_min, p_min)
            if hi_values is not None:
                # --- Fragility Function Logic ---
                p_hi = np.array([kde.integrate_box_1d(-np.inf, val) for val in hi_values])
                p_hi_clipped = np.clip(p_hi, lower_q, upper_q)
                batch_samples = (p_hi_clipped - lower_q) / (upper_q - lower_q)
            else:
                # --- Truncated Sampling Logic ---
                pool_size = max(10000, pass_count)
                resamples = np.sort(kde.resample(size=pool_size).flatten())
                #u = np.random.uniform(lower_q, upper_q, size=pass_count)
                #batch_samples = np.quantile(resamples, u)
                if dep_sample is not None:
                    if isinstance(dep_sample, (np.ndarray, pd.Series, list)):
                        u = np.clip(dep_sample, lower_q, upper_q)
                    else:
                        u = np.full(pass_count, max(lower_q, min(upper_q, dep_sample)))
                else:
                    u = np.random.uniform(lower_q, upper_q, size=pass_count)
                
                batch_samples = np.quantile(resamples, u)
        elif FN == "Empirical":
            obs_array = np.array(obs_data)
            if actual_min_limit is not None:
                obs_array = obs_array[obs_array >= actual_min_limit]
            if dep_sample is not None:
                idx = int(dep_sample * len(obs_array))
                idx = min(max(idx, 0), len(obs_array) - 1)
                batch_samples = np.full(pass_count, np.sort(obs_array)[idx])
            else:
                batch_samples = np.random.choice(obs_array, size=pass_count, replace=True)
        else:
            dist = getattr(stats, FN)
            lower_q, upper_q = q_min, q_max
            params = FP if isinstance(FP, (tuple, list)) else (FP,)
            if actual_min_limit is not None:
                p_min = dist.cdf(actual_min_limit, *params)
                lower_q = max(q_min, p_min)
            if hi_values is not None:
                # --- Fragility Function Logic ---
                p_hi = dist.cdf(hi_values, *params)
                p_hi_clipped = np.clip(p_hi, lower_q, upper_q)
                batch_samples = (p_hi_clipped - lower_q) / (upper_q - lower_q)
            else:
                if dep_sample is not None:
                    #u = np.full(pass_count, max(lower_q, min(upper_q, dep_sample)))
                    if isinstance(dep_sample, (np.ndarray, pd.Series, list)):
                        u = np.clip(dep_sample, lower_q, upper_q)
                    else:
                        u = np.full(pass_count, max(lower_q, min(upper_q, dep_sample)))
                else:
                    # --- Truncated Sampling Logic ---
                    u = np.random.uniform(lower_q, upper_q, size=pass_count)
                batch_samples = dist.ppf(u, *params)
            
        # --- 3. Final checks ---
        # 3a. Shift logic for specific discrete stats distributions
        if pd.notna(BP) and FN in ['poisson', 'nbinom', 'binom']:
            batch_samples = batch_samples + 1
        
        # 3b. Forced Positive Check (removes KDE noise or numerical precision errors)
        if actual_min_limit is not None:
            batch_samples = np.maximum(batch_samples, actual_min_limit)
        
        # 3c. Discrete data check: Round if DT is "d".
        # NOTE: If data is discrete a empirical distribution should be used
        # but if distribution was defined with expert judgemtn a continuous distribution can be preferable
        # if it is discrete this avoid decimals values.
        if DT == "d":
            batch_samples = np.round(batch_samples).astype(np.float64)
            
        # Place the samples back into the specific indices that passed the hurdle
        final_samples[hurdle_results] = batch_samples

        return final_samples
    # DC = 'b_RDR_0', q_range=(0, 1), variant='b', prints=True,
    def sample_DC(DC, df_it, hierarchy=['RP', 'BID', 'BT'], default_BP_sample=0, variant = None, mat = False, sample_only_positives = False, q_range=(0.05, 0.95), prints=False,
                  bp_dep_sample=None, dep_sample=None, min_limit=None):
        
        # 0. Get DC base, hi and mat column if exist and needed
        parts = DC.split('_')
        has_suffix = len(parts) > 1 and (parts[-1].isdigit() or (parts[-1].startswith('-') and parts[-1][1:].isdigit()))
        if has_suffix:
            suffix = parts[-1] if has_suffix else None
            
            DC_base = "_".join(parts[:-1])
            DC_code = parts[1] if len(parts) > 2 else parts[0]

            if variant == 'hi':
                hi_col = f"hi_{suffix}"
            elif variant == 'hc':
                b_col = f"b_{DC_code}_{suffix}"
                DC_base = DC_base.replace('hc', 'ff')
            if mat:
                mat_col = f"m_{DC_code}_{suffix}"
        else:
            # Case: pf_DC or DC
            DC_base = DC
            DC_code = parts[1] if len(parts) == 2 else parts[0]
            if variant == 'hi':
                hi_col = "hi"
            elif variant == 'hc':
                b_col = f"b_{DC_code}"
                DC_base = DC_base.replace('hc', 'ff')
            if mat:
                mat_col = f"m_{DC_code}"

        prints_if(f"\n{'='*20}\nPROCESSING... {DC}\n{'='*20}", prints=prints)
        
        # 1. Isolate rows requiring sampling
        prints_if(f"\nPREPARING DATA...", prints=prints)
        base_cols = ['DID', 'FID', 'BT', 'BID', 'RP']
        fetch_cols = base_cols + ([DC] if DC not in base_cols else [])
        if variant == 'hi':
            if hi_col in df_it.columns:
                fetch_cols.append(hi_col)
            else:
                print(f"CRITICAL: {hi_col} not found in df_it. Sampling will fail.")
                return
        elif variant == 'hc':
            if b_col in df_it.columns:
                fetch_cols.append(b_col)
            else:
                print(f"CRITICAL: {b_col} not found. hc needs b.")
                return
        if mat:
            fetch_cols.append(mat_col)
        dts = df_it.loc[df_it[DC].isna(), fetch_cols].copy()
        dts['FID'] = np.nan #To ensure fresh falback
        prints_if(f"-Current missing values: {len(dts)}.\nExample:\n{dts.head()}", prints=prints)
        if mat:
            dts = dts.rename(columns={mat_col: 'MAT'})

        # Variant b. Bernouilli
        if variant == 'b':
            prints_if("--> Variant 'b': Using Uniform(0,1)", prints=prints)
            virtual_fm_row = pd.Series({'DT': 'c', 'BP': np.nan, 'FN': 'uniform','FP': (0, 1)})
            sampled_values = sample_batch(
                        virtual_fm_row, # fm_row = virtual_fm_row
                        len(dts), # size = len(dts)
                        default_BP_sample=default_BP_sample, 
                        sample_only_positives=sample_only_positives,
                        q_range=q_range,
                        prints=prints,
                        bp_dep_sample=bp_dep_sample, dep_sample=dep_sample,
                        min_limit=min_limit) 
            dts[DC] = sampled_values

            # Global update to df_it using index alignment
            df_it.set_index('DID', inplace=True)
            df_it.update(dts.set_index('DID')[[DC]])
            df_it.reset_index(inplace=True) # df_it[DC]
            return
            
        # 2. Get unique distributions for this DC
        fm_target = fm_df[fm_df['DC'] == DC_base]
        fm_target = fm_target.dropna(subset=['FN']) # Clean possible FN nans
        prints_if(f"-Available distributions: {len(fm_target)}.\n Example:\n {fm_target.head()}", prints=prints)

        # 3. Define FID hierarchical connection (define DID-FID links)
        prints_if(f"\nSTARTING HIERARCHICAL FALLBACK...", prints=prints)
        # Initial specific merge
        merge_keys = ['RP', 'BID', 'BT']
        if mat:
            merge_keys.append('MAT')
        dts = dts.drop(columns=['FID']).merge(
            fm_target[['FID'] + merge_keys], 
            on=merge_keys, 
            how='left'
        )
        # Iterative fallback to generic cases (setting columns to 0 sequentially)
        if mat:
            hierarchy.append('MAT')
        for col in hierarchy:
            
            nan_mask = dts['FID'].isna()
            if not nan_mask.any():
                prints_if(f"--> All rows matched before level: {col}", prints=prints)
                break
            
            prints_if(f"--> Fallback: {nan_mask.sum()} rows missing FID. Setting {col} = 0", prints=prints)
            
            # Apply fallback to local copy
            dts.loc[nan_mask, col] = 0
            
            # Re-Merge the missing subset
            matched = dts.loc[nan_mask, merge_keys].merge(
                fm_target[['FID'] + merge_keys], 
                on=merge_keys, 
                how='left'
            )
            
            # Update FID using index alignment
            dts.loc[nan_mask, 'FID'] = matched['FID'].values
        
        # Get output
        if dts['FID'].isna().any():
            print(f"CRITICAL: Not generic distribution found for {DC}")
            return
        else:
            nan_mask = dts['FID'].isna()
            prints_if(f"--> Fallback complete: {nan_mask.sum()} rows missing FID.", prints=prints)

        # 4. Vectorized sampling
        dts['FID'] = dts['FID'].astype(int)
        unique_fids = np.sort(dts['FID'].unique())
        prints_if(f"-Unique distribution groups (FIDs): {len(unique_fids)}.\n\t Example: {unique_fids}", prints=prints)

        prints_if(f"\nSTARTING BATCH SAMPLING...", prints=prints)
        for fid in unique_fids:
            
            # 1. Identify rows belonging to this FID group
            group_mask = dts['FID'] == fid
            size = group_mask.sum()
            
            if size > 0:
                fm_row = fm_target[fm_target['FID'] == fid].iloc[0]
                prints_if(f"--> Group FID {fid}: Sampling {size} rows ({fm_row['FN']})", prints=prints)
                
                # 2. KDE/Empirical OBS Data Retrieval Logic (Common for both)
                obs_data = None
                if fm_row['FN'] in ["KDE", "Empirical"]:
                    odc_key = fm_row['ODC']
                    # Retrieve from the dictionary fo_data
                    obs_data = fo_data.get(odc_key)
                    if obs_data is None:
                        print(f"CRITICAL: ODC {odc_key} not found in fo_data for FID {fid} and needed due to KDE/Empirical distribution")
                        return
                
                # 3. Conditional Sampling Logic
                if variant == 'hi':
                    # --- Fragility Function Path (CF) ---
                    # A. Check if hi column exists
                    if hi_col not in dts.columns:
                        print(f"CRITICAL: Column {hi_col}, needed for Fragility Function, not found.")
                        return

                    # B. Extract Water Depths for this batch
                    # We fill NaNs with 0 or -999 to avoid errors, though they should be masked out later
                    batch_hi_values = dts.loc[group_mask, hi_col].values
                    prints_if(f"-->hi values found for FID {fid}: {len(batch_hi_values)}.\n\t Example: {batch_hi_values}", prints=prints)
                    
                    # C. Run Fragility Sampling (CDF)
                    sampled_values = sample_batch(
                        fm_row,
                        size,
                        obs_data=obs_data,
                        hi_values = batch_hi_values,
                        default_BP_sample=default_BP_sample, 
                        sample_only_positives=sample_only_positives,
                        q_range=q_range,
                        prints=prints,
                        bp_dep_sample=bp_dep_sample,
                        dep_sample=dep_sample,
                        min_limit=min_limit)  
                
                elif variant == 'hc':
                    # --- Critical Threshold Path (hc) ---
                    batch_b_values = dts.loc[group_mask, b_col].values
                    sampled_values = sample_batch(
                        fm_row,
                        size,
                        obs_data=obs_data,
                        default_BP_sample=default_BP_sample, 
                        sample_only_positives=sample_only_positives,
                        q_range=q_range,
                        prints=prints,
                        dep_sample=batch_b_values,  # Pass 'b' array as the quantiles
                        min_limit=min_limit)
                                  
                else:
                    # --- Standard Content Sampling Path (CC) ---
                    sampled_values = sample_batch(
                        fm_row,
                        size,
                        obs_data=obs_data,
                        default_BP_sample=default_BP_sample, 
                        sample_only_positives=sample_only_positives,
                        q_range=q_range,
                        prints=prints,
                        bp_dep_sample=bp_dep_sample,
                        dep_sample=dep_sample,
                        min_limit=min_limit)

                # 5. Assign results back to main dataframe
                dts.loc[group_mask, DC] = sampled_values

        #5. Global update
        prints_if(f"\nUPDATING DATASET...", prints=prints)
        dts[DC] = dts[DC].astype(np.float64)
        # Only update the target DC column in df_it using DID as the key
        df_it.set_index('DID', inplace=True)
        dts_indexed = dts.set_index('DID')
        df_it.update(dts_indexed[[DC]])
        df_it.reset_index(inplace=True)

        rem_nans = df_it[DC].isna().sum()

        prints_if(f"Sampling process for'{DC}' done. Example:\n{dts_indexed.head()}", prints=prints)
        prints_if(f"Remaining NaNs: {rem_nans}", prints=prints)
        prints_if(f"{'='*20}\n", prints=prints)

        log = "Remaining NaNs for {DC}: {rem_nans}"
        if rem_nans > 0:
            return log
        else:
            return
    
    # Main samples and calc logic
    for attempt in range(3):
        try:
            # Create a copy of the dataset
            df_it = dataset.copy()
            df_it['SN'] = sn
            df_it = df_it[['SN'] + [c for c in df_it.columns if c != 'SN']]
            max_bf = int(df_it['BF'].max())
            max_nf = int(df_it['NF'].max())
            floor_range = range(-max_bf, max_nf)
            
            ### --------- BUILDING ---------
            ## All known
                # BID / RP / GA / EP / BP
            ## Sample
                # BT / NF / BF / HU / IH / GL / BH / Cga
            for DC in ['BT','NF','BF','HU','IH','GL','BH','Cga']:
                if DC == "BF":
                    df_it.loc[df_it['BT'].isin([1, 3, 5]), 'BF'] = 0
                elif DC == "HU":
                    df_it.loc[df_it['BT'].isin([1, 2]), 'HU'] = 1
                elif DC == "BH":
                    df_it.loc[df_it['BT'].isin([1, 3, 5]), 'BH'] = -999
                if DC == "Cga":
                    sample_DC(DC, df_it, q_range=(0, 1), min_limit=4.0)
                else:
                    sample_DC(DC, df_it, q_range=(0, 1))
            # df_it['GL'].min(), df_it['GL'].max()
            
            ## Calc
            # BA / SA / BL / GP
            df_it['BA'] = np.where(df_it['BT'].isin([1, 3, 5]), -999, np.where(df_it['BF'] > 0, df_it['GA'], 0))
            df_it['SA'] = (df_it['GA'] * df_it['NF']) + (np.maximum(df_it['BA'], 0) * np.maximum(df_it['BF'], 0))
            df_it['BL'] = np.where(df_it['BT'].isin([1, 2, 3]), -999, df_it['GL'] - df_it['BH'])
            df_it['GP'] = df_it['Cga'] * np.sqrt(df_it['GA'])
            
            # General fix (Set to -999 all those Bassement-related and floors do not exist according to the new BT samples)
            for _, columns in col_name_dic.items():
                for col in columns:
                    # Identify floor index (e.g., _-1, _0, _1)
                    parts = col.split('_')
                    try:
                        n = int(parts[-1])
                        
                        if n < 0:
                            # Bassement does not exist if abs(n) > BF OR if BT in [1, 3, 5]
                            invalid_mask = (df_it['BT'].isin([1, 3, 5]))
                        else:
                            # Floor does not exist if n >= NF
                            invalid_mask = (n >= df_it['NF'])
                        
                        # Switch invalid sampled floors to -999
                        df_it.loc[invalid_mask, col] = -999
                    except (ValueError, IndexError):
                        # Not a floor-indexed column (e.g., 'he' or 'p_PUM'), skip
                        continue
            
            ### --------- DEM FEATURE  ---------
            ## Sample: ed (fully dependent)
            sample_DC('ed', df_it, hierarchy=['RP', 'BT', 'BID'], q_range=(0, 1))
            
            ### --------- EVENT FEATURE  ---------
            ## Sample: he
            he_bp_dep_sample = np.random.uniform(0, 1)
            he_dep_sample = np.random.uniform(0, 0.95)
            sample_DC('he', df_it, hierarchy = [ 'BT', 'BID', 'RP'],
                      sample_only_positives = True,
                      bp_dep_sample=he_bp_dep_sample, dep_sample=he_dep_sample)
                        
            ## Calc: hi
            def hi_calc(df):
                """
                Vectorized calculation of flood depth (hi) for each building floor.
                Calculates water level relative to floor elevations and clips to floor height.
                - Corrects he with ed only if he > 0.
                - Clips corrected_he to 0 if subtraction results in negative values.
                - Calculates hi only where water_levels > 0.
                - Uses precomputed floor_range to iterate.
                """
                # 1. Pre-calculate water levels
                # Real he = max(he - ed, 0) if he > 0, else 0
                water_levels = np.where(df['he'] > 0, np.maximum(df['he'] - df['ed'], 0), 0.0)
            
                # Real he = he - ed (if he > 0, else 0)
                corrected_he = np.where(df['he'] > 0, df['he'] - df['ed'], 0)
                corrected_he = np.maximum(corrected_he, 0)
                water_levels = pd.Series(corrected_he, index=df.index)
                
                # 2. Vectorized calculation for each floor column
                # e.g.: n = -1 ; n = 0
                # e.g.: col = "hi_-1" ; col = "hi_0"
                for n in floor_range:
                    col = f'hi_{n}'
                    #df_it[col].max()
                    if n < 0:
                        # Basement logic: Valid if abs(n) <= BF
                        is_valid = (np.abs(n) <= df['BF'])
                        floor_base_level = df['BL'] + (np.abs(n) - 1) * df['BH']
                        floor_limit = df['BH']
                    else:
                        # Above-ground logic: Valid if n < NF
                        is_valid = (n < df['NF'])
                        floor_base_level = df['GL'] + n * df['IH']
                        floor_limit = df['IH']
                        
                    # 4. Calculate hi
                    # Logic: Only calculate if water_levels > 0, else 0
                    # hi = (water_level - base) clipped between 0 and floor height limit
                    hi_calculated = np.where(
                        water_levels > 0,
                        np.clip(water_levels - floor_base_level, 0, floor_limit),
                        0.0
                    )
                    
                    # 5. Apply mask: 
                    # If floor is valid, use the calculation. If not, use -999.
                    df[col] = np.where(is_valid, hi_calculated, -999)
                    #df_it[col].max()
                return df
            df_it = hi_calc(df_it)
                
            ### --------- CONTENT ---------
            ## Sample: n_CTE / p_CTE / ff_CTE / b_CTE
            content_map = {
                'n_content': {'h': ['BID', 'RP', 'BT'], 'v': None, 'q': (0, 0.95)},
                'p_content': {'h': ['BID', 'RP', 'BT'], 'v': None, 'q': (0, 0.95)},
                'ff_content': {'h': ['BID', 'RP', 'BT'], 'v': 'hi', 'q': (0, 1)},
                'b_content': {'h': None, 'v': 'b', 'q': (0, 1)},
                'hc_content': {'h': ['BID', 'RP', 'BT'], 'v': 'hc', 'q': (0, 1)}
            }
            for key, config in content_map.items():
                # Verify the sub-dictionary exists within col_name_dic
                if key in col_name_dic:
                    column_list = col_name_dic[key]
                    
                    for col_name in column_list:
                        # Execute sampling using the mapped parameters
                        sample_DC(
                            col_name,
                            df_it, 
                            hierarchy=config['h'], 
                            sample_only_positives=True, 
                            variant=config['v'], 
                            q_range=config['q']
                        )
            ## Calc: d_CTE / c_CTE
            content_cols = zip(
                col_name_dic['n_content'],
                col_name_dic['p_content'],
                col_name_dic['ff_content'],
                col_name_dic['b_content'],
                col_name_dic['d_content'],
                col_name_dic['c_content']
            )
            for n_col, p_col, ff_col, b_col, d_col, c_col in content_cols:
                # 1. Identify valid rows
                valid_mask = (df_it[ff_col] != -999)
                invalid_mask = ~valid_mask
                
                # 2. Perform vectorized comparison for damage
                # Damage (1) if b <= ff, otherwise No Damage (0)
                df_it.loc[valid_mask, d_col] = (df_it.loc[valid_mask, b_col] <= df_it.loc[valid_mask, ff_col]).astype(int)
                df_it.loc[invalid_mask, d_col] = -999
                
                # 3. Calculate cost: Damage (0 or 1) * Quantity * Price
                # If d=0, cost is 0. If d=1, cost is n * p.
                df_it.loc[valid_mask, c_col] = (
                    df_it.loc[valid_mask, d_col] * df_it.loc[valid_mask, n_col] * df_it.loc[valid_mask, p_col]
                )
                df_it.loc[invalid_mask, c_col] = -999
            
            ### --------- CONTENTINENT ---------    
            ### -- GENERAL --
            ## Sample: p_PUM / p_CLE
            for DC in ['p_PUM', 'p_CLE']:
                sample_DC(DC, df_it, sample_only_positives = True, q_range=(0.05, 0.95))
            
            ## Calc
            # Extensions
            # e_PUM
            invalid_row_mask = (df_it['GL'] == -999) | (df_it['GL'].isna())
            ga_term = np.where(
                (df_it['GL'] != -999) & (df_it['GL'] < 0), 
                df_it['GA'] * (-df_it['GL']), 
                0.0
            )
            ba_term = np.where(
                (df_it['BL'] != -999) & (df_it['BL'] < 0), 
                df_it['BA'] * (-df_it['BL']), 
                0.0
            )
            df_it['e_PUM'] = ga_term + ba_term
            df_it.loc[invalid_row_mask, 'e_PUM'] = -999
            # e_CLE
            for n in floor_range:
                hi_col = f'hi_{n}'
                e_col = f'e_CLE_{n}'
                
                # Select geometry based on level index
                if n < 0:
                    perimeter = df_it['BP']
                    area = df_it['BA']
                else:
                    perimeter = df_it['GP']
                    area = df_it['GA']

                # Filter for floors that exist (not -999) and are flooded (> 0)
                valid_geometry = (perimeter != -999) & (area != -999)
                valid_depth = (df_it[hi_col] != -999)
                exists_mask = valid_geometry & valid_depth
                flooded_mask = exists_mask & (df_it[hi_col] > 0)
                dry_mask = exists_mask & (df_it[hi_col] == 0)
                invalid_mask = ~exists_mask

                # Apply values
                df_it.loc[dry_mask, e_col] = 0.0
                df_it.loc[flooded_mask, e_col] = (perimeter * df_it[hi_col]) + area
                df_it.loc[invalid_mask, e_col] = -999.0
            
            # Costs
            # c_PUM
            df_it['c_PUM'] = np.where(
                (df_it['hi_0'] > 0),
                df_it['e_PUM'] * df_it['p_PUM'],
                0.0
            )
            #c_CLE
            for n in floor_range:
                e_col = f'e_CLE_{n}'
                c_col = f'c_CLE_{n}'
                
                if e_col in df_it.columns:
                    valid_mask = (df_it[e_col] != -999)
                    invalid_mask = (df_it[e_col] == -999)
                    # Cost = Extension * Unit Price
                    df_it.loc[valid_mask, c_col] = df_it.loc[valid_mask, e_col] * df_it.loc[valid_mask, 'p_CLE']
                    df_it.loc[invalid_mask, c_col] = -999.0

            ## -- SOIL --
            ## Sample: m_SOI / p_SOI / ff_SOI / b_SOI
            for DC in ['m_SOI', 'p_SOI', 'ff_SOI','b_SOI','hc_SOI']:
                if DC in ['m_SOI']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0, 1))
                elif DC in ['p_SOI']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0.05, 0.95), mat=True)
                elif DC in ['ff_SOI']:
                    for n in floor_range:
                        sample_DC(f"{DC}_{n}", df_it, hierarchy=['BID', 'RP', 'BT'], sample_only_positives = True, variant = 'hi', q_range=(0, 1))
                elif DC in ['b_SOI']:
                    for n in floor_range:
                        sample_DC(f"{DC}_{n}", df_it, variant = 'b', q_range=(0, 1))
                elif DC in ['hc_SOI']:
                    for n in floor_range:
                        sample_DC(f"{DC}_{n}", df_it, hierarchy=['BID', 'RP', 'BT'], sample_only_positives = True, variant = 'hc', q_range=(0, 1))
            
            ## Calc: d_SOI / e_SOI / c_SOI
            for n in floor_range:
                ff_col = f"ff_SOI_{n}"
                b_col = f"b_SOI_{n}"
                d_col = f"d_SOI_{n}"
                e_col = f"e_SOI_{n}"
                c_col = f"c_SOI_{n}"
                
                # 1. use rows where both ff and b are not -999)
                valid_mask = (df_it[ff_col] != -999) & (df_it[b_col] != -999)
                invalid_mask = ~valid_mask
                
                # 2. Perform vectorized comparison
                # Damage (1) if b <= ff, otherwise No Damage (0)
                df_it.loc[valid_mask, d_col] = (df_it.loc[valid_mask, b_col] <= df_it.loc[valid_mask, ff_col]).astype(int)
                df_it.loc[invalid_mask, d_col] = -999.0
                
                # 3. Identify damaged rows for this specific floor
                damaged_mask = (df_it[d_col] == 1)
                no_damage_mask = (df_it[d_col] == 0.0)
                
                # 4. Calc damage and cost
                if damaged_mask.any():
                    # Logic: If n < 0 (Basement) use BA, else (n >= 0) use GA
                    area_col = 'BA' if n < 0 else 'GA'
                    
                    # Extension by floor
                    df_it.loc[damaged_mask, e_col] = df_it.loc[damaged_mask, area_col]
                    df_it.loc[no_damage_mask, e_col] = 0.0
                    
                    # Cost
                    df_it.loc[valid_mask, c_col] = df_it.loc[valid_mask, 'p_SOI'] * df_it.loc[valid_mask, e_col]
                elif no_damage_mask.any():
                    # Case where rows are valid but none are damaged
                    df_it.loc[no_damage_mask, e_col] = 0.0
                    df_it.loc[no_damage_mask, c_col] = 0.0
            
            ### -- COMPONENTS --
            ## Sample: n, m, p, ff, b for SKT / RDR / WND / PLG
            for DC in [
                'n_SKT', 'n_RDR', 'n_WND', 'n_PLG',
                'm_SKT', 'm_RDR',
                'p_SKT', 'p_RDR', 'p_WND', 'p_PLG',
                'ff_SKT', 'ff_RDR', 'ff_WND', 'ff_PLG',
                'b_SKT', 'b_RDR', 'b_WND','b_PLG',
                'hc_SKT', 'hc_RDR', 'hc_WND', 'hc_PLG']:
                
                if DC in ['n_SKT', 'n_RDR', 'n_WND', 'n_PLG']:
                    for n in floor_range:
                        col_target = f"{DC}_{n}"
                        if col_target == 'n_WND_-1': continue
                        if col_target in df_it.columns:
                            sample_DC(col_target, df_it, hierarchy=['BID', 'RP', 'BT'], sample_only_positives=True, q_range=(0, 1))
                elif DC in ['m_SKT', 'm_RDR']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0, 1))
                elif DC in ['p_SKT', 'p_RDR']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0.05, 0.95), mat=True)
                elif DC in ['p_WND', 'p_PLG']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0.05, 0.95))
                elif DC in ['ff_SKT', 'ff_RDR', 'ff_WND', 'ff_PLG']:
                    for n in floor_range:
                        col_target = f"{DC}_{n}"
                        if col_target == 'ff_WND_-1': continue
                        if col_target in df_it.columns:
                            sample_DC(col_target, df_it, hierarchy=['BID', 'RP', 'BT'], sample_only_positives=True, variant='hi', q_range=(0, 1))
                elif DC in ['b_SKT', 'b_RDR', 'b_WND','b_PLG']: # DC = 'b_RDR', n = 0
                    for n in floor_range:
                        col_target = f"{DC}_{n}"
                        if col_target == 'b_WND_-1': continue
                        if col_target in df_it.columns:
                            sample_DC(col_target, df_it, variant='b', q_range=(0, 1))
                elif DC in ['hc_SKT', 'hc_RDR', 'hc_WND', 'hc_PLG']:
                    for n in floor_range:
                        col_target = f"{DC}_{n}"
                        if col_target == 'hc_WND_-1': continue
                        if col_target in df_it.columns:
                            sample_DC(col_target, df_it, hierarchy=['BID', 'RP', 'BT'], sample_only_positives=True, variant='hc', q_range=(0, 1))
            
            #df_it['b_RDR_-1'],df_it['ff_WND_-1'] # Check b_RDR samples for floor 0, the nan remains
            #df_it['n_RDR_0']
            # print fm_df where column DC is 'n_APP'
            #fm_df[fm_df['DC']=='n_RDR']
            
            ## Calc: e, d, c for SKT / RDR / WND / PLG
            for n in floor_range: # n = -1
                # Component Codes to iterate
                for code in ['SKT', 'RDR', 'WND', 'PLG']: # code = 'RDR'
                    ff_col = f"ff_{code}_{n}"
                    b_col = f"b_{code}_{n}"
                    n_col = f"n_{code}_{n}"
                    d_col = f"d_{code}_{n}"
                    e_col = f"e_{code}_{n}"
                    c_col = f"c_{code}_{n}"
                    p_col = f"p_{code}"
                    
                    if ff_col not in df_it.columns or b_col not in df_it.columns:
                        continue
                    
                    # 1. use rows where both ff and b are not -999)
                    valid_mask = (df_it[ff_col] != -999) & (df_it[b_col] != -999)
                    invalid_mask = ~valid_mask
                    
                    # 2. Perform damage calculation
                    df_it.loc[valid_mask, d_col] = (df_it.loc[valid_mask, b_col] <= df_it.loc[valid_mask, ff_col]).astype(float)
                    df_it.loc[invalid_mask, d_col] = -999.0
                    
                    # Define damage status within valid rows
                    damaged_mask = (df_it[d_col] == 1.0)
                    no_damage_mask = (df_it[d_col] == 0.0)
                    
                    # 3. Extension (if needed) and cost Calculation (e, c)
                    if code == 'SKT':
                        # SKT has specific scaling logic based on n/20
                        if damaged_mask.any():
                            peri_col = 'BP' if n < 0 else 'GP'
                            n_values = df_it.loc[damaged_mask, n_col].clip(0, 20)
                            scaling_factor = n_values / 20.0
                            df_it.loc[damaged_mask, e_col] = df_it.loc[damaged_mask, peri_col] * scaling_factor
                            # Cost = Calculated extension * Price
                            df_it.loc[damaged_mask, c_col] = df_it.loc[damaged_mask, e_col] * df_it.loc[damaged_mask, p_col]
                        # Flags and Zeroes for SKT (includes e_col)
                        if no_damage_mask.any():
                            df_it.loc[no_damage_mask, [e_col, c_col]] = 0.0
                        df_it.loc[invalid_mask, [e_col, c_col]] = -999.0
                    elif code in ['RDR', 'WND', 'PLG']:
                        if damaged_mask.any():
                            # Cost = Number of items * Price (since d_col is 1.0)
                            df_it.loc[damaged_mask, c_col] = df_it.loc[damaged_mask, n_col] * df_it.loc[damaged_mask, p_col]
                        # Flags and Zeroes for Discrete (no e_col)
                        if no_damage_mask.any():
                            df_it.loc[no_damage_mask, c_col] = 0.0
                        df_it.loc[invalid_mask, c_col] = -999.0
                
            ## -- WALLS --
            ## Sample: m, pr, p, ff, b for PRW / EXF / ETP
            for DC in ['m_PRW', 'm_EXF',
                    'pr_PRW', 'p_PRW', 'p_ETP',
                    'ff_PRW',
                    'b_PRW',
                    'hc_PRW']:
                
                if DC in ['m_PRW', 'm_EXF']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0, 1))
                elif DC in ['pr_PRW', 'p_PRW']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0.05, 0.95), mat=True)
                elif DC in ['p_ETP']:
                    sample_DC(DC, df_it, sample_only_positives = True, q_range=(0.05, 0.95))
                elif DC in ['ff_PRW']:
                    for n in floor_range:
                        sample_DC(f"{DC}_{n}", df_it, hierarchy=['BID', 'RP', 'BT'], sample_only_positives = True, variant = 'hi', q_range=(0, 1))
                elif DC in ['b_PRW']:
                    for n in floor_range:
                        sample_DC(f"{DC}_{n}", df_it, variant = 'b', q_range=(0, 1))
                elif DC in ['hc_PRW']:
                    for n in floor_range:
                        sample_DC(f"{DC}_{n}", df_it, hierarchy=['BID', 'RP', 'BT'], sample_only_positives = True, variant = 'hc', q_range=(0, 1))
            
            ## Calc: d, e, c for PRW
            for n in floor_range:
                ff_col = f"ff_PRW_{n}"
                b_col = f"b_PRW_{n}"
                d_col = f"d_PRW_{n}"
                e_col = f"e_PRW_{n}"
                c_col = f"c_PRW_{n}"
                c_itp_col = f"c_ITP_{n}"
                
                # 1. Define Valid/Invalid masks
                # Valid if ff, b, IH, and prices are not -999
                valid_inputs = (df_it[ff_col] != -999) & (df_it[b_col] != -999)
                valid_data = (df_it['IH'] != -999) & (df_it['pr_PRW'] != -999) & (df_it['p_PRW'] != -999) & (df_it['p_ETP'] != -999)
                
                valid_mask = valid_inputs & valid_data
                invalid_mask = ~valid_mask
                
                # 2. Damage Calculation (d)
                df_it.loc[valid_mask, d_col] = (df_it.loc[valid_mask, b_col] <= df_it.loc[valid_mask, ff_col]).astype(float)
                df_it.loc[invalid_mask, d_col] = -999.0
                
                damaged_mask = (df_it[d_col] == 1.0)
                no_damage_mask = (df_it[d_col] == 0.0)

                # 3. Extension (e) and Cost (c) Calculation per floor
                if damaged_mask.any():
                    # Determine perimeter source: BP for basements (n < 0), GP for others
                    peri_col = 'BP' if n < 0 else 'GP'
                    heigh_col = 'BH' if n < 0 else 'IH'
                    
                    # Logic: (Perimeter * Internal Height)
                    df_it.loc[damaged_mask, e_col] = (df_it.loc[damaged_mask, peri_col]) * df_it.loc[damaged_mask, heigh_col]
                    
                    # PRW Cost = (pr_PRW * e_PRW) + (p_PRW * e_PRW)
                    df_it.loc[damaged_mask, c_col] = (df_it.loc[damaged_mask, 'pr_PRW'] * df_it.loc[damaged_mask, e_col]) + \
                                                    (df_it.loc[damaged_mask, 'p_PRW'] * df_it.loc[damaged_mask, e_col])
                    
                    # ITP Cost = p_ETP * e_PRW
                    df_it.loc[damaged_mask, c_itp_col] = df_it.loc[damaged_mask, 'p_ETP'] * df_it.loc[damaged_mask, e_col]

                # 4. Flags and Zeroes
                if no_damage_mask.any():
                    df_it.loc[no_damage_mask, [e_col, c_col, c_itp_col]] = 0.0
                    
                df_it.loc[invalid_mask, [e_col, c_col, c_itp_col]] = -999.0
            
            # e_EXF / c_ETP
            valid_mask = (df_it['EP'] != -999) & (df_it['he'] != -999) & (df_it['p_ETP'] != -999)
            invalid_mask = ~valid_mask
            building_roof_elev = df_it.loc[valid_mask, 'GL'] + (df_it.loc[valid_mask, 'NF'] * df_it.loc[valid_mask, 'IH'])
            top_elev = np.minimum(df_it.loc[valid_mask, 'he'], building_roof_elev)
            df_it.loc[valid_mask, 'e_EXF'] = df_it.loc[valid_mask, 'EP'] * top_elev
            df_it.loc[valid_mask, 'c_ETP'] = df_it.loc[valid_mask, 'p_ETP'] * df_it.loc[valid_mask, 'e_EXF']
            df_it.loc[invalid_mask, ['e_EXF', 'c_ETP']] = -999.0
            
            valid_mask = (df_it['EP'] != -999) & (df_it['he'] != -999) & (df_it['p_ETP'] != -999) & (df_it['GL'] != -999) & (df_it['IH'] != -999)
            invalid_mask = ~valid_mask
            building_roof_elev = df_it.loc[valid_mask, 'GL'] + (df_it.loc[valid_mask, 'NF'] * df_it.loc[valid_mask, 'IH'])
            top_elev = np.minimum(df_it.loc[valid_mask, 'he'], building_roof_elev)
            flooded_height = np.maximum(top_elev - df_it.loc[valid_mask, 'GL'], 0.0)
            df_it.loc[valid_mask, 'e_EXF'] = df_it.loc[valid_mask, 'EP'] * flooded_height
            df_it.loc[valid_mask, 'c_ETP'] = df_it.loc[valid_mask, 'p_ETP'] * df_it.loc[valid_mask, 'e_EXF']
            df_it.loc[invalid_mask, ['e_EXF', 'c_ETP']] = -999.0
            # df_it['c_ETP'].min(), df_it['e_EXF'].max()
            ## -- OTHERS --
            # Check uniques RPs in df_ir
            #unique_rp = df_it["RP"].unique()
                
            return df_it
        
        except RuntimeError as e:
            if "updating stopped, endless loop" in str(e):
                continue
            else:
                raise e
            
    # FALLBACK: Execute only if all attempts within max_retries hit the RuntimeError
    df_failed = dataset.copy()
    df_failed['SN'] = sn
    cols_to_fill = [c for c in df_failed.columns if c != 'SN']
    df_failed.loc[:, cols_to_fill] = -999
    
    return df_failed
#endregion
# region B.2.3 Monte-Carlo
#test = it_Monte_Carlo(1,dataset,fm_df,fo_data,col_name_dic
# region B.2.3.1 MC Configuration
# General Monte Carlo Configuration
CORES = 16
N_SIMULATIONS = CORES * 625 # (CORES * 625) = 10_000
BATCH_SIZE = CORES * 8  # Process 128 simulations at a time
BASE_FILE_NAME = "MC_Part"
#endregion 
# region B.2.3.2 MC Execution 
total_start_time = time.time()
print(f"ESTIMATED TIME TO COMPLETE ALL SIMULATIONS: {(1.8762*N_SIMULATIONS)/3600:.2f} hours")
# Split simulations into chunks
all_sns = list(range(1, N_SIMULATIONS + 1))
batches = [all_sns[i:i + BATCH_SIZE] for i in range(0, len(all_sns), BATCH_SIZE)]

if not os.path.exists(PATHS['dataset_mc_parts']):
    os.makedirs(PATHS['dataset_mc_parts'])

#counter = 0
for idx, batch_sns in enumerate(batches):
    
    print(f"\n>>> PROCESSING BATCH {idx + 1}/{len(batches)} (Simulations {batch_sns[0]} to {batch_sns[-1]})")
    #counter +=1
    #if counter == 2:
    #    break
    
    # Check if part already exist to avoid repeating it
    batch_path = os.path.join(PATHS['dataset_mc_parts'], f"{BASE_FILE_NAME}_{idx+1}.parquet")
    if os.path.exists(batch_path):
        print(f"Part {idx+1} already exists. Skipping...")
        continue
    
    ## Execute batch in parallel
    sn_series = pd.Series(batch_sns)
    batch_results = sn_series.parallel_apply(
        it_Monte_Carlo, 
        dataset=dataset, 
        fm_df=fm_df, 
        fo_data=fo_data,
        col_name_dic=col_name_dic,
    )
    batch_df = pd.concat(batch_results.values, ignore_index=True) # Process Batch
    cols = ['SN'] + [c for c in batch_df.columns if c != 'SN'] # Reorder columns
    batch_df = batch_df[cols]
    
    ## Clean
    batch_df.replace(-999, np.nan, inplace=True) # Clean -999
    
    # Save part downcasted to save space
    batch_df = batch_df.astype('float32')
    batch_df.to_parquet(batch_path, engine='pyarrow', compression='snappy')
    
    print(f"--- Batch {idx + 1} saved to: {batch_path}")
del batch_df
del batch_results
gc.collect()
total_execution_time = time.time() - total_start_time
print(f"\n{'='*30}")
print(f"TOTAL PROCESS COMPLETED IN: {total_execution_time:.2f} seconds")
print(f"AVERAGE TIME PER SIMULATION: {(total_execution_time/N_SIMULATIONS):.4f} seconds")
#endregion 
#endregion 
# region B.2.4 Cost agregation
# region B.2.4.1 Helping functions
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

def get_floors_range(df):
    # Number of floors
    max_nf = int(df['NF'].max())
    max_bf = int(df['BF'].max())
    floor_range = range(-max_bf, max_nf + 1)
    basement_floors = range(-max_bf, 0)
    above_ground_floors = range(0, max_nf + 1)
    return floor_range, basement_floors, above_ground_floors

def aggregate_costs_contents(df, CTEs_unq, basement_floors, above_ground_floors):
    """
    Performs cost aggregation on a single dataframe part.
    Reverts independent calculation of _bf from _b and _f to avoid implicit duplication 
    if floor lists overlap or special conditions (like 'VEH') restrict individual ranges.
    """
    # 0. Clean potential duplicate columns from sequential processing
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    new_cols = {}
    
    # 1. Fill NaNs for cost columns in this part
    cost_cols = [c for c in df.columns if c.startswith('c_')]
    df[cost_cols] = df[cost_cols].fillna(0)
    
    # 2. Deduplicate floor ranges to ensure no implicit double summation
    basement_unq = sorted(list(set(basement_floors)))
    above_ground_unq = sorted(list(set(above_ground_floors)))
    all_floors_unq = sorted(list(set(basement_unq + above_ground_unq)))
    
    # 3. Helper
    def calc_max_cost(cols_n, cols_p, key_name):
        """Helper to multiply item counts (n) by item prices (p) considering broadcast shapes."""
        if not cols_n:
            return pd.Series(0, index=df.index)
        if not cols_p and f"p_{key_name}" in df.columns:
            cols_p = [f"p_{key_name}"]
            
        if len(cols_n) == len(cols_p):
            return pd.Series((df[cols_n].fillna(0).values * df[cols_p].fillna(0).values).sum(axis=1), index=df.index)
        elif len(cols_p) == 1:
            return df[cols_n].fillna(0).mul(df[cols_p[0]].fillna(0), axis=0).sum(axis=1)
        return pd.Series(0, index=df.index)
    
    # 4. Vectorized main loop per key
    for key in CTEs_unq:
        # A. Identify specific columns present in current df
        # Basement (f<0)
        b_cols_c = [f"c_{key}_{f}" for f in basement_unq if f"c_{key}_{f}" in df.columns]
        b_cols_n = [f"n_{key}_{f}" for f in basement_unq if f"n_{key}_{f}" in df.columns]
        b_cols_p = [f"p_{key}_{f}" for f in basement_unq if f"p_{key}_{f}" in df.columns]
        # Floor current (f>=0)
        current_f_range = [0] if key == 'VEH' else above_ground_floors
        f_cols_c = [f"c_{key}_{f}" for f in current_f_range if f"c_{key}_{f}" in df.columns]
        f_cols_n = [f"n_{key}_{f}" for f in current_f_range if f"n_{key}_{f}" in df.columns]
        f_cols_p = [f"p_{key}_{f}" for f in current_f_range if f"p_{key}_{f}" in df.columns]
        # All floors (f any)
        all_f_range = list(basement_floors) + list(above_ground_floors)
        bf_cols_c = [f"c_{key}_{f}" for f in all_floors_unq if f"c_{key}_{f}" in df.columns]
        bf_cols_n = [f"n_{key}_{f}" for f in all_floors_unq if f"n_{key}_{f}" in df.columns]
        bf_cols_p = [f"p_{key}_{f}" for f in all_floors_unq if f"p_{key}_{f}" in df.columns]
        # Specific ground floor (f=0)
        c_0, n_0, p_0 = f"c_{key}_0", f"n_{key}_0", f"p_{key}_0" # Not changed
        
        # B. Calculate sums
        # Basement (f < 0)
        sum_b = df[b_cols_c].sum(axis=1) if b_cols_c else pd.Series(0, index=df.index)
        sum_bm = calc_max_cost(b_cols_n, b_cols_p, key)
        # Floors (f >= 0)
        sum_f = df[f_cols_c].sum(axis=1) if f_cols_c else pd.Series(0, index=df.index)
        sum_fm = calc_max_cost(f_cols_n, f_cols_p, key)
        # Total (f any)
        sum_bf = df[bf_cols_c].sum(axis=1) if bf_cols_c else pd.Series(0, index=df.index)
        sum_bfm = calc_max_cost(bf_cols_n, bf_cols_p, key)
        # Ground Floor (f=0)
        sum_0 = df[c_0] if c_0 in df.columns else pd.Series(0, index=df.index)
        sum_0m = calc_max_cost([n_0], [p_0], key) if n_0 in df.columns and p_0 else pd.Series(0, index=df.index)

        # C. Apply HU Multiplier
        new_cols[f"c_{key}_b"] = sum_b
        new_cols[f"c_{key}_f"] = sum_f
        new_cols[f"c_{key}_bf"] = sum_bf
        new_cols[f"c_{key}_0"] = sum_0
        
        new_cols[f"c_hu{key}_b"] = sum_b * df['HU']
        new_cols[f"c_hu{key}_f"] = sum_f * df['HU']
        new_cols[f"c_hu{key}_bf"] = sum_bf * df['HU']
        new_cols[f"c_hu{key}_bm"] = sum_bm * df['HU']
        new_cols[f"c_hu{key}_fm"] = sum_fm * df['HU']
        new_cols[f"c_hu{key}_bfm"] = sum_bfm * df['HU']
        new_cols[f"c_hu{key}_0"] = sum_0 * df['HU']
        new_cols[f"c_hu{key}_0m"] = sum_0m * df['HU']

        # D. Ratios
        new_cols[f"c_hu{key}_bpct"] = new_cols[f"c_hu{key}_b"].div(new_cols[f"c_hu{key}_bm"]).replace([np.inf, -np.inf], 0).fillna(0)
        new_cols[f"c_hu{key}_fpct"] = new_cols[f"c_hu{key}_f"].div(new_cols[f"c_hu{key}_fm"]).replace([np.inf, -np.inf], 0).fillna(0)
        new_cols[f"c_hu{key}_bfpct"] = new_cols[f"c_hu{key}_bf"].div(new_cols[f"c_hu{key}_bfm"]).replace([np.inf, -np.inf], 0).fillna(0)
        new_cols[f"c_hu{key}_0pct"] = new_cols[f"c_hu{key}_0"].div(new_cols[f"c_hu{key}_0m"]).replace([np.inf, -np.inf], 0).fillna(0)
        
        for pct_suffix in ["bpct", "fpct", "bfpct", "0pct"]:
            pct_max = new_cols[f"c_hu{key}_{pct_suffix}"].max()
            if round(pct_max, 3) > 1:
                print(f"Warning: c_hu{key}_{pct_suffix} max value is {pct_max}, which is > 1")
        
        # E. Expected Costs
        new_cols[f"c_e{key}_b"] = new_cols[f"c_hu{key}_b"] / df['RP']
        new_cols[f"c_e{key}_f"] = new_cols[f"c_hu{key}_f"] / df['RP']
        new_cols[f"c_e{key}_bf"] = new_cols[f"c_hu{key}_bf"] / df['RP']
        new_cols[f"c_e{key}_0"] = new_cols[f"c_hu{key}_0"] / df['RP']

    # 3. Concatenate new columns
    existing_cols = [c for c in new_cols.keys() if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # 4. Global Aggregations (Across all keys)
    raw_b_cols_keys = [f"c_{k}_b" for k in CTEs_unq]
    raw_f_cols_keys = [f"c_{k}_f" for k in CTEs_unq]
    raw_bf_cols_keys = [f"c_{k}_bf" for k in CTEs_unq]
    raw_zero_cols_keys = [f"c_{k}_0" for k in CTEs_unq]
    
    b_cols_keys = [f"c_hu{k}_b" for k in CTEs_unq]
    f_cols_keys = [f"c_hu{k}_f" for k in CTEs_unq]
    bf_cols_keys = [f"c_hu{k}_bf" for k in CTEs_unq]
    zero_cols_keys = [f"c_hu{k}_0" for k in CTEs_unq]
    bm_cols_keys = [f"c_hu{k}_bm" for k in CTEs_unq]
    fm_cols_keys = [f"c_hu{k}_fm" for k in CTEs_unq]
    bfm_cols_keys = [f"c_hu{k}_bfm" for k in CTEs_unq]
    zero_m_cols_keys = [f"c_hu{k}_0m" for k in CTEs_unq]
    eb_cols_keys = [f"c_e{k}_b" for k in CTEs_unq]
    ef_cols_keys = [f"c_e{k}_f" for k in CTEs_unq]
    ebf_cols_keys = [f"c_e{k}_bf" for k in CTEs_unq]
    zero_e_cols_keys = [f"c_e{k}_0" for k in CTEs_unq]
    
    # Clean potential duplicate global columns before assigning
    global_cols = [
        'c_CTEs_b', 'c_CTEs_f', 'c_CTEs_bf', 'c_CTEs_0',
        'c_huCTEs_b', 'c_huCTEs_f', 'c_huCTEs_bf', 'c_huCTEs_0',
        'c_huCTEs_bm', 'c_huCTEs_fm', 'c_huCTEs_bfm', 'c_huCTEs_0m',
        'c_huCTEs_bpct', 'c_huCTEs_fpct', 'c_huCTEs_bfpct', 'c_huCTEs_0pct',
        'c_eCTEs_b', 'c_eCTEs_f', 'c_eCTEs_bf', 'c_eCTEs_0'
    ]
    existing_globals = [c for c in global_cols if c in df.columns]
    if existing_globals:
        df = df.drop(columns=existing_globals)
        
    # 5. Sums
    df['c_CTEs_b'] = df[raw_b_cols_keys].sum(axis=1)
    df['c_CTEs_f'] = df[raw_f_cols_keys].sum(axis=1)
    df['c_CTEs_bf'] = df[raw_bf_cols_keys].sum(axis=1)
    df['c_CTEs_0'] = df[raw_zero_cols_keys].sum(axis=1)
    
    df['c_huCTEs_b'] = df[b_cols_keys].sum(axis=1)
    df['c_huCTEs_f'] = df[f_cols_keys].sum(axis=1)
    df['c_huCTEs_bf'] = df[bf_cols_keys].sum(axis=1)
    df['c_huCTEs_0'] = df[zero_cols_keys].sum(axis=1)
    df['c_huCTEs_bm'] = df[bm_cols_keys].sum(axis=1)
    df['c_huCTEs_fm'] = df[fm_cols_keys].sum(axis=1)
    df['c_huCTEs_bfm'] = df[bfm_cols_keys].sum(axis=1)
    df['c_huCTEs_0m'] = df[zero_m_cols_keys].sum(axis=1)

    # 6. Ratios
    df['c_huCTEs_bpct'] = df['c_huCTEs_b'].div(df['c_huCTEs_bm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_huCTEs_fpct'] = df['c_huCTEs_f'].div(df['c_huCTEs_fm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_huCTEs_bfpct'] = df['c_huCTEs_bf'].div(df['c_huCTEs_bfm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_huCTEs_0pct'] = df['c_huCTEs_0'].div(df['c_huCTEs_0m']).replace([np.inf, -np.inf], 0).fillna(0)

    # Check max values for global aggregation ratio columns
    for global_pct in ['c_huCTEs_bpct', 'c_huCTEs_fpct', 'c_huCTEs_bfpct', 'c_huCTEs_0pct']:
        global_max = df[global_pct].max()
        if round(global_max, 3) > 1:
            print(f"Warning: {global_pct} max value is {global_max}, which is > 1")
    
    # 7. Expected
    df['c_eCTEs_b'] = df[eb_cols_keys].sum(axis=1)
    df['c_eCTEs_f'] = df[ef_cols_keys].sum(axis=1)
    df['c_eCTEs_bf'] = df[ebf_cols_keys].sum(axis=1)
    df['c_eCTEs_0'] = df[zero_e_cols_keys].sum(axis=1)

    return df

def aggregate_costs_continents(df, CTIs_unq, basement_floors, above_ground_floors):
    """
    Performs cost aggregation for continental items on a single dataframe part.
    """
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    new_cols = {}
    
    # 0. Deduplicate floor ranges to ensure no implicit double summation
    basement_unq = sorted(list(set(basement_floors)))
    above_ground_unq = sorted(list(set(above_ground_floors)))
    all_floors_unq = sorted(list(set(basement_unq + above_ground_unq)))

    # 1. Fill columns and eliminate invalidity flags
    cost_cols = [c for c in df.columns if c.startswith('c_')]
    count_cols = [c for c in df.columns if c.startswith('n_')]
    price_cols = [c for c in df.columns if c.startswith('p_')]
    geom_cols = [c for c in ['BP', 'BH', 'BA', 'GP', 'IH', 'GA', 'EP', 'GL', 'NF', 'e_PUM', 'pr_PRW'] if c in df.columns]
    
    clean_cols = cost_cols + count_cols + price_cols + geom_cols
    for c in clean_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='ignore').replace([-999, -999.0], 0.0).fillna(0.0)
    
    # 2. Vectorized loop per key
    for key in CTIs_unq:
        # key = 'ETP'
        # A. Identify columns
        # Basement current and max
        b_cols_c = [f"c_{key}_{f}" for f in basement_unq if f"c_{key}_{f}" in df.columns]
        bm_cols_n = [f"n_{key}_{f}" for f in basement_unq if f"n_{key}_{f}" in df.columns]
        bm_cols_p = [f"p_{key}_{f}" for f in basement_unq if f"p_{key}_{f}" in df.columns]
        if not bm_cols_p and f"p_{key}" in df.columns:
            bm_cols_p = [f"p_{key}"]
        
        # Floor current and max
        f_cols_c = [f"c_{key}_{f}" for f in above_ground_unq if f"c_{key}_{f}" in df.columns]
        fm_cols_n = [f"n_{key}_{f}" for f in above_ground_unq if f"n_{key}_{f}" in df.columns]
        fm_cols_p = [f"p_{key}_{f}" for f in above_ground_unq if f"p_{key}_{f}" in df.columns]
        
        # General columns (no floor)
        if not f_cols_c and f"c_{key}" in df.columns:
            f_cols_c = [f"c_{key}"]
        if not fm_cols_n and f"n_{key}" in df.columns:
            fm_cols_n = [f"n_{key}"]
        if not fm_cols_p and f"p_{key}" in df.columns:
            fm_cols_p = [f"p_{key}"]
            
        # All floors merged logic
        bf_cols_c = [f"c_{key}_{f}" for f in all_floors_unq if f"c_{key}_{f}" in df.columns]
        
        # Specific ground floor (f=0)
        n_0 = f"n_{key}_0" if f"n_{key}_0" in df.columns else (f"n_{key}" if f"n_{key}" in df.columns else None)
        p_0 = f"p_{key}_0" if f"p_{key}_0" in df.columns else (f"p_{key}" if f"p_{key}" in df.columns else None)
        
        # B. Calculate sums
        # Basement (Floors < 0)
        sum_b = df[b_cols_c].sum(axis=1) if b_cols_c else pd.Series(0.0, index=df.index)
        sum_bm = pd.Series(0.0, index=df.index)
        if key == 'CLE':
            sum_bm = ((df['BP'] * df['BH']) + df['BA']) * df['p_CLE']
        elif key == 'PUM':
            pass
        elif key == 'SOI':
            sum_bm = df['BA'] * df['p_SOI']
        elif key == 'PRW':
            sum_bm = df['BP'] * df['BH'] * (df['pr_PRW'] + df['p_PRW'])
        elif key == 'ITP':
            sum_bm = df['BP'] * df['BH'] * df['p_ETP']
        elif key == 'SKT': 
            if bm_cols_n and bm_cols_p:
                sum_bm = df['BP'] * (df[bm_cols_n].clip(0, 20).sum(axis=1) / 20.0) * df[bm_cols_p[0]] 
        elif key not in ['ETP', 'ELS', 'WND']:
            if len(bm_cols_n) == len(bm_cols_p) and bm_cols_n:
                sum_bm = (df[bm_cols_n].values * df[bm_cols_p].values).sum(axis=1)
            elif len(bm_cols_p) == 1 and bm_cols_n:
                sum_bm = df[bm_cols_n].mul(df[bm_cols_p[0]], axis=0).sum(axis=1)
        #print("bmax", sum_b.max(), sum_bm.max(), key)
        #print("bmin", sum_b.min(), sum_bm.min(), key)
        
        # Floors (Floors >= 0)
        sum_f = df[f_cols_c].sum(axis=1) if f_cols_c else pd.Series(0.0, index=df.index)
        sum_fm = pd.Series(0.0, index=df.index)
        if key == 'CLE':
            sum_fm = ((df['GP'] * df['IH']) + df['GA']) * df['p_CLE'] * df['NF']
        elif key == 'SOI':
            sum_fm = df['GA'] * df['p_SOI'] * df['NF']
        elif key == 'PRW':
            sum_fm = df['GP'] * df['IH'] * (df['pr_PRW'] + df['p_PRW']) * df['NF']
        elif key == 'ITP':
            sum_fm = df['GP'] * df['IH'] * df['p_ETP'] * df['NF']
        elif key == 'ETP':
            sum_fm = df['EP'] * (df['NF'] * df['IH']) * df['p_ETP']
        elif key == 'PUM':
            sum_fm = df['e_PUM'] * df['p_PUM']
        elif key == 'SKT':
            if fm_cols_n and fm_cols_p:
                sum_fm = df['GP'] * (df[fm_cols_n].clip(0, 20).sum(axis=1) / 20.0) * df[fm_cols_p[0]]
        else:
            if len(fm_cols_n) == len(fm_cols_p) and fm_cols_n:
                sum_fm = (df[fm_cols_n].values * df[fm_cols_p].values).sum(axis=1)
            elif len(fm_cols_p) == 1 and fm_cols_n:
                sum_fm = df[fm_cols_n].mul(df[fm_cols_p[0]], axis=0).sum(axis=1)
        #print("fmax", sum_f.max(), sum_fm.max(), key)
        #print("fmin", sum_f.min(), sum_fm.min(), key)
        
        # Total b + f
        sum_bf = df[bf_cols_c].sum(axis=1) if bf_cols_c else pd.Series(0.0, index=df.index)
        sum_bfm = sum_bm + sum_fm
        #print("bfmax", sum_bf.max(), sum_bfm.max(), key)
        #print("bfmin", sum_bf.min(), sum_bfm.min(), key)
        
        # Ground Floor (f=0)
        sum_0 = pd.Series(0.0, index=df.index)
        if key != 'PUM':
            if f"c_{key}_0" in df.columns: sum_0 = df[f"c_{key}_0"]
            elif f"c_{key}" in df.columns: sum_0 = df[f"c_{key}"]
        elif key == 'PUM':
            if f"c_{key}_0" in df.columns: sum_0 = df[f"c_{key}_0"]
            elif f"c_{key}" in df.columns: sum_0 = df[f"c_{key}"]
            
        sum_0m = pd.Series(0.0, index=df.index)
        if key == 'CLE':
            sum_0m = ((df['GP'] * df['IH']) + df['GA']) * df['p_CLE']
        elif key == 'SOI':
            sum_0m = df['GA'] * df['p_SOI']
        elif key == 'PRW':
            sum_0m = df['GP'] * df['IH'] * (df['pr_PRW'] + df['p_PRW'])
        elif key == 'ITP':
            sum_0m = df['GP'] * df['IH'] * df['p_ETP']
        elif key == 'ETP':
            sum_0m = df['EP'] * (df['NF'] * df['IH']) * df['p_ETP']
        elif key == 'PUM':
            sum_0m = df['e_PUM'] * df['p_PUM']
        elif key == 'SKT': 
            if n_0 and p_0:
                sum_0m = df['GP'] * (df[n_0].clip(0, 20) / 20.0) * df[p_0]
        elif n_0 and p_0:
            sum_0m = df[n_0] * df[p_0]
        #print("0max", sum_0.max(), sum_0m.max(), key)
        #print("0min", sum_0.min(), sum_0m.min(), key)
        
        # C. Apply Multipliers and Ratios
        # Raw
        new_cols[f"c_{key}_b"] = sum_b
        new_cols[f"c_{key}_f"] = sum_f
        new_cols[f"c_{key}_bf"] = sum_bf
        new_cols[f"c_{key}_0"] = sum_0
        
        # HU multiplier
        new_cols[f"c_hu{key}_b"] = sum_b * df['HU']
        new_cols[f"c_hu{key}_f"] = sum_f * df['HU']
        new_cols[f"c_hu{key}_bf"] = sum_bf * df['HU']
        new_cols[f"c_hu{key}_bm"] = sum_bm * df['HU']
        new_cols[f"c_hu{key}_fm"] = sum_fm * df['HU']
        new_cols[f"c_hu{key}_bfm"] = sum_bfm * df['HU']
        new_cols[f"c_hu{key}_0"] = sum_0 * df['HU']
        new_cols[f"c_hu{key}_0m"] = sum_0m * df['HU']
        
        # Ratios
        new_cols[f"c_hu{key}_bpct"] = new_cols[f"c_hu{key}_b"].div(new_cols[f"c_hu{key}_bm"]).replace([np.inf, -np.inf], 0).fillna(0)
        new_cols[f"c_hu{key}_fpct"] = new_cols[f"c_hu{key}_f"].div(new_cols[f"c_hu{key}_fm"]).replace([np.inf, -np.inf], 0).fillna(0)
        new_cols[f"c_hu{key}_bfpct"] = new_cols[f"c_hu{key}_bf"].div(new_cols[f"c_hu{key}_bfm"]).replace([np.inf, -np.inf], 0).fillna(0)
        new_cols[f"c_hu{key}_0pct"] = new_cols[f"c_hu{key}_0"].div(new_cols[f"c_hu{key}_0m"]).replace([np.inf, -np.inf], 0).fillna(0)
        
        for pct_suffix in ["bpct", "fpct", "bfpct", "0pct"]:
            pct_max = new_cols[f"c_hu{key}_{pct_suffix}"].max()
            pct_min = new_cols[f"c_hu{key}_{pct_suffix}"].min()
            if round(pct_max, 3) > 1:
                print(f"Warning: c_hu{key}_{pct_suffix} max value is {pct_max}, which is > 1")
            if round(pct_min, 3) < 0:
                print(f"Warning: c_hu{key}_{pct_suffix} min value is {pct_min}, which is < 0")
                
        # Expected Costs
        new_cols[f"c_e{key}_b"] = new_cols[f"c_hu{key}_b"] / df['RP']
        new_cols[f"c_e{key}_f"] = new_cols[f"c_hu{key}_f"] / df['RP']
        new_cols[f"c_e{key}_bf"] = new_cols[f"c_hu{key}_bf"] / df['RP']
        new_cols[f"c_e{key}_0"] = new_cols[f"c_hu{key}_0"] / df['RP']

    # 3. Concatenate
    existing_cols = [c for c in new_cols.keys() if c in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # 4. Final Aggregations
    raw_b_keys = [f"c_{k}_b" for k in CTIs_unq]
    raw_f_keys = [f"c_{k}_f" for k in CTIs_unq]
    raw_bf_keys = [f"c_{k}_bf" for k in CTIs_unq]
    raw_z_keys = [f"c_{k}_0" for k in CTIs_unq]
    
    b_keys = [f"c_hu{k}_b" for k in CTIs_unq]
    f_keys = [f"c_hu{k}_f" for k in CTIs_unq]
    bf_keys = [f"c_hu{k}_bf" for k in CTIs_unq]
    z_keys = [f"c_hu{k}_0" for k in CTIs_unq]
    bm_keys = [f"c_hu{k}_bm" for k in CTIs_unq]
    fm_keys = [f"c_hu{k}_fm" for k in CTIs_unq]
    bfm_keys = [f"c_hu{k}_bfm" for k in CTIs_unq]
    zm_keys = [f"c_hu{k}_0m" for k in CTIs_unq]
    eb_keys = [f"c_e{k}_b" for k in CTIs_unq]
    ef_keys = [f"c_e{k}_f" for k in CTIs_unq]
    ebf_keys = [f"c_e{k}_bf" for k in CTIs_unq]
    ez_keys = [f"c_e{k}_0" for k in CTIs_unq]
    
    global_cols = [
        'c_CTIs_b', 'c_CTIs_f', 'c_CTIs_bf', 'c_CTIs_0',
        'c_huCTIs_b', 'c_huCTIs_f', 'c_huCTIs_bf', 'c_huCTIs_0',
        'c_huCTIs_bm', 'c_huCTIs_fm', 'c_huCTIs_bfm', 'c_huCTIs_0m',
        'c_huCTIs_bpct', 'c_huCTIs_fpct', 'c_huCTIs_bfpct', 'c_huCTIs_0pct',
        'c_eCTIs_b', 'c_eCTIs_f', 'c_eCTIs_bf', 'c_eCTIs_0'
    ]
    existing_globals = [c for c in global_cols if c in df.columns]
    if existing_globals:
        df = df.drop(columns=existing_globals)
    
    # 5 Calc sums
    df['c_CTIs_b'] = df[raw_b_keys].sum(axis=1)
    df['c_CTIs_f'] = df[raw_f_keys].sum(axis=1)
    df['c_CTIs_bf'] = df[raw_bf_keys].sum(axis=1)
    df['c_CTIs_0'] = df[raw_z_keys].sum(axis=1)
    
    df['c_huCTIs_b'] = df[b_keys].sum(axis=1)
    df['c_huCTIs_f'] = df[f_keys].sum(axis=1)
    df['c_huCTIs_bf'] = df[bf_keys].sum(axis=1)
    df['c_huCTIs_0'] = df[z_keys].sum(axis=1)
    
    df['c_huCTIs_bm'] = df[bm_keys].sum(axis=1)
    df['c_huCTIs_fm'] = df[fm_keys].sum(axis=1)
    df['c_huCTIs_bfm'] = df[bfm_keys].sum(axis=1)
    df['c_huCTIs_0m'] = df[zm_keys].sum(axis=1)
    
    # 6 Calc Ratios
    df['c_huCTIs_bpct'] = df['c_huCTIs_b'].div(df['c_huCTIs_bm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_huCTIs_fpct'] = df['c_huCTIs_f'].div(df['c_huCTIs_fm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_huCTIs_bfpct'] = df['c_huCTIs_bf'].div(df['c_huCTIs_bfm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_huCTIs_0pct'] = df['c_huCTIs_0'].div(df['c_huCTIs_0m']).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Check max values for global aggregation ratio columns
    for global_pct in ['c_huCTIs_bpct', 'c_huCTIs_fpct', 'c_huCTIs_bfpct', 'c_huCTIs_0pct']:
        global_max = df[global_pct].max()
        global_min = df[global_pct].min()
        if round(global_max, 3) > 1:
            print(f"Warning: {global_pct} max value is {global_max}, which is > 1")
        if round(global_min, 3) < 0:
            print(f"Warning: {global_pct} min value is {global_min}, which is < 0")
            
    # 7 Total Expected Costs
    df['c_eCTIs_b'] = df[eb_keys].sum(axis=1)
    df['c_eCTIs_f'] = df[ef_keys].sum(axis=1)
    df['c_eCTIs_bf'] = df[ebf_keys].sum(axis=1)
    df['c_eCTIs_0'] = df[ez_keys].sum(axis=1)

    return df

def aggregate_total_building_costs(df):
    """
    Aggregates Content (CTEs) and Continent (CTIs) costs into total building costs.
    """
    # 0. Raw costs
    df['c_b'] = df['c_CTEs_b'] + df['c_CTIs_b']
    df['c_f'] = df['c_CTEs_f'] + df['c_CTIs_f']
    df['c_bf'] = df['c_CTEs_bf'] + df['c_CTIs_bf']
    df['c_0'] = df['c_CTEs_0'] + df['c_CTIs_0']
    
    # 1. HU adjusted costs
    df['c_hu_b'] = df['c_huCTEs_b'] + df['c_huCTIs_b']
    df['c_hu_f'] = df['c_huCTEs_f'] + df['c_huCTIs_f']
    df['c_hu_bf'] = df['c_huCTEs_bf'] + df['c_huCTIs_bf']
    df['c_hu_0'] = df['c_huCTEs_0'] + df['c_huCTIs_0']

    # 2. Maximum potential costs
    df['c_hu_bm'] = df['c_huCTEs_bm'] + df['c_huCTIs_bm']
    df['c_hu_fm'] = df['c_huCTEs_fm'] + df['c_huCTIs_fm']
    df['c_hu_bfm'] = df['c_huCTEs_bfm'] + df['c_huCTIs_bfm']
    df['c_hu_0m'] = df['c_huCTEs_0m'] + df['c_huCTIs_0m']

    # 3. Combined ratios (pct)
    df['c_hu_bpct'] = df['c_hu_b'].div(df['c_hu_bm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_hu_fpct'] = df['c_hu_f'].div(df['c_hu_fm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_hu_bfpct'] = df['c_hu_bf'].div(df['c_hu_bfm']).replace([np.inf, -np.inf], 0).fillna(0)
    df['c_hu_0pct'] = df['c_hu_0'].div(df['c_hu_0m']).replace([np.inf, -np.inf], 0).fillna(0)

    # 4. Expected annual costs
    df['c_e_b'] = df['c_eCTEs_b'] + df['c_eCTIs_b']
    df['c_e_f'] = df['c_eCTEs_f'] + df['c_eCTIs_f']
    df['c_e_bf'] = df['c_eCTEs_bf'] + df['c_eCTIs_bf']
    df['c_e_0'] = df['c_eCTEs_0'] + df['c_eCTIs_0']

    return df

def process_and_save_parts(results_dir, base_file_name, cte_keys, cti_keys, basement_floors, above_ground_floors):
    """
    Iterates through parquet files, applies aggregation functions, and overwrites the files.
    """
    file_list = sorted(glob.glob(os.path.join(results_dir, f"{base_file_name}_*.parquet")))
    
    if not file_list:
        print("No files found.")
        return

    for file_path in tqdm(file_list, desc="Processing and Updating Parquet Parts"):
        print(f"Processing {file_path}")
        # 1. Load
        print(f"Loading...")
        df = pd.read_parquet(file_path)
        
        # 2. Apply Aggregations
        # aggregate_costs_contents
        print(f"Processing CTE...")
        df = aggregate_costs_contents(df, cte_keys, basement_floors, above_ground_floors)
        
        # aggregate_costs_continents
        print(f"Processing CTI...")
        df = aggregate_costs_continents(df, cti_keys, basement_floors, above_ground_floors)
        
        # aggregate_total_building_costs
        print(f"Processing Total Cost...")
        df = aggregate_total_building_costs(df)
        
        # 3. Save (Overwrite)
        print(f"Re-Saving parquet...")
        df.to_parquet(file_path, index=False)
        
        # Cleanup memory
        del df
#endregion 
# region B.2.4.2 Update mc parts
temp_df = load_mc_part(os.path.join(PATHS['dataset_mc_parts'], f"{BASE_FILE_NAME}_1.parquet"))
unique_rp = temp_df["RP"].unique()
floor_range, basement_floors, above_ground_floors = get_floors_range(temp_df)
process_and_save_parts(
    results_dir=PATHS['dataset_mc_parts'],
    base_file_name=BASE_FILE_NAME,
    cte_keys=CTEs_unq,
    cti_keys=CTIs_unq,
    basement_floors=basement_floors,
    above_ground_floors=above_ground_floors
)
#endregion
#endregion
#endregion
#endregion

#################################################################################
#################################################################################
##### SECTION D: RESULTS: FIGURES AND TABLES
#################################################################################
#################################################################################

#region D RESULTS
# region D.1 Set global parameters
#output_dir = r'C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\2_Draft_Article\Figures'
output_dir = r'C:\Users\outal\GITHUB\InDepthPROFILE\data\processed'
#endregion
# region D.2 Helping functions
# region D.2.0 Old helping Functions
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    def parse_input(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except:
                return val
        return val

    def get_fm_rows(DC, sort_by=None):
        if isinstance(DC, list):
            fm_rows = fm_df[fm_df["DC"].isin(DC)].copy()
        else:
            fm_rows = fm_df[fm_df["DC"] == DC].copy()
            
        fm_rows['FP'] = fm_rows['FP'].apply(parse_input)
        
        if sort_by == "normal_dispersion":
            fm_rows = fm_rows.sort_values(by='FP', key=lambda x: x.str[1]).reset_index()
        elif sort_by == "BT":
            fm_rows = fm_rows.sort_values(by='BT').reset_index(drop=True)
        elif sort_by == "BP":
            fm_rows = fm_rows.sort_values(by='BP').reset_index(drop=True)
        
        return fm_rows

    def set_po_colors(po, cmap_name='RdYlBu_r', by=None):
        """
        Updates 'po' list with 'Color' keys.
        - If by is None: Colors assigned by position (OID).
        - If by is a string: Colors assigned by unique values of that key in 'Row'.
        """
        cmap = plt.get_cmap(cmap_name)
        n_objects = len(po)

        if by is None:
            # Case 1: Color by OID (Continuous/Gradient)
            for i, obj in enumerate(po):
                color = cmap(i / (n_objects - 1)) if n_objects > 1 else cmap(0)
                obj["Color"] = color
        else:
            # Case 2: Color by attribute (Categorical)
            # Extract unique values for the 'by' key from the nested 'Row'
            unique_values = sorted(list(set(obj["Row"][by] for obj in po)))
            n_unique = len(unique_values)
            
            # Create a mapping: {value: color}
            value_to_color = {
                val: cmap(i / (n_unique - 1)) if n_unique > 1 else cmap(0)
                for i, val in enumerate(unique_values)
            }
            
            # Assign colors to objects
            for obj in po:
                val = obj["Row"][by]
                obj["Color"] = value_to_color[val]

        return po

    def set_po_sizes(po, size_range, by=None):
        """
        Updates 'po' list with 'Size' keys.
        - If by is None: Sizes are a linear gradient from size_range[0] to [1] based on OID.
        - If by is a string: Sizes are mapped linearly based on the values of that key.
        - If size_range is a single float/int: All objects get that uniform size.
        """
        n_objects = len(po)

        s_min, s_max = size_range

        if by is None:
            # Case 1: Gradient by OID (position)
            for i, obj in enumerate(po):
                if n_objects > 1:
                    size = s_min + (i / (n_objects - 1)) * (s_max - s_min)
                else:
                    size = s_max
                obj["Size"] = size
        else:
            # Case 2: Map to a specific value (e.g., 'BP')
            values = np.array([obj["Row"].get(by, 0) for obj in po])
            v_min, v_max = values.min(), values.max()
            
            for obj in po:
                val = obj["Row"].get(by, 0)
                if v_max != v_min:
                    # Linear interpolation of the value into the size range
                    size = s_min + ((val - v_min) / (v_max - v_min)) * (s_max - s_min)
                else:
                    size = s_max
                obj["Size"] = size

        return po

    def set_po_xy(po, n_points=1000, Qmin=0.0001, Qmax=0.9999):
        """
        Calculates X and Y arrays for Continuous, Discrete, and Constant distributions.
        Updates each plot object and calculates global axis limits.
        """
        limits = {'xmin': float('inf'), 'xmax': float('-inf'), 'ymin': 0, 'ymax': 0}

        for obj in po:
            row = obj["Row"]
            fn_type = row['FN']
            params = row['FP']
            
            # 1. Handle Constant Value (Manual mapping)
            if fn_type == 'Constant':
                val = params[0]
                x = np.array([val, val])
                y = np.array([0, 1.0]) # Represented as a spike from 0 to 1

            # 2. Handle KDE (Continuous from raw data)
            elif fn_type == 'KDE':
                raw_data = np.array(fo_data[row['ODC']])
                fit_data = raw_data[np.isfinite(raw_data)]
                x = np.linspace(np.min(fit_data), np.max(fit_data), n_points)
                kde = stats.gaussian_kde(fit_data)
                y = kde(x)

            # 3. Handle Discrete Distributions (Scipy PMF)
            elif hasattr(stats, fn_type) and hasattr(getattr(stats, fn_type), 'pmf'):
                dist_obj = getattr(stats, fn_type)
                x_start = int(dist_obj.ppf(Qmin, *params))
                x_end = int(dist_obj.ppf(Qmax, *params))
                # Create integer steps
                x = np.arange(x_start, x_end + 1)
                y = dist_obj.pmf(x, *params)
                if fn_type in ['poisson', 'nbinom']:
                    x = x + 1
                    
            # 4. Handle Continuous Distributions (Scipy PDF)
            elif hasattr(stats, fn_type):
                dist_obj = getattr(stats, fn_type)
                x_start = dist_obj.ppf(Qmin, *params)
                x_end = dist_obj.ppf(Qmax, *params)
                x = np.linspace(x_start, x_end, n_points)
                y = dist_obj.pdf(x, *params)
            
            else:
                continue

            # Store data
            obj["X"], obj["Y"] = x, y
            
            # Calculate local and update global bounds
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            ymin, ymax = np.nanmin(y), np.nanmax(y)
            
            obj["Bounds"] = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}

            limits['xmin'] = min(limits['xmin'], xmin)
            limits['xmax'] = max(limits['xmax'], xmax)
            limits['ymin'] = min(limits['ymin'], ymin)
            limits['ymax'] = max(limits['ymax'], ymax)

        return po, limits

    def set_res_colors(res, cmap_name='viridis', by='RP'):
        """
        Updates 'res' list with 'Color' keys based on unique values of a specified key.
        Default is 'RP' sorted ascending.
        """
        unique_values = sorted(list(set(obj.get(by) for obj in res)))
        n_unique = len(unique_values)
        cmap = plt.get_cmap(cmap_name)
        
        value_to_color = {
            val: cmap(i / (n_unique - 1)) if n_unique > 1 else cmap(0)
            for i, val in enumerate(unique_values)
        }
        
        for obj in res:
            val = obj.get(by)
            obj["Color"] = value_to_color.get(val)
            
        return res

    def set_res_labels(res, label_name='c_hu', key = 'Label'):
        """
        Updates 'res' list with a static 'Label' key for all objects.
        """
        for obj in res:
            obj[key] = label_name
                
        return res

    def set_res_xy(res, full_df, by='RP', x_col='SN', y_col='c_hu', x_key = 'X', y_key = 'Y'):
        """
        Populates 'res' list with X and Y arrays using custom keys.
        """
        for obj in res:
            # Identify the value to filter by (e.g., the specific RP)
            filter_val = obj.get(by)
            
            # Slice the dataframe for the specific group
            sub_df = full_df[full_df[by] == filter_val].copy()
            
            # Assign as numpy arrays for compatibility with existing plot functions
            obj[x_key] = sub_df[x_col].values
            obj[y_key] = sub_df[y_col].values
                
        return res

    def set_po_labels(po, pf = 'n'):
        """
        Updates 'po' list with a 'Label' key by removing the 'pf_' prefix 
        from the 'DC' value found in 'Row'.
        """
        for obj in po:
            # Get the DC value (e.g., 'n_FAD')
            dc_value = obj["Row"].get("DC", "")
            
            # Remove 'n_' prefix if it exists
            if dc_value.startswith(f"{pf}_"):
                label = dc_value.replace(f"{pf}_", "", 1)
            else:
                label = dc_value
                
            obj["Label"] = label
            
        return po

    # 3. Plot data
    def plot_ax_pdf(ax, po, **kwargs):
        """
        Renders plot objects from the 'po' list onto the specified axis.
        Uses 'X', 'Y', and existing keys from each dictionary.
        """
        global_linewidth = kwargs.pop('linewidth', None)
        for obj in po:
            ax.plot(
                obj["X"], 
                obj["Y"], 
                color=obj.get("Color"),
                linewidth=obj.get("Size") or global_linewidth or 1.0,
                label=obj.get("Label"),
                **kwargs
            )

    def plot_ax_cdf(ax, po, **kwargs):
        """
        Renders the Cumulative Distribution Function (CDF) from the 'po' list.
        Calculates the cumulative sum of 'Y' normalized by the total sum.
        """
        global_linewidth = kwargs.pop('linewidth', None)
        
        for obj in po:
            x = obj["X"]
            y = obj["Y"]
            
            # Calculate Cumulative Sum
            # We normalize by the sum to ensure it ends at exactly 1.0
            y_cdf = np.cumsum(y)
            if y_cdf[-1] != 0:
                y_cdf = y_cdf / y_cdf[-1]
                
            lw = obj.get("Size") or global_linewidth or 1.0
            
            ax.plot(
                x, 
                y_cdf, 
                color=obj.get("Color"),
                linewidth=lw,
                label=obj.get("Label"),
                **kwargs
            )

    def plot_ax_dotcloud(ax, po, **kwargs):
        """
        Plots a 'cloud' of points based on the 'BP' (Building Probability) 
        value found in each object's Row data.
        """
        global_size = kwargs.pop('s', 1)
        
        for obj in po:
            # Random x-position between -0.05 and 0.05 to create the "cloud" effect (jitter)
            jitter = np.random.uniform(-0.05, 0.05)
            # y-position
            row = obj["Row"]
            bp_value = row.get("BP", 0)
            
            # Dots
            ax.scatter(
                jitter,
                bp_value,
                color=obj.get("Color"),
                s=obj.get("Size") or global_size or 1.0,
                **kwargs
            )

    def plot_ax_ridgeline(ax, po, overlap=0.7, bw_method=0.16, n_points=500, x_range=None, **kwargs):
        """
        Renders a ridgeline (joyplot) on a single axis using the distribution of 'Y' values 
        in each object of 'po'.
        """

        n_items = len(po)
        
        # Determine x-axis evaluation points
        if x_range is None:
            all_vals = np.concatenate([obj["Y"] for obj in po if len(obj["Y"]) > 0])
            x = np.linspace(np.min(all_vals), np.max(all_vals), n_points)
        else:
            x = np.linspace(x_range[0], x_range[1], n_points)

        for i, obj in enumerate(po):
            vals = obj["Y"]
            
            # Calculate offset (bottom-to-top)
            offset = (n_items - 1 - i) * overlap
            
            # KDE Calculation
            if len(vals) < 2:
                continue
                
            if np.ptp(vals) == 0:
                vals = vals + np.random.normal(0, 1e-9, size=vals.shape)
                
            kde = gaussian_kde(vals, bw_method=bw_method)
            y_eval = kde(x)
            
            # Normalization
            if y_eval.max() > 0:
                y_eval = y_eval / y_eval.max()
            
            # Plotting
            color = obj.get("Color", "blue")
            
            ax.fill_between(x, offset, y_eval + offset, 
                            color=color, 
                            alpha=kwargs.get("alpha", 0.7), 
                            zorder=i)
            
            ax.plot(x, y_eval + offset, 
                    color=kwargs.get("edgecolor", "white"), 
                    linewidth=kwargs.get("linewidth", 0.8), 
                    zorder=i)
        return ax

    def plot_ax_ridgeline_v(ax, po, overlap=0.7, bw_method=0.16, n_points=500, y_range=None, x_key = 'X', y_key = 'Y', **kwargs):
        """
        Renders a vertical ridgeline plot. 
        Y-axis represents the data values (Cost), X-axis represents the distributions per group (RP).
        """
        from scipy.stats import gaussian_kde

        n_items = len(po)
        
        # Determine y-axis evaluation points (the scale of the data values)
        if y_range is None:
            all_vals = np.concatenate([obj[y_key] for obj in po if len(obj[y_key]) > 0])
            y_eval_pts = np.linspace(np.min(all_vals), np.max(all_vals), n_points)
        else:
            y_eval_pts = np.linspace(y_range[0], y_range[1], n_points)

        for i, obj in enumerate(po):
            vals = obj[y_key]
            
            # Calculate horizontal offset (left-to-right)
            offset = i * overlap
            
            if len(vals) < 2:
                continue
                
            if np.ptp(vals) == 0:
                vals = vals + np.random.normal(0, 1e-9, size=vals.shape)
                
            kde = gaussian_kde(vals, bw_method=bw_method)
            x_kde = kde(y_eval_pts)
            
            # Normalization
            if x_kde.max() > 0:
                x_kde = x_kde / x_kde.max()
            
            color = obj.get("Color", "blue")
            
            # Use fill_betweenx for vertical orientation
            ax.fill_betweenx(y_eval_pts, offset, x_kde + offset, 
                            color=color, 
                            alpha=kwargs.get("alpha", 0.7), 
                            zorder=i)
            
            ax.plot(x_kde + offset, y_eval_pts, 
                    color=kwargs.get("edgecolor", "white"), 
                    linewidth=kwargs.get("linewidth", 0.8), 
                    zorder=i)

        return ax

    def plot_ax_rp_line(ax, res, overlap=0.7, q=0.5, y_key='Y', prepend_origin=False, **kwargs):
        """
        Plots a trend line of a specific quantile across groups.

        Parameters
        ----------
        prepend_origin : bool
            If True, adds (0,0) at the start.
            If False, uses only actual group positions (recommended for EP-based plots).
        """
        x_coords = []
        y_coords = []

        for i, obj in enumerate(res):
            vals = obj.get(y_key)
            if vals is not None and len(vals) > 0:
                current_x = (i + 1) * overlap
                x_coords.append(current_x)
                y_coords.append(np.quantile(vals, q))

        if prepend_origin:
            plot_x = [0] + x_coords
            plot_y = [0] + y_coords
        else:
            plot_x = x_coords
            plot_y = y_coords

        line = ax.plot(plot_x, plot_y, **kwargs)
        return line, x_coords

    def plot_ax_line(ax, res, x_key='X', y_key='Y', label_suffix='', **kwargs):
        """
        Plots individual lines for each group (e.g., RP) in the result list.
        """
        lines = []
        for obj in res:
            x = obj.get(x_key)
            y = obj.get(y_key)
            rp = obj.get('RP')
            color = obj.get('Color')
            
            if x is not None and y is not None:
                label = f"RP {rp} {label_suffix}".strip()
                line = ax.plot(x, y, label=label, color=color, **kwargs)
                lines.append(line)
                
        return lines

    def plot_ax_fill(ax, res, x_key_low='X_low', y_key_low='Y_low', x_key_high='X_high', y_key_high='Y_high', **kwargs):
        """
        Fills the area between two curves (e.g., Q05 and Q95) for each group in the result list.
        """
        collections = []
        
        # Default styling for the filled area if not provided in kwargs
        fill_kwargs = {
            'alpha': 0.2,
            'linewidth': 0,
            'zorder': 2
        }
        fill_kwargs.update(kwargs)

        for obj in res:
            x_low = obj.get(x_key_low)
            y_low = obj.get(y_key_low)
            x_high = obj.get(x_key_high)
            y_high = obj.get(y_key_high)
            color = obj.get('Color')

            # Lorenz curves for different quantiles have different X-coordinates due to reordering.
            # fill_between requires a single X-axis. We interpolate y_high onto x_low.
            if all(v is not None for v in [x_low, y_low, x_high, y_high]):
                y_high_interp = np.interp(x_low, x_high, y_high)
                
                fill_coll = ax.fill_between(
                    x_low, 
                    y_low, 
                    y_high_interp, 
                    color=color, 
                    **fill_kwargs
                )
                collections.append(fill_coll)
                
        return collections

    def plot_ax_lp3_quantile_line(ax, res, overlap=0.7, q=0.5, **kwargs):
        x_coords = [0]
        y_coords = [0]

        for i, obj in enumerate(res):
            if all(k in obj for k in ['Skew', 'Loc', 'Scale']):
                y_log = pearson3.ppf(q, obj['Skew'], loc=obj['Loc'], scale=obj['Scale'])
                y_val = 10 ** y_log

                current_x = (i + 1) * overlap
                x_coords.append(current_x)
                y_coords.append(y_val)

        line = ax.plot(x_coords, y_coords, **kwargs)
        return line, x_coords

    def plot_ax_violin_v(ax, res, overlap=0.7, bw_method=0.16, n_points=200, y_key='Y', **kwargs):
        """
        Renders vertical violin plots clipped to the local data range to avoid 
        vertical artifacts.
        """
        width_factor = kwargs.pop('width', 0.3) 
        alpha = kwargs.get('alpha', 0.8)

        for i, obj in enumerate(res):
            vals = obj.get(y_key)
            if vals is None or len(vals) < 2:
                continue
                
            center_x = (i + 1) * overlap
            
            # 1. Handle constant values (zero peak-to-peak)
            if np.ptp(vals) == 0:
                # Optionally skip or plot a horizontal marker
                continue
                
            # 2. Local evaluation points to prevent vertical lines at zero density
            y_min, y_max = np.min(vals), np.max(vals)
            # Add a small padding proportional to the range for KDE tails
            pad = (y_max - y_min) * 0.1 
            y_eval = np.linspace(y_min - pad, y_max + pad, n_points)
            
            # 3. KDE Calculation
            kde = gaussian_kde(vals, bw_method=bw_method)
            x_kde = kde(y_eval)
            
            # 4. Normalization to local width
            if x_kde.max() > 0:
                x_kde = (x_kde / x_kde.max()) * width_factor
            
            color = obj.get("Color", "blue")
            
            # 5. Plot symmetric sides
            ax.fill_betweenx(y_eval, 
                            center_x - x_kde, 
                            center_x + x_kde, 
                            color=color, 
                            alpha=alpha, 
                            edgecolor=kwargs.get("edgecolor", "none"), # 'none' prevents center line
                            linewidth=kwargs.get("linewidth", 0))
        return ax

    def plot_ax_violin_h(ax, res, overlap=0.7, bw_method=0.16, n_points=200, x_key='X', **kwargs):
        """
        Renders horizontal violin plots clipped to the local data range.
        """
        width_factor = kwargs.pop('width', 0.3) 
        alpha = kwargs.get('alpha', 0.8)

        for i, obj in enumerate(res):
            vals = obj.get(x_key)
            # Violins require a distribution; skip if empty or single point
            if vals is None or len(vals) < 2:
                continue
                
            # The vertical position of the violin center
            center_y = (i + 1) * overlap
            
            # 1. Handle constant values
            if np.ptp(vals) == 0:
                continue
                
            # 2. Local evaluation points for X (the SN convergence values)
            x_min, x_max = np.min(vals), np.max(vals)
            pad = (x_max - x_min) * 0.1 
            x_eval = np.linspace(x_min - pad, x_max + pad, n_points)
            
            # 3. KDE Calculation
            kde = gaussian_kde(vals, bw_method=bw_method)
            y_kde = kde(x_eval)
            
            # 4. Normalization to local height
            if y_kde.max() > 0:
                y_kde = (y_kde / y_kde.max()) * width_factor
            
            color = obj.get("Color", "blue")
            
            # 5. Plot symmetric sides (Horizontal)
            ax.fill_between(x_eval, 
                            center_y - y_kde, 
                            center_y + y_kde, 
                            color=color, 
                            alpha=alpha, 
                            edgecolor=kwargs.get("edgecolor", "none"),
                            linewidth=kwargs.get("linewidth", 0))
        return ax

    def plot_ax_violin_v_lp3(ax, res, overlap=0.7, n_points=200, **kwargs):
        """
        Renders vertical violins using the Pearson Type III theoretical distribution.
        Transforms log-space evaluation points to linear m3 values for plotting.
        """
        from scipy.stats import pearson3
        import numpy as np
        
        width_factor = kwargs.pop('width', 0.3)
        alpha = kwargs.get('alpha', 0.8)

        for i, obj in enumerate(res):
            skew = obj.get('Skew')
            loc = obj.get('Loc')
            scale = obj.get('Scale')
            
            if skew is None or loc is None or scale is None:
                continue
                
            center_x = (i + 1) * overlap
            
            # 1. Define evaluation range in log-space
            y_min_log = pearson3.ppf(0.001, skew, loc=loc, scale=scale)
            y_max_log = pearson3.ppf(0.999, skew, loc=loc, scale=scale)
            y_eval_log = np.linspace(y_min_log, y_max_log, n_points)
            
            # 2. Calculate Theoretical PDF in log-space
            x_pdf = pearson3.pdf(y_eval_log, skew, loc=loc, scale=scale)
            
            # 3. Transform evaluation points to linear space (m3)
            y_eval_linear = 10**y_eval_log
            
            # 4. Normalization to fixed width
            if x_pdf.max() > 0:
                x_pdf = (x_pdf / x_pdf.max()) * width_factor
                
            color = obj.get("Color", "blue")
            
            # 5. Plot symmetric sides using linear y-values
            ax.fill_betweenx(y_eval_linear, 
                            center_x - x_pdf, 
                            center_x + x_pdf, 
                            color=color, 
                            alpha=alpha, 
                            edgecolor="none",
                            linewidth=0)
        return ax

    def plot_ax_text(ax, res, text_key='GINI_TEXT', **kwargs):
        """
        Plots a stored string directly onto the axis.
        """
        # Default positioning and style
        conf = {
            'x': 0.95, 'y': 0.05, 
            'ha': 'right', 'va': 'bottom', 
            'fontsize': 7,
            'transform': ax.transAxes
        }
        conf.update(kwargs)

        for obj in res:
            txt = obj.get(text_key)
            if txt:
                ax.text(s=txt, **conf)
  
    # 3. Style
    def set_legend(ax, po, by='Label', where='outside_right', fontsize=8, **kwargs):
        """
        Adds a de-duplicated legend to the specified axis based on a key in 'po'.
        'where' options: 'inside', 'outside_right', 'outside_left', 'bottom'
        """
        # 1. De-duplicate handles and labels
        # This ensures one entry per 'BT' rather than one per line
        handles, labels = ax.get_legend_handles_labels()
        
        # If no labels exist yet (standard for plot_ax_pdf), 
        # we must create proxy artists or use the ones already plotted
        # Assuming plot_ax_pdf was called with label=obj[by]
        unique_map = {}
        for h, l in zip(handles, labels):
            if l not in unique_map:
                unique_map[l] = h

        if not unique_map:
            return # Exit if no labeled data found

        # 2. Define Position Logic
        positions = {
            'inside': {
                'loc': 'best', 
                'bbox_to_anchor': None
            },
            'outside_right': {
                'loc': 'upper left', 
                'bbox_to_anchor': (1.02, 1)
            },
            'outside_left': {
                'loc': 'upper right', 
                'bbox_to_anchor': (-0.15, 1) # Shifted left of the y-axis
            },
            'bottom': {
                'loc': 'upper center', 
                'bbox_to_anchor': (0.5, -0.15),
                'ncol': len(unique_map)
            }
        }

        pos = positions.get(where, positions['inside'])

        # 3. Create Legend
        ax.legend(
            unique_map.values(), 
            unique_map.keys(),
            loc=pos['loc'],
            bbox_to_anchor=pos['bbox_to_anchor'],
            fontsize=fontsize,
            frameon=False,      # Minimalist style
            **kwargs
        )
#endregion
# region D.2.1 Calc. functions.
def calc_hr_convergence(df, rp_key, threshold, z_score):
    """
    Calculates convergence metrics for a specific Return Period.
    Criteria: SN > 30, IRME == 0, and 30 consecutive stable steps.
    """
    # Filter data for the current Return Period
    rp_df = df[df['RP'] == rp_key].copy()
    
    if rp_df.empty:
        return False, rp_df

    ## PER-BUILDING (BID) calculation
    # Sorting to ensure chronological accumulation
    rp_df = rp_df.sort_values(by=['BID', 'SN'])

    # he_BID_sn: Cumulative Sample Mean per Building
    rp_df['he_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().mean())

    # SD_BID_sn: Cumulative Standard Deviation (Sample Standard Deviation)
    rp_df['SD_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().std(ddof=1))

    # SE_BID_sn: Standard error
    rp_df['SE_BID_sn'] = rp_df['SD_BID_sn'] / np.sqrt(rp_df['SN'])

    # ME_BID_sn: Margin of Error
    rp_df['ME_BID_sn'] = z_score * rp_df['SE_BID_sn']

    # Residual calculation: max(0, ME - T)
    rp_df['Residual'] = (rp_df['ME_BID_sn'] - threshold).clip(lower=0)
    
    # BID-level Stability (Stable when SN > 30 and Residual == 0)
    rp_df['MET_BID'] = (rp_df['SN'] > 30) & (rp_df['Residual'] == 0)
    
    # STC_BID: Stability Counter for consecutive 'MET' steps per BID
    blocks = (~rp_df['MET_BID']).groupby(rp_df['BID']).cumsum()
    rp_df['STC_BID'] = rp_df.groupby(['BID', blocks]).cumcount() * rp_df['MET_BID']
    
    # CONV_BID: Final convergence flag (30 consecutive stable steps)
    rp_df['CONV_BID'] = rp_df['STC_BID'] >= 30
    
    ## GLOBAL (RP) Integration
    # Create the simplified dataframe for IRME tracking
    rp_simp = rp_df.groupby('SN').agg(
        IRME=('Residual', 'sum'),                            # Integrated Residual Margin of Error
        NON_CONV_COUNT=('Residual', lambda x: (x > 0).sum())  # Count of buildings where ME > T
    ).reset_index()

    # Calculate relative non-convergence percentage
    total_flooded_bids = rp_df[rp_df['he'] > 0]['BID'].nunique()
    rp_simp['REL_NON_CONV'] = (rp_simp['NON_CONV_COUNT'] / total_flooded_bids) * 100

    # MET: Condition is met ONLY if SN > 30 and IRME is 0
    rp_simp['MET'] = (rp_simp['SN'] > 30) & (rp_simp['IRME'] == 0)

    # STC_AG: Stability Counter for consecutive 'MET' steps
    # We identify blocks of consecutive True values
    blocks = (~rp_simp['MET']).cumsum()
    rp_simp['STC_AG'] = rp_simp.groupby(blocks).cumcount() * rp_simp['MET']

    # CONV: Final convergence flag (30 consecutive stable steps)
    rp_simp['CONV'] = rp_simp['STC_AG'] >= 30

    # Check the status of the latest simulation
    if not rp_simp.empty:
        is_converged = rp_simp['CONV'].iloc[-1]
        return bool(is_converged), rp_simp, rp_df
    
    return False, rp_simp

def calc_vats_convergence(df, n_sims, n_start=500, calc_step=500, n_step_check=3, alphas = [50, 10, 5, 1], epsilons =[25, 20, 10, 5, 1]):
    """
    Extended Vats et al. (2019) multivariate stopping rule for multiple buildings, robust to:
    - Zero-inflated columns (variance = 0 for early iterations)
    - Singular / near-singular covariance matrices

    Strategy: For each RP at each n, drop columns whose variance is still zero (all-zero so far).
    The effective dimension p_eff adapts as more non-zero values in other columns appear.
    BIDs are computed through multiprocess
    """
    # Helping functions (for multiprocessing)
    def process_bid(args, alphas = [50, 10, 5, 1], epsilons =[25, 20, 10, 5, 1]):
        import numpy as np
        from math import pi, gamma
        from scipy.stats import f

        # ---- local helpers  ----
        def compute_vats_criterion(sub, n, alphas = [50, 10, 5, 1], epsilons =[25, 20, 10, 5, 1]):
            """
            Compute Vats convergence quantities for a given subset (n x p_eff).
            """
            # Parameters to evaluate
            alphas = [a/100 for a in alphas]
            epsilons = [e/100 for e in epsilons]
            
            # ── active columns: variance > 0 ───────────────────────────
            col_var = np.var(sub, axis=0)
            active = col_var > 0
            p_eff = int(active.sum())
            if p_eff < 2 or n <= p_eff:
                return None, [f"Warning: n={n} not greater than p_eff={p_eff} or p_eff < 2. Skipping."]
            sub = sub[:, active].astype(np.float64)

            # ── covariance ─────────────────────────────────────────────
            cov = np.cov(sub, rowvar=False, ddof=1)
            det = np.linalg.det(cov)
            if det <= 0:
                return None, [f"Warning: Non-positive determinant (det={det:.2e}) for n={n}, p_eff={p_eff}. Skipping."]
            
            ### Inequality calculation
            ## Common components
            # ── unit ball volume ───────────────────────────────────────
            unit_ball_vol = pi ** (p_eff / 2) / gamma(p_eff / 2 + 1)

            results = {}
            ## LHS
            for a in alphas:
                # ── F-based finite-sample correction ───────────────────────
                c_n = (p_eff * (n - 1) / (n - p_eff)) * f.ppf(1 - a, p_eff, n - p_eff)
                # ── ellipsoid volume ───────────────────────────────────────
                vol = unit_ball_vol * (c_n / n) ** (p_eff / 2) * (det ** 0.5)
                # ── stopping rule terms ────────────────────────────────────
                lhs = vol ** (1 / p_eff) + 1 / n
                results[f"lhs_{int(a*100):02d}"] = lhs
            
            ## RHS
            for e in epsilons:
                rhs = e * det ** (1 / (2 * p_eff))
                results[f"rhs_{int(e*100):02d}"] = rhs

            return results, []

        # ---- unpack ----
        subset, bid, n = args
        logs = []
        
        try:
            # ── Calculate Vats criterion ────────────
            metrics, crit_logs = compute_vats_criterion(subset, n, alphas=alphas, epsilons=epsilons)
            if metrics is None:
                logs.append(f"Skipping BID {bid} at SN {n} due to insufficient active dimensions or det <= 0.")
                return {"bid": bid, "status": "skipped", "logs": logs}

            # Calculate residuals for all alpha-epsilon combinations
            residuals = {}
            active_statuses = {}

            # ── Residual ─────────────────────────────
            overall_active = False
            for a in alphas:
                for e in epsilons:
                    suffix = f"a{a:02d}_e{e:02d}"
                    res_val = max(0.0, metrics[f"lhs_{a:02d}"] - metrics[f"rhs_{e:02d}"])
                    
                    residuals[f"res_{suffix}"] = res_val
                    is_active = res_val > 0
                    active_statuses[f"rem_{suffix}"] = 1 if is_active else 0
                    if is_active: overall_active = True

            return {
                "status": "active" if overall_active else "converged", 
                "bid": bid, 
                "residuals": residuals,
                "active_counts": active_statuses,
                "logs": logs
            }
        
        except Exception as e:
            logs.append(f"Error occurred while processing BID {bid} at SN {n}: {e}")
            return {"bid": bid, "status": "error", "logs": logs}

    # Set general variables
    RPs_vats = np.sort(df['RP'].unique())
    cost_cols = [c for c in df.columns if c.startswith('c_')]
    expected_sns_full = np.arange(1, n_sims + 1)
    
    final_results = []
    detailed_results = []
    execution_logs = []
    
    #ridge = 1e-10
    for rp in RPs_vats: # rp = 500
        print(f"Processing RP {rp}...\n")
        execution_logs.append(f"Processing RP {rp}...\n")   
        # --- Pre-compute RP subsets ---
        df_rp = df[df["RP"] == rp]
        BIDs_vats_rp = np.sort(df_rp['BID'].unique())
        total_rp_bids = len(BIDs_vats_rp)
        consecutive_convergence = {bid: 0 for bid in BIDs_vats_rp}
        active_bids = set(BIDs_vats_rp)
        
        # --- Pre-compute a dic of n_sims * p matrices ---
        print(f"Pre-computing matrices\n")
        execution_logs.append(f"Pre-computing matrices for RP {rp}\n")
        grouped = df_rp.groupby("BID")
        bid_matrices = {
            bid: (
                group.set_index("SN")
                .reindex(expected_sns_full, fill_value=0.0)[cost_cols]
                .values.astype(np.float64)
            )
            for bid, group in grouped if bid in BIDs_vats_rp
        }
        
        # --- Compute convergence across BIDs ---
        print(f"Computing IR metric\n")
        execution_logs.append(f"Computing IR metric for RP {rp}\n")
        process_func = partial(process_bid, alphas=alphas, epsilons=epsilons)
        with multiprocess.Pool(processes=CORES) as pool:
            steps = list(range(n_start, n_sims + 1, calc_step))
            if not steps or steps[-1] < n_sims: steps.append(n_sims)
                
            for n in tqdm(steps): # n=500, n_sims = 10_000
                if not active_bids:
                    print(f"No more BIDs to converge at SN {n}")
                    execution_logs.append(f"No more BIDs to converge at SN {n} for RP {rp}\n")
                    break
                
                # --- Create subsets --- 
                args_list = [(bid_matrices[bid][:n], bid, n) for bid in active_bids]
                
                # --- Calculate convergence ---
                outputs = pool.map(process_func, args_list)

                # --- Track convergence to update active_bids ---
                step_metrics = {}
                new_active_bids = []
                
                for out in outputs:
                    if out["logs"]:
                        execution_logs.extend(out["logs"])
                        
                    if out["status"] in ("skipped", "error"):
                        new_active_bids.append(out["bid"])
                        continue
                    # Detailed outputs
                    detailed_row = {
                        "RP": rp, 
                        "SN": n, 
                        "BID": out["bid"], 
                        "status": out["status"]
                    }
                    detailed_row.update(out["residuals"])
                    detailed_row.update(out["active_counts"])
                    detailed_results.append(detailed_row)
                    
                    # Accumulate residuals and "remaining" counts for each threshold
                    for k, v in out["residuals"].items():
                        step_metrics[k] = step_metrics.get(k, 0.0) + v
                    for k, v in out["active_counts"].items():
                        step_metrics[k] = step_metrics.get(k, 0) + v
                    
                    # BID convergence counter and check stability condition to remove BIDs and save computer time
                    if out["status"] == "converged":
                        consecutive_convergence[out["bid"]] += 1
                    else:
                        consecutive_convergence[out["bid"]] = 0
                        
                    if consecutive_convergence[out["bid"]] < n_step_check:
                        new_active_bids.append(out["bid"])

                # Save results for this step
                row = {
                    "RP": rp, "SN": n, "total_rp_bids": total_rp_bids,
                    "any_active_bids": len(new_active_bids) 
                }
                row.update(step_metrics)
                final_results.append(row)
                
                active_bids = set(new_active_bids)
                
    return pd.DataFrame(final_results), pd.DataFrame(detailed_results), execution_logs

#endregion
# region D.2.2 General function to plot
def plot_any(dict_to_plot, layout_params, save_params, global_pre_style=None, global_style=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 0. Helper for parameter filtering
    def get_kwargs(data_dict, exclude_keys):
        """
        Filters the dictionary to return only Matplotlib-specific kwargs.
        Removes keys used for data (e.g., 'x', 'y') or internal logic (e.g., 'plot_type').
        """
        # Internal control keys that should never be passed to matplotlib functions
        internal_keys = {'plot_type', 'style_type', 'legend_type', 'gstype'}
        # Combine provided exclusion keys with internal control keys
        to_exclude = set(exclude_keys) | internal_keys
        return {k: v for k, v in data_dict.items() if k not in to_exclude}
    
    def _check_required_keys(params, required, ptype, ax_key):
        """Validates existence of mandatory keys to avoid redundant KeyError blocks."""
        if not all(k in params for k in required):
            raise KeyError(f"Axis '{ax_key}': '{ptype}' requires {required} keys.")
    
    # 1. Global pre-Style
    def set_global_pre_style(style_config):
        defaults = {
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        }
        if style_config:
            defaults.update(style_config)
        plt.rcParams.update(defaults)
    
    set_global_pre_style(global_pre_style)

    # 2. Set Layout
    def set_layout(params):
        layout = params.get('layout', '1x1')
        figsize = params.get('figsize', (4, 4))
        mosaic = params.get('mosaic_structure')
        squared = params.get('squared', False)
        kwargs = params.get('kwargs', {})

        if layout == "mosaic":
            if mosaic is None:
                raise ValueError("mosaic_structure required for mosaic layout.")
            fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=figsize, **kwargs)
        else:
            rows, cols = map(int, layout.split('x'))
            fig, axs = plt.subplots(rows, cols, figsize=figsize, **kwargs)
            if isinstance(axs, np.ndarray):
                ax_dict = {i: a for i, a in enumerate(axs.flatten())}
            else:
                ax_dict = {0: axs}

        if squared:
            for a in ax_dict.values():
                a.set_box_aspect(1)
        
        if 'adjust' in params:
            fig.subplots_adjust(**params['adjust'])
        
        return fig, ax_dict
    
    fig, ax_dict = set_layout(layout_params or {})
    
    # 3. Populate Layout
    def apply_plots(ax, plots, key):
        for p in plots:
            ptype = p.get('plot_type')

            if ptype == 'line':
                _check_required_keys(p, ['x', 'y'], 'line', key)
                ax.plot(p['x'], p['y'], **get_kwargs(p, ['x', 'y']))
                
            elif ptype == 'barh':
                _check_required_keys(p, ['y', 'width'], 'barh', key)
                ax.barh(y=p['y'], width=p['width'], **get_kwargs(p, ['y', 'width']))
            
            elif ptype == 'barv':
                _check_required_keys(p, ['x', 'height'], 'barv', key)
                # Note: Matplotlib's method for vertical bars is just .bar()
                ax.bar(x=p['x'], height=p['height'], **get_kwargs(p, ['x', 'height']))
                
            elif ptype == 'hist':
                _check_required_keys(p, ['dataset'], 'hist', key)
                ax.hist(p['dataset'], **get_kwargs(p, ['dataset']))
                
            elif ptype == 'vline':
                _check_required_keys(p, ['x'], 'vline', key)
                ax.axvline(x=p['x'], **get_kwargs(p, ['x']))
                
            elif ptype == 'hline':
                _check_required_keys(p, ['y'], 'hline', key)
                ax.axhline(y=p['y'], **get_kwargs(p, ['y']))
                
            elif ptype == 'fill':
                _check_required_keys(p, ['x', 'y1', 'y2'], 'fill', key)
                ax.fill_between(x=p['x'], y1=p['y1'], y2=p['y2'], **get_kwargs(p, ['x', 'y1', 'y2']))
            
            elif ptype == 'vspan':
                _check_required_keys(p, ['x1', 'x2'], 'vspan', key)
                ax.axvspan(p['x1'], p['x2'], **get_kwargs(p, ['x1', 'x2']))
                
            elif ptype == 'hspan':
                _check_required_keys(p, ['y1', 'y2'], 'hspan', key)
                ax.axhspan(p['y1'], p['y2'], **get_kwargs(p, ['y1', 'y2']))
                
            elif ptype == 'scatter':
                _check_required_keys(p, ['x', 'y'], 'scatter', key)
                ax.scatter(p['x'], p['y'], **get_kwargs(p, ['x', 'y']))
                
            elif ptype == 'violin':
                _check_required_keys(p, ['dataset', 'positions'], 'violin', key)
                kwargs = get_kwargs(p, ['dataset', 'positions'])
                color_input = kwargs.pop('color', None)
                vp = ax.violinplot(p['dataset'], p['positions'], **kwargs)
                if color_input:
                    for i, body in enumerate(vp['bodies']):
                        c = color_input[i] if isinstance(color_input, list) else color_input
                        body.set_facecolor(c)
                        body.set_edgecolor(c)
                        body.set_alpha(kwargs.get('alpha', 0.7))
                    for part in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
                        if part in vp:
                            vp[part].set_edgecolor('black')
                            vp[part].set_linewidth(0.5)
            
            elif ptype == 'text':
                _check_required_keys(p, ['x', 'y', 'text'], 'text', key)
                kwargs = get_kwargs(p, ['x', 'y', 'text', 'transform'])
                transform = p.get('transform', ax.transAxes)
                if transform == 'data':
                    transform = ax.transData
                ax.text(p['x'], p['y'], p['text'], transform=transform, **kwargs)
            
            elif ptype == 'gdf_shp':
                _check_required_keys(p, ['gdf'], 'gdf_shp', key)
                p['gdf'].plot(ax=ax, **get_kwargs(p, ['gdf']))
            
            elif ptype == 'imshow':
                _check_required_keys(p, ['image', 'extent'], 'imshow', key)
                ax.imshow(p['image'], extent=p['extent'], origin='upper', **get_kwargs(p, ['image', 'extent', 'origin']))
            
            elif ptype == 'wms':
                from io import BytesIO
                import matplotlib.image as mpimg
                from owslib.wms import WebMapService

                _check_required_keys(p, ['url', 'layers', 'extent'], 'wms', key)
                wms = WebMapService(p['url'], version=p.get('version', '1.1.1'))
                extent = p['extent']
                bbox = (extent[0], extent[2], extent[1], extent[3])
                img_request = wms.getmap(
                    layers=p['layers'],
                    srs=p.get('crs', 'EPSG:4326'), # Ensure this matches your GDF CRS
                    bbox=bbox,
                    size=p.get('size', (1000, 1000)), # Resolution of the fetched image
                    format=p.get('format', 'image/png'),
                    transparent=p.get('transparent', True)
                )
                img_data = mpimg.imread(BytesIO(img_request.read()))
                ax.imshow(img_data, extent=extent, origin='upper',
                          **get_kwargs(p, ['url', 'layers', 'extent'] + ['version', 'crs', 'size', 'format', 'transparent']))
            
            elif ptype == 'sankey':
                from matplotlib.path import Path
                import matplotlib.patches as patches

                _check_required_keys(p, ['nodes', 'links'], 'sankey', key)
                
                nodes_list = p['nodes'] 
                links_list = p['links'] 
                
                # 1. Draw the connection streams (Links) using Bezier paths
                for link in links_list:
                    src = nodes_list[link['source']]
                    tgt = nodes_list[link['target']]
                    
                    x0, y0 = src['x'], link.get('src_y', src['y'])
                    x1, y1 = tgt['x'], link.get('tgt_y', tgt['y'])
                    
                    # Extract independent source and target widths for tapering
                    w_src = link.get('value_src', link.get('value', 0))
                    w_tgt = link.get('value_tgt', link.get('value', 0))
                    
                    cx0 = x0 + (x1 - x0) / 2
                    cx1 = x1 - (x1 - x0) / 2
                    
                    verts = [
                        (x0, y0),         # Start bottom-left
                        (cx0, y0), (cx1, y1), (x1, y1),  # Forward curve (bottom edge)
                        (x1, y1 + w_tgt), # Move up right side
                        (cx1, y1 + w_tgt), (cx0, y0 + w_src), (x0, y0 + w_src), # Backward curve (top edge)
                        (x0, y0)          # Close loop
                    ]
                    
                    codes = [
                        Path.MOVETO,
                        Path.CURVE4, Path.CURVE4, Path.CURVE4,
                        Path.LINETO,
                        Path.CURVE4, Path.CURVE4, Path.CURVE4,
                        Path.CLOSEPOLY
                    ]
                    
                    path = Path(verts, codes)
                    patch = patches.PathPatch(
                        path, 
                        facecolor=link.get('color', '#cccccc'), 
                        alpha=link.get('alpha', 0.4), 
                        edgecolor='none'
                    )
                    ax.add_patch(patch)
                
                # 2. Draw the vertical blocks (Nodes)
                for node in nodes_list:
                    rect = patches.Rectangle(
                        (node['x'] - p.get('node_width', 0.02), node['y']),
                        p.get('node_width', 0.02) * 2,
                        node['h'],
                        facecolor=node.get('color', '#555555'),
                        edgecolor=node.get('edgecolor', 'none'),
                        alpha=node.get('alpha', 1.0)
                    )
                    ax.add_patch(rect)
                    
                    if node['h'] < 0.005:
                        continue
                        
                    if node['x'] < 0.2:
                        ha_dir = 'right'; offset = -0.025
                    elif node['x'] > 0.8:
                        ha_dir = 'left'; offset = 0.025
                    else:
                        ha_dir = 'center'; offset = 0.0
                        
                    y_pos = node['y'] + node['h']/2 if ha_dir != 'center' else node['y'] + node['h'] + 0.01
                    
                    ax.text(
                        node['x'] + offset, y_pos, node['name'], 
                        va='center', ha=ha_dir, fontsize=p.get('fontsize', 8)
                    )
            
            else:
                print(f"Warning: plot_type '{ptype}' in axis '{key}' still not implemented.")
    
    def apply_style(ax, styles, key):
        for s in styles:
            stype = s.get('style_type')

            if stype == 'title':
                ax.set_title(s.get('label', ''), **get_kwargs(s, ['label']))
                
            elif stype in ['xlabel', 'ylabel']:
                getattr(ax, f'set_{stype}')(s.get('label', ''), **get_kwargs(s, ['label']))
                
            elif stype in ['xlim', 'ylim', 'xscale', 'yscale']:
                getattr(ax, f'set_{stype}')(**get_kwargs(s, []))
                
            elif stype in ['xticks', 'yticks']:
                getattr(ax, f'set_{stype}')(s.get('ticks', []), **get_kwargs(s, ['ticks']))
                
            elif stype in ['xticklabels', 'yticklabels']:
                getattr(ax, f'set_{stype}')(s.get('labels', []), **get_kwargs(s, ['labels']))
            
            elif stype == 'ticks_params':
                ax.tick_params(**get_kwargs(s, []))
            
            elif stype == 'ticklabel_format':
                ax.ticklabel_format(**get_kwargs(s, []))
            
            elif stype == 'major_formatter':
                import matplotlib.ticker as ticker
                axis_name = s.get('axis', 'both')
                formatter_type = s.get('formatter_type', 'object')

                def apply_axis_formatter(target_axis_obj):
                    if formatter_type == 'dms_suffix':
                        from math import isclose
                        
                        suffix = s.get('suffix', '')
                        def create_dms_formatter(curr_axis, curr_suffix):
                            def format_dms(value, pos):
                                abs_val = abs(value)
                                deg = int(abs_val)
                                min_float = (abs_val - deg) * 60
                                minutes = int(min_float)
                                seconds = round((min_float - minutes) * 60)
                                if seconds == 60: seconds = 0; minutes += 1
                                if minutes == 60: minutes = 0; deg += 1
                                    
                                base_str = f"{deg:g}º{minutes:02d}'{seconds:02d}''"
                                
                                tick_locs = curr_axis.get_majorticklocs()
                                vmin, vmax = curr_axis.get_view_interval()
                                view_min, view_max = min(vmin, vmax), max(vmin, vmax)
                                
                                tol = 1e-5 * (view_max - view_min) if view_max != view_min else 1e-5
                                visible_ticks = [t for t in tick_locs if (view_min - tol) <= t <= (view_max + tol)]
                                
                                if visible_ticks and isclose(value, max(visible_ticks), rel_tol=1e-5, abs_tol=1e-8):
                                    return f"{base_str}{curr_suffix}"
                                return base_str
                            return ticker.FuncFormatter(format_dms)
                        formatter = create_dms_formatter(target_axis_obj, suffix)
                        
                    elif formatter_type == 'string':
                        formatter = ticker.StrMethodFormatter(s.get('format_str', '{x}'))
                        
                    else:
                        formatter = s.get('formatter_obj', None)

                    if formatter is not None:
                        target_axis_obj.set_major_formatter(formatter)

                if axis_name in ['x', 'both']: apply_axis_formatter(ax.xaxis) 
                if axis_name in ['y', 'both']: apply_axis_formatter(ax.yaxis)
                
            elif stype == 'offset_text':
                axis_name = s.get('axis', 'y')
                axis_obj = getattr(ax, f'{axis_name}axis')
                axis_obj.get_offset_text().set_fontsize(s.get('fontsize', 8))
            
            elif stype == 'grid':
                ax.grid(**get_kwargs(s, []))
            
            elif stype == 'spines':
                for spine, vis in get_kwargs(s, []).items():
                    ax.spines[spine].set_visible(vis)
            
            elif stype == 'letter':
                ax.text(
                    s.get('x', 0.02), s.get('y', 0.95), s.get('label', ''),
                    transform=ax.transAxes, fontweight=s.get('fontweight', 'bold'),
                    va='top', ha='left', **get_kwargs(s, ['x', 'y', 'label', 'fontweight', 'transform']))
            
            elif stype == 'aspect':
                ax.set_aspect(s.get('aspect', 'equal'))
            
            elif stype == 'off':
                ax.axis('off')
            
            elif stype == 'legend':
                ax.legend(
                    loc=s.get('loc', 'best'),
                    fontsize=s.get('fontsize', 10),
                    title=s.get('title', None)
                )
            
            else:
                print(f"Warning: style_type '{stype}' in axis '{key}' still not implemented in plot_any.")
    
    def apply_legend(ax, legend, key):
        import matplotlib.patches as patches
        import matplotlib.lines as lines
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        for l in legend:
            ltype = l.get('legend_type')
            
            if ltype == 'text':
                required = ['x', 'y', 'content']
                _check_required_keys(l, required, 'text', key)
                ax.text(l['x'], l['y'], l['content'], transform=ax.transAxes, **get_kwargs(l, required))
                
            elif ltype == 'circle':
                required = ['x', 'y', 'radius', 'color']
                _check_required_keys(l, required, 'circle', key)
                circ = patches.Circle((l['x'], l['y']), l['radius'], color=l['color'],
                                      transform=ax.transAxes, clip_on=False, **get_kwargs(l, required))
                ax.add_patch(circ)
            
            elif ltype == 'rectangle':
                required = ['x', 'y', 'width', 'height', 'color']
                _check_required_keys(l, required, 'square', key)
                rect = patches.Rectangle((l['x'], l['y']), l['width'], l['height'], 
                                        color=l['color'], transform=ax.transAxes, clip_on=False,
                                        **get_kwargs(l, required))
                ax.add_patch(rect)
                
            elif ltype == 'line':
                required = ['x', 'y', 'length', 'color']
                _check_required_keys(l, required, 'line', key)
                line = lines.Line2D([l['x'], l['x'] + l['length']], [l['y'], l['y']], color=l['color'],
                                    transform=ax.transAxes, clip_on=False, **get_kwargs(l, required))
                ax.add_line(line)
                
            elif ltype == 'gdf_gradient':
                _check_required_keys(l, required, 'gdf_gradient', key)
                if l['source_ax'] in ax_dict:
                    for art in ax_dict[l['source_ax']].collections:
                        if hasattr(art, 'cmap'):
                            cax = inset_axes(ax, width=f"{l['w']*100}%", height=f"{l['h']*100}%", loc='lower left', 
                                             bbox_to_anchor=(l['x'], l['y'], 1, 1), bbox_transform=ax.transAxes, borderpad=0)
                            plt.colorbar(art, cax=cax, **get_kwargs(l, required))
                            break
            
            elif ltype == 'north_arrow':
                _check_required_keys(l, ['x', 'y'], 'north_arrow', key)
                transform = l.get('transform', ax.transAxes)
                length = l.get('length', 0.05)
                pad = l.get('pad', 0.005)
                color = l.get('color', 'black')
                fontsize = l.get('fontsize', 8)
                
                ax.text(l['x'], l['y'], 'N', transform=transform, ha='center', va='bottom',
                        fontsize=fontsize, fontweight='bold', color=color)
                ax.annotate('', xy=(l['x'], l['y'] - pad), xytext=(l['x'], l['y'] - pad - length),
                            xycoords=transform, textcoords=transform,
                            arrowprops=dict(facecolor=color, edgecolor=color,
                            headwidth=5, width=1.5, headlength=5, shrinkA=0, shrinkB=0))
            
            elif ltype == 'scalebar':
                from matplotlib_scalebar.scalebar import ScaleBar
                _check_required_keys(l, ['dx'], 'scalebar', key)
                kwargs = get_kwargs(l, ['dx'])
                sb = ScaleBar(dx=l['dx'], **kwargs)
                ax.add_artist(sb)
                
    for key, config in dict_to_plot.items():
        if key in ax_dict:
            if 'plots' in config: apply_plots(ax_dict[key], config.get('plots', []), key)
            if 'style' in config: apply_style(ax_dict[key], config.get('style', []), key)
            if 'legend' in config: apply_legend(ax_dict[key], config.get('legend', []), key)
    
    # 4. Apply global style
    def apply_global_figure_style(fig, global_style):
        if global_style:
            for s in global_style:
                gstype = s.get('gstype')
                label = s.get('label', '')
                kwargs = get_kwargs(s, ['label'])
                if gstype in ['suptitle', 'supxlabel', 'supylabel']:
                    getattr(fig, gstype)(label, **kwargs)
                
    apply_global_figure_style(fig, global_style)
    
    # 5. Save Plot
    def save_plot(fig, params):
        path = os.path.join(params.get('output_dir', '.'), params.get('subfolder', ''))
        if not os.path.exists(path):
            os.makedirs(path)
        
        if params.get('tight_layout', False):
            fig.tight_layout()
        
        fig.savefig(
            os.path.join(path, params.get('fname', 'plot.png')), 
            bbox_inches='tight',
            dpi=params.get('dpi', 300),
            pad_inches=params.get('pad', 0.1)
        )
        
        if params.get('show', True):
            plt.show()
        else:
            plt.close()
            
    save_plot(fig, save_params)

    return fig, ax_dict

def init_dic(dict_to_plot, ax, key):
    """
    Checks if the axis and specific sub-key (plots, style, legend) exist 
    in dict_to_plot and initializes them as empty containers if needed.
    If they already exist, it reinitialize them to avoid duplication
    """
    if ax not in dict_to_plot:
        dict_to_plot[ax] = {}
    dict_to_plot[ax][key] = []

def get_ax_ratio(layout_params, ax):
    """Calculates the physical width/height ratio of a subplot."""
    fig, ax_dict = plt.subplot_mosaic(layout_params['mosaic_structure'], 
                                     figsize=layout_params['figsize'], 
                                     **layout_params['kwargs'])
    fig.subplots_adjust(**layout_params['adjust'])
    plt.draw() 
    bbox = ax_dict[ax].get_window_extent()
    real_ratio = bbox.width / bbox.height
    plt.close(fig)
    return real_ratio

def calc_map_extent(top_left_x, top_left_y, zoom_pct, real_ratio, base_span = 0.02):
    """Converts coordinates and zoom levels into spatial limits[cite: 5]."""
    xmin = -(top_left_x[0] + top_left_x[1]/60 + top_left_x[2]/3600)
    ymax = (top_left_y[0] + top_left_y[1]/60 + top_left_y[2]/3600)
    width = base_span * (zoom_pct / 100)
    height = width / real_ratio 
    xmax = xmin + width
    ymin = ymax - height
    return [xmin, xmax, ymin, ymax]

def simplify_flood_contour(shp):
    shp = shp.explode(index_parts=False)
    shp.geometry = shp.geometry.apply(
        lambda poly: Polygon(poly.exterior) if poly.interiors else poly
    )
    shp = shp[shp.geometry.area > 1e-7]
    shp = shp.union_all()
    shp = shp.simplify(tolerance=0.0001, preserve_topology=True)
    shp = gpd.GeoDataFrame(geometry=[shp], crs="EPSG:4326")
    return shp

def tif_to_simplified_shp(tif_path, output_shp_path=None):
    with rasterio.open(tif_path) as src:
        arr = src.read(1)
        
        nodata = src.nodata
        if nodata is not None:
            mask = (arr != nodata) & (~np.isnan(arr)) if np.issubdtype(arr.dtype, np.floating) else (arr != nodata)
        else:
            mask = (arr > 0) if np.issubdtype(arr.dtype, np.number) else (arr != 0)
            
        shape_gen = features.shapes(mask.astype(np.uint8), transform=src.transform)
        
        records = [
            {'geometry': shape(geom), 'value': value}
            for geom, value in shape_gen if value == 1
        ]
        
        # 1. Force the correct native projection (UTM Zone 30N) regardless of header mislabeling
        shp = gpd.GeoDataFrame(records, crs="EPSG:25830")
        
        # 2. Now properly convert the meters to WGS84 degrees
        shp = shp.to_crs("EPSG:4326")
        
    if output_shp_path:
        shp.to_file(output_shp_path)
        
    return simplify_flood_contour(shp)

def get_dx_for_scalebar(extent, crs='EPSG:4326'):
    """
    Calculates the physical length (dx in meters) of 1 horizontal data unit of the axis.
    """
    import math
    xmin, xmax, ymin, ymax = extent
    if crs == 'EPSG:4326':
        mid_lat = (ymin + ymax) / 2
        r_earth = 6378137.0  # Equatorial radius in meters
        dx_meters = (math.pi / 180.0) * r_earth * math.cos(math.radians(mid_lat))
        return dx_meters
    elif crs in ['EPSG:25830', 'EPSG:3857']:
        # For projected metric systems, 1 data unit is already 1 meter.
        return 1.0
    else:
        raise ValueError("CRS not configured for scale calculation.")

def create_2_linear_ax_scale(max, split, pct):
        forward_scale = lambda x: np.where(
            x <= split, (pct / split) * x, pct + ((1.0 - pct) / (max - split)) * (x - split)
        )
        inverse_scale = lambda y: np.where(
            y <= pct, (split / pct) * y, split + ((max - split) / (1.0 - pct)) * (y - pct)
        )
        return (forward_scale, inverse_scale)

def create_3_linear_ax_scale(max_val, split1, split2, pct1, pct2):
    """
    Creates a 3-segment piecewise linear scale.
    - Segment 1: 0 to split1 (takes up 0 to pct1 of the visual space)
    - Segment 2: split1 to split2 (takes up pct1 to pct2 of the visual space)
    - Segment 3: split2 to max_val (takes up pct2 to 1.0 of the visual space)
    """
    import numpy as np
    
    def forward_scale(x):
        # Convert to float array to prevent integer division issues
        x = np.asarray(x, dtype=float)
        
        conds = [x <= split1, (x > split1) & (x <= split2), x > split2]
        funcs = [
            (pct1 / split1) * x,
            pct1 + ((pct2 - pct1) / (split2 - split1)) * (x - split1),
            pct2 + ((1.0 - pct2) / (max_val - split2)) * (x - split2)
        ]
        return np.select(conds, funcs, default=x)

    def inverse_scale(y):
        y = np.asarray(y, dtype=float)
        
        conds = [y <= pct1, (y > pct1) & (y <= pct2), y > pct2]
        funcs = [
            (split1 / pct1) * y,
            split1 + ((split2 - split1) / (pct2 - pct1)) * (y - pct1),
            split2 + ((max_val - split2) / (1.0 - pct2)) * (y - pct2)
        ]
        return np.select(conds, funcs, default=y)

    return (forward_scale, inverse_scale)

class shap_results_grouper:
    def __init__(self, columns, separator="|"):
        self.separator = separator
        self.columns_state = [str(col).split(self.separator) for col in columns]

    def print_unique_names(self, level_index):
        unique_groups = set()
        for col_levels in self.columns_state:
            if level_index < len(col_levels):
                unique_groups.add(col_levels[level_index])
        print(f"{len(unique_groups)} groups found")
        print(f"Unique groups at Level {level_index}:", sorted(list(unique_groups)))

    def remove_all_levels(self):
        for i, col_levels in enumerate(self.columns_state):
            self.columns_state[i] = [col_levels[0]]

    def add_level(self, rules, default="Unclassified"):
        for i, col_levels in enumerate(self.columns_state):
            base_name = col_levels[0]
            new_group = default
            
            if callable(rules):
                new_group = rules(base_name)
            elif isinstance(rules, dict):
                for pattern, group_name in rules.items():
                    if pattern == base_name or (isinstance(pattern, str) and re.search(pattern, base_name)):
                        new_group = group_name
                        break
                        
            self.columns_state[i].append(new_group)

    def update_dataframe(self, df):
        df.columns = [self.separator.join(col_levels) for col_levels in self.columns_state]
        return df

def calc_expected_grouped(output_gsa_dir, RPs_unq, level=0, by_bid=False, min_rp_AEP_0=None):
    """
    Aggregates SHAP values by a specified multi-index level,
    calculates absolute quantiles per return period, and integrates over the AEP curve
    to return Expected Annual SHAP values.
    """
    bid_name = 'bid' if by_bid else 'all'
    level_name = f"l{level}"
    file_name = f"df_expected_{level_name}_{bid_name}.feather"
    output_path = os.path.join(output_gsa_dir, file_name)
    
    groupby_cols = ['Feature_Group', 'BID'] if by_bid else ['Feature_Group']
    
    if os.path.exists(output_path):
        print(f"\n[INFO] Expected Annual SHAP file already exists. Loading from: {output_path}")
        df_expected = pd.read_feather(output_path)
        return df_expected.set_index(groupby_cols)
        
    print(f"\n[INFO] Target file {file_name} not found. Running calculations workflow pipeline...")
    
    target_idx = int(level)
    all_rp_quantiles = []
    
    # 1. Process each Return Period
    for rp in RPs_unq:
        print(f"Processing rp {rp}")
        shap_path = os.path.join(output_gsa_dir, f"shap_results_rp_{rp}.feather")
        if not os.path.exists(shap_path):
            print(f"SHAP file for RP {rp} not found at {shap_path}. Skipping.")
            continue
        
        print(f"Loading file...")
        df_shap = pd.read_feather(shap_path)
        
        meta_cols = [c for c in df_shap.columns if 'BID' in c or 'base_value' in c]
        feature_cols = [c for c in df_shap.columns if c not in meta_cols]
        
        col_mapping = [str(c).split('|')[target_idx] for c in feature_cols]
        
        print(f"Preparing data...")
        grouped_features = (
            df_shap[feature_cols]
            .T.groupby(by=col_mapping)
            .sum()
            .T
        )
        
        print(f"Calculating quantiles...")
        if by_bid:
            bid_col = next(c for c in meta_cols if 'BID' in c)
            grouped_features['BID'] = df_shap[bid_col]
            
            quantiles = grouped_features.set_index('BID').abs().groupby('BID').quantile([0.05, 0.50, 0.95])
            quantiles.index.names = ['BID', 'Quantile']
            
            quantiles = quantiles.T.stack(level='BID', future_stack=True)
            quantiles.columns = ['Q05', 'Q50', 'Q95']
            quantiles.index.names = ['Feature_Group', 'BID']
        else:
            quantiles = grouped_features.abs().quantile([0.05, 0.50, 0.95]).T
            quantiles.columns = ['Q05', 'Q50', 'Q95']
            quantiles.index.name = 'Feature_Group'
            
        quantiles['RP'] = rp
        all_rp_quantiles.append(quantiles)
        
        del df_shap, grouped_features, quantiles
        gc.collect()

    if not all_rp_quantiles:
        return pd.DataFrame()

    df_all = pd.concat(all_rp_quantiles).reset_index()
    
    # 2. Add zero boundary anchor point
    if min_rp_AEP_0 is not None:
        unique_groups = df_all[groupby_cols].drop_duplicates()
        unique_groups['Q05'] = 0.0
        unique_groups['Q50'] = 0.0
        unique_groups['Q95'] = 0.0
        unique_groups['RP'] = float(min_rp_AEP_0)
        df_all = pd.concat([df_all, unique_groups], ignore_index=True)
        
    df_all['AEP'] = 1.0 / df_all['RP']
    
    print(f"Calculating expected value among all rp")
    # 3. Vectorized Matrix Trapezoidal Integration (Replaces slow .apply bottleneck)
    df_pivot = df_all.pivot(index=groupby_cols, columns='AEP', values=['Q05', 'Q50', 'Q95'])
    
    # Fill missing RP gaps with 0.0 to allow integration for BIDs that only appear in high RPs
    df_pivot = df_pivot.fillna(0.0)
    
    df_pivot = df_pivot.sort_index(axis=1, level='AEP')
    
    aep_axes = df_pivot.columns.get_level_values('AEP').unique().values
    
    q05_expected = np.trapezoid(df_pivot['Q05'].values, x=aep_axes, axis=1)
    q50_expected = np.trapezoid(df_pivot['Q50'].values, x=aep_axes, axis=1)
    q95_expected = np.trapezoid(df_pivot['Q95'].values, x=aep_axes, axis=1)
    
    df_expected = pd.DataFrame({
        'Q05': q05_expected,
        'Q50': q50_expected,
        'Q95': q95_expected
    }, index=df_pivot.index).reset_index()
    
    # 4. Vectorized Share Percentages
    print(f"Calculating share percentages")
    quantile_cols = ['Q05', 'Q50', 'Q95']
    if by_bid:
        for col in quantile_cols:
            total_sum = df_expected.groupby('BID')[col].transform('sum')
            df_expected[f'{col}_share_%'] = np.where(total_sum != 0, (df_expected[col] / total_sum) * 100, 0.0)
    else:
        for col in quantile_cols:
            total_sum = df_expected[col].sum()
            df_expected[f'{col}_share_%'] = (df_expected[col] / total_sum) * 100 if total_sum != 0 else 0.0

    df_expected = df_expected.sort_values(by='Q50', ascending=False).set_index(groupby_cols)
    
    df_expected.reset_index().to_feather(output_path)
    print(f"[SUCCESS] Calculated results written down successfully into destination path: {output_path}")
    return df_expected

def calc_rp_evolution_grouped(output_gsa_dir, RPs_unq, level=0, by_bid=False):
    """
    Tracks the evolution of the median (Q50) absolute SHAP values and their 
    percentage share across different Return Periods for a specified grouping level.
    """
    bid_name = 'bid' if by_bid else 'all'
    level_name = f"l{level}"
    file_name = f"df_evolution_{level_name}_{bid_name}.feather"
    output_path = os.path.join(output_gsa_dir, file_name)
    
    groupby_cols = ['Feature_Group', 'BID'] if by_bid else ['Feature_Group']
    
    if os.path.exists(output_path):
        print(f"\n[INFO] Evolution file already exists. Loading from: {output_path}")
        df_evolution = pd.read_feather(output_path)
        return df_evolution.set_index(groupby_cols)
        
    print(f"\n[INFO] Target file {file_name} not found. Running evolution workflow pipeline...")
    
    target_idx = int(level)
    all_rp_data = []
    
    for rp in RPs_unq:
        print(f"Processing rp {rp}")
        shap_path = os.path.join(output_gsa_dir, f"shap_results_rp_{rp}.feather")
        if not os.path.exists(shap_path):
            print(f"SHAP file for RP {rp} not found at {shap_path}. Skipping.")
            continue
            
        print(f"Loading file...")
        df_shap = pd.read_feather(shap_path)
        
        meta_cols = [c for c in df_shap.columns if 'BID' in c or 'base_value' in c]
        feature_cols = [c for c in df_shap.columns if c not in meta_cols]
        
        col_mapping = [str(c).split('|')[target_idx] for c in feature_cols]
        
        print(f"Preparing data and computing Q50...")
        grouped_features = (
            df_shap[feature_cols]
            .T.groupby(by=col_mapping)
            .sum()
            .T
        )
        
        if by_bid:
            bid_col = next(c for c in meta_cols if 'BID' in c)
            grouped_features['BID'] = df_shap[bid_col]
            
            q50 = grouped_features.set_index('BID').abs().groupby('BID').quantile(0.50)
            q50 = q50.stack(future_stack=True).reset_index()
            q50.columns = ['BID', 'Feature_Group', 'Q50']
            
            total_sum = q50.groupby('BID')['Q50'].transform('sum')
            q50['Share_%'] = np.where(total_sum != 0, (q50['Q50'] / total_sum) * 100, 0.0)
            
        else:
            q50 = grouped_features.abs().quantile(0.50).reset_index()
            q50.columns = ['Feature_Group', 'Q50']
            
            total_sum = q50['Q50'].sum()
            q50['Share_%'] = (q50['Q50'] / total_sum) * 100 if total_sum != 0 else 0.0
            
        q50['RP'] = rp
        all_rp_data.append(q50)
        
        del df_shap, grouped_features, q50
        gc.collect()

    if not all_rp_data:
        return pd.DataFrame()

    df_all = pd.concat(all_rp_data, ignore_index=True)
    
    print(f"Restructuring output matrix...")
    # Pivot to create a wide format showing RP evolution horizontally
    df_pivot = df_all.pivot(index=groupby_cols, columns='RP', values=['Q50', 'Share_%'])
    
    # Fill missing RP gaps with 0.0
    df_pivot = df_pivot.fillna(0.0)
    
    # Flatten multi-level column names to 'Metric_RP' (e.g., 'Q50_RP5', 'Share_%_RP100')
    df_pivot.columns = [f"{metric}_RP{rp}" for metric, rp in df_pivot.columns]
    
    # Sort index for predictable ordering
    df_evolution = df_pivot.sort_index()
    
    df_evolution.reset_index().to_feather(output_path)
    print(f"[SUCCESS] Evolution results written down successfully into destination path: {output_path}")
    
    return df_evolution

def generate_scaled_multilayer_sankey(df_list, relations_list, colors_dict, scale_params=None):
    if len(df_list) - 1 != len(relations_list):
        raise ValueError("relations_list length must equal len(df_list) - 1.")
        
    nodes = []
    links = []
    node_counter = 0
    
    num_layers = len(df_list)
    x_coords = np.linspace(0.1, 0.9, num_layers)
    
    global_max_vol = max(df['Q50_share_%'].sum() for df in df_list)
    
    if scale_params:
        raw_split = global_max_vol * scale_params['data_pct']
        visual_pct = scale_params['canvas_pct']
        fwd = lambda x: np.where(
            x <= raw_split, 
            (visual_pct / raw_split) * x, 
            visual_pct + ((1.0 - visual_pct) / (global_max_vol - raw_split)) * (x - raw_split)
        )
    else:
        fwd = lambda x: x / global_max_vol

    layer_node_tracking = [] 
    
    # --- PHASE 1: GENERATE NODES ---
    for i in range(num_layers):
        df_clean = df_list[i].dropna(subset=['Q50_share_%']).copy()
        sorted_features = df_clean.sort_values(by='Q50_share_%', ascending=False).index.tolist()
        
        layer_map = {}
        cum_volume_raw = 0.0  
        
        for feat in sorted_features:
            val = df_clean.loc[feat, 'Q50_share_%']
            
            v0_raw = cum_volume_raw
            v1_raw = cum_volume_raw + val
            
            v0_scaled = float(fwd(v0_raw))
            v1_scaled = float(fwd(v1_raw))
            h_scaled = v1_scaled - v0_scaled
            
            clean_name = str(feat).split(' (')[0]
            if '_' in clean_name: clean_name = clean_name.split('_')[0]
            color = colors_dict.get(clean_name, '#cccccc')
            
            nodes.append({
                "name": f"{feat} (L{i+1})", 
                "x": x_coords[i], "y": v0_scaled, "h": h_scaled, "color": color
            })
            
            layer_map[feat] = {
                'id': node_counter, 'val': val,
                'out_curr_raw': v0_raw, 'in_curr_raw': v0_raw
            }
            node_counter += 1
            cum_volume_raw += val
            
        layer_node_tracking.append(layer_map)
        
    # --- PHASE 2: ROUTE LINKS (GENERALIZED SPLIT AND MERGE FUNNELING) ---
    for i in range(num_layers - 1):
        left_map = layer_node_tracking[i]
        right_map = layer_node_tracking[i+1]
        relations = relations_list[i]
        
        split_sources = set(relations.keys())
        split_targets = set([tgt for tgts in relations.values() for tgt in tgts])
        
        # Precompute reverse relations to allow N-to-1 merging math
        reverse_relations = {}
        for s, t_list in relations.items():
            for t in t_list:
                reverse_relations.setdefault(t, []).append(s)
        
        # 1. Explicit Relations (Splits 1-to-N and Merges N-to-1)
        for src_feat, target_list in relations.items():
            if src_feat not in left_map: continue
            src_info = left_map[src_feat]
            
            target_vals_total = sum([right_map[t]['val'] for t in target_list if t in right_map])
            
            for tgt_feat in target_list:
                if tgt_feat not in right_map: continue
                tgt_info = right_map[tgt_feat]
                
                # Evaluate source-side capacity for multiple incoming links to one target
                sources_for_this_tgt = reverse_relations[tgt_feat]
                source_vals_total = sum([left_map[s]['val'] for s in sources_for_this_tgt if s in left_map])
                
                src_fraction = tgt_info['val'] / target_vals_total if target_vals_total > 0 else 0
                tgt_fraction = src_info['val'] / source_vals_total if source_vals_total > 0 else 0
                
                src_link_val = src_info['val'] * src_fraction
                tgt_link_val = tgt_info['val'] * tgt_fraction
                
                s0_raw = src_info['out_curr_raw']
                s1_raw = s0_raw + src_link_val
                t0_raw = tgt_info['in_curr_raw']
                t1_raw = t0_raw + tgt_link_val
                
                s0_sc = float(fwd(s0_raw))
                s1_sc = float(fwd(s1_raw))
                t0_sc = float(fwd(t0_raw))
                t1_sc = float(fwd(t1_raw))
                
                links.append({
                    "source": src_info['id'], "target": tgt_info['id'], 
                    "value_src": s1_sc - s0_sc, "value_tgt": t1_sc - t0_sc,  
                    "src_y": s0_sc, "tgt_y": t0_sc, "color": nodes[src_info['id']]['color']
                })
                
                src_info['out_curr_raw'] += src_link_val
                tgt_info['in_curr_raw'] += tgt_link_val

        # 2. Standard 1-to-1 Connections
        for feat in left_map.keys():
            if feat in split_sources or feat not in right_map or feat in split_targets: continue
                
            src_info = left_map[feat]
            tgt_info = right_map[feat]
            
            s0_raw = src_info['out_curr_raw']
            s1_raw = s0_raw + src_info['val']
            t0_raw = tgt_info['in_curr_raw']
            t1_raw = t0_raw + tgt_info['val']
            
            s0_sc = float(fwd(s0_raw))
            s1_sc = float(fwd(s1_raw))
            t0_sc = float(fwd(t0_raw))
            t1_sc = float(fwd(t1_raw))
            
            links.append({
                "source": src_info['id'], "target": tgt_info['id'], 
                "value_src": s1_sc - s0_sc, "value_tgt": t1_sc - t0_sc, 
                "src_y": s0_sc, "tgt_y": t0_sc, "color": nodes[src_info['id']]['color']
            })
            
            src_info['out_curr_raw'] += src_info['val']
            tgt_info['in_curr_raw'] += tgt_info['val']
            
    return nodes, links

#endregion
#endregion
# region D.3 Data Preprocessing
# region D.3.1 Configurations
# region D.3.1.1 Global all Pre Style
global_pre_style_all = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.frameon': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
}
#endregion
# region D.3.1.2 Return Period
RP_color_palette = 'viridis'
palette = plt.get_cmap(RP_color_palette, len(RPs_unq))
rp_colors = {rp: palette(i) for i, rp in enumerate(RPs_unq)}
rp_indices = range(len(RPs_unq))
#endregion
# region D.3.1.3 Flood plains
q_colors = {
    'sto_Q05_contour': "#0084FF",
    'sto_Q50_contour': "#2F00FF",
    'det_Q50_contour': "#16B302",
    'sto_Q95_contour': "#6F00FF",
}
#endregion
# region D.3.1.4 CTEs and CTIs
CCC_color_pallete = 'turbo'
component_groups = {
    'g1_Furnishing': ['FUR', 'HHG', 'DEC', 'OTH'],
    'g2_Technology': ['APP', 'COM', 'ELE','ENG'], # 'ENG' not pressent in dataset
    'g3_Personal':   ['CLO', 'FAD', 'HHB', 'LEI'],
    'g4_Tools_Veh':  ['INS', 'TOO', 'VEH', 'SPE'],
    'g5_Surfaces':   ['PRW', 'SOI', 'ETP', 'ITP', 'PUM', 'CLE'],
    'g6_Fixtures':   ['WND', 'RDR', 'SKT', 'PLG']
}
group_titles = {
    'g1_Furnishing': 'Furnishing & Decor (Content)',
    'g2_Technology': 'Tech & Appliances (Content)',
    'g3_Personal':   'Personal & Lifestyle (Content)',
    'g4_Tools_Veh':  'Tools & Vehicles (Content)',
    'g5_Surfaces':   'Surfaces & Structural (Continent)',
    'g6_Fixtures':   'Openings & Fixtures (Continent)'
}
groups_list = list(component_groups.values())
max_group_size = max(len(g) for g in groups_list)
CCCs = []
for i in range(max_group_size):
    for g in groups_list:
        if i < len(g):
            CCCs.append(g[i])
palette = plt.get_cmap(CCC_color_pallete, len(CCCs))
ccc_colors = {ccc: palette(i) for i, ccc in enumerate(CCCs)}
#endregion
# region D.3.1.5 Maps
dst_crs = 'EPSG:4326'
#endregion
#endregion
#endregion

mocalos_results_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\.cache\mocaloss_results"

# region D.F2 - Convergence Analysis
RUN_BLOCK = True
if RUN_BLOCK:
    # HEC-RAS Convergence
    threshold, z_score = 0.1, 1.96
    hr_conv_list = []
    bid_conv_map = {}
    all_steps = sorted(df_depth_samples['SN'].unique())
    sn_step_map = {all_steps[i]: all_steps[i+1] for i in range(len(all_steps)-1)}
    for rp in tqdm(sorted(df_depth_samples['RP'].unique())):
        # 1. Calculate convergence metrics
        _, rp_simp, rp_detailed = calc_hr_convergence(df_depth_samples, rp, threshold, z_score)
        # 2. Calculate spatial average depth (he) per SN
        spatial_mean = (
            df_depth_samples[df_depth_samples['RP'] == rp]
            .groupby('SN')['he']
            .mean()
            .reset_index()
        )
        # 3. Calculate Cumulative Average Depth (accounting for all he where SN <= current SN)
        spatial_mean['he_cum_mean'] = spatial_mean['he'].expanding().mean()
        # 4. Combine results
        rp_combined = pd.merge(rp_simp, spatial_mean, on='SN')
        rp_combined['RP'] = rp
        hr_conv_list.append(rp_combined)
        # 5. Filter for BID active steps among flooded buildings
        # We find the MAX SN where it was still NOT converged (CONV_BID == False)
        last_active = rp_detailed[
            (rp_detailed['CONV_BID'] == False) & 
            (rp_detailed['BID'].isin(BIDs_flooded_unq_byRP.get(rp, [])))
        ].groupby('BID')['SN'].max().to_dict()
        
        rp_conv = {}
        for bid, last_sn in last_active.items():
            # Stable convergence is the step immediately following the last active record
            stable_sn = sn_step_map.get(last_sn)
            if stable_sn:
                rp_conv[bid] = stable_sn
                
        bid_conv_map[rp] = rp_conv
    df_hr_convergence = pd.concat(hr_conv_list, ignore_index=True)

    df_irme_wide = df_hr_convergence.pivot(index='SN', columns='RP', values='IRME')
    df_irme_wide.columns = [f'Y_IRME_{c}' for c in df_irme_wide.columns]
    df_he_wide = df_hr_convergence.pivot(index='SN', columns='RP', values='he_cum_mean')
    df_he_wide.columns = [f'Y_he_cum_mean_{c}' for c in df_he_wide.columns]
    df_to_plot_hr = pd.concat([df_irme_wide, df_he_wide], axis=1).reset_index().rename(columns={'SN': 'X_SN'})

    list_to_plot_hr = [list(bid_conv_map[rp].values()) for rp in RPs_unq]

    conv_map = {
        rp: df_to_plot_hr.loc[df_to_plot_hr[f'Y_IRME_{rp}'] > 0, 'X_SN'].max() + 31
        for rp in RPs_unq
    }


    # Economic Monte Carlo Convergence
    temp_df = load_mc_part(os.path.join(PATHS['dataset_mc_parts'], f"{BASE_FILE_NAME}_1.parquet"))
    cost_cols = [c for c in temp_df.columns if c.startswith('c_')]
    cost_cols = ['c_APP_-1', 'c_APP_0', 'c_APP_1', 'c_APP_2', 'c_APP_3',
                'c_CLO_-1', 'c_CLO_0', 'c_CLO_1', 'c_CLO_2', 'c_CLO_3',
                'c_COM_-1', 'c_COM_0', 'c_COM_1', 'c_COM_2', 'c_COM_3',
                'c_DEC_-1', 'c_DEC_0', 'c_DEC_1', 'c_DEC_2', 'c_DEC_3',
                'c_ELE_-1', 'c_ELE_0', 'c_ELE_1', 'c_ELE_2', 'c_ELE_3',
                'c_ENG_-1', 'c_ENG_0', 'c_ENG_1', 'c_ENG_2', 'c_ENG_3',
                'c_FAD_-1', 'c_FAD_0', 'c_FAD_1', 'c_FAD_2', 'c_FAD_3',
                'c_FUR_-1', 'c_FUR_0', 'c_FUR_1', 'c_FUR_2', 'c_FUR_3',
                'c_HHG_-1', 'c_HHG_0', 'c_HHG_1', 'c_HHG_2', 'c_HHG_3',
                'c_HHB_-1', 'c_HHB_0', 'c_HHB_1', 'c_HHB_2', 'c_HHB_3',
                'c_INS_-1', 'c_INS_0', 'c_INS_1', 'c_INS_2', 'c_INS_3',
                'c_LEI_-1', 'c_LEI_0', 'c_LEI_1', 'c_LEI_2', 'c_LEI_3',
                'c_OTH_-1', 'c_OTH_0', 'c_OTH_1', 'c_OTH_2', 'c_OTH_3',
                'c_SPE_-1', 'c_SPE_0', 'c_SPE_1', 'c_SPE_2', 'c_SPE_3',
                'c_TOO_-1', 'c_TOO_0', 'c_TOO_1', 'c_TOO_2', 'c_TOO_3',
                'c_VEH_-1', 'c_VEH_0',
                'c_PUM',
                'c_CLE_-1', 'c_CLE_0', 'c_CLE_1', 'c_CLE_2','c_CLE_3',
                'c_SOI_-1', 'c_SOI_0', 'c_SOI_1', 'c_SOI_2', 'c_SOI_3',
                'c_SKT_-1', 'c_SKT_0', 'c_SKT_1', 'c_SKT_2', 'c_SKT_3',
                'c_RDR_-1', 'c_RDR_0', 'c_RDR_1', 'c_RDR_2', 'c_RDR_3',
                'c_WND_0', 'c_WND_1', 'c_WND_2', 'c_WND_3',
                'c_PLG_-1', 'c_PLG_0', 'c_PLG_1', 'c_PLG_2', 'c_PLG_3',
                'c_PRW_-1', 'c_PRW_0', 'c_PRW_1', 'c_PRW_2', 'c_PRW_3',
                'c_ETP',
                'c_ITP_-1', 'c_ITP_0', 'c_ITP_1', 'c_ITP_2', 'c_ITP_3']
    total_cost_col = 'c_hu_bf'
    target_cols = ['SN', 'BID', 'RP', 'he', 'NF', 'BF'] + cost_cols + [total_cost_col]
    full_df = load_and_concatenate_mc_parts(PATHS['dataset_mc_parts'], target_cols)
    full_df = full_df.sort_values(by=['BID', 'SN', 'RP'])
    full_df[cost_cols] = full_df[cost_cols].fillna(0) # Fullfill cost nans with 0
    bids_to_remove = dataset.loc[dataset['BT'].isna(), 'BID'].unique() # Remove undefined BIDs
    full_df = full_df[~full_df['BID'].isin(bids_to_remove)]
    floor_range, basement_floors, above_ground_floors = get_floors_range(temp_df) # Number of floors
    df_summed = (
        full_df.groupby(['RP', 'SN'])['c_hu_bf']
        .sum()
        .reset_index()
        .sort_values(['RP', 'SN'])
    )
    df_summed['avg_hu'] = (
        df_summed.groupby('RP')[total_cost_col]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    df_emc_convergence, df_emc_convergence_bid, execution_logs = calc_vats_convergence(
        full_df, n_sims = 10_000, n_start = 200, calc_step = 500, n_step_check = 1, alphas = [5], epsilons =[5])
    # df_emc_convergence['res_a05_e05'] = df_emc_convergence['res_a05_e05'].round(2)
    # df_emc_convergence.loc[df_emc_convergence['SN'] == 10000, 'res_a05_e05'] = 0.00
    bid_conv_map = {}
    steps = sorted(df_emc_convergence_bid['SN'].unique())
    sn_step_map = {steps[i]: steps[i+1] for i in range(len(steps)-1)}
    for rp in tqdm(sorted(df_emc_convergence_bid['RP'].unique())): # rp=10
        # 1. Isolate the data for the current RP
        rp_detailed = df_emc_convergence_bid[df_emc_convergence_bid['RP'] == rp]
        
        # 2. Group by BID to find the last convergence point SN filtering only for flood buildings
        last_active = rp_detailed[
            (rp_detailed['status'] == 'active') & 
            (rp_detailed['BID'].isin(BIDs_flooded_unq_byRP.get(rp, [])))
        ].groupby('BID')['SN'].max().to_dict()
        
        rp_conv = {}
        for bid, last_sn in last_active.items():
            # The stable convergence is the next step after the last active state
            stable_sn = sn_step_map.get(last_sn)
            if stable_sn:
                rp_conv[bid] = stable_sn
                
        bid_conv_map[rp] = rp_conv

    df_vats_wide = df_emc_convergence.pivot(index='SN', columns='RP', values='res_a05_e05')
    df_vats_wide.columns = [f'Y_IRVATS_{int(rp)}' for rp in df_vats_wide.columns]
    df_vats_wide = df_vats_wide.sort_index().reset_index().rename(columns={'SN': 'X_SN'})
    df_avg_hu_wide = df_summed.pivot(index='SN', columns='RP', values='avg_hu')
    df_avg_hu_wide.columns = [f'Y_avg_c_hu_{int(rp)}' for rp in df_avg_hu_wide.columns]
    df_avg_hu_wide = df_avg_hu_wide.sort_index().reset_index().rename(columns={'SN': 'X_SN'})

    list_to_plot_emc = [list(bid_conv_map[rp].values()) for rp in RPs_unq]

    conv_map_emc = {
        rp: df_emc_convergence.loc[
            (df_emc_convergence['RP'] == rp) & (df_emc_convergence['res_a05_e05'] > 0), 'SN'
        ].max()# + (np.random.uniform(-500, 500) if rp != 5 else 0)
        for rp in RPs_unq
    }

    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "x", "b", "xx", "leg"],
            ["c", "x", "d", "xx", "leg"],
            ["e", "x", "f", "xx", "leg"]
        ],
        'figsize': (7, 5),
        'kwargs': {
            'gridspec_kw': {
                'hspace': 0.25,
                'wspace': 0.05,
                'width_ratios': [1, 0.35, 1, 0.01, 0.45],
                'height_ratios': [1, 1, 1]
            },
        },
        'adjust': {'bottom': 0.105, 'top': 0.945, 'left': 0.115, 'right': 0.97}
    }
    
    # Plots
    palette = plt.get_cmap(RP_color_palette, len(RPs_unq))
    rp_colors = {rp: palette(i) for i, rp in enumerate(RPs_unq)}
    dict_to_plot = {
        'a': {'plots': [], 'style': []},
        'b': {'plots': [], 'style': []},
        'c': {'plots': [], 'style': []},
        'd': {'plots': [], 'style': []},
        'e': {'plots': [], 'style': []},
        'f': {'plots': [], 'style': []},
        'leg': {'style': [], 'legend': []},
        'x': {'style': []},
        'xx': {'style': []},
    }
    for rp in RPs_unq:
        color = rp_colors[rp]
        dict_to_plot['a']['plots'].append({'plot_type': 'line',
            'x': df_to_plot_hr['X_SN'],
            'y': df_to_plot_hr[f'Y_IRME_{rp}'],
            'label': f'RP {rp}',
            'linewidth': 1,
            'color': color
        })
        dict_to_plot['b']['plots'].append({'plot_type': 'line',
            'x': df_vats_wide['X_SN'],
            'y': df_vats_wide[f'Y_IRVATS_{rp}'],
            'label': f'RP {rp}',
            'linewidth': 1,
            'color': color
        })
        dict_to_plot['c']['plots'].append({'plot_type': 'line',
            'x': df_to_plot_hr['X_SN'],
            'y': df_to_plot_hr[f'Y_he_cum_mean_{rp}'],
            'label': f'RP {rp}',
            'linewidth': 1,
            'color': color
        })
        dict_to_plot['d']['plots'].append({'plot_type': 'line',
            'x': df_avg_hu_wide['X_SN'],
            'y': df_avg_hu_wide[f'Y_avg_c_hu_{rp}'], # rp=5
            'label': f'RP {rp}',
            'linewidth': 1,
            'color': color
        })
    for rp, sn in conv_map.items():
        for key in ['a', 'c']:
            dict_to_plot[key]['plots'].append({'plot_type': 'vline',
                'x': sn,
                'color': rp_colors[rp],
                'linestyle': '--',
                'linewidth': 0.8,
                'alpha': 0.6
            })
    for rp, sn in conv_map_emc.items():
        for key in ['b', 'd']:
            dict_to_plot[key]['plots'].append({'plot_type': 'vline',
                'x': sn,
                'color': rp_colors[rp],
                'linestyle': '--',
                'linewidth': 0.8,
                'alpha': 0.6
            })
    for key in ['a', 'c', 'e']:
        dict_to_plot[key]['plots'].insert(0, {'plot_type': 'vspan',
            'x1': 0, 'x2': 30,
            'color': '#E0E0E0', 'alpha': 0.5, 'zorder': 0})
    for key in ['b', 'd', 'f']:
        dict_to_plot[key]['plots'].insert(0, {'plot_type': 'vspan',
            'x1': 0, 'x2': 200,
            'color': '#E0E0E0', 'alpha': 0.5, 'zorder': 0})
    
    violin_colors = [rp_colors[rp] for rp in RPs_unq]
    dict_to_plot['e']['plots'].append({'plot_type': 'violin',
            'dataset': list_to_plot_hr,
            'positions': list(rp_indices),
            'widths': 0.9,
            'color': violin_colors,
            'vert': False,
            'showmeans': False,
            'showmedians': True,
            'showextrema': True,
            'points': 10000,
            'bw_method': 0.1
        })
    dict_to_plot['f']['plots'].append({'plot_type': 'violin',
            'dataset': list_to_plot_emc,
            'positions': list(rp_indices),
            'widths': 0.9,
            'color': violin_colors,
            'vert': False,
            'showmeans': False,
            'showmedians': True,
            'showextrema': True,
            'points': 10000,
            'bw_method': 0.3
        })
    
    # Style
    dict_to_plot['a']['style'].extend([
        {'style_type': 'title', 'label': 'HEC-RAS', 'fontsize': 8, 'pad': 4},
        {'style_type': 'ylabel', 'label': 'Integrated\nResidual', 'fontsize': 8},
        {'style_type': 'xlim', 'left': 2, 'right': 1_200},
        {'style_type': 'ylim', 'bottom': 0, 'top': 1_500},
        {'style_type': 'xscale', 'value': 'symlog', 'linthresh': 10, 'linscale': 0.1},
        {'style_type': 'yscale', 'value': 'symlog', 'linthresh': 0.1, 'linscale': 0.4},
        {'style_type': 'xticks', 'ticks': [0, 10, 100, 1_000]},
        {'style_type': 'xticklabels', 'labels': []},
        {'style_type': 'yticks', 'ticks': [0, 0.1, 1, 10, 100, 1_000]},
        {'style_type': 'yticklabels', 'labels': ['$0$', '', '$10^0$', '$10^1$', '$10^2$', '$10^3$']},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['b']['style'].extend([
        {'style_type': 'title', 'label': 'Economic Assessment', 'fontsize': 8, 'pad': 4},
        {'style_type': 'ylabel', 'label': 'Integrated\nResidual', 'fontsize': 8},
        {'style_type': 'xlim', 'left': 2, 'right': 10_000},
        {'style_type': 'ylim', 'bottom': 0, 'top': 20_000},
        {'style_type': 'xscale', 'value': 'symlog', 'linthresh': 100, 'linscale': 0.1},
        {'style_type': 'yscale', 'value': 'symlog', 'linthresh': 1, 'linscale': 0.5},
        {'style_type': 'xticks', 'ticks': [0, 100, 1_000, 10_000]},
        {'style_type': 'xticklabels', 'labels': []},
        {'style_type': 'yticks', 'ticks': [0, 1, 10, 100, 1_000, 10_000]},
        {'style_type': 'yticklabels', 'labels': ['$0$', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$']},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['c']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Average\nSpatial\nDepth (m)', 'fontsize': 8, 'labelpad': 0},
        {'style_type': 'xlim', 'left': 2, 'right': 1200},
        {'style_type': 'ylim', 'bottom': 0, 'top': 0.85},
        {'style_type': 'xscale', 'value': 'symlog', 'linthresh': 10, 'linscale': 0.1},
        {'style_type': 'yscale', 'value': 'symlog', 'linthresh': 0.005, 'linscale': 0.6},
        {'style_type': 'xticks', 'ticks': [0, 10, 100, 1_000]},
        {'style_type': 'xticklabels', 'labels': []},
        {'style_type': 'yticks', 'ticks': [0, 0.01, 0.1, 0.8]},
        {'style_type': 'yticklabels', 'labels': ['$0$', '$0.01$', '$0.1$', '$0.8$']},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['d']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Average\nTotal\nCost (€)', 'fontsize': 8},
        {'style_type': 'xlim', 'left': 2, 'right': 10_000},
        {'style_type': 'ylim', 'bottom': 100, 'top': 10_000_000},
        {'style_type': 'xscale', 'value': 'symlog', 'linthresh': 100, 'linscale': 0.15},
        {'style_type': 'yscale', 'value': 'symlog', 'linthresh': 10_000, 'linscale': 0.35},
        {'style_type': 'xticks', 'ticks': [0, 100, 1_000, 10_000]},
        {'style_type': 'xticklabels', 'labels': []},
        {'style_type': 'yticks', 'ticks': [0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]},
        {'style_type': 'yticklabels', 'labels': ['$0$', '', '', '', '', '$10^4$', '$10^5$', '$10^6$', '$10^7$']},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['e']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Distribution of\nachieved\nconvergence', 'fontsize': 8},
        {'style_type': 'xlabel', 'label': 'Simulation Number', 'fontsize': 8},
        {'style_type': 'xlim', 'left': 2, 'right': 1_200},
        {'style_type': 'xscale', 'value': 'symlog', 'linthresh': 10, 'linscale': 0.1},
        {'style_type': 'xticks', 'ticks': [0, 10, 100, 1_000]},
        {'style_type': 'xticklabels', 'labels': ['', '$10^1$', '$10^2$', '$10^3$']},
        {'style_type': 'yticks', 'ticks': list(rp_indices)},
        {'style_type': 'yticklabels', 'labels': [f'${rp}$' for rp in RPs_unq]},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['f']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Distribution of\nachieved\nconvergence', 'fontsize': 8},
        {'style_type': 'xlabel', 'label': 'Simulation Number', 'fontsize': 8},
        {'style_type': 'xlim', 'left': 2, 'right': 10_000},
        {'style_type': 'xscale', 'value': 'symlog', 'linthresh': 100, 'linscale': 0.1},
        {'style_type': 'xticks', 'ticks': [0, 100, 1_000, 10_000]},
        {'style_type': 'xticklabels', 'labels': ['', '$10^2$', '$10^3$', '$10^4$']},
        {'style_type': 'yticks', 'ticks': list(rp_indices)},
        {'style_type': 'yticklabels', 'labels': [f'${rp}$' for rp in RPs_unq]},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    for key in dict_to_plot.keys():
        if key not in ['leg', 'x', 'xx']:
            dict_to_plot[key]['style'].append({
                'style_type': 'letter', 
                'label': f'({key})',
                'fontsize': 8,
                'x': 0,
                'y': 1.13
            })
    dict_to_plot['leg']['style'].append({'style_type': 'off'})
    dict_to_plot['x']['style'].append({'style_type': 'off'})
    dict_to_plot['xx']['style'].append({'style_type': 'off'})
    
    # Legend
    # Return Periods
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Return Period',
            'x': 0, 'y': 0.97,
            'ha': 'left', 'va': 'bottom', 'fontsize': 8,
        },   
    ])
    
    x_line_value = 0.0
    x_text = 0.45
    x_line_stable = 0.75
    line_length = 0.25
    y_start = 0.93
    y_step = 0.05
    for i, rp in enumerate(RPs_unq):
        y_pos = y_start - (i * y_step)
        color = rp_colors[rp]
        dict_to_plot['leg']['legend'].append({
            'legend_type': 'line',
            'x': x_line_value,
            'y': y_pos,
            'length': line_length,
            'color': color,
            'linestyle': '-',
            'linewidth': 1
        })
        dict_to_plot['leg']['legend'].append({
            'legend_type': 'text',
            'x': x_text,
            'y': y_pos,
            'content': f'{rp}',
            'ha': 'center',
            'va': 'center',
            'fontsize': 8
        })
    
    # Convergence
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Convergence',
            'x': 0, 'y': 0.53,
            'ha': 'left', 'va': 'bottom', 'fontsize': 8,
        },    
    ])
    x_line_value = 0.0
    x_line_valueb = 0.5
    line_length = 0.25
    y_start = 0.5
    y_startb = 0.5
    y_step = 0.03
    for i, rp in enumerate(RPs_unq):
        x_pos = x_line_value if rp in [5, 10, 20] else x_line_valueb
        y_pos = y_start - (i * y_step) if rp in [5, 10, 20] else y_startb - ((i-len([5, 10, 20])) * y_step)
        color = rp_colors[rp]
        dict_to_plot['leg']['legend'].append({
            'legend_type': 'line',
            'x': x_pos,
            'y': y_pos,
            'length': line_length,
            'color': color,
            'linestyle': '--',
            'linewidth': 1,
            'alpha': 1
        })
    
    # Warm Up
    dict_to_plot['leg']['legend'].extend([
        {
            'legend_type': 'text',
            'x': x_line_value,
            'y': 0.32,
            'content': 'Warm Up',
            'ha': 'left', 'va': 'bottom',
            'fontsize': 8
        },
        {
            'legend_type': 'line',
            'x': x_line_value+0.04,
            'y': 0.28,
            'length': line_length+0.4,
            'color': '#E0E0E0',
            'alpha': 0.5,
            'linestyle': '-',
            'linewidth': 10
        },
    ])
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'F2_Convergence Analysis',
        'fname': 'F5_Damage_Functions_v3.1.png',
        'show': True,
        'dpi': 300,
    }
    
    # Plot
    plot_any(dict_to_plot, layout_params, save_params)
#endregion

# region D.T2 - Estimated Damage (Table)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    # Data sources
    c_cols = ['c_hu','c_huCTEs','c_huCTIs']
    c_cols_CTE = [
        'c_huAPP_f', 'c_huCLO_f', 'c_huCOM_f', 'c_huDEC_f', 'c_huELE_f',
        'c_huFAD_f', 'c_huFUR_f', 'c_huHHG_f', 'c_huHHB_f', 'c_huINS_f',
        'c_huLEI_f', 'c_huOTH_f', 'c_huSPE_f', 'c_huTOO_f', 'c_huVEH_f']
    c_cols_CTI = [
        'c_huETP_f', 'c_huWND_f', 'c_huPRW_f', 'c_huITP_f',
        'c_huSOI_f', 'c_huSKT_f', 'c_huRDR_f', 'c_huPLG_f']
    full_df = load_and_concatenate_mc_parts(
        PATHS['dataset_mc_parts'], ['SN','BID','RP'] + c_cols + c_cols_CTE + c_cols_CTI)
    valid_bids = set(full_df['BID'].drop_duplicates())
    buildings_subset = gdf_buildings[gdf_buildings['BID'].isin(valid_bids)]
    full_df = pd.merge(
        left=full_df, 
        right=buildings_subset[['BID', 'GA', 'RPZone', 'NEAR_DIST']],
        on='BID', 
        how='left'
    )
    full_df['c_hu_by_ga'] = full_df['c_hu'] / full_df['GA']
    
    # Data Frame Modification (Desired Table)
    # Average Quantile cost by meter and zone
    exclude_cols = ['SN', 'BID', 'RP', 'GA', 'RPZone', 'NEAR_DIST'] + c_cols_CTE + c_cols_CTI + c_cols
    cost_cols = [c for c in full_df.columns if c not in exclude_cols]
    reduced_df = full_df.groupby(['BID', 'RP', 'GA', 'RPZone', 'NEAR_DIST'])[cost_cols].quantile([0.05, 0.50, 0.95])
    reduced_df = reduced_df.unstack()
    reduced_df.columns = reduced_df.columns.set_levels(['Q05', 'Q50', 'Q95'], level=1)
    reduced_df = reduced_df.reset_index()
    
    rpzone_df = reduced_df.reset_index().groupby(['RPZone', 'RP']).mean()
    rpzone_df = rpzone_df.round(0).astype(int)
    rpzone_df = rpzone_df.drop(columns=['BID', 'GA', 'NEAR_DIST', 'index'], errors='ignore')
    rpzone_df = rpzone_df.reset_index()
    rpzone_df['RPZone'] = pd.to_numeric(rpzone_df['RPZone'])
    rpzone_df = rpzone_df.set_index(['RP', 'RPZone']).sort_index(level=['RP', 'RPZone'])
    rps = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    rpzones = [5, 10, 20, 50, 100, 200, 500, 9999]    
    full_index = pd.MultiIndex.from_product([rps, rpzones], names=['RP', 'RPZone'])
    rpzone_df_complete = rpzone_df.reindex(full_index, fill_value=0).reset_index()
    rpzone_labels = {
        5: "a<=5",
        10: "5<a<=10",
        20: "10<a<=20",
        50: "20<a<=50",
        100: "50<a<=100",
        200: "100<a<=200",
        500: "200<a<=500",
        9999: "a>500"
    }
    rpzone_df_complete['RPZone_Label'] = rpzone_df_complete['RPZone'].map(rpzone_labels)
    
    title = "Cost by building per return period zone"
    
    
    file_path = r"C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\2_Draft_Article\Tables\T2 Estimated Damage\T2_Estimated_Damage.xlsx"
    rpzone_df_complete.columns = [
        '_'.join(filter(None, col)).strip() if isinstance(col, tuple) else col 
        for col in rpzone_df_complete.columns.values
    ]
    rpzone_df_complete.to_excel(file_path, index=False)
    print(f"File saved: {file_path}")

    # Chart    
    rp_map = {val: i for i, val in enumerate(rps)}
    rpzone_map = {val: i for i, val in enumerate(rpzone_labels.keys())}
    
    # 2. Map raw data to integer indices
    # Assuming rpzone_df_complete contains the original numeric 'RP' and 'RPZone'
    x_idx = rpzone_df_complete['RP'].map(rp_map).values
    y_idx = rpzone_df_complete['RPZone'].map(rpzone_map).values

    # 3. Create grid based on index ranges
    xi_idx = np.arange(len(rps))
    yi_idx = np.arange(len(rpzone_labels))
    XI, YI = np.meshgrid(xi_idx, yi_idx)

    z_max_raw = rpzone_df_complete['c_hu_by_ga'].values.max()
    z_max_log = np.log10(z_max_raw + 1)


    fig = plt.figure(figsize=(24, 8))
    quantiles = ['Q05', 'Q50', 'Q95']

    for i, q in enumerate(quantiles):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        z_raw_q = rpzone_df_complete[('c_hu_by_ga', q)].values
        z_log_q = np.log10(z_raw_q + 1)
        
        # Interpolate using integer indices
        zi = griddata((x_idx, y_idx), z_log_q, (XI, YI), method='linear')
        
        # Surface
        surf = ax.plot_surface(XI, YI, zi, cmap='viridis', edgecolor='none', alpha=0.6, vmin=0, vmax=z_max_log)
        
        # Stem plot using indices
        for xj, yj, zj in zip(x_idx, y_idx, z_log_q):
            ax.plot([xj, xj], [yj, yj], [0, zj], color='black', linewidth=0.5, alpha=0.5)
            ax.scatter(xj, yj, zj, color='red', s=10, alpha=0.8)
        
        # 4. Formatting with categorical labels
        # X Formatting (RP)
        ax.set_xticks(xi_idx)
        ax.set_xticklabels([str(int(val)) for val in rps], fontsize=8)
        
        # Y Formatting (RPZone with new categorical labels)
        ax.set_yticks(yi_idx)
        ax.set_yticklabels(list(rpzone_labels.values()), rotation=-30, ha='left', va='center', fontsize=8)
        
        # Z Formatting (Log Scale)
        ax.set_zticks([0, 1, 2, 3, 4])
        ax.set_zticklabels(['0', '10', '100', '1k', '10k'], fontsize=8)
        
        # Maintain orientation or invert as needed
        ax.set_xlim(len(rps) - 1, 0)
        ax.set_ylim(len(rpzones) - 1, 0)
        ax.set_zlim(0, z_max_log)
        
        ax.set_title(f'Quantile {q}')
        ax.set_xlabel('Return Period', fontsize=8)
        ax.set_ylabel('Return Period Area (a)', labelpad=25, fontsize=8)
        ax.set_zlabel('Total Average Damage by BID (€/m2)', fontsize=8)

    plt.tight_layout()
    plt.show()
#endregion

# region D.F3 - Stochastic Total Damage Map (reference)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    # Data Sources
    value_col = 'c_hu_bf'
    PATHS['dataset_mc_parts'] = r"C:\Users\outal\GITHUB\Jose - UCLM\In-Depth-PROFILE\data\processed\MC_Parts"
    full_df_old = load_and_concatenate_mc_parts(PATHS['dataset_mc_parts'], ['SN','BID','RP', value_col])
    
    # ADAPTER for new results
    value_col = 'c_hu_total'
    cols = ['it','BID','RP', value_col]
    # scan lazy and collect only the cols, then rename it to SN
    full_df_pl = (
        pl.scan_ipc(os.path.join(mocalos_results_path, "*.ipc"))
        .select(cols)
        .with_columns(pl.col(value_col).fill_null(0))
        .rename({"it": "SN"})
        .collect()
    )
    full_df = full_df_pl.to_pandas()
    
    
    valid_bids = set(full_df['BID'].drop_duplicates())
    gdf_buildings_subset = gdf_buildings[gdf_buildings['BID'].isin(valid_bids)]
    
    # Total cost by BID
    df_cost = full_df.groupby(['BID', 'RP'])[value_col].quantile([0.05, 0.50, 0.95]).unstack().reset_index()
    df_cost.columns = ['BID', 'RP', 'Q05', 'Q50', 'Q95']
    df_wide = df_cost.pivot(index='BID', columns='RP', values=['Q05', 'Q50', 'Q95'])
    df_wide.columns = [f'RP{int(rp)}_{q}' for q, rp in df_wide.columns]
    df_wide = df_wide.reset_index()

    gdf_cost = gdf_buildings_subset.merge(df_wide, on='BID', how='left')
    gdf_cost_centroids = gdf_cost.copy()
    gdf_cost_centroids['geometry'] = gdf_cost.centroid
    gdf_cost_centroids_4326 = gdf_cost_centroids.to_crs(epsg=4326)
    max_rp = int(full_df['RP'].max())
    sort_col = f'RP{max_rp}_Q95'
    if sort_col in gdf_cost_centroids_4326.columns:
        gdf_cost_centroids_4326 = gdf_cost_centroids_4326.sort_values(by=sort_col, ascending=True)
    
    bins = [0, 100, 1_000, 10_000, 100_000, np.inf]
    n_bins = len(bins) - 1
    labels = [f"{bins[i]}-{bins[i+1]}" if bins[i+1] != np.inf else f">{bins[i]}" for i in range(n_bins)]
    
    cmap = plt.get_cmap('YlOrRd')
    colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0.2, 1, n_bins)]
    color_map = dict(zip(labels, colors))
    
    target_cols = [col for col in gdf_cost_centroids_4326.columns if '_Q' in col]
    for col in target_cols:
        cat_col = f"{col}_cat"
        color_col = f"color_{col}_cat"
        gdf_cost_centroids_4326[cat_col] = pd.cut(
            gdf_cost_centroids_4326[col], 
            bins=bins, 
            labels=labels, 
            right=False
        )
        gdf_cost_centroids_4326[color_col] = gdf_cost_centroids_4326[cat_col].map(color_map)
    
    # Terrain
    minx, miny, maxx, maxy = gdf_cost_centroids_4326.total_bounds
    buffer = 0.01
    plot_extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]
    
    # Flood Plains
    det_RP500_path = r"C:\Users\outal\GITHUB\Jose - UCLM\In-Depth-PROFILE\data\inputs\gis\OficialFloods\Q500_2Ciclo_PB_2026_Navaluenga.shp"
    
    # Find the SN whose IF is clossest to 1751,  3671, and  7908,
    targets = [1751, 3671, 7908]
    indices = [(df_depth_samples['IF'] - target).abs().idxmin() for target in targets]
    closest_rows = df_depth_samples.loc[indices]
    
    sto_RP500_Q05_path = r"C:\Users\outal\GITHUB\Jose - UCLM\In-Depth-PROFILE\models\HEC-RAS_6.6\RC_2_Output_WSE\WSE_RP500_SN213.tif"
    sto_RP500_path = r"C:\Users\outal\GITHUB\Jose - UCLM\In-Depth-PROFILE\models\HEC-RAS_6.6\RC_2_Output_WSE\WSE_RP500_SN801.tif"
    sto_RP500_Q95_path = r"C:\Users\outal\GITHUB\Jose - UCLM\In-Depth-PROFILE\models\HEC-RAS_6.6\RC_2_Output_WSE\WSE_RP500_SN242.tif"
    
    det_RP500 = gpd.read_file(det_RP500_path).to_crs("EPSG:4326")

    sto_RP500_Q05 = tif_to_simplified_shp(sto_RP500_Q95_path)
    sto_RP500 = tif_to_simplified_shp(sto_RP500_path)
    sto_RP500_Q95 = tif_to_simplified_shp(sto_RP500_Q05_path)

    # Plot both together
    det_RP500.crs
    sto_RP500.crs
    
    fig, ax = plt.subplots(figsize=(10, 10))
    det_RP500.plot(ax=ax, color='blue', alpha=0.4, edgecolor='darkblue', label='det_RP500')
    sto_RP500.plot(ax=ax, color='red', alpha=0.4, edgecolor='darkred', label='sto_RP500')
    plt.show()
    
    with rasterio.open(sto_RP500_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        depth_array_4326 = np.zeros((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=depth_array_4326,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )
        depth_extent_4326 = [
            transform[2],                             # xmin
            transform[2] + transform[0] * width,      # xmax
            transform[5] + transform[4] * height,     # ymin
            transform[5]                              # ymax
        ]
        depth_array_4326 = np.where(depth_array_4326 == src.nodata, np.nan, depth_array_4326)
    
    # Lorenz Curves
    df_dam_ineq = full_df.groupby(['RP', 'BID'])[value_col].quantile([0.05, 0.50, 0.95]).unstack()
    df_dam_ineq.columns = ['Q05', 'Q50', 'Q95']
    df_dam_ineq = df_dam_ineq.reset_index()
    
    lorenz_list = []
    quantiles = ['Q05', 'Q50', 'Q95']
    unique_rps = sorted(df_dam_ineq['RP'].unique())

    for rp in tqdm(unique_rps):
        rp_df = df_dam_ineq[df_dam_ineq['RP'] == rp].copy()
        n = len(rp_df)
        
        # Base dataframe for this RP
        rp_res = pd.DataFrame({'RP': [rp] * n})
        
        for q in quantiles:
            # Sort values for this specific quantile to define the curve
            sorted_vals = rp_df[q].sort_values().values
            
            # X: Cumulative proportion of population (BIDs)
            x_coords = np.arange(1, n + 1) / n
            
            # Y: Cumulative proportion of damage
            total_damage = sorted_vals.sum()
            if total_damage > 0:
                y_coords = np.cumsum(sorted_vals) / total_damage
            else:
                y_coords = np.zeros(n)
                
            rp_res[f'X_{q}'] = x_coords
            rp_res[f'Y_{q}'] = y_coords
            
        lorenz_list.append(rp_res)
    df_lorenz = pd.concat(lorenz_list, ignore_index=True)
    
    df_lorenz_final = df_lorenz.pivot(columns='RP')
    df_lorenz_final.columns = [f"{col}_{int(rp)}" for col, rp in df_lorenz_final.columns]
    df_lorenz_final = df_lorenz_final.apply(lambda x: pd.Series(x.dropna().values))
    
    # GINI Coeficients
    gini_dict = {}
    for rp in unique_rps:
        rp_data = df_lorenz[df_lorenz['RP'] == rp]
        gini_dict[int(rp)] = {}
        
        for q in quantiles:
            # Prepend 0 to coordinates for integration
            x = np.concatenate([[0], rp_data[f'X_{q}'].values])
            y = np.concatenate([[0], rp_data[f'Y_{q}'].values])
            
            # Calculate Area Under the Curve using trapezoidal rule
            auc = np.trapezoid(y, x)
            
            # Gini = (Area between identity and Lorenz) / (Area under identity)
            # Area under identity (0,0 to 1,1) is 0.5. 
            # Gini = (0.5 - auc) / 0.5 => 1 - 2*auc
            gini_val = 1 - 2 * auc
            gini_dict[int(rp)][q] = float(gini_val)
    gini_texts = {}
    for rp, gini_vals in gini_dict.items():
        q05 = f"{gini_vals['Q05']:.2f}" if isinstance(gini_vals['Q05'], float) else str(gini_vals['Q05'])
        q50 = f"{gini_vals['Q50']:.2f}" if isinstance(gini_vals['Q50'], float) else str(gini_vals['Q50'])
        q95 = f"{gini_vals['Q95']:.2f}" if isinstance(gini_vals['Q95'], float) else str(gini_vals['Q95'])
        
        gini_str = f"({q05}) {q50} ({q95})"
        gini_texts[int(rp)] = gini_str
    
    # Total cost
    df_total_cost = full_df.groupby(['SN', 'RP'])[[value_col]].sum().reset_index()
    wide_data = {}
    for rp in RPs_unq:
        rp_group = df_total_cost[df_total_cost['RP'] == rp]
        col_name = f"Y_{rp}_{value_col}"
        wide_data[col_name] = rp_group[value_col].reset_index(drop=True)
    df_total_cost_wide = pd.DataFrame(wide_data)
    df_indep_cost_wide = df_total_cost_wide.copy()
    
    df_qs = df_total_cost_wide.quantile([0.05, 0.50, 0.95])
    total_cost_quantiles_dict = {
        col: {
            'Q05': df_qs.loc[0.05, col],
            'Q50': df_qs.loc[0.50, col],
            'Q95': df_qs.loc[0.95, col]
        }
        for col in df_total_cost_wide.columns
    }
    
    bids_inside_det = gdf_cost_centroids_4326[
        gdf_cost_centroids_4326.within(det_RP500.union_all())
    ]['BID'].unique()
    
    # 2. Prepare configuration
    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "a",  "a", "a",  "a", "a",  "a", "x4", "leg"],
            ["b", "x1", "c", "x2", "d", "x3", "e", "x4", "leg"],
        ],
        'figsize': (6, 5.5),
        'kwargs': {
            'gridspec_kw': {
                'wspace': 0.05,
                'hspace': 0.16,
                #               "b", "x1","c", "x2","d", "x3","e", "x4","leg"
                'width_ratios': [1,  0.08,  0.2,  0.25,  0.2,  0.28,  1,  0.1,  0.35],
                'height_ratios': [2.25, 1],
            },
        },
        'adjust': {'bottom': 0.1, 'top': 0.9, 'left': 0.1, 'right': 0.9}
    }
    
    # Global style
    global_pre_style = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'cm',
        'axes.unicode_minus': False,
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    
    # Initialize dict_to_plot
    dict_to_plot = {}
    
    # Plots
    rp_to_plot = '500'
    q_to_plot = '50'
    column_to_plot = 'RP' + rp_to_plot + '_Q' + q_to_plot
    q05_col_to_plot = 'RP' + rp_to_plot + '_Q05'
    q95_col_to_plot = 'RP' + rp_to_plot + '_Q95'
    
    # a
    init_dic(dict_to_plot, 'a', 'plots')
    dict_to_plot['a']['plots'].extend([
        {'plot_type': 'wms',
            'url': 'https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS',
            'layers': ['OI.OrthoimageCoverage'],
            'extent': plot_extent,
            'crs': 'EPSG:4326',
            'size': (1000, 1000),
            'alpha': 1,
            'zorder': 1
        },
        {'plot_type': 'imshow',
            'image': depth_array_4326,
            'extent': depth_extent_4326,
            'cmap': 'Blues',
            'alpha': 0.6,
            'zorder': 2,
        },
        {'plot_type': 'gdf_shp',
            'gdf': det_RP500,
            'facecolor': 'none',
            'edgecolor': q_colors['det_Q50_contour'],
            'linewidth': 1.5,
            'linestyle': '--',
            'zorder': 4
        },
        {'plot_type': 'gdf_shp',
            'gdf': sto_RP500,
            'facecolor': 'none',
            'edgecolor': q_colors['sto_Q50_contour'],
            'linewidth': 1.5,
            'linestyle': '--',
            'zorder': 4
        },
        {'plot_type': 'gdf_shp',
            'gdf': sto_RP500_Q05,
            'facecolor': 'none',
            'edgecolor': q_colors['sto_Q05_contour'],
            'linewidth': 1.5,
            'linestyle': '--',
            'zorder': 4
        },
        {'plot_type': 'gdf_shp',
            'gdf': sto_RP500_Q95,
            'facecolor': 'none',
            'edgecolor': q_colors['sto_Q95_contour'],
            'linewidth': 1.5,
            'linestyle': '--',
            'zorder': 4
        },
        {'plot_type': 'gdf_shp',
            'gdf': gdf_cost_centroids_4326,
            'column': column_to_plot,
            'color': gdf_cost_centroids_4326[color_col],
            'markersize': 3,
            'zorder': 3
        },
    ])
    
    # b
    init_dic(dict_to_plot, 'b', 'plots')
    current_color = rp_colors[int(rp_to_plot)]
    dict_to_plot['b']['plots'].extend([
        {'plot_type': 'line',
            'x': df_lorenz_final[f'X_Q{q_to_plot}_{rp_to_plot}'],
            'y': df_lorenz_final[f'Y_Q{q_to_plot}_{rp_to_plot}'],
            'color': current_color,
            'linewidth': 1,
            'label': f'RP {rp_to_plot}'
        },
        {'plot_type': 'fill',
            'x': df_lorenz_final[f'X_Q{q_to_plot}_{rp_to_plot}'],
            'y1': df_lorenz_final[f'Y_Q05_{rp_to_plot}'],
            'y2': df_lorenz_final[f'Y_Q95_{rp_to_plot}'],
            'color': current_color,
            'alpha': 0.3
        },
        {'plot_type': 'line',
            'x': [0, 1],
            'y': [0, 1],
            'color': 'grey',
            'linestyle': '--',
            'linewidth': 0.8,
            'alpha': 1
        }
    ])
    
    # c
    init_dic(dict_to_plot, 'c', 'plots')
    current_color = rp_colors[int(rp_to_plot)]
    gini_vals = gini_dict[int(rp_to_plot)]
    dict_to_plot['c']['plots'].extend([
        {'plot_type': 'hspan',
            'y1': gini_vals['Q95'],
            'y2': gini_vals['Q05'],
            'color': current_color,
            'alpha': 0.3
        },
        {'plot_type': 'hline',
            'y': gini_vals['Q50'],
            'color': current_color,
            'linewidth': 2
        }
    ])
    
    # d
    init_dic(dict_to_plot, 'd', 'plots')
    target_col_d = f'Y_{rp_to_plot}_{value_col}'
    
    data_d = df_indep_cost_wide[target_col_d].dropna()
    
    current_qs = df_qs[target_col_d]
    
    q_configs = [
        {'val': current_qs[0.05], 'ls': '--', 'lw': 1.0, 'color': q_colors['sto_Q05_contour']},
        {'val': current_qs[0.50], 'ls': '-',  'lw': 1.5, 'color': q_colors['sto_Q50_contour']},
        {'val': current_qs[0.95], 'ls': '--', 'lw': 1.0, 'color': q_colors['sto_Q95_contour']}
    ]
    
    current_color = rp_colors[int(rp_to_plot)]
    dict_to_plot['d']['plots'].append({'plot_type': 'violin',
        'dataset': [data_d.values],
        'positions': [0],
        'widths': 0.95,
        'vert': True,
        'color': current_color,
        'showmeans': False,
        'showmedians': False,
        'showextrema': False,
        'points': 10000,
        'bw_method': 0.1,
    })
    
    for cfg in q_configs:
        dict_to_plot['d']['plots'].append({'plot_type': 'hline',
            'y': cfg['val'],
            'color': cfg['color'],
            'linestyle': cfg['ls'],
            'linewidth': cfg['lw'],
            'zorder': 5
        })
    
    
    df_det_subset_cost = full_df[
        (full_df['RP'] == int(rp_to_plot)) & 
        (full_df['BID'].isin(bids_inside_det))
    ].groupby('SN')[value_col].sum()
    det_q50_value = df_det_subset_cost.median()
    dict_to_plot['d']['plots'].append({'plot_type': 'hline',
        'y': det_q50_value,
        'color': q_colors['det_Q50_contour'],
        'linestyle': '-',
        'linewidth': 1.5,
        'label': 'Det. Boundary Q50',
        'zorder': 10 
    })
    
    # e
    init_dic(dict_to_plot, 'e', 'plots')
    def get_kde_jittered_x(y_series, base_x, max_jitter=0.08):
        """
        Calculates horizontal jitter based on Kernel Density Estimation (KDE).
        Falls back to uniform jitter if variance is zero/singular.
        """
        y_clean = y_series.dropna()
        
        # Fallback for empty or single-value series
        if len(y_clean) < 2:
            return y_clean, np.full(len(y_clean), base_x)
        
        try:
            # Fit KDE and evaluate density
            kde = gaussian_kde(y_clean)
            density = kde(y_clean)
            
            # Normalize density to maximum jitter threshold
            jitter_amp = (density / density.max()) * max_jitter
            jittered_x = base_x + np.random.uniform(-jitter_amp, jitter_amp)
            
        except np.linalg.LinAlgError:
            # Occurs when data variance is 0 (all values are constant)
            jittered_x = base_x + np.random.uniform(-max_jitter, max_jitter, size=len(y_clean))
            
        return y_clean, jittered_x
    y_q05, x_q05 = get_kde_jittered_x(df_wide[q05_col_to_plot], base_x=1, max_jitter=0.2)
    y_q50, x_q50 = get_kde_jittered_x(df_wide[column_to_plot],  base_x=2, max_jitter=0.2)
    y_q95, x_q95 = get_kde_jittered_x(df_wide[q95_col_to_plot], base_x=3, max_jitter=0.2)
    
    c_q05 = pd.cut(df_wide[q05_col_to_plot], bins=bins, labels=colors, right=False)
    c_q50 = pd.cut(df_wide[column_to_plot], bins=bins, labels=colors, right=False)
    c_q95 = pd.cut(df_wide[q95_col_to_plot], bins=bins, labels=colors, right=False)
    
    dict_to_plot['e']['plots'].extend([
        {'plot_type': 'scatter', 'x': x_q05, 'y': df_wide[q05_col_to_plot], 
            'c': c_q05, 's': 4, 'alpha': 0.7, 'edgecolor': 'none', 'zorder': 2
        },
        {'plot_type': 'scatter', 'x': x_q50, 'y': df_wide[column_to_plot], 
            'c': c_q50, 's': 4, 'alpha': 0.7, 'edgecolor': 'none', 'zorder': 2
        },
        {'plot_type': 'scatter', 'x': x_q95, 'y': df_wide[q95_col_to_plot], 
            'c': c_q95, 's': 4, 'alpha': 0.7, 'edgecolor': 'none', 'zorder': 2
        }
    ])
    
    # Style
    # a
    top_left_coords, zoom_level = {'x': (4, 42, 50), 'y': (40, 24, 52)}, 78.5
    xmin, xmax, ymin, ymax = calc_map_extent(
        top_left_coords['x'], top_left_coords['y'], zoom_level,
        get_ax_ratio(layout_params, 'a'))

    init_dic(dict_to_plot, 'a', 'style')
    dict_to_plot['a']['style'].extend([
        {'style_type': 'title', 'label': f'Quantile Q{q_to_plot} Map (RP {rp_to_plot})', 'fontsize': 8, 'pad': 10},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3, 'zorder': 1},
        {'style_type': 'xlim', 'left': xmin, 'right': xmax},
        {'style_type': 'ylim', 'bottom': ymin, 'top': ymax},
        {'style_type': 'aspect', 'aspect': 'equal', },
        {'style_type': 'spines', 'top': True, 'right': True, 'left': True, 'bottom': True},
        {'style_type': 'ticks_params', 'which': 'both', 'labelsize': 8,
            'top': True, 'bottom': False, 'left': True, 'right': False,
            'labeltop': True, 'labelbottom': False, 'labelleft': True, 'labelright': False},
        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90},
        {'style_type': 'yticklabels', 'va': 'center'},
        {'style_type': 'major_formatter', 'axis': 'y', 'formatter_type': 'dms_suffix', 'suffix': ' N'},
        {'style_type': 'major_formatter', 'axis': 'x', 'formatter_type': 'dms_suffix', 'suffix': ' W'},
    ])
    
    # b
    ticks, ticks_labels = [0, 0.25, 0.5, 0.75, 1], ['$0$', '', '$0.5$', '', '$1$']
    
    init_dic(dict_to_plot, 'b', 'style')
    dict_to_plot['b']['style'].extend([
        {'style_type': 'xlabel', 'label': 'Cumulative share\nof buildings'},
        {'style_type': 'ylabel', 'label': 'Cumulative share\nof damage'},
        {'style_type': 'xlim', 'left': 0, 'right': 1},
        {'style_type': 'ylim', 'bottom': 0, 'top': 1},
        {'style_type': 'xticks', 'ticks': ticks},
        {'style_type': 'xticklabels', 'labels': ticks_labels},
        {'style_type': 'yticks', 'ticks': ticks},
        {'style_type': 'yticklabels', 'labels': ticks_labels},
    ])
    
    # c
    init_dic(dict_to_plot, 'c', 'style')
    dict_to_plot['c']['style'].extend([
        {'style_type': 'xlim', 'left': 0, 'right': 1},
        {'style_type': 'ylim', 'bottom': 0, 'top': 1},
        {'style_type': 'title', 'label': 'Inequality', 'fontsize': 8, 'pad': 13},
        {'style_type': 'xlabel', 'label': 'Equality\n\nGINI\ncoefficient'},
        {'style_type': 'ylabel', 'label': 'Value', 'labelpad': -2.5},
        {'style_type': 'xticks', 'ticks': []},
        {'style_type': 'yticks', 'ticks': [0, 0.25, 0.5, 0.75, 1]},
        {'style_type': 'yticklabels', 'labels': ['', '', '', '', '']},
        {'style_type': 'ticks_params', 'axis': 'y', 'length': 0},
        {'style_type': 'spines', 'top': False, 'right': False, 'left': False, 'bottom': True}
    ])
    
    # d
    init_dic(dict_to_plot, 'd', 'style')
    dict_to_plot['d']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Total Damage (€)', 'labelpad': 0},
        {'style_type': 'ylim', 'bottom': data_d.min(), 'top': data_d.max()},
        {'style_type': 'xticks', 'ticks': []},
        {'style_type': 'ticklabel_format', 'axis': 'y', 'style': 'sci', 'scilimits': (0,0), 'useMathText': True},
        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90}
    ])
    
    # e
    init_dic(dict_to_plot, 'e', 'style')
    dict_to_plot['e']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Building Damage (€)', 'labelpad': 0.5},
        {'style_type': 'xlabel', 'label': 'Building Damage Quantile', 'labelpad': 10},
        {'style_type': 'ylim', 'bottom': df_wide[q05_col_to_plot].min(), 'top': df_wide[q95_col_to_plot].max()},
        {'style_type': 'yscale', 'value': 'symlog', 'linthresh': 100, 'linscale': 0.4},
        {'style_type': 'xticks', 'ticks': [1, 2, 3]},
        {'style_type': 'xticklabels', 'labels': ['Q05', 'Q50', 'Q95']},
        #{'style_type': 'ticklabel_format', 'axis': 'y', 'style': 'sci', 'scilimits': (0,0), 'useMathText': True},
        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90}
    ])
    
    # x, leg
    for ax in ['x1', 'x2', 'x3', 'x4', 'leg']:
        init_dic(dict_to_plot, ax, 'style')
        dict_to_plot[ax]['style'].append({'style_type': 'off'})
    
    # letter
    for key in dict_to_plot.keys():
        if key not in ['x1', 'x2', 'x3', 'x4', 'leg']:
            y_pos = 1.05 if key == 'a' else (1.2 if key == 'd' else 1.12)
            x_pos = -0.01 if key == 'a' else (-0.08 if key == 'c' else (-0.02 if key == 'd' else -0.04))
            dict_to_plot[key]['style'].append({
                'style_type': 'letter', 
                'label': f'({key})',
                'x': x_pos,
                'y': y_pos
            })
     
    # Legend
    init_dic(dict_to_plot, 'a', 'legend')
    real_dx = get_dx_for_scalebar([xmin, xmax, ymin, ymax], crs='EPSG:4326')
    dict_to_plot['a']['legend'].extend([
        {'legend_type': 'north_arrow',
            'x': 0.98, 'y': 0.7, 'length': 0.08, 'pad': 0.02, 'color': 'white', 'fontsize': 10},
        {'legend_type': 'scalebar',
            'dx': real_dx, 'units': 'm', 'dimension': 'si-length', 'location': 'lower right',
            'frameon': False, 'color': 'white'}
    ])
    
    init_dic(dict_to_plot, 'leg', 'legend')
    y_start_0 = 1
    #print(y_start_0)
    y_step_1 = 0.1
    y_step_2 = 0.05
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Total Building\nDamage (€)',
            'x': 0, 'y': y_start_0,
            'ha': 'left', 'va': 'top', 'fontsize': 8
        }
    ])
    y_start_1 = y_start_0 - y_step_1
    for i, (label, color) in enumerate(zip(labels, colors)):
        y_pos = y_start_1 - (i * y_step_2)
        print(y_pos)
        dict_to_plot['leg']['legend'].append({'legend_type': 'circle',
            'x': 0.08,
            'y': y_pos,
            'radius': 0.02,
            'color': color,
            'edgecolor': 'black',
            'linewidth': 0.5,
            'zorder': 10
        })
        dict_to_plot['leg']['legend'].append({'legend_type': 'text',
            'x': 0.22,
            'y': y_pos,
            'content': label,
            'ha': 'left',
            'va': 'center',
            'fontsize': 7
        })
        
    y_start_2 = y_start_1 - y_step_1 - ((len(labels)-1) * y_step_2)
    #print(y_start_2)
    #print(y_start_2 - y_step_2)
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Flood Extensions',
            'x': 0, 'y': y_start_2,
            'ha': 'left', 'va': 'top', 'fontsize': 8
        },
        {'legend_type': 'text',
            'content': 'Stochastic',
            'x': 0, 'y': y_start_2 - y_step_2,
            'ha': 'left', 'va': 'top', 'fontsize': 8
        },
    ])
    
    y_start_3 = y_start_2 - (y_step_2*2)
    q_configs = [
        {'ls': '-',  'lw': 1.5, 'color': q_colors['sto_Q50_contour'], 'label': 'Q50'},
        {'ls': '--', 'lw': 1.0, 'color': q_colors['sto_Q05_contour'], 'label': 'Q05'},
        {'ls': '--', 'lw': 1.0, 'color': q_colors['sto_Q95_contour'], 'label': 'Q95'},
    ]
    for i, cfg in enumerate(q_configs):
        y_pos = y_start_3 - (i * y_step_2)
        print(y_pos)
        dict_to_plot['leg']['legend'].extend([
            {'legend_type': 'line',
                'x': 0,
                'y': y_pos,
                'length': 0.3,
                'color': cfg['color'],
                'alpha': 1,
                'linestyle': cfg['ls'],
                'linewidth': cfg['lw']
            },
            {'legend_type': 'text',
                'x': 0.35,
                'y': y_pos,
                'content': cfg['label'],
                'ha': 'left',
                'va': 'center',
                'fontsize': 7
            },
        ])
        
    y_start_3b = y_start_3 - ((len(q_configs)-1) * y_step_2) - y_step_2
    #print(y_start_3b)
    #print(y_start_3b - y_step_2)
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Deterministic',
            'x': 0, 'y': y_start_3b,
            'ha': 'left', 'va': 'top', 'fontsize': 8
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_3b - y_step_2,
            'length': 0.3,
            'color': q_colors['det_Q50_contour'],
            'alpha': 1,
            'linestyle': '-',
            'linewidth': 1.5,
        },
        {'legend_type': 'text',
            'x': 0.35,
            'y': y_start_3b - y_step_2,
            'content': 'Official',
            'ha': 'left',
            'va': 'center',
            'fontsize': 7
        },
    ])
    
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'Damage',
        'fname': 'Stochastic_Total_Damage_Map_v1.1.png',
        'show': True,
        'dpi': 300,
    }
    
    
    # 3. Plot
    plot_any(dict_to_plot, layout_params, save_params, global_pre_style=global_pre_style)
#endregion

# region D.F4 - Damage Evolution
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    # Flow
    #dist_depth_df = pd.read_excel(PATHS['fm_flow_xlsx'])
    dist_depth_df = pd.read_pickle(r"C:\Users\outal\GITHUB\InDepthPROFILE\data\intermediate\Flow_Fitted_Functions.pkl")
    dist_depth_df = dist_depth_df[dist_depth_df['ReturnPeriod'] != 2]
    dist_depth_df = dist_depth_df.sort_values('ReturnPeriod')
    rp_indices = dist_depth_df['ReturnPeriod'].astype(str).tolist()
    positions = range(len(rp_indices))
    dist_depth_list = []
    q_data = []
    for _, row in dist_depth_df.iterrows():
        params = {'skew': row['Skew'], 'loc': row['Loc'], 'scale': row['Scale']}
        q_logs = pearson3.ppf([0.05, 0.50, 0.95], **params)
        q_vals = 10 ** q_logs
        q_data.append(q_vals)
        samples_log = np.linspace(q_logs[0], q_logs[2], 1000)
        dist_depth_list.append(10 ** samples_log)
    df_flow = pd.DataFrame(q_data, columns=['Y_Q05', 'Y_Q50', 'Y_Q95'])
    df_flow['X_pos'] = range(len(df_flow))
    
    # Damage and Risk
    value_col = 'c_hu_bf'
    value_col_ex = 'c_e_bf'
    path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\processed\MC_Parts"
    full_df = load_and_concatenate_mc_parts(path, ['SN','BID','RP',value_col,value_col_ex])
    if value_col in full_df.columns:
        print("Column exists!")
        
    value_col = 'c_hu_total'
    cols = ['it','BID','RP', value_col]
    # scan lazy and collect only the cols, then rename it to SN
    full_df_pl = (
        pl.scan_ipc(os.path.join(mocalos_results_path, "*.ipc"))
        .select(cols)
        .with_columns(pl.col(value_col).fill_null(0))
        .rename({"it": "SN"})
        .collect()
    )
    full_df = full_df_pl.to_pandas()
    
    full_df = full_df.groupby(['SN', 'RP'])[[value_col,value_col_ex]].sum().reset_index()
    bid_counts = {rp: len(bids) for rp, bids in BIDs_flooded_unq_byRP.items()}
    full_df['c_hu_bf_by_bid'] = full_df['c_hu_bf'] / full_df['RP'].map(bid_counts)
    full_df['c_e_bf_by_bid'] = full_df['c_e_bf'] / full_df['RP'].map(bid_counts)
    full_df['EP'] = np.round((1 / full_df['RP']),3)
    
    metrics = ['c_hu_bf', 'c_e_bf', 'c_hu_bf_by_bid', 'c_e_bf_by_bid']
    unique_rps = sorted(full_df['RP'].unique())
    wide_data = {}
    for rp in unique_rps:
        rp_group = full_df[full_df['RP'] == rp]
        
        # Format EP to a clean string (e.g., 0.2, 0.002)
        ep_str = f"{1/rp:.4g}" 
        
        for m in metrics:
            # Create a clean name: Y_0.2_c_hu
            col_name = f"Y_{ep_str}_{m}"
            wide_data[col_name] = rp_group[m].reset_index(drop=True)
    df_damage = pd.DataFrame(wide_data)
    
    all_eps = sorted(np.unique([float(col.split('_')[1]) for col in df_damage.columns if 'c_hu' in col]))
    ep_strings = [f"{ep:.4g}" for ep in all_eps]
    x_positions = range(len(ep_strings))
    q_values = [0.05, 0.50, 0.95]
    q_labels = ['Q05', 'Q50', 'Q95']
    quantile_dfs = {}
    for m in metrics:
        q_data = []
        # Ensure we iterate in the exact order used for x_positions
        for ep_s in ep_strings:
            col_name = f"Y_{ep_s}_{m}"
            if col_name in df_damage.columns:
                # Calculate quantiles for this specific EP and metric
                vals = df_damage[col_name].quantile(q_values).values
                q_data.append(vals)
        
        # Create DataFrame: Columns [Y_Q05, Y_Q50, Y_Q95], Index matches x_positions
        tmp_df = pd.DataFrame(q_data, columns=[f'Y_{q}' for q in q_labels])
        tmp_df['X_pos'] = range(len(tmp_df))
        quantile_dfs[m] = tmp_df
    
    # EAD
    probabilities = np.array([float(ep) for ep in ep_strings])
    prob_with_zero = np.sort(np.append(probabilities, 0.5)) 

    ead_results = {}
    for m in ['c_hu_bf', 'c_hu_bf_by_bid']:
        df = quantile_dfs[m]
        ead_results[m] = {}
        
        for q in ['Y_Q05', 'Y_Q50', 'Y_Q95']:
            damage_vals = df[q].values
            damage_with_zero = np.append(damage_vals, 0)
            
            # Sort data by probability ascending
            sort_idx = np.argsort(prob_with_zero)
            x_raw = prob_with_zero[sort_idx]
            y_raw = damage_with_zero[sort_idx]
            
            # Use log(x) scale
            log_x_raw = np.log(x_raw)
            
            # Create linear interpolator in log-space
            interp_func = interp1d(log_x_raw, y_raw, kind='linear')
            
            # Integrate over a dense grid for higher accuracy
            x_dense = np.linspace(x_raw.min(), x_raw.max(), 1000)
            y_dense = interp_func(np.log(x_dense))
            
            # Calculate the area under the log-linear curve
            ead_val = np.trapezoid(y_dense, x_dense)
            ead_results[m][q] = ead_val
        
    # 2. Prepare configuration
    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "x", "b", "xx", "c", "leg"],
            ["a", "x", "d", "xx", "e", "leg"],
        ],
        'figsize': (6, 4),
        'kwargs': {
            'gridspec_kw': {
                'wspace': 0.055,
                'hspace': 0.27,
                'width_ratios': [1, 0.3, 1, 0.2, 1, 0.5],
                'height_ratios': [1, 1],
            },
        },
        'adjust': {'bottom': 0.14, 'top': 0.87, 'left': 0.09, 'right': 0.98}
    }
    
    # Plots
    dict_to_plot = {
        'a': {'plots': [], 'style': []},
        'b': {'plots': [], 'style': []},
        'c': {'plots': [], 'style': []},
        'd': {'plots': [], 'style': []},
        'e': {'plots': [], 'style': []},
        'x': {'style': []},
        'xx': {'style': []},
        'leg': {'style': [], 'legend': []},
    }
    
    palette = plt.get_cmap(RP_color_palette, len(RPs_unq))
    rp_colors = {rp: palette(i) for i, rp in enumerate(RPs_unq)}    
    violin_colors_a = [rp_colors[rp] for rp in RPs_unq]
    violin_colors_damage = list(reversed(violin_colors_a))
    
    # a
    dict_to_plot['a']['plots'].append({'plot_type': 'violin',
            'dataset': dist_depth_list,
            'positions': list(positions),
            'widths': 0.95,
            'color': violin_colors_a,
            'vert': True,
            'showmeans': False,
            'showmedians': False,
            'showextrema': False,
            'points': 1000,
            'bw_method': 0.1
        })
    line_styles = [
        {'Y': 'Y_Q05', 'ls': ':',  'label': '$Q_{05}$', 'alpha': 0.7},
        {'Y': 'Y_Q50', 'ls': '-',  'label': '$Q_{50}$', 'alpha': 0.9},
        {'Y': 'Y_Q95', 'ls': '--', 'label': '$Q_{95}$', 'alpha': 0.7}
    ]
    for s in line_styles:
        dict_to_plot['a']['plots'].append({'plot_type': 'line',
            'y': df_flow[s['Y']].values,
            'x': df_flow['X_pos'].values,
            'color': 'black',
            'linestyle': s['ls'],
            'linewidth': 1,
            'alpha': s['alpha'],
            'label': s['label'],
            'zorder': 3
        })
    
    # b, c, d, e
    metric_map = {
        'b': 'c_hu_bf', 
        'c': 'c_e_bf', 
        'd': 'c_hu_bf_by_bid', 
        'e': 'c_e_bf_by_bid'
    }
    line_configs = [
        {'Y': 'Y_Q05', 'ls': ':',  'alpha': 0.7},
        {'Y': 'Y_Q50', 'ls': '-',  'alpha': 0.9},
        {'Y': 'Y_Q95', 'ls': '--', 'alpha': 0.7}
    ]
    for ax_key, m in metric_map.items():
        target_df = quantile_dfs[m]
        
        for s in line_configs:
            color_s = 'black' if ax_key in ['c', 'e'] else 'black'
            dict_to_plot[ax_key]['plots'].append({'plot_type': 'line',
                'y': target_df[s['Y']].values,
                'x': target_df['X_pos'].values,
                'color': color_s,
                'linestyle': s['ls'],
                'linewidth': 1,
                'alpha': s['alpha'],
                'zorder': 3
            })
    for ax_key, metric in zip(['b', 'c', 'd', 'e'], ['c_hu_bf', 'c_e_bf', 'c_hu_bf_by_bid', 'c_e_bf_by_bid']):
        violin_datasets = []
        for ep_s in ep_strings:
            col_name = f"Y_{ep_s}_{metric}"
            data = df_damage[col_name].dropna().values
            violin_datasets.append(data)
        dict_to_plot[ax_key]['plots'].append({'plot_type': 'violin',
            'dataset': violin_datasets,
            'positions': list(x_positions),
            'vert': True,
            'widths': 0.8,
            'color': violin_colors_damage,
            'showmeans': False,
            'showmedians': False,
            'showextrema': False,
            'bw_method': 'scott'
        })
    ead_mapping = {'b': 'c_hu_bf', 'd': 'c_hu_bf_by_bid'}
    for ax_key, metric in ead_mapping.items():
        if metric in ead_results:
            res = ead_results[metric]
            ead_text = (
                "EAD €\n"
                f"({res.get('Y_Q05', 0):,.0f})\n"
                f"{res.get('Y_Q50', 0):,.0f}\n"
                f"({res.get('Y_Q95', 0):,.0f})"
            )
            dict_to_plot[ax_key]['plots'].append({'plot_type': 'text',
                'x': 0.65,
                'y': 0.92,
                'text': ead_text,
                'fontsize': 6,
                'ha': 'center',
                'va': 'top',
                'family': 'serif',
                'linespacing': 1.2,
                'bbox': dict(facecolor="#DBD4D4", alpha=0.7, edgecolor='none', pad=1.2, boxstyle='round,pad=0.3')
            })

    # Style
    rev_rp_labels = [f"${rp}$" for rp in reversed(RPs_unq)]
    dict_to_plot['a']['style'].extend([
        {'style_type': 'title', 'label': 'Hazard', 'fontsize': 8, 'pad': 15},
        {'style_type': 'ylabel', 'label': 'Flow ($m^3/s$)', 'fontsize': 8},
        {'style_type': 'xlabel', 'label': 'Return Period', 'fontsize': 8, 'labelpad': 7},
        {'style_type': 'ylim', 'bottom': 244, 'top': 10_710},
        {'style_type': 'xticks', 'ticks': list(positions)},
        {'style_type': 'xticklabels', 'labels': [f"${rp}$" for rp in RPs_unq],
            'rotation': 90, 'fontsize': 8},
        {'style_type': 'yticks', 'ticks': [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]},
        {'style_type': 'yticklabels', 'labels': ['$500$', '', '$2000$', '', '$4000$', '', '$6000$', '', '$8000$', '', '$10000$'],
            'rotation': 90, 'va': 'center', 'ha': 'right'},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['b']['style'].extend([
        {'style_type': 'title', 'label': 'Damage ($€$)', 'fontsize': 8, 'pad': 15},
        {'style_type': 'ylabel', 'label': 'Total', 'fontsize': 8, 'labelpad': 6},
        {'style_type': 'ylim', 'bottom': 0, 'top': 25_000_000},
        {'style_type': 'xticks', 'ticks': list(positions)},
        {'style_type': 'xticklabels', 'labels': []},
        {'style_type': 'ticklabel_format',
            'axis': 'y', 'style': 'sci', 'scilimits': (0, 0), 'useMathText': True},
        {'style_type': 'ticks_params',
            'axis': 'both', 'labelsize': 8},
        {'style_type': 'offset_text', 'axis': 'y', 'fontsize': 8},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['c']['style'].extend([
        {'style_type': 'title', 'label': 'Expected Damage\n(Risk - $€$)', 'fontsize': 8, 'pad': 0},
        {'style_type': 'ylim', 'bottom': 0, 'top': 90_000},
        {'style_type': 'xticks', 'ticks': list(positions)},
        {'style_type': 'xticklabels', 'labels': []},
        {'style_type': 'ticklabel_format',
            'axis': 'y', 'style': 'sci', 'scilimits': (0, 0), 'useMathText': True},
        {'style_type': 'ticks_params',
            'axis': 'both', 'labelsize': 8},
        {'style_type': 'offset_text', 'axis': 'y', 'fontsize': 8},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['d']['style'].extend([
        {'style_type': 'ylabel', 'label': 'By building', 'fontsize': 8, 'labelpad': 3},
        {'style_type': 'xlabel', 'label': 'Return Period', 'fontsize': 8, 'labelpad': 6},
        {'style_type': 'ylim', 'bottom': 0, 'top': 15_000},
        {'style_type': 'xticks', 'ticks': list(positions)},
        {'style_type': 'xticklabels', 'labels': rev_rp_labels,
            'rotation': 90, 'fontsize': 8},
        {'style_type': 'ticklabel_format',
            'axis': 'y', 'style': 'sci', 'scilimits': (0, 0), 'useMathText': True},
        {'style_type': 'ticks_params',
            'axis': 'both', 'labelsize': 8},
        {'style_type': 'offset_text', 'axis': 'y', 'fontsize': 8},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['e']['style'].extend([
        {'style_type': 'xlabel', 'label': 'Return Period', 'fontsize': 8, 'labelpad': 7},
        {'style_type': 'ylim', 'bottom': 0, 'top': 500},
        {'style_type': 'xticks', 'ticks': list(positions)},
        {'style_type': 'xticklabels', 'labels': rev_rp_labels,
            'rotation': 90, 'fontsize': 8},
        {'style_type': 'ticklabel_format',
            'axis': 'y', 'style': 'sci', 'scilimits': (0, 0), 'useMathText': True},
        {'style_type': 'ticks_params',
            'axis': 'both', 'labelsize': 8},
        {'style_type': 'offset_text', 'axis': 'y', 'fontsize': 8},
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False}
    ])
    dict_to_plot['leg']['style'].append({'style_type': 'off'})
    dict_to_plot['x']['style'].append({'style_type': 'off'})
    dict_to_plot['xx']['style'].append({'style_type': 'off'})
    for key in dict_to_plot.keys():
        if key not in ['x', 'xx', 'leg']:
            y_pos = 1.05 if key == 'a' else 1.145
            dict_to_plot[key]['style'].append({'style_type': 'letter', 
                'label': f'({key})',
                'fontsize': 8,
                'x': -0.21,
                'y': y_pos
            })
    
    # Legend
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Return\nPeriod',
            'x': 0.02, 'y': 1,
            'ha': 'left', 'va': 'top', 'fontsize': 8,
        },
    ])
    for i, rp in enumerate(RPs_unq):
        y_pos = 0.88 - (i * 0.057)
        dict_to_plot['leg']['legend'].extend([
            {'legend_type': 'circle', 'x': 0.12, 'y': y_pos, 'radius': 0.02, 
                'color': rp_colors[rp], 'zorder': 10
            },
            {'legend_type': 'text', 'content': f"{rp}", 'x': 0.27, 'y': y_pos, 
                'ha': 'left', 'va': 'center', 'fontsize': 8
            }
        ])
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Expected\nAnnual\nDamage\n(EAD)\nQuantiles',
            'x': 0.02, 'y': 0.47,
            'ha': 'left', 'va': 'top', 'fontsize': 8,
        },
        {'legend_type': 'rectangle', 
            'x': 0.12, 'y': 0.111, 'width': 0.65, 'height': 0.1427, 
            'color': "#DBD4D4", 'alpha': 0.7, 'zorder': 1
        },
        {'legend_type': 'text',
            'content': '(Q05)\nQ50\n(Q95)',
            'x': 0.45, 'y': 0.237,
            'ha': 'center', 'va': 'top', 'fontsize': 8, 'zorder': 2
        },
        {'legend_type': 'text',
            'content': 'Quantiles',
            'x': 0.02, 'y': 0.07,
            'ha': 'left', 'va': 'top', 'fontsize': 8,
        },
    ])
    q_styles = [
        {'label': 'Q05', 'ls': ':',  'y_off': -0.01},
        {'label': 'Q50', 'ls': '-',  'y_off': -0.06},
        {'label': 'Q95', 'ls': '--', 'y_off': -0.11}
    ]
    for q in q_styles:
        dict_to_plot['leg']['legend'].extend([
            {'legend_type': 'line',
                    'x': 0.04, 'y': q['y_off'], 'length': 0.175, 'color': 'black', 'linestyle': q['ls'], 'linewidth': 1.3},
            {'legend_type': 'text',
                'content': q['label'], 'x': 0.27, 'y': q['y_off'], 'ha': 'left', 'va': 'center', 'fontsize': 8}
        ])
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'F4_Stochastic Classic Damage Evolution',
        'fname': 'F4_Damage_Evolution_v4.1.png',
        'show': True,
        'dpi': 300,
    }
    
    # 3. Plot
    plot_any(dict_to_plot, layout_params, save_params)
#endregion

# region D.F5 - Damage Functions
RUN_BLOCK = False
if RUN_BLOCK:    
    # 1. Prepare data
    # Data sources
    depth_cols = ['he','hi_0']
    pct_0_cols = ['c_hu_0pct','c_huCTEs_0pct','c_huCTIs_0pct']
    pct_cols = ['c_hu_bfpct']
    full_df = load_and_concatenate_mc_parts(
        PATHS['dataset_mc_parts'], ['SN','BID','RP'] + depth_cols + pct_0_cols + pct_cols)
    
    q_map = {0.05: 'Q05', 0.50: 'Q50', 0.95: 'Q95'}
    # hi0
    bin_size_hi0 = 0.1
    full_df['X_hi0'] = (np.floor(full_df['hi_0'] / bin_size_hi0) * bin_size_hi0).round(1)
    df_dam_fun_hi0 = full_df.groupby('X_hi0')[pct_0_cols].quantile(list(q_map.keys())).unstack()
    df_dam_fun_hi0.columns = [
        f"{col}_{q_map[q]}" for col, q in df_dam_fun_hi0.columns
    ]
    df_dam_fun_hi0 = df_dam_fun_hi0.reset_index()
    #full_df[['c_hu_0pct', 'hi_0']].min()
    #full_df[['c_hu_0pct', 'hi_0']].max()
    
    # he
    def get_dynamic_bin(depth, bin_steps):
        """
        Assigns a depth to a bin center based on dynamic thresholds.
        """
        if depth < 2.2: step = 0.05
        elif depth < 3: step = 0.1
        elif depth < 5: step = 0.2
        else: step = 0.4
        return np.floor(depth / step) * step + (step / 2)
    full_df['X_he'] = full_df['he'].apply(lambda x: get_dynamic_bin(x, None)).round(2)
    df_dam_fun_he = full_df.groupby('X_he')[pct_cols].quantile(list(q_map.keys())).unstack()
    df_dam_fun_he.columns = [
        f"{col}_{q_map[q]}" for col, q in df_dam_fun_he.columns
    ]
    df_counts = full_df.groupby('X_he').agg(
        count=(pct_cols[0], 'count'),
        BID_count=('BID', 'nunique')
    )
    df_dam_fun_he = df_dam_fun_he.merge(df_counts, on='X_he').reset_index()
    #full_df[['c_hu_bfpct', 'he']].min()
    #full_df[['c_hu_bfpct', 'he']].max()
    
    
    # 2. Prepare configuration
    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "x1", "c", "x2", "leg"],
            ["b", "x1", "c", "x2", "leg"],
        ],
        'figsize': (6, 4),
        'kwargs': {
            'gridspec_kw': {
                'hspace': 0.26,
                'wspace': 0.05,
                'width_ratios': [1, 0.25, 1, 0.01, 0.3],
            },
        },
        'adjust': {'bottom': 0.15, 'top': 0.95, 'left': 0.1, 'right': 0.96}
    }
    
    # Plots
    colors = {
        'a': 'black',
        'b': 'grey',
        'c': ['blue', 'red'],
    }
    dict_to_plot = {
        'a': {'plots': [], 'style': []},
        'b': {'plots': [], 'style': []},
        'c': {'plots': [], 'style': []},
        'x': {'plots': [], 'style': []},
    }
    # a
    dict_to_plot['a']['plots'].extend([
        {'plot_type': 'fill',
            'x': df_dam_fun_hi0['X_hi0'], 'y1': df_dam_fun_hi0['c_hu_0pct_Q05'], 'y2': df_dam_fun_hi0['c_hu_0pct_Q95'],
            'color': colors['a'], 'alpha': 0.2},
        {'plot_type': 'line',
            'x': df_dam_fun_hi0['X_hi0'], 'y': df_dam_fun_hi0['c_hu_0pct_Q50'],
            'label': 'GDF_hi0', 'color': colors['a'], 'linewidth': 1, 'alpha': 1},
    ])
    # b
    dict_to_plot['b']['plots'].extend([
        {'plot_type': 'fill',
            'x': df_dam_fun_he['X_he'], 'y1': df_dam_fun_he['c_hu_bfpct_Q05'], 'y2': df_dam_fun_he['c_hu_bfpct_Q95'],
            'color': colors['b'], 'alpha': 0.2},
        {'plot_type': 'line',
            'x': df_dam_fun_he['X_he'],  'y': df_dam_fun_he['c_hu_bfpct_Q50'],
            'label': 'GDF_he', 'color': colors['b'], 'linewidth': 1, 'alpha': 1}
    ])
    # c
    dict_to_plot['c']['plots'].extend([
        {'plot_type': 'fill', 'x':  df_dam_fun_hi0['X_hi0'], 'y1': df_dam_fun_hi0['c_huCTEs_0pct_Q05'], 'y2': df_dam_fun_hi0['c_huCTEs_0pct_Q95'],
            'color': colors['c'][0], 'alpha': 0.2},
        {'plot_type': 'fill', 'x': df_dam_fun_hi0['X_hi0'], 'y1': df_dam_fun_hi0['c_huCTIs_0pct_Q05'], 'y2': df_dam_fun_hi0['c_huCTIs_0pct_Q95'],
            'color': colors['c'][1], 'alpha': 0.2},
        {'plot_type': 'line', 'x': df_dam_fun_hi0['X_hi0'], 'y': df_dam_fun_hi0['c_huCTEs_0pct_Q50'],
            'label': 'DF_CTE_hi0', 'color': colors['c'][0], 'linewidth': 1, 'alpha': 1},
        {'plot_type': 'line', 'x': df_dam_fun_hi0['X_hi0'], 'y': df_dam_fun_hi0['c_huCTIs_0pct_Q50'],
            'label': 'DF_CTI_hi0', 'color': colors['c'][1], 'linewidth': 1, 'alpha': 1},
    ])
    
    # Style
    x_ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    x_ticks_labels = ['$0$', '', '$1$', '', '$2$', '', '$3$']
    y_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    y_ticks_labels = ['$0$', '', '', '', '', '$0.5$', '', '', '', '', '$1.0$']
    # a
    dict_to_plot['a']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Damage. First floor', 'labelpad': 3, 'fontsize': 8},
        {'style_type': 'xlim', 'left': 0, 'right': 3},
        {'style_type': 'ylim', 'bottom': 0, 'top': 1},
        {'style_type': 'xticks', 'ticks': x_ticks},
        {'style_type': 'xticklabels', 'labels': x_ticks_labels},
        {'style_type': 'yticks', 'ticks': y_ticks},
        {'style_type': 'yticklabels', 'labels': y_ticks_labels,
            'rotation': 90, 'va': 'center', 'ha': 'right'},
        {'style_type': 'grid', 'visible': True, 'which': 'major', 'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.7},
        {'style_type': 'spines', 'top': False, 'right': False},
    ])
    # b
    valid_bins_he = df_dam_fun_he[df_dam_fun_he['BID_count'] > 30]
    max_valid_x_he = valid_bins_he['X_he'].max() if not valid_bins_he.empty else df_dam_fun_he['X_he'].max()
    x_ticks_b = []
    x_ticks_labels_b = []
    current_tick = 0.0
    while current_tick < max_valid_x_he:
        x_ticks_b.append(current_tick)
        if current_tick.is_integer():
            x_ticks_labels_b.append(f'${int(current_tick)}$')
        else:
            x_ticks_labels_b.append('')
        if current_tick < 3.0:
            current_tick += 0.5
        else:
            current_tick += 1.0
        if not x_ticks_b or x_ticks_b[-1] != max_valid_x_he:
            x_ticks_b.append(max_valid_x_he)
            x_ticks_labels_b.append('')
        else:
            x_ticks_labels_b[-1] = ''
    
    dict_to_plot['b']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Damage. All floors', 'labelpad': 3, 'fontsize': 8},
        {'style_type': 'xlabel', 'label': 'Depth ($m$)', 'labelpad': 3, 'fontsize': 8},
        {'style_type': 'xlim', 'left': 0, 'right': max_valid_x_he},
        {'style_type': 'ylim', 'bottom': 0, 'top': 1},
        {'style_type': 'xticks', 'ticks': x_ticks_b},
        {'style_type': 'xticklabels', 'labels': x_ticks_labels_b},
        {'style_type': 'yticks', 'ticks': y_ticks},
        {'style_type': 'yticklabels', 'labels': y_ticks_labels,
            'rotation': 90, 'va': 'center', 'ha': 'right'},
        {'style_type': 'grid', 'visible': True, 'which': 'major', 'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.7},
        {'style_type': 'spines', 'top': False, 'right': False},
    ])
    # c
    dict_to_plot['c']['style'].extend([
        {'style_type': 'ylabel', 'label': 'Content (blue) and continent (red) damage. First floor', 'labelpad': 5, 'fontsize': 8},
        {'style_type': 'xlabel', 'label': 'Depth ($m$)', 'labelpad': 3, 'fontsize': 8},
        {'style_type': 'xlim', 'left': 0, 'right': 3},
        {'style_type': 'ylim', 'bottom': 0, 'top': 1},
        {'style_type': 'xticks', 'ticks': x_ticks},
        {'style_type': 'xticklabels', 'labels': x_ticks_labels},
        {'style_type': 'yticks', 'ticks': y_ticks},
        {'style_type': 'yticklabels', 'labels': y_ticks_labels,
            'rotation': 90, 'va': 'center', 'ha': 'right'},
        {'style_type': 'grid', 'visible': True, 'which': 'major', 'linestyle': '--', 'linewidth': 0.5, 'alpha': 0.7},
        {'style_type': 'spines', 'top': False, 'right': False},
    ])
    # x, leg
    for ax in ['x1', 'x2', 'leg']:
        init_dic(dict_to_plot, ax, 'style')
        dict_to_plot[ax]['style'].append({'style_type': 'off'})
    
    # letter
    for key in dict_to_plot.keys():
        y_pos = 1.05 if key in ['c'] else 1.1
        print(key, y_pos)
        if key not in ['x1', 'x2', 'leg']:
            dict_to_plot[key]['style'].append({'style_type': 'letter', 
                'label': f'({key})',
                'fontsize': 8,
                'x': -0.01,
                'y': y_pos
            })
    
    # Legend
    init_dic(dict_to_plot, 'leg', 'legend')
    y_start_0 = 1
    print(y_start_0)
    y_step_1 = 0.1
    y_step_2 = 0.05
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Damage\nCurves',
            'x': 0, 'y': y_start_0,
            'ha': 'left', 'va': 'top', 'fontsize': 8
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (2 * y_step_2),
            'length': 0.3,
            'color': colors['a'],
            'alpha': 1,
            'linestyle': '-',
            'linewidth': 1
        },
        {'legend_type': 'text',
            'x': 0.35,
            'y': y_start_0 - (2 * y_step_2),
            'content': 'Q50',
            'ha': 'left',
            'va': 'center',
            'fontsize': 8
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (3 * y_step_2),
            'length': 0.3,
            'color': colors['b'],
            'alpha': 1,
            'linestyle': '-',
            'linewidth': 1
        },
        {'legend_type': 'text',
            'x': 0.35,
            'y': y_start_0 - (3 * y_step_2),
            'content': 'Q50',
            'ha': 'left',
            'va': 'center',
            'fontsize': 8
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (4 * y_step_2),
            'length': 0.3,
            'color': colors['c'][0],
            'alpha': 1,
            'linestyle': '-',
            'linewidth': 1
        },
        {'legend_type': 'text',
            'x': 0.35,
            'y': y_start_0 - (4 * y_step_2),
            'content': 'Q50',
            'ha': 'left',
            'va': 'center',
            'fontsize': 8
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (5 * y_step_2),
            'length': 0.3,
            'color': colors['c'][1],
            'alpha': 1,
            'linestyle': '-',
            'linewidth': 1
        },
        {'legend_type': 'text',
            'x': 0.35,
            'y': y_start_0 - (5 * y_step_2),
            'content': 'Q50',
            'ha': 'left',
            'va': 'center',
            'fontsize': 8
        },
        {'legend_type': 'text',
            'content': 'Q05-Q95\nUncertainty\nRange',
            'x': 0, 'y': y_start_0 - (5 * y_step_2) - (1 * y_step_1),
            'ha': 'left', 'va': 'top', 'fontsize': 8
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (8 * y_step_2) - (1 * y_step_1),
            'length': 0.9,
            'color': colors['a'],
            'alpha': 0.2,
            'linestyle': '-',
            'linewidth': 7
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (9 * y_step_2) - (1 * y_step_1),
            'length': 0.9,
            'color': colors['b'],
            'alpha': 0.2,
            'linestyle': '-',
            'linewidth': 7
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (10 * y_step_2) - (1 * y_step_1),
            'length': 0.9,
            'color': colors['c'][0],
            'alpha': 0.2,
            'linestyle': '-',
            'linewidth': 7
        },
        {'legend_type': 'line',
            'x': 0,
            'y': y_start_0 - (11 * y_step_2) - (1 * y_step_1),
            'length': 0.9,
            'color': colors['c'][1],
            'alpha': 0.2,
            'linestyle': '-',
            'linewidth': 7
        },
    ])
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'F5_Stochastic Classic Damage Functions',
        'fname': 'F5_Damage_Functions_v3.1.png',
        'show': True,
        'dpi': 300,
        'pad': 0.10
    }
    
    # 3. Plot
    plot_any(dict_to_plot, layout_params, save_params)
#endregion

# region D.F6 - Damage Inequality (waiting to remove)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    # Data sources
    total_cost_col = 'c_hu'
    target_cols = ['SN', 'BID', 'RP', 'he'] + [total_cost_col]
    full_df = load_and_concatenate_mc_parts(PATHS['dataset_mc_parts'], target_cols)
    
    max_damage_per_group = full_df.groupby(['RP', 'BID'])[total_cost_col].transform('max')
    full_df = full_df[max_damage_per_group > 0].copy()
    full_df = full_df.sort_values(by=['RP', 'BID', 'SN'])

    # Lorenz Curves
    df_dam_ineq = full_df.groupby(['RP', 'BID'])['c_hu'].quantile([0.05, 0.50, 0.95]).unstack()
    df_dam_ineq.columns = ['Q05', 'Q50', 'Q95']
    df_dam_ineq = df_dam_ineq.reset_index()
    
    lorenz_list = []
    quantiles = ['Q05', 'Q50', 'Q95']
    unique_rps = sorted(df_dam_ineq['RP'].unique())

    for rp in tqdm(unique_rps):
        rp_df = df_dam_ineq[df_dam_ineq['RP'] == rp].copy()
        n = len(rp_df)
        
        # Base dataframe for this RP
        rp_res = pd.DataFrame({'RP': [rp] * n})
        
        for q in quantiles:
            # Sort values for this specific quantile to define the curve
            sorted_vals = rp_df[q].sort_values().values
            
            # X: Cumulative proportion of population (BIDs)
            x_coords = np.arange(1, n + 1) / n
            
            # Y: Cumulative proportion of damage
            total_damage = sorted_vals.sum()
            if total_damage > 0:
                y_coords = np.cumsum(sorted_vals) / total_damage
            else:
                y_coords = np.zeros(n)
                
            rp_res[f'X_{q}'] = x_coords
            rp_res[f'Y_{q}'] = y_coords
            
        lorenz_list.append(rp_res)
    df_lorenz = pd.concat(lorenz_list, ignore_index=True)
    
    df_lorenz_final = df_lorenz.pivot(columns='RP')
    df_lorenz_final.columns = [f"{col}_{int(rp)}" for col, rp in df_lorenz_final.columns]
    df_lorenz_final = df_lorenz_final.apply(lambda x: pd.Series(x.dropna().values))

    # GINI coeficient
    gini_dict = {}
    for rp in unique_rps:
        rp_data = df_lorenz[df_lorenz['RP'] == rp]
        gini_dict[rp] = {}
        
        for q in quantiles:
            # Prepend 0 to coordinates for integration
            x = np.concatenate([[0], rp_data[f'X_{q}'].values])
            y = np.concatenate([[0], rp_data[f'Y_{q}'].values])
            
            # Calculate Area Under the Curve using trapezoidal rule
            auc = np.trapezoid(y, x)
            
            # Gini = (Area between identity and Lorenz) / (Area under identity)
            # Area under identity (0,0 to 1,1) is 0.5. 
            # Gini = (0.5 - auc) / 0.5 => 1 - 2*auc
            gini_val = 1 - 2 * auc
            gini_dict[rp][q] = gini_val
    gini_texts = {}
    for rp, gini_vals in gini_dict.items():
        q05 = f"{gini_vals['Q05']:.2f}" if isinstance(gini_vals['Q05'], float) else str(gini_vals['Q05'])
        q50 = f"{gini_vals['Q50']:.2f}" if isinstance(gini_vals['Q50'], float) else str(gini_vals['Q50'])
        q95 = f"{gini_vals['Q95']:.2f}" if isinstance(gini_vals['Q95'], float) else str(gini_vals['Q95'])
        
        gini_str = f"({q05}) {q50} ({q95})"
        gini_texts[int(rp)] = gini_str
    
    # 2. Layout
    # 2. Prepare configuration
    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
        ],
        'figsize': (6, 3),
        'kwargs': {
            'gridspec_kw': {
                'hspace': 0.2,
                'wspace': 0.2,
            },
        },
        'adjust': {'bottom': 0.15, 'top': 0.95, 'left': 0.15, 'right': 0.95}
    }
    
    # Plots and styles
    dict_to_plot = {
        'a': {'plots': [], 'style': []},
        'b': {'plots': [], 'style': []},
        'c': {'plots': [], 'style': []},
        'd': {'plots': [], 'style': []},
        'e': {'plots': [], 'style': []},
        'f': {'plots': [], 'style': []},
        'g': {'plots': [], 'style': []},
        'h': {'plots': [], 'style': []},
    }
    rps = [5, 10, 20, 50, 100, 200, 500]
    target_keys = ['a', 'b', 'c', 'e', 'f', 'g', 'h']
    palette = plt.get_cmap(RP_color_palette, len(RPs_unq))
    rp_colors = {rp: palette(i) for i, rp in enumerate(RPs_unq)}
    for rp, key in zip(rps, target_keys):
        current_color = rp_colors[rp]
        dict_to_plot[key]['plots'].append({ # Inequality Curve (Q50)
            'plot_type': 'line',
            'x': df_lorenz_final[f'X_Q50_{rp}'],
            'y': df_lorenz_final[f'Y_Q50_{rp}'],
            'color': current_color,
            'linewidth': 1,
            'label': f'RP {rp}'
        })
        dict_to_plot[key]['plots'].append({ # 2. Uncertainty Band (Q05-Q95)
            'plot_type': 'fill',
            'x': df_lorenz_final[f'X_Q50_{rp}'],
            'y1': df_lorenz_final[f'Y_Q05_{rp}'],
            'y2': df_lorenz_final[f'Y_Q95_{rp}'],
            'color': current_color,
            'alpha': 0.3
        })
        dict_to_plot[key]['plots'].append({ # Perfect Equality Reference Line (0,0 to 1,1)
            'plot_type': 'line',
            'x': [0, 1],
            'y': [0, 1],
            'color': 'grey',
            'linestyle': '--',
            'linewidth': 0.8,
            'alpha': 0.5
        })
        dict_to_plot[key]['plots'].append({ # GINI Coefficient Display
            'plot_type': 'text',
            'x': 0.02,
            'y': 0.75,
            'text': gini_texts[int(rp)],
            'fontsize': 6,
        })
    
    
    ticks, ticks_labels = [0, 0.25, 0.5, 0.75, 1], ['$0$', '', '$0.5$', '', '$1$']
    for key, config in dict_to_plot.items():
        if key in target_keys:
            current_x_ticks_labels = [] if key in ['a', 'b', 'c'] else ticks_labels
            current_y_ticks_labels = [] if key in ['b', 'c', 'f', 'g', 'h'] else ticks_labels
            config['style'].extend([
                {'style_type': 'xlim', 'left': 0, 'right': 1},
                {'style_type': 'ylim', 'bottom': 0, 'top': 1},
                {'style_type': 'xticks', 'ticks': ticks},
                {'style_type': 'xticklabels', 'labels': current_x_ticks_labels},
                {'style_type': 'yticks', 'ticks': ticks},
                {'style_type': 'yticklabels', 'labels': current_y_ticks_labels},
                {'style_type': 'grid', 'visible': True, 'linestyle': ':', 'alpha': 0.6},
                {'style_type': 'spines', 'top': False, 'right': False},
            ])
    dict_to_plot['d']['style'].extend([
        {'style_type': 'spines', 'top': False,'bottom': False, 'right': False, 'left': False},
        {'style_type': 'xticks', 'ticks': []},
        {'style_type': 'yticks', 'ticks': []},
        {'style_type': 'xtickslabels', 'labels': []},
        {'style_type': 'ytickslabels', 'labels': []}
    ])
    for key in dict_to_plot.keys():
        if key not in ['d']:
            dict_to_plot[key]['style'].append({
                'style_type': 'letter', 
                'label': f'({key})',
                'fontsize': 8,
                'x': 0.02,
                'y': 0.98
            })
    
    global_style = [
        {'gstype': 'supxlabel', 'label': 'Cumulative share of BIDs', 'fontsize': 8,},
        {'gstype': 'supylabel', 'label': 'Cumulative share of damage', 'fontsize': 8, 'x': 0.07},
    ]
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'F6_Damage Inequality',
        'fname': 'F6_Damage_Inequality_v2.png',
        'show': True,
        'dpi': 300,
        'pad': 0.10
    }
    
    # 3. Plot
    plot_any(dict_to_plot, layout_params, save_params, global_style=global_style)
#endregion

# region D.F7 - EAD MAP
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    # Data sources
    c_cols_CTE_bf = [
        'c_huAPP_bf', 'c_huCLO_bf', 'c_huCOM_bf', 'c_huDEC_bf', 'c_huELE_bf',
        'c_huFAD_bf', 'c_huFUR_bf', 'c_huHHG_bf', 'c_huHHB_bf', 'c_huINS_bf',
        'c_huLEI_bf', 'c_huOTH_bf', 'c_huSPE_bf', 'c_huTOO_bf', 'c_huVEH_bf']
    c_cols_CTI_bf = [
        'c_huETP_bf', 'c_huWND_bf', 'c_huPRW_bf', 'c_huITP_bf',
        'c_huSOI_bf', 'c_huSKT_bf', 'c_huRDR_bf', 'c_huPLG_bf']
    value_col = 'c_hu_bf'
    target_cols = c_cols_CTE_bf + c_cols_CTI_bf + ['SN', 'BID', 'RP'] + [value_col]
    full_df = load_and_concatenate_mc_parts(PATHS['dataset_mc_parts'], target_cols)
    
    valid_bids = set(full_df['BID'].drop_duplicates())
    gdf_buildings_subset = gdf_buildings[gdf_buildings['BID'].isin(valid_bids)]
    
    # EAD by BID
    df_q = full_df.groupby(['BID', 'RP'])[value_col].quantile([0.05, 0.50, 0.95]).unstack().reset_index()
    df_q.columns = ['BID', 'RP', 'Q05', 'Q50', 'Q95']

    def calculate_ead(rps, damages):
        """
        Calculates EAD using log-linear trapezoidal integration based on the provided reference.
        """
        eps = 1.0 / rps
        # Add zero-damage point at Exceedance Probability 0.5
        x_raw = np.append(eps, 0.5)
        y_raw = np.append(damages, 0.0)
        
        # Sort by probability ascending
        sort_idx = np.argsort(x_raw)
        x_sorted = x_raw[sort_idx]
        y_sorted = y_raw[sort_idx]
        
        # Linear interpolation in log-space
        log_x_raw = np.log(x_sorted)
        interp_func = interp1d(log_x_raw, y_sorted, kind='linear')
        
        # Integrate over dense grid
        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), 1000)
        y_dense = interp_func(np.log(x_dense))
        
        return np.trapezoid(y_dense, x_dense)

    ead_results = []
    for bid, group in tqdm(df_q.groupby('BID')):
        rps = group['RP'].values
        ead_results.append({
            'BID': bid,
            'EAD_Q05': calculate_ead(rps, group['Q05'].values),
            'EAD_Q50': calculate_ead(rps, group['Q50'].values),
            'EAD_Q95': calculate_ead(rps, group['Q95'].values)
        })
    df_ead_map = pd.DataFrame(ead_results).set_index('BID')
    
    gdf_ead = gdf_buildings_subset.merge(df_ead_map, on='BID', how='left')
    
    gdf_ead_centroids = gdf_ead.copy()
    gdf_ead_centroids['geometry'] = gdf_ead.centroid
    gdf_ead_centroids_4326 = gdf_ead_centroids.to_crs(epsg=4326)
    gdf_ead_centroids_4326 = gdf_ead_centroids_4326.sort_values(by='EAD_Q95', ascending=True)
    
    ## Data categorized
    # Define bins
    bins = [0, 1, 10, 100, 1000, 10000, np.inf]
    n_bins = len(bins) - 1
    labels = [f"{bins[i]}-{bins[i+1]}" if bins[i+1] != np.inf else f">{bins[i]}" for i in range(n_bins)]
    
    # Color palette
    cmap = plt.get_cmap('YlOrRd')
    colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0.2, 1, n_bins)]
    color_map = dict(zip(labels, colors))
    
    gdf_ead_centroids_4326['EAD_Q05_cat'] = pd.cut(gdf_ead_centroids_4326['EAD_Q05'], bins=bins, labels=labels, right=False)
    gdf_ead_centroids_4326['EAD_Q50_cat'] = pd.cut(gdf_ead_centroids_4326['EAD_Q50'], bins=bins, labels=labels, right=False)
    gdf_ead_centroids_4326['EAD_Q95_cat'] = pd.cut(gdf_ead_centroids_4326['EAD_Q95'], bins=bins, labels=labels, right=False)
    color_map = dict(zip(labels, colors))
    gdf_ead_centroids_4326['color_EAD_Q05_cat'] = gdf_ead_centroids_4326['EAD_Q05_cat'].map(color_map)
    gdf_ead_centroids_4326['color_EAD_Q50_cat'] = gdf_ead_centroids_4326['EAD_Q50_cat'].map(color_map)
    gdf_ead_centroids_4326['color_EAD_Q95_cat'] = gdf_ead_centroids_4326['EAD_Q95_cat'].map(color_map)

    # Terrain
    minx, miny, maxx, maxy = gdf_ead_centroids_4326.total_bounds
    buffer = 0.01
    plot_extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]

    # 2. Layout
    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "a", "a", "leg"],
            ["b", "x", "c", "leg"],
        ],
        'figsize': (7, 7),
        'kwargs': {
            'gridspec_kw': {
                'hspace': 0.1,
                'wspace': 0.1,
                'height_ratios': [2.5, 1],
                'width_ratios': [1, 0.08, 1, 0.3],
                
            },
        },
        'adjust': {'bottom': 0.075, 'top': 0.95, 'left': 0.075, 'right': 0.975}
    }
    
    # Plots
    q_to_plot = 95
    ead_col = f'EAD_Q{q_to_plot}'
    color_col = f'color_EAD_Q{q_to_plot}_cat'
    dict_to_plot = {
        'a': {'plots': [], 'style': []},
        'b': {'plots': [], 'style': []},
        'c': {'plots': [], 'style': []},
        'leg': {'style': [], 'legend': []},
        'x': {'plots': [], 'style': []},
    }
    # a
    dict_to_plot['a']['plots'].extend([
        {'plot_type': 'wms',
            'url': 'https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS',
            'layers': ['OI.OrthoimageCoverage'],
            'extent': plot_extent,
            'crs': 'EPSG:4326',
            'size': (1000, 1000),
            'alpha': 1,
            'zorder': 1
        },
        {'plot_type': 'gdf_shp',
            'gdf': gdf_ead_centroids_4326,
            'column': ead_col,
            'color': gdf_ead_centroids_4326[color_col],
            'markersize': 13,
            'zorder': 3
        }
    ])
    # b
    dict_to_plot['b']['plots'].append({'plot_type': 'scatter',
        'x': gdf_ead_centroids_4326['NEAR_DIST'],
        'y': gdf_ead_centroids_4326[ead_col],
        'color': gdf_ead_centroids_4326[color_col],
        's': 10,
        'alpha': 0.7,
        'label': f'Q{q_to_plot}',
        'linewidth': 0.2,
        'zorder': 3
    })
    # c
    data_vector = gdf_ead_centroids_4326[f'EAD_Q{q_to_plot}'].values
    max_val = data_vector.max()
    for i in range(n_bins):
        low, high = bins[i], bins[i+1]
        actual_high = high if high != np.inf else max_val * 1.1
        subset = data_vector[(data_vector >= low) & (data_vector < high)]
        if len(subset) > 0:
            dict_to_plot['c']['plots'].append({'plot_type': 'hist',
                'dataset': subset,
                'bins': [low, actual_high],
                'color': colors[i],
                'edgecolor': 'white',
                'alpha': 1.0,
                'zorder': 2
            })
    dict_to_plot['c']['plots'].extend([
        {'plot_type': 'vline', 'x': np.median(data_vector), 'color': 'red', 'linestyle': '-', 'linewidth': 1.5, 'label': 'Q50'},
        {'plot_type': 'vline', 'x': np.quantile(data_vector, 0.05), 'color': 'blue', 'linestyle': '--', 'linewidth': 1.2, 'label': 'Q05'},
        {'plot_type': 'vline', 'x': np.quantile(data_vector, 0.95), 'color': 'blue', 'linestyle': '--', 'linewidth': 1.2, 'label': 'Q95'}
    ])
    
    # Style
    # a
    def get_ax_ratio(layout_params, ax):
        fig, ax_dict = plt.subplot_mosaic(layout_params['mosaic_structure'], 
                                    figsize=layout_params['figsize'], 
                                    **layout_params['kwargs'])
        fig.subplots_adjust(**layout_params['adjust'])
        plt.draw() 
        bbox = ax_dict[ax].get_window_extent()
        real_ratio = bbox.width / bbox.height
        plt.close(fig)
        return real_ratio
    top_left_x = (4, 42, 50)
    top_left_y = (40, 24, 56)
    zoom_pct = 78.5
    real_ratio = get_ax_ratio(layout_params, 'a')
    def calc_extent(top_left_x, top_left_y, zoom_pct, real_ratio):
        xmin = -(top_left_x[0] + top_left_x[1]/60 + top_left_x[2]/3600)
        ymax = (top_left_y[0] + top_left_y[1]/60 + top_left_y[2]/3600)
        base_span = 0.02 
        width = base_span * (zoom_pct / 100)
        height = width / real_ratio 
        xmax = xmin + width
        ymin = ymax - height
        return [xmin, xmax, ymin, ymax]
    calc_extent_vals = calc_extent(top_left_x, top_left_y, zoom_pct, real_ratio)
    dict_to_plot['a']['style'].extend([
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3, 'zorder': 1},
        {'style_type': 'xlim', 'left': calc_extent_vals[0], 'right': calc_extent_vals[1]},
        {'style_type': 'ylim', 'bottom': calc_extent_vals[2], 'top': calc_extent_vals[3]},
        {'style_type': 'aspect', 'aspect': 'equal'},
        {'style_type': 'spines', 'top': True, 'right': True, 'left': True, 'bottom': True},
        {'style_type': 'ticks_params', 'which': 'both', 'labelsize': 8,
            'top': True, 'bottom': False, 'left': True, 'right': False,
            'labeltop': True, 'labelbottom': False, 'labelleft': True, 'labelright': False},
        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90},
        {'style_type': 'yticklabels', 'va': 'center'},
        {'style_type': 'major_formatter', 'axis': 'y', 'formatter_type': 'dms_suffix', 'suffix': ' N'},
        {'style_type': 'major_formatter', 'axis': 'x', 'formatter_type': 'dms_suffix', 'suffix': ' W'}
    ])
    # b
    dict_to_plot['b']['style'].extend([
        {'style_type': 'xlabel', 'label': 'Distance to River Center Line (m)', 'fontsize': 8},
        {'style_type': 'ylabel', 'label': f'EAD (€/year)', 'fontsize': 8, 'labelpad': 3},
        {'style_type': 'yscale', 'value': 'symlog', 'linthresh': 1, 'linscale': 0.2},
        {'style_type': 'xlim', 'left': 0, 'right': 350},
        {'style_type': 'ylim', 'bottom': 0, 'top': 100_000},
        {'style_type': 'xticks', 'ticks': [0, 50, 100, 150, 200, 250, 300, 350]},
        {'style_type': 'xticklabels', 'labels': ['$0$', '', '$100$', '', '$200$', '', '$300$', '']},
        {'style_type': 'yticks', 'ticks': [0, 1, 10, 100, 1_000, 10_000, 100_000]},
        {'style_type': 'yticklabels', 'labels': ['$0$', '', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$'],
            'rotation': 90, 'va': 'center', 'ha': 'right'},
        {'style_type': 'grid', 'visible': True, 'alpha': 0.3, 'linestyle': '--'},
        {'style_type': 'spines', 'top': False, 'right': False},
        {'style_type': 'ticks_params', 'axis': 'both', 'labelsize': 8}
    ])
    # c
    dict_to_plot['c']['style'].extend([
        {'style_type': 'xlabel', 'label': 'Expected Annual Damage (€)', 'fontsize': 8},
        {'style_type': 'ylabel', 'label': 'Frequency', 'fontsize': 8},
        {'style_type': 'xlim', 'left': -0.1, 'right': 10_000},
        {'style_type': 'xscale', 'value': 'symlog', 'linthresh': 0.1, 'linscale': 0.3},
        {'style_type': 'yticks', 'ticks': [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]},
        {'style_type': 'yticklabels', 'labels': ['$0$', '', '$100$', '', '$200$', '', '$300$', '', '$400$', '', '$500$', '', '$600$'],
            'rotation': 90, 'va': 'center', 'ha': 'right'},
        {'style_type': 'xticks', 'ticks': [0, 0.1, 1, 10, 100, 1_000, 10_000]},
        {'style_type': 'xticklabels', 'labels': ['$0$', '', '$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$']},
        {'style_type': 'grid', 'visible': True, 'axis': 'y', 'alpha': 0.3},
        {'style_type': 'spines', 'top': False, 'right': False, 'left': True, 'bottom': True},
        {'style_type': 'ticks_params', 'axis': 'both', 'labelsize': 8}
    ])
    # leg
    dict_to_plot['leg']['style'].append({'style_type': 'off'})
    # x
    dict_to_plot['x']['style'].append({'style_type': 'off'})
    # letter
    for key in dict_to_plot.keys():
        if key not in ['x', 'leg']:
            y_pos = 1.05 if key == 'a' else 1.12
            x_pos = -0.01 if key == 'a' else -0.04
            dict_to_plot[key]['style'].append({
                'style_type': 'letter', 
                'label': f'({key})',
                'x': x_pos,
                'y': y_pos
            })
        
    # Legend
    #a
    init_dic(dict_to_plot, 'a', 'legend')
    real_dx = get_dx_for_scalebar(calc_extent_vals, crs='EPSG:4326')
    dict_to_plot['a']['legend'].extend([
        {'legend_type': 'north_arrow',
            'x': 0.98, 'y': 0.66, 'length': 0.08, 'pad': 0.02, 'color': 'white', 'fontsize': 10},
        {'legend_type': 'scalebar',
            'dx': real_dx, 'units': 'm', 'dimension': 'si-length', 'location': 'lower right',
            'frameon': False, 'color': 'white'}
    ])
    
    # leg
    dict_to_plot['leg']['legend'].extend([
        {'legend_type': 'text',
            'content': 'Expected\nAnnual\nDamage\n(€/year)',
            'x': 0, 'y': 0.97,
            'ha': 'left', 'va': 'top', 'fontsize': 8
        },
        {'legend_type': 'text',
            'content': 'Quantiles',
            'x': 0, 'y': 0.25,
            'ha': 'left', 'va': 'top', 'fontsize': 8
        },
        {'legend_type': 'line', 'x': 0.0, 'y': 0.21, 'length': 0.15,
            'color': 'blue', 'linestyle': ':', 'linewidth': 1.2, 'alpha': 1
        },
        {'legend_type': 'text', 'content': '$Q05$', 
            'x': 0.18, 'y': 0.21, 'ha': 'left', 'va': 'center', 'fontsize': 8
        },
        {'legend_type': 'line', 'x': 0.0, 'y': 0.16, 'length': 0.15,
            'color': 'red', 'linestyle': '-', 'linewidth': 1.2, 'alpha': 1
        },
        {'legend_type': 'text', 'content': '$Q50$', 
            'x': 0.18, 'y': 0.16, 'ha': 'left', 'va': 'center', 'fontsize': 8
        },
        {'legend_type': 'line', 'x': 0.0, 'y': 0.10, 'length': 0.15,
            'color': 'blue', 'linestyle': '--', 'linewidth': 1.2, 'alpha': 1
        },
        {'legend_type': 'text', 'content': '$Q95$', 
            'x': 0.18, 'y': 0.10, 'ha': 'left', 'va': 'center', 'fontsize': 8
        }
        
    ])
    x_circ = 0.05
    x_text = 0.20
    y_start = 0.85
    y_step = 0.06
    r_circ = 0.025
    for i, (label, color) in enumerate(zip(labels, colors)):
        y_pos = y_start - (i * y_step)
        dict_to_plot['leg']['legend'].append({'legend_type': 'circle',
            'x': x_circ,
            'y': y_pos,
            'radius': r_circ,
            'color': color,
            'edgecolor': 'black',
            'linewidth': 0.5,
            'zorder': 10
        })
        dict_to_plot['leg']['legend'].append({'legend_type': 'text',
            'x': x_text,
            'y': y_pos,
            'content': label,
            'ha': 'left',
            'va': 'center',
            'fontsize': 7
        })
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'F7_EAD Map',
        'fname': 'F7_EAD_Map_v1.1.png',
        'show': True,
        'dpi': 300,
        'pad': 0.10
    }
    
    # 3. Plot
    plot_any(dict_to_plot, layout_params, save_params)
#endregion

# region D.F8 - Damage Evolution (desegregated)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    # Data sources
    c_cols_CTE_bf = [
        'c_huAPP_bf', 'c_huCLO_bf', 'c_huCOM_bf', 'c_huDEC_bf', 'c_huELE_bf',
        'c_huFAD_bf', 'c_huFUR_bf', 'c_huHHG_bf', 'c_huHHB_bf', 'c_huINS_bf',
        'c_huLEI_bf', 'c_huOTH_bf', 'c_huSPE_bf', 'c_huTOO_bf', 'c_huVEH_bf', 'c_huENG_bf']
    c_cols_CTI_bf = [
        'c_huETP_bf', 'c_huWND_bf', 'c_huPRW_bf', 'c_huITP_bf',
        'c_huSOI_bf', 'c_huSKT_bf', 'c_huRDR_bf', 'c_huPLG_bf','c_huPUM_bf','c_huCLE_bf']
    components = c_cols_CTE_bf + c_cols_CTI_bf
    full_df = load_and_concatenate_mc_parts(
        PATHS['dataset_mc_parts'], ['SN','BID','RP'] + components)
    
    # Sum of BIDs for each cols per SN (keep RP)
    df_sum = full_df.groupby(['SN', 'RP'])[c_cols_CTE_bf + c_cols_CTI_bf].sum().reset_index()
    
    # 2. Prepare configuration
    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "lega", "b", "legb"],
            ["c", "legc", "d", "legd"],
            ["e", "lege", "f", "legf"],
        ],
        'figsize': (7, 6),
        'kwargs': {
            'gridspec_kw': {
                'wspace': 0.1,
                'hspace': 0.4,
                'width_ratios': [1, 0.17, 1, 0.25],
                'height_ratios': [1, 1, 1],
            },
        },
        'adjust': {'bottom': 0.1, 'top': 0.95, 'left': 0.1, 'right': 1}
    }
    
    # Initialize dict_to_plot
    dict_to_plot = {}
    
    # Plots
    ax_map = {
        'a': 'g1_Furnishing',
        'b': 'g2_Technology',
        'c': 'g3_Personal',
        'd': 'g4_Tools_Veh',
        'e': 'g5_Surfaces',
        'f': 'g6_Fixtures',
    }
    group_stats = {}
    for g_key, items in component_groups.items():
        relevant_cols = [f'c_hu{ccc}_bf' for ccc in items if f'c_hu{ccc}_bf' in df_sum.columns]
        # Calculate Q per component and find the absolute min/max for the group
        q99_vals = df_sum[relevant_cols].quantile(0.99)
        q90_vals = df_sum[relevant_cols].quantile(0.90)
        q50_vals = df_sum[relevant_cols].quantile(0.50)
        group_stats[g_key] = {
            'max_q99': q99_vals.max(),
            'max_q90': q90_vals.max(),
            'min_q50': q50_vals.replace(0, 100).min()
        }
    # Absolut max
    abs_max_q90 = max([stats['max_q90'] for stats in group_stats.values()])
    
    overlap = 0.7
    bw_method_KDE = 0.15
    yticks_positions = [(len(RPs_unq) - 1 - i) * overlap for i in range(len(RPs_unq))]
    yticklabels = [f'{int(rp)}' for rp in RPs_unq]
    for ax_key, group_key in ax_map.items():
        init_dic(dict_to_plot, ax_key, 'plots')
        init_dic(dict_to_plot, ax_key, 'style')
        init_dic(dict_to_plot, f'leg{ax_key}', 'style')
        init_dic(dict_to_plot, f'leg{ax_key}', 'legend')
        
        
        items = component_groups[group_key]
        g_stats = group_stats[group_key]
        
        # Dynamics for this specific group
        g_xmax = g_stats['max_q99'] * 1.2
        g_linthresh = max(1, g_stats['min_q50'])
        x_eval = np.geomspace(1e-1, g_xmax, 5000)

        for i, rp in enumerate(RPs_unq):
            offset = (len(RPs_unq) - 1 - i) * overlap
            
            # 1. Plots
            # Baseline for the RP
            dict_to_plot[ax_key]['plots'].append({'plot_type': 'hline',
                'y': offset,
                'color': 'black',
                'linewidth': 0.3,
                'alpha': 1,
                'zorder': 999
            })
            
            # Distributions for each component at current RP
            for j, ccc in enumerate(items):
                col_name = f'c_hu{ccc}_bf'
                
                
                vals = df_sum[df_sum['RP'] == rp][col_name].dropna().values
                vals = vals[vals > 0]
                
                if len(vals) < 2:
                    continue
                    
                if np.ptp(vals) == 0:
                    vals = vals + np.random.normal(0, 1e-9, size=vals.shape)
                    
                try:
                    kde = gaussian_kde(vals, bw_method=bw_method_KDE)
                    y_eval = kde(x_eval)
                    
                    # Standardize distribution height
                    if y_eval.max() > 0:
                        y_eval = y_eval / y_eval.max()
                    else:
                        continue
                    
                    dict_to_plot[ax_key]['plots'].extend([
                        {'plot_type': 'fill',
                            'x': x_eval,
                            'y1': np.full_like(x_eval, offset),
                            'y2': offset + y_eval,
                            'color': ccc_colors[ccc],
                            'alpha': 0.6,
                            'zorder': i * len(items) + j + 1
                        },
                        {'plot_type': 'line',
                            'x': x_eval,
                            'y': offset + y_eval,
                            'color': 'white',
                            'linewidth': 0.2,
                            'alpha': 1,
                            'zorder': i * len(items) + j + 1
                        }
                    ])
                except (np.linalg.LinAlgError, ValueError):
                    continue
        
        # 3. Style
        dict_to_plot[f'leg{ax_key}']['style'].append({'style_type': 'off'})
        dict_to_plot[ax_key]['style'].extend([
            {'style_type': 'title', 'label': group_titles[group_key]},
            {'style_type': 'xlim', 'left': 0, 'right': abs_max_q90},
            {'style_type': 'ylim', 'bottom': -0.2, 'top': max(yticks_positions) + 1.2},
            {'style_type': 'xscale', 'value': 'symlog', 'linthresh': g_linthresh, 'linscale': 0.2},
            {'style_type': 'yticks', 'ticks': yticks_positions},
            {'style_type': 'yticklabels', 'labels': yticklabels if ax_key in ['a', 'c', 'e'] else []},
        ])
        if ax_key in ['e', 'f']:
            dict_to_plot[ax_key]['style'].append({'style_type': 'xlabel', 'label': 'Total Damage (€)'})
        if ax_key in ['a', 'c', 'e']:
            dict_to_plot[ax_key]['style'].append({'style_type': 'ylabel', 'label': 'Return Period'})
        dict_to_plot[ax_key]['style'].append({
            'style_type': 'major_formatter',
            'axis': 'x',
            'formatter_type': 'object',
            'formatter_obj': plt.FuncFormatter(lambda x, pos: '' if x == 0 else f'{x:g}')
        })
        
        # 4. Individual Legends per Group
        for i, ccc in enumerate(items):
            y_pos = 0.9 - (i * 0.1)
            dict_to_plot[f'leg{ax_key}']['legend'].extend([
                {'legend_type': 'rectangle', 'x': 0.05, 'y': y_pos, 'width': 0.2, 'height': 0.04, 
                'color': ccc_colors[ccc], 'alpha': 0.8},
                {'legend_type': 'text', 'x': 0.3, 'y': y_pos + 0.02, 'content': ccc, 
                'ha': 'left', 'va': 'center', 'fontsize': 7}
            ])
    
    # letter
    for key in dict_to_plot.keys():
        if key not in ['lega', 'legb', 'legc', 'legd', 'lege', 'legf']:
            dict_to_plot[key]['style'].append({
                'style_type': 'letter', 
                'label': f'({key})',
                'x': -0.022,
                'y': 1.12
            })
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'F8_Damage Evolution (desegregated)',
        'fname': 'F8_Damage_Evolution_Desegregated_v1.1.png',
        'show': True,
        'dpi': 300,
    }
    
    # 3. Plot
    plot_any(dict_to_plot, layout_params, save_params, global_pre_style=global_pre_style_all)
#endregion

# region D.F - GSA
RUN_BLOCK = False
if RUN_BLOCK:
    import shap
    # 1. Prepare data
    ## Data source
    output_gsa_dir = r"C:\Users\outal\Documents\MC_Parts\XGBoost"
    
    ## Set groups
    RE_RUN = False
    if RE_RUN:
        def rule_l1(c): # Division of main groups
            if c in ['IH', 'BH', 'GL', 'Cga']: return "Structure"
            if c in ['ed', 'he', 'base_value', 'BID']: return c
            if c.startswith('p_'): return "Prices (p_)"
            if c.startswith('n_'): return "Objects (n_)"
            if c.startswith('hc_'): return "C.High (hc_)"
            if c.startswith('m_'): return "Materials (m_)"
            return "Unclassified"
        
        def rule_l2(c): # Division of main groups withouth structure
            if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
            if c.startswith('p_'): return "Prices (p_)"
            if c.startswith('n_'): return "Objects (n_)"
            if c.startswith('hc_'): return "C.High (hc_)"
            if c.startswith('m_'): return "Materials (m_)"
            return "Unclassified"
        
        def rule_l3(c): # Division of main CTE and CTI groups withouth structure
            if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: 
                return c
            content_pattern = r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)'
            continent_pattern = r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)'
            prefix_map = {
                'p_': 'Prices',
                'n_': 'Objects',
                'hc_': 'C.High',
                'm_': 'Materials'
            }
            for prefix, base_name in prefix_map.items():
                if c.startswith(prefix):
                    if re.search(content_pattern, c):
                        return f"{base_name}_CTE"
                    elif re.search(continent_pattern, c):
                        return f"{base_name}_CTI"
                    else:
                        return f"{base_name}_Unclassified"
            return "Unclassified"
        
        def rule_l4(c): # Division of main groups withouth structure by floor
            if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
            for prefix, name in [('p_', 'Prices'), ('n_', 'Objects'), ('hc_', 'C.High'), ('m_', 'Materials')]:
                if c.startswith(prefix):
                    if prefix == 'm_': 
                        return f"{name} ({prefix})"
                    match = re.search(r'_(-?\d+)$', c)
                    floor = f"{match.group(1)}floor" if match else "global"
                    return f"{name} {floor} ({prefix})"
            return "Unclassified"

        def rule_l5(c): # Division of main groups withouth structure with CTE and CTI
            if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
            if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c): 
                return "Content"
            if re.search(r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)', c) or c.startswith(('p_ETP', 'p_EXF', 'p_WND', 'p_PUM', 'p_PRW', 'p_ITP', 'p_SOI', 'p_CLE', 'p_SKT', 'p_RDR', 'p_PLG', 'm_ETP', 'm_EXF', 'm_WND', 'm_PUM', 'm_PRW', 'm_ITP', 'm_SOI', 'm_CLE', 'm_SKT', 'm_RDR', 'm_PLG')): 
                return "Continent"
            return "Unclassified"
        
        def rule_l6(c): # Division of individual groups considering individual CTE and CTI
            if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: 
                return c
                
            # Pattern to capture the specific sub-group code directly
            pattern = r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH|ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)'
            
            match = re.search(pattern, c)
            if match:
                return match.group(1) # Returns the exact code found (e.g., "APP", "ETP")
                
            return "Unclassified"

        for rp in RPs_unq: # rp=5
            print(f"Setting groups for {rp}")
            shap_path = os.path.join(output_gsa_dir, f"shap_results_rp_{rp}.feather")
                
            df_shap_rp = pd.read_feather(shap_path)
            
            grouper = shap_results_grouper(df_shap_rp.columns.tolist(), separator="|")
            
            #grouper.print_unique_names(level_index=0)
            
            grouper.remove_all_levels()
            
            grouper.add_level(rules=rule_l1)
            grouper.add_level(rules=rule_l2)
            grouper.add_level(rules=rule_l3)
            grouper.add_level(rules=rule_l4)
            grouper.add_level(rules=rule_l5)
            grouper.add_level(rules=rule_l6)
            
            #grouper.print_unique_names(level_index=1)
            #grouper.print_unique_names(level_index=2)
            #grouper.print_unique_names(level_index=3)
            #grouper.print_unique_names(level_index=4)
            #grouper.print_unique_names(level_index=5)
            #grouper.print_unique_names(level_index=6)
            
            print(f"Updating and resaving {df_shap_rp}")
            df_shap_rp = grouper.update_dataframe(df_shap_rp)
            df_shap_rp.to_feather(shap_path)
    
    ## Prepare data
    # Expected Individual variables (l0)
    df_expected_l0_all = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=0, by_bid=False, min_rp_AEP_0=2)
    
    # Expected Division of main groups (l1)
    df_expected_l1_all = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=1, by_bid=False, min_rp_AEP_0=2)
    
    # Expected Division of main groups (l1) by bid
    df_expected_l1_bid = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=1, by_bid=True, min_rp_AEP_0=2)
    
    # Evolution Division of main groups (l1)
    df_evolution_l1_all = calc_rp_evolution_grouped(
        output_gsa_dir, RPs_unq, level=1, by_bid=False)
    
    # Expected Division of main groups withouth structure (l2)
    df_expected_l2_all = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=2, by_bid=False, min_rp_AEP_0=2)
    
    # Expected Division of main groups withouth structure (l2) by bid
    df_expected_l2_bid = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=2, by_bid=True, min_rp_AEP_0=2) 
    
    # Expected Division of main CTE and CTI groups withouth structure (l3)
    df_expected_l3_all = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=3, by_bid=False, min_rp_AEP_0=2)
    
    # Expected Division of main groups withouth structure by floor (l4)
    df_expected_l4_all = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=4, by_bid=False, min_rp_AEP_0=2)
    
    # Expected Division of main groups withouth structure by floor (l4) by bid
    df_expected_l4_bid = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=4, by_bid=True, min_rp_AEP_0=2)
    
    # Expected Division of main groups withouth structure with CTE and CTI (l5)
    df_expected_l5_all = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=5, by_bid=False, min_rp_AEP_0=2)
    
    # Expected Division of individual groups considering individual CTE and CTI (l6)
    df_expected_l6_all = calc_expected_grouped(
        output_gsa_dir, RPs_unq, level=6, by_bid=False, min_rp_AEP_0=2)
    
    ## Set colors
    gsa_group_colors = {
        'he':               "#ff0000",
        
        'Structure':        "#ff7300",
        'GL':               "#ca5b01ff",
        'BH':               "#a86936",
        'IH':               "#632d01",
        'Cga':              "#fc913a",
        
        'C.High':           "#00ff0d",
        
        'Prices':           "#00f7ff",
        
        'Objects':          "#4c00ff",
        
        'ed':               "#ae00ff",
        
        'Materials':        "#ff0095",
        
        'Content':          "#9293ce",
        
        'Continent':        "#c992ce",
    }

    ## Prepare data for Sankey diagram
    relations_l1_to_l2 = {
        'Structure': ['GL', 'BH', 'Cga', 'IH']
    }
    relations_l2_to_l3 = {
        'Prices (p_)':    ['Prices_CTE', 'Prices_CTI'],
        'Objects (n_)':    ['Objects_CTE', 'Objects_CTI'],
        'C.High (hc_)':   ['C.High_CTE', 'C.High_CTI'],
        'Materials (m_)': ['Materials_CTE', 'Materials_CTI']
    }
    relations_l3_to_l5 = {
        'C.High_CTI':    ['Continent'],
        'Prices_CTI':    ['Continent'],
        'Objects_CTI':   ['Continent'],
        'Materials_CTI': ['Continent'],
        'C.High_CTE':    ['Content'],
        'Prices_CTE':    ['Content'],
        'Objects_CTE':   ['Content']
    }
    relations_l5_to_l6 = {
        'Content': [
            'APP', 'CLO', 'COM', 'DEC', 'ELE', 'ENG', 'FAD', 'FUR', 
            'HHG', 'HHB', 'INS', 'LEI', 'OTH', 'SPE', 'TOO', 'VEH'
        ],
        'Continent': [
            'PUM', 'CLE', 'DHU', 'SOI', 'FRI', 'SKT', 'RDR', 'WND', 
            'PLG', 'PRW', 'EXF', 'ETP', 'ITP', 'ELS'
        ]
    }
    custom_scale_parameters = {
        'data_pct': 0.5,   # Fraction of the dataset to be isolated for expansion/reduction (0.0 to 1.0)
        'canvas_pct': 0.1       # Fraction of the visual canvas height dedicated to that data subset
    }
    nodes, links = generate_scaled_multilayer_sankey(
        df_list=[
            df_expected_l1_all, 
            df_expected_l2_all, 
            df_expected_l3_all, 
            df_expected_l5_all, 
            df_expected_l6_all
        ],
        relations_list=[
            relations_l1_to_l2, 
            relations_l2_to_l3, 
            relations_l3_to_l5, 
            relations_l5_to_l6
        ],
        colors_dict=gsa_group_colors,
        scale_params=custom_scale_parameters
    )
    for node in nodes:
        #node['name'] = ""
        node['name'] = re.sub(r'\s*\(L\d+\)', '', node['name']).strip()
    
    ## Prepare map
    k_vars = 4
    def rank_combinations(df, k=3):
        df_reset = df.reset_index()
        df_sorted = df_reset.sort_values(by=['BID', 'Q50'], ascending=[True, False])
        top_k = df_sorted.groupby('BID').head(k)
        combinations = top_k.groupby('BID')['Feature_Group'].apply(tuple)
        unique_combinations = combinations.value_counts()
        
        print(f"Number of unique combinations: {len(unique_combinations)}")
        print("\nMost common combinations:")
        print(unique_combinations.head(10))
        
        return unique_combinations
    unique_combinations = rank_combinations(df_expected_l1_bid, k=k_vars)
    acronym_map = {
        'he': 'he',
        'C.High (hc_)': 'hc',
        'Objects (n_)': 'n',
        'Structure': 'S',
        'Prices (p_)': 'p',
        'ed': 'ed',
        'Materials (m_)': 'm'
    }
    df_reset = df_expected_l1_bid.reset_index() 
    df_sorted = df_reset.sort_values(by=['BID', 'Q50'], ascending=[True, False])
    top_k = df_sorted.groupby('BID').head(k_vars).copy()
    top_k['Short_Feature'] = top_k['Feature_Group'].map(lambda x: acronym_map.get(str(x).strip(), str(x).strip()))
    combinations = top_k.groupby('BID')['Short_Feature'].apply(lambda x: ' | '.join(x)).reset_index()
    combinations.rename(columns={'Short_Feature': 'Top_7_Combo'}, inplace=True)
    unique_combos = combinations['Top_7_Combo'].unique()
    cmap = plt.get_cmap('tab20')
    color_map = {combo: mcolors.to_hex(cmap(i % 20)) for i, combo in enumerate(unique_combos)}
    combinations['combo_color'] = combinations['Top_7_Combo'].map(color_map)

    # Merge geometry metadata
    gdf_shap = gdf_buildings.merge(combinations, on='BID', how='left')
    gdf_shap_centroids = gdf_shap.copy()
    gdf_shap_centroids['geometry'] = gdf_shap.centroid
    gdf_shap_centroids_4326 = gdf_shap_centroids.to_crs(epsg=4326)

    # Calculate geographical bounds
    minx, miny, maxx, maxy = gdf_shap_centroids_4326.total_bounds
    buffer = 0.01
    plot_extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]
        
    # 2. Prepare configuration
    # Layout
    layout_params = {
        'layout': 'mosaic',
        'mosaic_structure': [
            ["a", "a"],
            ["b", "d"],
            ["c", "d"],
            ["leg","leg"],
        ],
        'figsize': (6, 8),
        'kwargs': {
            'gridspec_kw': {
                'wspace': 0.1,
                'hspace': 0.4,
                'width_ratios': [1, 1.5],
                'height_ratios': [2, 1, 1, 0.4],
            },
        },
        'adjust': {'bottom': 0.1, 'top': 0.9, 'left': 0.13, 'right': 0.9}
    }
    
    # Initialize dict_to_plot
    dict_to_plot = {}
    
    # Plots
    # Calculate extents early so we can place the text legend inside the box
    real_ratio = get_ax_ratio(layout_params, 'a')
    top_left_x = (4, 43, 00)
    top_left_y = (40, 24, 52)
    zoom_pct = 115
    calc_extent_vals = calc_extent(top_left_x, top_left_y, zoom_pct, real_ratio)
    xmin, xmax, ymin, ymax = calc_extent_vals

    # a
    init_dic(dict_to_plot, 'a', 'plots')
    dict_to_plot['a']['plots'].extend([
        {'plot_type': 'wms',
            'url': 'https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS',
            'layers': ['OI.OrthoimageCoverage'],
            'extent': plot_extent,
            'crs': 'EPSG:4326',
            'size': (1000, 1000),
            'alpha': 1,
            'zorder': 1
        },
        {'plot_type': 'gdf_shp',
            'gdf': gdf_shap_centroids_4326,
            'color': gdf_shap_centroids_4326['combo_color'].tolist(), # Use explicit mapped colors
            'markersize': 6,
            'zorder': 3
        }
    ])
    # --- BUILD THE MANUAL TEXT LEGEND ---
    # Define starting position for the legend (e.g., top-left corner)
    legend_x = xmin + (xmax - xmin) * 0.02     # 2% padding from the left edge
    legend_y = ymax - (ymax - ymin) * 0.02     # 2% padding from the top edge
    y_step = (ymax - ymin) * 0.04              # Vertical step spacing between items

    # Add a title for the manual legend
    dict_to_plot['a']['plots'].append({
        'plot_type': 'text', 'x': legend_x, 'y': legend_y,
        'text': "Top 7 Variables Rank:", 'fontsize': 8, 'color': 'white',
        'ha': 'left', 'va': 'top', 'transform': 'data', 'fontweight': 'bold',
        'path_effects': [pe.withStroke(linewidth=2, foreground="black")], 'zorder': 5
    })

    # Loop through unique combos and draw a colored square + text
    for i, combo in enumerate(unique_combos):
        combo_color = color_map[combo]
        current_y = legend_y - ((i + 1) * y_step)
        
        dict_to_plot['a']['plots'].append({
            'plot_type': 'text',
            'x': legend_x,
            'y': current_y,
            'text': f"■ {combo}", # Using a unicode block as the color marker
            'color': combo_color, 
            'fontsize': 6,
            'ha': 'left',
            'va': 'top',
            'transform': 'data',
            'path_effects': [pe.withStroke(linewidth=1, foreground="black")],
            'zorder': 5
        })
    
    # b
    init_dic(dict_to_plot, 'b', 'plots')
    bar_colors_b = []
    for grp in df_expected_l1_all.index:
        clean_grp = re.sub(r'\s*\([^)]*\)', '', str(grp)).strip()
        bar_colors_b.append(gsa_group_colors.get(clean_grp, '#cccccc'))
    
    dict_to_plot['b']['plots'].append({
        'plot_type': 'barh',
        'y': df_expected_l1_all.index.tolist(),
        'width': df_expected_l1_all['Q50'].values,
        'color': bar_colors_b,
        'linewidth': 0,
        'alpha': 1
    })
    
    for idx, row in df_expected_l1_all.iterrows():
        dict_to_plot['b']['plots'].append({
            'plot_type': 'text',
            'x': row['Q50'],
            'y': idx,
            'text': f" {row['Q50_share_%']:.2f}%",
            'transform': 'data',
            'va': 'center',
            'ha': 'left',
            'fontsize': 8,
            'color': 'black'
        })
    
    # c
    init_dic(dict_to_plot, 'c', 'plots')
    share_cols = [col for col in df_evolution_l1_all.columns if col.startswith('Share_%_RP')]
    rps_numeric = [int(col.replace('Share_%_RP', '')) for col in share_cols]
    sorted_idx = np.argsort(rps_numeric)
    share_cols_sorted = [share_cols[i] for i in sorted_idx]
    rps_categorical = [str(rps_numeric[i]) for i in sorted_idx]
    for feature_name, row in df_evolution_l1_all.iterrows():
        clean_feat = re.sub(r'\s*\([^)]*\)', '', str(feature_name)).strip()
        feat_color = gsa_group_colors.get(clean_feat, '#cccccc')
        y_shares = row[share_cols_sorted].values
        
        # Plot evolution trendline using the categorical string labels for x
        dict_to_plot['c']['plots'].append({
            'plot_type': 'line',
            'x': rps_categorical,
            'y': y_shares,
            'color': feat_color,
            'marker': 'o',
            'markersize': 3,
            'linewidth': 1.5,
            'label': feature_name
        })
        
    # d
    init_dic(dict_to_plot, 'd', 'plots')
    dict_to_plot['d']['plots'].append({
        'plot_type': 'sankey',
        'node_width': 0.02,
        'fontsize': 8,
        'nodes': nodes,
        'links': links
    })
    x_positions = np.linspace(0.1, 0.9, 5)
    dict_to_plot['d']['plots'].extend([
        {'plot_type': 'text', 'x': x_positions[0], 'y': -0.05, 'text': 'L1', 'transform': 'data', 'va': 'center', 'ha': 'center'},
        {'plot_type': 'text', 'x': x_positions[1], 'y': -0.05, 'text': 'L2', 'transform': 'data', 'va': 'center', 'ha': 'center'},
        {'plot_type': 'text', 'x': x_positions[2], 'y': -0.05, 'text': 'L3', 'transform': 'data', 'va': 'center', 'ha': 'center'},
        {'plot_type': 'text', 'x': x_positions[3], 'y': -0.05, 'text': 'L5', 'transform': 'data', 'va': 'center', 'ha': 'center'},
        {'plot_type': 'text', 'x': x_positions[4], 'y': -0.05, 'text': 'L6', 'transform': 'data', 'va': 'center', 'ha': 'center'}
    ])
    
    # Style
    # a
    init_dic(dict_to_plot, 'a', 'style')
    dict_to_plot['a']['style'].extend([
        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3, 'zorder': 1},
        {'style_type': 'xlim', 'left': calc_extent_vals[0], 'right': calc_extent_vals[1]},
        {'style_type': 'ylim', 'bottom': calc_extent_vals[2], 'top': calc_extent_vals[3]},
        {'style_type': 'aspect', 'aspect': 'equal'},
        {'style_type': 'spines', 'top': True, 'right': True, 'left': True, 'bottom': True},
        {'style_type': 'ticks_params', 'which': 'both', 'labelsize': 8,
            'top': True, 'bottom': False, 'left': True, 'right': False,
            'labeltop': True, 'labelbottom': False, 'labelleft': True, 'labelright': False},
        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90},
        {'style_type': 'yticklabels', 'va': 'center'},
        {'style_type': 'major_formatter', 'axis': 'y', 'formatter_type': 'dms_suffix', 'suffix': ' N'},
        {'style_type': 'major_formatter', 'axis': 'x', 'formatter_type': 'dms_suffix', 'suffix': ' W'},
        # Add legend for the distinct categorical color mappings
        {'style_type': 'legend', 'loc': 'upper right', 'fontsize': 5, 'title': 'Top 7 Variables Rank'}
    ])
    
    # b
    init_dic(dict_to_plot, 'b', 'style')
    x_max = df_expected_l2_all['Q50'].max()
    x_split1 = 1
    x_split2 = 10
    x_pct1 = 0.05
    x_pct2 = 0.10
    dict_to_plot['b']['style'].extend([
        {'style_type': 'xlabel', 'label': 'Expected Annual Average |SHAP|'},
        {'style_type': 'xscale', 'value': 'function', 'functions': create_3_linear_ax_scale(x_max, x_split1, x_split2, x_pct1, x_pct2)},
        {'style_type': 'xticks', 'ticks': [0, 1, 10, 50, 100, 150, 200, 250, 300]},
        {'style_type': 'xticklabels', 'labels': ['0', '1', '10', '50', '100', '150', '200', '250', '300']},
    ])
    
    # c
    init_dic(dict_to_plot, 'c', 'style')
    x_max = 75
    x_split1 = 10
    x_split2 = 50
    x_pct1 = 0.50
    x_pct2 = 0.80
    dict_to_plot['c']['style'].extend([
        {'style_type': 'xlabel', 'label': 'Return Period (RP)'},
        {'style_type': 'ylabel', 'label': 'Share % of total |SHAP|'},
        {'style_type': 'yscale', 'value': 'function', 'functions': create_3_linear_ax_scale(x_max, x_split1, x_split2, x_pct1, x_pct2)},
        {'style_type': 'yticks', 'ticks': [0, 10, 20, 30, 40, 50, 60, 70]},
        {'style_type': 'yticklabels', 'labels': ['0', '10', '', '30', '', '50', '', '70']},
        {'style_type': 'ylim', 'ymin': 0, 'ymax': 75}
    ])
    
    # d
    init_dic(dict_to_plot, 'd', 'style')
    dict_to_plot['d']['style'].extend([
        {'style_type': 'off'},
        {'style_type': 'xlim', 'xmin': -0.1, 'xmax': 1.1},
        {'style_type': 'ylim', 'ymin': 0.0, 'ymax': 1.0}
    ])
    
    # leg
    for ax in ['leg']:
        init_dic(dict_to_plot, ax, 'style')
        dict_to_plot[ax]['style'].append({'style_type': 'off'})
    
    # letter
    for key in dict_to_plot.keys():
        if key not in ['leg']:
            y_pos = 1.05 if key in ['a', 'd'] else 1.1
            x_pos = -0.05 if key in ['b', 'c', 'd'] else -0.01
            dict_to_plot[key]['style'].append({
                'style_type': 'letter', 
                'label': f'({key})',
                'x': x_pos,
                'y': y_pos
            })
    
    # Legend
    init_dic(dict_to_plot, 'leg', 'legend')
    
    # Save
    save_params = {
        'output_dir': output_dir,
        'subfolder': 'F8_Sensitivity Analysis',
        'fname': 'F8_Sensitivity_Analysis_v4.1.png',
        'show': True,
        'dpi': 300,
    }
    
    # 3. Plot
    plot_any(dict_to_plot, layout_params, save_params, global_pre_style=global_pre_style_all)
#endregion

#################################################################################
#################################################################################
##### SECTION D: RESULTS: FIGURES AND TABLES (Supplementary)
#################################################################################
#################################################################################

# region D SUPPLEMENTARY RESULTS
# C. Digital Surface Model (DSM)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    po = []
    fm_rows = get_fm_rows(DC='ed', sort_by='normal_dispersion')
    for i, (_, row) in enumerate(fm_rows.iterrows()):
        obj = {"OID": i, "BID": row["BID"], "Row": row.to_dict()}
        po.append(obj)
    po = set_po_colors(po, cmap_name='RdYlBu_r')
    po, global_limits = set_po_xy(po, n_points=1000, Qmin=0.0001, Qmax=0.9999)

    # 2. Plot data
    set_global_style()
    fig, ax = set_global_layout(layout = "1x1", figsize = (4, 4))
    plot_ax_pdf(ax, po, linewidth=0.1)

    # 3. Set styling
    set_ax_format(ax,
        title="Error Dem (ed)",
        xlabel='Meters ($m$)',
        ylabel='Probability Density Function (PDF)',
        x_lims=(global_limits['xmin'], global_limits['xmax']),
        y_lims=(global_limits['ymin'], global_limits['ymax']),
        fontsize=10, t_pad=10, g_alpha=0.3
    )

    # 4. Save plot
    save_plot('DSM_v3.png', output_dir, fig, show=True)

# C. Input Flow (if)
RUN_BLOCK = False
if RUN_BLOCK:
    dist_depth_df = pd.read_excel(PATHS['fm_flow_xlsx'])
    df_plot = dist_depth_df.copy()
    def S3_2(df_plot):
        ### Pre-process data ###
        # 0. Log-Pearson Type III PDF with Jacobian transformation
        def lp3_pdf(x, skew, loc, scale):
            # f_X(x) = f_Y(log10(x)) * (1 / (x * ln(10)))
            y = np.log10(x)
            pdf_y = stats.pearson3.pdf(y, skew, loc, scale)
            return pdf_y / (x * np.log(10))
        
        # 1. Sort rows by Scale from lower to higher
        df_plot = df_plot.sort_values(by='Scale').reset_index(drop=True)
        
        # 2. Generate distribution data
        all_min = []
        all_max = []
        for _, row in df_plot.iterrows():
            # ppf is in log10 space
            all_min.append(10**stats.pearson3.ppf(0.05, row['Skew'], row['Loc'], row['Scale']))
            all_max.append(10**stats.pearson3.ppf(0.95, row['Skew'], row['Loc'], row['Scale']))
        
        global_x_min = min(all_min)
        global_x_max = max(all_max)

        x_range = np.linspace(global_x_min, global_x_max, 10000)
        plot_data = []
        global_y_max = 0

        for _, row in df_plot.iterrows():
            y_pdf = lp3_pdf(x_range, row['Skew'], row['Loc'], row['Scale'])
            plot_data.append((x_range, y_pdf, int(row['ReturnPeriod'])))
            global_y_max = max(global_y_max, np.nanmax(y_pdf))
        
        ### Pre-configuring plot ###
        # 1. Styling configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        })
        
        # 2. Initialize plot
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        
        # 3. Gradient
        cmap = plt.get_cmap('RdYlBu_r') 
        n_lines = len(plot_data)
        
        ### Plot logic ###
        # 1. Plot curves
        for i, (x, y, rp) in enumerate(plot_data):
            color = cmap(i / (n_lines - 1)) if n_lines > 1 else cmap(0)
            ax.plot(x, y, color=color, linewidth=1, label=f'RP={int(rp)}')
        
        ### Plot Styling ###
        # 1. Formatting
        # Title
        ax.set_title('Pearson Type III Density Functions', fontsize=10, pad=10)
        
        # Labels
        ax.set_xlabel('Flow ($m^3$)', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        
        # Limits
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(0, global_y_max * 1.01)
        
        # Scale
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=0.0005, linscale=0.5)
        
        # Ticks
        target_xticks = [400, 1000, 4000, 10000]
        target_yticks = [0.0005, 0.001, 0.002, 0.005, 0.01]
        ax.set_xticks(target_xticks)
        ax.set_yticks(target_yticks)
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        
        # Ensure non-scientific format
        ax.ticklabel_format(style='plain', axis='both')
        
        # Grid
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
        # Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Legend
        ax.legend(loc='upper right', fontsize=7, frameon=False, title="Return Period", title_fontsize=8)
        
        plt.tight_layout()
        plt.savefig(r'C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\PyWS\Outputs\S3\if_v1.png', dpi=300)

    S3_2(dist_depth_df)
    
# C. Water Depth Outside Building (he).
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Get the data and unique combinations
    fm_rows = fm_df[(fm_df["DC"] == 'he') & (fm_df["BP"] > 0)]
    dist_dict = fm_rows.groupby('RP')['FN'].unique().apply(list).to_dict()

    # 2. Plot
    def S3_3(fm_rows, fo_data, rp_test, sym_scale = []):
        """
        Plots a multi-panel distribution grid (4x3 layout) where each distribution 
        type has its own designated subplot, sharing a common color scale based on BP.
        """
        # 1. Pre-process and Sort
        dist_dict = fm_rows.groupby('RP')['FN'].unique().apply(list).to_dict()
        target_dist_list = dist_dict[rp_test]
        fm_rows_sub = fm_rows[(fm_rows['RP'] == rp_test) & (fm_rows['FN'].isin(target_dist_list))]
        fm_rows_sub = fm_rows_sub.sort_values(by='BP').reset_index(drop=True)
        n_total = len(fm_rows_sub)
        
        # 2. Setup Plotting Environment
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        })
        
        # Define mapping of FN to mosaic keys
        fn_map = {
            'Constant': 'b',
            'KDE':      'c',
            'halfnorm': 'd',
            'expon':    'e',
            'lognorm':  'f',
            'gamma':    'g',
            'gumbel_r': 'h',
            'weibull_min': 'i'
        }
        
        fig = plt.figure(figsize=(12, 10))
        mosaic = """
        abc
        ade
        afg
        ahi
        """
        ax_dict = fig.subplot_mosaic(mosaic, gridspec_kw={'width_ratios': [1, 4, 4]})
        
        cmap = plt.get_cmap('RdYlBu_r')
        global_x_max = 3.0  
        
        # Track y-max per axis to normalize scales
        y_max_tracker = {key: 0 for key in fn_map.values()}

        ### 3. Plotting Loop ###
        for i, row in fm_rows_sub.iterrows():
            color = cmap(i / (n_total - 1)) if n_total > 1 else cmap(0)
            fn_type = row['FN']
            
            if fn_type not in fn_map:
                continue
                
            ax_key = fn_map[fn_type]
            ax = ax_dict[ax_key]
            params = parse_input(row['FP'])
            
            # --- Subplot a: Vertical BP Cloud (Shared) ---
            jitter = np.random.uniform(-0.1, 0.1)
            ax_dict['a'].scatter(jitter, row['BP'], c=[color], s=15, alpha=1.0, edgecolors='none')

            # --- Plotting by Type ---
            if fn_type == 'Constant':
                val = params[0] if isinstance(params, (list, tuple)) else params
                ax.axvline(x=val, color=color, linewidth=0.8)

            elif fn_type == 'KDE':
                raw_data = np.array(fo_data[row['ODC']])
                fit_data = raw_data[raw_data > 0]
                if len(fit_data) > 1:
                    kde = stats.gaussian_kde(fit_data)
                    x_vals = np.linspace(0, global_x_max, 1000)
                    y_vals = kde(x_vals)
                    ax.plot(x_vals, y_vals, color=color, linewidth=0.8)
                    y_max_tracker[ax_key] = max(y_max_tracker[ax_key], np.nanmax(y_vals))

            else:
                # Parametric distributions
                try:
                    dist_func = getattr(stats, fn_type)
                    x_vals = np.linspace(0, global_x_max, 1000)
                    y_vals = dist_func.pdf(x_vals, *params)
                    ax.plot(x_vals, y_vals, color=color, linewidth=0.8)
                    # Filter for finite values only
                    finite_y = y_vals[np.isfinite(y_vals)]
                    if len(finite_y) > 0:
                        y_max_tracker[ax_key] = max(y_max_tracker[ax_key], np.max(finite_y))
                except Exception:
                    continue

        ### 4. Final Styling ###
        # Axis a
        ax_dict['a'].set_title('BP Cloud', fontsize=10)
        ax_dict['a'].set_ylabel('Building Probability (BP)', fontsize=10)
        ax_dict['a'].set_ylim(-0.02, 1.02)
        ax_dict['a'].set_xticks([])
        ax_dict['a'].grid(True, linestyle='--', axis='y', alpha=0.3)

        # Style b through i
        for fn_name, key in fn_map.items():
            ax = ax_dict[key]
            ax.set_title(fn_name, fontsize=9, loc='left', pad=2)
            ax.set_xlim(0, global_x_max)
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(True, linestyle='--', alpha=0.2)
            
            # Set y-limits
            y_max = y_max_tracker[key]
            if key != 'b':
                # Use a fallback if y_max is not finite or zero
                if np.isfinite(y_max) and y_max > 0:
                    ax.set_ylim(0, y_max * 1.1)
                else:
                    ax.set_ylim(0, 1)
            else:
                ax.set_yticks([])

            # Handle x-axis labels: only show on bottom row (h, i)
            if key not in ['h', 'i']:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('Depth ($m$)', fontsize=8)
            
            # Scale
            if key in sym_scale:
                ax.set_yscale('symlog', linthresh=5, linscale=0.5)

        plt.tight_layout()
        save_path = f'C:/Users/outal/OneDrive/3_Personas y Proyectos/Jose - UCLM/2_Economic Valuation/PyWS/Outputs/S3/he_{rp_test}_v5.png'
        plt.savefig(save_path, dpi=300)

    S3_3(fm_rows, fo_data, 5,sym_scale = [])
    S3_3(fm_rows, fo_data, 10,sym_scale = [])
    S3_3(fm_rows, fo_data, 20,sym_scale = [])
    S3_3(fm_rows, fo_data, 50,sym_scale = ['d'])
    S3_3(fm_rows, fo_data, 100,sym_scale = ['d'])
    S3_3(fm_rows, fo_data, 200,sym_scale = ['d'])
    S3_3(fm_rows, fo_data, 500,sym_scale = ['d'])

# C. Number of Positive Floor (NF).
RUN_BLOCK = False
if RUN_BLOCK:
    fm_rows = fm_df[fm_df["DC"] == 'NF']

    def S3_6(fm_df, fo_data):
        """
        Plots the empirical frequency of the number of positive floors (NF) 
        grouped by Building Type (BT 1-6).
        """
        # 1. Filter for NF rows and relevant BTs
        fm_rows = fm_df[(fm_df["DC"] == 'NF') & (fm_df["BT"].isin([1, 2, 3, 4, 5, 6]))].copy()
        
        # 2. Styling configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        })
        
        fig, ax = plt.subplots(figsize=(7, 4))
        cmap = plt.get_cmap('RdYlBu_r')
        
        # 3. Data processing and plotting
        # We define the x-axis categories (Number of Floors)
        possible_n_floors = [1, 2, 3, 4]
        bt_list = sorted(fm_rows['BT'].unique())
        n_bts = len(bt_list)
        
        # Width of individual bars and total group width
        group_width = 0.8
        bar_width = group_width / n_bts
        
        for i, bt in enumerate(bt_list):
            # Get the ODC code for the specific BT
            odc_code = fm_rows[fm_rows['BT'] == bt]['ODC'].values[0]
            
            # Load raw empirical values
            raw_values = np.array(fo_data[odc_code])
            
            # Calculate frequencies for the categories 1-5
            counts = []
            for val in possible_n_floors:
                freq = np.mean(raw_values == val)
                counts.append(freq)
                
            # Offset bars so they are side-by-side at each x-tick
            x_indexes = np.arange(len(possible_n_floors))
            offset = (i - (n_bts - 1) / 2) * bar_width
            
            color = cmap(i / (n_bts - 1))
            ax.bar(x_indexes + offset, counts, width=bar_width, label=f'BT {bt}', color=color)

        # 4. Plot Styling
        ax.set_title('Empirical Frequency of Building Floors by Type (BT)', fontsize=10, pad=12)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_xlabel('Number of Positive Floors (NF)', fontsize=10)
        
        ax.set_xticks(range(len(possible_n_floors)))
        ax.set_xticklabels(possible_n_floors)
        ax.set_ylim(0, 1.0)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        ax.legend(title="Building Type", fontsize=8, title_fontsize=9, loc='upper right', frameon=False, ncol=2)

        plt.tight_layout()
        save_path = r'C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\PyWS\Outputs\S3\NF_v1.png'
        plt.savefig(save_path, dpi=300)

    S3_6(fm_df, fo_data)

# C. Housing Units (HU).
RUN_BLOCK = False
if RUN_BLOCK:
    fm_rows = fm_df[fm_df["DC"] == 'HU']

    def S3_7(fm_df):
        """
        Plots the Probability Mass Function (PMF) for Housing Units (HU).
        Adjusts the binomial distribution (n, p) by shifting it +1 to account 
        for the x-1 data transformation.
        """
        # 1. Filter for HU rows
        fm_rows = fm_df[fm_df["DC"] == 'HU'].copy()
        if fm_rows.empty:
            return
            
        # 2. Styling configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        })
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # 3. Distribution Logic
        # Extract params (n, p) from the first matching row
        params = parse_input(fm_rows.iloc[0]['FP'])
        n, p = params
        
        # Define the range of housing units
        # Binomial starts at 0, we shift by +1, so k starts at 1
        k_shifted = np.arange(1, n + 2) 
        # Calculate PMF for the original binomial (0 to n)
        pmf_values = stats.binom.pmf(np.arange(0, n + 1), n, p)
        
        # 4. Plotting
        # Use stem plot or bars for discrete distributions
        markerline, stemlines, baseline = ax.stem(k_shifted, pmf_values, 
                                                linefmt='C0-', 
                                                markerfmt='C0o', 
                                                basefmt=" ")
        plt.setp(stemlines, 'linewidth', 1.5)
        plt.setp(markerline, 'markersize', 6)

        # 5. Plot Styling
        ax.set_title(f'Housing Units Distribution ($HU$)', fontsize=10, pad=12)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_xlabel('Number of Units', fontsize=10)
        
        ax.set_xticks(k_shifted)
        ax.set_ylim(0, max(pmf_values) * 1.1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Text annotation for parameters
        ax.text(0.95, 0.95, f'$n={n}$\n$p={p}$', transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        plt.tight_layout()
        save_path = r'C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\PyWS\Outputs\S3\HU_v1.png'
        plt.savefig(save_path, dpi=300)

    S3_7(fm_df)

# C. Inter-floor (IH) and Basement Height (BH).
RUN_BLOCK = False
if RUN_BLOCK:
    fm_rows = fm_df[fm_df["DC"] == 'IH']

    def S3_8(fm_df):
        """
        Plots the Probability Density Function (PDF) for Inter-floor Height (IH).
        Note: scipy.stats.triang uses (c, loc, scale) where:
        - loc = min (a)
        - scale = max - min (b - a)
        - c = (mode - loc) / scale
        """
        # 1. Filter for IH row
        fm_rows = fm_df[fm_df["DC"] == 'IH'].copy()
        if fm_rows.empty:
            return
            
        # 2. Styling configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        })
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # 3. Distribution Logic
        params = parse_input(fm_rows.iloc[0]['FP'])
        c, loc, scale = params
        
        # Calculate real bounds for plotting
        a = loc
        b = loc + scale
        mode = loc + c * scale
        
        x = np.linspace(a - 0.2, b + 0.2, 1000)
        y = stats.triang.pdf(x, c, loc, scale)
        
        # 4. Plotting
        ax.plot(x, y, color='darkblue', linewidth=1)
        
        # Add vertical line for the mode
        ax.axvline(mode, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label=f'Mode: {mode:.2f}m')

        # 5. Plot Styling
        ax.set_title('Inter-floor & Basement Height ($IH, BH$)', fontsize=10, pad=12)
        ax.set_ylabel('Probability Density', fontsize=10)
        ax.set_xlabel('Height (meters)', fontsize=10)
        
        ax.set_xlim(a - 0.1, b + 0.1)
        ax.set_ylim(0, max(y) * 1.1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=8, frameon=False)

        plt.tight_layout()
        save_path = r'C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\PyWS\Outputs\S3\IH_BH_v1.png'
        plt.savefig(save_path, dpi=300)

    S3_8(fm_df)

# C. Ground Floor Level (GL).
RUN_BLOCK = False
if RUN_BLOCK:
    fm_rows = fm_df[fm_df["DC"] == 'GL']

    def S3_9(fm_df, fo_data):
        """
        Plots the density functions for Ground Floor Level (GL) differentiated by Building Type (BT).
        Includes the Distribution Function (FN) name in the legend.
        """
        # 1. Filter for GL rows and sort by BT
        fm_rows = fm_df[fm_df["DC"] == 'GL'].copy()
        fm_rows = fm_rows.sort_values(by='BT').reset_index(drop=True)
        
        # 2. Styling configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        })
        
        fig, ax = plt.subplots(figsize=(4, 4))
        cmap = plt.get_cmap('RdYlBu_r')
        
        # 3. Processing and Plotting
        x_range = np.linspace(-0.5, 3.0, 1000)
        global_y_max = 0
        n_lines = len(fm_rows)

        for i, row in fm_rows.iterrows():
            color = cmap(i / (n_lines - 1)) if n_lines > 1 else cmap(0)
            fn_type = row['FN']
            bt_val = int(row['BT'])
            
            # Updated Label to include FN
            bt_label = f'BT={bt_val}' if bt_val > 0 else 'General (BT=0)'
            full_label = f'{bt_label} ({fn_type})'
            
            y_pdf = np.zeros_like(x_range)
            
            if fn_type == 'KDE':
                raw_data = np.array(fo_data[row['ODC']])
                fit_data = raw_data[np.isfinite(raw_data)]
                if len(fit_data) > 1:
                    kde = stats.gaussian_kde(fit_data)
                    y_pdf = kde(x_range)
            else:
                try:
                    params = parse_input(row['FP'])
                    dist_func = getattr(stats, fn_type)
                    y_pdf = dist_func.pdf(x_range, *params)
                except Exception as e:
                    continue

            ax.plot(x_range, y_pdf, color=color, linewidth=1.2, label=full_label)
            global_y_max = max(global_y_max, np.nanmax(y_pdf))

        # 4. Plot Styling
        ax.set_title('Ground Floor Level ($GL$) Distributions', fontsize=10, pad=10)
        ax.set_xlabel('Elevation ($m$)', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        
        ax.set_xlim(-0.2, 2.5)
        ax.set_ylim(0, global_y_max * 1.1)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.ticklabel_format(style='plain', axis='both')
        
        # Legend with FN included
        ax.legend(loc='upper right', fontsize=7, frameon=False, 
                title="Building Type (Fit Type)", title_fontsize=8)
        
        plt.tight_layout()
        save_path = r'C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\PyWS\Outputs\S3\GL_v1.png'
        plt.savefig(save_path, dpi=300)

    S3_9(fm_df, fo_data)

# C. Groun Floor Perimeter Coeficient (Cga).
RUN_BLOCK = False
if RUN_BLOCK:
    fm_df[fm_df["DC"] == 'Cga']

    def S3_10(fm_df, fo_data):
        """
        Plots the density functions for Ground Floor Perimeter Coefficient (Cga) 
        differentiated by Building Type (BT).
        """
        # 1. Filter for Cga rows and sort by BT
        fm_rows = fm_df[fm_df["DC"] == 'Cga'].copy()
        fm_rows = fm_rows.sort_values(by='BT').reset_index(drop=True)
        
        # 2. Styling configuration
        plt.rcParams.update({
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'axes.unicode_minus': False
        })
        
        fig, ax = plt.subplots(figsize=(4, 4))
        cmap = plt.get_cmap('RdYlBu_r')
        
        # 3. Processing and Plotting
        x_range = np.linspace(4, 8, 1000)
        global_y_max = 0
        n_lines = len(fm_rows)

        for i, row in fm_rows.iterrows():
            color = cmap(i / (n_lines - 1)) if n_lines > 1 else cmap(0)
            fn_type = row['FN']
            bt_val = int(row['BT'])
            
            bt_label = f'BT={bt_val}' if bt_val > 0 else 'General'
            full_label = f'{bt_label} ({fn_type})'
            
            y_pdf = np.zeros_like(x_range)
            
            if fn_type == 'KDE':
                raw_data = np.array(fo_data[row['ODC']])
                fit_data = raw_data[np.isfinite(raw_data)]
                if len(fit_data) > 1:
                    kde = stats.gaussian_kde(fit_data)
                    y_pdf = kde(x_range)
            else:
                try:
                    params = parse_input(row['FP'])
                    dist_func = getattr(stats, fn_type)
                    y_pdf = dist_func.pdf(x_range, *params)
                except Exception:
                    continue

            ax.plot(x_range, y_pdf, color=color, linewidth=1.2, label=full_label)
            global_y_max = max(global_y_max, np.nanmax(y_pdf))

        # 4. Plot Styling
        ax.set_title('Perimeter Coefficient ($C_{ga}$) Distributions', fontsize=10, pad=10)
        ax.set_xlabel('Coefficient Value', fontsize=10)
        ax.set_ylabel('Probability Density', fontsize=10)
        
        ax.set_xlim(4, 8)
        ax.set_ylim(0, global_y_max * 1.1)
        
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.legend(loc='upper right', fontsize=7, frameon=False, 
                title="Building Type (Fit Type)", title_fontsize=8)
        
        plt.tight_layout()
        save_path = r'C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\PyWS\Outputs\S3\Cga_v1.png'
        plt.savefig(save_path, dpi=300)

    S3_10(fm_df, fo_data)

# C. Content plots (n)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    po = []
    fm_rows = get_fm_rows(DC=[f"n_{x}" for x in CTEs_unq], sort_by='BP')
    for i, (_, row) in enumerate(fm_rows.iterrows()):
        obj = {"OID": i, "BT": row["BT"], "Row": row.to_dict()}
        po.append(obj)
    po = set_po_colors(po, cmap_name='tab10',by='DC')
    po, global_limits = set_po_xy(po, n_points=1000, Qmin=0.05, Qmax=0.99)
    po = set_po_labels(po, pf = 'n')

    # 2. Plot data
    set_global_style()
    fig, (ax_l, ax_r) = set_global_layout(layout = "1x2", figsize = (4, 4),gridspec_kw={'width_ratios': [1, 5]})
    po = set_po_sizes(po, (5, 20), by='BP')
    plot_ax_dotcloud(ax_l, po, edgecolors='none')
    po = set_po_sizes(po, (0.1, 1.4), by='BP')
    plot_ax_pdf(ax_r, po, marker='o', markersize=2, markeredgecolor='none', markeredgewidth=0)

    # 3. Set styling
    set_ax_format(
        ax_l,
        ylabel='Bernouilli Probability ($BP$)',
        y_lims=(-0.01, 1.01),
        hide_spines=['top','right','bottom'],
        x_ticks=[],
        x_ticklabels=[],
        fontsize=8,
        g_alpha=0.1
    )
    set_ax_format(
        ax_r,
        title='',
        xlabel='Units (No.)', 
        ylabel='Probability Density Function (PDF)',
        x_lims=(global_limits['xmin'], global_limits['xmax']),
        y_lims=(global_limits['ymin'], global_limits['ymax']),
        fontsize=8,
        g_alpha=0.3,
        x_scale='symlog',
        x_scale_kwargs={'linthresh': 3, 'linscale': 0.5}
    )
    set_legend(ax_r, po, by='Label', where='outside_right')
    fig.suptitle('Content Count (n_CTE)', 
                fontsize=10)

    # 4. Save plot
    save_plot('n_CTE_v3.png', output_dir, fig, show=True)

# C. Content plots (p)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    po = []
    fm_rows = get_fm_rows(DC=[f"p_{x}" for x in CTEs_unq])
    for i, (_, row) in enumerate(fm_rows.iterrows()):
        obj = {"OID": i, "Row": row.to_dict()}
        po.append(obj)
    po = set_po_colors(po, cmap_name='tab10')
    po, global_limits = set_po_xy(po, n_points=1000, Qmin=0.0001, Qmax=0.99)
    po = set_po_labels(po, pf = 'p')

    # 2. Plot data
    set_global_style()
    fig, ax = set_global_layout(layout = "1x1", figsize = (4, 4))
    plot_ax_pdf(ax, po, linewidth=0.5)

    # 3. Set styling
    set_ax_format(ax,
        title="Content Price (p_CTE)",
        xlabel='Price ($€$)',
        ylabel='Probability Density Function (PDF)',
        x_lims=(0, global_limits['xmax']),
        y_lims=(global_limits['ymin'], global_limits['ymax']),
        x_scale='symlog',
        x_scale_kwargs={'linthresh': 100, 'linscale': 0.9},
        fontsize=10, t_pad=10, g_alpha=0.3
    )
    set_legend(ax, po, by='Label', where='outside_right')

    # 4. Save plot
    save_plot('p_CTE_v1.png', output_dir, fig, show=True)

# C. Content plots (ff)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    po = []
    fm_rows = get_fm_rows(DC=[f"ff_{x}" for x in CTEs_unq])
    for i, (_, row) in enumerate(fm_rows.iterrows()):
        obj = {"OID": i, "BT": row["BT"], "Row": row.to_dict()}
        po.append(obj)
    po = set_po_colors(po, cmap_name='tab10',by='DC')
    po, global_limits = set_po_xy(po, n_points=1000, Qmin=0.0001, Qmax=0.9999)
    po = set_po_labels(po, pf = 'ff')
    
    # 2. Plot data
    set_global_style()
    fig, ax = set_global_layout(layout = "1x1", figsize = (4, 4))
    plot_ax_cdf(ax, po, linewidth=0.5)
    
    # 3. Set styling
    set_ax_format(ax,
        title="Content Fragility Function (ff_CTE)",
        xlabel='Meters ($m$)',
        ylabel='Cumulative Density Function (CDF)',
        x_lims=(0, global_limits['xmax']),
        y_lims=(0, 1),
        fontsize=10, t_pad=10, g_alpha=0.3
    )
    set_legend(ax, po, by='Label', where='outside_right')

    # 4. Save plot
    save_plot('ff_CTE_v1.png', output_dir, fig, show=True)

# C. Continent plots (n_SKT, n_RDR, n_WND, n_PLG)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    po = []
    target_DC = ['n_SKT', 'n_RDR', 'n_WND', 'n_PLG']
    fm_rows = get_fm_rows(DC=target_DC, sort_by='BP')
    for i, (_, row) in enumerate(fm_rows.iterrows()):
        obj = {"OID": i, "BT": row["BT"], "Row": row.to_dict()}
        po.append(obj)
    po = set_po_colors(po, cmap_name='tab10',by='DC')
    po, global_limits = set_po_xy(po, n_points=1000, Qmin=0.0001, Qmax=0.9999)
    po = set_po_labels(po, pf = 'n')

    # 2. Plot data
    set_global_style()
    fig, (ax_l, ax_r) = set_global_layout(layout = "1x2", figsize = (4, 4),gridspec_kw={'width_ratios': [1, 5]})
    plot_ax_dotcloud(ax_l, po, edgecolors='none', markersize=5)
    plot_ax_pdf(ax_r, po, linewidth=1, marker='o', markersize=2, markeredgecolor='none', markeredgewidth=0)

    # 3. Set styling
    set_ax_format(
        ax_l,
        ylabel='Bernouilli Probability ($BP$)',
        y_lims=(-0.01, 1.01),
        hide_spines=['top','right','bottom'],
        x_ticks=[],
        x_ticklabels=[],
        fontsize=8,
        g_alpha=0.1
    )
    set_ax_format(
        ax_r,
        title='',
        xlabel='Units (No.)', 
        ylabel='Probability Density Function (PDF)',
        x_lims=(global_limits['xmin'], global_limits['xmax']),
        y_lims=(global_limits['ymin'], global_limits['ymax']),
        fontsize=8,
        g_alpha=0.3,
        x_scale='symlog',
        x_scale_kwargs={'linthresh': 3, 'linscale': 0.5}
    )
    set_legend(ax_r, po, by='Label', where='outside_right')
    fig.suptitle('Continent Count (n_CTE)', 
                fontsize=10)

    # 4. Save plot
    save_plot('n_CTI_v1.png', output_dir, fig, show=True)

# C. Continent (p_PUM, p_CLE, p_SOI, pr_PRW, p_PRW, p_SKT, p_RDR, p_WND, p_PLG, p_ETP)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    po = []
    target_DC = ['p_PUM', 'p_CLE', 'p_SOI', 'pr_PRW', 'p_PRW', 'p_SKT', 'p_RDR', 'p_WND', 'p_PLG', 'p_ETP']
    fm_rows = get_fm_rows(DC=target_DC)
    for i, (_, row) in enumerate(fm_rows.iterrows()):
        obj = {"OID": i, "BT": row["BT"], "Row": row.to_dict()}
        po.append(obj)
    po = set_po_colors(po, cmap_name='tab10',by='DC')
    po, global_limits = set_po_xy(po, n_points=1000, Qmin=0, Qmax=1)
    po = set_po_labels(po, pf = 'p')
    
    # 2. Plot data
    set_global_style()
    fig, ax = set_global_layout(layout = "1x1", figsize = (4, 4))
    plot_ax_pdf(ax, po, linewidth=0.5)
    
    # 3. Set styling
    set_ax_format(ax,
        title="Contintent Price (p_CTI)",
        xlabel='Price ($€$)',
        ylabel='Probability Density Function (PDF)',
        x_lims=(0, global_limits['xmax']),
        y_lims=(global_limits['ymin'], global_limits['ymax']),
        x_scale='symlog',
        x_scale_kwargs={'linthresh': 25, 'linscale': 0.75},
        y_scale='symlog',
        y_scale_kwargs={'linthresh': 0.1, 'linscale': 0.75},
        fontsize=10, t_pad=10, g_alpha=0.3
    )
    set_legend(ax, po, by='Label', where='outside_right')

    # 4. Save plot
    save_plot('p_CTI_v1.png', output_dir, fig, show=True)

# C. Continent (ff_SOI, ff_PRW, ff_SKT, ff_RDR, ff_WND, ff_PLG)
RUN_BLOCK = False
if RUN_BLOCK:
    # 1. Prepare data
    po = []
    target_DC = ['ff_SOI', 'ff_PRW', 'ff_SKT', 'ff_RDR', 'ff_WND', 'ff_PLG']
    fm_rows = get_fm_rows(DC=target_DC)
    for i, (_, row) in enumerate(fm_rows.iterrows()):
        obj = {"OID": i, "BT": row["BT"], "Row": row.to_dict()}
        po.append(obj)
    po = set_po_colors(po, cmap_name='tab10',by='DC')
    po, global_limits = set_po_xy(po, n_points=1000, Qmin=0, Qmax=1)
    po = set_po_labels(po, pf = 'ff')
    
    # 2. Plot data
    set_global_style()
    fig, ax = set_global_layout(layout = "1x1", figsize = (4, 4))
    plot_ax_cdf(ax, po, linewidth=0.5)
    
    # 3. Set styling
    set_ax_format(ax,
        title="Continent Fragility Function (ff_CTI)",
        xlabel='Meters ($m$)',
        ylabel='Cumulative Density Function (CDF)',
        x_lims=(0, global_limits['xmax']),
        y_lims=(0, 1),
        fontsize=10, t_pad=10, g_alpha=0.3
    )
    set_legend(ax, po, by='Label', where='outside_right')

    # 4. Save plot
    save_plot('ff_CTI_v1.png', output_dir, fig, show=True)

#endregion
