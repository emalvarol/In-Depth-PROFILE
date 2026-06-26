"""
In-Depth-PROFILE Distribution preparation for economic valuation Module

This module handles the pre-treatment and probability distribution fitting 
for the economic valuation model.
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats
from scipy.optimize import minimize
from tqdm import tqdm
from pandarallel import pandarallel

class DistributionFitter:
    """
    Handles the pre-treatment and function fitting for the economic valuation model.
    Extracts, prepares, and fits statistical distributions for building, content, 
    continent, event features, and DEM errors.
    """
    def __init__(self, config_paths, config_codes, config_return_periods, config_dist_catalog):
        """
        Initializes the fitter with centralized framework paths and catalogs.
        """
        self.paths = config_paths
        self.codes = config_codes
        self.return_periods = config_return_periods
        self.dist_catalog = config_dist_catalog
        
        # Paths for saving objects
        self.fm_path = self.paths['fm_pkl']
        self.fo_path = self.paths['fo_pkl']
        
        # General params
        self.prints = False
        self.boorstrap_n = 5000

    # =========================================================================
    # A.1 Create files to save fitted functions and load input data
    # =========================================================================
    def initialize_saving_objects(self):
        """
        Initializes saving objects (fm - function main, fo - function observed).
        Pre-saves an empty framework setup if files do not exist, or loads 
        existing tracking databases from disk.
        """
        # Define tracking structure schemas
        fm_col_fid = ['FID']      # Unique identifier for each fitted function
        fm_col_obs = ['ODC']      # Observed Data Code (Links to fo_data columns if raw obs exist)
        fm_col_dataset = ['DC']   # Dataset Column identifier matching core config codes
        fm_col_filters = ['BT', 'BID', 'RP']
        fm_col_samples = ['MAT']  # Special filter tracking: material code must be sampled first
        fm_col_data = [
            'DT',                 # Distribution Type ('d' = discrete, 'c' = continuous)
            'BP',                 # Bernoulli Probability (Hurdle threshold entry rate)
            'FN',                 # Function Name (Scipy family string or 'empirical'/'KDE')
            'FP',                 # Serialized Function Parameters
        ]
        
        fm_columns = fm_col_fid + fm_col_obs + fm_col_dataset + fm_col_filters + fm_col_samples + fm_col_data

        # 1. Create baseline binary files if they are missing from storage directory
        if not os.path.exists(self.fm_path) and not os.path.exists(self.fo_path):
            # Initialize empty structural tracking table
            fm_df_init = pd.DataFrame(columns=fm_columns)
            fm_df_init.to_pickle(self.fm_path)
            
            # Initialize empty observed tracking catalog
            fo_data_init = {}
            with open(self.fo_path, 'wb') as f:
                pickle.dump(fo_data_init, f)

        # 2. Load persistent data trackers into class attributes for downstream modifications
        self.fm_df = pd.read_pickle(self.fm_path)
        with open(self.fo_path, 'rb') as f:
            self.fo_data = pickle.load(f)
    
    def load_input_data(self):
        """
        Loads and pre-processes input data tables including surveys, web-scraped asset prices,
        HEC-RAS flood depth samples, and cadaster building geometries.
        Stores the loaded data as instance attributes.
        """
        # ---------------------------------------------------------------------
        # 1. Load and Transform Survey Data
        # ---------------------------------------------------------------------
        print("Transforming and loading survey data")
        def transform_survey_to_long(df):
            # Rename columns using reverse lookup of self.codes['Survey']
            rename_dict = {v: k for k, v in self.codes['Survey'].items()}
            df = df.rename(columns=rename_dict)
            
            # Map content groups if applicable
            content_map = {v: k for k, v in self.codes['Content'].items()}
            df['GRP'] = df['GRP'].map(content_map).fillna(df['GRP'])
            
            # Melt survey data from wide to long format
            value_vars = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
            df_survey_long = pd.melt(
                df,
                id_vars=list(self.codes['Survey'].keys()),
                value_vars=value_vars,
                var_name='SU',
                value_name='VAL'
            )
            
            # Format Survey Unit (SU) identifiers and map Building Type (BT)
            df_survey_long['SU'] = df_survey_long['SU'].str[1:].astype(int)
            bt_map = df_survey_long[df_survey_long['ATR'] == 'BT'].set_index('SU')['VAL']
            df_survey_long['BT'] = df_survey_long['SU'].map(bt_map).astype(int)
            
            # Reorder columns to place key identifiers at the front
            meta_cols = [c for c in df_survey_long.columns if c not in ['SU', 'BT', 'VAL']]
            ordered_cols = ['SU', 'BT'] + meta_cols + ['VAL']
            return df_survey_long[ordered_cols]

        df_survey = pd.read_excel(self.paths['survey'], sheet_name="Data")
        self.df_survey_long = transform_survey_to_long(df_survey)

        # ---------------------------------------------------------------------
        # 2. Load Scraped Content Prices
        # ---------------------------------------------------------------------
        print("Loading price data")
        self.df_prices = pd.read_excel(self.paths['prices_content'], sheet_name="Sheet1")
        
        # ---------------------------------------------------------------------
        # 3. Load and Filter HEC-RAS Depth Samples
        # ---------------------------------------------------------------------
        print("Loading depth data")
        df_depth_raw = pd.read_pickle(self.paths['depth_samples_pkl'])
        
        # Keep only buildings flooded (depth > 0) in at least one Return Period (RP)
        bids_flooded = df_depth_raw.groupby('BID')['he'].transform('max') > 0
        self.df_depth_samples = df_depth_raw[bids_flooded].reset_index(drop=True)
        
        # ---------------------------------------------------------------------
        # 4. Load Building Geometries and Filter by Flooded Asset Subset
        # ---------------------------------------------------------------------
        print("Loading building data")
        gdf_buildings_all = gpd.read_file(self.paths['buildings_shp'])
        
        # Optimize boundary footprints to track unique flooded building IDs (BIDs)
        flooded_bids_unq = self.df_depth_samples['BID'].unique()
        self.gdf_buildings = gdf_buildings_all[gdf_buildings_all['BID'].isin(flooded_bids_unq)].copy()
        self.gdf_buildings_sample = gpd.read_file(self.paths['buildings_sample_shp'])
        
        # ---------------------------------------------------------------------
        # 5. Extract and Set Operational Framework Codes
        # ---------------------------------------------------------------------
        self.sus_unq = self.df_survey_long['SU'].unique()
        self.bts_unq = self.df_survey_long['BT'].unique()
        self.ctes_unq = list(self.codes['Content'].keys())
        self.ctis_unq = ['ETP', 'WND', 'PUM', 'PRW', 'ITP', 'SOI', 'CLE', 'SKT', 'RDR', 'PLG']
        self.rps_unq = list(self.return_periods.keys())
        
        # Generate return-period specific flooded tracking dictionaries
        self.bids_flooded_unq_by_rp = {
            rp: self.df_depth_samples[
                (self.df_depth_samples['RP'] == rp) & (self.df_depth_samples['he'] > 0)
            ]['BID'].unique().tolist()
            for rp in self.rps_unq
        }
        
        # Identify buildings requiring default structural characterization fallbacks
        self.bids_non_characterized = self.gdf_buildings[
            self.gdf_buildings['BT'] == 0
        ]['BID'].unique().tolist()
        
        return (self.df_survey_long, self.df_prices, self.df_depth_samples, self.gdf_buildings)
    
    # =========================================================================
    # A.2 Define functions (General, Check, Data prep, Fitting, Table/Plots)
    # =========================================================================
    # --- Data preparation functions ---
    def prints_if(self, p):
        if self.prints: print(p)

    def serialize_params(self, params):
        return str(tuple(params)) if isinstance(params, (tuple, list)) else str(params)

    def _expand_df_survey_long_by_weights(self, df_survey_long, df_filtered, weight_by):
        """
        Expands survey long dataframe by replacing a subset with an expanded version.
        
        Parameters:
        -----------
        df_survey_long : pd.DataFrame
            The master long-format survey DataFrame.
        df_filtered : pd.DataFrame
            The pre-filtered subset of rows to be expanded.
        weight_by : tuple
            A pair containing (val_attr, count_attr) strings, e.g., ('Mat', 'N').
            
        Returns:
        --------
        pd.DataFrame
            A combined master DataFrame with the original filtered subset rows 
            removed and replaced by the expanded multi-row observations.
        """
        # 0. Work on isolated local copies to prevent unintended mutations
        df_master = df_survey_long.copy()
        df_sub = df_filtered.copy()
        val_attr, count_attr = weight_by

        # 1. Define identifying contextual index columns
        idx_cols = [col for col in df_sub.columns if col not in ['ATR', 'VAL']]

        # 2. Extract specific values and count attributes from subset rows
        vals_df = df_sub[df_sub['ATR'] == val_attr].copy().rename(columns={'VAL': val_attr})
        counts_df = df_sub[df_sub['ATR'] == count_attr].copy().rename(columns={'VAL': count_attr})
        
        # 3. Join target values with their corresponding occurrence multipliers
        df_merged = vals_df.drop(columns='ATR').merge(
            counts_df[idx_cols + [count_attr]], 
            on=idx_cols, 
            how='inner'
        )
        
        # 4. Perform row replication according to the count vector
        df_merged = df_merged.dropna(subset=[val_attr, count_attr])
        df_expanded = df_merged.loc[df_merged.index.repeat(df_merged[count_attr].astype(int))].copy()
        
        # 5. Restructure dataframe schema back to original master shape matching df_survey_long
        df_expanded['ATR'] = val_attr
        df_expanded['VAL'] = df_expanded[val_attr]
        df_expanded = df_expanded[df_survey_long.columns]
        
        # 6. Recombine: drop stale original records via source indices and append expanded variants
        df_final = pd.concat([
            df_master[~df_master.index.isin(df_filtered.index)], 
            df_expanded
        ], ignore_index=True)
        
        return df_final

    def _expand_df_survey_long_by_mat(self, df_filtered, mat_codes_dict):
        """
        Decodes binary material combinations and expands the DataFrame.
        
        Parameters:
        -----------
        df_filtered : pd.DataFrame
            The pre-filtered subset containing encoded material combinations in the 'VAL' column.
        mat_codes_dict : dict
            Standardized bitwise data dictionary mapping power-of-two keys to 
            material descriptors (e.g., self.codes['MAT']).
            
        Returns:
        --------
        pd.DataFrame
            An expanded DataFrame where bitwise material combinations are exploded 
            into dedicated individual material asset rows.
        """
        # 0. Work on a local isolated copy to prevent side-effects on the source dataframe
        df_sub = df_filtered.copy()

        # 1. Bitwise decoding logic helper
        def decode_to_list(code):
            try:
                val_int = int(code)
                # Use bitwise AND operation to extract all constituent power-of-two flag keys
                return [k for k in mat_codes_dict.keys() if (val_int & k) == k]
            except (ValueError, TypeError):
                return []

        # 2. Expand the dataset structure
        # Compile lists of individual bit flags matching the asset entry
        df_sub['VAL_LIST'] = df_sub['VAL'].apply(decode_to_list)
        
        # Explode the lists into separate, distinct observation rows
        df_expanded = df_sub.explode('VAL_LIST')
        
        # 3. Restructure and clean temporary tracking artifacts
        df_expanded['VAL'] = df_expanded['VAL_LIST']
        df_expanded = df_expanded.drop(columns=['VAL_LIST'])
        
        # Safeguard: purge records that failed bit deconstruction (resulting in null states)
        df_expanded = df_expanded.dropna(subset=['VAL'])
        
        return df_expanded

    def _fullfill_df_survey_long_with_zeros(self, df_filtered, full_index):
        """
        Appends missing combinations from full_index to df_filtered with VAL=0,
        preserving SU-level metadata and setting default item placeholders.

        Parameters:
        -----------
        df_filtered : pd.DataFrame
            The long-format survey subset DataFrame containing existing observations.
        full_index : pd.DataFrame
            The complete lookup tracking matrix containing all target structural combinations 
            (e.g., cross-product of valid SUs, Groups, and Attributes).

        Returns:
        --------
        pd.DataFrame
            The structural union of original data entries concatenated with missing rows 
            initialized to zero.
        """
        # 0. Work on a local isolated copy to safeguard original dataframe states
        df = df_filtered.copy()
        merge_cols = full_index.columns.tolist()
        
        # 1. Identify missing combination rows using an asymmetric left join indicator
        missing = full_index.merge(
            df[merge_cols + ['VAL']], 
            on=merge_cols, 
            how='left', 
            indicator=True
        ).query("_merge == 'left_only'").drop(columns=['_merge'])
        
        # 2. Assign zero asset values and apply default tracking placeholders
        missing['VAL'] = 0
        missing['ITV'] = 0
        missing['ITM'] = '-'

        # 3. Retrieve and project missing multi-level context metadata via SU index matching
        metadata_cols = ['BT', 'VTP', 'DOM', 'ATR']
        available_metadata = [c for c in metadata_cols if c in df.columns]
        
        if available_metadata:
            su_map = df[['SU'] + available_metadata].drop_duplicates('SU').set_index('SU')
            
            for col in available_metadata:
                missing[col] = missing['SU'].map(su_map[col])

        # 4. Return combined aligned dataframe matrix
        return pd.concat([df, missing], ignore_index=True)

    def _sum_df_survey_long_by_su(self, df_filtered, sum_by=['SU']):
        """
        Aggregates room-level counts to house-level (SU) counts while 
        preserving key survey contextual metadata.

        Parameters:
        -----------
        df_filtered : pd.DataFrame
            The pre-filtered long-format survey data subset to be aggregated.
        sum_by : list, default=['SU']
            The columns used to group the aggregation.

        Returns:
        --------
        pd.DataFrame
            An aggregated DataFrame with unique grouping coordinates, summed values, 
            and structural metadata preserved.
        """
        # 1. Identify contextual metadata to preserve (assumed constant within each group)
        # Exclude columns actively used in the 'sum_by' list to prevent grouping collisions
        available_meta = ['SU', 'BT', 'VTP', 'DOM', 'GRP', 'ITM', 'ITV', 'ATR']
        meta_to_keep = [m for m in available_meta if m in df_filtered.columns and m not in sum_by]

        # 2. Enforce numeric typing on a standalone local data slice to avoid warnings
        df_working = df_filtered.copy()
        df_working['VAL'] = pd.to_numeric(df_working['VAL'], errors='coerce')

        # 3. Construct multi-column aggregation mapping definitions
        agg_dict = {'VAL': 'sum'}
        for col in meta_to_keep:
            agg_dict[col] = 'first'

        # 4. Execute group-by reduction and reset index geometry
        df_summed = (
            df_working.groupby(sum_by)
            .agg(agg_dict)
            .reset_index()
        )
        
        return df_summed

    def _create_dist_from_df(self, df_filtered, group_cols=['BT'], val_col='VAL', add_default_dist=None, astype=float):
        """
        Creates a distribution dataframe from a pre-filtered survey DataFrame.
        Groups empirical values into tuples and handles compiling default zero-coded 
        macro aggregates to prevent downstream simulation gaps.

        Parameters:
        -----------
        df_filtered : pd.DataFrame
            The pre-filtered long-format survey subset.
        group_cols : list, default=['BT']
            The columns used to group the observation arrays.
        val_col : str, default='VAL'
            The target column holding quantitative measurements to fit.
        add_default_dist : str or list, optional
            Column names to isolate for specific macro-level aggregates.
        astype : type, default=float
            Numeric type conversion target for quantitative elements (e.g., int, float).

        Returns:
        --------
        pd.DataFrame
            A structural tracking table containing unique combination coordinates 
            and their associated raw sample tuples mapped under the 'OBS' column.
        """
        # 1. Fetch raw data and enforce casting on an isolated local copy
        data_raw = df_filtered.copy()
        data_raw[val_col] = data_raw[val_col].astype(astype)
        
        # 2. Extract and package empirical sample observations into coordinate group tuples
        obs_df = (
            data_raw.groupby(group_cols)[val_col]
            .apply(lambda x: tuple(x.values))
            .reset_index(name='OBS')
        )

        # 3. Add default zeroed fallback aggregates if requested
        if add_default_dist is not None:
            # Normalize single string input variables into list containers
            defaults_to_keep = [add_default_dist] if isinstance(add_default_dist, str) else add_default_dist
            aggregates = []

            # Create sub-global baseline datasets (e.g., BT=0, GRP=actual_code)
            for col_to_keep in defaults_to_keep:
                if col_to_keep in group_cols:
                    # Aggregate sample records strictly using the defined target column
                    marg = (
                        data_raw.groupby(col_to_keep)[val_col]
                        .apply(lambda x: tuple(x.values))
                        .reset_index(name='OBS')
                    )
                    
                    # Force all remaining coordinate attributes inside the grouping space to zero
                    for col in group_cols:
                        if col != col_to_keep:
                            marg[col] = 0
                    aggregates.append(marg[group_cols + ['OBS']])

            # 4. Create Global-Global baseline aggregate row (e.g., BT=0, GRP=0)
            if len(defaults_to_keep) > 1 or (len(group_cols) == 1 and defaults_to_keep[0] in group_cols):
                global_obs = tuple(data_raw[val_col].values)
                global_row = {col: [0] for col in group_cols}
                global_row['OBS'] = [global_obs]
                aggregates.append(pd.DataFrame(global_row))

            # 5. Concatenate original matrices and purge duplicate tracking coordinates
            if aggregates:
                obs_df = pd.concat([obs_df] + aggregates, ignore_index=True)
                obs_df = obs_df.drop_duplicates(subset=group_cols).reset_index(drop=True)

        return obs_df

    def _ensure_default_dist(self, dist_df):
        """
        Ensures default zeroed aggregates are included in the distribution dataframe.
        Identifies missing content group codes (GRP) against the framework's configuration codes,
        duplicates the generic baseline row template (GRP=0), and appends the corrected placeholders.

        Parameters:
        -----------
        dist_df : pd.DataFrame
            The target distribution tracking DataFrame to sweep for group code completeness.

        Returns:
        --------
        pd.DataFrame
            The expanded distribution tracking DataFrame guaranteeing full category coverage.
        """
        # 1. Cross-reference existing groups against the single source of truth configuration keys
        content_codes = set(self.codes["Content"].keys())
        existing_groups = set(dist_df['GRP'].unique())
        missing_grps = content_codes - existing_groups

        # 2. Extract the generic baseline template row corresponding to the global indicator (0)
        default_template = dist_df[dist_df['GRP'] == 0].copy()
        
        # 3. Compile duplicate structural records to bridge missing data gaps
        new_rows = []
        if not default_template.empty:
            for grp in missing_grps:
                temp_df = default_template.copy()
                temp_df['GRP'] = grp
                new_rows.append(temp_df)

        # 4. Concatenate back into a complete, non-destructive execution array
        if new_rows:
            dist_df = pd.concat([dist_df] + new_rows, ignore_index=True)
            
        return dist_df

    # --- Fitting functions ---
    def _get_bernoulli_p(self, obs_tuple: tuple) -> float:
        """
        Calculates the exact ratio of non-zero observations to total observations.
        This provides the baseline probability entry parameter for zero-inflated Hurdle Models.

        Parameters:
        -----------
        obs_tuple : tuple
            A collection of quantitative empirical sample values.

        Returns:
        --------
        float
            The calculated success probability value bounded between 0.0 and 1.0.
        """
        if not obs_tuple:
            return 0.0
            
        return sum(1 for x in obs_tuple if x > 0) / len(obs_tuple)

    def _fit_discrete_distribution(self, obs_tuple, hurdle_fix=False, fit_only_empirical=False, min_sample_size=5, n_mc_samples=999, alpha=0.05, accepted_max=None):
        """
        Fits discrete distributions to non-zero item counts.
        Uses a shift (data - 1) to handle the hurdle model requirement (intensity >= 1).
        
        Parameters:
        -----------
        obs_tuple : tuple
            A collection of quantitative empirical sample values.
        hurdle_fix : bool, default=False
            Whether to apply a zero-inflated hurdle model shift adjustment (data - 1).
        fit_only_empirical : bool, default=False
            If True, skips parametric checks and forces an Empirical distribution fallback.
        min_sample_size : int, default=5
            Minimum required sample size to execute a rigorous Chi-Squared evaluation.
        n_mc_samples : int, default=999
            Number of Monte Carlo bootstrap replications for goodness-of-fit evaluations.
        alpha : float, default=0.05
            Significance threshold marker below which candidate functions are rejected.
        accepted_max : int/float, optional
            Physical validation upper boundary to filter extreme tail anomalies.

        Returns:
        --------
        pd.Series
            Contains ['FN', 'FP'] where 'FN' is the selected function name string 
            and 'FP' contains the serialized parameter string.
        """
        import numpy as np
        import pandas as pd
        from scipy import stats
        from scipy.optimize import minimize
        from scipy.special import beta

        def to_py(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                return [to_py(i) for i in x]
            return x.item() if hasattr(x, 'item') else float(x)
        
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
        
        def perform_discrete_chisquare_test(data, dist_name, params, min_sample_size=5):
            n_samples = len(data)

            if min_sample_size <= 5:
                return 0.0, False

            obs_values, obs_counts = np.unique(data, return_counts=True)
            obs_dict = dict(zip(obs_values.tolist(), obs_counts.tolist()))

            dist = getattr(stats, dist_name)
            v_min, _ = dist.support(*params)
            if dist_name in ['dlaplace']:
                v_min = int(max(0, v_min))
            v_max = int(max(data))
            
            all_values = np.arange(v_min, v_max + 1)

            expected_probs = dist.pmf(all_values, *params)
            
            left_tail_prob = dist.cdf(v_min - 1, *params) if v_min > 0 else 0
            expected_probs[0] += left_tail_prob

            right_tail_prob = 1.0 - dist.cdf(v_max, *params)
            expected_probs[-1] += right_tail_prob
            
            expected_counts = expected_probs * n_samples
            observed_counts = np.array([obs_dict.get(v, 0) for v in all_values])

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
                    
            if current_exp > 0:
                if len(f_obs_folded) > 0:
                    f_obs_folded[-1] += current_obs
                    f_exp_folded[-1] += current_exp
                else:
                    return 0.0, False

            n_params = len(params)
            ddof = len(f_obs_folded) - 1 - n_params

            if ddof <= 0:
                return 0.0, False

            chi_stat, p_value = stats.chisquare(
                to_py(f_obs_folded),
                to_py(f_exp_folded),
                ddof=n_params
            )

            return to_py(chi_stat), to_py(p_value)

        def sample_from_distribution(dist_name, params, size):
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
                return stats.randint.rvs(low, high, size=size)

            raise ValueError(f"Unsupported distribution: {dist_name}")

        def perform_discrete_bootstrap_chisquare_test(data, dist_name, n_mc_samples=999, hurdle_fix=False, min_sample_size=5):
            data = np.asarray(data, dtype=int)

            fit_res = fit_one_distribution(data, dist_name, hurdle_fix)
            if fit_res is not None:
                params, k, log_l = fit_res
            else:
                return None

            chi_stat_obs, p_value_obs = perform_discrete_chisquare_test(data, dist_name, params, min_sample_size=min_sample_size)

            boot_stats = []
            size = len(data)
            for _ in range(n_mc_samples):
                data_sample = sample_from_distribution(dist_name, params, size)
                fit_res_sample = fit_one_distribution(data_sample, dist_name)
                if fit_res_sample is not None:
                    params_sample, k_sample, log_l_sample = fit_res
                    chi_stat_sample, p_value_sample = perform_discrete_chisquare_test(
                        data_sample,
                        dist_name,
                        params_sample,
                        min_sample_size=min_sample_size
                    )
                    boot_stats.append(chi_stat_sample)
                else:
                    continue
            
            if len(boot_stats) < 20:
                return 0.0, p_value_obs

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
                return pd.Series(["Constant", self.serialize_params((0,))])
            return pd.Series(["Empirical", None])
        
        # --- 2: Constant Check ---
        if n_samples == 0:
            return pd.Series(["Constant", self.serialize_params((0,))])
        unique_vals = np.unique(data)
        if len(unique_vals) == 1:
            return pd.Series(["Constant", self.serialize_params((int(unique_vals[0]),))])
        
        # --- 3: Distribution Fitting & Bootstrap Evaluation ---
        final_candidates_chi = []
        dist_names = ['poisson', 'geom', 'nbinom', 'logser', 'randint']
        for d_name in dist_names:
            fit_res = fit_one_distribution(data, d_name, hurdle_fix)
            if fit_res is None:
                continue
            params, k, log_l = fit_res
            
            if accepted_max is not None:
                dist = getattr(stats, d_name)
                try:
                    q999 = dist.ppf(0.999, *params)
                    if hurdle_fix:
                        if d_name in ['poisson', 'nbinom']:
                            q999 += 1
                    
                    if q999 > accepted_max:
                        continue
                except RuntimeError:
                    continue

            bootstrap_res = perform_discrete_bootstrap_chisquare_test(
                data, 
                d_name, 
                n_mc_samples=n_mc_samples, 
                hurdle_fix=hurdle_fix,
                min_sample_size=min_sample_size
            )
            
            if bootstrap_res is not None:
                p_val_bs, p_val_obs = bootstrap_res
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
            best = min(final_candidates_chi, key=lambda x: x['AIC'])
            return pd.Series([best['FN'], self.serialize_params(best['FP'])])

        # --- 5: Empirical Frequency (Fallback) ---
        return pd.Series(["Empirical", None])

    def _define_discrete_distribution(self, function, params):
        """
        Returns the distribution name and serialized parameters for discrete data.
        
        Parameters:
        -----------
        function : str
            The short statistical identifier family name ('randint', 'triang', 'binom').
        params : tuple or list
            The raw engineering bounds or sample shape dimensions.
            
        Returns:
        --------
        pd.DataFrame
            A single-row DataFrame containing columns ['FN', 'FP'] where 'FN' is the 
            distribution family name and 'FP' holds the serialized parameters.
        """
        if function == 'randint':
            min_val, max_val = params
            # scipy.stats.randint params: low (inclusive), high (exclusive)
            new_params = (int(min_val), int(max_val) + 1)
            dist_info = pd.Series(['randint', self.serialize_params(new_params)])
            
        elif function == 'triang':
            # params: (low, mode, high)
            # Note: Discrete triangular is often handled via custom logic or 
            # as a continuous distribution rounded to integers.
            low, mode, high = params
            new_params = (float(low), float(mode), float(high))
            dist_info = pd.Series(['triang', self.serialize_params(new_params)])
            
        elif function == 'binom':
            n, p = params
            # scipy.stats.binom params: n, p
            new_params = (int(n), float(p))
            dist_info = pd.Series(['binom', self.serialize_params(new_params)])
            
        dist_df = pd.DataFrame([dist_info.values], columns=['FN', 'FP'])
        return dist_df

    def _fit_continuous_distribution(
        self,
        obs_tuple,
        hurdle_fix=False,
        dist_names_to_fit=['uniform'],
        fit_only_KDE=False,
        min_sample_size=5,
        n_mc_samples=5000,
        alpha=0.05,
        accepted_max=None,
        accepted_min=None,
        avoid_bs_size=500
    ):
        """
        Fits continuous distributions using the Cramer-von Mises (CvM) statistic and bootstrap.
        Validates candidate profiles dynamically, dropping curves whose physical boundaries 
        exceed pre-established thresholds to avoid sampling highly divergent tail anomalies.
        """
        import numpy as np
        import pandas as pd
        from scipy import stats
        import warnings

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
                return pd.Series(["Constant", self.serialize_params((0.0,))])
            return pd.Series(["KDE", None])

        # --- 2: Constant Check ---
        if n_samples == 0:
            return pd.Series(["Constant", self.serialize_params((0.0,))])
        unique_vals = np.unique(data)
        if len(unique_vals) == 1:
            return pd.Series(["Constant", self.serialize_params((float(unique_vals[0]),))])
        if n_samples < min_sample_size:
            return pd.Series(["KDE", None])

        # --- 3: Distribution Fitting & Bootstrap Evaluation ---
        candidates = []
        for d_name in dist_names_to_fit:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')  # Silence fitting noise
                try:
                    if not hasattr(stats, d_name):
                        continue
                    dist = getattr(stats, d_name)

                    # HURDLE check: Force floc=0 for positive-only distributions
                    positive_only = ['lognorm', 'gamma', 'weibull_min', 'expon', 'genpareto', 'pareto']
                    fit_kwargs = {'floc': 0} if d_name in positive_only else {}
                    
                    # 3.1 Fit the distribution parameters
                    try:
                        params = dist.fit(data, **fit_kwargs)
                        k = len(params)
                        log_l = np.sum(dist.logpdf(data, *params))
                        aic = get_aic(log_l, k)
                        res_1s = stats.cramervonmises(data, d_name, args=params)
                        p_val_1s = float(res_1s.pvalue)
                    except Exception:
                        continue

                    # 3.2 Check for extreme values (Q05 and/or Q95) if limits are provided
                    q05 = float(dist.ppf(0.05, *params))
                    q95 = float(dist.ppf(0.95, *params))
                    if accepted_min is not None and q05 < accepted_min:
                        continue
                    if accepted_max is not None and q95 > accepted_max:
                        continue
                    
                    # 3.3. Perform Bootstrap CvM Test directly (if size allows bypass threshold)
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
                        # Extract parameter positional layout mapping names
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

        # --- 4: Selection ---
        valid_candidates = [
            c for c in candidates 
            if float(c['P_BS']) > float(alpha)
        ]
        if valid_candidates:
            best = min(valid_candidates, key=lambda x: x['AIC'])
            return pd.Series([best['FN'], self.serialize_params(best['FP'])])

        # --- 5: KDE (Fallback) ---
        return pd.Series(["KDE", None])

    def _define_continuous_distribution(self, function, params, dist_df=None):
        """
        Returns the distribution name and serialized parameters for a continuous distribution.
        If dist_df is provided, it applies the distribution to every row of the DataFrame.
        
        Parameters:
        -----------
        function : str
            Distribution family name short identifier ('triang', 'truncnorm', 'lognorm', 'norm').
        params : tuple or list
            Raw continuous physical parameter dimensions or bounds.
        dist_df : pd.DataFrame, optional
            DataFrame to append the configuration results to as columns 'FN' and 'FP'.
            
        Returns:
        --------
        pd.DataFrame
            An updated DataFrame with assigned columns if dist_df was passed, 
            or a standalone single-row tracking DataFrame.
        """
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
            fn, fp = 'triang', self.serialize_params(to_py(final_params))

        elif function == 'truncnorm':
            t_min, t_mode, t_max = params
            def objective(s):
                if s <= 0: 
                    return 1e10
                loc = t_mode
                a_gen, b_gen = (t_min - loc) / s, (t_max - loc) / s
                current_coverage = stats.truncnorm.cdf(t_max, a_gen, b_gen, loc=loc, scale=s) - \
                                   stats.truncnorm.cdf(t_min, a_gen, b_gen, loc=loc, scale=s)
                return (current_coverage - 0.99)**2
            res = minimize(objective, x0=[(t_max - t_min) / 4], method='Nelder-Mead')
            s_opt, loc_opt = res.x[0], t_mode
            dist_params = ((t_min - loc_opt) / s_opt, (t_max - loc_opt) / s_opt, loc_opt, s_opt)
            fn, fp = 'truncnorm', self.serialize_params(to_py(dist_params))

        elif function == 'lognorm':
            t_min, t_mode, t_max = params
            def objective(vars):
                s, scale = vars
                if s <= 0 or scale <= 0: 
                    return 1e10
                current_mode = scale * np.exp(-(s**2))
                current_coverage = stats.lognorm.cdf(t_max, s, scale=scale) - \
                                   stats.lognorm.cdf(t_min, s, scale=scale)
                return (current_mode - t_mode)**2 + (current_coverage - 0.99)**2
            res = minimize(objective, x0=[0.5, t_mode], method='Nelder-Mead')
            s_opt, scale_opt = res.x
            dist_params = (s_opt, 0, scale_opt)
            fn, fp = 'lognorm', self.serialize_params(to_py(dist_params))

        elif function == 'norm':
            fn, fp = 'norm', self.serialize_params(to_py(params))
        
        else:
            raise ValueError(f"Function {function} not supported.")

        # 2. Return Logic
        if dist_df is not None:
            result_df = dist_df.copy()
            result_df['FN'] = fn
            result_df['FP'] = fp
            return result_df
        else:
            return pd.DataFrame([[fn, fp]], columns=['FN', 'FP'])

    # --- Functions updating ---
    def _bulk_update_functions_files(self, dist_df, mapping_config):
        """
        Bulk updates the fm_df and fo_data objects with new distribution fits.
        Parses dynamic configuration mappings, pairs parameters with raw sample arrays,
        and saves persistent trackers back to disk binary matrices.

        Parameters:
        -----------
        dist_df : pd.DataFrame
            The distribution table containing optimized parameters and empirical coordinates.
        mapping_config : dict
            A specification schema mapping fm_df columns to fixed constants or source columns.
            Example: {'BT': ('col', 'BT'), 'DC': 'NF', 'DT': 'd', 'FN': ('col', 'FN')}
        """
        start_time = time.time()
        self.prints_if("=" * 60)
        self.prints_if("Executing bulk framework registration updates...")

        # 1. Initialize an empty workspace copy matching the framework schema columns
        new_records = []

        # 2. Iterate through optimized distribution parameter rows
        for idx, row in dist_df.iterrows():
            record = {col: np.nan for col in self.fm_df.columns}
            
            # Extract identifiers or map inline constants as specified by configuration rules
            for fm_col, map_rule in mapping_config.items():
                if isinstance(map_rule, tuple) and map_rule[0] == 'col':
                    src_col = map_rule[1]
                    record[fm_col] = row[src_col] if src_col in dist_df.columns else np.nan
                else:
                    record[fm_col] = map_rule

            # 3. Synchronize raw sample arrays with the observed dictionary warehouse (fo_data)
            if 'OBS' in row and isinstance(row['OBS'], tuple):
                dc_code = record.get('DC', 'UNKNOWN')
                
                # Check for and resolve np.nan constraints cleanly prior to int evaluation
                bt_val = record.get('BT', 0)
                bt_code = int(bt_val) if pd.notna(bt_val) else 0
                
                bid_val = record.get('BID', 0)
                bid_code = int(bid_val) if pd.notna(bid_val) else 0
                
                rp_val = record.get('RP', 0)
                rp_code = int(rp_val) if pd.notna(rp_val) else 0
                
                # Construct a unique, traceable observed data code identifier key
                odc_key = f"{dc_code}_BT{bt_code}_BID{bid_code}_RP{rp_code}"
                
                # Store sample array and bind key back to main index data row
                self.fo_data[odc_key] = row['OBS']
                record['ODC'] = odc_key

            new_records.append(record)

        # 4. Integrate data arrays into class memory spaces
        df_new_updates = pd.DataFrame(new_records)
        
        # Eliminate legacy matching configuration lines to prevent duplicate overlaps
        filter_keys = [k for k in ['DC', 'BT', 'BID', 'RP'] if k in mapping_config]
        if filter_keys and not self.fm_df.empty:
            # Drop obsolete matching index profiles safely across dimensions
            cond = np.ones(len(self.fm_df), dtype=bool)
            for key in filter_keys:
                if key in df_new_updates.columns and not df_new_updates[key].isna().all():
                    unique_vals = df_new_updates[key].unique()
                    cond = cond & (self.fm_df[key].isin(unique_vals))
            
            self.fm_df = self.fm_df[~cond].copy()

        # Concatenate records and synchronize structural identification keys
        self.fm_df = pd.concat([self.fm_df, df_new_updates], ignore_index=True)
        self.fm_df['FID'] = range(len(self.fm_df))

        # 5. Commit state variations down to persistent binary files on disk
        self.fm_df.to_pickle(self.fm_path)
        with open(self.fo_path, 'wb') as f:
            pickle.dump(self.fo_data, f)

        execution_delta = time.time() - start_time
        self.prints_if(f"Successfully serialized records to disk files in {execution_delta:.4f}s.")
        self.prints_if("=" * 60)
    
    # =========================================================================
    # A.3 to A.7 Specific Preparation Methods
    # =========================================================================
    
    def prepare_building_distributions(self, var_to_fit = []):
        """
        A.3 Prepare distributions for building variables (e.g., BT, NF, BF, HU, IH, GL, BH, Cga)
        Extracts, formats, fits, and updates the structural distribution parameters to the persistent
        functions framework storage.
        """
        # ---------------------------------------------------------------------
        # BT - Building Type
        # ---------------------------------------------------------------------
        if "BT" in var_to_fit:
            self.prints_if("Preparing data for BT")
            df_filtered = self.gdf_buildings[
                (self.gdf_buildings['BT'].notna()) & 
                (self.gdf_buildings['BT'] != 0)
            ].copy()
            self.prints_if(df_filtered)
            general_obs = tuple(df_filtered['BT'].astype(int).values)
            dist_df = pd.DataFrame({
                'BID': [0], 
                'OBS': [general_obs]
            })
            self.prints_if("Fitting discrete dist for BT")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for BT")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'BT',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # NF - Number of Floors
        # ---------------------------------------------------------------------
        if "NF" in var_to_fit:
            self.prints_if("Preparing data for NF")
            df_filtered = gdf_buildings[
                (gdf_buildings['BT'].notna()) & 
                (gdf_buildings['BT'] != 0)
            ].copy()
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=["BT"],
                val_col="NF",
                add_default_dist=["BT"],
                astype=int
            )
            self.prints_if("Fitting discrete dist for NF")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for NF")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'NF',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # BF - Basement Floors
        # ---------------------------------------------------------------------
        if "BF" in var_to_fit:
            self.prints_if("Preparing data for BF")
            df_filtered = gdf_buildings[
                (gdf_buildings['BT'].notna()) & 
                (gdf_buildings['BT'] != 0)
            ].copy()
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=["BT"],
                val_col="BF",
                add_default_dist=["BT"],
                astype=int
            )
            self.prints_if("Fitting discrete dist for BF")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for BF")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'BF',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # HU - Number of Households
        # ---------------------------------------------------------------------
        if "HU" in var_to_fit:
            self.prints_if("Defining discrete dist for HU")
            dist_df = self._define_discrete_distribution(
                function='binom',
                params=(3, 0.45)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for HU")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'HU',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # IH - Inter-floor Height
        # ---------------------------------------------------------------------
        if "IH" in var_to_fit:
            self.prints_if("Defining continuous dist for IH")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(2.40, 2.50, 3.00)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for IH")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'IH',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )
        
        # ---------------------------------------------------------------------
        # GL - Ground Floor Level
        # ---------------------------------------------------------------------
        if "GL" in var_to_fit:
            self.prints_if("Preparing data for GL")
            df_filtered = self.df_survey_long[
                (self.df_survey_long['GRP'] == 'Difcota') & 
                (~self.df_survey_long['ROM'].isin(['Sotano'])) & 
                (self.df_survey_long['VAL'].notna())
            ]
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=["BT"],
                val_col="VAL",
                add_default_dist=["BT"],
                astype=float
            )
            self.prints_if("Fitting continuous dist for GL")
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_continuous_distribution,
                dist_names_to_fit=self.dist_catalog['C_SS_i'],
                fit_only_KDE=False,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=5,
                accepted_min=-5
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for GL")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'GL',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # BH - Basement Height
        # ---------------------------------------------------------------------
        if "BH" in var_to_fit:
            self.prints_if("Defining continuos dist for BH")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(2.40, 2.50, 3.00)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for BH")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'BH',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # Cga - Ground Floor Perimeter Coeficient
        # ---------------------------------------------------------------------
        if "Cga" in var_to_fit:
            self.prints_if("Preparing data for Cga")
            df_filtered = self.df_survey_long[
                (self.df_survey_long['ATR'] == 'Cga') & 
                (self.df_survey_long['VAL'].notna())
            ]
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=float
            )
            self.prints_if("Fitting continuous dist for Cga")
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_continuous_distribution,
                dist_names_to_fit=self.dist_catalog['C_SS_i_vff3'],
                fit_only_KDE=False,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=10,
                accepted_min=1
            )
            self.prints_if(dist_df)
            self.prints_if("Updating file for Cga")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'Cga',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

    def prepare_content_distributions(self, var_to_fit = []):
        """
        A.4 Prepare distributions for content asset features (n_, p_, ff_ for CTEs).
        Pre-processes, formats, and fits parametric statistical models for item quantities,
        market prices, and vulnerability altitude parameters.
        """
        # ---------------------------------------------------------------------
        # 1. Item Quantities (n_): Discrete Counts with Hurdle Bernoulli Probabilities
        # ---------------------------------------------------------------------
        if "n_CTE" in var_to_fit:
            self.prints_if("Preparing data for Content Quantities (n)")
            df_filtered = self._sum_df_survey_long_by_su(
                self.df_survey_long[
                    (self.df_survey_long['DOM'] == 'Contenido') & 
                    (self.df_survey_long['ATR'] == 'N') & 
                    (self.df_survey_long['VAL'].notna())
                ],
                sum_by=['SU', 'GRP']
            )
            full_index_n = (
                df_filtered[['SU']]
                .drop_duplicates()
                .merge(
                    pd.DataFrame({'GRP': self.ctes_unq}), 
                    how='cross'
                )
            )
            df_filtered = self._fullfill_df_survey_long_with_zeros(df_filtered, full_index_n)
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT', 'GRP'],
                add_default_dist='GRP',
                astype=int
            )
            
            self.prints_if("Fitting discrete dist & bernouilli for n_CTE")
            dist_df['BP'] = dist_df['OBS'].apply(self._get_bernoulli_p)
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                hurdle_fix=True,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.025,
                accepted_max=500
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for n_CTE")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': ('col', 'GRP'),
                    'DCV': 'n',
                    'DT': 'd',
                    'BP': ('col', 'BP'),
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                }
            )

        # ---------------------------------------------------------------------
        # 2. Asset Unit Prices (p_): Continuous Bounds
        # ---------------------------------------------------------------------
        if "p_CTE" in var_to_fit:
            self.prints_if("Preparing data for Content Unit Prices (p)")
            dist_df = self._create_dist_from_df(
                self.df_prices,
                group_cols=['Group_Code'],
                val_col='Scraped_Price',
                add_default_dist='Group_Code',
                astype=float
            )
            self.prints_if("Fitting continuous dist for p_CTE")
            # Segment vehicles into an independent high-variance fitting vector
            dist_df_a = dist_df.loc[dist_df['Group_Code'] != 'VEH'].copy()
            dist_df_b = dist_df.loc[dist_df['Group_Code'] == 'VEH'].copy()
            
            dist_df_a[['FN', 'FP']] = dist_df_a['OBS'].parallel_apply(
                self._fit_continuous_distribution,
                dist_names_to_fit=self.dist_catalog['C_BS_p'],
                fit_only_KDE=False,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=2000,
                accepted_min=0,
                avoid_bs_size=300
            )
            dist_df_b[['FN', 'FP']] = dist_df_b['OBS'].apply(
                self._fit_continuous_distribution,
                dist_names_to_fit=self.dist_catalog['C_BS_p'],
                fit_only_KDE=False,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=70000,
                accepted_min=0,
                avoid_bs_size=300
            )
            dist_df = pd.concat([dist_df_a, dist_df_b], axis=0).sort_index()
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for Content Unit Prices (p)")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': ('col', 'Group_Code'),
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                }
            )

        # ---------------------------------------------------------------------
        # 3. Fragility Altitude Ranges (ff_): Continuous Truncated Boundaries
        # ---------------------------------------------------------------------
        if "ff_CTE" in var_to_fit:
            self.prints_if("Preparing data for Vulnerability Fragility Altitudes (ff)")
            df_filtered = self.df_survey_long[
                (self.df_survey_long['DOM'] == 'Contenido') & 
                (self.df_survey_long['ATR'] == 'Alt') & 
                (self.df_survey_long['VAL'].notna())
            ]
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT', 'GRP'],
                val_col='VAL',
                add_default_dist=['BT', 'GRP'],
                astype=float
            )
            dist_df = dist_df[dist_df['OBS'].apply(len) >= 5].reset_index(drop=True).copy()
            self.prints_if("Fitting discrete dist for ff_CTE")
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_continuous_distribution,
                dist_names_to_fit=self.dist_catalog['C_SS_i_vff3'],
                fit_only_KDE=False,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=3,
                accepted_min=0,
                avoid_bs_size=300
            )
            dist_df = self._ensure_default_dist(dist_df)
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for Content Fragility Altitudes (ff)")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': ('col', 'GRP'),
                    'DCV': 'ff',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                }
            )

    def prepare_continent_distributions(self, var_to_fit=[]):
        """
        A.5 Prepare distributions for continent subcomponents (structural mitigation, finishes, and utilities).
        Pre-processes, formats, and fits statistical parameters for pumping, dehumidification, cleaning, 
        flooring, partition walls, painting, openings, and electrical arrays.
        """
        # ---------------------------------------------------------------------
        # PUM - Pumping Costs
        # ---------------------------------------------------------------------
        if "p_PUM" in var_to_fit:
            self.prints_if("Defining continuous dist for p_PUM")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(1.21, 2.84, 4.87)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_PUM")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'PUM',
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # DHU - Dehumidification Costs
        # ---------------------------------------------------------------------
        if "p_DHU" in var_to_fit:
            self.prints_if("Defining continuous dist for p_DHU")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(18, 23.66, 35)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_DHU")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'DHU',
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # CLE - Cleaning Costs
        # ---------------------------------------------------------------------
        if "p_CLE" in var_to_fit:
            self.prints_if("Defining continuous dist for p_CLE")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(1.45, 2.73, 3.64)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_CLE")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'CLE',
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # SOI - Soil / Flooring Systems
        # ---------------------------------------------------------------------
        if "p_SOI" in var_to_fit:
            self.prints_if("Preparing continuous cost matrices for p_SOI by materials")
            soi_mat = {
                0: 40,  # None
                512: 27.5,  # Wood
                4096: 35,   # Epoxi
                8: 50,      # Cement
                16: 40      # Ceramic
            }
            dist_list = []
            for mat_code, mean_val in soi_mat.items():
                temp_df = self._define_continuous_distribution(
                    function='triang',
                    params=(15, mean_val, 80)
                )
                temp_df['MAT'] = mat_code
                dist_list.append(temp_df)
            dist_df = pd.concat(dist_list, ignore_index=True)
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_SOI")
            self._bulk_update_functions_files(
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

        if "m_SOI" in var_to_fit:
            self.prints_if("Preparing data for material distribution m_SOI")
            df_filtered = self._expand_df_survey_long_by_mat(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Suelo') & 
                    (self.df_survey_long['ATR'] == 'Mat') & 
                    (self.df_survey_long['VAL'].notna())
                ],
                self.codes['MAT']
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete dist for m_SOI")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for m_SOI")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'SOI',
                    'DCV': 'm',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "ff_SOI" in var_to_fit:
            self.prints_if("Defining continuous threshold model for ff_SOI")
            dist_df = self._define_continuous_distribution(
                function='truncnorm',
                params=(0.2, 0.4, 0.6)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for ff_SOI")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'SOI',
                    'DCV': 'ff',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # PRW - Partition Walls
        # ---------------------------------------------------------------------
        if "pr_PRW" in var_to_fit:
            self.prints_if("Defining continuous removal cost distribution for pr_PRW")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(14.4, 20.0, 28.0)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for pr_PRW")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'PRW',
                    'DCV': 'pr',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "m_PRW" in var_to_fit:
            self.prints_if("Preparing weighted material dataset for m_PRW")
            df_expanded_w = self._expand_df_survey_long_by_weights(
                self.df_survey_long, 
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Paredes Interiores') & 
                    (self.df_survey_long['ITM'] == 'Pared')
                ], 
                ('Mat', 'N')
            )
            df_filtered = self._expand_df_survey_long_by_mat(
                df_expanded_w[
                    (df_expanded_w['GRP'] == 'Paredes Interiores') & 
                    (df_expanded_w['ITM'] == 'Pared') & 
                    (df_expanded_w['ATR'] == 'Mat') & 
                    (df_expanded_w['VAL'].notna())
                ],
                self.codes['MAT']
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete dist for m_PRW")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for m_PRW")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'PRW',
                    'DCV': 'm',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "ff_PRW" in var_to_fit:
            self.prints_if("Defining continuous fragility threshold model for ff_PRW")
            dist_df = self._define_continuous_distribution(
                function='truncnorm',
                params=(1.5, 1.7, 1.9)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for ff_PRW")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'PRW',
                    'DCV': 'ff',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "p_PRW" in var_to_fit:
            self.prints_if("Defining continuous reconstruction cost distribution for p_PRW")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(50, 90, 150)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_PRW")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'PRW',
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # ETP / EXF / FRI - Painting & Wall Linings
        # ---------------------------------------------------------------------
        if "p_ETP" in var_to_fit:
            self.prints_if("Defining continuous painting unit cost for p_ETP")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(4, 12, 50)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_ETP")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'ETP',
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "m_EXF" in var_to_fit:
            self.prints_if("Preparing dataset for facade material distribution m_EXF")
            df_filtered = self._expand_df_survey_long_by_mat(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Revestimiento Fachada') & 
                    (self.df_survey_long['ATR'] == 'Mat') & 
                    (self.df_survey_long['VAL'].notna())
                ],
                self.codes['MAT']
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete dist for m_EXF")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for m_EXF")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'EXF',
                    'DCV': 'm',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "ff_FRI" in var_to_fit:
            self.prints_if("Preparing data for continuous wainscoting height threshold ff_FRI")
            df_filtered = self.df_survey_long[
                (self.df_survey_long['GRP'] == 'Paredes Interiores') & 
                (self.df_survey_long['ITM'] == 'Friso') & 
                (self.df_survey_long['ATR'] == 'Alt') & 
                (self.df_survey_long['VAL'].notna())
            ]
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=float
            )
            self.prints_if("Define continuous fit for ff_FRI")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(df_filtered['VAL'].min(), df_filtered['VAL'].mean(), df_filtered['VAL'].max()),
                dist_df=dist_df
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for ff_FRI")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'FRI',
                    'DCV': 'ff',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "n_FRI" in var_to_fit:
            self.prints_if("Preparing data for wainscoting structural count aggregation n_FRI")
            df_filtered = self._sum_df_survey_long_by_su(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Paredes Interiores') & 
                    (self.df_survey_long['ITM'] == 'Friso') & 
                    (self.df_survey_long['ATR'] == 'N')
                ],
                sum_by=['SU']
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete hurdle distribution pipeline for n_FRI")
            dist_df['BP'] = dist_df['OBS'].apply(self._get_bernoulli_p)
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_discrete_distribution,
                hurdle_fix=True,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=500
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for n_FRI")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'FRI',
                    'DCV': 'n',
                    'DT': 'd',
                    'BP': ('col', 'BP'),
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # SKT - Skirting Boards
        # ---------------------------------------------------------------------
        if "p_SKT" in var_to_fit:
            self.prints_if("Preparing continuous cost profiles for p_SKT based on materials")
            skt_mat = {0: 5, 512: 7, 4096: 2.8, 1: 8.4, 2048: 4}
            dist_list = []
            for mat_code, mean_val in skt_mat.items():
                temp_df = self._define_continuous_distribution(
                    function='triang',
                    params=(2.79, mean_val, 10)
                )
                temp_df['MAT'] = mat_code
                dist_list.append(temp_df)
            dist_df = pd.concat(dist_list, ignore_index=True)
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_SKT")
            self._bulk_update_functions_files(
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

        if "n_SKT" in var_to_fit:
            self.prints_if("Preparing numerical survey counts for skirting board array n_SKT")
            df_filtered = self._sum_df_survey_long_by_su(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Paredes Interiores') & 
                    (self.df_survey_long['ITM'] == 'Rodapies') & 
                    (self.df_survey_long['ATR'] == 'N')
                ],
                sum_by=['SU']
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete hurdle distribution for n_SKT")
            dist_df['BP'] = dist_df['OBS'].apply(self._get_bernoulli_p)
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_discrete_distribution,
                hurdle_fix=True,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.025,
                accepted_max=500
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for n_SKT")
            self._bulk_update_functions_files(
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

        if "m_SKT" in var_to_fit:
            self.prints_if("Preparing material metrics for skirting profiles m_SKT")
            df_expanded_w = self._expand_df_survey_long_by_weights(
                self.df_survey_long, 
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Paredes Interiores') & 
                    (self.df_survey_long['ITM'] == 'Rodapies')
                ], 
                ('Mat', 'N')
            )
            df_filtered = self._expand_df_survey_long_by_mat(
                df_expanded_w[
                    (df_expanded_w['GRP'] == 'Paredes Interiores') & 
                    (df_expanded_w['ITM'] == 'Rodapies') & 
                    (df_expanded_w['ATR'] == 'Mat') & 
                    (df_expanded_w['VAL'].notna())
                ],
                self.codes['MAT']
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete mapping arrays for m_SKT")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for m_SKT")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BT': ('col', 'BT'),
                    'DC': 'SKT',
                    'DCV': 'm',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "ff_SKT" in var_to_fit:
            self.prints_if("Defining continuous vertical limit scale for ff_SKT")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(0, 0.1, 0.2)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for ff_SKT")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'SKT',
                    'DCV': 'ff',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # RDR - Doors Array
        # ---------------------------------------------------------------------
        if "p_RDR" in var_to_fit:
            self.prints_if("Preparing continuous pricing profiles for door systems p_RDR")
            rdr_mat = {0: 400, 512: 235, 4096: 860, 1: 120, 4: 600}
            dist_list = []
            for mat_code, mean_val in rdr_mat.items():
                temp_df = self._define_continuous_distribution(
                    function='triang',
                    params=(119.9, mean_val, 1200)
                )
                temp_df['MAT'] = mat_code
                dist_list.append(temp_df)
            dist_df = pd.concat(dist_list, ignore_index=True)
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_RDR")
            self._bulk_update_functions_files(
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

        if "n_RDR" in var_to_fit:
            self.prints_if("Preparing door quantity samples for n_RDR matching building profiles")
            df_filtered = self._sum_df_survey_long_by_su(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Aperturas') & 
                    (self.df_survey_long['ITM'] == 'Regular doors') & 
                    (self.df_survey_long['ATR'] == 'N') & 
                    (self.df_survey_long['VAL'].notna())
                ]
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete hurdle distribution matrix for n_RDR")
            dist_df['BP'] = dist_df['OBS'].apply(self._get_bernoulli_p)
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_discrete_distribution,
                hurdle_fix=True,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=500
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for n_RDR")
            self._bulk_update_functions_files(
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

        if "m_RDR" in var_to_fit:
            self.prints_if("Preparing weighting structural material arrays for m_RDR")
            df_expanded_w = self._expand_df_survey_long_by_weights(
                self.df_survey_long, 
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Aperturas') & 
                    (self.df_survey_long['ITM'] == 'Regular doors')
                ], 
                ('Mat', 'N')
            )
            df_filtered = self._expand_df_survey_long_by_mat(
                df_expanded_w[
                    (df_expanded_w['GRP'] == 'Aperturas') & 
                    (df_expanded_w['ITM'] == 'Regular doors') & 
                    (df_expanded_w['ATR'] == 'Mat') & 
                    (df_expanded_w['VAL'].notna())
                ],
                self.codes['MAT']
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting empirical discrete structure mappings for m_RDR")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for m_RDR")
            self._bulk_update_functions_files(
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

        if "ff_RDR" in var_to_fit:
            self.prints_if("Defining continuous fragility threshold vertical boundary for ff_RDR")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(0.4, 0.6, 0.8)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for ff_RDR")
            self._bulk_update_functions_files(
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

        # ---------------------------------------------------------------------
        # WND - Window Openings
        # ---------------------------------------------------------------------
        if "p_WND" in var_to_fit:
            self.prints_if("Defining continuous replacement price matrix for p_WND")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(250, 350, 600)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_WND")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'WND',
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "n_WND" in var_to_fit:
            self.prints_if("Preparing survey asset counts for window distribution n_WND")
            df_filtered = self._sum_df_survey_long_by_su(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Aperturas') & 
                    (self.df_survey_long['ITM'] == 'Windows') & 
                    (self.df_survey_long['ATR'] == 'N') & 
                    (self.df_survey_long['VAL'].notna())
                ]
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete distribution profiles for n_WND")
            dist_df['BP'] = dist_df['OBS'].apply(self._get_bernoulli_p)
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_discrete_distribution,
                hurdle_fix=False,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=500
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for n_WND")
            self._bulk_update_functions_files(
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

        if "ff_WND" in var_to_fit:
            self.prints_if("Defining continuous structural fragility altitude for ff_WND")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(1.3, 1.5, 1.7)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for ff_WND")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'WND',
                    'DCV': 'ff',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # PLG - Power Plugs & Electricity Outlets
        # ---------------------------------------------------------------------
        if "p_PLG" in var_to_fit:
            self.prints_if("Defining continuous unit costs metrics for outlet replacement p_PLG")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(35, 40, 90)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_PLG")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'PLG',
                    'DCV': 'p',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "n_PLG" in var_to_fit:
            self.prints_if("Preparing quantitative counts for electricity terminals n_PLG")
            df_filtered = self._sum_df_survey_long_by_su(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Electricidad') & 
                    (self.df_survey_long['ITM'] == 'Enchufes') & 
                    (self.df_survey_long['ATR'] == 'N') & 
                    (self.df_survey_long['VAL'].notna())
                ]
            )
            self.prints_if(df_filtered)
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete hurdle distribution equations for n_PLG")
            dist_df['BP'] = dist_df['OBS'].apply(self._get_bernoulli_p)
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_discrete_distribution,
                hurdle_fix=True,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=500
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for n_PLG")
            self._bulk_update_functions_files(
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

        if "ff_PLG" in var_to_fit:
            self.prints_if("Preparing data for electrical outlet installation height ff_PLG")
            df_filtered = self.df_survey_long[
                (self.df_survey_long['GRP'] == 'Electricidad') & 
                (self.df_survey_long['ITM'] == 'Enchufes') & 
                (self.df_survey_long['ATR'] == 'Alt') & 
                (self.df_survey_long['VAL'].notna())
            ]
            df_filtered
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=float
            )
            self.prints_if("Fitting continuous distribution profile lines for ff_PLG")
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_continuous_distribution,
                dist_names_to_fit=self.dist_catalog['C_MI_B'],
                fit_only_KDE=False,
                min_sample_size=5,
                n_mc_samples=self.boorstrap_n,
                alpha=0.05,
                accepted_max=3,
                accepted_min=0,
                avoid_bs_size=300
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for ff_PLG")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'PLG',
                    'DCV': 'ff',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        # ---------------------------------------------------------------------
        # ELS - Main Electrical Infrastructure Network
        # ---------------------------------------------------------------------
        if "p_ELS" in var_to_fit:
            self.prints_if("Defining continuous infrastructure replacement price scaling for p_ELS")
            dist_df = self._define_continuous_distribution(
                function='triang',
                params=(15, 25, 55)
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for p_ELS")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'DC': 'ELS',
                    'DCV': 'p',
                    'DT': 'd',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                },
            )

        if "t_ELS" in var_to_fit:
            self.prints_if("Preparing asset network categorical classification datasets for t_ELS")
            dist_df = self._create_dist_from_df(
                self.df_survey_long[
                    (self.df_survey_long['GRP'] == 'Electricidad') & 
                    (self.df_survey_long['ITM'] == 'Inst. Eléctrica') & 
                    (self.df_survey_long['ATR'] == 'Tipo') & 
                    (self.df_survey_long['VAL'].notna())
                ],
                group_cols=['BT'],
                val_col='VAL',
                add_default_dist=['BT'],
                astype=int
            )
            self.prints_if("Fitting discrete empirical categorization array lines for t_ELS")
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                self._fit_discrete_distribution,
                fit_only_empirical=True,
            )
            self.prints_if(dist_df)
            self.prints_if("Updating framework files for t_ELS")
            self._bulk_update_functions_files(
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

    def prepare_event_features_distributions(self, var_to_fit=[]):
        """
        A.6 Prepare distributions for event features (e.g., water depth 'he').
        Extracts, formats, fits, and updates hydraulic inundation depth continuous parameters 
        conditioned on structural assets (BID) and recurrence thresholds (RP).
        """
        # ---------------------------------------------------------------------
        # he - Inundation Water Depth (Hurdle Continuous Fit)
        # ---------------------------------------------------------------------
        if "he" in var_to_fit:
            self.prints_if("Preparing spatial hydraulic data slices for depth parameter 'he'")
            
            # Isolate structures that display an active inundation signature across the dataset
            bid_max_values = self.df_depth_samples.groupby('BID')['he'].max()
            bids_to_keep = bid_max_values[bid_max_values > 0].index
            df_filtered = self.df_depth_samples[self.df_depth_samples['BID'].isin(bids_to_keep)].copy()
            
            # Compile numerical observation arrays split by structural asset identity and return period
            dist_df = self._create_dist_from_df(
                df_filtered,
                group_cols=['BID', 'RP'],
                val_col='he',
                astype=float
            )
            
            self.prints_if("Fitting hurdle continuous distributions for 'he' via parallelization")
            dist_df['BP'] = dist_df['OBS'].parallel_apply(self._get_bernoulli_p)
            dist_df[['FN', 'FP']] = dist_df['OBS'].parallel_apply(
                self._fit_continuous_distribution,
                dist_names_to_fit=self.dist_catalog['C_BS_p_vB'],
                hurdle_fix=True,
                fit_only_KDE=False,
                min_sample_size=5,
                n_mc_samples=999,
                alpha=0.05,
                accepted_max=20,
                accepted_min=0,
                avoid_bs_size=30
            )
            
            self.prints_if(dist_df)
            self.prints_if("Updating framework indexing registries for 'he'")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BID': ('col', 'BID'),
                    'RP': ('col', 'RP'),
                    'DC': 'he',
                    'DT': 'c',
                    'BP': ('col', 'BP'),
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                }
            )
        
    def prepare_dem_error_distributions(self, var_to_fit=[]):
        """
        A.7 Prepare distributions for DEM error extracting raster stats per building.
        Overlays cadaster geometries against the geospatial terrain error raster to extract 
        zonal structural offset configurations, assigning zero-mean Gaussian uncertainty kernels.
        """
        # ---------------------------------------------------------------------
        # ed - Digital Elevation Model Spatial Altimetric Errors
        # ---------------------------------------------------------------------
        if "ed" in var_to_fit:
            import rasterio
            from rasterio.mask import mask

            self.prints_if("Extracting terrain uncertainty statistics from DEM error raster matrix")
            dem_errors = []
            
            with rasterio.open(self.paths["dem_error_tif"]) as src:
                for geom in self.gdf_buildings.geometry:
                    try:
                        # Extract bounded cell data arrays intersected by the polygon boundary footprint
                        out_image, out_transform = mask(src, [geom], crop=True)
                        data = out_image[0]
                        
                        # Purge unreferenced background nodata cells
                        valid_data = data[data != src.nodata] if src.nodata is not None else data[~np.isnan(data)]
                        
                        # Compute spatial average metric
                        avg_se = np.mean(valid_data) if valid_data.size > 0 else 0
                        dem_errors.append(avg_se)
                    except (ValueError, Exception):
                        dem_errors.append(0)

            dem_error_df = pd.DataFrame({
                'BID': self.gdf_buildings['BID'],
                'SE': dem_errors
            })

            dist_df = self._create_dist_from_df(
                dem_error_df, 
                group_cols=['BID'], 
                val_col='SE',
                astype=float
            )

            self.prints_if("Projecting zero-mean Normal Gaussian distribution parameters for 'ed'")
            # Applies a clean normal profile: mean=0, scale=extracted_structural_standard_error
            dist_df[['FN', 'FP']] = dist_df['OBS'].apply(
                lambda obs_tuple: self._define_continuous_distribution('norm', (0.0, float(obs_tuple[0]))).iloc[0]
            )

            self.prints_if(dist_df)
            self.prints_if("Updating framework indexing registries for 'ed'")
            self._bulk_update_functions_files(
                dist_df,
                {
                    'BID': ('col', 'BID'),
                    'DC': 'ed',
                    'DT': 'c',
                    'FN': ('col', 'FN'),
                    'FP': ('col', 'FP')
                }
            )


if __name__ == "__main__":
    # Centralized imports from config
    from src.config import PATHS, CODES, RETURN_PERIODS, DIST_CATALOG
    
    pandarallel.initialize(progress_bar=True)
    
    # Initialize the fitter class
    fitter = DistributionFitter(PATHS, CODES, RETURN_PERIODS, DIST_CATALOG)
    df_survey_long, df_prices, df_depth_samples, gdf_buildings = fitter.load_input_data()
    fitter.prepare_building_distributions(
        var_to_fit = ["BT", "NF", "BF", "HU", "IH", "GL", "BH", "Cga"])
    fitter.prepare_content_distributions(
        var_to_fit=["n_CTE", "p_CTE", "ff_CTE"]
    )
    fitter.prepare_continent_distributions(
        var_to_fit=[
            "p_PUM", "p_DHU", "p_CLE", 
            "p_SOI", "m_SOI", "ff_SOI", 
            "pr_PRW", "m_PRW", "ff_PRW", "p_PRW", 
            "p_ETP", "m_EXF", "ff_FRI", "n_FRI", 
            "p_SKT", "n_SKT", "m_SKT", "ff_SKT", 
            "p_RDR", "n_RDR", "m_RDR", "ff_RDR", 
            "p_WND", "n_WND", "ff_WND", 
            "p_PLG", "n_PLG", "ff_PLG", 
            "p_ELS", "t_ELS"
        ]
    )
    fitter.prepare_event_features_distributions(
        var_to_fit=["he"]
    )
    fitter.prepare_dem_error_distributions(
        var_to_fit=["ed"]
    )