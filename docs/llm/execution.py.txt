"""
In-Depth-PROFILE economic monte carlo valuation Module

This module handles the dataset creation and the economic monte carlo process,
including the post-treatment of the results.
"""

import os
import gc
import ast
import time
import glob
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

# ==============================================================================
# MODULE LEVEL MULTIPROCESSING WORKERS
# Must remain outside class definitions to ensure pickle compatibility.
# ==============================================================================

def _ensure_numeric_params(p):
    """
    Converts string representations of tuples/lists back to numeric native objects
    while wiping out localized numpy metadata string additions.
    """
    if isinstance(p, str):
        clean_p = (p.replace('np.float64', '')
                    .replace('np.int64', '')
                    .replace('np.float32', ''))
        try:
            p = ast.literal_eval(clean_p)
        except (ValueError, SyntaxError):
            return p

    if isinstance(p, (tuple, list)):
        try:
            return tuple(float(x) for x in p)
        except (TypeError, ValueError):
            return p
            
    try:
        return float(p)
    except (TypeError, ValueError):
        return p

def _sample_batch(fm_row, size, obs_data=None, hi_values=None, default_BP_sample=0, 
                  sample_only_positives=False, q_range=(0.05, 0.95), bp_dep_sample=None, 
                  dep_sample=None, min_limit=None):
    """
    Generates a vectorized batch of samples utilizing either parametric distribution curves,
    KDE envelopes, hurdle models, or empirical discrete frequencies based on standard framework bounds.
    
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
    
    DT = fm_row['DT']
    BP = fm_row['BP']
    FN = fm_row['FN']
    FP = _ensure_numeric_params(fm_row['FP'])
    
    q_min = max(q_range[0], 0.001)
    q_max = min(q_range[1], 0.999)
    actual_min_limit = 0 if sample_only_positives else min_limit
    
    # Hurdle Model Bernoulli Trail Configuration
    if pd.notna(BP):
        if bp_dep_sample is not None:
            hurdle_results = np.full(size, bp_dep_sample <= BP, dtype=bool)
        else:
            hurdle_results = np.random.binomial(1, BP, size).astype(bool)
            
        pass_count = hurdle_results.sum()
        final_samples = np.full(size, default_BP_sample, dtype=float)
        if pass_count == 0:
            return final_samples
    else:
        hurdle_results = np.ones(size, dtype=bool)
        pass_count = size
        final_samples = np.full(size, np.nan)

    # Theoretical curve mapping routes
    if FN == "Constant":
        val = float(FP[0]) if isinstance(FP, (tuple, list, np.ndarray)) and len(FP) > 0 else float(FP)
        batch_samples = np.full(pass_count, val)
        
    elif FN == "KDE":
        kde = stats.gaussian_kde(obs_data)
        lower_q, upper_q = q_min, q_max
        if actual_min_limit is not None:
            p_min = kde.integrate_box_1d(-np.inf, actual_min_limit)
            lower_q = max(q_min, p_min)
            
        if hi_values is not None:
            p_hi = np.array([kde.integrate_box_1d(-np.inf, val) for val in hi_values])
            batch_samples = (np.clip(p_hi, lower_q, upper_q) - lower_q) / (upper_q - lower_q)
        else:
            pool_size = max(10000, pass_count)
            resamples = np.sort(kde.resample(size=pool_size).flatten())
            u = np.clip(dep_sample, lower_q, upper_q) if dep_sample is not None else np.random.uniform(lower_q, upper_q, size=pass_count)
            batch_samples = np.quantile(resamples, u)
            
    elif FN == "Empirical":
        obs_array = np.array(obs_data)
        if actual_min_limit is not None:
            obs_array = obs_array[obs_array >= actual_min_limit]
        if dep_sample is not None:
            idx = np.clip(int(dep_sample * len(obs_array)), 0, len(obs_array) - 1)
            batch_samples = np.full(pass_count, np.sort(obs_array)[idx])
        else:
            batch_samples = np.random.choice(obs_array, size=pass_count, replace=True)
            
    else:
        dist = getattr(stats, FN)
        lower_q, upper_q = q_min, q_max
        params = FP if isinstance(FP, (tuple, list)) else (FP,)
        if actual_min_limit is not None:
            lower_q = max(q_min, dist.cdf(actual_min_limit, *params))
            
        if hi_values is not None:
            batch_samples = (np.clip(dist.cdf(hi_values, *params), lower_q, upper_q) - lower_q) / (upper_q - lower_q)
        else:
            u = np.clip(dep_sample, lower_q, upper_q) if dep_sample is not None else np.random.uniform(lower_q, upper_q, size=pass_count)
            batch_samples = dist.ppf(u, *params)
        
    if pd.notna(BP) and FN in ['poisson', 'nbinom', 'binom']:
        batch_samples = batch_samples + 1
        
    if actual_min_limit is not None:
        batch_samples = np.maximum(batch_samples, actual_min_limit)
        
    if DT == "d":
        batch_samples = np.round(batch_samples).astype(np.float64)
        
    final_samples[hurdle_results] = batch_samples
    return final_samples

def _sample_DC(DC, df_it, fm_df, fo_data, hierarchy=['RP', 'BID', 'BT'], 
               default_BP_sample=0, variant=None, mat=False, sample_only_positives=False, 
               q_range=(0.05, 0.95), bp_dep_sample=None, dep_sample=None, min_limit=None):
    """
    Resolves data matrix code mappings and handles the cascade fallback tracking strategy
    down to the general population metrics to avoid analytical simulation gaps.
    """
    parts = DC.split('_')
    has_suffix = len(parts) > 1 and (parts[-1].isdigit() or (parts[-1].startswith('-') and parts[-1][1:].isdigit()))
    
    if has_suffix:
        suffix = parts[-1]
        DC_base = "_".join(parts[:-1])
        DC_code = parts[1] if len(parts) > 2 else parts[0]
        hi_col = f"hi_{suffix}" if variant == 'hi' else None
        b_col = f"b_{DC_code}_{suffix}" if variant == 'hc' else None
        if variant == 'hc': DC_base = DC_base.replace('hc', 'ff')
        mat_col = f"m_{DC_code}_{suffix}" if mat else None
    else:
        DC_base = DC
        DC_code = parts[1] if len(parts) == 2 else parts[0]
        hi_col = "hi" if variant == 'hi' else None
        b_col = f"b_{DC_code}" if variant == 'hc' else None
        if variant == 'hc': DC_base = DC_base.replace('hc', 'ff')
        mat_col = f"m_{DC_code}" if mat else None

    base_cols = ['DID', 'FID', 'BT', 'BID', 'RP']
    fetch_cols = base_cols + ([DC] if DC not in base_cols else [])
    if variant == 'hi' and hi_col in df_it.columns: fetch_cols.append(hi_col)
    if variant == 'hc' and b_col in df_it.columns: fetch_cols.append(b_col)
    if mat: fetch_cols.append(mat_col)
    
    dts = df_it.loc[df_it[DC].isna(), fetch_cols].copy()
    dts['FID'] = np.nan
    if mat: dts = dts.rename(columns={mat_col: 'MAT'})

    if variant == 'b':
        virtual_fm_row = pd.Series({'DT': 'c', 'BP': np.nan, 'FN': 'uniform', 'FP': (0, 1)})
        df_it.loc[df_it[DC].isna(), DC] = _sample_batch(
            virtual_fm_row, len(dts), default_BP_sample=default_BP_sample, 
            sample_only_positives=sample_only_positives, q_range=q_range, 
            bp_dep_sample=bp_dep_sample, dep_sample=dep_sample, min_limit=min_limit
        )
        return
        
    fm_target = fm_df[(fm_df['DC'] == DC_base)].dropna(subset=['FN'])
    merge_keys = ['RP', 'BID', 'BT'] + (['MAT'] if mat else [])
    dts = dts.drop(columns=['FID']).merge(fm_target[['FID'] + merge_keys], on=merge_keys, how='left')
    
    local_hierarchy = hierarchy + (['MAT'] if mat else [])
    for col in local_hierarchy:
        nan_mask = dts['FID'].isna()
        if not nan_mask.any(): break
        dts.loc[nan_mask, col] = 0
        matched = dts.loc[nan_mask, merge_keys].merge(fm_target[['FID'] + merge_keys], on=merge_keys, how='left')
        dts.loc[nan_mask, 'FID'] = matched['FID'].values
        
    if dts['FID'].isna().any():
        return

    dts['FID'] = dts['FID'].astype(int)
    for fid in dts['FID'].unique():
        group_mask = dts['FID'] == fid
        size = group_mask.sum()
        if size > 0:
            fm_row = fm_target[fm_target['FID'] == fid].iloc[0]
            obs_data = fo_data.get(fm_row['ODC']) if fm_row['FN'] in ["KDE", "Empirical"] else None
            
            if variant == 'hi':
                batch_args = {'hi_values': dts.loc[group_mask, hi_col].values}
            elif variant == 'hc':
                batch_args = {'dep_sample': dts.loc[group_mask, b_col].values}
            else:
                batch_args = {'bp_dep_sample': bp_dep_sample, 'dep_sample': dep_sample}
                
            sampled_values = _sample_batch(
                fm_row, size, obs_data=obs_data, default_BP_sample=default_BP_sample,
                sample_only_positives=sample_only_positives, q_range=q_range, min_limit=min_limit, **batch_args
            )
            dts.loc[group_mask, DC] = sampled_values

    df_it.set_index('DID', inplace=True)
    df_it.update(dts.set_index('DID')[[DC]])
    df_it.reset_index(inplace=True)

def _calc_max_cost(df, cols_n, cols_p, key_name):
    """Calculates ultimate asset exposure baselines."""
    if not cols_n: return pd.Series(0.0, index=df.index)
    if not cols_p and f"p_{key_name}" in df.columns: cols_p = [f"p_{key_name}"]
    if len(cols_n) == len(cols_p):
        return pd.Series((df[cols_n].fillna(0).values * df[cols_p].fillna(0).values).sum(axis=1), index=df.index)
    elif len(cols_p) == 1:
        return df[cols_n].fillna(0).mul(df[cols_p[0]].fillna(0), axis=0).sum(axis=1)
    return pd.Series(0.0, index=df.index)

def _aggregate_batch_metrics(df, cte_keys, cti_keys, basement_floors, above_ground_floors):
    """Executes vectorized in-memory cost combinations across all groups."""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    cost_cols = [c for c in df.columns if c.startswith('c_')]
    df[cost_cols] = df[cost_cols].fillna(0)
    
    b_unq, f_unq = sorted(list(set(basement_floors))), sorted(list(set(above_ground_floors)))
    all_unq = sorted(list(set(b_unq + f_unq)))
    new_cols = {}

    # --- 1. Content Aggregations (CTEs) ---
    for key in cte_keys:
        b_cols_c = [f"c_{key}_{f}" for f in b_unq if f"c_{key}_{f}" in df.columns]
        b_cols_n = [f"n_{key}_{f}" for f in b_unq if f"n_{key}_{f}" in df.columns]
        b_cols_p = [f"p_{key}_{f}" for f in b_unq if f"p_{key}_{f}" in df.columns]
        
        f_range = [0] if key == 'VEH' else f_unq
        f_cols_c = [f"c_{key}_{f}" for f in f_range if f"c_{key}_{f}" in df.columns]
        f_cols_n = [f"n_{key}_{f}" for f in f_range if f"n_{key}_{f}" in df.columns]
        f_cols_p = [f"p_{key}_{f}" for f in f_range if f"p_{key}_{f}" in df.columns]
        
        bf_cols_c = [f"c_{key}_{f}" for f in all_unq if f"c_{key}_{f}" in df.columns]
        bf_cols_n = [f"n_{key}_{f}" for f in all_unq if f"n_{key}_{f}" in df.columns]
        bf_cols_p = [f"p_{key}_{f}" for f in all_unq if f"p_{key}_{f}" in df.columns]
        
        c_0, n_0, p_0 = f"c_{key}_0", f"n_{key}_0", f"p_{key}_0"
        
        sum_b = df[b_cols_c].sum(axis=1) if b_cols_c else pd.Series(0.0, index=df.index)
        sum_bm = _calc_max_cost(df, b_cols_n, b_cols_p, key)
        sum_f = df[f_cols_c].sum(axis=1) if f_cols_c else pd.Series(0.0, index=df.index)
        sum_fm = _calc_max_cost(df, f_cols_n, f_cols_p, key)
        sum_bf = df[bf_cols_c].sum(axis=1) if bf_cols_c else pd.Series(0.0, index=df.index)
        sum_bfm = _calc_max_cost(df, bf_cols_n, bf_cols_p, key)
        sum_0 = df[c_0] if c_0 in df.columns else pd.Series(0.0, index=df.index)
        sum_0m = _calc_max_cost(df, [n_0], [p_0], key) if n_0 in df.columns else pd.Series(0.0, index=df.index)

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

        new_cols[f"c_hu{key}_bpct"] = new_cols[f"c_hu{key}_b"].div(new_cols[f"c_hu{key}_bm"]).fillna(0).clip(0, 1)
        new_cols[f"c_hu{key}_fpct"] = new_cols[f"c_hu{key}_f"].div(new_cols[f"c_hu{key}_fm"]).fillna(0).clip(0, 1)
        new_cols[f"c_hu{key}_bfpct"] = new_cols[f"c_hu{key}_bf"].div(new_cols[f"c_hu{key}_bfm"]).fillna(0).clip(0, 1)
        new_cols[f"c_hu{key}_0pct"] = new_cols[f"c_hu{key}_0"].div(new_cols[f"c_hu{key}_0m"]).fillna(0).clip(0, 1)

        new_cols[f"c_e{key}_b"] = new_cols[f"c_hu{key}_b"] / df['RP']
        new_cols[f"c_e{key}_f"] = new_cols[f"c_hu{key}_f"] / df['RP']
        new_cols[f"c_e{key}_bf"] = new_cols[f"c_hu{key}_bf"] / df['RP']
        new_cols[f"c_e{key}_0"] = new_cols[f"c_hu{key}_0"] / df['RP']

    # --- 2. Continent Structural Aggregations (CTIs) ---
    count_cols = [c for c in df.columns if c.startswith('n_')]
    price_cols = [c for c in df.columns if c.startswith('p_')]
    geom_cols = [c for c in ['BP', 'BH', 'BA', 'GP', 'IH', 'GA', 'EP', 'GL', 'NF', 'e_PUM', 'pr_PRW'] if c in df.columns]
    
    for c in (cost_cols + count_cols + price_cols + geom_cols):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='ignore').replace([-999, -999.0], 0.0).fillna(0.0)

    for key in cti_keys:
        b_cols_c = [f"c_{key}_{f}" for f in b_unq if f"c_{key}_{f}" in df.columns]
        bm_cols_n = [f"n_{key}_{f}" for f in b_unq if f"n_{key}_{f}" in df.columns]
        bm_cols_p = [f"p_{key}_{f}" for f in b_unq if f"p_{key}_{f}" in df.columns]
        if not bm_cols_p and f"p_{key}" in df.columns:
            bm_cols_p = [f"p_{key}"]
        
        f_cols_c = [f"c_{key}_{f}" for f in f_unq if f"c_{key}_{f}" in df.columns]
        if not f_cols_c and f"c_{key}" in df.columns: f_cols_c = [f"c_{key}"]
        fm_cols_n = [f"n_{key}_{f}" for f in f_unq if f"n_{key}_{f}" in df.columns] or ([f"n_{key}"] if f"n_{key}" in df.columns else [])
        fm_cols_p = [f"p_{key}"] if f"p_{key}" in df.columns else []
        
        bf_cols_c = [f"c_{key}_{f}" for f in all_unq if f"c_{key}_{f}" in df.columns]
        n_0 = f"n_{key}_0" if f"n_{key}_0" in df.columns else (f"n_{key}" if f"n_{key}" in df.columns else None)
        p_0 = f"p_{key}_0" if f"p_{key}_0" in df.columns else (f"p_{key}" if f"p_{key}" in df.columns else None)

        sum_b = df[b_cols_c].sum(axis=1) if b_cols_c else pd.Series(0.0, index=df.index)
        sum_bm = pd.Series(0.0, index=df.index)
        if key == 'CLE': sum_bm = ((df['BP'] * df['BH']) + df['BA']) * df['p_CLE']
        elif key == 'SOI': sum_bm = df['BA'] * df['p_SOI']
        elif key == 'PRW': sum_bm = df['BP'] * df['BH'] * (df['pr_PRW'] + df['p_PRW'])
        elif key == 'ITP': sum_bm = df['BP'] * df['BH'] * df['p_ETP']
        elif key == 'SKT' and bm_cols_n and bm_cols_p: sum_bm = df['BP'] * (df[bm_cols_n].clip(0, 20).sum(axis=1) / 20.0) * df[bm_cols_p[0]]
        elif key not in ['ETP', 'ELS', 'WND'] and bm_cols_n and bm_cols_p: sum_bm = df[bm_cols_n].mul(df[bm_cols_p[0]], axis=0).sum(axis=1)

        sum_f = df[f_cols_c].sum(axis=1) if f_cols_c else pd.Series(0.0, index=df.index)
        sum_fm = pd.Series(0.0, index=df.index)
        if key == 'CLE': sum_fm = ((df['GP'] * df['IH']) + df['GA']) * df['p_CLE'] * df['NF']
        elif key == 'SOI': sum_fm = df['GA'] * df['p_SOI'] * df['NF']
        elif key == 'PRW': sum_fm = df['GP'] * df['IH'] * (df['pr_PRW'] + df['p_PRW']) * df['NF']
        elif key == 'ITP': sum_fm = df['GP'] * df['IH'] * df['p_ETP'] * df['NF']
        elif key == 'ETP': sum_fm = df['EP'] * (df['NF'] * df['IH']) * df['p_ETP']
        elif key == 'PUM': sum_fm = df['e_PUM'] * df['p_PUM']
        elif key == 'SKT' and fm_cols_n and fm_cols_p: sum_fm = df['GP'] * (df[fm_cols_n].clip(0, 20).sum(axis=1) / 20.0) * df[fm_cols_p[0]]

        sum_bf = df[bf_cols_c].sum(axis=1) if bf_cols_c else pd.Series(0.0, index=df.index)
        sum_bfm = sum_bm + sum_fm

        sum_0 = df[f"c_{key}_0"] if f"c_{key}_0" in df.columns else (df[f"c_{key}"] if f"c_{key}" in df.columns else pd.Series(0.0, index=df.index))
        sum_0m = pd.Series(0.0, index=df.index)
        if key == 'CLE': sum_0m = ((df['GP'] * df['IH']) + df['GA']) * df['p_CLE']
        elif key == 'SOI': sum_0m = df['GA'] * df['p_SOI']
        elif key == 'PRW': sum_0m = df['GP'] * df['IH'] * (df['pr_PRW'] + df['p_PRW'])
        elif key == 'ITP': sum_0m = df['GP'] * df['IH'] * df['p_ETP']
        elif key == 'ETP': sum_0m = df['EP'] * (df['NF'] * df['IH']) * df['p_ETP']
        elif key == 'PUM': sum_0m = df['e_PUM'] * df['p_PUM']
        elif key == 'SKT' and n_0 and p_0: sum_0m = df['GP'] * (df[n_0].clip(0, 20) / 20.0) * df[p_0]
        elif n_0 and p_0: sum_0m = df[n_0] * df[p_0]

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
        
        new_cols[f"c_hu{key}_bpct"] = new_cols[f"c_hu{key}_b"].div(new_cols[f"c_hu{key}_bm"]).fillna(0).clip(0, 1)
        new_cols[f"c_hu{key}_fpct"] = new_cols[f"c_hu{key}_f"].div(new_cols[f"c_hu{key}_fm"]).fillna(0).clip(0, 1)
        new_cols[f"c_hu{key}_bfpct"] = new_cols[f"c_hu{key}_bf"].div(new_cols[f"c_hu{key}_bfm"]).fillna(0).clip(0, 1)
        new_cols[f"c_hu{key}_0pct"] = new_cols[f"c_hu{key}_0"].div(new_cols[f"c_hu{key}_0m"]).fillna(0).clip(0, 1)
        
        new_cols[f"c_e{key}_b"] = new_cols[f"c_hu{key}_b"] / df['RP']
        new_cols[f"c_e{key}_f"] = new_cols[f"c_hu{key}_f"] / df['RP']
        new_cols[f"c_e{key}_bf"] = new_cols[f"c_hu{key}_bf"] / df['RP']
        new_cols[f"c_e{key}_0"] = new_cols[f"c_hu{key}_0"] / df['RP']

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)

    # --- 3. Global Framework Summaries ---
    for prefix, keys in [('CTEs', cte_keys), ('CTIs', cti_keys)]:
        df[f'c_{prefix}_b'] = df[[f"c_{k}_b" for k in keys]].sum(axis=1)
        df[f'c_{prefix}_f'] = df[[f"c_{k}_f" for k in keys]].sum(axis=1)
        df[f'c_{prefix}_bf'] = df[[f"c_{k}_bf" for k in keys]].sum(axis=1)
        df[f'c_{prefix}_0'] = df[[f"c_{k}_0" for k in keys]].sum(axis=1)
        
        df[f'c_hu{prefix}_b'] = df[[f"c_hu{k}_b" for k in keys]].sum(axis=1)
        df[f'c_hu{prefix}_f'] = df[[f"c_hu{k}_f" for k in keys]].sum(axis=1)
        df[f'c_hu{prefix}_bf'] = df[[f"c_hu{k}_bf" for k in keys]].sum(axis=1)
        df[f'c_hu{prefix}_0'] = df[[f"c_hu{k}_0" for k in keys]].sum(axis=1)
        df[f'c_hu{prefix}_bm'] = df[[f"c_hu{k}_bm" for k in keys]].sum(axis=1)
        df[f'c_hu{prefix}_fm'] = df[[f"c_hu{k}_fm" for k in keys]].sum(axis=1)
        df[f'c_hu{prefix}_bfm'] = df[[f"c_hu{k}_bfm" for k in keys]].sum(axis=1)
        df[f'c_hu{prefix}_0m'] = df[[f"c_hu{k}_0m" for k in keys]].sum(axis=1)

        df[f'c_hu{prefix}_bpct'] = df[f'c_hu{prefix}_b'].div(df[f'c_hu{prefix}_bm']).fillna(0).clip(0, 1)
        df[f'c_hu{prefix}_fpct'] = df[f'c_hu{prefix}_f'].div(df[f'c_hu{prefix}_fm']).fillna(0).clip(0, 1)
        df[f'c_hu{prefix}_bfpct'] = df[f'c_hu{prefix}_bf'].div(df[f'c_hu{prefix}_bfm']).fillna(0).clip(0, 1)
        df[f'c_hu{prefix}_0pct'] = df[f'c_hu{prefix}_0'].div(df[f'c_hu{prefix}_0m']).fillna(0).clip(0, 1)

        df[f'c_e{prefix}_b'] = df[[f"c_e{k}_b" for k in keys]].sum(axis=1)
        df[f'c_e{prefix}_f'] = df[[f"c_e{k}_f" for k in keys]].sum(axis=1)
        df[f'c_e{prefix}_bf'] = df[[f"c_e{k}_bf" for k in keys]].sum(axis=1)
        df[f'c_e{prefix}_0'] = df[[f"c_e{k}_0" for k in keys]].sum(axis=1)

    # --- 4. Final Total Building Aggregations ---
    df['c_b'], df['c_f'] = df['c_CTEs_b'] + df['c_CTIs_b'], df['c_CTEs_f'] + df['c_CTIs_f']
    df['c_bf'], df['c_0'] = df['c_CTEs_bf'] + df['c_CTIs_bf'], df['c_CTEs_0'] + df['c_CTIs_0']
    
    df['c_hu_b'], df['c_hu_f'] = df['c_huCTEs_b'] + df['c_huCTIs_b'], df['c_huCTEs_f'] + df['c_huCTIs_f']
    df['c_hu_bf'], df['c_hu_0'] = df['c_huCTEs_bf'] + df['c_huCTIs_bf'], df['c_huCTEs_0'] + df['c_huCTIs_0']
    
    df['c_hu_bm'], df['c_hu_fm'] = df['c_huCTEs_bm'] + df['c_huCTIs_bm'], df['c_huCTEs_fm'] + df['c_huCTIs_fm']
    df['c_hu_bfm'], df['c_hu_0m'] = df['c_huCTEs_bfm'] + df['c_huCTIs_bfm'], df['c_huCTEs_0m'] + df['c_huCTIs_0m']

    df['c_hu_bpct'] = df['c_hu_b'].div(df['c_hu_bm']).fillna(0).clip(0, 1)
    df['c_hu_fpct'] = df['c_hu_f'].div(df['c_hu_fm']).fillna(0).clip(0, 1)
    df['c_hu_bfpct'] = df['c_hu_bf'].div(df['c_hu_bfm']).fillna(0).clip(0, 1)
    df['c_hu_0pct'] = df['c_hu_0'].div(df['c_hu_0m']).fillna(0).clip(0, 1)

    df['c_e_b'], df['c_e_f'] = df['c_eCTEs_b'] + df['c_eCTIs_b'], df['c_eCTEs_f'] + df['c_eCTIs_f']
    df['c_e_bf'], df['c_e_0'] = df['c_eCTEs_bf'] + df['c_eCTIs_bf'], df['c_eCTEs_0'] + df['c_eCTIs_0']

    return df

def it_Monte_Carlo(sn, dataset, fm_df, fo_data, col_name_dic):
    """
    Core vectorized Standard Monte Carlo (SMC) simulation execution worker. Performs
    stochastic field parsing, geometric translations, and structural asset risk evaluations.
    """
    try:
        df_it = dataset.copy()
        df_it['SN'] = sn
        df_it = df_it[['SN'] + [c for c in df_it.columns if c != 'SN']]
        max_bf = int(df_it['BF'].max()) if df_it['BF'].notna().any() else 0
        max_nf = int(df_it['NF'].max()) if df_it['NF'].notna().any() else 0
        floor_range = range(-max_bf, max_nf)
        
        # --- Building Structural Field Sampling ---
        for DC in ['BT', 'NF', 'BF', 'HU', 'IH', 'GL', 'BH', 'Cga']:
            if DC == "BF":
                df_it.loc[df_it['BT'].isin([1, 3, 5]), 'BF'] = 0
            elif DC == "HU":
                df_it.loc[df_it['BT'].isin([1, 2]), 'HU'] = 1
            elif DC == "BH":
                df_it.loc[df_it['BT'].isin([1, 3, 5]), 'BH'] = -999
            
            limit_kwargs = {'min_limit': 4.0} if DC == "Cga" else {}
            _sample_DC(DC, df_it, fm_df, fo_data, q_range=(0, 1), **limit_kwargs)
        
        # Geometrical translation derivations
        df_it['BA'] = np.where(df_it['BT'].isin([1, 3, 5]), -999, np.where(df_it['BF'] > 0, df_it['GA'], 0))
        df_it['SA'] = (df_it['GA'] * df_it['NF']) + (np.maximum(df_it['BA'], 0) * np.maximum(df_it['BF'], 0))
        df_it['BL'] = np.where(df_it['BT'].isin([1, 2, 3]), -999, df_it['GL'] - df_it['BH'])
        df_it['GP'] = df_it['Cga'] * np.sqrt(df_it['GA'])
        
        # Deactivate dimensions across physically impossible fields
        for _, columns in col_name_dic.items():
            for col in columns:
                parts = col.split('_')
                try:
                    n = int(parts[-1])
                    invalid_mask = (df_it['BT'].isin([1, 3, 5])) if n < 0 else (n >= df_it['NF'])
                    df_it.loc[invalid_mask, col] = -999
                except (ValueError, IndexError):
                    continue
        
        # --- DEM and Environmental Hazard Parsing ---
        _sample_DC('ed', df_it, fm_df, fo_data, hierarchy=['RP', 'BT', 'BID'], q_range=(0, 1))
        
        he_bp = np.random.uniform(0, 1)
        he_dep = np.random.uniform(0, 0.95)
        _sample_DC('he', df_it, fm_df, fo_data, hierarchy=['BT', 'BID', 'RP'], 
                   sample_only_positives=True, bp_dep_sample=he_bp, dep_sample=he_dep)
        
        # Inside flood depth routing logic calculations
        water_levels = np.where(df_it['he'] > 0, np.maximum(df_it['he'] - df_it['ed'], 0.0), 0.0)
        for n in floor_range:
            col = f'hi_{n}'
            if n < 0:
                is_valid = (np.abs(n) <= df_it['BF'])
                floor_base_level = df_it['BL'] + (np.abs(n) - 1) * df_it['BH']
                floor_limit = df_it['BH']
            else:
                is_valid = (n < df_it['NF'])
                floor_base_level = df_it['GL'] + n * df_it['IH']
                floor_limit = df_it['IH']
                
            hi_calc = np.where(water_levels > 0, np.clip(water_levels - floor_base_level, 0, floor_limit), 0.0)
            df_it[col] = np.where(is_valid, hi_calc, -999)

        # --- Content Matrix Hazard Processing ---
        content_map = {
            'n_content': {'h': ['BID', 'RP', 'BT'], 'v': None, 'q': (0, 0.95)},
            'p_content': {'h': ['BID', 'RP', 'BT'], 'v': None, 'q': (0, 0.95)},
            'ff_content': {'h': ['BID', 'RP', 'BT'], 'v': 'hi', 'q': (0, 1)},
            'b_content': {'h': None, 'v': 'b', 'q': (0, 1)},
            'hc_content': {'h': ['BID', 'RP', 'BT'], 'v': 'hc', 'q': (0, 1)}
        }
        for key, config in content_map.items():
            if key in col_name_dic:
                for col_name in col_name_dic[key]:
                    _sample_DC(col_name, df_it, fm_df, fo_data, hierarchy=config['h'], 
                               sample_only_positives=True, variant=config['v'], q_range=config['q'])
                    
        for n_col, p_col, ff_col, b_col, d_col, c_col in zip(
            col_name_dic['n_content'], col_name_dic['p_content'], col_name_dic['ff_content'],
            col_name_dic['b_content'], col_name_dic['d_content'], col_name_dic['c_content']
        ):
            valid_mask = (df_it[ff_col] != -999)
            df_it.loc[valid_mask, d_col] = (df_it.loc[valid_mask, b_col] <= df_it.loc[valid_mask, ff_col]).astype(int)
            df_it.loc[~valid_mask, d_col] = -999
            df_it.loc[valid_mask, c_col] = df_it.loc[valid_mask, d_col] * df_it.loc[valid_mask, n_col] * df_it.loc[valid_mask, p_col]
            df_it.loc[~valid_mask, c_col] = -999

        # --- Envelope Continent Asset Valuation ---
        for DC in ['p_PUM', 'p_CLE']:
            _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0.05, 0.95))
            
        df_it['e_PUM'] = np.where((df_it['GL'] != -999) & (df_it['GL'] < 0), df_it['GA'] * (-df_it['GL']), 0.0) + \
                         np.where((df_it['BL'] != -999) & (df_it['BL'] < 0), df_it['BA'] * (-df_it['BL']), 0.0)
        df_it.loc[(df_it['GL'] == -999) | (df_it['GL'].isna()), 'e_PUM'] = -999
        df_it['c_PUM'] = np.where((df_it['hi_0'] > 0), df_it['e_PUM'] * df_it['p_PUM'], 0.0)

        for n in floor_range:
            hi_col, e_col, c_col = f'hi_{n}', f'e_CLE_{n}', f'c_CLE_{n}'
            p, a = (df_it['BP'], df_it['BA']) if n < 0 else (df_it['GP'], df_it['GA'])
            exists = (p != -999) & (a != -999) & (df_it[hi_col] != -999)
            df_it.loc[exists & (df_it[hi_col] > 0), e_col] = (p * df_it[hi_col]) + a
            df_it.loc[exists & (df_it[hi_col] == 0), e_col] = 0.0
            df_it.loc[~exists, e_col] = -999
            df_it.loc[exists, c_col] = df_it[e_col] * df_it['p_CLE']
            df_it.loc[~exists, c_col] = -999

        # --- Soil Layer Intervention Calculations ---
        for DC in ['m_SOI', 'p_SOI', 'ff_SOI', 'b_SOI', 'hc_SOI']:
            if DC == 'm_SOI': _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0, 1))
            elif DC == 'p_SOI': _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0.05, 0.95), mat=True)
            else:
                for n in floor_range:
                    v_type = 'hi' if DC == 'ff_SOI' else ('b' if DC == 'b_SOI' else 'hc')
                    _sample_DC(f"{DC}_{n}", df_it, fm_df, fo_data, hierarchy=['BID', 'RP', 'BT'], sample_only_positives=True, variant=v_type, q_range=(0, 1))

        for n in floor_range:
            ff, b, d, e, c = f"ff_SOI_{n}", f"b_SOI_{n}", f"d_SOI_{n}", f"e_SOI_{n}", f"c_SOI_{n}"
            v_mask = (df_it[ff] != -999) & (df_it[b] != -999)
            df_it.loc[v_mask, d] = (df_it.loc[v_mask, b] <= df_it.loc[v_mask, ff]).astype(int)
            df_it.loc[~v_mask, d] = -999
            area_src = 'BA' if n < 0 else 'GA'
            df_it.loc[df_it[d] == 1, e] = df_it[area_src]
            df_it.loc[df_it[d] == 0, e] = 0.0
            df_it.loc[v_mask, c] = df_it['p_SOI'] * df_it[e]
            df_it.loc[~v_mask, [e, c]] = -999

        # --- Fixtures & Component Group Sampling ---
        for DC in ['n_SKT', 'n_RDR', 'n_WND', 'n_PLG', 'm_SKT', 'm_RDR', 'p_SKT', 'p_RDR', 'p_WND', 'p_PLG', 'ff_SKT', 'ff_RDR', 'ff_WND', 'ff_PLG', 'b_SKT', 'b_RDR', 'b_WND', 'b_PLG', 'hc_SKT', 'hc_RDR', 'hc_WND', 'hc_PLG']:
            if DC in ['m_SKT', 'm_RDR']: _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0, 1))
            elif DC in ['p_SKT', 'p_RDR']: _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0.05, 0.95), mat=True)
            elif DC in ['p_WND', 'p_PLG']: _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0.05, 0.95))
            else:
                for n in floor_range:
                    if f"{DC}_{n}" == f"{DC}_WND_-1": continue
                    v_type = 'hi' if 'ff_' in DC else ('b' if 'b_' in DC else ('hc' if 'hc_' in DC else None))
                    _sample_DC(f"{DC}_{n}", df_it, fm_df, fo_data, hierarchy=['BID', 'RP', 'BT'], sample_only_positives=True, variant=v_type, q_range=(0, 1))

        for n in floor_range:
            for code in ['SKT', 'RDR', 'WND', 'PLG']:
                ff, b, n_c, d, e, c, p = f"ff_{code}_{n}", f"b_{code}_{n}", f"n_{code}_{n}", f"d_{code}_{n}", f"e_{code}_{n}", f"c_{code}_{n}", f"p_{code}"
                if ff not in df_it.columns: continue
                v_mask = (df_it[ff] != -999) & (df_it[b] != -999)
                df_it.loc[v_mask, d] = (df_it.loc[v_mask, b] <= df_it.loc[v_mask, ff]).astype(float)
                df_it.loc[~v_mask, d] = -999
                
                if code == 'SKT':
                    df_it.loc[df_it[d] == 1, e] = df_it['BP' if n < 0 else 'GP'] * (df_it[n_c].clip(0, 20) / 20.0)
                    df_it.loc[df_it[d] == 0, e] = 0.0
                    df_it.loc[v_mask, c] = df_it[e] * df_it[p]
                else:
                    df_it.loc[df_it[d] == 1, c] = df_it[n_c] * df_it[p]
                    df_it.loc[df_it[d] == 0, c] = 0.0
                df_it.loc[~v_mask, [e, c] if code == 'SKT' else [c]] = -999

        # --- Partition Walls Structural Operations ---
        for DC in ['m_PRW', 'm_EXF', 'pr_PRW', 'p_PRW', 'p_ETP', 'ff_PRW', 'b_PRW', 'hc_PRW']:
            if DC in ['m_PRW', 'm_EXF']: _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0, 1))
            elif DC in ['pr_PRW', 'p_PRW']: _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0.05, 0.95), mat=True)
            elif DC == 'p_ETP': _sample_DC(DC, df_it, fm_df, fo_data, sample_only_positives=True, q_range=(0.05, 0.95))
            else:
                for n in floor_range:
                    v_type = 'hi' if DC == 'ff_PRW' else ('b' if DC == 'b_PRW' else 'hc')
                    _sample_DC(f"{DC}_{n}", df_it, fm_df, fo_data, hierarchy=['BID', 'RP', 'BT'], sample_only_positives=True, variant=v_type, q_range=(0, 1))

        for n in floor_range:
            ff, b, d, e, c, c_itp = f"ff_PRW_{n}", f"b_PRW_{n}", f"d_PRW_{n}", f"e_PRW_{n}", f"c_PRW_{n}", f"c_ITP_{n}"
            v_mask = (df_it[ff] != -999) & (df_it[b] != -999) & (df_it['IH'] != -999) & (df_it['pr_PRW'] != -999)
            df_it.loc[v_mask, d] = (df_it.loc[v_mask, b] <= df_it.loc[v_mask, ff]).astype(float)
            df_it.loc[~v_mask, d] = -999
            df_it.loc[df_it[d] == 1, e] = df_it['BP' if n < 0 else 'GP'] * df_it['BH' if n < 0 else 'IH']
            df_it.loc[df_it[d] == 0, e] = 0.0
            df_it.loc[df_it[d] == 1, c] = (df_it['pr_PRW'] * df_it[e]) + (df_it['p_PRW'] * df_it[e])
            df_it.loc[df_it[d] == 1, c_itp] = df_it['p_ETP'] * df_it[e]
            df_it.loc[df_it[d] == 0, [c, c_itp]] = 0.0
            df_it.loc[~v_mask, [e, c, c_itp]] = -999

        # External Façade and Structural Paint Layers calculations
        v_mask = (df_it['EP'] != -999) & (df_it['he'] != -999) & (df_it['p_ETP'] != -999) & (df_it['GL'] != -999)
        b_roof = df_it['GL'] + (df_it['NF'] * df_it['IH'])
        f_height = np.maximum(np.minimum(df_it['he'], b_roof) - df_it['GL'], 0.0)
        df_it['e_EXF'] = np.where(v_mask, df_it['EP'] * f_height, -999)
        df_it['c_ETP'] = np.where(v_mask, df_it['p_ETP'] * df_it['e_EXF'], -999)

        return df_it

    except RuntimeError as e:
        if "updating stopped, endless loop" in str(e):
            return _generate_failed_fallback(dataset, sn)
        raise e

def _generate_failed_fallback(dataset, sn):
    """Generates an insulated fallback dataframe row matrix if runtime anomalies occur."""
    df_failed = dataset.copy()
    df_failed['SN'] = sn
    cols_to_fill = [c for c in df_failed.columns if c != 'SN']
    df_failed.loc[:, cols_to_fill] = -999
    return df_failed

class LossModelExecutionEngine:
    """
    Orchestrates the Economic Monte Carlo Simulation framework and data post-processing 
    pipelines for the In-Depth-PROFILE inundation loss estimation engine.
    """
    def __init__(self, config_paths, config_codes, config_return_periods):
        """
        Initializes the execution engine with centralized project tracking constants.
        """
        self.paths = config_paths
        self.codes = config_codes
        self.return_periods = config_return_periods
        
        # Core data structures placeholders
        self.dataset = None
        self.col_name_dic = {}

    def create_base_dataset(self, gdf_buildings):
        """
        Initializes the master evaluation matrix by binding geospatial cadastre attributes,
        structural footprint indices, and multi-floor building variants.
        
        Establishes vectorized placeholder grids using framework layout rules:
          - np.nan: Reserved for unobserved parameters sampled during Monte Carlo loops.
          - -999: Reserved for structurally impossible attributes (e.g., higher storeys).
        
        Parameters:
        -----------
        gdf_buildings : geopandas.GeoDataFrame
            The cleaned spatial cadastre layers containing structural profiles.
        Returns:
        --------
        tuple (pd.DataFrame, dict)
            The structured base database matrix and its grouped field identifier catalog.
        """
        self.col_name_dic = {}
        
        # --- B.1.1 Structural Base and Tracking Keys ---
        RPs_unq = list(self.return_periods.keys())
        df_depth_samples = pd.read_pickle(self.paths['depth_samples_pkl'])
        BIDs_flooded_unq_byRP = {
            rp: df_depth_samples[(df_depth_samples['RP'] == rp) & (df_depth_samples['he'] > 0)]['BID'].unique().tolist()
            for rp in RPs_unq
        }
        base_records = []
        for rp, bids in BIDs_flooded_unq_byRP.items():
            base_records.append(pd.DataFrame({'BID': bids, 'RP': rp}))
            
        dataset = pd.concat(base_records, ignore_index=True).copy()
        
        # Map Dwelling Categories (BT) and replace null footprints
        building_bt_map = gdf_buildings.set_index('BID')['BT']
        dataset['BT'] = dataset['BID'].map(building_bt_map)
        dataset.loc[dataset['BT'] == 0, 'BT'] = np.nan
        
        # Assign primary matrix tracking identifiers
        dataset['DID'] = range(len(dataset))
        dataset['FID'] = np.nan
        
        codes_order = ['DID', 'FID', 'BT', 'BID', 'RP']
        dataset = dataset[codes_order]
        self.col_name_dic["codes"] = codes_order
        
        # --- B.1.2 Structure ---
        building_nf_map = gdf_buildings.set_index('BID')['NF']
        building_bf_map = gdf_buildings.set_index('BID')['BF']
        building_hu_map = gdf_buildings.set_index('BID')['HU']
        
        dataset['NF'] = dataset['BID'].map(building_nf_map)
        dataset.loc[dataset['BT'].isna(), 'NF'] = np.nan
        max_nf = int(dataset['NF'].max()) if dataset['NF'].notna().any() else 0

        dataset['BF'] = dataset['BID'].map(building_bf_map)
        dataset.loc[dataset['BT'].isna(), 'BF'] = np.nan
        max_bf = int(dataset['BF'].max()) if dataset['BF'].notna().any() else 0

        dataset['HU'] = dataset['BID'].map(building_hu_map)
        dataset.loc[dataset['BT'].isna(), 'HU'] = np.nan
        dataset.loc[dataset['BT'].isin([5, 6]), 'HU'] = np.nan

        dataset['IH'] = np.nan
        dataset['BH'] = np.nan
        dataset['GL'] = np.nan
        dataset['BL'] = np.nan

        structure_order = ['HU', 'NF', 'BF', 'IH', 'BH', 'GL', 'BL']
        dataset = dataset[codes_order + structure_order]
        self.col_name_dic["structure"] = structure_order

        # --- B.1.3 Geometry and Volumetric Area Elements ---
        building_ga_map = gdf_buildings.set_index('BID')['GA']
        dataset['GA'] = dataset['BID'].map(building_ga_map)

        # Baseline basement area conditional extraction
        dataset['BA'] = np.where(dataset['BF'] > 0, dataset['GA'], 0)
        dataset.loc[dataset['GA'].isna(), 'BA'] = np.nan
        dataset.loc[dataset['BT'].isna(), 'BA'] = np.nan

        # Calculate absolute exposed surface area
        dataset['SA'] = (dataset['GA'] * dataset['NF']) + (dataset['BA'] * dataset['BF'])
        
        areas_order = ['GA', 'BA', 'SA']
        self.col_name_dic["areas"] = areas_order
        
        # --- B.1.4 Linear Perimeter Metrics ---
        building_ep_map = gdf_buildings.set_index('BID')['EP']
        dataset['EP'] = dataset['BID'].map(building_ep_map)
        dataset['GP'] = np.nan
        dataset['Cga'] = np.nan
        dataset['BP'] = 4 * np.sqrt(dataset['BA'].astype(float))
        
        perimeters_order = ['EP', 'GP', 'Cga', 'BP']
        self.col_name_dic["perimeters"] = perimeters_order
        
        # --- B.1.5 Elevation Models Error Layers ---
        dataset['ed'] = np.nan
        self.col_name_dic["dem"] = ['ed']
        
        # --- B.1.6 Hydraulic Flood Hazards Variant Fields ---
        dataset['he'] = np.nan
        event_order = ['he']
        
        nf_is_nan = dataset['NF'].isna()
        bf_is_nan = dataset['BF'].isna()
        
        for n in range(-max_bf, max_nf):
            col_name = f'hi_{n}'
            event_order.append(col_name)
            if n < 0:
                is_valid_floor = (abs(n) <= dataset['BF'])
                is_unknown = bf_is_nan
            else:
                is_valid_floor = (n < dataset['NF'])
                is_unknown = nf_is_nan
                
            dataset[col_name] = np.where(
                is_unknown, 
                np.nan, 
                np.where(is_valid_floor, np.nan, -999)
            )
        self.col_name_dic["event"] = event_order
        
        # --- B.1.7 Content Exposure Items ---
        new_content_dict = {}
        content_prefixes = ["n", "p", "ff", "b", "hc", "d", "c"]
        content_keys = list(self.codes['Content'].keys())
        
        for pf in content_prefixes:
            pf_group_cols = []
            for code in content_keys:
                for n in range(-max_bf, max_nf):
                    # Enforce vehicle restrictions to ground floor and basement boundaries
                    if code == "VEH" and n not in [-1, 0]:
                        continue
                        
                    col_name = f"{pf}_{code}_{n}"
                    pf_group_cols.append(col_name)
                    
                    if n < 0:
                        floor_exists = (abs(n) <= dataset['BF'])
                        unknown_floor = bf_is_nan
                    else:
                        floor_exists = (n < dataset['NF'])
                        unknown_floor = nf_is_nan
                    
                    new_content_dict[col_name] = np.where(
                        unknown_floor, 
                        np.nan, 
                        np.where(floor_exists, np.nan, -999)
                    )
            self.col_name_dic[f"{pf}_content"] = pf_group_cols

        new_cols_df = pd.DataFrame(new_content_dict, index=dataset.index)
        dataset = pd.concat([dataset, new_cols_df], axis=1)
        
        # --- B.1.8 Envelope and Structural Continent Components ---
        new_continent_dict = {}
        continent_config = {
            'PUM': {'general': ['p', 'e', 'c'], 'by_floor': []},
            'CLE': {'general': ['p'], 'by_floor': ['e', 'c']},
            'SOI': {'general': ['m', 'p'], 'by_floor': ['e', 'ff', 'b', "hc", 'd', 'c']},
            'SKT': {'general': ['m', 'p'], 'by_floor': ['e', 'n', 'ff', 'b', "hc", 'd', 'c']},
            'RDR': {'general': ['m', 'p'], 'by_floor': ['n', 'ff', 'b', "hc", 'd', 'c']},
            'WND': {'general': ['p'], 'by_floor': ['n', 'ff', 'b', "hc", 'd', 'c']},
            'PLG': {'general': ['p'], 'by_floor': ['n', 'ff', 'b', "hc", 'd', 'c']},
            'PRW': {'general': ['m', 'pr', 'p'], 'by_floor': ['e', 'ff', 'b', "hc", 'd', 'c']},
            'EXF': {'general': ['m', 'e'], 'by_floor': []},
            'ETP': {'general': ['p', 'c'], 'by_floor': []},
            'ITP': {'general': [], 'by_floor': ['c']},
            'FRI': {'general': [], 'by_floor': ['n', 'ff']}
        }
        
        continent_prefixes = ["n", 'e', 'm', 't', "p", 'pr', "ff", "b", "hc", "d", "c"]
        
        for pf in continent_prefixes:
            pf_group_cols = []
            for continent, cfg in continent_config.items():
                if pf in cfg['general']:
                    col_name = f"{pf}_{continent}"
                    new_continent_dict[col_name] = np.nan
                    pf_group_cols.append(col_name)
                    
                if pf in cfg['by_floor']:
                    for n in range(-max_bf, max_nf):
                        # Skip window/friso components for subterranean basements
                        if continent in ['WND', 'FRI'] and n < 0:
                            continue
                            
                        col_name = f"{pf}_{continent}_{n}"
                        pf_group_cols.append(col_name)
                        
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
            self.col_name_dic[f"{pf}_continent"] = pf_group_cols
            
        continent_df = pd.DataFrame(new_continent_dict, index=dataset.index)
        self.dataset = pd.concat([dataset, continent_df], axis=1).copy()
        
        return self.dataset, self.col_name_dic

    def run_economic_monte_carlo(self, n_simulations=10000, cores=16, batch_size=128):
        """Generates Monte Carlo slices and computes economic aggregations natively in-memory."""
        fm_df = pd.read_pickle(self.paths['fm_pkl'])
        fm_df['FID'] = range(len(fm_df))
        fm_df = fm_df[['FID'] + [c for c in fm_df.columns if c != 'FID']]
        
        with open(self.paths['fo_pkl'], 'rb') as f: fo_data = pickle.load(f)
        output_dir = self.paths['dataset_mc_parts']
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        # Dynamic structural bounds detection from evaluation footprint mapping matrices
        max_bf = int(self.dataset['BF'].max()) if self.dataset['BF'].notna().any() else 0
        max_nf = int(self.dataset['NF'].max()) if self.dataset['NF'].notna().any() else 0
        b_floors, f_floors = range(-max_bf, 0), range(0, max_nf + 1)

        cte_keys = list(self.codes['Content'].keys())
        cti_keys = ['PUM', 'CLE', 'SOI', 'SKT', 'RDR', 'WND', 'PLG', 'PRW', 'EXF', 'ETP', 'ITP', 'FRI']

        all_sns = list(range(1, n_simulations + 1))
        batches = [all_sns[i:i + batch_size] for i in range(0, len(all_sns), batch_size)]
        
        print(f"[ENGINE INITIALIZATION] Processing {n_simulations} iterations directly to merged targets...")
        
        for idx, batch_sns in enumerate(batches):
            batch_path = os.path.join(output_dir, f"MC_Part_{idx + 1}.parquet")
            if os.path.exists(batch_path): continue
            
            sn_series = pd.Series(batch_sns)
            batch_results = sn_series.parallel_apply(
                it_Monte_Carlo, dataset=self.dataset, fm_df=fm_df, fo_data=fo_data, col_name_dic=self.col_name_dic
            )
            
            # Combine individual simulations in memory
            batch_df = pd.concat(batch_results.values, ignore_index=True)
            batch_df = batch_df[['SN'] + [c for c in batch_df.columns if c != 'SN']].replace(-999, np.nan)
            
            # Integrated in-memory postprocessing calculations step (no extra read/write)
            batch_df = _aggregate_batch_metrics(
                df=batch_df, cte_keys=cte_keys, cti_keys=cti_keys, 
                basement_floors=b_floors, above_ground_floors=f_floors
            )
            
            # Downcast directly to optimized float32 and save completed matrix segment
            batch_df = batch_df.astype({c: 'float32' for c in batch_df.select_dtypes(include=['float64']).columns})
            batch_df.to_parquet(batch_path, engine='pyarrow', compression='snappy', index=False)
            
            del batch_df, batch_results
            gc.collect()

    def compile_sensitivity_dataset(self, cte_keys, cti_keys):
        """
        Dynamically extracts, filters, and assembles the surrogate modeling dataset 
        for XGBoost using low-overhead PyArrow disk streaming to protect RAM.
        """
        import glob
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_path = self.paths['xgb_dataset']
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

        # 1. Discover all simulation batch segments
        file_list = sorted(glob.glob(os.path.join(self.paths['dataset_mc_parts'], "MC_Part_*.parquet")))
        if not file_list:
            print("[SENSITIVITY ERROR] No generated Monte Carlo segments found.")
            return

        print(f"[SENSITIVITY DATASET] Scanning {len(file_list)} parts to build surrogate matrix...")

        # 2. Establish uniform structural features tracking keys
        x_cols_a = ['SN', 'BID', 'RP', 'BT', 'ed', 'he', 'IH', 'BH', 'GL', 'Cga']
        
        # Read schema of the first file to extract available stochastic columns dynamically
        first_schema = pq.read_schema(file_list[0])
        all_columns = first_schema.names
        
        x_cols_b = [c for c in all_columns if c.startswith(('n_', 'p_', 'hc_', 'm_'))]
        target_columns = x_cols_a + x_cols_b + ['c_bf']

        # 3. Stream chunks sequentially to disk via PyArrow Writer Channel
        writer = None
        try:
            for file_path in file_list:
                # Read only requested features to keep RAM footprint minimal
                table = pq.read_table(file_path, columns=target_columns)
                
                # Open streaming channel on first valid table block slice
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                
                writer.write_table(table)
                del table
                
            print(f"[SUCCESS] Sensitivity dataset compiled cleanly to disk -> {output_path}")
            
        finally:
            if writer is not None:
                writer.close()

if __name__ == "__main__":
    # Centralized imports from config
    from src.config import PATHS, CODES, RETURN_PERIODS
    
    # 1. Initialize the engine class
    engine = LossModelExecutionEngine(PATHS, CODES, RETURN_PERIODS)
    
    # 2. Build evaluation matrix structures dynamically inside the engine
    print("Building master base dataset evaluation matrix arrays...")
    dataset, col_name_dic = engine.create_base_dataset(gdf_buildings=gpd.read_file(PATHS['buildings_shp']))
    
    # 3. Execute economic Monte Carlo simulation engine and runtime post-processors
    print("Executing economic monte carlo simulation engine...")
    engine.run_economic_monte_carlo(
        n_simulations=10000, 
        cores=16, 
        batch_size=128
    )
