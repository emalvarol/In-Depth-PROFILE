"""
In-Depth-PROFILE Plotting and Results Visualization Module
This module encapsulates all helper analytical parsing functions, data preparation
routines, and vectorized visualization components to render figures, maps, and tables
for flood loss estimation and global sensitivity analyses.
"""

import os
import glob
import re
import math
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import matplotlib.lines as lines
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm

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

class DataPreparator:
    """
    Handles data aggregation, mathematical convergence checking, and formatting 
    for plotting matrices and output tables.
    """
    def __init__(self, config_paths, config_codes, config_return_periods):
        self.paths = config_paths
        self.codes = config_codes
        self.return_periods = config_return_periods

    @staticmethod
    def load_mc_part(part_path):
        return pd.read_parquet(part_path)
    
    @staticmethod
    def load_and_concatenate_mc_parts(results_dir, target_cols, base_file_name, bids_to_exclude=None):
        """
        Loads specific columns from parquet files to minimize memory footprint.
        Only SN, BID, RP, and columns starting with 'c_' are retained.
        """
        if bids_to_exclude is None:
            bids_to_exclude = []
            
        file_list = sorted(glob.glob(os.path.join(results_dir, f"{base_file_name}_*.parquet")))
        
        if not file_list:
            return pd.DataFrame()

        parts = []
        exclude_set = set(bids_to_exclude)
        
        for file in tqdm(file_list, desc="Merging MC Parts"):
            temp_df = pd.read_parquet(file, columns=target_cols)
            if exclude_set:
                temp_df = temp_df[~temp_df['BID'].isin(exclude_set)]
            parts.append(temp_df)
            
        df = pd.concat(parts, ignore_index=True)
        
        del temp_df
        del parts
        
        return df
    
    @staticmethod
    def load_flooded_buildings_gdf(building_path, depth_samples_path):
        gdf_buildings = gpd.read_file(building_path)
        df_depth_samples = pd.read_pickle(depth_samples_path)
        BIDs_flooded = df_depth_samples.groupby('BID')['he'].transform('max') > 0
        df_depth_samples = df_depth_samples[BIDs_flooded].reset_index(drop=True)
        BIDs_flooded_unq = df_depth_samples['BID'].unique()
        gdf_buildings = gdf_buildings[gdf_buildings['BID'].isin(BIDs_flooded_unq)].copy()
        return gdf_buildings
    
    @staticmethod
    def calc_hr_convergence(self, df, rp_key, threshold=0.1, z_score=1.96):
        """
        Calculates convergence metrics for a specific Return Period.
        Criteria: SN > 30, IRME == 0, and 30 consecutive stable steps.
        """
        rp_df = df[df['RP'] == rp_key].copy()
        
        if rp_df.empty:
            return False, rp_df

        rp_df = rp_df.sort_values(by=['BID', 'SN'])

        rp_df['he_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().mean())
        rp_df['SD_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().std(ddof=1))
        rp_df['SE_BID_sn'] = rp_df['SD_BID_sn'] / np.sqrt(rp_df['SN'])
        rp_df['ME_BID_sn'] = z_score * rp_df['SE_BID_sn']
        rp_df['Residual'] = (rp_df['ME_BID_sn'] - threshold).clip(lower=0)
        
        rp_df['MET_BID'] = (rp_df['SN'] > 30) & (rp_df['Residual'] == 0)
        
        blocks = (~rp_df['MET_BID']).groupby(rp_df['BID']).cumsum()
        rp_df['STC_BID'] = rp_df.groupby(['BID', blocks]).cumcount() * rp_df['MET_BID']
        rp_df['CONV_BID'] = rp_df['STC_BID'] >= 30
        
        rp_simp = rp_df.groupby('SN').agg(
            IRME=('Residual', 'sum'),
            NON_CONV_COUNT=('Residual', lambda x: (x > 0).sum())
        ).reset_index()

        total_flooded_bids = rp_df[rp_df['he'] > 0]['BID'].nunique()
        if total_flooded_bids > 0:
            rp_simp['REL_NON_CONV'] = (rp_simp['NON_CONV_COUNT'] / total_flooded_bids) * 100
        else:
            rp_simp['REL_NON_CONV'] = 0.0

        rp_simp['MET'] = (rp_simp['SN'] > 30) & (rp_simp['IRME'] == 0)

        blocks = (~rp_simp['MET']).cumsum()
        rp_simp['STC_AG'] = rp_simp.groupby(blocks).cumcount() * rp_simp['MET']
        rp_simp['CONV'] = rp_simp['STC_AG'] >= 30

        if not rp_simp.empty:
            is_converged = rp_simp['CONV'].iloc[-1]
            return bool(is_converged), rp_simp, rp_df
        
        return False, rp_simp

    @staticmethod
    def calc_vats_convergence(self, df, n_sims, n_start=500, calc_step=500, n_step_check=3, 
                              alphas=[50, 10, 5, 1], epsilons=[25, 20, 10, 5, 1], cores=4):
        """
        Extended Vats et al. (2019) multivariate stopping rule for multiple buildings.
        """
        import multiprocess
        
        RPs_vats = np.sort(df['RP'].unique())
        cost_cols = [c for c in df.columns if c.startswith('c_')]
        expected_sns_full = np.arange(1, n_sims + 1)
        
        final_results = []
        detailed_results = []
        execution_logs = []
        
        process_func = partial(_process_bid_worker, alphas=alphas, epsilons=epsilons)
        
        for rp in RPs_vats:
            df_rp = df[df["RP"] == rp]
            BIDs_vats_rp = np.sort(df_rp['BID'].unique())
            total_rp_bids = len(BIDs_vats_rp)
            consecutive_convergence = {bid: 0 for bid in BIDs_vats_rp}
            active_bids = set(BIDs_vats_rp)
            
            grouped = df_rp.groupby("BID")
            bid_matrices = {
                bid: (
                    group.set_index("SN")
                    .reindex(expected_sns_full, fill_value=0.0)[cost_cols]
                    .values.astype(np.float64)
                )
                for bid, group in grouped if bid in BIDs_vats_rp
            }
            
            with multiprocess.Pool(processes=cores) as pool:
                steps = list(range(n_start, n_sims + 1, calc_step))
                if not steps or steps[-1] < n_sims: 
                    steps.append(n_sims)
                    
                for n in steps:
                    if not active_bids:
                        break
                    
                    args_list = [(bid_matrices[bid][:n], bid, n) for bid in active_bids]
                    outputs = pool.map(process_func, args_list)

                    step_metrics = {}
                    new_active_bids = []
                    
                    for out in outputs:
                        if out["logs"]:
                            execution_logs.extend(out["logs"])
                            
                        if out["status"] in ("skipped", "error"):
                            new_active_bids.append(out["bid"])
                            continue

                        detailed_row = {
                            "RP": rp, "SN": n, "BID": out["bid"], "status": out["status"]
                        }
                        detailed_row.update(out["residuals"])
                        detailed_row.update(out["active_counts"])
                        detailed_results.append(detailed_row)
                        
                        for k, v in out["residuals"].items():
                            step_metrics[k] = step_metrics.get(k, 0.0) + v
                        for k, v in out["active_counts"].items():
                            step_metrics[k] = step_metrics.get(k, 0) + v
                        
                        if out["status"] == "converged":
                            consecutive_convergence[out["bid"]] += 1
                        else:
                            consecutive_convergence[out["bid"]] = 0
                            
                        if consecutive_convergence[out["bid"]] < n_step_check:
                            new_active_bids.append(out["bid"])

                    row = {
                        "RP": rp, "SN": n, "total_rp_bids": total_rp_bids,
                        "any_active_bids": len(new_active_bids) 
                    }
                    row.update(step_metrics)
                    final_results.append(row)
                    active_bids = set(new_active_bids)
                    
        return pd.DataFrame(final_results), pd.DataFrame(detailed_results), execution_logs
    
    @staticmethod
    def calc_result_expected(df, target_col='c_bf', min_rp_AEP_0=None):
        import numpy as np
        import pandas as pd
        
        df_q50 = df.groupby(['BID', 'RP'])[target_col].median().reset_index()
        df_q50.rename(columns={target_col: 'Q50'}, inplace=True)
        
        if min_rp_AEP_0 is not None:
            unique_bids = df_q50[['BID']].drop_duplicates()
            unique_bids['RP'] = float(min_rp_AEP_0)
            unique_bids['Q50'] = 0.0
            df_q50 = pd.concat([df_q50, unique_bids], ignore_index=True)
            
        df_q50['AEP'] = 1.0 / df_q50['RP']
        
        df_pivot = df_q50.pivot(index='BID', columns='AEP', values='Q50').fillna(0.0)
        df_pivot = df_pivot.sort_index(axis=1)
        
        aep_axes = df_pivot.columns.values
        expected_values = np.trapezoid(df_pivot.values, x=aep_axes, axis=1)
        
        df_expected = pd.DataFrame({
            f'Expected_{target_col}': expected_values
        }, index=df_pivot.index).reset_index()
        
        return df_expected
    
    @staticmethod
    def calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=0, by_bid=False, min_rp_AEP_0=None):
        """
        Aggregates SHAP values by a specified multi-index level,
        calculates absolute quantiles per return period, and integrates over the AEP curve
        to return Expected Annual SHAP values.
        """
        import os
        import gc
        import numpy as np
        import pandas as pd

        bid_name = 'bid' if by_bid else 'all'
        level_name = f"l{level}"
        file_name = f"df_expected_{level_name}_{bid_name}.feather"
        
        # Uses class instance path instead of function argument
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
    
    @staticmethod
    def calc_shap_rp_evolution_grouped(output_gsa_dir, rps_list, level=0, by_bid=False):
        """Extracts and formats intermediate variance trends across distinct return period horizons."""
        bid_name = 'bid' if by_bid else 'all'
        output_path = os.path.join(output_gsa_dir, f"df_evolution_l{level}_{bid_name}.feather")
        groupby_cols = ['Feature_Group', 'BID'] if by_bid else ['Feature_Group']
        
        if os.path.exists(output_path):
            return pd.read_feather(output_path).set_index(groupby_cols)
            
        all_rp_data = []
        for rp in rps_list:
            shap_path = os.path.join(output_gsa_dir, f"shap_results_rp_{rp}.feather")
            if not os.path.exists(shap_path): continue
            
            df_shap = pd.read_feather(shap_path)
            meta_cols = [c for c in df_shap.columns if 'BID' in c or 'base_value' in c]
            feature_cols = [c for c in df_shap.columns if c not in meta_cols]
            col_mapping = [str(c).split('|')[int(level)] for c in feature_cols]
            
            grouped_features = df_shap[feature_cols].T.groupby(by=col_mapping).sum().T
            if by_bid:
                bid_col = next(c for c in meta_cols if 'BID' in c)
                grouped_features['BID'] = df_shap[bid_col]
                q50 = grouped_features.set_index('BID').abs().groupby('BID').quantile(0.50).stack(future_stack=True).reset_index()
                q50.columns = ['BID', 'Feature_Group', 'Q50']
                total = q50.groupby('BID')['Q50'].transform('sum')
                q50['Share_%'] = np.where(total != 0, (q50['Q50'] / total) * 100, 0.0)
            else:
                q50 = grouped_features.abs().quantile(0.50).reset_index()
                q50.columns = ['Feature_Group', 'Q50']
                total = q50['Q50'].sum()
                q50['Share_%'] = (q50['Q50'] / total) * 100 if total != 0 else 0.0
                
            q50['RP'] = rp
            all_rp_data.append(q50)
            
        if not all_rp_data: return pd.DataFrame()
        df_pivot = pd.concat(all_rp_data, ignore_index=True).pivot(index=groupby_cols, columns='RP', values=['Q50', 'Share_%']).fillna(0.0)
        df_pivot.columns = [f"{m}_RP{rp}" for m, rp in df_pivot.columns]
        df_evolution = df_pivot.sort_index()
        df_evolution.reset_index().to_feather(output_path)
        return df_evolution

    @staticmethod
    def generate_scaled_multilayer_sankey(df_list, relations_list, CODES, colors_dict=None, scale_params=None):
        """Constructs unified structural flow charts linking structural subsets together securely."""
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
            v_pct = scale_params['canvas_pct']
            fwd = lambda x: np.where(
                x <= raw_split, 
                (v_pct / raw_split) * x, 
                v_pct + ((1.0 - v_pct) / (global_max_vol - raw_split)) * (x - raw_split)
            )
        else:
            fwd = lambda x: x / global_max_vol

        layer_node_tracking = []
        
        # --- PHASE 1: GENERATE NODES ---                
        for i in range(num_layers):
            df_clean = df_list[i].dropna(subset=['Q50_share_%']).copy()
            sorted_features = df_clean.sort_values(by='Q50_share_%', ascending=True).index.tolist()
            layer_map = {}
            cum_vol = 0.0
                
            for feat in sorted_features:
                val = df_clean.loc[feat, 'Q50_share_%']
                v0 = float(fwd(cum_vol))
                v1 = float(fwd(cum_vol + val))
                clean_name = str(feat).split(' (')[0].split('_')[0]
                                
                if clean_name in colors_dict:
                    node_color = colors_dict[clean_name]
                elif clean_name in CODES.get("Content", []):
                    node_color = colors_dict.get("Content", "#cccccc")
                elif clean_name in CODES.get("Continent", []):
                    node_color = colors_dict.get("Continent", "#cccccc")
                else:
                    node_color = "#cccccc"
                        
                nodes.append({
                    "name": f"{feat} (L{i+1})", 
                    "x": x_coords[i], 
                    "y": v0, 
                    "h": v1 - v0, 
                    "color": node_color,
                    "edgecolor": 'white',
                    "linewidth": 0.5,
                })
                
                layer_map[feat] = {
                    'id': node_counter, 
                    'val': val, 
                    'out_curr_raw': cum_vol, 
                    'in_curr_raw': cum_vol
                }
                node_counter += 1
                cum_vol += val
                
            layer_node_tracking.append(layer_map)

        # --- PHASE 2: ROUTE LINKS ---
        for i in range(num_layers - 1):
            left_map = layer_node_tracking[i]
            right_map = layer_node_tracking[i+1]
            relations = relations_list[i]
            
            split_srcs = set(relations.keys())
            split_tgts = set([t for ts in relations.values() for t in ts])
            
            rev_rels = {}
            for s, ts in relations.items():
                for t in ts: 
                    rev_rels.setdefault(t, []).append(s)
                
            # 1. Explicit Relations (Splits 1-to-N and Merges N-to-1)
            for src_feat, target_list in relations.items():
                if src_feat not in left_map: continue
                s_info = left_map[src_feat]
                t_total = sum([right_map[t]['val'] for t in target_list if t in right_map])
                
                for tgt_feat in target_list:
                    if tgt_feat not in right_map: continue
                    t_info = right_map[tgt_feat]
                    src_total = sum([left_map[s]['val'] for s in rev_rels[tgt_feat] if s in left_map])
                    
                    s_link = s_info['val'] * (t_info['val'] / t_total if t_total > 0 else 0)
                    t_link = t_info['val'] * (s_info['val'] / src_total if src_total > 0 else 0)
                    
                    s0 = float(fwd(s_info['out_curr_raw']))
                    s1 = float(fwd(s_info['out_curr_raw'] + s_link))
                    t0 = float(fwd(t_info['in_curr_raw']))
                    t1 = float(fwd(t_info['in_curr_raw'] + t_link))
                    
                    links.append({
                        "source": s_info['id'], 
                        "target": t_info['id'], 
                        "value_src": s1 - s0, 
                        "value_tgt": t1 - t0, 
                        "src_y": s0, 
                        "tgt_y": t0, 
                        "color": nodes[s_info['id']]['color']
                    })
                    s_info['out_curr_raw'] += s_link
                    t_info['in_curr_raw'] += t_link

            # 2. Standard 1-to-1 Connections
            for feat in left_map.keys():
                if feat in split_srcs or feat not in right_map or feat in split_tgts: continue
                
                s_info = left_map[feat]
                t_info = right_map[feat]
                
                s0 = float(fwd(s_info['out_curr_raw']))
                s1 = float(fwd(s_info['out_curr_raw'] + s_info['val']))
                t0 = float(fwd(t_info['in_curr_raw']))
                t1 = float(fwd(t_info['in_curr_raw'] + t_info['val']))
                
                links.append({
                    "source": s_info['id'], 
                    "target": t_info['id'], 
                    "value_src": s1 - s0, 
                    "value_tgt": t1 - t0, 
                    "src_y": s0, 
                    "tgt_y": t0, 
                    "color": nodes[s_info['id']]['color']
                })
                s_info['out_curr_raw'] += s_info['val']
                t_info['in_curr_raw'] += t_info['val']
                
        return nodes, links

    @staticmethod
    def simplify_flood_contour(self, shp):
        """Dissolves interiors and eliminates micro-scale polygon slivers from inundation layer sheets."""
        shp = shp.explode(index_parts=False)
        shp.geometry = shp.geometry.apply(lambda poly: Polygon(poly.exterior) if poly.interiors else poly)
        shp = shp[shp.geometry.area > 1e-7]
        union_geom = shp.union_all()
        simplified = union_geom.simplify(tolerance=0.0001, preserve_topology=True)
        return gpd.GeoDataFrame(geometry=[simplified], crs="EPSG:4326")

class Plotter:
    """
    Handles Matplotlib rendering, axes scales, styling configurations, 
    and the centralized vectorized plot_any engine.
    """
    def __init__(self, config_paths, config_codes, config_return_periods):
        """Initializes the visualizer with framework metadata and single-source truth catalogs."""
        self.paths = config_paths
        self.codes = config_codes
        self.return_periods = config_return_periods
        
        # Single Source of Truth Style Configuration
        self.global_pre_style = {
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
        
        # Return Period Mapping
        rps = list(config_return_periods.keys())
        cmap_rp = plt.get_cmap('viridis', len(rps))
        self.rp_colors = {rp: cmap_rp(i) for i, rp in enumerate(rps)}
        self.rp_indices = range(len(rps))
        
        # Floodplain Contours Mapping
        self.q_colors = {
            'sto_Q05_contour': "#0084FF",
            'sto_Q50_contour': "#2F00FF",
            'det_Q50_contour': "#16B302",
            'sto_Q95_contour': "#6F00FF",
        }
        
        # Sensitivity Analysis Core Mapping Tracker
        self.gsa_group_colors = {
            'he': "#ff0000",
            'Structure': "#ff7300",
            'GL': "#ca5b01ff",
            'BH': "#a86936",
            'IH': "#632d01",
            'Cga': "#fc913a",
            'C.High': "#00ff0d",
            'Prices': "#00f7ff",
            'Objects': "#4c00ff",
            'ed': "#ae00ff",
            'Materials': "#ff0095",
            'Content': "#9293ce",
            'Continent': "#c992ce",
        }
        
        # Maps
        self.dst_crs = 'EPSG:4326'
        self.gdf_buildings = gpd.read_file(config_paths['buildings_shp'])
        
    def plot_any(self, layout_params, dict_to_plot, save_params, global_pre_style=None, global_style=None):
        """
        Populates multi-panel mosaic or standard grid structures dynamically via configured dictionaries.
        Recovers the modular structure and comprehensive plot/style/legend options from the original implementation.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        
        # 0. Helper for parameter filtering
        def get_kwargs(data_dict, exclude_keys):
            internal_keys = {'plot_type', 'style_type', 'legend_type', 'gstype'}
            to_exclude = set(exclude_keys) | internal_keys
            return {k: v for k, v in data_dict.items() if k not in to_exclude}
        
        def _check_required_keys(params, required, ptype, ax_key):
            if not all(k in params for k in required):
                raise KeyError(f"Axis '{ax_key}': '{ptype}' requires {required} keys.")
        
        # 1. Global pre-Style
        def set_global_pre_style(style_config):
            defaults = self.global_pre_style.copy()
            if style_config:
                defaults.update(style_config)
            plt.rcParams.update(defaults)
        print(f"\n[Plotter.plot_any] Setting global pre style...")
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
        print(f"\n[Plotter.plot_any] Setting layout...")
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
                    if p.get('reverse', False):
                        p['y'] = p['y'][::-1]
                        p['width'] = p['width'][::-1]
                        if 'color' in p and isinstance(p['color'], (list, np.ndarray, tuple)):
                            p['color'] = p['color'][::-1]
                        if 'edgecolor' in p and isinstance(p['edgecolor'], (list, np.ndarray, tuple)):
                            p['edgecolor'] = p['edgecolor'][::-1]
                    ax.barh(y=p['y'], width=p['width'], **get_kwargs(p, ['y', 'width', 'reverse']))
                
                elif ptype == 'barv':
                    _check_required_keys(p, ['x', 'height'], 'barv', key)
                    if p.get('reverse', False):
                        p['x'] = p['x'][::-1]
                        p['height'] = p['height'][::-1]
                        if 'color' in p and isinstance(p['color'], (list, np.ndarray, tuple)):
                            p['color'] = p['color'][::-1]
                        if 'edgecolor' in p and isinstance(p['edgecolor'], (list, np.ndarray, tuple)):
                            p['edgecolor'] = p['edgecolor'][::-1]
                    ax.bar(x=p['x'], height=p['height'], **get_kwargs(p, ['x', 'height', 'reverse']))
                    
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
                        srs=p.get('crs', 'EPSG:4326'), 
                        bbox=bbox,
                        size=p.get('size', (1000, 1000)), 
                        format=p.get('format', 'image/png'),
                        transparent=p.get('transparent', True)
                    )
                    img_data = mpimg.imread(BytesIO(img_request.read()))
                    ax.set_facecolor('black')
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
                        # FIX: Correctly extract the individual dictionary node using its integer ID
                        src = nodes_list[link['source']]
                        tgt = nodes_list[link['target']]
                        
                        x0, y0 = src['x'], link.get('src_y', src['y'])
                        x1, y1 = tgt['x'], link.get('tgt_y', tgt['y'])
                        
                        w_src = link.get('value_src', link.get('value', 0))
                        w_tgt = link.get('value_tgt', link.get('value', 0))
                        
                        cx0 = x0 + (x1 - x0) / 2
                        cx1 = x1 - (x1 - x0) / 2
                        
                        verts = [
                            (x0, y0),
                            (cx0, y0), (cx1, y1), (x1, y1),
                            (x1, y1 + w_tgt),
                            (cx1, y1 + w_tgt), (cx0, y0 + w_src), (x0, y0 + w_src),
                            (x0, y0)
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
                            edgecolor=node.get('edgecolor', p.get('node_edgecolor', 'black')),
                            linewidth=node.get('linewidth', p.get('node_linewidth', 0.5)),
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
                
                elif stype == 'minor_locator':
                    import matplotlib.ticker as ticker
                    axis_name = s.get('axis', 'both')
                    loc_type = s.get('locator_type', 'log')
                    
                    def apply_axis_locator(target_axis_obj):
                        if loc_type == 'log':
                            base_val = s.get('base', 10.0)
                            subs_val = s.get('subs', range(1, 10))
                            target_axis_obj.set_minor_locator(ticker.LogLocator(base=base_val, subs=subs_val))
                        elif loc_type == 'null':
                            target_axis_obj.set_minor_locator(ticker.NullLocator())

                    if axis_name in ['x', 'both']: apply_axis_locator(ax.xaxis)
                    if axis_name in ['y', 'both']: apply_axis_locator(ax.yaxis)
                
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
                    required = ['x', 'y', 'w', 'h', 'source_ax']
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
                
                else:
                    print(f"Warning: legend_type '{ltype}' in axis '{key}' still not implemented in plot_any.")
                
        for key, config in dict_to_plot.items():
            if key in ax_dict:
                print(f"\n[Plotter.plot_any] Applying plots to axis '{key}'...")
                if 'plots' in config: apply_plots(ax_dict[key], config.get('plots', []), key)
                print(f"\n[Plotter.plot_any] Applying style to axis '{key}'...")
                if 'style' in config: apply_style(ax_dict[key], config.get('style', []), key)
                print(f"\n[Plotter.plot_any] Applying legend to axis '{key}'...")
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
        print(f"\n[Plotter.plot_any] Setting global style...")
        apply_global_figure_style(fig, global_style)
        
        # 5. Save Plot
        print(f"\n[Plotter.plot_any] Saving plot...")
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
            print(f"\n[Plotter.plot_any.save_plot] Plot saved at {path}")
            if params.get('show', True):
                plt.show()
            else:
                plt.close()
        
        save_plot(fig, save_params)

        return fig, ax_dict
    
    def init_dic(self, dict_to_plot, ax, key):
        """Initializes empty metadata sub-containers safely within target configurations."""
        if ax not in dict_to_plot:
            dict_to_plot[ax] = {}
        dict_to_plot[ax][key] = []

    def get_ax_ratio(self, layout_params, ax):
        """Calculates the physical window aspect bounding-box ratio of a designated layout canvas."""
        fig, ax_dict = plt.subplot_mosaic(layout_params['mosaic_structure'], figsize=layout_params['figsize'], **layout_params.get('kwargs', {}))
        fig.subplots_adjust(**layout_params.get('adjust', {}))
        plt.draw() 
        bbox = ax_dict[ax].get_window_extent()
        real_ratio = bbox.width / bbox.height
        plt.close(fig)
        return real_ratio

    def calc_map_extent(self, top_left_x, top_left_y, zoom_pct, real_ratio, base_span=0.02):
        """Transforms degree minute coordinates into geographic coordinate system map bounds."""
        xmin = -(top_left_x[0] + top_left_x[1]/60 + top_left_x[2]/3600)
        ymax = (top_left_y[0] + top_left_y[1]/60 + top_left_y[2]/3600)
        width = base_span * (zoom_pct / 100)
        height = width / real_ratio 
        return [xmin, xmin + width, ymax - height, ymax]

    def get_dx_for_scalebar(self, extent, crs='EPSG:4326'):
        """Calculates physical conversion factors per data coordinate unit for map scalebars."""
        if crs == 'EPSG:4326':
            mid_lat = (extent[2] + extent[3]) / 2
            return (math.pi / 180.0) * 6378137.0 * math.cos(math.radians(mid_lat))
        return 1.0

    def create_2_linear_ax_scale(self, max_val, split, pct):
        """Builds forwarding/reversing structural mappings for segmented axis stretching."""
        fwd = lambda x: np.where(x <= split, (pct / split) * x, pct + ((1.0 - pct) / (max_val - split)) * (x - split))
        rev = lambda y: np.where(y <= pct, (split / pct) * y, split + ((max_val - split) / (1.0 - pct)) * (y - pct))
        return (fwd, rev)

    def create_3_linear_ax_scale(self, max_val, split1, split2, pct1, pct2):
        """Constructs three-segment piecewise scales to zoom axis sub-segments."""
        def forward_scale(x):
            x = np.asarray(x, dtype=float)
            conds = [x <= split1, (x > split1) & (x <= split2), x > split2]
            funcs = [(pct1 / split1) * x, pct1 + ((pct2 - pct1) / (split2 - split1)) * (x - split1), pct2 + ((1.0 - pct2) / (max_val - split2)) * (x - split2)]
            return np.select(conds, funcs, default=x)
        def inverse_scale(y):
            y = np.asarray(y, dtype=float)
            conds = [y <= pct1, (y > pct1) & (y <= pct2), y > pct2]
            funcs = [(split1 / pct1) * y, split1 + ((split2 - split1) / (pct2 - pct1)) * (y - pct1), split2 + ((max_val - split2) / (1.0 - pct2)) * (y - pct2)]
            return np.select(conds, funcs, default=y)
        return (forward_scale, inverse_scale)

def _process_bid_worker(args, alphas, epsilons):
    """Module-level un-nested standalone worker function for Vats criterion execution mapping."""
    import numpy as np
    from math import pi, gamma
    from scipy.stats import f

    subset, bid, n = args
    alphas_processed = [a/100 for a in alphas]
    epsilons_processed = [e/100 for e in epsilons]
    
    col_var = np.var(subset, axis=0)
    active = col_var > 0
    p_eff = int(active.sum())
    
    if p_eff < 2 or n <= p_eff:
        return {"bid": bid, "status": "skipped", "logs": [f"Warning: n={n} <= p_eff={p_eff}"]}
        
    subset = subset[:, active].astype(np.float64)
    cov = np.cov(subset, rowvar=False, ddof=1)
    det = np.linalg.det(cov)
    
    if det <= 0:
        return {"bid": bid, "status": "skipped", "logs": [f"Warning: Non-positive det={det:.2e}"]}
        
    unit_ball_vol = (pi ** (p_eff / 2)) / gamma(p_eff / 2 + 1)
    metrics = {}
    
    for a in alphas_processed:
        c_n = (p_eff * (n - 1) / (n - p_eff)) * f.ppf(1 - a, p_eff, n - p_eff)
        metrics[f"lhs_{int(a*100):02d}"] = (unit_ball_vol * (c_n / n) ** (p_eff / 2) * (det ** 0.5)) ** (1 / p_eff) + 1 / n
    for e in epsilons_processed:
        metrics[f"rhs_{int(e*100):02d}"] = e * det ** (1 / (2 * p_eff))

    residuals, active_statuses, overall_active = {}, {}, False
    for a in alphas:
        for e in epsilons:
            suffix = f"a{a:02d}_e{e:02d}"
            res_val = max(0.0, metrics[f"lhs_{a:02d}"] - metrics[f"rhs_{e:02d}"])
            residuals[f"res_{suffix}"] = res_val
            active_statuses[f"rem_{suffix}"] = 1 if res_val > 0 else 0
            if res_val > 0: overall_active = True

    return {
        "status": "active" if overall_active else "converged", 
        "bid": bid, "residuals": residuals, "active_counts": active_statuses, "logs": []
    }

if __name__ == "__main__":
    None
    # TEST in terminal: python -m src.valuation.results