"""
src/sensitivity.py

Surrogate Modeling and Global Sensitivity Analysis (GSA) pipeline for the
In-Depth-PROFILE framework. Manages multi-seed Optuna Bayesian hyperparameter search,
XGBoost model training across individual return periods, and SHAP value generation
with hierarchical MultiIndex feature classification.
"""

import os
import re
import gc
import time
import torch
import optuna
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score

# ==============================================================================
# CORE OBJECTIVE FUNCTIONS (OPTUNA WORKERS)
# ==============================================================================

def _optuna_objective(trial, X, y, current_seed):
    """
    Objective function for Optuna hyperparameter optimization trials.
    Executes a 3-fold cross-validation routine overcpu channels.
    """
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.65, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'min_split_loss': trial.suggest_float('min_split_loss', 0.3, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 200, 400),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }
    
    model = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        device='cpu',
        random_state=current_seed,
        booster='gbtree',
        enable_categorical=True,
        n_jobs=1
    )
    
    scores = cross_val_score(
        model, X, y, cv=3, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=1
    )
    return scores.mean()

# ==============================================================================
# PIPELINE MANAGEMENT CLASS
# ==============================================================================

class SurrogateSensitivityEngine:
    """
    Manages the lifecycle of hyperparameter optimization, training, validation,
    and structural SHAP value grouping for XGBoost surrogate models.
    """
    def __init__(self, config_paths, config_return_periods):
        """Initializes the pipeline engine with systemic directory and metadata paths."""
        self.paths = config_paths
        self.return_periods = list(config_return_periods.keys())
        self.output_dir = os.path.dirname(self.paths['xgb_dataset'])
        
        # Operational variables configuration constants
        self.y_col = 'c_bf'
        self.cols_to_drop_initial = ['SN', 'BID', 'RP', 'BT']
        self.seeds = [29, 6, 1997]
        self.sample_size = 30000
        self.n_trials = 90
        
        # Target asset configuration outputs
        self.output_bayesian_path = os.path.join(self.output_dir, "xgb_bayesian_search_results.feather")
        self.validation_results_path = os.path.join(self.output_dir, "xgb_best_fit_validation_results.feather")
        
        # System setup adjustments
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def shard_dataset_by_return_period(self):
        """Splits the unified dataset into independent shards by return period to minimize RAM spikes."""
        print("[SURROGATE] Slicing master dataset into return period parquet/feather shards...")
        full_df = pd.read_parquet(self.paths['xgb_dataset'])
        
        for rp in self.return_periods:
            output_path = os.path.join(self.output_dir, f"xgb_dataset_{rp}.feather")
            if not os.path.exists(output_path):
                df_rp = full_df[full_df['RP'] == rp].reset_index(drop=True)
                df_rp.to_feather(output_path)
                del df_rp
                gc.collect()
        del full_df
        gc.collect()

    def _preprocess_features(self, df):
        """Isolates targets and encodes material categories as categorical variants."""
        df = df.dropna(axis=1, how='all')
        df = df.drop(columns=[c for c in self.cols_to_drop_initial if c in df.columns])
        
        y = df[self.y_col]
        X = df.drop(columns=[self.y_col])
        
        mat_cols = [c for c in X.columns if c.startswith('m_')]
        for col in mat_cols:
            X[col] = X[col].map(lambda x: str(int(x)) if pd.notna(x) else x).astype('category')
            
        return X, y

    def run_bayesian_optimization(self):
        """Executes multi-seed Optuna hyperparameter optimization campaigns across all return periods."""
        if os.path.exists(self.output_bayesian_path):
            print("[SURROGATE] Bayesian parameters matrix already exists. Skipping optimizations.")
            return pd.read_feather(self.output_bayesian_path)

        results_list = []
        for rp in self.return_periods:
            start_time = time.time()
            rp_file_path = os.path.join(self.output_dir, f"xgb_dataset_{rp}.feather")
            df_rp = pd.read_feather(rp_file_path)
            
            for seed in self.seeds:
                print(f"[TUNING] Return Period {rp} | Estimating parameters using Seed {seed}...")
                df_sampled = df_rp.sample(min(self.sample_size, len(df_rp)), random_state=seed)
                X, y = self._preprocess_features(df_sampled)
                
                study = optuna.create_study(
                    direction='maximize', 
                    sampler=optuna.samplers.TPESampler(seed=seed)
                )
                study.optimize(lambda trial: _optuna_objective(trial, X, y, seed), n_trials=self.n_trials, n_jobs=-1)
                
                rmse = abs(study.best_value)
                y_mean = y.mean()
                y_std = y.std()
                
                result_dict = study.best_params.copy()
                result_dict.update({
                    'rp': rp, 'seed': seed, 'best_score': study.best_value,
                    'rRMSE': rmse / y_mean if y_mean != 0 else np.nan,
                    'NRMSE_std': rmse / y_std if y_std != 0 else np.nan
                })
                results_list.append(result_dict)
            
            # Compute operational Coefficient of Variation metrics across seeds
            rp_scores = [res['best_score'] for res in results_list if res['rp'] == rp]
            cv_rp = (np.std(rp_scores) / np.abs(np.mean(rp_scores))) * 100
            for res in results_list:
                if res['rp'] == rp: res['CV_seeds_%'] = cv_rp
                
            print(f"[TUNING] Complete for RP {rp} in {time.time() - start_time:.2f} seconds.")

        results_df = pd.DataFrame(results_list)
        cols = ['rp', 'seed', 'best_score'] + [c for c in results_df.columns if c not in ['rp', 'seed', 'best_score']]
        results_df = results_df[cols]
        results_df.to_feather(self.output_bayesian_path)
        return results_df

    def validate_and_fit_models(self, tuning_results):
        """Fits validation splits using GPU environments to measure baseline $R^2$ accuracy values."""
        best_params_df = tuning_results.loc[tuning_results.groupby('rp')['best_score'].idxmax()].copy()
        exclude_cols = ['rp', 'seed', 'best_score', 'rRMSE', 'CV_seeds_%', 'NRMSE_std']
        param_cols = [c for c in best_params_df.columns if c not in exclude_cols]
        
        if not os.path.exists(self.validation_results_path):
            r2_scores_dict = {}
            for rp in self.return_periods:
                start_time = time.time()
                model_path = os.path.join(self.output_dir, f"xgb_model_validation_rp_{rp}.json")
                rp_file_path = os.path.join(self.output_dir, f"xgb_dataset_{rp}.feather")
                
                df_rp = pd.read_feather(rp_file_path)
                X, y = self._preprocess_features(df_rp)
                del df_rp
                gc.collect()
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                del X, y
                gc.collect()
                
                best_params = best_params_df[best_params_df['rp'] == rp].iloc[0].to_dict()
                xgb_params = {k: v for k, v in best_params.items() if k in param_cols}
                for int_param in ['max_depth', 'min_child_weight', 'n_estimators']:
                    xgb_params[int_param] = int(xgb_params[int_param])
                
                if not os.path.exists(model_path):
                    xgb_model = xgb.XGBRegressor(
                        **xgb_params,
                        random_state=42,
                        device='cuda',
                        eval_metric='rmse',
                        tree_method='hist',
                        booster='gbtree',
                        objective='reg:squarederror',
                        enable_categorical=True
                    )
                    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
                    xgb_model.save_model(model_path)
                else:
                    xgb_model = xgb.XGBRegressor()
                    xgb_model.load_model(model_path)
                    
                dtest = xgb.DMatrix(X_test, enable_categorical=True)
                y_pred = xgb_model.get_booster().predict(dtest)
                r2_scores_dict[rp] = r2_score(y_test, y_pred)
                
                del X_train, X_test, y_train, y_test, xgb_model, y_pred, dtest
                gc.collect()
                print(f"[VALIDATION] Return Period {rp} complete. R2 Score: {r2_scores_dict[rp]:.3f} in {time.time() - start_time:.2f}s")
                
            best_params_df['R2'] = best_params_df['rp'].map(r2_scores_dict)
            best_params_df.reset_index(drop=True).to_feather(self.validation_results_path)
        else:
            best_params_df = pd.read_feather(self.validation_results_path)
            
        return best_params_df

    def fit_complete_surrogates(self, best_params_df):
        """Fits finalized XGBoost models on the entire architectural footprint database."""
        exclude_cols = ['rp', 'seed', 'best_score', 'rRMSE', 'CV_seeds_%', 'NRMSE_std', 'R2']
        param_cols = [c for c in best_params_df.columns if c not in exclude_cols]
        
        for rp in self.return_periods:
            model_path = os.path.join(self.output_dir, f"xgb_model_complete_rp_{rp}.json")
            if os.path.exists(model_path): continue
            
            rp_file_path = os.path.join(self.output_dir, f"xgb_dataset_{rp}.feather")
            df_rp = pd.read_feather(rp_file_path)
            X, y = self._preprocess_features(df_rp)
            del df_rp
            gc.collect()
            
            best_params = best_params_df[best_params_df['rp'] == rp].iloc[0].to_dict()
            xgb_params = {k: v for k, v in best_params.items() if k in param_cols}
            for int_param in ['max_depth', 'min_child_weight', 'n_estimators']:
                xgb_params[int_param] = int(xgb_params[int_param])
                
            xgb_model = xgb.XGBRegressor(
                **xgb_params,
                random_state=42,
                device='cuda', eval_metric='rmse',
                tree_method='hist', booster='gbtree', objective='reg:squarederror', enable_categorical=True
            )
            print(f"[MODEL FIT] Executing complete spatial matrix fit for RP {rp} over CUDA channels...")
            xgb_model.fit(X, y, verbose=False)
            xgb_model.save_model(model_path)
            
            del X, y, xgb_model
            gc.collect()

    def compute_and_group_shap_values(self):
        """Generates raw SHAP contribution limits and packages them into classified structural groups."""
        structure_vars = ['IH', 'BH', 'GL', 'Cga']
        individual_vars = ['ed', 'he']
        
        for rp in self.return_periods:
            start_time = time.time()
            shap_path = os.path.join(self.output_dir, f"shap_results_rp_{rp}.feather")
            rp_file_path = os.path.join(self.output_dir, f"xgb_dataset_{rp}.feather")
            
            if not os.path.exists(shap_path):
                df_rp = pd.read_feather(rp_file_path)
                bids = df_rp['BID'].copy()
                X, _ = self._preprocess_features(df_rp)
                del df_rp
                gc.collect()
                
                model_path = os.path.join(self.output_dir, f"xgb_model_complete_rp_{rp}.json")
                xgb_model = xgb.XGBRegressor()
                xgb_model.load_model(model_path)
                
                print(f"[SHAP] Driving variance metrics projections for RP {rp}...")
                dtest_shap = xgb.DMatrix(X)
                shap_preds = xgb_model.get_booster().predict(dtest_shap, pred_contribs=True)
                
                shap_cols = X.columns.tolist() + ['base_value']
                df_shap = pd.DataFrame(shap_preds, columns=shap_cols)
                df_shap['BID'] = bids.values
                df_shap.to_feather(shap_path)
                
                del X, bids, xgb_model, dtest_shap, shap_preds, df_shap
                gc.collect()
                
            print(f"[SHAP CLASSIFICATION] Restructuring feature layout index metrics for RP {rp}...")
            df_shap = pd.read_feather(shap_path)
            metadata_cols = {col: df_shap[col].values for col in ['base_value', 'BID'] if col in df_shap.columns}
            df_shap = df_shap.drop(columns=list(metadata_cols.keys()))
            
            features = df_shap.columns.tolist()
            gc_l1, gc_l2, gc_l3 = [], [], []
            
            for c in features:
                l1, l2, l3 = "Unclassified", "Unclassified", "Unclassified"
                if c in structure_vars:
                    l1, l2, l3 = "Structure", "Structure", "Structure"
                elif c in individual_vars:
                    l1, l2, l3 = c, c, c
                elif c.startswith('p_'):
                    l3 = "Prices (p_)"
                    for f in ['-1', '0', '1', '2', '3']:
                        if c.endswith(f'_{f}'): l1 = f"Prices {f}floor (p_)"
                    if not re.search(r'_-?\d+$', c): l1 = "Prices global (p_)"
                    l2 = "Content" if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c) else "Continent"
                elif c.startswith('n_'):
                    l3 = "Objects (n_)"
                    for f in ['-1', '0', '1', '2', '3']:
                        if c.endswith(f'_{f}'): l1 = f"Objects {f}floor (n_)"
                    if not re.search(r'_-?\d+$', c): l1 = "Objects global (n_)"
                    l2 = "Content" if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c) else "Continent"
                elif c.startswith('hc_'):
                    l3 = "C.High (hc_)"
                    for f in ['-1', '0', '1', '2', '3']:
                        if c.endswith(f'_{f}'): l1 = f"C.High {f}floor (hc_)"
                    if not re.search(r'_-?\d+$', c): l1 = "C.High global (hc_)"
                    l2 = "Content" if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c) else "Continent"
                elif c.startswith('m_'):
                    l1, l3 = "Materials (m_)", "Materials (m_)"
                    l2 = "Content" if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c) else "Continent"
                    
                gc_l1.append(l1); gc_l2.append(l2); gc_l3.append(l3)

            if "Unclassified" in (gc_l1 + gc_l2 + gc_l3):
                raise ValueError("Group mappings failed to categorize all dataframe columns.")
                
            df_shap.columns = pd.MultiIndex.from_tuples(list(zip(features, gc_l1, gc_l2, gc_l3)), names=['Feature', 'GC_L1', 'GC_L2', 'GC_L3'])
            for meta_col, vals in metadata_cols.items(): df_shap[(meta_col, meta_col, meta_col, meta_col)] = vals
            
            df_shap_flat = df_shap.copy()
            df_shap_flat.columns = ["|".join(col) if isinstance(col, tuple) else col for col in df_shap_flat.columns]
            df_shap_flat.to_feather(shap_path)
            
            del df_shap, df_shap_flat
            gc.collect()
            print(f"[SHAP SUCCESS] Structural mapping matrix compiled cleanly for RP {rp} in {time.time() - start_time:.2f}s.")

# ==============================================================================
# PIPELINE INITIALIZATION EXECUTION INTERFACE
# ==============================================================================

if __name__ == "__main__":
    from src.config import PATHS, RETURN_PERIODS

    # 1. Initialize the Sensitivity Engine class
    sensitivity_engine = SurrogateSensitivityEngine(PATHS, RETURN_PERIODS)
    
    # 2. Divide the master database into processing return period segments
    sensitivity_engine.shard_dataset_by_return_period()
    
    # 3. Execute Bayesian Hyperparameter search routines across Optuna channels
    tuning_results = sensitivity_engine.run_bayesian_optimization()
    
    # 4. Extract optimized baseline parameter sets and validate R2 distributions
    best_params_df = sensitivity_engine.validate_and_fit_models(tuning_results)
    
    # 5. Fit global surrogate models across complete cadastre boundaries
    sensitivity_engine.fit_complete_surrogates(best_params_df)
    
    # 6. Execute global sensitivity SHAP analysis metrics summaries
    sensitivity_engine.compute_and_group_shap_values()
