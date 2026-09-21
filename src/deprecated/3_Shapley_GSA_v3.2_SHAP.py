# region 0 Import libraries
# Import libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.add_dll_directory(r"C:\Users\outal\anaconda3\DLLs")
os.add_dll_directory(r"C:\Users\outal\anaconda3\envs\FLoMiM\DLLs")
import gc
import re

import torch
import numpy as np

print(f"NumPy Version: {np.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# --- Needed for GSA ---
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score #
from sklearn.metrics import r2_score
import time
import optuna

# --- Third-Party Libraries: Data Handling ---
import pandas as pd

# --- Project-Specific Imports ---
import User_Paths_and_Inputs
# endregion

# region 1 Set Basic variables
# region 1.1 General global
workspace = User_Paths_and_Inputs.workspace
PATHS = User_Paths_and_Inputs.PATHS
CODES = User_Paths_and_Inputs.CODES
BASE_FILE_NAME = "MC_Part"
RETURN_PERIODS = User_Paths_and_Inputs.RETURN_PERIODS
RPs_unq = list(RETURN_PERIODS.keys())
# endregion
# region 1.2 Paths
input_path = r"C:\Users\outal\Documents\MC_Parts\XGBoost\xgb_dataset.feather"
output_dir = r"C:\Users\outal\Documents\MC_Parts\XGBoost"
output_bayesian_path = os.path.join(output_dir, "xgb_bayesian_search_results.feather")
validation_results_path = os.path.join(output_dir, "xgb_best_fit_validation_results.feather")

# endregion
# region 1.3 General script
y_col = 'c_bf'
cols_to_drop_initial = ['SN', 'BID', 'RP', 'BT']
# Bayesian search
seeds = [29, 6, 1997]  #seeds=[7, 42, 123, 404, 666, 1024, 2026] # seeds=[42]
sample_size = 30_000 # sample_size=5_000 (>10_000)
n_trials = 90 # n_trials=10 (90-180, 10 per each variable searched)

# endregion
# region 1.4 Load df and split it for each RP to save RAM and sppeed up the code
full_df = pd.read_feather(input_path)
for rp in RPs_unq:
    output_path = os.path.join(output_dir, f"xgb_dataset_{rp}.feather")
    if not os.path.exists(output_path):
        df_rp = full_df[full_df['RP'] == rp].reset_index(drop=True)
        df_rp.to_feather(output_path)
        del df_rp
        gc.collect()
del full_df
gc.collect()
# endregion
# endregion

# region 2. Bayesian search
# Perform the fine tuning
if not os.path.exists(output_bayesian_path):
    
    results_list = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Helping functions
    def objective(trial, X, y, current_seed):
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
            device='cpu', # Safe execution, task will spand long time
            random_state=current_seed,
            booster='gbtree',
            enable_categorical=True,
            n_jobs=1
        )
        scores = cross_val_score(
            model, 
            X, 
            y, 
            cv=3, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=1
        )
        
        return scores.mean()

    # Repeat search for each rp
    for rp in RPs_unq: # rp = 5
        start_time = time.time()
        # Get tabular data for rp
        print(f"\nPreparing tabular data for RP {rp}...")
        rp_file_path = os.path.join(output_dir, f"xgb_dataset_{rp}.feather")
        df_rp = pd.read_feather(rp_file_path)
        df_rp = df_rp.dropna(axis=1, how='all')
        df_rp = df_rp.drop(columns=[c for c in cols_to_drop_initial if c in df_rp.columns])
        n_rows = len(df_rp)
        print(f"\nLenght (rows) of tabular data for RP {rp}: {n_rows}")
        
        # Perform searchs for each seed
        for seed in seeds: #seed = 7
            print(f"\nRP {rp}: Estimating seed {seed}...")
            # Take subsample
            df_rp_sampled = df_rp.sample(min(sample_size, len(df_rp)), random_state=seed)
            
            # Check for columns with >90% NaN values in the specific sample
            #nan_percentages = df_rp_sampled.isna().mean()
            #high_nan_cols = nan_percentages[nan_percentages > 0.90].index.tolist()
            #if high_nan_cols:
            #    print(f"  [WARNING] Seed {seed} has the following columns with >90% NaNs: {high_nan_cols}")
                
            # Isolate target variable
            y = df_rp_sampled[y_col]
            X = df_rp_sampled.drop(columns=[y_col])
            
            mat_cols = [c for c in X.columns if c.startswith('m_')]
            for col in mat_cols:
                # Converts float 2.0 to string "2", but keeps actual NaNs as NaN
                X[col] = X[col].map(lambda x: str(int(x)) if pd.notna(x) else x).astype('category')
            
            # Perform search
            study = optuna.create_study(
                direction='maximize', 
                sampler=optuna.samplers.TPESampler(seed=seed)
            )
            study.optimize(
                lambda trial: objective(trial, X, y, seed),
                n_trials=n_trials,
                n_jobs=-1
            )

            # Calculate relative RMSE to y
            rmse = abs(study.best_value)
            y_mean = y.mean()
            r_rmse = rmse / y_mean if y_mean != 0 else np.nan
            
            # Calculate NRMSE normalized by standard deviation
            rmse = abs(study.best_value)
            y_std = y.std()
            nrmse_std = rmse / y_std if y_std != 0 else np.nan
            
            # Append results (include relative RMSE)
            result_dict = study.best_params.copy()
            result_dict['rp'] = rp
            result_dict['seed'] = seed
            result_dict['best_score'] = study.best_value
            result_dict['rRMSE'] = r_rmse
            result_dict['NRMSE_std'] = nrmse_std
            results_list.append(result_dict)
        
        # Coefficient of Variation across all seeds
        current_rp_scores = [res['best_score'] for res in results_list if res['rp'] == rp]
        cv_rp = (np.std(current_rp_scores) / np.abs(np.mean(current_rp_scores))) * 100
        
        # Append CV to results
        for res in results_list:
            if res['rp'] == rp:
                res['CV_seeds_%'] = cv_rp
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nRP {rp}: Execution time: {execution_time:.2f} seconds\n({execution_time/len(seeds):.2f} per seed)")

    # Results to df
    results_df = pd.DataFrame(results_list)
    cols = ['rp', 'seed', 'best_score'] + [c for c in results_df.columns if c not in ['rp', 'seed', 'best_score']]
    results_df = results_df[cols]

    # Save results
    results_df.to_feather(output_bayesian_path)
else:
    # Load results_df
    results_df = pd.read_feather(output_bayesian_path)
# Create a df of the best params
best_params_df = results_df.loc[results_df.groupby('rp')['best_score'].idxmax()]
print(best_params_df)
# endregion

# region 3 XGBoost fit
exclude_cols = ['rp', 'seed', 'best_score', 'rRMSE', 'CV_seeds_%', 'NRMSE_std']
#param_cols = [c for c in results_df.columns if c not in exclude_cols]
param_cols = [c for c in best_params_df.columns if c not in exclude_cols]
results_df = pd.read_feather(output_bayesian_path)
best_params_df = results_df.loc[results_df.groupby('rp')['best_score'].idxmax()]
# region 3.1 For Validation (R2)
if not os.path.exists(validation_results_path):
    r2_scores_dict = {}
    
    for rp in RPs_unq: #rp=500
        start_time = time.time()
        model_path = os.path.join(output_dir, f"xgb_model_validation_rp_{rp}.json")
        rp_file_path = os.path.join(output_dir, f"xgb_dataset_{rp}.feather")
        
        print(f"\nLoading and precomputing tabular data for validation from: {rp_file_path}")
        df_rp = pd.read_feather(rp_file_path)
        df_rp = df_rp.dropna(axis=1, how='all')
        df_rp = df_rp.drop(columns=[c for c in cols_to_drop_initial if c in df_rp.columns])
        
        y = df_rp[y_col]
        X = df_rp.drop(columns=[y_col])
        
        mat_cols = [c for c in X.columns if c.startswith('m_')]
        for col in mat_cols:
            # Converts float 2.0 to string "2", but keeps actual NaNs as NaN
            X[col] = X[col].map(lambda x: str(int(x)) if pd.notna(x) else x).astype('category')
            

        # Free memory before splitting
        del df_rp
        gc.collect()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Free memory of full X and y
        del X, y
        gc.collect()
        
        # Get parameters from the best_params_df
        best_params = best_params_df[best_params_df['rp'] == rp].iloc[0].to_dict()
        xgb_params = {k: v for k, v in best_params.items() if k in param_cols}
        xgb_params['max_depth'] = int(xgb_params['max_depth'])
        xgb_params['min_child_weight'] = int(xgb_params['min_child_weight'])
        xgb_params['n_estimators'] = int(xgb_params['n_estimators'])
        
        if not os.path.exists(model_path):
            xgb_model = xgb.XGBRegressor(
                **xgb_params, random_state=42, device='cuda', eval_metric='rmse', tree_method='hist',
                booster='gbtree', objective='reg:squarederror', enable_categorical=True
            )
            print(f"Fitting RP {rp} XGBoost model...")
            xgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
            xgb_model.save_model(model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Model for RP {rp} already exists. Loading...")
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(model_path)
            
        print(f"Validating RP {rp} XGBoost model...")
        dtest_validation = xgb.DMatrix(X_test, enable_categorical=True)
        y_pred = xgb_model.get_booster().predict(dtest_validation)
        # y_pred = xgb_model.predict(X_test) # Alternative
        r2 = r2_score(y_test, y_pred)
        print(f"RP {rp}: R2 Score: {r2:.3f}")
        
        r2_scores_dict[rp] = r2
        
        # Clear memory
        del X_train, X_test, y_train, y_test, xgb_model, y_pred, dtest_validation
        gc.collect()
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for RP {rp}: {execution_time:.2f} seconds")
        
    # Update best_params_df with R2 scores and save to validation_results_path
    best_params_df['R2'] = best_params_df['rp'].map(r2_scores_dict)
    best_params_df.reset_index(drop=True).to_feather(validation_results_path)
    print(f"\nSaved updated best parameters with R2 scores to {validation_results_path}")
    print(best_params_df)
else:
    best_params_df = pd.read_feather(validation_results_path)
    print(f"\nValidation results already exist. Loaded from {validation_results_path}")
    print(best_params_df)
#endregion
# region 3.2 For Final SHAP (all BIDs, all dataset)
for rp in RPs_unq: #rp=5
    start_time = time.time()
    model_path = os.path.join(output_dir, f"xgb_model_complete_rp_{rp}.json")
    rp_file_path = os.path.join(output_dir, f"xgb_dataset_{rp}.feather")
    
    if not os.path.exists(model_path):
        print(f"\nLoading complete tabular data for RP {rp} from: {rp_file_path}")
        df_rp = pd.read_feather(rp_file_path)
        df_rp = df_rp.dropna(axis=1, how='all')
        df_rp = df_rp.drop(columns=[c for c in cols_to_drop_initial if c in df_rp.columns])
        
        y = df_rp[y_col]
        X = df_rp.drop(columns=[y_col])
        
        mat_cols = [c for c in X.columns if c.startswith('m_')]
        for col in mat_cols:
            # Converts float 2.0 to string "2", but keeps actual NaNs as NaN
            X[col] = X[col].map(lambda x: str(int(x)) if pd.notna(x) else x).astype('category')
        
        del df_rp
        gc.collect()
        
        # Get parameters from the best_params_df
        best_params = best_params_df[best_params_df['rp'] == rp].iloc[0].to_dict()
        xgb_params = {k: v for k, v in best_params.items() if k in param_cols}
        xgb_params['max_depth'] = int(xgb_params['max_depth'])
        xgb_params['min_child_weight'] = int(xgb_params['min_child_weight'])
        xgb_params['n_estimators'] = int(xgb_params['n_estimators'])
        
        xgb_model = xgb.XGBRegressor(
            **xgb_params, random_state=42, device='cuda', eval_metric='rmse', tree_method='hist',
            booster='gbtree', objective='reg:squarederror', enable_categorical=True
        )
        print(f"Fitting complete RP {rp} XGBoost model...")
        xgb_model.fit(
            X,
            y,
            verbose=False
        )
        xgb_model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        del X, y, xgb_model
        gc.collect()
    else:
        print(f"\nComplete model for RP {rp} already exists. Skipping fit.")
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for RP {rp}: {execution_time:.2f} seconds")
# endregion
# endregion

# region 4 SHAP
# region 4.1 SHAP Calculation
for rp in RPs_unq: #rp=5
    start_time = time.time()
    shap_path = os.path.join(output_dir, f"shap_results_rp_{rp}.feather")
    
    if not os.path.exists(shap_path):
        rp_file_path = os.path.join(output_dir, f"xgb_dataset_{rp}.feather")
        print(f"\nLoading tabular data for SHAP from: {rp_file_path}")
        
        # Load dataset and extract 'BID' before dropping metadata columns
        df_rp = pd.read_feather(rp_file_path)
        bids = df_rp['BID'].copy()
        
        df_rp = df_rp.dropna(axis=1, how='all')
        df_rp = df_rp.drop(columns=[c for c in cols_to_drop_initial if c in df_rp.columns])
        
        # Separate features (target variable is dropped to match features used in training)
        X = df_rp.drop(columns=[y_col])
        
        del df_rp
        gc.collect()
        
        # Verify existence of BH
        if 'BH' not in X.columns:
            print("No BH")
        
        # Load model
        model_path = os.path.join(output_dir, f"xgb_model_complete_rp_{rp}.json")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
        
        # SHAP calculation
        print(f"Calculating RP {rp} SHAP values...")
        dtest_shap = xgb.DMatrix(X)
        shap_preds = xgb_model.get_booster().predict(dtest_shap, pred_contribs=True)    

        # Convert to DataFrame to save as feather
        shap_cols = X.columns.tolist() + ['base_value']
        shap_df = pd.DataFrame(shap_preds, columns=shap_cols)
        
        # Add back the BID codes
        shap_df['BID'] = bids.values
        
        # Save directly to disk
        shap_df.to_feather(shap_path)
        
        # Clear memory
        del X, bids, xgb_model, dtest_shap, shap_preds, shap_df
        gc.collect()
    else:
        print(f"SHAP results for RP {rp} already exist. Skipping.")
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution time for RP {rp}: {execution_time:.2f} seconds")
# endregion
# region 4.2 SHAP Groups
for rp in RPs_unq:
    start_time = time.time()
    shap_path = os.path.join(output_dir, f"shap_results_rp_{rp}.feather")
    
    if not os.path.exists(shap_path):
        print(f"SHAP file for RP {rp} not found at {shap_path}. Skipping.")
        continue
        
    print(f"\nProcessing MultiIndex feature groupings for RP {rp}...")
    df_shap = pd.read_feather(shap_path)
    
    # Store metadata columns to restore them later without groupings
    metadata_cols = {}
    for meta_col in ['base_value', 'BID']:
        if meta_col in df_shap.columns:
            metadata_cols[meta_col] = df_shap[meta_col].values
            df_shap = df_shap.drop(columns=[meta_col])
            
    features = df_shap.columns.tolist()
    print(f"Number of features: {len(features)}")
    
    # Initialize classification lists
    gc_l1_list = []
    gc_l2_list = []
    gc_l3_list = []
    
    structure_vars = ['IH', 'BH', 'GL', 'Cga']
    individual_vars = ['ed', 'he',]
    
    for c in features:
        l1, l2, l3 = "Unclassified", "Unclassified", "Unclassified"
        
        # Rule 1: Structure Group
        if c in structure_vars:
            l1, l2, l3 = "Structure", "Structure", "Structure"
            
        # Rule 2: Individual variables
        elif c in individual_vars:
            l1, l2, l3 = c, c, c
            
        # Rule 3: Prices (p_)
        elif c.startswith('p_'):
            l3 = "Prices (p_)"
            if c.endswith('_-1'):
                l1 = "Prices -1floor (p_)"
            elif c.endswith('_0'):
                l1 = "Prices 0floor (p_)"
            elif c.endswith('_1'):
                l1 = "Prices 1floor (p_)"
            elif c.endswith('_2'):
                l1 = "Prices 2floor (p_)"
            elif c.endswith('_3'):
                l1 = "Prices 3floor (p_)"
            elif not re.search(r'_-?\d+$', c):
                l1 = "Prices global (p_)"
                
            if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c):
                l2 = "Content"
            elif re.search(r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)', c) or c.startswith('p_ETP') or c.startswith('p_EXF') or c.startswith('p_WND') or c.startswith('p_PUM') or c.startswith('p_PRW') or c.startswith('p_ITP') or c.startswith('p_SOI') or c.startswith('p_CLE') or c.startswith('p_SKT') or c.startswith('p_RDR') or c.startswith('p_PLG'):
                l2 = "Continent"
                
        # Rule 4: Objects (n_)
        elif c.startswith('n_'):
            l3 = "Objects (n_)"
            if c.endswith('_-1'):
                l1 = "Objects -1floor (n_)"
            elif c.endswith('_0'):
                l1 = "Objects 0floor (n_)"
            elif c.endswith('_1'):
                l1 = "Objects 1floor (n_)"
            elif c.endswith('_2'):
                l1 = "Objects 2floor (n_)"
            elif c.endswith('_3'):
                l1 = "Objects 3floor (n_)"
            elif not re.search(r'_-?\d+$', c):
                l1 = "Objects global (n_)"
                
            if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c):
                l2 = "Content"
            elif re.search(r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)', c):
                l2 = "Continent"
                
        # Rule 5: critic high (hc_)
        elif c.startswith('hc_'):
            l3 = "C.High (hc_)"
            if c.endswith('_-1'):
                l1 = "C.High -1floor (hc_)"
            elif c.endswith('_0'):
                l1 = "C.High 0floor (hc_)"
            elif c.endswith('_1'):
                l1 = "C.High 1floor (hc_)"
            elif c.endswith('_2'):
                l1 = "C.High 2floor (hc_)"
            elif c.endswith('_3'):
                l1 = "C.High 3floor (hc_)"
            elif not re.search(r'_-?\d+$', c):
                l1 = "C.High global (hc_)"
                
            if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c):
                l2 = "Content"
            elif re.search(r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)', c):
                l2 = "Continent"
                
        # Rule 6: Materials (m_)
        elif c.startswith('m_'):
            l3 = "Materials (m_)"
            l1 = "Materials (m_)"
            if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c):
                l2 = "Content"
            elif re.search(r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)', c) or c.startswith('m_ETP') or c.startswith('m_EXF') or c.startswith('m_WND') or c.startswith('m_PUM') or c.startswith('m_PRW') or c.startswith('m_ITP') or c.startswith('m_SOI') or c.startswith('m_CLE') or c.startswith('m_SKT') or c.startswith('m_RDR') or c.startswith('m_PLG'):
                l2 = "Continent"
                
        gc_l1_list.append(l1)
        gc_l2_list.append(l2)
        gc_l3_list.append(l3)
    
    if "Unclassified" in gc_l1_list or "Unclassified" in gc_l2_list or "Unclassified" in gc_l3_list:
        unclassified_idx = [i for i, x in enumerate(gc_l1_list) if x == "Unclassified"]
        unclassified_features = [features[i] for i in unclassified_idx]
        print(f"❌ Structural Mapping Error: Unclassified columns found: {unclassified_features}")
        raise ValueError("Group mappings failed to categorize all dataframe columns.")
    print(f"Number of features (L1): {len(gc_l1_list)}")
    print(f"Number of features (L2): {len(gc_l2_list)}")
    print(f"Number of features (L3): {len(gc_l3_list)}")
    
    # Create the MultiIndex for the feature columns
    multi_idx_tuples = list(zip(features, gc_l1_list, gc_l2_list, gc_l3_list))
    feature_multiindex = pd.MultiIndex.from_tuples(
        multi_idx_tuples, 
        names=['Feature', 'GC_L1', 'GC_L2', 'GC_L3']
    )
    df_shap.columns = feature_multiindex
    
    # Append metadata columns with matching MultiIndex levels filled with metadata label
    for meta_col, values in metadata_cols.items():
        df_shap[(meta_col, meta_col, meta_col, meta_col)] = values
        
    # Flatten MultiIndex into flat strings before writing to feather format
    df_shap_flat = df_shap.copy()
    df_shap_flat.columns = [
        "|".join(col) if isinstance(col, tuple) else col 
        for col in df_shap_flat.columns
    ]
    
    df_shap_flat.to_feather(shap_path)
    print(f"✅ Mapping complete. Flattened MultiIndex columns written to file: {shap_path}")
    
    del df_shap, df_shap_flat
    gc.collect()
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for grouping RP {rp}: {execution_time:.2f} seconds")

# endregion
# endregion
