"""
src/sensitivity.py

Surrogate Modeling and Global Sensitivity Analysis (GSA) pipeline for the
In-Depth-PROFILE framework. Manages multi-seed Optuna Bayesian hyperparameter search,
XGBoost model training across individual return periods, and SHAP value.
"""

import os
import gc
import time
import mocaloss as mcl
import numpy as np
import polars as pl
import polars.selectors as cs
import pandas as pd
import xgboost as xgb
import optuna
import shap
import re
import math
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score

# ------------------------------
## 2. GSA
# ------------------------------
# --- USER HELPERS ---
def sample_gsa_assets(
    X, 
    y, 
    meta_df,
    asset_col="BID", 
    it_col="it",
    max_its_per_asset=None,
    min_total_rows=None,
    seed=42
):
    if max_its_per_asset is None:
        return X, y, meta_df

    # 1. Normalize inputs to Polars DataFrames
    pl_X = pl.from_pandas(X) if isinstance(X, pd.DataFrame) else X
    pl_meta = pl.from_pandas(meta_df) if isinstance(meta_df, pd.DataFrame) else meta_df
    
    if isinstance(y, pd.Series):
        pl_y = pl.from_pandas(y.to_frame())
    else:
        pl_y = pl.from_pandas(y) if isinstance(y, pd.DataFrame) else (y if isinstance(y, pl.DataFrame) else pl.DataFrame(y))

    # 2. Check total rows and evaluate min_total_rows constraint
    total_rows = pl_meta.height
    adapted_max_its = max_its_per_asset

    if min_total_rows is not None:
        if total_rows <= min_total_rows:
            print(f"[sample_gsa_assets] Total rows ({total_rows}) <= min_total_rows ({min_total_rows}). Skipping stratification.")
            return X, y, meta_df
        
        num_assets = pl_meta.get_column(asset_col).n_unique()
        adapted_max_its = max(max_its_per_asset, math.ceil(min_total_rows / num_assets))
        
        if adapted_max_its > max_its_per_asset:
            print(f"[sample_gsa_assets] Adapting max_its_per_asset from {max_its_per_asset} to {adapted_max_its} to meet min_total_rows ({min_total_rows}).")
        else:
            print(f"[sample_gsa_assets] Applying stratified sample (max {adapted_max_its} per asset)...")
    else:
        print(f"[sample_gsa_assets] Applying stratified sample (max {adapted_max_its} per asset)...")

    # 3. Concatenate, shuffle row-wise, and apply stratified sampling
    sampled = (
        pl.concat([pl_meta, pl_X, pl_y], how="horizontal_extend")
        .sample(fraction=1.0, shuffle=True, seed=seed)
        .group_by(asset_col, maintain_order=True)
        .head(adapted_max_its)
    )

    # 4. Split back to original structures
    out_X = sampled.select(pl_X.columns)
    out_meta = sampled.select(pl_meta.columns)
    out_y = sampled.get_column(pl_y.columns[0])

    # 5. Restore original Pandas types if they were provided as Pandas
    if isinstance(X, pd.DataFrame):
        out_X = out_X.to_pandas()
    if isinstance(y, (pd.Series, pd.DataFrame)):
        out_y = out_y.to_pandas()
    if isinstance(meta_df, pd.DataFrame):
        out_meta = out_meta.to_pandas()

    return out_X, out_y, out_meta

def get_gsa_data(
    return_period,
    shap_target_col,
    shap_input_groups,
    categorical_cols,
    rp_col = "RP",
    it_col = "it",
    asset_col = "BID",
    col_to_agg = "FID",
    max_its_per_asset = None
):
    print(f"[get_gsa_data] Preparing gsa data")
    meta_cols = [rp_col, it_col, asset_col, col_to_agg]
    shap_input_cols = [col for group in shap_input_groups.values() for col in group]
    gsa_data = (
        results
        .collect_data(columns=meta_cols + shap_input_cols + [shap_target_col])
        .filter(pl.col(rp_col) == return_period) # type: ignore
        .pipe(lambda df: 
            (lambda counts: 
                # 1. Pivot variant columns ONLY on 'FID', keeping 'it' and 'BID' as rows
                df.pivot(
                    index=[it_col, asset_col],
                    on=col_to_agg,
                    values=[col for col, count in counts.items() if count > 1 and col not in [it_col, asset_col]],
                    aggregate_function="first"
                ).join(
                    # 2. Aggregate the constant columns and sum the output column
                    df.group_by([it_col, asset_col]).agg(
                        [pl.col(col).first() for col, count in counts.items() if count == 1 and col not in [it_col, asset_col]] + 
                        [pl.col(shap_target_col).sum()]
                    ),
                    on=[it_col, asset_col],
                    how="inner"
                )
            )(
                # Evaluate unique counts to separate constant vs. variant columns dynamically
                df.group_by([it_col, asset_col])
                .agg(pl.exclude(it_col, asset_col, col_to_agg, shap_target_col).n_unique())
                .max()
                .to_dicts()[0]
            )
        )
        # 3. Drop columns that have ZERO variance across the entire wide dataset
        .pipe(lambda wide_df:
            wide_df.drop([
                col for col, unique_count in 
                wide_df.select([
                    # drop_nulls() ensures all-null columns result in 0 unique values
                    pl.col(c).drop_nulls().n_unique().alias(c) 
                    for c in wide_df.columns 
                    if c not in [it_col, asset_col, shap_target_col]
                ]).to_dicts()[0].items()
                # Drop if n_unique is 0 (all nulls) or 1 (all zeros / all identical constants)
                if unique_count <= 1
            ])
        )
        .with_columns([
            pl.col(f"^{cat_col}(\\_.*)?$")
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64, strict=False)
            .cast(pl.String)
            .cast(pl.Categorical)
            for cat_col in categorical_cols
        ])
        .with_columns(pl.col(shap_target_col).log1p().alias(shap_target_col))
    )
    
    meta_df = gsa_data.select([c for c in meta_cols if c in gsa_data.columns])
    X = gsa_data.drop([c for c in meta_cols if c in gsa_data.columns] + [shap_target_col]).to_pandas()
    y = gsa_data.get_column(shap_target_col).to_pandas()
    
    # Stratified sample per asset
    X, y, meta_df = sample_gsa_assets(
        X, y, meta_df, 
        asset_col=asset_col, 
        it_col=it_col, 
        max_its_per_asset=max_its_per_asset
    )
    
    return meta_df, X, y

def objective(trial, X_opt, y_opt, seed):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 0.85),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.85),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'min_split_loss': trial.suggest_float('min_split_loss', 0.4, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-6, 10.0, log=True),
    }

    model = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='hist',
        enable_categorical=True,
        device='cpu',
        booster='gbtree',
        random_state=seed,
        n_jobs=1
    )

    scores = cross_val_score(
        model, X_opt, y_opt, 
        cv=3, 
        scoring='neg_root_mean_squared_error', 
        n_jobs=-1
    )
    return float(scores.mean())

def run_bayesian_optimization(
    X,
    y,
    max_rows = 30000,
    n_trials = 90,
    seeds = (29, 6, 1997),
    output_path=None
) -> pl.DataFrame:
    """Executes multi-seed Optuna hyperparameter optimization campaigns
    and returns the combined execution metrics as a Polars DataFrame.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    if os.path.exists(optuna_path):
        print(f"[run_bayesian_optimization] Cached study found at {optuna_path}. Loading existing DataFrame.")
        return pl.read_ipc(optuna_path)

    start_time = time.time()
    results_list = []
    for seed in seeds:
        print(f"[run_bayesian_optimization] Using seed {seed}")
        if len(X) <= max_rows:
            print(f"[run_bayesian_optimization] Dataset size ({len(X)}) <= max_rows ({max_rows}). Using full dataset.")
            X_opt = X
            y_opt = y
        else:
            print(f"[run_bayesian_optimization] Subsampling {max_rows} rows from total {len(X)}")
            sampled_idx = X.sample(n=max_rows, random_state=seed).index
            X_opt = X.loc[sampled_idx]
            y_opt = y.loc[sampled_idx]

        print(f"[run_bayesian_optimization] Estimating parameters")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed)
        )

        study.optimize(
            lambda trial: objective(trial, X_opt, y_opt, seed),
            n_trials=n_trials,
            n_jobs=-1,
            show_progress_bar=True,
        )

        best_score = study.best_value
        best_trial_number = study.best_trial.number
        rmse = abs(best_score)
        y_mean = float(y_opt.mean())
        y_std = float(y_opt.std())

        # Build metric record
        result_dict = {
            "rp": 500,
            "seed": seed,
            "best_trial": best_trial_number,
            "best_score": best_score,
            "rRMSE": rmse / y_mean if y_mean != 0 else np.nan,
            "NRMSE_std": rmse / y_std if y_std != 0 else np.nan,
        }
        result_dict.update(study.best_params)
        results_list.append(result_dict)

    # 3. Compute Coefficient of Variation (CV %) across seed scores
    scores = [res["best_score"] for res in results_list]
    mean_score = np.mean(scores)
    cv_seeds = (
        (np.std(scores, ddof=1) / abs(mean_score)) * 100
        if mean_score != 0
        else np.nan
    )

    for res in results_list:
        res["CV_seeds_%"] = cv_seeds
        
    # Construct Polars DataFrame
    results_df = pl.DataFrame(results_list)

    # Reorder key columns to the front
    lead_cols = ["rp", "seed", "best_score", "rRMSE", "NRMSE_std", "CV_seeds_%"]
    param_cols = [c for c in results_df.columns if c not in lead_cols]
    results_df = results_df.select(lead_cols + param_cols)

    print(f"[run_bayesian_optimization] Completed in: {time.time() - start_time:.2f}s.")

    if output_path:
        try: results_df.write_ipc(output_path)
        except Exception as e: print(f"Save failed: {e}")
    return results_df

def get_best_fit(optuna_df):
    # Retrieve best parameter row for target return period
    best_row = (
        optuna_df.filter(pl.col("rp") == 500)
        .sort("best_score", descending=True)
        .head(1)
        .to_dicts()[0]
    )

    # Isolate XGBoost hyperparameters from execution metadata
    exclude_cols = {"rp", "seed", "best_trial", "best_score", "rRMSE", "CV_seeds_%", "NRMSE_std", "R2"}
    xgb_params = {k: v for k, v in best_row.items() if k not in exclude_cols}

    # Cast integer parameters required by XGBoost
    for int_param in ["max_depth", "min_child_weight", "n_estimators"]:
        if int_param in xgb_params:
            xgb_params[int_param] = int(xgb_params[int_param])
    
    return best_row, xgb_params

def validate_best_model(X, y, optuna_path) -> pl.DataFrame:
    """Fits an 80/20 train-test validation split using the optimal hyperparameter profile,

    evaluates $R^2$ performance, saves the fitted model, and outputs an updated metrics DataFrame.
    """
    if not os.path.exists(optuna_path):
        raise FileNotFoundError(f"Optuna file not found at: {optuna_path}. Run run_bayesian_optimization")
    else:
        optuna_df = pl.read_ipc(optuna_path)
        best_row, xgb_params = get_best_fit(optuna_df)

        if "R2" in optuna_df.columns:
            best_r2 = optuna_df.filter(
                (pl.col("seed") == best_row["seed"]) & 
                (pl.col("best_trial") == best_row["best_trial"])
            ).get_column("R2").to_list()
            
            if best_r2 and best_r2[0] is not None:
                print(f"[VALIDATION] R2 score ({best_r2[0]:.4f}) already exists in {optuna_path}. Skipping evaluation.")
                return optuna_df
    
    start_time = time.time()
    
    # Get params
    optuna_df = pl.read_ipc(optuna_path)
    best_row, xgb_params = get_best_fit(optuna_df)
    
    # 80/20 Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=best_row["seed"]
    )

    # Fit
    print(f"[validate_best_model] Fitting XGBoost validation model")
    xgb_model = xgb.XGBRegressor(
        **xgb_params,
        early_stopping_rounds=15,
        random_state=best_row["seed"],
        eval_metric="rmse",
        tree_method="hist",
        booster='gbtree',
        objective="reg:squarederror",
        enable_categorical=True,
        device="cuda",
    )
    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
    )

    # Compute predictions and R2 score
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    y_pred = xgb_model.get_booster().predict(dtest)
    #y_pred = xgb_model.predict(X_test)
    r2_val = float(r2_score(y_test, y_pred))

    print(
        f"[validate_best_model] Complete in {time.time() - start_time:.2f}s"
        f"R2 Score: {r2_val:.4f}"
    )

    if "R2" not in optuna_df.columns:
        optuna_df = optuna_df.with_columns(pl.lit(None).cast(pl.Float64).alias("R2"))
    
    optuna_df = optuna_df.with_columns(
        pl.when(
            (pl.col("seed") == best_row["seed"]) & 
            (pl.col("best_trial") == best_row["best_trial"])
        )
        .then(pl.lit(r2_val))
        .otherwise(pl.col("R2"))
        .alias("R2")
    )

    # Save updated dataframe back to optuna_path directly
    optuna_df.write_ipc(optuna_path)
    print(f"[validate_best_model] Updated R2 metric saved directly to {optuna_path}")

    # Cleanup
    del X_train, X_test, y_train, y_test, xgb_model, y_pred, dtest
    gc.collect()

    return optuna_df

def validate_best_model_kfold(X, y, optuna_path, n_splits=5) -> pl.DataFrame:
    """Fits a K-Fold cross-validation using the optimal hyperparameter profile,
    evaluates R2 performance (mean and std), and outputs an updated metrics DataFrame.
    """
    if not os.path.exists(optuna_path):
        raise FileNotFoundError(f"Optuna file not found at: {optuna_path}. Run run_bayesian_optimization")
    
    optuna_df = pl.read_ipc(optuna_path)
    best_row, xgb_params = get_best_fit(optuna_df)

    if "R2_mean" in optuna_df.columns:
        best_r2 = optuna_df.filter(
            (pl.col("seed") == best_row["seed"]) & 
            (pl.col("best_trial") == best_row["best_trial"])
        ).get_column("R2_mean").to_list()
        
        if best_r2 and best_r2[0] is not None:
            print(f"[VALIDATION] R2_mean score ({best_r2[0]:.4f}) already exists in {optuna_path}. Skipping evaluation.")
            return optuna_df
    
    start_time = time.time()
    
    # K-Fold setup
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=best_row["seed"])
    r2_scores = []
    
    print(f"[validate_best_model] Fitting XGBoost with {n_splits}-Fold Cross Validation")
    
    # Iterate over folds
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        # Pandas indexing (use X[train_idx] if X is a numpy array)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        xgb_model = xgb.XGBRegressor(
            **xgb_params,
            early_stopping_rounds=15,
            random_state=best_row["seed"],
            eval_metric="rmse",
            tree_method="hist",
            booster='gbtree',
            objective="reg:squarederror",
            enable_categorical=True,
            device="cuda",
        )
        
        xgb_model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False,
        )

        # Compute predictions and R2 score
        dtest = xgb.DMatrix(X_test, enable_categorical=True)
        y_pred = xgb_model.get_booster().predict(dtest)
        
        r2_val = float(r2_score(y_test, y_pred))
        r2_scores.append(r2_val)
        
        print(f"  - Fold {fold + 1}/{n_splits} | R2: {r2_val:.4f}")
        
        # Cleanup fold memory
        del X_train, X_test, y_train, y_test, xgb_model, dtest, y_pred
        gc.collect()

    r2_mean = float(np.mean(r2_scores))
    r2_std = float(np.std(r2_scores))

    print(
        f"[validate_best_model] Complete in {time.time() - start_time:.2f}s | "
        f"R2 Mean: {r2_mean:.4f} ± {r2_std:.4f}"
    )

    # Initialize new columns if missing
    for col in ["R2_mean", "R2_std"]:
        if col not in optuna_df.columns:
            optuna_df = optuna_df.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    
    # Update Polars DataFrame
    optuna_df = optuna_df.with_columns(
        pl.when(
            (pl.col("seed") == best_row["seed"]) & 
            (pl.col("best_trial") == best_row["best_trial"])
        )
        .then(pl.lit(r2_mean))
        .otherwise(pl.col("R2_mean"))
        .alias("R2_mean"),
        
        pl.when(
            (pl.col("seed") == best_row["seed"]) & 
            (pl.col("best_trial") == best_row["best_trial"])
        )
        .then(pl.lit(r2_std))
        .otherwise(pl.col("R2_std"))
        .alias("R2_std")
    )

    # Save updated dataframe back to optuna_path directly
    optuna_df.write_ipc(optuna_path)
    print(f"[validate_best_model] Updated R2_mean and R2_std metrics saved directly to {optuna_path}")

    return optuna_df

def fit_surrogate_model(X, y, optuna_path, output_path):
    """Loads a pre-trained surrogate JSON model if present.

    Otherwise, retrieves optimal parameters from optuna_path, fits on full (X, y), and saves to disk.
    """
    if os.path.exists(output_path):
        print(f"[fit_surrogate_model] Cached surrogate model found at {output_path}. Loading model.")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(output_path)
        return xgb_model
    
    start_time = time.time()
    # Get params
    optuna_df = pl.read_ipc(optuna_path)
    best_row, xgb_params = get_best_fit(optuna_df)

    # Fit
    print(f"[fit_surrogate_model] Fitting XGBoost surrogate model")
    xgb_model = xgb.XGBRegressor(
        **xgb_params,
        random_state=best_row["seed"],
        eval_metric="rmse",
        tree_method="hist",
        booster='gbtree',
        objective="reg:squarederror",
        enable_categorical=True,
        device="cuda",
    )
    xgb_model.fit(X, y, verbose=False,)
    xgb_model.save_model(output_path)
    
    gc.collect()
    print(f"[fit_surrogate_model] Completed in {time.time() - start_time:.2f}s")

    return xgb_model

def calculate_shap_groups(xgb_model, X, meta_df, shap_groups, output_path):
    if os.path.exists(output_path):
        print(f"[calculate_shap_groups] Cached SHAP groups file found at {output_path}. Loading IPC.")
        return pl.read_ipc(output_path)
    
    print(f"[calculate_shap_groups] Calculating SHAP values")
    start_time = time.time()
    explainer = shap.TreeExplainer(xgb_model)
    raw_shaps = explainer.shap_values(X)
    shap_df = pl.DataFrame(raw_shaps, schema=list(X.columns))

    # Group by FID
    base_var_mapping = {}
    for col in shap_df.columns:
        base_name = re.sub(r'_-?\d+$', '', col)
        base_var_mapping.setdefault(base_name, []).append(col)
    unpivoted_exprs = [
        pl.sum_horizontal([pl.col(c) for c in cols]).alias(base_var)
        for base_var, cols in base_var_mapping.items()
    ]
    df_unpivoted = shap_df.select(unpivoted_exprs)

    # Group by shap groups
    group_exprs = []
    unclassified_vars = set(df_unpivoted.columns)
    for group_name, vars_in_group in shap_groups.items():
        existing_vars = [v for v in vars_in_group if v in df_unpivoted.columns]
        
        if existing_vars:
            group_exprs.append(
                pl.sum_horizontal([pl.col(v) for v in existing_vars]).alias(group_name)
            )
            unclassified_vars.difference_update(existing_vars)
    shap_groups_df = df_unpivoted.select(group_exprs)
    
    # Include meta
    final_df = pl.concat([meta_df, shap_groups_df], how="horizontal_extend")
    
    # Save groups using output_path
    final_df.write_ipc(output_path)
    print(f"[SHAP] Calculation complete in {time.time() - start_time:.2f}s. Saved to {output_path}")
    
    return final_df
