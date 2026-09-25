import os
import re
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from pandarallel import pandarallel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import colorsys
import mocaloss as mcl

from src.config import PATHS, CODES, RETURN_PERIODS, DIST_CATALOG, RST

def run_pipeline():
    # 0. Config in config.py the script wants to be run using real data
    # (USE_TEST_DATA = False) or test data (USE_TEST_DATA = True)
    
    ## STAGE 1: Scrape data
    RUN = False
    if RUN:
        # 1. Extract Data
        print("### STAGE 1: Extract Data ###")
        from src.scraping import EconomicScraper
        scraper = EconomicScraper(PATHS, CODES)
        scraper.run()
    
        # 2. Clean and Validate Data
        print("### STAGE 1: Clean and Validate Data ###")
        from src.scraping import PriceDataCleaner
        cleaner = PriceDataCleaner(PATHS)
        cleaner.clean_data()
    
    ## STAGE 2: HEC-RAS
    RUN = False
    if RUN:
        # 1. Fit Hydrological Data
        RUN = False
        if RUN:
            print("### STAGE 2: Fit Hydrological Data ###")
            from src.modeling import HydrologicalFitter
            fitter = HydrologicalFitter(PATHS)
            fitted_parameters_df = fitter.fit_flood_frequency(RETURN_PERIODS)
        
        # 2. Prepare Building Data
        RUN = False
        if RUN:
            print("### STAGE 2: Prepare Building Data ###")
            from src.modeling import GeospatialProcessor
            geo_processor = GeospatialProcessor(PATHS)
            gdf_buildings = geo_processor.prepare_building_dem_medians()
        
        # 3. Run HEC-RAS Monte Carlo
        RUN = False
        if RUN:
            print("### STAGE 2: Run HEC-RAS Monte Carlo ###")
            from src.modeling import HecRasMonteCarloEngine
            engine = HecRasMonteCarloEngine(PATHS, RETURN_PERIODS, RST)
            engine.run_monte_carlo()
            engine.get_representative_floods()
            
    ## STAGE 3: Economic Valuation
    RUN = False
    if RUN:
        
        # 1. Fit data
        RUN = False
        if RUN:
            print("### STAGE 3: Fit Data ###")
            from src.valuation.preparation import DistributionFitter
            
            # NOTE: This a newer implementation example using mocaloss library
            # sampling_table_v1.4.pkl is already populated, original fit from survey
            # was done with preparation_old.py, maintained for crosscomparision
            
            # Initialize the fitter class
            fitter = DistributionFitter(PATHS, CODES, RETURN_PERIODS, DIST_CATALOG,)
            
            # Prepare and save sampling rules
            fitter.fit_hazard() 
            fitter.complete_sampling_rules()
            fitter.save() 

        # 2. Run Loss Model
        RUN = False
        if RUN:
            print("### STAGE 3: Run Loss Model ###")
            from src.valuation.execution import LossModelExecutionEngine
            
            # NOTE: This is a newer implementation done with mocaloss
                
            # 1. Initialize the engine class
            engine = LossModelExecutionEngine(PATHS, CODES, RETURN_PERIODS)
            
            # 2. Run mocaloss model
            engine.run_model()
            
    ## STAGE 4: Sensitivity Analysis
    RUN = False
    if RUN:
        print("### STAGE 3: Run Sensitivity Analysis ###")
        from src.valuation.sensitivity import (
            get_gsa_data,
            run_bayesian_optimization,
            validate_best_model,
            sample_gsa_assets,
            fit_surrogate_model,
            calculate_shap_groups
        )
        # NOTE: This is a newer implementation done with mocaloss, the original implementation code is in sensitivity_old.py
        
        # ------------------------------
        ## 1. SET-UP
        # ------------------------------
        # --- INITIALIZATION ---
        dir_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss"
        results = mcl.PostProcessor(working_dir=dir_path)
        results.load(chunk_dirs = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\.cache\mocaloss_results")
        results.print_columns()

        # paths
        gsa_dir = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\gsa"
        os.makedirs(gsa_dir, exist_ok=True)

        # gsa cols
        shap_target_col = 'c_total'
        shap_input_groups = {
            "he": ["he"],
            "ed": ["ed"],
            "Cga": ["Cga"],
            "building_elevation": ['GL', 'BH', 'IH', 'BL',],
            "price_CTE" : [
                'p_APP', 'p_CLO', 'p_COM', 'p_DEC', 'p_ELE',
                'p_FAD', 'p_FUR', 'p_HHB', 'p_HHG', 'p_INS',
                'p_LEI', 'p_OTH', 'p_SPE', 'p_TOO', 'p_VEH',],
            "price_CTI" : [
                'p_CLE', 'p_ETP', 'p_PLG', 'p_PRW', 'p_PUM',
                'p_WND', 'pr_PRW', 'p_RDR', 'p_SKT', 'p_SOI',
                'p_ENG'],
            "material" : ['m_RDR', 'm_SKT', 'm_SOI',],
            "count_CTE": [
                'n_APP', 'n_CLO', 'n_COM', 'n_DEC', 'n_ELE',
                'n_FAD', 'n_FUR', 'n_HHB', 'n_HHG', 'n_INS',
                'n_LEI', 'n_OTH', 'n_SPE', 'n_TOO', 'n_VEH',],
            "count_CTI": [
                'n_PLG', 'n_ENG', 'n_RDR', 'n_SKT', 'n_WND',],
            "ch_CTE": [
                'hc_APP', 'hc_CLO', 'hc_COM', 'hc_DEC', 'hc_ELE',
                'hc_FAD', 'hc_FUR', 'hc_HHB', 'hc_HHG', 'hc_INS',
                'hc_LEI', 'hc_OTH', 'hc_TOO', 'hc_VEH', 'hc_SPE',],
            "ch_CTI": [
                'hc_ENG', 'hc_PLG', 'hc_PRW', 'hc_RDR', 'hc_SKT',
                'hc_SOI', 'hc_WND',],
        }
        categorical_cols = ['m_RDR', 'm_SKT', 'm_SOI']

        for rp in [5, 10, 25, 50, 100, 200, 500]:
            # 0. Define file paths for this return period
            optuna_path = os.path.join(gsa_dir, f"optuna_rp_{rp}.ipc")
            surrogate_path = os.path.join(gsa_dir, f"surrogate_rp_{rp}.json")
            shap_path = os.path.join(gsa_dir, f"shap_rp_{rp}.ipc")
            
            # By-pass calculated
            if os.path.exists(shap_path):
                print(f"Cached SHAP groups file found at {shap_path}. Loading IPC.")
                continue
                
            # 1. Get data
            meta_df, X, y = get_gsa_data(
                rp,
                shap_target_col,
                shap_input_groups,
                categorical_cols,
                rp_col = "RP",
                it_col = "it",
                asset_col = "BID",
                col_to_agg = "FID"
            )

            # 2. Bayesian optimization
            optuna_df = run_bayesian_optimization(
                X,
                y,
                max_rows=30000,
                n_trials=30,
                seeds=(7, 42, 123, 404, 666),
                output_path=optuna_path
            )
            
            # 3. R2 validation
            validation_df = validate_best_model(X, y, optuna_path)

            # 4. Final model fit
            X, y, meta_df = sample_gsa_assets(
                    X, y, meta_df, 
                    asset_col="BID", 
                    it_col="it", 
                    max_its_per_asset=2000,
                    min_total_rows = 100_000,
                    seed = get_best_fit(optuna_df)[0]["seed"]  
                )
            xgb_model = fit_surrogate_model(X, y, optuna_path, surrogate_path)
            
            # 5. SHAP calculation
            raw_shaps = calculate_shap_groups(xgb_model, X, meta_df, shap_input_groups, shap_path)
    
    ## STAGE 5: Results
    RUN = False
    if RUN:
        from src.valuation.results import run_all_plots
        
        # NOTE: This is sections is specifically created for the model results and it is not 
        # easylly generalizable. All plots used for the published article are included in results.py
        # It is recomended to use that file running plots individually, otherwise you can uncomment next
        # line and run it.
        
        # run_all_plots()
                
        
if __name__ == "__main__":
    run_pipeline()
    
# TEST in terminal: python -m main
