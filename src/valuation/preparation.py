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
from pathlib import Path
import mocaloss as mcl
from mocaloss.preprocessor.fitter import FitOptions

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
        self.sampling_rules_file = self.paths["sampling_rules_name"]
        
        # Initialize mocaloss PreProcessor
        self.preprocessor = mcl.PreProcessor(working_dir=self.paths['mocaloss_working_dir'])
        self.preprocessor.load(self.sampling_rules_file)
    
    def fit_hazard(self):
        df_depth_samples=pd.read_pickle(open(self.paths['depth_samples_pkl'],"rb"))
        bid_max_values = df_depth_samples.groupby('BID')['he'].max()
        bids_to_keep = bid_max_values[bid_max_values > 0].index
        df_filtered = df_depth_samples[df_depth_samples['BID'].isin(bids_to_keep)].copy()
        tasks = {
            (int(rp), int(bid)): group["he"].to_numpy()
            for (rp, bid), group in df_filtered.groupby(["RP", "BID"])
        }
        opts = FitOptions(
            dist_names=["lognorm","weibull_min","gumbel_r","expon","gamma","halfnorm"],
            hurdle=True, min_sample_size=5, n_mc_samples=1000,
            alpha=0.05, accepted_min=0.0, accepted_max=20.0,
            avoid_bs_size=30, n_jobs=-1
        )
        self.preprocessor.add_fit_batch(
            var_name="he",
            tasks=tasks,
            fallback_cols=["RP", "BID"],
            fit_options=opts,
            n_jobs=-1,
        )
    
    def complete_sampling_rules(self):
        content_list = ["APP", "CLO", "COM", "DEC", "ELE", "ENG", "FAD", "FUR", "HHG", "HHB", "INS", "LEI", "OTH", "SPE", "TOO", "VEH"]
        continent_list_items = ["SKT", "RDR", "WND", "PLG",]
        continent_list_general = ["SOI", "PRW"]
        continent_list_general_fix = ["CLE", "PUM"]
        continent_list_paints = ["ITP","ETP"]
        continent_list = continent_list_items + continent_list_general + continent_list_general_fix + continent_list_paints

        self.preprocessor.set_truncation(
            quantiles={
                "he": {"upper": 0.95},
                **{f"p_{code}": {"upper": 0.95} for code in content_list},
            },
            values={
                "he": {"lower": 0.0},
                "Cga": {"lower": 4},
                **{f"p_{code}": {"lower": 0.0} for code in content_list},
            },
        )
        self.preprocessor.set_resolution(
            level_col="BID",
            independent=[
                "IH", "Cga", "GL", "BL", "BH", "ed",
                "m_SOI", "m_SKT", "m_RDR",
                "p_PUM", "p_CLE", "p_SOI", "p_SKT", "p_RDR", "p_WND", "p_PLG", "pr_PRW", "p_PRW", "p_ETP"
            ],
            coupled=["he"]
        )
        self.preprocessor.set_resolution(
            level_col="FID",
            independent=[f"{prefix}_{code}" for prefix in ["n", "p", "f", "bp", "hc"] for code in content_list] + 
                        [f"{prefix}_{code}" for prefix in ["f", "bp", "hc"] for code in continent_list_general] +
                        [f"{prefix}_{code}" for prefix in ["n", "f", "bp", "hc"] for code in continent_list_items], 
            coupled=[]
        )
        self.preprocessor.set_targeted_sample({
            **{f"f_{code}": {"target": "hi", "method": "cdf"} for code in content_list + continent_list_items + continent_list_general},
            **{f"hc_{code}": {"target": f"bp_{code}", "method": "ppf"} for code in content_list + continent_list_items + continent_list_general}
        })
    
    def save(self):
        self.preprocessor.save(self.sampling_rules_file)

if __name__ == "__main__":
    # Centralized imports from config
    from src.config import PATHS, CODES, RETURN_PERIODS, DIST_CATALOG
    
    # NOTE: This a newer implementation example using mocaloss library
    # sampling_table_v1.4.pkl is already populated, original fit from survey
    # was done with preparation_old.py, maintained for crosscomparision
    
    # Initialize the fitter class
    fitter = DistributionFitter(PATHS,CODES,RETURN_PERIODS,DIST_CATALOG,)
    
    # Prepare and save sampling rules
    fitter.fit_hazard() 
    fitter.complete_sampling_rules()
    fitter.save() 
    