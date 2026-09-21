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

from pathlib import Path
import polars as pl
import mocaloss as mcl


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

    def run_model(self):
        # 1. Prepare Base Data Frame
        # Base polar dataframe
        df_ds = pd.read_pickle(self.paths["depth_samples_pkl"])
        df_ds = pl.from_pandas(df_ds)
        BIDs_flooded_any_rp = df_ds.filter(pl.col("he") > 0).get_column("BID").unique().to_list()

        gdf_buildings = gpd.read_file(self.paths["buildings_sample_med_shp"])
        cols_to_keep = ["BID", "BT", "NF", "BF", "HU", "GA", "EP"]
        df_pandas = gdf_buildings[cols_to_keep]
        df_buildings = pl.from_pandas(df_pandas)
        df_buildings = df_buildings.with_columns([pl.col("GA").round(2), pl.col("EP").round(2)])
        df_buildings = (
            df_buildings
            .filter(
                pl.col("BID").is_in(BIDs_flooded_any_rp) & 
                (pl.col("BT") != 0)
            )
            .with_columns([
                pl.when(pl.col(col) == 0)
                .then(None)
                .otherwise(pl.col(col))
                .alias(col) 
                for col in ["NF", "BF", "HU"]
            ])
            .with_columns(
                (pl.col("BF") * -1).alias("BF")
            )
        )

        # mapping for explosion by RP (BIDs flooded some time at specefic rp)
        agg_df = (
            df_ds.filter(pl.col("he") > 0.0)
            .group_by("RP")
            .agg(pl.col("BID").unique())
        )
        rp_mapping = dict(zip(agg_df["RP"], agg_df["BID"]))
        
        # variable lists
        content_list = ["APP", "CLO", "COM", "DEC", "ELE", "ENG", "FAD", "FUR", "HHG", "HHB", "INS", "LEI", "OTH", "SPE", "TOO", "VEH"]
        continent_list_items = ["SKT", "RDR", "WND", "PLG",]
        continent_list_general = ["SOI", "PRW"]
        continent_list_general_fix = ["CLE", "PUM"]
        continent_list_paints = ["ITP","ETP"]
        continent_list = continent_list_items + continent_list_general + continent_list_general_fix + continent_list_paints

        # 2. Model Stage
        # Model
        model = mcl.Model(max_iterations=5_000, working_dir=self.paths['mocaloss_working_dir'], measure_time=True)
        model.add_base_dataframe(df_buildings)
        model.add_explosion(new_col="RP", target_col="BID", mapping=rp_mapping)
        model.add_explosion(new_col="FID", bounds=["BF", "NF"], lower_bound_default_value = 0)

        # Sampling rules
        preprocessor = mcl.PreProcessor(working_dir=self.paths['mocaloss_working_dir'])
        preprocessor.load(self.paths["sampling_rules_name"])
        model.add_sampling_rules(preprocessor)

        # Calculations
        def hi_expr_logic():
            wl_expr = pl.when(pl.col("he") > 0).then((pl.col("he") - pl.col("ed")).clip(lower_bound=0.0)).otherwise(0.0)
            base_expr = pl.when(pl.col("FID") < 0).then(pl.col("BL") + (pl.col("FID").abs() - 1) * pl.col("BH")).otherwise(pl.col("GL") + pl.col("FID") * pl.col("IH"))
            limit_expr = pl.when(pl.col("FID") < 0).then(pl.col("BH")).otherwise(pl.col("IH"))
            raw_hi = wl_expr - base_expr
            clipped_hi = pl.min_horizontal(raw_hi.clip(lower_bound=0.0), limit_expr)
            return pl.when(wl_expr > 0).then(clipped_hi).otherwise(0.0).alias("hi")

        peri_expr = pl.when(pl.col("FID") < 0).then(pl.col("BP")).otherwise(pl.col("GP"))
        area_expr = pl.when(pl.col("FID") < 0).then(pl.col("BA")).otherwise(pl.col("GA"))
        heigh_expr = pl.when(pl.col("FID") < 0).then(pl.col("BH")).otherwise(pl.col("IH"))
        ga_term = pl.when(pl.col("GL") < 0).then(pl.col("GA") * pl.col("GL").abs()).otherwise(0.0)
        ba_term = pl.when(pl.col("BL") < 0).then(pl.col("BA") * pl.col("BL").abs()).otherwise(0.0)
        roof_elev = pl.col("GL") + (pl.col("NF") * pl.col("IH"))
        top_elev = pl.min_horizontal(pl.col("he"), roof_elev)
        flooded_height = (top_elev - pl.col("GL")).clip(lower_bound=0.0)

        model.add_calculations([
            # Building calculations
            (pl.when(pl.col("FID") < 0).then(pl.col("GA")).otherwise(0.0)).alias("BA"),
            (pl.col("Cga") * pl.col("GA").sqrt()).alias("GP"),
            (4 * pl.col("BA").cast(pl.Float64).sqrt()).alias("BP"),
            
            # Water depth inside each floor propagation
            hi_expr_logic(),
            
            # Fragility function aplication to identify damaged items
            *[(pl.col(f"bp_{comp}") <= pl.col(f"f_{comp}")).cast(pl.Int8).alias(f"d_{comp}") for comp in content_list + continent_list_items + continent_list_general],
            
            # Calculation of extensions
            pl.when(pl.col("d_SOI") == 1).then(area_expr).otherwise(0.0).alias("e_SOI"),
            pl.when(pl.col("d_PRW") == 1).then(peri_expr * heigh_expr).otherwise(0.0).alias("e_PRW"),
            pl.when(pl.col("d_SKT") == 1).then(peri_expr * (pl.col("n_SKT").clip(lower_bound=0, upper_bound=20) / 20.0)).otherwise(0.0).alias("e_SKT"),
            (ga_term + ba_term).alias("e_PUM"),
            pl.when(pl.col("hi") > 0).then((peri_expr * pl.col("hi")) + area_expr).otherwise(0.0).alias("e_CLE"),
            (pl.col("EP") * flooded_height).alias("e_EXF"),
            
            ## Calculation of costs
            # Calculation of actual costs (c_)
            *[(pl.col(f"d_{comp}") * pl.col(f"n_{comp}") * pl.col(f"p_{comp}")).alias(f"c_{comp}") for comp in content_list + continent_list_items],
            *[(pl.col(f"e_{comp}") * pl.col(f"p_{comp}")).alias(f"c_{comp}") for comp in ["SOI", "CLE"]],
            (pl.col("e_PRW") * (pl.col("pr_PRW") + pl.col("p_PRW"))).alias("c_PRW"),
            (pl.col("e_PRW") * pl.col("p_ETP")).alias("c_ITP"),
            pl.when((pl.col("FID") == 0) & (pl.col("hi") > 0)).then(pl.col("e_PUM") * pl.col("p_PUM")).otherwise(0.0).alias("c_PUM"),
            pl.when((pl.col("FID") == 0) & (pl.col("he") > 0)).then(pl.col("e_EXF") * pl.col("p_ETP")).otherwise(0.0).alias("c_ETP"),
            
            # Calculation of maximum possible costs (cM_)
            *[(pl.col(f"n_{comp}") * pl.col(f"p_{comp}")).alias(f"cM_{comp}") for comp in content_list + continent_list_items],
            (area_expr * pl.col("p_SOI")).alias("cM_SOI"),
            ((peri_expr * heigh_expr) * (pl.col("pr_PRW") + pl.col("p_PRW"))).alias("cM_PRW"),
            ((peri_expr * heigh_expr) * pl.col("p_ETP")).alias("cM_ITP"),
            
            ## Agregation of costs
            # Agregation of actual costs (c_)
            pl.sum_horizontal([pl.col(f"c_{comp}") for comp in content_list]).alias("c_CTEs"),
            pl.sum_horizontal([pl.col(f"c_{comp}") for comp in continent_list]).alias("c_CTIs"),    
            (pl.col("c_CTEs") * pl.col("HU")).alias("c_huCTEs"),
            (pl.col("c_CTIs") * pl.col("HU")).alias("c_huCTIs"),
            (pl.col("c_CTEs") + pl.col("c_CTIs")).alias("c_total"),
            (pl.col("c_huCTEs") + pl.col("c_huCTIs")).alias("c_hu_total"),
            
            # Agregation of maximum possible costs (c_)
            pl.sum_horizontal([pl.col(f"cM_{comp}") for comp in content_list]).alias("cM_CTEs"),
            pl.sum_horizontal(
                [pl.col(f"cM_{comp}") for comp in continent_list if comp not in ("CLE", "PUM", "ETP")]
                + [pl.col("c_CLE"), pl.col("c_PUM"), pl.col("c_ETP")]
            ).alias("cM_CTIs"),
            (pl.col("cM_CTEs") * pl.col("HU")).alias("cM_huCTEs"),
            (pl.col("cM_CTIs") * pl.col("HU")).alias("cM_huCTIs"),
            (pl.col("cM_CTEs") + pl.col("cM_CTIs")).alias("cM_total"),
            (pl.col("cM_huCTEs") + pl.col("cM_huCTIs")).alias("cM_hu_total"),
        ])

        # Convergence (non deterministic costs)
        cost_cols = [f"c_{comp}" for comp in content_list + continent_list if comp not in ("CLE", "PUM", "ETP")]
        model.set_convergence(
            convergence_targets=["BID"],
            output_targets=cost_cols,
            method="clt",
            return_period_col = "RP",
            metric="mean",
            tolerance_type="relative",
            tolerance_value=0.05,
            significance_level=0.05,
            check_frequency=50
        )

        # Summary
        model.print_summary()
        model.print_dag(target_col = 'c_total', show_dependencies=True, show_blocks= True)
        #model.print_sampling_rules(only_vars=["GL"])

        # Run
        results = model.run()
        results.save()
    
if __name__ == "__main__":
    # Centralized imports from config
    from src.config import PATHS, CODES, RETURN_PERIODS
    
    # NOTE: This is a newer implementation done with mocaloss
    
    # 1. Initialize the engine class
    engine = LossModelExecutionEngine(PATHS, CODES, RETURN_PERIODS)
    
    # 2. Run mocaloss model
    engine.run_model()
