import os
from pathlib import Path
from io import BytesIO
import mocaloss as mcl
from mocaloss import PostProcessor
from mocaloss import CLTUnivariateStrategy, CLTMultivariateStrategy
import pandas as pd
import geopandas as gpd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.ticker import FuncFormatter
from matplotlib_scalebar.scalebar import ScaleBar
from owslib.wms import WebMapService
from scipy.stats import gaussian_kde
from scipy.stats import pearson3
import rasterio
from rasterio import features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry import shape

def run_all_plots():
    # ------------------------------
    ## 1. SET-UP
    # ------------------------------
    # --- INITIALIZATION ---
    results = PostProcessor(working_dir=r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss")
    results.load(chunk_dirs = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\.cache\mocaloss_results")

    # --- SUMARY ---
    all_cols = results.print_columns()
    def print_complete_summary(all_cols, results):
        summary_data = []

        for col in all_cols:
            print(f"[print_complete_summary] Preparing column: {col}")
            # 1. Iteratively load each column one-by-one
            df_col = results.collect_data(columns=[col])
            s = df_col[col]
            
            total_len = len(s)
            pct_null = (s.null_count() / total_len) * 100 if total_len > 0 else 0.0
            
            # Check if the column is numeric to avoid errors on strings (like Building Type 'BT')
            is_numeric = s.dtype in [
                pl.Float64, pl.Float32, pl.Int64, pl.Int32, 
                pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8
            ]
            
            if is_numeric:
                min_val = s.min()
                max_val = s.max()
                q05 = s.quantile(0.05)
                q50 = s.quantile(0.50)
                q95 = s.quantile(0.95)
                pct_zero = ((s == 0).sum() / total_len) * 100 if total_len > 0 else 0.0
            else:
                min_val = None
                max_val = None
                q05 = None
                q50 = None
                q95 = None
                pct_zero = None

            # 2. Print metrics for the column
            print(f"--- {col} ---")
            print(f"Min: {min_val} | Max: {max_val}")
            print(f"Q05: {q05} | Q50 (Median): {q50} | Q95: {q95}")
            zero_str = f"{pct_zero:.2f}%" if pct_zero is not None else "N/A (Non-numeric)"
            print(f"% Zeros: {zero_str} | % Nulls: {pct_null:.2f}%\n")
            
            summary_data.append({
                "column": col,
                "min": min_val,
                "max": max_val,
                "q05": q05,
                "q50": q50,
                "q95": q95,
                "pct_zero": pct_zero,
                "pct_null": pct_null
            })

        # 3. Create and print the final Polars DataFrame showing all columns and rows
        summary_df = pl.DataFrame(summary_data)
        
        print("=== FINAL SUMMARY DATAFRAME ===")
        with pl.Config(tbl_rows=len(all_cols), fmt_str_lengths=50):
            print(summary_df)
        
        return summary_data
    # summary_data = print_complete_summary(all_cols, results)


    # --- GENERAL ---
    DET_IF = {
        10:     461.8,
        50:     817.6,
        100:    991.5,
        500:    1441.5,
    }
    DET_SHP =  {
        10:     r"C:\Users\outal\GITHUB\InDepthPROFILE\data\inputs\gis\OficialFloods\Q10_2Ciclo_PB_2026_Navaluenga.shp",
        50:     r"C:\Users\outal\GITHUB\InDepthPROFILE\data\inputs\gis\OficialFloods\Q50_2Ciclo_PB_2026_Navaluenga.shp",
        100:    r"C:\Users\outal\GITHUB\InDepthPROFILE\data\inputs\gis\OficialFloods\Q100_2Ciclo_PB_2026_Navaluenga.shp",
        500:    r"C:\Users\outal\GITHUB\InDepthPROFILE\data\inputs\gis\OficialFloods\Q500_2Ciclo_PB_2026_Navaluenga.shp",
    }
    RETURN_PERIODS = {
        #2:    [115,   159,    219], # Excluded from the analysis, any damage generated.
        5:    [279,   399,    570],
        10:   [442,   644,    980],
        25:   [652,   1075,   1796],
        50:   [850,   1493,   2663],
        100:  [1082,  2007,   3800],
        200:  [1347,  2638,   5275],
        500:  [1751,  3671,   7908],
    }
    sto_base = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\intermediate\gis\ModeledFloods"
    gsa_dir = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\gsa"
    STO_TIF = {
        10: {
            "Q05": sto_base + "/WSE_RP10_SN23_Q05.tif",
            "Q50": sto_base + "/WSE_RP10_SN79_Q50.tif",
            "Q95": sto_base + "/WSE_RP10_SN68_Q95.tif",
        },
        50: {
            "Q05": sto_base + "/WSE_RP50_SN542_Q05.tif",
            "Q50": sto_base + "/WSE_RP50_SN26_Q50.tif",
            "Q95": sto_base + "/WSE_RP50_SN343_Q95.tif",
        },
        100: {
            "Q05": sto_base + "/WSE_RP100_SN277_Q05.tif",
            "Q50": sto_base + "/WSE_RP100_SN503_Q50.tif",
            "Q95": sto_base + "/WSE_RP100_SN193_Q95.tif",
        },
        500: {
            "Q05": sto_base + "/WSE_RP500_SN213_Q05.tif",
            "Q50": sto_base + "/RP500_Q50.tif",
            "Q95": sto_base + "/WSE_RP500_SN242_Q95.tif",
        }
    }
    color_dic = {
        "RP": {
            5:   "#440154",  # viridis: deep purple
            10:  "#443983",  # purple-blue
            25:  "#31688e",  # blue
            50:  "#21918c",  # teal
            100: "#35b779",  # green
            200: "#90d743",  # lime
            500: "#fde725",  # yellow
        },
        "shap_groups": {
            "he":                 "#E60000",  # Saturated Red
            "ed":                 "#FF6600",  # Vivid Orange
            "Cga":                "#FFD700",  # Bright Gold / Yellow
            "HU":                 "#00E676",  # Bright Spring Green
            "building_elevation": "#00E5FF",  # Electric Cyan
            "price_CTE":          "#0066FF",  # Saturated Royal Blue
            "price_CTI":          "#6600FF",  # Vivid Electric Violet
            "material":           "#FF00FF",  # Saturated Magenta / Fuchsia
            "count_CTE":          "#FF007F",  # Vivid Deep Pink
            "count_CTI":          "#990000",  # Dark Crimson
            "ch_CTE":             "#008080",  # Deep Teal
            "ch_CTI":             "#8B00FF",  # Electric Indigo
        }
    }
    custom_style = {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "black",
        # Spines & Ticks pointing IN on all 4 borders
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Grid aesthetics
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "grid.color": "#bbb8b8",
        "grid.alpha": 0.7,
    }

    # ------------------------------
    ## 2. RESULTS
    # ------------------------------
    # region --- Add external results ---
    # depth_samples and IRME_Ev_by_BID
    depht_samples_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\intermediate\Depth_Samples.pkl"
    pd_df = pd.read_pickle(depht_samples_path)
    depht_samples_df = pl.from_pandas(pd_df) if isinstance(pd_df, pd.DataFrame) else pl.DataFrame(pd_df)
    depht_samples_df = depht_samples_df.rename({"SN": "it"})

    results.add_result(
        res_name = "depth_samples",
        res_description = "Depth samples from HEC-RAS Montecarlo",
        data = depht_samples_df,
    )
    config = {
        "convergence_targets": ["BID"],
        "output_targets": ["he"],    
        "return_period_col": "RP",
        "tolerance_type": "absolut",
        "tolerance_value": 0.1,
        "metric": "mean",
        "significance_level": 0.05,
        "check_frequency": 50,
    }
    strategy = CLTUnivariateStrategy()
    is_converged, evolution_df = strategy.calculate(depht_samples_df, config)
    results.add_result(
        res_name="IRME_Ev_by_BID",
        res_description="""(external) HEC-RAS Monte Carlo IRME univariate metric evolution considering all BID""",
        data=evolution_df,
        overwrite=True
    )

    # IRRSDFV
    try:
        conv_df = results.get_result_as_df(res_name="convergence")
        print("[IRRSDFV] 'convergence' already exists. Loaded from registry.")
    except KeyError:
        print("[IRRSDFV] 'convergence' not found. Calculating manually using CLTMultivariateStrategy...")
        cost_cols = [
            'c_APP', 'c_CLO', 'c_COM', 'c_DEC', 'c_ELE', 'c_ENG', 'c_FAD', 'c_FUR',
            'c_HHG', 'c_HHB', 'c_INS', 'c_LEI', 'c_OTH', 'c_SPE', 'c_TOO', 'c_VEH',
            'c_SKT', 'c_RDR', 'c_WND', 'c_PLG', 'c_SOI', 'c_PRW', 'c_ITP'
        ]
        multi_config = {
            "convergence_targets": ["BID"],
            "output_targets": cost_cols,
            "method": "clt",
            "return_period_col": "RP",
            "metric": "mean",
            "tolerance_value": 0.05,
            "significance_level": 0.05,
            "first_check_point": 100,
            "check_frequency": 1000,
        }
        
        needed_cols = list(dict.fromkeys(
            multi_config.get("convergence_targets", []) + 
            multi_config.get("output_targets", []) + 
            [multi_config.get("return_period_col"), "it"]
        ))
        needed_cols = [col for col in needed_cols if col is not None]
        
        print(f"[IRRSDFV] Collecting required columns from chunked lazy frame: {needed_cols}")
        df_conv_input = results.collect_data(columns=needed_cols)
        
        strategy_multi = CLTMultivariateStrategy()
        is_converged, calculated_conv_df = strategy_multi.calculate(df_conv_input, multi_config) # type: ignore
        
        # 5. Save the processed metric directly into the PostProcessor registry
        results.add_result(
            res_name="convergence",
            res_description="Manual CLT Multivariate convergence (IRRSDFV) fallback",
            data=calculated_conv_df,
            overwrite=True
        )
        print(f"[IRRSDFV] Manual calculation complete. Global convergence status: {is_converged}")

    # input_flow_fit
    if_fit_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\intermediate\Flow_Fitted_Functions.pkl"
    pd_df = pd.read_pickle(if_fit_path)
    pl_df = pl.from_pandas(pd_df) if isinstance(pd_df, pd.DataFrame) else pl.DataFrame(pd_df)
    pl_df = pl_df.rename({"ReturnPeriod": "RP"}).with_columns(
        pl.col("RP").replace(20, 25)
    )
    results.add_result(
        res_name = "input_flow_fit",
        res_description = """Input flow log-pearson type III parameters (Skew, Loc, Scale) by return
        period used in HEC-RAS Montecarlo""",
        data = pl_df,
        overwrite=True
    )

    # Building shapefile (IDEA: Allow add to postprocessor)
    bid_shp = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\inputs\gis\BID\BIDs_v2.3.shp"
    raw_gdf = gpd.read_file(bid_shp)
    raw_gdf["geometry"] = raw_gdf.geometry.centroid
    raw_gdf = raw_gdf.to_crs(epsg=4326)
    # endregion

    # region --- Internal helpers ---
    def _init_fig(figsize, clean:bool):
        if clean:
            fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        else: 
            fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    def _savefig(fig, png_path, clean, show):
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        if clean:
            fig.savefig(png_path, dpi=300)
        else:
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _simplify_flood_contour(shp, clip_bounds=None):
        if clip_bounds is not None:
            clip_box = box(*clip_bounds)
            shp = shp.clip(clip_box)
        shp = shp.explode(index_parts=False)
        shp.geometry = shp.geometry.apply(
            lambda poly: Polygon(poly.exterior) if poly.interiors else poly
        )
        shp = shp[shp.geometry.area > 1e-7]
        shp = shp.union_all()
        shp = shp.simplify(tolerance=0.00011, preserve_topology=True)
        shp = gpd.GeoDataFrame(geometry=[shp], crs="EPSG:4326")
        return shp

    def _tif_to_simplified_shp(tif_path, clip_bounds=None):
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
            
            shp = gpd.GeoDataFrame(records, crs="EPSG:25830")
            
            # 2. Now properly convert the meters to WGS84 degrees
            shp = shp.to_crs("EPSG:4326")
            
        return _simplify_flood_contour(shp, clip_bounds=clip_bounds)

    def _to_dms(val, pos):
        """Convert decimal degrees to standard DMS (º ' '') string."""
        sign = "-" if val < 0 else ""
        abs_val = abs(val)
        
        degrees = int(abs_val)
        minutes_float = (abs_val - degrees) * 60
        minutes = int(minutes_float)
        seconds = round((minutes_float - minutes) * 60)
        
        # Handle rollover from rounding seconds
        if seconds == 60:
            seconds = 0
            minutes += 1
        if minutes == 60:
            minutes = 0
            degrees += 1
            
        if seconds == 0:
            return f"{sign}{degrees}º{minutes:02d}'"
        return f"{sign}{degrees}º{minutes:02d}'{seconds:02d}''"
    # endregion

    # ------------------------------
    # region 2.1 CONVERGENCE (Figure 5)
    # ------------------------------
    # depth_Ev_by_BID
    res = results.calc_metric(
        res_name="depth_Ev_by_BID",
        res_description= """(external) HEC-RAS Monte Carlo average value evolution at global level, summing the FID and averaging the BID
        levels, calculatin comulative over iterations for each return period. Commonly used to visuallyze
        estability of monte carlo as more simulations are added""",
        target_col="he",
        levels={1: "RP", 2: "BID"},
        level = 1,
        level_calc_type={1: None, 2: "mean",},
        return_period_col="RP",
        evolution_col="it",
        custom_data = depht_samples_df,
    )
    df1=results.get_result_as_df(res_name="IRME_Ev_by_BID")
    df2=results.get_result_as_df(res_name="depth_Ev_by_BID")
    df2_fix = df2.join(df1.select(["RP", "it", "IRME", "CONV"]), on=["RP", "it"], how="left")
    results.add_result(
        res_name="depth_Ev_by_BID",
        res_description="""(external) HEC-RAS Monte Carlo average value evolution at global level, summing the FID and averaging the BID
        levels, calculatin comulative over iterations for each return period. Commonly used to visuallyze
        estability of monte carlo as more simulations are added""",
        data=df2_fix,
        overwrite=True,
    )

    # convergence (IRRSDFV)
    # plot
    print(results.get_result_as_df(res_name="convergence"))
    results.plot(
        res_name = "convergence",
        chart_type = "montecarlo_convergence_evolution",
        figsize = (3.5, 2),
        color_dic = color_dic,
        custom_style = custom_style,
        return_period = "RP",
        x_col = "it",
        y_col = "IRRSDFV",
        warm_up = 100,
        convergence_col_val = {"CONV":True},
        ax_properties={
            "constant":{
                "xscale": "log",
                "yscale": "log",
                "xlim": (2, 10000),
                "ylim": (1, 41000),
            },
        }
    )

    # cost_Ev_by_BID
    results.add_result(
        res_name="cost_Ev_by_BID",
        res_description="""Monte Carlo cost evolution (cumulative average cost per BID) for each return period over iterations""",
        data=(
            results.collect_data(columns=['it', 'RP', 'BID', 'FID', 'c_total'])
            .with_columns( # type: ignore
                pl.col("BID").n_unique().over("RP").alias("num_bids")
            )
            .group_by(["RP", "it", "num_bids"]) 
            .agg(pl.col("c_total").sum().alias("municipality_damage"))
            .with_columns(
                (pl.col("municipality_damage") / pl.col("num_bids")).alias("iteration_avg_damage")
            )
            .sort(["RP", "it"])
            .with_columns(
                (pl.col("iteration_avg_damage").cum_sum().over("RP") / pl.col("iteration_avg_damage").cum_count().over("RP"))
                .alias("cumulative_mean_cost")
            )
            .select(["RP", "it", "cumulative_mean_cost"])
        ),
        overwrite=True,
    )

    # Plot
    cost_Ev_by_BID = results.get_result_as_df(res_name="cost_Ev_by_BID")
    def cost_evolution_plot(data, figsize, clean: bool, show: bool, warm_up: int | None = 100):
        return_periods = sorted(data["RP"].unique().to_list())

        with plt.rc_context(custom_style): # type: ignore
            fig, ax = _init_fig(figsize, clean)

            for rp in return_periods:
                df_rp = data.filter(pl.col("RP") == rp)
                x = df_rp["it"].to_numpy()
                y = df_rp["cumulative_mean_cost"].to_numpy()

                rp_color = color_dic["RP"].get(rp, "gray")

                # PLOT Evolution Line for each RP
                ax.plot(
                    x,
                    y,
                    color=rp_color,
                    linestyle="-",
                    linewidth=1.2,
                    label=f"RP {rp}"
                )

            # Warm-up shading on the left side
            if warm_up is not None and warm_up > 0:
                ax.axvspan(1, warm_up, alpha=0.2, color="#cccccc", label="warm-up", zorder=0)

            # Use log scale for simulation number to match standard convergence layout
            ax.set_xscale("log")
            ax.set_xlim(left=2, right=12000)

            # STYLE
            if not clean:
                ax.set(
                    xlabel="Simulation Number",
                    ylabel="Average Spatial Damage (€/building)",
                    title="Cost Evolution Convergence",
                )
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.04, 1.0),
                    fontsize=8,
                    framealpha=0.9,
                    borderaxespad=0,
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[],
                )
                ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)

            # SAVE
            base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\convergence\F5_CostEvolution"
            png_path = (
                f"{base_path}.png"
                if not clean
                else f"{base_path}_clean.png"
            )
            _savefig(fig, png_path, clean, show)
    cost_evolution_plot(cost_Ev_by_BID, figsize=(2.8, 1.6), clean=False, show=False)
    cost_evolution_plot(cost_Ev_by_BID, figsize=(2.8, 1.6), clean=True, show=False)

    # Convergence: IRME
    print(results.get_result_as_df(res_name="IRME_Ev_by_BID"))
    results.plot(
        res_name = "IRME_Ev_by_BID",
        chart_type = "montecarlo_convergence_evolution",
        figsize = (3.5, 2),
        color_dic = color_dic,
        custom_style = custom_style,
        return_period = "RP",
        x_col = "it",
        y_col = "IRME",
        warm_up = 30,
        convergence_col_val = {"CONV":True},
        ax_properties={
            "constant":{
                "xscale": "log",
                "yscale": "log",
                "xlim": (2, 1296),
                "ylim": (1, 555),
            },
        }
    )

    # Convergence: Average Spatial Depht
    print(results.get_result_as_df(res_name="depth_Ev_by_BID"))
    results.plot(
        res_name = "depth_Ev_by_BID",
        chart_type = "montecarlo_convergence_evolution",
        figsize = (3.5, 2),
        color_dic = color_dic,
        custom_style = custom_style,
        return_period = "RP",
        x_col = "it",
        y_col = "he",
        warm_up = 30,
        convergence_col_val = {"CONV":True},
        ax_properties={
            "constant":{
                "xscale": "log",
                "yscale": {"value": "symlog", "linthresh": 0.01, "linscale": 0.5},
                "xlim": (2, 1296),
                "ylim": (0, 0.6),
            },
        }
    )
    # endregion

    # ------------------------------
    # region 2.2 DAMAGE MAP (Figure 6)
    # ------------------------------
    # --- LORENZ CURVE (BID) ---
    # lorenz_curve_cost_BID
    results.add_result(
        res_name="lorenz_curve_cost_BID",
        res_description="""Monte Carlo lorenz curve for total damage by bid for each return period""",
        data=(
            results.collect_data(columns = ['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it", "BID"]) # type: ignore
            .agg(pl.col("c_total").sum().alias("bid_damage"))
            .group_by(["RP", "BID"])
            .agg([
                pl.col("bid_damage").quantile(0.05).alias("q05"),
                pl.col("bid_damage").quantile(0.50).alias("q50"),
                pl.col("bid_damage").quantile(0.95).alias("q95"),
            ])
            .unpivot(
                index=["RP", "BID"],
                on=["q05", "q50", "q95"],
                variable_name="quantile",
                value_name="damage",
            )
            .sort(["RP", "quantile", "damage"])
            .with_columns(
                pl.col("damage").sum().over(["RP", "quantile"]).alias("total_damage"),
                (
                    pl.int_range(1, pl.len() + 1).over(["RP", "quantile"])
                    / pl.len().over(["RP", "quantile"])
                ).alias("lorenz_x"),
            )
            .with_columns(
                (pl.col("damage").cum_sum().over(["RP", "quantile"]) / pl.col("total_damage"))
                .fill_nan(0.0)
                .alias("lorenz_y")
            )
            .select(["RP", "quantile", "BID", "damage", "total_damage", "lorenz_x", "lorenz_y"])
            .sort(["RP", "quantile", "lorenz_y"])
        ),
        overwrite=True,
    )

    # Plot
    lorenz_curve_cost_BID = results.get_result_as_df(res_name = "lorenz_curve_cost_BID")
    def lorenz_curve_plot(data, figsize, clean: bool, show: bool):
        # Get source data from results object
        return_periods = data["RP"].unique().to_list()

        with plt.rc_context(custom_style):  # type: ignore
            for rp in return_periods:
                # FIG
                fig, ax = _init_fig(figsize, clean)

                # Filter dataset for the specific RP
                df_rp = data.filter(pl.col("RP") == rp)

                # Extract x and y coordinates per quantile
                x = df_rp.filter(pl.col("quantile") == "q50")["lorenz_x"].to_numpy()
                q05_y = df_rp.filter(pl.col("quantile") == "q05")["lorenz_y"].to_numpy()
                q50_y = df_rp.filter(pl.col("quantile") == "q50")["lorenz_y"].to_numpy()
                q95_y = df_rp.filter(pl.col("quantile") == "q95")["lorenz_y"].to_numpy()

                # PLOT
                # 1. 1:1 Equality line (Diagonal)
                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    label="Equality",
                )

                # 2. Lorenz Curve Confidence Interval (Q05 to Q95)
                rp_color = color_dic["RP"].get(rp, "gray")
                ax.fill_between(
                    x,
                    q05_y,
                    q95_y,
                    color=rp_color,
                    alpha=0.35,
                    edgecolor="none",
                    label="Q05-Q95",
                )

                # 3. Lorenz Curve Median (Q50)
                ax.plot(
                    x,
                    q50_y,
                    color=rp_color,
                    linestyle="-",
                    linewidth=1.5,
                    label="Q50",
                )

                # STYLE
                ax.set(
                    xlim=(0, 1),
                    ylim=(0, 1),
                    aspect="equal",
                )

                if not clean:
                    ax.set(
                        xlabel="Cumulative Share of BIDs",
                        ylabel="Cumulative Share of Damage",
                        title=f"Lorenz Curve (RP {rp})",
                    )
                else:
                    ax.set(
                        xticklabels=[],
                        yticklabels=[],
                    )

                # SAVE
                base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_Lorenz"
                png_path = (
                    f"{base_path}_RP{rp}.png"
                    if not clean
                    else f"{base_path}_RP{rp}_clean.png"
                )
                _savefig(fig, png_path, clean, show)
    lorenz_curve_plot(lorenz_curve_cost_BID, figsize=(1.8, 1.8), clean=False, show=False)
    lorenz_curve_plot(lorenz_curve_cost_BID, figsize=(1.8, 1.8), clean=True, show=False)

    # --- GINI COEFFICIENT (BID) ---
    # gini_coefficient_cost_BID
    results.add_result(
        res_name="gini_coefficient_cost_BID",
        res_description="""Monte Carlo gini coefficient for total damage by bid for each return period""",
        data=(
            lorenz_curve_cost_BID
            .sort(["RP", "quantile", "lorenz_x"]) # type: ignore
            .with_columns(
                # Previous x and y points within each (RP, quantile) group
                pl.col("lorenz_x").shift(1).fill_null(0.0).over(["RP", "quantile"]).alias("x_prev"),
                pl.col("lorenz_y").shift(1).fill_null(0.0).over(["RP", "quantile"]).alias("y_prev"),
            )
            .with_columns(
                # Trapezoidal area under the segment: (x_i - x_{i-1}) * (y_i + y_{i-1}) / 2
                (
                    (pl.col("lorenz_x") - pl.col("x_prev")) 
                    * (pl.col("lorenz_y") + pl.col("y_prev")) 
                    / 2.0
                ).alias("trapezoid_area")
            )
            .group_by(["RP", "quantile"])
            .agg([
                # Gini = 1 - 2 * (Area under Lorenz Curve)
                (1.0 - 2.0 * pl.col("trapezoid_area").sum())
                .fill_nan(0.0)
                .alias("gini_coefficient")
            ])
            .sort(["RP", "quantile"])
        ),
        overwrite=True,
    )

    # Plot
    gini_coefficient_cost_BID = results.get_result_as_df(res_name = "gini_coefficient_cost_BID")
    def gini_plot(data, figsize, clean: bool, show: bool):
        # Pivot quantile values side-by-side per RP
        df_gini = data.pivot(
            on="quantile",
            index="RP",
            values="gini_coefficient"
        ).sort("RP")

        return_periods = df_gini["RP"].to_list()

        with plt.rc_context(custom_style):  # type: ignore
            for rp in return_periods:
                # FIG
                fig, ax = _init_fig(figsize, clean)

                # Filter row for current RP
                df_rp = df_gini.filter(pl.col("RP") == rp)

                q05_val = df_rp["q05"].to_numpy()[0]
                q50_val = df_rp["q50"].to_numpy()[0]
                q95_val = df_rp["q95"].to_numpy()[0]

                # PLOT
                rp_color = color_dic["RP"].get(rp, "gray")
                
                # Single bar position centered at x = 0
                x_center = 0.0
                half_width = 0.35

                # 1. Fill range between Q05 and Q95
                y_min, y_max = min(q05_val, q95_val), max(q05_val, q95_val)
                ax.fill_between(
                    [x_center - half_width, x_center + half_width],
                    y_min,
                    y_max,
                    color=rp_color,
                    alpha=0.35,
                    edgecolor="none",
                    label="Q05-Q95"
                )

                # 2. Horizontal line for Q50 median
                ax.hlines(
                    y=q50_val,
                    xmin=x_center - half_width,
                    xmax=x_center + half_width,
                    color=rp_color,
                    linewidth=3.0,
                    label="Q50"
                )

                # STYLE
                ax.set(
                    xlim=(-0.5, 0.5),
                    ylim=(0, 1),
                    xticks=[x_center],
                    xticklabels=[f"RP {rp}"],
                )

                if not clean:
                    ax.set(
                        ylabel="Gini Coefficient",
                        title=f"Gini Coefficient (RP {rp})",
                    )
                else:
                    ax.set(
                        xticklabels=[],
                        yticklabels=[],
                    )
                
                # SAVE
                base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_Gini"
                png_path = (
                    f"{base_path}_RP{rp}.png"
                    if not clean
                    else f"{base_path}_RP{rp}_clean.png"
                )
                _savefig(fig, png_path, clean, show)
    gini_plot(gini_coefficient_cost_BID, figsize=(0.7, 1.8), clean=False, show=False)
    gini_plot(gini_coefficient_cost_BID, figsize=(0.7, 1.8), clean=True, show=False)

    # --- TOTAL DAMAGE DISTRIBUTION (BID STO) ---
    # total_damage_cost_BID
    results.add_result(
        res_name="total_damage_cost_BID",
        res_description="""Monte Carlo lorenz curve for total damage by bid for each return period""",
        data=(
            results.collect_data(columns = ['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it", "BID"]) # type: ignore
            .agg(pl.col("c_total").sum().alias("bid_damage"))
            .group_by(["RP", "BID"])
            .agg([
                pl.col("bid_damage").quantile(0.05).alias("q05"),
                pl.col("bid_damage").quantile(0.50).alias("q50"),
                pl.col("bid_damage").quantile(0.95).alias("q95"),
            ])
            .sort(["RP"])
        ),
        overwrite=True,
    )

    # Plot
    total_damage_cost_BID = results.get_result_as_df(res_name = "total_damage_cost_BID")
    def building_damage_quantile_plot(data, figsize, clean: bool, show: bool):
        # Get all unique return periods ordered
        return_periods = data["RP"].unique().sort().to_list()

        # Dynamic global Y-limit
        positive_damages = data.select(["q05", "q50", "q95"]).to_numpy().flatten()
        positive_damages = positive_damages[positive_damages > 0]

        if len(positive_damages) > 0:
            global_max = positive_damages.max()
            y_lim_max = 10 ** (np.ceil(np.log10(global_max)))
        else:
            y_lim_max = 10**5.5
        
        # Dynamic color pallete
        base_cmap = plt.get_cmap("YlOrRd")
        colors = base_cmap(np.linspace(0, 1, 256)) ** 2.1
        darkened_YlOrRd = mcolors.LinearSegmentedColormap.from_list("DarkYlOrRd", colors)

        # Plot
        quantiles = ["Q05", "Q50", "Q95"]
        x_positions = [0, 1, 2]
        max_jitter = 0.35
        
        with plt.rc_context(custom_style):  # type: ignore
            for rp in return_periods:
                # FIG
                fig, ax = _init_fig(figsize, clean)

                # Filter data for current RP and unpivot to long format
                df_rp = (
                    data.filter(pl.col("RP") == rp)
                    .unpivot(
                        index=["RP", "BID"],
                        on=["q05", "q50", "q95"],
                        variable_name="quantile",
                        value_name="damage",
                    )
                    .with_columns(
                        pl.col("quantile").str.to_uppercase().alias("quantile")
                    )
                )
                
                # Define color
                q50_vals = df_rp.filter(
                    (pl.col("quantile") == "Q50") & (pl.col("damage") > 0)
                )["damage"].to_numpy()
                
                if len(q50_vals) > 0:
                    q50_log = np.log10(q50_vals)
                    vmin = float(np.percentile(q50_log, 5))
                    vcenter = float(np.percentile(q50_log, 50))
                    vmax = float(np.percentile(q50_log, 95))
                    
                    if vmin >= vcenter:
                        vmin = vcenter - 0.1
                    if vmax <= vcenter:
                        vmax = vcenter + 0.1
                else:
                    vmin, vcenter, vmax = 0, 0.5, 1
                
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                                
                # PLOT each quantile column (Q05, Q50, Q95)
                for i, q in enumerate(quantiles):
                    # Filter positive damage values for log scale
                    damage_vals = df_rp.filter(
                        (pl.col("quantile") == q) & (pl.col("damage") > 0)
                    )["damage"].to_numpy()

                    if len(damage_vals) == 0:
                        continue

                    log_vals = np.log10(damage_vals)

                    # Calculate density along log damage axis using histogram bins
                    counts, bin_edges = np.histogram(log_vals, bins=35)
                    bin_indices = np.digitize(log_vals, bin_edges[:-1]) - 1
                    bin_indices = np.clip(bin_indices, 0, len(counts) - 1)

                    # Scale jitter width proportionally to density
                    densities = counts[bin_indices]
                    jitter_widths = (
                        np.sqrt(densities / densities.max()) * max_jitter
                    )

                    # Generate random offset based on density
                    x_offsets = np.random.uniform(-jitter_widths, jitter_widths)
                    x_coords = x_positions[i] + x_offsets

                    # Scatter plot
                    ax.scatter(
                        x_coords,
                        damage_vals,
                        c=log_vals,
                        cmap=darkened_YlOrRd,
                        s=6,
                        alpha=0.75,
                        edgecolors="none",
                        norm=norm,
                    )

                # STYLE
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
                ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)) # type: ignore
                ax.set(
                    xticks=x_positions,
                    ylim=(10, y_lim_max),
                )
                ax.tick_params(
                    axis="x",
                    which="minor",
                    bottom=False,
                    top=False,
                )

                if not clean:
                    ax.set(
                        xticklabels=quantiles,
                        xlabel="Building Damage Quantile",
                        ylabel="Building Damage (€)",
                        title=f"Building Damage Quantiles (RP {rp})",
                    )
                    
                    sm = cm.ScalarMappable(cmap=darkened_YlOrRd, norm=norm)
                    sm.set_array([])  # Dummy array for scalar mappable
                    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                    cbar_ticks = [vmin, vcenter, vmax]
                    cbar.set_ticks(cbar_ticks)
                    cbar.set_ticklabels([f"€10$^{{{t:.1f}}}$" for t in cbar_ticks])
                    cbar.set_label("Damage Intensity (Log10 €)", rotation=270, labelpad=15)
                    
                else:
                    ax.set(
                        xticklabels=[],
                        yticklabels=[],
                    )

                # SAVE
                base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_QuantileDist"
                png_path = (
                    f"{base_path}_RP{rp}.png"
                    if not clean
                    else f"{base_path}_RP{rp}_clean.png"
                )
                _savefig(fig, png_path, clean, show)
    building_damage_quantile_plot(total_damage_cost_BID, figsize=(1.8, 1.8), clean=False, show=False)
    building_damage_quantile_plot(total_damage_cost_BID, figsize=(1.8, 1.8), clean=True, show=False)

    # --- TOTAL DAMAGE MAP COMPARISION (STO vs DET) ---
    # Data
    total_damage_cost_BID = results.get_result_as_df(res_name = "total_damage_cost_BID")
    total_damage_cost_BID_pandas = (
        total_damage_cost_BID.to_pandas()
        .pivot(index="BID", columns="RP", values="q50")
        .add_prefix("q50_RP_")
        .rename_axis(columns=None)[["q50_RP_10", "q50_RP_50", "q50_RP_100", "q50_RP_500"]]
        .reset_index()
    )
    gdf = raw_gdf.merge(
        total_damage_cost_BID_pandas,
        on="BID",
        how="inner"
    )

    # Plot
    def damage_map_plot(gdf_data, figsize, clean: bool, show: bool):
        # Get all unique return periods ordered
        return_periods = [500]
        
        # WMS request
        wms_url = "https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS"
        wms = WebMapService(wms_url, version="1.1.1")
        xmin, ymin, xmax, ymax = gdf_data.total_bounds
        x_pad = (xmax - xmin) * 0.05 if xmax != xmin else 0.01
        y_pad = (ymax - ymin) * 0.05 if ymax != ymin else 0.01
        xmin, xmax = xmin - x_pad, xmax + x_pad
        ymin, ymax = ymin - y_pad, ymax + y_pad
        extent = [xmin, xmax, ymin, ymax]
        bbox = (extent[0], extent[2], extent[1], extent[3])
        img_request = wms.getmap(
            layers=["OI.OrthoimageCoverage"],
            srs="EPSG:4326",
            bbox=bbox,
            size=(1000, 1000),
            format="image/png",
            transparent=True
        )
        img_data = mpimg.imread(BytesIO(img_request.read()))
        
        base_cmap = plt.get_cmap("YlOrRd")
        colors = base_cmap(np.linspace(0, 1, 256)) ** 2.1
        darkened_YlOrRd = mcolors.LinearSegmentedColormap.from_list("DarkYlOrRd", colors)
        
        clip_xmin, clip_ymin, clip_xmax, clip_ymax = gdf_data.total_bounds
        print(clip_xmin, clip_ymin, clip_xmax, clip_ymax)
        chart_style = {
            **custom_style,
            "xtick.top": True,
            "xtick.bottom": False,
            "xtick.labeltop": True,
            "xtick.labelbottom": False,
        }
        with plt.rc_context(chart_style):  # type: ignore
            for rp in return_periods:
                # FIG
                fig, ax = _init_fig(figsize, clean)

                # Filter data
                rp_col = f"q50_RP_{rp}"
                gdf_valid = gdf_data.dropna(subset=[rp_col])
                gdf_zero = gdf_valid[gdf_valid[rp_col] == 0]
                gdf_pos = gdf_valid[gdf_valid[rp_col] > 0].copy()
                gdf_pos["q50_log"] = np.log10(gdf_pos[rp_col])

                # PLOT
                # SHP (Damage)
                q50_log_pos = gdf_pos["q50_log"].to_numpy()
                vmin = float(np.percentile(q50_log_pos, 5))
                vcenter = float(np.median(q50_log_pos))
                vmax = float(np.percentile(q50_log_pos, 95))
                if vmin >= vcenter:
                    vmin = vcenter - 0.1
                if vmax <= vcenter:
                    vmax = vcenter + 0.1
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

                if not gdf_zero.empty:
                    gdf_zero.plot(
                        ax=ax,
                        color="white",
                        edgecolor="none",
                        linewidth=0,
                        markersize=6,
                        zorder=2,
                    )

                if not gdf_pos.empty:
                    gdf_pos.plot(
                        column="q50_log",
                        cmap=darkened_YlOrRd,
                        norm=norm,
                        markersize=9,
                        legend=False,
                        ax=ax,
                        edgecolor="none",
                        linewidth=0,
                        zorder=3,
                    )
                
                # WMS
                ax.imshow(
                    img_data, 
                    extent=extent, # type: ignore
                    origin="upper", 
                    alpha=1.0, 
                    zorder=0
                )
                
                # Q50 depth
                tif_path = STO_TIF[rp]["Q50"]
                dst_crs = "EPSG:4326"
                with rasterio.open(tif_path) as src:
                    def_transform, _, _ = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds
                    )
                    res_x = abs(def_transform.a)
                    res_y = abs(def_transform.e)
                    width = max(1, int(np.ceil((clip_xmax - clip_xmin) / res_x)))
                    height = max(1, int(np.ceil((clip_ymax - clip_ymin) / res_y)))
                    dst_transform = from_bounds(clip_xmin, clip_ymin, clip_xmax, clip_ymax, width, height)
                    tif_extent = [clip_xmin, clip_xmax, clip_ymin, clip_ymax]
                    tif_data = np.zeros((height, width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=tif_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=np.nan
                    )
                    tif_data = np.ma.masked_invalid(tif_data)
                    tif_data = np.ma.masked_less_equal(tif_data, 0)

                ax.imshow(
                    tif_data,
                    extent=tif_extent, # type: ignore
                    origin="upper",
                    cmap="Blues",
                    alpha=0.6,
                    zorder=1
                )
                
                # DET Official Contour
                det_shp_path = DET_SHP[rp]
                raw_gdf = gpd.read_file(det_shp_path)
                if raw_gdf.crs is not None and raw_gdf.crs.to_string() != "EPSG:4326":
                    raw_gdf = raw_gdf.to_crs("EPSG:4326")
                gdf_contour = _simplify_flood_contour(raw_gdf, clip_bounds=(clip_xmin, clip_ymin, clip_xmax, clip_ymax))
                
                gdf_contour.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="#8400DB",
                    linewidth=1.3,
                    linestyle="-",
                    alpha=0.7,
                    zorder=4
                )
                
                # STO Contours
                sto_tif_path_q05 = STO_TIF[rp]["Q05"]
                sto_tif_path_q50 = STO_TIF[rp]["Q50"]
                sto_tif_path_q95 = STO_TIF[rp]["Q95"]
                bounds = (clip_xmin, clip_ymin, clip_xmax, clip_ymax)
                gdf_sto_contour_q05 = _tif_to_simplified_shp(sto_tif_path_q05, clip_bounds=bounds)
                gdf_sto_contour_q50 = _tif_to_simplified_shp(sto_tif_path_q50, clip_bounds=bounds)
                gdf_sto_contour_q95 = _tif_to_simplified_shp(sto_tif_path_q95, clip_bounds=bounds)
                gdf_sto_contour_q05.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="#003CFF",
                    linewidth=1.3,
                    linestyle=":",
                    alpha=0.7,
                    zorder=4
                )
                gdf_sto_contour_q50.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="#003CFF",
                    linewidth=1.3,
                    linestyle="-",
                    alpha=0.7,
                    zorder=4
                )
                gdf_sto_contour_q95.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="#003CFF",
                    linewidth=1.3,
                    linestyle="--",
                    alpha=0.7,
                    zorder=4
                )
        
                # STYLE
                plt.grid(True, linestyle="--", alpha=0.5)
                ax.set_xlim(clip_xmin, clip_xmax)
                ax.set_ylim(clip_ymin, clip_ymax)
                ax.set_aspect("equal", adjustable="box")
                
                if not clean:
                    title_str = (
                        f"BID Centroids - Damage Map (RP {rp})"
                        if rp is not None
                        else "BID Centroids (WGS 84 / EPSG:4326)"
                    )
                    ax.set(
                        xlabel="Longitude",
                        ylabel="Latitude",
                        title=title_str,
                    )
                    
                    # Colorbar
                    sm = cm.ScalarMappable(cmap=darkened_YlOrRd, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax)
                    cbar_ticks = [vmin, vcenter, vmax]
                    cbar.set_ticks(cbar_ticks)
                    cbar.set_ticklabels([f"€10$^{{{t:.1f}}}$" for t in cbar_ticks])
                    cbar.set_label("Damage Intensity (Log10 €)", rotation=270, labelpad=15)
                    
                    # Format labels
                    ax.xaxis.set_major_formatter(FuncFormatter(_to_dms))
                    ax.yaxis.set_major_formatter(FuncFormatter(_to_dms))
                    
                    # Graphic scale
                    center_lat = (ymin + ymax) / 2.0
                    dx_meters = 111_320 * np.cos(np.radians(center_lat))
                    scalebar = ScaleBar(
                        dx=dx_meters,
                        units="m",
                        dimension="si-length",
                        location="lower left",  # 'lower left', 'lower right', 'upper left', etc.
                        box_alpha=0.7,
                        box_color="white",
                        color="black",
                        scale_loc="bottom",
                        length_fraction=0.2,    # Scale bar will take ~20% of map width
                    )
                    ax.add_artist(scalebar)
                    
                else:
                    ax.set(
                        xticklabels=[],
                        yticklabels=[],
                    )
                
                # SAVE
                base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F6_DamageMap"
                rp_suffix = f"_RP{rp}" if rp is not None else ""
                clean_suffix = "_clean.png" if clean else ".png"
                png_path = f"{base_path}{rp_suffix}{clean_suffix}"

                _savefig(fig, png_path, clean, show)
    damage_map_plot(gdf, (6.5, 2.9), clean = False, show = False)
    damage_map_plot(gdf, (6.5, 2.9), clean = True, show = False)

    # --- TOTAL DAMAGE DISTRIBUTION (Municipality STO) ---
    # cost_Municipality_by_it
    results.add_result(
        res_name="cost_Municipality_by_it",
        res_description="""Monte Carlo raw values at municipality level for each return period""",
        data=(
            results.collect_data(columns = ['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it"]) # type: ignore
            .agg(pl.col("c_total").sum().alias("y"))
            .select(["RP", "y"])
        ),
        overwrite=True,
    )

    # Plot
    cost_Municipality_by_it = results.get_result_as_df(res_name = "cost_Municipality_by_it")
    def damage_plot_0(data, figsize, clean: bool, show: bool):
        # Unique Return Periods in order
        return_periods = sorted(data["RP"].unique().to_list(), reverse=True)
        
        with plt.rc_context(custom_style): # type: ignore
            for rp in return_periods:
                # FIG
                fig, ax = _init_fig(figsize, clean)
                
                # Extract damage values for current RP
                y_data = data.filter(pl.col("RP") == rp)["y"].to_numpy()
                
                # Compute percentiles directly from empirical data
                q05 = np.percentile(y_data, 5)
                q50 = np.percentile(y_data, 50)
                q95 = np.percentile(y_data, 95)
                
                # Fast Kernel Density Estimation for empirical violin shape
                qlower, qupper = np.percentile(y_data, [0.1, 99.9])
                y_clipped = y_data[(y_data >= qlower) & (y_data <= qupper)]
                
                def calculate_iqr_bandwidth(y_clipped):
                    n = len(y_clipped)
                    if n < 2:
                        return 0.2
                    q25, q75 = np.percentile(y_clipped, [25, 75])
                    iqr = q75 - q25
                    std_dev = np.std(y_clipped, ddof=1)
                    
                    # Robust standard deviation estimator using IQR
                    sigma_robust = min(std_dev, iqr / 1.34) if iqr > 0 else std_dev
                    
                    # Silverman's bandwidth formula
                    h = 0.9 * sigma_robust * (n ** (-1 / 5))
                    
                    # Convert h to bandwidth factor expected by scipy (h / std_dev)
                    bw_factor = h / std_dev if std_dev > 0 else 0.2
                    
                    return bw_factor * 2
                    
                bw_factor = calculate_iqr_bandwidth(y_clipped)
                kde = gaussian_kde(y_clipped, bw_method=bw_factor)
                val_grid = np.linspace(qlower, qupper, 300)
                pdf = kde(val_grid)
                
                # Normalize width
                pdf_norm = (pdf / pdf.max()) * 0.38
                x_center = 0
                
                # Fill density violin profile vertically
                ax.fill_betweenx(
                    val_grid,
                    x_center - pdf_norm,
                    x_center + pdf_norm,
                    color=color_dic["RP"].get(rp),
                    alpha=0.85,
                    edgecolor='none'
                )
                
                # Plot Horizontal Quantile Lines
                ax.axhline(q95, color='black', linestyle='--', linewidth=1, label='0.95')
                ax.axhline(q50, color='black', linestyle='-', linewidth=1, label='0.50')
                ax.axhline(q05, color='black', linestyle=':', linewidth=1, label='0.05')
                
                # STYLE
                ax.set(
                    xticks=[x_center],
                    xticklabels=[str(rp)],
                    ylim=(q05 * 0.95, q95 * 1.05)
                )
                if not clean:
                    ax.set(
                        xlabel='Return Period',
                        ylabel='Total (€)',
                        title=f'Damage (€) - RP {rp}',
                    )
                else:
                    ax.set(
                        xticklabels=[],
                        yticklabels=[]
                    )
                    ax.tick_params(axis='x', which='minor', bottom=False)
                
                # SAVE per Return Period
                base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F6_TotalDamage"
                png_path = f"{base_path}_RP_{rp}"
                png_path = png_path + f".png" if clean else png_path + f"_clean.png"
                _savefig(fig, png_path, clean, show)
                plt.close(fig)
    damage_plot_0(cost_Municipality_by_it, (0.7, 1.8), clean = False, show = False)
    damage_plot_0(cost_Municipality_by_it, (0.7, 1.8), clean = True, show = False)

    # endregion

    # ------------------------------
    # region 2.3 DAMAGE EVOLUTION (Figure 7)
    # ------------------------------
    # --- HAZARD ---
    # Plot
    df_if_fit = results.get_result_as_df(res_name = "input_flow_fit")
    def hazard_plot(figsize, clean:bool, show:bool):
        # Get source
        return_periods = df_if_fit["RP"].to_list()
        skews = df_if_fit["Skew"].to_numpy()
        locs = df_if_fit["Loc"].to_numpy()
        scales = df_if_fit["Scale"].to_numpy()
        
        # X
        x_pos = np.arange(len(return_periods))
        
        # Y
        q05 = 10 ** pearson3.ppf(0.05, skew=skews, loc=locs, scale=scales)
        q50 = 10 ** pearson3.ppf(0.50, skew=skews, loc=locs, scale=scales)
        q95 = 10 ** pearson3.ppf(0.95, skew=skews, loc=locs, scale=scales)
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            for i, rp in enumerate(return_periods):
                p_grid = np.linspace(0.01, 0.99, 300)
                val_grid = 10 ** pearson3.ppf(p_grid, skew=skews[i], loc=locs[i], scale=scales[i])
                
                pdf_log = pearson3.pdf(np.log10(val_grid), skew=skews[i], loc=locs[i], scale=scales[i])
                pdf_norm = (pdf_log / pdf_log.max()) * 0.38
                
                ax.fill_betweenx(
                    val_grid,
                    x_pos[i] - pdf_norm,
                    x_pos[i] + pdf_norm,
                    color=color_dic["RP"].get(rp),
                    alpha=0.85,
                    edgecolor='none'
                )

            ax.plot(x_pos, q95, color='black', linestyle='--', linewidth=1, label='0.95')
            ax.plot(x_pos, q50, color='black', linestyle='-', linewidth=1, label='0.50')
            ax.plot(x_pos, q05, color='black', linestyle=':', linewidth=1, label='0.05')
            
            # STYLE
            ax.set(
                xticks=x_pos,
                xticklabels=return_periods,
                ylim=(0, 10000)
            )
            if not clean:
                ax.set(
                    xlabel='Return Period',
                    ylabel='Flow ($m^3/s$)',
                    title='Hazard',
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_Hazard"
            png_path = png_path + f".png" if clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    hazard_plot(figsize=(1.6, 3.4), clean=False, show=False)
    hazard_plot(figsize=(1.6, 3.4), clean=True, show=False)

    # --- DAMAGE (Total) ---
    # Plot
    cost_Municipality_by_it = results.get_result_as_df(res_name = "cost_Municipality_by_it")
    def damage_plot_1(data, figsize, clean: bool, show: bool):
        # Unique Return Periods in order
        return_periods = sorted(data["RP"].unique().to_list(), reverse=True)
        x_pos = np.arange(len(return_periods))
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            q05, q50, q95 = [], [], []
            for i, rp in enumerate(return_periods):
                # Extract damage values for current RP
                y_data = data.filter(pl.col("RP") == rp)["y"].to_numpy()
                
                # Compute percentiles directly from empirical data
                q05.append(np.percentile(y_data, 5))
                q50.append(np.percentile(y_data, 50))
                q95.append(np.percentile(y_data, 95))
                
                # Fast Kernel Density Estimation for empirical violin shape
                qlower, qupper = np.percentile(y_data, [0.1, 99.9])
                y_clipped = y_data[(y_data >= qlower) & (y_data <= qupper)]
                def calculate_iqr_bandwidth(y_clipped):
                    n = len(y_clipped)
                    q25, q75 = np.percentile(y_clipped, [25, 75])
                    iqr = q75 - q25
                    std_dev = np.std(y_clipped, ddof=1)
                    
                    # Robust standard deviation estimator using IQR
                    sigma_robust = min(std_dev, iqr / 1.34) if iqr > 0 else std_dev
                    
                    # Silverman's bandwidth formula
                    h = 0.9 * sigma_robust * (n ** (-1 / 5))
                    
                    # Convert h to bandwidth factor expected by scipy (h / std_dev)
                    bw_factor = h / std_dev if std_dev > 0 else 0.2
                    
                    return bw_factor * 3
                bw_factor = calculate_iqr_bandwidth(y_clipped)
                kde = gaussian_kde(y_clipped, bw_method=bw_factor)
                val_grid = np.linspace(qlower, qupper, 300)
                pdf = kde(val_grid)
                
                # Normalize width
                pdf_norm = (pdf / pdf.max()) * 0.38
                
                # Fill density violin profile
                ax.fill_betweenx(
                    val_grid,
                    x_pos[i] - pdf_norm,
                    x_pos[i] + pdf_norm,
                    color=color_dic["RP"].get(rp),
                    alpha=0.85,
                    edgecolor='none'
                )

            # Plot Quantile Lines
            ax.plot(x_pos, q95, color='black', linestyle='--', linewidth=1, label='0.95')
            ax.plot(x_pos, q50, color='black', linestyle='-', linewidth=1, label='0.50')
            ax.plot(x_pos, q05, color='black', linestyle=':', linewidth=1, label='0.05')
            
            # STYLE
            ax.set(
                xticks=x_pos,
                xticklabels=return_periods,
                ylim=(0, max(q95) * 1.01)
            )
            if not clean:
                ax.set(
                    xlabel='Return Period',
                    ylabel='Total (€)',
                    title='Damage (€)',
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_TotalDamage"
            png_path = png_path + f".png" if clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    damage_plot_1(cost_Municipality_by_it, figsize=(1.6, 1.6), clean=False, show=False)
    damage_plot_1(cost_Municipality_by_it, figsize=(1.6, 1.6), clean=True, show=False)

    # --- DAMAGE (By building) ---
    # cost_by_BID_Municipality_by_it
    results.add_result(
        res_name="cost_by_BID_Municipality_by_it",
        res_description="""Monte Carlo average (by BID) raw values at municipality for each return period""",
        data=(
            results.collect_data(columns = ['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it"]) # type: ignore
            .agg((pl.col("c_total").sum() / pl.col("BID").n_unique()).alias("y"))
            .select(["RP", "y"])
        ),
        overwrite=True,
    )

    # Plot
    cost_by_BID_Municipality_by_it = results.get_result_as_df(res_name = "cost_by_BID_Municipality_by_it")
    def damage_plot_2(data, figsize, clean: bool, show: bool):
        # Unique Return Periods in order
        return_periods = sorted(data["RP"].unique().to_list(), reverse=True)
        x_pos = np.arange(len(return_periods))
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            q05, q50, q95 = [], [], []
            for i, rp in enumerate(return_periods):
                # Extract damage values for current RP
                y_data = data.filter(pl.col("RP") == rp)["y"].to_numpy()
                
                # Compute percentiles directly from empirical data
                q05.append(np.percentile(y_data, 5))
                q50.append(np.percentile(y_data, 50))
                q95.append(np.percentile(y_data, 95))
                
                # Fast Kernel Density Estimation for empirical violin shape
                qlower, qupper = np.percentile(y_data, [0.5, 99.5])
                y_clipped = y_data[(y_data >= qlower) & (y_data <= qupper)]
                def calculate_iqr_bandwidth(y_clipped):
                    n = len(y_clipped)
                    q25, q75 = np.percentile(y_clipped, [25, 75])
                    iqr = q75 - q25
                    std_dev = np.std(y_clipped, ddof=1)
                    
                    # Robust standard deviation estimator using IQR
                    sigma_robust = min(std_dev, iqr / 1.34) if iqr > 0 else std_dev
                    
                    # Silverman's bandwidth formula
                    h = 0.9 * sigma_robust * (n ** (-1 / 5))
                    
                    # Convert h to bandwidth factor expected by scipy (h / std_dev)
                    bw_factor = h / std_dev if std_dev > 0 else 0.2
                    
                    return bw_factor * 4
                bw_factor = calculate_iqr_bandwidth(y_clipped)
                kde = gaussian_kde(y_clipped, bw_method=bw_factor)
                val_grid = np.linspace(qlower, qupper, 300)
                pdf = kde(val_grid)
                
                # Normalize width
                pdf_norm = (pdf / pdf.max()) * 0.38
                
                # Fill density violin profile
                ax.fill_betweenx(
                    val_grid,
                    x_pos[i] - pdf_norm,
                    x_pos[i] + pdf_norm,
                    color=color_dic["RP"].get(rp),
                    alpha=0.85,
                    edgecolor='none'
                )

            # Plot Quantile Lines
            ax.plot(x_pos, q95, color='black', linestyle='--', linewidth=1, label='0.95')
            ax.plot(x_pos, q50, color='black', linestyle='-', linewidth=1, label='0.50')
            ax.plot(x_pos, q05, color='black', linestyle=':', linewidth=1, label='0.05')
            
            # STYLE
            ax.set(
                xticks=x_pos,
                xticklabels=return_periods,
                ylim=(0, max(q95) * 1.5)
            )
            if not clean:
                ax.set(
                    xlabel='Return Period',
                    ylabel='Total (€)',
                    title='Damage (€)',
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_ByBIDDamage"
            png_path = png_path + f".png" if clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    damage_plot_2(cost_by_BID_Municipality_by_it, figsize=(1.6, 1.6), clean=False, show=False)
    damage_plot_2(cost_by_BID_Municipality_by_it, figsize=(1.6, 1.6), clean=True, show=False)

    # --- Expected DAMAGE (Total) ---
    # expected_cost_Municipality_by_it
    results.add_result(
        res_name="expected_cost_Municipality_by_it",
        res_description="""Monte Carlo raw expected values at municipality level for each return period""",
        data=(
            results.collect_data(columns = ['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it"]) # type: ignore
            .agg(pl.col("c_total").sum().alias("total_damage"))
            .with_columns((pl.col("total_damage") / pl.col("RP")).alias("y"))
            .select(["RP", "y"])
        ),
        overwrite=True,
    )

    # Plot
    expected_cost_Municipality_by_it = results.get_result_as_df(res_name = "expected_cost_Municipality_by_it")
    def damage_plot_3(data, figsize, clean: bool, show: bool):
        # Unique Return Periods in order
        return_periods = sorted(data["RP"].unique().to_list(), reverse=True)
        x_pos = np.arange(len(return_periods))
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            q05, q50, q95 = [], [], []
            for i, rp in enumerate(return_periods):
                # Extract damage values for current RP
                y_data = data.filter(pl.col("RP") == rp)["y"].to_numpy()
                
                # Compute percentiles directly from empirical data
                q05.append(np.percentile(y_data, 5))
                q50.append(np.percentile(y_data, 50))
                q95.append(np.percentile(y_data, 95))
                
                # Fast Kernel Density Estimation for empirical violin shape
                qlower, qupper = np.percentile(y_data, [0.5, 99.5])
                y_clipped = y_data[(y_data >= qlower) & (y_data <= qupper)]
                def calculate_iqr_bandwidth(y_clipped):
                    n = len(y_clipped)
                    q25, q75 = np.percentile(y_clipped, [25, 75])
                    iqr = q75 - q25
                    std_dev = np.std(y_clipped, ddof=1)
                    
                    # Robust standard deviation estimator using IQR
                    sigma_robust = min(std_dev, iqr / 1.34) if iqr > 0 else std_dev
                    
                    # Silverman's bandwidth formula
                    h = 0.9 * sigma_robust * (n ** (-1 / 5))
                    
                    # Convert h to bandwidth factor expected by scipy (h / std_dev)
                    bw_factor = h / std_dev if std_dev > 0 else 0.2
                    
                    return bw_factor * 6
                bw_factor = calculate_iqr_bandwidth(y_clipped)
                kde = gaussian_kde(y_clipped, bw_method=bw_factor)
                val_grid = np.linspace(qlower, qupper, 300)
                pdf = kde(val_grid)
                
                # Normalize width
                pdf_norm = (pdf / pdf.max()) * 0.38
                
                # Fill density violin profile
                ax.fill_betweenx(
                    val_grid,
                    x_pos[i] - pdf_norm,
                    x_pos[i] + pdf_norm,
                    color=color_dic["RP"].get(rp),
                    alpha=0.85,
                    edgecolor='none'
                )

            # Plot Quantile Lines
            ax.plot(x_pos, q95, color='black', linestyle='--', linewidth=1, label='0.95')
            ax.plot(x_pos, q50, color='black', linestyle='-', linewidth=1, label='0.50')
            ax.plot(x_pos, q05, color='black', linestyle=':', linewidth=1, label='0.05')
            
            # STYLE
            ax.set(
                xticks=x_pos,
                xticklabels=return_periods,
                ylim=(0, max(q95) * 1.5)
            )
            if not clean:
                ax.set(
                    xlabel='Return Period',
                    ylabel='Total (€)',
                    title='Damage (€)',
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_ExpectedDamage"
            png_path = png_path + f".png" if clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    damage_plot_3(expected_cost_Municipality_by_it, figsize=(1.6, 1.6), clean=False, show=False)
    damage_plot_3(expected_cost_Municipality_by_it, figsize=(1.6, 1.6), clean=True, show=False)

    # --- Expected DAMAGE (By BID) ---
    # expected_cost_by_BID_Municipality_by_it
    results.add_result(
        res_name="expected_cost_by_BID_Municipality_by_it",
        res_description="""Monte Carlo expected average (by BID) raw values at municipality for each return period""",
        data=(
            results.collect_data(columns = ['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it"]) # type: ignore
            .agg((pl.col("c_total").sum() / pl.col("BID").n_unique()).alias("total_average_damage"))
            .with_columns((pl.col("total_average_damage") / pl.col("RP")).alias("y"))
            .select(["RP", "y"])
        ),
        overwrite=True,
    )

    # Plot
    expected_cost_by_BID_Municipality_by_it = results.get_result_as_df(res_name = "expected_cost_by_BID_Municipality_by_it")
    def damage_plot_4(data, figsize, clean: bool, show: bool):
        # Unique Return Periods in order
        return_periods = sorted(data["RP"].unique().to_list(), reverse=True)
        x_pos = np.arange(len(return_periods))
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            q05, q50, q95 = [], [], []
            for i, rp in enumerate(return_periods):
                # Extract damage values for current RP
                y_data = data.filter(pl.col("RP") == rp)["y"].to_numpy()
                
                # Compute percentiles directly from empirical data
                q05.append(np.percentile(y_data, 5))
                q50.append(np.percentile(y_data, 50))
                q95.append(np.percentile(y_data, 95))
                
                # Fast Kernel Density Estimation for empirical violin shape
                qlower, qupper = np.percentile(y_data, [0.5, 99.5])
                y_clipped = y_data[(y_data >= qlower) & (y_data <= qupper)]
                def calculate_iqr_bandwidth(y_clipped):
                    n = len(y_clipped)
                    q25, q75 = np.percentile(y_clipped, [25, 75])
                    iqr = q75 - q25
                    std_dev = np.std(y_clipped, ddof=1)
                    
                    # Robust standard deviation estimator using IQR
                    sigma_robust = min(std_dev, iqr / 1.34) if iqr > 0 else std_dev
                    
                    # Silverman's bandwidth formula
                    h = 0.9 * sigma_robust * (n ** (-1 / 5))
                    
                    # Convert h to bandwidth factor expected by scipy (h / std_dev)
                    bw_factor = h / std_dev if std_dev > 0 else 0.2
                    
                    return bw_factor * 6
                bw_factor = calculate_iqr_bandwidth(y_clipped)
                kde = gaussian_kde(y_clipped, bw_method=bw_factor)
                val_grid = np.linspace(qlower, qupper, 300)
                pdf = kde(val_grid)
                
                # Normalize width
                pdf_norm = (pdf / pdf.max()) * 0.38
                
                # Fill density violin profile
                ax.fill_betweenx(
                    val_grid,
                    x_pos[i] - pdf_norm,
                    x_pos[i] + pdf_norm,
                    color=color_dic["RP"].get(rp),
                    alpha=0.85,
                    edgecolor='none'
                )

            # Plot Quantile Lines
            ax.plot(x_pos, q95, color='black', linestyle='--', linewidth=1, label='0.95')
            ax.plot(x_pos, q50, color='black', linestyle='-', linewidth=1, label='0.50')
            ax.plot(x_pos, q05, color='black', linestyle=':', linewidth=1, label='0.05')
            
            # STYLE
            ax.set(
                xticks=x_pos,
                xticklabels=return_periods,
                ylim=(0, max(q95) * 1.5)
            )
            if not clean:
                ax.set(
                    xlabel='Return Period',
                    ylabel='Total (€)',
                    title='Damage (€)',
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_ExpectedByBIDDamage"
            png_path = png_path + f".png" if clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    damage_plot_4(expected_cost_by_BID_Municipality_by_it, figsize=(1.6, 1.6), clean=False, show=False)
    damage_plot_4(expected_cost_by_BID_Municipality_by_it, figsize=(1.6, 1.6), clean=True, show=False)

    # --- Expected Annual DAMAGE (EAD Total) ---
    # expected_annual_damage_Municipality
    results.add_result(
        res_name="expected_annual_damage_Municipality",
        res_description="""Monte Carlo expected annual damage at municipality for each return period""",
        data=(
            results.collect_data(columns=['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it"]) # type: ignore
            .agg(pl.col("c_total").sum().alias("total_damage"))
            .group_by("RP")
            .agg([
                pl.col("total_damage").quantile(0.05).alias("q05_damage"),
                pl.col("total_damage").mean().alias("q50_damage"),
                pl.col("total_damage").quantile(0.95).alias("q95_damage"),
            ])
            .extend(
                pl.DataFrame(
                    {
                        "RP": [2],
                        "q05_damage": [0.0],
                        "q50_damage": [0.0],
                        "q95_damage": [0.0],
                    }
                )
            )
            .sort("RP")
            .with_columns(p=1 / pl.col("RP"))
            .select(
                [
                    (
                        (pl.col(col) + pl.col(col).shift(-1))
                        / 2
                        * (pl.col("p") - pl.col("p").shift(-1))
                    )
                    .sum()
                    .alias(f"ead_{col}")
                    for col in ["q05_damage", "q50_damage", "q95_damage"]
                ]
            )
        ),
        overwrite=True,
    )
    expected_annual_damage_Municipality = results.get_result_as_df(res_name = "expected_annual_damage_Municipality")

    # --- Expected Annual DAMAGE (EAD by BID) ---
    # expected_annual_damage_by_BID_Municipality
    results.add_result(
        res_name="expected_annual_damage_by_BID_Municipality",
        res_description="""Monte Carlo expected annual damage at municipality (by BID) for each return period""",
        data=(
            results.collect_data(columns=['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it"]) # type: ignore
            .agg((pl.col("c_total").sum() / pl.col("BID").n_unique()).alias("total_damage"))
            .group_by("RP")
            .agg([
                pl.col("total_damage").quantile(0.05).alias("q05_damage"),
                pl.col("total_damage").mean().alias("q50_damage"),
                pl.col("total_damage").quantile(0.95).alias("q95_damage"),
            ])
            .extend(
                pl.DataFrame(
                    {
                        "RP": [2],
                        "q05_damage": [0.0],
                        "q50_damage": [0.0],
                        "q95_damage": [0.0],
                    }
                )
            )
            .sort("RP")
            .with_columns(p=1 / pl.col("RP"))
            .select(
                [
                    (
                        (pl.col(col) + pl.col(col).shift(-1))
                        / 2
                        * (pl.col("p") - pl.col("p").shift(-1))
                    )
                    .sum()
                    .alias(f"ead_{col}")
                    for col in ["q05_damage", "q50_damage", "q95_damage"]
                ]
            )
        ),
        overwrite=True,
    )
    expected_annual_damage_by_BID_Municipality = results.get_result_as_df(res_name = "expected_annual_damage_by_BID_Municipality")

    # endregion

    # ------------------------------
    # region 2.4 DAMAGE FUNCTION (Figure 8)
    # ------------------------------
    # --- % of Total Damage (up to first floor) ---
    # pct_hi_damage_up_to_first_floor
    results.add_result(
        res_name="pct_hi_damage_up_to_first_floor",
        res_description="""Estimated depth curve from model for first floor and bassement
        if exist considering internal water depth at first floor""",
        data=(
            results.collect_data(columns=['it', 'RP', 'BID', 'FID', 'hi', 'c_total', 'cM_total'])
            .filter(pl.col("FID").is_in([-1, 0])) # type: ignore [-1, 0]
            .group_by(["it", 'RP', "BID"])
            .agg([
                pl.col("c_total").sum().alias("c_sum"),
                pl.col("cM_total").sum().alias("cM_sum"),
                pl.col("hi").filter(pl.col("FID") == 0).first().alias("hi"),
            ])
            .with_columns(
                pl.when(pl.col("cM_sum") > 0)
                .then(pl.col("c_sum") / pl.col("cM_sum"))
                .otherwise(0.0)
                .alias("damage_pct")
            )
            .with_columns(
                ((pl.col("hi") / 0.1).floor() * 0.1 + 0.05).round(2).alias("hi_bin")
            )
            .group_by("hi_bin")
            .agg([
                pl.col("damage_pct").quantile(0.05).alias("q05_damage"),
                pl.col("damage_pct").quantile(0.50).alias("q50_damage"),
                pl.col("damage_pct").quantile(0.95).alias("q95_damage"),
            ])
            .vstack(
                pl.DataFrame({
                    "hi_bin": [0.0],
                    "q05_damage": [0.0],
                    "q50_damage": [0.0],
                    "q95_damage": [0.0],
                })
            )
            .sort("hi_bin")
        ),
        overwrite=True,
    )

    # Plot
    pct_hi_damage_up_to_first_floor = results.get_result_as_df(res_name = "pct_hi_damage_up_to_first_floor")
    def depth_damage_plot_1(data, figsize, clean: bool, show: bool):
        # Extract data series from Polars DataFrame
        x = data["hi_bin"].to_numpy()
        q05 = data["q05_damage"].to_numpy()
        q50 = data["q50_damage"].to_numpy()
        q95 = data["q95_damage"].to_numpy()
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            # Fill standard 5th to 95th quantile interval
            ax.fill_between(
                x, 
                q05, 
                q95, 
                color='#C0C0C0', 
                alpha=0.7, 
                edgecolor='none'
            )
            
            # Plot Median (q50) line
            ax.plot(
                x, 
                q50, 
                color='black', 
                linewidth=1.2, 
                linestyle='-'
            )
            
            # GRID & AXIS SETUP
            ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, max(x))
            
            # STYLE
            if not clean:
                ax.set(
                    xlabel='Water depth (m)',
                    ylabel='Total damage, First floor',
                )
                ax.text(
                    0.02, 0.95, '(a)', 
                    transform=ax.transAxes, 
                    fontsize=10, 
                    fontweight='bold', 
                    va='top'
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_DamageCurve"
            png_path = png_path + f".png" if not clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    depth_damage_plot_1(pct_hi_damage_up_to_first_floor, figsize=(2.3, 1.2), clean=False, show=False)
    depth_damage_plot_1(pct_hi_damage_up_to_first_floor, figsize=(2.3, 1.2), clean=True, show=False)

    # --- % of Total Damage (all building) ---
    # pct_he_damage_all_building
    results.add_result(
        res_name="pct_he_damage_all_building",
        res_description="""Estimated depth curve from model for all building using external water depth""",
        data=(
            results.collect_data(columns=['it', 'RP', 'BID', 'FID', 'he', 'c_total', 'cM_total'])
            .group_by(["it", 'RP', "BID"]) # type: ignore
            .agg([
                pl.col("c_total").sum().alias("c_sum"),
                pl.col("cM_total").sum().alias("cM_sum"),
                pl.col("he").first().alias("he"),
            ])
            .with_columns(
                pl.when(pl.col("cM_sum") > 0)
                .then(pl.col("c_sum") / pl.col("cM_sum"))
                .otherwise(0.0)
                .alias("damage_pct")
            )
            .with_columns(
                pl.when(pl.col("he") < 0.2)
                .then(((pl.col("he") / 0.1).floor() * 0.1 + 0.05).round(2))
                .when(pl.col("he") < 0.6)
                .then(((pl.col("he") / 0.2).floor() * 0.2 + 0.1).round(2))
                .when(pl.col("he") < 1.2)
                .then(((pl.col("he") / 0.3).floor() * 0.3 + 0.15).round(2))
                .when(pl.col("he") < 2)
                .then(((pl.col("he") / 0.4).floor() * 0.4 + 0.2).round(2))
                .otherwise(((pl.col("he") / 0.5).floor() * 0.5 + 0.25).round(2))
                .alias("he_bin")
            )
            .group_by("he_bin")
            .agg([
                pl.col("damage_pct").quantile(0.05).alias("q05_damage"),
                pl.col("damage_pct").quantile(0.50).alias("q50_damage"),
                pl.col("damage_pct").quantile(0.95).alias("q95_damage"),
            ])
            .vstack(
                pl.DataFrame({
                    "he_bin": [0.0],
                    "q05_damage": [0.0],
                    "q50_damage": [0.0],
                    "q95_damage": [0.0],
                })
            )
            .sort("he_bin")
        ),
        overwrite=True,
    )

    # Plot
    pct_he_damage_all_building = results.get_result_as_df(res_name = "pct_he_damage_all_building")
    def depth_damage_plot_2(data, figsize, clean: bool, show: bool):
        # Extract data series from Polars DataFrame
        x = data["he_bin"].to_numpy()
        q05 = data["q05_damage"].to_numpy()
        q50 = data["q50_damage"].to_numpy()
        q95 = data["q95_damage"].to_numpy()
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            # Fill standard 5th to 95th quantile interval
            ax.fill_between(
                x, 
                q05, 
                q95, 
                color='#C0C0C0', 
                alpha=0.7, 
                edgecolor='none'
            )
            
            # Plot Median (q50) line
            ax.plot(
                x, 
                q50, 
                color='black', 
                linewidth=1.2, 
                linestyle='-'
            )
            
            # GRID & AXIS SETUP
            ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, max(x))
            
            # STYLE
            if not clean:
                ax.set(
                    xlabel='Water depth (m)',
                    ylabel='Total damage, First floor',
                )
                ax.text(
                    0.02, 0.95, '(a)', 
                    transform=ax.transAxes, 
                    fontsize=10, 
                    fontweight='bold', 
                    va='top'
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_DamageCurveAll"
            png_path = png_path + f".png" if not clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    depth_damage_plot_2(pct_he_damage_all_building, figsize=(2.3, 1.2), clean=False, show=False)
    depth_damage_plot_2(pct_he_damage_all_building, figsize=(2.3, 1.2), clean=True, show=False)

    # --- % of Total Damage (up to first floor, content and continent) ---
    # pct_hi_damage_up_to_first_floor_cte_cti
    results.add_result(
        res_name="pct_hi_damage_up_to_first_floor_cte_cti",
        res_description="""Estimated depth curve from model for all building using external water depth""",
        data=(
            results.collect_data(columns=['it', 'RP', 'BID', 'FID', 'hi', 'c_CTEs', 'c_CTIs', 'cM_CTEs', 'cM_CTIs'])
            .filter(pl.col("FID").is_in([-1, 0])) # type: ignore
            .group_by(["it", 'RP', "BID"])
            .agg([
                pl.col("c_CTEs").sum().alias("c_sum_CTE"),
                pl.col("cM_CTEs").sum().alias("cM_sum_CTE"),
                pl.col("c_CTIs").sum().alias("c_sum_CTI"),
                pl.col("cM_CTIs").sum().alias("cM_sum_CTI"),
                pl.col("hi").filter(pl.col("FID") == 0).first().alias("hi"),
            ])
            .with_columns(
                pl.when(pl.col("cM_sum_CTE") > 0)
                .then(pl.col("c_sum_CTE") / pl.col("cM_sum_CTE"))
                .otherwise(0.0)
                .alias("damage_pct_CTE"),
                
                pl.when(pl.col("cM_sum_CTI") > 0)
                .then(pl.col("c_sum_CTI") / pl.col("cM_sum_CTI"))
                .otherwise(0.0)
                .alias("damage_pct_CTI"),
            )
            .with_columns(
                ((pl.col("hi") / 0.1).floor() * 0.1 + 0.05).round(2).alias("hi_bin")
            )
            .group_by("hi_bin")
            .agg([
                # CTE Quantiles
                pl.col("damage_pct_CTE").quantile(0.05).alias("q05_damage_CTE"),
                pl.col("damage_pct_CTE").quantile(0.50).alias("q50_damage_CTE"),
                pl.col("damage_pct_CTE").quantile(0.95).alias("q95_damage_CTE"),
                # CTI Quantiles
                pl.col("damage_pct_CTI").quantile(0.05).alias("q05_damage_CTI"),
                pl.col("damage_pct_CTI").quantile(0.50).alias("q50_damage_CTI"),
                pl.col("damage_pct_CTI").quantile(0.95).alias("q95_damage_CTI"),
            ])
            .vstack(
                pl.DataFrame({
                    "hi_bin": [0.0],
                    "q05_damage_CTE": [0.0],
                    "q50_damage_CTE": [0.0],
                    "q95_damage_CTE": [0.0],
                    "q05_damage_CTI": [0.0],
                    "q50_damage_CTI": [0.0],
                    "q95_damage_CTI": [0.0],
                })
            )
            .sort("hi_bin")
        ),
        overwrite=True,
    )

    # Plot
    pct_hi_damage_up_to_first_floor_cte_cti = results.get_result_as_df(res_name = "pct_hi_damage_up_to_first_floor_cte_cti")
    def depth_damage_plot_3(data, figsize, clean: bool, show: bool):
        # Extract data series from Polars DataFrame (using hi_bin as x)
        x = data["hi_bin"].to_numpy()
        
        # CTE series
        q05_cte = data["q05_damage_CTE"].to_numpy()
        q50_cte = data["q50_damage_CTE"].to_numpy()
        q95_cte = data["q95_damage_CTE"].to_numpy()
        
        # CTI series
        q05_cti = data["q05_damage_CTI"].to_numpy()
        q50_cti = data["q50_damage_CTI"].to_numpy()
        q95_cti = data["q95_damage_CTI"].to_numpy()
        
        with plt.rc_context(custom_style): # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT CTE
            ax.fill_between(
                x, 
                q05_cte, 
                q95_cte, 
                color='#1f77b4', 
                alpha=0.25, 
                edgecolor='none',
                label='CTE 5-95%'
            )
            ax.plot(
                x, 
                q50_cte, 
                color='#1f77b4', 
                linewidth=1.2, 
                linestyle='-',
                label='CTE Median'
            )
            
            # PLOT CTI
            ax.fill_between(
                x, 
                q05_cti, 
                q95_cti, 
                color='#ff7f0e', 
                alpha=0.25, 
                edgecolor='none',
                label='CTI 5-95%'
            )
            ax.plot(
                x, 
                q50_cti, 
                color='#ff7f0e', 
                linewidth=1.2, 
                linestyle='--',
                label='CTI Median'
            )
            
            # GRID & AXIS SETUP
            ax.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.8)
            ax.set_axisbelow(True)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlim(0, max(x))
            
            # STYLE
            if not clean:
                ax.set(
                    xlabel='Water depth (m)',
                    ylabel='Total damage, First floor',
                )
                ax.text(
                    0.02, 0.95, '(a)', 
                    transform=ax.transAxes, 
                    fontsize=10, 
                    fontweight='bold', 
                    va='top'
                )
                ax.legend(fontsize=6, loc='lower right', frameon=False)
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis='x', which='minor', bottom=False)
            
            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F5_DamageCurve_CTE_CTI"
            png_path = png_path + f".png" if not clean else png_path + f"_clean.png"
            _savefig(fig, png_path, clean, show)
    depth_damage_plot_3(pct_hi_damage_up_to_first_floor_cte_cti, figsize=(3.2, 2.7), clean=False, show=False)
    depth_damage_plot_3(pct_hi_damage_up_to_first_floor_cte_cti, figsize=(3.2, 2.7), clean=True, show=False)

    # endregion

    # ------------------------------
    # region 2.5 EAD (Figure 9)
    # ------------------------------
    # --- EAD (Map) ---
    # expected_annual_damage_by_BID
    results.add_result(
        res_name="expected_annual_damage_by_BID",
        res_description="""Monte Carlo expected average (by BID) with quantiles""",
        data=(
            results.collect_data(columns = ['it', 'RP', 'BID', 'FID', 'c_total'])
            .group_by(["RP", "it", "BID"]) # type: ignore
            .agg((pl.col("c_total").sum()).alias("c_sum"))
            .group_by(["RP", "BID"])
            .agg([
                pl.col("c_sum").quantile(0.05).alias("q05_c_sum"),
                pl.col("c_sum").quantile(0.50).alias("q50_c_sum"),
                pl.col("c_sum").quantile(0.95).alias("q95_c_sum"),
            ])
            .pipe(
                lambda df: pl.concat([
                    df.select("BID")
                    .unique()
                    .with_columns(
                        pl.lit(2).cast(pl.Int64).alias("RP"),
                        pl.lit(0.0).alias("q05_c_sum"),
                        pl.lit(0.0).alias("q50_c_sum"),
                        pl.lit(0.0).alias("q95_c_sum"),
                    )
                    .select(df.columns),
                    df.filter(pl.col("RP") != 2),
                ])
            )
            .with_columns((1 / pl.col("RP")).alias("p"))
            .sort(["BID", "RP"])
            .group_by("BID")
            .agg([
                (
                    0.5 * (pl.col(c) + pl.col(c).shift(-1)) * 
                    (pl.col("p") - pl.col("p").shift(-1))
                ).sum().alias(f"ead_{c}")
                for c in ["q05_c_sum", "q50_c_sum", "q95_c_sum"]
            ])
            .sort("BID")
        ),
        overwrite=True,
    )

    # Data
    expected_annual_damage_by_BID = results.get_result_as_df(res_name = "expected_annual_damage_by_BID")
    expected_annual_damage_by_BID_pandas = expected_annual_damage_by_BID.to_pandas()
    ead_gdf = raw_gdf.merge(
        expected_annual_damage_by_BID_pandas,
        on="BID",
        how="inner"
    )


    def _get_highlight_mask(df, highlight):
        """Helper function to create a combined boolean mask from a highlight dict.
        Handles both GeoDataFrame/Pandas and Polars DataFrames.
        """
        if not highlight:
            return None

        is_polars = isinstance(df, pl.DataFrame)
        mask = None

        for op, conds in highlight.items():
            for col, val in conds.items():
                if is_polars:
                    if op == ">":
                        curr = df[col] > val
                    elif op == ">=":
                        curr = df[col] >= val
                    elif op == "<":
                        curr = df[col] < val
                    elif op == "<=":
                        curr = df[col] <= val
                    elif op == "==":
                        curr = df[col] == val
                    else:
                        continue
                else:  # Pandas / GeoPandas
                    if op == ">":
                        curr = df[col] > val
                    elif op == ">=":
                        curr = df[col] >= val
                    elif op == "<":
                        curr = df[col] < val
                    elif op == "<=":
                        curr = df[col] <= val
                    elif op == "==":
                        curr = df[col] == val
                    else:
                        continue

                mask = curr if mask is None else (mask & curr)

        return mask
    hig = {">": {"ead_q50_c_sum": 2700}, "<": {"NEAR_DIST": 200}}

    # Plot
    def ead_map_plot(gdf_data, figsize, clean: bool, show: bool, highlight: dict | None = None):
        # WMS request
        wms_url = "https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS"
        wms = WebMapService(wms_url, version="1.1.1")
        xmin, ymin, xmax, ymax = gdf_data.total_bounds
        x_pad = (xmax - xmin) * 0.05 if xmax != xmin else 0.01
        y_pad = (ymax - ymin) * 0.05 if ymax != ymin else 0.01
        xmin, xmax = xmin - x_pad, xmax + x_pad
        ymin, ymax = ymin - y_pad, ymax + y_pad
        extent = [xmin, xmax, ymin, ymax]
        bbox = (extent[0], extent[2], extent[1], extent[3])
        img_request = wms.getmap(
            layers=["OI.OrthoimageCoverage"],
            srs="EPSG:4326",
            bbox=bbox,
            size=(1000, 1000),
            format="image/png",
            transparent=True
        )
        img_data = mpimg.imread(BytesIO(img_request.read()))
        
        #base_cmap = plt.get_cmap("YlOrRd")
        #colors = base_cmap(np.linspace(0, 1, 256)) ** 2.1
        #darkened_YlOrRd = mcolors.LinearSegmentedColormap.from_list("DarkYlOrRd", colors)
        
        clip_xmin, clip_ymin, clip_xmax, clip_ymax = gdf_data.total_bounds
        print(clip_xmin, clip_ymin, clip_xmax, clip_ymax)
        chart_style = {
            **custom_style,
            "xtick.top": True,
            "xtick.bottom": False,
            "xtick.labeltop": True,
            "xtick.labelbottom": False,
        }
        with plt.rc_context(chart_style):  # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # Filter data
            col = f"ead_q50_c_sum"
            gdf_zero = gdf_data[gdf_data[col] == 0]
            gdf_pos = gdf_data[gdf_data[col] > 0].copy()
            #gdf_pos["q50_log"] = np.log10(gdf_pos[rp_col])

            # PLOT
            # SHP (EAD)
            q50_pos = gdf_pos[col].to_numpy()
            vmin = float(np.percentile(q50_pos, 5))
            vcenter = float(np.percentile(q50_pos, 50))
            vmax = float(np.percentile(q50_pos, 95))
            if vmin >= vcenter:
                vmin = vcenter - 0.1
            if vmax <= vcenter:
                vmax = vcenter + 0.1
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            
            if not gdf_zero.empty:
                gdf_zero.plot(
                    ax=ax,
                    color="white",
                    edgecolor="none",
                    linewidth=0,
                    markersize=6,
                    zorder=2,
                )
                
            if not gdf_pos.empty:
                gdf_pos.plot(
                    column=col,
                    cmap="YlOrRd",
                    norm=norm,
                    markersize=9,
                    legend=False,
                    ax=ax,
                    edgecolor="none",
                    linewidth=0,
                    zorder=3,
                )
            
            # HIGHLIGHT CONTOUR LAYER
            hl_mask = _get_highlight_mask(gdf_data, highlight)
            if hl_mask is not None and hl_mask.any():
                gdf_highlight = gdf_data[hl_mask]
                gdf_highlight.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor="blue",
                    linewidth=1.5,
                    markersize=12,
                    zorder=4,
                )
            
            # WMS
            ax.imshow(
                img_data, 
                extent=extent, # type: ignore
                origin="upper", 
                alpha=1.0, 
                zorder=0
            )
            
            # STYLE
            plt.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlim(clip_xmin, clip_xmax)
            ax.set_ylim(clip_ymin, clip_ymax)
            ax.set_aspect("equal", adjustable="box")
            
            if not clean:
                ax.set(
                    xlabel="Longitude",
                    ylabel="Latitude",
                    title="BID Centroids EAD (WGS 84 / EPSG:4326)",
                )
                
                # Format labels
                ax.xaxis.set_major_formatter(FuncFormatter(_to_dms))
                ax.yaxis.set_major_formatter(FuncFormatter(_to_dms))
                
                # Colorbar
                sm = cm.ScalarMappable(cmap="YlOrRd", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar_ticks = [vmin, vcenter, vmax]
                print(cbar_ticks)
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([f"€{t:.1f}$" for t in cbar_ticks])
                cbar.set_label("Damage Intensity (€)", rotation=270, labelpad=15)
                
                # Format labels
                ax.xaxis.set_major_formatter(FuncFormatter(_to_dms))
                ax.yaxis.set_major_formatter(FuncFormatter(_to_dms))
                
                # Graphic scale
                center_lat = (ymin + ymax) / 2.0
                dx_meters = 111_320 * np.cos(np.radians(center_lat))
                scalebar = ScaleBar(
                    dx=dx_meters,
                    units="m",
                    dimension="si-length",
                    location="lower left",  # 'lower left', 'lower right', 'upper left', etc.
                    box_alpha=0.7,
                    box_color="white",
                    color="black",
                    scale_loc="bottom",
                    length_fraction=0.2,    # Scale bar will take ~20% of map width
                )
                ax.add_artist(scalebar)
                
                
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[],
                )
            
            # SAVE
            base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F9_EADMap"
            clean_suffix = "_clean.png" if clean else ".png"
            png_path = f"{base_path}{clean_suffix}"

            _savefig(fig, png_path, clean, show)
    ead_map_plot(ead_gdf, (6.5, 2.9), clean = False, show = False)
    ead_map_plot(ead_gdf, (6.5, 2.9), clean = True, show = False)

    # --- EAD x DISTANCE (by BID) ---
    # Plot
    data = pl.from_pandas(ead_gdf.drop(columns=["geometry"], errors="ignore"))
    def ead_dist_plot(data, figsize, clean: bool, show: bool, highlight: dict | None = None):
        # Convert GeoDataFrame to Polars DataFrame (ignoring spatial geometry for plotting)
        with plt.rc_context(custom_style):  # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)

            # Filter data into zero and positive values
            col = "ead_q50_c_sum"
            data_zero = data.filter(pl.col(col) == 0)
            data_pos = data.filter(pl.col(col) > 0)

            # Normaliation
            q50_pos = data_pos[col].to_numpy()
            vmin = float(np.percentile(q50_pos, 5))
            vcenter = float(np.percentile(q50_pos, 50))
            vmax = float(np.percentile(q50_pos, 95))
            if vmin >= vcenter:
                vmin = vcenter - 0.1
            if vmax <= vcenter:
                vmax = vcenter + 0.1
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

            # PLOT
            ax.scatter(
                data_zero["NEAR_DIST"].to_numpy(),
                data_zero[col].to_numpy(),
                color="white",
                edgecolor="none",
                s=12,
                zorder=2,
            )

            # HIGHLIGHT CONTOUR LAYER
            if not data_pos.is_empty():
                ax.scatter(
                    data_pos["NEAR_DIST"].to_numpy(),
                    data_pos[col].to_numpy(),
                    c=data_pos[col].to_numpy(),
                    cmap="YlOrRd",
                    norm=norm,
                    edgecolor="none",
                    linewidth=0,
                    s=25,
                    zorder=3,
                )
            
            hl_mask = _get_highlight_mask(data, highlight)
            if hl_mask is not None and hl_mask.any():
                data_hl = data.filter(hl_mask)
                if not data_hl.is_empty():
                    ax.scatter(
                        data_hl["NEAR_DIST"].to_numpy(),
                        data_hl[col].to_numpy(),
                        facecolor="none",
                        edgecolor="blue",
                        linewidth=1.5,
                        s=35,
                        zorder=4,
                    )
                
            # STYLE
            plt.grid(True, linestyle="--", alpha=0.5)
            
            
            ax.yaxis.set_major_locator(ticker.SymmetricalLogLocator(base=10, linthresh=10))
            ax.yaxis.set_minor_locator(
                ticker.SymmetricalLogLocator(
                    base=10, 
                    linthresh=10, 
                    subs=np.arange(2, 10) # type: ignore
                )
            )
            ax.tick_params(axis='y', which='minor', left=True, right=False)
            
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=0)
            
            if not clean:
                ax.set(
                    xlabel="Near Distance (m)",
                    ylabel="Expected Annual Damage (€)",
                    title="EAD vs Distance",
                )

                # Colorbar 
                sm = cm.ScalarMappable(cmap="YlOrRd", norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar_ticks = [vmin, vcenter, vmax]
                cbar.set_ticks(cbar_ticks)
                cbar.set_ticklabels([f"€{t:.1f}" for t in cbar_ticks])
                cbar.set_label("Damage Intensity (€)", rotation=270, labelpad=15)
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[],
                )
                ax.tick_params(axis="x", which="minor", bottom=False)
            
            # SAVE
            base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\damage\F9_EADDist"
            clean_suffix = "_clean.png" if clean else ".png"
            png_path = f"{base_path}{clean_suffix}"

            _savefig(fig, png_path, clean, show)
    ead_dist_plot(data, (6.5, 1.8), clean = False, show = False)
    ead_dist_plot(data, (6.5, 1.8), clean = True, show = False)

    data_df = (
        data
        .select("NEAR_DIST", "ead_q50_c_sum")
        .filter((pl.col("NEAR_DIST") < 19, pl.col("ead_q50_c_sum") > 600))
    )
    # endregion

    # HERE ------------------------------
    # region 2.6 SHAP (Figure 10)
    # ------------------------------
    # Prepare raw data
    sample_file = os.path.join(gsa_dir, f"shap_rp_{next(iter(RETURN_PERIODS))}.ipc")
    available_cols = set(pl.read_ipc_schema(sample_file).keys())
    shap_groups = [g for g in color_dic["shap_groups"].keys() if g in available_cols]

    # --- EAS (Expected Annual SHAP, Municipality) ---
    # expected_annual_shap_groups_municipality
    results.add_result(
        res_name="expected_annual_shap_groups_municipality",
        res_description="""Expected Annual SHAP (EAS) by group at municipality level with percentage contributions""",
        data=(
            pl.concat([
                pl.read_ipc(os.path.join(gsa_dir, f"shap_rp_{rp}.ipc"))
                .with_columns(pl.lit(rp).cast(pl.Int64).alias("RP"))
                for rp in RETURN_PERIODS.keys()
            ])
            .group_by("RP")
            .agg([pl.col(group).abs().mean().cast(pl.Float64).alias(group) for group in shap_groups])
            .pipe(
                lambda df: pl.concat([
                    pl.DataFrame({
                        "RP": [2],
                        **{group: [0.0] for group in shap_groups},
                    }).select(df.columns),
                    df.filter(pl.col("RP") != 2),
                ])
            )
            .with_columns((1 / pl.col("RP")).alias("p"))
            .sort("RP")
            .select([
                (
                    0.5 * (pl.col(group) + pl.col(group).shift(-1)) * 
                    (pl.col("p") - pl.col("p").shift(-1))
                ).sum().alias(group)
                for group in shap_groups
            ])
            .unpivot(on=shap_groups, variable_name="group", value_name="value")
            .with_columns(
                pct=(pl.col("value") / pl.col("value").sum()) * 100
            )
            .sort("value", descending=True)
        ),
        overwrite=True,
    )

    # Plot
    eas_data = results.get_result_as_df(res_name="expected_annual_shap_groups_municipality")
    def eas_shap_plot_1(data: pl.DataFrame, figsize, clean: bool, show: bool, group_labels: dict | None = None):
        # Ensure dataset is sorted descending by value
        df_plot = data.sort("value", descending=True)
        n_groups = len(df_plot)
        y_pos = np.arange(n_groups)

        with plt.rc_context(custom_style):  # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)

            # PLOT
            for i, row in enumerate(df_plot.to_dicts()):
                grp = row["group"]
                val = row["value"]
                pct = row["pct"]
                color = color_dic["shap_groups"].get(grp, "#333333")
                label_text = group_labels.get(grp, str(grp)) if group_labels else str(grp)

                # Horizontal bar
                ax.barh(
                    y=i,
                    width=val,
                    left=0,
                    height=0.75,
                    color=color,
                    edgecolor="none",
                    zorder=2,
                    label=label_text
                )
                
                # Pct
                if not clean:
                    ax.text(
                        x= 0.0001 * 1.2,
                        y=i,
                        s=f"{pct:.2f} %",
                        va="center",
                        ha="left",
                        fontsize=8,
                        zorder=3
                    )
                
            # STYLE
            ax.invert_yaxis()
            ax.grid(True, axis="x", which="both", linestyle=":", linewidth=0.6, color="#bbb8b8", alpha=0.7, zorder=1)
            ax.set_axisbelow(True)
            
            ax.set_xscale("log")
            ax.set_xlim(0.0001, 1)

            ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=10))
            ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10)) # type: ignore
            
            ax.set(yticks=y_pos)
            
            if not clean:
                ax.set(
                    xlabel=r"Expected Annual |SHAP| (log(€/year))",
                    ylabel="Grouped Input Uncertainty",
                    yticks=y_pos,
                    yticklabels=[f"C{i+1}" for i in range(n_groups)],
                )
                
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.04, 1.0),
                    fontsize=6,
                    framealpha=0.9,
                    borderaxespad=0,
                )
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[]
                )
                ax.tick_params(axis="y", which="minor", left=False)

            # SAVE
            png_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\gsa\F10_ExpectedAnnualSHAP"
            png_path = png_path + ".png" if not clean else png_path + "_clean.png"
            _savefig(fig, png_path, clean, show)
    eas_shap_plot_1(eas_data, figsize=(2.8, 3.1), clean=False, show=False)
    eas_shap_plot_1(eas_data, figsize=(2.8, 3.1), clean=True, show=False)

    # --- SHAP (by RP) ---
    # shap_groups_municipality_by_rp
    results.add_result(
        res_name="shap_groups_municipality_by_rp",
        res_description="""Absolute mean SHAP values by group at municipality level for each return period with percentage contributions""",
        data=(
            pl.concat([
                pl.read_ipc(os.path.join(gsa_dir, f"shap_rp_{rp}.ipc"))
                .with_columns(pl.lit(rp).cast(pl.Int64).alias("RP"))
                for rp in RETURN_PERIODS.keys()
            ])
            .group_by("RP")
            .agg([pl.col(group).abs().mean().cast(pl.Float64).alias(group) for group in shap_groups])
            .with_columns(
                pl.sum_horizontal(shap_groups).alias("total_shap")
            )
            .with_columns([
                # Create the _pct columns relative to that RP's total
                ((pl.col(group) / pl.col("total_shap")) * 100).alias(f"{group}_pct")
                for group in shap_groups
            ])
            .drop("total_shap")
            .sort("RP")
        ),
        overwrite=True,
    )

    # Plot
    shap_groups_municipality_by_rp = results.get_result_as_df(res_name="shap_groups_municipality_by_rp")
    def shap_plot_2(data: pl.DataFrame, figsize, clean: bool, show: bool, group_labels: dict | None = None):
        # Extract RPs for the x-axis mapping
        rps = data["RP"].to_list()
        x_pos = np.arange(len(rps))
        
        # Identify all percentage columns
        pct_cols = [c for c in data.columns if c.endswith("_pct")]
        groups = [c.replace("_pct", "") for c in pct_cols]
        
        with plt.rc_context(custom_style):  # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)
            
            # PLOT
            for grp in groups:
                y = data[f"{grp}_pct"].to_numpy()
                
                color = color_dic["shap_groups"].get(grp, "#333333")
                label_text = group_labels.get(grp, str(grp)) if group_labels else str(grp)
                
                ax.plot(
                    x_pos, 
                    y, 
                    color=color, 
                    marker='o', 
                    markersize=2, 
                    linewidth=1.0, 
                    label=label_text,
                    zorder=3
                )
                
            # STYLE
            ax.set_yscale("symlog", linthresh=0.01, linscale=0.3, base=10)
            ax.set_ylim(0, 100)
            
            y_ticks = [0, 0.01, 0.1, 1, 10, 100]
            ax.set_yticks(y_ticks)
            
            ax.yaxis.set_minor_locator(
                ticker.SymmetricalLogLocator(
                    base=10, 
                    linthresh=0.01, 
                    subs=np.arange(2, 10) # type: ignore
                )
            )
            
            ax.set_xticks(x_pos)
            
            ax.tick_params(axis='x', which='minor', bottom=False, top=False)
            ax.tick_params(axis='y', which='minor', left=True, right=False)
            
            if not clean:
                # Custom formatting to prevent '10^1' default symlog string formats
                ax.set_yticklabels([f"{t:g}" for t in y_ticks])
                ax.set_xticklabels(rps)
                ax.set(
                    xlabel="Return Period (year)",
                    ylabel="Relative total |SHAP| contribution (%)"
                )
                
                # External legend
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(1.04, 1.0),
                    fontsize=8,
                    framealpha=0.9,
                    borderaxespad=0,
                )
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.tick_params(axis="x", which="minor", left=False)

            # SAVE
            base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\gsa\F10_SHAP_Evolution"
            png_path = base_path + "_clean.png" if clean else base_path + ".png"
            _savefig(fig, png_path, clean, show)
    shap_plot_2(shap_groups_municipality_by_rp, figsize=(2.8, 3.1), clean=False, show=False)
    shap_plot_2(shap_groups_municipality_by_rp, figsize=(2.8, 3.1), clean=True, show=False)

    # --- EAS (Expected Annual SHAP, BID) ---
    # expected_annual_shap_groups_bid
    results.add_result(
        res_name="expected_annual_shap_groups_bid",
        res_description="""Expected Annual SHAP (EAS) by group at BID level for each building with percentage contributions""",
        data=(
            pl.concat([
                pl.read_ipc(os.path.join(gsa_dir, f"shap_rp_{rp}.ipc"))
                .group_by("BID")
                .agg([pl.col(group).abs().mean().cast(pl.Float64).alias(group) for group in shap_groups])
                .with_columns(pl.lit(rp).cast(pl.Int64).alias("RP"))
                for rp in RETURN_PERIODS.keys()
            ])
            .pipe(
                lambda df: pl.concat([
                    # Get unique BIDs, add RP=2 and 0.0 for all SHAP groups
                    df.select("BID").unique().with_columns([
                        pl.lit(2).cast(pl.Int64).alias("RP"),
                        *[pl.lit(0.0).alias(group) for group in shap_groups]
                    ]).select(df.columns),
                    df.filter(pl.col("RP") != 2),
                ])
            )
            .with_columns((1 / pl.col("RP")).alias("p"))
            .sort(["BID", "RP"])
            .group_by("BID")
            .agg([
                # Calculate Expected Annual SHAP (trapezoidal integration) per BID
                (
                    0.5 * (pl.col(group) + pl.col(group).shift(-1)) * 
                    (pl.col("p") - pl.col("p").shift(-1))
                ).sum().alias(group)
                for group in shap_groups
            ])
            .with_columns(
                # Calculate the sum of all absolute EAS means for the current BID
                pl.sum_horizontal(shap_groups).alias("total_eas")
            )
            .with_columns([
                # Create the _pct columns relative to that BID's total EAS (safeguard against division by zero)
                pl.when(pl.col("total_eas") > 0)
                .then((pl.col(group) / pl.col("total_eas")) * 100)
                .otherwise(0.0)
                .alias(f"{group}_pct")
                for group in shap_groups
            ])
            .drop("total_eas")
            .sort("BID")
        ),
        overwrite=True,
    )

    # --- EAS RANK (BID, Rank first 4th) ---
    # expected_annual_shap_groups_bid_rank
    expected_annual_shap_groups_bid = results.get_result_as_df(res_name="expected_annual_shap_groups_bid")
    results.add_result(
        res_name="expected_annual_shap_groups_bid_rank",
        res_description="EAS at BID level with top 4 ranked SHAP groups concatenated and their dominance percentage",
        data=(
            expected_annual_shap_groups_bid.join(
                expected_annual_shap_groups_bid
                .select(["BID"] + shap_groups)
                .unpivot(index="BID", on=shap_groups, variable_name="group", value_name="eas_value")
                .sort(["BID", "eas_value"], descending=[False, True])
                .with_columns([
                    pl.int_range(1, pl.len() + 1).over("BID").alias("rank_num"),
                    pl.col("eas_value").sum().over("BID").alias("total_eas")  # Calculate total before filtering
                ])
                .filter(pl.col("rank_num") <= 3)
                .group_by("BID")
                .agg([
                    pl.col("group").str.join(">").alias("rank"),
                    # Calculate % explained by the top 4, safeguarding against division by zero
                    pl.when(pl.col("total_eas").first() > 0)
                    .then((pl.col("eas_value").sum() / pl.col("total_eas").first()) * 100)
                    .otherwise(0.0)
                    .alias("top4_explained_pct")
                ])
                .with_columns([
                    (pl.len().over("rank") / expected_annual_shap_groups_bid.height * 100).alias("rank_pct")
                ]),
                on="BID",
                how="left"
            )
        ),
        overwrite=True,
    )

    # Plot
    expected_annual_shap_groups_bid_rank = results.get_result_as_df(res_name="expected_annual_shap_groups_bid_rank")
    rank_summary_table = (
        expected_annual_shap_groups_bid_rank
        .group_by(["rank", "rank_pct"])
        .agg(
            pl.col("top4_explained_pct").min().alias("mean_top4_explained_pct")
        )
        .sort("rank_pct", descending=True)
    )
    with pl.Config(fmt_str_lengths=150, tbl_rows=-1):
        print(rank_summary_table)
    rank_gdf = raw_gdf.merge(
        expected_annual_shap_groups_bid_rank.to_pandas(),
        on="BID",
        how="inner"
    )

    def rank_map_plot(gdf_data, figsize, clean: bool, show: bool):
        # WMS request setup
        wms_url = "https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS"
        wms = WebMapService(wms_url, version="1.1.1")
        xmin, ymin, xmax, ymax = gdf_data.total_bounds
        x_pad = (xmax - xmin) * 0.05 if xmax != xmin else 0.01
        y_pad = (ymax - ymin) * 0.05 if ymax != ymin else 0.01
        xmin, xmax = xmin - x_pad, xmax + x_pad
        ymin, ymax = ymin - y_pad, ymax + y_pad
        extent = [xmin, xmax, ymin, ymax]
        bbox = (extent[0], extent[2], extent[1], extent[3])
        
        img_request = wms.getmap(
            layers=["OI.OrthoimageCoverage"],
            srs="EPSG:4326",
            bbox=bbox,
            size=(1000, 1000),
            format="image/png",
            transparent=True
        )
        img_data = mpimg.imread(BytesIO(img_request.read()))
        
        clip_xmin, clip_ymin, clip_xmax, clip_ymax = gdf_data.total_bounds
        
        # Identify unique categories and create a desaturated color palette
        unique_ranks = gdf_data["rank"].dropna().unique().tolist()
        cmap = plt.get_cmap("Set3")
        rank_colors = {rank: cmap(i % cmap.N) for i, rank in enumerate(unique_ranks)}
        
        chart_style = {
            **custom_style,
            "xtick.top": True,
            "xtick.bottom": False,
            "xtick.labeltop": True,
            "xtick.labelbottom": False,
        }
        
        with plt.rc_context(chart_style):  # type: ignore
            # FIG
            fig, ax = _init_fig(figsize, clean)

            # PLOT Geometries grouped by rank
            for rank_val in unique_ranks:
                subset = gdf_data[gdf_data["rank"] == rank_val]
                subset.plot(
                    ax=ax,
                    color=rank_colors[rank_val],
                    markersize=9,
                    edgecolor="black",
                    linewidth=0.1,
                    label=rank_val,
                    zorder=3,
                )
            # WMS Background
            ax.imshow(
                img_data, 
                extent=extent, # type: ignore
                origin="upper", 
                alpha=1.0, 
                zorder=0
            )
            
            # STYLE
            plt.grid(True, linestyle="--", alpha=0.5)
            ax.set_xlim(clip_xmin, clip_xmax)
            ax.set_ylim(clip_ymin, clip_ymax)
            ax.set_aspect("equal", adjustable="box")
            
            if not clean:
                ax.set(
                    xlabel="Longitude",
                    ylabel="Latitude",
                    title="EAS Dominance Rank Map (WGS 84 / EPSG:4326)",
                )
                
                # Format labels
                ax.xaxis.set_major_formatter(FuncFormatter(_to_dms))
                ax.yaxis.set_major_formatter(FuncFormatter(_to_dms))
                
                # Categorical Legend (replaces colorbar)
                ax.legend(
                    title="Top 4 SHAP Rank",
                    loc="center left", 
                    bbox_to_anchor=(1.02, 0.5), 
                    fontsize=7,
                    title_fontsize=8,
                    framealpha=0.9,
                    edgecolor="black"
                )
                
                # Graphic scale
                center_lat = (ymin + ymax) / 2.0
                dx_meters = 111_320 * np.cos(np.radians(center_lat))
                scalebar = ScaleBar(
                    dx=dx_meters,
                    units="m",
                    dimension="si-length",
                    location="lower left",
                    box_alpha=0.7,
                    box_color="white",
                    color="black",
                    scale_loc="bottom",
                    length_fraction=0.2,
                )
                ax.add_artist(scalebar)
                
            else:
                ax.set(
                    xticklabels=[],
                    yticklabels=[],
                )
            
            # SAVE
            base_path = r"C:\Users\outal\GITHUB\InDepthPROFILE\data\mocaloss\postprocessor\gsa\F10_SHAPRankMap"
            clean_suffix = "_clean.png" if clean else ".png"
            png_path = f"{base_path}{clean_suffix}"

            _savefig(fig, png_path, clean, show)
    rank_map_plot(rank_gdf, (6.5, 2.9), clean=False, show=False)
    rank_map_plot(rank_gdf, (6.5, 2.9), clean=True, show=False)

    # endregion

    # Final export
    results.export_res_to_xlsx()

## FINISH HELPERS
pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(10)
def summary(data_to_summarize, col):
    df_summary = (
        data_to_summarize
        .group_by("RP")
        .agg([
            pl.len().alias("count"),
            #pl.col(col).mean().alias("mean"),
            #pl.col(col).std().alias("std"),
            pl.col(col).min().alias("min"),
            #pl.col(col).quantile(0.01).alias("q01"),
            pl.col(col).quantile(0.05).alias("q05"),
            pl.col(col).median().alias("q50"),
            pl.col(col).quantile(0.95).alias("q95"),
            #pl.col(col).quantile(0.99).alias("q99"),
            pl.col(col).max().alias("max"),
        ])
        .sort("RP", descending=True)
    )
    pl.Config.set_tbl_cols(-1)
    print(df_summary)




