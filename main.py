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
          
    ## STAGE 3: Economic Valuation
    RUN = False
    if RUN:
        
        # 1. Fit data
        RUN = False
        if RUN:
            print("### STAGE 3: Fit Data ###")
            from src.valuation.preparation import DistributionFitter
            pandarallel.initialize(progress_bar=True)
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

        # 2. Run Loss Model
        RUN = False
        if RUN:
            print("### STAGE 3: Run Loss Model ###")
            from src.valuation.execution import LossModelExecutionEngine
            engine = LossModelExecutionEngine(PATHS, CODES, RETURN_PERIODS)
            dataset, col_name_dic = engine.create_base_dataset(gdf_buildings=gpd.read_file(PATHS['buildings_shp']))
            engine.run_economic_monte_carlo(
                n_simulations=10000, 
                cores=16, 
                batch_size=128
            )
            
    ## STAGE 4: Sensitivity Analysis
    RUN = False
    if RUN:
        print("### STAGE 3: Run Sensitivity Analysis ###")
        from src.valuation.sensitivity import SurrogateSensitivityEngine
        sensitivity_engine = SurrogateSensitivityEngine(PATHS, RETURN_PERIODS)
        sensitivity_engine.shard_dataset_by_return_period()
        tuning_results = sensitivity_engine.run_bayesian_optimization()
        best_params_df = sensitivity_engine.validate_and_fit_models(tuning_results)
        sensitivity_engine.fit_complete_surrogates(best_params_df)
        sensitivity_engine.compute_and_group_shap_values()
    
    ## STAGE 5: Results
    RUN = True
    if RUN:
        from src.valuation.results import DataPreparator, Plotter
        
        ## EAD
        RUN = False
        if RUN:
            # Paths and Variables
            output_gsa_dir = PATHS['ead_map']
            
            # fig_verion, re_run_grouping, clean = 2, False, True
            def plot_ead_map(fig_version, clean=False):
                ## Prepare data
                print(f"\n[STAGE 5. EAD_Map] Preparing data...")
                
                full_df = DataPreparator.load_and_concatenate_mc_parts(
                    PATHS['dataset_mc_parts'],
                    ['SN', 'BID', 'RP', 'c_hu_bf'],
                    "MC_Part")
                
                gdf_buildings = DataPreparator.load_flooded_buildings_gdf(PATHS['buildings_shp'], PATHS['depth_samples_pkl'])
                
        ## GSA
        RUN = True
        if RUN:
            from src.valuation.results import shap_results_grouper
            # Paths and Variables
            output_gsa_dir = PATHS['xgb']
            RPs_unq = list(RETURN_PERIODS.keys())
            # Colors
            gsa_group_colors = {
                # Hydrology / Hazard State
                'he': "#FF0000",           # Pure Red
                # Structural Building Components
                'Structure': "#FF6600",    # Intense Orange
                'IH': "#993300",           # Deep Burnt Orange
                'BH': "#B35900",           # Medium Brown-Orange
                'GL': "#FFCC00",           # Vivid Yellow-Gold
                'Cga': "#FFAA00",          # Dark Amber
                # Parameters / Constants
                'C.High': "#00FFCC",       # Ultra Cyan/Teal
                'C.High_CTE': "#48FFDA", 
                'C.High_CTI': "#00A181",
                'Prices': "#0000FF",       # Pure Blue
                'Prices_CTE': "#4B4BFF", 
                'Prices_CTI': "#0000A8",
                'Objects': "#9900FF",      # Vivid Purple
                'Objects_CTE': "#B547FF", 
                'Objects_CTI': "#5E009C",
                'ed': "#FF00FF",           # Neon Magenta
                'Materials': "#FF0066",    # Hot Pink/Crimson
                'Materials_CTI': "#FF0066",
                # Aggregated Damage Assets
                'Content': "#33FF00",      # Bright Lime Green
                'Continent': "#006600",    # Deep Forest Green
            }
            # Data Grouping (Optional Re-run)
            RUN = False
            if RUN:
                def rule_l1(c): # Division of main groups
                    if c in ['IH', 'BH', 'GL', 'Cga']: return "Structure"
                    if c in ['ed', 'he', 'base_value', 'BID']: return c
                    if c.startswith('p_'): return "Prices (p_)"
                    if c.startswith('n_'): return "Objects (n_)"
                    if c.startswith('hc_'): return "C.High (hc_)"
                    if c.startswith('m_'): return "Materials (m_)"
                    return "Unclassified"

                def rule_l2(c): # Division of main groups withouth structure
                    if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
                    if c.startswith('p_'): return "Prices (p_)"
                    if c.startswith('n_'): return "Objects (n_)"
                    if c.startswith('hc_'): return "C.High (hc_)"
                    if c.startswith('m_'): return "Materials (m_)"
                    return "Unclassified"

                def rule_l3(c): # Division of main CTE and CTI groups withouth structure
                    if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
                    content_pattern = r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)'
                    continent_pattern = r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)'
                    prefix_map = {
                        'p_': 'Prices',
                        'n_': 'Objects',
                        'hc_': 'C.High',
                        'm_': 'Materials'
                    }
                    for prefix, base_name in prefix_map.items():
                        if c.startswith(prefix):
                            if re.search(content_pattern, c):
                                return f"{base_name}_CTE"
                            elif re.search(continent_pattern, c):
                                return f"{base_name}_CTI"
                    return "Unclassified"

                def rule_l4(c): 
                    if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
                    
                    content_pattern = r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)'
                    continent_pattern = r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)'
                    prefix_map = {
                        'p_': 'Prices',
                        'n_': 'Objects',
                        'hc_': 'C.High',
                        'm_': 'Materials'
                    }
                    
                    for prefix, base_name in prefix_map.items():
                        if c.startswith(prefix):
                            if re.search(content_pattern, c):
                                suffix = "CTE"
                            elif re.search(continent_pattern, c):
                                suffix = "CTI"
                            else:
                                continue

                            match = re.search(r'_(-?\d+)(_|$)', c)
                            if match:
                                floor_num = int(match.group(1))
                                floor_group = "-" if floor_num < 0 else "+"
                            else:
                                floor_group = "gen"
                                
                            return f"{base_name}_{floor_group}_{suffix}"
                            
                    return "Unclassified"
            
                def rule_l5(c): # Division of main groups withouth structure with CTE and CTI
                    if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
                    if re.search(r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH)(_|$)', c):
                        return "Content"
                    if re.search(r'_(ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)', c) or c.startswith(('p_ETP', 'p_EXF', 'p_WND', 'p_PUM', 'p_PRW', 'p_ITP', 'p_SOI', 'p_CLE', 'p_SKT', 'p_RDR', 'p_PLG', 'm_ETP', 'm_EXF', 'm_WND', 'm_PUM', 'm_PRW', 'm_ITP', 'm_SOI', 'm_CLE', 'm_SKT', 'm_RDR', 'm_PLG')):
                        return "Continent"
                    return "Unclassified"

                def rule_l6(c): # Division of individual groups considering individual CTE and CTI
                    if c in ['ed', 'he', 'IH', 'BH', 'GL', 'Cga', 'base_value', 'BID']: return c
                    pattern = r'_(APP|CLO|COM|DEC|ELE|ENG|FAD|FUR|HHG|HHB|INS|LEI|OTH|SPE|TOO|VEH|ETP|EXF|WND|PUM|PRW|ITP|SOI|CLE|SKT|RDR|PLG|FRI)(_|$)'
                    match = re.search(pattern, c)
                    if match:
                        return match.group(1) # Returns the exact code found (e.g., "APP", "ETP")
                    return "Unclassified"

                for rp in RPs_unq:
                    print(f"Setting groups for RP {rp}")
                    # Ensure using the resolved output_gsa_dir pathlib object
                    shap_path = output_gsa_dir / f"shap_results_rp_{rp}.feather"
                    
                    if not os.path.exists(shap_path):
                        print(f"File not found: {shap_path}. Skipping.")
                        continue
                    
                    df_shap_rp = pd.read_feather(shap_path)
                    grouper = shap_results_grouper(df_shap_rp.columns.tolist(), separator="|")
                    grouper.remove_all_levels()
                    
                    grouper.add_level(rules=rule_l1)
                    grouper.add_level(rules=rule_l2)
                    grouper.add_level(rules=rule_l3)
                    grouper.add_level(rules=rule_l4)
                    grouper.add_level(rules=rule_l5)
                    grouper.add_level(rules=rule_l6)
                    
                    print(f"Updating and resaving {shap_path}")
                    df_shap_rp = grouper.update_dataframe(df_shap_rp)
                    # get a list of column names
                    column_names = df_shap_rp.columns.tolist()
                    
                    df_shap_rp.to_feather(shap_path)
            
            # Chart
            RUN = True
            if RUN:
                # fig_verion, re_run_grouping, clean = 5, False, True
                def plot_gsa(fig_version, clean=False):                    
                    ## Prepare data
                    print(f"\n[STAGE 5. GSA] Preparing data...")
                    
                    # Dataframes
                    df_expected_l3_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=3, by_bid=False, min_rp_AEP_0=2)
                    df_evolution_l3_all = DataPreparator.calc_shap_rp_evolution_grouped(output_gsa_dir, RPs_unq, level=3, by_bid=False)
                    df_expected_l3_bid = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=3, by_bid=True, min_rp_AEP_0=2)
                    gdf_buildings = DataPreparator.load_flooded_buildings_gdf(PATHS['buildings_shp'], PATHS['depth_samples_pkl'])
                    
                    # Geodataframes
                    df_filtered = df_expected_l3_bid.reset_index()
                    k_vars = len(df_filtered["Feature_Group"].unique())
                    cumulative_threshold = 80
                    acronym_map = {'he': 'he', 'C.High (hc_)': 'hc', 'Objects (n_)': 'n', 'Structure': 'S', 'Prices (p_)': 'p', 'ed': 'ed', 'Materials (m_)': 'm'}
                    df_filtered = df_filtered[df_filtered['Q50'] > 0].copy()
                    df_filtered = df_filtered.dropna(subset=['Q50'])
                    df_filtered = df_filtered.sort_values(by=['BID', 'Q50_share_%'], ascending=[True, False])
                    df_filtered['cumsum_share'] = df_filtered.groupby('BID')['Q50_share_%'].cumsum()
                    df_filtered['prev_cumsum'] = df_filtered['cumsum_share'] - df_filtered['Q50_share_%']
                    df_filtered = df_filtered[df_filtered['prev_cumsum'] < cumulative_threshold]
                    
                    df_map = df_filtered.groupby('BID').head(k_vars).copy()
                    df_map['Short_Feature'] = df_map['Feature_Group'].map(lambda x: acronym_map.get(str(x).strip(), str(x).strip()))

                    combinations = df_map.groupby('BID')['Short_Feature'].apply(lambda x: ' | '.join(x)).reset_index()
                    combinations.rename(columns={'Short_Feature': f'Top_{cumulative_threshold}%_Combo'}, inplace=True)
                    df_combo_counts = combinations[f'Top_{cumulative_threshold}%_Combo'].value_counts().reset_index()
                    df_combo_counts.columns = [f'Top_{cumulative_threshold}%_Combo', 'BID_count']
                    unique_combos = df_combo_counts[f'Top_{cumulative_threshold}%_Combo'].unique()
                    
                    def sort_hierarchical_combos(combos_list):
                        """
                        Sorts combinations hierarchically by frequency at each position, 
                        placing longer combinations before shorter ones sharing the same prefix.
                        """
                        split_combos = [c.split(' | ') for c in combos_list]
                        
                        def _sort_recursive(items):
                            if not items:
                                return []
                            
                            counts = {}
                            for item in items:
                                if len(item) > 0:
                                    counts[item[0]] = counts.get(item[0], 0) + 1
                                    
                            sorted_keys = sorted(counts.keys(), key=lambda x: (-counts[x], x))
                            
                            result = []
                            for key in sorted_keys:
                                continuations = [item[1:] for item in items if len(item) > 0 and item[0] == key and len(item) > 1]
                                terminating = [item for item in items if len(item) > 0 and item[0] == key and len(item) == 1]
                                
                                sorted_continuations = _sort_recursive(continuations)
                                
                                for sub in sorted_continuations:
                                    result.append([key] + sub)
                                for _ in terminating:
                                    result.append([key])
                                    
                            return result
                        
                        sorted_splits = _sort_recursive(split_combos)
                        
                        # Remove duplicates preserving order
                        seen = set()
                        final_combos = []
                        for c in sorted_splits:
                            combo_str = ' | '.join(c)
                            if combo_str not in seen:
                                seen.add(combo_str)
                                final_combos.append(combo_str)
                                
                        return final_combos
                    sorted_unique_combos = sort_hierarchical_combos(df_combo_counts[f'Top_{cumulative_threshold}%_Combo'].unique())
                    df_combo_counts[f'Top_{cumulative_threshold}%_Combo'] = pd.Categorical(
                        df_combo_counts[f'Top_{cumulative_threshold}%_Combo'], 
                        categories=sorted_unique_combos, 
                        ordered=True
                    )
                    df_combo_counts = df_combo_counts.sort_values(by=f'Top_{cumulative_threshold}%_Combo').reset_index(drop=True)
                    total_BIDS = df_combo_counts["BID_count"].sum()
                    df_combo_counts["BID_count_%"] = round((df_combo_counts["BID_count"] / total_BIDS * 100),1)
                    
                    num_combos = len(sorted_unique_combos)
                    cmap_name = 'tab10' if num_combos <= 10 else 'tab20'
                    cmap = plt.get_cmap(cmap_name)
                    pastel_colors = []
                    for i in range(num_combos):
                        rgb_orig = cmap(i % cmap.N)[:3]
                        h, l, s = colorsys.rgb_to_hls(*rgb_orig)
                        l_soft = 0.82  
                        s_soft = 0.65 
                        rgb_soft = colorsys.hls_to_rgb(h, l_soft, s_soft)
                        pastel_colors.append(mcolors.to_hex(rgb_soft))
                    color_map = {combo: pastel_colors[i] for i, combo in enumerate(sorted_unique_combos)}
                    
                    combinations[f'Top_{cumulative_threshold}%_Combo'] = pd.Categorical(
                        combinations[f'Top_{cumulative_threshold}%_Combo'], 
                        categories=sorted_unique_combos, 
                        ordered=True
                    )
                    combinations['combo_color'] = combinations[f'Top_{cumulative_threshold}%_Combo'].map(color_map)
                    
                    gdf_shap = gdf_buildings.merge(combinations, on='BID', how='inner')
                    gdf_shap = gdf_shap.merge(df_map[df_map["Feature_Group"]=="he"], on='BID', how='inner')
                    gdf_shap_centroids = gdf_shap.copy()
                    gdf_shap_centroids['geometry'] = gdf_shap.centroid
                    gdf_shap_centroids_4326 = gdf_shap_centroids.to_crs(epsg=4326)
                    gdf_shap_centroids_4326 = gdf_shap_centroids_4326.merge(
                        df_combo_counts[[f'Top_{cumulative_threshold}%_Combo', 'BID_count']], 
                        on=f'Top_{cumulative_threshold}%_Combo', 
                        how='left'
                    )
                    gdf_shap_centroids_4326 = gdf_shap_centroids_4326.sort_values(by='BID_count', ascending=False).reset_index(drop=True)
                    
                    minx, miny, maxx, maxy = gdf_shap_centroids_4326.total_bounds
                    buffer = 0.01
                    plot_extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]
                    
                    
                    ## Prepare Plot
                    print(f"\n[STAGE 5. GSA] Preparing layout...")
                    # Plotter
                    plotter = Plotter(
                        config_paths=PATHS,
                        config_codes=CODES,
                        config_return_periods=RETURN_PERIODS
                    )
                    
                    ## Layout
                    layout_params = {
                        'layout': 'mosaic',
                        'mosaic_structure': [
                            ["a", "b"],
                            ["c", "c"],],
                        'figsize': (6, 8),
                        'kwargs': {'gridspec_kw': {
                            'wspace': 0.1,
                            'hspace': 0.1,
                            'width_ratios': [0.5, 0.5],
                            'height_ratios': [0.55, 0.45]}},
                        'adjust': {'bottom': 0.1, 'top': 0.9, 'left': 0.1, 'right': 0.9}
                    }
                    dict_to_plot = {}
                    
                    ## Plots
                    print(f"\n[STAGE 5. GSA] Preparing plots...")
                    
                     # a
                    ax = 'a'
                    plotter.init_dic(dict_to_plot, ax, 'plots')
                    bar_colors = []
                    for grp in df_expected_l3_all.index:
                        grp_str = str(grp)
                        bar_colors.append(gsa_group_colors.get(grp_str, '#cccccc'))
                    dict_to_plot[ax]['plots'].append({
                        'plot_type': 'barh',
                        'y': df_expected_l3_all.index.tolist(),
                        'width': df_expected_l3_all['Q50'].values,
                        'color': bar_colors,
                        'alpha': 1,
                        'reverse': True
                    })
                    if not clean:
                        for idx, row in df_expected_l3_all.iterrows():
                            dict_to_plot[ax]['plots'].append({
                                'plot_type': 'text', 'x': row['Q50'], 'y': idx,
                                'text': f" {row['Q50_share_%']:.2f}%", 'transform': 'data',
                                'va': 'center', 'ha': 'left', 'fontsize': 8, 'color': 'black'
                            })

                    # b
                    ax = 'b'
                    plotter.init_dic(dict_to_plot, ax, 'plots')
                    share_cols = [col for col in df_evolution_l3_all.columns if col.startswith('Share_%_RP')]
                    rps_numeric = [int(col.replace('Share_%_RP', '')) for col in share_cols]
                    sorted_idx = np.argsort(rps_numeric)
                    share_cols_sorted = [share_cols[i] for i in sorted_idx]
                    rps_categorical = [str(rps_numeric[i]) for i in sorted_idx]
                    for feature_name, row in df_evolution_l3_all.iterrows():
                        feat_color = gsa_group_colors.get(feature_name, '#cccccc')
                        y_shares = row[share_cols_sorted].values
                        dict_to_plot[ax]['plots'].append({
                            'plot_type': 'line', 
                            'x': rps_categorical, 
                            'y': y_shares, 
                            'color': feat_color, 
                            'marker': 'o', 
                            'markersize': 3, 
                            'linewidth': 2,
                            'label': feature_name
                        })
                    
                    # c
                    ax = 'c'
                    plotter.init_dic(dict_to_plot, ax, 'plots')
                    real_ratio = plotter.get_ax_ratio(layout_params, ax)
                    top_left_x = (4, 42, 45)
                    top_left_y = (40, 24, 53) #50
                    zoom_pct = 90 #100
                    calc_extent_vals = plotter.calc_map_extent(top_left_x, top_left_y, zoom_pct, real_ratio)
                    xmin, xmax, ymin, ymax = calc_extent_vals
                    dict_to_plot[ax]['plots'].extend([
                        {'plot_type': 'wms',
                            'url': 'https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS',
                            'layers': ['OI.OrthoimageCoverage'],
                            'extent': plot_extent,
                            'crs': 'EPSG:4326',
                            'size': (1000, 1000),
                            'alpha': 0.7, # Lower alpha to blend with the black background, creating a darker image
                            'zorder': 1
                        },
                        {'plot_type': 'gdf_shp',
                            'gdf': gdf_shap_centroids_4326,
                            'color': gdf_shap_centroids_4326['combo_color'].tolist(),
                            'markersize': 2,
                            'alpha': 1,
                            'zorder': 3
                        }
                    ])
                    if not clean:
                        xmin, xmax, ymin, ymax = plot_extent
                        legend_x = xmin + (xmax - xmin) * 0.02
                        legend_y = ymax - (ymax - ymin) * 0.02
                        y_step = (ymax - ymin) * 0.04
                        
                        dict_to_plot[ax]['plots'].append({
                            'plot_type': 'text', 'x': legend_x, 'y': legend_y,
                            'text': "Top 7 Variables Rank:", 'fontsize': 8, 'color': 'white',
                            'ha': 'left', 'va': 'top', 'transform': 'data', 'fontweight': 'bold',
                            'path_effects': [pe.withStroke(linewidth=2, foreground="black")], 'zorder': 5
                        })
                        
                        # Changed from unique_combos to sorted_unique_combos
                        for i, combo in enumerate(sorted_unique_combos):
                            combo_color = color_map[combo]
                            current_y = legend_y - ((i + 1) * y_step)
                            
                            dict_to_plot[ax]['plots'].append({
                                'plot_type': 'text',
                                'x': legend_x,
                                'y': current_y,
                                'text': f"■ {combo}",
                                'color': combo_color,
                                'fontsize': 6,
                                'ha': 'left',
                                'va': 'top',
                                'transform': 'data',
                                'path_effects': [pe.withStroke(linewidth=1, foreground="black")],
                                'zorder': 5
                            })

                    # Style
                    print(f"\n[STAGE 5. GSA] Preparing style...")
                    
                    # a
                    ax = 'a'
                    plotter.init_dic(dict_to_plot, ax, 'style')
                    dict_to_plot[ax]['style'].extend([
    {'style_type': 'xscale', 'value': 'log'},
])
                    if clean:
                        dict_to_plot[ax]['style'].extend([
                            {'style_type': 'yticklabels', 'labels': []},
                            {'style_type': 'xticklabels', 'labels': []}, # Clear log minor/major labels
                            {'style_type': 'ticks_params', 'labelleft': False, 'labelbottom': False} # Force hide
                        ])
                    else:
                        dict_to_plot[ax]['style'].extend([
                            {'style_type': 'xlabel', 'label': 'Expected Annual Average |SHAP|'},
                            {'style_type': 'xticks', 'ticks': [1e-1, 1e0, 1e1, 1e2, 1e3]},
                        ])
    
                    # b
                    ax = 'b'
                    plotter.init_dic(dict_to_plot, ax, 'style')
                    dict_to_plot[ax]['style'].extend([
                        {'style_type': 'yscale', 'value': 'symlog', 'linthresh': 0.01, 'linscale': 0.12},
                        {'style_type': 'ylim', 'ymin': 0, 'ymax': 1e2},
                        {'style_type': 'minor_locator', 'axis': 'y', 'locator_type': 'log', 'base': 10.0, 'subs': range(1, 10)},
                        
                    ])
                    if clean:
                        dict_to_plot[ax]['style'].extend([
                            {'style_type': 'xticklabels', 'labels': []},
                            {'style_type': 'yticklabels', 'labels': []},
                            {'style_type': 'ticks_params', 'labelbottom': False, 'labelleft': False},
                        ])
                    else:
                        dict_to_plot[ax]['style'].extend([
                            {'style_type': 'xlabel', 'label': 'Return Period (RP)'},
                            {'cite_type': 'ylabel', 'label': 'Share % of total |SHAP|'},
                            {'style_type': 'yticks', 'ticks': [0, 0.01, 0.1, 1, 10, 100]},
                            {'style_type': 'yticklabels', 'labels': ['0','0.01', '0.1', '1', '10', '100']},
                        ])
                    
                    # c
                    ax = 'c'
                    plotter.init_dic(dict_to_plot, ax, 'style')
                    label_vis = not clean  # Determines if top/left labels are shown
                    dict_to_plot[ax]['style'].extend([
                        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3, 'zorder': 1},
                        {'style_type': 'xlim', 'left': calc_extent_vals[0], 'right': calc_extent_vals[1]},
                        {'style_type': 'ylim', 'bottom': calc_extent_vals[2], 'top': calc_extent_vals[3]},
                        {'style_type': 'aspect', 'aspect': 'equal'},
                        {'style_type': 'spines', 'top': True, 'right': True, 'left': True, 'bottom': True},
                        # Modified tick parameters behavior:
                        {'style_type': 'ticks_params', 'which': 'both', 'labelsize': 8,
                        'top': False, 'bottom': True, 'left': True, 'right': False,
                        'labeltop': False, 'labelbottom': label_vis, 'labelleft': label_vis, 'labelright': False},
                        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90},
                        {'style_type': 'yticklabels', 'va': 'center'},
                        {'style_type': 'major_formatter', 'axis': 'y', 'formatter_type': 'dms_suffix', 'suffix': ' N'},
                        {'style_type': 'major_formatter', 'axis': 'x', 'formatter_type': 'dms_suffix', 'suffix': ' W'},
                    ])
                    if clean:
                        dict_to_plot[ax]['style'].extend([
                            {'style_type': 'xticklabels', 'labels': []},
                            {'style_type': 'yticklabels', 'labels': []}
                        ])
                    else:
                        dict_to_plot[ax]['style'].extend([
                            {'style_type': 'legend', 'loc': 'upper right', 'fontsize': 5, 'title': 'Top 7 Variables Rank'}
                        ])

                    # Save
                    print(f"\n[STAGE 5. GSA] Preparing saving...")
                    gsa_target_path = Path(PATHS['gsa'])
                    save_params = {
                        'output_dir': str(gsa_target_path),  # Saves in output/gsa/ folder dynamically
                        'subfolder': '',
                        'fname': f"{gsa_target_path.name}_v{fig_version}.png",
                        'show': False,
                        'dpi': 300,
                    }

                    # 3. Execute Plot Rendering via the Engine
                    print(f"\n[STAGE 5. GSA] Running plot any...")
                    plotter.plot_any(layout_params, dict_to_plot, save_params)
                    
                plot_gsa(fig_version="5", clean=False)
                plot_gsa(fig_version="5.c", clean=True)    

                    
            # Charts
            RUN = False
            if RUN:
                # fig_verion, re_run_grouping, clean = 4.4, False, True
                def plot_gsa_charts_sensitivity_analysis(fig_version, clean=False):                    
                    ## Prepare data
                    print(f"\n[STAGE 5. GSA] Preparing data...")                   
                                    
                    # 4. Load Expected SHAP Data
                    df_expected_l1_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=1, by_bid=False, min_rp_AEP_0=2)
                    df_evolution_l1_all = DataPreparator.calc_shap_rp_evolution_grouped(output_gsa_dir, RPs_unq, level=1, by_bid=False)
                    df_expected_l2_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=2, by_bid=False, min_rp_AEP_0=2)
                    df_expected_l2_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=2, by_bid=False, min_rp_AEP_0=2)
                    df_expected_l4_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=4, by_bid=False, min_rp_AEP_0=2)
                    df_expected_l3_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=3, by_bid=False, min_rp_AEP_0=2)
                    df_expected_l5_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=5, by_bid=False, min_rp_AEP_0=2)
                    df_expected_l6_all = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=6, by_bid=False, min_rp_AEP_0=2)
                    
                    # 6. Sankey Diagram Setup
                    relations_l1_to_l2 = {'Structure': [ 'BH', 'IH', 'Cga', 'GL']}
                    relations_l2_to_l3 = {
                        'Prices (p_)': ['Prices_CTI','Prices_CTE'],
                        'Objects (n_)': ['Objects_CTI','Objects_CTE'],
                        'C.High (hc_)': ['C.High_CTE', 'C.High_CTI'],
                        'Materials (m_)': ['Materials_CTE', 'Materials_CTI']
                    }
                    relations_l3_to_l4 = {
                        'Prices_CTE': ['Prices_+_CTE', 'Prices_gen_CTE', 'Prices_-_CTE'],
                        'Objects_CTE': ['Objects_+_CTE', 'Objects_gen_CTE','Objects_-_CTE'],
                        'C.High_CTE': ['C.High_-_CTE', 'C.High_+_CTE', 'C.High_gen_CTE'],
                        'Materials_CTE': ['Materials_-_CTE', 'Materials_+_CTE', 'Materials_gen_CTE'],
                        
                        'Prices_CTI': ['Prices_+_CTI', 'Prices_gen_CTI','Prices_-_CTI'],
                        'Objects_CTI': ['Objects_-_CTI', 'Objects_+_CTI', 'Objects_gen_CTI'],
                        'C.High_CTI': ['C.High_+_CTI', 'C.High_gen_CTI', 'C.High_-_CTI'],
                        'Materials_CTI': ['Materials_-_CTI', 'Materials_+_CTI', 'Materials_gen_CTI'],
                    }
                    relations_l4_to_l5 = {
                        'Prices_gen_CTE': ['Content'],
                        'Objects_gen_CTE': ['Content'],
                        'C.High_-_CTE': ['Content'],
                        'C.High_gen_CTE': ['Content'],
                        'Materials_-_CTE': ['Content'],
                        'Materials_+_CTE': ['Content'],
                        'Materials_gen_CTE': ['Content'],
                        
                        'Objects_+_CTE': ['Content'],
                        'Prices_+_CTE': ['Content'],
                        'C.High_+_CTE': ['Content'],
                        'Prices_-_CTE': ['Content'],
                        'Objects_-_CTE': ['Content'],
                        
                        
                        'Prices_-_CTI': ['Continent'],
                        'Prices_+_CTI': ['Continent'],
                        'Objects_-_CTI': ['Continent'],
                        'Objects_gen_CTI': ['Continent'],
                        'C.High_+_CTI': ['Continent'],
                        'C.High_gen_CTI': ['Continent'],
                        'Materials_-_CTI': ['Continent'],
                        'Materials_+_CTI': ['Continent'],
                        
                        'Materials_gen_CTI': ['Continent'],
                        'Objects_+_CTI': ['Continent'],
                        'Prices_gen_CTI': ['Continent'],
                        'C.High_-_CTI': ['Continent'],
                    }
                    relations_l5_to_l6 = {
                        'Content': ['CLO',  'DEC', 'FAD', 'HHB', 'INS', 'OTH', 'TOO',      'ENG','SPE','COM','ELE','HHG','LEI','VEH','APP','FUR'],
                        'Continent': ['PUM', 'CLE', 'DHU', 'FRI', 'SKT',  'EXF', 'ETP', 'ITP', 'ELS',     'WND','RDR','SOI','PLG','PRW']
                    }
                    
                    custom_scale_parameters = {'data_pct': 0.5, 'canvas_pct': 0.9}
                    desired_data_ticks = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])

                    data_pct = custom_scale_parameters['data_pct']
                    canvas_pct = custom_scale_parameters['canvas_pct']

                    scaled_yticks = np.where(
                        desired_data_ticks <= data_pct,
                        (canvas_pct / data_pct) * desired_data_ticks,
                        canvas_pct + ((1.0 - canvas_pct) / (1.0 - data_pct)) * (desired_data_ticks - data_pct)
                    ).tolist()

                    tick_labels = [str(val) for val in desired_data_ticks]
                    
                    nodes, links = DataPreparator.generate_scaled_multilayer_sankey(
                        df_list=[df_expected_l1_all, df_expected_l2_all, df_expected_l3_all, df_expected_l4_all, df_expected_l5_all, df_expected_l6_all],
                        relations_list=[relations_l1_to_l2, relations_l2_to_l3, relations_l3_to_l4, relations_l4_to_l5, relations_l5_to_l6],
                        CODES = CODES,
                        colors_dict=gsa_group_colors,
                        scale_params=custom_scale_parameters
                    )
                    for node in nodes:
                        if clean:
                            node['name'] = ""
                        else:
                            node['name'] = re.sub(r'\s*(L\d+)', '', node['name']).strip()

                    ## Prepare Plot
                    print(f"\n[STAGE 5. GSA] Preparing layout...")
                    # Plotter
                    plotter = Plotter(
                        config_paths=PATHS,
                        config_codes=CODES,
                        config_return_periods=RETURN_PERIODS
                    )
                    
                    # Layout
                    layout_params = {
                        'layout': 'mosaic',
                        'mosaic_structure': [
                            ["a", "c"],
                            ["b", "c"],],
                        'figsize': (6, 8),
                        'kwargs': {'gridspec_kw': {
                            'wspace': 0.1,
                            'hspace': 0.2,
                            'width_ratios': [0.4, 0.6],
                            'height_ratios': [0.6, 0.4]}},
                        'adjust': {'bottom': 0.1, 'top': 0.9, 'left': 0.1, 'right': 0.9}
                    }
                    dict_to_plot = {}
                    
                    # Plots
                    print(f"\n[STAGE 5. GSA] Preparing plots...")
                    # a
                    plotter.init_dic(dict_to_plot, 'a', 'plots')
                    bar_colors_b = []
                    for grp in df_expected_l3_all.index:
                        grp_str = str(grp)
                        # Base color from the prefix (e.g., Prices, Materials)
                        base_name = grp_str.split('_CT')[0].strip()
                        bar_colors_b.append(gsa_group_colors.get(base_name, '#cccccc'))
                    dict_to_plot['a']['plots'].append({
                        'plot_type': 'barh',
                        'y': df_expected_l3_all.index.tolist(),
                        'width': df_expected_l3_all['Q50'].values,
                        'color': bar_colors_b,
                        'alpha': 1,
                        'reverse': True
                    })
                    if not clean:
                        for idx, row in df_expected_l3_all.iterrows():
                            dict_to_plot['a']['plots'].append({
                                'plot_type': 'text', 'x': row['Q50'], 'y': idx,
                                'text': f" {row['Q50_share_%']:.2f}%", 'transform': 'data',
                                'va': 'center', 'ha': 'left', 'fontsize': 8, 'color': 'black'
                            })

                    # b
                    plotter.init_dic(dict_to_plot, 'b', 'plots')
                    share_cols = [col for col in df_evolution_l1_all.columns if col.startswith('Share_%_RP')]
                    rps_numeric = [int(col.replace('Share_%_RP', '')) for col in share_cols]
                    sorted_idx = np.argsort(rps_numeric)
                    share_cols_sorted = [share_cols[i] for i in sorted_idx]
                    rps_categorical = [str(rps_numeric[i]) for i in sorted_idx]
                    for feature_name, row in df_evolution_l1_all.iterrows():
                        clean_feat = re.sub(r'\s*\([^)]*\)', '', str(feature_name)).strip()
                        feat_color = gsa_group_colors.get(clean_feat, '#cccccc')
                        y_shares = row[share_cols_sorted].values
                        dict_to_plot['b']['plots'].append({
                            'plot_type': 'line', 
                            'x': rps_categorical, 
                            'y': y_shares, 
                            'color': feat_color, 
                            'marker': 'o', 
                            'markersize': 3, 
                            'linewidth': 1.5, 
                            'label': feature_name
                        })

                    # c
                    plotter.init_dic(dict_to_plot, 'c', 'plots')
                    dict_to_plot['c']['plots'].append({
                        'plot_type': 'sankey',
                        'node_width': 0.02,
                        'fontsize': 8,
                        'nodes': nodes,
                        'links': links,
                        'edgecolor': 'white',
                        "linewidth": 0.5,
                    })

                    # Style
                    print(f"\n[STAGE 5. GSA] Preparing style...")
                    # a
                    plotter.init_dic(dict_to_plot, 'a', 'style')
                    x_max = df_expected_l2_all['Q50'].max()
                    x_split1 = 1
                    x_split2 = 10
                    x_pct1 = 0.05
                    x_pct2 = 0.10
                    dict_to_plot['a']['style'].extend([
                        {'style_type': 'xscale', 'value': 'function', 'functions': plotter.create_3_linear_ax_scale(x_max, x_split1, x_split2, x_pct1, x_pct2)},
                        {'style_type': 'xticks', 'ticks': [0, 1, 10, 50, 100, 150, 200, 250, 300]},
                        {'style_type': 'xticklabels', 'labels': ['', '', '', '', '', '', '', '', '']},
                    ])
                    if clean:
                        dict_to_plot['a']['style'].extend([
                            {'style_type': 'yticklabels', 'labels': []},
                            {'style_type': 'ticks_params', 'labelleft': False}
                        ])
                    else:
                        dict_to_plot['a']['style'].extend([
                            {'style_type': 'xlabel', 'label': 'Expected Annual Average |SHAP|'},
                            {'style_type': 'xticklabels', 'labels': ['0', '1', '10', '50', '100', '150', '200', '250', '300']},
                        ])
                        
                    # b
                    plotter.init_dic(dict_to_plot, 'b', 'style')
                    x_max = 75
                    x_split1 = 10
                    x_split2 = 50
                    x_pct1 = 0.50
                    x_pct2 = 0.80
                    dict_to_plot['b']['style'].extend([
                        {'style_type': 'yscale', 'value': 'function', 'functions': plotter.create_3_linear_ax_scale(x_max, x_split1, x_split2, x_pct1, x_pct2)},
                        {'style_type': 'yticks', 'ticks': [0, 10, 20, 30, 40, 50, 60, 70]},
                        {'style_type': 'yticklabels', 'labels': ['', '', '', '', '', '', '', '']},
                        {'style_type': 'ylim', 'ymin': 0, 'ymax': 75}
                    ])
                    if clean:
                        dict_to_plot['b']['style'].extend([
                            {'style_type': 'xticklabels', 'labels': []},
                            {'style_type': 'ticks_params', 'labelbottom': False}
                        ])
                    else:
                        dict_to_plot['b']['style'].extend([
                            {'style_type': 'xlabel', 'label': 'Return Period (RP)'},
                            {'style_type': 'ylabel', 'label': 'Share % of total |SHAP|'},
                            {'style_type': 'yticklabels', 'labels': ['0', '10', '', '30', '', '50', '', '70']},
                        ])

                    # c
                    plotter.init_dic(dict_to_plot, 'c', 'style')
                    dict_to_plot['c']['style'].extend([
                        {'style_type': 'xlim', 'xmin': 0, 'xmax': 1.0},
                        {'style_type': 'ylim', 'ymin': 0.0, 'ymax': 1.0},
                        {'style_type': 'yticks', 'ticks': scaled_yticks},
                        {'style_type': 'yticklabels', 'labels': tick_labels},
                        {'style_type': 'grid', 'visible': True, 'axis': 'y', 'linestyle': '--', 'alpha': 0.5, 'color': 'gray', 'zorder': 0},
                        {'style_type': 'grid', 'visible': False, 'axis': 'x', 'linestyle': '--', 'alpha': 0, 'color': 'white', 'zorder': 0},
                        {'style_type': 'spines', 'top': False, 'right': False, 'left': False, 'bottom': False},
                    ])
                    if clean:
                        dict_to_plot['c']['style'].extend([
                            {'style_type': 'ticks_params', 'left': False, 'bottom': False, 'labelleft': False, 'labelbottom': False}
                        ])
                    else:
                        dict_to_plot['c']['style'].extend([
                            {'style_type': 'ticks_params', 'left': False, 'bottom': False, 'labelleft': True, 'labelbottom': False}
                        ])
                        
                    # Save
                    print(f"\n[STAGE 5. GSA] Preparing saving...")
                    gsa_target_path = Path(PATHS['gsa'])
                    save_params = {
                        'output_dir': str(gsa_target_path),  # Saves in output/gsa/ folder dynamically
                        'subfolder': '',
                        'fname': f"{gsa_target_path.name}_charts_v{fig_version}.png",
                        'show': False,
                        'dpi': 300,
                    }

                    # 3. Execute Plot Rendering via the Engine
                    print(f"\n[STAGE 5. GSA] Running plot any...")
                    plotter.plot_any(layout_params, dict_to_plot, save_params)

                plot_gsa_charts_sensitivity_analysis(fig_version="4.4", clean=False)
                plot_gsa_charts_sensitivity_analysis(fig_version="4.4.c", clean=True)
            
            # Map
            RUN = False
            if RUN:
                def plot_gsa_maps_sensitivity_analysis(fig_version, clean=False):
                    """
                    Builds the configuration dictionaries for the GSA mosaic plot and passes them
                    to the Plotter engine.
                    """
                    
                    ## Prepare data
                    print(f"\n[STAGE 5. GSA] Preparing data...")
                    # 1. Load Needed Data
                    gdf_buildings = DataPreparator.load_flooded_buildings_gdf(PATHS['buildings_shp'], PATHS['depth_samples_pkl'])
                    
                    # 4. Load Expected SHAP Data
                    df_expected_l3_bid = DataPreparator.calc_shap_expected_grouped(output_gsa_dir, RPs_unq, level=3, by_bid=True, min_rp_AEP_0=2)
                    
                    # 7. Map Setup & Geographic Combos
                    df_filtered = df_expected_l3_bid.reset_index()
                    k_vars = len(df_filtered["Feature_Group"].unique())
                    cumulative_threshold = 90
                    acronym_map = {'he': 'he', 'C.High (hc_)': 'hc', 'Objects (n_)': 'n', 'Structure': 'S', 'Prices (p_)': 'p', 'ed': 'ed', 'Materials (m_)': 'm'}
                    df_filtered = df_filtered[df_filtered['Q50'] > 0].copy()
                    df_filtered = df_filtered.dropna(subset=['Q50'])
                    df_filtered = df_filtered.sort_values(by=['BID', 'Q50_share_%'], ascending=[True, False])
                    df_filtered['cumsum_share'] = df_filtered.groupby('BID')['Q50_share_%'].cumsum()
                    df_filtered['prev_cumsum'] = df_filtered['cumsum_share'] - df_filtered['Q50_share_%']
                    df_filtered = df_filtered[df_filtered['prev_cumsum'] < cumulative_threshold]
                    
                    df_map = df_filtered.groupby('BID').head(k_vars).copy()
                    df_map['Short_Feature'] = df_map['Feature_Group'].map(lambda x: acronym_map.get(str(x).strip(), str(x).strip()))

                    combinations = df_map.groupby('BID')['Short_Feature'].apply(lambda x: ' | '.join(x)).reset_index()
                    combinations.rename(columns={'Short_Feature': f'Top_{cumulative_threshold}%_Combo'}, inplace=True)
                    df_combo_counts = combinations[f'Top_{cumulative_threshold}%_Combo'].value_counts().reset_index()
                    df_combo_counts.columns = [f'Top_{cumulative_threshold}%_Combo', 'BID_count']
                    unique_combos = df_combo_counts[f'Top_{cumulative_threshold}%_Combo'].unique()
                    
                    def sort_hierarchical_combos(combos_list):
                        """
                        Sorts combinations hierarchically by frequency at each position, 
                        placing longer combinations before shorter ones sharing the same prefix.
                        """
                        split_combos = [c.split(' | ') for c in combos_list]
                        
                        def _sort_recursive(items):
                            if not items:
                                return []
                            
                            counts = {}
                            for item in items:
                                if len(item) > 0:
                                    counts[item[0]] = counts.get(item[0], 0) + 1
                                    
                            sorted_keys = sorted(counts.keys(), key=lambda x: (-counts[x], x))
                            
                            result = []
                            for key in sorted_keys:
                                continuations = [item[1:] for item in items if len(item) > 0 and item[0] == key and len(item) > 1]
                                terminating = [item for item in items if len(item) > 0 and item[0] == key and len(item) == 1]
                                
                                sorted_continuations = _sort_recursive(continuations)
                                
                                for sub in sorted_continuations:
                                    result.append([key] + sub)
                                for _ in terminating:
                                    result.append([key])
                                    
                            return result
                        
                        sorted_splits = _sort_recursive(split_combos)
                        
                        # Remove duplicates preserving order
                        seen = set()
                        final_combos = []
                        for c in sorted_splits:
                            combo_str = ' | '.join(c)
                            if combo_str not in seen:
                                seen.add(combo_str)
                                final_combos.append(combo_str)
                                
                        return final_combos
                    sorted_unique_combos = sort_hierarchical_combos(df_combo_counts[f'Top_{cumulative_threshold}%_Combo'].unique())
                    df_combo_counts[f'Top_{cumulative_threshold}%_Combo'] = pd.Categorical(
                        df_combo_counts[f'Top_{cumulative_threshold}%_Combo'], 
                        categories=sorted_unique_combos, 
                        ordered=True
                    )
                    df_combo_counts = df_combo_counts.sort_values(by=f'Top_{cumulative_threshold}%_Combo').reset_index(drop=True)
                    total_BIDS = df_combo_counts["BID_count"].sum()
                    df_combo_counts["BID_count_%"] = round((df_combo_counts["BID_count"] / total_BIDS * 100),1)
                    
                    num_combos = len(sorted_unique_combos)
                    cmap_name = 'tab10' if num_combos <= 10 else 'tab20'
                    cmap = plt.get_cmap(cmap_name)
                    color_map = {combo: mcolors.to_hex(cmap(i % cmap.N)) for i, combo in enumerate(sorted_unique_combos)}
                    combinations[f'Top_{cumulative_threshold}%_Combo'] = pd.Categorical(
                        combinations[f'Top_{cumulative_threshold}%_Combo'], 
                        categories=sorted_unique_combos, 
                        ordered=True
                    )
                    combinations['combo_color'] = combinations[f'Top_{cumulative_threshold}%_Combo'].map(color_map)
                    
                    # Merge geometries and safely fill missing colors to prevent RGBA crash
                    gdf_shap = gdf_buildings.merge(combinations, on='BID', how='inner')
                    
                    gdf_shap = gdf_shap.merge(df_map[df_map["Feature_Group"]=="he"], on='BID', how='inner')
                    
                    gdf_shap_centroids = gdf_shap.copy()
                    gdf_shap_centroids['geometry'] = gdf_shap.centroid
                    gdf_shap_centroids_4326 = gdf_shap_centroids.to_crs(epsg=4326)

                    minx, miny, maxx, maxy = gdf_shap_centroids_4326.total_bounds
                    buffer = 0.01
                    plot_extent = [minx - buffer, maxx + buffer, miny - buffer, maxy + buffer]

                    ## Prepare Plot
                    print(f"\n[STAGE 5. GSA] Preparing layout...")
                    # Plotter
                    plotter = Plotter(
                        config_paths=PATHS,
                        config_codes=CODES,
                        config_return_periods=RETURN_PERIODS
                    )
                    
                    # Layout
                    layout_params = {
                        'layout': 'mosaic',
                        'mosaic_structure': [["a"],["b"]],
                        'figsize': (6, 8),
                        'kwargs': {'gridspec_kw': {'hspace': 0.15}},
                        'adjust': {'bottom': 0.1, 'top': 0.9, 'left': 0.1, 'right': 0.9}
                    }
                    dict_to_plot = {}
                    
                    # Plots
                    print(f"\n[STAGE 5. GSA] Preparing plots...")
                    # Calculate extents early so we can place the text legend inside the box
                    real_ratio = plotter.get_ax_ratio(layout_params, 'a')
                    top_left_x = (4, 42, 45)
                    top_left_y = (40, 25, 00)
                    zoom_pct = 120
                    calc_extent_vals = plotter.calc_map_extent(top_left_x, top_left_y, zoom_pct, real_ratio)
                    xmin, xmax, ymin, ymax = calc_extent_vals

                    # a
                    plotter.init_dic(dict_to_plot, 'a', 'plots')
                    dict_to_plot['a']['plots'].extend([
                        {'plot_type': 'wms',
                            'url': 'https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS',
                            'layers': ['OI.OrthoimageCoverage'],
                            'extent': plot_extent,
                            'crs': 'EPSG:4326',
                            'size': (1000, 1000),
                            'alpha': 1,
                            'zorder': 1
                        },
                        {'plot_type': 'gdf_shp',
                            'gdf': gdf_shap_centroids_4326,
                            'color': gdf_shap_centroids_4326['combo_color'].tolist(),
                            'markersize': 2,
                            'alpha': 1,
                            'zorder': 3
                        }
                    ])
                    if not clean:
                        xmin, xmax, ymin, ymax = plot_extent
                        legend_x = xmin + (xmax - xmin) * 0.02
                        legend_y = ymax - (ymax - ymin) * 0.02
                        y_step = (ymax - ymin) * 0.04
                        
                        dict_to_plot['a']['plots'].append({
                            'plot_type': 'text', 'x': legend_x, 'y': legend_y,
                            'text': "Top 7 Variables Rank:", 'fontsize': 8, 'color': 'white',
                            'ha': 'left', 'va': 'top', 'transform': 'data', 'fontweight': 'bold',
                            'path_effects': [pe.withStroke(linewidth=2, foreground="black")], 'zorder': 5
                        })
                        
                        # Changed from unique_combos to sorted_unique_combos
                        for i, combo in enumerate(sorted_unique_combos):
                            combo_color = color_map[combo]
                            current_y = legend_y - ((i + 1) * y_step)
                            
                            dict_to_plot['a']['plots'].append({
                                'plot_type': 'text',
                                'x': legend_x,
                                'y': current_y,
                                'text': f"■ {combo}",
                                'color': combo_color,
                                'fontsize': 6,
                                'ha': 'left',
                                'va': 'top',
                                'transform': 'data',
                                'path_effects': [pe.withStroke(linewidth=1, foreground="black")],
                                'zorder': 5
                            })

                    # b
                    plotter.init_dic(dict_to_plot, 'b', 'plots')
                    show_legend_vis = not clean
                    dict_to_plot['b']['plots'].extend([
                        {'plot_type': 'wms',
                            'url': 'https://www.ign.es/wms-inspire/pnoa-ma?request=GetCapabilities&service=WMS',
                            'layers': ['OI.OrthoimageCoverage'],
                            'extent': plot_extent,
                            'crs': 'EPSG:4326',
                            'size': (1000, 1000),
                            'alpha': 1,
                            'zorder': 1
                        },
                        {'plot_type': 'gdf_shp',
                            'gdf': gdf_shap_centroids_4326,
                            'column': 'Q50',
                            'cmap': 'viridis',
                            'markersize': 2,
                            'alpha': 1,
                            'zorder': 3,
                            'legend': show_legend_vis,
                        }
                    ])
                        
                    # Style
                    print(f"\n[STAGE 5. GSA] Preparing style...")
                    # a
                    plotter.init_dic(dict_to_plot, 'a', 'style')
                    label_vis = not clean  # Determines if top/left labels are shown
                    dict_to_plot['a']['style'].extend([
                        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3, 'zorder': 1},
                        {'style_type': 'xlim', 'left': calc_extent_vals[0], 'right': calc_extent_vals[1]},
                        {'style_type': 'ylim', 'bottom': calc_extent_vals[2], 'top': calc_extent_vals[3]},
                        {'style_type': 'aspect', 'aspect': 'equal'},
                        {'style_type': 'spines', 'top': True, 'right': True, 'left': True, 'bottom': True},
                        {'style_type': 'ticks_params', 'which': 'both', 'labelsize': 8,
                            'top': True, 'bottom': False, 'left': True, 'right': False,
                            'labeltop': label_vis, 'labelbottom': False, 'labelleft': label_vis, 'labelright': False},
                        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90},
                        {'style_type': 'yticklabels', 'va': 'center'},
                        {'style_type': 'major_formatter', 'axis': 'y', 'formatter_type': 'dms_suffix', 'suffix': ' N'},
                        {'style_type': 'major_formatter', 'axis': 'x', 'formatter_type': 'dms_suffix', 'suffix': ' W'},
                    ])
                    if clean:
                        dict_to_plot['a']['style'].extend([
                            {'style_type': 'xticklabels', 'labels': []},
                            {'style_type': 'yticklabels', 'labels': []}
                        ])
                    else:
                        dict_to_plot['a']['style'].extend([
                            {'style_type': 'legend', 'loc': 'upper right', 'fontsize': 5, 'title': 'Top 7 Variables Rank'}
                        ])

                    # b
                    plotter.init_dic(dict_to_plot, 'b', 'style')
                    label_vis = not clean  # Determines if top/left labels are shown
                    dict_to_plot['b']['style'].extend([
                        {'style_type': 'grid', 'visible': True, 'linestyle': '--', 'alpha': 0.3, 'zorder': 1},
                        {'style_type': 'xlim', 'left': calc_extent_vals[0], 'right': calc_extent_vals[1]},
                        {'style_type': 'ylim', 'bottom': calc_extent_vals[2], 'top': calc_extent_vals[3]},
                        {'style_type': 'aspect', 'aspect': 'equal'},
                        {'style_type': 'spines', 'top': True, 'right': True, 'left': True, 'bottom': True},
                        {'style_type': 'ticks_params', 'which': 'both', 'labelsize': 8,
                            'top': True, 'bottom': False, 'left': True, 'right': False,
                            'labeltop': label_vis, 'labelbottom': False, 'labelleft': label_vis, 'labelright': False},
                        {'style_type': 'ticks_params', 'axis': 'y', 'labelrotation': 90},
                        {'style_type': 'yticklabels', 'va': 'center'},
                        {'style_type': 'major_formatter', 'axis': 'y', 'formatter_type': 'dms_suffix', 'suffix': ' N'},
                        {'style_type': 'major_formatter', 'axis': 'x', 'formatter_type': 'dms_suffix', 'suffix': ' W'},
                    ])
                    if clean:
                        dict_to_plot['b']['style'].extend([
                            {'style_type': 'xticklabels', 'labels': []},
                            {'style_type': 'yticklabels', 'labels': []}
                        ])
                    else:
                        dict_to_plot['b']['style'].extend([
                            {'style_type': 'legend', 'loc': 'upper right', 'fontsize': 5, 'title': 'Top 7 Variables Rank'}
                        ])
                    
                    # letter
                    if not clean:
                        print(f"\n[STAGE 5. GSA] Preparing letters...")
                        for key in dict_to_plot.keys():
                            if key not in ['leg']:
                                y_pos = 1.05 if key in ['a', 'd'] else 1.1
                                x_pos = -0.05 if key in ['b', 'c', 'd'] else -0.01
                                dict_to_plot[key]['style'].append({
                                    'style_type': 'letter', 
                                    'label': f'({key})',
                                    'x': x_pos,
                                    'y': y_pos
                                })
                    
                    # Save
                    print(f"\n[STAGE 5. GSA] Preparing saving...")
                    gsa_target_path = Path(PATHS['gsa'])
                    save_params = {
                        'output_dir': str(gsa_target_path),
                        'subfolder': '',
                        'fname': f"{gsa_target_path.name}_maps_v{fig_version}.png",
                        'show': False,
                        'dpi': 300,
                    }

                    # 3. Execute Plot Rendering via the Engine
                    print(f"\n[STAGE 5. GSA] Running plot any...")
                    plotter.plot_any(layout_params, dict_to_plot, save_params)

                plot_gsa_maps_sensitivity_analysis(fig_version="4.4", clean=False)
                plot_gsa_maps_sensitivity_analysis(fig_version="4.4.c", clean=True)
            
if __name__ == "__main__":
    run_pipeline()
    
# TEST in terminal: python -m main

# TESTS in VS
