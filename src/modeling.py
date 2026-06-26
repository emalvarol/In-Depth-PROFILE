"""
In-Depth-PROFILE HEC-RAS Monte Carlo Module
Handles automated, parallelized execution of HEC-RAS hydraulic models
integrated with statistical flood frequency distributions and spatial analysis

Requirements:
1- A functional proyects must be created first on HEC-RAS
2- Restart Files should be created mannually to ensure good results
They must be renamed as "RST_n.rst" where n is the flow stabilized
3- Once the hecral models is approved, it must be prepared to automattically
compute and save outputs wsl tiff for depht (this simplify the use of ras commander).
The use and creation of restart files must be switched off (ras commander will create it)


Note: On this work the same project, gemometry and plan (labeled as "01") was used to run
all the deterministic and stochastic simulations, and all intermediate simulations to
generate restart files. Further only one file for unsteady, geometry, etc was
created to simplify the use of hecras commander

Note 2: This module process spatial data. Spatial data is tricky when using python since it
relay on C++ (GDAL libraries). If you have any problem check all your shp have a correct
geometry, otherwise fix it (e.g.: tool "Repair geometry" on ArcMAP)
"""

import os
import sys
import shutil
import stat
import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import pearson3, qmc
from scipy.optimize import minimize
from tqdm import tqdm
import multiprocess
from rasterstats import zonal_stats

# Project-specific imports
from ras_commander import init_ras_project, RasCmdr, RasUnsteady, RasProcess

logging.getLogger("ras_commander").setLevel(logging.WARNING)

class HydrologicalFitter:
    """
    Handles Log-Pearson Type III distribution fitting, parameter extraction,
    and Latin Hypercube Sampling (LHS) with truncation for flood frequency analysis.
    """
    def __init__(self, config_paths):
        """
        Initializes the fitter with centralized framework paths.
        """
        self.paths = config_paths
        self.output_path = self.paths['fm_flow_xlsx']

    @staticmethod
    def _fit_pearson3(x_vals, tail_percentile=0.05):
        """
        Fits a Pearson Type III distribution parameters (skew, loc, scale) 
        to log10-transformed flow values using Nelder-Mead optimization.
        """
        y_vals = np.log10(x_vals)
        probs = [tail_percentile, 0.50, 1 - tail_percentile]

        def objective(params):
            skew, loc, scale = params
            if scale <= 0: 
                return 1e9
            quantiles = pearson3.ppf(probs, skew, loc, scale)
            return np.sum((quantiles - y_vals)**2)

        initial_guess = [0.1, np.mean(y_vals), np.std(y_vals)]
        res = minimize(objective, initial_guess, method='Nelder-Mead')
        return res.x

    def fit_flood_frequency(self, return_periods_dict):
        """
        Fits Pearson III distributions for all return periods specified in the configuration,
        appends historical/legacy Return Period 2 distribution settings, and exports to Excel.
        
        Args:
            return_periods_dict (dict): Dictionary mapping return periods to flow records.
        """
        print("Starting Hydrological Frequency Analysis Fitting...")
        results = []
        
        # Fit distributions for configured return periods
        for rp, flows in tqdm(return_periods_dict.items(), desc="Fitting Distributions"):
            skew, loc, scale = self._fit_pearson3(flows)
            results.append({
                'ReturnPeriod': rp,
                'Skew': skew,
                'Loc': loc,
                'Scale': scale
            })
            
        dist_depth_df = pd.DataFrame(results)
        
        # Dynamic directory creation and export using pathlib
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        dist_depth_df.to_excel(self.output_path, index=False)
        print(f"Hydrological distribution mapping exported successfully to: {self.output_path}")
        
        return dist_depth_df

class GeospatialProcessor:
    """
    Handles spatial raster-vector interactions, specifically calculating baseline 
    building elevations (DEM) and extracting water surface elevations (WSE) per asset.
    """
    def __init__(self, config_paths):
        """
        Initializes the geospatial processor with framework data paths.
        """
        self.paths = config_paths

    def prepare_building_dem_medians(self):
        """
        Calculates the median DEM altitude for each building polygon and saves an updated Shapefile.
        Executed only if the target file does not exist yet.
        """
        target_shp = self.paths['buildings_sample_med_shp']
        input_shp = self.paths['buildings_sample_shp']
        terrain_tif = self.paths['terrain_tif']

        if not target_shp.exists():
            print("\nStarting Pre-analysis for Buildings (DEM Median Calculation)...")
            
            print("Loading building shapefile...")
            gdf = gpd.read_file(input_shp)

            print("Calculating DEM Medians via zonal_stats...")
            stats = zonal_stats(gdf, terrain_tif, stats="median", nodata=np.nan)
            
            gdf['med_dem'] = [x['median'] for x in stats]
            
            target_shp.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(target_shp, driver='ESRI Shapefile')

            print(f"Median DEM successfully calculated for {len(gdf)} buildings.")
            print(f"Target verification path: {target_shp}")
            
        else:
            print(f"\n[SKIPPED] Median DEM already calculated and saved at: {target_shp}")
            gdf = gpd.read_file(target_shp)
            
        return gdf

    @staticmethod
    def extract_building_depths(wsl_tif_path, gdf):
        """
        Extracts hydraulic metrics (WSE and water depth 'he') for each building from a maximum WSE raster.
        Replaced rasterio mask with rasterstats zonal_stats for stability.
        """
        all_metrics = []
        
        # Calculate mean WSE for all polygons in the gdf
        stats = zonal_stats(gdf, wsl_tif_path, stats="mean", nodata=np.nan)

        for (idx, row), stat in zip(gdf.iterrows(), stats):
            building_id = row['BID']
            avg_wsl = stat['mean']
            
            if avg_wsl is not None and not np.isnan(avg_wsl):
                he = max(0, round(float(avg_wsl - row['med_dem']), 2))
            else:
                he = 0
                
            all_metrics.append({
                'BID': building_id,
                'he': he
            })
            
        return all_metrics

class HecRasMonteCarloEngine:
    """
    Orchestrates the parallel HEC-RAS execution pipeline, folder management,
    incremental storage, and global convergence checking (IRME criteria).
    """
    def __init__(self, config_paths, config_return_periods, config_rst_list):
        self.paths = config_paths
        self.return_periods = config_return_periods
        self.rst_values = config_rst_list
        
        # Hardware & Engine execution settings
        self.phy_cores = 16
        self.hr_cores_by_sim = 6
        self.hr_parallel_sims = 8
        
        # Sampling Settings (set according to the flood frequency analyst)
        self.p_lower = 0.05
        self.p_upper = 0.95
        
        # HEC-RAS Model settings
        self.n_rows_unsteady = 7 
        
        # Convergence Settings
        self.threshold = 0.1
        self.z_score = 1.96
        self.max_sims = 1600
               
        # Optimize Numexpr for parallelization
        os.environ['NUMEXPR_MAX_THREADS'] = f'{self.phy_cores}'

    @staticmethod
    def _remove_readonly(func, path, excinfo):
        """Error handler for shutil.rmtree to remove read-only attributes."""
        os.chmod(path, stat.S_IWRITE)
        func(path)
    
    def _manage_worker_folders(self):
        """Cleans and initializes isolated workspaces for worker threads."""
        worker_paths = []
        for i in range(1, self.phy_cores + 1):
            dest_path = os.path.join(self.paths['HR_sample_project'], f'Worker_{i}')
            
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path, onexc=self._remove_readonly)
            
            shutil.copytree(self.paths['HR_base_project'], dest_path)
            worker_paths.append(dest_path)
            
        os.makedirs(self.paths['HR_output_maps'], exist_ok=True)
        return worker_paths

    @staticmethod
    def sample_lhs_truncated(skew, loc, scale, size=16, p_range=(0.05, 0.95)):
        """Generates Latin Hypercube Samples restricted to a probability window."""
        p_min, p_max = p_range
        sampler = qmc.LatinHypercube(d=1)
        u_samples = sampler.random(n=size).flatten()
        p_samples = p_min + u_samples * (p_max - p_min)
        log_samples = pearson3.ppf(p_samples, skew, loc, scale)
        return 10**log_samples

    def _update_worker_unsteady_file(self, worker_path, fw_sample):
        """Injects flow boundaries and assigns the correct HEC-RAS restart file."""
        ras_instance = init_ras_project(worker_path, self.paths['HR_exe'])
        unsteady_path = ras_instance.unsteady_df.loc[0, 'full_path']
        
        lower_rst = max([r for r in self.rst_values if r <= fw_sample], default=min(self.rst_values))
        new_data = pd.DataFrame({'Value': [
            float(lower_rst),                                   # Row 1: Closest restart file flow
            *[round(fw_sample, 2)] * (self.n_rows_unsteady-1)   # Remaining rows: Sampled flow
        ]})

        with open(unsteady_path, 'r') as f:
            lines = f.readlines()
        
        tables = RasUnsteady.identify_tables(lines)
        table_name, start_line, _ = tables[0]
        
        RasUnsteady.write_table_to_file(
            unsteady_file=unsteady_path,
            table_name=table_name,
            df=new_data,
            start_line=start_line
        )

        restart_name = f"RST_{int(lower_rst)}.rst"
        RasUnsteady.update_restart_settings(
            unsteady_file=unsteady_path,
            use_restart=True,
            restart_filename=restart_name,
            ras_object=ras_instance
        )

    def _get_latest_sn(self, return_period):
        """Retrieves the highest Simulation Number (SN) executed for a given Return Period."""
        pkl_path = self.paths['depth_samples_pkl']
        last_sn = 0

        if os.path.exists(pkl_path):
            try:
                df_existing = pd.read_pickle(pkl_path)
                if not df_existing.empty and 'RP' in df_existing.columns:
                    df_rp = df_existing[df_existing['RP'] == return_period]
                    if not df_rp.empty and 'SN' in df_rp.columns:
                        last_sn = int(df_rp['SN'].max())
            except (pd.errors.EmptyDataError, EOFError, KeyError):
                pass
        
        return last_sn
    
    def check_convergence(self, df, rp_key):
        """
        Calculates statistical convergence metrics for a specific Return Period.
        Criteria: SN > 30, IRME == 0, and 30 consecutive stable steps.
        """
        rp_df = df[df['RP'] == rp_key].copy()
        if rp_df.empty:
            return False, rp_df

        # Per-building (BID) cumulative calculation
        rp_df = rp_df.sort_values(by=['BID', 'SN'])
        rp_df['he_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().mean())
        rp_df['SD_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().std(ddof=1))
        rp_df['SE_BID_sn'] = rp_df['SD_BID_sn'] / np.sqrt(rp_df['SN'])
        rp_df['ME_BID_sn'] = self.z_score * rp_df['SE_BID_sn']

        # Residual margin calculation: max(0, ME - Threshold)
        rp_df['Residual'] = (rp_df['ME_BID_sn'] - self.threshold).clip(lower=0)

        # Global Integration by Simulation Number (SN)
        rp_simp = rp_df.groupby('SN').agg(
            IRME=('Residual', 'sum'),
            NON_CONV_COUNT=('Residual', lambda x: (x > 0).sum())
        ).reset_index()

        total_flooded_bids = rp_df[rp_df['he'] > 0]['BID'].nunique()
        if total_flooded_bids > 0:
            rp_simp['REL_NON_CONV'] = (rp_simp['NON_CONV_COUNT'] / total_flooded_bids) * 100
        else:
            rp_simp['REL_NON_CONV'] = 0.0

        # Stability rule evaluation
        rp_simp['MET'] = (rp_simp['SN'] > 30) & (rp_simp['IRME'] == 0)
        blocks = (~rp_simp['MET']).cumsum()
        rp_simp['STC_AG'] = rp_simp.groupby(blocks).cumcount() * rp_simp['MET']
        rp_simp['CONV'] = rp_simp['STC_AG'] >= 30

        is_converged = bool(rp_simp['CONV'].iloc[-1]) if not rp_simp.empty else False
        return is_converged, rp_simp

    def run_monte_carlo(self):
        """
        Main orchestration execution thread. Iterates across hydrological parameters
        and triggers multi-threaded HEC-RAS worker sub-processes.
        """
        print("--------------------------------------------\n")
        print("--------------------------------------------\n")
        print("Initializing HEC-RAS Multi-Engine Monte Carlo Pipeline...")
        print("--------------------------------------------\n")
        print("--------------------------------------------\n")
        gdf_buildings = gpd.read_file(self.paths['buildings_sample_med_shp'])
        dist_depth_df = pd.read_excel(self.paths['fm_flow_xlsx'])
        pkl_path = self.paths['depth_samples_pkl']

        for rp_key in sorted(self.return_periods.keys(), reverse=True):
            print("--------------------------------------------\n")
            print("--------------------------------------------\n")
            print(f"\nProcessing Return Period Execution Group: {rp_key}")
            print("--------------------------------------------\n")
            print("--------------------------------------------\n")
            converged = False
            total_sims_done = self._get_latest_sn(rp_key)

            while not converged and total_sims_done < self.max_sims:
                total_sims_done = self._get_latest_sn(rp_key)
                worker_paths = self._manage_worker_folders()

                # Generate sampling arrays
                p_dist = dist_depth_df[dist_depth_df['ReturnPeriod'] == rp_key].iloc[0]
                fw_samples = self.sample_lhs_truncated(
                    p_dist['Skew'], p_dist['Loc'], p_dist['Scale'], size=self.phy_cores, p_range=(self.p_lower, self.p_upper))

                # Prepare parameters and boundary condition files per thread
                last_sn = self._get_latest_sn(rp_key)
                task_inputs = []
                for i, path in enumerate(worker_paths):
                    self._update_worker_unsteady_file(path, fw_samples[i])
                    task_inputs.append({
                        'worker_path': path,
                        'SN': last_sn + i + 1,
                        'RP': rp_key,
                        'IF': fw_samples[i],
                        'ras_exe': self.paths['HR_exe'],
                        'num_cores': self.hr_cores_by_sim,
                        'gdf': gdf_buildings
                    })

                # Process concurrent execution blocks
                all_results_list = []
                for idx in range(0, len(task_inputs), self.hr_parallel_sims):
                    block = task_inputs[idx : idx + self.hr_parallel_sims]
                    with multiprocess.Pool(processes=len(block)) as pool:
                        block_results = pool.map(_mp_hydraulic_worker, block)
                        for sim_metrics in block_results:
                            all_results_list.extend(sim_metrics)

                # Relocate and format spatial rasters (.tif outputs)
                for i, path in enumerate(worker_paths):
                    sim_sn = last_sn + i + 1
                    src_tif = os.path.join(path, '01', f"WSE (Max).Terrain.{os.path.basename(self.paths['terrain_tif'])}")
                    dest_tif = os.path.join(self.paths['HR_output_maps'], f"WSE_RP{rp_key}_SN{sim_sn}.tif")
                    if os.path.exists(src_tif):
                        shutil.move(src_tif, dest_tif)

                # Save intermediate updates incrementally to storage
                new_df = pd.DataFrame(all_results_list)
                if os.path.exists(pkl_path):
                    old_df = pd.read_pickle(pkl_path)
                    sample_df = pd.concat([old_df, new_df], ignore_index=True)
                else:
                    sample_df = new_df
                sample_df.to_pickle(pkl_path)

                # Verification evaluation step
                converged, _ = self.check_convergence(sample_df, rp_key)
                total_sims_done = self._get_latest_sn(rp_key)
                print("--------------------------------------------\n")
                print("--------------------------------------------\n")
                print(f"RP={rp_key} Progress. Total Simulated: {total_sims_done}. Converged={converged}")
                print("--------------------------------------------\n")
                print("--------------------------------------------\n")
        
        print("HEC-RAS Monte Carlo Engine executed and converged successfully.")
   
# ==============================================================================
# GLOBAL MULTIPROCESSING WORKER
# Must remain at module level for pickling compatibility
# ==============================================================================
def _mp_hydraulic_worker(input_dict):
    """
    Independent worker function triggered inside the multiprocessing pool.
    Runs a single simulation and computes spatial depth metrics on the fly.
    """
    # Force localized imports inside the worker for interactive shell compatibility
    import numpy as np
    from rasterstats import zonal_stats
    from ras_commander import init_ras_project, RasCmdr, RasProcess

    worker_path = input_dict['worker_path']
    sim_id = input_dict['SN']
    return_period = input_dict['RP']
    fw_sample = input_dict['IF']
    ras_exe = input_dict['ras_exe']
    num_cores = input_dict['num_cores']
    gdf = input_dict['gdf']

    # 1. Initialize RAS for the specific worker
    ras_worker = init_ras_project(worker_path, ras_exe)
    
    # 2. Run RAS Simulation
    _ = RasCmdr.compute_plan("01", num_cores=num_cores, ras_object=ras_worker)

    # 3. Process max WSE result
    map_results = RasProcess.store_maps(
        plan_number="01",
        wse=True,
        depth=False,
        velocity=False,
        profile="Max",
        ras_object=ras_worker
    )
    wsl_tif_path = map_results['wse'][0]

    # 4. Calculate depth metrics for buildings using zonal_stats
    all_metrics = []
    stats = zonal_stats(gdf, wsl_tif_path, stats="mean", nodata=np.nan)

    for (idx, row), stat in zip(gdf.iterrows(), stats):
        building_id = row['BID']
        avg_wsl = stat['mean']
        
        if avg_wsl is not None and not np.isnan(avg_wsl):
            he = max(0, round(float(avg_wsl - row['med_dem']), 2))
        else:
            he = 0
            
        all_metrics.append({
            'SN': sim_id,
            'RP': return_period,
            'BID': building_id,
            'IF': round(float(fw_sample), 2),
            'he': he
        })
            
    return all_metrics

if __name__ == "__main__":
    from src.config import PATHS, RETURN_PERIODS, RST
    
    # 1. Initialize and execute the fitter sub-module
    fitter = HydrologicalFitter(PATHS)
    fitted_parameters_df = fitter.fit_flood_frequency(RETURN_PERIODS)
    
    # 2. Initialize and execute geoespatial sub-module
    geo_processor = GeospatialProcessor(PATHS)
    gdf_buildings = geo_processor.prepare_building_dem_medians()
    
    # 3. Initialize and execute the Monte Carlo sub-module
    engine = HecRasMonteCarloEngine(PATHS, RETURN_PERIODS, RST)
    #engine.threshold = 0.5
    #engine.z_score = 1.28
    #engine.max_sims = 500
    # DEBUG: _manage_worker_folders
    # test_worker_paths = engine._manage_worker_folders()
    # DEBUG: sample_lhs_truncated
    # sample_rp = 100 # Change to any existing return period key in your config
    # p_dist = fitted_parameters_df[fitted_parameters_df['ReturnPeriod'] == sample_rp].iloc[0]
    # mock_flows = engine.sample_lhs_truncated(p_dist['Skew'], p_dist['Loc'], p_dist['Scale'], size=4)
    # DEBUG: _update_worker_unsteady_file
    # target_worker = os.path.join(PATHS['HR_sample_project'], 'Worker_1')
    # engine._update_worker_unsteady_file(target_worker, fw_sample=mock_flows[0])
    # DEBUG:_mp_hydraulic_worker
    # test_flow = float(mock_flows[0])
    # mock_payload = {'worker_path': target_worker,'SN': 9999,'RP': sample_rp,'IF': test_flow,'ras_exe': PATHS['HR_exe'],'num_cores': 2, 'gdf': gdf_buildings}
    # all_metrics = _mp_hydraulic_worker(mock_payload)
    # DEBUG:_mp_hydraulic_worker (parallel validation)
    # last_sn = engine._get_latest_sn(sample_rp)
    # p_dist = fitted_parameters_df[fitted_parameters_df['ReturnPeriod'] == sample_rp].iloc[0]
    # mock_flows_16 = engine.sample_lhs_truncated(p_dist['Skew'], p_dist['Loc'], p_dist['Scale'], size=16)
    # test_worker_paths = engine._manage_worker_folders()
    ''' # Comment to run the block
    task_inputs_16 = []
    for i, path in tqdm(enumerate(test_worker_paths)):
        engine._update_worker_unsteady_file(path, mock_flows_16[i])
        task_inputs_16.append({
            'worker_path': path,
            'SN': last_sn + i + 1,
            'RP': sample_rp,
            'IF': mock_flows_16[i],
            'ras_exe': PATHS['HR_exe'],
            'num_cores': engine.hr_cores_by_sim,
            'gdf': gdf_buildings
        })
    all_results_list = []
    for idx in tqdm(range(0, len(task_inputs_16), engine.hr_parallel_sims)):
        block = task_inputs_16[idx : idx + engine.hr_parallel_sims]
        with multiprocess.Pool(processes=len(block)) as pool:
            block_results = pool.map(_mp_hydraulic_worker, block)
            for sim_metrics in block_results:
                all_results_list.extend(sim_metrics)
    # Done
    ''' # Comment to run the block
    # DEBUG: check_convergence
    # sample_df = pd.DataFrame(all_results_list)
    # converged, stats_summary = engine.check_convergence(sample_df, sample_rp)
    # DEBUG: run_monte_carlo
    engine.run_monte_carlo()
    # pkl_path = PATHS['depth_samples_pkl']
    # sample_df = pd.read_pickle(pkl_path)
    # sample_df_rp = sample_df["RP"].unique()
    # sample_rp = 500
    # converged, stats_summary = engine.check_convergence(sample_df, sample_rp)
    
## BIG-DEBUG:
