import os
import shutil
import stat
from scipy.stats import pearson3, qmc
import geopandas as gpd
import pandas as pd
from src.config import PATHS, RETURN_PERIODS, RST

from ras_commander import init_ras_project, RasCmdr, RasUnsteady, RasProcess

paths = PATHS
return_periods = RETURN_PERIODS
rst_values = RST

# Hardware & Engine execution settings
phy_cores = 16
hr_cores_by_sim = 6
hr_parallel_sims = 8

# Sampling Settings (set according to the flood frequency analyst)
p_lower = 0.05
p_upper = 0.95

# HEC-RAS Model settings
n_rows_unsteady = 10 # IMPORTANT, fails sylently and its defined by the base project

# Convergence Settings
threshold = 0.1
z_score = 1.96
max_sims = 1600
        
# Optimize Numexpr for parallelization
os.environ['NUMEXPR_MAX_THREADS'] = f'{phy_cores}'

def _get_latest_sn(return_period):
    """Retrieves the highest Simulation Number (SN) executed for a given Return Period."""
    pkl_path = PATHS['depth_samples_pkl']
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

def _remove_readonly(func, path, excinfo):
    """Error handler for shutil.rmtree to remove read-only attributes.
    Supports both Python <3.12 (onerror) and >=3.12 (onexc)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _manage_worker_folders():
    """Cleans and initializes isolated workspaces for worker threads."""
    worker_paths = []
    base_proj = paths['HR_base_project']

    print("\n--- [DEBUG] Starting Worker Workspace Initialization ---")
    print(f"[DEBUG] Base Project Directory: {base_proj}")

    # Check source base folder for critical .hdf files
    if os.path.exists(base_proj):
        base_files = os.listdir(base_proj)
        hdf_files = [f for f in base_files if f.endswith('.hdf')]
        print(f"[DEBUG] Files in base folder ({len(base_files)} total): {base_files}")
        print(f"[DEBUG] .hdf files found in base folder: {hdf_files}")
    else:
        print(f"[ERROR] Base project path does NOT exist: {base_proj}")

    for i in range(1, phy_cores + 1):
        dest_path = os.path.join(paths['HR_sample_project'], f'Worker_{i}')
        
        # Remove old folder using flexible error handler to ensure full deletion
        if os.path.exists(dest_path):
            try:
                shutil.rmtree(dest_path, onexc=_remove_readonly)
            except TypeError:
                # Fallback for Python < 3.12
                shutil.rmtree(dest_path, onerror=_remove_readonly)
        
        # Copy base project to worker folder
        shutil.copytree(base_proj, dest_path)
        worker_paths.append(dest_path)

        # Verification of copy action
        worker_files = os.listdir(dest_path)
        target_hdf = [f for f in worker_files if f.endswith('.hdf')]
        
        print(f"[DEBUG] Worker_{i} created at: {dest_path}")
        print(f"        -> Total items copied: {len(worker_files)}")
        print(f"        -> .hdf files present: {target_hdf}")

        # Specific check for missing HDF condition
        if not target_hdf:
            print(f"[WARNING] Worker_{i} is missing .hdf files after copytree!")

    os.makedirs(paths['HR_output_maps'], exist_ok=True)
    print("--- [DEBUG] Workspace Initialization Complete ---\n")
    return worker_paths

def sample_lhs_truncated(skew, loc, scale, size=16, p_range=(0.05, 0.95)):
    """Generates Latin Hypercube Samples restricted to a probability window."""
    p_min, p_max = p_range
    sampler = qmc.LatinHypercube(d=1)
    u_samples = sampler.random(n=size).flatten()
    p_samples = p_min + u_samples * (p_max - p_min)
    log_samples = pearson3.ppf(p_samples, skew, loc, scale)
    return 10**log_samples

#worker_path, fw_sample = path, fw_samples[i]
def _update_worker_unsteady_file(worker_path, fw_sample):
    """Injects flow boundaries and assigns the correct HEC-RAS restart file."""
    ras_instance = init_ras_project(worker_path, paths['HR_exe'])
    unsteady_path = ras_instance.unsteady_df.loc[0, 'full_path']
    
    lower_rst = max([r for r in rst_values if r <= fw_sample], default=min(rst_values))
    new_data = pd.DataFrame({'Value': [
        float(lower_rst),                                   # Row 1: Closest restart file flow
        *[round(fw_sample, 2)] * (n_rows_unsteady-1)   # Remaining rows: Sampled flow
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

# input_dict = task_inputs[0]
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


print("--------------------------------------------\n")
print("--------------------------------------------\n")
print("Initializing HEC-RAS Multi-Engine Monte Carlo Pipeline...")
print("--------------------------------------------\n")
print("--------------------------------------------\n")
gdf_buildings = gpd.read_file(paths['buildings_sample_med_shp'])
dist_depth_df = pd.read_pickle(paths['fm_flow_pkl'])
pkl_path = paths['depth_samples_pkl']

for rp_key in sorted(return_periods, reverse=True):
    print("--------------------------------------------\n")
    print("--------------------------------------------\n")
    print(f"\nProcessing Return Period Execution Group: {rp_key}")
    print("--------------------------------------------\n")
    print("--------------------------------------------\n")
    converged = False
    total_sims_done = _get_latest_sn(rp_key)
    

    while not converged and total_sims_done < max_sims:
        total_sims_done = _get_latest_sn(rp_key)
        worker_paths = _manage_worker_folders()

        # Generate sampling arrays
        p_dist = dist_depth_df[dist_depth_df['ReturnPeriod'] == rp_key].iloc[0]
        fw_samples = sample_lhs_truncated(
            p_dist['Skew'], p_dist['Loc'], p_dist['Scale'], size=phy_cores, p_range=(p_lower, p_upper))
        print("flow inputs generated")
        
        # Prepare parameters and boundary condition files per thread
        last_sn = _get_latest_sn(rp_key)
        task_inputs = []
        for i, path in enumerate(worker_paths):
            _update_worker_unsteady_file(path, fw_samples[i]) # Works fine
            task_inputs.append({
                'worker_path': path,
                'SN': last_sn + i + 1,
                'RP': rp_key,
                'IF': fw_samples[i],
                'ras_exe': paths['HR_exe'],
                'num_cores': hr_cores_by_sim,
                'gdf': gdf_buildings
            })
        print("Unsteady file updated")
        
        # Process concurrent execution blocks
        all_results_list = []
        '''
        for idx in range(0, len(task_inputs), self.hr_parallel_sims):
            block = task_inputs[idx : idx + self.hr_parallel_sims]
            with multiprocess.Pool(processes=len(block)) as pool:
                block_results = pool.map(_mp_hydraulic_worker, block)
                for sim_metrics in block_results:
                    all_results_list.extend(sim_metrics)
        '''
        
        print("Simulation executed")



        
path1 = r"C:\Users\outal\GITHUB\Jose - UCLM\In-Depth-PROFILE\data\intermediate\Functions_Main.pkl"
path2 = r"C:\Users\outal\GITHUB\Jose - UCLM\In-Depth-PROFILE\data\intermediate\Functions_Observed.pkl"

df1 = pd.read_pickle(path1)

he_df = df1[df1["DC"] == "he"]

# Count of occurrences
dc_he_count = len(he_df)
print(f"Count of DC == 'he': {dc_he_count}")

# Unique FN values for DC == 'he'
unique_fn = he_df["FN"].unique()
print(f"Unique FN values: {unique_fn}")
