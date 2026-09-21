####################### INSYDE-Content Flood Damage Model: HEC-RAS Monte Carlo #######################
# NOTE: This script is designed to be run in a terminal using a python editor (VSCode).
# NOTE: However, it can be easyly tested with interactive cells (Jupyter support in VSCode), except for multiprocessing cells.

# Region 0: Prepare configuration and data
# %% 0.1 Import libraries
# --- Standard Library Imports ---
import os
import sys
import shutil
import stat
import time
from datetime import datetime, timedelta
import logging

# --- Third-Party Libraries: Data Handling ---
import numpy as np
import pandas as pd
import geopandas as gpd

# --- Third-Party Libraries: Raster and Spatial ---
import rasterio
from rasterio.mask import mask

# --- Third-Party Libraries: Statistics and Sampling ---
from scipy.stats import pearson3, qmc
from scipy.optimize import minimize

# --- Third-Party Libraries: Progress and Visualization ---
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Multiprocessing ---
import multiprocess

# --- Project-Specific Imports ---
from ras_commander import init_ras_project, RasCmdr, RasUnsteady, RasProcess
import User_Paths_and_Inputs

# %% 0.2 Set up logging for ETA/Progress tracking
# Set up a dedicated logger for your ETA/Progress
progress_logger = logging.getLogger('ProgressTracker')
progress_logger.setLevel(logging.INFO)

# Create a file handler (this creates/overwrites the log file)
fh = logging.FileHandler('progress.log', mode='w') 
fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
progress_logger.addHandler(fh)

# Prevent this logger from printing to the main console
progress_logger.propagate = False

# Needed in splited terminal: Get-Content progress.log -Wait

# %% 0.3 Prepare configuration and data
workspace = User_Paths_and_Inputs.workspace
PATHS = User_Paths_and_Inputs.PATHS
CODES = User_Paths_and_Inputs.CODES
RETURN_PERIODS = User_Paths_and_Inputs.RETURN_PERIODS
RETURN_PERIODS.pop(2, None) # Identified that any input flow in RP=2 flood any building
RST = [50,250,500,750,1000,1250,1500,2000,2500,3000,5000,7000,9000]
#endregion

# region 1. Prepare data and functions
# %% 1.1 Prepare input flow functions based on flood frequency analysis
# Helping functions
def fit_pearson3(x_vals, tail_percentile=0.05):
    y_vals = np.log10(x_vals)
    probs = [tail_percentile, 0.50, 1 - tail_percentile]

    def objective(params):
        skew, loc, scale = params
        if scale <= 0: return 1e9
        quantiles = pearson3.ppf(probs, skew, loc, scale)
        return np.sum((quantiles - y_vals)**2)

    initial_guess = [0.1, np.mean(y_vals), np.std(y_vals)]
    res = minimize(objective, initial_guess, method='Nelder-Mead')
    return res.x

# Fit
results = []
for rp, flows in tqdm(RETURN_PERIODS.items()):
    skew, loc, scale = fit_pearson3(flows)
    results.append({
        'ReturnPeriod': rp,
        'Skew': skew,
        'Loc': loc,
        'Scale': scale
})
dist_depth_df = pd.DataFrame(results)
old_RP2 = [106,   147,    203]
skew, loc, scale = fit_pearson3(old_RP2)
rp2_row = pd.DataFrame({
    'ReturnPeriod': [2],
    'Skew': [skew],
    'Loc': [loc],
    'Scale': [scale]
})
dist_depth_df = pd.concat([dist_depth_df, rp2_row], ignore_index=True)

# Save
os.makedirs(os.path.dirname(PATHS['fm_flow_xlsx']), exist_ok=True)
dist_depth_df.to_excel(PATHS['fm_flow_xlsx'], index=False)

# %% OPTIONAL - Save plot with the function of all the return periods together
RUN_BLOCK = False
if RUN_BLOCK:
    def lp3_pdf(x, skew, loc, scale):
        """
        Calculates the Probability Density Function (PDF) for Log-Pearson Type III.
        Includes the Jacobian transformation: f_X(x) = f_Y(log10(x)) * (1 / (x * ln(10)))
        """
        y = np.log10(x)
        pdf_y = pearson3.pdf(y, skew, loc, scale)
        return pdf_y / (x * np.log(10))

    def plot_lp3_distributions_mpl(df, label='ReturnPeriod'):
        """
        Generates a Matplotlib chart with the PDF for all return periods.
        X-axis range is determined by the extreme quantiles of all distributions.
        """
        plt.figure(figsize=(12, 7))
        
        # Calculate global range based on extreme probabilities (e.g., 0.001 to 0.999)
        # in the log-space to ensure we cover low probability tails
        all_min = []
        all_max = []
        
        for _, row in df.iterrows():
            # ppf works in log10 space for LP3
            all_min.append(10**pearson3.ppf(0.05, row['Skew'], row['Loc'], row['Scale']))
            all_max.append(10**pearson3.ppf(0.95, row['Skew'], row['Loc'], row['Scale']))
        
        x_range = np.linspace(0, max(all_max), 10000)

        for _, row in df.iterrows():
            rp = row[label]
            y_pdf = lp3_pdf(x_range, row['Skew'], row['Loc'], row['Scale'])
            
            plt.plot(x_range, y_pdf, label=f"T={rp}", linewidth=2)

        plt.xscale('log')
        plt.title("Log-Pearson Type III Probability Density Functions by Return Period")
        plt.xlabel("Flow (m³/s)")
        plt.ylabel("Probability Density")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Return Period (T)")
        
        # Adjust scale if distributions are highly concentrated
        plt.xlim(left=0) 
        
        plt.tight_layout()
        plt.show()

    # Execution
    plot_lp3_distributions_mpl(dist_depth_df, label='ReturnPeriod')
    
    p_min, p_max = 0.05, 0.95
    bounds = []
    for _, r in dist_depth_df.iterrows():
        q_min = 10**pearson3.ppf(p_min, r['Skew'], r['Loc'], r['Scale'])
        q_max = 10**pearson3.ppf(p_max, r['Skew'], r['Loc'], r['Scale'])
        bounds.append((q_min, q_max))

    global_min = min(b[0] for b in bounds)
    global_max = max(b[1] for b in bounds)

    print(f"Global Theoretical Min (0.1%): {global_min:.2f} m³/s")
    print(f"Global Theoretical Max (99.9%): {global_max:.2f} m³/s")
#endregion

# region 2. Run HECRAS within Monte Carlo
# %% 1.2 Pre-analysis for buildings (DEM Median Calculation)
# Run this if PATHS['buildings_ua_sample_med_shp'] does not exist yet
if not os.path.exists(PATHS['buildings_ua_sample_med_shp']):
    #1 Load the shapefile
    gdf = gpd.read_file(PATHS['buildings_ua_sample_shp'])

    #2 With rasterio load DEM and calculate median for each building polygon
    with rasterio.open(PATHS['terrain_tif']) as src:
        no_data = src.nodata
        
        # Create a function to calculate median DEM for each polygon
        def calculate_median_dem(poly_geom, nodata):
            """Calculate median DEM value within a polygon"""
            out_image, _ = mask(src, [poly_geom], crop=True)
            out_image = out_image[0]
            out_image = out_image[out_image != nodata]
            
            if out_image.size > 0:
                return np.median(out_image)
            else:
                return np.nan
        
        #3 Calculate median for each building polygon and add to gdf
        medians_list = []
        
        for idx, row in gdf.iterrows():
            poly_geom = row.geometry
            median_val = calculate_median_dem(poly_geom, no_data)
            medians_list.append(median_val)
        
        #4 Add the median DEM values to a new column
        gdf['med_dem'] = medians_list
        
    #5 Save with _med prefix
    gdf.to_file(PATHS['buildings_ua_sample_med_shp'], driver='ESRI Shapefile')

    print(f"\nMedian DEM calculated for {len(gdf)} buildings")
    print(f"Sample med_dem values:\n{gdf['med_dem'].head()}")
else:
    print(f"\nMedian DEM already calculated and saved at {PATHS['buildings_ua_sample_med_shp']}")
#endregion

# region 2. Run HECRAS within Monte Carlo


# %% 2.1 Set parameters to run simulations
# Convergence criteria
# Criteria: Integrated Residual Margin of error (IRME)
THRESHOLD = 0.1 # Margin of error for the mean depth (he) accepted for convergence
Z_SCORE = 1.96  # Z alpha/2  (95% confidence -1.96; 90% confidence -1.645; etc)

# Max number of simulations to run (to avoid infinite loops in case of non-convergence)
MAX_SIMS = (16*100)-(16*0)

# Computer specifications (used for multiprocessing in python)
PHY_CORES = 16 # Physical cores of the computer (not logical/virtual cores)
os.environ['NUMEXPR_MAX_THREADS'] = f'{PHY_CORES}'

# HEC-RAS computing specifications
HR_CORES_BY_SIM = 6 # Cores to use in each HEC-RAS simulation (set in the HEC-RAS plan settings, not in python)
HR_PARALLEL_SIMS = 8 # Number of parallel simulations will be lauch simultanously from python.
# Note: I tested 6 cores and 8 simulation gives the maximun performance by simulation using aroun 90% of the computer resources.

# %% 2.2 Helping functions
#* Used Outside multiprocessing context
#--- To copy HEC-RAS project allowing multiprocessing ---
def remove_readonly(func, path, excinfo):
    """
    Error handler for shutil.rmtree to remove read-only attributes.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)

def copy_project(project_path, dest_path):
    """
    Cleans the destination and copies the full project.
    """
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path, onexc=remove_readonly)
    
    shutil.copytree(project_path, dest_path)

#--- To sample from the fitted distribution with LHS and truncation to avoid extreme tails ---
def sample_pearson3_lhs_truncated(skew, loc, scale, size=16, p_range=(0.05, 0.95)):
    """
    LHS sampling restricted to a specific probability window to avoid extreme tails.
    p_range: tuple (lower_bound_quantile, upper_bound_quantile)
    """
    p_min, p_max = p_range
    
    # 1. Generate LHS points in [0, 1]
    sampler = qmc.LatinHypercube(d=1)
    u_samples = sampler.random(n=size).flatten()
    
    # 2. Rescale U[0, 1] to the target probability range [p_min, p_max]
    p_samples = p_min + u_samples * (p_max - p_min)
    
    # 3. Map rescaled probabilities to Pearson3
    log_samples = pearson3.ppf(p_samples, skew, loc, scale)
    
    return 10**log_samples

#--- To update unsteady flow table and restart settings for each worker ---
def update_unsteady_flow(worker_path, fw_sample):
    """
    Updates the first flow table and restart settings for a specific worker path.
    """
    # Initialize ras object temporarily to get unsteady metadata
    ras_instance = init_ras_project(worker_path, PATHS['HR_exe'])
    unsteady_path = ras_instance.unsteady_df.loc[0, 'full_path']
    
    # Prepare input flow data
    lower_rst = max([r for r in RST if r <= fw_sample], default=min(RST))
    new_data = pd.DataFrame({'Value': [
        float(lower_rst),
        *[round(fw_sample, 2)] * 9
    ]})

    # Update flow table
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

    # Update restart file settings
    restart_name = f"RST_{int(lower_rst)}.rst"
    RasUnsteady.update_restart_settings(
        unsteady_file=unsteady_path,
        use_restart=True,
        restart_filename=restart_name,
        ras_object=ras_instance
    )

#--- To get the latest higher SN for the working return_period ---
def get_latest_higher_sn(return_period):
    """
    Gets the latest higher SN for the working return period.
    """
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
            last_sn = 0
    
    return last_sn

#* Used inside multiprocessing context
#--- To run HEC-RAS for each worker ---
def run_worker(input_dict):
    '''
    HEC-RAS simulation for a specific worker with the given input dictionary.
    The function will:
    1. Initialize RAS for the specific worker
    2. Run the simulation
    3. Process the max wse result to get the wsl tif path
    4. Copy wsl tif to folder PATHS['HR_output_maps'] modifying the
    '''
    import os
    import shutil
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask
    from ras_commander import init_ras_project, RasCmdr, RasProcess

    # Unpack inputs
    worker_path = input_dict['worker_path']
    output_path = input_dict['output_path']
    sim_id = input_dict['SN']
    return_period = input_dict['RP']
    fw_sample = input_dict['IF']
    buildings_shp = input_dict['buildings_shp']
    ras_exe = input_dict['ras_exe']
    num_cores = input_dict['num_cores']
    gdf = input_dict['gdf']

    # Initialize RAS for the specific worker
    print("Step 1: Initializing RAS project for worker at path:", worker_path)
    ras_worker = init_ras_project(worker_path, ras_exe)
    
    # Run RAS
    print(f"Step 2: Running HEC-RAS simulation for SN={sim_id}, RP={return_period}, IF={fw_sample:.2f}")
    result = RasCmdr.compute_plan("01", num_cores=num_cores, ras_object=ras_worker)

    # Process max wse result
    print("Step 3: Processing HEC-RAS output maps for worker at path:", worker_path)
    map_results = RasProcess.store_maps(
        plan_number="01",
        wse=True,
        depth=False,
        velocity=False,
        profile="Max",
        ras_object=ras_worker
    )
    wsl_tif_path = map_results['wse'][0]

    # Process output depht (he) by building
    print("Step 4: Calculating depth metrics for buildings using the WSL map")
    # he = wsl average value (from wsl_tif_path) - med_dem (from gdf)
    # negative he values will be set to 0 (not flooded)
    all_metrics = []
    with rasterio.open(wsl_tif_path) as src_depth:
        nodata_depth = src_depth.nodata

        for _, row in gdf.iterrows():
            building_id = row['BID']
            
            wsl_image, _ = mask(src_depth, [row.geometry], crop=True)
            wsl_values = wsl_image[0]
            
            if nodata_depth is not None:
                valid_wsl = wsl_values[wsl_values != nodata_depth]
            else:
                valid_wsl = wsl_values[~np.isnan(wsl_values)]
            
            if valid_wsl.size > 0:
                avg_wsl = np.mean(valid_wsl)
                he = max(0, round(float(avg_wsl - row['med_dem']), 2))
                
                all_metrics.append({
                    'SN': sim_id,
                    'RP': return_period,
                    'BID': building_id,
                    'IF': round(float(fw_sample), 2),
                    'he': round(he, 2)
                })
            else:
                all_metrics.append({
                    'SN': sim_id,
                    'RP': return_period,
                    'BID': building_id,
                    'IF': round(float(fw_sample), 2),
                    'he': 0
                })
    return all_metrics

#* Used outside multiprocessing context
#--- To check convergence and calculate final metrics ---
# test df = final_df, threshold = THRESHOLD, z_score = Z_SCORE
def check_convergence(df, rp_key, threshold, z_score):
    """
    Calculates convergence metrics for a specific Return Period.
    Criteria: SN > 30, IRME == 0, and 30 consecutive stable steps.
    """
    # Filter data for the current Return Period
    rp_df = df[df['RP'] == rp_key].copy()
    
    if rp_df.empty:
        return False, rp_df

    ## PER-BUILDING (BID) calculation
    # Sorting to ensure chronological accumulation
    rp_df = rp_df.sort_values(by=['BID', 'SN'])

    # he_BID_sn: Cumulative Sample Mean per Building
    rp_df['he_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().mean())

    # SD_BID_sn: Cumulative Standard Deviation (Sample Standard Deviation)
    rp_df['SD_BID_sn'] = rp_df.groupby('BID')['he'].transform(lambda x: x.expanding().std(ddof=1))

    # SE_BID_sn: Standard error
    rp_df['SE_BID_sn'] = rp_df['SD_BID_sn'] / np.sqrt(rp_df['SN'])

    # ME_BID_sn: Margin of Error
    rp_df['ME_BID_sn'] = z_score * rp_df['SE_BID_sn']

    # Residual calculation: max(0, ME - T)
    rp_df['Residual'] = (rp_df['ME_BID_sn'] - threshold).clip(lower=0)

    ## GLOBAL (RP) Integration
    # Create the simplified dataframe for IRME tracking
    rp_simp = rp_df.groupby('SN').agg(
        IRME=('Residual', 'sum'),                            # Integrated Residual Margin of Error
        NON_CONV_COUNT=('Residual', lambda x: (x > 0).sum())  # Count of buildings where ME > T
    ).reset_index()

    # Calculate relative non-convergence percentage
    total_flooded_bids = rp_df[rp_df['he'] > 0]['BID'].nunique()
    rp_simp['REL_NON_CONV'] = (rp_simp['NON_CONV_COUNT'] / total_flooded_bids) * 100

    # MET: Condition is met ONLY if SN > 30 and IRME is 0
    rp_simp['MET'] = (rp_simp['SN'] > 30) & (rp_simp['IRME'] == 0)

    # STC_AG: Stability Counter for consecutive 'MET' steps
    # We identify blocks of consecutive True values
    blocks = (~rp_simp['MET']).cumsum()
    rp_simp['STC_AG'] = rp_simp.groupby(blocks).cumcount() * rp_simp['MET']

    # CONV: Final convergence flag (30 consecutive stable steps)
    rp_simp['CONV'] = rp_simp['STC_AG'] >= 30

    # Check the status of the latest simulation
    if not rp_simp.empty:
        is_converged = rp_simp['CONV'].iloc[-1]
        return bool(is_converged), rp_simp
    
    return False, rp_simp

#--- To plot the convergence analysis with the detailed metrics ---
def plot_detailed_convergence(sample_df, threshold=0.1, z_score=2.0):
    """
    Processes and plots 4 convergence metrics for all Return Periods.
    Added Column 4: Spatial Average of Sample Mean across all BIDs.
    """
    rps = sorted(sample_df['RP'].unique().tolist())
    
    # Color palette
    colors = cm.get_cmap('tab10', len(rps))
    rp_colors = {rp: colors(i) for i, rp in enumerate(rps)}

    # Start layout
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, figsize=(12, 12), sharex=True)

    abs_max_sn = sample_df['SN'].max()

    for rp in rps:
        # 1. Run the existing convergence logic
        # Note: We need the full df to get spatial means per BID, 
        # but check_convergence returns the summary. 
        # We perform the BID-level mean calculation here for plotting.
        
        is_conv, rp_simp = check_convergence(sample_df, rp, threshold, z_score)
        
        if rp_simp.empty:
            continue
            
        # Re-calculating Spatial Mean for Chart 4
        # Grouping original data to get the average 'he' across all BIDs at each SN
        rp_raw = sample_df[sample_df['RP'] == rp].copy()
        rp_raw['he_BID_sn'] = rp_raw.groupby('BID')['he'].transform(lambda x: x.expanding().mean())
        spatial_mean_series = rp_raw.groupby('SN')['he_BID_sn'].mean()

        current_color = rp_colors[rp]
        label_name = f"RP {rp}"

        # --- CHART 1: IRME ---
        ax1.plot(rp_simp['SN'], rp_simp['IRME'], 
                 color=current_color, linestyle='-', linewidth=1.5, label=label_name)
        
        # --- CHART 2: % Non-Converged ---
        ax2.plot(rp_simp['SN'], rp_simp['REL_NON_CONV'], 
                 color=current_color, linestyle='--', linewidth=1)

        # --- CHART 3: Stability Counter (STC_AG) ---
        ax3.plot(rp_simp['SN'], rp_simp['STC_AG'], 
                 color=current_color, linestyle='-', linewidth=1)

        # --- CHART 4: Spatial Average Sample Mean ---
        ax4.plot(spatial_mean_series.index, spatial_mean_series.values,
                 color=current_color, linestyle='-', linewidth=1.5)

        # --- VERTICAL STABILITY LINES ---
        converged_points = rp_simp[rp_simp['CONV'] == True]
        if not converged_points.empty:
            stable_sn = converged_points['SN'].min()
            for ax in [ax1, ax2, ax3, ax4]:
                ax.axvline(x=stable_sn, color=current_color, linestyle=':', linewidth=2, alpha=0.5)

    # Global Max for IRME
    global_max_irme = sample_df.get('IRME', pd.Series([10])).max() 

    # Formatting
    ax1.set_yscale('log')
    ax1.set_ylabel('IRME (Log)', fontweight='bold')
    ax1.set_title(f'Multi-Metric Convergence Analysis (T={threshold}, Z={z_score})', fontsize=14, fontweight='bold')
    ax1.legend(title="Return Periods", loc='upper right', ncol=2, fontsize='small')
    
    ax2.set_ylabel('% BIDs < T', fontweight='bold')
    ax2.set_ylim(-1, 101)

    ax3.set_ylabel('Stable Steps', fontweight='bold')
    ax3.axhline(y=30, color='red', linestyle='--', alpha=0.4)

    ax4.set_ylabel('Spatial Avg mu (m)', fontweight='bold')
    ax4.set_xlabel('Simulation Number (SN) [Log Scale]', fontweight='bold')

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xscale('log')
        ax.set_xlim(left=1, right=abs_max_sn)
        ax.grid(True, which="both", ls="-", alpha=0.15)
        ax.axvspan(1, 30, color='gray', alpha=0.1)

    plt.tight_layout()

    # Save in PATHS['convergence_charts'] with name including threshold and z_score
    os.makedirs(PATHS['convergence_charts'], exist_ok=True)
    chart_path = os.path.join(PATHS['convergence_charts'], f'convergence_analysis_T{threshold}_Z{z_score}.png')
    plt.savefig(chart_path, dpi=300)

# %% 2.3 (MULTIPROCESSING!) Run simulations in batches with multiprocessing and track progress/ETA
# test: rp_key = 500
# Clean some rp_key from pkl if needed
RUN_BLOCK = False
if RUN_BLOCK:
    rp_key = 10
    pkl_path = PATHS['depth_samples_pkl']
    if os.path.exists(pkl_path):
        try:
            df_existing = pd.read_pickle(pkl_path)
            df_existing = df_existing[df_existing['RP'] != rp_key]  # Remove RP=5 data
            df_existing.to_pickle(pkl_path)
            print(f"Cleaned RP={rp_key} data from existing pickle.")
        except (pd.errors.EmptyDataError, EOFError, KeyError):
            print("Existing pickle is empty or malformed. No cleaning needed.")
    else:
        print("No existing pickle found. No cleaning needed.")
# Read the building shape
gdf = gpd.read_file(PATHS['buildings_ua_sample_med_shp'])

# Start a loop for each return period.
for rp_key in sorted(RETURN_PERIODS.keys(), reverse=True):
    
    # Start loop for each batch of simulations for the return period
    # until num_simulations or MAX_SIMS is reached
    converged = False
    total_sims_done = get_latest_higher_sn(rp_key)
    while not converged and total_sims_done < MAX_SIMS+total_sims_done:
        total_sims_done = get_latest_higher_sn(rp_key)
        #* Outside multiprocessing context
        #2.3.1 Create/clean working folders
        worker_paths = []
        for i in range(1, PHY_CORES + 1):
            dest_path = rf'{PATHS["HR_sample_project"]}\Worker_{i}'
            copy_project(PATHS['HR_base_project'], dest_path)
            worker_paths.append(dest_path)
        os.makedirs(PATHS['HR_output_maps'], exist_ok=True)
        #2.3.2 Prepare sample for the return period
        p = dist_depth_df[dist_depth_df['ReturnPeriod'] == rp_key].iloc[0]
        fw_samples = sample_pearson3_lhs_truncated(p['Skew'], p['Loc'], p['Scale'], size=PHY_CORES)
        #2.3.3 Update unsteady file for each worker
        for i, path in enumerate(worker_paths):
            update_unsteady_flow(path, fw_samples[i])
        #2.3.4 Get lastest higuer SN for the working return_period
        last_sn = get_latest_higher_sn(rp_key)
        #2.3.5 Prepare input_dict for each worker
        task_inputs = []
        for i, path in enumerate(worker_paths):
            task_inputs.append({
                'worker_path': path,
                'output_path' : PATHS['HR_output_maps'],
                'SN': last_sn + i + 1,
                'RP': rp_key,
                'IF': fw_samples[i],
                'buildings_shp': PATHS['buildings_ua_sample_med_shp'],
                'ras_exe': PATHS['HR_exe'],
                'num_cores': HR_CORES_BY_SIM,
                'gdf': gdf
            })
        
        #* Inside multiprocessing context
        #2.3.6 Run simulations in parallel
        all_results_list = []
        for i in range(0, len(task_inputs), HR_PARALLEL_SIMS):
            block = task_inputs[i : i + HR_PARALLEL_SIMS]
            with multiprocess.Pool(processes=len(block)) as pool:
                # pool.map returns a list of lists (all_metrics from run_worker)
                block_results = pool.map(run_worker, block)
                # Flatten and collect
                for sim_metrics in block_results:
                    all_results_list.extend(sim_metrics)
        
        #* Outside multiprocessing context
        #2.3.7 Save outputs maps into results folder with a name including RP and SN
        for i, path in enumerate(worker_paths):
            sim_sn = last_sn + i + 1
            sim_rp = rp_key
            worker_path = path
            src_tif = os.path.join(worker_path,'01', f"WSE (Max).Terrain.{os.path.basename(PATHS['terrain_tif'])}")
            dest_tif = os.path.join(PATHS['HR_output_maps'], f"WSE_RP{sim_rp}_SN{sim_sn}.tif")
            if os.path.exists(src_tif):
                shutil.move(src_tif, dest_tif)
            
        #2.3.8 Update and save the pkl with the new results
        new_df = pd.DataFrame(all_results_list)
        if os.path.exists(PATHS['depth_samples_pkl']):
            old_df = pd.read_pickle(PATHS['depth_samples_pkl'])
            sample_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            sample_df = new_df
        sample_df.to_pickle(PATHS['depth_samples_pkl'])

        #2.3.9 Check convergence
        converged, check_df = check_convergence(sample_df, rp_key, THRESHOLD, Z_SCORE)
        total_sims_done = get_latest_higher_sn(rp_key)

# %% 2.4 Save convergence analysis plot with detailed metrics
sample_df = pd.read_pickle(PATHS['depth_samples_pkl'])
plot_detailed_convergence(sample_df, threshold=THRESHOLD, z_score=Z_SCORE)
#endregion
# %% End of script
print("End of script")
