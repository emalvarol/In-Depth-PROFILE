"""
In-Depth-PROFILE Configuration Module

This module serves as the single source of truth for the framework's metadata, 
geospatial assets, configuration variables, and mathematical catalogs. 

It contains:
1. PATHS: A dictionary managing dynamic path definitions using pathlib.Path, 
   covering data surveys, item prices, HEC-RAS projects/executables, 
   GIS assets (buildings/terrains), and Monte Carlo output directories.
2. CODES: Standardized data dictionaries mapping short alpha codes to human-readable 
   descriptions for surveys, cadastre/building attributes, interior contents, 
   structural building components, and material classifications.
3. RETURN_PERIODS: Hydrological parameter mappings defining discrete discharge 
   or boundary conditions for statistical return periods (5 to 500 years).
4. DIST_CATALOG: Comprehensive statistical distribution catalogs mapping Scipy 
   distribution identifiers into tailored continuous and discrete parameter groups 
   for probabilistic Monte Carlo sampling.
"""

from pathlib import Path

# Dynamically resolve the project root directory (In-Depth-PROFILE/)
# src/config.py -> parent.parent yields the root directory
if '__file__' in globals():
    BASE_DIR = Path(__file__).resolve().parent.parent
else:
    # Fallback to current working directory if run in REPL
    BASE_DIR = Path.cwd()
    
# ==============================================================================
# ENVIRONMENT TOGGLE SWITCH
# ==============================================================================
# Set to True to execute the pipeline with lightweight mock data for testing.
# Set to False to run the full methodology with large production datasets.
USE_TEST_DATA = False  
# ==============================================================================

# Define the root data directory based on the toggle switch
DATA_DIR = BASE_DIR / 'tests' / 'test_data' if USE_TEST_DATA else BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'tests' / 'test_models' if USE_TEST_DATA else BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'tests' / 'test_outputs' if USE_TEST_DATA else BASE_DIR / 'outputs'

print(f"[{'TESTING' if USE_TEST_DATA else 'PRODUCTION'} MODE] Data directory set to: {DATA_DIR}")
print(f"[{'TESTING' if USE_TEST_DATA else 'PRODUCTION'} MODE] Model directory set to: {MODEL_DIR}")
print(f"[{'TESTING' if USE_TEST_DATA else 'PRODUCTION'} MODE] Output directory set to: {OUTPUT_DIR}")

### PATHS
PATHS = {
    # Data survey
    'survey': DATA_DIR / 'inputs' / 'Formulario_encuestas(v2022)_long_clean_v2.1.xlsx',
    
    # Item prices (from webscraping)
    'prices_content': DATA_DIR / 'processed' / 'Prices_Content.xlsx',
    
    # Depth samples (for/from HEC-RAS Monte Carlo)
    'fm_flow_pkl': DATA_DIR / 'intermediate' / 'Flow_Fitted_Functions.pkl',
    'depth_samples_pkl': DATA_DIR / 'intermediate' / 'Depth_Samples.pkl',
    
    # NEW! mocaloss
    'mocaloss_working_dir' : DATA_DIR / 'mocaloss',
    'sampling_rules_name' : "sampling_table_v1.4.pkl",
    
    # NEW! Official floods
    'det_RP10' : DATA_DIR / 'inputs' / "gis" / "OficialFloods" / "Q10_2Ciclo_PB_2026_Navaluenga.shp",
    'det_RP50' : DATA_DIR / 'inputs' / "gis" / "OficialFloods" / "Q50_2Ciclo_PB_2026_Navaluenga.shp",
    'det_RP100' : DATA_DIR / 'inputs' / "gis" / "OficialFloods" / "Q100_2Ciclo_PB_2026_Navaluenga.shp",
    'det_RP500' : DATA_DIR / 'inputs' / "gis" / "OficialFloods" / "Q500_2Ciclo_PB_2026_Navaluenga.shp",
    
    # HEC-RAS Configuration
    'HR_base_project': MODEL_DIR / 'HEC-RAS_6.6' / 'RC_0_Base_Project',
    'HR_sample_project': MODEL_DIR / 'HEC-RAS_6.6' / 'RC_1_Sample_Projects',
    'HR_output_maps': MODEL_DIR / 'HEC-RAS_6.6' / 'RC_2_Output_WSE',
    'HR_exe': Path(r"C:\Program Files (x86)\HEC\HEC-RAS\6.6\Ras.exe"),
    
    # NEW! Modeled floods
    'representative_floods' : DATA_DIR / 'intermediate' / "gis" / "ModeledFloods",
    
    # Buildings (Exposure Data)
    'buildings_shp': DATA_DIR / 'inputs' / 'gis' / 'BID' / 'BIDs_v2.3.shp',
    'buildings_sample_shp': DATA_DIR / 'inputs' / 'gis' / 'BID' / 'BIDs_sampling_v2.3.shp',
    'buildings_sample_med_shp': DATA_DIR / 'intermediate' / 'gis' / 'BID' / 'BIDs_sampling_v2.3_med.shp',
    
    # Terrains & Hazard Input
    'terrain_tif': DATA_DIR / 'inputs' / 'gis' / 'DSM' / 'dsm_cs1_v23.tif',
    
    # DEM Error (Geostatistics)
    'dem_error_tif': DATA_DIR / 'inputs' / 'Geostatistics' / 'Outputs' / 'kv1_se_c.tif',
    
    # Economic Monte Carlo (Resolves dynamically to the current user's home directory)
    'dataset_mc_parts': DATA_DIR / 'processed' / 'MC_Parts',
    
    # GSA XGBoost dataset
    'xgb': DATA_DIR / 'processed' / 'MC_Parts' / 'XGBoost',
    'xgb_dataset': DATA_DIR / 'processed' / 'MC_Parts' / 'XGBoost' / 'xgb_dataset.feather',
    
    # Figures
    'convergence': OUTPUT_DIR / 'convergence',
    'damage_map': OUTPUT_DIR / 'damage_map',
    'damage_evolution': OUTPUT_DIR / 'damage_evolution',
    'damage_functions': OUTPUT_DIR / 'damage_functions',
    'ead_map': OUTPUT_DIR / 'ead_map',
    'gsa': OUTPUT_DIR / 'gsa',
}

### CODES
CODES = {
    'Survey':{
        'ROM':'1_Room',
        'RMV':'1b_Room_Variant',
        'VTP':'2_Variable_Type',
        'DOM':'3_Domain',
        'GRP':'4_Group',
        'ITM':'5_Item',
        'ITV':'5b_Item_Variant',
        'ATR':'8_Attribute',
    },
    #RID (Row ID) 
    #DC (Dataset codes)
    'Building':{
        # Codes
        'BID': 'Building ID',
        'BT': 'Dwelling Type',
        'RP': 'Return Period',
        ## Structure
        'NF': 'Number of Floors',
        'BF': 'Basement Floors',
        'HU': 'Number of Households',
        'IH': 'Inter-floor Height',
        ## Areas
        'GA': 'Ground Floor Area', # IA in INSYDE and FA in INSYDE-Content
        'BA': 'Basement Area',
        'SA': 'Total Surface Area',
        ## Levels and heights
        'GL': 'Ground floor Level',
        'BL': 'Basement Level',
        'BH': 'Basement Height',
        ## Perimeters
        'EP': 'External Perimeter',
        'GP': 'Ground Floor Perimeter', # IP in INSYDE
        'BP': 'Basement Perimeter',
        ## Not used (optional for further development)
        #'FL': 'Finishing Level'
        #'GU': 'Ground floor use'
        #'BS': 'Building structure'
        #'LM': 'Level of maintenance'
        #'YY': 'Year of construction'
    },
    'Content':{ # All have n, p, d, c
        'APP': 'Appliance',
        'CLO': 'Clothing',
        'COM': 'Computer',
        'DEC': 'Decorative or collectible item',
        'ELE': 'Electronic devices',
        'ENG': 'Energy equipment',
        'FAD': 'Food and drink',
        'FUR': 'Furniture',
        'HHG': 'Household goods',
        'HHB': 'Hygiene, health, or beauty item',
        'INS': 'Instruments',
        'LEI': 'Leisure equipment',
        'OTH': 'Others',
        'SPE': 'Specialized equipment',
        'TOO': 'Tools',
        'VEH': 'Vehicle',
    },
    'Continent':{
        # General
        'PUM': 'Pumping', #p,e,c
        'CLE': 'Cleaning', #p,e,c
        'DHU': 'Dehumidification', #p, e, c
        # Soil
        'SOI': 'Soil', # m, p_r, p , e , c_r, c
        # Components
        'FRI': 'Friso', # ff, n
        'SKT': 'Skirting', #n,m,p,d,c
        'RDR': 'Regular doors', #n,m,p,d,c
        'WND': 'Windows', #n,p,d,c
        'PLG': 'Plugs', #n,p,d,c
        # Walls
        'PRW': 'Partition walls', #m,p_r,e,d,p,n,c
        'EXF': 'External Façade', #m, e
        'ETP': 'External Paint', #p, c
        'ITP': 'Interior Paint', #c
        # Others
        'ELS': 'Electrical system', #t,d,p,c
    },
    'Event':{
        'he': 'Depth outside the building',
        'hi': 'Depth inside the building',
    },
    'Terrain':{
        'ed': 'DEM error altitude outside the building',
    },
    'Dataset_Variants':{
        #DCV (Dataset variants)
        'n': 'Number',
        'ff': 'Fragility Function',
        't': 'Type',
        'p': 'Unitarian prices',
        'pr': 'Unitarian removal prices',
        'm': 'Material',
        'e': 'Extension',
        'd': 'Damage state',
        'c': 'Cost damaged',
        'cr': 'Removal cost damaged',
    },
    'MAT': { # Binary code, allow any key combination for any combination of materials
        1: 'Aluminio', # Aluminum
        2: 'Barro', # Mud / Clay
        4: 'Metal', # Metal
        8: 'Cemento, hormigón', # Cement, Concrete
        16: 'Cerámica, gres, terrazo, teja, azulejo', # Ceramic, Stoneware, Terrazzo, Roof tile, Wall tile
        32: 'Aglomerado, contrachapado, conglomerado', # Chipboard, Pied-wood (Plywood), Conglomerate
        64: 'Corcho', # Cork
        128: 'Cristal, vidrio', # Crystal, Glass
        256: 'Escayola, yeso', # Plaster, Gypsum / Plasterboard
        512: 'Madera, parquet, plaqueta, tarima', # Wood, Parquet, Floor tile (small), Floorboards / Decking
        1024: 'Metracilato', # Methacrylate (Acrylic)
        2048: 'Piedra, mampostería', # Stone, Masonry
        4096: 'Plástico, vinilo', # Plastic, Vinyl
        8192: 'Pintura', # Paint
        16384: 'Ladrillo', # Brick
        32768: 'Papel', # Paper / Wallpaper
    }
}

### RETURN PERIODS
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

### DISTRIBUTIONS
'''
Dic structure:
Data Type ("c: continuous", "d: discrete")
    Range ("i: -inf,inf", "p: 0,inf", "b: a,b")
        Parameters ("1", "2", "3", "4+")
            Funct Type ("co: Common", "ht: heavy tailed", "sp: specialized", "ge: generalized")
'''
dist_families = {
    "c": {
        "i": {  # [-inf, inf]
            "2": {
                "co": ["norm", "laplace", "logistic", "uniform"],
                "sp": ["gumbel_r"]
            },
            "3": {
                "co": ["triang"],
                "ht": ["t", "lognorm"],
                "ge": ["gennorm", "genlogistic", "genextreme"],
                "sp": ["jhonsonsb", "jhonsonsu"]
            },
            "4+": {
                "ge": ["genhyperbolic"]
            }
        },
        "p": {  # [0, inf]
            "1": {
                "co": ["expon", "halfnorm"]
            },
            "2": {
                "co": ["gamma", "pareto"],
                "ht": ["fisk"],
                "sp": ["weibull_min", "nakagami"]
            },
            "3+": {
                "ge": ["gengamma", "genexpon", "genpareto"],
                "he": ["burr"]
            }
        },
        "b": { # [a, b]
            "4": {
                "sp": ["truncnorm"]
            }
        }
    },
    "d": {
        "p": {
            "1": {"co": ["poisson", "geom"]},
            "2": {"co": ["binom"]}
        }
    }
}
ci = dist_families["c"]["i"]
cp = dist_families["c"]["p"]
cb = dist_families["c"]["b"]
dp = dist_families["d"]["p"]
DIST_CATALOG = {
    "C_BS_i": ( # dist_names_to_fit_C_II
        ci["2"]["co"] + ci["2"]["sp"] + 
        ci["3"]["co"] + ci["3"]["ge"] + ci["3"]["sp"] +
        [f for f in ci["3"]["ht"] if f != 'lognorm'] +
        ci["4+"]["ge"]
    ),
    "C_SS_i": ( # dist_names_to_fit_C_II_A
        ci["2"]["co"] +
        [f for f in ci["3"]["ht"] if f != 'lognorm'] # t
    ),
    "C_SS_i_vB": ( # dist_names_to_fit_C_II_B
        ci["2"]["co"] +                             # norm, laplace, logistic, uniform
        ci["3"]["co"] +                             # triang
        [f for f in ci["3"]["ht"] if f != 'lognorm'] # t
    ),
    "C_SS_i_vff": ( # dist_names_to_fit_C_MI
        [ci["2"]["co"][0]] + cb["4"]["sp"] + ci["2"]["co"][1:] + # norm, truncnorm, laplace, logistic, uniform
        [f for f in ci["3"]["ht"] if f != 'lognorm'] +          # t
        [f for f in ci["3"]["ht"] if f == 'lognorm'] +          # lognorm
        cp["1"]["co"] + [cp["2"]["co"][0]] +                    # expon, halfnorm, gamma
        [cp["2"]["co"][1]]                                      # pareto
    ),
    "C_SS_i_vff2": ( # dist_names_to_fit_C_MI_A
        ci["2"]["co"] +                                         # norm, laplace, logistic, uniform
        [f for f in ci["3"]["ht"] if f != 'lognorm'] +          # t
        [f for f in ci["3"]["ht"] if f == 'lognorm'] +          # lognorm
        cp["1"]["co"] + [cp["2"]["co"][0]] +                    # expon, halfnorm, gamma
        [cp["2"]["co"][1]]                                      # pareto
    ),
    "C_SS_i_vff3": ( # dist_names_to_fit_C_MI_B
        [ci["2"]["co"][0], ci["2"]["co"][1]] +                  # norm, laplace
        ci["3"]["co"] + [ci["2"]["co"][2], ci["2"]["co"][3]] + # triang, logistic, uniform
        [f for f in ci["3"]["ht"] if f != 'lognorm'] +          # t
        [f for f in ci["3"]["ht"] if f == 'lognorm'] +          # lognorm
        cp["1"]["co"] + [cp["2"]["co"][0]]                      # expon, halfnorm, gamma
    ),
    "C_BS_p": ( # dist_names_to_fit_C_0I
        [f for f in ci["3"]["ht"] if f == 'lognorm'] + # lognorm from inf/3/ht
        [f for f in ci["3"]["ge"] if f == 'genextreme'] + # genextreme from inf/3/ge
        cp["2"]["sp"][:1] + # weibull_min
        cp["3+"]["ge"][:2] + # gengamma, genexpon
        cp["1"]["co"] + # expon, halfnorm
        cp["2"]["co"] + # gamma, pareto
        cp["2"]["sp"][1:] + # nakagami
        cp["3+"]["ge"][2:] + # genpareto
        cp["2"]["ht"] + # fisk
        cp["3+"]["he"] # burr
    ),
    "C_BS_p_vB": ( # dist_names_to_fit_C_0I_A
        [f for f in ci["3"]["ht"] if f == 'lognorm'] + # lognorm
        [cp["2"]["sp"][0]] +                           # weibull_min
        ci["2"]["sp"] +                                # gumbel_r
        [cp["1"]["co"][0]] +                           # expon
        [cp["2"]["co"][0]] +                           # gamma
        [cp["1"]["co"][1]]                             # halfnorm
    )
}

### HEC-RAS
RST = [ # Restart Files flow m3/s
    90, 250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 5000, 7000
]