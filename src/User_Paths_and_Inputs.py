'''
Here the user must set the diferent paths of the data.
Both input and output paths
'''
import os

workspace = r"C:\Users\outal\OneDrive\3_Personas y Proyectos\Jose - UCLM\2_Economic Valuation\PyWS"
os.chdir(workspace)

### PATHS
PATHS = {
    # Data survey
    'survey': os.path.join(
        'Data',
        'Formulario_encuestas(v2022)_long_clean_v2.1.xlsx'
    ),
    'survey_summary': os.path.join(
        'Data',
        'Formulario_encuestas(v2022)_Summary.xlsx'
    ),
    # Item prices (from webscraping)
    'prices_content': os.path.join(
        'Data',
        'Prices_Content.xlsx'
    ),
    'prices_continents': os.path.join(
        'Data',
        'Prices_Contentinent.xlsx'
    ),
    # Depth samples (from hec-ras montecarlo)
    'depth_samples': os.path.join(
        'Data',
        'Depth_Samples.xlsx'
    ),
    'depth_samples_pkl': os.path.join(
        'Data',
        'Depth_Samples.pkl'
    ),
    # Data functions (for monte carlo)
    'input_flow_distributions_outputs': os.path.join(
        'Outputs',
        'Distribution HECRAS InputFlow'
    ),
    'fm_flow_xlsx': os.path.join(
        'Data',
        'Functions_Input_Flow.xlsx'
    ),
    'fm_xlsx': os.path.join(
        'Data',
        'Functions_Main.xlsx'
    ),
    'fm_pkl': os.path.join(
        'Data',
        'Functions_Main.pkl'
    ),
    'fo_xlsx': os.path.join(
        'Data',
        'Functions_Observed.xlsx'
    ),
    'fo_pkl': os.path.join(
        'Data',
        'Functions_Observed.pkl'
    ),
    # HecRas
    'HR_base_project': os.path.join(
        'HEC-RAS_6',
        'RC_0_Base_Project'
    ),
    'HR_sample_project': os.path.join(
        'HEC-RAS_6',
        'RC_1_Sample_Projets'
    ),
    'HR_output_maps': os.path.join(
        'HEC-RAS_6',
        'RC_2_Output_WSE'
    ),
    'HR_exe': r"C:\Program Files (x86)\HEC\HEC-RAS\6.6\Ras.exe",
    'convergence_charts': os.path.join(
        'Outputs',
        'HC_Convergence',
    ),
    # Buildings
    'buildings_ua_shp': os.path.join(
        'Data',
        'GIS_Inputs',
        'Catastro',
        'UA',
        'BIDs_v2.3.shp',
    ),
    'buildings_ua_sample_shp': os.path.join(
        'Data',
        'GIS_Inputs',
        'Catastro',
        'UA',
        'BIDs_sampling_v2.3.shp',
    ),
    'buildings_ua_sample_med_shp': os.path.join(
        'Data',
        'GIS_Inputs',
        'Catastro',
        'UA_Process',
        'CONSTRU_UA_v23_p9_med.shp',
    ),
    # Terrains
    'terrain_tif': os.path.join(
        'Data',
        'GIS_Inputs',
        'DSM',
        'DSM_GeoFixed',
        'dsm_cs1_v23.tif',
    ),
    # DEM Error
    'dem_error_tif': os.path.join(
        'Data',
        'GIS_Inputs',
        'Geostatistics',
        'Raster',
        'kv1_se_c.tif',
    ),
    # Economic Monte Carlo
    'dataset_mc_parts': os.path.join(
        r"C:\Users\outal\Documents",
        'MC_Parts'
    ),
    'fitted_distributions': os.path.join(
        'Outputs',
        'PT_Fitted_Distributions',
    ),
    # GSA XGBoost dataset
    'xgb_dataset': os.path.join(
        'Data',
        'xgb_dataset.parquet',
    )
        
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
        ## Not used
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
    'MAT': { # Material, Binary code, allow any key combination for any combination of materials
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
    #2:    [106,   147,    203],
    5:    [244,   340,    507],
    10:   [367,   526,    877],
    20:   [552,   838,    1686],
    50:   [707,   1132,   2671],
    100:  [871,   1483,   4136],
    200:  [1044,  1898,   6288],
    500:  [1284,  2560,   10710],
}
### DISTRIBUTIONS
'''
Dic structure:
Data Type ("cont", "disc")
    Range ("-inf,inf", "0,inf", "0,a", "a,b")
        Parameters ("1", "2", "3", etc)
            Funct Type ("Common", "heavy tailed", "specialized", "generalized")
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




