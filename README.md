# In-Depth-PROFILE
In-Depth PRobabilistic Object-oriented Framework for Inundation Loss Estimation

## Structure of this repository:
```text
In-Depth-PROFILE/
├── .gitignore                      <- For ignoring large datasets and temporary files/folders
├── LICENSE                         <- Specify the license of the repository
├── README.md                       <- The main information file
├── requirements.txt                <- List of Python dependencies for reproducibility
│
├── src/                            <- Main source code, ordered sequentially
│   ├── User_Paths_and_Inputs.py    <- Centralized configuration and path management
│   ├── 0_Web_Scraping_v2.py        <- Data extraction of economic data (Amazon/Wallapop)
│   ├── 1_HEC-RAS_MonteCarlo_v4.py  <- Flood hazard modeling (HEC-RAS in Monte Carlo mode)
│   ├── 2_INSYDEPlus_Compiled_v3.py <- Core code: includes data preprocessing, economic damage Monte Carlo and data post-processing
│   ├── 3_Shapley_GSA_v3.1_PrepareDataset.py <- Data prep for sensitivity analysis
│   └── 3_Shapley_GSA_v3.2_SHAP.py  <- XGBoost training and SHAP value calculation (use a different environment)
│
├── data/                           <- Data directory (tracked files only; large files ignored)
│   ├── raw/                        <- E.g., survey data, uncorrected DSMs
│   ├── processed/                  <- E.g., scraped prices, fitted distributions
│   └── GIS_Inputs/                 <- Shapefiles, Cadastre data, Corine Land Cover
│
├── models/                         <- External model configurations
│   ├── HEC-RAS_6/                  <- Base and sample HEC-RAS projects
│   └── XGBoost/                    <- Trained surrogate models (if small enough)
│
├── outputs/                        <- Generated results (tracked as examples)
│   ├── figures/                    <- Damage maps, convergence charts, SHAP plots
│   └── tables/                     <- Extracted CSVs and summary data
│
└── docs/                           <- Documentation and manuscript drafts
    ├── 2026_Navaluenga_Draft.docx
    ├── Supplementary_Data.docx
    └── Figures_and_tables.docx
```

## Notes:
1. This repository and python code has not been designed as a python library neither as a fully automatic pip-line, rather as a worspace to use dynamicly on a code editor such as Visual Studio Code together with Conda environment. For example, python file
2_INSYDEPlus_Compiled_v3.py is executed up to a point, where the SHAPs values needs to be calculated using the file 3_Shapley_GSA_v3.1_... and 3_Shapley_GSA_v3.2_... Then the 2_INSYDEPlus_Compiled_v3.py code can be used to get the SHAP figure.

2. Large files used in the process are ommited on this repository. Instead they are updated as part of the research on: ... (free download available)
