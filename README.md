# In-Depth-PROFILE
In-Depth PRobabilistic Object-oriented Framework for Inundation Loss Estimation

## Structure
### Structure of this repository:
```text
In-Depth-PROFILE/
├── .gitignore                      <- For ignoring large datasets and temporary files/folders
├── LICENSE                         <- Specify the license of the repository
├── README.md                       <- The main information file
├── requirements.txt                <- List of Python dependencies for reproducibility
├── main.py                         <- Framework orchestrator and main execution script
│
├── src/                            <- Main source code package
│   ├── __init__.py                 <- Makes src a recognizable Python package
│   ├── config.py                   <- Dynamic pathing, global dictionaries, and setups
│   ├── scraping.py                 <- Web scraping and market price data mining
│   ├── modeling.py                 <- HEC-RAS automation and Monte Carlo hazard engine
│   │
│   ├── valuation/                  <- Loss valuation engine (INSYDE Core)
│   │   ├── __init__.py
│   │   ├── preparation.py          <- Preprocessing (spatial joins, land registries)
│   │   └── execution.py            <- Monte Carlo damage simulation and post-processing
│   │
│   └── sensitivity/                <- Global Sensitivity Analysis (GSA) module
│       ├── __init__.py
│       ├── preparation.py          <- Dataset building and preparation for XGBoost
│       └── analysis.py             <- Surrogate model training and SHAP values calculation
│
├── data/                           <- Data directory (tracked files only; large files ignored)
│   ├── raw/                        <- E.g., survey data, uncorrected DSMs
│   ├── processed/                  <- E.g., scraped prices, fitted distributions
│   └── GIS_Inputs/                 <- Shapefiles, Cadastre data, Corine Land Cover
│
├── models/                         <- External model configurations
│   ├── HEC-RAS_6.6./               <- Base and sample HEC-RAS projects
│   └── XGBoost/                    <- Trained surrogate models (if small enough)
│
├── outputs/                        <- Generated results (tracked as examples)
│   ├── figures/                    <- Damage maps, convergence charts, SHAP plots
│   ├── hecras/                     <- Results from hecras models (deterministic or stochastic)
│   └── tables/                     <- Extracted CSVs and summary data
│
├── tests/                          <- Isolated test suite 
│   ├── test_data/                  <- Mock datasets for verification
│   ├── test_models/                <- Mock models for verification
│   └── test_outputs/               <- Mock outputs for verification
│
└── docs/                           <- Documentation and manuscript drafts
    ├── 2026_Navaluenga_Draft.docx
    ├── Supplementary_Data.docx
    └── Figures_and_tables.docx
```

### Notes:
1. This repository and python code has not been designed as a python library neither as a fully automatic pip-line, rather as a worspace to use dynamicly on a code editor such as Visual Studio Code together with Conda environment. For example, python file
2_INSYDEPlus_Compiled_v3.py is executed up to a point, where the SHAPs values needs to be calculated using the file 3_Shapley_GSA_v3.1_... and 3_Shapley_GSA_v3.2_... Then the 2_INSYDEPlus_Compiled_v3.py code can be used to get the SHAP figure.

2. Large files used in the process are ommited on this repository. Instead they are updated as part of the research on: ... (free download available)

## Requirements & Environments (Working space configuration)
### Overview
This projects uses two dedicated conda environments, lets name them environment A and B. Most of the work can be run on environment A, but the script dedicated to the global sensitivity analysis needs the environment B. Both environment description and configurations are explained below, but both needs this common steps
```text
1. Install Anaconda:
https://www.anaconda.com/download
2. (Optional) Install Anaconda.Navigator:
https://www.anaconda.com/products/navigator
```

### Environment A
It has libraries related to Data Handling & Computation, Spatial & GIS, Statistics, Machine Learning & Optimization, Parallelization, Progress Tracking, Web Scraping, HEC-RAS, and Visualization. Note the libraries provided in Requirements.txt file are those belonging to this environment. To create the environment run these commands oppening 'Anaconda Prompt' on windows search:
```text
1. Create the environment with conda forge with all these libraries:
conda create --name env_a -c conda-forge python=3.12 numpy pandas pyarrow geopandas OWSLib scipy shap multiprocess pandarallel tqdm beautifulsoup4 thefuzz matplotlib matplotlib-scalebar openpyxl shapely gdal fiona rasterio rasterstats pyogrio -y
It will be created at: C:\Users\<Your_User>\anaconda3\envs\env_a

2. Activate the environment:
conda activate env_a

3. Install libraries not available in conda forge
pip install ras_commander==0.89.2 thefuzz==0.22.1 playwright

4. Install navigator for web scrapping:
playwright install
```

### Environment B
It has libraries focused on Machine Learning & Optimization but the main difference is that it requiers further configuration to be able of working with GPU though Cuda. Note the libraries needed for this environment are also provided in Requirements.txt file but withouth cuda support. To create this environment with cuda support the steps are:
```text
1. Install GPU Prerequisites (Windows):
- Visual Studio Community: Download (https://visualstudio.microsoft.com/vs/community/) and install it. During installation, you must check the box for "Desktop development with C++" to install the necessary compilers.
- NVIDIA CUDA Toolkit: Download and install the CUDA toolkit compatible with your GPU (https://developer.nvidia.com/cuda-downloads).     

2. Create the environment with PyTorch, XGBoost, and CUDA 11.8 support:
conda create -n env_b python=3.11 pandas numpy optuna==4.9.0 shap xgboost==3.2.0 pyarrow==21.0.0 scikit-learn fastparquet matplotlib pytorch torchvision torchaudio pytorch-cuda=11.8 mkl mkl-include -c pytorch -c nvidia -c conda-forge --override-channels -y

3. Activate the environment:
conda activate env_b
```
