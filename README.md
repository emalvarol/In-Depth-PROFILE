# In-Depth-PROFILE
In-Depth PRobabilistic Object-oriented Framework for Inundation Loss Estimation.
A framework result of an extension of Acharya et al. (2025) and Dottori et al. (2016) originally wrote in R lenguage. This framework integrates all steps neccesary to develope a Stochastic Flood Loss model considering: (1) HEC-RAS Monte Carlo Modelling; (2) survey data management; (3) web scraping for prices; (4) economic Monte Carlo; (4) convergence analysis; (5) sensitivity analysis though XGBoost subrogated model and SHAP calculation.

## NOTE!
This repository is part of an active research. We are in process of writting and plublishing the manuscript. Furthermore, this repository was build after the research process and the code has been changed during the creation of this repository. Due this unplanned workflow the repository may contain bugs which will be fixed according our possibilities.

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
│   └── valuation/                  <- Loss valuation engine (INSYDE Core)
│       ├── __init__.py
│       ├── preparation.py          <- Preprocessing (spatial joins, land registries)
│       ├── execution.py            <- Monte Carlo damage simulation and post-processing
│       ├── sensitivity.py          <- Global Sensitivity Analysis (GSA) module for SHAP
│       └── results.py              <- Figure creation
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
    ├── Draft.docx
    ├── Supplementary_Data.docx
    └── llm/                        <- A txt copy of each script to allow parsing them as context for the AI use
```

### Notes:
1. This repository and python code is not a python library neither a fully automatic pip-line, rather it is a worspace to use dynamicly on a code editor such as Visual Studio Code together with Conda environment. All workflow can be done trough main.py

2. Large files used in the process are ommited on this repository. Instead they are updated as part of the research on: ... (free download available)

## Requirements & Environments
### Overview
This projects uses two dedicated conda environments due to its complexity and library conflicts, lets name them environment A and B. Most of the work can be run on environment A, but the script dedicated to the global sensitivity analysis needs the environment B. Both environment description and configurations are explained below, but both needs this common steps
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

5.  Add environment to exception in Windows Security
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

## Strenghts and Limitations
What can and what can not be done with this repository
### Web Scraping
scraping.py module contains the EconomicScraper and PriceDataCleaner classes. They were prepared to scrape multiple items on two general retailers with modern webpages and limitations. They use of playwright was mandatory to overpass those limitation. Please note the use of web scraping its delicated and may lead to legal concerns. The data scraped for this research was limited and used for that propose.

### HEC-RAS Monte Carlo
modeling.py module contains HydrologicalFitter, GeospatialProcessor and HecRasMonteCarloEngine classes. They allow a fully automation of HEC-RAS launches, including the sample of the water depth outside the buildings. Its relay heavely on Ras Commander library (https://github.com/gpt-cmdr/ras-commander). This (amazing) library is still on development and may change. Further a new modern version of HEC-RAS is being developed and will may simply this stage.

### Input uncertainties distribution fitting
preparation.py module contains the DistributionFitter class. An stochastics or uncertainty analysis involve defining the distribution of each input with available data. To do so the mentioned class allow doing bootstrap across many specified functions to use survey data to create the input distribution. It heavy relay on scipy library. Alternativelly, the user can directly create the fm_pkl and fo_pkl object defining the functions name and params according to scipy.

### Economic Monte-Carlo
execution.py module contains the LossModelExecutionEngine class. This is the adaptation to python and extension of the work developed by Acharya et al. (2025) and Dottori et al. (2016). It works with a defined dataset including all the input variables, its distributions, its relationships and calculations and automatically executes a monte carlo process using vectorized samples, calculations and paralellization. It can handle multiple return periods, multiple buildings and building type, multiple distributions, missing distribution with fallback logic, multiple building floors and more.

### Sensitivty Analysis
sensitivity.py module contain SurrogateSensitivityEngine class. This is the unique module that need the environment B. Sensitivity analysis on complex multivariates models are still a frontier in science and particularly in flood loss models. This module solve it with an highly efficient XGBoost surrogate model running on the GPU to calculate then the SHAP values. This allow analyzing the sensitivity in details for each input variable considered in a few hours of computing (in the case of this research we used 304 input variables).

### Convergence analysis and other results
results.py include shap_result_grouper, DataPreparator and Plotter. All of them help to visualize the convergence of both the HEC-RAS and the Economic Monte Carlo, and other results such us the sensitivity analysis and damage analisys. Its semi-automate the creation of the figures including charts and maps with its neccesary results data proccesing.

## References
1. Dottori, F., Figueiredo, R., Martina, M. L., Molinari, D., & Scorzini, A. R. (2016). INSYDE: a synthetic, probabilistic flood damage model based on explicit cost analysis. Natural Hazards and Earth System Sciences, 16(12), 2577-2591.
2. Acharya, P., Di Bacco, M., Molinari, D., & Scorzini, A. R. (2025). INSYDE-content: a synthetic, multi-variable flood damage model for household contents. Natural Hazards and Earth System Sciences, 25(11), 4317-4330.

