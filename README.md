# Efficient and Reliable Predictive Maintenance in Trains based on BiLSTM Model

## Overview

This project implements a predictive maintenance system to estimate the Remaining Useful Life (RUL) of train components, specifically air compressors, by combining the MetroPT dataset from Zenodo and the MetroPT-3 dataset from Kaggle. It compares traditional machine learning models (Linear Regression, Random Forest, SVR, KNN) with a Bidirectional Long Short-Term Memory (BiLSTM) neural network. SHAP (SHapley Additive exPlanations) is used to interpret model predictions, focusing on the BiLSTM model for enhanced explainability.

## Project Structure

```plaintext
Efficient-and-Reliable-Predictive-Maintenance-in-Trains-based-on-BiLSTM-Model/
├── config/
│   └── config.yaml                # Configuration file for paths and parameters
│
├── data/
│   ├── raw/                       # Raw datasets (MetroPT.csv, MetroPT3(AirCompressor).csv)
│   ├── predictions/               # Stored model predictions
│   └── processed/                 # Processed data (X_train, X_test, y_train, y_test)
│
├── models/                        # Store trained models (pickle, HDF5)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Exploratory Data Analysis (EDA)
│   ├── 02_prepare_data.ipynb            # Data preprocessing and dataset combination
│   ├── 03_train_evaluate_models.ipynb   # BiLSTM and baseline models training and evaluation
│   ├── 04_shap_analysis.ipynb           # SHAP analysis (for BiLSTM only)
│
├── scripts/
│   ├── data_preprocessing.py      # Preprocessing functions
│   ├── evaluate_models.py         # Evaluation functions
│   ├── utils.py                   # Utility functions for plotting
│   └── model_training.py          # Functions for training different models
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## Datasets

This project combines two datasets for robust predictive maintenance:

### 1. MetroPT
- **Source**: Download from [Zenodo](https://zenodo.org/records/6854240) and save as `data/raw/MetroPT.csv`.
- **Features**: Analog sensors (pressure, temperature, current), digital signals, GPS (latitude, longitude, speed).
- **Citation**:
  > Veloso, B., Ribeiro, R.P., Gama, J. et al. The MetroPT dataset for predictive maintenance. Sci Data 9, 764 (2022). https://doi.org/10.1038/s41597-022-01877-3

### 2. MetroPT-3
- **Source**: Download from [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/metropt-3-dataset?select=MetroPT3%28AirCompressor%29.csv) and save as `data/raw/MetroPT3(AirCompressor).csv`.
- **Features**: Analog sensors (pressure, temperature, motor current), digital signals (air intake valves).
- **Citation**:
  > N. Davari, B. Veloso, R. P. Ribeiro, P. M. Pereira and J. Gama, "Predictive maintenance based on anomaly detection using deep learning for air production unit in the railway industry," 2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA), Porto, Portugal, 2021, pp. 1-10, doi: 10.1109/DSAA53316.2021.9564181.

## Requirements

- **Python 3.7** (required for `tensorflow==1.14.0`)
- **Anaconda** (optional, recommended for Conda environment management)
- **Git** (required to clone the repository)

### Dependencies listed in `requirements.txt`:
```plaintext
tensorflow==1.14.0
shap==0.42.1
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
pyyaml>=6.0
joblib>=1.2.0
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YahiaouiLydia/Efficient-and-Reliable-Predictive-Maintenance-in-Trains-based-on-BiLSTM-Model.git
   cd Efficient-and-Reliable-Predictive-Maintenance-in-Trains-based-on-BiLSTM-Model
   ```

2. (Optional) Create a virtual environment:
   - **Using Conda** (recommended for managing Python 3.7):
     ```bash
     conda create -n pyota_env python=3.7
     conda activate pyota_env
     ```
   - **Using Python `venv`**:
     ```bash
     python -m venv pyota_env
     source pyota_env/bin/activate  # On Windows: pyota_env\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

This project is structured into four steps, each represented by a notebook. Execute all cells in each notebook sequentially to avoid runtime errors. Notebooks can be run via terminal or manually through Jupyter Notebook, JupyterLab, or an IDE like VS Code.

> ⚠️ **Important**: Ensure datasets are downloaded and `config.yaml` is configured before running notebooks.

### Step 1 – Exploratory Data Analysis (EDA)
- **Purpose**:
  - Understand the structure of MetroPT and MetroPT-3 datasets.
  - Visualize sensor data distributions.
  - Identify anomalies or missing values.
  - Explore correlations between variables.
- **How to Run**:
  ```bash
  jupyter notebook notebooks/01_data_exploration.ipynb
  ```
  Alternatively, open Jupyter Notebook, JupyterLab, or an IDE (e.g., VS Code) and navigate to `notebooks/01_data_exploration.ipynb`.
- **Output**: Visualizations and statistics (e.g., histograms, correlation matrices).

### Step 2 – Data Preprocessing
- **Purpose**:
  - Load and combine `MetroPT.csv` and `MetroPT3(AirCompressor).csv`.
  - Standardize column names and units.
  - Merge datasets based on timestamps.
  - Compute RUL for each component.
  - Save preprocessed datasets for model training.
- **How to Run**:
  ```bash
  jupyter notebook notebooks/02_prepare_data.ipynb
  ```
  Alternatively, open Jupyter Notebook, JupyterLab, or an IDE and navigate to `notebooks/02_prepare_data.ipynb`.
- **Output**:
  - Baseline models: `data/processed/x_train_baseline.csv`, `y_train_baseline.csv`, `x_test_baseline.csv`, `y_test_baseline.csv`.
  - BiLSTM: `data/processed/X_train_bilstm.joblib`, `y_train_bilstm.joblib`, `X_test_bilstm.joblib`, `y_test_bilstm.joblib`, `scaler_x_bilstm.joblib`.

### Step 3 – Model Training and Evaluation
- **Purpose**:
  - Train baseline models: Linear Regression, Random Forest, SVR, KNN, Gradient Boosting.
  - Train a BiLSTM model on time-series data.
  - Evaluate models using MAE, RMSE, and R².
  - Save trained models and results.
- **How to Run**:
  ```bash
  jupyter notebook notebooks/03_train_evaluate_models.ipynb
  ```
  Alternatively, open Jupyter Notebook, JupyterLab, or an IDE and navigate to `notebooks/03_train_evaluate_models.ipynb`.
- **Output**:
  - Trained models: `models/bilstm_model.h5`, `models/random_forest_model.pkl`, etc.
  - Predictions: `data/predictions/` (model-specific files).
  - Plots (e.g., prediction comparisons).

### Step 4 – SHAP Analysis (Explainability)
- **Purpose**:
  - Use SHAP to explain BiLSTM predictions.
  - Visualize global and local feature impacts.
  - Enhance model interpretability.
- **How to Run**:
  ```bash
  jupyter notebook notebooks/04_shap_analysis.ipynb
  ```
  Alternatively, open Jupyter Notebook, JupyterLab, or an IDE and navigate to `notebooks/04_shap_analysis.ipynb`.
- **Output**:
  - SHAP summary plots.
  - SHAP dependence plots.

### Alternative: Run Scripts
For advanced users, scripts in `scripts/` provide modular functions:
- `data_preprocessing.py`: Preprocessing functions.
- `evaluate_models.py`: Evaluation metrics.
- `utils.py`: Plotting utilities.
- `model_training.py`: Model training functions.

## Configuration

The `config.yaml` file (`config/config.yaml`) defines paths and parameters. Example:
```yaml
paths:
  raw:
    metropt: data/raw/MetroPT.csv
    metropt3: data/raw/MetroPT3(AirCompressor).csv
  processed:
    x_train_baseline: data/processed/x_train_baseline.csv
    y_train_baseline: data/processed/y_train_baseline.csv
    X_test_bilstm: data/processed/X_test_bilstm.joblib
    scaler_x_bilstm: data/processed/scaler_x_bilstm.joblib
  models:
    bilstm_model: models/bilstm_model.h5
bilstm:
  test_size: 0.3
  random_state: 42
```

## Troubleshooting

1. **FileNotFoundError: Missing Raw Data Files**
   - Ensure `data/raw/MetroPT.csv` and `data/raw/MetroPT3(AirCompressor).csv` exist in `data/raw/`.
   - Download from [Zenodo](https://zenodo.org/records/6854240) and [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/metropt-3-dataset).

2. **Dataset Mismatch: Feature Alignment**
   - Check feature alignment in `02_prepare_data.ipynb` (e.g., consistent sensor names, units).
   - Verify timestamps and sensor compatibility.

3. **Dependency Issues: Python Version and Package Installation**
   - Use Python 3.7 for compatibility with `tensorflow==1.14.0`.
   - Reinstall dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Check TensorFlow version:
     ```bash
     pip show tensorflow
     ```

4. **SHAP Errors: Explanation Model Issues**
   - If `shap.DeepExplainer` fails, try `shap.KernelExplainer`.
   - Ensure `tensorflow==1.14.0` is installed.

5. **Output Verification: Check Generated Files**
   - Verify:
     - `data/processed/`: Processed datasets.
     - `models/`: Trained models.
     - `data/predictions/`: Model predictions.

## Contributing

This project is part of a doctoral research effort on predictive maintenance in trains. Contributions or suggestions are welcome via GitHub issues: [Issues](https://github.com/YahiaouiLydia/Efficient-and-Reliable-Predictive-Maintenance-in-Trains-based-on-BiLSTM-Model/issues).


## License

This project is for academic purposes and not distributed under a specific license.
