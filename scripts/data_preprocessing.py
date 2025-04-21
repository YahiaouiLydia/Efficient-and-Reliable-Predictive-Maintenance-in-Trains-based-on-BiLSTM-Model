import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pickle
import os
import joblib

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess_common(config):
    """Common preprocessing steps for baseline and BiLSTM."""
    # Load and clean data
    project_root = Path().resolve().parent
    path_metropt3 = project_root / config["paths"]["raw"]["metropt3"]
    path_metropt = project_root / config["paths"]["raw"]["metropt"]

    MetroPT3 = pd.read_csv(path_metropt3)
    MetroPT1 = pd.read_csv(path_metropt)

    columns_to_remove1 = ['Flowmeter', 'gpsQuality', 'gpsLat', 'gpsSpeed', 'gpsLong']
    MetroPT1 = MetroPT1.drop(columns=[col for col in columns_to_remove1 if col in MetroPT1.columns])
    columns_to_remove3 = ['Unnamed: 0']
    MetroPT3 = MetroPT3.drop(columns=[col for col in columns_to_remove3 if col in MetroPT3.columns])

    MetroPT = pd.concat([MetroPT1, MetroPT3], ignore_index=True)
    MetroPT['timestamp'] = pd.to_datetime(MetroPT['timestamp'], format="%Y-%m-%d %H:%M:%S")

    # Add state and failure_component
    state_time_ranges = [
        ('2022-02-28 21:53', '2022-03-01 02:00', 'Air Leak'),
        ('2022-03-23 14:54', '2022-03-23 15:24', 'Air Leak'),
        ('2022-05-30 12:00', '2022-06-02 06:18', 'Oil Leak'),
        ('2020-04-18 00:00', '2020-04-18 23:59', 'Air Leak'),
        ('2020-05-29 23:30', '2020-05-30 06:00', 'Air Leak'),
        ('2020-06-05 10:00', '2020-06-07 14:30', 'Air Leak'),
        ('2020-07-15 14:30', '2020-07-15 19:00', 'Air Leak')
    ]

    component_time_ranges = [
        ('2022-02-28 21:53', '2022-03-01 02:00', 'Clients'),
        ('2022-03-23 14:54', '2022-03-23 15:24', 'Air Dryer'),
        ('2022-05-30 12:00', '2022-06-02 06:18', 'Compressor'),
        ('2020-04-18 00:00', '2020-04-18 23:59', 'Compressor'),
        ('2020-05-29 23:30', '2020-05-30 06:00', 'Compressor'),
        ('2020-06-05 10:00', '2020-06-07 14:30', 'Compressor'),
        ('2020-07-15 14:30', '2020-07-15 19:00', 'Compressor')
    ]

    MetroPT['state'] = 'Normal'
    for start, end, state in state_time_ranges:
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        mask = (MetroPT['timestamp'] >= start_time) & (MetroPT['timestamp'] <= end_time)
        MetroPT.loc[mask, 'state'] = state

    MetroPT['failure_component'] = 'Operational'
    for start, end, component in component_time_ranges:
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        mask = (MetroPT['timestamp'] >= start_time) & (MetroPT['timestamp'] <= end_time)
        MetroPT.loc[mask, 'failure_component'] = component

    # Calculate RUL
    failure_times = [
        ('2022-02-28 21:53', '2022-03-01 02:00'),
        ('2022-03-23 14:54', '2022-03-23 15:24'),
        ('2022-05-30 12:00', '2022-06-02 06:18'),
        ('2020-04-18 00:00', '2020-04-18 23:59'),
        ('2020-05-29 23:30', '2020-05-30 06:00'),
        ('2020-06-05 10:00', '2020-06-07 14:30'),
        ('2020-07-15 14:30', '2020-07-15 19:00')
    ]

    MetroPT['RUL'] = 0
    for start, end in failure_times:
        start_time = pd.to_datetime(start)
        end_time = pd.to_datetime(end)
        mask = (MetroPT['timestamp'] >= start_time) & (MetroPT['timestamp'] <= end_time)
        MetroPT.loc[mask, 'RUL'] = 0

    latest_failure_time = pd.to_datetime(max(end for _, end in failure_times))
    outside_failure_mask = MetroPT['RUL'] == 0
    MetroPT.loc[outside_failure_mask, 'RUL'] = (
        latest_failure_time - MetroPT.loc[outside_failure_mask, 'timestamp']
    ).dt.total_seconds() / (60 * 60 * 24)

    # Remove outliers
    for col in MetroPT.select_dtypes(include=['number']).columns:
        Q1 = MetroPT[col].quantile(0.25)
        Q3 = MetroPT[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (MetroPT[col] < lower_bound) | (MetroPT[col] > upper_bound)
        MetroPT = MetroPT[~outliers]

    MetroPT['RUL'] = MetroPT['RUL'].astype(int)

    MetroPT['timestamp'] = pd.to_datetime(MetroPT['timestamp'])
    # Extract timestamp features and group data
    MetroPT['year'] = MetroPT['timestamp'].dt.year
    MetroPT['month'] = MetroPT['timestamp'].dt.month
    MetroPT['day'] = MetroPT['timestamp'].dt.day
    MetroPT['hour'] = MetroPT['timestamp'].dt.hour
    MetroPT['minute'] = MetroPT['timestamp'].dt.minute
    MetroPT['second'] = MetroPT['timestamp'].dt.second

    grouped_data = MetroPT.groupby(['month', 'day', 'hour', 'minute']).agg({
        'TP2': 'mean',
        'TP3': 'mean',
        'H1': 'mean',
        'DV_pressure': 'mean',
        'Reservoirs': 'mean',
        'Oil_temperature': 'mean',
        'Motor_current': 'mean',
        'COMP': 'mean',
        'DV_eletric': 'mean',
        'Towers': 'mean',
        'MPG': 'mean',
        'LPS': 'mean',
        'Pressure_switch': 'mean',
        'Oil_level': 'mean',
        'Caudal_impulses': 'mean',
        'state': 'first',
        'failure_component': 'first',
        'RUL': 'mean'
    }).reset_index()
    grouped_data = grouped_data.sort_values(by=['month', 'day', 'hour', 'minute'])
    return grouped_data

def prepare_data_for_model(grouped_data, n_top_features=8):
    """Prepare data for baseline models, including feature selection and scaling."""
    # Prepare features and target
    X = grouped_data.drop(columns=['RUL', 'state', 'failure_component'])
    y = grouped_data['RUL']

    # Normalize data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature selection with GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(X_scaled, y)
    importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print("Feature Importance:\n", importance_df)

    # Select top features
    top_features = importance_df['Feature'].head(n_top_features).values
    print(f"Top {n_top_features} Features: {top_features}")
    X_top = X[top_features]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, top_features
def prepare_data_for_bilstm(grouped_data, n_top_features=8, n_steps=50):
    """Prepare data for BiLSTM, using same feature selection as baseline."""
    # Prepare features and target
    X = grouped_data.drop(columns=['RUL', 'state', 'failure_component'])
    y = grouped_data['RUL'].values.reshape(-1, 1)

    # Feature selection with GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=100)
    model.fit(X, y.flatten())
    importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    top_features = importance_df['Feature'].head(n_top_features).values
    print(f"Top {n_top_features} Features: {top_features}")

    # Select top features
    X_top = X[top_features]

    # Normalize features and target
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_top_scaled = x_scaler.fit_transform(X_top)
    y_scaled = y_scaler.fit_transform(y)

    return X_top_scaled, y_scaled, x_scaler, y_scaler, top_features


def create_and_save_bilstm_sequences(X_scaled, y_scaled, x_scaler, y_scaler, config, n_steps):
    """Create and save BiLSTM sequences for training and testing in Joblib format."""
    # Get project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Create processed directory if it doesn't exist
    processed_dir = project_root / "data" / "processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create time sequences
    def create_sequences(X, y, n_steps):
        X_seq, y_seq = [], []
        for i in range(len(X) - n_steps):
            X_seq.append(X[i:i+n_steps])
            y_seq.append(y[i+n_steps])
        return np.array(X_seq), np.array(y_seq)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, n_steps)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=config["bilstm"]["test_size"], random_state=config["bilstm"]["random_state"]
    )

    # Save processed datasets as Joblib
    joblib.dump(X_train, processed_dir / config["paths"]["processed"]["X_train_bilstm"])
    print(f"Saved {config['paths']['processed']['X_train_bilstm']} successfully.")
    joblib.dump(X_test, processed_dir / config["paths"]["processed"]["X_test_bilstm"])
    print(f"Saved {config['paths']['processed']['X_test_bilstm']} successfully.")
    joblib.dump(y_train, processed_dir / config["paths"]["processed"]["y_train_bilstm"])
    print(f"Saved {config['paths']['processed']['y_train_bilstm']} successfully.")
    joblib.dump(y_test, processed_dir / config["paths"]["processed"]["y_test_bilstm"])
    print(f"Saved {config['paths']['processed']['y_test_bilstm']} successfully.")

    # Save scalers as Joblib
    joblib.dump(x_scaler, processed_dir / config["paths"]["processed"]["scaler_x_bilstm"])
    print(f"Saved {config['paths']['processed']['scaler_x_bilstm']} successfully.")
    joblib.dump(y_scaler, processed_dir / config["paths"]["processed"]["scaler_y_bilstm"])
    print(f"Saved {config['paths']['processed']['scaler_y_bilstm']} successfully.")

    return X_train, X_test, y_train, y_test


def save_processed_data_baseline(X_train, X_test, y_train, y_test, scaler, config):
    """Save processed data and scaler for baseline models."""
    project_root = Path().resolve().parent
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train).to_csv(project_root / config["paths"]["processed"]["x_train_baseline"], index=False)
    pd.DataFrame(X_test).to_csv(project_root / config["paths"]["processed"]["x_test_baseline"], index=False)
    pd.DataFrame(y_train).to_csv(project_root / config["paths"]["processed"]["y_train_baseline"], index=False)
    pd.DataFrame(y_test).to_csv(project_root / config["paths"]["processed"]["y_test_baseline"], index=False)

    with open(project_root / config["paths"]["processed"]["scaler_baseline"], 'wb') as f:
        pickle.dump(scaler, f)

def preprocess_pipeline_baseline(config_path, n_top_features=8):
    """Preprocessing pipeline for baseline models."""
    config = load_config(config_path)
    grouped_data = preprocess_common(config)
    X_train, X_test, y_train, y_test, scaler, top_features = prepare_data_for_model(grouped_data, n_top_features)
    save_processed_data_baseline(X_train, X_test, y_train, y_test, scaler, config)
    return X_train, X_test, y_train, y_test, scaler, top_features

def preprocess_pipeline_bilstm(config_path, n_top_features=8, n_steps=50):
    """Preprocessing pipeline for BiLSTM."""
    config = load_config(config_path)
    grouped_data = preprocess_common(config)
    X_scaled, y_scaled, x_scaler, y_scaler, top_features = prepare_data_for_bilstm(grouped_data, n_top_features, n_steps)
    X_train, X_test, y_train, y_test = create_and_save_bilstm_sequences(X_scaled, y_scaled, x_scaler, y_scaler, config, n_steps)

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler, top_features