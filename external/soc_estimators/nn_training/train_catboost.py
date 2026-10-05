"""
================================================================================
SOC Prediction Framework for LFP Battery Cells
================================================================================

This work was originally developed in the context of the Huawei Tech Arena
Challenge (Nuremberg, 2024) for State of Charge (SOC) prediction of LFP
(Lithium Iron Phosphate) battery cells, where it was awarded First Prize.

The framework was subsequently extended and refined by researchers at the
Energy Center Lab, Department of Energy (DENERG), Politecnico di Torino.
The codebase, particularly the AI/Machine Learning model implementations,
was primarily developed by Lorenzo Giannuzzo.

Authors:
    Lorenzo Giannuzzo        - lorenzo.giannuzzo@polito.it
    Elena Rozzi               - elena.rozzi@polito.it
    Marcel Stolte             - marcel.stolte@polito.it
    Daniele Salvatore Schiera - daniele.schiera@polito.it
    Francesco D. Minuto       - francesco.minuto@polito.it
    Andrea Lanzini            - andrea.lanzini@polito.it

Supervision:
    Mansour Alramlawi         - Huawei Technologies
    (supervision of the related research paper and selected code modules)

Affiliation:
    Politecnico di Torino
    Department of Energy (DENERG) - Energy Center Lab
    Corso Duca degli Abruzzi 24, 10129 Torino, Italy

================================================================================
"""

"""
CatBoost SOC Prediction - Training Script
@authors: Lorenzo Giannuzzo, Marcel Stolte, Elena Rozzi
Contact: lorenzo.giannuzzo@polito.it, marcel.stolte@polito.it, elena.rozzi@polito.it

This script trains a CatBoost model for battery SOC prediction.
It uses the same base features as ANN/CNN/LSTM plus engineered lag features.
Training data is passed through the Kalman Filter first to generate realistic
SOC_Calc values, ensuring consistency between training and inference conditions.
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Lorenzo Giannuzzo: Use the local KalmanFilter_module.py (in nn_training/) which has correct
# Lorenzo Giannuzzo: relative paths for ../train_data/ and ../data/
from KalmanFilter_module import KalmanFilter

# ============================================================================
# CONFIGURATION
# ============================================================================

# Lorenzo Giannuzzo: Base features (same as ANN/CNN/LSTM models)
# Lorenzo Giannuzzo:  Raw measurements
RAW_FEATURES = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc']

# Computed features (same as other models - created during preprocessing)
COMPUTED_FEATURES = [
    'Delta_Temperature',
    'Delta_Current',
    'Delta_Voltage',
    'Delta_SoC_Calc',
    'Delta_Voltage_Current_Ratio',
    'Current_Voltage_Ratio'
]

# Lorenzo Giannuzzo: All base features (same 10 as ANN/CNN/LSTM)
BASE_FEATURES = RAW_FEATURES + COMPUTED_FEATURES

# Lorenzo Giannuzzo: Lag timesteps in seconds (assuming 1Hz sampling)
# Applied only to raw features to capture temporal dynamics
# - 1s: immediate dynamics
# - 5s: short-term variations
# - 10s: short-medium term trends
# - 30s: medium term trends
# - 60s: longer term trends (1 minute)
LAG_STEPS = [1, 5, 10, 30, 60]

# Lorenzo Giannuzzo: Lag features are applied only to RAW_FEATURES
# (no rolling windows needed - Delta features already capture temporal info)

# Lorenzo Giannuzzo: CatBoost hyperparameters (standard values)
CATBOOST_PARAMS = {
    'iterations': 2000,           # Lorenzo Giannuzzo: Number of trees (all will be trained)
    'learning_rate': 0.1,         # Lorenzo Giannuzzo: Learning rate
    'depth': 6,                   # Lorenzo Giannuzzo: Maximum tree depth
    'l2_leaf_reg': 3,             # Lorenzo Giannuzzo: L2 regularization coefficient
    'loss_function': 'RMSE',      # Lorenzo Giannuzzo: Loss function for regression
    'random_seed': 42,            # Lorenzo Giannuzzo: Random seed for reproducibility
    'verbose': 200,               # Lorenzo Giannuzzo: Print progress every 100 iterations
    'task_type': 'CPU',           # Lorenzo Giannuzzo: Use CPU (change to 'GPU' if available)
}


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def compute_delta_features(df):
    """
    Compute the same Delta features used by ANN/CNN/LSTM models.

    These are the 6 computed features that, together with the 4 raw features,
    form the standard 10-feature set used across all models.

    Args:
        df: DataFrame with raw columns (Voltage, Current_inv, Temp, SOC_Calc)

    Returns:
        DataFrame with Delta features added
    """
    df = df.copy()

    # Lorenzo Giannuzzo: Delta features (same as ANN/CNN/LSTM preprocessing)
    df['Delta_Temperature'] = df['Temp'].diff().fillna(0)
    df['Delta_Current'] = df['Current_inv'].diff().fillna(0)
    df['Delta_Voltage'] = df['Voltage'].diff().fillna(0)
    df['Delta_SoC_Calc'] = df['SOC_Calc'].diff().fillna(0)

    # Lorenzo Giannuzzo: Ratio features
    df['Delta_Voltage_Current_Ratio'] = np.where(
        df['Delta_Current'].abs() > 1e-6,
        df['Delta_Voltage'] / df['Delta_Current'],
        0.0
    )
    df['Current_Voltage_Ratio'] = np.where(
        df['Voltage'].abs() > 1e-6,
        df['Current_inv'] / df['Voltage'],
        0.0
    )

    return df


def create_lag_features(df, raw_features, lag_steps):
    """
    Create lag features for raw measurement features.

    Lag features are applied only to raw measurements (not Delta features)
    since Deltas are already temporal in nature.

    Args:
        df: DataFrame with features
        raw_features: List of raw feature column names to create lags for
        lag_steps: List of lag timesteps

    Returns:
        DataFrame with lag features added
    """
    df = df.copy()

    for feature in raw_features:
        for lag in lag_steps:
            col_name = f'{feature}_lag_{lag}'
            df[col_name] = df[feature].shift(lag)

    return df


def engineer_features(df, raw_features=RAW_FEATURES, lag_steps=LAG_STEPS):
    """
    Apply all feature engineering steps.

    1. Compute Delta features (same as ANN/CNN/LSTM)
    2. Create lag features for raw measurements

    Args:
        df: Input DataFrame with raw columns
        raw_features: Raw feature columns for lag creation
        lag_steps: Lag timesteps

    Returns:
        DataFrame with all engineered features
    """
    print(f"         Base features: {len(BASE_FEATURES)} (same as ANN/CNN/LSTM)")
    print(f"         Lag steps: {lag_steps} (on raw features only)")

    # Lorenzo Giannuzzo: Step 1: Compute Delta features (same as other models)
    df = compute_delta_features(df)

    # Lorenzo Giannuzzo: Step 2: Create lag features for raw measurements
    df = create_lag_features(df, raw_features, lag_steps)

    # Lorenzo Giannuzzo: Drop rows with NaN from lag features
    max_lag = max(lag_steps)
    df = df.iloc[max_lag:].reset_index(drop=True)

    feature_cols = get_feature_columns()
    print(f"         Total features: {len(feature_cols)} "
          f"(10 base + {len(feature_cols) - 10} lag)")

    return df


def get_feature_columns(base_features=BASE_FEATURES, raw_features=RAW_FEATURES,
                        lag_steps=LAG_STEPS):
    """
    Get list of all feature column names after engineering.

    Returns:
        List of feature column names
    """
    features = list(base_features)

    # Lag features (only on raw measurements)
    for feature in raw_features:
        for lag in lag_steps:
            features.append(f'{feature}_lag_{lag}')

    return features


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def standardize_columns(df):
    """Standardize column names to expected format."""
    column_mappings = {
        'Voltage(V)': 'Voltage',
        'Current_inv(A)': 'Current_inv',
        'SOC_true(%)': 'SOC_true',
        'SOC_True(%)': 'SOC_true',
        'SOC_True': 'SOC_true',
        'Temp1': 'Temp',
        'Step(s)': 'Step',
    }
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Lorenzo Giannuzzo: Convert numeric columns to float
    numeric_columns = ['Voltage', 'Current_inv', 'SOC_true', 'Temp', 'Step', 'SOC_Calc']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def run_kalman_filter_on_file(file_name, scenario_name=''):
    """
    Run Kalman Filter on a single training file to generate realistic SOC_Calc.

    Uses the local KalmanFilter_module.py which loads files from ../train_data/
    automatically given just the filename.

    Args:
        file_name: Name of the Excel file (e.g. 'GenerateTestData_S1_DAY0to4.xlsx')
        scenario_name: Name of the file (for capacity detection)

    Returns:
        DataFrame with SOC_Calc column from Kalman Filter
    """
    # Lorenzo Giannuzzo: Determine capacity based on scenario
    if 'HPPC' in scenario_name and 'DSG' in scenario_name:
        capacity_charge = 3600
        capacity_discharge = 3600
    else:
        capacity_charge = 280
        capacity_discharge = 280

    # Lorenzo Giannuzzo: The local KalmanFilter_module loads the file from ../train_data/ internally
    # Lorenzo Giannuzzo: We need to read the file first to get initial SOC
    file_path = os.path.join('..', 'train_data', file_name)
    temp_df = pd.read_excel(file_path, skiprows=2).iloc[1:]
    temp_df = standardize_columns(temp_df)

    soc_initial = float(temp_df['SOC_true'].iloc[0])
    if np.isnan(soc_initial):
        soc_initial = 50.0

    # Lorenzo Giannuzzo:  Run KF with same parameters as main evaluation script
    KF = KalmanFilter(file_name, dt=1, capacity_charge=capacity_charge,
                      capacity_discharge=capacity_discharge,
                      opt=False, initial_SOC=soc_initial)
    P = np.array([[3.393996e-03, 0, 0], [0, 6.662388e-04, 0], [0, 0, 6.946038e-03]])
    Q = np.array([[4.034329e-07, 0, 0], [0, 8.628196e-05, 0], [0, 0, 9.095089e-05]])
    R = 3.743003e-03

    updated_data = KF.kalman_filter(Q=Q, R=R, initial_covariance=P)

    # Lorenzo Giannuzzo: Standardize columns on the KF output
    updated_data = standardize_columns(updated_data)

    return updated_data


def load_training_data(data_path):
    """
    Load training data from folder.
    Each file is passed through the Kalman Filter to generate realistic SOC_Calc.

    Args:
        data_path: Path to training data folder

    Returns:
        DataFrame with training data (SOC_Calc from Kalman Filter)
    """
    data_path = Path(data_path)

    if data_path.is_dir():
        all_dfs = []

        files = list(data_path.glob('*.xlsx'))
        # Exclude HPPC and OCV files for training
        files = [f for f in files if 'HPPC' not in f.name and 'OCV' not in f.name]

        print(f"         Found {len(files)} training files:")

        for file_path in files:
            try:
                # Run Kalman Filter (KF loads the file internally from ../train_data/)
                df = run_kalman_filter_on_file(file_path.name, scenario_name=file_path.stem)

                # Check required columns
                required = ['Voltage', 'Current_inv', 'Temp', 'SOC_true', 'SOC_Calc']
                if all(col in df.columns for col in required):
                    all_dfs.append(df)
                    print(f"           - {file_path.name} ({len(df):,} samples) [KF applied]")
                else:
                    missing = [c for c in required if c not in df.columns]
                    print(f"           - {file_path.name} [SKIP: missing {missing}]")
            except Exception as e:
                print(f"           - {file_path.name} [ERROR: {e}]")

        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            print(f"         Total: {len(combined):,} samples")
            return combined
        else:
            raise ValueError("No valid training files found")

    else:
        raise ValueError(f"Invalid data path: {data_path}")


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_catboost_model(X_train, y_train, X_val, y_val, params=CATBOOST_PARAMS):
    """
    Train CatBoost model.

    Args:
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        params: CatBoost parameters

    Returns:
        Trained CatBoost model
    """
    print(f"\n  Training CatBoost model...")
    print(f"    Parameters:")
    for key, value in params.items():
        print(f"      {key}: {value}")

    # Lorenzo Giannuzzo: Create CatBoost pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Lorenzo Giannuzzo: Initialize and train model
    model = CatBoostRegressor(**params)

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        plot=False
    )

    print(f"    Best iteration: {model.best_iteration_}")

    return model


def evaluate_model(model, X, y, dataset_name="Test"):
    """
    Evaluate model performance.

    Args:
        model: Trained model
        X: Features
        y: True values
        dataset_name: Name for printing

    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    max_ae = np.max(np.abs(y - y_pred))
    r2 = r2_score(y, y_pred)

    print(f"\n  {dataset_name} Metrics:")
    print(f"    RMSE:   {rmse:.4f}")
    print(f"    MAE:    {mae:.4f}")
    print(f"    MaxAE:  {max_ae:.4f}")
    print(f"    R²:     {r2:.4f}")

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MaxAE': max_ae,
        'R2': r2
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_feature_importance(model, feature_names, save_path):
    """
    Plot and save feature importance.

    Args:
        model: Trained CatBoost model
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    importance = model.feature_importances_

    # Lorenzo Giannuzzo: Sort by importance
    indices = np.argsort(importance)[::-1]

    # Lorenzo Giannuzzo: Plot top 30 features
    n_features = min(30, len(feature_names))

    fig, ax = plt.subplots(figsize=(10, 12))

    top_indices = indices[:n_features]
    top_names = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]

    # Lorenzo Giannuzzo: Use magma colormap
    colors = plt.cm.magma(np.linspace(0.2, 0.8, n_features))

    bars = ax.barh(range(n_features), top_importance, color=colors)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('CatBoost Feature Importance (Top 30)')
    ax.grid(True, alpha=0.3, axis='x')

    # Lorenzo Giannuzzo: Add percentage labels
    total_importance = np.sum(importance)
    for i, (bar, imp) in enumerate(zip(bars, top_importance)):
        pct = imp / total_importance * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"         Feature importance plot saved")


def plot_predictions(y_true, y_pred, save_path, title="CatBoost SOC Prediction"):
    """
    Plot true vs predicted values.

    Args:
        y_true: True SOC values
        y_pred: Predicted SOC values
        save_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lorenzo Giannuzzo: Time series plot (first 1000 samples)
    ax = axes[0]
    ax.plot(y_true[:1000], label='True SOC', linewidth=1)
    ax.plot(y_pred[:1000], label='Predicted SOC', linewidth=1, alpha=0.8)
    ax.set_xlabel('Sample')
    ax.set_ylabel('SOC [%]')
    ax.set_title(f'{title} - Time Series (first 1000 samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Lorenzo Giannuzzo: Scatter plot (subsampled to max 5000 points to avoid huge PDF)
    ax = axes[1]
    n_points = len(y_true)
    max_scatter = 5000
    if n_points > max_scatter:
        idx = np.random.choice(n_points, max_scatter, replace=False)
        scatter_true = y_true[idx]
        scatter_pred = y_pred[idx]
    else:
        scatter_true = y_true
        scatter_pred = y_pred

    ax.scatter(scatter_true, scatter_pred, alpha=0.3, s=1, rasterized=True)
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlabel('True SOC [%]')
    ax.set_ylabel('Predicted SOC [%]')
    ax.set_title(f'{title} - Scatter Plot ({min(n_points, max_scatter):,} points)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"         Predictions plot saved")


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

if __name__ == '__main__':

    total_start = time.time()

    # Lorenzo Giannuzzo: Paths - resolve to handle PyCharm file:/ URI issue
    script_file = os.path.abspath(__file__)
    script_dir = Path(script_file).parent

    # Lorenzo Giannuzzo: Training data path (relative, one level up)
    data_path = script_dir.parent / 'train_data'

    # Lorenzo Giannuzzo: Change working directory to nn_training/ so KalmanFilter_module.py
    # Lorenzo Giannuzzo: can resolve its relative paths (../train_data/, ../data/)
    os.chdir(script_dir)
    print(f"  Working directory set to: {script_dir}")

    # Output directories
    models_dir = script_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = models_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    print("\n" + "="*70)
    print("  CATBOOST SOC PREDICTION - TRAINING")
    print("="*70)

    # ========================================================================
    # 1. LOAD DATA
    # ========================================================================
    print("\n  [1/5] Loading training data...")
    print(f"         Path: {data_path}")

    try:
        df = load_training_data(data_path)
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print(f"  Resolved path: {data_path.resolve()}")
        print("  Please update 'data_path' variable to point to your training data.")
        exit(1)

    # Lorenzo Giannuzzo: Verify SOC_Calc was generated by Kalman Filter
    if 'SOC_Calc' not in df.columns:
        print("\n  ERROR: SOC_Calc not found after Kalman Filter processing.")
        print("  This should not happen. Check KalmanFilter_module_all_data.")
        exit(1)
    else:
        # Verify SOC_Calc differs from SOC_true (KF was actually applied)
        soc_diff = np.mean(np.abs(df['SOC_Calc'].astype(float) - df['SOC_true'].astype(float)))
        print(f"         SOC_Calc from Kalman Filter (avg deviation from true: {soc_diff:.4f}%)")

    # ========================================================================
    # 2. FEATURE ENGINEERING
    # ========================================================================
    print("\n  [2/5] Engineering features...")

    df_features = engineer_features(df)
    feature_columns = get_feature_columns()

    # Lorenzo Giannuzzo: Prepare X and y
    X = df_features[feature_columns].values.astype(float)
    y = df_features['SOC_true'].values.astype(float)

    # Lorenzo Giannuzzo: Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=50.0)

    print(f"         Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # ========================================================================
    # 3. SCALE DATA
    # ========================================================================
    print("\n  [3/5] Scaling features...")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Lorenzo Giannuzzo: Save scaler and feature columns
    scaler_path = models_dir / 'catboost_scaler.joblib'
    feature_cols_path = models_dir / 'catboost_feature_columns.joblib'
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_columns, feature_cols_path)

    print(f"         Scaler saved:   {scaler_path.name}")
    print(f"         Features saved: {feature_cols_path.name}")

    # ========================================================================
    # 4. TRAIN MODEL ON 100% DATA
    # ========================================================================
    print("\n  [4/5] Training CatBoost model on 100% data...")
    print(f"         Iterations: {CATBOOST_PARAMS['iterations']}")
    print(f"         Samples:    {X_scaled.shape[0]:,}")

    # Lorenzo Giannuzzo: Save catboost_info inside models/ folder
    params_with_dir = CATBOOST_PARAMS.copy()
    params_with_dir['train_dir'] = str(models_dir / 'catboost_info')

    model = CatBoostRegressor(**params_with_dir)
    model.fit(Pool(X_scaled, y), plot=False)

    # Lorenzo Giannuzzo: Save model
    model_path = models_dir / 'catboost_model.cbm'
    model.save_model(str(model_path))

    # ========================================================================
    # 5. EVALUATE AND SAVE PLOTS
    # ========================================================================
    print("\n  [5/5] Evaluating and saving plots...")

    y_pred = model.predict(X_scaled)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    max_ae = np.max(np.abs(y - y_pred))
    r2 = r2_score(y, y_pred)

    plot_feature_importance(
        model, feature_columns,
        plots_dir / 'catboost_feature_importance.pdf'
    )

    plot_predictions(
        y, y_pred,
        plots_dir / 'catboost_predictions.pdf'
    )

    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed = time.time() - total_start

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)

    print(f"""
  Time:          {elapsed:.1f}s
  Iterations:    {CATBOOST_PARAMS['iterations']}
  Total samples: {X_scaled.shape[0]:,}
  Features:      {X_scaled.shape[1]}

  Training Metrics (sanity check):
  ┌──────────┬────────────┐
  │ RMSE     │ {rmse:>10.4f} │
  │ MAE      │ {mae:>10.4f} │
  │ MaxAE    │ {max_ae:>10.4f} │
  │ R²       │ {r2:>10.4f} │
  └──────────┴────────────┘

  Output files:
    {model_path}
    {scaler_path}
    {feature_cols_path}
    {plots_dir / 'catboost_feature_importance.pdf'}
    {plots_dir / 'catboost_predictions.pdf'}""")

    print("\n" + "="*70)