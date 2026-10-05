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
================================================================================
Ensemble Meta-Model Training for SOC Prediction - V2
================================================================================
CHANGES FROM V1:
- Removed ALL early stopping callbacks (full epoch training)
- Set 100 epochs for ALL models (Neural, Dynamic, Selector)
- No ReduceLROnPlateau callbacks

This script creates a meta-model that learns to optimally combine predictions
from multiple base models (KF, ANN, CNN, LSTM, CatBoost, TFT) using:

1. Stacking with XGBoost meta-learner
2. Stacking with Neural Network meta-learner
3. Dynamic Weighting Network (learns context-dependent weights)
4. Model Selector Network (learns to SELECT the best model per timestep)

The meta-model learns WHEN to trust each base model based on operating conditions.

Authors:
    Lorenzo Giannuzzo - lorenzo.giannuzzo@polito.it

Affiliation:
    Politecnico di Torino - Energy Center Lab
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump, load

# TensorFlow/Keras for Neural Network meta-learner
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model, regularizers

# XGBoost for tree-based meta-learner
try:
    import xgb as xgb
    HAS_XGBOOST = True
except ImportError:
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
    except ImportError:
        HAS_XGBOOST = False
        print("Warning: XGBoost not installed. Using Ridge regression instead.")


# ============================================================================
# CONFIGURATION
# ============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5  # For cross-validation stacking

# Training epochs - ALL MODELS USE 100 EPOCHS (no early stopping)
EPOCHS = 100

# Base model names
BASE_MODELS = ['KF', 'ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']

# Features used for dynamic weighting (operating conditions)
CONTEXT_FEATURES = [
    'Temp', 'Voltage', 'Current_inv', 'SOC_Calc',
    'Delta_Temperature', 'Delta_Current', 'Delta_Voltage',
    'Delta_SoC_Calc', 'Current_Voltage_Ratio'
]

# Training files (in train_data/ folder)
TRAINING_FILES = [
    'train_data/GenerateTestData_S1_DAY0to4.xlsx',
    'train_data/GenerateTestData_S2_DAY4to7.xlsx',
    'train_data/GenerateTestData_S3_DAY0to4.xlsx',
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def preprocess_data(df):
    """Add delta features to dataframe."""
    df = df.copy()

    if 'Current_Voltage_Ratio' not in df.columns:
        df['Current_Voltage_Ratio'] = (df['Current_inv'] / df['Voltage']).fillna(0)

    if 'Delta_Current' not in df.columns:
        df['Delta_Current'] = df['Current_inv'].diff().fillna(0)

    if 'Delta_Voltage' not in df.columns:
        df['Delta_Voltage'] = df['Voltage'].diff().fillna(0)

    if 'Delta_Voltage_Current_Ratio' not in df.columns:
        df['Delta_Voltage_Current_Ratio'] = (
            df['Delta_Voltage'] / df['Delta_Current'].replace(0, 1)
        ).fillna(0)

    if 'Delta_SoC_Calc' not in df.columns:
        df['Delta_SoC_Calc'] = df['SOC_Calc'].diff().fillna(0)

    if 'Delta_Temperature' not in df.columns:
        df['Delta_Temperature'] = df['Temp'].diff().fillna(0)

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


# ============================================================================
# BASE MODEL PREDICTION FUNCTIONS
# ============================================================================
def get_kf_predictions(df):
    """Get Kalman Filter predictions (SOC_Calc)."""
    return df['SOC_Calc'].values


def get_ann_predictions(df, model_path, scaler_path):
    """Get ANN predictions."""
    from ANN_prediction_module_all_data import ANN_BatteryAnalysis
    model = ANN_BatteryAnalysis(model_path, scaler_path)
    result = model.make_predictions(df.copy())
    return result['Predicted_SOC'].values


def get_cnn_predictions(df, model_path, scaler_path, sequence_length=10):
    """Get CNN predictions."""
    from CNN_prediction_module_all_data import CNN_BatteryAnalysis
    model = CNN_BatteryAnalysis(model_path, scaler_path, sequence_length=sequence_length)
    result = model.make_predictions(df.copy())
    return result['Predicted_SOC'].values


def get_lstm_predictions(df, model_path, scaler_path, sequence_length=10):
    """Get LSTM predictions."""
    from LSTM_prediction_module_all_data import LSTM_BatteryAnalysis
    model = LSTM_BatteryAnalysis(model_path, scaler_path, sequence_length=sequence_length)
    result = model.make_predictions(df.copy())
    return result['Predicted_SOC'].values


def get_catboost_predictions(df, model_path, scaler_path, features_path=None):
    """Get CatBoost predictions."""
    from CATBOOST_prediction_module_all_data import CATBOOST_BatteryAnalysis
    model = CATBOOST_BatteryAnalysis(model_path, scaler_path, feature_columns_path=features_path)
    result = model.make_predictions(df.copy())
    return result['Predicted_SOC'].values


def get_tft_predictions(df, model_path, scaler_path, config_path=None):
    """Get TFT predictions."""
    from TFT_prediction_module_all_data import TFT_BatteryAnalysis
    model = TFT_BatteryAnalysis(model_path, scaler_path, config_path=config_path)
    result = model.make_predictions(df.copy())
    return result['Predicted_SOC'].values


# ============================================================================
# META-LEARNER ARCHITECTURES
# ============================================================================

class StackingMetaLearner:
    """
    Classic stacking meta-learner using XGBoost or Ridge regression.
    Input: predictions from all base models + context features
    Output: final SOC prediction
    """

    def __init__(self, use_xgboost=True):
        self.use_xgboost = use_xgboost and HAS_XGBOOST
        self.model = None
        self.scaler = MinMaxScaler()

    def fit(self, X, y):
        """
        X: array of shape (n_samples, n_base_models + n_context_features)
        y: true SOC values
        """
        X_scaled = self.scaler.fit_transform(X)

        if self.use_xgboost:
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0
            )
        else:
            self.model = Ridge(alpha=1.0)

        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, model_path, scaler_path):
        if self.use_xgboost:
            self.model.save_model(model_path)
        else:
            dump(self.model, model_path)
        dump(self.scaler, scaler_path)

    def load(self, model_path, scaler_path):
        if self.use_xgboost:
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
        else:
            self.model = load(model_path)
        self.scaler = load(scaler_path)
        return self


class NeuralStackingMetaLearner:
    """
    Neural network meta-learner for stacking.
    A small MLP that learns to combine base model predictions.

    V2: No early stopping, full 100 epochs training.
    """

    def __init__(self, n_inputs, hidden_units=[64, 32]):
        self.n_inputs = n_inputs
        self.hidden_units = hidden_units
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None

    def _build_model(self):
        inputs = layers.Input(shape=(self.n_inputs,))
        x = inputs

        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization()(x)

        outputs = layers.Dense(1, activation='linear')(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def fit(self, X, y, epochs=EPOCHS, batch_size=256, validation_split=0.1, verbose=1):
        X_scaled = self.scaler.fit_transform(X)

        self.model = self._build_model()

        # V2: NO CALLBACKS - full epoch training
        self.history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

    def save(self, model_path, scaler_path):
        self.model.save(model_path)
        dump(self.scaler, scaler_path)

    def load(self, model_path, scaler_path):
        self.model = keras.models.load_model(model_path)
        self.scaler = load(scaler_path)
        return self


class DynamicWeightingNetwork:
    """
    Dynamic Weighting Network (Mixture of Experts style).

    Instead of learning a fixed combination, this network learns
    CONTEXT-DEPENDENT WEIGHTS for each base model.

    Architecture:
    - Gating Network: context_features -> weights (softmax)
    - Output: sum(weights * base_predictions)

    This allows the model to learn patterns like:
    - "At low SOC, trust LSTM more"
    - "At high temperature, trust CatBoost more"

    V2: No early stopping, full 100 epochs training.
    """

    def __init__(self, n_base_models, n_context_features, hidden_units=[64, 32]):
        self.n_base_models = n_base_models
        self.n_context_features = n_context_features
        self.hidden_units = hidden_units
        self.model = None
        self.context_scaler = MinMaxScaler()
        self.history = None

    def _build_model(self):
        # Input: base model predictions
        pred_input = layers.Input(shape=(self.n_base_models,), name='predictions')

        # Input: context features
        context_input = layers.Input(shape=(self.n_context_features,), name='context')

        # Gating network: learns weights based on context
        x = context_input
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization()(x)

        # Output weights (softmax ensures they sum to 1)
        weights = layers.Dense(self.n_base_models, activation='softmax', name='weights')(x)

        # Weighted combination of predictions
        weighted_pred = layers.Multiply()([pred_input, weights])
        # Use Dense(1) with fixed weights=1 instead of Lambda (better serialization)
        output = layers.Dense(1, use_bias=False,
                              kernel_initializer='ones',
                              trainable=False,
                              name='sum_output')(weighted_pred)

        model = Model(inputs=[pred_input, context_input], outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Also create a model to extract weights for analysis
        self.weight_model = Model(inputs=context_input, outputs=weights)

        return model

    def fit(self, base_predictions, context_features, y_true,
            epochs=EPOCHS, batch_size=256, validation_split=0.1, verbose=1):
        """
        base_predictions: array of shape (n_samples, n_base_models)
        context_features: array of shape (n_samples, n_context_features)
        y_true: true SOC values
        """
        context_scaled = self.context_scaler.fit_transform(context_features)

        self.model = self._build_model()

        # V2: NO CALLBACKS - full epoch training
        self.history = self.model.fit(
            [base_predictions, context_scaled], y_true,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return self

    def predict(self, base_predictions, context_features):
        context_scaled = self.context_scaler.transform(context_features)
        return self.model.predict([base_predictions, context_scaled], verbose=0).flatten()

    def get_weights(self, context_features):
        """Get the learned weights for given context."""
        context_scaled = self.context_scaler.transform(context_features)
        return self.weight_model.predict(context_scaled, verbose=0)

    def save(self, model_path, scaler_path):
        self.model.save(model_path)
        dump(self.context_scaler, scaler_path)

    def load(self, model_path, scaler_path):
        self.model = keras.models.load_model(model_path)
        self.context_scaler = load(scaler_path)
        # Rebuild weight model
        context_input = self.model.get_layer('context').input
        weights_output = self.model.get_layer('weights').output
        self.weight_model = Model(inputs=context_input, outputs=weights_output)
        return self


class ModelSelectorNetwork:
    """
    Model Selector Network - HARD SELECTION instead of soft weighting.

    This network learns to SELECT the single best model for each timestep,
    rather than computing a weighted average of all models.

    Architecture:
    - Classifier Network: context_features + predictions -> class probabilities
    - Output: prediction from the SELECTED model only

    Training:
    - For each sample, find which model has minimum error
    - That model index becomes the classification target
    - Train with CrossEntropyLoss

    Inference:
    - Predict which model to use (argmax of class probabilities)
    - Return prediction from ONLY that model

    Advantages:
    - No "dilution" from weak models
    - Can completely ignore models that don't work well
    - More interpretable: "I chose TFT because..."

    V2: No early stopping, full 100 epochs training.
    """

    def __init__(self, n_base_models, n_context_features, hidden_units=[128, 64, 32]):
        self.n_base_models = n_base_models
        self.n_context_features = n_context_features
        self.hidden_units = hidden_units
        self.model = None
        self.context_scaler = MinMaxScaler()
        self.history = None
        self.selection_stats = None  # Track which models get selected

    def _build_model(self):
        # Input: base model predictions (used as features to help decide)
        pred_input = layers.Input(shape=(self.n_base_models,), name='predictions')

        # Input: context features
        context_input = layers.Input(shape=(self.n_context_features,), name='context')

        # Concatenate predictions and context for the classifier
        # The predictions help the network learn patterns like:
        # "when models disagree a lot, trust TFT"
        # "when KF and ANN are close, use KF"
        combined = layers.Concatenate()([pred_input, context_input])

        # Classifier network
        x = combined
        for units in self.hidden_units:
            x = layers.Dense(units, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001))(x)
            x = layers.BatchNormalization()(x)

        # Output: probability distribution over models (which one to select)
        selection_probs = layers.Dense(self.n_base_models, activation='softmax',
                                        name='selection')(x)

        model = Model(inputs=[pred_input, context_input], outputs=selection_probs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',  # Classification loss
            metrics=['accuracy']
        )

        return model

    def _compute_best_model_labels(self, base_predictions, y_true):
        """
        For each sample, find which model has the minimum absolute error.
        Returns: array of model indices (0 to n_base_models-1)
        """
        # Compute absolute errors for each model
        errors = np.abs(base_predictions - y_true.reshape(-1, 1))  # (n_samples, n_models)

        # Find model with minimum error for each sample
        best_model_indices = np.argmin(errors, axis=1)  # (n_samples,)

        return best_model_indices

    def fit(self, base_predictions, context_features, y_true,
            epochs=EPOCHS, batch_size=256, validation_split=0.1, verbose=1):
        """
        Train the selector network.

        base_predictions: array of shape (n_samples, n_base_models)
        context_features: array of shape (n_samples, n_context_features)
        y_true: true SOC values (used to compute which model was best)
        """
        # Scale context features
        context_scaled = self.context_scaler.fit_transform(context_features)

        # Compute target labels: which model is best for each sample
        y_labels = self._compute_best_model_labels(base_predictions, y_true)

        # Print label distribution
        unique, counts = np.unique(y_labels, return_counts=True)
        print(f"\n    Best model distribution in training data:")
        for idx, count in zip(unique, counts):
            pct = count / len(y_labels) * 100
            print(f"      Model {idx}: {count:,} samples ({pct:.1f}%)")

        self.model = self._build_model()

        # V2: NO CALLBACKS - full epoch training
        self.history = self.model.fit(
            [base_predictions, context_scaled], y_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return self

    def predict(self, base_predictions, context_features):
        """
        Make predictions using hard model selection.

        For each sample:
        1. Predict which model to use
        2. Return that model's prediction
        """
        context_scaled = self.context_scaler.transform(context_features)

        # Get selection probabilities
        probs = self.model.predict([base_predictions, context_scaled], verbose=0)

        # Hard selection: pick the model with highest probability
        selected_models = np.argmax(probs, axis=1)

        # Get prediction from selected model for each sample
        n_samples = len(selected_models)
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            y_pred[i] = base_predictions[i, selected_models[i]]

        # Store selection stats
        self.selection_stats = selected_models

        return y_pred

    def get_selection_probabilities(self, context_features, base_predictions):
        """Get the selection probabilities for analysis."""
        context_scaled = self.context_scaler.transform(context_features)
        return self.model.predict([base_predictions, context_scaled], verbose=0)

    def get_selected_models(self, context_features, base_predictions):
        """Get which model was selected for each sample."""
        probs = self.get_selection_probabilities(context_features, base_predictions)
        return np.argmax(probs, axis=1)

    def save(self, model_path, scaler_path):
        self.model.save(model_path)
        dump(self.context_scaler, scaler_path)

    def load(self, model_path, scaler_path):
        self.model = keras.models.load_model(model_path)
        self.context_scaler = load(scaler_path)
        return self


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_ensemble_dataset(file_paths, model_paths, sequence_length=10):
    """
    Prepare dataset for ensemble training.
    Returns: base_predictions, context_features, y_true
    """
    print("\n" + "="*70)
    print("  PREPARING ENSEMBLE TRAINING DATA")
    print("="*70)

    # Import Kalman Filter module (same directory as this script)
    from KalmanFilter_module_all_data import KalmanFilter

    all_base_predictions = []
    all_context_features = []
    all_y_true = []

    for file_idx, file_path in enumerate(file_paths):
        print(f"\n  [{file_idx+1}/{len(file_paths)}] Processing: {file_path}")

        # Load raw data
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = Path(__file__).parent / file_path

        # Read Excel file (same format as main.py: skiprows=2, then skip first row)
        bat_data_df = pd.read_excel(full_path, skiprows=2).iloc[1:]
        bat_data_df = bat_data_df.reset_index(drop=True)

        print(f"    Original columns: {list(bat_data_df.columns)}")

        # Standardize columns BEFORE Kalman Filter
        column_mappings = {
            'Voltage(V)': 'Voltage',
            'Current_inv(A)': 'Current_inv',
            'Current(A)': 'Current_inv',
            'SOC_true(%)': 'SOC_true',
            'SOC_True(%)': 'SOC_true',
            'SOC_True': 'SOC_true',
            'Temp1': 'Temp',
            'Temperature': 'Temp',
            'Step(s)': 'Step',
        }
        for old_name, new_name in column_mappings.items():
            if old_name in bat_data_df.columns:
                bat_data_df[new_name] = bat_data_df[old_name]

        print(f"    After mapping: {list(bat_data_df.columns)}")

        # Convert numeric columns
        numeric_columns = ['Voltage', 'Current_inv', 'SOC_true', 'Temp', 'Step']
        for col in numeric_columns:
            if col in bat_data_df.columns:
                bat_data_df[col] = pd.to_numeric(bat_data_df[col], errors='coerce')

        if 'Temp' not in bat_data_df.columns:
            bat_data_df['Temp'] = 25.0

        # Check required columns
        required_cols = ['Voltage', 'Current_inv', 'SOC_true']
        missing = [c for c in required_cols if c not in bat_data_df.columns]
        if missing:
            print(f"    ERROR: Missing columns: {missing}")
            print(f"    Available columns: {list(bat_data_df.columns)}")
            continue

        # Get initial SOC
        soc_initial = float(bat_data_df['SOC_true'].iloc[0]) if 'SOC_true' in bat_data_df.columns else 100.0
        if np.isnan(soc_initial):
            soc_initial = 100.0

        # Run Kalman Filter
        print("    Running Kalman Filter...")
        KF = KalmanFilter(bat_data_df, dt=1, capacity_charge=280, capacity_discharge=280,
                          opt=False, initial_SOC=soc_initial)
        P = np.array([[3.393996e-03, 0, 0], [0, 6.662388e-04, 0], [0, 0, 6.946038e-03]])
        Q = np.array([[4.034329e-07, 0, 0], [0, 8.628196e-05, 0], [0, 0, 9.095089e-05]])
        R = 3.743003e-03
        df = KF.kalman_filter(Q=Q, R=R, initial_covariance=P)

        df = preprocess_data(df)

        y_true = df['SOC_true'].values
        n_samples = len(y_true)

        # Get predictions from each base model
        predictions = {}

        # KF predictions
        print("    Getting KF predictions...")
        predictions['KF'] = get_kf_predictions(df)

        # ANN predictions
        if model_paths['ANN']['model'].exists():
            print("    Getting ANN predictions...")
            predictions['ANN'] = get_ann_predictions(
                df, str(model_paths['ANN']['model']), str(model_paths['ANN']['scaler'])
            )

        # CNN predictions
        if model_paths['CNN']['model'].exists():
            print("    Getting CNN predictions...")
            predictions['CNN'] = get_cnn_predictions(
                df, str(model_paths['CNN']['model']), str(model_paths['CNN']['scaler']),
                sequence_length
            )

        # LSTM predictions
        if model_paths['LSTM']['model'].exists():
            print("    Getting LSTM predictions...")
            predictions['LSTM'] = get_lstm_predictions(
                df, str(model_paths['LSTM']['model']), str(model_paths['LSTM']['scaler']),
                sequence_length
            )

        # CatBoost predictions
        if model_paths['CatBoost']['model'].exists():
            print("    Getting CatBoost predictions...")
            features_path = str(model_paths['CatBoost']['features']) if model_paths['CatBoost']['features'].exists() else None
            predictions['CatBoost'] = get_catboost_predictions(
                df, str(model_paths['CatBoost']['model']), str(model_paths['CatBoost']['scaler']),
                features_path
            )

        # TFT predictions (if available)
        if model_paths['TFT']['model'].exists():
            print("    Getting TFT predictions...")
            config_path = str(model_paths['TFT']['config']) if model_paths['TFT']['config'].exists() else None
            predictions['TFT'] = get_tft_predictions(
                df, str(model_paths['TFT']['model']), str(model_paths['TFT']['scaler']),
                config_path
            )

        # Build prediction matrix
        available_models = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[m] for m in available_models])

        # Get context features
        context_cols = [c for c in CONTEXT_FEATURES if c in df.columns]
        context_matrix = df[context_cols].values

        # Handle NaN values
        valid_mask = ~np.isnan(pred_matrix).any(axis=1) & ~np.isnan(context_matrix).any(axis=1) & ~np.isnan(y_true)

        all_base_predictions.append(pred_matrix[valid_mask])
        all_context_features.append(context_matrix[valid_mask])
        all_y_true.append(y_true[valid_mask])

        print(f"    Valid samples: {valid_mask.sum():,} / {n_samples:,}")

    # Concatenate all data
    base_predictions = np.vstack(all_base_predictions)
    context_features = np.vstack(all_context_features)
    y_true = np.concatenate(all_y_true)

    print(f"\n  Total samples: {len(y_true):,}")
    print(f"  Base models: {available_models}")
    print(f"  Context features: {context_cols}")

    return base_predictions, context_features, y_true, available_models, context_cols


# ============================================================================
# TRAINING
# ============================================================================

def train_ensemble_models(base_predictions, context_features, y_true,
                          available_models, context_cols, output_dir):
    """
    Train all four types of meta-learners and compare.

    V2: All models train for full 100 epochs (no early stopping).
    """
    print("\n" + "="*70)
    print("  TRAINING ENSEMBLE META-MODELS (V2 - Full Epochs)")
    print(f"  All neural models will train for {EPOCHS} epochs (no early stopping)")
    print("="*70)

    # Split data
    X_stacking = np.hstack([base_predictions, context_features])

    (X_stack_train, X_stack_test,
     pred_train, pred_test,
     ctx_train, ctx_test,
     y_train, y_test) = train_test_split(
        X_stacking, base_predictions, context_features, y_true,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = {}

    # ========================================================================
    # 1. STACKING META-LEARNER (XGBoost/Ridge) - unchanged
    # ========================================================================
    print("\n  [1/4] Training Stacking Meta-Learner (XGBoost)...")
    start_time = time.time()

    stacking_model = StackingMetaLearner(use_xgboost=True)
    stacking_model.fit(X_stack_train, y_train)

    y_pred_stack = stacking_model.predict(X_stack_test)

    results['Stacking'] = {
        'RMSE': root_mean_squared_error(y_test, y_pred_stack),
        'MAE': mean_absolute_error(y_test, y_pred_stack),
        'R2': r2_score(y_test, y_pred_stack),
        'time': time.time() - start_time
    }

    # Save
    stacking_model.save(
        str(output_dir / 'ensemble_stacking_model.json'),
        str(output_dir / 'ensemble_stacking_scaler.joblib')
    )

    print(f"    RMSE: {results['Stacking']['RMSE']:.4f}")
    print(f"    MAE:  {results['Stacking']['MAE']:.4f}")
    print(f"    R²:   {results['Stacking']['R2']:.4f}")
    print(f"    Time: {results['Stacking']['time']:.2f}s")

    # ========================================================================
    # 2. NEURAL STACKING META-LEARNER (100 epochs, no early stopping)
    # ========================================================================
    print(f"\n  [2/4] Training Neural Stacking Meta-Learner ({EPOCHS} epochs, no early stopping)...")
    start_time = time.time()

    neural_model = NeuralStackingMetaLearner(
        n_inputs=X_stacking.shape[1],
        hidden_units=[128, 64, 32]
    )
    neural_model.fit(X_stack_train, y_train, epochs=EPOCHS, batch_size=256, verbose=1)

    y_pred_neural = neural_model.predict(X_stack_test)

    results['Neural'] = {
        'RMSE': root_mean_squared_error(y_test, y_pred_neural),
        'MAE': mean_absolute_error(y_test, y_pred_neural),
        'R2': r2_score(y_test, y_pred_neural),
        'time': time.time() - start_time
    }

    # Save
    neural_model.save(
        str(output_dir / 'ensemble_neural_model.keras'),
        str(output_dir / 'ensemble_neural_scaler.joblib')
    )

    print(f"    RMSE: {results['Neural']['RMSE']:.4f}")
    print(f"    MAE:  {results['Neural']['MAE']:.4f}")
    print(f"    R²:   {results['Neural']['R2']:.4f}")
    print(f"    Time: {results['Neural']['time']:.2f}s")

    # ========================================================================
    # 3. DYNAMIC WEIGHTING NETWORK (100 epochs, no early stopping)
    # ========================================================================
    print(f"\n  [3/4] Training Dynamic Weighting Network ({EPOCHS} epochs, no early stopping)...")
    start_time = time.time()

    dynamic_model = DynamicWeightingNetwork(
        n_base_models=base_predictions.shape[1],
        n_context_features=context_features.shape[1],
        hidden_units=[128, 64, 32]
    )
    dynamic_model.fit(pred_train, ctx_train, y_train, epochs=EPOCHS, batch_size=256, verbose=1)

    y_pred_dynamic = dynamic_model.predict(pred_test, ctx_test)

    results['Dynamic'] = {
        'RMSE': root_mean_squared_error(y_test, y_pred_dynamic),
        'MAE': mean_absolute_error(y_test, y_pred_dynamic),
        'R2': r2_score(y_test, y_pred_dynamic),
        'time': time.time() - start_time
    }

    # Save
    dynamic_model.save(
        str(output_dir / 'ensemble_dynamic_model.keras'),
        str(output_dir / 'ensemble_dynamic_scaler.joblib')
    )

    print(f"    RMSE: {results['Dynamic']['RMSE']:.4f}")
    print(f"    MAE:  {results['Dynamic']['MAE']:.4f}")
    print(f"    R²:   {results['Dynamic']['R2']:.4f}")
    print(f"    Time: {results['Dynamic']['time']:.2f}s")

    # ========================================================================
    # 4. MODEL SELECTOR NETWORK (100 epochs, no early stopping)
    # ========================================================================
    print(f"\n  [4/4] Training Model Selector Network ({EPOCHS} epochs, no early stopping)...")
    start_time = time.time()

    selector_model = ModelSelectorNetwork(
        n_base_models=base_predictions.shape[1],
        n_context_features=context_features.shape[1],
        hidden_units=[128, 64, 32]
    )
    selector_model.fit(pred_train, ctx_train, y_train, epochs=EPOCHS, batch_size=256, verbose=1)

    y_pred_selector = selector_model.predict(pred_test, ctx_test)

    results['Selector'] = {
        'RMSE': root_mean_squared_error(y_test, y_pred_selector),
        'MAE': mean_absolute_error(y_test, y_pred_selector),
        'R2': r2_score(y_test, y_pred_selector),
        'time': time.time() - start_time
    }

    # Save
    selector_model.save(
        str(output_dir / 'ensemble_selector_model.keras'),
        str(output_dir / 'ensemble_selector_scaler.joblib')
    )

    print(f"    RMSE: {results['Selector']['RMSE']:.4f}")
    print(f"    MAE:  {results['Selector']['MAE']:.4f}")
    print(f"    R²:   {results['Selector']['R2']:.4f}")
    print(f"    Time: {results['Selector']['time']:.2f}s")

    # Analyze selector choices
    selected_models = selector_model.get_selected_models(ctx_test, pred_test)
    print(f"\n    Model selection distribution on test set:")
    unique, counts = np.unique(selected_models, return_counts=True)
    for idx, count in zip(unique, counts):
        model_name = available_models[idx]
        pct = count / len(selected_models) * 100
        print(f"      {model_name}: {count:,} times ({pct:.1f}%)")

    # ========================================================================
    # ANALYZE DYNAMIC WEIGHTS (for comparison)
    # ========================================================================
    print("\n  Analyzing learned weights from Dynamic Weighting Network...")

    weights = dynamic_model.get_weights(ctx_test)
    avg_weights = weights.mean(axis=0)

    print(f"\n  Average weights per base model (Dynamic):")
    for i, model_name in enumerate(available_models):
        print(f"    {model_name}: {avg_weights[i]:.4f} ({avg_weights[i]*100:.1f}%)")

    # ========================================================================
    # COMPARE WITH INDIVIDUAL MODELS
    # ========================================================================
    print("\n  Comparison with individual base models:")

    base_results = {}
    for i, model_name in enumerate(available_models):
        y_pred_base = pred_test[:, i]
        base_results[model_name] = {
            'RMSE': root_mean_squared_error(y_test, y_pred_base),
            'MAE': mean_absolute_error(y_test, y_pred_base),
            'R2': r2_score(y_test, y_pred_base)
        }

    print(f"\n  {'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print(f"  {'-'*47}")

    for model_name in available_models:
        r = base_results[model_name]
        print(f"  {model_name:<15} {r['RMSE']:>10.4f} {r['MAE']:>10.4f} {r['R2']:>10.4f}")

    print(f"  {'-'*47}")

    for ensemble_name in ['Stacking', 'Neural', 'Dynamic', 'Selector']:
        r = results[ensemble_name]
        print(f"  {f'Ensemble ({ensemble_name})':<15} {r['RMSE']:>10.4f} {r['MAE']:>10.4f} {r['R2']:>10.4f}")

    return results, base_results, dynamic_model, selector_model, weights, selected_models, available_models


# ============================================================================
# PLOTTING
# ============================================================================

def create_plots(results, base_results, weights, selected_models, available_models, output_dir):
    """Create visualization plots."""

    # Plot 1: RMSE comparison (all models)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Combine results
    all_models = list(base_results.keys()) + ['Stacking', 'Neural', 'Dynamic', 'Selector']
    all_rmse = [base_results[m]['RMSE'] for m in base_results] + \
               [results[m]['RMSE'] for m in ['Stacking', 'Neural', 'Dynamic', 'Selector']]
    all_mae = [base_results[m]['MAE'] for m in base_results] + \
              [results[m]['MAE'] for m in ['Stacking', 'Neural', 'Dynamic', 'Selector']]
    all_r2 = [base_results[m]['R2'] for m in base_results] + \
             [results[m]['R2'] for m in ['Stacking', 'Neural', 'Dynamic', 'Selector']]

    colors = ['#1E88E5', '#42A5F5', '#E65100', '#FB8C00', '#7B1FA2', '#4CAF50'] + \
             ['#2E7D32', '#1B5E20', '#004D40', '#BF360C']  # Last one is red-orange for Selector
    colors = colors[:len(all_models)]

    # RMSE
    ax = axes[0]
    bars = ax.bar(all_models, all_rmse, color=colors, edgecolor='white')
    ax.set_ylabel('RMSE [%]')
    ax.set_title('RMSE Comparison')
    ax.set_xticklabels(all_models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, all_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # MAE
    ax = axes[1]
    bars = ax.bar(all_models, all_mae, color=colors, edgecolor='white')
    ax.set_ylabel('MAE [%]')
    ax.set_title('Mean Absolute Error Comparison')
    ax.set_xticklabels(all_models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, all_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # R²
    ax = axes[2]
    bars = ax.bar(all_models, all_r2, color=colors, edgecolor='white')
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score Comparison')
    ax.set_xticklabels(all_models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    for bar, val in zip(bars, all_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Ensemble vs Base Models Performance (V2 - Full Epochs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_comparison.pdf', dpi=500, bbox_inches='tight')
    plt.savefig(output_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: ENSEMBLE ONLY comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ensemble_models = ['Stacking', 'Neural', 'Dynamic', 'Selector']
    ensemble_rmse = [results[m]['RMSE'] for m in ensemble_models]
    ensemble_mae = [results[m]['MAE'] for m in ensemble_models]
    ensemble_r2 = [results[m]['R2'] for m in ensemble_models]

    ensemble_colors = ['#2E7D32', '#1B5E20', '#004D40', '#BF360C']  # Green shades + red for Selector

    # RMSE
    ax = axes[0]
    bars = ax.bar(ensemble_models, ensemble_rmse, color=ensemble_colors, edgecolor='white', width=0.6)
    ax.set_ylabel('RMSE [%]')
    ax.set_title('RMSE Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, ensemble_rmse):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # MAE
    ax = axes[1]
    bars = ax.bar(ensemble_models, ensemble_mae, color=ensemble_colors, edgecolor='white', width=0.6)
    ax.set_ylabel('MAE [%]')
    ax.set_title('Mean Absolute Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, ensemble_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # R²
    ax = axes[2]
    bars = ax.bar(ensemble_models, ensemble_r2, color=ensemble_colors, edgecolor='white', width=0.6)
    ax.set_ylabel('R² Score')
    ax.set_title('R² Score Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(min(ensemble_r2) - 0.001, 1.001)
    for bar, val in zip(bars, ensemble_r2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{val:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Ensemble Models Comparison (V2 - Full Epochs)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_only_comparison.pdf', dpi=500, bbox_inches='tight')
    plt.savefig(output_dir / 'ensemble_only_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: Dynamic weights distribution + Selector choices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    base_colors = ['#0A2463', '#1E88E5', '#42A5F5', '#E65100', '#FB8C00', '#7B1FA2']
    base_colors = base_colors[:len(available_models)]

    # Average weights pie chart (Dynamic)
    ax = axes[0]
    avg_weights = weights.mean(axis=0)
    ax.pie(avg_weights, labels=available_models, autopct='%1.1f%%',
           colors=base_colors)
    ax.set_title('Average Model Weights\n(Dynamic Weighting Network)')

    # Selector choices pie chart
    ax = axes[1]
    unique, counts = np.unique(selected_models, return_counts=True)
    selector_labels = [available_models[i] for i in unique]
    selector_colors = [base_colors[i] for i in unique]
    ax.pie(counts, labels=selector_labels, autopct='%1.1f%%',
           colors=selector_colors)
    ax.set_title('Model Selection Frequency\n(Selector Network - Hard Selection)')

    # Weight distribution boxplot (Dynamic)
    ax = axes[2]
    bp = ax.boxplot([weights[:, i] for i in range(len(available_models))],
                    labels=available_models, patch_artist=True)
    for patch, color in zip(bp['boxes'], base_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Weight')
    ax.set_title('Weight Distribution per Model\n(Dynamic Weighting Network)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Dynamic Weighting vs Hard Selection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_weights_vs_selection.pdf', dpi=500, bbox_inches='tight')
    plt.savefig(output_dir / 'ensemble_weights_vs_selection.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved to: {output_dir}")


def save_results_excel(results, base_results, weights, selected_models, available_models, output_dir):
    """Save results to Excel."""

    excel_path = output_dir / 'ensemble_results.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Comparison sheet
        comparison_data = []
        for model_name in base_results:
            r = base_results[model_name]
            comparison_data.append({
                'Model': model_name,
                'Type': 'Base',
                'RMSE': r['RMSE'],
                'MAE': r['MAE'],
                'R2': r['R2']
            })
        for ensemble_name in ['Stacking', 'Neural', 'Dynamic', 'Selector']:
            r = results[ensemble_name]
            comparison_data.append({
                'Model': f'Ensemble ({ensemble_name})',
                'Type': 'Ensemble',
                'RMSE': r['RMSE'],
                'MAE': r['MAE'],
                'R2': r['R2'],
                'Training_time_s': r['time']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Comparison', index=False)

        # Dynamic weights statistics
        weights_data = []
        for i, model_name in enumerate(available_models):
            w = weights[:, i]
            weights_data.append({
                'Model': model_name,
                'Mean_weight': w.mean(),
                'Std_weight': w.std(),
                'Min_weight': w.min(),
                'Max_weight': w.max(),
            })

        weights_df = pd.DataFrame(weights_data)
        weights_df.to_excel(writer, sheet_name='Dynamic_Weights', index=False)

        # Selector statistics
        unique, counts = np.unique(selected_models, return_counts=True)
        selector_data = []
        for idx, count in zip(unique, counts):
            model_name = available_models[idx]
            selector_data.append({
                'Model': model_name,
                'Selection_count': count,
                'Selection_percentage': count / len(selected_models) * 100
            })
        selector_df = pd.DataFrame(selector_data)
        selector_df.to_excel(writer, sheet_name='Selector_Choices', index=False)

    print(f"\n  Results saved to: {excel_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    total_start = time.time()

    print("\n" + "="*70)
    print("  ENSEMBLE META-MODEL TRAINING (V2 - Full Epochs)")
    print("  Combining: KF, ANN, CNN, LSTM, CatBoost, TFT")
    print("  Methods: Stacking, Neural, Dynamic Weighting, Model Selector")
    print(f"  Training: {EPOCHS} epochs for all neural models (NO early stopping)")
    print("="*70)

    # Setup paths - ensemble_train.py is now in SOC_prediction/
    script_dir = Path(__file__).parent  # SOC_prediction/

    # Output directory for ensemble models (same as other models)
    output_dir = script_dir / 'nn_training' / 'models'
    output_dir.mkdir(exist_ok=True)

    # Model paths (all in nn_training/models/)
    model_paths = {
        'ANN': {
            'model': script_dir / 'nn_training/models/my_model.keras',
            'scaler': script_dir / 'nn_training/models/scaler.joblib',
        },
        'CNN': {
            'model': script_dir / 'nn_training/models/cnn_model.keras',
            'scaler': script_dir / 'nn_training/models/cnn_scaler.joblib',
        },
        'LSTM': {
            'model': script_dir / 'nn_training/models/lstm_model.keras',
            'scaler': script_dir / 'nn_training/models/lstm_scaler.joblib',
        },
        'CatBoost': {
            'model': script_dir / 'nn_training/models/catboost_model.cbm',
            'scaler': script_dir / 'nn_training/models/catboost_scaler.joblib',
            'features': script_dir / 'nn_training/models/catboost_feature_columns.joblib',
        },
        'TFT': {
            'model': script_dir / 'nn_training/models/tft_model.ckpt',
            'scaler': script_dir / 'nn_training/models/tft_scaler.joblib',
            'config': script_dir / 'nn_training/models/tft_config.joblib',
        },
    }

    # Check which models are available
    print("\n  Checking available base models:")
    for model_name, paths in model_paths.items():
        status = "✓" if paths['model'].exists() else "✗"
        print(f"    {status} {model_name}")

    # Prepare data
    base_predictions, context_features, y_true, available_models, context_cols = \
        prepare_ensemble_dataset(TRAINING_FILES, model_paths)

    # Train ensemble models
    results, base_results, dynamic_model, selector_model, weights, selected_models, available_models = \
        train_ensemble_models(
            base_predictions, context_features, y_true,
            available_models, context_cols, output_dir
        )

    # Create plots
    create_plots(results, base_results, weights, selected_models, available_models, output_dir)

    # Save Excel
    save_results_excel(results, base_results, weights, selected_models, available_models, output_dir)

    # Save config (updated for selector)
    config = {
        'available_models': available_models,
        'context_features': context_cols,
        'n_base_models': len(available_models),
        'n_context_features': len(context_cols),
        'ensemble_types': ['stacking', 'neural', 'dynamic', 'selector'],
        'training_epochs': EPOCHS,
        'early_stopping': False,  # V2: no early stopping
    }
    dump(config, output_dir / 'ensemble_config.joblib')

    # Summary
    total_time = time.time() - total_start

    print("\n" + "="*70)
    print("  ENSEMBLE TRAINING COMPLETE (V2)")
    print("="*70)
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"  Epochs per model: {EPOCHS} (full training, no early stopping)")
    print(f"\n  Best ensemble model: ", end='')

    best_ensemble = min(results, key=lambda x: results[x]['RMSE'])
    best_base = min(base_results, key=lambda x: base_results[x]['RMSE'])

    if results[best_ensemble]['RMSE'] < base_results[best_base]['RMSE']:
        improvement = (base_results[best_base]['RMSE'] - results[best_ensemble]['RMSE']) / base_results[best_base]['RMSE'] * 100
        print(f"Ensemble ({best_ensemble})")
        print(f"  Improvement over best base model ({best_base}): {improvement:.2f}%")
    else:
        print(f"Base model {best_base} is still better than ensembles")
        gap = (results[best_ensemble]['RMSE'] - base_results[best_base]['RMSE']) / base_results[best_base]['RMSE'] * 100
        print(f"  Gap: {gap:.2f}% (best ensemble: {best_ensemble})")

    print(f"\n  Output files:")
    print(f"    - models/ensemble_stacking_model.json")
    print(f"    - models/ensemble_neural_model.keras")
    print(f"    - models/ensemble_dynamic_model.keras")
    print(f"    - models/ensemble_selector_model.keras")
    print(f"    - models/ensemble_results.xlsx")
    print(f"    - models/ensemble_comparison.pdf")
    print(f"    - models/ensemble_weights_vs_selection.pdf")
    print("="*70)