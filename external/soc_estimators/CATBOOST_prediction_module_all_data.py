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
CatBoost Battery SOC Prediction Module
@authors: Lorenzo Giannuzzo, Marcel Stolte, Elena Rozzi
Contact: lorenzo.giannuzzo@polito.it, marcel.stolte@polito.it, elena.rozzi@polito.it

This module loads a trained CatBoost model and makes SOC predictions
using the same feature engineering pipeline as training.

Feature pipeline (must match train_catboost.py exactly):
  1. Raw features:    Temp, Voltage, Current_inv, SOC_Calc          (4)
  2. Delta features:  Delta_Temperature, Delta_Current, Delta_Voltage,
                      Delta_SoC_Calc, Delta_Voltage_Current_Ratio,
                      Current_Voltage_Ratio                          (6)
  3. Lag features:    raw features × lag steps [1,5,10,30,60]       (20)
  Total: 30 features
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import joblib


class CATBOOST_BatteryAnalysis:
    """CatBoost-based SOC prediction using Kalman Filter output."""

    # Lorenzo Giannuzzo: Feature engineering parameters
    # Lorenzo Giannuzzo: Must match the values used during training (train_catboost.py)
    DEFAULT_RAW_FEATURES = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc']
    DEFAULT_COMPUTED_FEATURES = [
        'Delta_Temperature',
        'Delta_Current',
        'Delta_Voltage',
        'Delta_SoC_Calc',
        'Delta_Voltage_Current_Ratio',
        'Current_Voltage_Ratio',
    ]
    DEFAULT_LAG_STEPS = [1, 5, 10, 30, 60]

    def __init__(self, model_path, scaler_path, feature_columns_path=None,
                 raw_features=None, computed_features=None, lag_steps=None):
        """
        Initialize CatBoost model, scaler, and feature engineering parameters.

        Args:
            model_path: Path to the trained CatBoost model (.cbm)
            scaler_path: Path to the fitted scaler (.joblib)
            feature_columns_path: Path to saved feature columns list (.joblib)
            raw_features: Raw measurement columns
            computed_features: Computed (Delta/Ratio) columns
            lag_steps: Lag timesteps for feature engineering
        """
        # Load model
        self.model = CatBoostRegressor()
        self.model.load_model(model_path)

        # Lorenzo Giannuzzo: Load scaler
        self.scaler = joblib.load(scaler_path)

        # Lorenzo Giannuzzo: Load or set feature columns
        if feature_columns_path is not None:
            try:
                self.feature_columns = joblib.load(feature_columns_path)
            except Exception:
                self.feature_columns = None
        else:
            self.feature_columns = None

        # Lorenzo Giannuzzo: Set feature engineering parameters
        self.raw_features = raw_features or self.DEFAULT_RAW_FEATURES
        self.computed_features = computed_features or self.DEFAULT_COMPUTED_FEATURES
        self.base_features = self.raw_features + self.computed_features
        self.lag_steps = lag_steps or self.DEFAULT_LAG_STEPS

        # Lorenzo Giannuzzo: If feature columns not loaded, generate them
        if self.feature_columns is None:
            self.feature_columns = self._get_feature_columns()

    def _get_feature_columns(self):
        """
        Generate list of all feature column names after engineering.
        Must match get_feature_columns() in train_catboost.py.

        Returns:
            List of feature column names (30 total)
        """
        features = list(self.base_features)  # 10 base features

        # Lorenzo Giannuzzo: Lag features (only on raw measurements, NOT on Delta/Ratio)
        for feature in self.raw_features:
            for lag in self.lag_steps:
                features.append(f'{feature}_lag_{lag}')

        return features

    def _compute_delta_features(self, df):
        """
        Compute the same Delta features used by all models (ANN/CNN/LSTM/CatBoost).
        Must match compute_delta_features() in train_catboost.py.

        Args:
            df: DataFrame with raw columns (Voltage, Current_inv, Temp, SOC_Calc)

        Returns:
            DataFrame with Delta features added
        """
        df = df.copy()

        # Lorenzo Giannuzzo: Delta features (first differences)
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

    def _create_lag_features(self, df):
        """
        Create lag features for raw measurement features only.
        Must match create_lag_features() in train_catboost.py.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with lag features added
        """
        df = df.copy()

        for feature in self.raw_features:
            for lag in self.lag_steps:
                col_name = f'{feature}_lag_{lag}'
                df[col_name] = df[feature].shift(lag)

        return df

    def engineer_features(self, df):
        """
        Apply all feature engineering steps.
        Must match engineer_features() in train_catboost.py.

        Args:
            df: Input DataFrame with raw columns

        Returns:
            DataFrame with all engineered features
        """
        # Lorenzo Giannuzzo: Step 1: Compute Delta features (same as ANN/CNN/LSTM)
        df = self._compute_delta_features(df)

        # Lorenzo Giannuzzo: Step 2: Create lag features for raw measurements only
        df = self._create_lag_features(df)

        return df

    def make_predictions(self, bat_data_df):
        """
        Make SOC predictions using the CatBoost model.

        For the first max_lag timesteps the lag features are unavailable
        (NaN / zero-filled), so CatBoost predictions are unreliable.
        Strategy: use Kalman Filter SOC_Calc for those initial samples,
        then switch to CatBoost from timestep max_lag onward.

        Args:
            bat_data_df: DataFrame with Kalman Filter results
                         (must contain: Voltage, Current_inv, Temp, SOC_Calc)

        Returns:
            DataFrame with Predicted_SOC column added
        """
        original_length = len(bat_data_df)
        max_lag = max(self.lag_steps)

        # Lorenzo Giannuzzo: Engineer features (Delta + Lag)
        df_features = self.engineer_features(bat_data_df.copy())

        # Lorenzo Giannuzzo: Extract features in the correct order
        try:
            X = df_features[self.feature_columns].values.astype(float)
        except KeyError as e:
            available_cols = [c for c in self.feature_columns if c in df_features.columns]
            missing_cols = [c for c in self.feature_columns if c not in df_features.columns]
            if missing_cols:
                print(f"[CatBoost] Warning: Missing columns: {missing_cols}")
            X = df_features[available_cols].values.astype(float)

            # Lorenzo Giannuzzo: Pad missing columns with zeros
            n_missing = len(self.feature_columns) - len(available_cols)
            if n_missing > 0:
                X = np.hstack([X, np.zeros((X.shape[0], n_missing))])

        # Lorenzo Giannuzzo: Handle NaN values
        X = np.nan_to_num(X, nan=0.0)

        # Lorenzo Giannuzzo: Scale features
        X_scaled = self.scaler.transform(X)

        # Lorenzo Giannuzzo: Make predictions
        predictions = self.model.predict(X_scaled)

        # Lorenzo Giannuzzo: Build final prediction array:
        # - First max_lag samples: use Kalman Filter SOC_Calc (lag features incomplete)
        # - From max_lag onward:   use CatBoost predictions
        kf_soc = bat_data_df['SOC_Calc'].values.astype(float)

        final_predictions = np.empty(original_length)
        final_predictions[:max_lag] = kf_soc[:max_lag]

        if len(predictions) < original_length:
            # Lorenzo Giannuzzo: If engineer_features dropped rows, predictions are shorter
            offset = original_length - len(predictions)
            final_predictions[max_lag:] = predictions[max_lag - offset:]
        else:
            final_predictions[max_lag:] = predictions[max_lag:]

        # Lorenzo Giannuzzo: Ensure predictions are in valid SOC range [0, 100]
        final_predictions = np.clip(final_predictions, 0, 100)

        # Lorenzo Giannuzzo: Add predictions to DataFrame
        bat_data_df = bat_data_df.copy()
        bat_data_df['Predicted_SOC'] = final_predictions

        return bat_data_df