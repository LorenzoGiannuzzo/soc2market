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
LSTM Battery SOC Prediction Module

This module loads a trained LSTM model and makes SOC predictions.
Uses the same features as ANN and CNN for fair comparison.
"""

import numpy as np
import pandas as pd
from tensorflow import keras
from joblib import load


class LSTM_BatteryAnalysis:
    """LSTM-based SOC prediction using Kalman Filter output."""

    def __init__(self, model_path, scaler_path, sequence_length=10):
        """
        Initialize the LSTM model and scaler.

        :param model_path: Path to the trained LSTM model (.keras)
        :param scaler_path: Path to the fitted scaler (.joblib)
        :param sequence_length: Number of time steps used during training
        """
        self.model = keras.models.load_model(model_path)
        self.scaler = load(scaler_path)
        self.sequence_length = sequence_length

        # Same features as CNN for fair comparison
        self.feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                                'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                                'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

    def _add_features(self, df):
        """
        Add calculated features to the DataFrame.
        Same preprocessing as CNN for fair comparison.
        """
        df = df.copy()
        
        # Lorenzo Giannuzzo: Calculate derived features
        df['Current_Voltage_Ratio'] = (df['Current_inv'] / df['Voltage']).fillna(0)
        df['Delta_Current'] = df['Current_inv'].diff().fillna(0)
        df['Delta_Voltage'] = df['Voltage'].diff().fillna(0)
        df['Delta_Voltage_Current_Ratio'] = (
            df['Delta_Voltage'] / df['Delta_Current'].replace(0, 1)
        ).fillna(0)
        df['Delta_SoC_Calc'] = df['SOC_Calc'].diff().fillna(0)
        df['Delta_Temperature'] = df['Temp'].diff().fillna(0)
        
        # Lorenzo Giannuzzo: Handle NaN and infinite values
        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        
        return df

    def prepare_sequences(self, features_scaled):
        """
        Create sequences for LSTM input.

        :param features_scaled: Scaled feature array
        :return: 3D array of shape (samples, sequence_length, features)
        """
        X = []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i - self.sequence_length:i])
        return np.array(X)

    def make_predictions(self, bat_data_df):
        """
        Make SOC predictions using the LSTM model.

        :param bat_data_df: DataFrame with Kalman Filter results
        :return: DataFrame with Predicted_SOC column added
        """
        # Lorenzo Giannuzzo: Add calculated features
        df = self._add_features(bat_data_df)
        
        # Lorenzo Giannuzzo: Extract and scale features
        features = df[self.feature_columns].values.astype(float)
        features_scaled = self.scaler.transform(features)

        # Lorenzo Giannuzzo: Create sequences
        X = self.prepare_sequences(features_scaled)

        # Lorenzo Giannuzzo: Make predictions
        predictions = self.model.predict(X, verbose=0).flatten()

        # Lorenzo Giannuzzo: Pad the beginning with the first prediction value
        padded_predictions = np.concatenate([
            np.full(self.sequence_length, predictions[0]),
            predictions
        ])

        # Lorenzo Giannuzzo: Ensure predictions are in valid SOC range [0, 100]
        padded_predictions = np.clip(padded_predictions, 0, 100)

        # Lorenzo Giannuzzo: Add predictions to original DataFrame
        bat_data_df = bat_data_df.copy()
        bat_data_df['Predicted_SOC'] = padded_predictions[:len(bat_data_df)]

        return bat_data_df
