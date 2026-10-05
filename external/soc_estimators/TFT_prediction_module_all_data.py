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
TFT Prediction Module for Battery SOC Analysis
================================================================================
Loads a trained TFT model and makes predictions on new battery data.
================================================================================
"""

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from joblib import load


class TFT_BatteryAnalysis:
    """
    TFT-based battery SOC prediction.
    Loads a trained TFT checkpoint and makes predictions.
    """

    def __init__(self, model_path, scaler_path, config_path=None):
        """
        Initialize the TFT model.

        :param model_path: Path to the TFT checkpoint (.ckpt file)
        :param scaler_path: Path to the fitted scaler (.joblib)
        :param config_path: Path to the config file (.joblib) - optional
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path

        # Load scaler
        self.scaler = load(scaler_path)

        # Load config if provided
        if config_path:
            self.config = load(config_path)
            self.max_encoder_length = self.config.get('max_encoder_length', 30)
            self.features = self.config.get('features', self._default_features())
        else:
            self.max_encoder_length = 30
            self.features = self._default_features()

        # Load model
        self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
        self.model.eval()

    def _default_features(self):
        """Default feature list (same as ANN/CNN/LSTM)."""
        return [
            'Temp', 'Voltage', 'Current_inv', 'SOC_Calc',
            'Delta_Temperature', 'Delta_Current', 'Delta_Voltage',
            'Delta_SoC_Calc', 'Delta_Voltage_Current_Ratio',
            'Current_Voltage_Ratio'
        ]

    def _preprocess_data(self, df):
        """Add delta features (same as other models)."""
        df = df.copy()

        # Add delta features if not present
        if 'Current_Voltage_Ratio' not in df.columns:
            df['Current_Voltage_Ratio'] = (
                df['Current_inv'] / df['Voltage']
            ).fillna(0)

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

    def make_predictions(self, bat_data):
        """
        Make predictions on battery data.

        :param bat_data: DataFrame with battery data (output of Kalman Filter)
        :return: DataFrame with added 'Predicted_SOC' column
        """
        df = self._preprocess_data(bat_data)

        # Add required columns for TFT
        df['group_id'] = '0'
        df['time_idx'] = np.arange(len(df))

        # Ensure SOC_true exists (use SOC_Calc as placeholder if not)
        if 'SOC_true' not in df.columns:
            df['SOC_true'] = df['SOC_Calc']

        # Scale features
        df_scaled = df.copy()
        df_scaled[self.features] = self.scaler.transform(df[self.features])

        # Create TimeSeriesDataSet for prediction
        # We need at least max_encoder_length + 1 samples
        if len(df_scaled) < self.max_encoder_length + 1:
            # Not enough data, pad with NaN predictions
            bat_data['Predicted_SOC'] = np.nan
            return bat_data

        try:
            prediction_dataset = TimeSeriesDataSet(
                df_scaled,
                time_idx='time_idx',
                target='SOC_true',
                group_ids=['group_id'],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=1,
                time_varying_known_reals=['time_idx'],
                time_varying_unknown_reals=self.features + ['SOC_true'],
                target_normalizer=GroupNormalizer(groups=['group_id']),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

            # Create dataloader
            predict_dataloader = prediction_dataset.to_dataloader(
                train=False, batch_size=128, num_workers=0
            )

            # Make predictions
            with torch.no_grad():
                predictions = self.model.predict(
                    predict_dataloader,
                    mode="prediction",
                    return_x=False,
                )

            y_pred = predictions.cpu().numpy().flatten()

            # Predictions start at index max_encoder_length
            # Pad beginning with NaN
            full_predictions = np.full(len(df), np.nan)
            start_idx = self.max_encoder_length
            end_idx = start_idx + len(y_pred)

            if end_idx > len(full_predictions):
                end_idx = len(full_predictions)
                y_pred = y_pred[:end_idx - start_idx]

            full_predictions[start_idx:end_idx] = y_pred

            # Fill initial NaN values with first valid prediction or SOC_Calc
            if not np.isnan(full_predictions[start_idx]):
                full_predictions[:start_idx] = full_predictions[start_idx]
            else:
                full_predictions[:start_idx] = df['SOC_Calc'].iloc[:start_idx].values

            bat_data['Predicted_SOC'] = full_predictions

        except Exception as e:
            print(f"TFT prediction error: {e}")
            # Fallback: use SOC_Calc as prediction
            bat_data['Predicted_SOC'] = bat_data['SOC_Calc'].values

        return bat_data
