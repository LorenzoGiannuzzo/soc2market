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
CNN Prediction Module for Battery SOC
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from joblib import load


class CNN_BatteryAnalysis:
    def __init__(self, file_path_model, file_path_scaler, sequence_length=10):
        self.model = load_model(file_path_model)
        self.scaler = load(file_path_scaler)
        self.sequence_length = sequence_length

    def preprocess_data(self):
        """
        Perform preprocessing on the DataFrame, adding calculated features.
        """
        try:
            # Convert to float
            self.data['Current_inv'] = self.data['Current_inv'].astype(float)
            self.data['Voltage'] = self.data['Voltage'].astype(float)
            self.data['SOC_Calc'] = self.data['SOC_Calc'].astype(float)
            self.data['Temp'] = self.data['Temp'].astype(float)

            # Protected division
            self.data['Current_Voltage_Ratio'] = np.where(
                self.data['Voltage'] != 0,
                self.data['Current_inv'] / self.data['Voltage'],
                0.0
            )

            self.data['Delta_Current'] = self.data['Current_inv'].diff().fillna(0)
            self.data['Delta_Voltage'] = self.data['Voltage'].diff().fillna(0)

            self.data['Delta_Voltage_Current_Ratio'] = np.where(
                self.data['Delta_Current'] != 0,
                self.data['Delta_Voltage'] / self.data['Delta_Current'],
                0.0
            )

            self.data['Delta_SoC_Calc'] = self.data['SOC_Calc'].diff().fillna(0)
            self.data['Delta_Temperature'] = self.data['Temp'].diff().fillna(0)

            self.data.replace([np.inf, -np.inf], 0, inplace=True)
            self.data.fillna(0, inplace=True)

        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            raise

    def create_sequences(self, features):
        """
        Create sequences for CNN input using sliding window.
        """
        X = []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i - self.sequence_length:i])
        return np.array(X)

    def make_predictions(self, data):
        """
        Make predictions on the entire dataset.
        """
        self.data = data
        self.preprocess_data()

        feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                           'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                           'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

        features = self.data[feature_columns]
        features_scaled = self.scaler.transform(features)

        # Lorenzo Giannuzzo: Create sequences
        X_seq = self.create_sequences(features_scaled)

        # Lorenzo Giannuzzo: Predict
        predictions = self.model.predict(X_seq, verbose=0).flatten()

        # Lorenzo Giannuzzo: Pad the beginning with KF predictions (for first sequence_length points)
        padded_predictions = np.concatenate([
            self.data['SOC_Calc'].iloc[:self.sequence_length].values,
            predictions
        ])

        self.data['Predicted_SOC'] = padded_predictions

        return self.data