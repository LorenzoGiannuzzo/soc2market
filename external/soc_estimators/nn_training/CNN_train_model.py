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

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from joblib import dump

"""
This script trains a Convolutional Neural Network (CNN) model on battery data.
Uses sliding window approach to create sequences for CNN input.
"""


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))


class BatteryAnalysisCNN:
    def __init__(self, file_paths, sequence_length=10):
        """
        Initialize the BatteryAnalysisCNN object.
        :param file_paths: List of Excel file paths to analyze.
        :param sequence_length: Number of time steps to use for each CNN input sequence.
        """
        self.file_paths = file_paths
        self.sequence_length = sequence_length
        self.df = None
        self.model = None
        self.scaler = MinMaxScaler()

    def load_and_combine_data(self):
        """
        Run Kalman Filter for each file, load the updated data, and combine into a single DataFrame.
        """
        all_dfs = []
        for file_path in self.file_paths:
            print(f"Running Kalman Filter for: {file_path}")
            subprocess.run(['python', 'run_KalmanFilter_module.py', file_path])

            updated_data_file = 'updated_bat_data.xlsx'
            df = pd.read_excel(updated_data_file)
            all_dfs.append(df)

        self.df = pd.concat(all_dfs, ignore_index=True)

    def preprocess_data(self):
        """
        Perform preprocessing on the DataFrame, adding calculated features.
        """
        try:
            self.df['Current_Voltage_Ratio'] = (self.df['Current_inv'] / self.df['Voltage']).fillna(0)
            self.df['Delta_Current'] = self.df['Current_inv'].diff().fillna(0)
            self.df['Delta_Voltage'] = self.df['Voltage'].diff().fillna(0)
            self.df['Delta_Voltage_Current_Ratio'] = (
                        self.df['Delta_Voltage'] / self.df['Delta_Current'].replace(0, 1)).fillna(0)
            self.df['Delta_SoC_Calc'] = self.df['SOC_Calc'].diff().fillna(0)
            self.df['Delta_Temperature'] = self.df['Temp'].diff().fillna(0)
            self.df.fillna(0, inplace=True)

            # Handle infinite values
            self.df.replace([np.inf, -np.inf], 0, inplace=True)
        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            raise

    def create_sequences(self, features, target):
        """
        Create sequences for CNN input using sliding window.
        :param features: Feature array (scaled)
        :param target: Target array
        :return: X sequences, y targets
        """
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i - self.sequence_length:i])
            y.append(target.iloc[i])
        return np.array(X), np.array(y)

    def prepare_data(self, test_split=False):
        """
        Prepare data for training. Optionally split into train/test.
        """
        feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                           'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                           'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

        features = self.df[feature_columns]
        target = self.df['SOC_true']

        # Lorenzo Giannuzzo: Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Lorenzo Giannuzzo: Create sequences
        X, y = self.create_sequences(features_scaled, target)

        if test_split:
            # Lorenzo Giannuzzo: Split data for testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            return X_train, X_test, y_train, y_test
        else:
            # Lorenzo Giannuzzo: Use all data for training
            return X, y

    def build_model(self, input_shape):
        self.model = keras.Sequential([
            layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.Conv1D(18, kernel_size=3, activation='relu', padding='same'),
            layers.GlobalAveragePooling1D(),  # Riduce a 1D
            layers.Dense(1)
        ])

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    def train_model(self, X_train, y_train, epochs=150, batch_size=64, use_early_stopping=False):
        """
        Train the CNN model.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        if use_early_stopping:
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            validation_split = 0.2
        else:
            callbacks = []
            validation_split = 0.0  # No validation, train on all data

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        return history

    def make_predictions(self):
        """
        Make predictions on the entire dataset.
        """
        feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                           'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                           'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

        features = self.df[feature_columns]
        features_scaled = self.scaler.transform(features)

        # Lorenzo Giannuzzo: Create sequences for all data
        X_all, _ = self.create_sequences(features_scaled, self.df['SOC_true'])

        # Lorenzo Giannuzzo: Predict
        predictions = self.model.predict(X_all).flatten()

        # Lorenzo Giannuzzo: Pad the beginning with NaN (no prediction for first sequence_length points)
        padded_predictions = np.concatenate([np.full(self.sequence_length, np.nan), predictions])

        self.df['Predicted_SOC'] = padded_predictions

    def save_results(self):
        """
        Save the results, including plots and the trained model.
        """
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/plots', exist_ok=True)

        # Plot comparison
        plt.figure(figsize=(14, 7))
        plt.plot(np.arange(0, len(self.df['SOC_Calc']), 1), self.df['SOC_true'], label='True SOC', color='blue',
                 alpha=0.6)
        plt.plot(np.arange(0, len(self.df['SOC_Calc']), 1), self.df['Predicted_SOC'], label='KF + CNN', color='orange',
                 alpha=0.6)
        plt.plot(np.arange(0, len(self.df['SOC_Calc']), 1), self.df['SOC_Calc'], label='KF', color='green', alpha=0.6)
        plt.title('State of Charge (SoC) Comparison - CNN Model')
        plt.xlabel('Index')
        plt.ylabel('State of Charge')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig('models/plots/CNN_Comparison_plot.png')
        plt.close()

        # Lorenzo Giannuzzo: Save model and scaler
        self.model.save('models/cnn_model.keras')
        dump(self.scaler, 'models/cnn_scaler.joblib')

        # Lorenzo Giannuzzo: Save sequence length for inference
        with open('models/cnn_config.txt', 'w') as f:
            f.write(f"sequence_length={self.sequence_length}")

        # Lorenzo Giannuzzo: Calculate metrics (excluding NaN values)
        valid_idx = ~np.isnan(self.df['Predicted_SOC'])
        mae = max_absolute_error(self.df.loc[valid_idx, 'SOC_true'], self.df.loc[valid_idx, 'Predicted_SOC'])
        rmse = root_mean_squared_error(self.df.loc[valid_idx, 'SOC_true'], self.df.loc[valid_idx, 'Predicted_SOC'])

        print(f"\nCNN Model Results:")
        print(f"  Max Absolute Error: {mae:.4f}")
        print(f"  Root Mean Squared Error: {rmse:.4f}")