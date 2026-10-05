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
from keras.models import load_model
from joblib import load

"""
This script trains an Artificial Neural Network (ANN) model on battery data.

The script reads battery data from multiple Excel files, runs the Kalman Filter on each file to estimate the State of Charge (SOC), and combines the updated data. 
It then preprocesses the data, builds and trains an ANN model to predict SOC, and saves the results, including plots and the trained model.

Input:
- List of paths to Excel files containing battery data.

Output:
- Combined and preprocessed battery data.
- Trained ANN model.
- Plots comparing true SOC, EKF SOC, and ANN-predicted SOC.
- RMSE and MAE error metrics.
"""

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))

class ANN_BatteryAnalysis:
    def __init__(self, file_path_model, file_path_scaler):
        self.model = load_model(file_path_model)    # Lorenzo Giannuzzo:  Load the model from the 'Results' directory
        self.scaler = MinMaxScaler()                # Lorenzo Giannuzzo:  Scaler for standardizing features
        self.scaler = load(file_path_scaler)        # Lorenzo Giannuzzo:  Carica lo scaler precedentemente addestrato e salvato

    def preprocess_data(self):
        """
        Perform preprocessing on the DataFrame, adding calculated features.
        """
        try:
            # Lorenzo Giannuzzo:  Convert to float to avoid object dtype issues
            self.data['Current_inv'] = self.data['Current_inv'].astype(float)
            self.data['Voltage'] = self.data['Voltage'].astype(float)
            self.data['SOC_Calc'] = self.data['SOC_Calc'].astype(float)
            self.data['Temp'] = self.data['Temp'].astype(float)

            # Lorenzo Giannuzzo:  Protected division - avoid division by zero
            self.data['Current_Voltage_Ratio'] = np.where(
                self.data['Voltage'] != 0,
                self.data['Current_inv'] / self.data['Voltage'],
                0.0
            )

            self.data['Delta_Current'] = self.data['Current_inv'].diff().fillna(0)
            self.data['Delta_Voltage'] = self.data['Voltage'].diff().fillna(0)

            # Lorenzo Giannuzzo:  Protected division - avoid division by zero
            self.data['Delta_Voltage_Current_Ratio'] = np.where(
                self.data['Delta_Current'] != 0,
                self.data['Delta_Voltage'] / self.data['Delta_Current'],
                0.0
            )

            self.data['Delta_SoC_Calc'] = self.data['SOC_Calc'].diff().fillna(0)
            self.data['Delta_Temperature'] = self.data['Temp'].diff().fillna(0)

            # Lorenzo Giannuzzo:  Replace any infinite values with 0
            self.data.replace([np.inf, -np.inf], 0, inplace=True)
            self.data.fillna(0, inplace=True)

        except Exception as e:
            print(f"An error occurred during preprocessing: {e}")
            raise

    def make_predictions(self, data):
        """
        Make predictions on the entire dataset and add them to the DataFrame.
        """
        self.data = data
        self.preprocess_data()

        features = self.data[['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                              'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                              'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']]

        features_scaled = self.scaler.transform(features)   # Lorenzo Giannuzzo:  Standardize the features

        predicted_SOC = self.model.predict(features_scaled) # Lorenzo Giannuzzo:  Make predictions

        self.data['Predicted_SOC'] = predicted_SOC

        return self.data

    def save_results(self):
        """
        Save the results, including plots and the trained model.
        """
        os.makedirs('Results', exist_ok=True) # Lorenzo Giannuzzo:  Create the Results directory if it does not exist

        plt.figure(figsize=(14, 7))
        plt.plot(np.arange(0, len(self.df['SOC_Calc']), 1), self.df['SOC_true'], label='True SOC', color='blue', alpha=0.6)
        plt.plot(np.arange(0, len(self.df['SOC_Calc']), 1), self.df['Predicted_SOC'], label='EKF + ANN', color='orange', alpha=0.6)
        plt.plot(np.arange(0, len(self.df['SOC_Calc']), 1), self.df['SOC_Calc'], label='EFK', color='green', alpha=0.6)
        plt.title('State of Charge (SoC) Comparison')
        plt.xlabel('Index')
        plt.ylabel('State of Charge')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        os.makedirs('Results', exist_ok=True)     # Lorenzo Giannuzzo:  Create the Results directory if it does not exist
        plt.savefig('Results/Comparison_plot.png')      # Lorenzo Giannuzzo:  Save the plot
        self.model.save('Results/my_model.keras')       # Lorenzo Giannuzzo:  Save the trained model
        dump(self.scaler, 'Results/scaler.joblib')

        mae = max_absolute_error(self.df['SOC_true'], self.df['Predicted_SOC'])         # Lorenzo Giannuzzo:  Calculate the max absolute error
        rmse = root_mean_squared_error(self.df['SOC_true'], self.df['Predicted_SOC'])   # Lorenzo Giannuzzo:  Calculate the root mean squared error
        print("Max Absolute Error:", mae)
        print("Root Mean Squared Error:", rmse)
