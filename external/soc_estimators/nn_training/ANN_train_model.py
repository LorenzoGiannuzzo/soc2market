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

class BatteryAnalysis:
    def __init__(self, file_paths):
        """
        Initialize the BatteryAnalysis object with a list of file paths.
        :param file_paths: List of Excel file paths to analyze.
        """
        self.file_paths = file_paths # List of file paths
        self.df = None # DataFrame to store the combined data
        self.model = None # Neural network model
        self.scaler = MinMaxScaler() # Scaler for standardizing features

    def load_and_combine_data(self):
        """
        Run Kalman Filter for each file, load the updated data, and combine into a single DataFrame.
        """
        all_dfs = [] # List to store all DataFrames
        for file_path in self.file_paths: # Loop through each file path
            print(f"Running Kalman Filter for: {file_path}") # Print the file path
            subprocess.run(['python', 'run_KalmanFilter_module.py', file_path])  # Assume run_KalmanFilter_module updates updated_bat_data.xlsx

            updated_data_file = 'updated_bat_data.xlsx' # Assume run_KalmanFilter_module updates updated_bat_data.xlsx
            df = pd.read_excel(updated_data_file) # Read the updated data
            all_dfs.append(df) # Append the DataFrame to the list

        self.df = pd.concat(all_dfs, ignore_index=True) # Combine all DataFrames into a single DataFrame

    def preprocess_data(self):
        """
        Perform preprocessing on the DataFrame, adding calculated features.
        """
        try:
            self.df['Current_Voltage_Ratio'] = (self.df['Current_inv'] / self.df['Voltage']).fillna(0) # Calculate the current-voltage ratio
            self.df['Delta_Current'] = self.df['Current_inv'].diff().fillna(0) # Calculate the change in current
            self.df['Delta_Voltage'] = self.df['Voltage'].diff().fillna(0) # Calculate the change in voltage
            self.df['Delta_Voltage_Current_Ratio'] = (self.df['Delta_Voltage'] / self.df['Delta_Current'].replace(0, 1)).fillna(0) # Calculate the change in voltage-current ratio
            self.df['Delta_SoC_Calc'] = self.df['SOC_Calc'].diff().fillna(0) # Calculate the change in SOC_Calc
            self.df['Delta_Temperature'] = self.df['Temp'].diff().fillna(0) # Calculate the change in temperature
            self.df.fillna(0, inplace=True) # Fill NaN values with 0
        except Exception as e: # Catch any exceptions
            print(f"An error occurred during preprocessing: {e}") # Print the error message
            raise

    def prepare_data(self):
        """
        Split the data into training and testing sets and standardize the features.
        """
        features = self.df[['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                            'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                            'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']]
        target = self.df['SOC_true']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42) # Split the data into training and testing sets

        X_train = self.scaler.fit_transform(X_train) # Standardize the training features
        X_test = self.scaler.transform(X_test) # Standardize the testing features

        return X_train, X_test, y_train, y_test

    def build_model(self, input_shape):
        """
        Build and compile the neural network model.
        :param input_shape: Shape of the input features.
        """
        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(18, activation='relu'),
            layers.Dense(1)
        ])

        optimizer = keras.optimizers.Adam(learning_rate=0.001) # Adam optimizer with learning rate of 0.001
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def train_model(self, X_train, y_train, epochs=150, batch_size=50):
        """
        Train the neural network model.
        :param X_train: Training features.
        :param y_train: Training target values.
        :param epochs: Number of epochs for training.
        :param batch_size: Batch size for training.
        """
        if self.model is None: # Lorenzo Giannuzzo:  Check if the model has been built
            raise ValueError("Model has not been built. Call build_model() first.")

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size) # Train the model

    def make_predictions(self):
        """
        Make predictions on the entire dataset and add them to the DataFrame.
        """
        features = self.df[['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                            'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                            'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']]
        features_scaled = self.scaler.transform(features)               # Lorenzo Giannuzzo:  Standardize the features
        self.df['Predicted_SOC'] = self.model.predict(features_scaled)  # Lorenzo Giannuzzo:  Make predictions

    def save_results(self):
        """
        Save the results, including plots and the trained model.
        """
        os.makedirs('models', exist_ok=True)        # Lorenzo Giannuzzo:  Create the Results directory if it does not exist
        os.makedirs('models/plots', exist_ok=True)  # Lorenzo Giannuzzo:  Create the Results directory if it does not exist

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

        os.makedirs('models', exist_ok=True)          # Lorenzo Giannuzzo:  Create the Results directory if it does not exist
        plt.savefig('models/plots/Comparison_plot.png')     # Lorenzo Giannuzzo: Save the plot
        self.model.save('models/my_model.keras')            # Lorenzo Giannuzzo:  Save the trained model
        dump(self.scaler, 'models/scaler.joblib')

        mae = max_absolute_error(self.df['SOC_true'], self.df['Predicted_SOC'])         # Lorenzo Giannuzzo:  Calculate the max absolute error
        rmse = root_mean_squared_error(self.df['SOC_true'], self.df['Predicted_SOC'])   # Lorenzo Giannuzzo:  Calculate the root mean squared error
        print("Max Absolute Error:", mae)
        print("Root Mean Squared Error:", rmse)
