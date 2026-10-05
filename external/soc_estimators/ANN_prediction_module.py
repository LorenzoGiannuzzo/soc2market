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
from keras.models import load_model
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from joblib import load

"""

Description:
This class performs predictive analysis of a battery's State of Charge (SoC) using an Artificial Neural Network (ANN).
The class loads a pre-trained neural network model and a scaler for data standardization. 
It preprocesses the input data by computing derived features, such as current-voltage ratios and changes in current, voltage, and temperature.
Finally, it predicts the SoC using the neural network model and appends the prediction to the input data.

Main Features:
1. **Load Pre-trained Model and Scaler**: 
   - Loads a trained ANN model (`.keras`) for SoC prediction.
   - Loads a MinMaxScaler or other scaling object for feature standardization.

2. **Data Preprocessing**:
   - Calculates derived features, including:
     - Current-Voltage Ratio
     - Change in Current (`Delta_Current`)
     - Change in Voltage (`Delta_Voltage`)
     - Change in Temperature (`Delta_Temperature`)
     - Change in SoC (`Delta_SoC_Calc`)
     - Delta Voltage/Current Ratio
   - Handles missing values by replacing NaN values with zeros.

3. **Prediction**:
   - Standardizes input features using the scaler.
   - Predicts the SoC using the ANN model.
   - Ensures the predicted SoC does not exceed 100%.

Input:
- `file_path_model` (str): Path to the pre-trained ANN model file (e.g., `.keras` format).
- `file_path_scaler` (str): Path to the pre-saved scaler file (e.g., `.joblib` format).
- `data` (pd.Series): A Pandas Series containing battery parameters at the current time step.
- `data_back` (pd.Series): A Pandas Series containing battery parameters at the previous time step.

Output:
- Returns the updated `data` (Pandas Series) with an additional column:
  - `'Predicted_SOC'`: The predicted State of Charge.

Usage:
The class can be instantiated with the model and scaler file paths, and predictions can be made using the `make_predictions` method.
"""

class ANN_BatteryAnalysis:
    def __init__(self, file_path_model, file_path_scaler):
        self.model = load_model(file_path_model)    # Lorenzo Giannuzzo:  Load the model from the 'Results' directory
        self.scaler = MinMaxScaler()                # Lorenzo Giannuzzo: Scaler for standardizing features
        self.scaler = load(file_path_scaler)


    def preprocess_data(self):
        """
        Perform preprocessing on the DataFrame, adding calculated features.
        """
        try:
            self.data['Current_Voltage_Ratio'] = (self.data['Current_inv'] / self.data['Voltage'])  # Lorenzo Giannuzzo: Calculate the current-voltage ratio
            self.data['Delta_Current'] = self.data['Current_inv'] - self.data_back['Current_inv']   # Lorenzo Giannuzzo:  Calculate the change in current
            self.data['Delta_Voltage'] = self.data['Voltage'] - self.data_back['Voltage']           # Lorenzo Giannuzzo:  Calculate the change in voltage
            self.data['Delta_Voltage_Current_Ratio'] = (self.data['Delta_Voltage'] / (self.data['Delta_Current'] if self.data['Delta_Current'] != 0 else 1)) # Calculate the change in voltage-current ratio
            self.data['Delta_SoC_Calc'] = self.data['SOC_Calc'] - self.data_back['SOC_Calc']        # Lorenzo Giannuzzo:  Calculate the change in SOC_Calc
            self.data['Delta_Temperature'] = self.data['Temp'] - self.data_back['Temp']             # Lorenzo Giannuzzo:  Calculate the change in temperature

            self.data.fillna(0, inplace=True) # Lorenzo Giannuzzo:  Fill NaN values with 0

        except Exception as e: # Lorenzo Giannuzzo:  Catch any exceptions
            print(f"An error occurred during preprocessing: {e}") # Lorenzo Giannuzzo:  Print the error message


    def make_predictions(self, data, data_back):
        """
        Make predictions on the entire dataset and add them to the DataFrame.
        """
        self.data = data
        self.data_back = data_back
        self.preprocess_data()


        features = self.data[['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                            'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                            'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']]

        features = pd.DataFrame([features])  # Lorenzo Giannuzzo:  Transform the features into a DataFrame

        features_scaled = self.scaler.transform(features)  # Lorenzo Giannuzzo:  Standardize the features

        predicted_SOC= self.model.predict(features_scaled, verbose=0).item()  # Lorenzo Giannuzzo:  Make predictions

        if predicted_SOC > 100:
            predicted_SOC = 100

        self.data['Predicted_SOC'] = predicted_SOC

        return self.data






