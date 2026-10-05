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
# libraries
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""
This script implements the HPPC_DSG model for analyzing battery discharge pulses and rest periods.
It includes methods for extracting segments, calculating parameters, and fitting a 2RC model to the data.

Input: 
- Excel file containing battery discharge data with columns for current, voltage, and state of charge (SOC).

Output:
- DataFrames containing extracted segments of discharge pulses and rest periods.
- Fitted parameters (R0, R1, R2, C1, C2) for the 2RC model.
- Plots of voltage and current over time, and fitted 2RC model curves.
"""

class HPPC_DSG: # Class for the HPPC_DSG model
    # Load the data
    def __init__(self):
        self.file_path = os.path.join('data', 'HPPC_25degree_DSG.xlsx')  # Cambiato da '../data'
        self.data = pd.read_excel(self.file_path, skiprows=2)
        self.data.drop(index=0, inplace=True)

    def extract_discharge_pulses(self): # Extract discharge pulses
        subdatasets = [] # List of subdatasets
        in_high_discharge = False # Flag for high discharge
        high_discharge_start = None # Start of high discharge
        low_discharge_count = 0 # Counter for low discharge

        for i in range(len(self.data)): # Loop over the data
            current_value = self.data.iloc[i]["Current_inv(A)"] # Current value
            if pd.isna(current_value): # Skip if NaN
                continue

            # For discharge, we assume high discharge is when current is significantly negative
            if current_value < -200: # High discharge
                if not in_high_discharge: # If not already in high discharge
                    # Checking previous values for stability before the pulse
                    if i >= 1 and all(self.data.iloc[i-1:i, 2] < -1): # If the previous value is also high discharge
                        in_high_discharge = True # Set flag to true
                        high_discharge_start = max(0, i - 1) # Include one value before the pulse
                low_discharge_count = 0 # Reset low discharge count
            else:
                if in_high_discharge: # If in high discharge
                    low_discharge_count += 1 # Increment low discharge count
                    if low_discharge_count > 1: # If the low discharge count is greater than 1
                        high_discharge_end = min(len(self.data), i) # Include one value after the pulse
                        subdataset = self.data.iloc[high_discharge_start:high_discharge_end] # Extract the subdataset
                        if len(subdataset) < 70: # If the subdataset is less than 70
                            subdatasets.append(subdataset) # Append the subdataset
                        in_high_discharge = False # Reset the flag
                        low_discharge_count = 0 # Reset the low discharge count

        # Handle case where the high discharge period goes till the end of the data
        if in_high_discharge: # If in high discharge
            high_discharge_end = len(self.data) # Include one value after the pulse
            subdataset = self.data.iloc[high_discharge_start:high_discharge_end] # Extract the subdataset

            if len(subdataset) < 70: # If the subdataset is less than 70
                subdatasets.append(subdataset) # Append the subdataset

        return subdatasets # Return the subdatasets

    def extract_discharge_rest_periods(self): # Extract discharge rest periods
        short_rest = [] # Short rest periods
        middle_rest = [] # Middle rest periods
        long_rest = [] # Long rest periods
        in_low_discharge = False # Flag for low discharge
        low_discharge_start = None # Start of low discharge

        for i in range(1, len(self.data.iloc[1:, :])): # Loop over the data
            current_value = self.data.iloc[i]["Current_inv(A)"] # Current value
            # Checking for low discharge (current close to zero)
            if abs(current_value) <= 1: # Low discharge
                if not in_low_discharge: # If not already in low discharge
                    in_low_discharge = True # Set flag to true
                    low_discharge_start = i-1 # Start of low discharge
            else:
                if in_low_discharge: # If in low discharge
                    low_discharge_end = i # End of low discharge
                    subdataset = self.data.iloc[low_discharge_start:low_discharge_end] # Extract the subdataset

                    if len(subdataset) < 200: # If the subdataset is less than 200
                        if subdataset.iloc[0]["Current_inv(A)"] < -200: # If the current is less than -200
                            short_rest.append(subdataset.iloc[1:, :]) # Append the subdataset
                    elif len(subdataset) <= 1000: # If the subdataset is less than or equal to 1000
                        middle_rest.append(subdataset) # Append the subdataset
                    else:
                        long_rest.append(subdataset) # Append the subdataset
                    in_low_discharge = False # Reset the flag

        # Handle case where the low discharge period goes till the end of the data
        if in_low_discharge:
            low_discharge_end = len(self.data) # End of low discharge
            subdataset = self.data.iloc[low_discharge_start:low_discharge_end] # Extract the subdataset
            if len(subdataset) < 200: # If the subdataset is less than 200
                short_rest.append(subdataset) # Append the subdataset
            elif len(subdataset) <= 1000: # If the subdataset is less than or equal to 1000
                middle_rest.append(subdataset) # Append the subdataset
            else:
                long_rest.append(subdataset) # Append the subdataset

        return short_rest, middle_rest, long_rest # Return the rest periods

    def segment_extraction(self): # Extract segments
        pulse = self.extract_discharge_pulses() # Extract discharge pulses
        short_rest, middle_rest, long_rest = self.extract_discharge_rest_periods() # Extract discharge rest periods
        V_model = [] # List of voltage models

        for i, _ in enumerate(pulse): # Loop over the pulses
            try:
                concatenated_df = pd.concat([pulse[i], short_rest[i]], ignore_index=True) # Concatenate the pulse and short rest
            except IndexError:
                continue
            V_model.append(concatenated_df) # Append the concatenated dataframe

        # Plot the voltage for each dataset
            # plt.show()

        self.AB = [V_model[i].iloc[1:3, :] for i in range(len(V_model))]  # Extract AB (see slides)
        self.BC = [V_model[i].iloc[2:52, :] for i in range(len(V_model))] # Extract BC (see slides)
        self.CD = [V_model[i].iloc[51:53, :] for i in range(len(V_model))] # Extract CD (see slides)
        self.DE = [V_model[i].iloc[53:, :] for i in range(len(V_model))] # Extract DE (see slides)

        # for i in range(len(V_model)): # Loop over the datasets for plotting secments
        #      fig, ax1 = plt.subplots()  # Create figure and first y-axis (ax1)
        #
        #      # Plot voltage on the primary y-axis
        #      ax1.plot(self.AB[i]['Step(s)'], self.AB[i]['Voltage(V)'], label='AB Voltage')
        #      ax1.plot(self.BC[i]['Step(s)'], self.BC[i]['Voltage(V)'], label='BC Voltage')
        #      ax1.plot(self.CD[i]['Step(s)'], self.CD[i]['Voltage(V)'], label='CD Voltage')
        #      ax1.plot(self.DE[i]['Step(s)'], self.DE[i]['Voltage(V)'], label='DE Voltage')
        #      ax1.set_xlabel('Step(s)')
        #      ax1.set_ylabel('Voltage (V)', color='blue')
        #      ax1.tick_params(axis='y', labelcolor='blue')
        #
        #      # Create a secondary y-axis for current
        #      ax2 = ax1.twinx()  # Instantiate a second y-axis that shares the same x-axis
        #      ax2.plot(V_model[i]['Step(s)'], V_model[i]['Current_inv(A)'], color='orange', label='Current')
        #      ax2.set_ylabel('Current (A)', color='orange')
        #      ax2.tick_params(axis='y', labelcolor='orange')
        #
        #      # Add legends for clarity
        #      ax1.legend(loc='lower right')
        #      ax2.legend(loc='upper right')
        #
        #      plt.title(f'Voltage and Current Plot for Dataset {i + 1}')
             # plt.show()

    def calculate_R0(self): # Calculate R0
        self.R0_ds = [((self.AB[i].iloc[0]["Voltage(V)"] - self.AB[i].iloc[1]["Voltage(V)"]) +  (self.CD[i].iloc[1]["Voltage(V)"] - self.CD[i].iloc[0]["Voltage(V)"])) / (2 * abs(self.AB[i].iloc[0]["Current_inv(A)"])) for i in range(len(self.AB))] # Calculate R0
        self.SOC = [self.AB[i].iloc[0]["SOC_true(%)"] for i in range(len(self.AB))] # Extract SOC

    # Calculate other parameters
    def two_RC_model(self, t, R1, C1, R2, C2, I, DT, V):   # fitting function
        return V[-1] - R1 * abs(I) * (1 - np.exp(-DT / (R1 * C1))) * np.exp(-t / (R1 * C1) - R2 * abs(I) * (1 - np.exp(-DT / (R2 * C2)))) * np.exp(-t / (R2 * C2))

    def fit_two_RC_model(self):
        R2_values = [] # List of R2 values
        C2_values = [] # List of C2 values
        R1_values = [] # List of R1 values
        C1_values = [] # List of C1 values
        SoC_values = [] # List of SOC values
        for i, soc in enumerate(self.SOC): # Loop over the SOC values

            V = self.DE[i]['Voltage(V)'].values  # Voltage in segment DE to be fitted for R1, R2, C1, C2 parameters
            t = np.arange(0, len(self.DE[i]), 1)  # Dummy time
            DT = len(self.BC[i]) # Discharge time (DT) is the length of segment BC (lenght of current pulse)
            I = self.AB[i].iloc[1]["Current_inv(A)"] # Current value

            try:
                # Fitting the 2RC model
                popt, pcov = curve_fit(lambda t, R1, C1, R2, C2: self.two_RC_model(t, R1, C1, R2, C2, I, DT, V), t, V, bounds=([1e-6, 10000, 1e-6, 10000], [1e-3, 1e6, 1e-3, 1e7]) )

                R1, C1, R2, C2 = popt # Extract the parameters

                R2_values.append(R2)  # Append the R2 value
                R1_values.append(R1) # Append the R1 value
                C2_values.append(C2) # Append the C2 value
                C1_values.append(C1) # Append the C1 value
                SoC_values.append(soc) # Append the SOC value


            except RuntimeError as e:
                 print(f"Fitting failed for SOC: {soc}%: {e}") # Print error message

        results_dict_two = pd.DataFrame({"SOC": np.arange(2.5, 98, 5), "R0": self.R0_ds, "R2": R2_values, "C2": C2_values, "R1": R1_values, "C1": C1_values}) # Create a dataframe with the results
        return results_dict_two


if __name__ == "__main__":
    hppc = HPPC_DSG()
    hppc.segment_extraction()
    hppc.calculate_R0()
    results = hppc.fit_two_RC_model()