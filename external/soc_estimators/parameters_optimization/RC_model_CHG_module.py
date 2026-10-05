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
This script implements the HPPC_CHG model for analyzing battery charge pulses and rest periods.
It includes methods for extracting segments, calculating parameters, and fitting a 2RC model to the data.

Input: 
- Excel file containing battery charge data with columns for current, voltage, and state of charge (SOC).

Output:
- DataFrames containing extracted segments of charge pulses and rest periods.
- Fitted parameters (R0, R1, R2, C1, C2) for the 2RC model.
- Plots of voltage and current over time, and fitted 2RC model curves.
"""

class HPPC_CHG: # Class for the HPPC_CHG model
    def __init__(self): # Constructor
        file_path = os.path.join('..', 'data', 'EVE_HPPC_1_25degree_CHG-injectionTemplate.xlsx') # Path of the file
        self.data = pd.read_excel(file_path, skiprows=2)  # Load data and Skip the first 3 rows
        self.data.drop(index=0, inplace=True)  # remove the first row

    def extract_charge_pulses(self): # Method to extract charge pulses
        subdatasets = [] # List to store the subdatasets
        in_high_current = False # Boolean variable to check if the current is high
        high_current_start = None # Variable to store the start of the high current
        low_current_count = 0 # Variable to count the low current

        for i in range(len(self.data)): # Loop over the data
            current_value = self.data.iloc[i]["Current_inv(A)"] # Get the current value

            if pd.isna(current_value): # Check if the value is NaN
                continue

            if current_value > 100: # Check if the current is greater than 100
                if not in_high_current: # Check if the current is not high
                    if i >= 3 and all(self.data.iloc[i-3:i, 2] > 1): # Check if the previous 3 values are greater than 1
                        in_high_current = True # Set the current to high
                        high_current_start = max(0, i - 4)  # Include one value before the pulse
                low_current_count = 0 # Reset the low current count
            else:
                if in_high_current: # Check if the current is high
                    low_current_count += 1 # Increment the low current count
                    if low_current_count > 3: # Check if the low current count is greater than 3
                        high_current_end = min(len(self.data), i - 2)  # Include one value after the pulse (3-2=1)
                        subdataset = self.data.iloc[high_current_start:high_current_end] # Get the subdataset
                        if len(subdataset) < 50: # Check if the length of the subdataset is less than 50
                            subdatasets.append(subdataset) # Append the subdataset
                        in_high_current = False # Set the current to low
                        low_current_count = 0 # Reset the low current count

        # Handle case where the high current period goes till the end of the data
        if in_high_current: # Check if the current is high
            high_current_end = len(self.data)  # Include one value after the pulse
            subdataset = self.data.iloc[high_current_start:high_current_end] # Get the subdataset
            if len(subdataset) < 50: # Check if the length of the subdataset is less than 50
                subdatasets.append(subdataset) # Append the subdataset

        return subdatasets # Return the subdatasets

    def extract_charge_rest_periods(self): # Method to extract charge rest periods
        short_rest = [] # List to store the short rest periods
        middle_rest = [] # List to store the middle rest periods
        long_rest = [] # List to store the long rest periods
        in_low_discharge = False # Boolean variable to check if the current is low
        low_discharge_start = None # Variable to store the start of the low discharge

        for i in range(1, len(self.data.iloc[1:, :])): # Loop over the data
            current_value = self.data.iloc[i]["Current_inv(A)"] # Get the current value
            # Checking for low discharge (current close to zero)
            if abs(current_value) <= 1: # Check if the absolute value of the current is less than or equal to 1
                if not in_low_discharge: # Check if the current is not low
                    in_low_discharge = True # Set the current to low
                    low_discharge_start = i-1 # Set the start of the low discharge
            else:
                if in_low_discharge: # Check if the current is low
                    low_discharge_end = i # Set the end of the low discharge
                    subdataset = self.data.iloc[low_discharge_start:low_discharge_end] # Get the subdataset


                    if len(subdataset) < 50: # Check if the length of the subdataset is less than 50
                        if subdataset.iloc[0]["Current_inv(A)"] > 100: # Check if the current is greater than 100
                            short_rest.append(subdataset.iloc[1:, :]) # Append the subdataset
                    elif len(subdataset) <= 1000: # Check if the length of the subdataset is less than or equal to 1000
                        middle_rest.append(subdataset) # Append the subdataset
                    else:
                        long_rest.append(subdataset) # Append the subdataset
                    in_low_discharge = False # Set the current to high

        # Handle case where the low discharge period goes till the end of the data
        if in_low_discharge: # Check if the current is low
            low_discharge_end = len(self.data) # Set the end of the low discharge
            subdataset = self.data.iloc[low_discharge_start:low_discharge_end] # Get the subdataset
            if len(subdataset) < 50: # Check if the length of the subdataset is less than 50
                short_rest.append(subdataset) # Append the subdataset
            elif len(subdataset) <= 1000: # Check if the length of the subdataset is less than or equal to 1000
                middle_rest.append(subdataset) # Append the subdataset
            else:
                long_rest.append(subdataset) # Append the subdataset

        return short_rest, middle_rest, long_rest # Return the short, middle and long rest periods


    def segment_extraction(self):  # Extract segments
        pulse = self.extract_charge_pulses() # Extract charge pulses
        short_rest, middle_rest, long_rest = self.extract_charge_rest_periods() # Extract charge rest periods
        V_model = [] # List to store the voltage model

        for i, _ in enumerate(pulse): # Loop over the charge pulses
            try:
                concatenated_df = pd.concat([pulse[i], short_rest[i]], ignore_index=True) # Concatenate the charge pulse and short rest period
            except IndexError:
                continue
            V_model.append(concatenated_df) # Append the concatenated dataframe to the voltage model


        self.AB = [V_model[i].iloc[0:2, :] for i in range(len(V_model))]
        self.BC = [V_model[i].iloc[1:12, :] for i in range(len(V_model))]
        self.CD = [V_model[i].iloc[11:13, :] for i in range(len(V_model))]
        self.DE = [V_model[i].iloc[13:, :] for i in range(len(V_model))]

        # for i in range(len(V_model)):
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
        #      ax1.legend(loc='upper left')
        #      ax2.legend(loc='upper right')
        #
        #      plt.title(f'Voltage and Current Plot for Dataset {i + 1}')
        #      plt.show()

    def calculate_R0(self): # Calculate R0
        self.R0_ds = [((self.AB[i].iloc[1]["Voltage(V)"] - self.AB[i].iloc[0]["Voltage(V)"]) +  (self.CD[i].iloc[0]["Voltage(V)"] - self.CD[i].iloc[1]["Voltage(V)"])) / (2 * abs(self.AB[i].iloc[1]["Current_inv(A)"])) for i in range(len(self.AB))]
        self.SOC = [self.AB[i].iloc[0]["SOC_true(%)"] for i in range(len(self.AB))]

    # Calculate other parameters
    def two_RC_model(self, t, R1, C1, R2, C2, I, DT, V):  # 2RC model fitting function
        return V[0] - R1 * abs(I) * (1-np.exp(-DT / (R1 * C1))) * (1-np.exp(-t / (R1 * C1))) - R2 * abs(I) * (1-np.exp(-DT / (R2 * C2))) * (1-np.exp(-t / (R2 * C2)))

    def fit_two_RC_model(self):
        R2_values = []  # List of R2 values
        C2_values = []  # List of C2 values
        R1_values = []  # List of R1 values
        C1_values = []  # List of C1 values
        SoC_values = []  # List of SOC values
        for i, soc in enumerate(self.SOC): # Loop over the SOC values
            V = self.DE[i]['Voltage(V)'].values  # Voltage values
            t = np.arange(0, len(self.DE[i]), 1)  # Dummy time
            DT = len(self.BC[i]) # Discharge time (DT) is the length of segment BC (lenght of current pulse)
            I = self.AB[i].iloc[1]["Current_inv(A)"] # Current value

            try:
                # Fitting the 2RC model
                popt, pcov = curve_fit(lambda t, R1, C1, R2, C2: self.two_RC_model(t, R1, C1, R2, C2, I, DT, V), t, V, bounds=([1e-6, 10000, 1e-6, 10000], [1e-3, 1e6, 1e-3, 1e7]) )

                R1, C1, R2, C2 = popt # Get the optimized parameters

                R2_values.append(R2) # Append the R2 value
                R1_values.append(R1) # Append the R1 value
                C2_values.append(C2) # Append the C2 value
                C1_values.append(C1) # Append the C1 value
                SoC_values.append(soc) # Append the SOC value

            except RuntimeError as e:
                 print(f"Fitting failed for SOC: {soc}%: {e}") # Print the error message


        results_dict_two = pd.DataFrame({"SOC": np.arange(5, 96, 10), "R0": self.R0_ds, "R2": R2_values, "C2": C2_values, "R1": R1_values, "C1": C1_values}) # Create a dataframe with the results
        return results_dict_two # Return the results


if __name__ == "__main__": # Main function
    hppc_chg = HPPC_CHG() # Create an object of the HPPC_CHG class
    hppc_chg.segment_extraction() # Extract segments
    hppc_chg.calculate_R0() # Calculate R0
    results = hppc_chg.fit_two_RC_model() # Fit the 2RC model


