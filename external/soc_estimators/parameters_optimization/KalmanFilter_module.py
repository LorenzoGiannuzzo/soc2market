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

import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

from RC_model_CHG_module import HPPC_CHG
from RC_model_DCH_module import HPPC_DSG

"""
This module contains the `Data` class and the `KalmanFilter` class.

The `Data` class is responsible for importing and processing battery data. It provides methods to import battery data from an Excel file, extract current, voltage, and SOC measurements, import circuit parameters from external modules, and import OCV data from an Excel file.

The `KalmanFilter` class inherits from the `Data` class and implements the Kalman Filter algorithm to estimate the State of Charge (SOC) of a battery. It uses the imported data and circuit parameters to perform the estimation.

Input:
- Excel file containing battery data.

Output:
- Updated battery data with estimated SOC and error metrics.
"""

class Data:
    # def __init__(self):

    def import_bat_data(self, file_name):
        file_path = os.path.join('..', 'data', file_name)  # Path to the data
        bat_data = pd.read_excel(file_path, skiprows=2).iloc[1:]  # Read the data
        return bat_data

    def extract_current_voltage(self, bat_data):
        I = bat_data['Current_inv']  # Extract the current
        Vmeas = bat_data['Voltage']  # Extract the voltage
        SOC_meas = bat_data['SOC_true']  # Extract the SOC
        return I, Vmeas, SOC_meas

    def import_circuit_params(self):
        """Import the circuit parameters from RC_model_CHG_module and RC_model_DCH_module."""
        hppc_chg = HPPC_CHG()  # Create an object of the HPPC_CHG class
        hppc_chg.segment_extraction()  # Extract segments
        hppc_chg.calculate_R0()  # Calculate R0
        df_charge = hppc_chg.fit_two_RC_model()  # Fit the 2RC model

        hppc = HPPC_DSG()  # Create an object of the HPPC_DSG class
        hppc.segment_extraction()  # Extract segments
        hppc.calculate_R0()  # Calculate R0
        df_discharge = hppc.fit_two_RC_model()  # Fit the 2RC model
        return df_charge, df_discharge

    def import_ocv(self):
        file_path_ocv = '../../data/Cha_Dis_OCV_SOC_Data.xlsx'  # Path to the OCV data
        df = pd.read_excel(file_path_ocv, skiprows=1)  # Read the OCV data
        ocv_charge = df.loc[:, ["data_SOC", "data_U"]][::-1]  # Extract the charge OCV data
        ocv_discharge = df.loc[:, ["data_SOC.1", "data_U.1"]][::-1]  # Extract the discharge OCV data
        return ocv_charge, ocv_discharge


class KalmanFilter(Data):
    def __init__(self, file_name, dt=1, initial_SOC=60, capacity=280, opt=False):
        self.dt = dt  # Time step
        self.opt = opt

        self.get_imports(file_name)  # Import the data
        self.initial_SOC = initial_SOC  # Initial SOC
        # self.Q = 0.0001  # Covariance Process Noise
        self.C = capacity  # Battery capacity Ah
        # self.R = 0.0001  # Measurement Noise
        self.soc_array_ch = self.OCV_gradient()[0] # Get the SOC array for charging
        self.dOCV_dSOC_array_ch = self.OCV_gradient()[1] # Get the OCV gradient for charging
        self.soc_array_dh = self.OCV_gradient()[2] # Get the SOC array for discharging
        self.dOCV_dSOC_array_dh = self.OCV_gradient()[3] # Get the OCV gradient for discharging

    def OCV_gradient(self):
        soc_array_ch = self.OCV_charge['data_SOC'] # Extract the SOC array for charging
        voc_array_ch = self.OCV_charge['data_U'] # Extract the OCV array for charging
        dOCV_dSOC_array_ch = np.gradient(voc_array_ch, soc_array_ch) # Calculate the OCV gradient for charging
        soc_array_dh = self.OCV_discharge['data_SOC.1'] # Extract the SOC array for discharging
        voc_array_dh = self.OCV_discharge['data_U.1'] # Extract the OCV array for discharging
        dOCV_dSOC_array_dh = np.gradient(voc_array_dh, soc_array_dh) # Calculate the OCV gradient for discharging

        return soc_array_ch, dOCV_dSOC_array_ch, soc_array_dh, dOCV_dSOC_array_dh

    def get_imports(self, file_name): # Import the
        self.bat_data = self.import_bat_data(file_name)  # Import the battery data
        """
        # Optimization flag (default is False). 
        With opt=True, only the first 100 data points are considered to estimate the initial SOC. 
        This reduces the error of the initial SOC estimation down to 5%, improving the overall performance of the Kalman filter.
        Muted as partially in contrast with the challenge requirements, which ask for considering only one sample at a time
        """
        self.N = 100 if self.opt else self.bat_data.shape[0] # Number of data points
        self.bat_data = self.bat_data.iloc[range(self.N),:].copy() # Consider only the first N data points
        self.I_all, self.Vmeas_all, self.SOC_all = self.extract_current_voltage(self.bat_data) # Extract current and voltage
        self.df_charge, self.df_discharge = self.import_circuit_params() # Import circuit parameters
        self.OCV_charge, self.OCV_discharge = self.import_ocv() # Import OCV data


    def get_ocv(self): # Get the OCV
        if self.I > 0: # Charging
            voc = np.interp(self.SOC_k, self.OCV_charge['data_SOC'], self.OCV_charge['data_U']) # Interpolate the OCV
        else: # Discharging
            voc = np.interp(self.SOC_k, self.OCV_discharge['data_SOC.1'], self.OCV_discharge['data_U.1']) # Interpolate the OCV
        return voc

    def RC_parameters(self):
        if self.I > 0: # Charging
            R0 = np.interp(self.SOC_k, self.df_charge['SOC'], self.df_charge['R0']) # Interpolate R0
            R1 = np.interp(self.SOC_k, self.df_charge['SOC'], self.df_charge['R1'])  # Interpolate R1
            R2 = np.interp(self.SOC_k, self.df_charge['SOC'], self.df_charge['R2'])  # Interpolate R2
            C1 = np.interp(self.SOC_k, self.df_charge['SOC'], self.df_charge['C1'])  # Interpolate C1
            C2 = np.interp(self.SOC_k, self.df_charge['SOC'], self.df_charge['C2'])  # Interpolate C2

        else: # Discharging
            R0 = np.interp(self.SOC_k, self.df_discharge['SOC'], self.df_discharge['R0']) # Interpolate R0
            R1 = np.interp(self.SOC_k, self.df_discharge['SOC'], self.df_discharge['R1'])  # Interpolate R1
            R2 = np.interp(self.SOC_k, self.df_discharge['SOC'], self.df_discharge['R2'])  # Interpolate R2
            C1 = np.interp(self.SOC_k, self.df_discharge['SOC'], self.df_discharge['C1'])  # Interpolate C1
            C2 = np.interp(self.SOC_k, self.df_discharge['SOC'], self.df_discharge['C2'])  # Interpolate C2

        return R0, R1, R2, C1, C2

    def terminal_voltage_estimation(self, voc, R0, R1, R2, C1, C2): # Estimate the terminal voltage
        # R0, R1, R2, C1, C2 = self.RC_parameters() # Get the RC parameters
        V1 = self.V1_back*np.exp(-1/(R1*C1)) + R1*self.I*(1-np.exp(-1/(R1*C1))) # Calculate V1
        V2 = self.V2_back*np.exp(-1/(R2*C2)) + R2*self.I*(1-np.exp(-1/(R2*C2))) # Calculate V2

        v_terminal = voc - self.I * R0 - V1 -  V2  #Calculate the terminal voltage

        return v_terminal # Return the terminal voltage

    def linearize_system(self):
        self.I = self.I_all.iloc[self.index] # Initial current
        self.I = gaussian_filter1d([self.I], 4.140949830648336)[0] # Apply Gaussian filter
        V_meas_k = self.Vmeas_all.iloc[self.index] # Initial voltage

        I = np.array([[self.I]])  # Ensure I is a 2D column array

        R0, R1, R2, C1, C2 = self.RC_parameters() # Get the RC parameters

        # State vector
        X_back = self.X_back # Initial state
        P_back =  self.P_back # Initial covariance

        # State transition matrix
        AA_k = np.array([
                        [1, 0, 0],
                        [0, np.exp(-self.dt / (R1 * C1)), 0],
                        [0, 0, np.exp(-self.dt / (R2 * C2))]
                        ])

        # Input matrix
        BB_k = np.array([[1 * self.dt / (3600 * self.C) * 100],
                        [R1 * (1- np.exp(-self.dt / (R1 * C1)))],
                        [R2 * (1 - np.exp(-self.dt / (R2 * C2)))],
                        ])

        X_pred = AA_k @ X_back + BB_k * I  # Predicted state
        P_pred = AA_k @ P_back @ AA_k.T + self.Q  # Predicted covariance


        # compute the gradient of the OCV wrt SOC at SOC_k
        if self.I > 0:
            dV_ocv_dSOC = np.interp(self.SOC_k, self.soc_array_ch, self.dOCV_dSOC_array_ch)

        else:
            dV_ocv_dSOC = np.interp(self.SOC_k, self.soc_array_dh, self.dOCV_dSOC_array_dh)


        V_ocv_k = self.get_ocv() # Get the OCV
        CC_k = np.array([[dV_ocv_dSOC, -1, -1]]) # Observation matrix
        K_pred = P_pred @ CC_k.T @ np.linalg.inv(CC_k @ P_pred @ CC_k.T + self.R)  # Kalman gain

        V_term_k = self.terminal_voltage_estimation(V_ocv_k, R0, R1, R2, C1, C2)  # Estimate the terminal voltage

        X_pred_post = X_pred + K_pred @  np.array([[V_meas_k - V_term_k]])   # Update the state
        I_mat = np.eye(3)  # Identity matrix
        P_pred_post = (I_mat - K_pred @ CC_k) @ P_pred # Update the covariance

        v_error = abs(V_meas_k - V_term_k) # Compute the voltage error

        return X_pred_post, P_pred_post, v_error

    def kalman_filter(self, Q=np.array([[0.1, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]), R=0.001, initial_covariance=np.array([[0.1, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])):  # Perform a Kalman filter step  def run_kalman_filter(self, initial_soc=0.5, initial_covariance=None):
        """
        Execute the Kalman filter on all data.

        Parameters:
        - initial_soc: initial value of the SOC (default 60)
        - initial_covariance: initial covariance matrix (optional)

        Returns:
        - bat_data updated with the SOC estimated by the KF
        """

        # Variable initialization
        SOC_est_cor_array = []  # list to store the SOC estimates
        error_array = []  # list to store the errors
        self.SOC_k = self.initial_SOC  # Initial SOC (input)
        self.P_back = initial_covariance  # Initial covariance
        self.V1_back = 0  # Initialize V1_back
        self.V2_back = 0   # Initialize V2_back
        self.X_back = np.array([[self.SOC_k], [self.V1_back], [self.V2_back]]) # Initial state
        self.Q = Q  # Process noise
        self.R= R  # Measurement noise

        self.V_error = 0 # Initialize the voltage error

        # Iterate over all data points
        for i in range(self.N):

            self.index = i  # update the index

            X_pred_post, P_post, v_error = self.linearize_system() # Call the linearize_system function

            # Extract the SOC estimate and error
            self.SOC_k = X_pred_post[0, 0]  # new SOC estimate
            self.V1_back = X_pred_post[1, 0]  # update V1_back
            self.V2_back = X_pred_post[2, 0]  # update V2_back

            SOC_est_cor_array.append(self.SOC_k)  # add the SOC estimate to the list
            SOC_meas_k = self.SOC_all.iloc[self.index]  # Real SOC
            error = (self.SOC_k - SOC_meas_k) / SOC_meas_k * 100   # compute the error
            error_array.append(error)  # add the error to the list

            self.V_error += v_error
            # update the state and covariance
            self.P_back = P_post # update the covariance
            self.X_back = X_pred_post # update the state

        self.V_error = self.V_error/self.N # Compute the average voltage error
        # add the SOC estimate and error to the battery data
        SOC_est_cor_array = np.array(SOC_est_cor_array) # Convert the SOC estimate to an array
        updated_bat_data = self.bat_data.copy() # Copy the battery data
        updated_bat_data['SOC_Calc'] = SOC_est_cor_array # Add the SOC estimate to the battery data
        updated_bat_data['Error'] = error_array # Add the error to the battery data

        return updated_bat_data