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
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from KalmanFilter_module import KalmanFilter
import argparse
"""
This script runs the Kalman Filter algorithm on battery data.

The script reads battery data from an Excel file, initializes the Kalman Filter with the data, and runs the filter to estimate the State of Charge (SOC). The results, including computed and real SOC, are plotted and saved. The script also calculates and prints error metrics such as RMSE and MAE, and writes the updated battery data to an Excel file.

Input:
- Path to the Excel file containing battery data.

Output:
- Plots of computed and real SOC.
- RMSE and MAE error metrics.
- Excel file with updated battery data including estimated SOC.
"""

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))


def run_KF(file_path, optimization=False):
    if optimization:
        pass
    else:

        def import_first_soc(file_name):
            file_path = os.path.join('..', 'train_data', file_name)  # Cambiato da 'data' a 'train_data'
            bat_data = pd.read_excel(file_path, skiprows=2).iloc[1:]
            return bat_data.loc[1, 'SOC_true']

        first_soc = import_first_soc(file_name=file_path)

        # Updated: use dynamic capacity
        updated_bat_data = KalmanFilter(
            file_path,
            dt=1,
            initial_SOC=first_soc,
            capacity_charge=280,
            capacity_discharge=280
        ).kalman_filter(
            Q=np.array([[4.034329e-7, 0, 0], [0, 8.628196e-5, 0], [0, 0, 9.095089e-05]]),
            R=3.743003e-03,
            initial_covariance=np.array([[3.393996e-03, 0, 0], [0, 6.662388e-04, 0], [0, 0, 6.946038e-03]])
        )
    # Plot the computed and real SOC
    f1 = plt.figure()
    plt.plot(updated_bat_data['Step'].astype(float), updated_bat_data['SOC_Calc'], label="EKF")
    plt.plot(updated_bat_data['Step'].astype(float), updated_bat_data['SOC_true'], label="Real SOC")

    plt.title('Computed and Real SOC')
    plt.xlabel('Step')
    plt.ylabel('SOC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../Results/{file_path}.png')

    # Calculate the RMSE and MAE
    rmse = root_mean_squared_error(updated_bat_data['SOC_true'], updated_bat_data['SOC_Calc'])  # Calculate the RMSE
    mae = max_absolute_error(updated_bat_data['SOC_true'], updated_bat_data['SOC_Calc'])  # Calculate the MAE
    print("RMSE: ", rmse)
    print("MAE: ", mae)

    # Write the updated battery data to an Excel file
    soc_est_cor_array = updated_bat_data['SOC_Calc'].values  # Assuming SOC_Calc is already calculated
    updated_bat_data['SOC_Calc'] = soc_est_cor_array  # Add the calculated SOC to the battery data

    with pd.ExcelWriter('updated_bat_data.xlsx', engine='xlsxwriter') as writer:  # Write the updated battery data to an Excel file
        updated_bat_data.to_excel(writer, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run analysis on battery data.') # Create an argument parser
    parser.add_argument('file_path', type=str, help='Path to the input Excel file') # Add an argument for the input file path
    args = parser.parse_args() # Parse the arguments

    run_KF(args.file_path) # run the Kalman Filter with the specified file path