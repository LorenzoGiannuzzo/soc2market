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
# Funzione obiettivo per PSO
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from KalmanFilter_module import KalmanFilter

"""
Description:
This script defines an `Optimization` class to perform **Particle Swarm Optimization (PSO)** on the parameters of a Kalman Filter. 
The optimization minimizes the Root Mean Squared Error (RMSE) between the true State of Charge (SOC) and the calculated SOC (`SOC_Calc`) produced by the Kalman Filter.

Key Components:
1. **Root Mean Squared Error (RMSE)**: A utility function to measure the difference between predicted and true values.
2. **Optimization Class**:
   - Manages the optimization process for the Kalman Filter parameters.
   - Evaluates the RMSE for different sets of parameters using the Kalman Filter.

Main Features:
1. **Parameter Initialization**:
   - Kalman Filter parameters include:
     - Process noise covariance matrix (`Q`)
     - Measurement noise covariance (`R`)
     - Initial covariance matrix (`P0`)

2. **RMSE Calculation**:
   - The `kalman_filter_error` method evaluates a given parameter set by:
     - Running the Kalman Filter with the provided parameters.
     - Calculating the RMSE between the true and estimated SOC values.

3. **Parallel Objective Function**:
   - The `objective_function_parallel` method evaluates RMSE for multiple sets of parameters in parallel using `ProcessPoolExecutor`, speeding up the optimization process.

Inputs:
- `params` (list of lists): A list where each entry contains a set of parameters to optimize, defined as:
  `[Q11, Q22, Q33, R, P011, P022, P033]`
  - `Q11, Q22, Q33`: Diagonal elements of the process noise covariance matrix (`Q`).
  - `R`: Measurement noise covariance.
  - `P011, P022, P033`: Diagonal elements of the initial covariance matrix (`P0`).

Outputs:
- `rmse_final` (list): A list of RMSE values calculated for each parameter set.

Usage:
- Instantiate the `Optimization` class.
- Call the `objective_function_parallel` method with a list of parameter sets.
- The method returns the RMSE values for all evaluated sets of parameters.

"""

def root_mean_squared_error(y_true, y_pred):
    """Calculate the Root Mean Squared Error (RMSE) between true and predicted values."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

class Optimization():
    """
    Optimization class for performing Particle Swarm Optimization (PSO) on Kalman Filter parameters.

    Attributes:
        current_iteration (int): The current iteration of the optimization process.
        params_rmse (list): A list to store RMSE values for each set of parameters.
        KF (KalmanFilter): An instance of the KalmanFilter class initialized with default parameters.
    """
    def __init__(self):
        """Initialize the Optimization class with default values and a KalmanFilter instance."""
        self.current_iteration = 0
        self.params_rmse = []
        self.KF = KalmanFilter('GenerateTestData_S1_DAY0to4.xlsx', dt=1, initial_SOC=60)

    def kalman_filter_error(self, params):
        """
        Calculate the RMSE for a given set of Kalman Filter parameters.

        Args:
            params (list): A list of parameters [Q11, Q22, Q33, R, P011, P022, P033].

        Returns:
            float: The RMSE value for the given parameters.
        """
        Q = np.array([[params[0], 0, 0], [0, params[1], 0], [0, 0, params[2]]])  # Process noise covariance
        R = params[3]  # Measurement noise covariance
        print(params)
        P0 = np.array([[params[4], 0, 0], [0, params[5], 0], [0, 0, params[6]]])  # Initial covariance
        # Perform a Kalman filter with the given parameters
        updated_bat_data = self.KF.kalman_filter(Q=Q, R=R, initial_covariance=P0)
        # Calculate the RMSE between true SOC and calculated SOC
        rmse = root_mean_squared_error(updated_bat_data['SOC_true'], updated_bat_data['SOC_Calc'])
        return rmse

    def objective_function_parallel(self, params):
        """
        Objective function for PSO that calculates RMSE in parallel for multiple sets of parameters.

        Args:
            params (list): A list of parameter sets to evaluate.

        Returns:
            list: A list of RMSE values for each set of parameters.
        """
        with ProcessPoolExecutor(max_workers=4) as executor:
            rmse_list = list(executor.map(self.kalman_filter_error, params))
        self.rmse_final = [rmse for rmse in rmse_list]
        self.current_iteration += 1
        self.params_rmse.append(self.rmse_final)
        return self.rmse_final