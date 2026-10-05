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
This script performs Particle Swarm Optimization (PSO) to optimize the parameters of the Kalman Filter to improve the SOC estimations.
It includes methods for defining the objective function, configuring the PSO, and plotting the optimization results.

Input:
- Excel file containing battery test data with columns for current, voltage, SOC.

Output:
- Optimized parameters (Q, R, P0) for the Kalman Filter.
- Plots of the cost and parameter values over the iterations of the PSO.

Usage:
- Configure the PSO with appropriate bounds and hyperparameters.
- Run the optimization to find the best parameters.
- Visualize the optimization process and results.
"""



# Import necessary libraries
import numpy as np
import pyswarms as ps  # Particle Swarm Optimization library
from Optimization_module import Optimization  # Contains the objective function for optimization
import matplotlib.pyplot as plt  # Plotting library
from KalmanFilter_module import KalmanFilter  # Implements the Kalman Filter logic

# Define utility functions for error evaluation
def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def max_absolute_error(y_true, y_pred):
    """Calculate Maximum Absolute Error (MAE)."""
    return np.max(np.abs(y_true - y_pred))

# Define the number of parameters for optimization (Q, R, P0)
n_params = 7  # Includes elements for Q, R, and initial covariance matrix P0

if __name__ == "__main__":
    # Configuration of the Particle Swarm Optimization (PSO)
    optimization_instance = Optimization()

    # Define bounds for the optimization parameters
    min_bound_opt = [1e-9, 1e-7, 1e-7, 1e-5, 1e-5, 1e-6, 1e-5]
    max_bound_opt = [1e-5, 1e-4, 1e-4, 1e-1, 1e-1, 1e-2, 1e-1]
    bounds = (min_bound_opt, max_bound_opt)

    # Seed for reproducibility
    np.random.seed(42)

    # Initialize the PSO optimizer with a single particle for debugging/testing
    optimizer = ps.single.GlobalBestPSO(
        n_particles=6,  # Single particle for quick testing
        dimensions=n_params,  # Number of parameters to optimize
        options={'c1': 2.5, 'c2': 0.5, 'w': 1},  # PSO hyperparameters
        bounds=bounds,  # Bounds for parameters
        oh_strategy={  # Linearly varying strategy for PSO coefficients
            "w": 'lin_variation',
            "c1": 'lin_variation',
            "c2": 'lin_variation'
        }
    )

    # Perform the optimization
    loss, pos = optimizer.optimize(
        optimization_instance.objective_function_parallel,  # Objective function
        iters = 5  # Number of iterations
    )

    # Retrieve the cost history
    cost_history = optimizer.cost_history  # History of costs
    position_history = optimizer.pos_history  # History of particle positions

    # Plot the cost history
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(cost_history)), cost_history, label="Cost")
    plt.title("Loss History During Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()