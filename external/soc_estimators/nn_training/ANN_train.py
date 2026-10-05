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

from ANN_train_model import BatteryAnalysis

"""
Main script for Battery Analysis using Artificial Neural Networks (ANN).

This script performs the following tasks:
1. Loads and combines data from specified Excel files containing battery-related data.
2. Preprocesses the data to prepare it for training (e.g., normalization, cleaning).
3. Splits the data into training and testing datasets.
4. Builds an Artificial Neural Network model to predict or classify battery-related outcomes.
5. Trains the model using the training data.
6. Makes predictions based on the test data.
7. Saves the results to a specified location for further analysis.

Input:
- A list of Excel file paths containing the battery data (e.g., 'GenerateTestData_S1_DAY0to4.xlsx').

Output:
- Trained neural network model.
- Predicted results and analysis saved to the specified output locations.

Usage:
Run the script with the necessary data files to train the model and evaluate its performance.

"""

# Lorenzo Giannuzzo: Main script
if __name__ == "__main__":
    # List of file paths used to train the ANN model
    file_paths = [
        'GenerateTestData_S1_DAY0to4.xlsx',
        'GenerateTestData_S2_DAY4to7.xlsx',
        'GenerateTestData_S3_DAY0to4.xlsx',
        # Lorenzo Giannuzzo: 'GenerateTestData_S4_DAY4to7.xlsx',
    ]

    analysis = BatteryAnalysis(file_paths)  # Lorenzo Giannuzzo:  Initialize the BatteryAnalysis object
    analysis.load_and_combine_data()        # Lorenzo Giannuzzo:  Load and combine the data from the file paths
    analysis.preprocess_data()              # Lorenzo Giannuzzo:  Preprocess the data

    X_train, X_test, y_train, y_test = analysis.prepare_data()  # Lorenzo Giannuzzo:  Prepare the data for training
    analysis.build_model(X_train.shape[1])                      # Lorenzo Giannuzzo:  Build the neural network model
    analysis.train_model(X_train, y_train)                      # Lorenzo Giannuzzo:  Train the model
    analysis.make_predictions()                                 # Lorenzo Giannuzzo:  Make predictions
    analysis.save_results()                                     # Lorenzo Giannuzzo:  Save the results


