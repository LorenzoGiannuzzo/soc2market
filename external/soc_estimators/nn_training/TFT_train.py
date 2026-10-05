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

from TFT_train_model import BatteryAnalysisTFT

"""
Main script for Battery Analysis using Temporal Fusion Transformer (TFT).

This script performs the following tasks:
1. Loads and combines data from specified Excel files containing battery-related data.
2. Runs the Kalman Filter on each file to estimate SOC_Calc.
3. Preprocesses the data to prepare it for training (feature engineering, normalization).
4. Prepares TimeSeriesDataSet with encoder/decoder structure for the TFT.
5. Builds a Temporal Fusion Transformer model.
6. Trains the model with early stopping and learning rate scheduling.
7. Makes predictions and evaluates on validation data.
8. Saves the results (model, scaler, plots, metrics) to a specified location.

The TFT provides several advantages over standard ANN/CNN/LSTM:
- Built-in variable selection (learns which features matter most)
- Multi-head attention mechanism for long-range temporal dependencies
- Interpretable outputs (variable importance, attention weights)
- Gated residual networks for robust feature processing

Features used (same as ANN/CNN/LSTM for fair comparison):
- Temp, Voltage, Current_inv, SOC_Calc
- Delta_Temperature, Delta_Current, Delta_Voltage
- Delta_SoC_Calc, Delta_Voltage_Current_Ratio, Current_Voltage_Ratio

Input:
- A list of Excel file paths containing the battery data.

Output:
- Trained TFT model (tft_model.pt, tft_model.ckpt).
- Scaler (tft_scaler.joblib).
- Config and dataset parameters (tft_config.joblib, tft_dataset_params.joblib).
- Comparison plots and variable importance plots.
- RMSE, MAE, MaxAE, R² metrics.

Usage:
    python TFT_train.py
"""

# Lorenzo Giannuzzo: Main script
if __name__ == "__main__":
    # Lorenzo Giannuzzo: List of file paths used to train the TFT model
    file_paths = [
        'GenerateTestData_S1_DAY0to4.xlsx',
        'GenerateTestData_S2_DAY4to7.xlsx',
        'GenerateTestData_S3_DAY0to4.xlsx',
        # 'GenerateTestData_S4_DAY4to7.xlsx',
    ]

    print("=" * 70)
    print("  TEMPORAL FUSION TRANSFORMER (TFT) - BATTERY SOC PREDICTION")
    print("=" * 70)

    # Lorenzo Giannuzzo: Initialize the BatteryAnalysisTFT object
    print("\n[1/6] Loading and combining data...")
    analysis = BatteryAnalysisTFT(file_paths)
    analysis.load_and_combine_data()

    # Lorenzo Giannuzzo: Preprocess the data
    print("\n[2/6] Preprocessing data...")
    analysis.preprocess_data()

    # Lorenzo Giannuzzo: Prepare the data for training (creates TimeSeriesDataSet)
    print("\n[3/6] Preparing TimeSeriesDataSet...")
    train_dataloader = analysis.prepare_data()

    # Lorenzo Giannuzzo: Build the TFT model
    print("\n[4/6] Building TFT model...")
    analysis.build_model()

    # Lorenzo Giannuzzo: Train the model
    print("\n[5/6] Training model...")
    analysis.train_model(train_dataloader)

    # Lorenzo Giannuzzo: Make predictions and save results
    print("\n[6/6] Making predictions and saving results...")
    analysis.make_predictions()
    analysis.save_results()

    print("\n  Training pipeline complete!")
    print("=" * 70)