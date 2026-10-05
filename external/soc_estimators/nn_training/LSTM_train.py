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
LSTM Training Script for Battery SOC Prediction

This script trains a Long Short-Term Memory (LSTM) neural network
for State of Charge prediction, using the same data and features
as the ANN and CNN models for fair comparison.

Training is performed on 100% of the data (no test split).
Testing will be done separately on different data files.
"""

from LSTM_train_model import BatteryAnalysisLSTM

if __name__ == "__main__":
    # Lorenzo Giannuzzo: List of file paths used to train the LSTM model
    # Lorenzo Giannuzzo: Same files as ANN and CNN for fair comparison
    file_paths = [
        'GenerateTestData_S1_DAY0to4.xlsx',
        'GenerateTestData_S2_DAY4to7.xlsx',
        'GenerateTestData_S3_DAY0to4.xlsx',
    ]

    # Lorenzo Giannuzzo: Initialize with sequence length (number of past time steps to consider)
    # Lorenzo Giannuzzo: Same as CNN for fair comparison
    sequence_length = 10
    
    print("="*60)
    print("  LSTM MODEL TRAINING FOR SOC PREDICTION")
    print("="*60)
    print(f"\nSequence length: {sequence_length}")
    print(f"Training files: {len(file_paths)}")
    print(f"Training mode: 100% data (no test split)")
    
    analysis = BatteryAnalysisLSTM(file_paths, sequence_length=sequence_length)
    
    print("\n[1/5] Loading and combining data...")
    analysis.load_and_combine_data()
    
    print("\n[2/5] Preprocessing data...")
    analysis.preprocess_data()

    print("\n[3/5] Preparing sequences (100% for training)...")
    # Lorenzo Giannuzzo: test_split=False means all data is used for training
    X_train, y_train = analysis.prepare_data(test_split=False)
    
    print(f"\n  Training data shape: {X_train.shape}")
    print(f"  Sequence length: {X_train.shape[1]}")
    print(f"  Features per timestep: {X_train.shape[2]}")
    
    # Lorenzo Giannuzzo: Build model with input shape (sequence_length, num_features)
    print("\n[4/5] Building LSTM model...")
    analysis.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Lorenzo Giannuzzo: Train model on all data (no early stopping, no validation)
    print("\n[5/5] Training model (150 epochs, no early stopping)...")
    history = analysis.train_model(
        X_train, y_train, 
        epochs=150,
        batch_size=64,
        use_early_stopping=False  # Train for all 150 epochs
    )
    
    # Lorenzo Giannuzzo: Make predictions on training data
    print("\n" + "="*60)
    print("  EVALUATION ON TRAINING DATA")
    print("="*60)
    print("\nMaking predictions on full dataset...")
    analysis.make_predictions()
    
    print("\nSaving results...")
    analysis.save_results()
    
    print("\n" + "="*60)
    print("  TRAINING COMPLETE!")
    print("="*60)
    print("\nOutput files saved in 'Results/' folder:")
    print("  - lstm_model.keras (trained model)")
    print("  - lstm_scaler.joblib (feature scaler)")
    print("  - lstm_config.txt (configuration)")
    print("  - lstm_metrics.txt (performance metrics on training data)")
    print("  - LSTM_Comparison_plot.png (SOC comparison)")
    print("  - lstm_training_history.png (training curves)")
    print("\nNote: These metrics are on TRAINING data.")
    print("Use separate test files for final evaluation.")
