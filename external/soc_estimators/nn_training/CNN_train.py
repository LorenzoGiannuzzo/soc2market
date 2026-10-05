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

from CNN_train_model import BatteryAnalysisCNN

if __name__ == "__main__":
    file_paths = [
        'GenerateTestData_S1_DAY0to4.xlsx',
        'GenerateTestData_S2_DAY4to7.xlsx',
        'GenerateTestData_S3_DAY0to4.xlsx',
    ]

    sequence_length = 10

    analysis = BatteryAnalysisCNN(file_paths, sequence_length=sequence_length)
    analysis.load_and_combine_data()
    analysis.preprocess_data()

    # Lorenzo Giannuzzo: Use ALL data for training (no test split)
    X_train, y_train = analysis.prepare_data(test_split=False)

    print(f"\nTraining data shape: {X_train.shape}")
    print(f"Training on 100% of data ({len(y_train)} samples)")

    # Lorenzo Giannuzzo: Build model
    analysis.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Lorenzo Giannuzzo: Train for all 150 epochs without early stopping
    history = analysis.train_model(X_train, y_train, epochs=150, batch_size=64, use_early_stopping=False)

    # Lorenzo Giannuzzo: Make predictions and save results
    analysis.make_predictions()
    analysis.save_results()

    print(f"\nTraining completed for {len(history.history['loss'])} epochs")