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
================================================================================
TFT - Create Model from Checkpoint
================================================================================
Carica un checkpoint e salva il modello TFT nei formati standard per l'uso
nel main.py e nel progressive_metrics_evaluation.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from joblib import dump
import subprocess

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = 'lightning_logs/version_1/checkpoints/epoch=10-step=79673.ckpt'
# Lorenzo Giannuzzo: Modify with the correct file path of the new checkpoint

# Lorenzo Giannuzzo: Same config as training
MAX_ENCODER_LENGTH = 30
MAX_PREDICTION_LENGTH = 1
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 32
LEARNING_RATE = 0.001

FEATURES = [
    'Temp', 'Voltage', 'Current_inv', 'SOC_Calc',
    'Delta_Temperature', 'Delta_Current', 'Delta_Voltage',
    'Delta_SoC_Calc', 'Delta_Voltage_Current_Ratio',
    'Current_Voltage_Ratio'
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":

    print("=" * 70)
    print("  TFT - CREATE MODEL FROM CHECKPOINT")
    print("=" * 70)
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 70)

    # Lorenzo Giannuzzo: Check checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please update CHECKPOINT_PATH with the correct location.")
        exit(1)

    # Lorenzo Giannuzzo: Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/plots', exist_ok=True)

    # ========================================================================
    # STEP 1: Load and preprocess data (needed for scaler and dataset params)
    # ========================================================================
    print("\n[1/5] Loading training data for scaler...")

    file_paths = [
        'GenerateTestData_S1_DAY0to4.xlsx',
        'GenerateTestData_S2_DAY4to7.xlsx',
        'GenerateTestData_S3_DAY0to4.xlsx',
    ]

    all_dfs = []
    for file_idx, file_path in enumerate(file_paths):
        print(f"  Running Kalman Filter for: {file_path}")
        subprocess.run(['python', 'run_KalmanFilter_module.py', file_path],
                       capture_output=True)

        updated_data_file = 'updated_bat_data.xlsx'
        df = pd.read_excel(updated_data_file)
        df['group_id'] = str(file_idx)
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    # Lorenzo Giannuzzo:  Preprocess (same as training)
    print("\n[2/5] Preprocessing and fitting scaler...")
    df['Current_Voltage_Ratio'] = (df['Current_inv'] / df['Voltage']).fillna(0)
    df['Delta_Current'] = df['Current_inv'].diff().fillna(0)
    df['Delta_Voltage'] = df['Voltage'].diff().fillna(0)
    df['Delta_Voltage_Current_Ratio'] = (
            df['Delta_Voltage'] / df['Delta_Current'].replace(0, 1)
    ).fillna(0)
    df['Delta_SoC_Calc'] = df['SOC_Calc'].diff().fillna(0)
    df['Delta_Temperature'] = df['Temp'].diff().fillna(0)
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df['time_idx'] = df.groupby('group_id').cumcount()
    df['group_id'] = df['group_id'].astype(str)

    # Lorenzo Giannuzzo:  Fit scaler
    scaler = MinMaxScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])

    # Lorenzo Giannuzzo: Create dataset (needed for parameters)
    training_dataset = TimeSeriesDataSet(
        df,
        time_idx='time_idx',
        target='SOC_true',
        group_ids=['group_id'],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=['time_idx'],
        time_varying_unknown_reals=FEATURES + ['SOC_true'],
        target_normalizer=GroupNormalizer(groups=['group_id']),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # ========================================================================
    # STEP 2: Load model from checkpoint
    # ========================================================================
    print("\n[3/5] Loading model from checkpoint...")

    model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model loaded with {n_params:,} parameters")

    # ========================================================================
    # STEP 3: Make predictions to verify model works
    # ========================================================================
    print("\n[4/5] Making predictions on training data...")

    predict_dataloader = training_dataset.to_dataloader(
        train=False, batch_size=128, num_workers=0
    )

    predictions = model.predict(
        predict_dataloader,
        mode="prediction",
        return_x=False,
    )
    y_pred = predictions.cpu().numpy().flatten()

    # Lorenzo Giannuzzo: Collect true values
    y_true_list = []
    for batch_x, batch_y in predict_dataloader:
        y_true_list.append(batch_y[0].cpu().numpy().flatten())
    y_true = np.concatenate(y_true_list)

    # Lorenzo Giannuzzo: Align lengths
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # Lorenzo Giannuzzo: Calculate metrics
    rmse = root_mean_squared_error(y_true, y_pred)
    mean_ae = np.mean(np.abs(y_true - y_pred))
    max_ae = max_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n  Metrics on training data (checkpoint epoch 10):")
    print(f"    RMSE:    {rmse:.4f}")
    print(f"    MeanAE:  {mean_ae:.4f}")
    print(f"    MaxAE:   {max_ae:.4f}")
    print(f"    R²:      {r2:.4f}")

    # ========================================================================
    # STEP 4: Save everything
    # ========================================================================
    print("\n[5/5] Saving model and artifacts...")

    # Lorenzo Giannuzzo: Save model state dict
    torch.save(model.state_dict(), 'models/tft_model.pt')
    print("  ✓ models/tft_model.pt")

    # Lorenzo Giannuzzo: Copy checkpoint as .ckpt
    import shutil

    shutil.copy(CHECKPOINT_PATH, 'models/tft_model.ckpt')
    print("  ✓ models/tft_model.ckpt")

    # Lorenzo Giannuzzo: Save scaler
    dump(scaler, 'models/tft_scaler.joblib')
    print("  ✓ models/tft_scaler.joblib")

    # Lorenzo Giannuzzo: Save config
    config = {
        'max_encoder_length': MAX_ENCODER_LENGTH,
        'max_prediction_length': MAX_PREDICTION_LENGTH,
        'hidden_size': HIDDEN_SIZE,
        'attention_head_size': ATTENTION_HEAD_SIZE,
        'dropout': DROPOUT,
        'hidden_continuous_size': HIDDEN_CONTINUOUS_SIZE,
        'features': FEATURES,
        'learning_rate': LEARNING_RATE,
        'checkpoint_epoch': 10,
    }
    dump(config, 'models/tft_config.joblib')
    print("  ✓ models/tft_config.joblib")

    # Lorenzo Giannuzzo: Save dataset parameters
    dump(training_dataset.get_parameters(), 'models/tft_dataset_params.joblib')
    print("  ✓ models/tft_dataset_params.joblib")

    # ========================================================================
    # STEP 5: Create comparison plot
    # ========================================================================
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(len(y_true)), y_true, label='True SOC', color='blue', alpha=0.6)
    plt.plot(np.arange(len(y_pred)), y_pred, label='EKF + TFT (epoch 10)', color='red', alpha=0.6)
    plt.title('State of Charge (SoC) Comparison - TFT (Checkpoint Epoch 10)')
    plt.xlabel('Index')
    plt.ylabel('State of Charge')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('models/plots/TFT_Comparison_plot.png', dpi=300)
    plt.savefig('models/plots/TFT_Comparison_plot.pdf', dpi=500)
    plt.close()
    print("  ✓ models/plots/TFT_Comparison_plot.png")
    print("  ✓ models/plots/TFT_Comparison_plot.pdf")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("  TFT MODEL CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Source: Checkpoint at epoch 10")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MeanAE: {mean_ae:.4f}")
    print(f"  MaxAE: {max_ae:.4f}")
    print(f"  R²: {r2:.4f}")
    print("=" * 70)
    print("\n  Model is ready to use!")
    print("  You can now add TFT to main.py for comparison with ANN/CNN/LSTM/CatBoost")
    print("=" * 70)