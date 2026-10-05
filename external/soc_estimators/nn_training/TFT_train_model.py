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

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, Callback
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, RMSE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from joblib import dump


warnings.filterwarnings("ignore", ".*does not have many workers.*")

"""
This script trains a Temporal Fusion Transformer (TFT) model on battery data.

The script reads battery data from multiple Excel files, runs the Kalman Filter on each file 
to estimate the State of Charge (SOC), and combines the updated data. It then preprocesses the 
data, builds and trains a TFT model to predict SOC, and saves the results, including plots and 
the trained model.

The TFT architecture provides:
- Variable Selection Networks: automatically learn which features are most relevant
- Temporal attention: capture long-range dependencies in time series
- Interpretable multi-horizon forecasting with attention weights

Input:
- List of paths to Excel files containing battery data.

Output:
- Combined and preprocessed battery data.
- Trained TFT model.
- Plots comparing true SOC, EKF SOC, and TFT-predicted SOC.
- Variable importance plots.
- RMSE and MAE error metrics.
"""

# ============================================================================
# CONFIGURATION
# ============================================================================
MAX_ENCODER_LENGTH = 30       # Lorenzo Giannuzzo: Number of past timesteps the model sees
MAX_PREDICTION_LENGTH = 1     # Lorenzo Giannuzzo: Predict 1 step ahead (single-step SOC prediction)
BATCH_SIZE = 128
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 32
GRADIENT_CLIP_VAL = 0.1
PRINT_EVERY_N_BATCHES = 500  # Print progress every N batches


# ============================================================================
# CUSTOM PROGRESS CALLBACK
# ============================================================================
class VerboseProgressCallback(Callback):
    """Custom callback to print detailed progress during training."""

    def __init__(self, print_every_n_batches=500):
        super().__init__()
        self.print_every_n_batches = print_every_n_batches
        self.epoch_start_time = None
        self.batch_start_time = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        print(f"\n  [Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] Started...")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.print_every_n_batches == 0:
            total_batches = trainer.num_training_batches
            elapsed = time.time() - self.epoch_start_time
            progress = (batch_idx + 1) / total_batches * 100
            loss = outputs['loss'].item() if isinstance(outputs, dict) and 'loss' in outputs else 0

            # Estimate remaining time for this epoch
            batches_remaining = total_batches - (batch_idx + 1)
            time_per_batch = elapsed / (batch_idx + 1)
            eta = batches_remaining * time_per_batch

            print(f"    Batch {batch_idx + 1:,}/{total_batches:,} ({progress:.1f}%) | "
                  f"Loss: {loss:.4f} | "
                  f"Elapsed: {elapsed/60:.1f}min | "
                  f"ETA: {eta/60:.1f}min")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = time.time() - self.epoch_start_time
        train_loss = trainer.callback_metrics.get('train_loss_epoch', 0)
        if hasattr(train_loss, 'item'):
            train_loss = train_loss.item()

        print(f"  [Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}] "
              f"Completed in {epoch_time/60:.1f}min | "
              f"Train Loss: {train_loss:.4f}")


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))


class BatteryAnalysisTFT:
    def __init__(self, file_paths):
        """
        Initialize the BatteryAnalysisTFT object with a list of file paths.
        :param file_paths: List of Excel file paths to analyze.
        """
        self.file_paths = file_paths
        self.df = None
        self.model = None
        self.trainer = None
        self.training_dataset = None
        self.predict_dataloader = None
        self.scaler = MinMaxScaler()
        self.device = "gpu" if torch.cuda.is_available() else "cpu"

        # Feature definitions (same as ANN/CNN/LSTM)
        self.features = [
            'Temp', 'Voltage', 'Current_inv', 'SOC_Calc',
            'Delta_Temperature', 'Delta_Current', 'Delta_Voltage',
            'Delta_SoC_Calc', 'Delta_Voltage_Current_Ratio',
            'Current_Voltage_Ratio'
        ]

    def load_and_combine_data(self):
        """
        Run Kalman Filter for each file, load the updated data, and combine into a single DataFrame.
        """
        all_dfs = []
        for file_idx, file_path in enumerate(self.file_paths):
            print(f"  Running Kalman Filter for: {file_path}")
            subprocess.run(['python', 'run_KalmanFilter_module.py', file_path])

            updated_data_file = 'updated_bat_data.xlsx'
            df = pd.read_excel(updated_data_file)
            df['group_id'] = str(file_idx)  # Assign group ID for TFT
            all_dfs.append(df)

        self.df = pd.concat(all_dfs, ignore_index=True)

    def preprocess_data(self):
        """
        Perform preprocessing on the DataFrame, adding calculated features.
        Same feature engineering as ANN/CNN/LSTM for fair comparison.
        """
        try:
            # Same features as ANN/CNN/LSTM
            self.df['Current_Voltage_Ratio'] = (
                self.df['Current_inv'] / self.df['Voltage']
            ).fillna(0)
            self.df['Delta_Current'] = self.df['Current_inv'].diff().fillna(0)
            self.df['Delta_Voltage'] = self.df['Voltage'].diff().fillna(0)
            self.df['Delta_Voltage_Current_Ratio'] = (
                self.df['Delta_Voltage'] / self.df['Delta_Current'].replace(0, 1)
            ).fillna(0)
            self.df['Delta_SoC_Calc'] = self.df['SOC_Calc'].diff().fillna(0)
            self.df['Delta_Temperature'] = self.df['Temp'].diff().fillna(0)
            self.df.fillna(0, inplace=True)

            # Lorenzo Giannuzzo: Replace infinities
            self.df.replace([np.inf, -np.inf], 0, inplace=True)

            # Lorenzo Giannuzzo: Add time index per group (required by TFT)
            self.df['time_idx'] = self.df.groupby('group_id').cumcount()

            # Lorenzo Giannuzzo: Ensure group_id is string type
            self.df['group_id'] = self.df['group_id'].astype(str)

            print(f"  Data shape after preprocessing: {self.df.shape}")
            print(f"  Groups: {self.df['group_id'].nunique()}")
            print(f"  Time steps per group: {self.df.groupby('group_id')['time_idx'].max().to_dict()}")

        except Exception as e:
            print(f"  An error occurred during preprocessing: {e}")
            raise

    def prepare_data(self):
        """
        Prepare TimeSeriesDataSet for TFT training.
        Uses 100% of data for training (no validation split), same as ANN/CNN/LSTM.
        Returns train dataloader.
        """
        # Lorenzo Giannuzzo: Scale features using MinMaxScaler (same as other models)
        self.df[self.features] = self.scaler.fit_transform(self.df[self.features])

        max_time_idx = self.df['time_idx'].max()
        print(f"  Max time index: {max_time_idx}")

        # Lorenzo Giannuzzo: Create training dataset using ALL data
        self.training_dataset = TimeSeriesDataSet(
            self.df,
            time_idx='time_idx',
            target='SOC_true',
            group_ids=['group_id'],
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=MAX_PREDICTION_LENGTH,
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=self.features + ['SOC_true'],
            target_normalizer=GroupNormalizer(groups=['group_id']),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Lorenzo Giannuzzo: Create dataloader
        train_dataloader = self.training_dataset.to_dataloader(
            train=True, batch_size=BATCH_SIZE, num_workers=0
        )

        # Lorenzo Giannuzzo: Also create a non-shuffled dataloader for predictions/metrics
        self.predict_dataloader = self.training_dataset.to_dataloader(
            train=False, batch_size=BATCH_SIZE, num_workers=0
        )

        print(f"  Training samples: {len(self.training_dataset)} (100% of data)")

        return train_dataloader

    def build_model(self):
        """
        Build and compile the TFT model.
        """
        if self.training_dataset is None:
            raise ValueError("Dataset has not been prepared. Call prepare_data() first.")

        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=LEARNING_RATE,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTENTION_HEAD_SIZE,
            dropout=DROPOUT,
            hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
            output_size=7,  # 7 quantiles by default
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=5,
        )

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  TFT model built with {n_params:,} trainable parameters")
        print(f"  Hidden size: {HIDDEN_SIZE}")
        print(f"  Attention heads: {ATTENTION_HEAD_SIZE}")
        print(f"  Encoder length: {MAX_ENCODER_LENGTH}")
        print(f"  Device: {self.device}")

    def train_model(self, train_dataloader):
        """
        Train the TFT model on 100% of data for all epochs (no early stopping).
        :param train_dataloader: Training dataloader.
        """
        if self.model is None:
            raise ValueError("Model has not been built. Call build_model() first.")

        # Lorenzo Giannuzzo: Callbacks
        lr_logger = LearningRateMonitor()
        progress_callback = VerboseProgressCallback(print_every_n_batches=PRINT_EVERY_N_BATCHES)

        # Lorenzo Giannuzzo: Trainer - no early stopping, trains for all MAX_EPOCHS
        self.trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator=self.device,
            enable_model_summary=True,
            gradient_clip_val=GRADIENT_CLIP_VAL,
            callbacks=[lr_logger, progress_callback],
            enable_progress_bar=True,
            log_every_n_steps=1,  # Lorenzo Giannuzzo:  Log every step for visibility
        )

        total_batches = len(train_dataloader)
        print(f"\n  Training TFT for {MAX_EPOCHS} epochs (no early stopping)...")
        print(f"  Total batches per epoch: {total_batches:,}")
        print(f"  Progress updates every {PRINT_EVERY_N_BATCHES} batches")
        print(f"  Estimated time per epoch on CPU: ~30-60 minutes")
        print("-" * 60)

        start_time = time.time()
        self.trainer.fit(
            self.model,
            train_dataloaders=train_dataloader,
        )
        train_time = time.time() - start_time

        print("-" * 60)
        print(f"  Training completed in {train_time/3600:.2f} hours ({train_time/60:.1f} minutes)")

    def make_predictions(self):
        """
        Make predictions on the full training dataset and compute metrics.
        """
        if self.model is None:
            raise ValueError("Model has not been trained.")

        self.model.eval()

        # Predict on full training data
        predictions = self.model.predict(
            self.predict_dataloader,
            mode="prediction",
            return_x=False,
        )
        self.y_pred = predictions.cpu().numpy().flatten()

        # Lorenzo Giannuzzo: Collect true values
        y_true_list = []
        for batch_x, batch_y in self.predict_dataloader:
            y_true_list.append(batch_y[0].cpu().numpy().flatten())
        self.y_true = np.concatenate(y_true_list)

        # Lorenzo Giannuzzo: Align lengths
        min_len = min(len(self.y_true), len(self.y_pred))
        self.y_true = self.y_true[:min_len]
        self.y_pred = self.y_pred[:min_len]

        # Lorenzo Giannuzzo: For comparison plot (same data since we train on 100%)
        self.full_true = self.y_true
        self.full_pred = self.y_pred

    def save_results(self):
        """
        Save the results, including plots, the trained model, and metrics.
        """
        os.makedirs('models', exist_ok=True)
        os.makedirs('models/plots', exist_ok=True)

        # ---- Comparison Plot ----
        plt.figure(figsize=(14, 7))
        plt.plot(
            np.arange(len(self.full_true)),
            self.full_true,
            label='True SOC', color='blue', alpha=0.6
        )
        plt.plot(
            np.arange(len(self.full_pred)),
            self.full_pred,
            label='EKF + TFT', color='red', alpha=0.6
        )
        plt.title('State of Charge (SoC) Comparison - TFT')
        plt.xlabel('Index')
        plt.ylabel('State of Charge')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('models/plots/TFT_Comparison_plot.png', dpi=300)
        plt.savefig('models/plots/TFT_Comparison_plot.pdf', dpi=500)
        plt.close()

        # ---- Variable Importance Plot ----
        try:
            self._plot_variable_importance()
        except Exception as e:
            print(f"  Warning: Could not generate variable importance plot: {e}")

        # ---- Save Model ----
        torch.save(self.model.state_dict(), 'models/tft_model.pt')

        # Lorenzo Giannuzzo: Save full model checkpoint for easier loading
        self.trainer.save_checkpoint('models/tft_model.ckpt')

        # Lorenzo Giannuzzo: Save scaler
        dump(self.scaler, 'models/tft_scaler.joblib')

        # Lorenzo Giannuzzo: Save config
        config = {
            'max_encoder_length': MAX_ENCODER_LENGTH,
            'max_prediction_length': MAX_PREDICTION_LENGTH,
            'hidden_size': HIDDEN_SIZE,
            'attention_head_size': ATTENTION_HEAD_SIZE,
            'dropout': DROPOUT,
            'hidden_continuous_size': HIDDEN_CONTINUOUS_SIZE,
            'features': self.features,
            'learning_rate': LEARNING_RATE,
        }
        dump(config, 'models/tft_config.joblib')

        # Lorenzo Giannuzzo: Save dataset parameters for inference reconstruction
        dump(self.training_dataset.get_parameters(), 'models/tft_dataset_params.joblib')

        # ---- Metrics ----
        mae_val = max_absolute_error(self.y_true, self.y_pred)
        rmse_val = root_mean_squared_error(self.y_true, self.y_pred)
        mean_ae = np.mean(np.abs(self.y_true - self.y_pred))
        r2 = r2_score(self.y_true, self.y_pred)

        print("\n" + "=" * 60)
        print("  TFT RESULTS (Training Set)")
        print("=" * 60)
        print(f"  Root Mean Squared Error:  {rmse_val:.4f}")
        print(f"  Mean Absolute Error:      {mean_ae:.4f}")
        print(f"  Max Absolute Error:       {mae_val:.4f}")
        print(f"  R² Score:                 {r2:.4f}")
        print("=" * 60)

        print("\n  Saved files:")
        print("    - models/tft_model.pt")
        print("    - models/tft_model.ckpt")
        print("    - models/tft_scaler.joblib")
        print("    - models/tft_config.joblib")
        print("    - models/tft_dataset_params.joblib")
        print("    - models/plots/TFT_Comparison_plot.png")
        print("    - models/plots/TFT_Comparison_plot.pdf")

    def _plot_variable_importance(self):
        """
        Plot TFT variable importance using the model's built-in interpretation.
        """
        interpretation = self.model.interpret_output(
            self.model.predict(
                self.predict_dataloader,
                mode="raw",
                return_x=True,
            ),
            reduction="sum",
        )

        # Lorenzo Giannuzzo: Extract attention and variable importance
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Encoder variable importance
        if 'encoder_variables' in interpretation:
            enc_importance = interpretation['encoder_variables']
            var_names = self.training_dataset.reals
            if len(enc_importance) == len(var_names):
                sorted_idx = np.argsort(enc_importance.cpu().numpy())
                axes[0].barh(
                    [var_names[i] for i in sorted_idx],
                    enc_importance.cpu().numpy()[sorted_idx],
                    color='steelblue'
                )
                axes[0].set_title('Encoder Variable Importance')
                axes[0].set_xlabel('Importance')

        # Lorenzo Giannuzzo: Decoder variable importance
        if 'decoder_variables' in interpretation:
            dec_importance = interpretation['decoder_variables']
            dec_names = self.training_dataset.reals
            if len(dec_importance) == len(dec_names):
                sorted_idx = np.argsort(dec_importance.cpu().numpy())
                axes[1].barh(
                    [dec_names[i] for i in sorted_idx],
                    dec_importance.cpu().numpy()[sorted_idx],
                    color='darkorange'
                )
                axes[1].set_title('Decoder Variable Importance')
                axes[1].set_xlabel('Importance')

        plt.suptitle('TFT Variable Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('models/plots/TFT_variable_importance.png', dpi=300)
        plt.savefig('models/plots/TFT_variable_importance.pdf', dpi=500)
        plt.close()
        print("  Variable importance plot saved.")
