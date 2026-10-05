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
TFT Resume Training from Checkpoint
================================================================================
Riprende il training del TFT da un checkpoint salvato.
================================================================================
"""

from TFT_train_model import BatteryAnalysisTFT, MAX_EPOCHS, BATCH_SIZE, GRADIENT_CLIP_VAL
from TFT_train_model import VerboseProgressCallback, PRINT_EVERY_N_BATCHES
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATH = 'lightning_logs/version_1/checkpoints/epoch=10-step=79673.ckpt'

# CHECKPOINT_PATH = 'path/to/epoch_10-step_79673.ckpt'

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Lorenzo Giannuzzo: List of file paths (same as original training)
    file_paths = [
        'GenerateTestData_S1_DAY0to4.xlsx',
        'GenerateTestData_S2_DAY4to7.xlsx',
        'GenerateTestData_S3_DAY0to4.xlsx',
    ]

    print("=" * 70)
    print("  TFT - RESUME TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Will continue until epoch {MAX_EPOCHS}")
    print("=" * 70)

    # Lorenzo Giannuzzo: Initialize and prepare data (same as original)
    print("\n[1/4] Loading and combining data...")
    analysis = BatteryAnalysisTFT(file_paths)
    analysis.load_and_combine_data()

    print("\n[2/4] Preprocessing data...")
    analysis.preprocess_data()

    print("\n[3/4] Preparing TimeSeriesDataSet...")
    train_dataloader = analysis.prepare_data()

    # Lorenzo Giannuzzo: Load model from checkpoint
    print("\n[4/4] Loading model from checkpoint and resuming training...")

    # Lorenzo Giannuzzo: Callbacks
    lr_logger = LearningRateMonitor()
    progress_callback = VerboseProgressCallback(print_every_n_batches=PRINT_EVERY_N_BATCHES)

    # Lorenzo Giannuzzo: Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=analysis.device,
        enable_model_summary=True,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[lr_logger, progress_callback],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    # Lorenzo Giannuzzo: Load model from checkpoint
    model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)
    analysis.model = model
    analysis.trainer = trainer

    total_batches = len(train_dataloader)
    print(f"\n  Resuming training...")
    print(f"  Total batches per epoch: {total_batches:,}")
    print(f"  Progress updates every {PRINT_EVERY_N_BATCHES} batches")
    print("-" * 60)

    start_time = time.time()

    # Lorenzo Giannuzzo: Resume training from checkpoint
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        ckpt_path=CHECKPOINT_PATH  # This resumes from checkpoint
    )

    train_time = time.time() - start_time
    print("-" * 60)
    print(f"  Training completed in {train_time / 3600:.2f} hours ({train_time / 60:.1f} minutes)")

    # Lorenzo Giannuzzo: Make predictions and save results
    print("\n[5/5] Making predictions and saving results...")
    analysis.make_predictions()
    analysis.save_results()

    print("\n  Training pipeline complete!")
    print("=" * 70)