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

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import os
from pathlib import Path
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.lines import Line2D
from scipy import stats
import logging
import sys

# Lorenzo Giannuzzo: Add subdirectory to path for imports
script_dir = Path(__file__).parent
sys.path.append(str(script_dir / 'model_execution_modules'))

from KalmanFilter_module_all_data import KalmanFilter
from ANN_prediction_module_all_data import ANN_BatteryAnalysis
from CNN_prediction_module_all_data import CNN_BatteryAnalysis
from LSTM_prediction_module_all_data import LSTM_BatteryAnalysis
from CATBOOST_prediction_module_all_data import CATBOOST_BatteryAnalysis
from TFT_prediction_module_all_data import TFT_BatteryAnalysis
from ensemble_prediction_module_all_data import Ensemble_BatteryAnalysis


# ============================================================================
# Lorenzo Giannuzzo: PLOT STYLE CONFIGURATION
# Blue/Cyan palette for first models, Orange palette for remaining
# ============================================================================

# Lorenzo Giannuzzo: Color palette
COLOR_TRUE_SOC = '#1A1A1A'       # Lorenzo Giannuzzo: Near-black for ground truth
COLOR_KF       = '#0A2463'       # Lorenzo Giannuzzo: Navy blue - Kalman Filter
COLOR_ANN      = '#1E88E5'       # Lorenzo Giannuzzo: Vivid blue - ANN
COLOR_CNN      = '#42A5F5'       # Lorenzo Giannuzzo: Medium blue - CNN
COLOR_LSTM     = '#E65100'       # Lorenzo Giannuzzo: Deep orange - LSTM
COLOR_CATBOOST = '#FB8C00'       # Lorenzo Giannuzzo: Amber orange - CatBoost
COLOR_TFT      = '#7B1FA2'       # Lorenzo Giannuzzo: Purple - TFT (Transformer)
COLOR_ENSEMBLE = '#2E7D32'       # Lorenzo Giannuzzo: Dark green - Ensemble

# Lorenzo Giannuzzo: Bar chart colors (same palette, ordered as ANN, CNN, LSTM, CatBoost, TFT, Ensemble)
BAR_COLORS = [COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT, COLOR_ENSEMBLE]

# Lorenzo Giannuzzo: Line styles for SOC time series (each model gets a unique style)
STYLE_TRUE_SOC = {'color': COLOR_TRUE_SOC, 'linestyle': '-',        'linewidth': 2.0, 'alpha': 1.0}
STYLE_KF       = {'color': COLOR_KF,       'linestyle': '--',       'linewidth': 1.3, 'alpha': 0.85}
STYLE_ANN      = {'color': COLOR_ANN,      'linestyle': '-.',       'linewidth': 1.3, 'alpha': 0.85}
STYLE_CNN      = {'color': COLOR_CNN,       'linestyle': ':',        'linewidth': 1.5, 'alpha': 0.85}
STYLE_LSTM     = {'color': COLOR_LSTM,      'linestyle': (0,(5,2)),  'linewidth': 1.3, 'alpha': 0.85}       # Lorenzo Giannuzzo:  custom dashed
STYLE_CATBOOST = {'color': COLOR_CATBOOST,  'linestyle': (0,(3,1,1,1)), 'linewidth': 1.3, 'alpha': 0.85}    # Lorenzo Giannuzzo:  dash-dot-dot
STYLE_TFT      = {'color': COLOR_TFT,       'linestyle': (0,(1,1)),  'linewidth': 1.5, 'alpha': 0.85}       # Lorenzo Giannuzzo:  dotted
STYLE_ENSEMBLE = {'color': COLOR_ENSEMBLE,  'linestyle': '-',        'linewidth': 2.0, 'alpha': 0.95}       # Lorenzo Giannuzzo:  solid thick

# Lorenzo Giannuzzo: Mapping model name -> style dict (for programmatic access)
MODEL_STYLES = {
    'ANN':      STYLE_ANN,
    'CNN':      STYLE_CNN,
    'LSTM':     STYLE_LSTM,
    'CatBoost': STYLE_CATBOOST,
    'TFT':      STYLE_TFT,
    'Ensemble': STYLE_ENSEMBLE,
}

# Lorenzo Giannuzzo: Colors for ensemble dominant model visualization
ENSEMBLE_BASE_COLORS = {
    'ANN': COLOR_ANN,
    'CNN': COLOR_CNN,
    'LSTM': COLOR_LSTM,
    'CatBoost': COLOR_CATBOOST,
    'TFT': COLOR_TFT,
}

MODEL_NAMES = ['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT', 'Ensemble']
BASE_MODEL_NAMES = ['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']  # Lorenzo Giannuzzo: Models used by Ensemble


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def max_absolute_error(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def standardize_columns(df):
    """Standardize column names to expected format."""
    column_mappings = {
        'Voltage(V)': 'Voltage',
        'Current_inv(A)': 'Current_inv',
        'SOC_true(%)': 'SOC_true',
        'SOC_True(%)': 'SOC_true',
        'SOC_True': 'SOC_true',
        'Temp1': 'Temp',
        'Step(s)': 'Step',
    }
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    # Lorenzo Giannuzzo: Convert numeric columns to float (handle string values)
    numeric_columns = ['Voltage', 'Current_inv', 'SOC_true', 'Temp', 'Step']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def print_file_progress(phase, scenario_name, model_type=''):
    """Print progress for individual file processing phases."""
    phases = {
        'load': ('Loading data...', 6),
        'kalman': ('Kalman Filter...', 12),
        'ann': ('ANN Prediction...', 26),
        'cnn': ('CNN Prediction...', 40),
        'lstm': ('LSTM Prediction...', 54),
        'catboost': ('CatBoost Prediction...', 68),
        'tft': ('TFT Prediction...', 82),
        'ensemble': ('Ensemble Prediction...', 94),
        'save': ('Saving results...', 100),
    }
    symbol, percent = phases[phase]
    bar_length = 20
    filled = int(bar_length * percent // 100)
    bar = '=' * filled + '-' * (bar_length - filled)
    model_str = f" [{model_type}]" if model_type else ""
    print(f'\r  [{bar}] {percent:>3}% {symbol}{model_str:<25}', end='', flush=True)


def run_kalman_filter(bat_data_df, SOC_initial, scenario_name):
    """Run Kalman Filter on data."""
    if 'Temp' not in bat_data_df.columns:
        bat_data_df["Temp"] = 25.0

    if 'HPPC' in scenario_name and 'DSG' in scenario_name:
        capacity_charge = 3600
        capacity_discharge = 3600
    else:
        capacity_charge = 280
        capacity_discharge = 280

    KF = KalmanFilter(bat_data_df, dt=1, capacity_charge=capacity_charge, capacity_discharge=capacity_discharge,
                      opt=False, initial_SOC=SOC_initial)
    P = np.array([[3.393996e-03, 0, 0], [0, 6.662388e-04, 0], [0, 0, 6.946038e-03]])
    Q = np.array([[4.034329e-07, 0, 0], [0, 8.628196e-05, 0], [0, 0, 9.095089e-05]])
    R = 3.743003e-03

    updated_bat_data = KF.kalman_filter(Q=Q, R=R, initial_covariance=P)
    return updated_bat_data


def run_model_prediction(updated_bat_data, model_type, file_path_model, file_path_scaler,
                         sequence_length=10, file_path_feature_columns=None, file_path_config=None,
                         ensemble_type='selector'):
    """Run ANN, CNN, LSTM, CatBoost, TFT or Ensemble prediction on KF output."""
    if model_type == 'ANN':
        model = ANN_BatteryAnalysis(file_path_model, file_path_scaler)
    elif model_type == 'CNN':
        model = CNN_BatteryAnalysis(file_path_model, file_path_scaler, sequence_length=sequence_length)
    elif model_type == 'LSTM':
        model = LSTM_BatteryAnalysis(file_path_model, file_path_scaler, sequence_length=sequence_length)
    elif model_type == 'CatBoost':
        model = CATBOOST_BatteryAnalysis(file_path_model, file_path_scaler,
                                         feature_columns_path=file_path_feature_columns)
    elif model_type == 'TFT':
        model = TFT_BatteryAnalysis(file_path_model, file_path_scaler, config_path=file_path_config)
    elif model_type == 'Ensemble':
        # Lorenzo Giannuzzo: Ensemble uses different constructor: (ensemble_type, model_dir)
        from pathlib import Path
        model_dir = str(Path(file_path_model).parent)
        model = Ensemble_BatteryAnalysis(ensemble_type=ensemble_type, model_dir=model_dir)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    bat_data_predicted = model.make_predictions(updated_bat_data.copy())

    # Lorenzo Giannuzzo: For Ensemble, also get the weights if available
    if model_type == 'Ensemble' and hasattr(model, 'get_model_weights'):
        try:
            weights_df = model.get_model_weights(updated_bat_data.copy())
            # Lorenzo Giannuzzo: Convert DataFrame to list of arrays for storage
            bat_data_predicted['Ensemble_Weights'] = weights_df.values.tolist()
        except Exception as e:
            print(f"\n    Warning: Could not get ensemble weights: {e}")
            bat_data_predicted['Ensemble_Weights'] = None

    return bat_data_predicted


def calculate_progressive_metrics(y_true, y_pred, percentages=[20, 40, 80, 100]):
    """Calculate metrics at different percentages of data."""
    # Lorenzo Giannuzzo: Handle NaN values
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    n = len(y_true_valid)
    metrics = {}

    for pct in percentages:
        end_idx = int(n * pct / 100)
        if end_idx == 0:
            end_idx = 1

        y_true_subset = y_true_valid[:end_idx]
        y_pred_subset = y_pred_valid[:end_idx]

        metrics[f'RMSE_{pct}pct'] = root_mean_squared_error(y_true_subset, y_pred_subset)
        metrics[f'MeanAE_{pct}pct'] = mean_absolute_error(y_true_subset, y_pred_subset)
        metrics[f'MaxAE_{pct}pct'] = max_absolute_error(y_true_subset, y_pred_subset)

    return metrics


# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_comparison(scenario_name, results_dir, all_metrics_dict, percentages=[20, 40, 80, 100]):
    """Plot comparison bar charts for all models."""

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    available_models = [m for m in MODEL_NAMES if m in all_metrics_dict]
    x = np.arange(len(percentages))
    n_models = len(available_models)
    width = 0.8 / n_models

    model_labels = [f'KF + {m}' for m in available_models]
    offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

    for ax_idx, (metric_key, ylabel, title) in enumerate([
        ('RMSE', 'RMSE [%]', 'RMSE Comparison'),
        ('MeanAE', 'Mean AE [%]', 'Mean Absolute Error Comparison'),
        ('MaxAE', 'Max AE [%]', 'Max Absolute Error Comparison'),
    ]):
        ax = axes[ax_idx]
        for i, (model_name, offset) in enumerate(zip(available_models, offsets)):
            if model_name in all_metrics_dict:
                metrics = all_metrics_dict[model_name]
                values = [metrics[f'{metric_key}_{p}pct'] for p in percentages]
                color_idx = MODEL_NAMES.index(model_name) if model_name in MODEL_NAMES else i
                ax.bar(x + offset, values, width, label=model_labels[i],
                       color=BAR_COLORS[color_idx], edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Data Percentage')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}%' for p in percentages])
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Model Comparison - {scenario_name}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_comparison.pdf', dpi=500, bbox_inches='tight')
    plt.close()


def plot_soc_comparison(scenario_name, results_dir, bat_data_kf, all_bat_data_dict):
    """Plot SOC comparison: True vs all models."""
    fig, ax = plt.subplots(figsize=(14, 6))

    step = bat_data_kf['Step'].astype(float)

    ax.plot(step, bat_data_kf['SOC_true'], label='True SOC', **STYLE_TRUE_SOC)

    for model_name in MODEL_NAMES:
        if model_name in all_bat_data_dict:
            bat_data = all_bat_data_dict[model_name]
            style = MODEL_STYLES[model_name]
            ax.plot(step, bat_data['Predicted_SOC'], label=f'KF + {model_name}', **style)

    ax.set_ylim(0, 100)
    ax.set_xlabel('Step [s]')
    ax.set_ylabel('SOC [%]')
    ax.set_title(f'SOC Prediction Comparison - {scenario_name}')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_SOC_comparison.pdf', dpi=500, bbox_inches='tight')
    plt.close()


def plot_single_model_soc(scenario_name, results_dir, bat_data_kf, bat_data_pred, model_type):
    """Plot SOC for single model: True vs KF vs Model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    step = bat_data_kf['Step'].astype(float)
    style = MODEL_STYLES[model_type]

    ax.plot(step, bat_data_kf['SOC_true'],         label='True SOC',           **STYLE_TRUE_SOC)
    ax.plot(step, bat_data_pred['Predicted_SOC'],   label=f'KF + {model_type}', **style)

    ax.set_ylim(0, 100)
    ax.set_xlabel('Step [s]')
    ax.set_ylabel('SOC [%]')
    ax.set_title(f'SOC Prediction - {model_type} - {scenario_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_SOC.pdf', dpi=500, bbox_inches='tight')
    plt.close()


def plot_residual_analysis(scenario_name, results_dir, bat_data_kf, bat_data_pred, model_type):
    """
    Lorenzo Giannuzzo: Advanced residual analysis visualization.

    Creates a professional multi-panel plot with:
    - Top: Residuals over time with rolling mean and confidence bands
    - Bottom-left: Histogram with normal distribution overlay
    - Bottom-right: Residuals vs Predicted SOC (heteroscedasticity check)
    """
    # Lorenzo Giannuzzo: Calculate residuals
    y_true = bat_data_kf['SOC_true'].astype(float).values
    y_pred = bat_data_pred['Predicted_SOC'].astype(float).values
    step = bat_data_kf['Step'].astype(float).values

    # Lorenzo Giannuzzo: Handle NaN
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    step = step[valid_mask]

    residuals = y_true - y_pred

    # Lorenzo Giannuzzo: Statistics
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    skewness = stats.skew(residuals)
    kurtosis = stats.kurtosis(residuals)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    # Lorenzo Giannuzzo: Get model color
    model_color = MODEL_STYLES[model_type]['color']

    # Lorenzo Giannuzzo: Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    # ================================================================
    # Lorenzo Giannuzzo: Top panel: Residuals over time (spans both columns)
    # ================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Lorenzo Giannuzzo: Calculate rolling statistics for smoothing
    window_size = max(len(residuals) // 100, 50)
    residuals_series = pd.Series(residuals)
    rolling_mean = residuals_series.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = residuals_series.rolling(window=window_size, center=True, min_periods=1).std()

    # Lorenzo Giannuzzo: Plot residuals as scatter (subsampled for performance)
    subsample = max(1, len(residuals) // 2000)
    ax1.scatter(step[::subsample], residuals[::subsample],
                alpha=0.3, s=8, color=model_color, label='Residuals', zorder=1)

    # Lorenzo Giannuzzo: Plot rolling mean
    ax1.plot(step, rolling_mean, color='#D32F2F', linewidth=2,
             label=f'Rolling Mean (w={window_size})', zorder=3)

    # Lorenzo Giannuzzo: Plot confidence bands (±2σ)
    ax1.fill_between(step, rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std,
                     alpha=0.2, color=model_color, label='±2σ band', zorder=2)

    # Lorenzo Giannuzzo: Zero line
    ax1.axhline(y=0, color='#1A1A1A', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)

    # Lorenzo Giannuzzo: Horizontal bands for reference
    ax1.axhline(y=1, color='#FFA726', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=-1, color='#FFA726', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=2, color='#EF5350', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=-2, color='#EF5350', linestyle=':', linewidth=1, alpha=0.5)

    ax1.set_xlabel('Step [s]', fontsize=11)
    ax1.set_ylabel('Residual (True - Predicted) [%]', fontsize=11)
    ax1.set_title(f'Residual Analysis Over Time - {model_type} - {scenario_name}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Lorenzo Giannuzzo: Set symmetric y-limits
    max_abs = max(abs(residuals.min()), abs(residuals.max())) * 1.1
    ax1.set_ylim(-max_abs, max_abs)

    # ================================================================
    # Bottom-left: Histogram with normal distribution overlay
    # ================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    # Lorenzo Giannuzzo: Histogram
    n_bins = min(100, len(residuals) // 50)
    n, bins, patches = ax2.hist(residuals, bins=n_bins, density=True,
                                 alpha=0.7, color=model_color, edgecolor='white', linewidth=0.5)

    # Lorenzo Giannuzzo: Fit and plot normal distribution
    x_norm = np.linspace(residuals.min(), residuals.max(), 200)
    y_norm = stats.norm.pdf(x_norm, mean_res, std_res)
    ax2.plot(x_norm, y_norm, 'r-', linewidth=2.5, label=f'Normal fit\n(μ={mean_res:.3f}, σ={std_res:.3f})')

    # Lorenzo Giannuzzo: KDE for actual distribution
    try:
        kde = stats.gaussian_kde(residuals)
        ax2.plot(x_norm, kde(x_norm), '--', color='#1A1A1A', linewidth=2, label='KDE')
    except:
        pass

    # Lorenzo Giannuzzo: Vertical lines for mean and ±σ
    ax2.axvline(x=mean_res, color='#D32F2F', linestyle='-', linewidth=2, alpha=0.8)
    ax2.axvline(x=mean_res + std_res, color='#FFA726', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvline(x=mean_res - std_res, color='#FFA726', linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Residual [%]', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ================================================================
    # Bottom-right: Residuals vs Predicted (heteroscedasticity check)
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    # Lorenzo Giannuzzo: Scatter plot
    ax3.scatter(y_pred[::subsample], residuals[::subsample],
                alpha=0.4, s=12, color=model_color, zorder=1)

    # Lorenzo Giannuzzo: LOWESS smoothing for trend
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        # Subsample more aggressively for LOWESS
        lowess_subsample = max(1, len(y_pred) // 500)
        smoothed = lowess(residuals[::lowess_subsample], y_pred[::lowess_subsample],
                         frac=0.3, return_sorted=True)
        ax3.plot(smoothed[:, 0], smoothed[:, 1], color='#D32F2F', linewidth=2.5,
                label='LOWESS trend', zorder=3)
    except ImportError:
        # Fallback: simple moving average binned by predicted value
        bins_pred = np.linspace(y_pred.min(), y_pred.max(), 20)
        bin_indices = np.digitize(y_pred, bins_pred)
        bin_means = [residuals[bin_indices == i].mean() if np.sum(bin_indices == i) > 0 else np.nan
                     for i in range(1, len(bins_pred))]
        bin_centers = (bins_pred[:-1] + bins_pred[1:]) / 2
        valid_bins = ~np.isnan(bin_means)
        ax3.plot(bin_centers[valid_bins], np.array(bin_means)[valid_bins],
                color='#D32F2F', linewidth=2.5, marker='o', markersize=4,
                label='Binned mean', zorder=3)

    # Lorenzo Giannuzzo: Zero line
    ax3.axhline(y=0, color='#1A1A1A', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

    # Lorenzo Giannuzzo: Reference bands
    ax3.axhline(y=1, color='#FFA726', linestyle=':', linewidth=1, alpha=0.5)
    ax3.axhline(y=-1, color='#FFA726', linestyle=':', linewidth=1, alpha=0.5)

    ax3.set_xlabel('Predicted SOC [%]', fontsize=11)
    ax3.set_ylabel('Residual [%]', fontsize=11)
    ax3.set_title('Residuals vs Predicted (Heteroscedasticity Check)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-max_abs, max_abs)

    # ================================================================
    # Statistics box
    # ================================================================
    stats_text = (
        f"Statistics:\n"
        f"─────────────\n"
        f"RMSE:      {rmse:.4f}%\n"
        f"MAE:       {mae:.4f}%\n"
        f"Mean:      {mean_res:.4f}%\n"
        f"Std Dev:   {std_res:.4f}%\n"
        f"Skewness:  {skewness:.3f}\n"
        f"Kurtosis:  {kurtosis:.3f}\n"
        f"Min:       {residuals.min():.4f}%\n"
        f"Max:       {residuals.max():.4f}%\n"
        f"N samples: {len(residuals):,}"
    )

    # Lorenzo Giannuzzo: Add text box
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace', bbox=props)

    plt.suptitle(f'Residual Analysis - {model_type} - {scenario_name}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_residuals.pdf', dpi=500, bbox_inches='tight')
    plt.savefig(results_dir / f'{scenario_name}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_standardized_residuals(scenario_name, results_dir, bat_data_kf, bat_data_pred, model_type):
    """
    Lorenzo Giannuzzo: Standardized (normalized) residuals visualization.

    Residuals are divided by standard deviation (z-scores), making it easy to:
    - Use fixed thresholds (±1σ, ±2σ, ±3σ)
    - Identify outliers immediately
    - Compare across different scenarios/models

    Layout:
    - Top: Standardized residuals over time with fixed threshold bands
    - Bottom-left: Histogram with standard normal overlay
    - Bottom-right: Q-Q plot for normality check
    """
    # Lorenzo Giannuzzo: Calculate residuals
    y_true = bat_data_kf['SOC_true'].astype(float).values
    y_pred = bat_data_pred['Predicted_SOC'].astype(float).values
    step = bat_data_kf['Step'].astype(float).values

    # Lorenzo Giannuzzo: Handle NaN
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    step = step[valid_mask]

    residuals = y_true - y_pred

    # Lorenzo Giannuzzo: Standardize residuals (z-scores)
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    z_residuals = (residuals - mean_res) / std_res

    # Lorenzo Giannuzzo: Statistics on standardized residuals
    pct_within_1sigma = np.mean(np.abs(z_residuals) <= 1) * 100
    pct_within_2sigma = np.mean(np.abs(z_residuals) <= 2) * 100
    pct_within_3sigma = np.mean(np.abs(z_residuals) <= 3) * 100
    n_outliers_3sigma = np.sum(np.abs(z_residuals) > 3)

    # Lorenzo Giannuzzo: Get model color
    model_color = MODEL_STYLES[model_type]['color']

    # Lorenzo Giannuzzo: Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.25)

    # ================================================================
    # Top panel: Standardized residuals over time
    # ================================================================
    ax1 = fig.add_subplot(gs[0, :])

    # Lorenzo Giannuzzo: Subsample for performance
    subsample = max(1, len(z_residuals) // 3000)

    # Lorenzo Giannuzzo: Color points by magnitude
    colors = np.where(np.abs(z_residuals[::subsample]) > 3, '#D32F2F',  # Red for |z| > 3
              np.where(np.abs(z_residuals[::subsample]) > 2, '#FF9800',  # Orange for |z| > 2
              np.where(np.abs(z_residuals[::subsample]) > 1, '#FFC107',  # Yellow for |z| > 1
                       model_color)))  # Model color for normal

    ax1.scatter(step[::subsample], z_residuals[::subsample],
                c=colors, alpha=0.5, s=10, zorder=2)

    # Lorenzo Giannuzzo: Fixed threshold bands
    ax1.axhline(y=0, color='#1A1A1A', linestyle='-', linewidth=2, alpha=0.8, zorder=3)

    # Lorenzo Giannuzzo: ±1σ band (68.3% of data expected)
    ax1.axhspan(-1, 1, alpha=0.15, color='#4CAF50', label='±1σ (68.3%)', zorder=1)
    ax1.axhline(y=1, color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=-1, color='#4CAF50', linestyle='--', linewidth=1.5, alpha=0.7)

    # Lorenzo Giannuzzo: ±2σ band (95.4% of data expected)
    ax1.axhline(y=2, color='#FF9800', linestyle='--', linewidth=1.5, alpha=0.7, label='±2σ (95.4%)')
    ax1.axhline(y=-2, color='#FF9800', linestyle='--', linewidth=1.5, alpha=0.7)

    # Lorenzo Giannuzzo: ±3σ band (99.7% of data expected)
    ax1.axhline(y=3, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.7, label='±3σ (99.7%)')
    ax1.axhline(y=-3, color='#D32F2F', linestyle='--', linewidth=1.5, alpha=0.7)

    ax1.set_xlabel('Step [s]', fontsize=11)
    ax1.set_ylabel('Standardized Residual (z-score)', fontsize=11)
    ax1.set_title(f'Standardized Residuals Over Time - {model_type} - {scenario_name}',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Lorenzo Giannuzzo: Dynamic y-limits based on data
    y_max = min(max(np.abs(z_residuals).max() * 1.1, 4), 10)
    ax1.set_ylim(-y_max, y_max)

    # ================================================================
    # Bottom-left: Histogram with standard normal overlay
    # ================================================================
    ax2 = fig.add_subplot(gs[1, 0])

    #Lorenzo Giannuzzo:  Histogram
    n_bins = min(80, len(z_residuals) // 50)
    n, bins, patches = ax2.hist(z_residuals, bins=n_bins, density=True,
                                 alpha=0.7, color=model_color, edgecolor='white', linewidth=0.5)

    # Lorenzo Giannuzzo: Color histogram bars by z-value
    for patch, left_edge in zip(patches, bins[:-1]):
        if abs(left_edge) > 3 or abs(left_edge + (bins[1]-bins[0])) > 3:
            patch.set_facecolor('#D32F2F')
        elif abs(left_edge) > 2:
            patch.set_facecolor('#FF9800')
        elif abs(left_edge) > 1:
            patch.set_facecolor('#FFC107')

    # Lorenzo Giannuzzo: Standard normal distribution overlay
    x_norm = np.linspace(-4, 4, 200)
    y_norm = stats.norm.pdf(x_norm, 0, 1)
    ax2.plot(x_norm, y_norm, 'r-', linewidth=2.5, label='Standard Normal N(0,1)')

    # Lorenzo Giannuzzo: Actual KDE
    try:
        kde = stats.gaussian_kde(z_residuals)
        ax2.plot(x_norm, kde(x_norm), '--', color='#1A1A1A', linewidth=2, label='Actual KDE')
    except:
        pass

    # Lorenzo Giannuzzo: Vertical reference lines
    ax2.axvline(x=0, color='#1A1A1A', linestyle='-', linewidth=1.5, alpha=0.5)
    ax2.axvline(x=-1, color='#4CAF50', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=1, color='#4CAF50', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=-2, color='#FF9800', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x=2, color='#FF9800', linestyle='--', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Standardized Residual (z-score)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Standardized Residual Distribution', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-4, 4)

    # ================================================================
    # Bottom-right: Q-Q Plot for normality check
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 1])

    # Lorenzo Giannuzzo: Calculate theoretical quantiles
    n_points = len(z_residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, min(n_points, 1000)))

    # Lorenzo Giannuzzo: Sample residuals if too many
    if n_points > 1000:
        sample_indices = np.linspace(0, n_points-1, 1000, dtype=int)
        z_sample = np.sort(z_residuals)[sample_indices]
    else:
        z_sample = np.sort(z_residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, len(z_sample)))

    # Lorenzo Giannuzzo: Q-Q scatter
    ax3.scatter(theoretical_quantiles, z_sample, alpha=0.5, s=15, color=model_color, zorder=2)

    # Lorenzo Giannuzzo: Perfect normal line
    lims = [-4, 4]
    ax3.plot(lims, lims, 'r-', linewidth=2, label='Perfect Normal', zorder=3)

    # Lorenzo Giannuzzo: Confidence bands (approximate)
    ax3.fill_between(lims, [l - 0.5 for l in lims], [l + 0.5 for l in lims],
                     alpha=0.1, color='red', zorder=1)

    ax3.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax3.set_ylabel('Sample Quantiles', fontsize=11)
    ax3.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.set_aspect('equal')

    # ================================================================
    # Statistics box
    # ================================================================
    stats_text = (
        f"Standardized Residuals:\n"
        f"────────────────────\n"
        f"Mean:     {np.mean(z_residuals):.4f}\n"
        f"Std Dev:  {np.std(z_residuals):.4f}\n"
        f"Skewness: {stats.skew(z_residuals):.3f}\n"
        f"Kurtosis: {stats.kurtosis(z_residuals):.3f}\n"
        f"────────────────────\n"
        f"Within ±1σ: {pct_within_1sigma:.1f}%\n"
        f"  (expected: 68.3%)\n"
        f"Within ±2σ: {pct_within_2sigma:.1f}%\n"
        f"  (expected: 95.4%)\n"
        f"Within ±3σ: {pct_within_3sigma:.1f}%\n"
        f"  (expected: 99.7%)\n"
        f"────────────────────\n"
        f"Outliers (|z|>3): {n_outliers_3sigma:,}\n"
        f"N samples: {len(z_residuals):,}"
    )

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=8,
             verticalalignment='top', fontfamily='monospace', bbox=props)

    plt.suptitle(f'Standardized Residual Analysis - {model_type} - {scenario_name}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_standardized_residuals.pdf', dpi=500, bbox_inches='tight')
    plt.savefig(results_dir / f'{scenario_name}_standardized_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ensemble_dominant_model(scenario_name, results_dir, bat_data_kf, bat_data_ensemble, ensemble_model=None):
    """
    Lorenzo Giannuzzo: Plot Ensemble SOC with color indicating dominant base model at each timestep.

    This creates a visually striking plot where the Ensemble prediction line changes color
    based on which base model (KF, ANN, CNN, LSTM, CatBoost, TFT) has the highest weight
    at each timestep.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    step = bat_data_kf['Step'].astype(float).values
    soc_true = bat_data_kf['SOC_true'].astype(float).values
    soc_pred = bat_data_ensemble['Predicted_SOC'].astype(float).values

    # Lorenzo Giannuzzo: Check if weights are available
    has_weights = ('Ensemble_Weights' in bat_data_ensemble.columns and
                   bat_data_ensemble['Ensemble_Weights'].iloc[0] is not None)

    ax1 = axes[0]

    # Lorenzo Giannuzzo: Plot True SOC
    ax1.plot(step, soc_true, label='True SOC', **STYLE_TRUE_SOC, zorder=1)

    if has_weights:
        # Lorenzo Giannuzzo: Get weights - they come as list of lists from DataFrame.values.tolist()
        weights_list = bat_data_ensemble['Ensemble_Weights'].tolist()
        weights_array = np.array(weights_list)

        n_base_models = weights_array.shape[1]

        # Lorenzo Giannuzzo: Determine dominant model (highest weight) at each timestep
        dominant_indices = np.argmax(weights_array, axis=1)

        # Lorenzo Giannuzzo: Create colored line segments
        points = np.array([step, soc_pred]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Lorenzo Giannuzzo: Colors for base models (KF, ANN, CNN, LSTM, CatBoost, TFT)
        if n_base_models == 6:
            colors_map = [COLOR_KF, COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT]
            model_names_ordered = ['KF', 'ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']
        else:
            colors_map = [COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT]
            model_names_ordered = ['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']

        segment_colors = [colors_map[dominant_indices[i]] for i in range(len(segments))]

        # Lorenzo Giannuzzo: Create LineCollection with varying colors
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, colors=segment_colors, linewidths=2.5, alpha=0.9, zorder=2)
        ax1.add_collection(lc)

        # Lorenzo Giannuzzo: Create custom legend
        legend_elements = [Line2D([0], [0], color=COLOR_TRUE_SOC, linewidth=2, label='True SOC')]
        for i, model_name in enumerate(model_names_ordered):
            legend_elements.append(
                Line2D([0], [0], color=colors_map[i], linewidth=2.5,
                       label=f'Ensemble → {model_name} dominant')
            )
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

        # ====================================================================
        # Bottom subplot: Stacked area chart of weights over time
        # ====================================================================
        ax2 = axes[1]

        # Lorenzo Giannuzzo: Create stacked area plot
        if n_base_models == 6:
            ax2.stackplot(step,
                          weights_array[:, 0],  # KF
                          weights_array[:, 1],  # ANN
                          weights_array[:, 2],  # CNN
                          weights_array[:, 3],  # LSTM
                          weights_array[:, 4],  # CatBoost
                          weights_array[:, 5],  # TFT
                          labels=['KF', 'ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT'],
                          colors=[COLOR_KF, COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT],
                          alpha=0.8)
        else:
            ax2.stackplot(step,
                          weights_array[:, 0],  # ANN
                          weights_array[:, 1],  # CNN
                          weights_array[:, 2],  # LSTM
                          weights_array[:, 3],  # CatBoost
                          weights_array[:, 4],  # TFT
                          labels=['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT'],
                          colors=[COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT],
                          alpha=0.8)

        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Step [s]', fontsize=11)
        ax2.set_ylabel('Model Weight', fontsize=11)
        ax2.set_title('Dynamic Model Weights Over Time', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8, ncol=n_base_models)
        ax2.grid(True, alpha=0.3)

        # Lorenzo Giannuzzo: Calculate and display dominance statistics
        dominance_counts = np.bincount(dominant_indices, minlength=n_base_models)
        dominance_pct = dominance_counts / len(dominant_indices) * 100

        stats_text = "Dominance: " + " | ".join([f"{model_names_ordered[i]}: {dominance_pct[i]:.1f}%"
                                                  for i in range(n_base_models)])
        ax2.text(0.5, -0.25, stats_text, transform=ax2.transAxes, fontsize=9,
                 ha='center', va='top', style='italic')

    else:
        # Lorenzo Giannuzzo: No weights available - plot simple ensemble line
        ax1.plot(step, soc_pred, label='KF + Ensemble', **STYLE_ENSEMBLE, zorder=2)
        ax1.legend(loc='best', fontsize=10)

        # Lorenzo Giannuzzo: Hide bottom subplot if no weights
        axes[1].set_visible(False)

    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Step [s]', fontsize=11)
    ax1.set_ylabel('SOC [%]', fontsize=11)
    ax1.set_title(f'Ensemble SOC Prediction with Dominant Model Visualization - {scenario_name}',
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_Ensemble_dominant.pdf', dpi=500, bbox_inches='tight')
    plt.savefig(results_dir / f'{scenario_name}_Ensemble_dominant.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_ensemble_weights_heatmap(scenario_name, results_dir, bat_data_ensemble, step_interval=100):
    """
    Lorenzo Giannuzzo: Create a heatmap showing how ensemble weights evolve over time.
    """
    if ('Ensemble_Weights' not in bat_data_ensemble.columns or
        bat_data_ensemble['Ensemble_Weights'].iloc[0] is None):
        return

    weights_list = bat_data_ensemble['Ensemble_Weights'].tolist()

    if isinstance(weights_list[0], np.ndarray):
        weights_array = np.array(weights_list)
    else:
        weights_array = np.array([w if w is not None else np.zeros(5) for w in weights_list])

    # Lorenzo Giannuzzo: Subsample for readability
    n_samples = len(weights_array)
    indices = np.arange(0, n_samples, step_interval)
    weights_subsampled = weights_array[indices]

    fig, ax = plt.subplots(figsize=(14, 4))

    model_names = ['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']
    im = ax.imshow(weights_subsampled.T, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)

    ax.set_yticks(range(5))
    ax.set_yticklabels(model_names)
    ax.set_xlabel(f'Time Step (×{step_interval})')
    ax.set_ylabel('Base Model')
    ax.set_title(f'Ensemble Weights Heatmap - {scenario_name}', fontsize=12, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='Weight')

    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_Ensemble_weights_heatmap.pdf', dpi=500, bbox_inches='tight')
    plt.close()


def plot_hppc_results(scenario_name, results_dir, bat_data_kf):
    """Plot HPPC validation results: True vs KF only."""
    fig, ax = plt.subplots(figsize=(12, 6))

    step = bat_data_kf['Step'].astype(float)

    ax.plot(step, bat_data_kf['SOC_true'], label='True SOC', **STYLE_TRUE_SOC)
    ax.plot(step, bat_data_kf['SOC_Calc'], label='KF',       **STYLE_KF)

    ax.set_ylim(0, 100)
    ax.set_xlabel('Step [s]')
    ax.set_ylabel('SOC [%]')
    ax.set_title(f'SOC Prediction (KF only) - {scenario_name}')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / f'{scenario_name}_SOC.pdf', dpi=500, bbox_inches='tight')
    plt.close()



# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    total_start = time.time()
    log_file = "battery_data_log.txt"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    # Lorenzo Giannuzzo: Settings
    SEQUENCE_LENGTH = 10
    PERCENTAGES = [20, 40, 80, 100]
    ENSEMBLE_TYPE = 'selector'  # Options: 'dynamic', 'selector', 'stacking', 'neural'

    # Lorenzo Giannuzzo: Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    results_dir = script_dir / 'results'

    # Lorenzo Giannuzzo: Create subfolders
    results_dir.mkdir(exist_ok=True)
    model_results_dirs = {}
    for model_name in MODEL_NAMES:
        model_results_dirs[model_name] = results_dir / f'{model_name}_test_results'
        model_results_dirs[model_name].mkdir(exist_ok=True)

    hppc_results_dir = results_dir / 'HPPC_validation_results'
    hppc_results_dir.mkdir(exist_ok=True)
    results_sheet_dir = results_dir / 'results_sheet'
    results_sheet_dir.mkdir(exist_ok=True)

    # Lorenzo Giannuzzo: Model paths
    model_paths = {
        'ANN': {
            'model': script_dir / 'nn_training/models/my_model.keras',
            'scaler': script_dir / 'nn_training/models/scaler.joblib',
        },
        'CNN': {
            'model': script_dir / 'nn_training/models/cnn_model.keras',
            'scaler': script_dir / 'nn_training/models/cnn_scaler.joblib',
        },
        'LSTM': {
            'model': script_dir / 'nn_training/models/lstm_model.keras',
            'scaler': script_dir / 'nn_training/models/lstm_scaler.joblib',
        },
        'CatBoost': {
            'model': script_dir / 'nn_training/models/catboost_model.cbm',
            'scaler': script_dir / 'nn_training/models/catboost_scaler.joblib',
            'features': script_dir / 'nn_training/models/catboost_feature_columns.joblib',
        },
        'TFT': {
            'model': script_dir / 'nn_training/models/tft_model.ckpt',
            'scaler': script_dir / 'nn_training/models/tft_scaler.joblib',
            'config': script_dir / 'nn_training/models/tft_config.joblib',
        },
    }

    # Lorenzo Giannuzzo: Ensemble model path depends on ENSEMBLE_TYPE
    ensemble_model_names = {
        'dynamic': 'ensemble_dynamic_model.keras',
        'selector': 'ensemble_selector_model.keras',
        'neural': 'ensemble_neural_model.keras',
        'stacking': 'ensemble_stacking_model.json',
    }
    model_paths['Ensemble'] = {
        'model': script_dir / 'nn_training/models' / ensemble_model_names[ENSEMBLE_TYPE],
        'scaler': None,  # Ensemble uses base model scalers
    }

    # Lorenzo Giannuzzo: Check which models exist
    available_models = []
    for model_name in MODEL_NAMES:
        model_path = model_paths[model_name]['model']
        if model_path.exists():
            available_models.append(model_name)
        else:
            print(f"WARNING: {model_name} model not found at {model_path} - skipping")

    if not available_models:
        print("ERROR: No models found!")
        sys.exit(1)

    # Lorenzo Giannuzzo: Get all xlsx files in data folder (excluding Cha_Dis_OCV files)
    data_files = [f for f in data_dir.glob('*.xlsx') if 'Cha_Dis_OCV' not in f.name]

    # Lorenzo Giannuzzo: Store results
    all_results = []
    comparison_results = []
    hppc_results = []

    # Lorenzo Giannuzzo: Accumulators for results_sheet Excel export
    excel_sheet_data = {m: {} for m in MODEL_NAMES}
    excel_sheet_metrics = {m: [] for m in MODEL_NAMES}

    # Lorenzo Giannuzzo: Timing tracking for each model
    execution_times = []  # List of dicts: {File, ANN_time, CNN_time, ...}

    total_files = len(data_files)

    print(f"\n{'='*70}")
    print(f"  SOC PREDICTION - MODEL COMPARISON")
    print(f"  Models: {', '.join(available_models)}")
    print(f"  Ensemble type: {ENSEMBLE_TYPE}")
    print(f"  Found {total_files} files to process")
    print(f"  Progressive evaluation at: {PERCENTAGES}% of data")
    print(f"{'='*70}\n")

    for idx, file_path in enumerate(data_files):
        scenario_name = file_path.stem
        file_start = time.time()

        is_hppc = 'HPPC' in file_path.name

        print(f"[{idx+1}/{total_files}] {scenario_name}")

        try:
            # Lorenzo Giannuzzo: Phase 1: Load data
            print_file_progress('load', scenario_name)
            bat_data_df = pd.read_excel(file_path, skiprows=2).iloc[1:]
            bat_data_df = standardize_columns(bat_data_df)

            if 'SOC_true' not in bat_data_df.columns:
                print(f'\r  [X] Skipped: Missing SOC_true column.')
                all_results.append({
                    'File': scenario_name,
                    'Status': 'error: Missing SOC_true column'
                })
                continue

            soc_initial = float(bat_data_df['SOC_true'].iloc[0])
            if np.isnan(soc_initial):
                soc_initial = 50.0
                print(f'\n    Warning: SOC_initial is NaN, using default 50%')

            # Lorenzo Giannuzzo:  Phase 2: Kalman Filter
            print_file_progress('kalman', scenario_name)
            bat_data_kf = run_kalman_filter(bat_data_df.copy(), soc_initial, scenario_name)

            if is_hppc:
                # Lorenzo Giannuzzo:  For HPPC files: only KF, no neural network comparison
                y_true = bat_data_kf['SOC_true'].values.astype(float)
                y_pred = bat_data_kf['SOC_Calc'].values.astype(float)

                rmse = root_mean_squared_error(y_true, y_pred)
                mean_ae = mean_absolute_error(y_true, y_pred)
                max_ae = max_absolute_error(y_true, y_pred)

                print_file_progress('save', scenario_name)
                plot_hppc_results(scenario_name, hppc_results_dir, bat_data_kf)

                print(f'\r  [OK] HPPC file - KF only | RMSE={rmse:.4f} | MeanAE={mean_ae:.4f} | MaxAE={max_ae:.4f} | Time: {time.time() - file_start:.1f}s')

                hppc_results.append({
                    'File': scenario_name,
                    'Method': 'KF only',
                    'SOC_initial': soc_initial,
                    'RMSE': rmse,
                    'MeanAE': mean_ae,
                    'MaxAE': max_ae,
                    'Status': 'success'
                })
                continue

            # Lorenzo Giannuzzo: Run predictions for all available models
            all_bat_data = {}
            all_metrics = {}
            model_times = {}  # Lorenzo Giannuzzo: Track execution time for each model
            y_true = bat_data_kf['SOC_true'].values.astype(float)

            for model_name in available_models:
                print_file_progress(model_name.lower(), scenario_name, model_name)

                model_start_time = time.time()  # Lorenzo Giannuzzo: Start timing

                paths = model_paths[model_name]

                # Lorenzo Giannuzzo: Special handling for Ensemble
                if model_name == 'Ensemble':
                    bat_data_pred = run_model_prediction(
                        bat_data_kf, model_name,
                        str(paths['model']),
                        None,  # Lorenzo Giannuzzo: Ensemble doesn't use its own scaler
                        sequence_length=SEQUENCE_LENGTH,
                        ensemble_type=ENSEMBLE_TYPE,
                    )
                else:
                    bat_data_pred = run_model_prediction(
                        bat_data_kf, model_name,
                        str(paths['model']),
                        str(paths['scaler']),
                        sequence_length=SEQUENCE_LENGTH,
                        file_path_feature_columns=str(paths.get('features')) if paths.get('features') and paths['features'].exists() else None,
                        file_path_config=str(paths.get('config')) if paths.get('config') and paths['config'].exists() else None,
                    )

                model_times[model_name] = time.time() - model_start_time  # Lorenzo Giannuzzo: End timing

                all_bat_data[model_name] = bat_data_pred

                y_pred = bat_data_pred['Predicted_SOC'].values.astype(float)
                all_metrics[model_name] = calculate_progressive_metrics(y_true, y_pred, PERCENTAGES)

            # Lorenzo Giannuzzo: Phase: Save results and plots
            print_file_progress('save', scenario_name)

            # Lorenzo Giannuzzo: Output columns
            output_columns = ['Step', 'Voltage', 'Current_inv', 'Temp', 'SOC_true', 'SOC_Calc', 'Predicted_SOC']

            for model_name in available_models:
                bat_data_pred = all_bat_data[model_name]
                metrics = all_metrics[model_name]

                export_cols = [c for c in output_columns if c in bat_data_pred.columns]
                excel_sheet_data[model_name][scenario_name] = bat_data_pred[export_cols].copy()

                excel_sheet_metrics[model_name].append({
                    'File': scenario_name,
                    'RMSE': metrics['RMSE_100pct'],
                    'MeanAE': metrics['MeanAE_100pct'],
                    'MaxAE': metrics['MaxAE_100pct'],
                })

                # Lorenzo Giannuzzo: Plot individual model results
                plot_single_model_soc(scenario_name, model_results_dirs[model_name],
                                     bat_data_kf, bat_data_pred, model_name)

                # Lorenzo Giannuzzo: Plot residual analysis
                plot_residual_analysis(scenario_name, model_results_dirs[model_name],
                                      bat_data_kf, bat_data_pred, model_name)

                # Lorenzo Giannuzzo: Plot standardized residuals (z-scores)
                plot_standardized_residuals(scenario_name, model_results_dirs[model_name],
                                           bat_data_kf, bat_data_pred, model_name)

                # Special plots for Ensemble
                if model_name == 'Ensemble':
                    plot_ensemble_dominant_model(scenario_name, model_results_dirs['Ensemble'],
                                                bat_data_kf, bat_data_pred)
                    plot_ensemble_weights_heatmap(scenario_name, model_results_dirs['Ensemble'],
                                                 bat_data_pred)

            # Lorenzo Giannuzzo: Plot comparison
            plot_comparison(scenario_name, results_dir, all_metrics, PERCENTAGES)
            plot_soc_comparison(scenario_name, results_dir, bat_data_kf, all_bat_data)

            file_elapsed = time.time() - file_start

            # Lorenzo Giannuzzo: Print metrics for all models
            print(f'\r  [OK] ', end='')
            for i, model_name in enumerate(available_models):
                metrics = all_metrics[model_name]
                suffix = f' | Time: {file_elapsed:.1f}s' if i == len(available_models) - 1 else ''
                if i == 0:
                    print(f'{model_name}: RMSE={metrics["RMSE_100pct"]:.4f} | MeanAE={metrics["MeanAE_100pct"]:.4f} | MaxAE={metrics["MaxAE_100pct"]:.4f}{suffix}')
                else:
                    print(f'       {model_name}: RMSE={metrics["RMSE_100pct"]:.4f} | MeanAE={metrics["MeanAE_100pct"]:.4f} | MaxAE={metrics["MaxAE_100pct"]:.4f}{suffix}')

            # Lorenzo Giannuzzo: Store results
            result_row = {
                'File': scenario_name,
                'SOC_initial': soc_initial,
            }

            for model_name in available_models:
                metrics = all_metrics[model_name]
                for pct in PERCENTAGES:
                    result_row[f'{model_name}_RMSE_{pct}pct'] = metrics[f'RMSE_{pct}pct']
                    result_row[f'{model_name}_MeanAE_{pct}pct'] = metrics[f'MeanAE_{pct}pct']
                    result_row[f'{model_name}_MaxAE_{pct}pct'] = metrics[f'MaxAE_{pct}pct']

            result_row['Elapsed_time_s'] = round(file_elapsed, 2)
            result_row['Status'] = 'success'

            all_results.append(result_row)

            # Lorenzo Giannuzzo: Determine winners
            rmse_values = {m: all_metrics[m]['RMSE_100pct'] for m in available_models}
            meanae_values = {m: all_metrics[m]['MeanAE_100pct'] for m in available_models}
            maxae_values = {m: all_metrics[m]['MaxAE_100pct'] for m in available_models}

            comparison_results.append({
                'File': scenario_name,
                **{f'{m}_RMSE': rmse_values[m] for m in available_models},
                **{f'{m}_MeanAE': meanae_values[m] for m in available_models},
                **{f'{m}_MaxAE': maxae_values[m] for m in available_models},
                'RMSE_Winner': min(rmse_values, key=rmse_values.get),
                'MeanAE_Winner': min(meanae_values, key=meanae_values.get),
                'MaxAE_Winner': min(maxae_values, key=maxae_values.get),
            })

            # Lorenzo Giannuzzo: Store execution times
            time_row = {'File': scenario_name}
            for model_name in available_models:
                time_row[f'{model_name}_time_s'] = round(model_times[model_name], 4)
            time_row['Total_time_s'] = round(file_elapsed, 4)
            execution_times.append(time_row)

        except Exception as e:
            import traceback
            file_elapsed = time.time() - file_start
            print(f'\r  [X] Error: {e}')
            traceback.print_exc()

            all_results.append({
                'File': scenario_name,
                'Status': f'error: {e}'
            })

    # Lorenzo Giannuzzo: Summary
    total_elapsed = time.time() - total_start
    successful = [r for r in comparison_results]

    print(f"\n{'='*70}")
    print(f"  SUMMARY - MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"  Files processed: {total_files}")
    print(f"  Successful comparisons: {len(successful)}")
    print(f"  HPPC files: {len(hppc_results)}")
    print(f"  Total time: {total_elapsed:.2f}s")

    if successful:
        # Lorenzo Giannuzzo: Calculate averages dynamically
        avg = {}
        for model in available_models:
            for metric in ['RMSE', 'MeanAE', 'MaxAE']:
                avg[f'{model}_{metric}'] = np.mean([r[f'{model}_{metric}'] for r in successful])

        # Lorenzo Giannuzzo: Count wins
        wins = {}
        for model in available_models:
            for metric in ['RMSE', 'MeanAE', 'MaxAE']:
                wins[f'{model}_{metric}'] = sum(1 for r in successful if r[f'{metric}_Winner'] == model)

        print(f"\n  Average Metrics:")
        header = f"  {'Metric':<15}" + ''.join(f'{m:>12}' for m in available_models) + f"{'Winner':>10}"
        print(header)
        print(f"  {'-'*(15 + 12*len(available_models) + 10)}")

        for metric in ['RMSE', 'MeanAE', 'MaxAE']:
            metric_avgs = {m: avg[f'{m}_{metric}'] for m in available_models}
            values_str = ''.join(f'{avg[f"{m}_{metric}"]:>12.4f}' for m in available_models)
            print(f"  {metric:<15} {values_str} {min(metric_avgs, key=metric_avgs.get):>10}")

        print(f"\n  Wins by File:")
        for metric in ['RMSE', 'MeanAE']:
            wins_str = ', '.join(f'{m}={wins[f"{m}_{metric}"]}' for m in available_models)
            print(f"    {metric}: {wins_str}")

    print(f"{'='*70}")

    # Lorenzo Giannuzzo: Save to Excel
    excel_path = results_dir / 'evaluation_results_comparison.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        results_df = pd.DataFrame(all_results)
        results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)

        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df.to_excel(writer, sheet_name='Comparison_Summary', index=False)

        if successful:
            progressive_data = []
            for pct in PERCENTAGES:
                row = {'Data_Percentage': f'{pct}%'}
                for model in available_models:
                    for metric in ['RMSE', 'MeanAE', 'MaxAE']:
                        row[f'{model}_Avg_{metric}'] = np.mean(
                            [r.get(f'{model}_{metric}_{pct}pct', np.nan)
                             for r in all_results if r.get('Status') == 'success'])
                progressive_data.append(row)
            progressive_df = pd.DataFrame(progressive_data)
            progressive_df.to_excel(writer, sheet_name='Progressive_Metrics', index=False)

        if successful:
            summary_rows = [
                ('Total files processed', total_files),
                ('Successful comparisons', len(successful)),
                ('HPPC files', len(hppc_results)),
                ('Failed', total_files - len(successful) - len(hppc_results)),
            ]
            for model in available_models:
                for metric in ['RMSE', 'MeanAE', 'MaxAE']:
                    summary_rows.append((f'{model} Avg {metric}', round(avg[f'{model}_{metric}'], 4)))

            for metric in ['RMSE', 'MeanAE']:
                metric_avgs = {m: avg[f'{m}_{metric}'] for m in available_models}
                summary_rows.append((f'{metric} Winner (overall)', min(metric_avgs, key=metric_avgs.get)))

            for metric in ['RMSE', 'MeanAE']:
                for model in available_models:
                    summary_rows.append((f'{model} wins ({metric})', wins[f'{model}_{metric}']))

            summary_rows.append(('Total time (s)', round(total_elapsed, 2)))

            summary_df = pd.DataFrame(summary_rows, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\n  Results saved to: {excel_path}")

    # Lorenzo Giannuzzo: Save HPPC results
    if hppc_results:
        hppc_excel_path = hppc_results_dir / 'HPPC_validation_results.xlsx'
        hppc_df = pd.DataFrame(hppc_results)
        hppc_df.to_excel(hppc_excel_path, index=False)
        print(f"  HPPC results saved to: {hppc_excel_path}")

    # Lorenzo Giannuzzo: Save per-model prediction sheets
    print(f"\n  Saving prediction sheets to: {results_sheet_dir}")
    for model_name in available_models:
        if not excel_sheet_data[model_name]:
            continue

        model_excel_path = results_sheet_dir / f'{model_name}.xlsx'
        with pd.ExcelWriter(model_excel_path, engine='openpyxl') as writer:
            for scenario_name, df_pred in excel_sheet_data[model_name].items():
                sheet_name = scenario_name[:31]
                df_pred.to_excel(writer, sheet_name=sheet_name, index=False)

            if excel_sheet_metrics[model_name]:
                metrics_df = pd.DataFrame(excel_sheet_metrics[model_name])
                avg_row = {
                    'File': 'AVERAGE',
                    'RMSE': metrics_df['RMSE'].mean(),
                    'MeanAE': metrics_df['MeanAE'].mean(),
                    'MaxAE': metrics_df['MaxAE'].mean(),
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_row])], ignore_index=True)
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

        print(f"    - {model_name}.xlsx ({len(excel_sheet_data[model_name])} scenarios)")

    # ========================================================================
    # Lorenzo Giannuzzo: OVERALL COMPARISON PLOT (all models including Ensemble)
    # ========================================================================
    if successful:
        fig, axes = plt.subplots(1, 3, figsize=(22, 5))

        files = [r['File'][:12] + '...' if len(r['File']) > 12 else r['File']
                 for r in comparison_results]

        x = np.arange(len(files))
        n_models = len(available_models)
        width = 0.8 / n_models
        offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

        for ax_idx, (metric_key, ylabel, title) in enumerate([
            ('RMSE', 'RMSE [%]', 'RMSE Comparison'),
            ('MeanAE', 'MeanAE [%]', 'Mean Absolute Error Comparison'),
            ('MaxAE', 'MaxAE [%]', 'Max Absolute Error Comparison'),
        ]):
            ax = axes[ax_idx]
            for i, (model, offset) in enumerate(zip(available_models, offsets)):
                values = [r[f'{model}_{metric_key}'] for r in comparison_results]
                color_idx = MODEL_NAMES.index(model) if model in MODEL_NAMES else i
                ax.bar(x + offset, values, width, label=model,
                       color=BAR_COLORS[color_idx], edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Dataset')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(files, rotation=45, ha='right', fontsize=6)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Overall Model Comparison ({", ".join(available_models)})',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'overall_comparison.pdf', dpi=500, bbox_inches='tight')
        plt.savefig(results_dir / 'overall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Overall comparison plot saved to: {results_dir / 'overall_comparison.pdf'}")

    # ========================================================================
    # Lorenzo Giannuzzo: BASE MODELS COMPARISON (without Ensemble)
    # ========================================================================
    base_models_only = [m for m in available_models if m != 'Ensemble']

    if successful and len(base_models_only) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(22, 5))

        files = [r['File'][:12] + '...' if len(r['File']) > 12 else r['File']
                 for r in comparison_results]

        x = np.arange(len(files))
        n_models = len(base_models_only)
        width = 0.8 / n_models
        offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

        for ax_idx, (metric_key, ylabel, title) in enumerate([
            ('RMSE', 'RMSE [%]', 'RMSE Comparison (Base Models)'),
            ('MeanAE', 'MeanAE [%]', 'Mean Absolute Error (Base Models)'),
            ('MaxAE', 'MaxAE [%]', 'Max Absolute Error (Base Models)'),
        ]):
            ax = axes[ax_idx]
            for i, (model, offset) in enumerate(zip(base_models_only, offsets)):
                values = [r[f'{model}_{metric_key}'] for r in comparison_results]
                color_idx = MODEL_NAMES.index(model) if model in MODEL_NAMES else i
                ax.bar(x + offset, values, width, label=model,
                       color=BAR_COLORS[color_idx], edgecolor='white', linewidth=0.5)
            ax.set_xlabel('Dataset')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(files, rotation=45, ha='right', fontsize=6)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Base Models Comparison (without Ensemble): {", ".join(base_models_only)}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'base_models_comparison.pdf', dpi=500, bbox_inches='tight')
        plt.savefig(results_dir / 'base_models_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Base models comparison plot saved to: {results_dir / 'base_models_comparison.pdf'}")

        # Lorenzo Giannuzzo: Print base models ranking
        print(f"\n  Base Models Ranking (without Ensemble):")
        for metric in ['RMSE', 'MeanAE', 'MaxAE']:
            metric_avgs = {m: np.mean([r[f'{m}_{metric}'] for r in comparison_results])
                          for m in base_models_only}
            sorted_models = sorted(metric_avgs.items(), key=lambda x: x[1])
            ranking_str = ' > '.join([f"{m}({v:.4f})" for m, v in sorted_models])
            print(f"    {metric}: {ranking_str}")

    # ========================================================================
    # EXECUTION TIME ANALYSIS
    # ========================================================================
    if execution_times:
        # Lorenzo Giannuzzo: Create execution time plot
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        files = [r['File'][:15] + '...' if len(r['File']) > 15 else r['File']
                 for r in execution_times]

        x = np.arange(len(files))
        n_models = len(available_models)
        width = 0.8 / n_models
        offsets = [(i - (n_models - 1) / 2) * width for i in range(n_models)]

        # Lorenzo Giannuzzo: Plot 1: Execution time per scenario (stacked or grouped)
        ax1 = axes[0]
        for i, (model, offset) in enumerate(zip(available_models, offsets)):
            values = [r.get(f'{model}_time_s', 0) for r in execution_times]
            color_idx = MODEL_NAMES.index(model) if model in MODEL_NAMES else i
            ax1.bar(x + offset, values, width, label=model,
                   color=BAR_COLORS[color_idx], edgecolor='white', linewidth=0.5)

        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Execution Time [s]')
        ax1.set_title('Model Execution Time per Dataset')
        ax1.set_xticks(x)
        ax1.set_xticklabels(files, rotation=45, ha='right', fontsize=7)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')

        # Lorenzo Giannuzzo: Plot 2: Average execution time per model
        ax2 = axes[1]
        avg_times = []
        for model in available_models:
            times = [r.get(f'{model}_time_s', 0) for r in execution_times]
            avg_times.append(np.mean(times))

        color_indices = [MODEL_NAMES.index(m) if m in MODEL_NAMES else i for i, m in enumerate(available_models)]
        bars = ax2.bar(available_models, avg_times,
                       color=[BAR_COLORS[idx] for idx in color_indices],
                       edgecolor='white', linewidth=0.5)

        # Lorenzo Giannuzzo: Add value labels on bars
        for bar, val in zip(bars, avg_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}s', ha='center', va='bottom', fontsize=9)

        ax2.set_xlabel('Model')
        ax2.set_ylabel('Average Execution Time [s]')
        ax2.set_title('Average Model Execution Time')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Model Execution Time Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(results_dir / 'execution_time_comparison.pdf', dpi=500, bbox_inches='tight')
        plt.savefig(results_dir / 'execution_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Execution time plot saved to: {results_dir / 'execution_time_comparison.pdf'}")

        # Lorenzo Giannuzzo: Save execution times to Excel
        exec_time_excel_path = results_dir / 'execution_times.xlsx'
        with pd.ExcelWriter(exec_time_excel_path, engine='openpyxl') as writer:
            # Lorenzo Giannuzzo: Detailed times per scenario
            exec_df = pd.DataFrame(execution_times)
            exec_df.to_excel(writer, sheet_name='Detailed_Times', index=False)

            # Lorenzo Giannuzzo: Summary statistics
            summary_data = []
            for model in available_models:
                times = [r.get(f'{model}_time_s', 0) for r in execution_times]
                summary_data.append({
                    'Model': model,
                    'Mean_time_s': round(np.mean(times), 4),
                    'Std_time_s': round(np.std(times), 4),
                    'Min_time_s': round(np.min(times), 4),
                    'Max_time_s': round(np.max(times), 4),
                    'Total_time_s': round(np.sum(times), 4),
                })

            # Lorenzo Giannuzzo: Add total row
            total_times = [r.get('Total_time_s', 0) for r in execution_times]
            summary_data.append({
                'Model': 'TOTAL (all models)',
                'Mean_time_s': round(np.mean(total_times), 4),
                'Std_time_s': round(np.std(total_times), 4),
                'Min_time_s': round(np.min(total_times), 4),
                'Max_time_s': round(np.max(total_times), 4),
                'Total_time_s': round(np.sum(total_times), 4),
            })

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Lorenzo Giannuzzo: Ranking by speed (fastest first)
            ranking_data = []
            for model in available_models:
                times = [r.get(f'{model}_time_s', 0) for r in execution_times]
                ranking_data.append({
                    'Model': model,
                    'Avg_time_s': np.mean(times),
                })
            ranking_df = pd.DataFrame(ranking_data).sort_values('Avg_time_s')
            ranking_df['Rank'] = range(1, len(ranking_df) + 1)
            ranking_df = ranking_df[['Rank', 'Model', 'Avg_time_s']]
            ranking_df.to_excel(writer, sheet_name='Speed_Ranking', index=False)

        print(f"  Execution times saved to: {exec_time_excel_path}")

    print(f"\n  Output folders:")
    for model_name in available_models:
        print(f"    - {model_name} results: {model_results_dirs[model_name]}")
    print(f"    - HPPC results: {hppc_results_dir}")
    print(f"    - Prediction sheets: {results_sheet_dir}")
    print(f"    - Comparison results: {results_dir}")