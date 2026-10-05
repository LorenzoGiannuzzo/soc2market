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
Comparison script: KF only vs KF+Denoise vs KF+Denoise+ANN
Uses the existing modules without code duplication.
"""
import numpy as np
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Add module path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / 'model_execution_modules'))

from KalmanFilter_module_all_data import KalmanFilter
from ANN_prediction_module_all_data import ANN_BatteryAnalysis


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
        'Temp1': 'Temp',
    }
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]
    return df


def plot_current_comparison(df, save_dir):
    """Debug plot: compare raw vs filtered current."""

    I_raw = df['Current_inv'].values.astype(float)
    I_filtered = gaussian_filter1d(I_raw, sigma=10)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Lorenzo Giannuzzo: Full signal
    ax.plot(I_raw, 'b-', label='Raw Current', alpha=0.7, linewidth=0.8)
    ax.plot(I_filtered, 'r-', label='Filtered Current (sigma=50)', alpha=0.7, linewidth=0.8)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Current (A)')
    ax.set_title('Current Signal: Raw vs Gaussian Filtered', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'current_denoise.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Lorenzo Giannuzzo: Print statistics
    diff = np.abs(I_raw - I_filtered)
    print("\n" + "="*60)
    print("DEBUG: Current Signal Comparison")
    print("="*60)
    print(f"  Max difference: {np.max(diff):.6f} A")
    print(f"  Mean difference: {np.mean(diff):.6f} A")
    print(f"  Std Raw: {np.std(I_raw):.4f} A")
    print(f"  Std Filtered: {np.std(I_filtered):.4f} A")
    print("="*60)


def run_comparison(file_path):
    """Run all 3 methods and return metrics."""

    # Lorenzo Giannuzzo: Load data
    df = pd.read_excel(file_path, skiprows=2).iloc[1:]
    df = standardize_columns(df)

    if 'Temp' not in df.columns:
        df['Temp'] = 25.0

    soc_initial = float(df['SOC_true'].iloc[0])

    # Lorenzo Giannuzzo: KF parameters (optimized values)
    Q = np.array([[4.034329e-07, 0, 0], [0, 8.628196e-05, 0], [0, 0, 9.095089e-05]])
    R = 3.743003e-03
    P = np.array([[3.393996e-03, 0, 0], [0, 6.662388e-04, 0], [0, 0, 6.946038e-03]])

    results = {}

    # 1. KF only (no denoise) - pass a copy of df
    print("Running KF only (no denoise)...")
    kf_no_denoise = KalmanFilter(df.copy(), initial_SOC=soc_initial, use_denoise=False)
    data_kf_only = kf_no_denoise.kalman_filter(Q=Q, R=R, initial_covariance=P)

    results['KF Only'] = {
        'RMSE': root_mean_squared_error(data_kf_only['SOC_true'], data_kf_only['SOC_Calc']),
        'MaxAE': max_absolute_error(data_kf_only['SOC_true'], data_kf_only['SOC_Calc']),
        'MeanAE': mean_absolute_error(data_kf_only['SOC_true'], data_kf_only['SOC_Calc']),
        'SOC_pred': data_kf_only['SOC_Calc'].values
    }

    # 2. KF + denoise - pass a copy of df
    print("Running KF + denoise...")
    kf_denoise = KalmanFilter(df.copy(), initial_SOC=soc_initial, use_denoise=True)
    data_kf_denoise = kf_denoise.kalman_filter(Q=Q, R=R, initial_covariance=P)

    results['KF + Denoise'] = {
        'RMSE': root_mean_squared_error(data_kf_denoise['SOC_true'], data_kf_denoise['SOC_Calc']),
        'MaxAE': max_absolute_error(data_kf_denoise['SOC_true'], data_kf_denoise['SOC_Calc']),
        'MeanAE': mean_absolute_error(data_kf_denoise['SOC_true'], data_kf_denoise['SOC_Calc']),
        'SOC_pred': data_kf_denoise['SOC_Calc'].values
    }

    # 3. KF + denoise + ANN
    print("Running KF + denoise + ANN...")
    file_path_model = script_dir / 'ann_training/Results/my_model.keras'
    file_path_scaler = script_dir / 'ann_training/Results/scaler.joblib'

    ANN = ANN_BatteryAnalysis(str(file_path_model), str(file_path_scaler))
    data_kf_ann = ANN.make_predictions(data_kf_denoise)

    results['KF + Denoise + ANN'] = {
        'RMSE': root_mean_squared_error(data_kf_ann['SOC_true'], data_kf_ann['Predicted_SOC']),
        'MaxAE': max_absolute_error(data_kf_ann['SOC_true'], data_kf_ann['Predicted_SOC']),
        'MeanAE': mean_absolute_error(data_kf_ann['SOC_true'], data_kf_ann['Predicted_SOC']),
        'SOC_pred': data_kf_ann['Predicted_SOC'].values
    }

    return results, df['SOC_true'].values, df


def plot_comparison(results, soc_true, save_dir):
    """Create bar plots comparing metrics."""

    methods = list(results.keys())
    metrics = ['RMSE', 'MaxAE', 'MeanAE']
    metric_labels = ['RMSE (%)', 'Max Absolute Error (%)', 'Mean Absolute Error (%)']
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    # Lorenzo Giannuzzo: Create figure with 3 subplots for metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in methods]

        # Lorenzo Giannuzzo: Bars without black edges
        bars = axes[idx].bar(methods, values, color=colors, edgecolor='none')
        axes[idx].set_ylabel(label, fontsize=12)
        axes[idx].set_title(f'{label}', fontsize=13, fontweight='bold', pad=20)
        axes[idx].tick_params(axis='x', rotation=15)
        axes[idx].grid(axis='y', alpha=0.3)

        # Lorenzo Giannuzzo: Add value labels on bars - positioned higher
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(values),
                           f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('SOC Estimation Methods Comparison\n(GenerateTestData_S4_DAY0to4)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison_barplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Lorenzo Giannuzzo: Create SOC time series comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))

    time_steps = np.arange(len(soc_true))
    ax.plot(time_steps, soc_true, 'k-', label='True SOC', linewidth=2)
    ax.plot(time_steps, results['KF Only']['SOC_pred'], '--', color=colors[0], label='KF Only', linewidth=1.5,
            alpha=0.8)
    ax.plot(time_steps, results['KF + Denoise']['SOC_pred'], '--', color=colors[1], label='KF + Denoise', linewidth=1.5,
            alpha=0.8)
    ax.plot(time_steps, results['KF + Denoise + ANN']['SOC_pred'], '--', color=colors[2], label='KF + Denoise + ANN',
            linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('SOC (%)', fontsize=12)
    ax.set_title('SOC Estimation Comparison - GenerateTestData_S4_DAY0to4', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_dir / 'soc_timeseries_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {save_dir}")


if __name__ == '__main__':

    # Lorenzo Giannuzzo: Paths
    script_dir = Path(__file__).parent
    data_file = script_dir / 'data' / 'GenerateTestData_S4_DAY0to4.xlsx'
    results_dir = script_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    print(f"Loading data from: {data_file}")

    # Lorenzo Giannuzzo: Run comparison
    results, soc_true, df = run_comparison(data_file)

    # Lorenzo Giannuzzo: Plot current comparison (debug)
    plot_current_comparison(df, results_dir)

    # Lorenzo Giannuzzo: Print results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  RMSE:   {metrics['RMSE']:.4f}%")
        print(f"  MaxAE:  {metrics['MaxAE']:.4f}%")
        print(f"  MeanAE: {metrics['MeanAE']:.4f}%")
    print("=" * 60)

    # Lorenzo Giannuzzo: Create plots
    plot_comparison(results, soc_true, results_dir)