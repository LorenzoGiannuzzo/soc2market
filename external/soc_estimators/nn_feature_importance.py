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
Feature Importance Analysis using Permutation Importance
Supports: ANN, CNN, LSTM
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from joblib import load as joblib_load
import os
from pathlib import Path


class FeatureImportanceAnalyzer:
    """
    Permutation Importance analyzer for neural network models.
    Supports ANN, CNN, and LSTM architectures.
    """

    def __init__(self, model_path, scaler_path, model_type='ANN', sequence_length=10):
        self.model = load_model(model_path)
        self.scaler = joblib_load(scaler_path)
        self.model_type = model_type
        self.sequence_length = sequence_length

        self.feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                                'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                                'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

    def preprocess_data(self, df):
        df = df.copy()
        for col in ['Voltage', 'Current_inv', 'SOC_Calc', 'Temp', 'SOC_true']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Current_Voltage_Ratio'] = np.where(df['Voltage'] != 0, df['Current_inv'] / df['Voltage'], 0)
        df['Delta_Current'] = df['Current_inv'].diff().fillna(0)
        df['Delta_Voltage'] = df['Voltage'].diff().fillna(0)
        df['Delta_Voltage_Current_Ratio'] = np.where(df['Delta_Current'] != 0,
                                                     df['Delta_Voltage'] / df['Delta_Current'], 0)
        df['Delta_SoC_Calc'] = df['SOC_Calc'].diff().fillna(0)
        df['Delta_Temperature'] = df['Temp'].diff().fillna(0)
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
        return df

    def create_sequences(self, features):
        X = []
        for i in range(self.sequence_length, len(features)):
            X.append(features[i - self.sequence_length:i])
        return np.array(X)

    def prepare_data(self, df):
        df = self.preprocess_data(df)
        features = df[self.feature_columns]
        features_scaled = self.scaler.transform(features)

        if self.model_type in ['CNN', 'LSTM']:
            X = self.create_sequences(features_scaled)
            y = df['SOC_true'].iloc[self.sequence_length:].values
        else:
            X = features_scaled
            y = df['SOC_true'].values

        return X, y

    def calculate_baseline_error(self, X, y):
        predictions = self.model.predict(X, verbose=0).flatten()
        return mean_squared_error(y, predictions)

    def permutation_importance(self, df, n_repeats=10):
        X, y = self.prepare_data(df)
        baseline_mse = self.calculate_baseline_error(X, y)

        importance_scores = {}

        for i, feature_name in enumerate(self.feature_columns):
            print(f"  Testing feature: {feature_name}...")

            mse_increases = []

            for repeat in range(n_repeats):
                X_permuted = X.copy()

                if self.model_type in ['CNN', 'LSTM']:
                    np.random.shuffle(X_permuted[:, :, i])
                else:
                    np.random.shuffle(X_permuted[:, i])

                permuted_mse = self.calculate_baseline_error(X_permuted, y)
                mse_increase = permuted_mse - baseline_mse
                mse_increases.append(mse_increase)

            importance_scores[feature_name] = {
                'mean': np.mean(mse_increases),
                'std': np.std(mse_increases)
            }

        return importance_scores, baseline_mse

    def plot_importance(self, importance_scores, save_path='feature_importance.pdf', title_suffix=''):
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1]['mean'], reverse=True)

        features = [f[0] for f in sorted_features]
        means = [f[1]['mean'] for f in sorted_features]
        stds = [f[1]['std'] for f in sorted_features]

        colors = {'ANN': 'steelblue', 'CNN': 'darkorange', 'LSTM': 'purple'}
        color = colors.get(self.model_type, 'gray')

        fig, ax = plt.subplots(figsize=(10, 6))

        y_pos = np.arange(len(features))
        ax.barh(y_pos, means, xerr=stds, align='center', alpha=0.8, color=color, capsize=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('MSE Increase (higher = more important)')
        ax.set_title(f'Feature Importance - {self.model_type}{title_suffix}')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()

        print(f"  Plot saved to: {save_path}")
        return sorted_features


def plot_comparison(all_importance, save_path='feature_importance_comparison.pdf'):
    feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                       'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                       'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

    avg_importance = {}
    for feature in feature_columns:
        values = []
        for model_type in all_importance:
            if feature in all_importance[model_type]:
                values.append(all_importance[model_type][feature]['mean'])
        avg_importance[feature] = np.mean(values) if values else 0

    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]

    n_features = len(features)
    n_models = len(all_importance)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_height = 0.25
    y_pos = np.arange(n_features)

    colors = {'ANN': 'steelblue', 'CNN': 'darkorange', 'LSTM': 'purple'}

    for idx, (model_type, importance_scores) in enumerate(all_importance.items()):
        means = [importance_scores.get(f, {'mean': 0})['mean'] for f in features]
        stds = [importance_scores.get(f, {'std': 0})['std'] for f in features]

        offset = (idx - (n_models - 1) / 2) * bar_height
        ax.barh(y_pos + offset, means, bar_height, xerr=stds,
                label=model_type, color=colors.get(model_type, 'gray'),
                alpha=0.8, capsize=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('MSE Increase (higher = more important)')
    ax.set_title('Feature Importance Comparison - ANN vs CNN vs LSTM')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

    print(f"\nComparison plot saved to: {save_path}")


def plot_normalized_comparison(all_importance, save_path='feature_importance_normalized.pdf'):
    feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                       'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                       'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

    normalized = {}
    for model_type, importance_scores in all_importance.items():
        total = sum(max(0, importance_scores.get(f, {'mean': 0})['mean']) for f in feature_columns)
        if total > 0:
            normalized[model_type] = {f: max(0, importance_scores.get(f, {'mean': 0})['mean']) / total * 100
                                      for f in feature_columns}
        else:
            normalized[model_type] = {f: 0 for f in feature_columns}

    avg_importance = {}
    for feature in feature_columns:
        values = [normalized[m].get(feature, 0) for m in normalized]
        avg_importance[feature] = np.mean(values)

    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    features = [f[0] for f in sorted_features]

    n_features = len(features)
    n_models = len(normalized)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_height = 0.25
    y_pos = np.arange(n_features)

    colors = {'ANN': 'steelblue', 'CNN': 'darkorange', 'LSTM': 'purple'}

    for idx, (model_type, norm_scores) in enumerate(normalized.items()):
        values = [norm_scores.get(f, 0) for f in features]
        offset = (idx - (n_models - 1) / 2) * bar_height
        ax.barh(y_pos + offset, values, bar_height, label=model_type,
                color=colors.get(model_type, 'gray'), alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance (%)')
    ax.set_title('Normalized Feature Importance - ANN vs CNN vs LSTM')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()

    print(f"Normalized comparison plot saved to: {save_path}")


def print_importance_table(all_importance, all_baselines):
    feature_columns = ['Temp', 'Voltage', 'Current_inv', 'SOC_Calc', 'Delta_Temperature',
                       'Delta_Current', 'Delta_Voltage', 'Delta_SoC_Calc',
                       'Delta_Voltage_Current_Ratio', 'Current_Voltage_Ratio']

    print("\n" + "=" * 80)
    print("  FEATURE IMPORTANCE SUMMARY")
    print("=" * 80)

    print("\n  Baseline MSE:")
    for model_type, baseline in all_baselines.items():
        print(f"    {model_type}: {baseline:.4f}")

    print("\n  " + "-" * 76)
    header = f"  {'Feature':<30}"
    for model_type in all_importance:
        header += f" | {model_type:^12}"
    print(header)
    print("  " + "-" * 76)

    totals = {}
    for feature in feature_columns:
        totals[feature] = sum(all_importance[m].get(feature, {'mean': 0})['mean']
                              for m in all_importance)

    sorted_features = sorted(totals.items(), key=lambda x: x[1], reverse=True)

    for feature, _ in sorted_features:
        row = f"  {feature:<30}"
        for model_type in all_importance:
            score = all_importance[model_type].get(feature, {'mean': 0, 'std': 0})
            row += f" | {score['mean']:>6.2f}+/-{score['std']:<4.2f}"
        print(row)

    print("  " + "-" * 76)


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    results_dir = script_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    test_file = data_dir / 'GenerateTestData_S4_DAY0to4.xlsx'

    if not test_file.exists():
        xlsx_files = list(data_dir.glob('*.xlsx'))
        xlsx_files = [f for f in xlsx_files if 'HPPC' not in f.name and 'Cha_Dis_OCV' not in f.name]
        if xlsx_files:
            test_file = xlsx_files[0]
            print(f"Using alternative file: {test_file.name}")
        else:
            print("ERROR: No suitable data file found!")
            exit(1)

    print(f"\nLoading data from: {test_file.name}")
    df = pd.read_excel(test_file, skiprows=2).iloc[1:]

    column_mappings = {
        'Voltage(V)': 'Voltage',
        'Current_inv(A)': 'Current_inv',
        'SOC_true(%)': 'SOC_true',
        'SOC_True(%)': 'SOC_true',
        'SOC_True': 'SOC_true',
        'Temp1': 'Temp',
    }
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df[new_name] = df[old_name]

    for col in ['Voltage', 'Current_inv', 'SOC_true', 'Temp']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Temp' not in df.columns:
        df['Temp'] = 25.0

    if 'SOC_Calc' not in df.columns:
        df['SOC_Calc'] = df['SOC_true']
        print("Warning: Using SOC_true as SOC_Calc (KF not run)")

    N_REPEATS = 5

    all_importance = {}
    all_baselines = {}

    # ANN
    ann_model_path = script_dir / 'ann_training/Results/my_model.keras'
    ann_scaler_path = script_dir / 'ann_training/Results/scaler.joblib'

    if ann_model_path.exists() and ann_scaler_path.exists():
        print("\n" + "=" * 60)
        print("  ANN FEATURE IMPORTANCE")
        print("=" * 60)

        ann_analyzer = FeatureImportanceAnalyzer(
            model_path=str(ann_model_path),
            scaler_path=str(ann_scaler_path),
            model_type='ANN'
        )

        ann_importance, ann_baseline = ann_analyzer.permutation_importance(df, n_repeats=N_REPEATS)
        all_importance['ANN'] = ann_importance
        all_baselines['ANN'] = ann_baseline

        print(f"\n  Baseline MSE: {ann_baseline:.4f}")
        ann_analyzer.plot_importance(ann_importance,
                                     save_path=results_dir / 'feature_importance_ANN.pdf')
    else:
        print("\nWARNING: ANN model not found, skipping...")

    # CNN
    cnn_model_path = script_dir / 'ann_training/Results/cnn_model.keras'
    cnn_scaler_path = script_dir / 'ann_training/Results/cnn_scaler.joblib'

    if cnn_model_path.exists() and cnn_scaler_path.exists():
        print("\n" + "=" * 60)
        print("  CNN FEATURE IMPORTANCE")
        print("=" * 60)

        cnn_analyzer = FeatureImportanceAnalyzer(
            model_path=str(cnn_model_path),
            scaler_path=str(cnn_scaler_path),
            model_type='CNN',
            sequence_length=10
        )

        cnn_importance, cnn_baseline = cnn_analyzer.permutation_importance(df, n_repeats=N_REPEATS)
        all_importance['CNN'] = cnn_importance
        all_baselines['CNN'] = cnn_baseline

        print(f"\n  Baseline MSE: {cnn_baseline:.4f}")
        cnn_analyzer.plot_importance(cnn_importance,
                                     save_path=results_dir / 'feature_importance_CNN.pdf')
    else:
        print("\nWARNING: CNN model not found, skipping...")

    # LSTM
    lstm_model_path = script_dir / 'ann_training/Results/lstm_model.keras'
    lstm_scaler_path = script_dir / 'ann_training/Results/lstm_scaler.joblib'

    if lstm_model_path.exists() and lstm_scaler_path.exists():
        print("\n" + "=" * 60)
        print("  LSTM FEATURE IMPORTANCE")
        print("=" * 60)

        lstm_analyzer = FeatureImportanceAnalyzer(
            model_path=str(lstm_model_path),
            scaler_path=str(lstm_scaler_path),
            model_type='LSTM',
            sequence_length=10
        )

        lstm_importance, lstm_baseline = lstm_analyzer.permutation_importance(df, n_repeats=N_REPEATS)
        all_importance['LSTM'] = lstm_importance
        all_baselines['LSTM'] = lstm_baseline

        print(f"\n  Baseline MSE: {lstm_baseline:.4f}")
        lstm_analyzer.plot_importance(lstm_importance,
                                      save_path=results_dir / 'feature_importance_LSTM.pdf')
    else:
        print("\nWARNING: LSTM model not found, skipping...")

    # Comparison
    if len(all_importance) >= 2:
        print("\n" + "=" * 60)
        print("  GENERATING COMPARISON PLOTS")
        print("=" * 60)

        plot_comparison(all_importance, save_path=results_dir / 'feature_importance_comparison.pdf')
        plot_normalized_comparison(all_importance, save_path=results_dir / 'feature_importance_normalized.pdf')
        print_importance_table(all_importance, all_baselines)

    print("\n" + "=" * 60)
    print("  FEATURE IMPORTANCE ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files in: {results_dir}/")