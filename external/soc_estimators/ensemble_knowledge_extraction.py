"""
================================================================================
Ensemble Meta-Model Knowledge Extraction
================================================================================

Lorenzo Giannuzzo - Politecnico di Torino, Energy Center Lab

This script extracts interpretable knowledge from the ensemble meta-model
(Dynamic Weighting or Model Selector) to understand:
- Which features drive model selection
- Under what conditions each base model is preferred
- Explicit rules for model preference

Techniques used:
1. Decision Tree Surrogate - Approximates neural network with interpretable tree
2. Partial Dependence Plots - How features affect model preference
3. Rule Extraction - Human-readable decision rules
4. Preference Heatmaps - 2D visualization of model dominance regions

Output:
- PDF report with visualizations
- Text file with extracted rules
- Excel file with feature importance rankings

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Deep learning
import tensorflow as tf
from tensorflow import keras


# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose ensemble type: 'dynamic' or 'selector'
ENSEMBLE_TYPE = 'selector'  # Options: 'dynamic', 'selector'

# Model subdirectory (use '' for default models, 'old_ensemble_models' for old)
MODEL_SUBDIR = ''  # Options: '' (default), 'old_ensemble_models'

# ============================================================================
# SURROGATE TREE CONFIGURATION
# ============================================================================
# Step 1 (interpretabile, ~75-85% fidelity):
#   TREE_MAX_DEPTH = 12, TREE_MIN_SAMPLES_LEAF = 10
#
# Step 2 (più accurato, ~85-92% fidelity):
#   TREE_MAX_DEPTH = 15, TREE_MIN_SAMPLES_LEAF = 5
# ============================================================================
TREE_MAX_DEPTH = 15          # Profondità massima albero (12 per Step 1, 15 per Step 2)
TREE_MIN_SAMPLES_LEAF = 10   # Minimo campioni per foglia (10 per Step 1, 5 per Step 2)
ADD_PREDICTION_DIFFS = True  # Aggiungi differenze tra predizioni come features

# Model names (must match training order - including KF if present)
BASE_MODEL_NAMES_5 = ['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']
BASE_MODEL_NAMES_6 = ['KF', 'ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']

# Will be set dynamically based on model output shape
BASE_MODEL_NAMES = None

# Context feature names - MUST MATCH ensemble_train.py exactly!
CONTEXT_FEATURES = [
    'Temp', 'Voltage', 'Current_inv', 'SOC_Calc',
    'Delta_Temperature', 'Delta_Current', 'Delta_Voltage',
    'Delta_SoC_Calc', 'Current_Voltage_Ratio'
]

# For rule extraction - updated for pointwise features
FEATURE_BINS = {
    'Temp': [0, 15, 25, 35, 45, np.inf],
    'Voltage': [2.8, 3.0, 3.2, 3.3, 3.4, 3.5],
    'Current_inv': [-np.inf, -100, -50, 0, 50, 100, np.inf],
    'SOC_Calc': [0, 20, 40, 60, 80, 100],
    'Delta_Temperature': [-np.inf, -0.1, -0.01, 0, 0.01, 0.1, np.inf],
    'Delta_Current': [-np.inf, -10, -1, 0, 1, 10, np.inf],
    'Delta_Voltage': [-np.inf, -0.01, -0.001, 0, 0.001, 0.01, np.inf],
    'Delta_SoC_Calc': [-np.inf, -0.1, -0.01, 0, 0.01, 0.1, np.inf],
    'Current_Voltage_Ratio': [-np.inf, -30, -10, 0, 10, 30, np.inf],
}

FEATURE_LABELS = {
    'Temp': ['Cold', 'Cool', 'Normal', 'Warm', 'Hot'],
    'Voltage': ['Very Low', 'Low', 'Mid-Low', 'Mid', 'Mid-High'],
    'Current_inv': ['Heavy Discharge', 'Moderate Disch.', 'Light Disch.', 'Light Charge', 'Moderate Charge', 'Heavy Charge'],
    'SOC_Calc': ['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    'Delta_Current': ['Large Drop', 'Small Drop', 'Stable', 'Small Rise', 'Large Rise'],
}


# ============================================================================
# LOAD META-MODEL AND DATA
# ============================================================================

def load_ensemble_model(model_dir, ensemble_type):
    """Load the ensemble meta-model and associated scalers."""
    model_dir = Path(model_dir)

    if ensemble_type == 'dynamic':
        model_path = model_dir / 'ensemble_dynamic_model.keras'
        scaler_path = model_dir / 'ensemble_dynamic_scaler.joblib'
    elif ensemble_type == 'selector':
        model_path = model_dir / 'ensemble_selector_model.keras'
        scaler_path = model_dir / 'ensemble_selector_scaler.joblib'
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    print(f"Loading {ensemble_type} model from: {model_path}")

    model = keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


def extract_weight_model(full_model, ensemble_type):
    """
    Extract the sub-model that maps to model weights/selection.

    For Dynamic: context → weights (only context input needed)
    For Selector: (predictions, context) → selection (both inputs needed)
    """
    if ensemble_type == 'dynamic':
        # Dynamic uses only context to determine weights
        context_input = full_model.inputs[1]
        weights_output = full_model.get_layer('weights').output
        weight_model = keras.Model(inputs=context_input, outputs=weights_output)
        return weight_model, False  # False = doesn't need predictions

    else:  # selector
        # Selector uses BOTH predictions and context (concatenated)
        selection_output = full_model.get_layer('selection').output
        weight_model = keras.Model(inputs=full_model.inputs, outputs=selection_output)
        return weight_model, True  # True = needs predictions


def generate_synthetic_predictions(n_samples, n_models=6, soc_values=None, seed=42):
    """
    Generate synthetic base model predictions for Selector analysis.
    Simulates realistic prediction patterns around true SOC values.

    If soc_values is provided, predictions are generated around those values.
    """
    np.random.seed(seed)

    # Generate "true" SOC values if not provided
    if soc_values is None:
        true_soc = np.random.uniform(10, 90, n_samples)
    else:
        true_soc = np.array(soc_values)

    # Generate predictions for each model with different error patterns
    predictions = np.zeros((n_samples, n_models))

    # Model characteristics (bias, noise_std)
    model_chars = [
        (0.0, 2.0),   # KF - low bias, moderate noise
        (0.5, 3.0),   # ANN - slight bias, higher noise
        (-0.3, 2.5),  # CNN - slight negative bias
        (0.2, 2.8),   # LSTM - slight bias
        (0.0, 1.5),   # CatBoost - low bias, low noise
        (-0.1, 1.8),  # TFT - very low bias, low noise
    ]

    for i, (bias, noise_std) in enumerate(model_chars[:n_models]):
        predictions[:, i] = true_soc + bias + np.random.normal(0, noise_std, n_samples)

    # Clip to valid SOC range
    predictions = np.clip(predictions, 0, 100)

    return predictions


def load_real_predictions(results_dir, base_model_names, subsample_rate=10):
    """
    Load REAL predictions from all base model Excel files.

    This loads predictions from results/results_sheet/{ModelName}.xlsx
    and aligns them by scenario and timestep.

    Returns:
        predictions: np.array of shape (n_samples, n_models)
        context_data: pd.DataFrame with context features

    Or (None, None) if loading fails.
    """
    results_dir = Path(results_dir)
    results_sheet_dir = results_dir / 'results_sheet'

    if not results_sheet_dir.exists():
        print(f"  Results sheet directory not found: {results_sheet_dir}")
        return None, None

    # Model file mapping (handle KF which might be in a different file)
    # KF predictions are in SOC_Calc column, others in Predicted_SOC
    model_files = {}
    for model_name in base_model_names:
        if model_name == 'KF':
            # KF uses SOC_Calc from any file (it's the Kalman Filter output)
            # We'll get it from the first available file
            model_files['KF'] = ('ANY', 'SOC_Calc')
        else:
            model_files[model_name] = (f'{model_name}.xlsx', 'Predicted_SOC')

    print(f"\n  Loading real predictions from {results_sheet_dir}")
    print(f"  Base models: {base_model_names}")

    # First, find all scenarios from any available model file
    available_files = list(results_sheet_dir.glob('*.xlsx'))
    if not available_files:
        print(f"  No Excel files found!")
        return None, None

    # Use a reference file to get scenario list
    ref_file = None
    for f in available_files:
        if f.stem in base_model_names or f.stem == 'Ensemble':
            ref_file = f
            break
    if ref_file is None:
        ref_file = available_files[0]

    print(f"  Reference file: {ref_file.name}")

    # Get scenario list
    try:
        xl_ref = pd.ExcelFile(ref_file)
        scenarios = [s for s in xl_ref.sheet_names if s != 'Metrics']
        print(f"  Found {len(scenarios)} scenarios")
    except Exception as e:
        print(f"  Error reading reference file: {e}")
        return None, None

    # Load predictions for each model
    all_data = []

    for scenario in scenarios:
        print(f"    Processing {scenario}...", end=" ")

        scenario_predictions = {}
        scenario_context = None
        n_rows = None

        for model_name in base_model_names:
            if model_name == 'KF':
                # KF is SOC_Calc - get from any file
                try:
                    df = pd.read_excel(ref_file, sheet_name=scenario)
                    if 'SOC_Calc' in df.columns:
                        scenario_predictions['KF'] = pd.to_numeric(df['SOC_Calc'], errors='coerce').values
                        n_rows = len(df)

                        # Also get context data from this file
                        if scenario_context is None:
                            scenario_context = df.copy()
                except:
                    pass
            else:
                # Other models - load from their specific file
                model_file = results_sheet_dir / f'{model_name}.xlsx'
                if not model_file.exists():
                    print(f"\n      Warning: {model_name}.xlsx not found")
                    continue

                try:
                    df = pd.read_excel(model_file, sheet_name=scenario)
                    if 'Predicted_SOC' in df.columns:
                        scenario_predictions[model_name] = pd.to_numeric(df['Predicted_SOC'], errors='coerce').values
                        if n_rows is None:
                            n_rows = len(df)

                        # Get context if not already loaded
                        if scenario_context is None:
                            scenario_context = df.copy()
                except Exception as e:
                    print(f"\n      Error loading {model_name} for {scenario}: {e}")
                    continue

        # Check we have all models
        if len(scenario_predictions) < len(base_model_names):
            missing = set(base_model_names) - set(scenario_predictions.keys())
            print(f"missing {missing}, skipping")
            continue

        # Verify all have same length
        lengths = [len(v) for v in scenario_predictions.values()]
        if len(set(lengths)) > 1:
            print(f"length mismatch {lengths}, skipping")
            continue

        # Preprocess context data
        if scenario_context is not None:
            # Ensure required columns exist
            required = ['Voltage', 'Current_inv', 'Temp']
            if not all(c in scenario_context.columns for c in required):
                print(f"missing context columns, skipping")
                continue

            # Get SOC_Calc
            if 'SOC_Calc' not in scenario_context.columns:
                if 'SOC_true' in scenario_context.columns:
                    scenario_context['SOC_Calc'] = scenario_context['SOC_true']
                else:
                    print(f"missing SOC column, skipping")
                    continue

            # Convert to numeric
            for col in ['Voltage', 'Current_inv', 'SOC_Calc', 'Temp']:
                if col in scenario_context.columns:
                    scenario_context[col] = pd.to_numeric(scenario_context[col], errors='coerce')

            # Add delta features
            scenario_context = preprocess_data(scenario_context)

            # Create predictions array in correct order
            pred_array = np.column_stack([scenario_predictions[m] for m in base_model_names])

            # Add to context dataframe
            for i, model_name in enumerate(base_model_names):
                scenario_context[f'Pred_{model_name}'] = pred_array[:, i]

            scenario_context['scenario'] = scenario

            # Subsample
            scenario_context = scenario_context.iloc[::subsample_rate].copy()
            scenario_context = scenario_context.dropna()

            all_data.append(scenario_context)
            print(f"{len(scenario_context)} samples")
        else:
            print("no context data, skipping")

    if not all_data:
        print("  No valid data loaded!")
        return None, None

    # Combine all scenarios
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n  Total samples loaded: {len(combined):,}")

    # Extract predictions array
    pred_cols = [f'Pred_{m}' for m in base_model_names]
    predictions = combined[pred_cols].values

    # Context data (only CONTEXT_FEATURES)
    context_data = combined[CONTEXT_FEATURES + ['scenario']].copy()

    return predictions, context_data


def generate_synthetic_context_data(n_samples=10000, seed=42):
    """Generate synthetic context data covering the operating range."""
    np.random.seed(seed)

    # Generate base features
    voltage = np.random.uniform(2.8, 3.5, n_samples)
    current = np.random.uniform(-150, 150, n_samples)
    soc = np.random.uniform(0, 100, n_samples)
    temp = np.random.uniform(10, 45, n_samples)

    # Generate delta features (small changes)
    delta_temp = np.random.normal(0, 0.1, n_samples)
    delta_current = np.random.normal(0, 5, n_samples)
    delta_voltage = np.random.normal(0, 0.005, n_samples)
    delta_soc = np.random.normal(0, 0.05, n_samples)

    # Current/Voltage ratio
    cv_ratio = current / voltage

    data = {
        'Temp': temp,
        'Voltage': voltage,
        'Current_inv': current,
        'SOC_Calc': soc,
        'Delta_Temperature': delta_temp,
        'Delta_Current': delta_current,
        'Delta_Voltage': delta_voltage,
        'Delta_SoC_Calc': delta_soc,
        'Current_Voltage_Ratio': cv_ratio,
    }

    return pd.DataFrame(data)


def preprocess_data(df):
    """Add delta features to dataframe - same as ensemble_train.py."""
    df = df.copy()

    if 'Current_Voltage_Ratio' not in df.columns:
        df['Current_Voltage_Ratio'] = (df['Current_inv'] / df['Voltage']).fillna(0)

    if 'Delta_Current' not in df.columns:
        df['Delta_Current'] = df['Current_inv'].diff().fillna(0)

    if 'Delta_Voltage' not in df.columns:
        df['Delta_Voltage'] = df['Voltage'].diff().fillna(0)

    if 'Delta_SoC_Calc' not in df.columns:
        df['Delta_SoC_Calc'] = df['SOC_Calc'].diff().fillna(0)

    if 'Delta_Temperature' not in df.columns:
        df['Delta_Temperature'] = df['Temp'].diff().fillna(0)

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def load_real_context_data(data_dir, results_dir):
    """
    Load real context data from the prediction results.
    Now loads POINTWISE data and computes delta features.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)

    # Check for cached data with correct features
    context_path = results_dir / 'ensemble_context_data_pointwise.csv'
    if context_path.exists():
        print(f"  Loading cached pointwise context data from: {context_path}")
        df = pd.read_csv(context_path)
        # Verify it has the correct features
        if all(f in df.columns for f in CONTEXT_FEATURES):
            return df
        else:
            print(f"  Cache has wrong features, regenerating...")

    results_sheet_dir = results_dir / 'results_sheet'

    if not results_sheet_dir.exists():
        print(f"  Results sheet directory not found: {results_sheet_dir}")
        return None

    excel_files = list(results_sheet_dir.glob('*.xlsx'))
    if not excel_files:
        print(f"  No Excel files found in: {results_sheet_dir}")
        return None

    # Use first available file (should be Ensemble.xlsx or similar)
    excel_file = excel_files[0]
    print(f"  Loading data from: {excel_file.name}")

    all_context_data = []

    try:
        xl = pd.ExcelFile(excel_file)

        for sheet_name in xl.sheet_names:
            if sheet_name == 'Metrics':
                continue

            try:
                df = pd.read_excel(xl, sheet_name=sheet_name)
            except Exception as e:
                print(f"    Error reading {sheet_name}: {e}")
                continue

            # Check required columns
            required_cols = ['Voltage', 'Current_inv', 'Temp']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"    Skipping {sheet_name}: missing columns {missing}")
                continue

            # Get SOC_Calc (may be called SOC_true in some files)
            if 'SOC_Calc' not in df.columns:
                if 'SOC_true' in df.columns:
                    df['SOC_Calc'] = df['SOC_true']
                else:
                    print(f"    Skipping {sheet_name}: missing SOC column")
                    continue

            # Convert to numeric
            for col in ['Voltage', 'Current_inv', 'SOC_Calc', 'Temp']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add delta features
            df = preprocess_data(df)

            # Add scenario column
            df['scenario'] = sheet_name

            # Select only the features we need
            cols_to_keep = CONTEXT_FEATURES + ['scenario']
            cols_available = [c for c in cols_to_keep if c in df.columns]

            # Subsample to avoid huge datasets (keep every 10th point)
            df_sampled = df[cols_available].iloc[::10].copy()
            df_sampled = df_sampled.dropna()

            all_context_data.append(df_sampled)
            print(f"    {sheet_name}: {len(df_sampled):,} samples")

        if all_context_data:
            context_df = pd.concat(all_context_data, ignore_index=True)
            context_df.to_csv(context_path, index=False)
            print(f"  Cached context data to: {context_path}")
            print(f"  Total samples: {len(context_df):,}")
            return context_df
        else:
            print("  No valid context data extracted from Excel files")
            return None

    except Exception as e:
        print(f"  Error loading Excel file: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# DECISION TREE SURROGATE
# ============================================================================

def train_surrogate_tree(weight_model, context_data, scaler, output_dir,
                         ensemble_type='dynamic', max_depth=None,
                         predictions=None, needs_predictions=False,
                         include_predictions_in_tree=True):
    """
    Train a Decision Tree to approximate the neural network meta-model.

    For Selector with real predictions:
    - If include_predictions_in_tree=True: Tree uses both predictions + context (best fidelity)
    - If include_predictions_in_tree=False: Tree uses only context (shows operating condition patterns)
    """
    print("\n" + "="*60)
    print("  DECISION TREE SURROGATE MODEL")
    print("="*60)

    # Use configuration values
    if max_depth is None:
        max_depth = TREE_MAX_DEPTH
    min_samples_leaf = TREE_MIN_SAMPLES_LEAF

    context_scaled = scaler.transform(context_data[CONTEXT_FEATURES].values)

    # Get NN predictions based on model type
    if needs_predictions and predictions is not None:
        nn_predictions = weight_model.predict([predictions, context_scaled], verbose=0)
    else:
        nn_predictions = weight_model.predict(context_scaled, verbose=0)

    y_target = np.argmax(nn_predictions, axis=1)

    # Decide what features to use for the surrogate tree
    if needs_predictions and predictions is not None and include_predictions_in_tree:
        # Start with predictions + context
        X_parts = [predictions, context_data[CONTEXT_FEATURES].values]
        feature_names = [f'Pred_{m}' for m in BASE_MODEL_NAMES] + CONTEXT_FEATURES

        # Add derived features: pairwise differences between predictions
        if ADD_PREDICTION_DIFFS:
            print(f"  Adding pairwise prediction differences...")
            diff_features = []
            diff_names = []
            n_models = len(BASE_MODEL_NAMES)
            for i in range(n_models):
                for j in range(i+1, n_models):
                    diff = predictions[:, i] - predictions[:, j]
                    diff_features.append(diff)
                    diff_names.append(f'Diff_{BASE_MODEL_NAMES[i]}_{BASE_MODEL_NAMES[j]}')

            if diff_features:
                X_parts.append(np.column_stack(diff_features))
                feature_names.extend(diff_names)

        X = np.hstack(X_parts)
        print(f"  Using predictions + context + diffs as features ({len(feature_names)} total)")
    else:
        # Context only
        X = context_data[CONTEXT_FEATURES].values
        feature_names = CONTEXT_FEATURES
        print(f"  Using context features only ({len(feature_names)} total)")

    print(f"  Tree config: max_depth={max_depth}, min_samples_leaf={min_samples_leaf}")

    # Train Decision Tree
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    tree.fit(X, y_target)

    actual_classes = tree.classes_
    actual_class_names = [BASE_MODEL_NAMES[i] for i in actual_classes]

    print(f"  Models found in data: {actual_class_names}")

    tree_predictions = tree.predict(X)
    fidelity = np.mean(tree_predictions == y_target) * 100
    print(f"  Surrogate tree fidelity: {fidelity:.1f}%")

    # Show per-class fidelity
    print(f"\n  Per-class fidelity:")
    for cls_idx, cls_name in zip(actual_classes, actual_class_names):
        mask = y_target == cls_idx
        if mask.sum() > 0:
            cls_fidelity = np.mean(tree_predictions[mask] == y_target[mask]) * 100
            print(f"    {cls_name}: {cls_fidelity:.1f}% ({mask.sum():,} samples)")

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(32, 24))
    plot_tree(tree,
              feature_names=feature_names,
              class_names=actual_class_names,
              filled=True,
              rounded=True,
              fontsize=6,
              ax=ax)

    tree_title = f'Decision Tree Surrogate (Fidelity: {fidelity:.1f}%)\nModels: {", ".join(actual_class_names)}'
    if needs_predictions and include_predictions_in_tree:
        tree_title += f"\n(Using {len(feature_names)} features: predictions + context"
        if ADD_PREDICTION_DIFFS:
            tree_title += " + diffs"
        tree_title += ")"
    ax.set_title(tree_title, fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch

    model_color_map = {
        'KF': '#8c564b',       # Brown
        'ANN': '#1f77b4',      # Azzurro (blue)
        'CNN': '#7fcdff',      # Celeste (light blue)
        'LSTM': '#d35400',     # Arancione scuro (dark orange)
        'CatBoost': '#f39c12', # Giallo/arancione (yellow-orange)
        'TFT': '#9b59b6',      # Violetto (purple)
    }
    legend_colors = [model_color_map.get(name, '#7f7f7f') for name in actual_class_names]

    legend_patches = []
    for i, class_name in enumerate(actual_class_names):
        legend_patches.append(Patch(facecolor=legend_colors[i], edgecolor='black',
                                    label=class_name, linewidth=1.5))

    ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=len(actual_class_names), fontsize=12, title='Preferred Model (class)',
              title_fontsize=13, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_dir / 'surrogate_tree_visualization.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'surrogate_tree_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: surrogate_tree_visualization.pdf")

    # Extract text rules
    tree_rules = export_text(tree, feature_names=feature_names, class_names=actual_class_names)

    rules_path = output_dir / 'surrogate_tree_rules.txt'
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("  DECISION TREE RULES (Surrogate Model)\n")
        f.write(f"  Fidelity: {fidelity:.1f}%\n")
        f.write(f"  max_depth: {max_depth}, min_samples_leaf: {min_samples_leaf}\n")
        if needs_predictions and include_predictions_in_tree:
            f.write("  Features: Predictions + Context")
            if ADD_PREDICTION_DIFFS:
                f.write(" + Pairwise Differences")
            f.write(f" ({len(feature_names)} total)\n")
        else:
            f.write("  Features: Context only\n")
        f.write("="*70 + "\n\n")
        f.write(tree_rules)
    print(f"  Saved: surrogate_tree_rules.txt")

    # Feature importance
    tree_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': tree.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = []
    for feat in tree_importance['Feature'].values:
        if feat.startswith('Pred_'):
            colors.append('#e74c3c')  # Red for predictions
        elif feat.startswith('Diff_'):
            colors.append('#27ae60')  # Green for differences
        else:
            colors.append('#3498db')  # Blue for context

    ax.barh(range(len(tree_importance)), tree_importance['Importance'].values, color=colors)
    ax.set_yticks(range(len(tree_importance)))
    ax.set_yticklabels(tree_importance['Feature'].values, fontsize=8)
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(f'Surrogate Tree - Feature Importance (Fidelity: {fidelity:.1f}%)', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend for colors
    if needs_predictions and include_predictions_in_tree:
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor='#e74c3c', label='Prediction features'),
            Patch(facecolor='#3498db', label='Context features'),
        ]
        if ADD_PREDICTION_DIFFS:
            legend_patches.append(Patch(facecolor='#27ae60', label='Prediction differences'))
        ax.legend(handles=legend_patches, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_dir / 'surrogate_tree_importance.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: surrogate_tree_importance.pdf")

    return tree, tree_rules, fidelity, tree_importance


# ============================================================================
# EXPLICIT RULE EXTRACTION
# ============================================================================

def extract_explicit_rules(weight_model, context_data, scaler, output_dir,
                           predictions=None, needs_predictions=False):
    """Extract explicit, human-readable rules."""
    print("\n" + "="*60)
    print("  EXPLICIT RULE EXTRACTION")
    print("="*60)

    context_scaled = scaler.transform(context_data[CONTEXT_FEATURES].values)

    # Get NN predictions based on model type
    if needs_predictions and predictions is not None:
        nn_weights = weight_model.predict([predictions, context_scaled], verbose=0)
    else:
        nn_weights = weight_model.predict(context_scaled, verbose=0)

    dominant_model = np.argmax(nn_weights, axis=1)

    rules = []

    print("\n  Analyzing single-feature preferences...")

    single_feature_prefs = {}

    for feature in CONTEXT_FEATURES:
        if feature not in FEATURE_BINS:
            continue

        bins = FEATURE_BINS[feature]
        feature_values = context_data[feature].values

        bin_indices = np.digitize(feature_values, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        prefs = []
        for bin_idx in range(len(bins) - 1):
            mask = bin_indices == bin_idx
            if mask.sum() < 10:
                continue

            models_in_bin = dominant_model[mask]
            counts = np.bincount(models_in_bin, minlength=len(BASE_MODEL_NAMES))
            percentages = counts / counts.sum() * 100

            preferred_model = np.argmax(counts)
            preferred_pct = percentages[preferred_model]

            bin_label = f"[{bins[bin_idx]:.2f}, {bins[bin_idx+1]:.2f})"
            if feature in FEATURE_LABELS and bin_idx < len(FEATURE_LABELS[feature]):
                bin_label = FEATURE_LABELS[feature][bin_idx]

            prefs.append({
                'bin_idx': bin_idx, 'bin_label': bin_label,
                'bin_range': (bins[bin_idx], bins[bin_idx+1]),
                'n_samples': mask.sum(),
                'preferred_model': BASE_MODEL_NAMES[preferred_model],
                'preferred_pct': preferred_pct,
            })

            if preferred_pct > 40:
                rule = {
                    'type': 'single', 'feature': feature, 'condition': bin_label,
                    'range': (bins[bin_idx], bins[bin_idx+1]),
                    'preferred_model': BASE_MODEL_NAMES[preferred_model],
                    'confidence': preferred_pct, 'n_samples': mask.sum(),
                }
                rules.append(rule)

        single_feature_prefs[feature] = prefs

    # Two-feature interactions
    print("  Analyzing two-feature interactions...")

    key_features = ['Current_inv', 'SOC_Calc', 'Voltage', 'Temp']

    for i, feat1 in enumerate(key_features):
        for feat2 in key_features[i+1:]:
            if feat1 not in FEATURE_BINS or feat2 not in FEATURE_BINS:
                continue

            bins1 = FEATURE_BINS[feat1]
            bins2 = FEATURE_BINS[feat2]

            bin_idx1 = np.clip(np.digitize(context_data[feat1].values, bins1) - 1, 0, len(bins1) - 2)
            bin_idx2 = np.clip(np.digitize(context_data[feat2].values, bins2) - 1, 0, len(bins2) - 2)

            for b1 in range(len(bins1) - 1):
                for b2 in range(len(bins2) - 1):
                    mask = (bin_idx1 == b1) & (bin_idx2 == b2)
                    if mask.sum() < 20:
                        continue

                    models_in_region = dominant_model[mask]
                    counts = np.bincount(models_in_region, minlength=len(BASE_MODEL_NAMES))
                    percentages = counts / counts.sum() * 100

                    preferred_model = np.argmax(counts)
                    preferred_pct = percentages[preferred_model]

                    if preferred_pct > 50:
                        label1 = FEATURE_LABELS.get(feat1, [f"bin_{b1}"])[min(b1, len(FEATURE_LABELS.get(feat1, [])) - 1)] if feat1 in FEATURE_LABELS else f"[{bins1[b1]:.1f},{bins1[b1+1]:.1f})"
                        label2 = FEATURE_LABELS.get(feat2, [f"bin_{b2}"])[min(b2, len(FEATURE_LABELS.get(feat2, [])) - 1)] if feat2 in FEATURE_LABELS else f"[{bins2[b2]:.1f},{bins2[b2+1]:.1f})"

                        rule = {
                            'type': 'interaction', 'features': (feat1, feat2),
                            'conditions': (label1, label2),
                            'ranges': ((bins1[b1], bins1[b1+1]), (bins2[b2], bins2[b2+1])),
                            'preferred_model': BASE_MODEL_NAMES[preferred_model],
                            'confidence': preferred_pct, 'n_samples': mask.sum(),
                        }
                        rules.append(rule)

    # Generate report
    report_path = output_dir / 'extracted_rules.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("  EXTRACTED KNOWLEDGE FROM ENSEMBLE META-MODEL\n")
        f.write(f"  Ensemble Type: {ENSEMBLE_TYPE}\n")
        f.write("="*70 + "\n\n")

        f.write("SINGLE-FEATURE RULES:\n")
        f.write("-"*50 + "\n\n")

        for rule in rules:
            if rule['type'] == 'single':
                f.write(f"IF {rule['feature']} is {rule['condition']}\n")
                f.write(f"   Range: {rule['range']}\n")
                f.write(f"   -> Prefer {rule['preferred_model']} ({rule['confidence']:.1f}% of cases)\n")
                f.write(f"   (Based on {rule['n_samples']:,} samples)\n\n")

        f.write("\n" + "="*70 + "\n")
        f.write("TWO-FEATURE INTERACTION RULES:\n")
        f.write("-"*50 + "\n\n")

        for rule in rules:
            if rule['type'] == 'interaction':
                f.write(f"IF {rule['features'][0]} is {rule['conditions'][0]}\n")
                f.write(f"   AND {rule['features'][1]} is {rule['conditions'][1]}\n")
                f.write(f"   -> Prefer {rule['preferred_model']} ({rule['confidence']:.1f}% of cases)\n")
                f.write(f"   (Based on {rule['n_samples']:,} samples)\n\n")

        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY STATISTICS:\n")
        f.write("-"*50 + "\n\n")

        model_counts = np.bincount(dominant_model, minlength=len(BASE_MODEL_NAMES))
        model_pcts = model_counts / model_counts.sum() * 100

        f.write("Overall model preference distribution:\n")
        for i, name in enumerate(BASE_MODEL_NAMES):
            f.write(f"  {name}: {model_pcts[i]:.1f}% ({model_counts[i]:,} samples)\n")

    print(f"  Saved: extracted_rules.txt")

    return rules, single_feature_prefs


# ============================================================================
# PARTIAL DEPENDENCE PLOTS
# ============================================================================

def create_partial_dependence_plots(weight_model, context_data, scaler, output_dir,
                                     predictions=None, needs_predictions=False):
    """Create partial dependence plots."""
    print("\n" + "="*60)
    print("  PARTIAL DEPENDENCE ANALYSIS")
    print("="*60)

    def predict_weights(X, preds=None):
        X_scaled = scaler.transform(X)
        if needs_predictions and preds is not None:
            return weight_model.predict([preds, X_scaled], verbose=0)
        else:
            return weight_model.predict(X_scaled, verbose=0)

    # Key features for PDP (pointwise values)
    key_features = ['Current_inv', 'SOC_Calc', 'Voltage', 'Temp']

    fig, axes = plt.subplots(len(key_features), len(BASE_MODEL_NAMES),
                             figsize=(20, 4 * len(key_features)))

    n_models = len(BASE_MODEL_NAMES)
    model_color_map = {
        'KF': '#8c564b',       # Brown
        'ANN': '#1f77b4',      # Azzurro (blue)
        'CNN': '#7fcdff',      # Celeste (light blue)
        'LSTM': '#d35400',     # Arancione scuro (dark orange)
        'CatBoost': '#f39c12', # Giallo/arancione (yellow-orange)
        'TFT': '#9b59b6',      # Violetto (purple)
    }
    model_colors = [model_color_map.get(name, '#7f7f7f') for name in BASE_MODEL_NAMES]

    for feat_idx, feature in enumerate(key_features):
        feat_col_idx = CONTEXT_FEATURES.index(feature)

        feat_min = context_data[feature].quantile(0.01)
        feat_max = context_data[feature].quantile(0.99)
        grid_values = np.linspace(feat_min, feat_max, 50)

        n_samples = min(500, len(context_data))
        sample_indices = np.random.choice(len(context_data), n_samples, replace=False)
        base_data = context_data[CONTEXT_FEATURES].iloc[sample_indices].values.copy()

        # Get corresponding predictions subset if needed
        sample_preds = None
        if needs_predictions and predictions is not None:
            sample_preds = predictions[sample_indices]

        pdp_values = np.zeros((len(grid_values), len(BASE_MODEL_NAMES)))

        for grid_idx, grid_val in enumerate(grid_values):
            modified_data = base_data.copy()
            modified_data[:, feat_col_idx] = grid_val
            preds_result = predict_weights(modified_data, sample_preds)
            pdp_values[grid_idx] = preds_result.mean(axis=0)

        for model_idx, model_name in enumerate(BASE_MODEL_NAMES):
            ax = axes[feat_idx, model_idx]
            color = model_colors[model_idx]
            ax.plot(grid_values, pdp_values[:, model_idx], linewidth=2, color=color)
            ax.fill_between(grid_values, 0, pdp_values[:, model_idx], alpha=0.3, color=color)
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Weight' if model_idx == 0 else '', fontsize=10)
            ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

    plt.suptitle('Partial Dependence: Feature Effect on Model Weights',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'partial_dependence_plots.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'partial_dependence_plots.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: partial_dependence_plots.pdf")


# ============================================================================
# PREFERENCE HEATMAPS
# ============================================================================

def create_preference_heatmaps(weight_model, context_data, scaler, output_dir,
                               predictions=None, needs_predictions=False):
    """Create heatmaps showing model preference."""
    print("\n" + "="*60)
    print("  PREFERENCE HEATMAPS")
    print("="*60)

    context_scaled = scaler.transform(context_data[CONTEXT_FEATURES].values)

    # Get NN predictions based on model type
    if needs_predictions and predictions is not None:
        nn_weights = weight_model.predict([predictions, context_scaled], verbose=0)
    else:
        nn_weights = weight_model.predict(context_scaled, verbose=0)

    dominant_model = np.argmax(nn_weights, axis=1)

    # Feature pairs using pointwise values
    feature_pairs = [
        ('Current_inv', 'SOC_Calc', 'Current [A]', 'SOC [%]'),
        ('Current_inv', 'Voltage', 'Current [A]', 'Voltage [V]'),
        ('SOC_Calc', 'Temp', 'SOC [%]', 'Temperature [°C]'),
        ('Voltage', 'Temp', 'Voltage [V]', 'Temperature [°C]'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    from matplotlib.colors import ListedColormap
    n_models = len(BASE_MODEL_NAMES)

    model_color_map = {
        'KF': '#8c564b',       # Brown
        'ANN': '#1f77b4',      # Azzurro (blue)
        'CNN': '#7fcdff',      # Celeste (light blue)
        'LSTM': '#d35400',     # Arancione scuro (dark orange)
        'CatBoost': '#f39c12', # Giallo/arancione (yellow-orange)
        'TFT': '#9b59b6',      # Violetto (purple)
    }
    custom_colors = [model_color_map.get(name, '#7f7f7f') for name in BASE_MODEL_NAMES]
    cmap = ListedColormap(custom_colors)

    for idx, (feat1, feat2, label1, label2) in enumerate(feature_pairs):
        ax = axes[idx]

        x = context_data[feat1].values
        y = context_data[feat2].values

        x_bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 20)
        y_bins = np.linspace(np.percentile(y, 1), np.percentile(y, 99), 20)

        heatmap = np.full((len(y_bins)-1, len(x_bins)-1), np.nan)

        for i in range(len(y_bins)-1):
            for j in range(len(x_bins)-1):
                mask = (x >= x_bins[j]) & (x < x_bins[j+1]) & \
                       (y >= y_bins[i]) & (y < y_bins[i+1])

                if mask.sum() >= 5:
                    models_in_cell = dominant_model[mask]
                    counts = np.bincount(models_in_cell, minlength=len(BASE_MODEL_NAMES))
                    heatmap[i, j] = np.argmax(counts)

        im = ax.imshow(heatmap, origin='lower', aspect='auto', cmap=cmap,
                      extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                      vmin=0, vmax=len(BASE_MODEL_NAMES)-1,
                      interpolation='nearest')

        ax.set_xlabel(label1, fontsize=11)
        ax.set_ylabel(label2, fontsize=11)
        ax.set_title(f'Preferred Model: {label1} vs {label2}', fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, ticks=range(len(BASE_MODEL_NAMES)))
        cbar.ax.set_yticklabels(BASE_MODEL_NAMES)

    plt.suptitle('Model Preference Heatmaps (Pointwise Values)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'preference_heatmaps.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'preference_heatmaps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: preference_heatmaps.pdf")

    # Create model distribution plot
    print("  Creating model distribution summary...")
    model_counts = np.bincount(dominant_model, minlength=len(BASE_MODEL_NAMES))
    model_pcts = model_counts / model_counts.sum() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(BASE_MODEL_NAMES, model_pcts,
                  color=[model_color_map.get(name, '#7f7f7f') for name in BASE_MODEL_NAMES],
                  edgecolor='black', linewidth=1)

    for bar, pct, count in zip(bars, model_pcts, model_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%\n({count:,})', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Selection Frequency [%]', fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_title('Model Selection Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(model_pcts) * 1.15)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: model_distribution.pdf")

    return dominant_model, model_pcts


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("  ENSEMBLE META-MODEL KNOWLEDGE EXTRACTION")
    print(f"  Ensemble Type: {ENSEMBLE_TYPE}")
    print(f"  Model Subdir: {MODEL_SUBDIR if MODEL_SUBDIR else 'default (current models)'}")
    print("="*70)

    script_dir = Path(__file__).parent

    if MODEL_SUBDIR:
        model_dir = script_dir / 'nn_training' / 'models' / MODEL_SUBDIR
        output_dir = script_dir / 'results' / f'ensemble_knowledge_{MODEL_SUBDIR}'
    else:
        model_dir = script_dir / 'nn_training' / 'models'
        output_dir = script_dir / 'results' / 'ensemble_knowledge'
    output_dir.mkdir(parents=True, exist_ok=True)

    if ENSEMBLE_TYPE == 'dynamic':
        model_file = model_dir / 'ensemble_dynamic_model.keras'
    else:
        model_file = model_dir / 'ensemble_selector_model.keras'

    if not model_file.exists():
        print(f"\nERROR: Model not found: {model_file}")
        print("Please train the ensemble model first using ensemble_train.py")
        return

    full_model, scaler = load_ensemble_model(model_dir, ENSEMBLE_TYPE)
    weight_model, needs_predictions = extract_weight_model(full_model, ENSEMBLE_TYPE)

    print(f"\nWeight model extracted:")
    print(f"  Input shape: {weight_model.input_shape}")
    print(f"  Output shape: {weight_model.output_shape}")
    print(f"  Needs predictions: {needs_predictions}")

    # Verify scaler matches our features
    print(f"\n  Scaler expects {scaler.n_features_in_} features")
    print(f"  We have {len(CONTEXT_FEATURES)} CONTEXT_FEATURES")
    if scaler.n_features_in_ != len(CONTEXT_FEATURES):
        print(f"  WARNING: Feature count mismatch!")
        print(f"  Expected features: {CONTEXT_FEATURES}")

    global BASE_MODEL_NAMES
    n_outputs = weight_model.output_shape[-1]
    if n_outputs == 6:
        BASE_MODEL_NAMES = BASE_MODEL_NAMES_6
        print(f"  Detected 6 base models (with KF): {BASE_MODEL_NAMES}")
    elif n_outputs == 5:
        BASE_MODEL_NAMES = BASE_MODEL_NAMES_5
        print(f"  Detected 5 base models (no KF): {BASE_MODEL_NAMES}")
    else:
        raise ValueError(f"Unexpected number of outputs: {n_outputs}. Expected 5 or 6.")

    results_dir = script_dir / 'results'

    # For Selector: try to load REAL predictions first
    predictions = None
    context_data = None

    if needs_predictions:
        print("\n" + "="*60)
        print("  LOADING REAL PREDICTIONS FOR SELECTOR ANALYSIS")
        print("="*60)

        predictions, context_data = load_real_predictions(
            results_dir, BASE_MODEL_NAMES, subsample_rate=10
        )

        if predictions is not None:
            print(f"\n  Successfully loaded REAL predictions!")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Context samples: {len(context_data):,}")
        else:
            print(f"\n  Could not load real predictions, falling back to synthetic...")
            # Fallback to loading context separately
            real_data = load_real_context_data(script_dir / 'data', results_dir)
            if real_data is not None:
                context_data = real_data
                soc_values = context_data['SOC_Calc'].values if 'SOC_Calc' in context_data.columns else None
                predictions = generate_synthetic_predictions(len(context_data), n_models=n_outputs, soc_values=soc_values)
                print(f"  Using synthetic predictions: {predictions.shape}")
            else:
                context_data = generate_synthetic_context_data(n_samples=50000)
                predictions = generate_synthetic_predictions(len(context_data), n_models=n_outputs)
                print(f"  Using fully synthetic data")
    else:
        # Dynamic model: just load context data
        real_data = load_real_context_data(script_dir / 'data', results_dir)
        if real_data is not None:
            context_data = real_data
            print(f"\nUsing real context data: {len(context_data):,} samples")
            if 'scenario' in context_data.columns:
                n_scenarios = context_data['scenario'].nunique()
                print(f"  From {n_scenarios} scenarios")
        else:
            context_data = generate_synthetic_context_data(n_samples=50000)
            print(f"\nUsing synthetic context data: {len(context_data):,} samples")
            print("  WARNING: Run main.py first to generate real data!")

    # Run analyses
    tree, tree_rules, fidelity, tree_importance = train_surrogate_tree(
        weight_model, context_data, scaler, output_dir, ENSEMBLE_TYPE,
        predictions=predictions, needs_predictions=needs_predictions,
        include_predictions_in_tree=needs_predictions)  # Include predictions for Selector
    tree_importance.to_excel(output_dir / 'tree_feature_importance.xlsx', index=False)

    rules, single_feature_prefs = extract_explicit_rules(
        weight_model, context_data, scaler, output_dir,
        predictions=predictions, needs_predictions=needs_predictions)

    create_partial_dependence_plots(weight_model, context_data, scaler, output_dir,
                                     predictions=predictions, needs_predictions=needs_predictions)

    dominant_model, model_pcts = create_preference_heatmaps(
        weight_model, context_data, scaler, output_dir,
        predictions=predictions, needs_predictions=needs_predictions)

    # Summary
    print("\n" + "="*70)
    print("  KNOWLEDGE EXTRACTION COMPLETE")
    print("="*70)
    print(f"\n  Output directory: {output_dir}")
    print(f"\n  Generated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"    - {f.name}")

    print(f"\n  Key findings:")
    print(f"    - Surrogate tree fidelity: {fidelity:.1f}%")
    print(f"    - Total rules extracted: {len(rules)}")
    print(f"    - Single-feature rules: {sum(1 for r in rules if r['type'] == 'single')}")
    print(f"    - Interaction rules: {sum(1 for r in rules if r['type'] == 'interaction')}")

    print(f"\n  Model selection distribution:")
    for i, name in enumerate(BASE_MODEL_NAMES):
        print(f"    - {name}: {model_pcts[i]:.1f}%")

    print(f"\n  Top features (from surrogate tree):")
    for _, row in tree_importance.head(3).iterrows():
        print(f"    - {row['Feature']}: {row['Importance']:.3f}")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()