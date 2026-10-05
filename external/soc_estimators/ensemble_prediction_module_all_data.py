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
Ensemble Prediction Module for Battery SOC Analysis
================================================================================
Uses trained ensemble meta-models to make predictions by combining
outputs from all base models (KF, ANN, CNN, LSTM, CatBoost, TFT).

Supports 4 ensemble types:
- 'stacking': XGBoost meta-learner
- 'neural': Neural network meta-learner
- 'dynamic': Dynamic weighting network (soft weights)
- 'selector': Model selector network (hard selection - picks ONE model)
================================================================================
"""

import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf  # Keep for compatibility
from tensorflow import keras
from keras import Model
from joblib import load

# Enable unsafe deserialization (kept for backwards compatibility)
keras.config.enable_unsafe_deserialization()

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class Ensemble_BatteryAnalysis:
    """
    Ensemble-based battery SOC prediction.
    Combines predictions from multiple base models using a meta-learner.
    """

    def __init__(self, ensemble_type='selector', model_dir='nn_training/models'):
        """
        Initialize the Ensemble model.

        :param ensemble_type: 'stacking', 'neural', 'dynamic', or 'selector'
        :param model_dir: Directory containing ensemble model files
        """
        self.ensemble_type = ensemble_type
        self.model_dir = model_dir
        self.weight_model = None  # For dynamic ensemble weight extraction

        # Load ensemble config
        config_path = os.path.join(model_dir, 'ensemble_config.joblib')
        self.config = load(config_path)
        self.available_models = self.config['available_models']
        self.context_features = self.config['context_features']

        # Load appropriate ensemble model
        if ensemble_type == 'stacking':
            model_path = os.path.join(model_dir, 'ensemble_stacking_model.json')
            scaler_path = os.path.join(model_dir, 'ensemble_stacking_scaler.joblib')

            if HAS_XGBOOST:
                self.model = xgb.XGBRegressor()
                self.model.load_model(model_path)
            else:
                self.model = load(model_path.replace('.json', '.joblib'))
            self.scaler = load(scaler_path)

        elif ensemble_type == 'neural':
            model_path = os.path.join(model_dir, 'ensemble_neural_model.keras')
            scaler_path = os.path.join(model_dir, 'ensemble_neural_scaler.joblib')
            self.model = keras.models.load_model(model_path)
            self.scaler = load(scaler_path)

        elif ensemble_type == 'dynamic':
            model_path = os.path.join(model_dir, 'ensemble_dynamic_model.keras')
            scaler_path = os.path.join(model_dir, 'ensemble_dynamic_scaler.joblib')
            self.model = keras.models.load_model(model_path)
            self.context_scaler = load(scaler_path)

            # Rebuild weight_model for weight extraction
            self._rebuild_weight_model()

        elif ensemble_type == 'selector':
            model_path = os.path.join(model_dir, 'ensemble_selector_model.keras')
            scaler_path = os.path.join(model_dir, 'ensemble_selector_scaler.joblib')
            self.model = keras.models.load_model(model_path)
            self.context_scaler = load(scaler_path)

        else:
            raise ValueError(f"Unknown ensemble_type: {ensemble_type}. "
                           f"Choose from: 'stacking', 'neural', 'dynamic', 'selector'")

        # Load base model predictors
        self._load_base_models(model_dir)

    def _rebuild_weight_model(self):
        """Rebuild the weight extraction model from the loaded dynamic model."""
        try:
            # After loading, use model.inputs instead of get_layer().input
            context_input = self.model.inputs[1]  # Second input is 'context'
            weights_output = self.model.get_layer('weights').output
            self.weight_model = Model(inputs=context_input, outputs=weights_output)
        except Exception as e:
            print(f"Warning: Could not rebuild weight_model: {e}")
            self.weight_model = None

    def _load_base_models(self, model_dir):
        """Load all base model predictors."""
        self.base_models = {}

        # Import base model classes
        from ANN_prediction_module_all_data import ANN_BatteryAnalysis
        from CNN_prediction_module_all_data import CNN_BatteryAnalysis
        from LSTM_prediction_module_all_data import LSTM_BatteryAnalysis
        from CATBOOST_prediction_module_all_data import CATBOOST_BatteryAnalysis
        from TFT_prediction_module_all_data import TFT_BatteryAnalysis

        model_classes = {
            'ANN': (ANN_BatteryAnalysis, 'my_model.keras', 'scaler.joblib'),
            'CNN': (CNN_BatteryAnalysis, 'cnn_model.keras', 'cnn_scaler.joblib'),
            'LSTM': (LSTM_BatteryAnalysis, 'lstm_model.keras', 'lstm_scaler.joblib'),
            'CatBoost': (CATBOOST_BatteryAnalysis, 'catboost_model.cbm', 'catboost_scaler.joblib'),
            'TFT': (TFT_BatteryAnalysis, 'tft_model.ckpt', 'tft_scaler.joblib'),
        }

        for model_name in self.available_models:
            if model_name == 'KF':
                continue  # KF is handled separately

            if model_name in model_classes:
                cls, model_file, scaler_file = model_classes[model_name]
                model_path = os.path.join(model_dir, model_file)
                scaler_path = os.path.join(model_dir, scaler_file)

                if os.path.exists(model_path):
                    if model_name == 'TFT':
                        config_path = os.path.join(model_dir, 'tft_config.joblib')
                        self.base_models[model_name] = cls(model_path, scaler_path, config_path)
                    elif model_name == 'CatBoost':
                        features_path = os.path.join(model_dir, 'catboost_feature_columns.joblib')
                        self.base_models[model_name] = cls(model_path, scaler_path, features_path)
                    elif model_name in ['CNN', 'LSTM']:
                        self.base_models[model_name] = cls(model_path, scaler_path, sequence_length=10)
                    else:
                        self.base_models[model_name] = cls(model_path, scaler_path)

    def _preprocess_data(self, df):
        """Add delta features to dataframe."""
        df = df.copy()

        if 'Current_Voltage_Ratio' not in df.columns:
            df['Current_Voltage_Ratio'] = (df['Current_inv'] / df['Voltage']).fillna(0)

        if 'Delta_Current' not in df.columns:
            df['Delta_Current'] = df['Current_inv'].diff().fillna(0)

        if 'Delta_Voltage' not in df.columns:
            df['Delta_Voltage'] = df['Voltage'].diff().fillna(0)

        if 'Delta_Voltage_Current_Ratio' not in df.columns:
            df['Delta_Voltage_Current_Ratio'] = (
                df['Delta_Voltage'] / df['Delta_Current'].replace(0, 1)
            ).fillna(0)

        if 'Delta_SoC_Calc' not in df.columns:
            df['Delta_SoC_Calc'] = df['SOC_Calc'].diff().fillna(0)

        if 'Delta_Temperature' not in df.columns:
            df['Delta_Temperature'] = df['Temp'].diff().fillna(0)

        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        return df

    def _get_base_predictions(self, df):
        """Get predictions from all base models."""
        predictions = {}

        for model_name in self.available_models:
            if model_name == 'KF':
                predictions['KF'] = df['SOC_Calc'].values
            elif model_name in self.base_models:
                result = self.base_models[model_name].make_predictions(df.copy())
                predictions[model_name] = result['Predicted_SOC'].values

        return predictions

    def make_predictions(self, bat_data):
        """
        Make ensemble predictions on battery data.

        :param bat_data: DataFrame with battery data (output of Kalman Filter)
        :return: DataFrame with added 'Predicted_SOC' column
        """
        df = self._preprocess_data(bat_data)

        # Get predictions from all base models
        predictions = self._get_base_predictions(df)

        # Build prediction matrix
        pred_matrix = np.column_stack([predictions[m] for m in self.available_models])

        # Get context features
        context_cols = [c for c in self.context_features if c in df.columns]
        context_matrix = df[context_cols].values

        # Handle NaN values
        nan_mask = np.isnan(pred_matrix).any(axis=1) | np.isnan(context_matrix).any(axis=1)

        # Make predictions based on ensemble type
        if self.ensemble_type in ['stacking', 'neural']:
            X = np.hstack([pred_matrix, context_matrix])
            X_scaled = self.scaler.transform(X)

            if self.ensemble_type == 'stacking':
                y_pred = self.model.predict(X_scaled)
            else:
                y_pred = self.model.predict(X_scaled, verbose=0).flatten()

        elif self.ensemble_type == 'dynamic':
            context_scaled = self.context_scaler.transform(context_matrix)
            y_pred = self.model.predict([pred_matrix, context_scaled], verbose=0).flatten()

        elif self.ensemble_type == 'selector':
            # HARD SELECTION: pick ONE model per timestep
            context_scaled = self.context_scaler.transform(context_matrix)

            # Get selection probabilities
            probs = self.model.predict([pred_matrix, context_scaled], verbose=0)

            # Hard selection: pick model with highest probability
            selected_models = np.argmax(probs, axis=1)

            # Store for visualization
            self._last_selected_models = selected_models
            self._last_selection_probs = probs

            # Get prediction from selected model for each sample
            n_samples = len(selected_models)
            y_pred = np.zeros(n_samples)
            for i in range(n_samples):
                y_pred[i] = pred_matrix[i, selected_models[i]]

        # Handle NaN positions (use KF as fallback)
        y_pred[nan_mask] = df['SOC_Calc'].values[nan_mask]

        bat_data['Predicted_SOC'] = y_pred

        return bat_data

    def get_model_weights(self, bat_data):
        """
        Get the dynamic weights for each base model.

        For 'dynamic': returns soft weights (sum to 1)
        For 'selector': returns ONE-HOT weights (hard selection: 0 or 1)

        :param bat_data: DataFrame with battery data
        :return: DataFrame with weights for each model at each timestep
        """
        if self.ensemble_type == 'dynamic':
            if self.weight_model is None:
                raise ValueError("Weight model not available.")

            df = self._preprocess_data(bat_data)
            context_cols = [c for c in self.context_features if c in df.columns]
            context_matrix = df[context_cols].values
            context_scaled = self.context_scaler.transform(context_matrix)

            weights = self.weight_model.predict(context_scaled, verbose=0)
            return pd.DataFrame(weights, columns=self.available_models)

        elif self.ensemble_type == 'selector':
            df = self._preprocess_data(bat_data)
            predictions = self._get_base_predictions(df)
            pred_matrix = np.column_stack([predictions[m] for m in self.available_models])

            context_cols = [c for c in self.context_features if c in df.columns]
            context_matrix = df[context_cols].values
            context_scaled = self.context_scaler.transform(context_matrix)

            # Get selection probabilities
            probs = self.model.predict([pred_matrix, context_scaled], verbose=0)

            # Convert to ONE-HOT (hard selection): only selected model gets 1, rest get 0
            selected_indices = np.argmax(probs, axis=1)
            n_samples = len(selected_indices)
            n_models = len(self.available_models)

            # Create one-hot weights
            weights = np.zeros((n_samples, n_models))
            weights[np.arange(n_samples), selected_indices] = 1.0

            return pd.DataFrame(weights, columns=self.available_models)

        else:
            raise ValueError(f"Weights not available for ensemble_type '{self.ensemble_type}'")

    def get_selection_probabilities(self, bat_data):
        """
        Get the soft selection probabilities (only for 'selector' type).
        Useful for analysis - shows confidence of selection, not just the final choice.

        :param bat_data: DataFrame with battery data
        :return: DataFrame with probabilities for each model at each timestep
        """
        if self.ensemble_type != 'selector':
            raise ValueError("get_selection_probabilities() only available for 'selector' ensemble type")

        df = self._preprocess_data(bat_data)
        predictions = self._get_base_predictions(df)
        pred_matrix = np.column_stack([predictions[m] for m in self.available_models])

        context_cols = [c for c in self.context_features if c in df.columns]
        context_matrix = df[context_cols].values
        context_scaled = self.context_scaler.transform(context_matrix)

        # Get selection probabilities (soft)
        probs = self.model.predict([pred_matrix, context_scaled], verbose=0)
        return pd.DataFrame(probs, columns=self.available_models)

    def get_selected_models(self, bat_data):
        """
        Get which model was selected for each timestep (only for 'selector' type).

        :param bat_data: DataFrame with battery data
        :return: array of model indices and DataFrame with model names
        """
        if self.ensemble_type != 'selector':
            raise ValueError("get_selected_models() only available for 'selector' ensemble type")

        df = self._preprocess_data(bat_data)
        predictions = self._get_base_predictions(df)
        pred_matrix = np.column_stack([predictions[m] for m in self.available_models])

        context_cols = [c for c in self.context_features if c in df.columns]
        context_matrix = df[context_cols].values
        context_scaled = self.context_scaler.transform(context_matrix)

        probs = self.model.predict([pred_matrix, context_scaled], verbose=0)
        selected_indices = np.argmax(probs, axis=1)
        selected_names = [self.available_models[i] for i in selected_indices]

        result_df = pd.DataFrame({
            'Selected_Model_Index': selected_indices,
            'Selected_Model_Name': selected_names,
        })

        return selected_indices, result_df

    def get_selection_statistics(self, bat_data):
        """
        Get statistics about model selection (only for 'selector' type).

        :param bat_data: DataFrame with battery data
        :return: dict with selection counts and percentages
        """
        if self.ensemble_type != 'selector':
            raise ValueError("get_selection_statistics() only available for 'selector' ensemble type")

        selected_indices, _ = self.get_selected_models(bat_data)

        unique, counts = np.unique(selected_indices, return_counts=True)
        total = len(selected_indices)

        stats = {}
        for idx, count in zip(unique, counts):
            model_name = self.available_models[idx]
            stats[model_name] = {
                'count': int(count),
                'percentage': count / total * 100
            }

        # Add zero counts for models that were never selected
        for model_name in self.available_models:
            if model_name not in stats:
                stats[model_name] = {'count': 0, 'percentage': 0.0}

        return stats