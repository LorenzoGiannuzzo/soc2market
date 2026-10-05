"""
Bridge to the frozen Huawei SOC estimation code stored in external/soc_estimators.

The original pipeline (SOC_prediction/main.py) is reused as-is, so that the Kalman filter
settings (initial covariance P, process noise Q, measurement noise R, cell capacity) and the
model wrappers are exactly those that produced the published results. The original code
reads its data through paths relative to the working directory, therefore every call runs
inside data/private/huawei_workdir, which mirrors the original SOC_prediction layout.
"""

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#Lorenzo Giannuzzo: repository layout, resolved from this file so that it works from any working directory
REPO_ROOT = Path(__file__).resolve().parents[3]
SOC_CODE_DIR = REPO_ROOT / "external" / "soc_estimators"
HUAWEI_WORKDIR = REPO_ROOT / "data" / "private" / "huawei_workdir"
MODEL_DIR = HUAWEI_WORKDIR / "nn_training" / "models"

#Lorenzo Giannuzzo: same file names and settings used in SOC_prediction/main.py
SEQUENCE_LENGTH = 10
ENSEMBLE_TYPE = "selector"
MODEL_FILES = {
    "ANN": {"model": "my_model.keras", "scaler": "scaler.joblib"},
    "CNN": {"model": "cnn_model.keras", "scaler": "cnn_scaler.joblib"},
    "LSTM": {"model": "lstm_model.keras", "scaler": "lstm_scaler.joblib"},
    "CatBoost": {"model": "catboost_model.cbm", "scaler": "catboost_scaler.joblib",
                 "features": "catboost_feature_columns.joblib"},
    "TFT": {"model": "tft_model.ckpt", "scaler": "tft_scaler.joblib", "config": "tft_config.joblib"},
    "Ensemble": {"model": "ensemble_selector_model.keras"},
}

_HUAWEI_MAIN = None


def load_huawei_main():
    """Import SOC_prediction/main.py under a unique module name and return it."""
    global _HUAWEI_MAIN
    if _HUAWEI_MAIN is None:
        #Lorenzo Giannuzzo: the frozen modules import each other with flat imports,
        # so their folder must be on sys.path; the explicit file location avoids clashes
        # with any other module called "main" in the repository
        if str(SOC_CODE_DIR) not in sys.path:
            sys.path.insert(0, str(SOC_CODE_DIR))
        spec = importlib.util.spec_from_file_location("huawei_main", SOC_CODE_DIR / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HUAWEI_MAIN = module
    return _HUAWEI_MAIN


@contextlib.contextmanager
def huawei_cwd():
    """Temporarily run inside the mirrored SOC_prediction layout."""
    previous = Path.cwd()
    os.chdir(HUAWEI_WORKDIR)
    try:
        yield
    finally:
        os.chdir(previous)


def load_scenario(path):
    """Read a scenario workbook exactly as SOC_prediction/main.py does."""
    main = load_huawei_main()
    df = pd.read_excel(path, skiprows=2).iloc[1:]
    return main.standardize_columns(df)


def run_ekf(df, scenario_name, initial_soc=None):
    """
    Run the extended Kalman filter with the published settings.

    When initial_soc is None the filter starts from the first true SOC value, as in the
    published benchmark; the market simulation will instead pass an explicit initial value.
    """
    main = load_huawei_main()
    if initial_soc is None:
        initial_soc = float(df["SOC_true"].iloc[0])
        if np.isnan(initial_soc):
            initial_soc = 50.0
    with huawei_cwd():
        return main.run_kalman_filter(df.copy(), initial_soc, scenario_name)


def _optional_path(name):
    """Return the artifact path as a string if it exists, otherwise None (as in main.py)."""
    if name is None:
        return None
    path = MODEL_DIR / name
    return str(path) if path.exists() else None


def run_model(ekf_output, model_name):
    """Apply one hybrid estimator (EKF + model) to the Kalman filter output."""
    main = load_huawei_main()
    files = MODEL_FILES[model_name]
    model_path = str(MODEL_DIR / files["model"])
    #Lorenzo Giannuzzo: the working directory is switched also here because the TFT
    # prediction step can write lightning logs to the current folder
    with huawei_cwd():
        if model_name == "Ensemble":
            return main.run_model_prediction(
                ekf_output, "Ensemble", model_path, None,
                sequence_length=SEQUENCE_LENGTH, ensemble_type=ENSEMBLE_TYPE,
            )
        return main.run_model_prediction(
            ekf_output, model_name, model_path, str(MODEL_DIR / files["scaler"]),
            sequence_length=SEQUENCE_LENGTH,
            file_path_feature_columns=_optional_path(files.get("features")),
            file_path_config=_optional_path(files.get("config")),
        )
