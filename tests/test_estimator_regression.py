"""
Regression test for the frozen pre-trained SOC estimators.

The copied code is run on one benchmark scenario and its output is compared with the
reference result sheet. Passing this test proves that code, trained artifacts and Python
environment reproduce the reference estimates.
"""

import numpy as np
import pandas as pd
import pytest

from soc2market.estimation import estimator_bridge as eb

SCENARIO = "S4_DAY1to8"
REGRESSION_DIR = eb.REPO_ROOT / "data" / "private" / "soc_regression"
SCENARIO_FILE = REGRESSION_DIR / f"{SCENARIO}.xlsx"

#Lorenzo Giannuzzo: tolerances in SOC percentage points; the EKF is pure NumPy and must match
# almost exactly, while TensorFlow and PyTorch inference on CPU can differ in the last digits
EKF_ATOL = 1e-9
MODEL_ATOL = 1e-4

#Lorenzo Giannuzzo: the skip message lists the missing paths, so a wrong folder is easy to spot
_MISSING = [str(p) for p in (SCENARIO_FILE, eb.MODEL_DIR, eb.ESTIMATOR_WORKDIR / "data") if not p.exists()]
pytestmark = pytest.mark.skipif(
    bool(_MISSING),
    reason="Private estimator artifacts not found: " + ", ".join(_MISSING),
)


@pytest.fixture(scope="module")
def ekf_output():
    """Run the Kalman filter once and share its output among all tests of this module."""
    df = eb.load_scenario(SCENARIO_FILE)
    return eb.run_ekf(df, SCENARIO)


def _reference(model_name, column):
    """Read one column of the reference result sheet (sheet names are cut at 31 characters)."""
    sheet = pd.read_excel(REGRESSION_DIR / f"{model_name}.xlsx", sheet_name=SCENARIO[:31])
    return sheet[column].to_numpy(dtype=float)


def _assert_close(actual, expected, atol, label):
    assert actual.shape == expected.shape, f"{label}: length {actual.shape} vs reference {expected.shape}"
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), f"{label}: NaN positions differ"
    max_diff = float(np.nanmax(np.abs(actual - expected)))
    print(f"\n{label}: max absolute difference = {max_diff:.3e} SOC points")
    assert max_diff <= atol, f"{label}: max difference {max_diff:.3e} exceeds {atol:.0e}"


def test_ekf_reproduces_reference_soc(ekf_output):
    expected = _reference("ANN", "SOC_Calc")
    actual = ekf_output["SOC_Calc"].to_numpy(dtype=float)
    _assert_close(actual, expected, EKF_ATOL, "EKF")


@pytest.mark.parametrize("model_name", [
    "ANN",
    "CNN",
    "LSTM",
    "CatBoost",
    pytest.param("TFT", marks=pytest.mark.slow),
    pytest.param("Ensemble", marks=pytest.mark.slow),
])
def test_model_reproduces_reference_soc(ekf_output, model_name):
    predicted = eb.run_model(ekf_output, model_name)
    expected = _reference(model_name, "Predicted_SOC")
    actual = predicted["Predicted_SOC"].to_numpy(dtype=float)
    _assert_close(actual, expected, MODEL_ATOL, model_name)
