# Provenance of external/soc_estimators

Source repository: Huawei_repo (SOC_prediction)
Source commit: <paste revision number here>
Copied on: 2026-10-05
Uncommitted changes in source at copy time: no

## Contents
- Top-level .py files from SOC_prediction/
- nn_training/ (.py files only; trained models moved to data/private/huawei_workdir/nn_training/models/)
- parameters_optimization/ (.py files only)

## Artifacts (not versioned, stored in data/private/)
- huawei_workdir/data/: Cha_Dis_OCV_SOC_Data.xlsx, HPPC_25degree_CHG.xlsx, HPPC_25degree_DSG.xlsx
- huawei_workdir/nn_training/models/: trained ANN, CNN, LSTM, CatBoost, TFT and ensemble models with scalers
- soc_regression/: scenario S4_DAY1to8 and published result sheets for regression testing

## Known issues (kept as-is to reproduce published results)
- RC_model_DCH_module.two_RC_model: misplaced parenthesis puts the second RC term inside the first exponential
- KalmanFilter_module_all_data: Gaussian current denoising is applied to the whole series (non-causal)

## Local modifications
None.

- KalmanFilter_module_all_data: the measurement model predicts V = OCV - R0*I - V1 - V2 with
  positive current charging the cell, whereas the HPPC data follow V = OCV + R0*I + V1 + V2
  (scripts/07_check_ekf_sign.py: RMSE 28 vs 61 mV in charge, 51 vs 130 mV in discharge).
  Kept as published; the study evaluates the estimators without modification.