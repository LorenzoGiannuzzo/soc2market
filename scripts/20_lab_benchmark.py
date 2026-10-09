"""
Laboratory accuracy of the seven estimators, for the comparison with the closed-loop market runs.

The result sheets of the original pipeline stored in data/private/soc_regression (one workbook per
model, one sheet per laboratory test scenario, the same sheets used by the regression test) hold
the true SOC, the EKF estimate (SOC_Calc) and the estimate of each hybrid model or of the ensemble
(Predicted_SOC). The script reads every sheet and reports RMSE, MAE, 95th percentile and maximum of
the absolute error and the bias, per scenario and pooled over all laboratory samples. The EKF is
taken from the ANN workbook (all workbooks carry the same SOC_Calc, as checked by the regression test).
"""

import numpy as np
import pandas as pd

from soc2market import paths

LAB_DIR = paths.REPO_ROOT / "data" / "private" / "soc_regression"
OUT_DIR = paths.REPO_ROOT / "results" / "lab_benchmark"
MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")
TRUE_NAMES = ("SOC_true", "SOC_True", "SOC_true(%)", "SOC_True(%)")


def stats(e):
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    return dict(rmse_pts=float(np.sqrt(np.mean(e ** 2))), mae_pts=float(np.mean(np.abs(e))),
                p95_pts=float(np.percentile(np.abs(e), 95)), max_pts=float(np.max(np.abs(e))),
                bias_pts=float(np.mean(e)), n=int(e.size))


def errors_of(sheet, column):
    true_col = next((c for c in TRUE_NAMES if c in sheet.columns), None)
    if true_col is None or column not in sheet.columns:
        return None
    t = pd.to_numeric(sheet[true_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(sheet[column], errors="coerce").to_numpy(float)
    return y - t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, pooled = [], {m: [] for m in MODELS}
    for model in MODELS:
        book = "ANN" if model == "EKF" else model
        column = "SOC_Calc" if model == "EKF" else "Predicted_SOC"
        sheets = pd.read_excel(LAB_DIR / f"{book}.xlsx", sheet_name=None)
        if model == "EKF":
            print(f"{len(sheets)} laboratory scenarios: {list(sheets)}")
            print(f"columns of the first sheet: {list(next(iter(sheets.values())).columns)}")
        for name, sheet in sheets.items():
            e = errors_of(sheet, column)
            if e is None:
                print(f"  {model} / {name}: true SOC or {column} column not found, skipped")
                continue
            pooled[model].append(e)
            rows.append(dict(scenario=name, model=model, **stats(e)))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "per_scenario.csv", index=False)
    summary = pd.DataFrame({m: stats(np.concatenate(pooled[m])) for m in MODELS if pooled[m]}).T
    summary.to_csv(OUT_DIR / "pooled.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\nRMSE per scenario [percentage points]:")
    print(table.pivot(index="scenario", columns="model", values="rmse_pts")[[m for m in MODELS if pooled[m]]]
          .round(2).to_string())
    print("\npooled over all laboratory samples [percentage points]:")
    print(summary[["rmse_pts", "mae_pts", "p95_pts", "max_pts", "bias_pts", "n"]].round(2).to_string())


if __name__ == "__main__":
    main()