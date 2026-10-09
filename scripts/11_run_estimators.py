"""
Run the published SOC estimators in closed loop on the reference duty cycles.

Each representative week is concatenated into one continuous series, so that the state of the
EKF carries from one day to the next as in operation, and every estimator is applied with the
published settings (EKF parameters, current denoising, start from the true SOC at the beginning
of the week). One file per week and estimator is written to paths.LOOP_DIR (folder of the MSD mode
set in paths.py); existing files are skipped, so an interrupted run resumes where it stopped.
"""

import time

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.estimation import huawei_bridge as hb

DUTY_DIR = paths.DUTY_DIR
OUT_DIR = paths.LOOP_DIR
WEEKS = ("winter", "spring", "summer", "autumn")
#Lorenzo Giannuzzo: "EKF" is the filter alone; the others are EKF + model, as published
MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")
#Lorenzo Giannuzzo: quick check before the full run (a few minutes instead of three hours)
QUICK = False
if QUICK:
    MODELS = ("EKF", "ANN", "CNN")


def load_week(name):
    """Concatenate the daily files of one week into the table the estimators read."""
    files = sorted((DUTY_DIR / name).glob("day_*.npz"))
    parts, day_index = [], []
    for k, f in enumerate(files):
        with np.load(f) as z:
            parts.append(pd.DataFrame({"Current_inv": z["current_a"].astype(float),
                                       "Voltage": z["voltage_v"].astype(float),
                                       "Temp": z["temp_c"].astype(float),
                                       "SOC_true": z["soc_true_pct"].astype(float)}))
            day_index.append(np.full(len(parts[-1]), k))
    table = pd.concat(parts, ignore_index=True)
    table.index = table.index + 1
    return table, np.concatenate(day_index)


def error_stats(soc_hat, soc_true):
    e = np.asarray(soc_hat, float) - np.asarray(soc_true, float)
    return dict(rmse_pts=float(np.sqrt(np.mean(e ** 2))), mae_pts=float(np.mean(np.abs(e))),
                p95_pts=float(np.percentile(np.abs(e), 95)), max_pts=float(np.max(np.abs(e))),
                bias_pts=float(np.mean(e)))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for week in WEEKS:
        table, day_index = load_week(week)
        soc_true = table["SOC_true"].to_numpy()
        ekf_out = None
        for model in MODELS:
            out_file = OUT_DIR / f"{week}_{model}.npz"
            if out_file.exists():
                with np.load(out_file) as z:
                    soc_hat, seconds = z["soc_hat"], float(z["seconds"])
                print(f"{week:7s} {model:9s} cached")
            else:
                t0 = time.perf_counter()
                if ekf_out is None:
                    ekf_out = hb.run_ekf(table.copy(), week)
                if model == "EKF":
                    soc_hat = ekf_out["SOC_Calc"].to_numpy(float)
                else:
                    soc_hat = hb.run_model(ekf_out.copy(), model)["Predicted_SOC"].to_numpy(float)
                seconds = time.perf_counter() - t0
                np.savez(out_file, soc_hat=soc_hat.astype(np.float64), soc_true=soc_true.astype(np.float64),
                         day_index=day_index, seconds=seconds)
                print(f"{week:7s} {model:9s} done in {seconds / 60:.1f} min", flush=True)
            rows.append(dict(week=week, model=model, minutes=seconds / 60, **error_stats(soc_hat, soc_true)))

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    table.to_csv(OUT_DIR / "summary.csv", index=False)
    overall = table.groupby("model")[["rmse_pts", "mae_pts", "p95_pts", "max_pts", "bias_pts"]].mean()
    print("\nmean over the four weeks:\n" + overall.to_string(float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
