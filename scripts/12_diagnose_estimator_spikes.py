"""
Diagnose the occasional absurd outputs of the ANN and CNN hybrids on the duty cycles.

Hypothesis: the feature Delta_Voltage_Current_Ratio = dV/dI (set to 0 when dI = 0) explodes on the
simulated measurements, because the simulated current sensor is never exactly zero at rest,
whereas in the laboratory data used for training the current is exactly zero during rests.
The script compares the measurement statistics and the feature range of the training-like
scenario S4 with those of the duty cycles, and lists the instants of the largest errors.
"""

import numpy as np
import pandas as pd

from soc2market.estimation import estimator_bridge as eb

DUTY_DIR = eb.REPO_ROOT / "results" / "duty_cycles"
LOOP_DIR = eb.REPO_ROOT / "results" / "closed_loop"
S4 = eb.REPO_ROOT / "data" / "private" / "soc_regression" / "S4_DAY1to8.xlsx"


def measurement_stats(label, current, voltage):
    di = np.diff(current)
    dv = np.diff(voltage)
    ratio = np.where(di != 0, dv / np.where(di != 0, di, 1.0), 0.0)
    nz_i = np.abs(di[di != 0])
    nz_v = np.abs(dv[dv != 0])
    q = np.percentile(np.abs(ratio), [50, 99, 99.9, 99.99])
    print(f"\n{label}")
    print(f"  rows {current.size}, current exactly zero {100 * np.mean(current == 0):.1f}%, "
          f"|I| < 1 A {100 * np.mean(np.abs(current) < 1):.1f}%")
    print(f"  dI exactly zero {100 * np.mean(di == 0):.1f}%, 0 < |dI| < 0.5 A {100 * np.mean((di != 0) & (np.abs(di) < 0.5)):.1f}%")
    print(f"  smallest non-zero |dI| {nz_i.min():.2e} A, smallest non-zero |dV| {nz_v.min():.2e} V")
    print(f"  |dV/dI| percentiles 50/99/99.9/99.99: {q[0]:.2e} {q[1]:.2e} {q[2]:.2e} {q[3]:.2e}, max {np.abs(ratio).max():.2e} V/A")
    return ratio


def main():
    s4 = eb.load_scenario(S4)
    measurement_stats("S4 laboratory scenario (training-like)", s4["Current_inv"].to_numpy(float), s4["Voltage"].to_numpy(float))

    #Lorenzo Giannuzzo: the winter week of the duty cycles, as fed to the estimators
    files = sorted((DUTY_DIR / "winter").glob("day_*.npz"))
    cur = np.concatenate([np.load(f)["current_a"].astype(float) for f in files])
    vol = np.concatenate([np.load(f)["voltage_v"].astype(float) for f in files])
    ratio = measurement_stats("duty cycles, winter week", cur, vol)

    for model in ("ANN", "CNN"):
        z = np.load(LOOP_DIR / f"winter_{model}.npz")
        err = np.abs(z["soc_hat"].astype(float) - z["soc_true"].astype(float))
        top = np.argsort(err)[::-1][:8]
        rows = []
        for k in sorted(top):
            rows.append(dict(index=int(k), soc_true=float(z["soc_true"][k]), soc_hat=float(z["soc_hat"][k]),
                             current_a=cur[k], dI_a=cur[k] - cur[k - 1] if k > 0 else np.nan,
                             dV_mv=1e3 * (vol[k] - vol[k - 1]) if k > 0 else np.nan,
                             ratio_v_per_a=ratio[k - 1] if k > 0 else np.nan))
        print(f"\nlargest errors of {model} (winter):")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print(f"  share of samples with error above 10 points: {100 * np.mean(err > 10):.3f}%")


if __name__ == "__main__":
    main()
