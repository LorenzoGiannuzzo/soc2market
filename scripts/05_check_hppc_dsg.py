"""
Diagnose the discharge HPPC replay: the plant empties after about 2 h of a 24 h test.

Prints the structure of the workbook (columns, sampling, current range) and compares, hour
by hour, the charge obtained by integrating the measured current with the change of the
SOC_true column. If the two disagree, the time base or the current column is being read
wrongly; if they agree, the problem is in the plant.
"""

import numpy as np
import pandas as pd

from soc2market.estimation import estimator_bridge as eb

FILES = ("HPPC_25degree_DSG.xlsx", "HPPC_25degree_CHG.xlsx")
CAPACITY_AH = 280.0


def main():
    for name in FILES:
        df = pd.read_excel(eb.ESTIMATOR_WORKDIR / "data" / name, skiprows=2).drop(index=0)
        print(f"\n=== {name} ===")
        print("columns:", list(df.columns))
        print(df.head(3).to_string())

        t = pd.to_numeric(df["Step(s)"], errors="coerce").to_numpy(float)
        i = pd.to_numeric(df["Current_inv(A)"], errors="coerce").to_numpy(float)
        soc = pd.to_numeric(df["SOC_true(%)"], errors="coerce").to_numpy(float)
        d = np.diff(t)
        print(f"rows {len(df)}, Step(s) from {np.nanmin(t):.1f} to {np.nanmax(t):.1f}")
        print(f"Step(s) differences: min {np.nanmin(d):.3f}, median {np.nanmedian(d):.3f}, "
              f"max {np.nanmax(d):.3f}, negative {int((d < 0).sum())}, zero {int((d == 0).sum())}")
        print(f"current: min {np.nanmin(i):.1f} A, max {np.nanmax(i):.1f} A, "
              f"share of rows with |I| > 1 A {np.mean(np.abs(i) > 1):.3f}")

        #Lorenzo Giannuzzo: same time base as 04_validate_plant.py
        dt = np.diff(t, prepend=np.nan)
        med = np.nanmedian(dt[1:])
        dt[0] = med
        dt[~np.isfinite(dt) | (dt <= 0)] = med
        ah = np.cumsum(np.nan_to_num(i) * dt) / 3600.0
        hours = np.cumsum(dt) / 3600.0

        print("\n hour | integrated charge [Ah] | SOC_true change x capacity [Ah] | SOC_true [%]")
        for h in range(0, int(hours[-1]) + 1, 2):
            k = min(np.searchsorted(hours, h), len(hours) - 1)
            print(f"{hours[k]:5.1f} | {ah[k]:22.2f} | {(soc[k] - soc[0]) / 100 * CAPACITY_AH:31.2f} | {soc[k]:8.2f}")


if __name__ == "__main__":
    main()
