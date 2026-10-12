"""
Estimate the real sampling interval of the HPPC workbooks from their own SOC_true column.

In the discharge workbook the charge obtained by integrating the current at one row per
second is about 11.9 times the change of SOC_true, while the charge workbook agrees to
within 0.5 %. For every row with current flowing, the interval implied by SOC_true is

    dt = (delta SOC_true / 100) * capacity * 3600 / |I|

and it is summarised by current band, to see whether the factor is uniform (a different
logging rate) or depends on the phase of the test.
"""

import numpy as np
import pandas as pd

from soc2market.estimation import estimator_bridge as eb

FILES = ("HPPC_25degree_DSG.xlsx", "HPPC_25degree_CHG.xlsx")
CAPACITY_AH = 280.0
BANDS = ((5.0, 100.0), (100.0, 250.0), (250.0, 1000.0))


def main():
    for name in FILES:
        df = pd.read_excel(eb.ESTIMATOR_WORKDIR / "data" / name, skiprows=2).drop(index=0)
        i = pd.to_numeric(df["Current_inv(A)"], errors="coerce").to_numpy(float)
        soc = pd.to_numeric(df["SOC_true(%)"], errors="coerce").to_numpy(float)
        #Lorenzo Giannuzzo: the SOC change between row k and k+1 is produced by the current of row k+1
        dsoc = np.diff(soc)
        cur = i[1:]
        implied = np.abs(dsoc) / 100.0 * CAPACITY_AH * 3600.0 / np.maximum(np.abs(cur), 1e-9)
        print(f"\n=== {name} ===")
        print(" current band [A] | rows  | implied dt [s]: p10    median   p90")
        for lo, hi in BANDS:
            mask = (np.abs(cur) > lo) & (np.abs(cur) <= hi) & np.isfinite(implied)
            if mask.sum() == 0:
                continue
            q = np.percentile(implied[mask], [10, 50, 90])
            print(f" {lo:5.0f} to {hi:5.0f}   | {mask.sum():5d} | {q[0]:10.4f} {q[1]:8.4f} {q[2]:8.4f}")
        flowing = np.abs(cur) > 5.0
        print(f" all rows with current: total implied time {implied[flowing].sum() / 3600:.2f} h "
              f"against {flowing.sum() / 3600:.2f} h at one row per second")


if __name__ == "__main__":
    main()
