"""
Check the sign of the ohmic term in the measurement model of the pre-trained EKF.

In the data, positive current charges the cell (the charge HPPC pulses are positive and the
voltage jumps up). The frozen EKF predicts the terminal voltage as OCV - R0*I - V1 - V2, which
with that convention places the voltage BELOW the OCV while charging. Here the measured
voltage under current is compared with both OCV(SOC_true) + R0*I and OCV(SOC_true) - R0*I,
using the OCV branch of the direction of the current, as the EKF does; the sign that fits the
data is the physical one.
"""

import numpy as np
import pandas as pd

from soc2market.plant.cell import CellParams
from soc2market.plant.identify import read_hppc_workbook

MIN_CURRENT_A = 50.0
SOC_RANGE = (0.05, 0.95)


def main():
    p = CellParams.from_dir()
    rows = []
    for name in ("HPPC_25degree_CHG.xlsx", "HPPC_25degree_DSG.xlsx"):
        data, _, _ = read_hppc_workbook(name)
        i, v, soc = data["i"].to_numpy(), data["v"].to_numpy(), data["soc"].to_numpy()
        mask = (np.abs(i) > MIN_CURRENT_A) & (soc > SOC_RANGE[0]) & (soc < SOC_RANGE[1])
        for direction, sel, table, branch in (("charge", i > 0, p.chg_table, p.ocv_chg),
                                              ("discharge", i < 0, p.dch_table, p.ocv_dch)):
            m = mask & sel
            if m.sum() == 0:
                continue
            ocv = np.interp(soc[m], p.ocv_soc, branch)
            r0 = np.interp(soc[m], table["soc"], table["r0"])
            #Lorenzo Giannuzzo: only the ohmic term is compared; the RC terms would add to it with the same sign
            plus = ocv + r0 * i[m]
            minus = ocv - r0 * i[m]
            rows.append(dict(workbook=name, direction=direction, samples=int(m.sum()),
                             mean_v_minus_ocv_mv=1e3 * np.mean(v[m] - ocv),
                             rmse_plus_mv=1e3 * np.sqrt(np.mean((v[m] - plus) ** 2)),
                             rmse_minus_mv=1e3 * np.sqrt(np.mean((v[m] - minus) ** 2))))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nIf rmse_plus is clearly smaller than rmse_minus in both directions, the data follow "
          "V = OCV + R0*I and the minus sign in the frozen EKF is inverted.")


if __name__ == "__main__":
    main()
