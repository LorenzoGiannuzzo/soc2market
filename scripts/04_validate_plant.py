"""
Validate the plant against measured data of the same cell.

The measured current is fed to the plant and the simulated terminal voltage is compared
with the measured one, on the two HPPC tests (from which the RC tables were identified)
and on the dynamic scenario S4_DAY1to8 (never used for identification). On S4 the
hysteresis rate, which the HPPC tests cannot identify, is swept and the best value reported.
"""

import numpy as np
import pandas as pd

from soc2market.estimation import huawei_bridge as hb
from soc2market.figures.style import legend_top, plt, save
from soc2market.plant.cell import REPO_ROOT, CellParams, CellState, simulate
from soc2market.plant.identify import read_hppc_workbook

FIG_DIR = REPO_ROOT / "results" / "plant"
SCENARIO = hb.REPO_ROOT / "data" / "private" / "soc_regression" / "S4_DAY1to8.xlsx"
GAMMAS = (3.0, 10.0, 30.0, 100.0, 300.0)

#Lorenzo Giannuzzo: during validation the plant must follow the measured current, so the
# voltage window is widened and no step is refused
REPLAY_LIMITS = dict(v_min=1.5, v_max=4.5)


def metrics(v_meas, v_sim):
    e = v_sim - v_meas
    return dict(rmse_mv=1e3 * np.sqrt(np.mean(e ** 2)), mae_mv=1e3 * np.mean(np.abs(e)), max_mv=1e3 * np.max(np.abs(e)))


def replay(params, data, h0, current, dt):
    state = CellState(soc=float(data["soc"].iloc[0]), h=h0, temp_c=params.ambient_c)
    return simulate(params, dt, state, current_a=current)


def main():
    base = CellParams.from_dir(**REPLAY_LIMITS)
    rows = []

    #Lorenzo Giannuzzo: HPPC tests, the charge test starts from the discharge branch and vice versa
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0))
    for ax, name, h0 in ((axes[0], "HPPC_25degree_CHG.xlsx", -1.0), (axes[1], "HPPC_25degree_DSG.xlsx", 1.0)):
        data, dt_s, source = read_hppc_workbook(name)
        print(f"{name}: time step {dt_s:.4f} s ({source})")
        data["dt"] = dt_s
        res = replay(base, data, h0, data["i"].to_numpy(), dt_s)
        m = metrics(data["v"].to_numpy(), res.voltage_v)
        m.update(case=name, gamma=base.hysteresis_gamma,
                 soc_end_error_pts=100 * (res.final_state.soc - data["soc"].iloc[-1]), refused=int(res.refused.sum()))
        rows.append(m)
        hours = np.cumsum(data["dt"].to_numpy()) / 3600.0
        ax.plot(hours, data["v"], lw=0.8, label="Measured")
        ax.plot(hours, res.voltage_v, lw=0.8, ls="--", label="Plant")
        ax.set_xlabel("Time [h]")
        ax.set_ylabel("Voltage [V]")
        legend_top(ax)
    fig.tight_layout()
    save(fig, FIG_DIR / "validation_hppc")

    #Lorenzo Giannuzzo: dynamic scenario never used for identification, with a sweep of the hysteresis rate
    scen = hb.load_scenario(SCENARIO)
    data = pd.DataFrame({"i": scen["Current_inv"].astype(float).to_numpy(),
                         "v": scen["Voltage"].astype(float).to_numpy(),
                         "soc": scen["SOC_true"].astype(float).to_numpy() / 100.0,
                         "temp": scen["Temp"].astype(float).to_numpy()})
    best = None
    for gamma in GAMMAS:
        params = CellParams.from_dir(**REPLAY_LIMITS, hysteresis_gamma=gamma,
                                     ambient_c=float(np.nanmean(data["temp"])))
        for h0 in (-1.0, 0.0, 1.0):
            res = replay(params, data, h0, data["i"].to_numpy(), 1.0)
            m = metrics(data["v"].to_numpy(), res.voltage_v)
            soc_err = 100 * (res.soc - data["soc"].to_numpy())
            m.update(case=f"S4 h0={h0:+.0f}", gamma=gamma, soc_end_error_pts=soc_err[-1],
                     refused=int(res.refused.sum()))
            rows.append(m)
            if best is None or m["rmse_mv"] < best[0]["rmse_mv"]:
                best = (m, res, gamma, h0, soc_err)

    m, res, gamma, h0, soc_err = best
    hours = np.arange(len(data)) / 3600.0
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0))
    axes[0].plot(hours, data["v"], lw=0.6, label="Measured")
    axes[0].plot(hours, res.voltage_v, lw=0.6, ls="--", label=f"Plant (gamma {gamma:g})")
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Voltage [V]")
    legend_top(axes[0])
    axes[1].plot(hours, soc_err, lw=0.8, label="Plant minus measured")
    axes[1].set_xlabel("Time [h]")
    axes[1].set_ylabel("State Of Charge Difference [%]")
    legend_top(axes[1], ncol=1)
    fig.tight_layout()
    save(fig, FIG_DIR / "validation_s4")

    table = pd.DataFrame(rows)[["case", "gamma", "rmse_mv", "mae_mv", "max_mv", "soc_end_error_pts", "refused"]]
    pd.set_option("display.width", 160)
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nbest on S4: gamma {gamma:g}, h0 {h0:+.0f}, voltage RMSE {m['rmse_mv']:.2f} mV")
    table.to_csv(FIG_DIR / "validation_summary.csv", index=False)
    print(f"figures and summary written to {FIG_DIR}")


if __name__ == "__main__":
    main()