"""
Generate representative operating weeks of a 1 MW / 2 MWh unit on the 2023 Italian markets.

For every day: the MILP plans the day from the true SOC left by the previous one, awards and
activations are realised (chronological MSD outcome of the same calendar day, or independent
draws, as set by MSD_MODE in paths.py), and the day is executed hour by hour with the measured grid frequency of
the same calendar day and intraday SOC management towards the planned trajectory (true SOC,
since these are reference duty cycles). The plant simulates the cell inside the operating SOC
window of the MILP, and the measurement chain produces the table the SOC estimators will read.
One file per day is written to paths.DUTY_DIR.
"""

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.bidding import marl_bridge as mb
from soc2market.estimation import estimator_bridge as eb
from soc2market.bidding.schedule import SERVICE_KEYS, solve_day_schedule
from soc2market.plant.cell import CellParams, CellState
from soc2market.plant.pack import PackSpec
from soc2market.plant.sensors import SensorSpec, measure
from soc2market.signals.duty_cycle import draw_realisation, draw_realisation_chronological
from soc2market.signals.execution import execute_day
from soc2market.signals.frequency import FREQ_DIR, FrequencyArchive
from soc2market.signals.msd import MSDArchive

OUT_DIR = paths.DUTY_DIR
YEAR = 2023
UNIT_INDEX = 8                      # medium_2h of the aggregation portfolio, 1.0 MW / 2.0 MWh
SEED = 2023

#Lorenzo Giannuzzo: one week per season, avoiding the daylight-saving days and the days with
# long frequency gaps (day indices from 1 January, 0-based)
WEEKS = {"winter": range(8, 15), "spring": range(99, 106), "summer": range(190, 197), "autumn": range(281, 288)}


def main():
    mb.ensure_import_path()
    from aggregation_value import build_portfolio
    from marl_milp_continuous import MultiBatteryParameters

    fleet_all, meta = build_portfolio()
    assert meta[UNIT_INDEX]["label"] == "medium_2h"
    fleet = MultiBatteryParameters([fleet_all.batteries[UNIT_INDEX]])
    pack = PackSpec(power_mw=1.0, energy_mwh=2.0)
    params = CellParams.from_dir()
    #Lorenzo Giannuzzo: measurement chain matched to the laboratory record the estimators were
    # trained on (resolution 0.1 mV / 0.1 mA, noise estimated on scenario S4, no offset or gain error)
    reference = eb.load_scenario(eb.REPO_ROOT / "data" / "private" / "soc_regression" / "S4_DAY1to8.xlsx")
    sensors = SensorSpec.matched_to(reference)
    print(f"sensor chain: current noise {1e3 * sensors.current_noise_a:.3f} mA, voltage noise "
          f"{1e3 * sensors.voltage_noise_v:.4f} mV, resolution 0.1 mA / 0.1 mV")
    arch = FrequencyArchive.load(FREQ_DIR / "2023.npz")
    msd = MSDArchive.load([mb.MARKET_DIR / n for n in mb.MSD_FILES], YEAR) if paths.MSD_MODE == "chronological" else None
    print(f"MSD mode: {paths.MSD_MODE}, duty cycles written to {OUT_DIR}")
    window = mb.market_window(YEAR, max(max(w) for w in WEEKS.values()) + 1)
    #Lorenzo Giannuzzo: separate random streams, so that changing the measurement chain never changes
    # the market draws, and therefore the operation of the battery
    rng = np.random.default_rng(SEED)
    rng_sensors = np.random.default_rng(SEED + 1)
    battery = fleet.batteries[0]
    soc_window = (battery.soc_min, battery.soc_max)
    print(f"operating SOC window of the MILP: {soc_window[0]:.2f} to {soc_window[1]:.2f}")

    rows = []
    for name, days in WEEKS.items():
        state = CellState(soc=0.5, h=0.0, temp_c=params.ambient_c)
        week_dir = OUT_DIR / name
        week_dir.mkdir(parents=True, exist_ok=True)
        for day in days:
            sched = solve_day_schedule(fleet, window, day, state.soc)
            if msd is None:
                real = draw_realisation(sched, rng)
                msd_up = msd_dn = msd_missing = -1
            else:
                msd_day = msd.day(day)
                real = draw_realisation_chronological(sched, msd_day, rng)
                msd_up, msd_dn, msd_missing = int(msd_day.called_up.sum()), int(msd_day.called_dn.sum()), msd_day.missing_hours
            dev, zeroed = arch.day(day)
            res = execute_day(params, state, pack, sched, real, dev, soc_window)
            power_mw = res.power_mw
            table = measure(res, sensors, rng_sensors)

            #Lorenzo Giannuzzo: planned SOC (MILP, hourly, expectation) against realised SOC (plant)
            realised_hourly = res.soc[::3600]
            np.savez(week_dir / f"day_{day:03d}.npz",
                     current_a=table["Current_inv"].to_numpy(np.float64),
                     voltage_v=table["Voltage"].to_numpy(np.float64),
                     temp_c=table["Temp"].to_numpy(np.float64),
                     soc_true_pct=table["SOC_true"].to_numpy(np.float64),
                     power_mw=power_mw.astype(np.float32), refused=res.refused,
                     base_mw=sched.base_mw, soc_plan=sched.soc_plan, intraday_mw=res.intraday_mw,
                     **{f"band_{k}": sched.band_mw[k] for k in SERVICE_KEYS},
                     **{f"share_{k}": real.share[k] for k in SERVICE_KEYS},
                     fcr_awarded=real.awarded["fcr"])
            rows.append(dict(week=name, day=day, profit_eur=sched.net_profit_eur,
                             soc_start=100 * realised_hourly[0], soc_min=100 * res.soc.min(), soc_max=100 * res.soc.max(),
                             plan_vs_real_max_pts=100 * np.max(np.abs(sched.soc_plan - realised_hourly)),
                             throughput_mwh=float(np.abs(power_mw).sum() / 3600.0),
                             fcr_hours=int(real.awarded["fcr"].sum()),
                             called_up_h=msd_up, called_dn_h=msd_dn, msd_missing_h=msd_missing,
                             restoration_h=int(sum(real.awarded[k].sum() for k in SERVICE_KEYS if k != "fcr")),
                             activated_mwh=float(real.act_up_mw.sum() + real.act_down_mw.sum()),
                             intraday_mwh=float(np.abs(res.intraday_mw).sum()),
                             refused_s=int(res.refused.sum()), freq_zeroed_s=zeroed,
                             temp_max_c=float(res.temp_c.max())))
            state = res.final_state

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    table.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"\n{len(table)} days written to {OUT_DIR}")


if __name__ == "__main__":
    main()
