"""
Execution of a day-ahead schedule, hour by hour, with intraday SOC management.

The MILP plans the day in expectation, whereas the battery lives a single realisation: an
aFRR hour activated well above its expected share moves a sizeable part of a 2 h battery. Without correction the
day-ahead energy schedule then drives the battery into its limits for the rest of the day, which
no operator would accept. At the start of every hour the operator therefore corrects the energy
schedule towards the planned SOC trajectory,

    P_base'(h) = P_base(h) + k * (SOC_plan(h) - SOC_seen(h)) * E / 1 h,

clipped to the power left free by the reserved bands, as an intraday trade settled at the
day-ahead price. SOC_seen is the SOC the operator believes: the true SOC when generating
reference duty cycles, the estimated one in the market study.
"""

from dataclasses import dataclass

import numpy as np

from soc2market.plant.cell import simulate
from soc2market.signals.profile import fcr_response

SECONDS_PER_HOUR = 3600
INTRADAY_GAIN = 0.5


@dataclass
class DayExecution:
    current_a: np.ndarray
    voltage_v: np.ndarray
    soc: np.ndarray
    temp_c: np.ndarray
    refused: np.ndarray
    power_mw: np.ndarray            # pack power requested, MW, positive charging
    intraday_mw: np.ndarray         # hourly correction of the base power, MW
    soc_seen: np.ndarray            # SOC used by the operator at the start of each hour
    final_state: object


def execute_day(params, state, pack, schedule, realisation, dev_hz, soc_window,
                gain=INTRADAY_GAIN, soc_seen_fn=None):
    """
    Simulate one day hour by hour. `soc_seen_fn(state, hour)` returns the SOC the operator uses;
    by default the true SOC of the plant. The plant refuses power that would leave `soc_window`.
    """
    n = SECONDS_PER_HOUR
    keys = tuple(schedule.band_mw)
    parts = {k: [] for k in ("current_a", "voltage_v", "soc", "temp_c", "refused", "power_mw")}
    intraday = np.zeros(24)
    seen = np.zeros(24)
    for h in range(24):
        seen[h] = state.soc if soc_seen_fn is None else soc_seen_fn(state, h)
        #Lorenzo Giannuzzo: correction towards the planned SOC, within the power left by the reservations
        reserved = sum(schedule.band_mw[k][h] for k in keys)
        headroom = max(pack.power_mw - reserved, 0.0)
        target = schedule.base_mw[h] + gain * (schedule.soc_plan[h] - seen[h]) * pack.energy_mwh
        base = float(np.clip(target, -headroom, headroom)) if headroom > 0 else schedule.base_mw[h]
        intraday[h] = base - schedule.base_mw[h]

        dev = np.asarray(dev_hz[h * n:(h + 1) * n], float)
        power = (base + realisation.fcr_band_mw[h] * fcr_response(dev)
                 + realisation.act_down_mw[h] - realisation.act_up_mw[h])
        power = np.clip(power, -pack.power_mw, pack.power_mw)
        res = simulate(params, 1.0, state, power_w=pack.cell_power_w(power), soc_limits=soc_window)
        for k, arr in (("current_a", res.current_a), ("voltage_v", res.voltage_v), ("soc", res.soc),
                       ("temp_c", res.temp_c), ("refused", res.refused), ("power_mw", power)):
            parts[k].append(arr)
        state = res.final_state
    return DayExecution(**{k: np.concatenate(v) for k, v in parts.items()},
                        intraday_mw=intraday, soc_seen=seen, final_state=state)
