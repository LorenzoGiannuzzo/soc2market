"""
Hourly schedule of one unit for one day, extracted from the daily MILP.

The schedule carries what the battery will be asked to do: the day-ahead base power, the
band reserved on each product, the energy the MILP plans to deliver if a product is called,
and the award and activation probabilities of every product in every hour.
"""

from dataclasses import dataclass, field

import numpy as np

from soc2market.bidding.feasibility import constraint_violations, max_violation
from soc2market.bidding.rolling import FEASIBILITY_TOL, make_optimizer

SERVICE_KEYS = ("fcr", "afrr_up", "afrr_dn", "mfrr_up", "mfrr_dn")


@dataclass
class HourlySchedule:
    """One unit, 24 hours; powers in MW, positive charging."""

    base_mw: np.ndarray                                   # P_charge - P_discharge
    band_mw: dict = field(default_factory=dict)           # reserved band per product
    activation_mw: dict = field(default_factory=dict)     # power delivered if the product is called
    award_prob: dict = field(default_factory=dict)
    activation_prob: dict = field(default_factory=dict)
    soc_plan: np.ndarray = None                           # planned SOC at the start of each hour
    net_profit_eur: float = 0.0


def clamp_soc(soc, battery):
    """The MILP requires the initial SOC inside its own bounds."""
    return float(np.clip(soc, battery.soc_min + 1e-4, battery.soc_max - 1e-4))


def solve_day_schedule(fleet, window, day, soc, sustain=None):
    """Solve the MILP of one day for a single-unit fleet and return its HourlySchedule."""
    opt = make_optimizer(fleet, sustain)
    prices, services = window.slice_day(day)
    opt.current_soc = [clamp_soc(soc, fleet.batteries[0])]
    res = opt.optimize(24, list(prices), list(services))
    if res.solver_status != "Optimal":
        raise RuntimeError(f"day {day}: MILP status {res.solver_status}")
    #Lorenzo Giannuzzo: same guard as the rolling loop, a schedule must satisfy its own constraints
    if max_violation(opt.problem) > FEASIBILITY_TOL:
        raise RuntimeError(f"day {day}: infeasible schedule {constraint_violations(opt.problem)[:5]}")

    v = opt.variables
    value = lambda var: float(var.varValue or 0.0)
    T = range(24)
    sched = HourlySchedule(
        base_mw=np.array([value(v["P_charge"][(0, t)]) - value(v["P_discharge"][(0, t)]) for t in T]),
        soc_plan=np.array([value(v["SOC"][(0, t)]) for t in T]),
        net_profit_eur=float(res.net_profit),
    )
    for key in SERVICE_KEYS:
        sched.band_mw[key] = np.array([value(v[f"R_{key}"][(0, t)]) for t in T])
        sched.activation_mw[key] = np.array([value(v[f"A_{key}"][(0, t)]) for t in T])
        enum = opt._key_to_service_enum(key)
        aw, ap = [], []
        for t in T:
            s = opt._service_at(t, enum)
            aw.append(float(s.award_probability) if s is not None else 0.0)
            ap.append(float(s.activation_probability) if s is not None else 0.0)
        sched.award_prob[key] = np.array(aw)
        sched.activation_prob[key] = np.array(ap)
    return sched
