"""
Day-by-day rolling MILP for a single unit, with a fresh optimizer for every day.

The frozen MultiBESSMILPOptimizer must not be reused across calls: from the second call
onwards it returns a solution of the FIRST day's constraints (initial SOC and SOC dynamics),
evaluated on the new day's objective, and it slows down by about 0.1 s per previous call
(see scripts/01_compare_fresh_reused.py and scripts/02_check_reused_feasibility.py).
A new optimizer is therefore built every day, and every daily solution is checked against
its own constraints before it is accepted.
"""

import contextlib
import io
import time

from soc2market.bidding import marl_bridge as mb
from soc2market.bidding.feasibility import constraint_violations, max_violation

#Lorenzo Giannuzzo: tolerance on constraint violations and on the initial SOC of each daily solution
FEASIBILITY_TOL = 1e-6


def make_optimizer(fleet, sustain=None):
    """Optimizer configured exactly as in aggregation_value._solve_window (solver banner silenced)."""
    mb.ensure_import_path()
    from fleet_coupling import FleetCoupling
    from market_constants import DEFAULT_SUSTAIN
    from marl_milp_continuous import MultiBESSMILPOptimizer

    sustain = sustain or DEFAULT_SUSTAIN
    with contextlib.redirect_stdout(io.StringIO()):
        return MultiBESSMILPOptimizer(
            fleet, None, use_nonlinear_degradation=False, directional_services=True,
            coupling=FleetCoupling.uncoupled(), **sustain.as_milp_kwargs())


def solve_rolling(fleet, window, n_days, sustain=None, initial_soc=None, verbose=False, strict=True):
    """
    Roll the daily MILP over n_days, carrying the end-of-day SOC into the next day.

    Returns the totals with the same keys used by aggregation_value (net, arb, flex, deg),
    the end-of-day SOC of every day, the largest constraint violation and initial-SOC error
    of every daily solution, and solver bookkeeping. With strict=True a daily solution that
    violates its own constraints raises immediately.
    """
    soc = list(initial_soc) if initial_soc is not None else [b.initial_soc for b in fleet.batteries]
    totals = dict(net=0.0, arb=0.0, flex=0.0, deg=0.0)
    end_of_day_soc, day_max_violation, day_init_soc_error = [], [], []
    n_optimal = 0
    t0 = time.perf_counter()

    for day in range(n_days):
        opt = make_optimizer(fleet, sustain)
        prices, services = window.slice_day(day)
        opt.current_soc = list(soc)
        res = opt.optimize(24, list(prices), list(services))

        if res.solver_status == "Optimal":
            n_optimal += 1
            #Lorenzo Giannuzzo: the daily solution must satisfy its own constraints and start
            # from the SOC that was carried in from the previous day
            viol = max_violation(opt.problem)
            init_err = max(abs(opt.variables["SOC"][(i, 0)].varValue - soc[i])
                           for i in range(fleet.n_batteries))
            if strict and (viol > FEASIBILITY_TOL or init_err > FEASIBILITY_TOL):
                worst = constraint_violations(opt.problem, FEASIBILITY_TOL)[:5]
                raise RuntimeError(f"day {day}: infeasible daily solution, initial SOC error "
                                   f"{init_err:.3g}, violations {worst}")
        else:
            viol, init_err = float("nan"), float("nan")

        totals["net"] += float(res.net_profit)
        totals["arb"] += float(res.arbitrage_profit)
        totals["flex"] += float(res.flexibility_revenue)
        totals["deg"] += float(res.degradation_cost)
        soc = [float(res.soc_trajectories[i][-1]) for i in range(fleet.n_batteries)]
        end_of_day_soc.append(list(soc))
        day_max_violation.append(viol)
        day_init_soc_error.append(init_err)

        if verbose and (day + 1) % 30 == 0:
            print(f"  [rolling] day {day + 1}/{n_days} | {time.perf_counter() - t0:.0f} s "
                  f"| net {totals['net']:,.0f} EUR", flush=True)

    return dict(totals=totals, end_of_day_soc=end_of_day_soc, max_violation=day_max_violation,
                init_soc_error=day_init_soc_error, n_optimal=n_optimal, n_solves=n_days,
                seconds=time.perf_counter() - t0)