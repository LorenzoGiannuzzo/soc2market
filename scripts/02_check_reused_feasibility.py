"""
Check whether the solution returned by a reused optimizer satisfies the constraints of the
problem it was asked to solve.

On day 1 the reused optimizer returned an objective above the zero-gap optimum of the same
daily problem, which no feasible solution can do. This script solves day 0 and day 1 with
one reused optimizer, then day 1 again with a fresh one, and for each day-1 answer reports
the value of the initial SOC variable and every violated constraint.
"""

import contextlib
import io

import pulp

from soc2market.bidding import marl_bridge as mb

mb.ensure_import_path()
from aggregation_value import build_portfolio                      # noqa: E402
from fleet_coupling import FleetCoupling                           # noqa: E402
from market_constants import DEFAULT_SUSTAIN                       # noqa: E402
from marl_milp_continuous import MultiBatteryParameters, MultiBESSMILPOptimizer  # noqa: E402

UNIT_INDEX = 8
YEAR = 2023
TOL = 1e-6


def make_optimizer(fleet):
    with contextlib.redirect_stdout(io.StringIO()):
        return MultiBESSMILPOptimizer(
            fleet, None, use_nonlinear_degradation=False, directional_services=True,
            coupling=FleetCoupling.uncoupled(), **DEFAULT_SUSTAIN.as_milp_kwargs())


def solve_day(opt, window, day, soc):
    prices, services = window.slice_day(day)
    opt.current_soc = list(soc)
    return opt.optimize(24, list(prices), list(services))


def violations(problem):
    """Return (magnitude, name) for every constraint violated by more than TOL."""
    out = []
    for name, c in problem.constraints.items():
        try:
            val = c.value()
        except TypeError:
            out.append((float("inf"), f"{name} (variable without value)"))
            continue
        if c.sense == pulp.LpConstraintEQ:
            v = abs(val)
        elif c.sense == pulp.LpConstraintLE:
            v = max(val, 0.0)
        else:
            v = max(-val, 0.0)
        if v > TOL:
            out.append((v, name))
    return sorted(out, reverse=True)


def report(label, opt, expected_soc0):
    soc_var = opt.variables["SOC"][(0, 0)]
    viol = violations(opt.problem)
    print(f"\n{label}")
    print(f"  initial SOC requested {expected_soc0:.4f}, initial SOC in the returned solution {soc_var.varValue:.4f}")
    print(f"  objective {pulp.value(opt.problem.objective):.2f}, constraints {len(opt.problem.constraints)}, "
          f"violated {len(viol)}")
    for v, name in viol[:15]:
        print(f"    {v:12.6f}  {name}")


def main():
    fleet_all, _ = build_portfolio()
    fleet = MultiBatteryParameters([fleet_all.batteries[UNIT_INDEX]])
    window = mb.market_window(YEAR, 2)

    #Lorenzo Giannuzzo: day 0 and day 1 with the same optimizer, as in the original rolling loop
    reused = make_optimizer(fleet)
    res0 = solve_day(reused, window, 0, [fleet.batteries[0].initial_soc])
    soc1 = [float(res0.soc_trajectories[0][-1])]
    report("REUSED optimizer, day 0", reused, fleet.batteries[0].initial_soc)
    solve_day(reused, window, 1, soc1)
    report("REUSED optimizer, day 1", reused, soc1[0])

    #Lorenzo Giannuzzo: the same day-1 problem with a brand new optimizer
    fresh = make_optimizer(fleet)
    solve_day(fresh, window, 1, soc1)
    report("FRESH optimizer, day 1", fresh, soc1[0])


if __name__ == "__main__":
    main()
