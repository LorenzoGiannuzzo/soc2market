"""
Find where the rolling MILP with a fresh optimizer per day departs from the rolling MILP
with one reused optimizer, and which of the two returns the optimum.

Both loops start from the same SOC on day 0, so day 0 must coincide. On the first day
where they differ, the same daily problem is solved again with a zero optimality gap,
which gives the true optimum against which both answers are judged.
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
N_DAYS = 40


class ExactMILP(MultiBESSMILPOptimizer):
    """Same model, solved to a zero relative gap."""
    SOLVER_GAP_REL = 0.0


def make_optimizer(fleet, cls=MultiBESSMILPOptimizer):
    #Lorenzo Giannuzzo: the constructor prints the solver banner every time, silenced here
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(fleet, None, use_nonlinear_degradation=False, directional_services=True,
                   coupling=FleetCoupling.uncoupled(), **DEFAULT_SUSTAIN.as_milp_kwargs())


def solve_day(opt, window, day, soc):
    prices, services = window.slice_day(day)
    opt.current_soc = list(soc)
    res = opt.optimize(24, list(prices), list(services))
    return dict(
        soc_start=soc[0], status=res.solver_status, objective=pulp.value(opt.problem.objective),
        net=float(res.net_profit), arb=float(res.arbitrage_profit),
        flex=float(res.flexibility_revenue), soc_end=float(res.soc_trajectories[0][-1]),
    )


def run(fleet, window, reuse):
    soc = [b.initial_soc for b in fleet.batteries]
    opt = make_optimizer(fleet) if reuse else None
    rows = []
    for day in range(N_DAYS):
        if not reuse:
            opt = make_optimizer(fleet)
        row = solve_day(opt, window, day, soc)
        rows.append(row)
        soc = [row["soc_end"]]
    return rows


def main():
    fleet_all, _ = build_portfolio()
    fleet = MultiBatteryParameters([fleet_all.batteries[UNIT_INDEX]])
    window = mb.market_window(YEAR, N_DAYS)

    fresh = run(fleet, window, reuse=False)
    reused = run(fleet, window, reuse=True)

    print("\nday | soc0 fresh  soc0 reused | obj fresh    obj reused   | net fresh    net reused  | soc_end f  soc_end r")
    first_diff = None
    for d, (f, r) in enumerate(zip(fresh, reused)):
        differs = abs(f["objective"] - r["objective"]) > 1e-6 or abs(f["soc_start"] - r["soc_start"]) > 1e-9
        if differs and first_diff is None:
            first_diff = d
        flag = "  <-- differs" if differs else ""
        print(f"{d:>3} | {f['soc_start']:10.4f} {r['soc_start']:12.4f} | {f['objective']:11.2f} {r['objective']:12.2f} "
              f"| {f['net']:11.2f} {r['net']:11.2f} | {f['soc_end']:9.4f} {r['soc_end']:9.4f}{flag}")

    if first_diff is None:
        print(f"\nNo difference over the first {N_DAYS} days.")
        return

    #Lorenzo Giannuzzo: judge both answers against the zero-gap optimum of the same daily problem,
    # solved from each loop's own starting SOC
    print(f"\nFirst difference on day {first_diff}. Zero-gap re-solve of that day:")
    for label, rows in (("fresh", fresh), ("reused", reused)):
        row = rows[first_diff]
        exact = solve_day(make_optimizer(fleet, ExactMILP), window, first_diff, [row["soc_start"]])
        print(f"  {label:>6}: start SOC {row['soc_start']:.4f} | returned objective {row['objective']:.2f} "
              f"| zero-gap optimum {exact['objective']:.2f} | status {row['status']}")


if __name__ == "__main__":
    main()
