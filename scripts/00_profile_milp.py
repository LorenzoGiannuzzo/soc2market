"""
Diagnose why the single-unit daily MILP slows down along a rolling year.

In the regression run the marginal cost per day rose from about 3 s in January to about
45 s in December, while the problem size never changes. Two explanations are possible:
(a) late-year days are intrinsically harder, or (b) something accumulates inside the
optimizer object that is reused for all 365 solves. The same late-year days are solved
here twice, with a fresh optimizer per day and with one optimizer reused for all of them.
Equal times point to (a), much lower times than in the regression run point to (b).
"""

import time

from soc2market.bidding import marl_bridge as mb

mb.ensure_import_path()
from aggregation_value import build_portfolio                      # noqa: E402
from fleet_coupling import FleetCoupling                           # noqa: E402
from market_constants import DEFAULT_SUSTAIN                       # noqa: E402
from marl_milp_continuous import MultiBatteryParameters, MultiBESSMILPOptimizer  # noqa: E402

#Lorenzo Giannuzzo: same unit as the regression test, late-year days where the slowdown was largest
UNIT_INDEX = 8
YEAR = 2023
FIRST_DAY = 300
N_DAYS = 20


def make_optimizer(fleet):
    return MultiBESSMILPOptimizer(
        fleet, None, use_nonlinear_degradation=False, directional_services=True,
        coupling=FleetCoupling.uncoupled(), **DEFAULT_SUSTAIN.as_milp_kwargs())


def solve_days(fleet, window, days, reuse):
    """Solve the given days carrying the SOC, returning the wall time of each solve."""
    soc = [b.initial_soc for b in fleet.batteries]
    opt = make_optimizer(fleet) if reuse else None
    times = []
    for day in days:
        if not reuse:
            opt = make_optimizer(fleet)
        prices, services = window.slice_day(day)
        opt.current_soc = list(soc)
        t0 = time.perf_counter()
        res = opt.optimize(24, list(prices), list(services))
        times.append(time.perf_counter() - t0)
        soc = [float(res.soc_trajectories[i][-1]) for i in range(fleet.n_batteries)]
    return times


def main():
    fleet_all, _ = build_portfolio()
    fleet = MultiBatteryParameters([fleet_all.batteries[UNIT_INDEX]])
    window = mb.market_window(YEAR, FIRST_DAY + N_DAYS)
    days = list(range(FIRST_DAY, FIRST_DAY + N_DAYS))

    fresh = solve_days(fleet, window, days, reuse=False)
    reused = solve_days(fleet, window, days, reuse=True)

    #Lorenzo Giannuzzo: per-day table and summary; the regression run took about 39 s per day here
    print("\nday   fresh [s]   reused [s]")
    for d, a, b in zip(days, fresh, reused):
        print(f"{d:>3}   {a:9.2f}   {b:10.2f}")
    print(f"\nmean  {sum(fresh) / len(fresh):9.2f}   {sum(reused) / len(reused):10.2f}")
    print("reference from the regression run over days 300-330: about 39 s per day")


if __name__ == "__main__":
    main()
