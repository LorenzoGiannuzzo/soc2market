"""
Regression and correctness tests for the frozen MARL market data and MILP code.

1. The original rolling loop (one optimizer reused for the whole year) reproduces the
   result submitted to the 2nd IEEE PES International Meeting exactly. This documents
   provenance only: those numbers come from infeasible daily solutions (test 2).
2. A reused optimizer returns, on its second call, a solution that starts from the first
   day's SOC and violates the constraints of the second day. If a future PuLP or Gurobi
   version removes this behaviour, this test fails and the note in PROVENANCE.md must change.
3. The rolling loop of this repository (fresh optimizer per day, every daily solution
   checked against its own constraints) reproduces its own reference year.
"""

import contextlib
import io

import pytest

from soc2market.bidding import marl_bridge as mb
from soc2market.bidding.feasibility import constraint_violations

REFERENCE_UNIT_INDEX = 8
REFERENCE_YEAR = 2023
REFERENCE_DAYS = 365

#Lorenzo Giannuzzo: values from aggregation_2023.json, units[8], standalone arm (reused optimizer)
PUBLISHED_EUR = {
    "net": 231311.20300032618,
    "arb": 136866.11662752857,
    "flex": 107456.97945859302,
    "deg": 13011.893085794984,
}

#Lorenzo Giannuzzo: values from the fresh-optimizer rolling loop, first run on 2026-10-06,
# every daily solution feasible with zero-gap agreement checked on day 1
FRESH_REFERENCE_EUR = {
    "net": 213948.0099,
    "arb": 57160.2988,
    "flex": 168440.6207,
    "deg": 11652.9096,
}

REL_TOL = 1e-6

_MISSING = mb.market_files_missing()
needs_market_data = pytest.mark.skipif(bool(_MISSING), reason="Market data not found: " + ", ".join(_MISSING))


def _reference_unit():
    mb.ensure_import_path()
    from aggregation_value import build_portfolio
    from marl_milp_continuous import MultiBatteryParameters

    fleet, meta = build_portfolio()
    assert meta[REFERENCE_UNIT_INDEX]["label"] == "medium_2h"
    return MultiBatteryParameters([fleet.batteries[REFERENCE_UNIT_INDEX]])


def _compare(out, reference):
    print(f"\nsolver: {out['n_optimal']}/{out['n_solves']} optimal days, {out['seconds']:.0f} s")
    failures = []
    for key, expected in reference.items():
        actual = float(out["totals"][key])
        rel = abs(actual - expected) / max(abs(expected), 1e-9)
        print(f"{key:>5}: {actual:14.4f} EUR vs reference {expected:14.4f} EUR (relative difference {rel:.2e})")
        if rel > REL_TOL:
            failures.append(key)
    assert not failures, f"components outside tolerance: {failures}"


@pytest.mark.slow
@needs_market_data
def test_original_loop_reproduces_published_result():
    mb.ensure_import_path()
    from aggregation_value import _solve_window
    from market_constants import DEFAULT_SUSTAIN

    window = mb.market_window(REFERENCE_YEAR, REFERENCE_DAYS)
    out = _solve_window(_reference_unit(), window, REFERENCE_DAYS, DEFAULT_SUSTAIN,
                        tag="regression", verbose=True)
    _compare(out, PUBLISHED_EUR)


@needs_market_data
def test_reused_optimizer_returns_stale_solution():
    mb.ensure_import_path()
    from fleet_coupling import FleetCoupling
    from market_constants import DEFAULT_SUSTAIN
    from marl_milp_continuous import MultiBESSMILPOptimizer

    fleet = _reference_unit()
    window = mb.market_window(REFERENCE_YEAR, 2)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = MultiBESSMILPOptimizer(
            fleet, None, use_nonlinear_degradation=False, directional_services=True,
            coupling=FleetCoupling.uncoupled(), **DEFAULT_SUSTAIN.as_milp_kwargs())

    prices, services = window.slice_day(0)
    opt.current_soc = [fleet.batteries[0].initial_soc]
    res0 = opt.optimize(24, list(prices), list(services))
    soc1 = float(res0.soc_trajectories[0][-1])
    assert abs(soc1 - fleet.batteries[0].initial_soc) > 1e-3, "day 0 must end away from the initial SOC"

    prices, services = window.slice_day(1)
    opt.current_soc = [soc1]
    opt.optimize(24, list(prices), list(services))
    returned_soc0 = opt.variables["SOC"][(0, 0)].varValue
    viol = constraint_violations(opt.problem)
    print(f"\nrequested initial SOC {soc1:.4f}, returned {returned_soc0:.4f}, violated constraints {len(viol)}")
    assert abs(returned_soc0 - soc1) > 1e-3 and viol, "reused optimizer behaved correctly: update PROVENANCE.md"


@needs_market_data
def test_fresh_rolling_reproduces_reference_and_is_feasible():
    from soc2market.bidding.rolling import solve_rolling

    window = mb.market_window(REFERENCE_YEAR, REFERENCE_DAYS)
    out = solve_rolling(_reference_unit(), window, REFERENCE_DAYS, verbose=True, strict=True)
    print(f"largest daily violation {max(out['max_violation']):.2e}, "
          f"largest initial SOC error {max(out['init_soc_error']):.2e}")
    _compare(out, FRESH_REFERENCE_EUR)
