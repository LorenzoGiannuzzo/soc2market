"""
Unit tests of the battery plant. They use synthetic parameter tables, so they run on any
machine and do not need the private Huawei data.
"""

import numpy as np
import pandas as pd
import pytest

from soc2market.plant.cell import CellParams, CellState, simulate
from soc2market.plant.identify import clean_table, fit_relaxation, hppc_time_step, relaxation_model
from soc2market.plant.pack import PackSpec
from soc2market.plant.sensors import SensorSpec, measure


def synthetic_params(**overrides):
    soc = np.linspace(0.0, 1.0, 101)
    ocv_dch = 3.0 + 0.3 * soc + 0.05 * np.tanh((soc - 0.5) * 8)
    ocv_chg = ocv_dch + 0.02
    table = pd.DataFrame({"soc": [0.0, 0.5, 1.0], "r0": [4e-4, 3e-4, 4e-4], "r1": [2e-4] * 3,
                          "c1": [5e4] * 3, "r2": [3e-4] * 3, "c2": [1e6] * 3})
    return CellParams(ocv_soc=soc, ocv_chg=ocv_chg, ocv_dch=ocv_dch, chg_table=table,
                      dch_table=table.copy(), **overrides)


def ocv_at(params, soc, h):
    mid = 0.5 * (np.interp(soc, params.ocv_soc, params.ocv_chg) + np.interp(soc, params.ocv_soc, params.ocv_dch))
    half = 0.5 * (np.interp(soc, params.ocv_soc, params.ocv_chg) - np.interp(soc, params.ocv_soc, params.ocv_dch))
    return mid + h * half


def test_power_mode_delivers_requested_power():
    p = synthetic_params()
    request = np.r_[np.full(600, 450.0), np.full(600, -450.0)]
    res = simulate(p, 1.0, CellState(soc=0.5), power_w=request)
    assert not res.refused.any()
    assert np.allclose(res.voltage_v * res.current_a, request, rtol=1e-9, atol=1e-9)


def test_voltage_above_ocv_in_charge_and_below_in_discharge():
    p = synthetic_params()
    res = simulate(p, 1.0, CellState(soc=0.5), current_a=np.r_[np.full(60, 140.0), np.full(60, -140.0)])
    ocv = ocv_at(p, res.soc, res.h)
    assert np.all(res.voltage_v[:60] > ocv[:60])
    assert np.all(res.voltage_v[61:] < ocv[61:])


def test_coulomb_balance_with_efficiency_and_ageing():
    p = synthetic_params(coulombic_efficiency=0.99, soh_capacity=0.9)
    current = np.r_[np.full(1800, 100.0), np.full(900, -100.0)]
    res = simulate(p, 1.0, CellState(soc=0.3), current_a=current)
    expected = 0.3 + (0.99 * 100.0 * 1800 - 100.0 * 900) / (0.9 * 280.0 * 3600.0)
    assert res.final_state.soc == pytest.approx(expected, abs=1e-12)


def test_rest_relaxes_to_ocv():
    p = synthetic_params()
    res = simulate(p, 1.0, CellState(soc=0.5), current_a=np.r_[np.full(600, -140.0), np.zeros(20000)])
    end = res.final_state
    assert abs(res.voltage_v[-1] - ocv_at(p, end.soc, end.h)) < 1e-5


def test_hysteresis_moves_to_the_branch_of_the_current():
    p = synthetic_params(hysteresis_gamma=30.0)
    up = simulate(p, 1.0, CellState(soc=0.2, h=-1.0), current_a=np.full(3600, 140.0))
    assert up.final_state.h > 0.95
    down = simulate(p, 1.0, up.final_state, current_a=np.full(3600, -140.0))
    assert down.final_state.h < -0.95


def test_request_beyond_limits_is_refused():
    p = synthetic_params()
    res = simulate(p, 1.0, CellState(soc=0.001), power_w=np.full(100, -450.0))
    assert res.refused.any()
    assert np.all(res.current_a[res.refused] == 0.0)
    assert res.final_state.soc >= 0.0
    assert res.shortfall_wh > 0.0


def test_array_and_scalar_time_step_agree():
    p = synthetic_params()
    current = np.sin(np.arange(500) / 30.0) * 100.0
    a = simulate(p, 1.0, CellState(soc=0.5), current_a=current)
    b = simulate(p, np.ones(500), CellState(soc=0.5), current_a=current)
    assert np.array_equal(a.voltage_v, b.voltage_v)


@pytest.mark.parametrize("is_charge", [True, False])
def test_relaxation_fit_recovers_known_parameters(is_charge):
    t = np.arange(0.0, 600.0, 1.0)
    truth = dict(v_inf=3.30, r1=2e-4, tau1=15.0, r2=4e-4, tau2=250.0)
    v = relaxation_model(t, *truth.values(), 140.0, 30.0, -1.0 if is_charge else 1.0)
    fit = fit_relaxation(t, v, 140.0, 30.0, is_charge)
    assert fit["rmse_v"] < 1e-6
    for key in ("r1", "tau1", "r2", "tau2"):
        assert fit[key] == pytest.approx(truth[key], rel=1e-3)


def test_pack_scaling_of_a_1_mw_2_mwh_unit():
    pack = PackSpec(power_mw=1.0, energy_mwh=2.0)
    assert pack.n_cells == pytest.approx(2e6 / (280.0 * 3.2))
    assert pack.cell_power_w(1.0) == pytest.approx(448.0)


def test_sensors_apply_offset_gain_and_quantisation():
    p = synthetic_params()
    res = simulate(p, 1.0, CellState(soc=0.5), current_a=np.full(100, 100.0))
    spec = SensorSpec(current_offset_a=0.5, current_gain_error=0.01, current_noise_a=0.0,
                      voltage_noise_v=0.0, voltage_resolution_v=0.001,
                      temperature_noise_c=0.0, temperature_resolution_c=0.1)
    table = measure(res, spec, np.random.default_rng(0))
    assert np.allclose(table["Current_inv"], 100.0 * 1.01 + 0.5)
    assert np.allclose(table["Voltage"] * 1000, np.round(table["Voltage"] * 1000))
    assert np.allclose(table["SOC_true"], res.soc * 100.0)


def test_degenerate_fit_is_replaced_by_neighbours():
    raw = pd.DataFrame({
        "soc": [0.90, 0.95, 0.995], "r0": [2.5e-4, 2.5e-4, 2.4e-4],
        "r1": [1.0e-4, 0.9e-4, 1.2e-4], "c1": [9e3, 1e4, 7e3],
        "r2": [3.0e-5, 3.1e-5, 1.0e-7], "c2": [8e5, 8e5, 2e11],
        "tau2": [24.0, 24.7, 2e4], "rmse_mv": [0.43, 0.40, 1.3], "success": [True, True, True]})
    table = clean_table(raw)
    assert list(table["replaced"]) == [False, False, True]
    assert table.loc[2, "r2"] == pytest.approx(3.1e-5)


def test_cell_above_full_can_still_discharge():
    p = synthetic_params()
    res = simulate(p, 1.0, CellState(soc=1.002, h=1.0), current_a=np.full(600, -140.0))
    assert not res.refused.any()
    assert res.final_state.soc < 1.0
    blocked = simulate(p, 1.0, CellState(soc=1.002, h=1.0), current_a=np.full(10, 140.0))
    assert blocked.refused.all()


def test_time_base_is_replaced_only_when_the_coulomb_balance_disagrees():
    n = 2000
    current = np.full(n, -140.0)
    step = np.arange(n, dtype=float)
    for true_dt, expected_source in ((1.0, "declared"), (0.084, "coulomb balance")):
        soc = 100.0 + np.cumsum(np.r_[0.0, current[1:]]) * true_dt / 3600.0 / 280.0 * 100.0
        dt, source = hppc_time_step(step, current, soc)
        assert source == expected_source
        assert dt == pytest.approx(true_dt, rel=1e-6)


def test_relaxation_fit_is_independent_of_the_sampling_interval():
    dt = 0.084
    t = np.arange(182) * dt
    truth = dict(v_inf=3.30, r1=1.2e-4, tau1=0.06, r2=4e-5, tau2=2.5)
    v = relaxation_model(t, *truth.values(), 300.0, 50 * dt, 1.0)
    fit = fit_relaxation(t, v, 300.0, 50 * dt, False, dt=dt)
    assert fit["rmse_v"] < 1e-6 and not fit["at_bound"]
    for key in ("r1", "tau1", "r2", "tau2"):
        assert fit[key] == pytest.approx(truth[key], rel=1e-3)


def test_slow_branch_adds_steady_polarisation_and_is_off_by_default():
    base = synthetic_params()
    current = np.full(3600, -100.0)
    off = simulate(base, 1.0, CellState(soc=0.9), current_a=current)
    on = simulate(synthetic_params(r3_dch=2e-4, tau3_dch_s=300.0), 1.0, CellState(soc=0.9), current_a=current)
    assert on.voltage_v[-1] - off.voltage_v[-1] == pytest.approx(-100.0 * 2e-4, rel=1e-3)
    assert np.array_equal(off.voltage_v, simulate(synthetic_params(r3_dch=0.0), 1.0, CellState(soc=0.9),
                                                   current_a=current).voltage_v)


def test_slow_branch_fit_recovers_known_values():
    from soc2market.plant.identify import fit_slow_branch
    truth = synthetic_params(r3_dch=1.5e-4, tau3_dch_s=200.0)
    current = np.tile(np.r_[np.full(300, -140.0), np.zeros(300)], 6)
    sim = simulate(truth, 1.0, CellState(soc=0.8, h=-1.0), current_a=current)
    data = pd.DataFrame({"i": current, "v": sim.voltage_v, "soc": sim.soc})
    fit = fit_slow_branch(synthetic_params(), data, 1.0, -1.0, "dch", stride=1)
    assert fit["r3"] == pytest.approx(1.5e-4, rel=1e-3)
    assert fit["tau3_s"] == pytest.approx(200.0, rel=1e-3)
    assert fit["rmse_after_mv"] < 1e-3
