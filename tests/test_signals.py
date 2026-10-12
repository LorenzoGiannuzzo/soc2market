"""Tests of the frequency conversion, the FCR droop and the one-second profile builder."""

import numpy as np
import pandas as pd
import pytest

from soc2market.signals.frequency import FrequencyArchive, convert_frequency_csv
from soc2market.signals.profile import build_power_profile, fcr_response


def test_fcr_droop_sign_deadband_and_saturation():
    dev = np.array([-0.5, -0.2, -0.1, -0.005, 0.0, 0.005, 0.1, 0.2, 0.5])
    share = fcr_response(dev)
    assert list(share) == [-1.0, -1.0, -0.5, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0]


def test_profile_combines_base_fcr_and_activations():
    base = np.r_[np.full(12, 0.5), np.full(12, -0.5)]
    fcr = np.full(24, 0.2)
    up = np.zeros(24); up[20] = 0.3
    dev = np.full(24 * 3600, 0.1)
    profile, clipped = build_power_profile(base, fcr, dev, act_up_mw=up, rated_mw=1.0)
    assert profile[0] == pytest.approx(0.5 + 0.2 * 0.5)
    assert profile[20 * 3600] == pytest.approx(-0.5 - 0.3 + 0.2 * 0.5)
    assert clipped == 0.0


def test_profile_clips_to_rated_power_and_reports_it():
    profile, clipped = build_power_profile(np.full(24, 1.0), np.full(24, 0.5), np.full(24 * 3600, 0.2), rated_mw=1.0)
    assert profile.max() == pytest.approx(1.0)
    assert clipped == pytest.approx(0.5 * 24)


def test_conversion_handles_daylight_saving_and_gaps(tmp_path):
    #Lorenzo Giannuzzo: a two-day file with a NaN sample, a short gap and a missing hour
    ts = pd.date_range("2023-03-26 00:00:00", "2023-03-27 23:59:59", freq="s")
    ts = ts[~((ts >= "2023-03-26 02:00:00") & (ts < "2023-03-26 03:00:00"))]
    f = np.full(len(ts), 50.02)
    f[10] = np.nan
    keep = np.ones(len(ts), bool); keep[100:130] = False
    csv = tmp_path / "f.csv"
    pd.DataFrame({"ts": ts[keep].strftime("%Y-%m-%d %H:%M:%S"), "f": f[keep]}).to_csv(csv, header=False, index=False)
    out = tmp_path / "f.npz"
    convert_frequency_csv(csv, out, 2023, chunksize=50_000)
    arch = FrequencyArchive.load(out)
    d = (pd.Timestamp("2023-03-26") - pd.Timestamp("2023-01-01")).days
    assert np.isnan(arch.dev_hz[d, 2 * 3600:3 * 3600]).all()
    series, zeroed = arch.day(d)
    assert series[10] == pytest.approx(0.02, abs=1e-6)
    assert series[115] == pytest.approx(0.02, abs=1e-6)
    assert zeroed == 3600
    full_day, zeroed_next = arch.day(d + 1)
    assert zeroed_next == 0 and np.allclose(full_day, 0.02, atol=1e-6)


def test_implausible_spikes_are_treated_as_missing_and_interpolated():
    dev = np.full((1, 86400), 0.02, dtype=np.float32)
    dev[0, 5000] = 0.8
    dev[0, 6000] = -0.83
    series, zeroed = FrequencyArchive(dev_hz=dev, year=2023).day(0)
    assert zeroed == 0
    assert series[5000] == pytest.approx(0.02, abs=1e-6) and series[6000] == pytest.approx(0.02, abs=1e-6)


def _toy_schedule():
    from soc2market.bidding.schedule import HourlySchedule
    keys = ("fcr", "afrr_up", "afrr_dn", "mfrr_up", "mfrr_dn")
    s = HourlySchedule(base_mw=np.zeros(24))
    for k in keys:
        s.band_mw[k] = np.zeros(24)
        s.activation_mw[k] = np.zeros(24)
        s.award_prob[k] = np.zeros(24)
        s.activation_prob[k] = np.zeros(24)
    s.band_mw["fcr"][:] = 0.4
    s.award_prob["fcr"][:] = 1.0
    s.band_mw["afrr_up"][5] = 0.3
    s.activation_mw["afrr_up"][5] = 0.3
    s.award_prob["afrr_up"][5] = 1.0
    s.activation_prob["afrr_up"][5] = 1.0
    s.band_mw["afrr_dn"][6] = 0.3
    s.activation_mw["afrr_dn"][6] = 0.2
    s.award_prob["afrr_dn"][6] = 0.0
    s.activation_prob["afrr_dn"][6] = 1.0
    return s


def test_realisation_respects_award_and_activation():
    from soc2market.signals.duty_cycle import draw_realisation
    r = draw_realisation(_toy_schedule(), np.random.default_rng(0))
    assert r.awarded["fcr"].all() and np.allclose(r.fcr_band_mw, 0.4)
    assert r.act_up_mw[5] == pytest.approx(0.3) and r.act_up_mw.sum() == pytest.approx(0.3)
    assert r.act_down_mw.sum() == 0.0


def test_day_profile_from_a_realisation():
    from soc2market.signals.duty_cycle import day_power_profile, draw_realisation
    s = _toy_schedule()
    r = draw_realisation(s, np.random.default_rng(0))
    profile, clipped = day_power_profile(s, r, np.full(86400, -0.1), rated_mw=1.0)
    assert profile[0] == pytest.approx(0.4 * -0.5)
    assert profile[5 * 3600] == pytest.approx(-0.3 + 0.4 * -0.5)
    assert clipped == 0.0


def _toy_cell():
    from soc2market.plant.cell import CellParams
    soc = np.linspace(0.0, 1.0, 101)
    ocv = 3.0 + 0.3 * soc
    table = pd.DataFrame({"soc": [0.0, 1.0], "r0": [3e-4] * 2, "r1": [2e-4] * 2, "c1": [5e4] * 2,
                          "r2": [3e-4] * 2, "c2": [1e6] * 2})
    return CellParams(ocv_soc=soc, ocv_chg=ocv + 0.01, ocv_dch=ocv - 0.01, chg_table=table, dch_table=table.copy())


def test_intraday_correction_brings_the_soc_back_to_plan():
    from soc2market.plant.cell import CellState
    from soc2market.plant.pack import PackSpec
    from soc2market.signals.duty_cycle import Realisation
    from soc2market.signals.execution import execute_day
    s = _toy_schedule()
    s.band_mw["fcr"][:] = 0.0
    s.soc_plan = np.full(24, 0.5)
    act_up = np.zeros(24); act_up[2] = 0.8
    real = Realisation(awarded={}, called={}, share={}, fcr_band_mw=np.zeros(24), act_up_mw=act_up, act_down_mw=np.zeros(24))
    pack = PackSpec(power_mw=1.0, energy_mwh=2.0)
    dev = np.zeros(86400)
    no_ctrl = execute_day(_toy_cell(), CellState(soc=0.5), pack, s, real, dev, (0.1, 0.9), gain=0.0)
    ctrl = execute_day(_toy_cell(), CellState(soc=0.5), pack, s, real, dev, (0.1, 0.9), gain=0.5)
    assert abs(no_ctrl.final_state.soc - 0.5) > 0.3
    assert abs(ctrl.final_state.soc - 0.5) < 0.02


def test_operating_window_is_enforced():
    from soc2market.plant.cell import CellState, simulate
    res = simulate(_toy_cell(), 1.0, CellState(soc=0.12), power_w=np.full(3600, -400.0), soc_limits=(0.1, 0.9))
    assert res.final_state.soc >= 0.1 - 1e-6 and res.refused.any()


def test_activated_share_has_the_mean_of_the_activation_probability():
    from soc2market.signals.duty_cycle import activated_share
    rng = np.random.default_rng(1)
    x = activated_share(np.full(200_000, 0.3), 200_000, rng)
    assert x.mean() == pytest.approx(0.3, abs=0.005)
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert list(activated_share(np.array([0.0, 1.0]), 2, rng)) == [0.0, 1.0]
