"""Settlement of one day and SOC margins of the market study, on a hand-made day (no MILP, no plant)."""

import enum
from types import SimpleNamespace as NS

import numpy as np
import pytest

from soc2market.simulation import market as M
from soc2market.signals.profile import fcr_response


class _ST(enum.Enum):
    FCR = 1
    AFRR_UP = 2
    AFRR_DN = 3
    MFRR_UP = 4
    MFRR_DN = 5


ENUM = {"fcr": _ST.FCR, "afrr_up": _ST.AFRR_UP, "afrr_dn": _ST.AFRR_DN, "mfrr_up": _ST.MFRR_UP, "mfrr_dn": _ST.MFRR_DN}
DEG = 1.5625


def _day(refused):
    #Lorenzo Giannuzzo: charge 0.5 MW for 12 h, discharge 0.5 MW for 12 h, FCR 0.2 MW in hour 3 with
    # +100 mHz, aFRR up 0.4 MW in hour 20 activated for half of the hour
    z = np.zeros(24)
    sched = NS(base_mw=np.r_[np.full(12, 0.5), np.full(12, -0.5)],
               band_mw={k: z.copy() for k in ENUM}, activation_mw={k: z.copy() for k in ENUM})
    sched.band_mw["afrr_up"][20] = sched.activation_mw["afrr_up"][20] = 0.4
    sched.band_mw["fcr"][3] = 0.2
    share = {k: z.copy() for k in ENUM}
    share["afrr_up"][20] = 0.5
    awarded = {k: sched.band_mw[k] > 0 for k in ENUM}
    real = NS(awarded=awarded, called={k: share[k] > 0 for k in ENUM}, share=share,
              fcr_band_mw=sched.band_mw["fcr"], act_up_mw=0.4 * share["afrr_up"], act_down_mw=z.copy())
    dev = np.zeros(86400)
    dev[3 * 3600:4 * 3600] = 0.1
    power = (np.repeat(sched.base_mw, 3600) + np.repeat(real.fcr_band_mw, 3600) * fcr_response(dev)
             - np.repeat(real.act_up_mw, 3600))
    services = [[NS(service_type=_ST.FCR, capacity_price=50.0, energy_price=0.0,
                    true_capacity_price=None, true_energy_price=None),
                 NS(service_type=_ST.AFRR_UP, capacity_price=10.0, energy_price=200.0,
                    true_capacity_price=None, true_energy_price=None)] for _ in range(24)]
    res = NS(power_mw=power, refused=refused, intraday_mw=z.copy())
    return M.settle_day(sched, real, res, dev, [100.0] * 24, services, None, DEG, ENUM)


def test_settlement_all_delivered():
    out = _day(np.zeros(86400, bool))
    assert out["capacity"] == pytest.approx(0.2 * 50 + 0.4 * 10)
    assert out["activation"] == pytest.approx(0.2 * 200)
    assert out["penalty"] == 0 and out["forfeited"] == 0 and out["reserve_failed_h"] == 0
    #Lorenzo Giannuzzo: base energy nets to zero, FCR charges 0.1 MWh at 100 EUR/MWh
    assert out["energy"] == pytest.approx(-10.0)
    assert out["net"] == pytest.approx(14 + 40 - 10 - DEG * 12.3)


def test_settlement_refused_activation():
    refused = np.zeros(86400, bool)
    refused[20 * 3600:20 * 3600 + 1800] = True          # half of hour 20 cannot discharge
    out = _day(refused)
    #Lorenzo Giannuzzo: 0.35 MWh missing, of which 0.2/0.7 belongs to the activation
    assert out["miss_mwh"] == pytest.approx(0.35)
    assert out["reserve_miss_mwh"] == pytest.approx(0.1)
    assert out["forfeited"] == pytest.approx(4.0) and out["reserve_failed_h"] == 1
    assert out["activation"] == pytest.approx(0.1 * 200)
    assert out["penalty"] == pytest.approx(M.PENALTY_K * 200 * 0.1)


def test_margins_direction():
    soc = np.r_[np.full(1000, 15.0), np.full(1000, 85.0)]
    err = np.r_[np.full(1000, 4.0), np.full(1000, -6.0)]       # over at the bottom, under at the top
    lib = {"soc_true": soc, "err_X": err}
    m = M.margins(lib, "X")
    assert m["band"][0] == pytest.approx(0.04) and m["band"][1] == pytest.approx(0.06)
    lo, hi = M.shrink(M.TECHNICAL_WINDOW, *m["band"])
    assert lo == pytest.approx(0.09) and hi == pytest.approx(0.89)
