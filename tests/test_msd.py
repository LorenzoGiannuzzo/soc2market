"""Chronological MSD reader and realisation on the MSD outcome of the day."""

from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from soc2market.signals.duty_cycle import draw_realisation, draw_realisation_chronological
from soc2market.signals.msd import COL_DN, COL_UP, MSDArchive


def _write(tmp_path):
    #Lorenzo Giannuzzo: two days in Terna format, decimal commas, a 25th hour, a missing hour and an
    # overlapping second file that corrects one value
    rows = []
    for d in ("02/01/2023", "03/01/2023"):
        for h in range(1, 25):
            if d == "03/01/2023" and h == 5:
                continue
            rows.append({"Data": d, "Ora": h, COL_UP: "250,5" if h in (8, 9) else "0",
                         COL_DN: "90,25" if h == 3 else "0"})
    rows.append({"Data": "02/01/2023", "Ora": 25, COL_UP: "999", COL_DN: "999"})
    a = tmp_path / "a.xlsx"
    pd.DataFrame(rows).to_excel(a, index=False)
    b = tmp_path / "b.xlsx"
    pd.DataFrame([{"Data": "02/01/2023", "Ora": 10, COL_UP: "1.200,0", COL_DN: "0"}]).to_excel(b, index=False)
    return [a, b]


def test_archive_reads_terna_format(tmp_path):
    msd = MSDArchive.load(_write(tmp_path), 2023)
    d = msd.day(1)                                   # 2 January
    assert list(np.flatnonzero(d.up_active)) == [7, 8, 9]
    assert d.up_price[7] == pytest.approx(250.5) and d.up_price[9] == pytest.approx(1200.0)
    assert list(np.flatnonzero(d.dn_active)) == [2] and d.dn_price[2] == pytest.approx(90.25)
    assert d.missing_hours == 0
    #Lorenzo Giannuzzo: one direction per hour; hours active in one direction only keep it
    assert not (d.called_up & d.called_dn).any()
    assert d.called_up[[7, 8, 9]].all() and d.called_dn[2]
    assert msd.day(2).missing_hours == 1 and msd.day(0).missing_hours == 24


def _schedule():
    z = np.zeros(24)
    band = {k: z.copy() for k in ("fcr", "afrr_up", "afrr_dn", "mfrr_up", "mfrr_dn")}
    band["afrr_up"][:] = 0.3
    band["mfrr_dn"][:] = 0.2
    band["fcr"][:] = 0.1
    prob = {k: np.full(24, 0.5) for k in band}
    return NS(band_mw=band, activation_mw={k: v.copy() for k, v in band.items()},
              award_prob=prob, activation_prob={k: np.full(24, 0.3) for k in band})


def test_chronological_awards_follow_msd_and_share_the_random_stream():
    up = np.zeros(24, bool); up[[8, 9]] = True
    dn = np.zeros(24, bool); dn[2] = True
    day = NS(up_active=up, dn_active=dn, called_up=up, called_dn=dn, up_price=np.where(up, 250.0, np.nan), dn_price=np.where(dn, 90.0, np.nan))
    s = _schedule()
    chrono = draw_realisation_chronological(s, day, np.random.default_rng(3))
    bern = draw_realisation(s, np.random.default_rng(3))
    assert list(np.flatnonzero(chrono.awarded["afrr_up"])) == [8, 9]
    assert not chrono.awarded["afrr_dn"].any()
    assert set(np.flatnonzero(chrono.awarded["mfrr_dn"])) <= {2}
    assert chrono.energy_price["afrr_up"][8] == 250.0
    #Lorenzo Giannuzzo: FCR is not in MSD, its award draw must be identical in the two modes
    assert np.array_equal(chrono.awarded["fcr"], bern.awarded["fcr"])
    assert np.allclose(chrono.act_up_mw[[8, 9]], 0.3 * chrono.share["afrr_up"][[8, 9]])
