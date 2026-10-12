"""
Realisation of a daily schedule: which offers are awarded and which are called, and the
resulting one-second power profile of the pack.

Each product is awarded in an hour with its award probability, one Bernoulli draw per product
and hour as in the MILP. An awarded FCR band responds second by second to the measured grid
frequency. For an awarded restoration product (aFRR, mFRR) the activated share of the hour is
drawn from a Beta distribution whose mean is the activation probability used by the MILP, so
that the expected activated energy equals the one the MILP plans for, while single hours range
from almost nothing to full activation. The concentration of the Beta sets that spread: a very
large value reproduces the expectation exactly, a value close to zero degenerates into calling
the full band for the whole hour or not at all, which overstates the energy swing of a single
hour (aFRR setpoints change every few seconds and are rarely at full band for a full hour).
"""

from dataclasses import dataclass, field

import numpy as np

from soc2market.signals.profile import build_power_profile

UPWARD = ("afrr_up", "mfrr_up")
#Lorenzo Giannuzzo: concentration of the Beta distribution of the activated share (modelling
# assumption, to be varied in a sensitivity analysis)
ACTIVATION_CONCENTRATION = 4.0
DOWNWARD = ("afrr_dn", "mfrr_dn")


@dataclass
class Realisation:
    awarded: dict          # product -> bool[24]
    called: dict           # product -> bool[24], awarded with a non-negligible activated share
    share: dict            # product -> activated share of the hour, float[24]
    fcr_band_mw: np.ndarray
    act_up_mw: np.ndarray
    act_down_mw: np.ndarray
    energy_price: dict = field(default=None)   # product -> realised EUR/MWh[24] (chronological MSD only)


def activated_share(p, size, rng, concentration=ACTIVATION_CONCENTRATION):
    """Beta-distributed share of the hour with mean p (0 and 1 handled exactly)."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    out = np.where(p >= 1.0, 1.0, 0.0)
    inner = (p > 0.0) & (p < 1.0)
    if inner.any():
        out[inner] = rng.beta(p[inner] * concentration, (1.0 - p[inner]) * concentration)
    return out


def draw_realisation(schedule, rng, concentration=ACTIVATION_CONCENTRATION):
    """Award draw for every product and hour, activated share for the restoration products."""
    awarded, called, share = {}, {}, {}
    for key, band in schedule.band_mw.items():
        offered = band > 1e-9
        awarded[key] = offered & (rng.random(band.size) < schedule.award_prob[key])
        frac = activated_share(schedule.activation_prob[key], band.size, rng, concentration)
        share[key] = np.where(awarded[key], frac, 0.0) if key != "fcr" else np.zeros(band.size)
        called[key] = share[key] > 1e-3
    fcr = schedule.band_mw["fcr"] * awarded["fcr"]
    up = sum(schedule.activation_mw[k] * share[k] for k in UPWARD)
    down = sum(schedule.activation_mw[k] * share[k] for k in DOWNWARD)
    return Realisation(awarded, called, share, fcr, np.asarray(up, float), np.asarray(down, float))


#Lorenzo Giannuzzo: same thinning of the upward and downward MSD activity used by the frozen market
# window to derive the mFRR award probability from the MSD profiles
MFRR_AWARD_SCALE = 0.6


def draw_realisation_chronological(schedule, msd_day, rng, concentration=ACTIVATION_CONCENTRATION,
                                   mfrr_scale=MFRR_AWARD_SCALE):
    """
    Realisation on the MSD outcome of the same calendar day: an offered aFRR band is awarded in
    the hours called in its direction (MSDDay.called_up / called_dn), an offered mFRR band in those hours with
    probability mfrr_scale; the activated share is drawn as in draw_realisation and the energy is
    settled at the MSD price of the hour. FCR is not traded in MSD and keeps its Bernoulli award.
    The random stream is consumed exactly as in draw_realisation, so that the two modes share
    their draws wherever they coincide.
    """
    awarded, called, share, price = {}, {}, {}, {}
    for key, band in schedule.band_mw.items():
        offered = band > 1e-9
        u = rng.random(band.size)
        frac = activated_share(schedule.activation_prob[key], band.size, rng, concentration)
        if key == "fcr":
            awarded[key] = offered & (u < schedule.award_prob[key])
            share[key] = np.zeros(band.size)
        else:
            up = key in UPWARD
            active = msd_day.called_up if up else msd_day.called_dn
            market = active & (u < mfrr_scale) if key.startswith("mfrr") else active
            awarded[key] = offered & market
            share[key] = np.where(awarded[key], frac, 0.0)
            price[key] = np.asarray(msd_day.up_price if up else msd_day.dn_price, float)
        called[key] = share[key] > 1e-3
    fcr = schedule.band_mw["fcr"] * awarded["fcr"]
    up = sum(schedule.activation_mw[k] * share[k] for k in UPWARD)
    down = sum(schedule.activation_mw[k] * share[k] for k in DOWNWARD)
    return Realisation(awarded, called, share, fcr, np.asarray(up, float), np.asarray(down, float),
                       energy_price=price)


def day_power_profile(schedule, realisation, dev_hz, rated_mw):
    """One-second pack power (MW, positive charging) and the energy clipped at the rating (MWh)."""
    return build_power_profile(schedule.base_mw, realisation.fcr_band_mw, dev_hz,
                               act_up_mw=realisation.act_up_mw, act_down_mw=realisation.act_down_mw,
                               rated_mw=rated_mw)
