"""
From an hourly market schedule to the one-second power profile of the pack.

Sign convention as in the plant: positive power charges the battery.

    P(t) = P_base(h) + R_FCR(h) * fcr_response(df(t)) + A_down(h) - A_up(h)

where P_base = P_charge - P_discharge of the schedule, R_FCR the FCR band, fcr_response the
droop below, and A_up / A_down the power of upward and downward aFRR/mFRR activations held
constant over the hour (whether an hour is activated is decided by the caller).
"""

import numpy as np

#Lorenzo Giannuzzo: FCR droop for Continental Europe (SO GL Annex V): full activation at 200 mHz, insensitivity of 10 mHz
FCR_FULL_ACTIVATION_HZ = 0.200
FCR_DEADBAND_HZ = 0.010


def fcr_response(dev_hz, full_activation_hz=FCR_FULL_ACTIVATION_HZ, deadband_hz=FCR_DEADBAND_HZ):
    """
    Share of the FCR band to deliver (positive = absorb, i.e. charge), from the frequency
    deviation: proportional to the deviation, zero inside the deadband, saturated at +/-1.
    """
    dev = np.asarray(dev_hz, dtype=float)
    share = np.clip(dev / full_activation_hz, -1.0, 1.0)
    share[np.abs(dev) <= deadband_hz] = 0.0
    return share


def build_power_profile(base_mw, fcr_band_mw, dev_hz, act_up_mw=None, act_down_mw=None,
                        rated_mw=None, seconds_per_hour=3600):
    """
    One-second pack power (MW, positive charging) for one day.

    base_mw, fcr_band_mw, act_up_mw, act_down_mw: 24 hourly values (MW).
    dev_hz: frequency deviation, 24 * seconds_per_hour values (Hz).
    rated_mw: if given, the profile is clipped to +/- rated_mw and the clipped energy reported.
    Returns (profile, clipped_mwh).
    """
    n_hours = len(base_mw)
    zeros = np.zeros(n_hours)
    up = np.asarray(act_up_mw if act_up_mw is not None else zeros, float)
    dn = np.asarray(act_down_mw if act_down_mw is not None else zeros, float)
    hourly = np.asarray(base_mw, float) + dn - up
    profile = np.repeat(hourly, seconds_per_hour)
    dev = np.asarray(dev_hz, float)
    if dev.size != profile.size:
        raise ValueError(f"frequency series has {dev.size} values, expected {profile.size}")
    profile = profile + np.repeat(np.asarray(fcr_band_mw, float), seconds_per_hour) * fcr_response(dev)
    clipped_mwh = 0.0
    if rated_mw is not None:
        excess = np.abs(profile) - rated_mw
        clipped_mwh = float(np.sum(excess[excess > 0]) / 3600.0)
        profile = np.clip(profile, -rated_mw, rated_mw)
    return profile, clipped_mwh
