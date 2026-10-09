"""
Chronological MSD ex-ante outcomes of the northern zone (Terna), hour by hour.

The MILP plans with award probabilities taken from MSD profiles conditioned on season, day type
and hour, which is what an operator can know the day before. The realisation, instead, follows
the hours in which Terna actually accepted offers in 2023: an hour is active upward when the
average price of the accepted sale offers (Prezzo Medio di Vendita, the price the provider
receives on upward regulation) is positive, and active downward when the average price of the
accepted purchase offers (Prezzo Medio di Acquisto, the price the provider pays on downward
regulation) is positive. Sequences of consecutive activations, which drain or fill a battery
far more than independent hourly draws, are thereby preserved. Direction mapping as in the
frozen italian_market_data.msd_hourly_profiles.

In most hours of 2023 MSD ex-ante accepted offers in BOTH directions (congestion and unit
commitment are resolved there too, not only balancing). The frozen market window handles this by
normalising the upward and downward rates so that each hour is called in one direction only,
and the MILP plans with those normalised probabilities. The realisation follows the same rule:
an hour active in one direction only is called in that direction; an hour active in both is
called in one of them, chosen with a fair coin that is fixed for that calendar hour (seeded by
year and day), so that every strategy, estimator and replicate faces the same history.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

COL_DATE, COL_HOUR = "Data", "Ora"
COL_UP = "Prezzo Medio di Vendita (€/MWh)"
COL_DN = "Prezzo Medio di Acquisto (€/MWh)"


def _to_float(series):
    """Numbers as stored by Terna: numeric cells, or text with a decimal comma."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip()
    comma = text.str.contains(",", regex=False)
    text = text.where(~comma, text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def _to_date(series):
    dates = pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")
    missing = dates.isna()
    if missing.any():
        dates[missing] = pd.to_datetime(series[missing], dayfirst=True, errors="coerce")
    return dates.dt.normalize()


@dataclass
class MSDDay:
    up_active: np.ndarray      # bool[24]
    up_price: np.ndarray       # EUR/MWh, NaN where not active
    dn_active: np.ndarray
    dn_price: np.ndarray
    missing_hours: int
    called_up: np.ndarray      # bool[24], hour called upward (mutually exclusive with called_dn)
    called_dn: np.ndarray


class MSDArchive:
    """Hourly MSD outcomes of one year, indexed by day of the year (0-based) and hour."""

    def __init__(self, year, up_price, dn_price):
        self.year = year
        self.up_price = up_price          # float[n_days, 24], 0 when not active, NaN when missing
        self.dn_price = dn_price

    @classmethod
    def load(cls, paths, year):
        frames = [pd.read_excel(p) for p in paths]
        df = pd.concat(frames, ignore_index=True)
        df["date"] = _to_date(df[COL_DATE])
        df["hour"] = pd.to_numeric(df[COL_HOUR], errors="coerce") - 1
        #Lorenzo Giannuzzo: the 25th hour of the daylight-saving day has no slot on the 24-hour grid
        df = df[(df["date"].dt.year == year) & df["hour"].between(0, 23)]
        df = df.drop_duplicates(subset=["date", "hour"], keep="last")
        start = datetime(year, 1, 1)
        n_days = (datetime(year + 1, 1, 1) - start).days
        up = np.full((n_days, 24), np.nan)
        dn = np.full((n_days, 24), np.nan)
        day = (df["date"] - pd.Timestamp(start)).dt.days.to_numpy()
        hour = df["hour"].to_numpy(int)
        up[day, hour] = _to_float(df[COL_UP]).fillna(0.0).to_numpy()
        dn[day, hour] = _to_float(df[COL_DN]).fillna(0.0).to_numpy()
        return cls(year, up, dn)

    def day(self, index):
        up, dn = self.up_price[index], self.dn_price[index]
        missing = int(np.isnan(up).sum())
        up_active = np.nan_to_num(up) > 0
        dn_active = np.nan_to_num(dn) > 0
        #Lorenzo Giannuzzo: one direction per hour, fair coin fixed by calendar hour when both are active
        coin_up = np.random.default_rng([self.year, index]).random(24) < 0.5
        both = up_active & dn_active
        called_up = (up_active & ~dn_active) | (both & coin_up)
        called_dn = (dn_active & ~up_active) | (both & ~coin_up)
        return MSDDay(up_active=up_active, up_price=np.where(up_active, up, np.nan),
                      dn_active=dn_active, dn_price=np.where(dn_active, dn, np.nan), missing_hours=missing,
                      called_up=called_up, called_dn=called_dn)

    def date_of(self, index):
        return datetime(self.year, 1, 1) + timedelta(days=index)