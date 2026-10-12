"""
Grid frequency of the Continental European synchronous area, one value per second.

Source: Zenodo record 15784548 (Nikoltchovska et al., pre-processed TransnetBW data), file
Data_cleansed/.../TransnetBW/2023, a header-less csv with a naive local timestamp (CET with
daylight saving time) and the frequency in Hz; corrupted samples are marked as NaN.

The csv is converted once into a matrix of days x 86400 seconds holding the deviation from
50 Hz in hertz, with NaN where no sample exists. Local time means that the last Sunday of
March lacks one hour (left as NaN) and the last Sunday of October repeats one (the later
sample overwrites the earlier one).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#Lorenzo Giannuzzo: repository layout, resolved from this file
REPO_ROOT = Path(__file__).resolve().parents[3]
FREQ_DIR = REPO_ROOT / "data" / "frequency"
NOMINAL_HZ = 50.0
SECONDS_PER_DAY = 86400
#Lorenzo Giannuzzo: deviations beyond this bound are physically implausible in normal operation of
# the Continental European area (under-frequency load shedding starts at 49.0 Hz) and are treated
# as missing samples; FCR saturates at 200 mHz, so the bound does not affect any genuine response
PLAUSIBLE_DEV_HZ = 0.5


def convert_frequency_csv(csv_path, out_path, year, chunksize=5_000_000):
    """Convert the yearly csv into an .npz with `dev_hz` (days x 86400, float32) and `year`."""
    first = pd.Timestamp(year=year, month=1, day=1)
    n_days = (pd.Timestamp(year=year + 1, month=1, day=1) - first).days
    dev = np.full((n_days, SECONDS_PER_DAY), np.nan, dtype=np.float32)
    rows = 0
    for chunk in pd.read_csv(csv_path, header=None, names=["ts", "f"], chunksize=chunksize,
                             na_values=["NaN", "nan"]):
        ts = pd.to_datetime(chunk["ts"], format="%Y-%m-%d %H:%M:%S")
        day = (ts.dt.normalize() - first).dt.days.to_numpy()
        sec = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).to_numpy()
        keep = (day >= 0) & (day < n_days)
        dev[day[keep], sec[keep]] = (chunk["f"].to_numpy(dtype=np.float64)[keep] - NOMINAL_HZ).astype(np.float32)
        rows += len(chunk)
    np.savez(out_path, dev_hz=dev, year=year)
    return dev, rows


@dataclass
class FrequencyArchive:
    """Daily access to the converted frequency deviation."""

    dev_hz: np.ndarray
    year: int

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(dev_hz=z["dev_hz"], year=int(z["year"]))

    @property
    def n_days(self):
        return self.dev_hz.shape[0]

    def missing_share(self):
        """Share of missing seconds, per day."""
        return np.isnan(self.dev_hz).mean(axis=1)

    def day(self, index, max_gap_s=60):
        """
        Deviation from 50 Hz for one day (86400 values, Hz). Samples beyond PLAUSIBLE_DEV_HZ are
        treated as missing. Gaps up to `max_gap_s` seconds are filled by linear interpolation;
        longer gaps are set to zero, that is to no FCR activation.
        Returns the series and the number of seconds set to zero.
        """
        x = self.dev_hz[index].astype(np.float64)
        x[np.abs(x) > PLAUSIBLE_DEV_HZ] = np.nan
        isnan = np.isnan(x)
        if not isnan.any():
            return x, 0
        #Lorenzo Giannuzzo: runs of missing seconds; short inner runs are interpolated, every other run
        # (long, or touching the start or end of the day) is set to zero as a whole
        edges = np.diff(np.r_[0, isnan.astype(int), 0])
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        good = np.flatnonzero(~isnan)
        out = x.copy()
        zeroed = 0
        for a, b in zip(starts, ends):
            inner = a > 0 and b < x.size
            if inner and (b - a) <= max_gap_s and good.size:
                out[a:b] = np.interp(np.arange(a, b), good, x[good])
            else:
                out[a:b] = 0.0
                zeroed += b - a
        return out, int(zeroed)
