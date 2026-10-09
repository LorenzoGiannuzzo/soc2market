"""
Hourly generator of SOC-estimation errors, resampled from the closed-loop error library.

The operator reads the SOC once per hour, so the market study needs the error at hourly resolution
only. The error of each estimator depends on where the battery is (true-SOC band) and persists
for hours (EKF: about a day), so it cannot be drawn independently hour by hour. The generator is a
k-nearest-neighbour conditional bootstrap (Lall and Sharma, 1996): given the current true SOC and
the error of the previous hour, it finds the library instants with the most similar pair and returns
the error observed one hour later at one of them, chosen with weights 1/j over the k neighbours.
State dependence and persistence are thereby inherited from the closed-loop runs, including the
heavy tails, without fitting a parametric model.

The library holds four seasonal weeks, and the slow part of the EKF error (a bias that drifts over
days) differs between them. Each generator is therefore built from the week that the market study
replays, which keeps the seasonal level of the error. With a one-hour memory the generator still
shortens the persistence of the EKF error (about 10 h against 28 h in closed loop), so `ReplayError`
is provided as a check: it returns the closed-loop error recorded at the same hour of the same week,
with its full memory but without any dependence on where the battery actually is.
"""

import numpy as np

SECONDS_PER_HOUR = 3600
SECONDS_PER_WEEK = 7 * 24 * SECONDS_PER_HOUR
PAIR_STRIDE_S = 600          # one transition pair every 10 min, all within the same week
SOC_SCALE_PTS = 5.0          # distance of 5 points of SOC counts as one unit
ERR_SCALE_FRACTION = 0.25    # error distance in units of 0.25 standard deviations of the error
DEFAULT_K = 10               # neighbours, chosen on the validation of 14_validate_error_generator.py


def load_library(path):
    """Error library written by 13_error_model.py: dict with soc_true, week and err_<model> [%]."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


class ErrorGenerator:
    """Hourly error sequence for one estimator, conditioned on the true SOC [%]."""

    def __init__(self, soc_true, week, err, k=None, rng=None):
        soc_true = np.asarray(soc_true, float)
        err = np.asarray(err, float)
        week = np.asarray(week)
        now, prev = [], []
        for w in np.unique(week):
            idx = np.flatnonzero(week == w)
            t = np.arange(idx[0] + SECONDS_PER_HOUR, idx[-1] + 1, PAIR_STRIDE_S)
            now.append(t)
            prev.append(t - SECONDS_PER_HOUR)
        now, prev = np.concatenate(now), np.concatenate(prev)
        self.soc = soc_true[now]
        self.err_prev = err[prev]
        self.err_next = err[now]
        self.err_scale = max(ERR_SCALE_FRACTION * np.std(err), 1e-3)
        self.k = min(k or DEFAULT_K, len(now) - 1)
        weights = 1.0 / np.arange(1, self.k + 1)
        self.weights = weights / weights.sum()
        self.rng = rng or np.random.default_rng()
        self.state = None

    @classmethod
    def from_library(cls, library, model, week=None, **kwargs):
        """Generator for `model`, built from one week of the library (all weeks if `week` is None)."""
        mask = np.ones(len(library["week"]), bool) if week is None else library["week"] == week
        return cls(library["soc_true"][mask], library["week"][mask], library[f"err_{model}"][mask], **kwargs)

    def reset(self):
        self.state = None

    def _pick(self, distance):
        nearest = np.argpartition(distance, self.k)[:self.k]
        nearest = nearest[np.argsort(distance[nearest])]
        return nearest[self.rng.choice(self.k, p=self.weights)]

    def next(self, soc_pct):
        """Error [percentage points] for the hour that starts at true SOC `soc_pct` [%]."""
        d_soc = (self.soc - soc_pct) / SOC_SCALE_PTS
        if self.state is None:
            #Lorenzo Giannuzzo: first hour, condition on the SOC only and take the error at that instant
            j = self._pick(np.abs(d_soc))
            e = self.err_prev[j]
        else:
            d_err = (self.err_prev - self.state) / self.err_scale
            j = self._pick(np.hypot(d_soc, d_err))
            e = self.err_next[j]
        self.state = float(e)
        return self.state

    def soc_seen_fn(self, lower=0.0, upper=1.0):
        """Callback for execute_day: the SOC the operator believes, as a fraction, clipped to [lower, upper]."""
        def seen(state, hour):
            return float(np.clip(state.soc + self.next(100.0 * state.soc) / 100.0, lower, upper))
        return seen


class ReplayError:
    """Closed-loop error of one week replayed hour by hour, independent of the simulated SOC."""

    def __init__(self, library, model, week):
        mask = library["week"] == week
        self.err = np.asarray(library[f"err_{model}"][mask], float)[::SECONDS_PER_HOUR]
        self.hour = 0

    def reset(self):
        self.hour = 0

    def next(self, soc_pct=None):
        e = float(self.err[self.hour % len(self.err)])
        self.hour += 1
        return e

    def soc_seen_fn(self, lower=0.0, upper=1.0):
        def seen(state, hour):
            return float(np.clip(state.soc + self.next() / 100.0, lower, upper))
        return seen
