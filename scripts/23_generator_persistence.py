"""
Persistence of the hourly SOC-estimation error, closed-loop library against generator.

For every estimator the script compares two measures of persistence between the hourly closed-loop
library and the synthetic sequences of the k-NN generator, each generator being built from one
reference week and driven by the true hourly SOC of that week:
- decorrelation time, the first lag at which the mean autocorrelation drops below 1/e [h];
- mean duration of the overestimation episodes (error above 2 points) while the true SOC lies
  below 30%, which are the episodes that cause failed upward deliveries [h].

Usage:
    python scripts/23_generator_persistence.py [path/to/error_library.npz]
The library defaults to the one written by 13_error_model.py.
"""

import sys

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.errormodel.generator import ErrorGenerator, load_library, SECONDS_PER_HOUR

MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")
REPLICATES = 20
SEED = 7
MAX_LAG_H = 72
OVER_THRESHOLD_PTS = 2.0
LOW_SOC_BAND = (5.0, 30.0)


def hourly(library, key, weeks):
    """Hourly snapshots of one library entry, one row per week."""
    return np.stack([library[key][library["week"] == w][::SECONDS_PER_HOUR] for w in weeks]).astype(float)


def decorrelation_h(rows, max_lag=MAX_LAG_H):
    """First lag at which the mean autocorrelation over the rows drops below 1/e [h]."""
    acf = []
    for x in rows:
        x = x - x.mean()
        c = np.correlate(x, x, "full")[len(x) - 1:len(x) + max_lag] / max(np.dot(x, x), 1e-12)
        acf.append(c)
    acf = np.mean(acf, axis=0)
    below = np.flatnonzero(acf < np.exp(-1))
    return float(below[0]) if below.size else float(max_lag)


def overestimation_episodes(soc, err, threshold=OVER_THRESHOLD_PTS, band=LOW_SOC_BAND):
    """Mean duration [h] and number of runs of consecutive hours with error > threshold inside the band."""
    lengths = []
    for s, e in zip(soc, err):
        mask = (e > threshold) & (s >= band[0]) & (s < band[1])
        n = 0
        for v in mask:
            if v:
                n += 1
            elif n:
                lengths.append(n)
                n = 0
        if n:
            lengths.append(n)
    return (float(np.mean(lengths)) if lengths else 0.0), len(lengths)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else paths.ERROR_DIR / "error_library.npz"
    library = load_library(path)
    weeks = np.unique(library["week"])
    soc_h = hourly(library, "soc_true", weeks)
    n_hours = soc_h.shape[1]
    rng = np.random.default_rng(SEED)
    rows = []
    for model in MODELS:
        err_h = hourly(library, f"err_{model}", weeks)
        synth = np.empty((REPLICATES,) + soc_h.shape)
        for w, soc_week in enumerate(soc_h):
            gen = ErrorGenerator.from_library(library, model, week=w, rng=rng)
            for r in range(REPLICATES):
                gen.reset()
                synth[r, w] = [gen.next(s) for s in soc_week]
        #Lorenzo Giannuzzo: synthetic sequences flattened to one row per (replicate, week)
        synth_rows = synth.reshape(-1, n_hours)
        soc_rows = np.broadcast_to(soc_h, synth.shape).reshape(-1, n_hours)
        dur_lib, n_lib = overestimation_episodes(soc_h, err_h)
        dur_gen, n_gen = overestimation_episodes(soc_rows, synth_rows)
        rows.append({"model": model,
                     "decorrelation_library_h": decorrelation_h(err_h),
                     "decorrelation_generator_h": decorrelation_h(synth_rows),
                     "episode_duration_library_h": dur_lib,
                     "episodes_library": n_lib,
                     "episode_duration_generator_h": dur_gen,
                     "episodes_generator_per_replicate": n_gen / REPLICATES})
    table = pd.DataFrame(rows).set_index("model")
    out = paths.ERROR_DIR / "generator_persistence.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 250,
                           "display.max_columns", None):
        print("persistence of the hourly error, library vs generator [h]:")
        print(table)
    print(f"\ntable written to {out}")


if __name__ == "__main__":
    main()
