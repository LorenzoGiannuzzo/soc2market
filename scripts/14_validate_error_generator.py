"""
Validation of the hourly error generator against the closed-loop error library.

Each generator is built from one reference week, driven by the true hourly SOC trajectory of that
week and run REPLICATES times. The hourly library and the synthetic sequences are then compared on the
quantities that matter to the operator: 5th and 95th percentile of the signed error per true-SOC
band, and decorrelation time of the error. The figure overlays the two bands per estimator.
"""

import numpy as np
import pandas as pd

from soc2market.errormodel.generator import ErrorGenerator, load_library, SECONDS_PER_HOUR
from soc2market import paths
from soc2market.figures.style import plt, save

ERR_DIR = paths.ERROR_DIR
MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")
LABELS = {"EKF": "Extended Kalman Filter", "ANN": "EKF + ANN", "CNN": "EKF + CNN", "LSTM": "EKF + LSTM",
          "CatBoost": "EKF + CatBoost", "TFT": "EKF + TFT", "Ensemble": "Ensemble Selector"}
SOC_EDGES = np.arange(10, 91, 10)
REPLICATES = 50
SEED = 2023 + 2


def hourly(library, key):
    """Hourly snapshots, one row per week."""
    weeks = np.unique(library["week"])
    return np.stack([library[key][library["week"] == w][::SECONDS_PER_HOUR] for w in weeks]).astype(float)


def decorrelation_h(rows, max_lag=48):
    """First lag at which the mean autocorrelation over the rows drops below 1/e [h]."""
    acf = []
    for x in rows:
        x = x - x.mean()
        c = np.correlate(x, x, "full")[len(x) - 1:len(x) + max_lag] / max(np.dot(x, x), 1e-12)
        acf.append(c)
    acf = np.mean(acf, axis=0)
    below = np.flatnonzero(acf < np.exp(-1))
    return float(below[0]) if below.size else float(max_lag)


def band_quantiles(soc, err):
    band = pd.cut(soc.ravel(), SOC_EDGES)
    g = pd.Series(err.ravel()).groupby(band, observed=False)
    return g.quantile(0.05).to_numpy(), g.quantile(0.95).to_numpy()


def main():
    library = load_library(ERR_DIR / "error_library.npz")
    soc_h = hourly(library, "soc_true")
    rng = np.random.default_rng(SEED)
    rows, curves = [], {}
    for model in MODELS:
        err_h = hourly(library, f"err_{model}")
        synth = np.empty((REPLICATES,) + soc_h.shape)
        for w, soc_week in enumerate(soc_h):
            gen = ErrorGenerator.from_library(library, model, week=w, rng=rng)
            for r in range(REPLICATES):
                gen.reset()
                synth[r, w] = [gen.next(s) for s in soc_week]
        lib_q = band_quantiles(soc_h, err_h)
        gen_q = band_quantiles(np.broadcast_to(soc_h, synth.shape), synth)
        curves[model] = (lib_q, gen_q)
        rows.append({"model": model,
                     "q05_library": np.quantile(err_h, 0.05), "q05_generator": np.quantile(synth, 0.05),
                     "q95_library": np.quantile(err_h, 0.95), "q95_generator": np.quantile(synth, 0.95),
                     "q99_abs_library": np.quantile(np.abs(err_h), 0.99),
                     "q99_abs_generator": np.quantile(np.abs(synth), 0.99),
                     "band_q_mae": float(np.nanmean(np.abs(np.concatenate(lib_q) - np.concatenate(gen_q)))),
                     "decorrelation_library_h": decorrelation_h(err_h),
                     "decorrelation_generator_h": decorrelation_h(synth.reshape(-1, soc_h.shape[1]))})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(ERR_DIR / "generator_validation.csv")
    with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 250,
                           "display.max_columns", None):
        print("hourly error, library vs generator [percentage points, hours]:")
        print(table)

    mid = 0.5 * (SOC_EDGES[:-1] + SOC_EDGES[1:])
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        (lo, hi), (glo, ghi) = curves[model]
        ax.fill_between(mid, lo, hi, color="#1f3b73", alpha=0.25, lw=0)
        ax.plot(mid, glo, color="#c0392b", lw=1.0, ls="--")
        ax.plot(mid, ghi, color="#c0392b", lw=1.0, ls="--")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(LABELS[model], fontsize=8)
    axes.flat[-1].axis("off")
    for ax in axes[1]:
        ax.set_xlabel("State Of Charge [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Estimation Error [%]")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f3b73", alpha=0.25),
               plt.Line2D([], [], color="#c0392b", ls="--")]
    leg = fig.legend(handles, ["Closed Loop (5th To 95th Percentile)", "Generator (5th And 95th Percentile)"],
                     loc="upper center", ncol=2, frameon=True, fancybox=False, edgecolor="black",
                     bbox_to_anchor=(0.5, 1.09))
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    save(fig, ERR_DIR / "generator_validation")
    print(f"\ntable and figure written to {ERR_DIR}")


if __name__ == "__main__":
    main()
