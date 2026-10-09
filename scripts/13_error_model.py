"""
Error model of the published SOC estimators on the reference duty cycles.

For every estimator the signed error (estimate minus true SOC, in percentage points) is
summarised by true-SOC band and by hourly operating regime, its persistence is measured as
the decorrelation time of its autocorrelation, and the full traces are stored in an error
library for the market study. A figure shows the median and the 5th to 95th percentile band
of the signed error against the true SOC for every estimator.
"""

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.figures.style import plt, save

DUTY_DIR = paths.DUTY_DIR
LOOP_DIR = paths.LOOP_DIR
OUT_DIR = paths.ERROR_DIR
WEEKS = ("winter", "spring", "summer", "autumn")
MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")
LABELS = {"EKF": "Extended Kalman Filter", "ANN": "EKF + ANN", "CNN": "EKF + CNN", "LSTM": "EKF + LSTM",
          "CatBoost": "EKF + CatBoost", "TFT": "EKF + TFT", "Ensemble": "Ensemble Selector"}
SOC_BINS = np.arange(10, 91, 10)
RESTORATION = ("afrr_up", "afrr_dn", "mfrr_up", "mfrr_dn")
#Lorenzo Giannuzzo: hourly regime thresholds, MW of mean pack power and activated share of the hour
POWER_THRESHOLD_MW = 0.05
SHARE_THRESHOLD = 0.05


def hourly_regimes(week):
    """One label per hour of the week, from the realised operation stored in the duty-cycle files."""
    labels = []
    for f in sorted((DUTY_DIR / week).glob("day_*.npz")):
        with np.load(f) as z:
            p_h = z["power_mw"].astype(float).reshape(24, 3600).mean(axis=1)
            activated = np.zeros(24, bool)
            for k in RESTORATION:
                activated |= z[f"share_{k}"] > SHARE_THRESHOLD
            fcr = z["fcr_awarded"].astype(bool)
        for h in range(24):
            if activated[h]:
                labels.append("Reserve Activation")
            elif p_h[h] > POWER_THRESHOLD_MW:
                labels.append("Charging")
            elif p_h[h] < -POWER_THRESHOLD_MW:
                labels.append("Discharging")
            elif fcr[h]:
                labels.append("FCR Only")
            else:
                labels.append("Rest")
    return np.repeat(np.array(labels), 3600)


def decorrelation_time_h(err, max_lag_h=48):
    """First lag (hours) at which the autocorrelation of the demeaned error falls below 1/e."""
    x = err - err.mean()
    n = x.size
    spec = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(spec * np.conj(spec))[: max_lag_h * 3600 + 1]
    acf = acf / acf[0]
    below = np.flatnonzero(acf < np.exp(-1.0))
    return float(below[0] / 3600.0) if below.size else float(max_lag_h)


def signed_stats(e):
    return dict(n=int(e.size), q05=float(np.percentile(e, 5)), median=float(np.median(e)),
                q95=float(np.percentile(e, 95)), rmse=float(np.sqrt(np.mean(e ** 2))),
                max_over=float(e.max()), max_under=float(e.min()))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regimes = {w: hourly_regimes(w) for w in WEEKS}
    library = {}
    band_rows, regime_rows, persistence_rows = [], [], []
    soc_true_all, week_all = None, None

    for model in MODELS:
        errs, socs, regs, weeks = [], [], [], []
        for w in WEEKS:
            with np.load(LOOP_DIR / f"{w}_{model}.npz") as z:
                e = z["soc_hat"].astype(float) - z["soc_true"].astype(float)
                s = z["soc_true"].astype(float)
            if e.size != regimes[w].size:
                raise ValueError(f"{w} {model}: {e.size} samples against {regimes[w].size} regime labels")
            persistence_rows.append(dict(model=model, week=w, decorrelation_h=decorrelation_time_h(e)))
            errs.append(e); socs.append(s); regs.append(regimes[w]); weeks.append(np.full(e.size, WEEKS.index(w)))
        e, s, r = np.concatenate(errs), np.concatenate(socs), np.concatenate(regs)
        library[f"err_{model}"] = e.astype(np.float32)
        if soc_true_all is None:
            soc_true_all, week_all = s, np.concatenate(weeks)

        for lo, hi in zip(SOC_BINS[:-1], SOC_BINS[1:]):
            m = (s >= lo) & (s < hi)
            if m.sum() > 0:
                band_rows.append(dict(model=model, soc_band=f"{lo}-{hi}", soc_mid=(lo + hi) / 2, **signed_stats(e[m])))
        for reg in ("Rest", "FCR Only", "Charging", "Discharging", "Reserve Activation"):
            m = r == reg
            if m.sum() > 0:
                regime_rows.append(dict(model=model, regime=reg, hours=m.sum() / 3600, **signed_stats(e[m])))

    bands = pd.DataFrame(band_rows)
    reg = pd.DataFrame(regime_rows)
    pers = pd.DataFrame(persistence_rows)
    bands.to_csv(OUT_DIR / "error_by_soc_band.csv", index=False)
    reg.to_csv(OUT_DIR / "error_by_regime.csv", index=False)
    pers.to_csv(OUT_DIR / "decorrelation_time.csv", index=False)
    np.savez_compressed(OUT_DIR / "error_library.npz", soc_true=soc_true_all.astype(np.float32),
                        week=week_all.astype(np.int8), **library)

    pd.set_option("display.width", 200)
    fmt = lambda x: f"{x:.2f}"
    print("signed error by true-SOC band [percentage points]:")
    print(bands.pivot(index="soc_band", columns="model", values="q95")[list(MODELS)].to_string(float_format=fmt))
    print("\n5th percentile (negative = underestimation):")
    print(bands.pivot(index="soc_band", columns="model", values="q05")[list(MODELS)].to_string(float_format=fmt))
    print("\nsigned error by operating regime, 5th / 95th percentile:")
    reg["q05/q95"] = reg["q05"].map(fmt) + " / " + reg["q95"].map(fmt)
    print(reg.pivot(index="regime", columns="model", values="q05/q95")[list(MODELS)].to_string())
    print("\nhours per regime:", reg[reg.model == "EKF"].set_index("regime")["hours"].round(1).to_dict())
    print("\ndecorrelation time of the error [h]:")
    print(pers.pivot(index="week", columns="model", values="decorrelation_h")[list(MODELS)].to_string(float_format=fmt))

    #Lorenzo Giannuzzo: paper figure, one panel per estimator, shared axes
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.6), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, MODELS):
        b = bands[bands.model == model]
        ax.fill_between(b.soc_mid, b.q05, b.q95, color="#4C72B0", alpha=0.35, linewidth=0)
        ax.plot(b.soc_mid, b["median"], color="#1f3b73", lw=1.2)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_title(LABELS[model], fontsize=7.5)
    axes.flat[-1].axis("off")
    for ax in axes[1]:
        ax.set_xlabel("State Of Charge [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Estimation Error [%]")
    handles = [plt.Line2D([], [], color="#1f3b73", lw=1.2),
               plt.Rectangle((0, 0), 1, 1, color="#4C72B0", alpha=0.35, linewidth=0)]
    leg = fig.legend(handles, ["Median", "5th To 95th Percentile"], loc="upper center", ncol=2, frameon=True,
                     fancybox=False, edgecolor="black", bbox_to_anchor=(0.5, 1.0))
    leg.get_frame().set_linewidth(0.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.3)
    save(fig, OUT_DIR / "error_vs_soc")
    print(f"\ntables, library and figure written to {OUT_DIR}")


if __name__ == "__main__":
    main()