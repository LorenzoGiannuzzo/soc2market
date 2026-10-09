"""
Figures of the PES GM 2027 paper, from the results of the chronological MSD run.

    fig_error_vs_soc       signed estimation error against the true SOC (median and 5th to 95th
                           percentile per 10-point band), one panel per published estimator
    fig_loss_vs_metrics    annual loss against the oracle of the naive strategy (no margin) against
                           the RMSE and the MAE of the closed-loop error, with one standard error
    fig_margin_sweep       net revenue change against the oracle without margin as a function of the
                           symmetric SOC margin, for the mildest and the most severe penalty factor
    fig_margin_mechanism   for the true SOC: energy revenue given up and penalties plus forfeited
                           capacity avoided by the margin, for the same two penalty factors

Single-column figures are 3.5 in wide and double-column figures 7.16 in, as in the IEEE template.
"""

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.errormodel.generator import load_library
from soc2market.figures.style import plt, save
from soc2market.simulation import market as M

OUT_DIR = paths.RUN_DIR / "paper_figures"
STUDY_DIR = paths.RUN_DIR / "market_study_year"
SWEEP_DIR = paths.RUN_DIR / "margin_sweep"
ALL_MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")
SWEEP_MODELS = ("true", "EKF", "TFT", "CatBoost", "CNN")
LABELS = {"true": "Perfect Knowledge", "EKF": "Extended Kalman Filter",
          "ANN": "EKF + Artificial Neural Network", "CNN": "EKF + Convolutional Neural Network",
          "LSTM": "EKF + Long Short-Term Memory", "CatBoost": "EKF + CatBoost",
          "TFT": "EKF + Temporal Fusion Transformer", "Ensemble": "Ensemble Selector"}
COLORS = {"true": "black", "EKF": "#1f3b73", "ANN": "#7d3c98", "CNN": "#d68910", "LSTM": "#17a589",
          "CatBoost": "#c0392b", "TFT": "#2e8b57", "Ensemble": "#7f8c8d"}
MARKERS = {"true": "s", "EKF": "o", "ANN": "v", "CNN": "^", "LSTM": "D", "CatBoost": "P", "TFT": "X", "Ensemble": "*"}
PENALTIES = (1.5, 10.0)
SOC_EDGES = np.arange(10, 91, 10)


def boxed_legend(fig, handles, labels, ncol, y=1.02, fontsize=7):
    leg = fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=True, fancybox=False,
                     edgecolor="black", bbox_to_anchor=(0.5, y), fontsize=fontsize)
    leg.get_frame().set_linewidth(0.8)
    return leg


def paired(days, k, reference, keys=("config",)):
    """Annualised net change against `reference`, paired on block and replicate, at penalty factor k."""
    net = days["net"] - days["penalty"] * (k / M.PENALTY_K - 1.0)
    d = days.assign(x=net).groupby(["config", "week", "rep"]).agg(x=("x", "sum"), n=("x", "size")).reset_index()
    d["x"] *= 365.0 / d["n"]
    ref = d[d["config"] == reference][["week", "rep", "x"]].rename(columns={"x": "r"})
    d = d.merge(ref, on=["week", "rep"])
    d["change"] = d["x"] - d["r"]
    g = d.groupby("config")["change"]
    return pd.DataFrame({"change": g.mean(), "se": g.std() / np.sqrt(g.size())})


def fig_error_vs_soc(library):
    soc = np.asarray(library["soc_true"], float)
    band = pd.cut(soc, SOC_EDGES)
    mid = 0.5 * (SOC_EDGES[:-1] + SOC_EDGES[1:])
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.8), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, ALL_MODELS):
        g = pd.Series(np.asarray(library[f"err_{m}"], float)).groupby(band, observed=False)
        ax.fill_between(mid, g.quantile(0.05), g.quantile(0.95), color="#1f3b73", alpha=0.25, lw=0)
        ax.plot(mid, g.median(), color="#1f3b73", lw=1.2)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(LABELS[m].replace("EKF + ", "EKF +\n"), fontsize=6.5)
    axes.flat[-1].axis("off")
    #Lorenzo Giannuzzo: the last panel of the top row has no panel below it, so it carries its own x axis
    axes[0, 3].tick_params(labelbottom=True)
    for ax in list(axes[1, :3]) + [axes[0, 3]]:
        ax.set_xlabel("State Of Charge [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Estimation Error [%]")
    handles = [plt.Line2D([], [], color="#1f3b73", lw=1.2), plt.Rectangle((0, 0), 1, 1, color="#1f3b73", alpha=0.25)]
    fig.tight_layout()
    boxed_legend(fig, handles, ["Median", "5th To 95th Percentile"], 2)
    save(fig, OUT_DIR / "fig_error_vs_soc")


def fig_loss_vs_metrics(library, days):
    res = paired(days, M.PENALTY_K, "oracle")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)
    for ax, metric in zip(axes, ("RMSE", "MAE")):
        for m in ALL_MODELS:
            e = np.asarray(library[f"err_{m}"], float)
            x = np.sqrt(np.mean(e ** 2)) if metric == "RMSE" else np.mean(np.abs(e))
            loss, se = -res.loc[f"{m}_naive", "change"], res.loc[f"{m}_naive", "se"]
            ax.errorbar(x, loss, yerr=se, color=COLORS[m], marker=MARKERS[m], ms=4, capsize=2, elinewidth=0.6,
                        lw=0, label=LABELS[m])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel(f"{metric} [%]")
    axes[0].set_ylabel("Annual Loss [€/MW/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    boxed_legend(fig, handles, labels, 2, fontsize=5.5)
    save(fig, OUT_DIR / "fig_loss_vs_metrics")


def fig_margin_sweep(sweep):
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    for ax, k in zip(axes, PENALTIES):
        res = paired(sweep, k, "true_m0")
        res["model"] = [c.rsplit("_m", 1)[0] for c in res.index]
        res["margin"] = [float(c.rsplit("_m", 1)[1]) for c in res.index]
        #Lorenzo Giannuzzo: markers shifted sideways by a fraction of a point so that the error bars of the
        # five configurations, which share the same margins, do not hide each other
        for j, m in enumerate(SWEEP_MODELS):
            sub = res[res["model"] == m].sort_values("margin")
            shift = 0.12 * (j - (len(SWEEP_MODELS) - 1) / 2)
            ax.errorbar(sub["margin"] + shift, sub["change"], yerr=sub["se"], color=COLORS[m], marker=MARKERS[m],
                        ms=3.5, lw=1.0, capsize=1.5, elinewidth=0.5, label=LABELS[m])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"Penalty Factor {k:g}", fontsize=8)
        ax.set_xlabel("SOC Margin [%]")
    axes[0].set_ylabel("Net Revenue Change [€/MW/yr]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    boxed_legend(fig, handles, labels, 5, fontsize=6.5)
    save(fig, OUT_DIR / "fig_margin_sweep")


def fig_margin_mechanism(sweep):
    d = sweep[sweep["model"] == "true"].copy()
    d["margin"] = d["config"].str.rsplit("_m", n=1).str[1].astype(float)
    units = d.groupby("margin")[["week", "rep"]].apply(lambda x: len(x.drop_duplicates()))
    tot = d.groupby("margin")[["energy", "activation", "penalty", "forfeited"]].sum()
    tot = tot.mul(365.0 / (7.0 * units), axis=0)
    base = tot.loc[0.0]
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    given_up = (base["energy"] + base["activation"]) - (tot["energy"] + tot["activation"])
    ax.plot(tot.index, given_up, color="#1f3b73", marker="o", ms=3.5, lw=1.2, label="Energy Revenue Given Up")
    for k, ls in zip(PENALTIES, ("--", ":")):
        cost = tot["penalty"] * k / M.PENALTY_K + tot["forfeited"]
        ax.plot(tot.index, cost.loc[0.0] - cost, color="#c0392b", marker="s", ms=3.5, lw=1.2, ls=ls,
                label=f"Penalty And Forfeit Avoided (Factor {k:g})")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("SOC Margin [%]")
    ax.set_ylabel("Annual Amount [€/MW/yr]")
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout()
    boxed_legend(fig, handles, labels, 1, fontsize=6.5)
    save(fig, OUT_DIR / "fig_margin_mechanism")


def main():
    library = load_library(paths.ERROR_DIR / "error_library.npz")
    fig_error_vs_soc(library)
    fig_loss_vs_metrics(library, pd.read_csv(STUDY_DIR / "days.csv"))
    sweep = pd.read_csv(SWEEP_DIR / "days.csv")
    fig_margin_sweep(sweep)
    fig_margin_mechanism(sweep)
    print(f"figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()