"""
Post-processing of the market study: penalty and imbalance sensitivity, and which property of the
estimation error explains its market cost.

The operation of the unit does not depend on the penalty factor (the expected penalty term of the
MILP is zero at its optimum) nor on the price of the base energy that could not be delivered, so
both are varied here on the settled days of 15_market_study.py without simulating again:
    penalty       scaled from PENALTY_K of the run to each factor in PENALTY_FACTORS
    imbalance     undelivered base energy (refused energy outside the reserve products) charged at
                  each spread in IMBALANCE_SPREADS [EUR/MWh] on top of the lost day-ahead revenue
For every scenario the loss of each configuration against the oracle is paired on block and
replicate. For the naive strategy, the losses of the seven estimators are rank-correlated
(Spearman) with error metrics of the closed-loop library: RMSE, MAE, 95th percentile and maximum
of the absolute error, bias, and the tails in the dangerous direction next to the edges
(overestimation with true SOC 10 to 20%, underestimation with true SOC 80 to 90%).
"""

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.errormodel.generator import load_library
from soc2market.simulation import market as M

SCOPE = "year"
STUDY_DIR = paths.RUN_DIR / ("market_study_year" if SCOPE == "year" else "market_study")
PENALTY_FACTORS = (1.5, 3.0, 5.0, 10.0)
IMBALANCE_SPREADS = (0.0, 50.0, 100.0)
MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")


def scenario_net(days, k, spread):
    base_miss = (days["miss_mwh"] - days["reserve_miss_mwh"]).clip(lower=0.0)
    return days["net"] - days["penalty"] * (k / M.PENALTY_K - 1.0) - spread * base_miss


def paired_loss(days, net):
    """Mean and standard error of the annualised loss against the oracle, paired on block and replicate."""
    d = days.assign(x=net).groupby(["config", "week", "rep"]).agg(x=("x", "sum"), n=("x", "size")).reset_index()
    d["x"] *= 365.0 / d["n"]
    o = d[d["config"] == "oracle"][["week", "rep", "x"]].rename(columns={"x": "o"})
    d = d.merge(o, on=["week", "rep"])
    d["loss"] = d["o"] - d["x"]
    g = d.groupby("config")["loss"]
    return pd.DataFrame({"loss": g.mean(), "se": g.std() / np.sqrt(g.size())})


def error_metrics(library):
    soc = np.asarray(library["soc_true"], float)
    low, high = (soc >= 10) & (soc < 20), (soc >= 80) & (soc < 90)
    rows = {}
    for m in MODELS:
        e = np.asarray(library[f"err_{m}"], float)
        rows[m] = {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
                   "P95 Absolute": np.quantile(np.abs(e), 0.95), "Max Absolute": np.abs(e).max(),
                   "Bias": e.mean(), "Bottom Overestimation P95": np.quantile(e[low], 0.95),
                   "Top Underestimation P95": np.quantile(-e[high], 0.95)}
        rows[m]["Edge Tail P95"] = max(rows[m]["Bottom Overestimation P95"], rows[m]["Top Underestimation P95"])
    return pd.DataFrame(rows).T


def main():
    days = pd.read_csv(STUDY_DIR / "days.csv")
    print(f"{len(days)} days from {STUDY_DIR}")
    tables = {}
    for k in PENALTY_FACTORS:
        for s in IMBALANCE_SPREADS:
            tables[(k, s)] = paired_loss(days, scenario_net(days, k, s))
    loss = pd.concat({f"k={k:g}, spread={s:g}": t["loss"] for (k, s), t in tables.items()}, axis=1)
    se = pd.concat({f"k={k:g}, spread={s:g}": t["se"] for (k, s), t in tables.items()}, axis=1)
    loss.to_csv(STUDY_DIR / "sensitivity_loss.csv")
    se.to_csv(STUDY_DIR / "sensitivity_loss_se.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    focus = [f"k={k:g}, spread=0" for k in PENALTY_FACTORS] + ["k=1.5, spread=100", "k=10, spread=100"]
    print("\nloss against the oracle [EUR/MW/yr] (standard error in the second table):")
    print(loss[focus].sort_values(focus[0]).round(0).to_string())
    print(se[focus].round(0).to_string())

    #Lorenzo Giannuzzo: best strategy of every estimator in every scenario
    best = []
    for col in focus:
        for m in MODELS + ("EKF-replay",):
            sub = loss[col][[c for c in loss.index if c.startswith(m + "_")]]
            best.append({"scenario": col, "estimator": m, "best": sub.idxmin(), "loss": round(sub.min())})
    print("\nbest strategy per estimator:")
    print(pd.DataFrame(best).pivot(index="estimator", columns="scenario", values="best")[focus].to_string())

    metrics = error_metrics(load_library(paths.ERROR_DIR / "error_library.npz"))
    rho = {}
    for col in focus:
        naive = pd.Series({m: loss.loc[f"{m}_naive", col] for m in MODELS})
        rho[col] = {name: naive.corr(metrics[name], method="spearman") for name in metrics.columns}
    rho = pd.DataFrame(rho)
    metrics.to_csv(STUDY_DIR / "error_metrics.csv")
    rho.to_csv(STUDY_DIR / "rank_correlation.csv")
    print("\nerror metrics of the closed-loop library [percentage points]:")
    print(metrics.round(2).to_string())
    print("\nSpearman correlation between naive loss and error metric (7 estimators):")
    print(rho.round(2).to_string())


if __name__ == "__main__":
    main()
