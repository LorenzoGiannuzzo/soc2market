"""
Margin sweep: net revenue of the unit as a function of a symmetric SOC margin, for the true SOC
and for a subset of the pre-trained estimators, over the whole of 2023 (52 blocks of seven days,
chronological MSD, REPLICATES draws), followed by the penalty sensitivity of 16_market_sensitivity.py.

For every penalty factor the script reports the margin that maximises the mean net revenue of each
estimator, and the value of perfect SOC knowledge once both the oracle and the estimator use their
own best margin. The figure shows the change of net revenue against the oracle without margin,
with one standard error of the paired difference, for every penalty factor.

Set ANALYSE_ONLY = True to redo the tables and the figure from an existing days.csv.
"""

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.figures.style import plt, save
from soc2market.simulation import market as M

ANALYSE_ONLY = False
REPLICATES = 2
MODELS = ("EKF", "TFT", "CatBoost", "CNN")
LABELS = {"true": "Perfect Knowledge", "EKF": "Extended Kalman Filter",
          "TFT": "EKF + Temporal Fusion Transformer", "CatBoost": "EKF + CatBoost",
          "CNN": "EKF + Convolutional Neural Network"}
COLORS = {"true": "black", "EKF": "#1f3b73", "TFT": "#2e8b57", "CatBoost": "#c0392b", "CNN": "#d68910"}
PENALTY_FACTORS = (1.5, 3.0, 5.0, 10.0)
LIBRARY = paths.ERROR_DIR / "error_library.npz"
OUT_DIR = paths.RUN_DIR / "margin_sweep"
REFERENCE = "true_m0"


def simulate():
    from soc2market.errormodel.generator import load_library  # noqa: F401  (library is loaded by the workers)
    configs = M.build_sweep_configs(MODELS)
    blocks = M.year_blocks()
    tasks = [(c, b, r) for r in range(REPLICATES) for b in blocks for c in configs]
    print(f"MSD mode: {paths.MSD_MODE}, {len(configs)} configurations, margins {M.SWEEP_MARGINS_PTS} points")
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"running {len(tasks)} blocks on {workers} processes, results in {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0, rows, step = time.time(), [], max(10, len(tasks) // 50)
    with mp.get_context("spawn").Pool(workers, initializer=M.init_worker, initargs=(LIBRARY,)) as pool:
        for i, block_rows in enumerate(pool.imap_unordered(M.run_week, tasks), 1):
            rows += block_rows
            if i % step == 0 or i == len(tasks):
                left = (time.time() - t0) / i * (len(tasks) - i)
                print(f"  {i}/{len(tasks)} blocks, {(time.time() - t0) / 60:.1f} min, about {left / 60:.0f} min left")
                pd.DataFrame(rows).to_csv(OUT_DIR / "days.csv", index=False)
    days = pd.DataFrame(rows)
    days.to_csv(OUT_DIR / "days.csv", index=False)
    print(f"{len(days)} simulated days ({int(days['solver_failed'].sum())} with a failed MILP) "
          f"in {(time.time() - t0) / 60:.1f} min")
    return days


def paired_change(days, k):
    """Annualised net change against REFERENCE on the same block and replicate: mean and standard error."""
    net = days["net"] - days["penalty"] * (k / M.PENALTY_K - 1.0)
    d = days.assign(x=net).groupby(["config", "week", "rep"]).agg(x=("x", "sum"), n=("x", "size")).reset_index()
    d["x"] *= 365.0 / d["n"]
    ref = d[d["config"] == REFERENCE][["week", "rep", "x"]].rename(columns={"x": "r"})
    d = d.merge(ref, on=["week", "rep"])
    d["change"] = d["x"] - d["r"]
    g = d.groupby("config")["change"]
    out = pd.DataFrame({"change": g.mean(), "se": g.std() / np.sqrt(g.size())})
    out["model"] = [c.rsplit("_m", 1)[0] for c in out.index]
    out["margin"] = [float(c.rsplit("_m", 1)[1]) for c in out.index]
    return out


def analyse(days):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    results, best_rows = {}, []
    for k in PENALTY_FACTORS:
        res = paired_change(days, k)
        results[k] = res
        table = res.pivot(index="model", columns="margin", values="change").reindex(("true",) + MODELS)
        print(f"\npenalty factor {k:g}: net change against the oracle without margin [EUR/MW/yr]")
        print(table.round(0).to_string())
        best_true = res[res["model"] == "true"]["change"].max()
        for m in ("true",) + MODELS:
            sub = res[res["model"] == m]
            i = sub["change"].idxmax()
            best_rows.append({"penalty": k, "model": m, "best_margin": sub.loc[i, "margin"],
                              "best_change": sub.loc[i, "change"], "se": sub.loc[i, "se"],
                              "value_of_perfect_soc": best_true - sub.loc[i, "change"]})
    best = pd.DataFrame(best_rows)
    best.to_csv(OUT_DIR / "best_margin.csv", index=False)
    for k, res in results.items():
        res.to_csv(OUT_DIR / f"change_k{k:g}.csv")
    print("\nbest margin per estimator and penalty factor, and value of perfect SOC knowledge "
          "(oracle at its best margin minus estimator at its best margin) [EUR/MW/yr]:")
    print(best.pivot(index="model", columns="penalty", values="best_margin").reindex(("true",) + MODELS).to_string())
    print(best.pivot(index="model", columns="penalty", values="value_of_perfect_soc")
          .reindex(("true",) + MODELS).round(0).to_string())

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0), sharex=True)
    for ax, k in zip(axes.flat, PENALTY_FACTORS):
        res = results[k]
        for m in ("true",) + MODELS:
            sub = res[res["model"] == m].sort_values("margin")
            ax.errorbar(sub["margin"], sub["change"], yerr=sub["se"], color=COLORS[m], marker="o", ms=3,
                        lw=1.0, capsize=2, elinewidth=0.6, label=LABELS[m])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"Penalty Factor {k:g}", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("SOC Margin [%]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Net Revenue Change [€/MW/yr]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, fancybox=False,
                     edgecolor="black", bbox_to_anchor=(0.5, 1.08), fontsize=7)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    save(fig, OUT_DIR / "margin_sweep")
    print(f"\ntables and figure written to {OUT_DIR}")


def main():
    days = pd.read_csv(OUT_DIR / "days.csv") if ANALYSE_ONLY else simulate()
    analyse(days)


if __name__ == "__main__":
    main()
