"""
Market study: value of SOC-estimation error for a 1 MW / 2 MWh unit on the 2023 Italian markets.

Runs every configuration of soc2market.simulation.market (oracle, oracle with the fixed margin,
and for each estimator: naive, fixed margin, quantile margin, edge-band margin; the EKF also with
its closed-loop error replayed hour by hour) in parallel over the cores of the machine, on one of
two scopes:
    SCOPE = "year"   the whole of 2023 as 52 blocks of seven days, error of each block drawn from
                     the library week of its season (main result)
    SCOPE = "weeks"  the four reference weeks of the error library
Every block starts from SOC 50% and carries the SOC from one day to the next. Writes one row per
day to days.csv and the annualised summary to summary.csv, in the folder of the MSD mode set in
paths.py. Losses against the oracle are paired on the same block and replicate (common random
numbers); their standard error is computed over those pairs.

Set QUICK = True for a short sequential check (one block, one replicate, five configurations).
"""

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from soc2market import paths
from soc2market.errormodel.generator import load_library
from soc2market.simulation import market as M

QUICK = False
SCOPE = "year"
REPLICATES = {"year": 2, "weeks": 16}[SCOPE]
MODELS = ("EKF", "ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")
LIBRARY = paths.ERROR_DIR / "error_library.npz"
OUT_DIR = paths.RUN_DIR / ("market_study_quick" if QUICK else
                           "market_study_year" if SCOPE == "year" else "market_study")
QUICK_CONFIGS = ("oracle", "oracle_fixed", "EKF_naive", "CatBoost_naive", "CatBoost_band")
MONEY = ["net", "capacity", "forfeited", "activation", "energy", "penalty", "degradation"]
COUNTS = ["reserve_awarded_h", "reserve_failed_h", "reserve_miss_mwh", "miss_mwh", "refused_s"]
KEYS = ["config", "strategy", "model", "error_mode", "window_lo", "window_hi"]


def unit_totals(days):
    """Totals per configuration and unit (block, replicate), annualised per MW of the unit."""
    per = days.groupby(KEYS + ["week", "rep"])[MONEY + COUNTS + ["day"]].agg(
        {**{c: "sum" for c in MONEY + COUNTS}, "day": "count"}).reset_index()
    scale = 365.0 / per["day"]
    for c in MONEY:
        per[c + "_eur_mw_yr"] = per[c] * scale
    return per


def summarise(days):
    """Mean over units of the annualised totals, paired loss against the oracle with its standard error."""
    per = unit_totals(days)
    agg = per.groupby(KEYS).agg(
        net_eur_mw_yr=("net_eur_mw_yr", "mean"),
        capacity_eur_mw_yr=("capacity_eur_mw_yr", "mean"), forfeited_eur_mw_yr=("forfeited_eur_mw_yr", "mean"),
        activation_eur_mw_yr=("activation_eur_mw_yr", "mean"), energy_eur_mw_yr=("energy_eur_mw_yr", "mean"),
        penalty_eur_mw_yr=("penalty_eur_mw_yr", "mean"), degradation_eur_mw_yr=("degradation_eur_mw_yr", "mean"),
        reserve_awarded_h=("reserve_awarded_h", "sum"), reserve_failed_h=("reserve_failed_h", "sum"),
        reserve_miss_mwh=("reserve_miss_mwh", "mean"), miss_mwh=("miss_mwh", "mean"),
        refused_s=("refused_s", "mean"), units=("net", "size")).reset_index()
    agg["failed_share_pct"] = 100 * agg["reserve_failed_h"] / agg["reserve_awarded_h"].clip(lower=1)
    #Lorenzo Giannuzzo: paired difference against the oracle on the same block and replicate
    o = per.loc[per["config"] == "oracle", ["week", "rep", "net_eur_mw_yr"]].rename(columns={"net_eur_mw_yr": "o"})
    paired = per.merge(o, on=["week", "rep"])
    paired["loss"] = paired["o"] - paired["net_eur_mw_yr"]
    stats = paired.groupby("config")["loss"].agg(loss_vs_oracle_eur_mw_yr="mean", loss_sd="std", n="size")
    stats["loss_se"] = stats["loss_sd"] / np.sqrt(stats["n"])
    return agg.merge(stats[["loss_vs_oracle_eur_mw_yr", "loss_se"]].reset_index(), on="config")


def main():
    library = load_library(LIBRARY)
    configs = M.build_configs(library, MODELS)
    blocks = M.year_blocks() if SCOPE == "year" else M.week_blocks()
    reps = range(REPLICATES)
    if QUICK:
        configs = [c for c in configs if c.name in QUICK_CONFIGS]
        blocks, reps = blocks[:1], range(1)
    print(f"MSD mode: {paths.MSD_MODE}, scope: {SCOPE}, results in {OUT_DIR}")
    print(f"{len(configs)} configurations, {len(blocks)} blocks of seven days, {len(reps)} replicates")
    for c in configs:
        print(f"  {c.name:24s} SOC window {100 * c.window[0]:5.1f} to {100 * c.window[1]:5.1f} %")
    #Lorenzo Giannuzzo: replicate-major order, so that the partial days.csv written during the run
    # already covers every configuration and block of the first replicate
    tasks = [(c, b, r) for r in reps for b in blocks for c in configs]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0, rows = time.time(), []
    if QUICK:
        M.init_worker(LIBRARY)
        for i, task in enumerate(tasks, 1):
            rows += M.run_week(task)
            print(f"  {i}/{len(tasks)} {task[0].name} done, {time.time() - t0:.0f} s")
    else:
        workers = max(1, (os.cpu_count() or 2) - 1)
        print(f"running {len(tasks)} blocks on {workers} processes")
        step = max(10, len(tasks) // 50)
        with mp.get_context("spawn").Pool(workers, initializer=M.init_worker, initargs=(LIBRARY,)) as pool:
            for i, block_rows in enumerate(pool.imap_unordered(M.run_week, tasks), 1):
                rows += block_rows
                if i % step == 0 or i == len(tasks):
                    left = (time.time() - t0) / i * (len(tasks) - i)
                    print(f"  {i}/{len(tasks)} blocks, {(time.time() - t0) / 60:.1f} min, about {left / 60:.0f} min left")
                    pd.DataFrame(rows).to_csv(OUT_DIR / "days.csv", index=False)

    days = pd.DataFrame(rows)
    days.to_csv(OUT_DIR / "days.csv", index=False)
    summary = summarise(days)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    with pd.option_context("display.float_format", "{:.1f}".format, "display.width", 250, "display.max_columns", None):
        cols = ["config", "window_lo", "window_hi", "net_eur_mw_yr", "loss_vs_oracle_eur_mw_yr", "loss_se",
                "penalty_eur_mw_yr", "forfeited_eur_mw_yr", "failed_share_pct", "reserve_miss_mwh", "refused_s"]
        #Lorenzo Giannuzzo: windows printed in percent (the csv keeps fractions)
        shown = summary.assign(window_lo=100 * summary["window_lo"], window_hi=100 * summary["window_hi"])
        print(shown[cols].sort_values("net_eur_mw_yr", ascending=False).to_string(index=False))
    failed = int(days["solver_failed"].sum())
    print(f"\n{len(days)} simulated days ({failed} with a failed MILP, run idle) in "
          f"{(time.time() - t0) / 60:.1f} min, written to {OUT_DIR}")


if __name__ == "__main__":
    main()