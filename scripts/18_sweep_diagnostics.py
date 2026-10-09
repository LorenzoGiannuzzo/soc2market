"""
Diagnostics of the margin sweep: decomposition of the annualised revenue of each configuration
into its components (planned profit of the MILP, capacity, forfeited capacity, activation energy,
energy at the PUN, penalty, degradation) and the operating indicators behind them, to explain
where the change of net revenue with the margin comes from.
"""

import pandas as pd

from soc2market import paths

SWEEP_DIR = paths.RUN_DIR / "margin_sweep"
MODELS = ("true", "EKF", "CNN")
MONEY = ["planned_eur", "net", "capacity", "forfeited", "activation", "energy", "penalty", "degradation"]
COUNTS = ["reserve_awarded_h", "reserve_failed_h", "miss_mwh", "reserve_miss_mwh", "refused_s"]


def main():
    days = pd.read_csv(SWEEP_DIR / "days.csv")
    days["margin"] = days["config"].str.rsplit("_m", n=1).str[1].astype(float)
    n_units = days.groupby("config")[["week", "rep"]].apply(lambda d: len(d.drop_duplicates())).rename("units")
    per = days.groupby(["model", "margin", "config"])[MONEY + COUNTS].sum().reset_index()
    per = per.merge(n_units, left_on="config", right_index=True)
    #Lorenzo Giannuzzo: every unit is a block of seven days, so annualise with 365 / (7 * units)
    scale = 365.0 / (7.0 * per["units"])
    for c in MONEY + COUNTS:
        per[c] = per[c] * scale
    per["failed_pct"] = 100 * per["reserve_failed_h"] / per["reserve_awarded_h"].clip(lower=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    for m in MODELS:
        sub = per[per["model"] == m].set_index("margin").sort_index()
        print(f"\n{m}: annualised components per margin [EUR/MW/yr], awarded and failed reserve hours per year, "
              f"undelivered energy [MWh/yr], refused seconds per year")
        print(sub[MONEY].round(0).to_string())
        print(sub[["reserve_awarded_h", "reserve_failed_h", "failed_pct", "miss_mwh", "reserve_miss_mwh",
                   "refused_s"]].round(1).to_string())
    per.to_csv(SWEEP_DIR / "components.csv", index=False)

    #Lorenzo Giannuzzo: is the step between consecutive margins larger than its sampling error? Paired on
    # block and replicate, for the net revenue and for the gross capacity revenue (capacity + forfeited)
    days["gross_capacity"] = days["capacity"] + days["forfeited"]
    unit = days.groupby(["model", "margin", "week", "rep"])[["net", "gross_capacity", "planned_eur"]].sum() * 365.0 / 7.0
    rows = []
    for m in MODELS:
        margins = sorted(days.loc[days["model"] == m, "margin"].unique())
        for a_m, b_m in zip(margins[:-1], margins[1:]):
            d = unit.loc[(m, b_m)] - unit.loc[(m, a_m)]
            n = len(d)
            rows.append({"model": m, "step": f"{a_m:g} to {b_m:g}",
                         **{f"{c}_change": d[c].mean() for c in d.columns},
                         **{f"{c}_se": d[c].std() / n ** 0.5 for c in d.columns}})
    steps = pd.DataFrame(rows)
    steps.to_csv(SWEEP_DIR / "margin_steps.csv", index=False)
    print("\nchange between consecutive margins, paired on block and replicate [EUR/MW/yr], with standard error:")
    print(steps.round(0).to_string(index=False))


if __name__ == "__main__":
    main()