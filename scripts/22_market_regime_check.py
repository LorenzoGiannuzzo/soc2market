"""
Assign the market reference weeks to the operating regimes of the laboratory benchmark.

The twelve laboratory test scenarios of the estimators were grouped by a hierarchical Ward
clustering on z-scored operating features into three regimes: Dynamic Wide-Range SOC (DWR-SOC),
High-SOC Plateau (HSP) and Quasi-Static Narrow (QSN). The paper compares laboratory and market
accuracy on the laboratory regime that the market operation actually resembles, so this script
computes the same features for the simulated market operation of the cell and assigns it to the
nearest regime centroid, in the z-space of the fifteen laboratory scenarios. The assignment is made
for every whole reference week and for every 48 h window (the length of the shortest laboratory
test), with both temperature features, so that the conclusion does not hinge on one choice.

Laboratory features: Table S.4 of the Supplementary Materials of the estimator benchmark.
"""

import numpy as np
import pandas as pd

from soc2market import paths

DUTY_DIR = paths.DUTY_DIR
OUT_DIR = paths.RUN_DIR / "regime_check"
WEEKS = ("winter", "spring", "summer", "autumn")
WINDOW_H = 48

#Lorenzo Giannuzzo: Table S.4 (soc_min, soc_max, dsoc in %, v_min, v_max in V, i_peak, i_std in A,
# t_mean, t_max in degC) and the cluster of each scenario (Table S.5 and Section 2.1 of the paper)
LAB = pd.DataFrame([
    ("S1_DAY0to4", "DWR-SOC", 20.00, 76.90, 56.90, 3.14, 3.40, 172.00, 16.00, 26.70, 29.30),
    ("S2_DAY4to7", "HSP", 62.70, 80.00, 17.30, 3.29, 3.92, 56.70, 8.36, 27.00, 27.40),
    ("S3_DAY0to4", "DWR-SOC", 20.00, 80.00, 60.00, 3.18, 3.41, 110.20, 20.10, 26.50, 27.70),
    ("S1_DAY4to7", "HSP", 45.40, 77.40, 32.00, 3.26, 3.38, 26.30, 9.53, 26.80, 27.30),
    ("S2_DAY0to4", "DWR-SOC", 20.00, 80.00, 60.00, 3.19, 3.45, 168.40, 16.09, 26.80, 28.50),
    ("S2_DAY7to14", "DWR-SOC", 20.00, 80.00, 60.00, 3.12, 3.42, 225.60, 25.78, 26.70, 30.30),
    ("S3_DAY4to7", "HSP", 35.70, 80.00, 44.30, 3.26, 3.40, 54.10, 19.44, 26.70, 27.30),
    ("S3_DAY7to8", "DWR-SOC", 41.30, 80.00, 38.70, 3.22, 3.46, 252.80, 28.49, 26.80, 35.10),
    ("S4_DAY0to4", "DWR-SOC", 20.00, 80.00, 60.00, 3.22, 3.37, 51.20, 18.19, 26.40, 27.00),
    ("S4_DAY1to8", "QSN", 20.10, 39.50, 19.40, 3.22, 3.32, 10.50, 2.70, 27.00, 28.10),
    ("S4_DAY4to7", "DWR-SOC", 20.00, 80.00, 60.00, 3.22, 3.37, 83.60, 19.50, 26.50, 27.30),
    ("S5_DAY1to2", "QSN", 20.10, 29.70, 9.60, 3.22, 3.30, 5.70, 2.58, 27.00, 28.10),
    ("S5_DAY3to4", "QSN", 20.90, 33.70, 12.70, 3.23, 3.31, 7.10, 2.10, 27.00, 27.70),
    ("S5_DAY5to6", "DWR-SOC", 20.60, 80.40, 59.80, 3.09, 3.45, 280.10, 39.88, 27.10, 36.60),
    ("S6_DAY5to6", "QSN", 23.80, 39.50, 15.70, 3.26, 3.32, 10.50, 3.19, 27.00, 27.70),
], columns=["scenario", "cluster", "soc_min", "soc_max", "dsoc", "v_min", "v_max", "i_peak", "i_std",
            "t_mean", "t_max"]).set_index("scenario")
BASE = ["soc_min", "soc_max", "dsoc", "v_min", "v_max", "i_peak", "i_std"]


def features(soc, v, i, t):
    return dict(soc_min=soc.min(), soc_max=soc.max(), dsoc=soc.max() - soc.min(), v_min=v.min(), v_max=v.max(),
                i_peak=np.abs(i).max(), i_std=i.std(), t_mean=t.mean(), t_max=t.max())


def load_week(week):
    parts = {k: [] for k in ("soc", "v", "i", "t")}
    for f in sorted((DUTY_DIR / week).glob("day_*.npz")):
        with np.load(f) as z:
            parts["soc"].append(z["soc_true_pct"]); parts["v"].append(z["voltage_v"])
            parts["i"].append(z["current_a"]); parts["t"].append(z["temp_c"])
    return {k: np.concatenate(x) for k, x in parts.items()}


def assign(table, feats):
    """Nearest centroid in the z-space of the laboratory scenarios; returns cluster and distances."""
    mu, sd = LAB[feats].mean(), LAB[feats].std(ddof=0)
    z_lab = (LAB[feats] - mu) / sd
    centroids = z_lab.groupby(LAB["cluster"]).mean()
    z = (table[feats] - mu) / sd
    dist = pd.DataFrame({c: np.sqrt(((z - centroids.loc[c]) ** 2).sum(axis=1)) for c in centroids.index})
    return dist.idxmin(axis=1), dist


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weekly, windows = [], []
    for week in WEEKS:
        d = load_week(week)
        weekly.append(dict(week=week, **features(**d)))
        n = WINDOW_H * 3600
        for k in range(d["soc"].size // n):
            sl = slice(k * n, (k + 1) * n)
            windows.append(dict(week=week, window=k, **features(**{x: d[x][sl] for x in d})))
    weekly = pd.DataFrame(weekly).set_index("week")
    windows = pd.DataFrame(windows).set_index(["week", "window"])

    #Lorenzo Giannuzzo: sanity check, the assignment must reproduce the clusters of the laboratory scenarios
    for tname in ("t_max", "t_mean"):
        lab_label, _ = assign(LAB, BASE + [tname])
        print(f"laboratory scenarios reassigned correctly with {tname}: {(lab_label == LAB['cluster']).sum()} of {len(LAB)}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\nfeatures of the market reference weeks (cell level):")
    print(weekly.round(2).to_string())
    for tname in ("t_max", "t_mean"):
        label, dist = assign(weekly, BASE + [tname])
        weekly[f"cluster_{tname}"] = label
        print(f"\nweekly assignment with {tname} (distance to each centroid in z units):")
        print(pd.concat([dist.round(2), label.rename("assigned")], axis=1).to_string())
        wlabel, _ = assign(windows, BASE + [tname])
        windows[f"cluster_{tname}"] = wlabel
        print(f"\n{WINDOW_H} h windows assigned to each cluster with {tname}:")
        print(wlabel.value_counts().to_string())
    print("\nthreshold description of DWR-SOC (SOC excursion >= 38 points and current std > 16 A):")
    print(((weekly["dsoc"] >= 38) & (weekly["i_std"] > 16)).rename("weekly_DWR_by_thresholds").to_string())
    weekly.to_csv(OUT_DIR / "weekly_regimes.csv")
    windows.to_csv(OUT_DIR / "window_regimes.csv")
    print(f"\nresults written to {OUT_DIR}")


if __name__ == "__main__":
    main()
