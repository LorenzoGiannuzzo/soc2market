"""
Identify the plant parameters (OCV branches and two-RC tables) from the laboratory data of the cell,
write them to data/private/plant/ and draw a diagnostic figure in results/plant/.
"""

import numpy as np
import pandas as pd

from soc2market.figures.style import legend_top, plt, save
from soc2market.plant.cell import PLANT_DATA_DIR, REPO_ROOT
from soc2market.plant.identify import run_identification, run_slow_branch_identification

FIG_DIR = REPO_ROOT / "results" / "plant"


def main():
    ocv, tables = run_identification()
    pd.set_option("display.width", 160)
    for name, table in tables.items():
        print(f"\n=== {name.upper()} relaxation fits ===")
        raw = pd.read_csv(PLANT_DATA_DIR / f"ecm_{name}_raw.csv")
        cols = ["soc", "r0", "r1", "tau1", "r2", "tau2", "rmse_mv", "n_points", "pulse_s", "current_a", "success", "at_bound"]
        print(raw[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        print(f"replaced by interpolation: {int(table['replaced'].sum())} of {len(table)}")

    half = 0.5 * (ocv["ocv_chg"] - ocv["ocv_dch"])
    print(f"\nOCV hysteresis half-width: median {1e3 * half.median():.1f} mV, "
          f"range {1e3 * half.min():.1f} to {1e3 * half.max():.1f} mV")
    print(f"tables written to {PLANT_DATA_DIR}")

    #Lorenzo Giannuzzo: slow (diffusion) branch, fitted on the full HPPC replays
    print("\n=== slow branch (fitted on the HPPC replays, SOC 5 to 95%) ===")
    slow = run_slow_branch_identification()
    print(slow.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    #Lorenzo Giannuzzo: diagnostic figure, OCV branches and RC parameters against SOC
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax = axes[0, 0]
    ax.plot(100 * ocv["soc"], ocv["ocv_chg"], label="Charge")
    ax.plot(100 * ocv["soc"], ocv["ocv_dch"], label="Discharge")
    ax.set_xlabel("State Of Charge [%]")
    ax.set_ylabel("Open Circuit Voltage [V]")
    legend_top(ax)
    for ax, key, label, scale in ((axes[0, 1], "r0", "Ohmic Resistance [mΩ]", 1e3),
                                  (axes[1, 0], "r1", "Resistances R1, R2 [mΩ]", 1e3),
                                  (axes[1, 1], "tau1", "Time Constants [s]", 1.0)):
        for name, marker in (("chg", "o"), ("dch", "s")):
            raw = pd.read_csv(PLANT_DATA_DIR / f"ecm_{name}_raw.csv")
            direction = "Charge" if name == "chg" else "Discharge"
            ax.plot(100 * raw["soc"], scale * raw[key], marker=marker, ms=3, lw=0.8, label=f"{direction} {key.upper()}")
            if key in ("r1", "tau1"):
                second = "r2" if key == "r1" else "tau2"
                ax.plot(100 * raw["soc"], scale * raw[second], marker=marker, ms=3, lw=0.8, ls="--",
                        label=f"{direction} {second.upper()}")
        if key == "tau1":
            ax.set_yscale("log")
        ax.set_xlabel("State Of Charge [%]")
        ax.set_ylabel(label)
        legend_top(ax, ncol=2)
    fig.tight_layout()
    save(fig, FIG_DIR / "identification")
    print(f"figure written to {FIG_DIR / 'identification.png'}")


if __name__ == "__main__":
    main()
