"""
Lorenzo Giannuzzo — SOC Prediction Comparison Plots
====================================================
Reads ANN.xlsx, CNN.xlsx, LSTM.xlsx, CatBoost.xlsx, TFT.xlsx from
./results/results_sheet/, iterates over every scenario sheet, and produces one
figure per scenario showing:
    - True SOC          (SOC_true column)
    - KF + <Model>      (Predicted_SOC column from each model file)

Output: ./soc_plots/<scenario_name>.png + .pdf  (500 dpi for PNG)

Usage:
    python plot_soc_comparisons.py
    python plot_soc_comparisons.py --fmt png
    python plot_soc_comparisons.py --scenarios S2_DAY0to4 S3_DAY4to7
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results") / "results_sheet"
OUTPUT_DIR  = Path("soc_plots")

MODELS = ["ANN", "CNN", "LSTM", "CatBoost", "TFT"]

# Colors matching the overall_comparison figure
MODEL_COLORS = {
    "True SOC":  "#000000",   # black
    "ANN":       "#1f77b4",   # blue
    "CNN":       "#17becf",   # cyan/light blue
    "LSTM":      "#ff7f0e",   # orange
    "CatBoost":  "#FFD700",   # yellow
    "TFT":       "#9467bd",   # purple
}

LINE_STYLES = {
    "True SOC": dict(lw=1.5
                     , ls="-",  zorder=5),
    "ANN":      dict(lw=1.2, ls="-",  zorder=3),
    "CNN":      dict(lw=1.2, ls="-",  zorder=3),
    "LSTM":     dict(lw=1.2, ls="-",  zorder=3),
    "CatBoost": dict(lw=1.2, ls="-",  zorder=3),
    "TFT":      dict(lw=1.2, ls="-",  zorder=3),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_sheets(model_name: str) -> dict[str, pd.DataFrame]:
    path = RESULTS_DIR / f"{model_name}.xlsx"
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping {model_name}")
        return {}
    xl = pd.ExcelFile(path)
    sheets = {}
    for sheet in xl.sheet_names:
        if sheet.strip().lower() == "metrics":
            continue
        try:
            df = xl.parse(sheet)
            df.columns = [c.strip() for c in df.columns]
            sheets[sheet] = df
        except Exception as e:
            print(f"  [WARN] Could not parse sheet '{sheet}' in {path}: {e}")
    return sheets


def get_col(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return df[lower_map[cand.lower()]]
    return None


def plot_scenario(scenario: str, data: dict[str, pd.DataFrame], fmt: str = "both"):
    fig, ax = plt.subplots(figsize=(12, 5))

    ref_df = next(iter(data.values()))
    x      = get_col(ref_df, "Step")
    y_true = get_col(ref_df, "SOC_true", "SOC_True", "soc_true")

    if x is None or y_true is None:
        print(f"  [WARN] Missing Step/SOC_true in '{scenario}' — skipping")
        plt.close(fig)
        return

    # True SOC
    ax.plot(x, y_true,
            label="True SOC",
            color=MODEL_COLORS["True SOC"],
            **LINE_STYLES["True SOC"])

    # Per-model predictions (no KF line)
    for model_name, df in data.items():
        y_pred = get_col(df, "Predicted_SOC", "predicted_soc", "PredictedSOC")
        if y_pred is None:
            print(f"  [WARN] No Predicted_SOC for {model_name} in '{scenario}'")
            continue
        x_m = get_col(df, "Step")
        ax.plot(x_m, y_pred,
                label=f"KF + {model_name}",
                color=MODEL_COLORS[model_name],
                **LINE_STYLES[model_name])

    # --- Formatting ---
    ax.set_title(
        f"SOC Prediction Comparison — {scenario}",
        fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xlabel("Step [s]", fontsize=11)
    ax.set_ylabel("SOC [%]", fontsize=11)
    ax.set_ylim(-2, 102)

    # Y grid every 20 % only — no minor ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))

    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.legend(
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
        ncol=1,
    )

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = scenario.replace("/", "_").replace(" ", "_")

    if fmt in ("png", "both"):
        fig.savefig(OUTPUT_DIR / f"{safe}.png", dpi=500, bbox_inches="tight")
        print(f"  Saved → {OUTPUT_DIR / f'{safe}.png'}")
    if fmt in ("pdf", "both"):
        fig.savefig(OUTPUT_DIR / f"{safe}.pdf", bbox_inches="tight")
        print(f"  Saved → {OUTPUT_DIR / f'{safe}.pdf'}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmt", choices=["png", "pdf", "both"], default="both")
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    print("Loading model Excel files...")
    all_data: dict[str, dict[str, pd.DataFrame]] = {}
    for model in MODELS:
        print(f"  {model}.xlsx ...", end=" ")
        sheets = load_model_sheets(model)
        print(f"{len(sheets)} sheets")
        all_data[model] = sheets

    all_scenarios = sorted({s for sheets in all_data.values() for s in sheets})
    if args.scenarios:
        all_scenarios = [s for s in all_scenarios if s in args.scenarios]

    print(f"\nGenerating {len(all_scenarios)} plot(s) → ./{OUTPUT_DIR}/")
    for scenario in all_scenarios:
        print(f"\n[{scenario}]")
        scenario_data = {m: all_data[m][scenario] for m in MODELS if scenario in all_data[m]}
        if not scenario_data:
            print(f"  [SKIP] No data")
            continue
        plot_scenario(scenario, scenario_data, fmt=args.fmt)

    print("\nDone.")


if __name__ == "__main__":
    main()