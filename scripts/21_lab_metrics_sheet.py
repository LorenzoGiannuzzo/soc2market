"""
Print the Metrics sheet that the original estimator pipeline wrote in each result workbook.

These are the laboratory metrics computed by the original code, to be compared with the ones
recomputed by scripts/20_lab_benchmark.py from the prediction sheets: if they agree, the
recomputation is correct; if they disagree, the definition (samples, warm-up, columns) differs.
Only the Metrics sheet is read, so the script takes seconds rather than minutes.
"""

import pandas as pd

from soc2market import paths

LAB_DIR = paths.REPO_ROOT / "data" / "private" / "soc_regression"
OUT_DIR = paths.REPO_ROOT / "results" / "lab_benchmark"
BOOKS = ("ANN", "CNN", "LSTM", "CatBoost", "TFT", "Ensemble")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    frames = []
    for book in BOOKS:
        sheet = pd.read_excel(LAB_DIR / f"{book}.xlsx", sheet_name="Metrics")
        print(f"\n=== {book}.xlsx, sheet Metrics ({sheet.shape[0]} rows x {sheet.shape[1]} columns)")
        print(sheet.to_string())
        frames.append(sheet.assign(workbook=book))
    pd.concat(frames, ignore_index=True).to_csv(OUT_DIR / "original_metrics_sheets.csv", index=False)
    print(f"\nall sheets written to {OUT_DIR / 'original_metrics_sheets.csv'}")


if __name__ == "__main__":
    main()
