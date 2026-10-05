"""
Lorenzo Giannuzzo — Scenario Statistics Table Generator
========================================================
Reads scenario Excel files from:
  - train_data/   → training scenarios (GenerateTestData_*.xlsx, excluding HPPC + OCV)
  - data/         → test scenarios (excluding HPPC, OCV, GenerateTestData)

Outputs: scenario_statistics.xlsx

Usage:
    python generate_scenario_table.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── Directory config ──────────────────────────────────────────────────────────
TRAIN_DIR  = Path("train_data")
TEST_DIR   = Path("data")
OUTPUT     = Path("scenario_statistics.xlsx")

# Patterns to skip (case-insensitive)
SKIP_PATTERNS = ["hppc", "ocv", "cha_dis", "f.zip"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def should_skip(name: str) -> bool:
    n = name.lower()
    return any(p in n for p in SKIP_PATTERNS)


def clean_scenario_name(stem: str) -> str:
    """Remove 'GenerateTestData_' prefix if present."""
    return re.sub(r"(?i)^generatetestdata_", "", stem)


def load_df(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    sheet = next(s for s in xl.sheet_names if s.lower() != "metrics")
    raw = xl.parse(sheet, header=None)
    # row 0=Name, row 1=Mode/Inputs, row 2=column names, row 3=types, data from row 4
    col_names = raw.iloc[2].tolist()
    df = raw.iloc[4:].copy()
    df.columns = col_names
    df = df.reset_index(drop=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compute_stats(name: str, split: str, df: pd.DataFrame) -> dict:
    v   = df["Voltage"].dropna()
    i   = df["Current_inv"].dropna()
    soc = df["SOC_true"].dropna()

    temp_cols = [c for c in df.columns if str(c).lower().startswith("temp")]
    if temp_cols:
        temp = df[temp_cols].apply(pd.to_numeric, errors="coerce").stack().dropna()
    else:
        temp = pd.Series(dtype=float)

    return {
        "Scenario":      name,
        "Split":         split,
        "Duration [h]":  round(len(df) / 3600, 1),
        "SOC_init [%]":  round(float(soc.iloc[0]), 1),
        "SOC_min [%]":   round(float(soc.min()), 1),
        "SOC_max [%]":   round(float(soc.max()), 1),
        "SOC Range [%]": round(float(soc.max() - soc.min()), 1),
        "V_min [V]":     round(float(v.min()), 3),
        "V_max [V]":     round(float(v.max()), 3),
        "|I|_max [A]":   round(float(i.abs().max()), 1),
        "σ_I [A]":       round(float(i.std()), 2),
        "T_mean [°C]":   round(float(temp.mean()), 1) if len(temp) > 0 else "—",
        "T_max [°C]":    round(float(temp.max()), 1)  if len(temp) > 0 else "—",
        "Temp. Sensors": len(temp_cols),
    }


def collect_scenarios(directory: Path, split: str) -> list[dict]:
    results = []
    files = sorted(directory.glob("*.xlsx"))
    for f in files:
        if should_skip(f.name):
            continue
        # For train_data keep only GenerateTestData files
        if split == "Train" and "generatetestdata" not in f.name.lower():
            continue
        name = clean_scenario_name(f.stem)
        print(f"  [{split}] {name} ← {f.name}")
        try:
            df = load_df(f)
            results.append(compute_stats(name, split, df))
        except Exception as e:
            print(f"    [WARN] Could not process {f.name}: {e}")
    return results


# ── Build Excel ───────────────────────────────────────────────────────────────

def build_excel(rows: list[dict], output: Path):
    headers = list(rows[0].keys())

    # ── Styles ────────────────────────────────────────────────────────────────
    thin = Side(style="thin", color="555555")
    BRD  = Border(left=thin, right=thin, top=thin, bottom=thin)
    CTR  = Alignment(horizontal="center", vertical="center", wrap_text=False)
    CTR_W= Alignment(horizontal="center", vertical="center", wrap_text=True)
    LFT  = Alignment(horizontal="left",   vertical="center")

    FILL_HDR         = PatternFill("solid", fgColor="1F4E79")
    FILL_TRAIN_A     = PatternFill("solid", fgColor="1E4620")
    FILL_TRAIN_B     = PatternFill("solid", fgColor="2D6A2D")
    FILL_TEST_A      = PatternFill("solid", fgColor="1A1A2E")
    FILL_TEST_B      = PatternFill("solid", fgColor="16213E")
    FILL_TAG_TRAIN   = PatternFill("solid", fgColor="27AE60")
    FILL_TAG_TEST    = PatternFill("solid", fgColor="2980B9")

    WH   = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    WH_N = Font(name="Arial", color="E8E8E8", size=10)
    AMB  = Font(name="Arial", bold=True, color="F5A623", size=10)

    wb = Workbook()
    ws = wb.active
    ws.title = "Scenario Statistics"

    # Header row
    ws.row_dimensions[1].height = 38
    for c_idx, hdr in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c_idx, value=hdr)
        cell.font      = WH
        cell.fill      = FILL_HDR
        cell.alignment = CTR_W
        cell.border    = BRD

    # Train rows first, then test
    train_rows = [r for r in rows if r["Split"] == "Train"]
    test_rows  = [r for r in rows if r["Split"] == "Test"]
    ordered    = train_rows + test_rows

    for r_idx, row in enumerate(ordered):
        excel_row = r_idx + 2
        ws.row_dimensions[excel_row].height = 18
        split = row["Split"]
        alt   = r_idx % 2 == 0

        fill_row = (FILL_TRAIN_A if alt else FILL_TRAIN_B) if split == "Train" \
                   else (FILL_TEST_A if alt else FILL_TEST_B)

        for c_idx, key in enumerate(headers, start=1):
            val  = row[key]
            cell = ws.cell(row=excel_row, column=c_idx, value=val)
            cell.fill   = fill_row
            cell.border = BRD

            if c_idx == 1:  # Scenario
                cell.font      = AMB
                cell.alignment = LFT
            elif c_idx == 2:  # Split tag
                cell.fill      = FILL_TAG_TRAIN if split == "Train" else FILL_TAG_TEST
                cell.font      = WH
                cell.alignment = CTR
            else:
                cell.font      = WH_N
                cell.alignment = CTR
                if isinstance(val, float):
                    if key in ("V_min [V]", "V_max [V]"):
                        cell.number_format = "0.000"
                    elif key in ("|I|_max [A]", "σ_I [A]"):
                        cell.number_format = "0.00"
                    else:
                        cell.number_format = "0.0"

    # Column widths
    col_widths = [18, 8, 11, 11, 10, 10, 12, 10, 10, 11, 9, 10, 10, 12]
    for c_idx, w in enumerate(col_widths, start=1):
        ws.column_dimensions[get_column_letter(c_idx)].width = w

    ws.freeze_panes = "C2"
    wb.save(output)
    print(f"\nSaved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Collecting training scenarios...")
    train_rows = collect_scenarios(TRAIN_DIR, "Train")

    print("Collecting test scenarios...")
    test_rows  = collect_scenarios(TEST_DIR, "Test")

    all_rows = train_rows + test_rows
    print(f"\nTotal scenarios: {len(all_rows)} ({len(train_rows)} train, {len(test_rows)} test)")

    build_excel(all_rows, OUTPUT)


if __name__ == "__main__":
    main()