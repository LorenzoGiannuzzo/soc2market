"""
Bridge to the frozen MARL market and MILP code stored in external/marl_bess.

Market data files are listed explicitly by name and never discovered by globbing the
folder, so that partial or overlapping workbooks left in data/market cannot enter a run.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

#Lorenzo Giannuzzo: repository layout, resolved from this file so that it works from any working directory
REPO_ROOT = Path(__file__).resolve().parents[3]
MARL_CODE_DIR = REPO_ROOT / "external" / "marl_bess"
MARKET_DIR = REPO_ROOT / "data" / "market"

PUN_FILES = {
    2023: "20230101_20231231_PUN.xlsx",
    2024: "20240101_20241231_PUN.xlsx",
}
MSD_FILES = (
    "EsitiMSD_PrezziExAnte_20230101_20230401_Nord.xlsx",
    "EsitiMSD_PrezziExAnte_20230401_20240331_Nord.xlsx",
    "EsitiMSD_PrezziExAnte_20240401_20241231_Nord.xlsx",
)


def ensure_import_path():
    """The frozen modules import each other with flat imports, so their folder must be on sys.path."""
    if str(MARL_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(MARL_CODE_DIR))


def market_files_missing():
    """Return the market workbooks that are not present, as strings."""
    names = list(PUN_FILES.values()) + list(MSD_FILES)
    return [str(MARKET_DIR / n) for n in names if not (MARKET_DIR / n).exists()]


def market_window(year, n_days=365, price_column="PUN"):
    """
    Build the market window with the same call used by aggregation_value.main for real data:
    day-ahead prices from the GME workbook of that year, MSD Nord workbooks for the reserve
    products, directional services on.
    """
    ensure_import_path()
    from italian_market_data import make_market_window_from_real_pun

    start = datetime(year, 1, 1)
    return make_market_window_from_real_pun(
        start_date=start,
        end_date=start + timedelta(days=n_days),
        pun_xlsx_path=str(MARKET_DIR / PUN_FILES[year]),
        price_column=price_column,
        msd_xlsx_paths=[str(MARKET_DIR / name) for name in MSD_FILES],
        directional_services=True,
    )
