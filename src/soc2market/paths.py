"""
Result folders of the pipeline, switched by the MSD realisation mode.

MSD_MODE = "chronological": awards of aFRR and mFRR follow the hours in which Terna actually
accepted upward or downward offers in MSD ex-ante (Nord, 2023), at the price of that hour.
MSD_MODE = "bernoulli": awards are independent draws from the conditioned MSD profiles (first
version of the pipeline). Its results stay in results/ for comparison; every new mode writes
to its own subfolder, so that runs of different modes never mix.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MSD_MODE = "chronological"

RUN_DIR = REPO_ROOT / "results" if MSD_MODE == "bernoulli" else REPO_ROOT / "results" / MSD_MODE
DUTY_DIR = RUN_DIR / "duty_cycles"
LOOP_DIR = RUN_DIR / "closed_loop_matched"
ERROR_DIR = RUN_DIR / "error_model"
MARKET_STUDY_DIR = RUN_DIR / "market_study"
