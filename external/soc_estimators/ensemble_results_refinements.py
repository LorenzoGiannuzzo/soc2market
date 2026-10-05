"""
================================================================================
SOC Prediction Framework for LFP Battery Cells
================================================================================

Authors:
    Lorenzo Giannuzzo        - lorenzo.giannuzzo@polito.it
    Elena Rozzi               - elena.rozzi@polito.it
    Marcel Stolte             - marcel.stolte@polito.it
    Daniele Salvatore Schiera - daniele.schiera@polito.it
    Francesco D. Minuto       - francesco.minuto@polito.it
    Andrea Lanzini            - andrea.lanzini@polito.it

Supervision:
    Mansour Alramlawi         - Huawei Technologies

Affiliation:
    Politecnico di Torino
    Department of Energy (DENERG) - Energy Center Lab
    Corso Duca degli Abruzzi 24, 10129 Torino, Italy

================================================================================
"""

"""
================================================================================
Ensemble Dominant Plot — Refined Re-generation
================================================================================

Standalone script that re-runs KF → Ensemble for every scenario and saves
the Ensemble dominant visualisation to results/results_refinements/.

Key difference from the original main.py output:
  - The stacked-area weight chart (bottom panel) is rendered with
    rasterized=True, so it is embedded in the PDF as a single raster
    block instead of thousands of filled polygons.  This prevents
    Affinity Designer 2 from crashing while leaving the upper line-plot
    panel (True SOC, dominant-colour Ensemble line), all legend entries,
    axis labels and titles fully editable as vectors.

  - The LineCollection in the upper panel is also rasterized by default
    (it can produce hundreds of tiny segments).  Set
    RASTERIZE_LINE_COLLECTION = False below if you want to keep it
    fully vectorised and editable in Affinity Designer.

Usage:
    python regenerate_ensemble_plots_refined.py

    Place this file in the same directory as main.py (project root).
    Reads data from:   data/
    Writes plots to:   results/results_refinements/
================================================================================
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless backend
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import sys
import time
from pathlib import Path

# ============================================================================
# CONFIGURATION — edit here if needed
# ============================================================================

SEQUENCE_LENGTH  = 10
ENSEMBLE_TYPE    = 'selector'          # same as in main.py
RASTER_DPI       = 600                 # DPI used for the rasterized bottom panel
VECTOR_DPI       = 500                 # DPI for the overall PDF save (same as main.py)

# Set to False to keep the multi-colour Ensemble line as vectors (heavier file,
# but every segment is individually editable in Affinity Designer)
RASTERIZE_LINE_COLLECTION = True

# ============================================================================
# PATHS  — resolved relative to this script (project root)
# ============================================================================
SCRIPT_DIR        = Path(__file__).resolve().parent
DATA_DIR          = SCRIPT_DIR / 'data'
RESULTS_DIR       = SCRIPT_DIR / 'results'
OUTPUT_DIR        = RESULTS_DIR / 'results_refinements'

MODEL_DIR         = SCRIPT_DIR / 'nn_training' / 'models'
ENSEMBLE_MODELS   = {
    'dynamic':  'ensemble_dynamic_model.keras',
    'selector': 'ensemble_selector_model.keras',
    'neural':   'ensemble_neural_model.keras',
    'stacking': 'ensemble_stacking_model.json',
}

# ============================================================================
# COLOUR / STYLE PALETTE  (identical to main.py)
# ============================================================================
COLOR_TRUE_SOC = '#1A1A1A'
COLOR_KF       = '#0A2463'
COLOR_ANN      = '#1E88E5'
COLOR_CNN      = '#42A5F5'
COLOR_LSTM     = '#E65100'
COLOR_CATBOOST = '#FB8C00'
COLOR_TFT      = '#7B1FA2'

STYLE_TRUE_SOC = {'color': COLOR_TRUE_SOC, 'linestyle': '-', 'linewidth': 2.0, 'alpha': 1.0}


# ============================================================================
# IMPORTS FROM PROJECT MODULES
# ============================================================================
def _import_project_modules():
    """Add the model_execution_modules sub-directory to sys.path."""
    module_dir = SCRIPT_DIR / 'model_execution_modules'
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))


def _load_modules():
    _import_project_modules()
    global KalmanFilter, Ensemble_BatteryAnalysis
    from KalmanFilter_module_all_data      import KalmanFilter
    from ensemble_prediction_module_all_data import Ensemble_BatteryAnalysis


# ============================================================================
# DATA UTILITIES
# ============================================================================
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the standard names expected by the pipeline."""
    renames = {
        'Voltage(V)':     'Voltage',
        'Current_inv(A)': 'Current_inv',
        'SOC_true(%)':    'SOC_true',
        'SOC_True(%)':    'SOC_true',
        'SOC_True':       'SOC_true',
        'Temp1':          'Temp',
        'Step(s)':        'Step',
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    for col in ['Voltage', 'Current_inv', 'SOC_true', 'Temp', 'Step']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def run_kalman_filter(bat_data_df: pd.DataFrame, soc_initial: float,
                      scenario_name: str) -> pd.DataFrame:
    if 'Temp' not in bat_data_df.columns:
        bat_data_df['Temp'] = 25.0

    if 'HPPC' in scenario_name and 'DSG' in scenario_name:
        cap_ch = cap_dch = 3600
    else:
        cap_ch = cap_dch = 280

    kf = KalmanFilter(bat_data_df, dt=1,
                      capacity_charge=cap_ch, capacity_discharge=cap_dch,
                      opt=False, initial_SOC=soc_initial)

    P = np.array([[3.393996e-03, 0, 0],
                  [0,  6.662388e-04, 0],
                  [0,  0,  6.946038e-03]])
    Q = np.array([[4.034329e-07, 0, 0],
                  [0,  8.628196e-05, 0],
                  [0,  0,  9.095089e-05]])
    R = 3.743003e-03

    return kf.kalman_filter(Q=Q, R=R, initial_covariance=P)


def run_ensemble(bat_data_kf: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """Run ensemble prediction and attach weights to the returned DataFrame."""
    model_dir = str(model_path.parent)
    model = Ensemble_BatteryAnalysis(ensemble_type=ENSEMBLE_TYPE, model_dir=model_dir)

    bat_pred = model.make_predictions(bat_data_kf.copy())

    if hasattr(model, 'get_model_weights'):
        try:
            weights_df = model.get_model_weights(bat_data_kf.copy())
            bat_pred['Ensemble_Weights'] = weights_df.values.tolist()
        except Exception as exc:
            print(f"    Warning: could not retrieve ensemble weights: {exc}")
            bat_pred['Ensemble_Weights'] = None

    return bat_pred


# ============================================================================
# PLOT FUNCTION  (refined version with rasterized bottom panel)
# ============================================================================
def plot_ensemble_dominant_refined(scenario_name: str,
                                   output_dir: Path,
                                   bat_data_kf: pd.DataFrame,
                                   bat_data_ensemble: pd.DataFrame) -> None:
    """
    Reproduce the Ensemble dominant visualisation from main.py with two changes:

    1. The stacked-area weight chart (bottom panel) is rasterized, so Affinity
       Designer 2 sees a single image block instead of thousands of filled
       polygons.  All text, axis frames, grid lines and the legend in the bottom
       panel remain fully vectorised.

    2. The LineCollection in the upper panel can optionally be rasterized via
       RASTERIZE_LINE_COLLECTION (see module-level config above).

    The True SOC line, every legend entry, every axis label / title, and all
    annotation text are always vectorised and therefore fully editable.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    step     = bat_data_kf['Step'].astype(float).values
    soc_true = bat_data_kf['SOC_true'].astype(float).values
    soc_pred = bat_data_ensemble['Predicted_SOC'].astype(float).values

    has_weights = (
        'Ensemble_Weights' in bat_data_ensemble.columns
        and bat_data_ensemble['Ensemble_Weights'].iloc[0] is not None
    )

    ax1 = axes[0]

    # ------------------------------------------------------------------
    # True SOC — always vectorised
    # ------------------------------------------------------------------
    ax1.plot(step, soc_true, label='True SOC', **STYLE_TRUE_SOC, zorder=1)

    if has_weights:
        weights_array = np.array(bat_data_ensemble['Ensemble_Weights'].tolist())
        n_base = weights_array.shape[1]

        dominant_idx = np.argmax(weights_array, axis=1)

        if n_base == 6:
            colors_map         = [COLOR_KF, COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT]
            model_names_ordered = ['KF', 'ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']
        else:
            colors_map         = [COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT]
            model_names_ordered = ['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT']

        # --------------------------------------------------------------
        # Dominant-colour Ensemble line (LineCollection)
        # rasterized=RASTERIZE_LINE_COLLECTION controls vector vs raster
        # --------------------------------------------------------------
        points   = np.array([step, soc_pred]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = [colors_map[dominant_idx[i]] for i in range(len(segments))]

        lc = LineCollection(segments, colors=seg_colors, linewidths=2.5, alpha=0.9,
                            zorder=2, rasterized=RASTERIZE_LINE_COLLECTION)
        ax1.add_collection(lc)

        # Legend (always vector)
        legend_handles = [Line2D([0], [0], color=COLOR_TRUE_SOC, linewidth=2, label='True SOC')]
        for i, name in enumerate(model_names_ordered):
            legend_handles.append(
                Line2D([0], [0], color=colors_map[i], linewidth=2.5,
                       label=f'Ensemble \u2192 {name} dominant')
            )
        ax1.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.95)

        # ==============================================================
        # Bottom panel — stacked area chart with rasterized=True
        # The axis frame, tick marks, labels, legend and title are NOT
        # rasterized (they are vector elements drawn by matplotlib after
        # the fill polygons); only the coloured fill areas are rasterized.
        # ==============================================================
        ax2 = axes[1]

        if n_base == 6:
            stacks = ax2.stackplot(
                step,
                weights_array[:, 0], weights_array[:, 1], weights_array[:, 2],
                weights_array[:, 3], weights_array[:, 4], weights_array[:, 5],
                labels=['KF', 'ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT'],
                colors=[COLOR_KF, COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT],
                alpha=0.8,
            )
        else:
            stacks = ax2.stackplot(
                step,
                weights_array[:, 0], weights_array[:, 1], weights_array[:, 2],
                weights_array[:, 3], weights_array[:, 4],
                labels=['ANN', 'CNN', 'LSTM', 'CatBoost', 'TFT'],
                colors=[COLOR_ANN, COLOR_CNN, COLOR_LSTM, COLOR_CATBOOST, COLOR_TFT],
                alpha=0.8,
            )

        # *** Key change: rasterize every filled polygon in the stackplot ***
        for poly in stacks:
            poly.set_rasterized(True)

        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Step [s]', fontsize=11)
        ax2.set_ylabel('Model Weight', fontsize=11)
        ax2.set_title('Dynamic Model Weights Over Time', fontsize=11, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8, ncol=n_base)
        ax2.grid(True, alpha=0.3)

        # Dominance statistics annotation (vector text)
        dominance_counts = np.bincount(dominant_idx, minlength=n_base)
        dominance_pct    = dominance_counts / len(dominant_idx) * 100
        stats_text = 'Dominance: ' + ' | '.join(
            [f'{model_names_ordered[i]}: {dominance_pct[i]:.1f}%' for i in range(n_base)]
        )
        ax2.text(0.5, -0.25, stats_text, transform=ax2.transAxes,
                 fontsize=9, ha='center', va='top', style='italic')

    else:
        # Fallback: no weights available
        ax1.plot(step, soc_pred, label='KF + Ensemble',
                 color=COLOR_ANN, linestyle='-', linewidth=2.0, alpha=0.95, zorder=2)
        ax1.legend(loc='best', fontsize=10)
        axes[1].set_visible(False)

    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Step [s]', fontsize=11)
    ax1.set_ylabel('SOC [%]', fontsize=11)
    ax1.set_title(
        f'Ensemble SOC Prediction with Dominant Model Visualization - {scenario_name}',
        fontsize=12, fontweight='bold'
    )
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save — PDF backend honours rasterized=True at VECTOR_DPI for the
    # rasterized portions and keeps everything else as proper vectors
    pdf_path = output_dir / f'{scenario_name}_Ensemble_dominant.pdf'
    png_path = output_dir / f'{scenario_name}_Ensemble_dominant.png'

    fig.savefig(pdf_path, dpi=RASTER_DPI, bbox_inches='tight',
                backend='pdf')          # explicit backend ensures rasterized is applied
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"    Saved: {pdf_path.name}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    # ------------------------------------------------------------------
    # Import project modules
    # ------------------------------------------------------------------
    try:
        _load_modules()
    except ImportError as exc:
        print(f"ERROR: Could not import project modules.\n"
              f"       Make sure this script is in the project root (same folder as main.py).\n"
              f"       Details: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Check model path
    # ------------------------------------------------------------------
    ensemble_model_path = MODEL_DIR / ENSEMBLE_MODELS[ENSEMBLE_TYPE]
    if not ensemble_model_path.exists():
        print(f"ERROR: Ensemble model not found: {ensemble_model_path}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Gather data files — only the four target scenarios
    # ------------------------------------------------------------------
    TARGET_SCENARIOS = {
        'S1_DAY4to7',
        'S2_DAY7to14',
        'S3_DAY4to7',
        'S4_DAY0to4',
    }

    data_files = sorted([
        f for f in DATA_DIR.glob('*.xlsx')
        if f.stem in TARGET_SCENARIOS
    ])

    if not data_files:
        print(f"ERROR: None of the target scenarios found in {DATA_DIR}")
        print(f"       Expected file names: {', '.join(sorted(TARGET_SCENARIOS))}")
        sys.exit(1)

    missing = TARGET_SCENARIOS - {f.stem for f in data_files}
    if missing:
        print(f"  WARNING: these scenarios were not found and will be skipped:")
        for name in sorted(missing):
            print(f"    \u2022 {name}")

    print(f"\n{'='*70}")
    print(f"  ENSEMBLE DOMINANT PLOT — REFINED REGENERATION")
    print(f"  Ensemble type : {ENSEMBLE_TYPE}")
    print(f"  Scenarios     : {len(data_files)}")
    print(f"  Raster DPI    : {RASTER_DPI}  (stackplot fill areas)")
    print(f"  LC rasterized : {RASTERIZE_LINE_COLLECTION}  (dominant-colour line)")
    print(f"  Output        : {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    total_start = time.time()
    ok = 0
    errors = []

    for idx, fpath in enumerate(data_files):
        scenario = fpath.stem
        t0 = time.time()
        print(f"[{idx+1}/{len(data_files)}] {scenario}")

        try:
            # Load & standardize
            df = pd.read_excel(fpath, skiprows=2).iloc[1:]
            df = standardize_columns(df)

            if 'SOC_true' not in df.columns:
                raise ValueError("Missing SOC_true column")

            soc_initial = float(df['SOC_true'].iloc[0])
            if np.isnan(soc_initial):
                soc_initial = 50.0

            # KF
            bat_kf = run_kalman_filter(df.copy(), soc_initial, scenario)

            # Ensemble
            bat_ens = run_ensemble(bat_kf, ensemble_model_path)

            # Plot
            plot_ensemble_dominant_refined(scenario, OUTPUT_DIR, bat_kf, bat_ens)

            ok += 1
            print(f"    Done in {time.time()-t0:.1f}s")

        except Exception as exc:
            import traceback
            print(f"    ERROR: {exc}")
            traceback.print_exc()
            errors.append((scenario, str(exc)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  DONE  —  {ok}/{len(data_files)} scenarios OK  |  total {elapsed:.1f}s")
    if errors:
        print(f"\n  Failed scenarios:")
        for name, msg in errors:
            print(f"    • {name}: {msg}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()