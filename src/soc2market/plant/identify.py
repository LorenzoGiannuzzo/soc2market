"""
Identification of the plant parameters from the laboratory data of the cell.

OCV: the charge and discharge branches of Cha_Dis_OCV_SOC_Data.xlsx, put on a common SOC grid.
RC branches: the pulse segmentation and the R0 estimate of the frozen HPPC modules are reused
as they are, while R1, C1, R2, C2 are fitted again on each relaxation with the correct model

    V(t) = V_inf - s * |I| * [R1 (1 - e^(-T/tau1)) e^(-t/tau1) + R2 (1 - e^(-T/tau2)) e^(-t/tau2)]

where T is the pulse length, s = +1 after a discharge pulse (the voltage recovers upwards)
and s = -1 after a charge pulse (it relaxes downwards). The asymptote V_inf is a free
parameter, because the rests are too short for the voltage to reach the OCV. The frozen
discharge fit places the second RC term inside the first exponential, so its R1/C1/R2/C2
are not used here.

The Step(s) column of the discharge workbook counts rows: its sampling interval, implied by
the coulomb balance against SOC_true, is about 0.084 s. Every workbook is therefore read with
a verified time base (hppc_time_step), shared by identification and validation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from soc2market.estimation import estimator_bridge as eb
from soc2market.plant.cell import PLANT_DATA_DIR, CellParams, CellState, simulate

#Lorenzo Giannuzzo: a relaxation fitted worse than this (RMS, volts) is reported and replaced
# by interpolation between its neighbours
MAX_FIT_RMSE_V = 0.002
DEGENERATE_R = 1e-6
DEGENERATE_TAU_S = 1e4
TIME_BASE_TOL = 0.02


def relaxation_model(t, v_inf, r1, tau1, r2, tau2, current_abs, pulse_s, sign):
    return v_inf - sign * current_abs * (
        r1 * (1.0 - np.exp(-pulse_s / tau1)) * np.exp(-t / tau1)
        + r2 * (1.0 - np.exp(-pulse_s / tau2)) * np.exp(-t / tau2))


def fit_relaxation(t, v, current_abs, pulse_s, is_charge, dt=1.0):
    """
    Fit one relaxation; returns v_inf, r1, tau1, r2, tau2 (seconds), the RMS error in volts
    and whether any parameter ended on a bound. The two time constants are kept ordered
    (tau2 = tau1 + d, d > 0) so that the branches cannot swap between neighbouring SOC points.

    The fit runs in units of samples (t / dt), so that bounds and initial guesses mean the same
    thing whatever the sampling interval; the time constants are converted back to seconds.
    """
    t = np.asarray(t, float) / dt
    pulse_s = pulse_s / dt
    v = np.asarray(v, float)
    sign = -1.0 if is_charge else 1.0

    def unpack(x):
        v_inf, r1, log_tau1, r2, log_d = x
        tau1 = np.exp(log_tau1)
        return v_inf, r1, tau1, r2, tau1 + np.exp(log_d)

    def residual(x):
        return relaxation_model(t, *unpack(x), current_abs, pulse_s, sign) - v

    span = max(t[-1], 1.0)
    x0 = [v[-1], 2e-4, np.log(min(10.0, span / 4)), 2e-4, np.log(max(span / 2, 5.0))]
    lb = [v.min() - 0.1, 1e-7, np.log(0.5), 1e-7, np.log(0.5)]
    ub = [v.max() + 0.1, 5e-3, np.log(600.0), 5e-3, np.log(2e4)]
    sol = least_squares(residual, x0, bounds=(lb, ub), x_scale="jac", max_nfev=5000)
    v_inf, r1, tau1, r2, tau2 = unpack(sol.x)
    rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    #Lorenzo Giannuzzo: a branch parameter sitting on its bound means the fit could not place it
    at_bound = any(abs(sol.x[k] - lb[k]) <= 1e-6 * max(1.0, abs(lb[k])) or
                   abs(sol.x[k] - ub[k]) <= 1e-6 * max(1.0, abs(ub[k])) for k in (1, 2, 3, 4))
    return dict(v_inf=v_inf, r1=r1, tau1=tau1 * dt, r2=r2, tau2=tau2 * dt, rmse_v=rmse,
                success=bool(sol.success), at_bound=bool(at_bound))


def load_ocv_branches(path):
    """Charge and discharge OCV branches on a common ascending SOC grid (fraction)."""
    df = pd.read_excel(path, skiprows=1)
    chg = df[["data_SOC", "data_U"]].dropna().astype(float).sort_values("data_SOC")
    dch = df[["data_SOC.1", "data_U.1"]].dropna().astype(float).sort_values("data_SOC.1")
    lo = max(chg["data_SOC"].min(), dch["data_SOC.1"].min())
    hi = min(chg["data_SOC"].max(), dch["data_SOC.1"].max())
    grid = np.linspace(lo, hi, 401)
    return pd.DataFrame({
        "soc": grid / 100.0,
        "ocv_chg": np.interp(grid, chg["data_SOC"], chg["data_U"]),
        "ocv_dch": np.interp(grid, dch["data_SOC.1"], dch["data_U.1"]),
    })


def _sample_period(segment):
    try:
        steps = pd.to_numeric(segment["Step(s)"], errors="coerce").to_numpy(float)
        d = np.diff(steps)
        d = d[np.isfinite(d) & (d > 0)]
        return float(np.median(d)) if d.size else 1.0
    except KeyError:
        return 1.0


def coulomb_time_step(current_a, soc_pct, capacity_ah=280.0, min_current_a=5.0):
    """
    Sampling interval implied by the SOC_true column: for every row with current flowing,
    dt = (delta SOC / 100) * capacity * 3600 / |I|; the median over those rows is returned.
    """
    current_a = np.asarray(current_a, float)
    soc_pct = np.asarray(soc_pct, float)
    dsoc = np.diff(soc_pct)
    cur = current_a[1:]
    mask = np.isfinite(dsoc) & np.isfinite(cur) & (np.abs(cur) > min_current_a)
    implied = np.abs(dsoc[mask]) / 100.0 * capacity_ah * 3600.0 / np.abs(cur[mask])
    return float(np.median(implied))


def hppc_time_step(step_s, current_a, soc_pct, capacity_ah=280.0, tol=TIME_BASE_TOL):
    """
    Time step of an HPPC workbook. The declared one (median difference of Step(s)) is
    accepted when it agrees with the coulomb balance within `tol`; otherwise the step
    implied by the balance is used. Returns (dt in seconds, source).
    """
    d = np.diff(np.asarray(step_s, float))
    d = d[np.isfinite(d) & (d > 0)]
    declared = float(np.median(d)) if d.size else 1.0
    implied = coulomb_time_step(current_a, soc_pct, capacity_ah)
    if abs(implied / declared - 1.0) <= tol:
        return declared, "declared"
    return implied, "coulomb balance"


def read_hppc_workbook(name):
    """Read an HPPC workbook as the frozen modules do, with a verified time base."""
    df = pd.read_excel(eb.ESTIMATOR_WORKDIR / "data" / name, skiprows=2).drop(index=0)
    out = pd.DataFrame({
        "step": pd.to_numeric(df["Step(s)"], errors="coerce"),
        "i": pd.to_numeric(df["Current_inv(A)"], errors="coerce"),
        "v": pd.to_numeric(df["Voltage(V)"], errors="coerce"),
        "soc": pd.to_numeric(df["SOC_true(%)"], errors="coerce") / 100.0,
    }).dropna().reset_index(drop=True)
    dt, source = hppc_time_step(out["step"], out["i"], 100.0 * out["soc"])
    return out, dt, source


def identify_direction(hppc, is_charge, dt_s=None):
    """
    Fit every pulse of one frozen HPPC object; returns one row per pulse. When dt_s is given
    it overrides the Step(s) column, which in the discharge workbook counts rows, not seconds.
    """
    rows = []
    for k in range(len(hppc.DE)):
        de = hppc.DE[k]
        dt = dt_s if dt_s is not None else _sample_period(de)
        v = pd.to_numeric(de["Voltage(V)"], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        v = v[ok]
        t = np.arange(ok.sum()) * dt
        current_abs = abs(float(hppc.AB[k].iloc[1]["Current_inv(A)"]))
        pulse_s = len(hppc.BC[k]) * (dt_s if dt_s is not None else _sample_period(hppc.BC[k]))
        soc = float(hppc.AB[k].iloc[0]["SOC_true(%)"]) / 100.0
        row = dict(soc=soc, r0=float(hppc.R0_ds[k]), current_a=current_abs, pulse_s=pulse_s,
                   n_points=int(v.size), dt_s=dt)
        if v.size >= 6:
            fit = fit_relaxation(t, v, current_abs, pulse_s, is_charge, dt=dt)
            row.update(r1=fit["r1"], c1=fit["tau1"] / fit["r1"], r2=fit["r2"], c2=fit["tau2"] / fit["r2"],
                       tau1=fit["tau1"], tau2=fit["tau2"], v_inf=fit["v_inf"], rmse_mv=1e3 * fit["rmse_v"],
                       success=fit["success"], at_bound=fit["at_bound"])
        else:
            row.update(r1=np.nan, c1=np.nan, r2=np.nan, c2=np.nan, tau1=np.nan, tau2=np.nan,
                       v_inf=np.nan, rmse_mv=np.nan, success=False, at_bound=True)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("soc").reset_index(drop=True)


def clean_table(raw):
    """
    Keep the fits that converged with an acceptable error and physically meaningful branches;
    replace the others by linear interpolation in SOC between the accepted neighbours (held
    constant beyond the last accepted point), and mark them in `replaced`.
    """
    table = raw.copy()
    #Lorenzo Giannuzzo: a branch whose resistance collapses onto its lower bound, or whose time
    # constant runs onto its upper bound, has vanished from the fit even if the error is small
    degenerate = (table["r1"] < DEGENERATE_R) | (table["r2"] < DEGENERATE_R) | (table["tau2"] > DEGENERATE_TAU_S)
    if "at_bound" in table:
        degenerate = degenerate | table["at_bound"].astype(bool)
    good = (table["success"] & (table["rmse_mv"] <= 1e3 * MAX_FIT_RMSE_V) & (table["r0"] > 0)
            & ~degenerate.fillna(True))
    table["replaced"] = ~good
    if good.sum() < 2:
        raise RuntimeError("fewer than two acceptable relaxation fits: check the HPPC segmentation")
    for col in ("r0", "r1", "c1", "r2", "c2"):
        table.loc[~good, col] = np.interp(table.loc[~good, "soc"], table.loc[good, "soc"], table.loc[good, col])
    return table


def run_identification(out_dir=PLANT_DATA_DIR):
    """Identify OCV and RC tables and write ocv.csv, ecm_chg.csv, ecm_dch.csv (plus raw fits)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    #Lorenzo Giannuzzo: the frozen HPPC modules read their workbooks through relative paths
    if str(eb.SOC_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(eb.SOC_CODE_DIR))
    from RC_model_CHG_module import HPPC_CHG
    from RC_model_DCH_module import HPPC_DSG

    ocv = load_ocv_branches(eb.ESTIMATOR_WORKDIR / "data" / "Cha_Dis_OCV_SOC_Data.xlsx")
    ocv.to_csv(out_dir / "ocv.csv", index=False)

    tables = {}
    workbooks = {"chg": "HPPC_25degree_CHG.xlsx", "dch": "HPPC_25degree_DSG.xlsx"}
    with eb.estimator_cwd():
        for name, cls, is_charge in (("chg", HPPC_CHG, True), ("dch", HPPC_DSG, False)):
            _, dt_s, source = read_hppc_workbook(workbooks[name])
            print(f"{workbooks[name]}: time step {dt_s:.4f} s ({source})")
            hppc = cls()
            hppc.segment_extraction()
            hppc.calculate_R0()
            raw = identify_direction(hppc, is_charge, dt_s=dt_s)
            raw.to_csv(out_dir / f"ecm_{name}_raw.csv", index=False)
            table = clean_table(raw)
            table[["soc", "r0", "r1", "c1", "r2", "c2", "replaced"]].to_csv(out_dir / f"ecm_{name}.csv", index=False)
            tables[name] = table
    return ocv, tables


#Lorenzo Giannuzzo: the slow branch is fitted only where the OCV table is reliable and the replay
# follows the measured current; replays use a widened voltage window so that no step is refused
SLOW_FIT_SOC_RANGE = (0.05, 0.95)
REPLAY_LIMITS = dict(v_min=1.5, v_max=4.5)


def replay_error(params, data, dt_s, h0):
    """Voltage residuals (simulated minus measured) of a current-driven replay, and the replay."""
    state = CellState(soc=float(data["soc"].iloc[0]), h=h0, temp_c=params.ambient_c)
    res = simulate(params, dt_s, state, current_a=data["i"].to_numpy())
    return res.voltage_v - data["v"].to_numpy(), res


def fit_slow_branch(params, data, dt_s, h0, direction, stride=10):
    """
    Fit R3 and tau3 of one direction ("chg" or "dch") by minimising the voltage error of the
    full replay of an HPPC test, on the samples inside SLOW_FIT_SOC_RANGE (every `stride`-th).
    Returns the fitted values and the RMS error before and after, in millivolts.
    """
    soc = data["soc"].to_numpy()
    mask = (soc >= SLOW_FIT_SOC_RANGE[0]) & (soc <= SLOW_FIT_SOC_RANGE[1])
    idx = np.flatnonzero(mask)[::stride]
    key_r, key_t = f"r3_{direction}", f"tau3_{direction}_s"

    def residual(x):
        trial = replace_params(params, **{key_r: x[0], key_t: float(np.exp(x[1]))})
        err, _ = replay_error(trial, data, dt_s, h0)
        return err[idx]

    before = float(np.sqrt(np.mean(residual([0.0, np.log(600.0)]) ** 2)))
    sol = least_squares(residual, [1e-4, np.log(120.0)], bounds=([0.0, np.log(5.0)], [2e-3, np.log(5000.0)]),
                        x_scale=[1e-4, 1.0], diff_step=[1e-2, 1e-2], max_nfev=200)
    after = float(np.sqrt(np.mean(sol.fun ** 2)))
    return dict(direction=direction, r3=float(sol.x[0]), tau3_s=float(np.exp(sol.x[1])),
                rmse_before_mv=1e3 * before, rmse_after_mv=1e3 * after, success=bool(sol.success))


def replace_params(params, **kw):
    from dataclasses import replace
    return replace(params, **kw)


def run_slow_branch_identification(out_dir=PLANT_DATA_DIR):
    """
    Fit the slow branch on the two HPPC replays: discharge first (that test has no charge
    current), then charge with the discharge branch fixed. Writes slow_branch.csv.
    """
    out_dir = Path(out_dir)
    params = CellParams.from_dir(out_dir, **REPLAY_LIMITS, r3_chg=0.0, r3_dch=0.0)
    rows = []
    for direction, name, h0 in (("dch", "HPPC_25degree_DSG.xlsx", 1.0), ("chg", "HPPC_25degree_CHG.xlsx", -1.0)):
        data, dt_s, _ = read_hppc_workbook(name)
        fit = fit_slow_branch(params, data, dt_s, h0, direction)
        rows.append(fit)
        params = replace_params(params, **{f"r3_{direction}": fit["r3"], f"tau3_{direction}_s": fit["tau3_s"]})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "slow_branch.csv", index=False)
    return table
