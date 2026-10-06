"""
Equivalent-circuit plant of one LFP cell, used as the "true" battery of the simulation.

States: true SOC, the voltages of two RC branches, a dynamic hysteresis state h in [-1, 1]
(-1 on the discharge OCV branch, +1 on the charge branch) and the cell temperature.
Sign convention as in the Huawei data: positive current charges the cell.

    V = OCV_mid(SOC) + h * OCV_half(SOC) + R0 * I + V1 + V2 + V3
    SOC' = SOC + eta(I) * I * dt / (3600 * Q)
    V1' = a1 * V1 + R1 * (1 - a1) * I,  a1 = exp(-dt / (R1 * C1))   (same for branches 2 and 3)

The two fast branches (tenths of a second to tens of seconds) come from the HPPC pulse
relaxations; the third, slow branch (minutes, diffusion) is constant over SOC and is fitted
on the full HPPC replays, because no rest in those tests is long enough to identify it.
    h'  = b * h + (1 - b) * sign(I),     b = exp(-gamma * |eta(I) * I| * dt / (3600 * Q))
    T'  = T + dt / (m * cp) * (I^2 R0 + V1^2 / R1 + V2^2 / R2 - hA * (T - T_amb))

The estimators' EKF switches instantly between the two OCV branches with the sign of the
current, whereas this plant moves between them gradually: the mismatch between plant and
estimator model is physical and stated, not an artefact of using the same equations twice.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

#Lorenzo Giannuzzo: identified tables are written here by identify.py and read by CellParams.from_dir
REPO_ROOT = Path(__file__).resolve().parents[3]
PLANT_DATA_DIR = REPO_ROOT / "data" / "private" / "plant"

_LUT_POINTS = 2001


@dataclass(frozen=True)
class CellParams:
    """Cell parameters; SOC grids are fractions in [0, 1] and must be ascending."""

    ocv_soc: np.ndarray
    ocv_chg: np.ndarray
    ocv_dch: np.ndarray
    chg_table: pd.DataFrame          # columns: soc, r0, r1, c1, r2, c2
    dch_table: pd.DataFrame
    capacity_ah: float = 280.0       # nominal capacity, the value assumed by the EKF
    soh_capacity: float = 1.0        # true capacity = soh_capacity * capacity_ah
    resistance_scale: float = 1.0    # ageing factor on R0, R1, R2
    coulombic_efficiency: float = 1.0  # applied to charging current only
    hysteresis_gamma: float = 100.0  # transition rate per unit of SOC exchanged, selected on scenario S4
    r3_chg: float = 0.0              # slow branch, charge (ohm); 0 disables it
    tau3_chg_s: float = 600.0
    r3_dch: float = 0.0              # slow branch, discharge (ohm)
    tau3_dch_s: float = 600.0
    v_min: float = 2.5
    v_max: float = 3.65
    mass_kg: float = 5.4
    cp_j_per_kg_k: float = 1100.0
    ha_w_per_k: float = 2.0
    ambient_c: float = 25.0

    @property
    def true_capacity_ah(self):
        return self.capacity_ah * self.soh_capacity

    def with_ageing(self, soh_capacity, resistance_scale):
        return replace(self, soh_capacity=soh_capacity, resistance_scale=resistance_scale)

    @classmethod
    def from_dir(cls, directory=PLANT_DATA_DIR, **overrides):
        """Load the tables written by identify.run_identification."""
        directory = Path(directory)
        ocv = pd.read_csv(directory / "ocv.csv")
        chg = pd.read_csv(directory / "ecm_chg.csv")
        dch = pd.read_csv(directory / "ecm_dch.csv")
        slow = {}
        if (directory / "slow_branch.csv").exists():
            sb = pd.read_csv(directory / "slow_branch.csv").set_index("direction")
            slow = dict(r3_chg=float(sb.loc["chg", "r3"]), tau3_chg_s=float(sb.loc["chg", "tau3_s"]),
                        r3_dch=float(sb.loc["dch", "r3"]), tau3_dch_s=float(sb.loc["dch", "tau3_s"]))
        slow.update(overrides)
        return cls(ocv_soc=ocv["soc"].to_numpy(float), ocv_chg=ocv["ocv_chg"].to_numpy(float),
                   ocv_dch=ocv["ocv_dch"].to_numpy(float), chg_table=chg, dch_table=dch, **slow)


@dataclass
class CellState:
    soc: float
    h: float = 0.0
    v1: float = 0.0
    v2: float = 0.0
    v3: float = 0.0
    temp_c: float = 25.0
    direction: int = 1               # last non-zero current sign, selects the RC table at rest


@dataclass
class SimulationResult:
    current_a: np.ndarray            # current actually flowing, A (0 where a request was refused)
    voltage_v: np.ndarray            # terminal voltage during each step, V
    soc: np.ndarray                  # true SOC at the start of each step, fraction
    h: np.ndarray
    temp_c: np.ndarray
    requested_power_w: np.ndarray    # NaN in current mode
    refused: np.ndarray              # True where the request would have violated a limit
    dt_s: np.ndarray = field(default=None)
    final_state: CellState = field(default=None)

    @property
    def shortfall_wh(self):
        """Energy requested but not delivered, Wh (power mode only)."""
        p = np.where(self.refused, np.abs(self.requested_power_w), 0.0)
        return float(np.nansum(p * self.dt_s) / 3600.0)


class _Lut:
    """Uniform-grid lookup tables, so that each step costs a few float operations."""

    def __init__(self, params):
        self.grid = np.linspace(0.0, 1.0, _LUT_POINTS)
        self.n = _LUT_POINTS - 1
        mid = 0.5 * (np.interp(self.grid, params.ocv_soc, params.ocv_chg)
                     + np.interp(self.grid, params.ocv_soc, params.ocv_dch))
        half = 0.5 * (np.interp(self.grid, params.ocv_soc, params.ocv_chg)
                      - np.interp(self.grid, params.ocv_soc, params.ocv_dch))
        self.ocv_mid = mid.tolist()
        self.ocv_half = half.tolist()
        rs = params.resistance_scale
        self.tables = {}
        for sign, table in ((1, params.chg_table), (-1, params.dch_table)):
            s = table["soc"].to_numpy(float)
            order = np.argsort(s)
            cols = {}
            for name in ("r0", "r1", "c1", "r2", "c2"):
                values = np.interp(self.grid, s[order], table[name].to_numpy(float)[order])
                if name.startswith("r"):
                    values = values * rs
                cols[name] = values.tolist()
            self.tables[sign] = cols

    def locate(self, soc):
        x = min(max(soc, 0.0), 1.0) * self.n
        i = min(int(x), self.n - 1)
        return i, x - i

    @staticmethod
    def interp(arr, i, w):
        return arr[i] + w * (arr[i + 1] - arr[i])


def simulate(params, dt, state, *, current_a=None, power_w=None):
    """
    Simulate the cell for a sequence of steps, driven either by current (A) or by power (W).

    dt is a scalar or an array of step lengths in seconds. In power mode the current of each
    step solves R0 I^2 + E I - P = 0 with E = OCV + V1 + V2 at the start of the step. A step
    whose current would push the voltage above v_max or the SOC above 1 while charging, or
    below v_min or 0 while discharging, is refused: the current is set to zero and the step
    is flagged in `refused`. A step that moves back towards the window is always allowed.
    """
    if (current_a is None) == (power_w is None):
        raise ValueError("give exactly one of current_a or power_w")
    drive = np.asarray(current_a if current_a is not None else power_w, dtype=float)
    n = drive.size
    dts = np.broadcast_to(np.asarray(dt, dtype=float), (n,))
    power_mode = power_w is not None

    lut = _Lut(params)
    q_as = params.true_capacity_ah * 3600.0
    eta = params.coulombic_efficiency
    gamma = params.hysteresis_gamma
    heat_cap = params.mass_kg * params.cp_j_per_kg_k

    soc, h, v1, v2, temp, direction = state.soc, state.h, state.v1, state.v2, state.temp_c, state.direction
    v3 = state.v3
    slow = {1: (params.r3_chg, params.tau3_chg_s), -1: (params.r3_dch, params.tau3_dch_s)}
    out_i = np.empty(n)
    out_v = np.empty(n)
    out_soc = np.empty(n)
    out_h = np.empty(n)
    out_t = np.empty(n)
    refused = np.zeros(n, dtype=bool)

    for k in range(n):
        step = dts[k]
        i_idx, w = lut.locate(soc)
        ocv = lut.interp(lut.ocv_mid, i_idx, w) + h * lut.interp(lut.ocv_half, i_idx, w)
        e = ocv + v1 + v2 + v3

        #Lorenzo Giannuzzo: current of the step, from the drive signal
        if power_mode:
            p = drive[k]
            sign = 1 if p > 0 else (-1 if p < 0 else direction)
            r0 = lut.interp(lut.tables[sign]["r0"], i_idx, w)
            disc = e * e + 4.0 * r0 * p
            current = 2.0 * p / (e + disc ** 0.5) if disc >= 0.0 else float("nan")
        else:
            current = drive[k]
            sign = 1 if current > 0 else (-1 if current < 0 else direction)
            r0 = lut.interp(lut.tables[sign]["r0"], i_idx, w)

        #Lorenzo Giannuzzo: refuse a step only if it pushes further out of the window in its own
        # direction (a full cell cannot charge but can discharge, an empty one the opposite)
        eff = eta if current > 0 else 1.0
        if current != current:
            ok = False
        else:
            v_try = e + r0 * current
            soc_try = soc + eff * current * step / q_as
            blocked_up = current > 0 and (v_try > params.v_max or soc_try > 1.0)
            blocked_dn = current < 0 and (v_try < params.v_min or soc_try < 0.0)
            ok = not (blocked_up or blocked_dn)
        if not ok:
            refused[k] = True
            current = 0.0
            sign = direction
            r0 = lut.interp(lut.tables[sign]["r0"], i_idx, w)

        tab = lut.tables[sign]
        r1 = lut.interp(tab["r1"], i_idx, w)
        r2 = lut.interp(tab["r2"], i_idx, w)
        c1 = lut.interp(tab["c1"], i_idx, w)
        c2 = lut.interp(tab["c2"], i_idx, w)

        out_i[k] = current
        out_v[k] = e + r0 * current
        out_soc[k] = soc
        out_h[k] = h
        out_t[k] = temp

        #Lorenzo Giannuzzo: state update over the step (zero-order hold on the current)
        eff = eta if current > 0 else 1.0
        dq = eff * current * step
        soc = soc + dq / q_as
        a1 = np.exp(-step / (r1 * c1))
        a2 = np.exp(-step / (r2 * c2))
        r3, tau3 = slow[sign]
        heat = current * current * r0 + v1 * v1 / r1 + v2 * v2 / r2 + (v3 * v3 / r3 if r3 > 0 else 0.0)
        v1 = a1 * v1 + r1 * (1.0 - a1) * current
        v2 = a2 * v2 + r2 * (1.0 - a2) * current
        a3 = np.exp(-step / tau3)
        v3 = a3 * v3 + r3 * (1.0 - a3) * current
        if current != 0.0:
            b = np.exp(-gamma * abs(dq) / q_as)
            h = b * h + (1.0 - b) * (1.0 if current > 0 else -1.0)
            direction = 1 if current > 0 else -1
        temp = temp + step / heat_cap * (heat - params.ha_w_per_k * (temp - params.ambient_c))

    result = SimulationResult(
        current_a=out_i, voltage_v=out_v, soc=out_soc, h=out_h, temp_c=out_t,
        requested_power_w=drive.copy() if power_mode else np.full(n, np.nan), refused=refused,
        dt_s=np.array(dts, dtype=float),
        final_state=CellState(soc=soc, h=h, v1=v1, v2=v2, v3=v3, temp_c=temp, direction=direction))
    return result
