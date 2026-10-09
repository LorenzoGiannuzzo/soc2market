"""
Market study: the 1 MW / 2 MWh unit operated on the 2023 Italian markets by an operator who sees
either the true SOC (oracle) or the SOC of one of the published estimators.

Every day the MILP plans from the SOC the operator believes, within the SOC window of the
strategy; awards and activations are drawn, and the day is executed second by second on the
plant with the measured grid frequency, the operator correcting the energy schedule every hour
from the SOC it believes. The plant enforces only its physical limits (SOC 0 to 100%, voltage
limits of the cell), so an estimation error turns into a refusal only when it exceeds the margin
that the strategy keeps between its window and those limits.

The day is then settled hour by hour on what the plant actually delivered:
    capacity      awarded band times capacity price, forfeited for the hour if a product that was
                  awarded (FCR) or called (aFRR, mFRR) in a direction could not be fully delivered
    activation    delivered activation energy at the product energy price, paid on upward
                  products and paid by the provider on downward ones (sign convention of the MILP)
    energy        all other delivered energy (base, intraday correction, FCR response) at the PUN
    penalty       undelivered reserve energy times PENALTY_K times the product energy price (PUN
                  for FCR), the expected-penalty convention of the MILP
    degradation   delivered throughput times the linear degradation rate of the MILP
Undelivered energy of an hour and direction is attributed to the reserve products in proportion to
the energy they requested in that direction, the rest to the base schedule.

Awards of aFRR and mFRR follow paths.MSD_MODE: in chronological mode the hours in which MSD was
actually active in 2023, at the MSD price of the hour; in bernoulli mode independent draws.

Common random numbers: for a given week and replicate, every strategy sees the same award and
activation draws, and every strategy of one estimator the same error-generator stream.
"""

import copy
import dataclasses
from dataclasses import dataclass

import numpy as np

from datetime import datetime, timedelta

from soc2market.bidding.schedule import SERVICE_KEYS, HourlySchedule, solve_day_schedule
from soc2market.errormodel.generator import ErrorGenerator, ReplayError
from soc2market import paths
from soc2market.signals.duty_cycle import DOWNWARD, UPWARD, draw_realisation, draw_realisation_chronological
from soc2market.signals.execution import execute_day
from soc2market.signals.profile import fcr_response

SECONDS_PER_HOUR = 3600
YEAR = 2023
UNIT_INDEX = 8                      # medium_2h of the aggregation portfolio, 1.0 MW / 2.0 MWh
SEED = 2023
WEEKS = {"winter": range(8, 15), "spring": range(99, 106), "summer": range(190, 197), "autumn": range(281, 288)}
SEASONS = tuple(WEEKS)
INITIAL_SOC = 0.5
DAYS_IN_YEAR = 365


@dataclass(frozen=True)
class Block:
    """Seven consecutive days simulated with the SOC carried from one day to the next."""

    label: str
    index: int
    days: tuple
    season: int          # week of the error library used for the error of these days


def season_of(day):
    """0 winter (Dec to Feb), 1 spring (Mar to May), 2 summer (Jun to Aug), 3 autumn (Sep to Nov)."""
    month = (datetime(YEAR, 1, 1) + timedelta(days=day)).month
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}.get(month, 3)


def week_blocks():
    """The four reference weeks of the error library."""
    return [Block(name, i, tuple(days), i) for i, (name, days) in enumerate(WEEKS.items())]


def year_blocks():
    """The whole year as 52 blocks of seven days (31 December left out), error library of the season."""
    blocks = []
    for i in range(DAYS_IN_YEAR // 7):
        days = tuple(range(7 * i, 7 * i + 7))
        blocks.append(Block(f"w{i + 1:02d}", i, days, season_of(days[0])))
    return blocks

#Lorenzo Giannuzzo: technical window the oracle plans on (the plant itself only stops at 0 and 100%),
# and the window of the fixed industrial margin, the default of the published MILP
TECHNICAL_WINDOW = (0.05, 0.95)
FIXED_WINDOW = (0.10, 0.90)
MARGIN_QUANTILE = 0.95
EDGE_BANDS_PCT = ((10.0, 20.0), (80.0, 90.0))   # bands next to the bottom and top of the library
MIN_WINDOW_WIDTH = 0.30
PENALTY_K = 1.5                     # same as penalty_k of the MILP
MISS_TOL_MWH = 1e-6

STRATEGIES = ("oracle", "oracle_fixed", "naive", "fixed", "quantile", "band")


@dataclass(frozen=True)
class Config:
    """One operating configuration: what the operator sees and which SOC window it plans on."""

    name: str
    strategy: str
    model: str            # estimator, "true" for the oracle strategies
    window: tuple         # SOC window of the MILP, fractions
    error_mode: str = "generator"   # "generator", "replay" or "none"


def margins(library, model, quantile=MARGIN_QUANTILE):
    """
    SOC margins [fractions] at the bottom and at the top of the window for one estimator.
    quantile: from the error over the whole library; band: from the error next to the edges only.
    Bottom margin guards against overestimation (positive error), top margin against
    underestimation (negative error).
    """
    soc = np.asarray(library["soc_true"], float)
    err = np.asarray(library[f"err_{model}"], float)
    lo_band = (soc >= EDGE_BANDS_PCT[0][0]) & (soc < EDGE_BANDS_PCT[0][1])
    hi_band = (soc >= EDGE_BANDS_PCT[1][0]) & (soc < EDGE_BANDS_PCT[1][1])
    q = lambda x, p: max(float(np.quantile(x, p)), 0.0) / 100.0
    return {"quantile": (q(err, quantile), q(-err, quantile)),
            "band": (q(err[lo_band], quantile), q(-err[hi_band], quantile))}


def shrink(window, bottom, top):
    lo, hi = window[0] + bottom, window[1] - top
    if hi - lo < MIN_WINDOW_WIDTH:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * MIN_WINDOW_WIDTH, mid + 0.5 * MIN_WINDOW_WIDTH
    return (round(lo, 4), round(hi, 4))


def build_configs(library, models, replay_models=("EKF",)):
    configs = [Config("oracle", "oracle", "true", TECHNICAL_WINDOW, "none"),
               Config("oracle_fixed", "oracle_fixed", "true", FIXED_WINDOW, "none")]
    for model in models:
        m = margins(library, model)
        windows = {"naive": TECHNICAL_WINDOW, "fixed": FIXED_WINDOW,
                   "quantile": shrink(TECHNICAL_WINDOW, *m["quantile"]),
                   "band": shrink(TECHNICAL_WINDOW, *m["band"])}
        for strategy, window in windows.items():
            configs.append(Config(f"{model}_{strategy}", strategy, model, window))
        if model in replay_models:
            for strategy, window in windows.items():
                configs.append(Config(f"{model}-replay_{strategy}", strategy, model, window, "replay"))
    return configs


SWEEP_MARGINS_PTS = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)


def build_sweep_configs(models, margins_pts=SWEEP_MARGINS_PTS):
    """
    Symmetric SOC margin added to both ends of the technical window, for the true SOC ("true")
    and for each estimator in `models`; margin 0 is the naive strategy, margin 5 the fixed window.
    """
    configs = []
    for model in ("true",) + tuple(models):
        for m in margins_pts:
            window = (round(TECHNICAL_WINDOW[0] + m / 100.0, 4), round(TECHNICAL_WINDOW[1] - m / 100.0, 4))
            configs.append(Config(f"{model}_m{m:g}", "sweep", model, window,
                                  "none" if model == "true" else "generator"))
    return configs


class SeenSOC:
    """SOC the operator believes; hour 0 is drawn once and shared by the MILP and the execution."""

    def __init__(self, error):
        self.error = error
        self.first = None

    def start_day(self, state):
        self.first = self._draw(state)
        return self.first

    def _draw(self, state):
        if self.error is None:
            return state.soc
        return float(np.clip(state.soc + self.error.next(100.0 * state.soc) / 100.0, 0.0, 1.0))

    def __call__(self, state, hour):
        return self.first if hour == 0 else self._draw(state)


def unit_fleet(fleet_template, window):
    """Single-unit fleet whose MILP plans on `window`."""
    from marl_milp_continuous import MultiBatteryParameters
    source = fleet_template.batteries[0]
    if dataclasses.is_dataclass(source):
        battery = dataclasses.replace(source, soc_min=window[0], soc_max=window[1])
    else:
        battery = copy.copy(source)
        battery.soc_min, battery.soc_max = window
    return MultiBatteryParameters([battery])


def service_enum():
    from flexibility_market import ServiceType
    return {"fcr": ServiceType.FCR, "afrr_up": ServiceType.AFRR_UP, "afrr_dn": ServiceType.AFRR_DN,
            "mfrr_up": ServiceType.MFRR_UP, "mfrr_dn": ServiceType.MFRR_DN}


def settle_day(sched, real, res, dev, prices, services, pack, deg_rate, enum):
    """Hour-by-hour settlement of one executed day; returns a dict of daily totals."""
    n = SECONDS_PER_HOUR
    out = dict(capacity=0.0, forfeited=0.0, activation=0.0, energy=0.0, penalty=0.0, degradation=0.0,
               reserve_awarded_h=0, reserve_failed_h=0, reserve_miss_mwh=0.0, miss_mwh=0.0,
               refused_s=int(res.refused.sum()))
    delivered = np.where(res.refused, 0.0, res.power_mw)
    for h in range(24):
        sl = slice(h * n, (h + 1) * n)
        by_type = {s.service_type: s for s in services[h]}

        def price(key, attr):
            #Lorenzo Giannuzzo: MSD price of the hour in chronological mode, otherwise the realised price
            # of the market window when it carries one, as in the MILP results
            realised = getattr(real, "energy_price", None)
            if attr == "energy_price" and realised and key in realised:
                p = realised[key][h]
                if np.isfinite(p):
                    return float(p)
            s = by_type.get(enum[key])
            if s is None:
                return 0.0
            true = getattr(s, "true_" + attr, None)
            return float(true if true is not None else getattr(s, attr))
        p_req, p_del, ref = res.power_mw[sl], delivered[sl], res.refused[sl]

        #Lorenzo Giannuzzo: requested energy of each component in each direction (charging = down)
        base = np.full(n, sched.base_mw[h] + res.intraday_mw[h])
        fcr = real.fcr_band_mw[h] * fcr_response(np.asarray(dev[sl], float))
        act = {"dn": real.act_down_mw[h], "up": real.act_up_mw[h]}
        e_dir = lambda x, d: float(np.sum(np.maximum(x if d == "dn" else -x, 0.0))) / n
        miss = {"dn": float(np.sum(np.where(ref & (p_req > 0), p_req, 0.0))) / n,
                "up": float(np.sum(np.where(ref & (p_req < 0), -p_req, 0.0))) / n}
        act_miss, fcr_miss, failed_dir = {}, {}, {}
        for d in ("dn", "up"):
            e_base, e_fcr, e_act = e_dir(base, d), e_dir(fcr, d), act[d]
            total = e_base + e_fcr + e_act
            reserve = e_fcr + e_act
            m_res = miss[d] * reserve / total if total > 0 else 0.0
            act_miss[d] = m_res * e_act / reserve if reserve > 0 else 0.0
            fcr_miss[d] = m_res - act_miss[d]
            failed_dir[d] = m_res > MISS_TOL_MWH
            out["reserve_miss_mwh"] += m_res
            out["miss_mwh"] += miss[d]

        for key in SERVICE_KEYS:
            if not real.awarded[key][h]:
                continue
            band = sched.band_mw[key][h]
            cap = band * price(key, "capacity_price")
            if key == "fcr":
                failed = failed_dir["dn"] or failed_dir["up"]
                out["penalty"] += PENALTY_K * prices[h] * (fcr_miss["dn"] + fcr_miss["up"])
            else:
                d = "up" if key in UPWARD else "dn"
                e_req = sched.activation_mw[key][h] * real.share[key][h]
                share = e_req / act[d] if act[d] > 0 else 0.0
                e_miss = act_miss[d] * share
                failed = real.called[key][h] and e_miss > MISS_TOL_MWH
                sign = 1.0 if d == "up" else -1.0
                out["activation"] += sign * price(key, "energy_price") * (e_req - e_miss)
                out["penalty"] += PENALTY_K * price(key, "energy_price") * e_miss
            out["reserve_awarded_h"] += 1
            out["reserve_failed_h"] += int(failed)
            out["forfeited" if failed else "capacity"] += cap

        #Lorenzo Giannuzzo: energy outside the activations is settled at the PUN
        act_del = (act["dn"] - act_miss["dn"]) - (act["up"] - act_miss["up"])
        net_charged = float(p_del.sum()) / n
        out["energy"] -= prices[h] * (net_charged - act_del)
        out["degradation"] += deg_rate * float(np.abs(p_del).sum()) / n
    out["net"] = out["capacity"] + out["activation"] + out["energy"] - out["penalty"] - out["degradation"]
    return out


# -------------------------------------------------------------------------------------------
# Worker: everything heavy is loaded once per process
# -------------------------------------------------------------------------------------------

_CTX = {}


def init_worker(library_path):
    from soc2market.bidding import marl_bridge as mb
    from soc2market.errormodel.generator import load_library
    from soc2market.plant.cell import CellParams
    from soc2market.plant.pack import PackSpec
    from soc2market.signals.frequency import FREQ_DIR, FrequencyArchive
    from soc2market.bidding.rolling import make_optimizer  # noqa: F401  (import path check)
    mb.ensure_import_path()
    from aggregation_value import build_portfolio
    from marl_milp_continuous import MultiBatteryParameters
    fleet_all, meta = build_portfolio()
    assert meta[UNIT_INDEX]["label"] == "medium_2h"
    battery = fleet_all.batteries[UNIT_INDEX]
    _CTX.update(
        template=MultiBatteryParameters([battery]),
        deg_rate=battery.degradation_cost_per_mwh / (2.0 * battery.cycle_life),
        pack=PackSpec(power_mw=1.0, energy_mwh=2.0),
        params=CellParams.from_dir(),
        arch=FrequencyArchive.load(FREQ_DIR / "2023.npz"),
        window=mb.market_window(YEAR, DAYS_IN_YEAR),
        library=load_library(library_path),
        enum=service_enum(),
        msd=load_msd() if paths.MSD_MODE == "chronological" else None)


def load_msd():
    from soc2market.bidding import marl_bridge as mb
    from soc2market.signals.msd import MSDArchive
    return MSDArchive.load([mb.MARKET_DIR / name for name in mb.MSD_FILES], YEAR)


def realise(sched, day, rng, msd):
    """Awards and activations of one day in the mode of paths.MSD_MODE."""
    if msd is None:
        return draw_realisation(sched, rng)
    return draw_realisation_chronological(sched, msd.day(day), rng)


def idle_schedule(soc):
    """Fallback for a day whose MILP fails: no market position, plan flat at the current SOC."""
    z = np.zeros(24)
    sched = HourlySchedule(base_mw=z.copy(), soc_plan=np.full(24, soc), net_profit_eur=0.0)
    for key in SERVICE_KEYS:
        for attr in ("band_mw", "activation_mw", "award_prob", "activation_prob"):
            getattr(sched, attr)[key] = z.copy()
    return sched


def run_week(task):
    """task = (config, block, replicate); returns one row per day."""
    from soc2market.plant.cell import CellState
    config, block, rep = task
    ctx = _CTX
    models = sorted({c for c in ctx["library"] if c.startswith("err_")})
    model_idx = models.index(f"err_{config.model}") if config.model != "true" else -1
    #Lorenzo Giannuzzo: streams keyed by block and replicate only, so every configuration shares them
    rng_market = np.random.default_rng([SEED, block.index, rep])
    if config.error_mode == "generator":
        error = ErrorGenerator.from_library(ctx["library"], config.model, week=block.season,
                                            rng=np.random.default_rng([SEED + 3, block.index, rep, model_idx]))
    elif config.error_mode == "replay":
        error = ReplayError(ctx["library"], config.model, block.season)
    else:
        error = None
    seen = SeenSOC(error)
    fleet = unit_fleet(ctx["template"], config.window)
    params, pack = ctx["params"], ctx["pack"]
    state = CellState(soc=INITIAL_SOC, h=0.0, temp_c=params.ambient_c)
    rows = []
    for day in block.days:
        soc_seen0 = seen.start_day(state)
        failed = False
        try:
            sched = solve_day_schedule(fleet, ctx["window"], day, soc_seen0)
        except RuntimeError:
            sched, failed = idle_schedule(soc_seen0), True
        real = realise(sched, day, rng_market, ctx["msd"])
        dev, _ = ctx["arch"].day(day)
        res = execute_day(params, state, pack, sched, real, dev, (0.0, 1.0), soc_seen_fn=seen)
        prices, services = ctx["window"].slice_day(day)
        kpi = settle_day(sched, real, res, dev, prices, services, pack, ctx["deg_rate"], ctx["enum"])
        err_h = 100.0 * (res.soc_seen - res.soc[::SECONDS_PER_HOUR])
        rows.append(dict(config=config.name, strategy=config.strategy, model=config.model,
                         error_mode=config.error_mode, window_lo=config.window[0], window_hi=config.window[1],
                         week=block.label, season=SEASONS[block.season], rep=rep, day=day,
                         planned_eur=sched.net_profit_eur, solver_failed=failed,
                         soc_min=100 * res.soc.min(), soc_max=100 * res.soc.max(),
                         err_mean_pts=float(err_h.mean()), err_absmax_pts=float(np.abs(err_h).max()), **kpi))
        state = res.final_state
    return rows
