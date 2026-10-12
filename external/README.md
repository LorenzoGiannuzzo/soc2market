# External code

Frozen copies of code developed outside this repository. These folders are imported as they are
and are never edited: any adaptation lives in `src/soc2market/` (bridges, wrappers and checks),
and regression tests verify that the frozen code still reproduces its reference outputs.

| Folder | Content | Public |
|---|---|---|
| `marl_bess/` | the authors' market model and daily MILP for battery fleets on the Italian markets (`italian_market_data`, `marl_milp_continuous`, `market_constants`, `fleet_coupling`, `aggregation_value`, ...) | yes |
| `soc_estimators/` | original pipeline of the pre-trained SOC estimators (EKF and hybrid models) | no, third-party code, excluded by `.gitignore` |

The modules of both folders import each other with flat imports; the bridges in
`src/soc2market/bidding/marl_bridge.py` and `src/soc2market/estimation/estimator_bridge.py` put
the right folder on `sys.path` before importing them.

Known behavior of the frozen MILP: a `MultiBESSMILPOptimizer` reused across calls returns, from
its second call onwards, a solution of the first day's constraints. The rolling scheduler in
`src/soc2market/bidding/rolling.py` therefore builds a fresh optimizer every day and checks every
daily solution against its own constraints (see scripts `00` to `02` and
`tests/test_marl_regression.py`).
