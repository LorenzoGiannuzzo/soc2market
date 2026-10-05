# Provenance of external/marl_bess

Source repository: MARL/BESS repository
Source branch: <branch name>
Source commit: <paste revision number here>
Copied on: 2026-10-05
Uncommitted changes in source at copy time: no

## Contents
- marl_milp_continuous.py: multi-BESS MILP, used here with a single unit (N=1)
- italian_market_data.py: GME day-ahead (PUN) and Terna MSD loaders, service catalogue
- market_constants.py: sustain durations (SO GL compliant baseline) and regulatory basis
- aggregation_value.py: aggregation study; its standalone arm is the reference single-unit run
- milp_optimizer.py: BatteryParameters
- flexibility_market.py: FlexibilityService, ServiceType, DIRECTIONAL_SERVICES_DN
- fleet_coupling.py: shared connection limit (uncoupled by default)
- degradation_model.py: physics-based LFP ageing (imported only if nonlinear degradation is enabled)
- italian_market_config.py: imported by milp_optimizer

## Corrections already present in this copy (verified on 2026-10-05)
- Downward activation energy enters the objective with a negative sign (DIRECTIONAL_SERVICES_DN)
- MSD column mapping: Prezzo Medio di Vendita -> upward products, Prezzo Medio di Acquisto -> downward products
- Activation energy enters the SOC balance in expectation (p_activation * p_award * A), consistent with the objective

## Market data used (not versioned, stored in data/market/)
- 20230101_20231231_PUN.xlsx, 20240101_20241231_PUN.xlsx
- EsitiMSD_PrezziExAnte_20230101_20230401_Nord.xlsx
- EsitiMSD_PrezziExAnte_20230401_20240331_Nord.xlsx
- EsitiMSD_PrezziExAnte_20240401_20241231_Nord.xlsx

## Environment of the source repository
Python 3.12.2, numpy 1.26.4, pandas 2.2.0, scipy 1.13.1, openpyxl 3.1.5, pulp 3.3.0, gurobipy 13.0.2

## Notes
- MultiBESSMILPOptimizer uses SOLVER_GAP_REL = 0.01 and SOLVER_TIME_LIMIT_S = 300.
  Tighter settings for this paper are applied in a subclass in src/soc2market/bidding, not here.

## Local modifications
None.