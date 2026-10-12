# Data

Nothing in this folder is tracked by git. The pipeline reads its inputs from the paths below;
file names are fixed in the code and are never discovered by globbing, so that partial or
overlapping files cannot enter a run.

## Public inputs

### `data/market/`: day-ahead prices and MSD outcomes

| File | Content | Source |
|---|---|---|
| `20230101_20231231_PUN.xlsx` | hourly single national price (PUN) of 2023 | GME, results of the day-ahead market (MGP), [mercatoelettrico.org](https://www.mercatoelettrico.org) |
| `20240101_20241231_PUN.xlsx` | same for 2024 (needed only by the regression tests of the MILP) | GME |
| `EsitiMSD_PrezziExAnte_20230101_20230401_Nord.xlsx` | MSD ex-ante results, northern zone | Terna, MSD ex-ante results, [terna.it](https://www.terna.it) |
| `EsitiMSD_PrezziExAnte_20230401_20240331_Nord.xlsx` | same, following period | Terna |
| `EsitiMSD_PrezziExAnte_20240401_20241231_Nord.xlsx` | same, following period | Terna |

All three MSD workbooks are read together whatever the simulated year, so all of them must be
present. They must contain the hourly columns `Prezzo Medio di Vendita (€/MWh)` (upward
regulation) and `Prezzo Medio di Acquisto (€/MWh)` (downward regulation), as exported by Terna.
An hour is considered active in a direction when the corresponding average price is positive.

### `data/frequency/`: grid frequency of Continental Europe, 1 s

1. Download the pre-processed frequency data of Nikoltchovska et al. from Zenodo,
   [doi:10.5281/zenodo.15784548](https://doi.org/10.5281/zenodo.15784548), and take the 2023
   series of the `Data_cleansed` folder (header-less csv, local timestamp, frequency in Hz).
2. Save it as `data/frequency/continental_europe_2023.csv`.
3. Run `scripts/09_convert_frequency.py`, which writes `data/frequency/2023.npz` (one row per day,
   86,400 deviations from 50 Hz, NaN where no sample exists) and prints a quality report.

The dataset is subject to the license stated on its Zenodo record and is not redistributed here.

## Private inputs (not available)

The laboratory data of the cell and the trained estimators belong to a third party and cannot
be shared. They are listed here so that the structure of the pipeline is fully documented.

```
data/private/
├── plant/                  parameter tables written by scripts/03_identify_plant.py
│   ├── ocv.csv             charge and discharge OCV branches on a common SOC grid
│   ├── ecm_chg.csv         R0, R1, C1, R2, C2 against SOC, charge direction
│   ├── ecm_dch.csv         same, discharge direction
│   └── slow_branch.csv     slow diffusion branch per direction
├── estimator_workdir/      working directory expected by the frozen estimator code
│   ├── data/               HPPC tests at 25 °C and OCV characterization of the cell
│   └── nn_training/models/ trained models and scalers of the hybrid estimators
└── soc_regression/         laboratory test profiles and reference outputs of the estimators
```

Without these files the scripts that need them stop with a message that lists the missing paths,
and the corresponding tests are skipped.
