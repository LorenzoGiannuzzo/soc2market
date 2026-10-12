# Changelog

All notable changes to this project are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-12

First public release, accompanying the submission to the 2027 IEEE PES General Meeting.

### Added
- Equivalent-circuit plant of the LFP cell with hysteresis, slow diffusion branch and thermal
  model, identification and validation scripts.
- Daily MILP scheduling with a fresh optimizer per day and feasibility checks of every solution.
- Chronological realisation of aFRR and mFRR calls from the 2023 MSD ex-ante history of the
  northern zone, FCR response to the measured grid frequency, hourly intraday correction.
- Closed-loop runs of seven pre-trained SOC estimators on four reference weeks.
- Hourly k-nearest-neighbor conditional bootstrap of the estimation error, with a replay check.
- Full-year market study, penalty and imbalance sensitivity, SOC margin sweep and the figures of
  the paper.
