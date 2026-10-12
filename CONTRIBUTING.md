# Contributing

Thank you for your interest in `soc2market`. This repository accompanies a research paper, so its
first goal is that the published results stay reproducible; contributions are welcome as long as
they keep that property.

## Reporting a problem

Open a GitHub issue with:

1. the script or test that fails and the full traceback;
2. your operating system, Python version and the output of `pip freeze`;
3. which inputs you have (public data only, or also private artifacts).

## Proposing a change

1. Fork the repository and create a branch from `main`.
2. Install the development environment with `pip install -e ".[solver,dev]"`.
3. Add or update tests in `tests/`; tests that need data not available to everyone must skip
   with a message that lists the missing paths.
4. Run `pytest` and make sure that no previously passing test fails.
5. Open a pull request that explains what changes and why, and whether any result of the paper
   would change.

## Conventions

- Code, comments and documentation in American English.
- Units in variable names or docstrings (`_mw`, `_mwh`, `_s`, `€/MW/yr`); SOC in percent unless
  stated otherwise; positive current and power charge the battery.
- Files in `external/` are frozen: adapt their behavior through the bridges in `src/soc2market/`.
- Scripts in `scripts/` read their settings from constants at the top of the file and write only
  to `results/`.
- Figures follow `src/soc2market/figures/style.py`: units in square brackets, capitalized words,
  boxed legend centered at the top.
