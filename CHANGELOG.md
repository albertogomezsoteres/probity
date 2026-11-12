# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [1.1.0] - 2026-11-12

Engineering release: the results are unchanged.

### Added
- `probity` command-line tool: `data`, `extract`, `train`, `run`, `check`, `audit`, `app`, `figures`,
  `robustness` and `info`.
- A reproducible pipeline in the package (data with checksums and fact-grouped splits, hidden states, probe
  training and calibration, evaluation with bootstrap intervals) that ends by comparing its numbers with the
  published ones.
- Typed configuration in `configs/default.yaml` with command-line overrides.
- Run records (config, metrics, log, git commit, data version, software and hardware) and optional Weights & Biases.
- The app as a package module (`probity.app`) with ZeroGPU support, a Hugging Face Space bundle and a deploy workflow.
- Dockerfile, `uv.lock`, Makefile, pre-commit hooks, ruff and mypy.
- CI with lint, type checks and tests on Python 3.10 to 3.12; tests for the data stage, the probe maths, the
  metrics, the reports and the whole app on a fake model.
- Pipeline documentation, model card, data card, citation file and this changelog.

### Changed
- `scripts/make_figures.py` and `scripts/robustness_checks.py` became `probity figures` and
  `probity robustness` (identical outputs); `scripts/check_claims.py` and `scripts/audit_log.py` became
  `probity check` and `probity audit`.
- Dependencies moved from `requirements.txt` to `pyproject.toml` extras.

### Fixed
- The developer docs of the app used backslashes inside f-string expressions, a syntax error before Python 3.12,
  and showed doubled braces in a code sample.

## [1.0.0] - 2026-10-19

First release: notebooks 01 to 06 with saved outputs, the Gradio app (notebook 04), the `probity` package with
`Polygraph` and Polygraph Guard, figures, poster and documentation.
