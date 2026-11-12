# Common tasks. Run `make help` to list them.
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-gpu lint format typecheck test check figures robustness clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with the CPU tools (pipeline, reports, app, dev)
	$(PY) -m pip install -e ".[pipeline,reports,explain,app,dev]"
	pre-commit install

install-gpu: ## Install everything, including PyTorch and Transformers
	$(PY) -m pip install -e ".[all,dev]"

lint: ## Lint with ruff
	ruff check .
	ruff format --check .

format: ## Format and auto-fix with ruff
	ruff check --fix .
	ruff format .

typecheck: ## Type-check the package with mypy
	mypy

test: ## Run the unit tests with coverage
	pytest --cov

check: lint typecheck test ## Run every check that CI runs

figures: ## Rebuild figures/ from results/ (no GPU)
	$(PY) scripts/make_figures.py

robustness: ## Re-run the robustness checks on results/ (no GPU)
	$(PY) scripts/robustness_checks.py

clean: ## Remove caches and build artifacts
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
