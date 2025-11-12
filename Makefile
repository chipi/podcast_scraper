PYTHON ?= python3
PACKAGE = podcast_scraper

.PHONY: help init format format-check lint type security security-bandit security-audit test coverage build ci clean

help:
	@echo "Common developer commands:"
	@echo "  make init            Install development dependencies"
	@echo "  make format          Apply formatting with black + isort"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make lint            Run flake8 linting"
	@echo "  make type            Run mypy type checks"
	@echo "  make security        Run bandit & pip-audit security scans"
	@echo "  make test            Run pytest with coverage"
	@echo "  make build           Build source and wheel distributions"
	@echo "  make ci              Run the full CI suite locally"

init:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install \
		black \
		isort \
		flake8 \
		pytest \
		pytest-cov \
		mypy \
		bandit \
		pip-audit \
		build
	@if [ -f requirements.txt ]; then $(PYTHON) -m pip install -r requirements.txt; fi
	$(PYTHON) -m pip install -e .

format:
	black .
	isort .

format-check:
	black --check .
	isort --check-only .

lint:
	flake8 --config .flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 --config .flake8 . --count --exit-zero --statistics

type:
	mypy --config-file pyproject.toml .

security: security-bandit security-audit

security-bandit:
	bandit -r . --exclude ./.venv

security-audit:
	$(PYTHON) -m pip install --upgrade setuptools
	pip-audit --skip-editable

test:
	pytest --cov=$(PACKAGE) --cov-report=term-missing

coverage: test

build:
	$(PYTHON) -m build

ci: format-check lint type security test build

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache .coverage
