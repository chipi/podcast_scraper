PYTHON ?= python3
PACKAGE = podcast_scraper

.PHONY: help init format format-check lint lint-markdown type security security-bandit security-audit test coverage docs build ci clean

help:
	@echo "Common developer commands:"
	@echo "  make init            Install development dependencies"
	@echo "  make format          Apply formatting with black + isort"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make lint            Run flake8 linting"
	@echo "  make lint-markdown   Run markdownlint on markdown files"
	@echo "  make type            Run mypy type checks"
	@echo "  make security        Run bandit & pip-audit security scans"
	@echo "  make test            Run pytest with coverage"
	@echo "  make docs            Build MkDocs site (strict mode)"
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
	@if [ -f docs/requirements.txt ]; then $(PYTHON) -m pip install -r docs/requirements.txt; fi
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

lint-markdown:
	@command -v markdownlint >/dev/null 2>&1 || { echo "markdownlint not found. Install with: npm install -g markdownlint-cli"; exit 1; }
	markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore site

type:
	mypy --config-file pyproject.toml .

security: security-bandit security-audit

security-bandit:
	bandit -r . --exclude ./.venv

security-audit:
	$(PYTHON) -m pip install --upgrade setuptools
	pip-audit --skip-editable

docs:
	mkdocs build --strict

test:
	pytest --cov=$(PACKAGE) --cov-report=term-missing

coverage: test

build:
	$(PYTHON) -m build

ci: format-check lint lint-markdown type security test docs build

clean:
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache .coverage site
