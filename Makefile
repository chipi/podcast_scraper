PYTHON ?= python3
PACKAGE = podcast_scraper

.PHONY: help init init-no-ml format format-check lint lint-markdown type security security-bandit security-audit test test-unit test-unit-no-ml test-integration test-ci test-workflow-e2e test-all test-parallel test-reruns coverage docs build ci ci-fast clean clean-cache clean-all docker-build docker-test docker-clean install-hooks

help:
	@echo "Common developer commands:"
	@echo "  make init            Install development dependencies"
	@echo "  make format          Apply formatting with black + isort"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make lint            Run flake8 linting"
	@echo "  make lint-markdown   Run markdownlint on markdown files"
	@echo "  make type            Run mypy type checks"
	@echo "  make security        Run bandit & pip-audit security scans"
	@echo ""
	@echo "Test commands:"
	@echo "  make test            Run pytest with coverage (default: unit tests only)"
	@echo "  make test-unit       Run unit tests only"
	@echo "  make test-unit-no-ml Run unit tests without ML dependencies (matches CI)"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-workflow-e2e Run workflow E2E tests only"
	@echo "  make test-all        Run all tests (unit + integration + workflow_e2e)"
	@echo "  make test-network    Run network tests only (requires internet connection)"
	@echo "  make test-parallel   Run tests with parallel execution (-n auto)"
	@echo "  make test-reruns     Run tests with reruns for flaky tests (2 retries, 1s delay)"
	@echo ""
	@echo "Other commands:"
	@echo "  make docs            Build MkDocs site (strict mode, outputs to .build/site/)"
	@echo "  make build           Build source and wheel distributions (outputs to .build/dist/)"
	@echo "  make ci              Run the full CI suite locally (cleans cache, unit + integration tests, excludes workflow_e2e)"
	@echo "  make ci-fast         Run fast CI checks (unit tests only, faster feedback)"
	@echo "  make docker-build    Build Docker image"
	@echo "  make docker-test     Build and test Docker image"
	@echo "  make docker-clean    Remove Docker test images"
	@echo "  make install-hooks   Install git pre-commit hook for automatic linting"
	@echo "  make clean           Remove build artifacts (.build/, .mypy_cache/, .pytest_cache/)"
	@echo "  make clean-cache     Remove ML model caches (Whisper, spaCy) to test network isolation"
	@echo "  make clean-all       Remove both build artifacts and ML model caches"

init:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e .[dev,ml]
	@if [ -f docs/requirements.txt ]; then $(PYTHON) -m pip install -r docs/requirements.txt; fi

init-no-ml:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e .[dev]

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
	markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site --config .markdownlint.json

type:
	mypy --config-file pyproject.toml .

security: security-bandit security-audit

security-bandit:
	bandit -r . --exclude ./.venv --skip B113,B108,B110,B310 --severity-level medium

security-audit:
	$(PYTHON) -m pip install --upgrade setuptools
	# Install ML dependencies to ensure they are audited
	# This ensures production dependencies like torch, transformers, spacy, openai-whisper are audited
	$(PYTHON) -m pip install --quiet -e .[ml] || \
		(echo "⚠️  Editable install failed, using non-editable install" && \
		 $(PYTHON) -m pip install --quiet .[ml])
	# Audit all installed packages (including ML dependencies from pyproject.toml)
	pip-audit --skip-editable

docs:
	mkdocs build --strict

test:
	pytest --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

test-unit:
	pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

test-unit-no-ml: init-no-ml
	@echo "Running unit tests without ML dependencies (matches CI test-unit job)..."
	@echo "This verifies that unit tests work when ML dependencies (spacy, torch) are not installed."
	@echo ""
	@echo "Step 1: Checking if modules can be imported without ML dependencies..."
	@$(PYTHON) scripts/check_unit_test_imports.py
	@echo ""
	@echo "Step 2: Running unit tests..."
	pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

test-integration:
	pytest tests/integration/ -m integration

test-ci:
	pytest --cov=$(PACKAGE) --cov-report=term-missing -m 'not workflow_e2e and not network'

test-workflow-e2e:
	pytest tests/workflow_e2e/ -m workflow_e2e

test-all:
	pytest tests/ -m "not network" --cov=$(PACKAGE) --cov-report=term-missing

test-network:
	pytest tests/ -m network

test-parallel:
	pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

test-reruns:
	pytest --reruns 2 --reruns-delay 1 --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not workflow_e2e and not network'

coverage: test

build:
	$(PYTHON) -m pip install --quiet build
	$(PYTHON) -m build
	@if [ -d dist ]; then mkdir -p .build && rm -rf .build/dist && mv dist .build/ && echo "Moved dist to .build/dist/"; fi

ci: clean-all format-check lint lint-markdown type security test-ci docs build

ci-fast: format-check lint lint-markdown type security test docs build

docker-build:
	docker build -t podcast-scraper:test -f docker/Dockerfile .

docker-test: docker-build
	@echo "Running Docker smoke tests..."
	@echo "Test 1: --help command"
	@docker run --rm podcast-scraper:test --help > /dev/null
	@echo "Test 2: --version command"
	@docker run --rm podcast-scraper:test --version
	@echo "Test 3: No args (should error)"
	@docker run --rm podcast-scraper:test 2>&1 | grep -q "required" && echo "[OK] Error handling works"
	@echo "Test 4: Building with multiple Whisper models..."
	@docker build --quiet --build-arg WHISPER_PRELOAD_MODELS="tiny.en,base.en" -t podcast-scraper:multi-model -f docker/Dockerfile . > /dev/null
	@docker run --rm podcast-scraper:multi-model --help > /dev/null
	@echo "[OK] All Docker tests passed"

docker-clean:
	docker rmi podcast-scraper:test podcast-scraper:multi-model 2>/dev/null || true

install-hooks:
	@if [ ! -d .git ]; then echo "Error: Not a git repository"; exit 1; fi
	@echo "Installing git pre-commit hook..."
	@cp .github/hooks/pre-commit .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✓ Pre-commit hook installed!"
	@echo ""
	@echo "The hook will automatically run before each commit to check:"
	@echo "  • Black & isort formatting"
	@echo "  • flake8 linting"
	@echo "  • markdownlint (if installed)"
	@echo "  • mypy type checking"
	@echo ""
	@echo "To skip the hook for a specific commit, use: git commit --no-verify"

clean:
	rm -rf build .build .mypy_cache .pytest_cache

clean-cache:
	@echo "Cleaning ML model caches..."
	@if [ -d "$$HOME/.cache/whisper" ]; then \
		echo "  Removing Whisper cache: $$HOME/.cache/whisper"; \
		rm -rf "$$HOME/.cache/whisper"; \
	fi
	@if [ -d "$$HOME/.cache/spacy" ]; then \
		echo "  Removing spaCy cache: $$HOME/.cache/spacy"; \
		rm -rf "$$HOME/.cache/spacy"; \
	fi
	@if [ -d "$$HOME/.cache/huggingface" ]; then \
		echo "  Removing HuggingFace cache: $$HOME/.cache/huggingface"; \
		rm -rf "$$HOME/.cache/huggingface"; \
	fi
	@if [ -d "$$HOME/.cache/torch" ]; then \
		echo "  Removing PyTorch cache: $$HOME/.cache/torch"; \
		rm -rf "$$HOME/.cache/torch"; \
	fi
	@echo "Cache cleaning complete. Run 'make test-unit' to verify network isolation."

clean-all: clean clean-cache
	@echo "All cleaning complete."
