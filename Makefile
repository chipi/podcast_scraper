PYTHON ?= python3
PACKAGE = podcast_scraper

.PHONY: help init init-no-ml format format-check lint lint-markdown type security security-bandit security-audit test-unit test-unit-sequential test-unit-no-ml test-integration test-integration-sequential test-integration-fast test-integration-slow test-ci test-ci-fast test-e2e test-e2e-sequential test-e2e-fast test-e2e-slow test-all test-all-sequential test-all-fast test-all-slow test-reruns coverage docs build ci ci-fast ci-full clean clean-cache clean-all docker-build docker-test docker-clean install-hooks

help:
	@echo "Common developer commands:"
	@echo "  make init            Install development dependencies"
	@echo "  make format          Apply formatting with black + isort"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make lint            Run flake8 linting"
	@echo "  make lint-markdown   Run markdownlint on markdown files"
	@echo "  make fix-md          Auto-fix common markdown linting issues"
	@echo "  make type            Run mypy type checks"
	@echo "  make security        Run bandit & pip-audit security scans"
	@echo ""
	@echo "Test commands:"
	@echo "  make test-unit            Run unit tests with coverage in parallel (default, matches CI)"
	@echo "  make test-unit-sequential Run unit tests sequentially (for debugging, slower but clearer output)"
	@echo "  make test-unit-no-ml Run unit tests without ML dependencies (matches CI)"
	@echo "  make test-integration            Run integration tests (parallel - 3.4x faster)"
	@echo "  make test-integration-sequential Run integration tests sequentially (for debugging)"
	@echo "  make test-integration-fast       Run fast integration tests only (excludes slow/ml_models)"
	@echo "  make test-integration-slow       Run slow integration tests only (includes slow/ml_models, requires ML deps)"
	@echo "  make test-e2e                   Run all E2E tests (parallel, with network guard)"
	@echo "  make test-e2e-sequential         Run all E2E tests sequentially (for debugging)"
	@echo "  make test-e2e-fast              Run fast E2E tests only (excludes slow/ml_models)"
	@echo "  make test-e2e-slow              Run slow E2E tests only (includes slow/ml_models, requires ML deps)"
	@echo "  make test-all            Run all tests (unit + integration + e2e, parallel)"
	@echo "  make test-all-sequential Run all tests sequentially (for debugging)"
	@echo "  make test-all-fast       Run fast tests (unit + fast integration + fast e2e, excludes slow/ml_models)"
	@echo "  make test-all-slow       Run slow tests (slow integration + slow e2e, includes slow/ml_models, requires ML deps)"
	@echo "  make test-reruns     Run tests with reruns for flaky tests (2 retries, 1s delay)"
	@echo ""
	@echo "Other commands:"
	@echo "  make docs            Build MkDocs site (strict mode, outputs to .build/site/)"
	@echo "  make build           Build source and wheel distributions (outputs to .build/dist/)"
	@echo "  make ci              Run the full CI suite locally (unit + integration + e2e-fast tests)"
	@echo "  make ci-fast         Run fast CI checks (unit + fast integration tests, faster feedback)"
	@echo "  make ci-full         Run complete CI suite with all tests (unit + integration + e2e, slower)"
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

fix-md:
	@echo "Fixing common markdown linting issues..."
	@python scripts/fix_markdown.py
	@echo "✓ Markdown fixes applied. Run 'make lint-markdown' to verify."

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

test-unit:
	# Unit tests: parallel execution for faster feedback
	pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n auto

test-unit-sequential:
	# Unit tests: sequential execution (slower but clearer output, useful for debugging)
	pytest --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

test-unit-no-ml: init-no-ml
	@echo "Running unit tests without ML dependencies (matches CI test-unit job)..."
	@echo "This verifies that unit tests work when ML dependencies (spacy, torch) are not installed."
	@echo ""
	@echo "Step 1: Checking if modules can be imported without ML dependencies..."
	@$(PYTHON) scripts/check_unit_test_imports.py
	@echo ""
	@echo "Step 2: Running unit tests..."
	pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

test-integration:
	# Integration tests: parallel execution (3.4x faster, significant benefit)
	# Includes reruns for flaky tests (matches CI behavior)
	pytest tests/integration/ -m integration -n auto --reruns 2 --reruns-delay 1

test-integration-sequential:
	# Integration tests: sequential execution (slower but clearer output, useful for debugging)
	pytest tests/integration/ -m integration

test-integration-fast:
	# Fast integration tests: excludes slow/ml_models tests (faster CI feedback)
	pytest tests/integration/ -m "integration and not slow and not ml_models" -n auto

test-integration-slow:
	# Slow integration tests: includes slow/ml_models tests (requires ML dependencies)
	pytest tests/integration/ -m "integration and (slow or ml_models)" -n auto

test-ci:
	# CI test suite: parallel execution (matches CI behavior, excludes slow/ml_models for faster PRs)
	# Includes: unit + fast integration + fast e2e (excludes slow/ml_models tests)
	# Note: Slow integration and slow e2e tests run on main branch only
	pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing -m '(not slow and not ml_models)' --disable-socket --allow-hosts=127.0.0.1,localhost

test-ci-fast:
	# Fast CI test suite: parallel execution (faster feedback, no coverage for speed)
	# Includes: unit + fast integration + fast e2e (excludes slow/ml_models tests)
	# Note: Coverage is excluded here for faster execution; full validation job includes unified coverage
	pytest tests/unit/ tests/integration/ tests/e2e/ -n auto -m '(not slow and not ml_models)' --disable-socket --allow-hosts=127.0.0.1,localhost

test-e2e:
	# E2E tests: parallel execution with network guard (faster for slow tests)
	# Includes reruns for flaky tests (matches CI behavior)
	pytest tests/e2e/ -m e2e -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

test-e2e-sequential:
	# E2E tests: sequential execution (slower but clearer output, useful for debugging)
	pytest tests/e2e/ -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

test-e2e-fast:
	# Fast E2E tests: excludes slow/ml_models tests (faster CI feedback)
	# Includes reruns for flaky tests (matches CI behavior)
	pytest tests/e2e/ -m "e2e and not slow and not ml_models" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

test-e2e-slow:
	# Slow E2E tests: includes slow/ml_models tests (requires ML dependencies)
	# Includes reruns for flaky tests (matches CI behavior)
	pytest tests/e2e/ -m "e2e and (slow or ml_models)" -n auto --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1

test-all:
	# All tests: parallel execution (includes unit + integration + e2e, all slow/fast variants)
	pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing -n auto --disable-socket --allow-hosts=127.0.0.1,localhost

test-all-sequential:
	# All tests: sequential execution (slower but clearer output, useful for debugging)
	pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing

test-all-fast:
	# Fast tests: unit + fast integration + fast e2e (excludes slow/ml_models tests)
	pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing -m '(not slow and not ml_models)' --disable-socket --allow-hosts=127.0.0.1,localhost

test-all-slow:
	# Slow tests: slow integration + slow e2e (includes slow/ml_models tests, requires ML dependencies)
	pytest -n auto --cov=$(PACKAGE) --cov-report=term-missing -m '((integration or e2e) and (slow or ml_models))' --disable-socket --allow-hosts=127.0.0.1,localhost

test-reruns:
	pytest --reruns 2 --reruns-delay 1 --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e'

coverage: test-unit

build:
	$(PYTHON) -m pip install --quiet build
	$(PYTHON) -m build
	@if [ -d dist ]; then mkdir -p .build && rm -rf .build/dist && mv dist .build/ && echo "Moved dist to .build/dist/"; fi

ci: format-check lint lint-markdown type security test-ci docs build

ci-fast: format-check lint lint-markdown type security test-ci-fast docs build

ci-full: clean-all format-check lint lint-markdown type security test-all docs build

docker-build:
	docker build -t podcast-scraper:test -f Dockerfile .

docker-test: docker-build
	@echo "Running Docker smoke tests..."
	@echo "Test 1: --help command"
	@docker run --rm podcast-scraper:test --help > /dev/null
	@echo "Test 2: --version command"
	@docker run --rm podcast-scraper:test --version
	@echo "Test 3: No args (should error)"
	@docker run --rm podcast-scraper:test 2>&1 | grep -q "required" && echo "[OK] Error handling works"
	@echo "Test 4: Building with multiple Whisper models..."
	@docker build --quiet --build-arg WHISPER_PRELOAD_MODELS="tiny.en,base.en" -t podcast-scraper:multi-model -f Dockerfile . > /dev/null
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
	# Coverage files: .coverage.* are created during parallel test execution (pytest -n auto)
	rm -f .coverage .coverage.*

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
