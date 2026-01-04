PYTHON ?= python3
PACKAGE = podcast_scraper

# Test parallelism: Smart default that adapts to CPU count
# Formula: min(max(1, cpu_count - 2), 8)
# - Reserves 2 cores for system operations
# - Caps at 8 to prevent excessive memory usage
# - Falls back to 2 if CPU detection fails
# Can be overridden: PYTEST_WORKERS=4 make test
PYTEST_WORKERS ?= $(shell python3 -c "import os; print(min(max(1, (os.cpu_count() or 4) - 2), 8))")

.PHONY: help init init-no-ml format format-check lint lint-markdown type security security-bandit security-audit complexity deadcode docstrings spelling quality test-unit test-unit-sequential test-unit-no-ml test-integration test-integration-sequential test-integration-fast test-ci test-ci-fast test-e2e test-e2e-sequential test-e2e-fast test-e2e-data-quality test-nightly test test-sequential test-fast test-reruns coverage coverage-check coverage-check-unit coverage-check-integration coverage-check-e2e coverage-check-combined coverage-report coverage-enforce docs build ci ci-fast ci-sequential ci-clean ci-nightly clean clean-cache clean-all docker-build docker-build-fast docker-build-full docker-test docker-clean install-hooks preload-ml-models preload-ml-models-production

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
	@echo "  make complexity      Run radon complexity analysis"
	@echo "  make deadcode        Run vulture dead code detection"
	@echo "  make docstrings      Run interrogate docstring coverage"
	@echo "  make spelling        Run codespell spell checking"
	@echo "  make quality         Run all code quality checks (complexity, deadcode, docstrings, spelling)"
	@echo ""
	@echo "Test commands:"
	@echo "  make test-unit            Run unit tests with coverage in parallel (default, matches CI)"
	@echo "  make test-unit-sequential Run unit tests sequentially (for debugging, slower but clearer output)"
	@echo "  make test-unit-no-ml Run unit tests without ML dependencies (matches CI)"
	@echo "  make test-integration            Run all integration tests (full suite, parallel)"
	@echo "  make test-integration-sequential Run all integration tests sequentially (for debugging)"
	@echo "  make test-integration-fast       Run fast integration tests (critical path only)"
	@echo "  make test-e2e                   Run all E2E tests (full suite, parallel, 1 episode per test)"
	@echo "  make test-e2e-sequential         Run all E2E tests sequentially (for debugging)"
	@echo "  make test-e2e-fast              Run fast E2E tests (critical path only, 1 episode per test)"
	@echo "  make test-e2e-data-quality      Run data quality E2E tests (multiple episodes, all original mock data, nightly only)"
	@echo "  make test-nightly                Run nightly-only tests (p01-p05 full suite, production models)"
	@echo "  make test                Run all tests (unit + integration + e2e, full suite, uses multi-episode feed)"
	@echo "  make test-sequential     Run all tests sequentially (for debugging, uses multi-episode feed)"
	@echo "  make test-fast           Run fast tests (unit + critical path integration + critical path e2e, uses fast feed)"
	@echo "  make test-reruns     Run tests with reruns for flaky tests (2 retries, 1s delay)"
	@echo "  Tip: For debugging, use pytest directly with -n 0 for sequential execution"
	@echo ""
	@echo "Coverage commands:"
	@echo "  make coverage                Run all tests with coverage (same as make test)"
	@echo "  make coverage-check          Verify all layers meet minimum thresholds"
	@echo "  make coverage-check-unit     Check unit coverage >= $(COVERAGE_THRESHOLD_UNIT)%"
	@echo "  make coverage-check-integration  Check integration coverage >= $(COVERAGE_THRESHOLD_INTEGRATION)%"
	@echo "  make coverage-check-e2e      Check E2E coverage >= $(COVERAGE_THRESHOLD_E2E)%"
	@echo "  make coverage-check-combined Check combined coverage >= $(COVERAGE_THRESHOLD_COMBINED)%"
	@echo "  make coverage-enforce        Fast threshold check on existing .coverage (used by ci)"
	@echo "  make coverage-report         Generate HTML coverage report from existing .coverage"
	@echo ""
	@echo "Other commands:
	@echo "  make docs            Build MkDocs site (strict mode, outputs to .build/site/)"
	@echo "  make build           Build source and wheel distributions (outputs to .build/dist/)"
	@echo "  make ci              Run the full CI suite locally (all tests: unit + integration + e2e, uses multi-episode feed)"
	@echo "  make ci-fast         Run fast CI checks (unit + critical path integration + critical path e2e, uses fast feed)"
	@echo "  make ci-clean        Run complete CI suite with clean first (same as ci but cleans build artifacts first)"
	@echo "  make ci-nightly      Run full nightly CI chain (unit + integration + e2e + nightly, production models)"
	@echo "  make docker-build       Build Docker image (default, with model preloading)"
	@echo "  make docker-build-fast  Build Docker image fast (no model preloading, <5min target)"
	@echo "  make docker-build-full  Build Docker image full (with model preloading, matches main)"
	@echo "  make docker-test        Build and test Docker image"
	@echo "  make docker-clean       Remove Docker test images"
	@echo "  make install-hooks   Install git pre-commit hook for automatic linting"
	@echo "  make clean           Remove build artifacts (.build/, .mypy_cache/, .pytest_cache/)"
	@echo "  make clean-cache     Remove ML model caches (Whisper, spaCy) to test network isolation"
	@echo "  make clean-all       Remove both build artifacts and ML model caches"
	@echo "  make preload-ml-models  Pre-download and cache all required ML models locally (test models)"
	@echo "  make preload-ml-models-production  Pre-download and cache production ML models (for nightly tests)"

init:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -e .[dev,ml]
	@if [ -f docs/requirements.txt ]; then $(PYTHON) -m pip install -r docs/requirements.txt; fi

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
	markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site --ignore "docs/wip/**" --ignore "tests/fixtures/**" --config .markdownlint.json

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

# Code quality analysis (RFC-031)
complexity:
	@echo "=== Cyclomatic Complexity Analysis ==="
	@radon cc src/podcast_scraper/ -a -s --total-average || true
	@echo ""
	@echo "=== Maintainability Index ==="
	@radon mi src/podcast_scraper/ -s || true

deadcode:
	@echo "=== Dead Code Detection ==="
	@vulture src/podcast_scraper/ .vulture_whitelist.py --min-confidence 80 || true

docstrings:
	@echo "=== Docstring Coverage ==="
	@interrogate src/podcast_scraper/ -v || true

spelling:
	@echo "=== Spell Checking ==="
	@codespell src/ docs/ --skip="*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl" || true

quality: complexity deadcode docstrings spelling
	@echo ""
	@echo "✓ All code quality checks completed"
	# This ensures production dependencies like torch, transformers, spacy, openai-whisper are audited
	$(PYTHON) -m pip install --quiet -e .[ml] || \
		(echo "⚠️  Editable install failed, using non-editable install" && \
		 $(PYTHON) -m pip install --quiet .[ml])
	# Audit all installed packages (including ML dependencies from pyproject.toml)
	pip-audit --skip-editable

docs:
	mkdocs build --strict

# Coverage thresholds per layer (minimums to ensure balanced coverage)
# These are ambitious but achievable targets based on current coverage levels
# Combined threshold is enforced in CI; per-layer thresholds ensure no layer is neglected
COVERAGE_THRESHOLD_UNIT := 70          # Current: ~74% local, ~70% CI
COVERAGE_THRESHOLD_INTEGRATION := 40   # Current: ~54% local, ~42% CI
COVERAGE_THRESHOLD_E2E := 40           # Current: ~53% local, ~50% CI
COVERAGE_THRESHOLD_COMBINED := 80      # Current: ~82% local

test-unit:
	# Unit tests: parallel execution for faster feedback
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost

test-integration:
	# Integration tests: parallel execution (3.4x faster, significant benefit)
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Integration tests load ML models which consume ~1-2 GB per worker
	# Includes reruns for flaky tests (matches CI behavior)
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	# Coverage: measured independently (not appended) to match CI per-job measurement
	pytest tests/integration/ -m integration -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --reruns 2 --reruns-delay 1 --disable-socket --allow-hosts=127.0.0.1,localhost

test-integration-fast:
	# Fast integration tests: critical path tests only (includes ML tests if models are cached)
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Includes reruns for flaky tests (matches CI behavior)
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Coverage: measured independently to match CI
	pytest tests/integration/ -m "integration and critical_path" -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 --durations=20

test-ci:
	# CI test suite: serial tests first (sequentially), then parallel execution for the rest
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Note: Non-critical path tests run on main branch only
	pytest -m 'serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' --disable-socket --allow-hosts=127.0.0.1,localhost --cov=$(PACKAGE) --cov-report=term-missing || true
	pytest -m 'not serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --cov=$(PACKAGE) --cov-report=term-missing --cov-append

test-ci-fast:
	# Fast CI test suite: serial tests first (sequentially), then parallel execution for the rest
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Note: Coverage is excluded here for faster execution; full validation job includes unified coverage
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Includes reruns for flaky tests (matches CI behavior) - increased to 3 retries for very flaky tests
	pytest tests/unit/ tests/integration/ tests/e2e/ -m 'serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20 --reruns 3 --reruns-delay 2 || true
	pytest tests/unit/ tests/integration/ tests/e2e/ -m 'not serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20 --reruns 3 --reruns-delay 2

test-e2e:
	# E2E tests: serial tests first (sequentially), then parallel execution for the rest
	# Includes reruns for flaky tests (matches CI behavior) - 3 retries for ML model variability
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	# Coverage: measured independently (serial first, then parallel appends) to match CI
	@E2E_TEST_MODE=multi_episode pytest tests/e2e/ -m "e2e and serial" --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 || true
	@E2E_TEST_MODE=multi_episode pytest tests/e2e/ -m "e2e and not serial" -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --cov-append --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1

test-e2e-sequential:
	# E2E tests: sequential execution (slower but clearer output, useful for debugging)
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	E2E_TEST_MODE=multi_episode pytest tests/e2e/ -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

test-e2e-fast:
	# Fast E2E tests: serial tests first (sequentially), then parallel execution for the rest
	# Critical path tests only (includes ML tests if models are cached)
	# Includes reruns for flaky tests (matches CI behavior) - 3 retries for ML model variability
	# Uses fast feed (1 episode) - set via E2E_TEST_MODE environment variable
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Coverage: measured independently (serial first, then parallel appends) to match CI
	@E2E_TEST_MODE=fast pytest tests/e2e/ -m "e2e and critical_path and serial" --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 --durations=20 || true
	@E2E_TEST_MODE=fast pytest tests/e2e/ -m "e2e and critical_path and not serial" -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --cov-append --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 --durations=20

test-e2e-data-quality:
	# Data quality E2E tests: full pipeline validation with multiple episodes
	# Uses all original mock data (not fast fixtures)
	# Runs with 3-5 episodes per test to validate data quality and consistency
	# For nightly builds only - not part of regular CI/CD code quality checks
	@E2E_TEST_MODE=data_quality pytest tests/e2e/ -m "e2e and data_quality" -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1

test-nightly:
	# Nightly-only tests: comprehensive tests with production ML models (p01-p05 full suite)
	# Uses production models: Whisper base, BART-large-cnn, LED-large-16384
	# Runs all 15 episodes across 5 podcasts (p01-p05)
	# Sequential execution per podcast, parallel episodes within podcast (2 workers)
	# NOT marked with @pytest.mark.e2e - separate category from regular E2E tests
	# Excludes LLM/OpenAI tests to avoid API costs (see issue #183)
	@echo "Running nightly tests with production models..."
	@echo "Podcasts: p01-p05 (15 episodes total)"
	@echo "Models: Whisper base, BART-large-cnn, LED-large-16384"
	@mkdir -p reports
	@E2E_TEST_MODE=nightly pytest tests/e2e/ -m "nightly and not llm" -v -n 2 --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20 --junitxml=reports/junit-nightly.xml --json-report --json-report-file=reports/pytest-nightly.json

test:
	# All tests: serial tests first (sequentially), then parallel execution for the rest
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Excludes nightly tests (run separately via make test-nightly)
	@E2E_TEST_MODE=multi_episode pytest tests/ -m "serial and not nightly" --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost || true
	@E2E_TEST_MODE=multi_episode pytest tests/ -m "not serial and not nightly" --cov=$(PACKAGE) --cov-report=term-missing --cov-append -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost

test-sequential:
	# All tests: sequential execution (slower but clearer output, useful for debugging)
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	E2E_TEST_MODE=multi_episode pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost

test-fast:
	# Fast tests: serial tests first (sequentially), then parallel execution for the rest
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Uses fast feed for E2E tests (1 episode) - set via E2E_TEST_MODE environment variable
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Use --durations=20 to monitor slow tests and optimize them separately
	@E2E_TEST_MODE=fast pytest -m 'serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20 || true
	@E2E_TEST_MODE=fast pytest -m 'not serial and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' --cov=$(PACKAGE) --cov-report=term-missing --cov-append -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20

test-reruns:
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	pytest --reruns 2 --reruns-delay 1 --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' --disable-socket --allow-hosts=127.0.0.1,localhost

coverage: test
	# Runs all tests with coverage (same as 'make test')

coverage-check: coverage-check-unit coverage-check-integration coverage-check-e2e
	# Verify all layers meet minimum coverage thresholds
	@echo "✅ All coverage checks passed!"

coverage-check-unit:
	# Check unit test coverage meets minimum threshold ($(COVERAGE_THRESHOLD_UNIT)%)
	@echo "Checking unit test coverage (minimum $(COVERAGE_THRESHOLD_UNIT)%)..."
	@pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_UNIT) -m 'not integration and not e2e' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost -q

coverage-check-integration:
	# Check integration test coverage meets minimum threshold ($(COVERAGE_THRESHOLD_INTEGRATION)%)
	@echo "Checking integration test coverage (minimum $(COVERAGE_THRESHOLD_INTEGRATION)%)..."
	@pytest tests/integration/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_INTEGRATION) -m 'integration' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 -q

coverage-check-e2e:
	# Check E2E test coverage meets minimum threshold ($(COVERAGE_THRESHOLD_E2E)%)
	@echo "Checking E2E test coverage (minimum $(COVERAGE_THRESHOLD_E2E)%)..."
	@E2E_TEST_MODE=multi_episode pytest tests/e2e/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_E2E) -m 'e2e and not nightly' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 -q

coverage-check-combined:
	# Check combined coverage meets threshold ($(COVERAGE_THRESHOLD_COMBINED)%)
	# This runs all tests and enforces the combined threshold
	@echo "Checking combined coverage (minimum $(COVERAGE_THRESHOLD_COMBINED)%)..."
	@E2E_TEST_MODE=multi_episode pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_COMBINED) -m 'not nightly' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1

coverage-report:
	# Generate coverage report without running tests (uses existing .coverage file)
	@coverage report --show-missing
	@coverage html -d .build/coverage-html
	@echo "HTML report: .build/coverage-html/index.html"

coverage-enforce:
	# Enforce combined coverage threshold on existing .coverage file (fast, no re-run)
	# Use this after 'make test' to verify coverage meets threshold
	@echo "Checking combined coverage threshold ($(COVERAGE_THRESHOLD_COMBINED)%)..."
	@coverage report --fail-under=$(COVERAGE_THRESHOLD_COMBINED) > /dev/null && \
		echo "✅ Coverage meets $(COVERAGE_THRESHOLD_COMBINED)% threshold" || \
		(echo "❌ Coverage below $(COVERAGE_THRESHOLD_COMBINED)% threshold" && exit 1)

build:
	$(PYTHON) -m pip install --quiet build
	$(PYTHON) -m build
	@if [ -d dist ]; then mkdir -p .build && rm -rf .build/dist && mv dist .build/ && echo "Moved dist to .build/dist/"; fi

ci: format-check lint lint-markdown type security complexity deadcode docstrings spelling preload-ml-models test coverage-enforce docs build

ci-fast: format-check lint lint-markdown type security complexity deadcode docstrings spelling test-fast docs build
	# Note: ci-fast skips coverage-enforce because fast tests have partial coverage

ci-clean: clean-all format-check lint lint-markdown type security preload-ml-models test docs build

ci-nightly: format-check lint lint-markdown type security complexity deadcode docstrings spelling preload-ml-models-production test-unit test-integration test-e2e test-nightly coverage-enforce docs build
	@echo ""
	@echo "✓ Full nightly CI chain completed"

docker-build:
	docker build -t podcast-scraper:test -f Dockerfile .

docker-build-fast:
	@echo "Building Docker image (fast mode - no model preloading, matches PR builds)..."
	@echo "This should complete in under 5 minutes..."
	@echo ""
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg PRELOAD_ML_MODELS=false \
		-t podcast-scraper:test-fast \
		-f Dockerfile .
	@echo ""
	@echo "✓ Fast build complete! Image tagged as: podcast-scraper:test-fast"

docker-build-full:
	@echo "Building Docker image (full mode - with model preloading, matches main builds)..."
	@echo "This will take longer due to ML model downloads..."
	@echo ""
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg PRELOAD_ML_MODELS=true \
		--build-arg WHISPER_MODELS=base.en \
		-t podcast-scraper:test \
		-f Dockerfile .
	@echo ""
	@echo "✓ Full build complete! Image tagged as: podcast-scraper:test"

docker-test: docker-build
	@echo "Running Docker tests..."
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
	docker rmi podcast-scraper:test podcast-scraper:test-fast podcast-scraper:multi-model 2>/dev/null || true

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
	# Coverage reports (XML, HTML)
	rm -rf reports/ htmlcov/

clean-cache:
	@echo "Cleaning ML model caches..."
	@if [ -d "$$HOME/.cache/whisper" ]; then \
		echo "  Removing Whisper cache: $$HOME/.cache/whisper"; \
		echo "    (includes tiny.en model)"; \
		rm -rf "$$HOME/.cache/whisper"; \
	fi
	@if [ -d "$$HOME/.cache/spacy" ]; then \
		echo "  Removing spaCy cache: $$HOME/.cache/spacy"; \
		echo "    (Note: spaCy model is installed as dependency, but cache may exist)"; \
		rm -rf "$$HOME/.cache/spacy"; \
	fi
	@if [ -d "$$HOME/.local/share/spacy" ]; then \
		echo "  Removing spaCy user cache: $$HOME/.local/share/spacy"; \
		echo "    (Note: spaCy model is installed as dependency, but cache may exist)"; \
		rm -rf "$$HOME/.local/share/spacy"; \
	fi
	@if [ -d "$$HOME/.cache/huggingface" ]; then \
		echo "  Removing HuggingFace cache: $$HOME/.cache/huggingface"; \
		echo "    (includes all Transformers models: facebook/bart-base, facebook/bart-large-cnn, sshleifer/distilbart-cnn-12-6)"; \
		rm -rf "$$HOME/.cache/huggingface"; \
	fi
	@if [ -d "$$HOME/.cache/torch" ]; then \
		echo "  Removing PyTorch cache: $$HOME/.cache/torch"; \
		rm -rf "$$HOME/.cache/torch"; \
	fi
	@echo "Cache cleaning complete. Run 'make test-unit' to verify network isolation."

clean-all: clean clean-cache
	@echo "All cleaning complete."

preload-ml-models:
	@echo "Preloading ML models for local development..."
	@$(PYTHON) scripts/preload_ml_models.py

preload-ml-models-production:
	@echo "Preloading production ML models for nightly tests..."
	@echo "Models: Whisper base, BART-large-cnn, LED-large-16384, en_core_web_sm"
	@$(PYTHON) scripts/preload_ml_models.py --production

