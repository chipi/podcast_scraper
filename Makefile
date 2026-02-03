# Auto-detect venv Python if .venv exists, otherwise use python3
ifeq ($(wildcard .venv/bin/python),)
PYTHON ?= python3
else
PYTHON ?= .venv/bin/python
endif
PACKAGE = podcast_scraper

# Test parallelism: Smart default that adapts to CPU count
# Formula: min(max(1, cpu_count - 2), 8)
# - Reserves 2 cores for system operations
# - Caps at 8 to prevent excessive memory usage
# - Falls back to 2 if CPU detection fails
# Can be overridden: PYTEST_WORKERS=4 make test
PYTEST_WORKERS ?= $(shell python3 -c "import os; print(min(max(1, (os.cpu_count() or 4) - 2), 8))")

.PHONY: help init init-no-ml format format-check lint lint-markdown lint-markdown-docs type security security-bandit security-audit complexity deadcode docstrings spelling spelling-docs quality check-unit-imports deps-analyze deps-check analyze-test-memory test-unit test-unit-sequential test-unit-no-ml test-integration test-integration-sequential test-integration-fast test-ci test-ci-fast test-e2e test-e2e-sequential test-e2e-fast test-e2e-data-quality test-nightly test test-sequential test-fast test-reruns test-track test-track-view test-openai test-openai-multi test-openai-all-feeds test-openai-real test-openai-real-multi test-openai-real-all-feeds test-openai-real-feed coverage coverage-check coverage-check-unit coverage-check-integration coverage-check-e2e coverage-check-combined coverage-report coverage-enforce docs docs-check build ci ci-fast ci-sequential ci-clean ci-nightly clean clean-cache clean-model-cache clean-all docker-build docker-build-fast docker-build-full docker-test docker-clean install-hooks preload-ml-models preload-ml-models-production backup-cache backup-cache-dry-run backup-cache-list backup-cache-cleanup restore-cache restore-cache-dry-run metadata-generate source-index dataset-create dataset-smoke dataset-benchmark dataset-raw dataset-materialize run-promote baseline-create experiment-run runs-list baselines-list runs-compare benchmark

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
	@echo "Verification commands:"
	@echo "  make check-unit-imports  Verify unit tests can import modules without ML dependencies"
	@echo "  make deps-analyze        Analyze module dependencies and detect architectural issues (with report)"
	@echo "  make deps-check          Check dependencies and exit with error if issues found"
	@echo ""
	@echo "Analysis commands:"
	@echo "  make analyze-test-memory [TARGET=test-unit] [WORKERS=N]  Analyze test memory usage and resource consumption"
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
	@echo "  make test-track          Run all test suites (unit + integration + e2e) and track execution times"
	@echo "                            Results saved to reports/test-timings.json for historical comparison"
	@echo "  make test-track-view     View test timing history and compare runs"
	@echo "                            Options: --last N, --compare, --stats, --regressions, --trends"
	@echo ""
	@echo "OpenAI test commands:"
	@echo "  make test-openai         Run OpenAI tests with fast feed (p01_fast.xml - 1 episode, 1 minute)"
	@echo "  make test-openai-multi   Run OpenAI tests with multi-episode feed (p01_multi.xml - 5 episodes)"
	@echo "  make test-openai-all-feeds Run OpenAI tests against all p01-p05 feeds (p01_mtb.xml through p05_investing.xml)"
	@echo "  make test-openai-real    Run OpenAI tests with REAL API using fast feed (fixture input, real API calls)"
	@echo "                            NOTE: Only runs test_openai_all_providers_in_pipeline to minimize costs"
	@echo "  make test-openai-real-multi Run OpenAI tests with REAL API using multi-episode feed"
	@echo "                            NOTE: Only runs test_openai_all_providers_in_pipeline to minimize costs"
	@echo "  make test-openai-real-all-feeds Run OpenAI tests with REAL API against all p01-p05 feeds"
	@echo "                            NOTE: Only runs test_openai_all_providers_in_pipeline to minimize costs"
	@echo "  make test-openai-real-feed Run OpenAI tests with REAL API using a real RSS feed"
	@echo "                            Usage: make test-openai-real-feed FEED_URL=\"https://...\" [MAX_EPISODES=5]"
	@echo "                            NOTE: Only runs test_openai_all_providers_in_pipeline to minimize costs"
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
	@echo "  make docs-check      Run all documentation checks (linting + spelling + build)"
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
	@echo "  make clean           Remove build artifacts (.build/, .mypy_cache/, .pytest_cache/, output/)"
	@echo "  make clean-cache     Remove ML model caches (Whisper, spaCy) to test network isolation"
	@echo "  make clean-model-cache  Delete cache for a specific Transformers model (forces re-download)"
	@echo "                            Usage: make clean-model-cache MODEL_NAME=google/pegasus-cnn_dailymail [FORCE=yes]"
	@echo "  make clean-all       Remove both build artifacts and ML model caches"
	@echo "  make preload-ml-models  Pre-download and cache all required ML models locally (test models)"
	@echo "  make preload-ml-models-production  Pre-download and cache production ML models (for nightly tests)"
	@echo "  make backup-cache    Backup .cache directory (ML models)"
	@echo "  make backup-cache-dry-run  Dry run: Check what would be backed up"
	@echo "  make backup-cache-list     List existing cache backups"
	@echo "  make backup-cache-cleanup   Clean up old cache backups (keeping 5 most recent)"
	@echo "  make restore-cache   Restore .cache directory from backup (use TARGET=path BACKUP=name)"
	@echo "  make restore-cache-dry-run  Dry run: Check what would be restored"
	@echo ""
	@echo "Experiment commands:"
	@echo "  make metadata-generate   Generate episode metadata JSON files from RSS XML files"
	@echo "                            Usage: make metadata-generate INPUT_DIR=data/eval/sources [OUTPUT_DIR=...]"
	@echo "  make source-index        Generate source index JSON files for inventory management"
	@echo "                            Usage: make source-index SOURCE_DIR=data/eval/sources/curated_5feeds_raw_v1 [ALL=1]"
	@echo "  make dataset-create      Create a canonical dataset from eval data"
	@echo "                            Usage: make dataset-create DATASET_ID=indicator_v1 [EVAL_DIR=data/eval] [OUTPUT_DIR=...]"
	@echo "  make dataset-smoke       Create smoke test dataset (first episode per feed)"
	@echo "                            Usage: make dataset-smoke [EVAL_DIR=data/eval] [OUTPUT_DIR=...]"
	@echo "  make dataset-benchmark    Create benchmark dataset (first 2 episodes per feed)"
	@echo "                            Usage: make dataset-benchmark [EVAL_DIR=data/eval] [OUTPUT_DIR=...]"
	@echo "  make dataset-raw          Create raw dataset (all episodes)"
	@echo "                            Usage: make dataset-raw [EVAL_DIR=data/eval] [OUTPUT_DIR=...]"
	@echo "  make dataset-materialize  Materialize a dataset (copy transcripts, validate hashes)"
	@echo "                            Usage: make dataset-materialize DATASET_ID=curated_5feeds_smoke_v1 [OUTPUT_DIR=...]"
	@echo "  make run-promote         Promote a run to baseline or reference"
	@echo "                            Usage: make run-promote RUN_ID=run_xxx --as baseline PROMOTED_ID=baseline_prod_v2 REASON=\"...\""
	@echo "  make baseline-create     Materialize a baseline (auto-creates and promotes)"
	@echo "                            Usage: make baseline-create BASELINE_ID=bart_led_baseline_v1 DATASET_ID=indicator_v1"
	@echo "  make experiment-run      Run an experiment using a config file"
	@echo "                            Usage: make experiment-run CONFIG=data/eval/configs/my_experiment.yaml"
	@echo "  make run-freeze          Freeze a run for baseline comparison"
	@echo "                            Usage: make run-freeze RUN_ID=run_name [REASON=\"...\"]"
	@echo "  make runs-delete         Delete experiment runs"
	@echo "                            Usage: make runs-delete RUN_IDS=\"run1 run2 run3\""
	@echo "  make configs-archive     Archive experiment configs to dated folder"
	@echo "                            Usage: make configs-archive [NAME=param_sweeps] [DATE=2026-01-30]"
	@echo "  make configs-clean       Remove experiment configs (use after archive)"
	@echo "                            Usage: make configs-clean [KEEP=baseline.yaml]"
	@echo "  make runs-archive        Archive experiment runs to named subfolder"
	@echo "                            Usage: make runs-archive FOLDER=name PATTERN=\"pattern\""
	@echo "                                  or: make runs-archive FOLDER=name RUNS=\"run1 run2\""

init:
	# Upgrade pip, setuptools, and wheel (required for PEP 660 editable installs with pyproject.toml)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e .[dev,ml]
	@if [ -f docs/requirements.txt ]; then $(PYTHON) -m pip install -r docs/requirements.txt; fi

format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

format-check:
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check-only .

lint:
	$(PYTHON) -m flake8 --config .flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 --config .flake8 . --count --exit-zero --statistics

lint-markdown:
	@command -v markdownlint >/dev/null 2>&1 || { echo "markdownlint not found. Install with: npm install -g markdownlint-cli"; exit 1; }
	markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site --ignore "docs/wip/**" --ignore "tests/fixtures/**" --config .markdownlint.json

lint-markdown-docs:
	@command -v markdownlint >/dev/null 2>&1 || { echo "markdownlint not found. Install with: npm install -g markdownlint-cli"; exit 1; }
	markdownlint "docs/**/*.md" --ignore "docs/wip/**" --config .markdownlint.json

fix-md:
	@echo "⚠️  WARNING: fix-md script is disabled due to issues."
	@echo "Use 'markdownlint --fix' instead for reliable markdown fixes."
	@echo ""
	@echo "Example:"
	@echo "  npx markdownlint-cli2 --fix '**/*.md'"
	@exit 1

type:
	$(PYTHON) -m mypy --config-file pyproject.toml .

security: security-bandit security-audit

security-bandit:
	$(PYTHON) -m bandit -r . --exclude ./.venv --skip B113,B108,B110,B310 --severity-level medium

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

spelling-docs:
	@echo "=== Spell Checking (Docs only) ==="
	@codespell docs/ --skip="*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl" || true

quality: complexity deadcode docstrings spelling
	@echo ""
	@echo "✓ All code quality checks completed"
	# This ensures production dependencies like torch, transformers, spacy, openai-whisper are audited
	$(PYTHON) -m pip install --quiet -e .[ml] || \
		(echo "⚠️  Editable install failed, using non-editable install" && \
		 $(PYTHON) -m pip install --quiet .[ml])
	# Audit all installed packages (including ML dependencies from pyproject.toml)
	# Ignore PYSEC-2022-42969: py package vulnerability (transitive dep of interrogate, deprecated, not exploitable here)
	# Ignore CVE-2026-0994: protobuf vulnerability (affects 6.33.4, fixed in later versions; transitive dep of ML packages)
	# Note: If protobuf is updated to >=6.33.5 or >=7.0.0, this ignore can be removed
	# Ignore en-core-web-sm: installed from GitHub (not PyPI), cannot be audited by pip-audit
	pip-audit --skip-editable --ignore-vuln PYSEC-2022-42969 --ignore-vuln CVE-2026-0994 --ignore-package en-core-web-sm

docs:
	$(PYTHON) -m mkdocs build --strict

docs-check: lint-markdown-docs spelling-docs docs
	@echo ""
	@echo "✓ Documentation validation complete (linting + spelling + build)"

# Coverage thresholds per layer (minimums to ensure balanced coverage)
# These are ambitious but achievable targets based on current coverage levels
# Combined threshold is enforced in CI; per-layer thresholds ensure no layer is neglected
COVERAGE_THRESHOLD_UNIT := 70          # Current: ~74% local, ~70% CI
COVERAGE_THRESHOLD_INTEGRATION := 40   # Current: ~54% local, ~42% CI
COVERAGE_THRESHOLD_E2E := 40           # Current: ~53% local, ~50% CI
COVERAGE_THRESHOLD_COMBINED := 80      # Current: ~82% local

check-unit-imports:
	# Verify that unit tests can import modules without ML dependencies
	# This ensures unit tests can run in CI without heavy ML dependencies installed
	# Run this when: adding new modules, refactoring imports, or debugging CI failures
	export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/check_unit_test_imports.py

test-unit:
	# Unit tests: parallel execution for faster feedback
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	$(PYTHON) -m pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost

test-integration:
	# Integration tests: parallel execution (3.4x faster, significant benefit)
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Integration tests load ML models which consume ~1-2 GB per worker
	# Includes reruns for flaky tests (matches CI behavior)
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	# Coverage: measured independently (not appended) to match CI per-job measurement
	$(PYTHON) -m pytest tests/integration/ -m integration -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --reruns 2 --reruns-delay 1 --disable-socket --allow-hosts=127.0.0.1,localhost

test-integration-fast:
	# Fast integration tests: critical path tests only (excludes ml_models for speed)
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Includes reruns for flaky tests (matches CI behavior)
	# Excludes ml_models marker - use test-integration for ML workflow tests
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Coverage: measured independently but no threshold (fast tests are a subset, full suite enforces threshold)
	$(PYTHON) -m pytest tests/integration/ -m "integration and critical_path and not ml_models" -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 --durations=20

test-ci:
	# CI test suite: parallel execution for speed
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Note: Non-critical path tests run on main branch only
	$(PYTHON) -m pytest -m '(not integration and not e2e) or (integration and critical_path) or (e2e and critical_path)' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --cov=$(PACKAGE) --cov-report=term-missing

test-ci-fast:
	# Fast CI test suite: parallel execution for speed
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Note: Coverage is excluded here for faster execution; full validation job includes unified coverage
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Includes reruns for flaky tests (matches CI behavior) - increased to 3 retries for very flaky tests
	$(PYTHON) -m pytest tests/unit/ tests/integration/ tests/e2e/ -m 'not nightly and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20 --reruns 3 --reruns-delay 2

test-e2e:
	# E2E tests: parallel execution for speed
	# Excludes analysis/diagnostic tests - these are slow diagnostic tools, not regular tests
	# Includes reruns for flaky tests (matches CI behavior) - 3 retries for ML model variability
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	@E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/e2e/ -m "e2e and not analysis" -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1

test-e2e-sequential:
	# E2E tests: sequential execution (slower but clearer output, useful for debugging)
	# Excludes analysis/diagnostic tests - these are slow diagnostic tools, not regular tests
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/e2e/ -m "e2e and not analysis" --disable-socket --allow-hosts=127.0.0.1,localhost

test-e2e-fast:
	# Fast E2E tests: parallel execution for speed
	# Critical path tests only (includes ML tests if models are cached)
	# Excludes analysis/diagnostic tests (p07/p08 threshold analysis) - these are slow and not critical path
	# Includes reruns for flaky tests (matches CI behavior) - 3 retries for ML model variability
	# Uses fast feed (1 episode) - set via E2E_TEST_MODE environment variable
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Coverage: measured independently but no threshold (fast tests are a subset, full suite enforces threshold)
	@E2E_TEST_MODE=fast $(PYTHON) -m pytest tests/e2e/ -m "e2e and critical_path and not analysis" -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 --durations=20

test-e2e-data-quality:
	# Data quality E2E tests: full pipeline validation with multiple episodes
	# Excludes analysis/diagnostic tests - these are slow diagnostic tools, not regular tests
	# Uses all original mock data (not fast fixtures)
	# Runs with 3-5 episodes per test to validate data quality and consistency
	# For nightly builds only - not part of regular CI/CD code quality checks
	@E2E_TEST_MODE=data_quality pytest tests/e2e/ -m "e2e and data_quality and not analysis" -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1

test-analytical:
	# Analytical/diagnostic tests: tools for investigating specific behaviors and thresholds
	# These are NOT regular tests - they are diagnostic tools for:
	# - Investigating threshold behavior (e.g., summarization thresholds)
	# - Capturing baseline metrics for comparison
	# - Diagnosing specific issues
	# - Performance analysis
	# Uses E2E infrastructure (E2E server, fixtures) but separate from regular E2E tests
	# Run explicitly when investigating specific issues or capturing metrics
	@E2E_TEST_MODE=fast $(PYTHON) -m pytest tests/analytical/ -m "analytical" -v --disable-socket --allow-hosts=127.0.0.1,localhost --durations=10

test-nightly:
	# Nightly-only tests: comprehensive tests with production ML models (p01-p05 full suite)
	# Uses production models: Whisper base.en, BART-large-cnn, LED-large-16384
	# Runs all 15 episodes across 5 podcasts (p01-p05)
	# Sequential execution per podcast, parallel episodes within podcast (2 workers)
	# NOT marked with @pytest.mark.e2e - separate category from regular E2E tests
	# Excludes LLM/OpenAI tests to avoid API costs (see issue #183)
	@echo "Running nightly tests with production models..."
	@echo "Podcasts: p01-p05 (15 episodes total)"
	@echo "Models: Whisper base.en, BART-large-cnn, LED-large-16384"
	@mkdir -p reports
	@E2E_TEST_MODE=nightly pytest tests/e2e/ -m "nightly and not llm" -v -n 2 --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20 --junitxml=reports/junit-nightly.xml --json-report --json-report-file=reports/pytest-nightly.json

test:
	# All tests: parallel execution for speed
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	# Parallelism: $(PYTEST_WORKERS) workers (adapts to CPU, reserves 2 cores, caps at 8)
	# Excludes nightly tests (run separately via make test-nightly)
	# Excludes analytical tests (diagnostic tools, run separately via make test-analytical)
	# Note: Coverage with pytest-xdist may show lower numbers due to parallel collection
	# Use 'make test-sequential' for accurate coverage measurement
	@E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/ -m "not nightly and not analytical" -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost

test-sequential:
	# All tests: sequential execution (slower but clearer output, useful for debugging)
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	# Excludes analytical tests (diagnostic tools, run separately via make test-analytical)
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/ -m "not analytical" --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost

test-fast:
	# Fast tests: parallel execution for speed
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Uses fast feed for E2E tests (1 episode) - set via E2E_TEST_MODE environment variable
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Excludes nightly tests (comprehensive tests run only in nightly builds)
	# Use --durations=20 to monitor slow tests and optimize them separately
	@E2E_TEST_MODE=fast $(PYTHON) -m pytest -m 'not nightly and ((not integration and not e2e) or (integration and critical_path) or (e2e and critical_path))' -n $(PYTEST_WORKERS) --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost --durations=20

test-reruns:
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	$(PYTHON) -m pytest --reruns 2 --reruns-delay 1 --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' --disable-socket --allow-hosts=127.0.0.1,localhost

test-track:
	# Run all test suites (unit + integration + e2e) and track execution times
	# Results saved to reports/test-timings.json for historical comparison
	# Use this to monitor test performance over time and detect regressions
	@mkdir -p reports
	$(PYTHON) scripts/tools/track_test_timings.py

test-track-view:
	# View test timing history and compare runs
	# Shows recent runs with timing comparisons
	@if [ ! -f reports/test-timings.json ]; then \
		echo "No timing data found. Run 'make test-track' first."; \
		exit 1; \
	fi
	$(PYTHON) scripts/tools/track_test_timings.py --view

test-openai:
	# Test OpenAI providers (uses E2E server by default, or real API if USE_REAL_OPENAI_API=1)
	# Feed selection via OPENAI_TEST_FEED environment variable (E2E mode):
	#   - fast: p01_fast.xml (1 episode, 1 minute) - DEFAULT
	#   - multi: p01_multi.xml (5 episodes, 10-15s each)
	#   - p01-p05: Individual podcast feeds (p01_mtb.xml, p02_software.xml, etc.)
	# For real API mode: Set USE_REAL_OPENAI_API=1 and OPENAI_TEST_RSS_FEED=<feed-url>
	# Examples:
	#   make test-openai                    # Uses fast feed with E2E server (1 episode, 1 minute)
	#   make test-openai-multi              # Uses multi-episode feed (5 episodes, 10-15s each)
	#   OPENAI_TEST_FEED=p02 make test-openai    # Uses podcast2 feed
	#   USE_REAL_OPENAI_API=1 OPENAI_TEST_RSS_FEED=https://example.com/podcast.rss make test-openai  # Real API
	@echo "Running OpenAI tests..."
	@echo "  Feed type: $${OPENAI_TEST_FEED:-fast} (use OPENAI_TEST_FEED to change)"
	@if [ "$${USE_REAL_OPENAI_API:-0}" = "1" ]; then \
		if [ -z "$$OPENAI_API_KEY" ] && ! grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then \
			echo "❌ Error: OPENAI_API_KEY not set in environment or .env file"; \
			echo "   Set it in your .env file or export it before running this target"; \
			exit 1; \
		fi; \
		echo "⚠️  WARNING: Running tests with REAL OpenAI API (will incur costs)"; \
		echo "⚠️  This will make actual API calls to OpenAI endpoints"; \
		echo "⚠️  Press Ctrl+C within 5 seconds to cancel..."; \
		sleep 5; \
	fi
	@echo ""
	E2E_TEST_MODE=fast USE_REAL_OPENAI_API=$${USE_REAL_OPENAI_API:-0} OPENAI_TEST_FEED=$${OPENAI_TEST_FEED:-fast} $(PYTHON) -m pytest \
		tests/e2e/test_openai_provider_integration_e2e.py \
		-m "openai" \
		-v \
		--tb=short \
		--durations=10 \
		--maxfail=1

test-openai-multi:
	# Test OpenAI providers with multi-episode feed (p01_multi.xml - 5 episodes, 10-15s each)
	# Uses E2E server by default (no API costs)
	@echo "Running OpenAI tests with multi-episode feed..."
	@if [ "$${USE_REAL_OPENAI_API:-0}" = "1" ]; then \
		if [ -z "$$OPENAI_API_KEY" ] && ! grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then \
			echo "❌ Error: OPENAI_API_KEY not set in environment or .env file"; \
			echo "   Set it in your .env file or export it before running this target"; \
			exit 1; \
		fi; \
		echo "⚠️  WARNING: Running tests with REAL OpenAI API (will incur costs)"; \
		echo "⚠️  This will make actual API calls to OpenAI endpoints"; \
		echo "⚠️  Press Ctrl+C within 5 seconds to cancel..."; \
		sleep 5; \
	fi
	@echo ""
	E2E_TEST_MODE=fast USE_REAL_OPENAI_API=$${USE_REAL_OPENAI_API:-0} OPENAI_TEST_FEED=multi $(PYTHON) -m pytest \
		tests/e2e/test_openai_provider_integration_e2e.py \
		-m "openai" \
		-v \
		--tb=short \
		--durations=10 \
		--maxfail=1

test-openai-all-feeds:
	@# Test OpenAI providers against all p01-p05 feeds (p01_mtb.xml through p05_investing.xml)
	@# Uses E2E server in nightly mode (allows all podcasts 1-5)
	@# Each feed is tested sequentially to ensure clean state between runs
	@echo "Running OpenAI tests against all p01-p05 feeds..."
	@echo "  This will test: p01 (mtb), p02 (software), p03 (scuba), p04 (photo), p05 (investing)"
	@echo ""
	@if [ "$${USE_REAL_OPENAI_API:-0}" = "1" ]; then \
		if [ -z "$$OPENAI_API_KEY" ] && ! grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then \
			echo "❌ Error: OPENAI_API_KEY not set in environment or .env file"; \
			echo "   Set it in your .env file or export it before running this target"; \
			exit 1; \
		fi; \
		echo "⚠️  WARNING: Running tests with REAL OpenAI API (will incur costs)"; \
		echo "⚠️  This will make actual API calls to OpenAI endpoints"; \
		echo "⚠️  Press Ctrl+C within 5 seconds to cancel..."; \
		sleep 5; \
	fi
	@for feed in p01 p02 p03 p04 p05; do \
		echo "========================================="; \
		echo "Testing feed: $$feed"; \
		echo "========================================="; \
		E2E_TEST_MODE=nightly USE_REAL_OPENAI_API=$${USE_REAL_OPENAI_API:-0} OPENAI_TEST_FEED=$$feed $(PYTHON) -m pytest \
			tests/e2e/test_openai_provider_integration_e2e.py \
			-m "openai" \
			-v \
			--tb=short \
			--durations=10 \
			--maxfail=1 || exit 1; \
		echo ""; \
	done
	@echo "========================================="
	@echo "All feeds tested successfully!"
	@echo "========================================="

test-openai-real:
	@# Test OpenAI providers with REAL API using fast feed (p01_fast.xml - 1 episode, 1 minute)
	@# Uses fixture feeds as input but makes real OpenAI API calls
	@# Requires OPENAI_API_KEY in .env file
	@# NOTE: Only runs test_openai_all_providers_in_pipeline to minimize API costs
	@echo "Running OpenAI tests with REAL API (using fast feed fixture)..."
	@if [ -z "$$OPENAI_API_KEY" ] && ! grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then \
		echo "❌ Error: OPENAI_API_KEY not set in environment or .env file"; \
		echo "   Set it in your .env file or export it before running this target"; \
		exit 1; \
	fi
	@echo "⚠️  WARNING: Running tests with REAL OpenAI API (will incur costs)"
	@echo "⚠️  This will make actual API calls to OpenAI endpoints"
	@echo "⚠️  Using fixture feed (p01_fast.xml) as input"
	@echo "⚠️  Running ONLY test_openai_all_providers_in_pipeline to minimize costs"
	@echo "⚠️  Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	@echo ""
	E2E_TEST_MODE=fast USE_REAL_OPENAI_API=1 OPENAI_TEST_FEED=fast $(PYTHON) -m pytest \
		tests/e2e/test_openai_provider_integration_e2e.py::TestOpenAIProviderE2E::test_openai_all_providers_in_pipeline \
		-v \
		-s \
		--tb=short \
		--durations=10 \
		--maxfail=1

test-openai-real-multi:
	@# Test OpenAI providers with REAL API using multi-episode feed (p01_multi.xml - 5 episodes)
	@# Uses fixture feeds as input but makes real OpenAI API calls
	@# Requires OPENAI_API_KEY in .env file
	@# NOTE: Only runs test_openai_all_providers_in_pipeline to minimize API costs
	@echo "Running OpenAI tests with REAL API (using multi-episode feed fixture)..."
	@if [ -z "$$OPENAI_API_KEY" ] && ! grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then \
		echo "❌ Error: OPENAI_API_KEY not set in environment or .env file"; \
		echo "   Set it in your .env file or export it before running this target"; \
		exit 1; \
	fi
	@echo "⚠️  WARNING: Running tests with REAL OpenAI API (will incur costs)"
	@echo "⚠️  This will make actual API calls to OpenAI endpoints"
	@echo "⚠️  Using multi-episode fixture feed (p01_multi.xml - 5 episodes) as input"
	@echo "⚠️  Running ONLY test_openai_all_providers_in_pipeline to minimize costs"
	@echo "⚠️  Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	@echo ""
	E2E_TEST_MODE=fast USE_REAL_OPENAI_API=1 OPENAI_TEST_FEED=multi $(PYTHON) -m pytest \
		tests/e2e/test_openai_provider_integration_e2e.py::TestOpenAIProviderE2E::test_openai_all_providers_in_pipeline \
		-v \
		--tb=short \
		--durations=10 \
		--maxfail=1

test-openai-real-all-feeds:
	@# Test OpenAI providers with REAL API against all p01-p05 feeds
	@# Uses fixture feeds as input but makes real OpenAI API calls
	@# Requires OPENAI_API_KEY in .env file
	@# NOTE: Only runs test_openai_all_providers_in_pipeline to minimize API costs
	@echo "Running OpenAI tests with REAL API against all p01-p05 feeds..."
	@echo "  This will test: p01 (mtb), p02 (software), p03 (scuba), p04 (photo), p05 (investing)"
	@echo "  Using fixture feeds as input but making REAL OpenAI API calls"
	@echo "  Running ONLY test_openai_all_providers_in_pipeline to minimize costs"
	@echo ""
	@if [ -z "$$OPENAI_API_KEY" ] && ! grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then \
		echo "❌ Error: OPENAI_API_KEY not set in environment or .env file"; \
		echo "   Set it in your .env file or export it before running this target"; \
		exit 1; \
	fi
	@echo "⚠️  WARNING: Running tests with REAL OpenAI API (will incur costs)"
	@echo "⚠️  This will make actual API calls to OpenAI endpoints"
	@echo "⚠️  Testing all 5 feeds (p01-p05) with fixture data"
	@echo "⚠️  Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	@echo ""
	@for feed in p01 p02 p03 p04 p05; do \
		echo "========================================="; \
		echo "Testing feed: $$feed (REAL API)"; \
		echo "========================================="; \
		E2E_TEST_MODE=nightly USE_REAL_OPENAI_API=1 OPENAI_TEST_FEED=$$feed $(PYTHON) -m pytest \
			tests/e2e/test_openai_provider_integration_e2e.py::TestOpenAIProviderE2E::test_openai_all_providers_in_pipeline \
			-v \
			--tb=short \
			--durations=10 \
			--maxfail=1 || exit 1; \
		echo ""; \
	done
	@echo "========================================="
	@echo "All feeds tested successfully with REAL API!"
	@echo "========================================="

test-openai-real-feed:
	@# Test OpenAI providers with REAL API using a real RSS feed
	@# Usage: make test-openai-real-feed FEED_URL="https://example.com/podcast.rss" [MAX_EPISODES=5]
	@# Requires OPENAI_API_KEY in .env file
	@# NOTE: Only runs test_openai_all_providers_in_pipeline to minimize API costs
	@if [ -z "$(FEED_URL)" ]; then \
		echo "❌ Error: FEED_URL is required"; \
		echo ""; \
		echo "Usage: make test-openai-real-feed FEED_URL=\"https://example.com/podcast.rss\" [MAX_EPISODES=5]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make test-openai-real-feed FEED_URL=\"https://example.com/podcast.rss\""; \
		echo "  make test-openai-real-feed FEED_URL=\"https://example.com/podcast.rss\" MAX_EPISODES=3"; \
		exit 1; \
	fi
	@if [ -z "$$OPENAI_API_KEY" ] && ! grep -q "^OPENAI_API_KEY=" .env 2>/dev/null; then \
		echo "❌ Error: OPENAI_API_KEY not set in environment or .env file"; \
		echo "   Set it in your .env file or export it before running this target"; \
		exit 1; \
	fi
	@MAX_EPISODES=$${MAX_EPISODES:-5}; \
	echo "Running OpenAI tests with REAL API using real RSS feed..."; \
	echo "⚠️  WARNING: Running tests with REAL OpenAI API (will incur costs)"; \
	echo "⚠️  This will make actual API calls to OpenAI endpoints"; \
	echo "⚠️  Feed URL: $(FEED_URL)"; \
	echo "⚠️  Max Episodes: $$MAX_EPISODES"; \
	echo "⚠️  Running ONLY test_openai_all_providers_in_pipeline to minimize costs"; \
	echo "⚠️  Press Ctrl+C within 5 seconds to cancel..."; \
	sleep 5; \
	echo ""; \
	USE_REAL_OPENAI_API=1 OPENAI_TEST_RSS_FEED="$(FEED_URL)" OPENAI_TEST_MAX_EPISODES=$$MAX_EPISODES $(PYTHON) -m pytest \
		tests/e2e/test_openai_provider_integration_e2e.py::TestOpenAIProviderE2E::test_openai_all_providers_in_pipeline \
		-v \
		-s \
		--tb=short \
		--durations=10 \
		--maxfail=1

coverage: test
	# Runs all tests with coverage (same as 'make test')

coverage-check: coverage-check-unit coverage-check-integration coverage-check-e2e
	# Verify all layers meet minimum coverage thresholds
	@echo "✅ All coverage checks passed!"

coverage-check-unit:
	# Check unit test coverage meets minimum threshold ($(COVERAGE_THRESHOLD_UNIT)%)
	@echo "Checking unit test coverage (minimum $(COVERAGE_THRESHOLD_UNIT)%)..."
	@$(PYTHON) -m pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_UNIT) -m 'not integration and not e2e' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost -q

coverage-check-integration:
	# Check integration test coverage meets minimum threshold ($(COVERAGE_THRESHOLD_INTEGRATION)%)
	@echo "Checking integration test coverage (minimum $(COVERAGE_THRESHOLD_INTEGRATION)%)..."
	@$(PYTHON) -m pytest tests/integration/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_INTEGRATION) -m 'integration' -n $(PYTEST_WORKERS) --disable-socket --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 -q

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

# Check if ML models are cached (used to conditionally run preload-ml-models)
# Returns "1" if models are missing, "0" if all cached
# This check runs at Makefile parse time, so it's fast and doesn't block
ML_MODELS_CACHED := $(shell $(PYTHON) -c "import sys; sys.path.insert(0, 'src'); \
	from tests.integration.ml_model_cache_helpers import _is_whisper_model_cached, _is_transformers_model_cached; \
	from podcast_scraper import config; \
	whisper_ok = _is_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL); \
	transformers_ok = _is_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None); \
	spacy_ok = False; \
	try: \
		import spacy; \
		spacy.load(config.DEFAULT_NER_MODEL); \
		spacy_ok = True; \
	except: \
		pass; \
	all_cached = whisper_ok and transformers_ok and spacy_ok; \
	print('1' if not all_cached else '0', end='')" 2>/dev/null || echo "1")

ci: format-check lint lint-markdown type security complexity deadcode docstrings spelling $(if $(filter 1,$(ML_MODELS_CACHED)),preload-ml-models,) test coverage-enforce docs build
	# Conditional preload: Only runs preload-ml-models if models are not cached
	# This makes ci seamless for new contributors (auto-downloads) and fast for experienced ones (skips if cached)
	@if [ "$(ML_MODELS_CACHED)" = "0" ]; then \
		echo ""; \
		echo "✓ ML models already cached, skipped preload"; \
	fi

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

deps-analyze:
	# Analyze module dependencies and detect architectural issues
	# Checks for circular imports, analyzes import patterns, checks thresholds
	# Generates detailed JSON report in reports/deps-analysis.json
	# Run this when: refactoring modules, adding new imports, or debugging architecture issues
	export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/analyze_dependencies.py --report

deps-check:
	# Check dependencies and exit with error if issues found
	# Use in CI or before committing to catch architectural issues early
	# Checks: circular imports, import thresholds (max 15 imports per module)
	export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/analyze_dependencies.py --check

analyze-test-memory:
	# Analyze test suite memory usage and resource consumption
	# Helps identify memory leaks, excessive resource usage, and optimization opportunities
	# Run this when: debugging memory issues, optimizing test performance, investigating leaks
	# Usage: make analyze-test-memory TARGET=test-unit WORKERS=4
	#   TARGET: Makefile test target (default: test-unit)
	#   WORKERS: Max parallel workers (optional, overrides Makefile setting)
	@if [ -z "$(TARGET)" ]; then \
		echo "Running memory analysis for default target (test-unit)..."; \
		export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/analyze_test_memory.py; \
	else \
		if [ -z "$(WORKERS)" ]; then \
			echo "Running memory analysis for $(TARGET)..."; \
			export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/analyze_test_memory.py --test-target $(TARGET); \
		else \
			echo "Running memory analysis for $(TARGET) with $(WORKERS) workers..."; \
			export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/analyze_test_memory.py --test-target $(TARGET) --max-workers $(WORKERS); \
		fi; \
	fi

clean:
	rm -rf build .build .mypy_cache .pytest_cache
	# Coverage files: .coverage.* are created during parallel test execution (pytest -n auto)
	rm -f .coverage .coverage.*
	# Coverage reports (XML, HTML)
	rm -rf reports/ htmlcov/
	# Test output directories (created during test runs)
	rm -rf output/
	# Temporary test directories (fingerprint validation tests)
	find /tmp -maxdepth 1 -type d -name "fingerprint_test_*" -exec rm -rf {} + 2>/dev/null || true
	# Temporary test config files (if any were left behind)
	find data/eval/configs -name "*_test_*.yaml" -type f -delete 2>/dev/null || true

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

clean-model-cache:
	@if [ -z "$(MODEL_NAME)" ]; then \
		echo "Error: MODEL_NAME is required"; \
		echo "Usage: make clean-model-cache MODEL_NAME=google/pegasus-cnn_dailymail [FORCE=yes]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make clean-model-cache MODEL_NAME=google/pegasus-cnn_dailymail"; \
		echo "  make clean-model-cache MODEL_NAME=facebook/bart-large-cnn FORCE=yes"; \
		exit 1; \
	fi
	@echo "Deleting cache for Transformers model: $(MODEL_NAME)"
	@if [ "$(FORCE)" = "yes" ]; then \
		$(PYTHON) -c "from podcast_scraper.cache import manager; success, freed = manager.delete_transformers_model_cache('$(MODEL_NAME)', confirm=False, force=True); exit(0)"; \
	else \
		$(PYTHON) -c "from podcast_scraper.cache import manager; success, freed = manager.delete_transformers_model_cache('$(MODEL_NAME)', confirm=True, force=False); exit(0)"; \
	fi

clean-all: clean clean-cache
	@echo "All cleaning complete."

preload-ml-models:
	@echo "Preloading ML models for local development..."
	@$(PYTHON) scripts/cache/preload_ml_models.py

preload-ml-models-production:
	@echo "Preloading production ML models for nightly tests..."
	@echo "Models: Whisper base, BART-large-cnn, LED-large-16384, en_core_web_sm"
	@$(PYTHON) scripts/cache/preload_ml_models.py --production

backup-cache:
	@echo "Backing up .cache directory (ML models)..."
	@$(PYTHON) scripts/cache/backup_cache.py

backup-cache-dry-run:
	@echo "Dry run: Checking what would be backed up..."
	@$(PYTHON) scripts/cache/backup_cache.py --dry-run --verbose

backup-cache-list:
	@echo "Listing existing cache backups..."
	@$(PYTHON) scripts/cache/backup_cache.py --list

backup-cache-cleanup:
	@echo "Cleaning up old cache backups (keeping 5 most recent)..."
	@$(PYTHON) scripts/cache/backup_cache.py --cleanup 5

restore-cache:
	@echo "Restoring .cache directory from backup..."
	@cmd="$(PYTHON) scripts/cache/restore_cache.py"; \
	if [ -n "$(TARGET)" ]; then \
		echo "  Target: $(TARGET)"; \
		cmd="$$cmd --target \"$(TARGET)\""; \
	fi; \
	if [ -n "$(BACKUP)" ]; then \
		echo "  Backup: $(BACKUP)"; \
		cmd="$$cmd --backup \"$(BACKUP)\""; \
	fi; \
	eval $$cmd

restore-cache-dry-run:
	@echo "Dry run: Checking what would be restored..."
	@cmd="$(PYTHON) scripts/cache/restore_cache.py --dry-run --verbose"; \
	if [ -n "$(TARGET)" ]; then \
		echo "  Target: $(TARGET)"; \
		cmd="$$cmd --target \"$(TARGET)\""; \
	fi; \
	if [ -n "$(BACKUP)" ]; then \
		echo "  Backup: $(BACKUP)"; \
		cmd="$$cmd --backup \"$(BACKUP)\""; \
	fi; \
	eval $$cmd

# Experiment commands (RFC-015, RFC-041)
metadata-generate:
	@# Generate episode metadata JSON files from RSS XML files
	@# Usage: make metadata-generate INPUT_DIR=data/eval/sources [OUTPUT_DIR=...] [LOG_LEVEL=INFO]
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "❌ Error: INPUT_DIR is required"; \
		echo ""; \
		echo "Usage: make metadata-generate INPUT_DIR=data/eval/sources"; \
		echo ""; \
		echo "Optional parameters:"; \
		echo "  OUTPUT_DIR=path/to/output    Output directory (default: same as XML file location)"; \
		echo "  LOG_LEVEL=INFO|DEBUG|WARNING|ERROR    Logging level (default: INFO)"; \
		exit 1; \
	fi
	@echo "Generating episode metadata from RSS XML files..."
	@echo "  Input directory: $(INPUT_DIR)"
	@cmd="$(PYTHON) scripts/eval/generate_episode_metadata.py --input-dir $(INPUT_DIR)"; \
	if [ -n "$(OUTPUT_DIR)" ]; then \
		echo "  Output directory: $(OUTPUT_DIR)"; \
		cmd="$$cmd --output-dir $(OUTPUT_DIR)"; \
	fi; \
	if [ -n "$(LOG_LEVEL)" ]; then \
		echo "  Log level: $(LOG_LEVEL)"; \
		cmd="$$cmd --log-level $(LOG_LEVEL)"; \
	fi; \
	eval $$cmd
	@echo ""
	@echo "✓ Metadata generation complete"

source-index:
	@# Generate source index JSON files for inventory management
	@# Usage: make source-index SOURCE_DIR=data/eval/sources/curated_5feeds_raw_v1 [ALL=1] [LOG_LEVEL=INFO]
	@if [ -z "$(SOURCE_DIR)" ]; then \
		echo "❌ Error: SOURCE_DIR is required"; \
		echo ""; \
		echo "Usage: make source-index SOURCE_DIR=data/eval/sources/curated_5feeds_raw_v1"; \
		echo ""; \
		echo "Optional parameters:"; \
		echo "  ALL=1                    Process all source directories in the given directory"; \
		echo "  LOG_LEVEL=INFO|DEBUG|WARNING|ERROR    Logging level (default: INFO)"; \
		exit 1; \
	fi
	@echo "Generating source index..."
	@echo "  Source directory: $(SOURCE_DIR)"
	@cmd="$(PYTHON) scripts/eval/generate_source_index.py --source-dir $(SOURCE_DIR)"; \
	if [ -n "$(ALL)" ] && [ "$(ALL)" = "1" ]; then \
		echo "  Processing all source directories"; \
		cmd="$$cmd --all"; \
	fi; \
	if [ -n "$(LOG_LEVEL)" ]; then \
		echo "  Log level: $(LOG_LEVEL)"; \
		cmd="$$cmd --log-level $(LOG_LEVEL)"; \
	fi; \
	eval $$cmd
	@echo ""
	@echo "✓ Source index generation complete"

dataset-create:
	@# Create a canonical dataset JSON from existing eval data
	@# Usage: make dataset-create DATASET_ID=indicator_v1 [EVAL_DIR=data/eval] [OUTPUT_DIR=...] [DESCRIPTION="..."] [CONTENT_REGIME="..."] [SMOKE_TEST=1]
	@if [ -z "$(DATASET_ID)" ]; then \
		echo "❌ Error: DATASET_ID is required"; \
		echo ""; \
		echo "Usage: make dataset-create DATASET_ID=indicator_v1"; \
		echo ""; \
		echo "Optional parameters:"; \
		echo "  EVAL_DIR=data/eval              Eval directory (default: data/eval)"; \
		echo "  OUTPUT_DIR=path/to/output       Output directory (default: benchmarks/datasets)"; \
		echo "  DESCRIPTION=\"Description text\"     Dataset description (default: auto-generated)"; \
		echo "  CONTENT_REGIME=\"narrative\"        Content regime (narrative, interview, etc.)"; \
		echo "  SMOKE_TEST=1                    Create smoke test dataset (first episode per feed)"; \
		exit 1; \
	fi
	@EVAL_DIR=$${EVAL_DIR:-data/eval}; \
	echo "Creating dataset: $(DATASET_ID)"; \
	echo "  Eval directory: $$EVAL_DIR"; \
	@# Generate default description if not provided
	@if [ -z "$(DESCRIPTION)" ]; then \
		DESCRIPTION="Dataset $(DATASET_ID)"; \
		echo "  Description: $$DESCRIPTION (auto-generated)"; \
	else \
		echo "  Description: $(DESCRIPTION)"; \
	fi; \
	cmd="$(PYTHON) scripts/eval/create_dataset_json.py --dataset-id $(DATASET_ID) --eval-dir $$EVAL_DIR --description \"$$DESCRIPTION\""; \
	if [ -n "$(OUTPUT_DIR)" ]; then \
		echo "  Output directory: $(OUTPUT_DIR)"; \
		cmd="$$cmd --output-dir $(OUTPUT_DIR)"; \
	fi; \
	if [ -n "$(CONTENT_REGIME)" ]; then \
		echo "  Content regime: $(CONTENT_REGIME)"; \
		cmd="$$cmd --content-regime $(CONTENT_REGIME)"; \
	fi; \
	if [ -n "$(SMOKE_TEST)" ] && [ "$(SMOKE_TEST)" = "1" ]; then \
		echo "  Smoke test mode: enabled (first episode per feed)"; \
		cmd="$$cmd --smoke-test"; \
	fi; \
	eval $$cmd
	@OUTPUT_DIR=$${OUTPUT_DIR:-benchmarks/datasets}; \
	echo ""; \
	echo "✓ Dataset created: $$OUTPUT_DIR/$(DATASET_ID).json"

dataset-smoke:
	@# Create smoke test dataset (first episode per feed)
	@# Usage: make dataset-smoke [EVAL_DIR=data/eval] [OUTPUT_DIR=...]
	@EVAL_DIR=$${EVAL_DIR:-data/eval}; \
	OUTPUT_DIR=$${OUTPUT_DIR:-data/eval/datasets}; \
	echo "Creating smoke test dataset: curated_5feeds_smoke_v1"; \
	echo "  Eval directory: $$EVAL_DIR"; \
	echo "  Output directory: $$OUTPUT_DIR"; \
	$(PYTHON) scripts/eval/create_dataset_json.py \
		--dataset-id curated_5feeds_smoke_v1 \
		--eval-dir $$EVAL_DIR \
		--output-dir $$OUTPUT_DIR \
		--description "Smoke test dataset: first episode per feed from curated_5feeds_raw_v1" \
		--max-episodes-per-feed 1
	@echo ""
	@echo "✓ Smoke test dataset created: $$OUTPUT_DIR/curated_5feeds_smoke_v1.json"

dataset-benchmark:
	@# Create benchmark dataset (first 2 episodes per feed)
	@# Usage: make dataset-benchmark [EVAL_DIR=data/eval] [OUTPUT_DIR=...]
	@EVAL_DIR=$${EVAL_DIR:-data/eval}; \
	OUTPUT_DIR=$${OUTPUT_DIR:-data/eval/datasets}; \
	echo "Creating benchmark dataset: curated_5feeds_benchmark_v1"; \
	echo "  Eval directory: $$EVAL_DIR"; \
	echo "  Output directory: $$OUTPUT_DIR"; \
	$(PYTHON) scripts/eval/create_dataset_json.py \
		--dataset-id curated_5feeds_benchmark_v1 \
		--eval-dir $$EVAL_DIR \
		--output-dir $$OUTPUT_DIR \
		--description "Benchmark dataset: first 2 episodes per feed from curated_5feeds_raw_v1" \
		--max-episodes-per-feed 2
	@echo ""
	@echo "✓ Benchmark dataset created: $$OUTPUT_DIR/curated_5feeds_benchmark_v1.json"

dataset-raw:
	@# Create raw dataset (all episodes from all feeds)
	@# Usage: make dataset-raw [EVAL_DIR=data/eval] [OUTPUT_DIR=...]
	@EVAL_DIR=$${EVAL_DIR:-data/eval}; \
	OUTPUT_DIR=$${OUTPUT_DIR:-data/eval/datasets}; \
	echo "Creating raw dataset: curated_5feeds_raw_v1"; \
	echo "  Eval directory: $$EVAL_DIR"; \
	echo "  Output directory: $$OUTPUT_DIR"; \
	$(PYTHON) scripts/eval/create_dataset_json.py \
		--dataset-id curated_5feeds_raw_v1 \
		--eval-dir $$EVAL_DIR \
		--output-dir $$OUTPUT_DIR \
		--description "Raw dataset: all episodes from all feeds in curated_5feeds_raw_v1"
	@echo ""
	@echo "✓ Raw dataset created: $$OUTPUT_DIR/curated_5feeds_raw_v1.json"

dataset-materialize:
	@# Materialize a dataset (copy transcripts, validate hashes)
	@# Usage: make dataset-materialize DATASET_ID=curated_5feeds_smoke_v1 [OUTPUT_DIR=...] [DATASET_FILE=...]
	@if [ -z "$(DATASET_ID)" ]; then \
		echo "❌ Error: DATASET_ID is required"; \
		echo ""; \
		echo "Usage: make dataset-materialize DATASET_ID=curated_5feeds_smoke_v1"; \
		echo ""; \
		echo "Optional parameters:"; \
		echo "  OUTPUT_DIR=path/to/output    Output directory (default: data/eval/materialized)"; \
		echo "  DATASET_FILE=path/to/file.json    Dataset JSON file (default: data/eval/datasets/{DATASET_ID}.json)"; \
		exit 1; \
	fi
	@OUTPUT_DIR=$${OUTPUT_DIR:-data/eval/materialized}; \
	echo "Materializing dataset: $(DATASET_ID)"; \
	echo "  Output directory: $$OUTPUT_DIR"; \
	cmd="$(PYTHON) scripts/eval/materialize_dataset.py --dataset-id $(DATASET_ID) --output-dir $$OUTPUT_DIR"; \
	if [ -n "$(DATASET_FILE)" ]; then \
		echo "  Dataset file: $(DATASET_FILE)"; \
		cmd="$$cmd --dataset-file $(DATASET_FILE)"; \
	fi; \
	eval $$cmd
	@echo ""
	@echo "✓ Dataset materialized: $$OUTPUT_DIR/$(DATASET_ID)/"


run-promote:
	@# Promote a run to baseline or reference
	@# Usage: make run-promote RUN_ID=run_2026-01-16_11-52-03 --as baseline PROMOTED_ID=baseline_prod_authority_v2 REASON="New production baseline" [RENAME_TO=baseline_ml_dev_authority_smoke_v1]
	@if [ -z "$(RUN_ID)" ]; then \
		echo "❌ Error: RUN_ID is required"; \
		echo ""; \
		echo "Usage: make run-promote RUN_ID=run_2026-01-16_11-52-03 --as baseline PROMOTED_ID=baseline_prod_authority_v2 REASON=\"...\" [RENAME_TO=...]"; \
		echo ""; \
		echo "For baselines:"; \
		echo "  make run-promote RUN_ID=run_xxx --as baseline PROMOTED_ID=baseline_prod_v2 REASON=\"...\" [RENAME_TO=baseline_new_name]"; \
		echo ""; \
		echo "For references:"; \
		echo "  make run-promote RUN_ID=run_xxx --as reference PROMOTED_ID=silver_gpt5_v1 REASON=\"...\" [REFERENCE_QUALITY=silver|gold] [RENAME_TO=...]"; \
		echo "    Note: DATASET_ID no longer required. Task type auto-detected from run."; \
		exit 1; \
	fi
	@if [ -z "$(PROMOTED_ID)" ]; then \
		echo "❌ Error: PROMOTED_ID is required"; \
		exit 1; \
	fi
	@if [ -z "$(REASON)" ]; then \
		echo "❌ Error: REASON is required (explains why this run is being promoted)"; \
		exit 1; \
	fi
	@if [ "$(AS)" != "baseline" ] && [ "$(AS)" != "reference" ]; then \
		echo "❌ Error: --as must be 'baseline' or 'reference'"; \
		exit 1; \
	fi
	@cmd="$(PYTHON) scripts/eval/promote_run.py --run-id $(RUN_ID) --as $(AS) --promoted-id $(PROMOTED_ID) --reason \"$(REASON)\""; \
	if [ "$(AS)" = "reference" ]; then \
		# DATASET_ID is optional now (task type auto-detected from run) \
		if [ -n "$(DATASET_ID)" ]; then \
			cmd="$$cmd --dataset-id $(DATASET_ID)"; \
		fi; \
		if [ -n "$(REFERENCE_QUALITY)" ]; then \
			cmd="$$cmd --reference-quality $(REFERENCE_QUALITY)"; \
		fi; \
	fi; \
	if [ -n "$(RENAME_TO)" ]; then \
		cmd="$$cmd --rename-to $(RENAME_TO)"; \
	fi; \
	eval $$cmd
	@echo ""
	@if [ -n "$(RENAME_TO)" ]; then \
		echo "✓ Run promoted to $(AS) and renamed: $(RENAME_TO)"; \
	else \
		echo "✓ Run promoted to $(AS): $(PROMOTED_ID)"; \
	fi

baseline-create:
	@# Materialize a baseline from current system state (creates run then auto-promotes)
	@# Usage: make baseline-create BASELINE_ID=bart_led_baseline_v1 DATASET_ID=indicator_v1 [EXPERIMENT_CONFIG=...] [PREPROCESSING_PROFILE=...] [REFERENCE=ref1,ref2]
	@if [ -z "$(BASELINE_ID)" ]; then \
		echo "❌ Error: BASELINE_ID is required"; \
		echo ""; \
		echo "Usage: make baseline-create BASELINE_ID=bart_led_baseline_v1 DATASET_ID=indicator_v1"; \
		echo ""; \
		echo "Note: This command creates a run then auto-promotes it to a baseline."; \
		echo "      For experiments with full evaluation, use 'make experiment-run' instead."; \
		exit 1; \
	fi
	@if [ -z "$(DATASET_ID)" ]; then \
		echo "❌ Error: DATASET_ID is required"; \
		echo ""; \
		echo "Usage: make baseline-create BASELINE_ID=bart_led_baseline_v1 DATASET_ID=indicator_v1"; \
		exit 1; \
	fi
	@RUN_ID=run_$$(date +%Y-%m-%d_%H-%M-%S); \
	echo "Creating baseline: $(BASELINE_ID)"; \
	echo "  Dataset: $(DATASET_ID)"; \
	echo "  Run ID (temporary): $$RUN_ID"; \
	cmd="$(PYTHON) scripts/eval/materialize_baseline.py --baseline-id $$RUN_ID --dataset-id $(DATASET_ID) --output-dir data/eval/runs"; \
	if [ -n "$(EXPERIMENT_CONFIG)" ]; then \
		echo "  Experiment config: $(EXPERIMENT_CONFIG)"; \
		cmd="$$cmd --experiment-config $(EXPERIMENT_CONFIG)"; \
	fi; \
	if [ -n "$(PREPROCESSING_PROFILE)" ]; then \
		echo "  Preprocessing profile: $(PREPROCESSING_PROFILE)"; \
		cmd="$$cmd --preprocessing-profile $(PREPROCESSING_PROFILE)"; \
	fi; \
	if [ -n "$(REFERENCE)" ]; then \
		echo "  Reference(s): $(REFERENCE)"; \
		for ref in $$(echo $(REFERENCE) | tr ',' ' '); do \
			cmd="$$cmd --reference $$ref"; \
		done; \
	fi; \
	eval $$cmd; \
	echo ""; \
	echo "Auto-promoting run to baseline..."; \
	$(MAKE) run-promote RUN_ID=$$RUN_ID AS=baseline PROMOTED_ID=$(BASELINE_ID) REASON="baseline-create command (auto-promoted)"; \
	echo ""; \
	echo "✓ Baseline created: data/eval/baselines/$(BASELINE_ID)/"

experiment-run:
	@# Run an experiment using a config file (complete evaluation loop: runner + scorer + comparator)
	@# Usage: make experiment-run CONFIG=data/eval/configs/my_experiment.yaml [BASELINE=baseline_id] [REFERENCE=ref_id1,ref_id2] [LOG_LEVEL=INFO] [DRY_RUN=1] [SCORE_ONLY=1]
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Error: CONFIG is required"; \
		echo ""; \
		echo "Usage: make experiment-run CONFIG=data/eval/configs/my_experiment.yaml"; \
		echo ""; \
		echo "Optional parameters:"; \
		echo "  BASELINE=baseline_id              Baseline ID for comparison (optional but recommended)"; \
		echo "  REFERENCE=ref_id1,ref_id2         Reference IDs for evaluation (comma-separated, can be silver/gold)"; \
		echo "  LOG_LEVEL=INFO|DEBUG|WARNING|ERROR    Logging level (default: INFO)"; \
		echo "  DRY_RUN=1                         Dry run mode: generate predictions only, skip metrics/comparison"; \
		echo "  SMOKE_INFERENCE_ONLY=1            Smoke test mode: same as DRY_RUN"; \
		echo "  SCORE_ONLY=1                      Score-only mode: skip inference, use existing predictions.jsonl"; \
		echo "  FORCE=1                           Delete existing run directory before starting (useful for re-runs)"; \
		exit 1; \
		fi
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "❌ Error: Config file not found: $(CONFIG)"; \
		exit 1; \
	fi
	@echo "Running experiment: $(CONFIG)"
	@cmd="$(PYTHON) scripts/eval/run_experiment.py $(CONFIG)"; \
	if [ -n "$(BASELINE)" ]; then \
		echo "  Baseline: $(BASELINE)"; \
		cmd="$$cmd --baseline $(BASELINE)"; \
	fi; \
	if [ -n "$(REFERENCE)" ]; then \
		echo "  References: $(REFERENCE)"; \
		for ref in $$(echo $(REFERENCE) | tr ',' ' '); do \
			cmd="$$cmd --reference $$ref"; \
		done; \
	fi; \
	if [ -n "$(LOG_LEVEL)" ]; then \
		echo "  Log level: $(LOG_LEVEL)"; \
		cmd="$$cmd --log-level $(LOG_LEVEL)"; \
	fi; \
	if [ "$(DRY_RUN)" = "1" ] || [ "$(SMOKE_INFERENCE_ONLY)" = "1" ]; then \
		echo "  Mode: dry-run (predictions only, skip metrics/comparison)"; \
		cmd="$$cmd --dry-run"; \
	fi; \
	if [ "$(FORCE)" = "1" ]; then \
		echo "  Force: deleting existing run directory"; \
		cmd="$$cmd --force"; \
	fi; \
	if [ "$(SCORE_ONLY)" = "1" ]; then \
		echo "  Mode: score-only (skip inference, use existing predictions)"; \
		cmd="$$cmd --score-only"; \
	fi; \
	eval $$cmd
	@echo ""
	@echo "✓ Experiment completed. Check data/eval/runs/ directory for output."

run-freeze:
	@# Freeze a run for baseline comparison (moves to _frozen_pre_cleanup/)
	@# Usage: make run-freeze RUN_ID=baseline_bart_small_led_long_fast [REASON="Pre-cleanup baseline"]
	@if [ -z "$(RUN_ID)" ]; then \
		echo "❌ Error: RUN_ID is required"; \
		echo "Usage: make run-freeze RUN_ID=baseline_bart_small_led_long_fast"; \
		echo ""; \
		echo "Available runs:"; \
		ls -d data/eval/runs/*/ 2>/dev/null | grep -v '_frozen' | xargs -I {} basename {} | sed 's/^/  /'; \
		exit 1; \
	fi
	@if [ ! -d "data/eval/runs/$(RUN_ID)" ]; then \
		echo "❌ Error: Run not found: data/eval/runs/$(RUN_ID)"; \
		exit 1; \
	fi
	@frozen_dir="data/eval/runs/_frozen_pre_cleanup"; \
	mkdir -p "$$frozen_dir"; \
	if [ -d "$$frozen_dir/$(RUN_ID)" ]; then \
		echo "❌ Error: Frozen run already exists: $$frozen_dir/$(RUN_ID)"; \
		exit 1; \
	fi; \
	mv "data/eval/runs/$(RUN_ID)" "$$frozen_dir/$(RUN_ID)"; \
	reason="$${REASON:-Frozen pre-cleanup baseline candidate for comparison.}"; \
	echo "# Frozen Run: $(RUN_ID)" > "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "**Status**: Frozen" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "**Frozen Date**: $$(date +%Y-%m-%d)" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "## Purpose" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "$$reason" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "You can also just rename the folder so you don't accidentally overwrite it." >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "This run will let you quantify improvement after cleanup:" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "- repetition down" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "- garbage tokens down" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "- coherence up" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "- speaker label leakage down" >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo "- etc." >> "$$frozen_dir/$(RUN_ID)/NOTE.md"; \
	echo ""; \
	echo "✓ Frozen run: $(RUN_ID)"; \
	echo "  Location: $$frozen_dir/$(RUN_ID)/"; \
	echo "  NOTE.md created with freeze reason"

runs-delete:
	@# Delete experiment runs (use with caution!)
	@# Usage: make runs-delete RUN_IDS="run1 run2 run3"
	@if [ -z "$(RUN_IDS)" ]; then \
		echo "❌ Error: RUN_IDS is required"; \
		echo "Usage: make runs-delete RUN_IDS=\"run1 run2 run3\""; \
		echo ""; \
		echo "Available runs:"; \
		ls -d data/eval/runs/*/ 2>/dev/null | grep -v '_frozen' | xargs -I {} basename {} | sed 's/^/  /'; \
		exit 1; \
	fi
	@for run_id in $(RUN_IDS); do \
		if [ -d "data/eval/runs/$$run_id" ]; then \
			rm -rf "data/eval/runs/$$run_id"; \
			echo "  Deleted: $$run_id"; \
		else \
			echo "  Skipped (not found): $$run_id"; \
		fi; \
	done; \
	echo ""; \
	echo "✓ Deletion complete"

configs-archive:
	@# Archive all experiment configs to a dated folder
	@# Usage: make configs-archive [NAME=param_sweeps] [DATE=2026-01-30]
	@# Defaults: NAME=experiment_configs, DATE=today
	@archive_name=$${NAME:-experiment_configs}; \
	archive_date=$${DATE:-$$(date +%Y-%m-%d)}; \
	archive_dir="data/eval/configs/_archive/$${archive_name}_$${archive_date}"; \
	configs=$$(ls data/eval/configs/baseline_*.yaml 2>/dev/null | grep -v '_archive'); \
	if [ -z "$$configs" ]; then \
		echo "❌ No experiment configs found to archive (baseline_*.yaml)"; \
		exit 1; \
	fi; \
	echo "Archiving configs to: $$archive_dir"; \
	mkdir -p "$$archive_dir"; \
	for cfg in $$configs; do \
		cp "$$cfg" "$$archive_dir/"; \
		echo "  Archived: $$(basename $$cfg)"; \
	done; \
	config_count=$$(ls "$$archive_dir"/*.yaml 2>/dev/null | wc -l | tr -d ' '); \
	echo ""; \
	echo "✓ Archived $$config_count config(s) to $$archive_dir"; \
	echo ""; \
	echo "To restore later:"; \
	echo "  cp $$archive_dir/*.yaml data/eval/configs/"

configs-clean:
	@# Remove experiment configs from main folder (keeps README and examples)
	@# Usage: make configs-clean [KEEP=baseline_bart_small_led_long_fast.yaml]
	@# Use after configs-archive to clean up
	@keep_file="$${KEEP:-}"; \
	configs=$$(ls data/eval/configs/baseline_*.yaml 2>/dev/null); \
	if [ -z "$$configs" ]; then \
		echo "No experiment configs to clean"; \
		exit 0; \
	fi; \
	for cfg in $$configs; do \
		if [ -n "$$keep_file" ] && [ "$$(basename $$cfg)" = "$$keep_file" ]; then \
			echo "  Keeping: $$(basename $$cfg)"; \
		else \
			rm "$$cfg"; \
			echo "  Removed: $$(basename $$cfg)"; \
		fi; \
	done; \
	echo ""; \
	echo "✓ Configs cleaned. Remaining:"; \
	ls data/eval/configs/*.yaml 2>/dev/null || echo "  (none)"

runs-list:
	@# List all experiment runs
	@# Usage: make runs-list [DATASET_ID=dataset_id]
	@$(PYTHON) scripts/eval/list_runs.py \
		$(if $(DATASET_ID),--dataset-id $(DATASET_ID))

baselines-list:
	@# List all baselines
	@# Usage: make baselines-list [DATASET_ID=dataset_id]
	@$(PYTHON) scripts/eval/list_runs.py --baselines \
		$(if $(DATASET_ID),--dataset-id $(DATASET_ID))

runs-compare:
	@# Compare two experiment runs
	@# Usage: make runs-compare RUN1=run_id1 RUN2=run_id2 [DATASET_ID=dataset_id] [OUTPUT=path/to/report.md]
	@if [ -z "$(RUN1)" ] || [ -z "$(RUN2)" ]; then \
		echo "❌ Error: RUN1 and RUN2 are required"; \
		echo "Usage: make runs-compare RUN1=run_id1 RUN2=run_id2"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/eval/compare_runs.py \
		--run1 $(RUN1) \
		--run2 $(RUN2) \
		$(if $(DATASET_ID),--dataset-id $(DATASET_ID)) \
		$(if $(OUTPUT),--output $(OUTPUT))

benchmark:
	@# Run benchmark across multiple datasets
	@# Usage: make benchmark CONFIG=config.yaml BASELINE=baseline_id [SMOKE=1|ALL=1|DATASETS=ds1,ds2] [REFERENCE=ref1,ref2] [OUTPUT_DIR=...]
	@if [ -z "$(CONFIG)" ] || [ -z "$(BASELINE)" ]; then \
		echo "❌ Error: CONFIG and BASELINE are required"; \
		echo "Usage: make benchmark CONFIG=data/eval/configs/my_experiment.yaml BASELINE=baseline_id"; \
		echo ""; \
		echo "Options:"; \
		echo "  SMOKE=1              Run on smoke test datasets only"; \
		echo "  ALL=1                Run on all available datasets"; \
		echo "  DATASETS=ds1,ds2     Run on specific datasets (comma-separated)"; \
		echo "  REFERENCE=ref1,ref2  Reference IDs for evaluation (comma-separated)"; \
		echo "  OUTPUT_DIR=...       Custom output directory"; \
		exit 1; \
	fi
	@if [ ! -f "$(CONFIG)" ]; then \
		echo "❌ Error: Config file not found: $(CONFIG)"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/eval/run_benchmark.py \
		--config $(CONFIG) \
		--baseline $(BASELINE) \
		$(if $(SMOKE),--smoke) \
		$(if $(ALL),--all) \
		$(if $(DATASETS),--datasets $(DATASETS)) \
		$(if $(REFERENCE),--reference $(REFERENCE)) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR)) \
		$(if $(LOG_LEVEL),--log-level $(LOG_LEVEL))
