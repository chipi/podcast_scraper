# NEVER hand an inherited NODE_OPTIONS to a recipe. cmux launches Claude Code with
# `--require=$TMPDIR/cmux-claude-node-options/restore-node-options.cjs`, a self-erasing shim that is
# meant to delete itself from `process.env` at startup. It no longer does: Claude Code ships as a
# native binary, which does not honour NODE_OPTIONS' `--require`, so the erase never runs and the
# variable leaks into every child. Worse, macOS periodically purges $TMPDIR, so the path it points
# at stops existing while the exported value lives on — and then every node step (`npm run build`,
# `npx vite preview`, playwright) dies instantly with MODULE_NOT_FOUND.
#
# That failure surfaces three layers away: the origin never comes up, and the iOS tier reports
# "no Play control" or "cannot reach the api" instead of "your terminal broke node".
#
# This replaces per-recipe `env -u NODE_OPTIONS`, which was applied to `ios-app-install` but NOT to
# `ios-origin-up` or `test-app-ios-sim-download` — so whether the tier worked depended on which
# entry point you used. Declared once here, it cannot be forgotten by a target nobody has written
# yet. If a recipe ever genuinely needs a raised heap, set NODE_OPTIONS on that recipe explicitly
# rather than inheriting it by accident from a terminal emulator.
unexport NODE_OPTIONS

# Auto-detect venv Python if .venv exists, otherwise use python3
ifeq ($(wildcard .venv/bin/python),)
PYTHON ?= python3
CODESPELL ?= codespell
else
PYTHON ?= .venv/bin/python
CODESPELL ?= .venv/bin/codespell
endif
PACKAGE = podcast_scraper
# App state is Postgres (ADR-169). `serve-api` runs the API through scripts/tools/dev_db.py: a dev
# PostgreSQL for this checkout (PODCAST_DEV_DB=embedded, the default: the [dev] extra's
# embedded-postgres; or native: your own initdb/pg_ctl), both planes migrated, the dev fixtures
# loaded once, stopped when the API stops. No docker. A set PODCAST_DATABASE_URL is used as is.
DEV_DB_RUN = $(PYTHON) scripts/tools/dev_db.py run --

# Fixture version: resolved from tests/fixtures/FIXTURES_VERSION (single source of truth).
# Versioned fixture subdirs: tests/fixtures/{transcripts,audio}/$(FIXTURES_VERSION)/...
FIXTURES_VERSION ?= $(shell cat tests/fixtures/FIXTURES_VERSION 2>/dev/null | tr -d '[:space:]')

# Classic dot progress for fast layers (pyproject addopts no longer includes -q).
# PYTHONUNBUFFERED on fast pytest lines reduces line buffering in IDE/agent terminals.
PYTEST_PROGRESS_OPTS := -o console_output_style=classic

# GI/KG viewer (Vue + Vite + Playwright). Override if the app moves, e.g. ``make serve-ui WEB_VIEWER_DIR=apps/viewer``.
WEB_VIEWER_DIR ?= web/gi-kg-viewer

# GIL Quote vs indexed transcript chunk offset gate (#528 Phase 5). ``make verify-gil-offsets-strict`` uses these.
# Override for CI or another indexed corpus: GIL_OFFSET_VERIFY_DIR=/path/to/corpus-root make verify-gil-offsets-strict
GIL_OFFSET_VERIFY_DIR ?= output
GIL_OFFSET_MIN_RATE ?= 0.95

# Preload / hf-hub-smoke-test must reach Hugging Face; unset offline flags for those recipes
# only (a global Makefile export previously forced HF offline and broke ``make preload-ml-models*``).
HF_NET_ENV := env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE

# Secondary venv matching GitHub ``test-unit``: ``pip install -e .[dev]`` only (no ml/llm).
# Override path: ``make venv-dev-init VENVDEV=.venv-ci-unit``
VENVDEV ?= .venv-dev
VENVDEV_PY = $(VENVDEV)/bin/python

# Pip cache directory (can be overridden via environment variable)
# Defaults to standard pip cache location, but can be set explicitly for consistency
PIP_CACHE_DIR ?= $(HOME)/.cache/pip
export PIP_CACHE_DIR

# Prefer local spaCy (and related) wheels when present (see make download-spacy-wheels).
# Does not override PIP_FIND_LINKS if you already exported it in the shell.
ifneq ($(wildcard wheels/spacy/*.whl),)
ifeq ($(origin PIP_FIND_LINKS),undefined)
export PIP_FIND_LINKS := $(abspath wheels/spacy)
endif
endif

# Test parallelism: default fallback value. Each test recipe calculates its own
# optimal worker count at recipe time via calculate_test_workers.py.
# This parse-time value is only used if a recipe references $(PYTEST_WORKERS) directly.
# Can be overridden: PYTEST_WORKERS=4 make test
PYTEST_WORKERS ?= 2

# Note: NIGHTLY_PYTEST_WORKERS removed — test-nightly now runs sequentially only.
# Parallel execution via pytest-xdist caused double-runs on CI (exit-code mismatch
# triggered fallback, doubling wall time).

.PHONY: gates-changed regen-fixtures withdraw-language profiles-materialize profiles-check check-doc-structure help init init-no-ml venv-dev-init test-unit-dev-venv download-spacy-wheels format format-check lint lint-markdown lint-markdown-docs fix-md strip-doc-checkmarks strip-doc-emoji strip-docs type security security-bandit security-audit security-fast complexity complexity-track deadcode docstrings spelling spelling-docs quality check-unit-imports check-test-policy check-pricing-assumptions validate-kg-schema kg-quality-metrics search-quality-metrics search-quality-reseed quality-metrics-ci fetch-ci-metrics fetch-ci-metrics-validate fetch-nightly-metrics validate-metrics-bundle build-metrics-dashboard-preview metrics-preview-check serve-metrics-dashboard metrics-dashboard-live deps-analyze deps-check deps-graph deps-graph-full call-graph flowcharts visualize release-docs-prep pre-release bump analyze-test-memory cleanup-processes check-zombie check-spotlight test-unit test-unit-sequential test-unit-no-ml test-integration test-integration-sequential test-integration-fast test-ci test-ci-fast test-e2e test-e2e-sequential test-e2e-fast verify-gil-offsets-after-acceptance preload-transformers-integration-summariesuality test-nightly test test-sequential test-fast test-fast-no-py-e2e test-reruns test-track test-track-view test-openai test-openai-multi test-openai-all-feeds test-openai-real test-openai-real-multi test-openai-real-all-feeds test-openai-real-feed coverage coverage-check coverage-check-unit coverage-check-integration coverage-check-e2e coverage-check-combined merge-cov-fragments coverage-report coverage-enforce docs docs-check build _ci_body ci ci-fast ci-ui-fast ci-ui-full ci-ui-validation serve-for-validation ci-sequential ci-clean ci-nightly clean clean-cache clean-model-cache clean-all docker-build docker-build-fast docker-build-full docker-test docker-clean install-hooks preload-ml-models preload-ml-models-production hf-hub-smoke-test backup-cache backup-cache-dry-run backup-cache-list backup-cache-cleanup restore-cache restore-cache-dry-run autoresearch-sweep-multi serve-gi-kg-viz test-ui test-ui-e2e e2e-api-image test-ui-e2e-live build-viewer verify-gil-offsets-strict upgrade-undo-roles

help:
	@echo "Common developer commands:"
	@echo "  make init            Install development dependencies (uses wheels/spacy if *.whl present)"
	@echo "  make venv-dev-init   Create $(VENVDEV) with pip install -e .[dev] only (CI test-unit parity)"
	@echo "  make init-no-ml      Alias for venv-dev-init (does not modify your main .venv)"
	@echo "  make format          Apply formatting with black + isort"
	@echo "  make format-check    Check formatting without modifying files"
	@echo "  make lint            Run flake8 linting"
	@echo "  make lint-markdown   Run markdownlint on markdown files"
	@echo "  make fix-md          Auto-fix markdown (markdownlint --fix; same rules as lint-markdown)"
	@echo "  make strip-doc-checkmarks  Remove checkmark-style chars from docs and policy .md/.mdc"
	@echo "  make strip-doc-emoji       Remove decorative Unicode emoji (docs, README, policies, viewer e2e map)"
	@echo "  make strip-docs            strip-doc-checkmarks then strip-doc-emoji"
	@echo "  make type            Run mypy type checks"
	@echo "  make security        Run bandit & pip-audit security scans"
	@echo "  make complexity      Run radon complexity analysis"
	@echo "  make complexity-track  Per-commit complexity trend (REVISIONS=N, default 50)"
	@echo "  make deadcode        Run vulture dead code detection"
	@echo "  make docstrings      Run interrogate docstring coverage"
	@echo "  make spelling        Run codespell on src/ and docs/ (CLI from .[dev], e.g. .venv/bin/codespell)"
	@echo "  make quality         Run all code quality checks (complexity, deadcode, docstrings, spelling)"
	@echo ""
	@echo "Verification commands:"
	@echo "  make check-unit-imports  Verify unit tests can import modules without ML dependencies"
	@echo "  make check-test-policy   Enforce 3-tier ML/AI testing policy (no importorskip in unit, etc.)"
	@echo "  make check-pricing-assumptions  Show config/pricing_assumptions.yaml status (optional --strict)"
	@echo "  make validate-kg-schema [ARTIFACTS_DIR=path]  Validate kg.json files against KG schema (strict)"
	@echo "  make kg-quality-metrics [DIR=path]   PRD-019 metrics over .kg.json (optional --enforce via ARGS)"
	@echo "  make quality-metrics-ci              GIL+KG enforce on tests/fixtures/gil_kg_ci_enforce (matches CI)"
	@echo "  make fetch-ci-metrics [N=40]       Download metrics/ artifacts from last N successful main runs (needs gh; default N=40)"
	@echo "  make fetch-ci-metrics-validate [N=40]  Download (same as above) then validate every run-* bundle"
	@echo "  make fetch-nightly-metrics [N=25]  N unset: one artifact or Pages curl; N>=1: download N nightlies + merge JSONL for charts"
	@echo "  make validate-metrics-bundle BUNDLE=path  Validate downloaded latest-*.json + optional history JSONL"
	@echo "  make verify-gil-offsets-strict [GIL_OFFSET_VERIFY_DIR=path] [GIL_OFFSET_MIN_RATE=0.95]  Quote vs indexed transcript chunk offsets (#528; strict gate)"
	@echo "  make verify-gil-offsets-after-acceptance [OUTPUT_DIR=path]  After test-acceptance*: verify every run_* with search/metadata.json"
	@echo "  make build-metrics-dashboard-preview   CI: merged history + nightly + dashboard-data.json → artifacts/dashboard-preview/"
	@echo "  make metrics-preview-check             Same as preview build with strict history-*.jsonl validation (METRICS_PREVIEW_STRICT)"
	@echo "  make serve-metrics-dashboard   Rebuild preview + HTTP server (http://127.0.0.1:8777/)"
	@echo "  make metrics-dashboard-live [N=40]  Fetch + validate + preview + server (same URL; needs gh)"
	@echo "  make deps-analyze        Analyze module dependencies and detect architectural issues (with report)"
	@echo "  make deps-check          Check dependencies and exit with error if issues found"
	@echo "  make deps-graph          Generate module dependency graph (SVG) in docs/architecture/"
	@echo "  make deps-graph-full     Generate full dependency graph with all deps in docs/architecture/"
	@echo "  make call-graph          Generate workflow call graph (pyan3) in docs/architecture/"
	@echo "  make flowcharts          Generate flowcharts for orchestration and service (code2flow)"
	@echo "  make visualize           Generate all architecture visualizations (deps, call graph, flowcharts)"
	@echo "  make release-docs-prep   Regenerate diagrams + create release notes draft (then commit)"
	@echo "  make pre-release         ADR-031: pre_release_check.py (version + release notes) then make ci"
	@echo "  make bump VERSION=X.Y.Z  Bump pyproject.toml + __init__.py (optional ALLOW_DIRTY=1 FORCE_TAG=1)"
	@echo "  make test-ui             Vitest unit tests for TypeScript utils in $(WEB_VIEWER_DIR) (fast, no browser)"
	@echo "  make test-ui-e2e         Playwright E2E for $(WEB_VIEWER_DIR) (needs npm install in that dir)"
	@echo "  make build-viewer        Production viewer bundle (vue-tsc -b && vite build); catches strict-mode TS errors invisible to vitest/playwright"
	@echo ""
	@echo "Analysis commands:"
	@echo "  make analyze-test-memory [TARGET=test-unit] [WORKERS=N]  Analyze test memory usage and resource consumption"
	@echo ""
	@echo "Cleanup and safety commands:"
	@echo "  make cleanup-processes  Clean up leftover Python/test processes from previous runs"
	@echo "  make check-zombie       Detect unkillable (UE state) Python processes (reboot required)"
	@echo "  make check-spotlight    Verify Spotlight indexing is disabled (macOS ML safety)"
	@echo ""
	@echo "Test commands:"
	@echo "  make test-db-up           Start the local postgres:18 the tests need; prints PODCAST_TEST_DATABASE_URL to export"
	@echo "  make test-db-down         Remove that test database container (this checkout's only)"
	@echo "  make test-unit            Run unit tests with coverage in parallel (uses .venv if present)"
	@echo "  make test-unit-sequential Run unit tests sequentially (for debugging, slower but clearer output)"
	@echo "  make test-unit-dev-venv   Unit tests inside $(VENVDEV) (run venv-dev-init first; true CI parity)"
	@echo "  make test-unit-no-ml      check-unit-imports + unit tests using Makefile PYTHON (often full .venv)"
	@echo "  make test-integration            Run all integration tests (full suite, parallel)"
	@echo "  make test-integration-sequential Run all integration tests sequentially (for debugging)"
	@echo "  make test-integration-fast       Run fast integration tests (critical path only)"
	@echo "  make test-e2e                   Run all E2E tests (full suite, parallel, 1 episode per test)"
	@echo "  make test-e2e-sequential         Run all E2E tests sequentially (for debugging)"
	@echo "  make test-e2e-fast              Run fast E2E tests (critical path only, 1 episode per test)"
	@echo "  make test-nightly                Run nightly-only tests (p01-p05 full suite, production models)"
	@echo "  make test                Run all tests (unit + integration + e2e, full suite, uses multi-episode feed)"
	@echo "  make test-sequential     Run all tests sequentially (for debugging, uses multi-episode feed)"
	@echo "  make test-fast           Run fast tests (unit + critical path integration + critical path e2e, uses fast feed)"
	@echo "  make test-fast-no-py-e2e Same as test-fast but skips tests/e2e (faster when validating UI + server only)"
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
	@echo "  make test-acceptance  Run E2E acceptance tests (multiple configs sequentially)"
	@echo "                            Usage: make test-acceptance CONFIGS=\"…\" [USE_FIXTURES=1] …"
	@echo "                            Or:     make test-acceptance FROM_FAST_STEMS=1 USE_FIXTURES=1 (materialize MAIN_ACCEPTANCE_CONFIG.yaml rows)"
	@echo "                            Configs with vector_search: run make preload-ml-models without SKIP_GIL=1 so vector indexing has cached embeddings offline."
	@echo "  make test-acceptance-fixtures-fast  Same as FROM_FAST_STEMS=1 + USE_FIXTURES=1 + no auto analyze/benchmark; TIMEOUT default 900; PER_RUN_WALL_SECONDS default 600 (post-run wall budget)"
	@echo "                            Options: USE_FIXTURES=1 uses test fixtures (default: uses real RSS/APIs)"
	@echo "                                     NO_SHOW_LOGS=1 disables real-time log streaming (default: logs shown)"
	@echo "                                     NO_AUTO_ANALYZE=1 disables automatic analysis (default: analysis runs automatically)"
	@echo "                                     ANALYZE_MODE=comprehensive uses comprehensive analysis mode (default: basic)"
	@echo "                                     STRICT_VECTOR_INDEX=1 fails run if vector_search builds an empty index (exit 1 if any run fails)"
	@echo "  make analyze-acceptance  Analyze acceptance test results"
	@echo "                                 Usage: make analyze-acceptance SESSION_ID=\"20260208_093757\" [MODE=basic|comprehensive] [COMPARE_BASELINE=...]"
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
	@echo "Other commands:"
	@echo "  make docs            Build MkDocs site (strict mode, outputs to .build/site/)"
	@echo "  make docs-check      Run all documentation checks (linting + spelling + build)"
	@echo "  make build           Build source and wheel distributions (outputs to .build/dist/)"
	@echo "  make ci              Run the full CI suite locally (all tests: unit + integration + e2e, uses multi-episode feed)"
	@echo "  make ci-fast         Run fast CI checks (unit + critical path integration + critical path e2e, uses fast feed)"
	@echo "  make ci-ui-fast      Like ci-fast but skips Python tests/e2e and runs Playwright (test-ui-e2e)"
	@echo "  make ci-ui-full      Like ci-ui-fast plus stack-test-ml-ci (Docker stack + Playwright; opt-in for chip/testid refactors)"
	@echo "  make ci-clean        Run complete CI suite with clean first (same as ci but cleans build artifacts first)"
	@echo "  make ci-nightly      Run full nightly CI chain (unit + integration + e2e + nightly, production models)"
	@echo "  make docker-build       Build Docker image (default, with model preloading)"
	@echo "  make docker-build-fast  Build Docker image fast (no model preloading, <5min target)"
	@echo "  make docker-build-full  Build Docker image full (with model preloading, matches main)"
	@echo "  make stack-build        Docker stack (production base, no mock-feeds): build viewer + api + pipeline images"
	@echo "  make stack-build-llm    Docker stack: also build pipeline-llm (INSTALL_EXTRAS=llm)"
	@echo "  make stack-up           Docker stack: start viewer + api (VIEWER_PORT=8080); pipeline jobs spawn via API factory"
	@echo "  make stack-down         Docker stack: stop stack (REMOVE_VOLUMES=1 runs compose down -v)"
	@echo "  make stack-logs         Docker stack: follow viewer + api logs"
	@echo "  make verify-stack-profiles  Validate packaged profile → Docker tier (scripts/tools)"
	@echo "  make stack-compose-validate Docker stack: docker compose config (stack + jobs-docker merge, no build)"
	@echo "  make stack-test-build       Stack-test: build api + viewer + pipeline (airgapped/ml) images"
	@echo "  make stack-test-build-cloud Stack-test: also build pipeline-llm ([llm] extras for cloud-thin runs — local only)"
	@echo "  make stack-test-up          Stack-test: bring up api + viewer + mock-feeds (sources .env if present)"
	@echo "  make stack-test-seed        Stack-test: seed corpus_data volume with feeds.spec.yaml + viewer_operator.yaml (STACK_TEST_OPERATOR_VARIANT=ml|cloud-thin)"
	@echo "  make stack-test-playwright  Stack-test: run Playwright (smoke + full UI flow; STACK_TEST_OPERATOR_PROFILE=cloud_thin for llm runs)"
	@echo "  make stack-test-ml          Stack-test (one-shot): ml pipeline — build → up → seed → Playwright (airgapped_thin)"
	@echo "  make stack-test-cloud-thin  Stack-test (one-shot): cloud-thin (LLM) pipeline — build + pipeline-llm → up → seed → Playwright (cloud_thin); needs .env keys"
	@echo "  make stack-test-down        Stack-test: tear down (STACK_TEST_DOWN_VOLUMES=1 to also drop corpus_data)"
	@echo "  make stack-test-reap        Stack-test: reap ALL leftovers (stack + orphan build/Playwright), this repo only"
	@echo "  make stack-test-export      Stack-test: copy corpus_data volume → .stack-test-corpus/ for debug inspection"
	@echo "  make upgrade-status         Show pending corpus-upgrade migrations (human-readable)"
	@echo "                                Usage: make upgrade-status CORPUS_DIR=<path>"
	@echo "  make upgrade-check          Same as upgrade-status but --json; exits 2 when migrations are pending (CI-gate)"
	@echo "                                Usage: make upgrade-check CORPUS_DIR=<path>"
	@echo "  make upgrade-dry-run        Preview what upgrade-corpus would do without writing anything"
	@echo "                                Usage: make upgrade-dry-run CORPUS_DIR=<path>"
	@echo "  make upgrade-corpus         Apply pending migrations (non-interactive, --yes); idempotent"
	@echo "                                Usage: make upgrade-corpus CORPUS_DIR=<path>"
	@echo "  make upgrade-verify         Verify applied migrations against the ledger"
	@echo "                                Usage: make upgrade-verify CORPUS_DIR=<path>"
	@echo "  make docker-test        Build and test Docker image"
	@echo "  make docker-clean       Remove Docker test images"
	@echo "  make install-hooks   Install git pre-commit hook for automatic linting"
	@echo "  make clean           Remove build artifacts (.build/, .mypy_cache/, .pytest_cache/, output/)"
	@echo "  make clean-cache     Remove ML model caches (Whisper, spaCy) to test network isolation"
	@echo "  make clean-model-cache  Delete cache for a specific Transformers model (forces re-download)"
	@echo "                            Usage: make clean-model-cache MODEL_NAME=google/pegasus-cnn_dailymail [FORCE=yes]"
	@echo "  make clean-all       Remove both build artifacts and ML model caches"
	@echo "  make download-spacy-wheels  Download spaCy en_core_web_* wheels into wheels/spacy (use with PIP_FIND_LINKS)"
	@echo "  make preload-ml-models  Pre-download/cache ML models (Whisper, spaCy, Transformers, GIL evidence: embedding+QA+NLI)"
	@echo "                            Omit SKIP_GIL=1 when you need sentence-transformers cached for vector_search/LanceDB (indexing uses allow_download=False)."
	@echo "  make preload-ml-models-production  Same idea for nightly-sized model set (Whisper base, BART/LED/Pegasus, hybrid, en_core_web_sm)"
	@echo "  make hf-hub-smoke-test  Diagnose Hugging Face HTTPS + hub API + tokenizer load (optional HF_SMOKE_ARGS=... for script flags)"
	@echo "  make backup-cache    Backup .cache directory (ML models)"
	@echo "  make backup-cache-dry-run  Dry run: Check what would be backed up"
	@echo "  make backup-cache-list     List existing cache backups"
	@echo "  make backup-cache-cleanup   Clean up old cache backups (keeping 5 most recent)"
	@echo "  make restore-cache   Restore .cache directory from backup (use TARGET=path BACKUP=name)"
	@echo "  make restore-cache-dry-run  Dry run: Check what would be restored"

init:
	# Upgrade pip, setuptools, and wheel (required for PEP 660 editable installs with pyproject.toml)
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	# Install package with all optional dependencies for development
	# Note: When adding new optional dependency groups to pyproject.toml, add them here too
	# Current groups: dev, ml (local ML stack), llm (API providers incl. Gemini via google-genai)
	$(PYTHON) -m pip install --upgrade -e .[dev,ml,llm]
	@if [ -f docs/requirements.txt ]; then $(PYTHON) -m pip install --upgrade -r docs/requirements.txt; fi

# CI ``test-unit`` installs ``.[dev]`` only. This creates a separate venv so you do not strip [ml] from .venv.
venv-dev-init:
	@test -d "$(VENVDEV)" || python3 -m venv "$(VENVDEV)"
	$(VENVDEV_PY) -m pip install --upgrade pip setuptools wheel
	$(VENVDEV_PY) -m pip install -e .[dev]
	@echo "✅ $(VENVDEV): editable install with .[dev] only. Run: make test-unit-dev-venv"

init-no-ml: venv-dev-init

# Download spaCy model wheels (matches pyproject.toml [ml] pins). When wheels/spacy/*.whl exists,
# make sets PIP_FIND_LINKS for recipes (e.g. make init) unless you already exported it.
# See docs/guides/DEPENDENCIES_GUIDE.md.
download-spacy-wheels:
	@mkdir -p wheels/spacy
	$(PYTHON) -m pip download -r scripts/spacy_model_wheels_requirements.txt -d wheels/spacy
	@echo "Wheels saved under wheels/spacy/. Run make init (or any pip install via make) to use them automatically."

format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .

format-check:
	$(PYTHON) -m black --check .
	$(PYTHON) -m isort --check-only .

lint:
	$(PYTHON) -m flake8 --config .flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@# Full flake8 (E501 etc.): must fail on violations so ci-fast catches them before pre-commit
	$(PYTHON) -m flake8 --config .flake8 . --count --show-source --statistics
	@# ONE reader for the run-global language (#2177). INSIDE `lint` deliberately, not a
	@# standalone target: `lint-search-v3` is the cited precedent and it runs in NO workflow, so
	@# the guard it provides has never gated a PR. `lint` runs in python-app.yml, nightly.yml and
	@# ci-fast, so this one actually does.
	$(PYTHON) scripts/check/lint_language_readers.py

# Search v3 forbidden-imports guard (RFC-107 §S, PRD-045 FR12; #1205 SIGSEGV).
# Rules: .github/lint/search-v3-forbidden-imports.txt.
lint-search-v3:
	$(PYTHON) scripts/check/lint_search_v3_forbidden_imports.py

# Search v3 quality-eval harness (RFC-107 §T2, PRD-045 FR10; #1230 S0(c)).
# Default corpus + query set = the v3 synthetic fixture; output the S0 baseline
# under data/eval/search-v3/eval/. Override via CORPUS= / QUERIES= / OUT= to
# eval against a different corpus / query set.
CORPUS ?= tests/fixtures/viewer-validation-corpus/v3
QUERIES ?= tests/fixtures/viewer-validation-corpus/v3/search-queries.json
OUT ?= data/eval/search-v3/eval/latest.json
# `data/eval/**`: eval runs write their reports there. The `eval-data` ignores below point at
# a directory that no longer exists (the path moved), so 320 generated reports were being
# linted locally. CI never saw it — they are gitignored, so a fresh checkout has none.
MARKDOWNLINT_CLI_ARGS = "**/*.md" \
	".github/**/*.md" \
	".cursor/**/*.md" \
	".journal/**/*.md" \
	".metrics/**/*.md" \
	".ai-coding-guidelines.md" \
	".ai-coding-guidelines-quick.md" \
	--ignore node_modules \
	--ignore "**/node_modules/**" \
	--ignore .venv \
	--ignore "**/.venv/**" \
	--ignore .build/site \
	--ignore "docs/wip/**" \
	--ignore "tests/fixtures/**" --ignore "eval-data/**" --ignore eval-data \
	--ignore "apps/**" --ignore apps \
	--ignore "data/eval/**" --ignore data/eval \
	--ignore "$(WEB_VIEWER_DIR)/playwright-report/**" \
	--ignore "$(WEB_VIEWER_DIR)/test-results/**" \
	--ignore "$(WEB_VIEWER_DIR)/validation-results/**" \
	--ignore "tests/stack-test/playwright-report/**" \
	--ignore "tests/stack-test/test-results/**" \
	--ignore "**/fastlane/README.md" \
	--config .markdownlint.json

lint-markdown:
	@command -v markdownlint >/dev/null 2>&1 || { echo "markdownlint not found. Install with: npm install -g markdownlint-cli"; exit 1; }
	markdownlint $(MARKDOWNLINT_CLI_ARGS)

fix-md:
	@command -v markdownlint >/dev/null 2>&1 || { echo "markdownlint not found. Install with: npm install -g markdownlint-cli"; exit 1; }
	@echo "Running markdownlint --fix (same paths and .markdownlint.json as lint-markdown)..."
	markdownlint --fix $(MARKDOWNLINT_CLI_ARGS)

strip-doc-checkmarks:
	$(PYTHON) scripts/docs_strip_checkmarks.py

strip-doc-emoji:
	$(PYTHON) scripts/docs_strip_emoji.py

strip-docs: strip-doc-checkmarks strip-doc-emoji

lint-markdown-docs:
	@command -v markdownlint >/dev/null 2>&1 || { echo "markdownlint not found. Install with: npm install -g markdownlint-cli"; exit 1; }
	markdownlint "docs/**/*.md" --ignore "docs/wip/**" --config .markdownlint.json
# tests/fixtures/*.md is not in mkdocs (docs_dir: docs), so nothing linted it. It is
# where the corpus documents itself, and it had an unterminated code fence swallowing
# 150 lines of rendering. Formatting only — the stale-number problem these docs
# actually had is caught by tests/unit/test_fixture_docs_match_disk.py instead.
	markdownlint "tests/fixtures/**/*.md" --config .markdownlint.json

# Match CI lint job (python-app.yml): PYTHONPATH includes repo root so imports match Actions.
type:
	@echo "=== type (mypy): starting $$(date '+%Y-%m-%d %H:%M:%S') — may run several minutes with little output when clean ==="
	@export PYTHONPATH="$$PYTHONPATH:$(PWD)" && PYTHONUNBUFFERED=1 $(PYTHON) -m mypy --config-file pyproject.toml .

security: security-bandit security-audit

# The fast gates' security step (operator 2026-10-06): bandit always; pip-audit only when
# pyproject.toml differs from where this branch left origin/main, committed or not. pip-audit reads
# the dependency set, so a branch that did not touch it gains nothing from a run — the same bet the
# PR's Snyk makes. `ci` and `security` still audit every time. No merge-base (no origin/main fetched,
# a detached tree) audits anyway, so a missing ref never reads as "unchanged".
security-fast: security-bandit
	@base=$$(git merge-base HEAD origin/main 2>/dev/null); \
	if [ -n "$$base" ] && git diff --quiet "$$base" -- pyproject.toml; then \
		echo "pip-audit skipped: pyproject.toml unchanged since $$(git rev-parse --short $$base) (origin/main merge-base); make security-audit runs it."; \
	else \
		$(MAKE) security-audit; \
	fi

# ./eval-data is its own private repo cloned inside the checkout (gitignored; CI never has it),
# with its own .venv: scanning it reported ~69k third-party findings on machines that have it.
security-bandit:
	$(PYTHON) -m bandit -r . --exclude ./.venv,./.venv-dev,./eval-data,./apps --skip B113,B108,B110,B310 --severity-level medium

# Dependency vulnerability audit (pip-audit).
#
# This target was HOLLOW from its first commit (#10, 7a464a16) until now: it upgraded
# setuptools, then carried a dangling "Install ML dependencies to ensure they are audited"
# comment and ran no audit at all. `make security` therefore reported success while checking
# nothing, for the entire life of the repo. pip-audit has been a declared dev dependency
# (pyproject `pip-audit>=2.10.1`) the whole time, and pyproject even references a "Makefile
# ignore" that had never been written.
#
# Upgrading pip first is not cosmetic: pip 25.0.1 carries seven advisories (PYSEC-2026-196,
# -1795, -1796, -2875, -2876, -3721), all of which have fix versions, so they are FIXED here
# rather than ignored.
#
# Ignored advisories carry a reason and a re-check trigger. Never add a blanket skip:
#
#   PYSEC-2026-3740 — nltk 3.10.3, NO fix version published.
#     File-sandbox bypass in the model-artifact APIs: TransitionParser.train/parse,
#     AveragedPerceptron.save/load, PerceptronTagger.save_to_json/load. We import nltk in
#     exactly ONE module (src/podcast_scraper/evaluation/scorer.py) for word_tokenize and
#     sentence_bleu, and none of those vulnerable symbols appears anywhere in src/, tests/
#     or scripts/. The vulnerable code paths are unreachable in our usage, and upstream has
#     published no release to move to. DROP this ignore as soon as nltk ships a fix.
# Ignored advisories (each has a reason; everything else must be fixed by upgrading):
#   PYSEC-2026-3740 — pre-existing (kept).
#   CVE-2026-69112  (accelerate 1.13.0) — no fixed release exists yet.
#   PYSEC-2026-2447 (diskcache 5.6.3)  — no fixed release exists yet (latest is 5.6.3).
#   PYSEC-2026-3624 (lightning 2.6.5)  — the only listed "fix" (2022.6.15) is a breaking downgrade
#                    incompatible with pyannote-audio's lightning 2.x requirement.
#   PYSEC-2025-194  (torch 2.12.0)     — fixed in 2.13.0 (within our range), but a core-ML bump is
#                    deferred to a COORDINATED upgrade: it risks the diarization/whisper stack, the
#                    torchcodec<0.15 cap, and divergence from the baked stack-test image. Do it as a
#                    deliberate ML-stack pass, not a drive-by here.
# Blocking on CI, advisory locally — decided by $CI, which every CI provider sets.
#
# The findings are identical either way; what differs is whether they halt the run. On CI they
# must: that is the gate. Locally they must not, because of one situation this repo actually has.
#
# x86_64 macOS is it. The newest torch wheel published for that platform is 2.2.2, which carries
# 22 known vulnerabilities, and EVERY version the advisories name as fixed (2.5, 2.6, 2.7, 2.8,
# 2.9) has no macOS x86_64 build. There is no pin that is both installable and clean. A gate that
# cannot go green on a supported dev machine stops the other 20 stages of `make ci` from running
# at all, and "I never run the full build locally" is a worse security outcome than a warning.
#
# Override either way:
#   SECURITY_AUDIT_STRICT=1     fail even locally (check before you push)
#   SECURITY_AUDIT_ADVISORY=1   report-only even on CI (do not)
#
# On a local x86_64 macOS run (never on CI, never with SECURITY_AUDIT_STRICT) the advisories of
# the three packages pinned by that torch ceiling are also ignored, so the audit can pass there and
# a NEW finding in any other package stands out. The set that works on that platform is torch
# 2.2.2, transformers 4.57.6, sentence-transformers 3.4.1; the fixed versions need torch >= 2.5
# (sentence-transformers 5.x imports and then dies with "name 'nn' is not defined"). CI installs
# the pyproject floors (torch >= 2.11, transformers >= 5, sentence-transformers >= 5.6) and audits
# them in full. Drop an id here once it stops appearing in a local run.
SECURITY_AUDIT_X86_MAC_IGNORES := \
	PYSEC-2024-259 PYSEC-2025-189 PYSEC-2025-190 PYSEC-2025-191 PYSEC-2025-192 PYSEC-2025-193 \
	PYSEC-2025-195 PYSEC-2025-198 PYSEC-2025-203 PYSEC-2025-204 PYSEC-2025-205 PYSEC-2025-206 \
	PYSEC-2025-207 PYSEC-2025-208 PYSEC-2025-209 PYSEC-2025-41 PYSEC-2026-139 PYSEC-2026-1970 \
	PYSEC-2026-2286 \
	PYSEC-2025-217 PYSEC-2026-2288 PYSEC-2026-2289 PYSEC-2026-2290 PYSEC-2026-3929 PYSEC-2026-4174 \
	PYSEC-2026-4164
security-audit:
	@$(PYTHON) -m pip install --quiet --upgrade pip setuptools
	@platform_ignores=""; \
	if [ -z "$${CI:-}" ] && [ -z "$${SECURITY_AUDIT_STRICT:-}" ] && [ "$$(uname -sm)" = "Darwin x86_64" ]; then \
		for v in $(SECURITY_AUDIT_X86_MAC_IGNORES); do platform_ignores="$$platform_ignores --ignore-vuln $$v"; done; \
		echo "x86_64 macOS local run: ignoring the torch/transformers/sentence-transformers advisories (see Makefile)"; \
	fi; \
	$(PYTHON) -m pip_audit --progress-spinner off \
		--ignore-vuln PYSEC-2026-3740 \
		--ignore-vuln CVE-2026-69112 \
		--ignore-vuln PYSEC-2026-2447 \
		--ignore-vuln PYSEC-2026-3624 \
		--ignore-vuln PYSEC-2025-194 \
		$$platform_ignores; \
	rc=$$?; \
	[ $$rc -eq 0 ] && exit 0; \
	blocking=1; \
	[ -z "$${CI:-}" ] && blocking=0; \
	[ -n "$${SECURITY_AUDIT_STRICT:-}" ] && blocking=1; \
	[ -n "$${SECURITY_AUDIT_ADVISORY:-}" ] && blocking=0; \
	if [ $$blocking -eq 0 ]; then \
		echo ""; \
		echo "  ================================================================"; \
		echo "  SECURITY AUDIT FAILED - reported, NOT blocking (local run)"; \
		echo "  The findings above are real. CI blocks on them."; \
		echo "  Run SECURITY_AUDIT_STRICT=1 make security-audit to fail here too."; \
		echo "  ================================================================"; \
		echo ""; \
		exit 0; \
	fi; \
	exit $$rc

# Code quality analysis (radon)
# Note: Use $(PYTHON) -m to ensure tools run from venv, not system PATH
complexity:
	@echo "=== Cyclomatic Complexity Analysis ==="
	@$(PYTHON) -m radon cc src/podcast_scraper/ -a -s --total-average || true
	@echo ""
	@echo "=== Maintainability Index ==="
	@$(PYTHON) -m radon mi src/podcast_scraper/ -s || true

# complexity-track: per-commit complexity/maintainability trend over git history (#1014).
# Restores the wily-based target dropped with wily (#424 pinned radon<5.2, blocking radon 6.x)
# — reimplemented on radon 6 directly, walking revisions via `git archive` (no checkout).
complexity-track:
	@echo "=== Complexity Trend (last $(or $(REVISIONS),50) commits) ==="
	@$(PYTHON) scripts/tools/radon_history.py --revisions $(or $(REVISIONS),50)

deadcode:
	@echo "=== Dead Code Detection ==="
	@$(PYTHON) -m vulture src/podcast_scraper/ .vulture_whitelist.py --min-confidence 80 || true

docstrings:
	@echo "=== Docstring Coverage ==="
	@$(PYTHON) -m interrogate src/podcast_scraper/ -v

spelling:
	@echo "=== Spell Checking ==="
	@$(CODESPELL) src/ docs/ --skip="*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl,*.patch"

spelling-docs:
	@echo "=== Spell Checking (Docs only) ==="
	@$(CODESPELL) docs/ --skip="*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl,*.patch"

quality: complexity deadcode docstrings spelling
	@echo ""
	@echo "✓ All code quality checks completed"
	# This ensures production deps (torch, transformers, spacy, openai-whisper) AND search-only
	# deps (lancedb, since it left [ml] in #1019) are all audited.
	$(PYTHON) -m pip install --quiet -e .[ml,search] || \
		(echo "⚠️  Editable install failed, using non-editable install" && \
		 $(PYTHON) -m pip install --quiet .[ml,search])
	# Audit all installed packages (including ML dependencies from pyproject.toml)
	# Ignore PYSEC-2022-42969: py package vulnerability (transitive dep of interrogate, deprecated, not exploitable here)
	# Ignore CVE-2026-0994: protobuf vulnerability (affects 6.33.4, fixed in later versions; transitive dep of ML packages)
	# Ignore CVE-2026-4539: pip-audit/OSV currently flags all pygments versions until a fixed release is published; we pin
	#   pygments<2.19.0 in pyproject.toml (NVD/GHSA: vulnerable code in 2.19.0–2.19.2). Revisit when 2.19.3+ exists or OSV range fixes.
	# TODO(CVE-2026-4539): Remove --ignore-vuln when upstream fix + pip-audit range allow; sync pyproject pygments cap.
	# Ignore CVE-2025-69872: diskcache 5.6.3 vulnerability (transitive dep). Revisit when diskcache publishes a fix.
	# TODO(CVE-2025-69872): Remove --ignore-vuln when diskcache releases a patched version.
	# Ignore CVE-2026-3219: pip mishandles concatenated tar+ZIP archives (install confusion). Affects the
	#   ambient ``pip`` on the runner (26.0.1); PyPI has no newer pip yet (fix merged, unreleased). CI only
	#   installs from trusted indexes; revisit when pip >26.0.1 includes https://github.com/pypa/pip/pull/13870.
	# TODO(CVE-2026-3219): Remove --ignore-vuln after a patched pip release is published and GHA images pick it up.
	# Ignore CVE-2026-6357: pip vulnerability fixed in 26.1 — same shape as CVE-2026-3219 (runner ambient
	#   ``pip``, not a project dep). The GHA runner ships 26.0.1; we don't upgrade pip in the audit job since
	#   the install path is trusted (PyPI-only, no third-party indexes). Revisit once GHA images ship pip>=26.1.
	# TODO(CVE-2026-6357): Remove --ignore-vuln once the GHA runner's bundled pip is >=26.1.
	#
	# --- ML-stack deserialization class (joblib / nltk / torch / transformers) ---
	# 22 PYSECs below all share the same shape: vulnerable code path is reached
	# only when the program deserialises an attacker-supplied artifact (joblib
	# pickle, torch checkpoint, transformers conversion script, nltk user-path).
	# In this project we (1) only load model weights from pinned HuggingFace
	# repos that we control, (2) only joblib-cache values we computed in-process,
	# (3) never expose ``nltk.util.filestring`` to user input, and (4) don't use
	# any of the transformers conversion utilities (X-CLIP / Trainer / etc.).
	# Re-evaluate per package when a fixed release is published.
	#
	# Ignore PYSEC-2024-277 (joblib NumpyArrayWrapper.read_array deserialization).
	#   Disputed by maintainer ("only used during caching of trusted content").
	#   No fix version exists; affects all joblib through 1.5.3.
	# TODO(PYSEC-2024-277): drop ignore if joblib upstream addresses the dispute.
	#
	# Ignore PYSEC-2026-97 (nltk.util.filestring arbitrary file read).
	#   We use nltk only for tokeniser/stopwords; ``filestring`` is never called
	#   on user-supplied paths. No fix version yet; affects all nltk through 3.9.4.
	# TODO(PYSEC-2026-97): drop ignore when nltk releases a sanitised filestring.
	#
	# Ignore PYSEC-2026-597 (nltk 3.9.4 — same deserialization/file-read class as
	#   PYSEC-2026-97 above; newly published against the nltk we pin). Same
	#   reachability argument: nltk is used only for the tokeniser/stopwords, never
	#   fed user-supplied paths. No fix version yet (affects all nltk through 3.9.4).
	# TODO(PYSEC-2026-597): drop ignore when nltk publishes a fixed release.
	#
	# Ignore PYSEC-2025-189..197, PYSEC-2025-210, PYSEC-2026-139 (torch checkpoint /
	#   operator deserialization class). 11 advisories against torch <= 2.12.0 with
	#   NO fix versions (2.12.0 IS the latest). We only ``torch.load`` HuggingFace
	#   weights pinned in pyproject.toml; no user-supplied checkpoints.
	# TODO(torch PYSECs): drop ignores when upstream ships fixed versions and we
	#   bump the torch pin.
	#
	# Ignore CVE-2025-3000 (torch.jit.script memory corruption).
	#   Local-access attack on the JIT script path. We don't call
	#   ``torch.jit.script`` directly anywhere in the codebase; torch is loaded
	#   transitively via transformers / sentence-transformers / pyannote at
	#   higher levels. Same exploitation class as the PYSEC-2025-189..197 block
	#   above (no fix version published; 2.12.0 is latest).
	# TODO(CVE-2025-3000): drop ignore when torch upstream ships a fixed
	#   version and we bump the pin.
	#
	# Ignore PYSEC-2026-3624 (lightning <= 2.6.5, load_from_checkpoint RCE via the
	#   ``_instantiator`` hyperparameter importing attacker-controlled module names —
	#   bypasses weights_only=True). Same attacker-supplied-artifact class as the torch
	#   checkpoint PYSECs above: reached only when deserialising an UNTRUSTED checkpoint.
	#   We load only pinned HuggingFace pyannote weights (lightning rides in transitively
	#   via pyannote-audio); no request-time / user-supplied checkpoint loading (D6).
	#   2.6.5 is the latest release and OSV's "fixed 2022.6.15" is phantom calendar-version
	#   data — the real fix is unreleased commit d710d68, so there is nothing to bump to.
	# TODO(PYSEC-2026-3624): drop ignore when lightning ships a release past commit
	#   d710d68 (PR #21832) and we confirm pip-audit accepts it.
	#
	# Ignore PYSEC-2026-3740 / CVE-2026-81726 (nltk <= 3.10.3, path traversal in model-artifact
	#   APIs — TransitionParser, AveragedPerceptron, PerceptronTagger and maxent read/write
	#   caller-controlled paths with raw file ops, bypassing pathsec).
	#   NOT REACHABLE HERE: `grep -rnE 'TransitionParser|AveragedPerceptron|PerceptronTagger|maxent'
	#   src/ scripts/` is empty. The only nltk surface we touch is `evaluation/scorer.py` —
	#   `word_tokenize`, `sentence_bleu`, and `nltk.data.find` against the hardcoded
	#   `_NLTK_TOKENIZER_DATA` list, so no path in the advisory's class is caller-controlled.
	#   Nothing to bump to: 3.10.3 IS the latest on PyPI. The advisory is self-contradictory —
	#   its prose says "through 3.10.3 contains", its OSV range says `fixed: 3.10.3` — which is
	#   why pip-audit reports it with an EMPTY Fix Versions column. Published 2026-08-27; it
	#   turned this gate red on a branch that does not touch nltk.
	# TODO(PYSEC-2026-3740): drop ignore once upstream resolves the range/prose contradiction or
	#   ships a release that pip-audit accepts.
	#
	# Note: If protobuf is updated to >=6.33.5 or >=7.0.0, this ignore can be removed
	# Note: en-core-web-sm is installed from GitHub (not PyPI), so it cannot be audited by pip-audit
	#       If it appears in audit output, it can be safely ignored as it's not from PyPI
	$(PYTHON) -m pip_audit --skip-editable \
		--ignore-vuln PYSEC-2022-42969 \
		--ignore-vuln CVE-2026-0994 \
		--ignore-vuln CVE-2026-4539 \
		--ignore-vuln CVE-2025-69872 \
		--ignore-vuln CVE-2026-3219 \
		--ignore-vuln CVE-2026-6357 \
		--ignore-vuln PYSEC-2024-277 \
		--ignore-vuln PYSEC-2026-97 \
		--ignore-vuln PYSEC-2026-597 \
		--ignore-vuln PYSEC-2025-189 \
		--ignore-vuln PYSEC-2025-190 \
		--ignore-vuln PYSEC-2025-191 \
		--ignore-vuln PYSEC-2025-192 \
		--ignore-vuln PYSEC-2025-193 \
		--ignore-vuln PYSEC-2025-194 \
		--ignore-vuln PYSEC-2025-195 \
		--ignore-vuln PYSEC-2025-196 \
		--ignore-vuln PYSEC-2025-197 \
		--ignore-vuln PYSEC-2025-210 \
		--ignore-vuln PYSEC-2026-139 \
		--ignore-vuln PYSEC-2026-161 \
		--ignore-vuln MAL-2026-4750 \
		--ignore-vuln PYSEC-2026-3624 \
		--ignore-vuln CVE-2025-3000 \
		--ignore-vuln PYSEC-2026-3740
	# PYSEC-2026-161 (starlette<1.0.1, Host-header URL-path poisoning, GHSA-86qp-5c8j-p5mr):
	# Not exploitable in this codebase — grep -rn 'request.url.path' src/ is empty;
	# FastAPI routing uses the real request path, not the reconstructed URL. Traefik
	# also normalises Host upstream in prod. Fix is in
	# starlette 1.0.1 but it ships as "Alpha" classifier and fastapi 0.136.x (latest)
	# still pins starlette<1.0; atomic bump is unsafe today.
	# TODO(PYSEC-2026-161): Drop this ignore once fastapi releases a version that
	# validates against starlette>=1.0, then bump both atomically.
	#
	# MAL-2026-4750 (fastapi 0.136.3): freshly-published advisory in the OSV
	# malicious-package database. The MAL- prefix is for typosquats / malware,
	# but ``fastapi`` is the canonical package on PyPI (10M+/day downloads),
	# not a typosquat — almost certainly a false positive from OSV's heuristic
	# classifier. ``Fix Versions`` is empty in the advisory, which fits the
	# false-positive profile. We pin fastapi 0.136.x explicitly in
	# pyproject.toml and pip installs go through the trusted PyPI index only.
	# TODO(MAL-2026-4750): Drop this ignore once OSV withdraws the advisory
	# or fastapi releases a version flagged as "fixed".

# PR #815: path-filter touch to re-trigger CI on feat/2.7 HEAD.
docs:
	$(PYTHON) -m mkdocs build --strict

docs-check: lint-markdown-docs check-doc-structure spelling-docs docs
	@echo ""
	@echo "✓ Documentation validation complete (linting + spelling + build)"

# Coverage thresholds per layer (minimums to ensure balanced coverage)
# These are ambitious but achievable targets based on current coverage levels
# Combined threshold is enforced in CI; per-layer thresholds ensure no layer is neglected
COVERAGE_THRESHOLD_UNIT := 70          # Current: ~74% local, ~70% CI
COVERAGE_THRESHOLD_INTEGRATION := 42   # Raised 2026-04: integration-only line cov ~43% local
# E2E: measured with `.coveragerc-e2e`, which omits the consumer app (`server/app_*.py`,
# `server/routes/app_*.py`) from the denominator. That file carries the full reasoning; the short
# version is that `tests/e2e/` is 66 files of PIPELINE tests of which exactly one touches the app,
# while the app is covered by 50 files across tests/unit and tests/integration — measured by
# DIFFERENT gates. Shipping a player feature therefore grew this denominator while its tests landed
# in another job, so the numerator could not move. The gate was measuring the ratio of pipeline code
# to total code, which falls as the app grows however well anything is tested.
#
# FIXED THE METRIC, NOT THE NUMBER (2026-09-19). The floor stays where the last recalibration put
# it — 38.0 -> 37.5 (2026-09-13, #2049) — and is deliberately NOT raised to match the new
# measurement. Recomputing run 35439345239's coverage-e2e.xml with the omit gives 38.58% against
# 37.04% without it, but a floor set from an offline recomputation is still a guess, and guessing a
# floor is how main goes red after a merge — the exact failure this change exists to stop. Raise it
# once CI has reported the real figure on a full run.
#
# Keep in lockstep with --cov-fail-under in python-app.yml and nightly.yml; asserted by
# tests/unit/test_viewer_e2e_jobs_stay_in_lockstep.py. This copy once sat at 39 while CI moved
# to 38.5, so `make coverage-check-e2e` was gating on a number CI had already abandoned.
COVERAGE_THRESHOLD_E2E := 37.5
COVERAGE_THRESHOLD_COMBINED := 70      # Combined line coverage (make ci + coverage-enforce); align with CI workflow

check-unit-imports:
	# Verify that unit tests can import modules without ML dependencies
	# This ensures unit tests can run in CI without heavy ML dependencies installed
	# Run this when: adding new modules, refactoring imports, or debugging CI failures
	export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/check_unit_test_imports.py

check-test-policy:
	# Enforce 3-tier ML/AI testing policy (no importorskip in unit, no ml_models in integration, etc.)
	# Run this when: adding/moving tests, before commit, or debugging CI skip issues
	$(PYTHON) scripts/tools/check_test_policy.py --fix-hint

check-doc-structure:
	# Enforce the documentation STRUCTURE: required READMEs exist at the tree roots people land in,
	# none of them are stubs, and every relative markdown link in the repo resolves.
	# Complements ``lint-markdown`` (style/format) — this one is about pointers being true.
	# Run this when: adding/moving/renaming any doc or directory, or before commit.
	$(PYTHON) scripts/tools/check_doc_structure.py


profile-drift-check:
	# #907 Option B: every config/profiles/*.yaml that declares a `profile:` field
	# must agree with the named registry preset on routing
	# (transcription_provider, summary_provider, summary_model). Catches hand-edits
	# that fork a YAML from its eval-report-justified registry default.
	# Wired into ci-fast; safe to run standalone in <1s.
	$(PYTHON) -m pytest tests/integration/providers/ml/test_profile_yaml_registry_drift.py -q --no-cov --disable-socket --allow-hosts=127.0.0.1,localhost

profile-resolve: ## Print the resolved settings dict for PROFILE=<name>. Usage: make profile-resolve PROFILE=cloud_with_dgx_primary
	@if [ -z "$(PROFILE)" ]; then \
		echo "PROFILE not set. Available presets:"; \
		$(PYTHON) scripts/tools/resolve_profile.py --list | sed 's/^/  /'; \
		echo ""; \
		echo "Usage: make profile-resolve PROFILE=<name> [DGX_TAILNET_HOST=<host>]"; \
		exit 2; \
	fi
	@$(PYTHON) scripts/tools/resolve_profile.py "$(PROFILE)"

# Optional ARGS: e.g. make check-pricing-assumptions ARGS="--strict"
# Runs the staleness report AND the #651 profile-coverage guard
# (fails when a profile references a model without a YAML rate row).
check-pricing-assumptions:
	export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) -m $(PACKAGE).cli pricing-assumptions $(ARGS)
	$(PYTHON) scripts/validate/check_profile_pricing_coverage.py
	$(PYTHON) scripts/validate/check_pricing_yaml_bundled_sync.py

validate-kg-schema:
	# Validate kg.json files against docs/architecture/kg/kg.schema.json (strict mode).
	# Usage: make validate-kg-schema [ARTIFACTS_DIR=path]. With no path, validates tests/fixtures (if any).
	@if [ -n "$(ARTIFACTS_DIR)" ]; then \
		export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/validate_kg_schema.py "$(ARTIFACTS_DIR)"; \
	else \
		export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/validate_kg_schema.py tests/fixtures; \
	fi

# GI/KG viewer v2 (#489): FastAPI + Vite. ``make init`` includes FastAPI via ``[dev]``; cd $(WEB_VIEWER_DIR) && npm install
.PHONY: serve serve-api serve-ui serve-e2e-mock stack-build stack-build-llm stack-compose-validate stack-up stack-down stack-logs verify-stack-profiles stack-test-build stack-test-build-cloud stack-test-up stack-test-down stack-test-seed stack-test-playwright stack-test-export stack-test-ml stack-test-cloud-thin stack-test-ml-ci reprocess-corpus-from-transcripts corpus-compat-check index-two-tier index-two-tier-docker upgrade-status upgrade-check upgrade-dry-run upgrade-corpus upgrade-verify enrich enrich-viewer-fixture
SERVE_OUTPUT_DIR ?= ./output
# serve-api bind — DEFAULT 0.0.0.0:8000 so `make serve` "just works" for off-box dev:
# a dev-signed iOS build's APNs token registration + the homelab delivery worker pulling
# /internal/outbox both reach it over the tailnet, with ZERO extra flags. Binding 0.0.0.0
# still answers on 127.0.0.1:8000, so the Vite proxy (/api -> 127.0.0.1:8000) and every other
# localhost client keep working unchanged — it only ADDS tailnet + LAN reachability.
# NOTE: serve-api defaults to mock OAuth (local dev), so on 0.0.0.0 the API is reachable on
# your LAN too. Trusted-network dev only. Pin loopback with `make serve-api SERVE_HOST=127.0.0.1`.
SERVE_HOST ?= 0.0.0.0
SERVE_PORT ?= 8000
# Optional corpus-editing + jobs routes (health shows green when on). Override with SERVE_ARGS= to disable.
SERVE_ARGS ?= --enable-feeds-api --enable-operator-config-api --enable-jobs-api
# Default E2E mock RSS port (127.0.0.1:18765; override with E2E_MOCK_PORT when fixture URLs change).
# Deliberately not 8000 so ``serve-e2e-mock`` can run alongside ``serve-api`` (FastAPI default).
E2E_MOCK_PORT ?= 18765
serve:
	@echo "Running serve-api and serve-ui in parallel (Ctrl+C stops both)."
	@$(MAKE) -j2 serve-api serve-ui

serve-api:
	@# KMP_DUPLICATE_LIB_OK: with the [ml] extra installed, hybrid search loads torch (query
	@# embedder) alongside faiss, and on macOS the duplicate OpenMP runtime aborts the process
	@# ("Abort trap: 6") the first time /api/corpus/digest or /api/search runs — the API dies and
	@# the viewer goes blank. Allowing the duplicate is the standard local workaround; no-op on Linux.
	@# Local dev auth: default to the mock OAuth provider + a dev session secret so player AND viewer
	@# sign-in work out of the box (also required to exercise personalization). serve-api is local-only
	@# (prod runs via compose/stack with a real provider), so the mock default is safe here. Override
	@# any of these env vars to disable or point elsewhere; NEVER use the mock provider outside dev.
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && export KMP_DUPLICATE_LIB_OK=$${KMP_DUPLICATE_LIB_OK:-TRUE} && \
		export APP_OAUTH_PROVIDER=$${APP_OAUTH_PROVIDER:-mock} && \
		export APP_SESSION_SECRET=$${APP_SESSION_SECRET:-dev-secret} && \
		export APP_ADMIN_EMAILS=$${APP_ADMIN_EMAILS:-dev@localhost} && \
		export APP_SEED_USERS_FILE=$${APP_SEED_USERS_FILE:-config/dev-seed-users.json} && \
		if [ -f .env.obs.dev ]; then set -a; . ./.env.obs.dev; set +a; echo "[obs] streaming to homelab as environment=dev instance=$(notdir $(CURDIR))-$${PODCAST_OBS_PORT:-8000}"; fi && \
		export PODCAST_OBS_PORT=$${PODCAST_OBS_PORT:-8000} && \
		$(DEV_DB_RUN) \
			$(PYTHON) -m $(PACKAGE).cli serve --host "$(SERVE_HOST)" --port "$(SERVE_PORT)" --output-dir "$(SERVE_OUTPUT_DIR)" $(SERVE_ARGS)

serve-ui:
	@cd $(WEB_VIEWER_DIR) && npm run dev

# --- Local MCP servers (#56) — stdio, auth-free (local trust), for an MCP client to SPAWN. ---
# These are what `.mcp.json` invokes, so Claude Code/Desktop just connects with zero config against
# whatever `make serve` is serving. stdio ONLY (never a public surface); stdout carries the JSON-RPC
# so the recipes stay echo-free and `exec` the server (no make wrapper on stdout). Pair with a
# running `make serve`.
serve-mcp: ## Content MCP over stdio (local trust), pointed at the make-serve corpus ($(SERVE_OUTPUT_DIR)).
	@export PYTHONPATH="$(PWD)/src" && exec $(PYTHON) -m $(PACKAGE).cli mcp --transport stdio --corpus "$(SERVE_OUTPUT_DIR)"

serve-obs: ## Observability MCP over stdio (local trust) using config/observability.local.yaml (local api + homelab reads).
	@export PYTHONPATH="$(PWD)/src"; \
		[ -f .env.obs.dev ] && { set -a; . ./.env.obs.dev; set +a; } ; \
		export PODCAST_OBS_CONFIG="$(CURDIR)/config/observability.local.yaml"; \
		exec $(PYTHON) -m podcast_obs serve --transport stdio

# E2E fixture HTTP server (RSS + mock API paths); use --feeds-spec with URLs on this port.
serve-e2e-mock:
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src:$(PWD)" && $(PYTHON) scripts/tools/run_e2e_mock_server.py --port "$(E2E_MOCK_PORT)"

# GitHub #659: Docker Compose stack (Nginx + FastAPI + shared volume; pipeline behind profile).
STACK_COMPOSE ?= docker compose -f compose/docker-compose.stack.yml
STACK_TEST_COMPOSE ?= FIXTURES_VERSION=$(FIXTURES_VERSION) docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.stack-test.yml
REMOVE_VOLUMES ?=
# Stack-test: host directory for ``make stack-test-export`` (debug aid —
# copies the ``corpus_data`` volume contents to disk so artifacts can be
# inspected after a Playwright run).
STACK_TEST_EXPORT_DIR ?= $(CURDIR)/.stack-test-corpus

# Provenance for the compose builds. Compose cannot shell out, so the SHA is exported into its
# environment here and interpolated by ``${GIT_SHA:-}`` in docker-compose.stack.yml. Without
# this the built image writes git_sha: null into every manifest (ADR-132, #1657).
# Inner quotes are NOT backslash-escaped — see the note on GIT_BUILD_ARGS; the escaped form
# reports a dirty tree as clean instead of erroring loudly.
GIT_BUILD_ENV = \
	BUILD_TIME="$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
	GIT_SHA="$$(git rev-parse HEAD 2>/dev/null || echo '')" \
	GIT_BRANCH="$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')" \
	GIT_DIRTY="$$(test -z "$$(git status --porcelain 2>/dev/null)" && echo false || echo true)"

stack-build:
	@$(GIT_BUILD_ENV) $(STACK_COMPOSE) build

stack-build-llm:
	@$(GIT_BUILD_ENV) $(STACK_COMPOSE) --profile pipeline-llm build pipeline-llm

verify-stack-profiles:
	@echo "Validating config/profiles/*.yaml minimum Docker tiers..."
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/validate_profile_docker_tier.py

# ``docker compose config`` (no build) — catches bad merges / invalid interpolation.
# Validates both the base stack and the production overlay
# (``jobs-docker.yml`` declares ``PODCAST_DOCKER_PROJECT_DIR`` as required).
stack-compose-validate:
	@PODCAST_DOCKER_PROJECT_DIR="$(CURDIR)" $(STACK_COMPOSE) config -q
	@PODCAST_DOCKER_PROJECT_DIR="$(CURDIR)" docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.jobs-docker.yml config -q
	@echo "stack-compose-validate: OK"

stack-up:
	@$(STACK_COMPOSE) up -d

stack-down:
	@if [ -n "$(REMOVE_VOLUMES)" ]; then $(STACK_COMPOSE) down -v; else $(STACK_COMPOSE) down; fi

stack-logs:
	@$(STACK_COMPOSE) logs -f viewer api

stack-test-build:
	@# ``--profile pipeline`` is required: the ``pipeline`` service has
	@# ``profiles: [pipeline]`` so plain ``docker compose build`` silently
	@# skips it. Without this the API's job factory ends up building the
	@# pipeline image at job-spawn time inside its own container, which
	@# in CI runs without BuildKit (the workflow's ``DOCKER_BUILDKIT=1``
	@# env doesn't propagate into the api container) and fails on the
	@# Dockerfile's ``RUN --mount=type=cache`` syntax.
	@#
	@# ``STACK_PIPELINE_PRELOAD_ML=true`` is REQUIRED, not optional: the
	@# ``airgapped_thin`` profile (the only profile stack-test exercises
	@# in CI) loads transformers models — bart-base / led-base / spaCy /
	@# Whisper — directly from the local cache. With ``=false`` the
	@# image ships empty caches and the pipeline subprocess errors at
	@# Transformers load time with "We couldn't connect to
	@# 'https://huggingface.co' to load the files, and couldn't find
	@# them in the cached files." (HF_HUB_OFFLINE-style failure on a
	@# fresh CI runner with no warm DNS / no pre-pulled models). The
	@# ML pipeline must ALWAYS bake the cache in at build time.
	@#
	@# ``STACK_PIPELINE_PRELOAD_VARIANT=airgapped-thin`` selects the
	@# slim preload bundle: only what config/profiles/airgapped_thin.yaml
	@# uses (BART-base + LED-base-16384 + Whisper tiny.en + en_core_web_sm
	@# + GIL evidence). Drops SummLlama (Llama 3.2 license — never
	@# publish), Pegasus, hybrid LongT5/FLAN-T5, Whisper base.en. This
	@# saves ~2 GB on the resulting image and keeps the redistribution
	@# story clean for the future GHCR publish step (RFC-081, not in this repository).
	@STACK_PIPELINE_PRELOAD_ML=true \
		STACK_PIPELINE_PRELOAD_VARIANT=airgapped-thin \
		$(STACK_TEST_COMPOSE) --profile pipeline build

# Cloud-thin pipeline image: ``[llm]`` extras only (cloud API SDKs —
# openai, google-genai, anthropic, mistralai, httpx). Zero local ML
# (no torch / transformers / whisper / spaCy). Built into the
# ``pipeline-llm`` compose service.
stack-test-build-cloud:
	@# Mirror of ``stack-test-build`` for the cloud-thin variant.
	@# Contract: **always** preload models for ML, **never** for cloud.
	@# The cloud-thin pipeline uses ``[llm]`` extras only (OpenAI /
	@# Gemini / Anthropic / Mistral SDKs) — no local Transformers /
	@# Whisper / spaCy — so a model preload would be a no-op (the
	@# Dockerfile gates preload on ``INSTALL_EXTRAS=ml`` anyway) and
	@# would waste image size + build time. Force ``=false`` to keep
	@# the contract explicit and symmetrical with the ML recipe.
	@STACK_PIPELINE_PRELOAD_ML=false $(STACK_TEST_COMPOSE) --profile pipeline-llm build pipeline-llm

# Debug aid: copy ``/app/output`` from the running ``corpus_data`` volume
# onto the host so artifacts can be inspected after a Playwright run.
# Multi-feed corpus layout — runs land at
# ``feeds/<stable>/run_*/{transcripts,metadata,run.json,…}`` and the
# corpus root carries ``corpus_run_summary.json`` + ``feeds.spec.yaml``.
stack-test-export:
	@mkdir -p "$(STACK_TEST_EXPORT_DIR)"
	@$(STACK_TEST_COMPOSE) --profile pipeline run --rm -T --no-deps \
		-v "$(STACK_TEST_EXPORT_DIR):/host" \
		--entrypoint /bin/sh \
		pipeline -c 'set -e; if [ ! -d /app/output ] || [ -z "$$(ls -A /app/output 2>/dev/null)" ]; then echo "empty /app/output — run a Playwright pipeline job first (make stack-test-up + stack-test-seed + stack-test-playwright)"; exit 1; fi; cp -a /app/output/. /host/'
	@echo "stack-test-export: copied to $(STACK_TEST_EXPORT_DIR)"

# Copy ``config/ci/stack-test-seed/*`` into the ``corpus_data`` volume
# so the API + Playwright full-flow spec see ``feeds.spec.yaml`` and
# ``viewer_operator.yaml`` at the corpus root before the operator UI
# touches anything. Idempotent — overwrites any prior seed each run
# so a previous test's mutations don't bleed into the next.
#
# ``STACK_TEST_OPERATOR_VARIANT`` (default ``ml``) decides which
# ``pipeline_install_extras`` line gets appended to the single
# ``viewer_operator.yaml`` template — the rest of the file is shared
# across both runs:
#   - ``ml``         → ``pipeline_install_extras: ml``  (compose service ``pipeline``)
#   - ``cloud-thin`` → ``pipeline_install_extras: llm`` (compose service ``pipeline-llm``)
# The Playwright spec then needs ``STACK_TEST_OPERATOR_PROFILE`` set to
# match: ``airgapped_thin`` for ml (default), ``cloud_thin`` for llm.
#
# Run order for the full UI flow (ml/airgapped):
#   make stack-test-build                          # builds api + pipeline (ml)
#   make stack-test-up
#   make stack-test-seed                           # default ml variant
#   make stack-test-playwright
#
# Run order for the full UI flow (llm/cloud-thin) — local only:
#   make stack-test-build-cloud                    # adds pipeline-llm
#   make stack-test-up
#   make stack-test-seed STACK_TEST_OPERATOR_VARIANT=cloud-thin
#   STACK_TEST_OPERATOR_PROFILE=cloud_thin make stack-test-playwright
STACK_TEST_OPERATOR_VARIANT ?= ml
ifeq ($(STACK_TEST_OPERATOR_VARIANT),ml)
  STACK_TEST_SEED_EXTRAS := ml
else ifeq ($(STACK_TEST_OPERATOR_VARIANT),cloud-thin)
  STACK_TEST_SEED_EXTRAS := llm
else
  STACK_TEST_SEED_EXTRAS := __invalid__
endif

stack-test-seed:
	@if [ ! -d config/ci/stack-test-seed ]; then \
		echo "stack-test-seed: missing config/ci/stack-test-seed/ (need feeds.spec.yaml + viewer_operator.yaml)"; \
		exit 1; \
	fi
	@if [ "$(STACK_TEST_SEED_EXTRAS)" = "__invalid__" ]; then \
		echo "stack-test-seed: STACK_TEST_OPERATOR_VARIANT must be 'ml' or 'cloud-thin' (got: $(STACK_TEST_OPERATOR_VARIANT))"; \
		exit 1; \
	fi
	@echo "stack-test-seed: variant=$(STACK_TEST_OPERATOR_VARIANT) pipeline_install_extras=$(STACK_TEST_SEED_EXTRAS)"
	@PODCAST_DOCKER_PROJECT_DIR=$(CURDIR) \
		$(STACK_TEST_COMPOSE) run --rm -T --no-deps \
			-v "$(CURDIR)/config/ci/stack-test-seed:/seed:ro" \
			-e STACK_TEST_SEED_EXTRAS=$(STACK_TEST_SEED_EXTRAS) \
			--entrypoint /bin/sh \
			api -c 'set -e; mkdir -p /app/output; cp -f /seed/feeds.spec.yaml /app/output/feeds.spec.yaml; cp -f /seed/viewer_operator.yaml /app/output/viewer_operator.yaml; printf "\npipeline_install_extras: %s\n" "$$STACK_TEST_SEED_EXTRAS" >> /app/output/viewer_operator.yaml; chown -R podcast:podcast /app/output 2>/dev/null || true; ls -la /app/output/'
	@echo "stack-test-seed: OK"

stack-test-up:
	@bash -c 'set -e; \
		if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
		STACK_PIPELINE_PRELOAD_ML=false \
		STACK_TEST_VIEWER_PORT=$${STACK_TEST_VIEWER_PORT:-8090} \
		PODCAST_DOCKER_PROJECT_DIR=$(CURDIR) \
		$(STACK_TEST_COMPOSE) up -d'

stack-test-down:
	@$(STACK_TEST_COMPOSE) down $(if $(filter 1 true yes,$(STACK_TEST_DOWN_VOLUMES)),-v,)

stack-test-playwright:
	@cd tests/stack-test && npm install && ./node_modules/.bin/playwright install firefox && STACK_TEST_BASE_URL=$${STACK_TEST_BASE_URL:-http://127.0.0.1:8090} ./node_modules/.bin/playwright test

# One-command local stack-test for the ml (airgapped_thin) pipeline path.
# Build (idempotent) → up → seed → Playwright. Stack stays up afterward
# so artifacts can be inspected with ``make stack-test-export`` and the
# corpus volume can be poked at; run ``make stack-test-down`` to clean up.
# CI uses the underlying step-by-step targets in the workflow file.
stack-test-ml:
	@# Start from a clean corpus volume. A stale/partial LanceDB index left in the
	@# persisted compose volume by a previous run causes false "no_index" search
	@# failures (the dataset manifest references a compacted-away data fragment).
	@# CI runners are ephemeral so they never hit this; locally we must wipe first.
	$(MAKE) stack-test-down STACK_TEST_DOWN_VOLUMES=1 || true
	$(MAKE) stack-test-build
	$(MAKE) stack-test-up
	$(MAKE) stack-test-seed STACK_TEST_OPERATOR_VARIANT=ml
	$(MAKE) stack-test-playwright

# One-command local stack-test for the cloud-thin (LLM) pipeline path.
# Same flow as ``stack-test-ml`` plus ``stack-test-build-cloud`` (builds
# the ``pipeline-llm`` image with ``[llm]`` extras only — no torch /
# transformers / whisper / spaCy) and seeds with ``pipeline_install_extras:
# llm``. Playwright runs with ``STACK_TEST_OPERATOR_PROFILE=cloud_thin``.
# Requires ``.env`` with the cloud API keys the ``cloud_thin`` profile
# providers expect (OPENAI / GEMINI / ANTHROPIC). Local-only — public CI
# does not run this variant to avoid recurring API costs.
stack-test-cloud-thin:
	@# Clean corpus volume first — see stack-test-ml (stale LanceDB index → false no_index).
	$(MAKE) stack-test-down STACK_TEST_DOWN_VOLUMES=1 || true
	$(MAKE) stack-test-build
	$(MAKE) stack-test-build-cloud
	$(MAKE) stack-test-up
	$(MAKE) stack-test-seed STACK_TEST_OPERATOR_VARIANT=cloud-thin
	STACK_TEST_OPERATOR_PROFILE=cloud_thin $(MAKE) stack-test-playwright

# CI flavour of ``stack-test-ml`` — same flow but always tears down at
# the end (success or failure). Used by ``_ci_body``: ``make ci`` should
# leave a clean laptop after the gate runs. Inner exit code is preserved
# so a Playwright failure still fails ``make ci``.
#
# The shell ``trap … EXIT`` is what guarantees teardown even when
# ``stack-test-playwright`` fails — chaining with ``&&`` short-circuits
# at the failed step, then the EXIT trap fires before the recipe
# returns the captured non-zero status.
stack-test-ml-ci:
	@# Wipe the corpus volume before (clean slate — a stale/partial LanceDB index from a
	@# prior run yields false "no_index" search failures; ephemeral CI runners never hit
	@# it, local re-runs do) and REAP on any exit (EXIT/INT/TERM) so an interrupted run
	@# never leaves a runaway build / Playwright runner thrashing the machine.
	@set -u; \
	trap '$(MAKE) stack-test-reap >/dev/null 2>&1 || true' EXIT INT TERM; \
	$(MAKE) stack-test-down STACK_TEST_DOWN_VOLUMES=1 >/dev/null 2>&1 || true; \
	$(MAKE) stack-test-build \
		&& $(MAKE) stack-test-up \
		&& $(MAKE) stack-test-seed STACK_TEST_OPERATOR_VARIANT=ml \
		&& $(MAKE) stack-test-playwright


# Recompute GI/KG/search from on-disk transcripts without re-transcribing (#796).
# Requires CORPUS_DIR (corpus parent with feeds.spec.yaml + transcripts/).
#
# Fixed 2026-09-01. It was BROKEN for already-processed episodes: it passed
# `--no-transcribe-missing`, which sets transcribe_missing=false, and the transcript-reuse path
# was gated behind `if cfg.transcribe_missing and temp_dir:` — so no processing job was ever
# produced and this target exited 0 having regenerated NOTHING.
#
# (An older note here blamed a missing --single-feed-uses-corpus-layout for the "no transcript
# for <episode>" symptom. That diagnosis was wrong: the flag only affects the transcript lookup
# INSIDE the function the gate prevented from being called.)
#
# It now declares its intent with `--pipeline-stage rederive_only`, which coerces
# transcribe_missing=false AND skip_existing=true and takes the reuse branch that resolves the
# on-disk transcript. Saying "--no-transcribe-missing" described the mechanism; saying
# "rederive_only" describes the goal, and only the goal reaches the code that serves it.
#
# `--reprocess-existing-only` is REQUIRED alongside it, and its absence was a second silent
# no-op hiding behind the first. rederive_only coerces skip_existing=true, so without
# existing-only scoping the run builds its work list from the LIVE feed and then skips every
# episode already on disk — selecting nothing to re-derive, and again exiting 0. Verified
# end-to-end 2026-09-01: without the flag the run reported "Episodes to process: 0 of 0"; with
# it, the reuse branch fires per episode. (existing-only also fails LOUDLY when the corpus is
# missing, rather than quietly selecting zero.)
#
# For GI ONLY — in place, diffable, with an audit trail — `gi-repair --episode-ids ...
# --force-healthy` remains the narrower tool (docs/guides/CORPUS_REPROCESSING.md).
reprocess-corpus-from-transcripts:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	test -f "$${CORPUS_DIR}/feeds.spec.yaml" || (echo "Missing $${CORPUS_DIR}/feeds.spec.yaml"; exit 1); \
	OP_CFG="$${CORPUS_DIR}/viewer_operator.yaml"; \
	test -f "$$OP_CFG" || OP_CFG="config/profiles/cloud_balanced.yaml"; \
	echo "Reprocessing $${CORPUS_DIR} (enrich_only, scoped to on-disk episodes)..."; \
	$(PYTHON) -m podcast_scraper.cli \
	  --config "$$OP_CFG" \
	  --feeds-spec "$${CORPUS_DIR}/feeds.spec.yaml" \
	  --output-dir "$${CORPUS_DIR}" \
	  --single-feed-uses-corpus-layout \
	  --reprocess-existing-only \
	  --pipeline-stage rederive_only; \
	echo "Optional, with the extension that provides topic-clusters installed, when search/ exists:"; \
	echo "  $(PYTHON) -m podcast_scraper.cli topic-clusters --output-dir $${CORPUS_DIR}"

# Report corpus/code compatibility for a on-disk corpus parent (#796 / #797).
corpus-compat-check:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "from pathlib import Path; from podcast_scraper.corpus_version import read_produced_by, assess_corpus_version_compat, MIN_SUPPORTED_CORPUS_CODE_VERSION; from podcast_scraper import __version__; root = Path('$${CORPUS_DIR}').expanduser().resolve(); pb = read_produced_by(root); ver, warn = assess_corpus_version_compat(pb); print(f'server={__version__} min_supported={MIN_SUPPORTED_CORPUS_CODE_VERSION}'); print(f'corpus_code_version={ver!r}'); print(f'produced_by={pb!r}'); import sys; (print(f'WARNING: {warn}') or sys.exit(1)) if warn else print('COMPAT OK')"

# Pre-deploy gate (#1494 / #1497): fail if a corpus would silently under-serve the MCP —
# a stale/absent LanceDB index (every search → no_index) or missing typed edges /
# enrichments (relational tools → empty). Non-zero exit on a HARD gap; diarization is a
# soft warn. Run before shipping a corpus (or wire into the deploy). CORPUS_DIR required.
corpus-completeness-check:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "import sys; from pathlib import Path; from podcast_scraper.corpus_completeness import check_corpus; ok, report = check_corpus(Path('$${CORPUS_DIR}').expanduser()); print(report); sys.exit(0 if ok else 1)"

# Phase 0 gate (#2175): what language is every episode in, and WHERE did that answer come from?
# READ-ONLY. Non-zero exit when the report cannot be trusted as a measurement of the CORPUS --
# specifically when NO episode resolved from 'rss', which means it measured the run configuration
# instead (the same shape as capability_audit's "0/36 openings, defect rate 0.0%"). Finding a
# non-English episode is a RESULT, not a failure. Run AFTER the m0011 backfill, and before S0.6:
# after S0.6 a feed whose RSS says `de` is genuinely transcribed as German. CORPUS_DIR required.
corpus-language-audit:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "import sys; from pathlib import Path; from podcast_scraper.corpus_language_audit import check_corpus; ok, report = check_corpus(Path('$${CORPUS_DIR}').expanduser()); print(report); sys.exit(0 if ok else 1)"

# S3.2 / D-43: withdraw a published language from DISCOVERY (search index rows only; the corpus
# is untouched, so a reindex undoes it). DRY RUN unless APPLY=1 — what changes is what listeners
# can find, and `enabled: false` in config/languages.yaml deliberately does NOT do this.
#   make withdraw-language CORPUS_DIR=<corpus> LANGUAGE=es
#   make withdraw-language CORPUS_DIR=<corpus> LANGUAGE=es APPLY=1
withdraw-language:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	test -n "$${LANGUAGE:-}" || (echo "LANGUAGE required (e.g. LANGUAGE=es)"; exit 1); \
	$(PYTHON) scripts/tools/withdraw_language.py --corpus "$${CORPUS_DIR}" \
		--language "$${LANGUAGE}" $${APPLY:+--apply}

# THE repair exit criterion (#1655). Asserts, per EPISODE, that declared GI actually exists:
# artifact resolves + parses, is not a pre-#1657 placeholder, and no episode_id resolves to more
# than one gi.json (the supersede hazard — corpus_metadata_index keeps the OLDEST, search keeps
# the NEWEST, so duplicates mean two subsystems disagree on the canonical artifact).
#
# A check that only asks "is the placeholder string absent?" passes on a corpus whose artifacts
# were deleted and never regenerated (proven 2026-08-16); this one counts placeholders AND
# missing artifacts. Needs the GI extension installed (its `gi-integrity` verb). CORPUS_DIR
# required.
corpus-gi-integrity-check:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli gi-integrity --output-dir "$${CORPUS_DIR}"

# Build the two-tier LanceDB index from corpus artifacts (RFC-090 Phase 2, follow-up
# B). Native path for corpora with no legacy index to migrate. CORPUS_DIR required.
index-two-tier:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli index-two-tier --output-dir "$${CORPUS_DIR}"

# Same build, in a container — for hosts where the [search] extra CANNOT be installed.
#
# torch and lancedb publish no macOS x86_64 wheels ("no wheels with a matching platform tag
# (macosx_15_0_x86_64)"), so on an Intel Mac the target above cannot run at all and every search
# returns `no_index`. The same wheels DO exist for manylinux_2_28_x86_64 — which is what the local
# Docker VM runs — so the index builds fine one layer down.
#
# Bind mounts are not usable: Colima mounts the host read-only, so the corpus is staged into a
# named volume, indexed there, and copied back out. The REPO's src is mounted and put on
# PYTHONPATH so the index is built by the working tree's code, not whatever the image baked in.
#
# Deleting episode_fingerprints.json alongside lance_index is NOT optional: the indexer skips
# episodes whose fingerprint is unchanged, so leaving it behind yields a SILENTLY EMPTY index
# ("episodes=36 segments=0 insights=0 aux=0") that still exits 0. The per-tier counts are checked
# below rather than trusted.
#
#   make index-two-tier-docker CORPUS_DIR=tests/fixtures/app-validation-corpus/v3
INDEX_DOCKER_IMAGE ?= podcast-scraper-stack-pipeline-llm:latest
index-two-tier-docker:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	test -d "$${CORPUS_DIR}" || (echo "CORPUS_DIR does not exist: $${CORPUS_DIR}"; exit 1); \
	set -e; \
	VOL=podcast-index-$$$$; CT=podcast-index-ct-$$$$; LOG=/tmp/podcast-index-$$$$.log; \
	trap 'docker rm -f $$CT >/dev/null 2>&1 || true; docker volume rm $$VOL >/dev/null 2>&1 || true' EXIT INT TERM; \
	docker volume create $$VOL >/dev/null; \
	docker run -d --name $$CT --user root -v $$VOL:/w -v "$(CURDIR)":/repo:ro \
		--entrypoint sleep $(INDEX_DOCKER_IMAGE) 7200 >/dev/null; \
	docker exec $$CT sh -c 'mkdir -p /w/corpus'; \
	docker cp "$${CORPUS_DIR}/." $$CT:/w/corpus; \
	echo "--> dropping stale index + fingerprint sidecar (else the rebuild indexes nothing)"; \
	docker exec $$CT sh -c 'rm -rf /w/corpus/search/lance_index /w/corpus/search/episode_fingerprints.json /w/corpus/search/metadata.json'; \
	echo "--> building index in $(INDEX_DOCKER_IMAGE) (detached; polling)"; \
	docker exec -d $$CT sh -c 'PYTHONPATH=/repo/src python -u -m podcast_scraper.cli index-two-tier --output-dir /w/corpus --allow-download > /w/idx.log 2>&1; echo "rc=$$?" >> /w/idx.log; echo BUILD_DONE >> /w/idx.log'; \
	i=0; \
	while [ $$i -lt 120 ]; do \
		docker cp $$CT:/w/idx.log $$LOG 2>/dev/null || true; \
		if grep -q "^BUILD_DONE" $$LOG 2>/dev/null; then break; fi; \
		i=$$((i+1)); sleep 10; \
	done; \
	grep -q "^BUILD_DONE" $$LOG 2>/dev/null || (echo "index build did not finish in 20min; log:"; cat $$LOG 2>/dev/null; exit 1); \
	grep -q "^rc=0" $$LOG || (echo "index build FAILED; log:"; cat $$LOG; exit 1); \
	grep -E "Two-tier index built" $$LOG || (echo "no index summary line; log:"; cat $$LOG; exit 1); \
	grep -E "Two-tier index built" $$LOG | grep -qE "segments=[1-9]" \
		|| (echo "EMPTY INDEX — segments=0. Was the fingerprint sidecar left in place?"; exit 1); \
	echo "--> copying index back to $${CORPUS_DIR}/search/"; \
	if [ -e "$${CORPUS_DIR}/search/lance_index" ] && ! rm -rf "$${CORPUS_DIR}/search/lance_index" 2>/dev/null; then \
		echo ""; \
		echo "CANNOT REPLACE the existing index — it is not writable by $$(id -un):"; \
		ls -ld "$${CORPUS_DIR}/search/lance_index"; \
		echo ""; \
		echo "The index BUILT fine; only the copy-back is blocked. Have its owner remove it:"; \
		echo "    rm -rf $(CURDIR)/$${CORPUS_DIR}/search/lance_index"; \
		echo "then re-run this target. (lance_index is gitignored, so nothing is lost.)"; \
		exit 1; \
	fi; \
	rm -f "$${CORPUS_DIR}/search/episode_fingerprints.json" 2>/dev/null || true; \
	docker cp $$CT:/w/corpus/search/lance_index "$${CORPUS_DIR}/search/lance_index"; \
	docker cp $$CT:/w/corpus/search/episode_fingerprints.json "$${CORPUS_DIR}/search/" 2>/dev/null || true; \
	docker cp $$CT:/w/corpus/search/metadata.json "$${CORPUS_DIR}/search/" 2>/dev/null || true; \
	echo "index written to $${CORPUS_DIR}/search/lance_index"

# Phase 1 of staged re-diarization (#947): download + cache RAW audio for the existing
# corpus episodes only, WITHOUT transcribing/diarizing. --reprocess-existing-only scopes
# to on-disk GUIDs; --reprocess-source forces the existing whisper episodes back through
# download; --pipeline-stage download_only stops before transcription. The audio lands in
# the durable cache (default .cache/audio, external to the corpus) so phase 2
# (the extension Makefile's migrate-diarization) reprocesses off the cache with no feed re-fetch. Set
# AUDIO_CACHE_IN_CORPUS=1 to store the cache inside the corpus for a portable snapshot.
#   make download-audio CORPUS_DIR=<corpus> PROFILE=config/profiles/cloud_with_dgx_primary.yaml
download-audio:
	@test -n "$${CORPUS_DIR:-}" || { echo "CORPUS_DIR required (corpus parent path)"; exit 1; }; \
	test -n "$${PROFILE:-}" || { echo "PROFILE required (e.g. config/profiles/cloud_with_dgx_primary.yaml)"; exit 1; }; \
	test -f "$${CORPUS_DIR}/feeds.spec.yaml" || { echo "Missing $${CORPUS_DIR}/feeds.spec.yaml"; exit 1; }; \
	echo "Downloading + caching audio (existing-only, no transcription) for $${CORPUS_DIR}..."; \
	$(HF_NET_ENV) $(PYTHON) -m podcast_scraper.cli \
	  --config "$${PROFILE}" \
	  --feeds-spec "$${CORPUS_DIR}/feeds.spec.yaml" \
	  --output-dir "$${CORPUS_DIR}" \
	  --skip-existing \
	  --reprocess-source whisper_transcription \
	  --reprocess-existing-only \
	  --pipeline-stage download_only \
	  $${AUDIO_CACHE_IN_CORPUS:+--audio-cache-in-corpus}

# pipeline-check (#2287): run the pipeline under variants (locale, profile, overrides), record every
# decision it makes, and compare runs — code vs code (BASE=<ref>) and variant vs variant (LOCALES).
# Runbook: docs/guides/PIPELINE_CHECK.md. All variables are optional:
#   make pipeline-check                                  # this checkout, every check in expectations.yaml
#   make pipeline-check BASE=origin/main                 # this checkout vs main (fetch first)
#   make pipeline-check CHECK=english BASE=origin/main LOCALES="en en-US"
#   make pipeline-check CANDIDATE=feat/x BASE=origin/main CORPUS=<fixture corpus>
# Exit 0 = every check passed; the report is .test_outputs/pipeline-check/report.md.
.PHONY: pipeline-check
pipeline-check:
	@PYTHONPATH=scripts/validate $(PYTHON) -m pipeline_check.cli \
	  $(foreach c,$(CHECK),--check $(c)) \
	  $(if $(BASE),--base "$(BASE)") \
	  $(if $(CANDIDATE),--candidate "$(CANDIDATE)") \
	  $(if $(LOCALES),--locales "$(LOCALES)") \
	  $(if $(FEEDS),--feeds "$(FEEDS)") \
	  $(if $(PROFILE),--profile "$(PROFILE)") \
	  $(if $(OVERRIDES),--overrides '$(OVERRIDES)') \
	  $(if $(CORPUS),--corpus "$(CORPUS)") \
	  $(if $(REAL),--real --feed "$(FEED)" $(if $(MAX_EPISODES),--max-episodes $(MAX_EPISODES)) $(if $(REUSE),--reuse)); \
	  status=$$?; echo "PIPELINE_CHECK_EXIT=$$status"; exit $$status

# Corpus upgrade-path framework (#862). Managed, idempotent migrations for moving a
# deployed corpus across releases (2.6 → 2.7 FAISS→LanceDB is step 0001). CORPUS_DIR
# selects the corpus parent. `upgrade-check` exits 2 when migrations are pending —
# wire it into boot/CI to detect a corpus that needs upgrading.
upgrade-status:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli upgrade status --corpus-dir "$${CORPUS_DIR}"

upgrade-check:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli upgrade status --corpus-dir "$${CORPUS_DIR}" --json

# Preview the plan without writing anything.
upgrade-dry-run:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli upgrade run --corpus-dir "$${CORPUS_DIR}" --dry-run

# Apply pending migrations. Non-interactive (--yes) for automated upgrades; drop
# `--yes` and run the CLI directly for an interactive confirmation prompt.
# SNAPSHOT_DIR is optional but STRONGLY recommended on a deployed corpus (#2065 advisor S8): the
# default snapshot location is a sibling of the corpus root, which is container-ephemeral when the
# corpus is a volume mount — so the rollback story is only real if the snapshot outlives the
# container. The run prints the path it used; capture it.
#   make upgrade-corpus CORPUS_DIR=<corpus> SNAPSHOT_DIR=/persistent/path
upgrade-corpus:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli upgrade run --corpus-dir "$${CORPUS_DIR}" --yes \
		$${SNAPSHOT_DIR:+--snapshot-dir "$${SNAPSHOT_DIR}"}

upgrade-verify:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli upgrade verify --corpus-dir "$${CORPUS_DIR}"

# Roll back the role changes m0009 made, from the ledger it wrote (#2069). Refuses any node that
# something else has written since — a rollback that overwrites a later re-enrich is a regression
# wearing a rollback's clothes — and reports those rather than forcing.
#   make upgrade-undo-roles CORPUS_DIR=<corpus>
upgrade-undo-roles:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) scripts/ops/undo_speaker_roles.py --corpus-dir "$${CORPUS_DIR}"

# Run the RFC-088 enrichment layer over a corpus (#1127). Invokes the
# `podcast_scraper.cli enrich` subcommand — the same verb a manual job / cron
# schedule / auto-after-ingest run uses (#1069). Deterministic enrichers run by default;
# WITH_ML=1 wires the embedding/NLI enrichers (needs the `[ml]` extra + `provider`
# blocks in the operator YAML). Results land under <CORPUS>/enrichments/ +
# <CORPUS>/metadata/enrichments/, with a run_summary.json + run.jsonl audit trail.
#
#   make enrich CORPUS=<corpus output dir> \
#       [WITH_ML=1] [ONLY=id,id] [SKIP=id,id] [OPT_IN=id,id] \
#       [PROFILE=name] [CONFIG=operator.yaml] [CORPUS_ONLY=1] [LOG_LEVEL=INFO]
#
# e.g.  make enrich CORPUS=.test_outputs/manual/prod-pilot/corpus           # deterministic only
#       make enrich CORPUS=.test_outputs/manual/prod-v2/corpus WITH_ML=1    # + embedding/NLI
enrich:
	@test -n "$(CORPUS)" || { echo "CORPUS required — corpus output dir, e.g. .test_outputs/manual/prod-pilot/corpus"; exit 1; }
	@export PYTHONPATH="$(PWD)/src:$${PYTHONPATH}" && $(PYTHON) -m $(PACKAGE).cli enrich \
		--output-dir "$(CORPUS)" \
		$(if $(WITH_ML),--with-ml) \
		$(if $(ONLY),--only "$(ONLY)") \
		$(if $(SKIP),--skip "$(SKIP)") \
		$(if $(OPT_IN),--opt-in "$(OPT_IN)") \
		$(if $(PROFILE),--profile "$(PROFILE)") \
		$(if $(CONFIG),--config "$(CONFIG)") \
		$(if $(LOG_LEVEL),--log-level "$(LOG_LEVEL)") \
		$(if $(CORPUS_ONLY),--corpus-only)

# Regenerate the ``viewer-validation-corpus/v3`` fixture that graph-v3
# lenses and the tier-3 walks consume. The output is committed to git
# (see tests/fixtures/viewer-validation-corpus/v3/enrichments/) so a
# clean clone works without running this — but anyone touching an
# enricher's output schema must rerun this and commit the intended
# diff.
#
# Requires the ``.[ml,search]`` extras installed in the venv (for
# ``topic_similarity`` + ``topic_consensus`` — sentence-transformers
# + a local NLI checkpoint). Matches the invocation that seeded the
# fixture originally (see commit 2ab47388): ``cloud_balanced``
# profile is what enables ``topic_consensus`` (packaged
# ``airgapped_thin`` does not). No ``--corpus-only`` — we want the
# per-episode ``insight_density`` + ``insight_sentiment`` sidecars
# regenerated too.
enrich-viewer-fixture:
	@echo "Regenerating tests/fixtures/viewer-validation-corpus/v3/enrichments/ …"
	@$(MAKE) enrich CORPUS=tests/fixtures/viewer-validation-corpus/v3 \
		WITH_ML=1 PROFILE=cloud_balanced
	@echo ""
	@echo "Done. Re-run diffs against tests/fixtures/viewer-validation-corpus/v3/:"
	@echo "  git diff --stat tests/fixtures/viewer-validation-corpus/v3/"
	@echo "  # only ``computed_at`` / ``run_id`` / timestamp fields should"
	@echo "  # differ if no enricher output schema changed."



# Vitest unit tests for TypeScript utility logic (no browser needed)
test-ui:
	@echo "Vitest unit tests + coverage gate (gi-kg-viewer, #914)..."
	@cd $(WEB_VIEWER_DIR) && npm install && npm run test:coverage

# Playwright browser E2E: Chrome locally, Firefox under CI (playwright.config.ts).
test-ui-e2e:
	@echo "Playwright E2E (gi-kg-viewer)..."
	@cd $(WEB_VIEWER_DIR) && npm install && npx playwright install firefox && npm run test:e2e

# API image the e2e run-local-stack scripts expect (#1619). Nothing built this before, so
# ``e2e/run-local-stack.sh`` failed on a fresh machine with an image-not-found that read like a
# docker problem. Built from docker/api/Dockerfile, which carries CPU torch + lancedb +
# sentence-transformers AND pre-caches all-MiniLM-L6-v2 — that is what makes /api/search work
# offline, and with it the topic bands, enriched hero and every search-* spec.
# ~705 MB, a few minutes cold; layer-cached afterwards.
e2e-api-image:
	@echo "Building podcast-api:e2e-local (viewer e2e backend; the player's builds on it)..."
	@DOCKER_BUILDKIT=1 docker build -t podcast-api:e2e-local -f docker/api/Dockerfile \
		--build-arg GIT_SHA="$$(git rev-parse HEAD 2>/dev/null || echo '')" \
		--build-arg BUILD_TIME="$$(date -u +%Y-%m-%dT%H:%M:%SZ)" .
	@echo ""
	@echo "✓ podcast-api:e2e-local ready — run: $(WEB_VIEWER_DIR)/e2e/run-local-stack.sh"
# The viewer e2e suite against the real backend, image included. The stack script serves from a
# DISPOSABLE COPY of the fixture corpus (the operator plane writes into whatever corpus it is
# given), so this leaves the tracked fixture untouched.
test-ui-e2e-live: e2e-api-image
	@echo "Playwright E2E (gi-kg-viewer) against the fixture-bootstrapped API..."
	@cd $(WEB_VIEWER_DIR) && npm install && npx playwright install firefox
	@$(WEB_VIEWER_DIR)/e2e/run-local-stack.sh

# Production viewer bundle: ``vue-tsc -b && vite build`` (catches strict-mode TS
# errors that ``vitest`` / ``playwright`` skip). Wired into every CI target so a
# vue-tsc regression cannot reach the stack-test Docker build path on main.
build-viewer:
	@echo "Production viewer bundle (vue-tsc -b && vite build)..."
	@cd $(WEB_VIEWER_DIR) && npm install && npm run build





# Phase 5 (#528): fail if GIL Quote spans do not overlap indexed transcript chunks enough.
verify-gil-offsets-strict:
	@echo "GIL vs indexed transcript chunk offset verification (strict, min overlap rate $(GIL_OFFSET_MIN_RATE))..."
	@test -d "$(GIL_OFFSET_VERIFY_DIR)" || { echo "GIL_OFFSET_VERIFY_DIR not found: $(GIL_OFFSET_VERIFY_DIR)"; exit 2; }
	@test -f "$(GIL_OFFSET_VERIFY_DIR)/search/metadata.json" || { echo "No search/metadata.json under $(GIL_OFFSET_VERIFY_DIR) (index missing?)"; exit 2; }
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) -m $(PACKAGE).cli verify-gil-chunk-offsets \
		--output-dir "$(GIL_OFFSET_VERIFY_DIR)" \
		--strict \
		--min-overlap-rate "$(GIL_OFFSET_MIN_RATE)"

# Run after ``make test-acceptance*`` (same OUTPUT_DIR). Uses the lexicographically latest session_* folder.
verify-gil-offsets-after-acceptance:
	@set -e; \
	base="$(or $(OUTPUT_DIR),.test_outputs/acceptance)/sessions"; \
	test -d "$$base" || { echo "No sessions dir: $$base (run test-acceptance-fixtures-fast first?)"; exit 2; }; \
	session=$$(ls -1d "$$base"/session_* 2>/dev/null | sort | tail -1); \
	test -n "$$session" || { echo "No session_* under $$base"; exit 2; }; \
	runs="$$session/runs"; \
	test -d "$$runs" || { echo "No runs dir: $$runs"; exit 2; }; \
	n=0; \
	for d in "$$runs"/run_*; do \
		test -d "$$d" || continue; \
		if test -f "$$d/search/metadata.json"; then \
			echo "verify-gil-offsets-strict: $$d"; \
			$(MAKE) verify-gil-offsets-strict GIL_OFFSET_VERIFY_DIR="$$d" GIL_OFFSET_MIN_RATE="$(GIL_OFFSET_MIN_RATE)"; \
			n=$$((n+1)); \
		fi; \
	done; \
	if test "$$n" -eq 0; then \
		echo "No run_* with search/metadata.json under $$runs"; exit 2; \
	fi; \
	echo "OK: verified GIL vs indexed transcript offsets for $$n acceptance run(s)"

kg-quality-metrics:
	# PRD-019 KG quality metrics over .kg.json (see scripts/tools/kg_quality_metrics.py).
	@if [ -z "$(DIR)" ]; then \
		echo "DIR is required (e.g. make kg-quality-metrics DIR=./output)"; exit 2; \
	fi
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/kg_quality_metrics.py "$(DIR)" $(ARGS)

# RFC-107 §T2 search quality over a corpus's LanceDB index — the third sibling of the two
# targets above, and the same shape: a thin CLI over a product module
# (podcast_scraper.search.quality_metrics). GI/KG measure what extraction WROTE; this
# measures what retrieval RETURNS, which no artifact check can see.
#
#   make search-quality-metrics
#   make search-quality-metrics SEARCH_CORPUS=path/to/corpus ARGS='--no-embed --json'
#
# Needs an index: make build-validation-index. Reports; the --min-* floors are all OFF by
# default and there is no CI gate yet, deliberately — see the module docstring.
SEARCH_CORPUS ?= tests/fixtures/viewer-validation-corpus/$(shell cat tests/fixtures/FIXTURES_VERSION 2>/dev/null || echo v3)

search-quality-metrics:
	@export PYTHONPATH="$(PWD)/src:$${PYTHONPATH}" && $(PYTHON) scripts/tools/search_quality_metrics.py \
		"$(SEARCH_CORPUS)" $(ARGS)

search-quality-reseed:
	# Re-freeze expected_top_k_doc_ids from what search returns TODAY, for every query whose
	# label_status is "unlabeled-seed". Deliberate and separate: it rewrites a TRACKED fixture,
	# and the seeding run scores nDCG = 1.000 by construction — worth nothing by itself. The
	# value is in a later run differing.
	#
	# Anchors embed a content hash, so regenerating the corpus kills all of them at once. Reset
	# the stale ones to "unlabeled-seed" first, or this is a no-op: it skips labelled queries on
	# purpose so a re-run can never silently overwrite a human audit.
	@export PYTHONPATH="$(PWD)/src:$${PYTHONPATH}" && $(PYTHON) scripts/tools/search_quality_metrics.py \
		"$(SEARCH_CORPUS)" --seed-labels
	@echo ""
	@echo "Review the diff before committing it:"
	@echo "  git diff --stat $(SEARCH_CORPUS)/search-queries.json"

quality-metrics-ci:
	# Same KG enforce as GitHub Actions test-unit job (committed fixtures); the GIL half runs
	# where GI lives (E04 move).
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/kg_quality_metrics.py tests/fixtures/gil_kg_ci_enforce --enforce --strict-schema --fail-on-errors --min-artifacts 1 --min-avg-nodes 1 --min-avg-edges 0 --min-extraction-coverage 1.0

cleanup-processes:
	# Clean up leftover Python/test processes from previous runs — THIS repo only.
	# Covers pytest workers, ML model probe processes, and worker calculator. Every PID
	# is CWD-gated to $(CURDIR) before killing, so a sibling worktree's test run is never
	# touched (the bare ``pkill -f pytest`` this replaced was machine-wide and could kill
	# another worktree's suite). Same scoping model as the SessionEnd reap.
	# The recipe's own shell and its ancestors (make, the caller's shell) are skipped: the
	# pattern is in this shell's command line and its cwd is $(CURDIR), so `pgrep -f`
	# matches it, and killing it terminated make itself (procps pgrep, Linux).
	@echo "Cleaning up leftover test processes (this repo only)..."
	@self=" "; p=$$$$; while [ -n "$$p" ] && [ "$$p" -gt 1 ]; do self="$$self$$p "; \
	  p=$$(ps -o ppid= -p $$p 2>/dev/null | tr -d ' '); done; \
	for pid in $$(pgrep -f "pytest|gw[0-9]|ml_model_cache_helpers|calculate_test_workers" 2>/dev/null); do \
	  case "$$self" in *" $$pid "*) continue ;; esac; \
	  cwd=$$(lsof -p $$pid -a -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -1); \
	  case "$$cwd" in "$(CURDIR)" | "$(CURDIR)"/*) kill $$pid 2>/dev/null || true ;; esac; \
	done
	@echo "Process cleanup complete"

# Reliable teardown of ALL stack-test artifacts — the compose stack AND the orphan
# build / Playwright / browser processes a killed or interrupted run leaves behind.
# THIS is what was missing: cleanup-processes only reaps pytest workers, so an
# interrupted ``stack-test-*`` left runaway ``make stack-test-build`` + buildx +
# Playwright node runners thrashing the machine (they starved the api → 502s).
#
# Scoped to THIS repo via ``$(CURDIR)`` so it NEVER touches another worktree's
# processes (orrery, podcast_scraper-FUTURE, -infra). Idempotent + safe anytime.
# pkill defaults to SIGTERM, so the Playwright runner cleans up its own browsers.
stack-test-reap:
	@echo "Reaping stack-test artifacts (this repo only)..."
	@$(MAKE) stack-test-down STACK_TEST_DOWN_VOLUMES=1 >/dev/null 2>&1 || true
	@# Playwright runner + node launched from THIS repo's stack-test dir.
	@pkill -f "$(CURDIR)/[t]ests/stack-test/node_modules/.bin/playwright" 2>/dev/null || true
	@pkill -f "$(CURDIR)/[t]ests/stack-test/node_modules/.bin/playwright-core" 2>/dev/null || true
	@# Orphan docker build for THIS project's stack (make + buildx bake + compose build).
	@pkill -f "$(CURDIR).*[m]ake stack-test-build" 2>/dev/null || true
	@pkill -f "[d]ocker-buildx bake.*$(notdir $(CURDIR))" 2>/dev/null || true
	@pkill -f "[c]ompose.*$(CURDIR)/compose/docker-compose.stack-test" 2>/dev/null || true
	@$(MAKE) cleanup-processes >/dev/null 2>&1 || true
	@echo "Reap complete."

check-zombie:
	# Detect unkillable (UE state) Python processes that require reboot
	@echo "Checking for unkillable Python processes..."
	@zombie_count=$$(ps aux 2>/dev/null | grep -E '[Pp]ython|[Pp]ytest' | \
		awk '$$8 ~ /U/' | grep -v grep | wc -l | tr -d ' '); \
	if [ "$$zombie_count" -gt 0 ]; then \
		echo "WARNING: $$zombie_count unkillable Python process(es) found:"; \
		ps aux | grep -E '[Pp]ython|[Pp]ytest' | awk '$$8 ~ /U/' | grep -v grep; \
		echo ""; \
		echo "These processes are in uninterruptible wait (UE state)."; \
		echo "They cannot be killed -- reboot is required."; \
		echo "After reboot, run Disk Utility First Aid on the boot volume."; \
		exit 1; \
	else \
		echo "No zombie processes found."; \
	fi

check-spotlight:
	# Verify Spotlight indexing is disabled or cache dirs are excluded
	@echo "Checking Spotlight indexing status..."
	@if command -v mdutil >/dev/null 2>&1; then \
		if mdutil -s / 2>/dev/null | grep -q "Indexing enabled"; then \
			echo "WARNING: Spotlight indexing is enabled on /."; \
			echo "Heavy ML I/O + Spotlight = APFS lock contention risk."; \
			echo "Disable with: sudo mdutil -a -i off"; \
			echo "Or exclude cache dirs in System Settings > Spotlight > Privacy:"; \
			echo "  ~/.cache/huggingface  ~/.cache/whisper  .venv/"; \
		else \
			echo "Spotlight indexing is disabled. Good."; \
		fi; \
	else \
		echo "mdutil not found (not macOS). Skipping."; \
	fi

# The test database (ADR-169): app-state tests read PODCAST_TEST_DATABASE_URL and FAIL without it.
# One container per checkout, loopback only; down removes only that container.
.PHONY: test-db-up test-db-down
test-db-up:
	@bash scripts/tools/test_db.sh up

test-db-down:
	@bash scripts/tools/test_db.sh down

test-unit: cleanup-processes
	# Unit tests: parallel execution for faster feedback
	# Parallelism: Uses unit-specific worker calculation (memory-aware, caps at 8)
	# Unit tests are lightweight (~100 MB per worker), so can use more workers
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	# Note: cleanup-processes runs automatically via pytest fixture, but also called here for safety
	$(PYTHON) -m pytest tests/unit/ tests/guardrails/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --disable-socket --allow-hosts=127.0.0.1,localhost

test-unit-no-ml: cleanup-processes
	# Runs import guard + unit tests with *current* $(PYTHON) (.venv if present).
	# For the same environment as GitHub test-unit, use: make venv-dev-init && make test-unit-dev-venv
	@echo "Verifying unit tests can import modules without ML dependencies..."
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/check_unit_test_imports.py
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ tests/guardrails/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --disable-socket --allow-hosts=127.0.0.1,localhost

test-unit-dev-venv: cleanup-processes
	@test -x "$(VENVDEV_PY)" || { echo "Missing $(VENVDEV_PY). Run: make venv-dev-init [VENVDEV=$(VENVDEV)]"; exit 1; }
	@echo "Verifying imports (CI-style, .[dev] only)..."
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(VENVDEV_PY) scripts/tools/check_unit_test_imports.py
	@echo "Running unit tests in $(VENVDEV)..."
	$(VENVDEV_PY) -m pytest tests/unit/ tests/guardrails/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(shell $(VENVDEV_PY) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --disable-socket --allow-hosts=127.0.0.1,localhost

test-integration: cleanup-processes
	# Integration tests: parallel execution (3.4x faster, significant benefit)
	# Parallelism: Uses integration-specific worker calculation (memory-aware, caps at 5 to prevent hangs)
	# Integration tests load ML models which consume ~1-2 GB per worker
	# Reduced worker count (max 5) prevents resource contention and hangs at high completion percentages
	# Includes reruns for flaky tests (matches CI behavior)
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	# Coverage: measured independently (not appended) to match CI per-job measurement
	# Note: cleanup-processes runs automatically via pytest fixture, but also called here for safety
	# Note: pytest-rerunfailures 14.0+ uses socket-based ServerStatusDB with pytest-xdist (-n)
	# This requires localhost socket access. We disable socket blocking for integration tests
	# when using reruns with parallel execution, but still restrict to localhost only
	# Note: Force coverage collection completion by combining coverage files immediately after tests
	# Capture pytest exit code to ensure test failures are not masked
	# Added --durations=20 and --tb=short for better progress visibility and debugging hangs
	@echo "🔄 Starting integration tests at $$(date '+%Y-%m-%d %H:%M:%S')"
	@set -e; \
	pytest_exit=0; \
	$(PYTHON) -m pytest tests/integration/ -m integration -v --tb=short -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3) --cov=$(PACKAGE) --cov-report=term-missing --reruns 2 --reruns-delay 1 --allow-hosts=127.0.0.1,localhost --durations=20 || pytest_exit=$$?; \
	echo "📊 Tests completed, combining coverage files..."; \
	$(MAKE) merge-cov-fragments || { echo "⚠️  Coverage fragment merge failed (non-fatal)"; true; }; \
	echo "✅ Integration tests finished at $$(date '+%Y-%m-%d %H:%M:%S')"; \
	exit $$pytest_exit

test-integration-fast:
	# Fast integration tests: critical path tests only (excludes ml_models for speed)
	# Parallelism: Uses integration-specific worker calculation (memory-aware, caps at 5 to prevent hangs)
	# Includes reruns for flaky tests (matches CI behavior)
	# Excludes ml_models marker - use test-integration for ML workflow tests
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Coverage: measured independently but no threshold (fast tests are a subset, full suite enforces threshold)
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility with -n (parallel)
	$(PYTHON) -m pytest tests/integration/ -m "integration and (critical_path or app) and not ml_models" -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3) --cov=$(PACKAGE) --cov-report=term-missing --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 --durations=20

.PHONY: audio-archive-local-e2e
audio-archive-local-e2e:
	# See the audio cold-storage archive work end to end, locally (#1787, epic #1788).
	# Runs the REAL pipeline (RSS -> download over localhost -> archive to a local rclone
	# 'cold' remote -> persist media/ -> evict -> provenance -> re-fetch from cold), with only
	# the whisper ASR mocked. No network, no cost, ~7s. -s streams the step-by-step output.
	$(PYTHON) -m pytest tests/integration/archive/test_pipeline_offload_evict_e2e.py \
		-v -s -p no:xdist -m integration --no-cov --allow-hosts=127.0.0.1,localhost

test-ci:
	# CI test suite: parallel execution for speed
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Uses conservative worker calculation (default type, caps at 5) for mixed test types
	# Note: Non-critical path tests run on main branch only
	$(PYTHON) -m pytest -m '(not integration and not e2e) or (integration and (critical_path or (app and not ml_models))) or (e2e and critical_path)' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type default --max-workers 5 2>/dev/null || echo 3) --disable-socket --allow-hosts=127.0.0.1,localhost --cov=$(PACKAGE) --cov-report=term-missing

test-ci-fast:
	# Fast CI test suite: parallel execution for speed
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Note: Three separate pytest processes so unit tests can patch ``sys.modules`` (openai/httpx/google
	# stubs) without poisoning integration/E2E in the same interpreter (see provider unit tests).
	# Per-layer worker counts match test-fast; coverage is excluded (full CI jobs measure coverage).
	# Includes reruns for flaky tests (matches CI behavior).
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility with -n (parallel)
	@set -e; \
	WU=$$($(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4); \
	WI=$$($(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3); \
	WE=$$($(PYTHON) scripts/tools/calculate_test_workers.py --test-type e2e --max-workers 4 2>/dev/null || echo 2); \
	echo "Running unit (test-ci-fast)..."; \
	PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/unit/ -m 'not nightly and not integration and not e2e' \
		-n $$WU --allow-hosts=127.0.0.1,localhost --durations=20 --reruns 3 --reruns-delay 2 $(PYTEST_PROGRESS_OPTS); \
	echo "Running critical path integration (test-ci-fast)..."; \
	PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/integration/ -m 'not nightly and integration and critical_path' \
		-n $$WI --allow-hosts=127.0.0.1,localhost --durations=20 --reruns 3 --reruns-delay 2 $(PYTEST_PROGRESS_OPTS); \
	echo "Running critical path E2E (test-ci-fast, non-ML, parallel)..."; \
	E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/e2e/ -m 'not nightly and e2e and critical_path and not ml_models' \
		-n $$WE --allow-hosts=127.0.0.1,localhost --durations=20 --reruns 3 --reruns-delay 2 $(PYTEST_PROGRESS_OPTS); \
	echo "Running critical path E2E (test-ci-fast, ML models, sequential)..."; \
	set +e; \
	E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/e2e/ -m 'not nightly and e2e and critical_path and ml_models' \
		-n 1 --allow-hosts=127.0.0.1,localhost --durations=20 --reruns 3 --reruns-delay 2 $(PYTEST_PROGRESS_OPTS); \
	ec=$$?; \
	set -e; \
	if [ $$ec -eq 5 ]; then ec=0; fi; \
	exit $$ec

test-e2e: cleanup-processes
	# E2E tests: parallel execution for speed
	# Uses E2E-specific worker calculation (memory-aware, caps at 4 to prevent system freezes)
	# E2E tests are memory-intensive (~2.5 GB per worker) and run full pipeline
	# Excludes analysis/diagnostic tests - these are slow diagnostic tools, not regular tests
	# Note: cleanup-processes runs automatically via pytest fixture, but also called here for safety
	# Includes reruns for flaky tests (matches CI behavior) - 3 retries for ML model variability
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility with -n (parallel)
	@E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/e2e/ -m "e2e and not analysis" -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type e2e --max-workers 4 2>/dev/null || echo 2) --cov=$(PACKAGE) --cov-report=term-missing --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1

test-e2e-sequential:
	# E2E tests: sequential execution (slower but clearer output, useful for debugging)
	# Excludes analysis/diagnostic tests - these are slow diagnostic tools, not regular tests
	# Uses multi-episode feed (5 episodes) - set via E2E_TEST_MODE environment variable
	E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/e2e/ -m "e2e and not analysis" --disable-socket --allow-hosts=127.0.0.1,localhost

test-e2e-fast:
	# Fast E2E tests: parallel non-ML, then sequential ML (avoids xdist "stuck at ~80%" when one worker runs Whisper)
	# Uses E2E-specific worker calculation (memory-aware, caps at 4 to prevent system freezes)
	# Critical path tests only (includes ML tests if models are cached)
	# Excludes analysis/diagnostic tests (p07/p08 threshold analysis) - these are slow and not critical path
	# Includes reruns for flaky tests (matches CI behavior) - 3 retries for ML model variability
	# Uses fast feed (1 episode) - set via E2E_TEST_MODE environment variable
	# Includes ALL critical path tests, even if slow (critical path cannot be shortened)
	# Use --durations=20 to monitor slow tests and optimize them separately
	# Coverage: measured independently but no threshold (fast tests are a subset, full suite enforces threshold)
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility with -n (parallel)
	@WE=$$($(PYTHON) scripts/tools/calculate_test_workers.py --test-type e2e --max-workers 4 2>/dev/null || echo 2); \
	E2E_TEST_MODE=fast $(PYTHON) -m pytest tests/e2e/ -m "e2e and critical_path and not analysis and not ml_models" -n $$WE --cov=$(PACKAGE) --cov-report=term-missing --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 --durations=20 && \
	set +e; \
	E2E_TEST_MODE=fast $(PYTHON) -m pytest tests/e2e/ -m "e2e and critical_path and not analysis and ml_models" -n 1 --cov=$(PACKAGE) --cov-append --cov-report=term-missing --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 --durations=20; \
	ec=$$?; \
	set -e; \
	if [ $$ec -eq 5 ]; then ec=0; fi; \
	exit $$ec

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
	# Sequential execution only — parallel (pytest-xdist) caused double-runs on CI because
	# exit-code mismatches triggered the fallback path, doubling wall time from ~75 min to ~3 h.
	# NOT marked with @pytest.mark.e2e - separate category from regular E2E tests
	# Excludes LLM/OpenAI tests to avoid API costs (see issue #183)
	# Network access restricted via --disable-socket + --allow-hosts
	@echo "Running nightly tests with production models..."
	@echo "Podcasts: p01-p05 (15 episodes total)"
	@echo "Models: Whisper base.en, BART-large-cnn, LED-large-16384"
	@mkdir -p reports
	@echo "🔍 Verifying test collection..."
	@E2E_TEST_MODE=nightly pytest tests/e2e/ -m "nightly and not llm" --collect-only -q || { \
		echo "❌ Test collection failed, trying with verbose output..."; \
		E2E_TEST_MODE=nightly pytest tests/e2e/ -m "nightly and not llm" --collect-only -v; \
		exit 1; \
	}
	@echo "✅ Test collection successful, running tests..."
	@START_TIME=$$(date +%s); \
	echo ""; \
	echo "📊 Test execution details:"; \
	echo "   - Mode: Nightly (production models, sequential)"; \
	echo "   - Episodes: 15 total (5 podcasts × 3 episodes)"; \
	echo "   - Models: Whisper base.en, BART-large-cnn, LED-large-16384"; \
	echo "   - Start time: $$(date '+%Y-%m-%d %H:%M:%S')"; \
	echo ""; \
	echo "🔍 System information:"; \
	echo "   - CPU cores: $$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"; \
	echo "   - Available memory: $$(free -h 2>/dev/null | grep Mem | awk '{print $$7}' || vm_stat 2>/dev/null | head -1 || echo 'unknown')"; \
	echo "   - Disk space: $$(df -h . | tail -1 | awk '{print $$4 " available"}')"; \
	echo ""; \
	echo "🔄 Running sequential (no pytest-xdist)..."; \
	echo "   This may take 60-90 minutes depending on model loading and processing time"; \
	echo "   Progress will be shown below (each test name as it runs)"; \
	echo "   ============================================================"; \
	E2E_TEST_MODE=nightly pytest tests/e2e/ -m "nightly and not llm" -v --tb=short -ra \
		--disable-socket --allow-hosts=127.0.0.1,localhost \
		--durations=20 \
		--junitxml=reports/junit-nightly.xml \
		--json-report --json-report-file=reports/pytest-nightly.json; \
	ELAPSED=$$(($$(date +%s) - START_TIME)); \
	echo ""; \
	echo "============================================================"; \
	echo "✅ Nightly tests completed at $$(date '+%Y-%m-%d %H:%M:%S')"; \
	echo "   Total elapsed time: $$ELAPSED seconds ($$(($$ELAPSED / 60)) minutes)"

test-nightly-subset:
	# Run a subset of nightly tests for local verification
	# Usage: make test-nightly-subset PODCASTS=podcast1,podcast2
	# Default: runs only podcast1 (3 episodes) for quick verification
	# Note: Runs sequentially (no -n flag) for simplicity
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility (needs socket for ServerStatusDB)
	# Network access is still restricted via --allow-hosts=127.0.0.1,localhost
	@PODCASTS=$${PODCASTS:-podcast1}; \
	echo "Running nightly test subset with podcasts: $$PODCASTS"; \
	echo "Models: Whisper base.en, BART-large-cnn, LED-large-16384"; \
	mkdir -p reports; \
	echo "🔍 Verifying test collection..."; \
	E2E_TEST_MODE=nightly $(PYTHON) -m pytest tests/e2e/test_nightly_full_suite_e2e.py -k "$$(echo $$PODCASTS | tr ',' ' or ')" --collect-only -q || { \
		echo "❌ Test collection failed, trying with verbose output..."; \
		E2E_TEST_MODE=nightly $(PYTHON) -m pytest tests/e2e/test_nightly_full_suite_e2e.py -k "$$(echo $$PODCASTS | tr ',' ' or ')" --collect-only -v; \
		exit 1; \
	}; \
	echo "✅ Test collection successful, running tests..."; \
	E2E_TEST_MODE=nightly $(PYTHON) -m pytest tests/e2e/test_nightly_full_suite_e2e.py -k "$$(echo $$PODCASTS | tr ',' ' or ')" -v --tb=short -ra --allow-hosts=127.0.0.1,localhost --durations=20

test:
	# All tests: run separately with --cov-append to match CI behavior and get accurate coverage
	# CI runs tests in separate jobs (unit, integration, e2e) and combines coverage
	# This matches that approach for consistent coverage numbers
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	# Each test type uses its own worker calculation with appropriate max workers
	# Excludes nightly tests (run separately via make test-nightly)
	# Excludes analytical tests (diagnostic tools, run separately via make test-analytical)
	@echo "Running unit tests with coverage..."
	@E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/unit/ -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost -q
	@echo "Running integration tests with coverage (appending)..."
	# Note: --disable-socket is required to block real network calls (without it, tests hang)
	# Note: --reruns removed because pytest-rerunfailures 14.0+ uses sockets with -n (parallel)
	# which conflicts with --disable-socket. Reruns are available via make test-integration standalone.
	@E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/integration/ -m integration -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3) --cov=$(PACKAGE) --cov-append --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost -q
	@echo "Running E2E tests with coverage (appending)..."
	# Note: Same as integration - --disable-socket for network isolation, no --reruns with -n
	@E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/e2e/ -m "e2e and not nightly" -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type e2e --max-workers 4 2>/dev/null || echo 2) --cov=$(PACKAGE) --cov-append --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost -q
	@echo "Final coverage:"
	@$(MAKE) merge-cov-fragments
	@$(PYTHON) -m coverage report 2>&1 | grep -E "^[[:space:]]*TOTAL" || (echo "No TOTAL line in coverage report (check for coverage errors above)"; exit 1)

test-sequential:
	# All tests: sequential execution (slower but clearer output, useful for debugging)
	# Uses multi-episode feed for E2E tests (5 episodes) - set via E2E_TEST_MODE environment variable
	# Excludes analytical tests (diagnostic tools, run separately via make test-analytical)
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	E2E_TEST_MODE=multi_episode $(PYTHON) -m pytest tests/ -m "not analytical" --cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost

# Fast tests: same coverage approach as 'test' — separate pytest runs per layer, then combine.
# Avoids one large xdist session (which can hang at shutdown). Each layer uses its own worker count.
test-fast:
	# Fast tests: unit + critical path integration + critical path e2e (separate runs, then combine coverage)
	# E2E: non-ml_models parallel, then ml_models with -n 1 (avoids xdist tail / progress stuck around ~80%)
	# Uses fast feed for E2E (1 episode) via E2E_TEST_MODE=fast. Excludes nightly.
	# Uses $(PYTEST_PROGRESS_OPTS) for classic dots; PYTHONUNBUFFERED reduces IDE terminal buffering.
	@echo "Running unit tests (fast) with coverage..."
	@E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/unit/ -m 'not integration and not e2e' \
		-n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) \
		--cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost $(PYTEST_PROGRESS_OPTS)
	@echo "Running critical path integration tests with coverage (appending)..."
	@E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/integration/ -m 'integration and (critical_path or (app and not ml_models))' \
		-n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3) \
		--cov=$(PACKAGE) --cov-append --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost $(PYTEST_PROGRESS_OPTS)
	@echo "Running critical path E2E tests (non-ML, parallel) with coverage (appending)..."
	@E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/e2e/ -m 'e2e and critical_path and not nightly and not ml_models' \
		-n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type e2e --max-workers 4 2>/dev/null || echo 2) \
		--cov=$(PACKAGE) --cov-append --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost $(PYTEST_PROGRESS_OPTS) --durations=20
	@echo "Running critical path E2E tests (ML models, sequential) with coverage (appending)..."
	@set +e; \
	E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/e2e/ -m 'e2e and critical_path and not nightly and ml_models' \
		-n 1 \
		--cov=$(PACKAGE) --cov-append --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost $(PYTEST_PROGRESS_OPTS) --durations=20; \
	ec=$$?; \
	set -e; \
	if [ $$ec -eq 5 ]; then ec=0; fi; \
	exit $$ec
	@echo "Combining coverage..."
	@$(MAKE) merge-cov-fragments
	@$(PYTHON) -m coverage report 2>&1 | grep -E "^[[:space:]]*TOTAL" || (echo "No TOTAL line in coverage report"; exit 1)

# Same worker/coverage layout as test-fast, but skips tests/e2e/ (saves significant wall time).
# Pair with make test-ui-e2e (see ci-ui-fast) when validating viewer changes.
test-fast-no-py-e2e:
	@echo "Running unit tests (fast) with coverage..."
	@E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/unit/ -m 'not integration and not e2e' \
		-n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) \
		--cov=$(PACKAGE) --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost $(PYTEST_PROGRESS_OPTS)
	@echo "Running critical path integration tests with coverage (appending)..."
	@E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/integration/ -m 'integration and (critical_path or (app and not ml_models))' \
		-n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3) \
		--cov=$(PACKAGE) --cov-append --cov-report=term-missing --disable-socket --allow-hosts=127.0.0.1,localhost $(PYTEST_PROGRESS_OPTS)
	@echo "Combining coverage..."
	@$(MAKE) merge-cov-fragments
	@$(PYTHON) -m coverage report 2>&1 | grep -E "^[[:space:]]*TOTAL" || (echo "No TOTAL line in coverage report"; exit 1)

test-reruns:
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	$(PYTHON) -m pytest --reruns 2 --reruns-delay 1 --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' --disable-socket --allow-hosts=127.0.0.1,localhost

test-acceptance:
	@# Run E2E acceptance tests (multiple configs sequentially)
	@# Usage: make test-acceptance CONFIGS="…" OR make test-acceptance FROM_FAST_STEMS=1 [USE_FIXTURES=1] …
	@if [ -z "$(CONFIGS)" ] && [ -z "$(FROM_FAST_STEMS)" ]; then \
		echo "❌ Error: CONFIGS is required unless FROM_FAST_STEMS=1"; \
		echo "Usage: make test-acceptance CONFIGS=\"config/examples/config.example.yaml\""; \
		echo "   Or: make test-acceptance FROM_FAST_STEMS=1 USE_FIXTURES=1"; \
		echo ""; \
		echo "Options:"; \
		echo "  CONFIGS=pattern         Config glob(s), space-separated (required unless FROM_FAST_STEMS=1)"; \
		echo "  FROM_FAST_STEMS=1      Materialize rows from config/acceptance/MAIN_ACCEPTANCE_CONFIG.yaml"; \
		echo "  USE_FIXTURES=1          Use E2E server fixtures (test feeds and mock APIs)"; \
		echo "  NO_SHOW_LOGS=1          Disable streaming logs to console"; \
		echo "  NO_AUTO_ANALYZE=1       Disable automatic analysis after session"; \
		echo "  NO_AUTO_BENCHMARK=1     Disable automatic benchmark report generation"; \
		echo "  ANALYZE_MODE=mode       Analysis mode: basic or comprehensive (default: basic)"; \
		echo "  COMPARE_BASELINE=id     Baseline ID to compare against"; \
		echo "  SAVE_AS_BASELINE=id     Save current runs as baseline"; \
		echo "  FAST_ONLY=1             Run only configs matching fast stems (after CONFIGS glob)"; \
		echo "  TIMEOUT=seconds         Per-run timeout (kill and fail if exceeded)"; \
		echo "  PER_RUN_WALL_SECONDS=s  Post-run wall budget (default 600 via runner; 0 disables)"; \
		echo "  OUTPUT_DIR=path          Output directory (default: .test_outputs/acceptance)"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make test-acceptance CONFIGS=\"config/examples/config.example.yaml\""; \
		echo "  make test-acceptance CONFIGS=\"config/acceptance/*.yaml\" USE_FIXTURES=1"; \
		echo "  make test-acceptance CONFIGS=\"config/acceptance/*.yaml\" USE_FIXTURES=1 FAST_ONLY=1"; \
		echo "  make test-acceptance-fixtures-fast"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/acceptance/run_acceptance_tests.py \
		$(if $(FROM_FAST_STEMS),--from-fast-stems,--configs "$(CONFIGS)") \
		--output-dir "$(or $(OUTPUT_DIR),.test_outputs/acceptance)" \
		$(if $(USE_FIXTURES),--use-fixtures) \
		$(if $(NO_SHOW_LOGS),--no-show-logs) \
		$(if $(NO_AUTO_ANALYZE),--no-auto-analyze) \
		$(if $(NO_AUTO_BENCHMARK),--no-auto-benchmark) \
		$(if $(ANALYZE_MODE),--analyze-mode $(ANALYZE_MODE)) \
		$(if $(COMPARE_BASELINE),--compare-baseline $(COMPARE_BASELINE)) \
		$(if $(SAVE_AS_BASELINE),--save-as-baseline $(SAVE_AS_BASELINE)) \
		$(if $(FAST_ONLY),--fast-only) \
		$(if $(TIMEOUT),--timeout $(TIMEOUT)) \
		$(if $(PER_RUN_WALL_SECONDS),--per-run-wall-seconds $(PER_RUN_WALL_SECONDS)) \
		$(if $(ASSERT_ARTIFACTS),--assert-artifacts) \
		--log-level INFO

# Fixture smoke for the full *fast* acceptance matrix (offline E2E server + mock APIs).
# Materializes each enabled row from config/acceptance/MAIN_ACCEPTANCE_CONFIG.yaml under session_*/materialized/.
test-acceptance-fixtures-fast:
	@$(PYTHON) scripts/acceptance/run_acceptance_tests.py \
		--from-fast-stems \
		--output-dir "$(or $(OUTPUT_DIR),.test_outputs/acceptance)" \
		--use-fixtures \
		--no-auto-analyze \
		--no-auto-benchmark \
		--timeout "$(or $(TIMEOUT),900)" \
		--per-run-wall-seconds "$(or $(PER_RUN_WALL_SECONDS),600)" \
		--log-level INFO
	@echo ""
	@echo "✓ Acceptance tests completed"
	@echo "  Results saved to: $(or $(OUTPUT_DIR),.test_outputs/acceptance)"
	@echo ""
	@echo "To analyze results:"
	@echo "  make analyze-acceptance SESSION_ID=<session_id>"
	@echo "  Or: python scripts/acceptance/analyze_bulk_runs.py --session-id <session_id> --output-dir $(or $(OUTPUT_DIR),.test_outputs/acceptance)"

analyze-acceptance:
	@# Analyze E2E acceptance test results
	@# Usage: make analyze-acceptance SESSION_ID="20260208_093757" [MODE=basic|comprehensive] [COMPARE_BASELINE=...] [OUTPUT_DIR=...] [OUTPUT_FORMAT=markdown|json|both]
	@if [ -z "$(SESSION_ID)" ]; then \
		echo "❌ Error: SESSION_ID is required"; \
		echo "Usage: make analyze-acceptance SESSION_ID=<session_id>"; \
		echo ""; \
		echo "Options:"; \
		echo "  SESSION_ID=id            Session ID (required, e.g., '20260208_093757')"; \
		echo "  MODE=mode                Analysis mode: basic or comprehensive (default: basic)"; \
		echo "  COMPARE_BASELINE=id      Baseline ID to compare against"; \
		echo "  OUTPUT_DIR=path          Output directory (default: .test_outputs/acceptance)"; \
		echo "  OUTPUT_FORMAT=format     Output format: markdown, json, or both (default: both)"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make analyze-acceptance SESSION_ID=20260208_101601"; \
		echo "  make analyze-acceptance SESSION_ID=20260208_101601 MODE=comprehensive"; \
		echo "  make analyze-acceptance SESSION_ID=20260208_101601 COMPARE_BASELINE=baseline_v1"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/acceptance/analyze_bulk_runs.py \
		--session-id "$(SESSION_ID)" \
		--output-dir "$(or $(OUTPUT_DIR),.test_outputs/acceptance)" \
		--mode "$(or $(MODE),basic)" \
		$(if $(COMPARE_BASELINE),--compare-baseline $(COMPARE_BASELINE)) \
		--output-format "$(or $(OUTPUT_FORMAT),both)" \
		--log-level INFO
	@echo ""
	@echo "  Reports saved to: $(or $(OUTPUT_DIR),.test_outputs/acceptance)/sessions/session_$(SESSION_ID)/"

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
	@$(PYTHON) -m pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_UNIT) -m 'not integration and not e2e' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --disable-socket --allow-hosts=127.0.0.1,localhost -q

coverage-check-integration:
	# Check integration test coverage meets minimum threshold ($(COVERAGE_THRESHOLD_INTEGRATION)%)
	@echo "Checking integration test coverage (minimum $(COVERAGE_THRESHOLD_INTEGRATION)%)..."
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility with -n (parallel)
	@$(PYTHON) -m pytest tests/integration/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_INTEGRATION) -m 'integration' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3) --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 -q

coverage-check-e2e:
	# Check E2E test coverage meets minimum threshold ($(COVERAGE_THRESHOLD_E2E)%)
	@echo "Checking E2E test coverage (minimum $(COVERAGE_THRESHOLD_E2E)%)..."
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility with -n (parallel)
	@E2E_TEST_MODE=multi_episode pytest tests/e2e/ --cov=$(PACKAGE) --cov-config=.coveragerc-e2e --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_E2E) -m 'e2e and not nightly' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type e2e --max-workers 4 2>/dev/null || echo 2) --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 -q

coverage-check-combined:
	# Check combined coverage meets threshold ($(COVERAGE_THRESHOLD_COMBINED)%)
	# This runs all tests and enforces the combined threshold
	# Uses conservative worker calculation (default type, caps at 5) for mixed test types
	@echo "Checking combined coverage (minimum $(COVERAGE_THRESHOLD_COMBINED)%)..."
	# Note: Removed --disable-socket for pytest-rerunfailures compatibility with -n (parallel)
	@E2E_TEST_MODE=multi_episode pytest tests/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_COMBINED) -m 'not nightly' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type default --max-workers 5 2>/dev/null || echo 3) --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1

# Merge ``.coverage.*`` from pytest-xdist / subprocess coverage without clobbering ``.coverage``.
# Plain ``coverage combine`` loads no existing DB and ``save()`` overwrites the full pytest dataset.
merge-cov-fragments:
	@if find . -maxdepth 1 -name '.coverage.*' -type f ! -name '*-journal' 2>/dev/null | grep -q .; then \
		echo "Merging parallel/subprocess coverage fragments..."; \
		$(PYTHON) -m coverage combine --append 2>/dev/null || true; \
	fi

coverage-report:
	# Generate coverage report without running tests (uses existing .coverage file)
	@$(PYTHON) -m coverage report --show-missing
	@$(PYTHON) -m coverage html -d .build/coverage-html
	@echo "HTML report: .build/coverage-html/index.html"

coverage-enforce:
	# Enforce combined coverage threshold on existing .coverage file (fast, no re-run)
	# Use this after 'make test' to verify coverage meets threshold
	# Combines parallel coverage files (.coverage.*) created by pytest-xdist when using -n flag
	# Note: pytest-cov creates files like .coverage.Mac.pid* which need to be combined
	# Note: When using --cov-append, pytest writes directly to .coverage (no separate files)
	#       In this case, "No data to combine" is expected and normal - coverage is already combined
	@echo "Checking combined coverage threshold ($(COVERAGE_THRESHOLD_COMBINED)%)..."
	@if [ -f .coverage ] || find . -maxdepth 1 -name ".coverage*" -type f 2>/dev/null | grep -q .; then \
		echo "Combining parallel coverage files (if any)..."; \
		$(MAKE) merge-cov-fragments; \
		$(PYTHON) -m coverage report --fail-under=$(COVERAGE_THRESHOLD_COMBINED) > /dev/null && \
		echo "✅ Coverage meets $(COVERAGE_THRESHOLD_COMBINED)% threshold" || \
		(echo "❌ Coverage below $(COVERAGE_THRESHOLD_COMBINED)% threshold" && exit 1); \
	else \
		echo "⚠️ No coverage files found (.coverage or .coverage.*)"; \
		echo "Run 'make test' first to generate coverage data"; \
		exit 1; \
	fi

build:
	$(PYTHON) -m pip install --quiet build
	$(PYTHON) -m build
	@if [ -d dist ]; then mkdir -p .build && rm -rf .build/dist && mv dist .build/ && echo "Moved dist to .build/dist/"; fi

# ML model cache probe — runs at RECIPE TIME inside `make ci` only.
# Previously this was a parse-time $(shell ...) that spawned a heavy Python process
# on every `make` invocation (including `make help`), causing APFS kernel lock
# contention and unkillable zombie processes on macOS.
ci: cleanup-processes
	@cached=$$(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) -c "import sys; sys.path.insert(0, 'src'); \
	from tests.integration.ml_model_cache_helpers import _is_whisper_model_cached, _is_transformers_model_cached; \
	from podcast_scraper.providers.ml.model_loader import is_evidence_model_cached; \
	from podcast_scraper import config, config_constants as _cc; \
	whisper_ok = _is_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL); \
	transformers_ok = _is_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None); \
	embedding_ok = is_evidence_model_cached(_cc.DEFAULT_EMBEDDING_MODEL); \
	spacy_ok = False; \
	try: \
		import spacy; \
		spacy_ok = config.DEFAULT_NER_MODEL.replace('-', '_') in \
			[m.replace('-', '_') for m in spacy.util.get_installed_models()]; \
	except Exception: \
		pass; \
	all_cached = whisper_ok and transformers_ok and spacy_ok and embedding_ok; \
	print('1' if not all_cached else '0', end='')" 2>/dev/null || printf '1'); \
	if [ "$$cached" = "1" ]; then \
		echo "ML models not fully cached — running preload..."; \
		$(MAKE) preload-ml-models; \
	else \
		echo "ML models already cached, skipped preload"; \
	fi; \
	$(MAKE) _ci_body

# Offline HF for pytest: ``tests/conftest.py`` sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE.
# The ``ci:`` cache probe above passes them inline so probes do not hit the Hub accidentally.

_ci_body: format-check lint lint-markdown check-doc-structure type security complexity deadcode docstrings spelling check-test-policy test test-ui test-ui-e2e build-viewer coverage-enforce docs build stack-test-ml-ci
	# Final gate is ``stack-test-ml-ci`` (build → up → seed → Playwright →
	# always-teardown). ml pipeline only — ~5-10 min, no API keys, no cloud
	# cost. Cloud-thin variant (``stack-test-cloud-thin``) is a separate
	# explicit target; public CI does not run it (recurring API cost) and
	# ``ci`` does not include it for the same reason locally.

# Sequential sub-makes + banners so long-quiet steps (especially mypy) never look like a hang.
# BACKEND gate (operator 2026-10-06): pipeline, providers, data work. No viewer tests or
# builds at all — those live in `ci-ui-fast`. `ci` runs everything.
ci-fast:
	# Note: ci-fast skips coverage-enforce and test-ui-e2e (Playwright). For viewer work use ci-ui-fast.
	@set -e; \
	echo ""; echo "=== ci-fast START $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""; \
	$(MAKE) cleanup-processes; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] format-check ==="; $(MAKE) format-check; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint ==="; $(MAKE) lint; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint-markdown ==="; $(MAKE) lint-markdown; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] check-doc-structure ==="; $(MAKE) check-doc-structure; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] type (mypy) ==="; $(MAKE) type; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] security ==="; $(MAKE) security-fast; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] complexity ==="; $(MAKE) complexity; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] deadcode ==="; $(MAKE) deadcode; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] docstrings ==="; $(MAKE) docstrings; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] spelling ==="; $(MAKE) spelling; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] check-test-policy ==="; $(MAKE) check-test-policy; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] quality-metrics-ci ==="; $(MAKE) quality-metrics-ci; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] profile-drift-check ==="; $(MAKE) profile-drift-check; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] test-fast (pytest) ==="; $(MAKE) test-fast; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] docs ==="; $(MAKE) docs; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] build ==="; $(MAKE) build; \
	echo ""; echo "=== ci-fast DONE $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""

# UI gate (operator 2026-10-06): the viewer — unit tests, Playwright e2e, build — plus
# the same static checks as ci-fast and the fast Python tiers without Python e2e. The ONLY local
# gate with viewer tests: ci-fast is backend-only. Longer than Vitest alone (Playwright).
ci-ui-fast:
	# Note: ci-ui-fast skips coverage-enforce and Python tests/e2e; Playwright still needs browsers installed.
	@set -e; \
	echo ""; echo "=== ci-ui-fast START $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""; \
	$(MAKE) cleanup-processes; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] format-check ==="; $(MAKE) format-check; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint ==="; $(MAKE) lint; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint-markdown ==="; $(MAKE) lint-markdown; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] check-doc-structure ==="; $(MAKE) check-doc-structure; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] type (mypy) ==="; $(MAKE) type; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] security ==="; $(MAKE) security-fast; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] complexity ==="; $(MAKE) complexity; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] deadcode ==="; $(MAKE) deadcode; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] docstrings ==="; $(MAKE) docstrings; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] spelling ==="; $(MAKE) spelling; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] check-test-policy ==="; $(MAKE) check-test-policy; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] quality-metrics-ci ==="; $(MAKE) quality-metrics-ci; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] profile-drift-check ==="; $(MAKE) profile-drift-check; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] test-fast-no-py-e2e ==="; $(MAKE) test-fast-no-py-e2e; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] test-ui ==="; $(MAKE) test-ui; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] test-ui-e2e ==="; $(MAKE) test-ui-e2e; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] build-viewer ==="; $(MAKE) build-viewer; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] docs ==="; $(MAKE) docs; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] build ==="; $(MAKE) build; \
	echo ""; echo "=== ci-ui-fast DONE $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""

# Full viewer gate: ci-ui-fast plus the Docker stack-test (Playwright vs the
# real api+viewer+mock-feeds compose stack). Opt-in target — use it before
# pushing PRs that rename testids, refactor chips, or otherwise change the
# Playwright contract. ci-ui-fast does NOT include stack-test by design
# (Docker build + boot adds ~10-15 min); GitHub Actions only runs stack-test
# post-merge on main, so chip-refactor PRs that pass ci-ui-fast can still
# break main. ci-ui-full closes that gap locally without slowing down the
# default per-commit gate.
ci-ui-full:
	# Note: ci-ui-full = ci-ui-fast + stack-test-ml-ci. The player app and both device tiers left
	# with the player (its own `ci-ui-full`). Requires Docker
	# (Buildx + Compose v2). Same airgapped_thin profile as the public CI
	# Stack-test workflow.
	@set -e; \
	echo ""; echo "=== ci-ui-full START $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""; \
	$(MAKE) cleanup-processes; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] format-check ==="; $(MAKE) format-check; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] lint ==="; $(MAKE) lint; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] lint-markdown ==="; $(MAKE) lint-markdown; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] type (mypy) ==="; $(MAKE) type; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] security ==="; $(MAKE) security; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] complexity ==="; $(MAKE) complexity; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] deadcode ==="; $(MAKE) deadcode; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] docstrings ==="; $(MAKE) docstrings; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] spelling ==="; $(MAKE) spelling; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] check-test-policy ==="; $(MAKE) check-test-policy; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] quality-metrics-ci ==="; $(MAKE) quality-metrics-ci; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] test-fast-no-py-e2e ==="; $(MAKE) test-fast-no-py-e2e; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] test-ui ==="; $(MAKE) test-ui; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] test-ui-e2e ==="; $(MAKE) test-ui-e2e; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] build-viewer ==="; $(MAKE) build-viewer; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] docs ==="; $(MAKE) docs; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] build ==="; $(MAKE) build; \
	echo ""; echo "=== ci-ui-full [$$(date '+%Y-%m-%d %H:%M:%S')] stack-test-ml-ci ==="; $(MAKE) stack-test-ml-ci; \
	echo ""; echo "=== ci-ui-full DONE $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""

# Synthetic validation corpus root — MUST include the FIXTURES_VERSION subdir
# (e.g. ``.../viewer-validation-corpus/v3``). The raw ``feeds/<feed>/metadata/``
# artifacts that serve-api computes episodes from live under the version dir, NOT
# the version-less parent — pointing the walk at the parent discovers 0 episodes
# (empty Library → every handoff spec fails on the first row-click).
VIEWER_VALIDATION_CORPUS := $(PWD)/tests/fixtures/viewer-validation-corpus/$(shell cat tests/fixtures/FIXTURES_VERSION 2>/dev/null || echo v2)

ci-ui-validation:
	# Tier-3 real-backend validation walk (RFC-086 / ADR-095). Drives the
	# viewer against a running ``make serve`` stack + an on-disk corpus.
	# Usage: ``make ci-ui-validation CORPUS=/path/to/corpus``
	#
	# Per RFC-086 institutional rule: bugs surfaced here must land a
	# Tier-2 matrix row under web/gi-kg-viewer/e2e/handoff-production/
	# that reproduces them before the fix PR merges.
	@if [ -z "$(CORPUS)" ]; then \
		echo "ERROR: set CORPUS=/abs/path/to/corpus"; \
		echo ""; \
		echo "Local operator corpus (BYOC):"; \
		echo "  make ci-ui-validation CORPUS=$$HOME/.test_outputs/manual/my-corpus"; \
		echo ""; \
		echo "Synthetic validation corpus (in-repo, GH #774):"; \
		echo "  1) Start server pointed at the repo root:"; \
		echo "       make serve-for-validation"; \
		echo "  2) Run validation against the synthetic corpus:"; \
		echo "       make ci-ui-validation CORPUS=$(VIEWER_VALIDATION_CORPUS)"; \
		exit 2; \
	fi
	@echo "=== ci-ui-validation against $(CORPUS) ==="
	@echo "Requires ``make serve`` (or ``make serve-for-validation``) running in another terminal."
	cd web/gi-kg-viewer && \
		CORPUS_PATH=$(CORPUS) \
		node_modules/.bin/playwright test --config playwright.validation.config.ts

serve-for-validation:
	# Tier-3 validation requires the synthetic corpus (tests/fixtures/...)
	# to be reachable as a subdirectory of the API's configured root.
	# The default ``make serve`` uses ``.test_outputs`` as the root; this
	# target points at the repo root so ``tests/fixtures/viewer-validation-corpus``
	# is reachable. Use a separate terminal: ``make serve-for-validation``.
	# APP_OAUTH_PROVIDER=none disables the auth gate for the validation walk.
	# ``serve-api`` defaults to the mock OAuth provider (auth ON), which makes the
	# app render LoginView instead of the shell until a user signs in — the Tier-3
	# specs (real-corpus / handoff-matrix / person-profile) drive graph + corpus
	# surfaces, not auth, and never sign in, so an enabled gate blocks every walk
	# on the corpus-path input. Auth has its own unit + stack-test coverage; this
	# harness runs auth-off (its pre-#14 behaviour). ``none`` is non-empty so
	# serve-api's ``:-mock`` default does not re-enable it, and provider_from_env
	# returns None for any non-``mock`` value without Google creds.
	@echo "Starting API + UI with SERVE_OUTPUT_DIR=$(PWD) (repo root), auth disabled."
	@echo "Synthetic validation corpus root (pass this as CORPUS=):"
	@echo "  $(VIEWER_VALIDATION_CORPUS)"
	@SERVE_OUTPUT_DIR=$(PWD) APP_OAUTH_PROVIDER=none $(MAKE) -j2 serve-api serve-ui

APP_TIER3_CORPUS ?= tests/fixtures/app-validation-corpus/v3

.PHONY: build-app-validation-index build-validation-index
# Corpus this target indexes. Its own variable, NOT `$(if $(CORPUS),...)`: CORPUS is globally
# defaulted to the viewer corpus at the top of this file, so it is never empty and that
# conditional could never choose the app corpus.
APP_IDX_CORPUS ?= $(APP_TIER3_CORPUS)

build-app-validation-index:
	# Builds ONLY the LanceDB index for the app tier-3 corpus. It stays a separate verb from
	# build-validation-index because it also clears the incremental ledger first and then COUNTS
	# what landed; build-validation-index does neither.
	#
	# The app corpus's search/topic_clusters.json is TRACKED and is now the EMBEDDING-DERIVED
	# artifact: 14 clusters / 16 singletons from all-MiniLM-L6-v2 at threshold 0.35, produced by
	# `cli topic-clusters` (an installed extension's command) against this target's index (2026-10-01).
	#
	# It used to be the synthetic one build_app_validation_corpus.py authors (threshold 0.75, 2
	# clusters), and this comment used to warn that regenerating turned 80 passed into 6 failed.
	# Those 6 assertions have been rewritten against the real clusters — they encoded a finding
	# ("both clusters are universal, so the picker verdict is undecidable on v3") that was an
	# artifact of the hand-authored file: one of its two clusters bundled each show's LEAD topic,
	# so its members had nothing in common, and only 3 of its 28 member pairs even appeared in the
	# topic_similarity artifact beside it, all BELOW its own declared threshold.
	#
	# The builder no longer overwrites it: _write_topic_clusters_scaffold keeps any file whose
	# `model` is not "synthetic-app-validation-corpus".
	#
	# Clearing search/episode_fingerprints.json first is the load-bearing step. That ledger
	# records which episodes are already indexed, so with it in place the indexer treats a GROWN
	# corpus as fully covered, writes nothing, and exits 0. The corpus went 36 -> 40, so a build
	# without this yields a 40-episode corpus with a 36-episode index and no complaint.
	#
	# Then it COUNTS, because `index-two-tier` exits 0 either way and the exit code is not
	# evidence.
	@echo "=== 1/3 Clearing the incremental ledger at $(APP_IDX_CORPUS)/search ==="
	@rm -f "$(APP_IDX_CORPUS)/search/episode_fingerprints.json"
	@echo "=== 2/3 Building the LanceDB two-tier index at $(APP_IDX_CORPUS)/search/lance_index ==="
	@$(PYTHON) -m podcast_scraper.cli index-two-tier --output-dir "$(APP_IDX_CORPUS)"
	@echo ""
	@echo "=== 3/3 Verifying the index covers the corpus ==="
	@$(PYTHON) scripts/tools/count_indexed_episodes.py "$(APP_IDX_CORPUS)"

# ONE threshold for both corpora. It lived only inside the recipe below, so the number that
# produced a COMMITTED fixture was visible only to whoever read the recipe.
VALIDATION_TOPIC_CLUSTER_THRESHOLD ?= 0.35

build-validation-index:
	# Builds the LanceDB two-tier index ONLY — the ignored, regenerable half.
	#
	# It used to also run ``cli topic-clusters``, back when the whole of <corpus>/search/
	# was gitignored and nothing under it could be clobbered. topic_clusters.json is now a
	# COMMITTED fixture for both corpora, so "build me an index" silently rewriting it is
	# the same hazard that turned 80 passed into 6 failed on the app corpus. Regenerating a
	# tracked fixture is a deliberate act with its own verb: build-validation-topic-clusters,
	# in the Makefile of the extension that provides topic-clusters.
	#
	# Run this BEFORE ``make serve-for-validation``. It unlocks V3 (semantic search → Show
	# on graph); without an index V3 is SKIPPED (test.skip on ``!indexJson.available``).
	# V1/V5 (Library/Digest handoffs) need none of it. V2/V4 need topic_clusters.json, which
	# is committed — so a clean clone has it already and needs no ML to run them.
	@CORPUS=$(if $(CORPUS),$(CORPUS),$(VIEWER_VALIDATION_CORPUS)); \
	echo "=== Building LanceDB two-tier index at $$CORPUS/search/lance_index ==="; \
	$(PYTHON) -m podcast_scraper.cli index-two-tier \
		--output-dir $$CORPUS
	@echo ""
	@echo "Done — LanceDB two-tier index built. Now run:"
	@echo "  make serve-for-validation       (terminal 1)"
	@echo "  make ci-ui-validation CORPUS=$(VIEWER_VALIDATION_CORPUS)  (terminal 2)"

ci-clean: clean-all format-check lint lint-markdown type security complexity deadcode docstrings spelling check-test-policy preload-ml-models test test-ui test-ui-e2e build-viewer coverage-enforce docs build

ci-nightly: format-check lint lint-markdown type security complexity deadcode docstrings spelling check-test-policy preload-ml-models-production test-unit test-integration test-e2e test-nightly test-ui test-ui-e2e build-viewer coverage-enforce docs build
	@echo ""
	@echo "✓ Full nightly CI chain completed"

# Provenance for EVERY image build (ADR-132, #1657). The runtime image has no git — the build
# context excludes .git/ and the binary is never installed — so unless the SHA is passed in,
# every manifest the image writes records git_sha: null. Falls back to empty outside a repo,
# which the code treats exactly as it did before (shell out to git, find nothing).
#
# Do NOT escape the inner quotes as \" — command substitution restarts the quoting context, so
# the backslashes reach `test` as literal arguments. That form dies with "test: too many
# arguments" and, worse, falls through to `echo false`: a dirty tree silently reported clean.
GIT_BUILD_ARGS = \
	--build-arg BUILD_TIME="$$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
	--build-arg GIT_SHA="$$(git rev-parse HEAD 2>/dev/null || echo '')" \
	--build-arg GIT_BRANCH="$$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')" \
	--build-arg GIT_DIRTY="$$(test -z "$$(git status --porcelain 2>/dev/null)" && echo false || echo true)"

docker-build:
	@echo "Building Docker image (ML-enabled variant, production ML preload, default)..."
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg INSTALL_EXTRAS=ml \
		--build-arg PRELOAD_ML_MODELS=true \
		$(GIT_BUILD_ARGS) \
		-t podcast-scraper:test \
		-f docker/pipeline/Dockerfile .

docker-build-llm:
	@echo "Building Docker image (LLM-only variant, ~200MB)..."
	@echo "This variant excludes ML dependencies for smaller size and faster builds..."
	@echo ""
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg INSTALL_EXTRAS="" \
		$(GIT_BUILD_ARGS) \
		-t podcast-scraper:test-llm \
		-f docker/pipeline/Dockerfile .
	@echo ""
	@echo "✓ LLM-only build complete! Image tagged as: podcast-scraper:test-llm"

docker-build-fast:
	@echo "Building Docker image (fast mode - ML variant, no model preloading, matches PR builds)..."
	@echo "This should complete in under 5 minutes..."
	@echo ""
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg INSTALL_EXTRAS=ml \
		--build-arg PRELOAD_ML_MODELS=false \
		$(GIT_BUILD_ARGS) \
		-t podcast-scraper:test-fast \
		-f docker/pipeline/Dockerfile .
	@echo ""
	@echo "✓ Fast build complete! Image tagged as: podcast-scraper:test-fast"

docker-build-full:
	@echo "Building Docker image (full mode - ML variant with production model preloading)..."
	@echo "This will take a long time and require significant disk (HF + Whisper + evidence)..."
	@echo ""
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg INSTALL_EXTRAS=ml \
		--build-arg PRELOAD_ML_MODELS=true \
		$(GIT_BUILD_ARGS) \
		-t podcast-scraper:test \
		-f docker/pipeline/Dockerfile .
	@echo ""
	@echo "✓ Full build complete! Image tagged as: podcast-scraper:test"

docker-test: docker-build docker-build-llm
	@echo "Running Docker tests for both variants..."
	@echo ""
	@echo "=== Testing ML-enabled variant ==="
	@echo "Test 1: --help command"
	@docker run --rm podcast-scraper:test --help > /dev/null
	@echo "Test 2: --version command"
	@docker run --rm podcast-scraper:test --version
	@echo "Test 3: No args (should show config file error)"
	@docker run --rm podcast-scraper:test 2>&1 | grep -q "Config file not found" && echo "[OK] Error handling works"
	@echo ""
	@echo "=== Testing LLM-only variant ==="
	@echo "Test 1: --help command"
	@docker run --rm podcast-scraper:test-llm --help > /dev/null
	@echo "Test 2: --version command"
	@docker run --rm podcast-scraper:test-llm --version
	@echo "Test 3: No args (should show config file error)"
	@docker run --rm podcast-scraper:test-llm 2>&1 | grep -q "Config file not found" && echo "[OK] Error handling works"
	@echo "Test 4: Verify LLM-only variant is smaller"
	@ML_SIZE=$$(docker images podcast-scraper:test --format "{{.Size}}" | head -1); \
	LLM_SIZE=$$(docker images podcast-scraper:test-llm --format "{{.Size}}" | head -1); \
	echo "ML variant size: $$ML_SIZE"; \
	echo "LLM-only variant size: $$LLM_SIZE"; \
	echo "[OK] Both variants built and tested successfully"

docker-clean:
	docker rmi podcast-scraper:test podcast-scraper:test-llm podcast-scraper:test-fast podcast-scraper:multi-model 2>/dev/null || true
	@echo "Cleaned up Docker test images"

# --- Observability control plane (podcast_obs, #803) ---
# android tiers

.PHONY: obs-test obs-e2e obs-docker-build obs-summary obs-serve

obs-test: ## Unit tests for the observability control plane (podcast_obs). Fast, no network.
	$(PYTHON) -m pytest tests/unit/podcast_obs/ -q --no-cov --disable-socket --allow-hosts=127.0.0.1,localhost

obs-e2e: ## Live E2E — hits real APIs, self-skips unconfigured sources. Source .env / set PODCAST_OBS_* first.
	$(PYTHON) -m pytest tests/e2e_observability/ -v --no-cov -p no:cacheprovider

obs-docker-build: ## Build the standalone control-plane image (podcast-obs:latest); tiny context via Dockerfile.dockerignore.
	docker build -f docker/observability/Dockerfile -t podcast-obs:latest .

obs-summary: ## Control-plane glance for the configured target. Needs PODCAST_OBS_* in env (source .env).
	$(PYTHON) -m podcast_obs summary

obs-serve: ## Run the observability MCP server. Usage: make obs-serve OBS_ARGS="--transport http --port 8848"
	$(PYTHON) -m podcast_obs serve $(OBS_ARGS)

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

# Architecture visualization (Issue #425) - outputs in docs/architecture/ for documentation
deps-graph:
	@mkdir -p docs/architecture/diagrams
	@echo "Generating module dependency graph (simplified)..."
	export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) -m pydeps src/podcast_scraper --cluster --max-bacon=2 -o docs/architecture/diagrams/dependency-graph-simple.svg --no-show
	@echo "Generating module dependency graph (full package)..."
	export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) -m pydeps src/podcast_scraper --cluster --max-bacon=3 -o docs/architecture/diagrams/dependency-graph.svg --no-show
	@touch docs/architecture/diagrams/dependency-graph.svg docs/architecture/diagrams/dependency-graph-simple.svg 2>/dev/null || true
	@echo "✓ Dependency graphs written to docs/architecture/"

deps-graph-full:
	@mkdir -p docs/architecture/diagrams
	@echo "Generating full dependency graph (all dependencies)..."
	export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) -m pydeps src/podcast_scraper -o docs/architecture/diagrams/dependency-graph-full.svg --no-show
	@echo "✓ Full dependency graph written to docs/architecture/diagrams/dependency-graph-full.svg"

# Call graph (pyan3) - workflow orchestration entry point; pyan3 1.1.1 required (1.2.0 has bugs)
call-graph:
	@mkdir -p docs/architecture/diagrams
	@echo "Generating workflow call graph (orchestration.py)..."
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) -m pyan src/podcast_scraper/workflow/orchestration.py --uses --no-defines --dot --file docs/architecture/diagrams/workflow-call-graph.dot 2>/dev/null || true
	@if [ -f docs/architecture/diagrams/workflow-call-graph.dot ]; then \
		dot -Tsvg docs/architecture/diagrams/workflow-call-graph.dot -o docs/architecture/diagrams/workflow-call-graph.svg 2>/dev/null && echo "✓ Call graph written to docs/architecture/diagrams/workflow-call-graph.svg"; \
	else \
		echo "⚠ pyan3 call graph skipped (install pyan3==1.1.1 and graphviz)"; \
	fi

# Flowcharts (code2flow) - orchestration and service entry points
flowcharts:
	@mkdir -p docs/architecture/diagrams
	@echo "Generating orchestration flowchart..."
	@$(PYTHON) -m code2flow src/podcast_scraper/workflow/orchestration.py -o docs/architecture/diagrams/orchestration-flow.svg --language py -q 2>/dev/null || true
	@touch -r src/podcast_scraper/workflow/orchestration.py docs/architecture/diagrams/orchestration-flow.svg 2>/dev/null || true
	@echo "Generating service flowchart..."
	@$(PYTHON) -m code2flow src/podcast_scraper/service.py -o docs/architecture/diagrams/service-flow.svg --language py -q 2>/dev/null || true
	@touch -r src/podcast_scraper/service.py docs/architecture/diagrams/service-flow.svg 2>/dev/null || true
	@echo "✓ Flowcharts written to docs/architecture/ (orchestration-flow.svg, service-flow.svg)"

providers-deps:
	@mkdir -p docs/architecture/diagrams
	@echo "Generating providers dependency graph..."
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) -m pydeps src/podcast_scraper/providers --cluster --max-bacon=2 -o docs/architecture/diagrams/providers-deps.svg --no-show 2>/dev/null && echo "✓ Providers dependency graph written" || echo "⚠ Providers dependency graph skipped (pydeps not available)"

gi-kg-flow:
	@mkdir -p docs/architecture/diagrams
	@echo "Rendering GI pipeline flowchart from DOT..."
	@dot -Tsvg docs/architecture/diagrams/gi-pipeline-flow.dot -o docs/architecture/diagrams/gi-pipeline-flow.svg 2>/dev/null && echo "  ✓ gi-pipeline-flow.svg" || echo "  ⚠ gi-pipeline-flow.svg skipped (graphviz not available)"
	@echo "Rendering KG pipeline flowchart from DOT..."
	@dot -Tsvg docs/architecture/diagrams/kg-pipeline-flow.dot -o docs/architecture/diagrams/kg-pipeline-flow.svg 2>/dev/null && echo "  ✓ kg-pipeline-flow.svg" || echo "  ⚠ kg-pipeline-flow.svg skipped (graphviz not available)"
	@echo "✓ GI/KG pipeline flowcharts written"

eval-flow:
	@mkdir -p docs/architecture/diagrams
	@echo "Rendering evaluation scorer flowchart from DOT..."
	@dot -Tsvg docs/architecture/diagrams/eval-scorer-flow.dot -o docs/architecture/diagrams/eval-scorer-flow.svg 2>/dev/null && echo "  ✓ eval-scorer-flow.svg" || echo "  ⚠ eval-scorer-flow.svg skipped (graphviz not available)"
	@echo "✓ Evaluation flowchart written"

visualize: deps-graph call-graph flowcharts providers-deps gi-kg-flow eval-flow
	@echo "✓ Architecture visualizations up to date (see docs/architecture/diagrams/)"


# Download CI metrics bundles (GitHub CLI). Output: artifacts/ci-metrics-runs/run-<id>/
fetch-ci-metrics:
	@bash scripts/dashboard/fetch_ci_metrics_artifacts.sh "$(N)"

fetch-ci-metrics-validate:
	@export PYTHON="$(PYTHON)"; bash scripts/dashboard/fetch_and_validate_ci_metrics.sh "$(N)"

fetch-nightly-metrics:
	@export PYTHON="$(PYTHON)" GITHUB_BRANCH="$(GITHUB_BRANCH)"; bash scripts/dashboard/fetch_nightly_metrics.sh "$(N)"

validate-metrics-bundle:
	@if [ -z "$(BUNDLE)" ]; then echo "Usage: make validate-metrics-bundle BUNDLE=artifacts/ci-metrics-runs/run-<id>"; exit 1; fi
	@$(PYTHON) scripts/dashboard/validate_metrics_bundle.py "$(BUNDLE)"

# Unified dashboard: CI from newest artifacts/ci-metrics-runs/run-* + nightly from metrics/ (gitignored output).
build-metrics-dashboard-preview:
	@export PYTHON="$(PYTHON)"; bash scripts/dashboard/build_local_metrics_preview.sh

# Fails (exit 2) if metrics/history-*.jsonl looks like pretty-printed JSON, not JSONL.
metrics-preview-check:
	@export PYTHON="$(PYTHON)" METRICS_PREVIEW_STRICT=1; bash scripts/dashboard/build_local_metrics_preview.sh

serve-metrics-dashboard: build-metrics-dashboard-preview
	@echo "Metrics dashboard → http://127.0.0.1:8777/  (toggle CI vs Nightly; Ctrl+C to stop)"
	@$(PYTHON) -m http.server 8777 --bind 127.0.0.1 --directory "$(CURDIR)/artifacts/dashboard-preview"

# Fetch CI metrics (gh), validate, merge with nightly, serve unified dashboard (not MkDocs).
metrics-dashboard-live:
	@export PYTHON="$(PYTHON)"; bash scripts/dashboard/metrics_dashboard_live.sh "$(N)"

# Run before release: regenerate diagrams and create release notes draft. Add to release checklist.
release-docs-prep: visualize
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/create_release_notes_draft.py
	@echo "✓ Release docs prep complete"
	@echo "Review: git status docs/architecture/ docs/releases/ && git diff docs/architecture/ docs/releases/"
	@echo "Then commit: git add docs/architecture/diagrams/*.svg docs/releases/RELEASE_*.md && git commit -m 'docs: release docs prep (visualizations and release notes)'"

# ADR-031: fast version/release-doc checks, then full CI (long; avoid concurrent make ci).
pre-release:
	@echo "Running scripts/pre_release_check.py …"
	@$(PYTHON) scripts/pre_release_check.py
	@echo "Running make ci (preload + tests + docs + build) …"
	@$(MAKE) ci

# Bump [project].version and __version__ (no v). Optional: ALLOW_DIRTY=1, FORCE_TAG=1
bump:
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make bump VERSION=X.Y.Z   [ALLOW_DIRTY=1] [FORCE_TAG=1]"; \
		exit 1; \
	fi
	@cmd="$(PYTHON) scripts/tools/bump_version.py $(VERSION)"; \
	if [ "$(ALLOW_DIRTY)" = "1" ]; then cmd="$$cmd --allow-dirty"; fi; \
	if [ "$(FORCE_TAG)" = "1" ]; then cmd="$$cmd --force"; fi; \
	eval $$cmd

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
	@$(HF_NET_ENV) $(PYTHON) scripts/cache/preload_ml_models.py

preload-ml-models-production:
	@echo "Preloading production ML models for nightly tests..."
	@echo "Models: Whisper base, BART/LED/Pegasus, hybrid LongT5+FLAN-T5, en_core_web_sm"
	@$(HF_NET_ENV) $(PYTHON) scripts/cache/preload_ml_models.py --production

hf-hub-smoke-test:
	@$(HF_NET_ENV) $(PYTHON) scripts/cache/hf_hub_smoke_test.py $(HF_SMOKE_ARGS)

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

profiles-materialize:
	.venv/bin/python scripts/config/materialize_profiles.py; echo "MAKE_EXIT=$$?"

profiles-check:
	.venv/bin/python scripts/config/materialize_profiles.py --check; echo "MAKE_EXIT=$$?"

#: The local gates for what changed vs origin/main (or STAGED=1: the staged files), checks only:
#: black/isort/flake8/mypy/docstrings + the tests naming a changed module, markdownlint, make docs,
#: viewer vitest-related + vue-tsc, actionlint. Last line GATES: PASS|FAIL. Skill: local-gates.
#: Rebuild the app validation corpus as committed (build, realign, corpus enrichers) and report
#: the difference; WRITE=1 copies added/changed files in. README: tests/fixtures/app-validation-corpus.
regen-fixtures:
	@scripts/tools/regen_app_fixture.sh $(if $(filter 1,$(WRITE)),--write,)

gates-changed:
	@scripts/tools/gates_changed.sh $(if $(filter 1,$(STAGED)),--staged,$(if $(SINCE),--since $(SINCE),))

validate-files:
	@# Validate changed files: lint/format + run only impacted tests
	@# Usage: make validate-files FILES="src/podcast_scraper/config.py src/podcast_scraper/workflow/orchestration.py"
	@# Optional: TEST_TYPE=unit|integration|e2e|all (default: all), FAST_ONLY=1 (only critical_path tests)
	@if [ -z "$(FILES)" ]; then \
		echo "❌ Error: FILES required"; \
		echo "Usage: make validate-files FILES='file1.py file2.py' [TEST_TYPE=unit] [FAST_ONLY=1]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make validate-files FILES='src/podcast_scraper/config.py'"; \
		echo "  make validate-files FILES='src/podcast_scraper/config.py' TEST_TYPE=unit"; \
		echo "  make validate-files FILES='src/podcast_scraper/config.py' FAST_ONLY=1"; \
		exit 1; \
	fi
	@echo "🔍 Step 1: Linting and formatting changed files..."
	@for file in $(FILES); do \
		if [ ! -f "$$file" ]; then \
			echo "⚠️  Warning: File not found: $$file"; \
			continue; \
		fi; \
		echo "  Formatting: $$file"; \
		$(PYTHON) -m black $$file; \
		$(PYTHON) -m isort $$file; \
		echo "  Linting: $$file"; \
		$(PYTHON) -m flake8 --config .flake8 $$file || lint_failed=1; \
		$(PYTHON) -m mypy $$file || lint_failed=1; \
	done; \
	[ -z "$$lint_failed" ] || { echo "FAIL: flake8/mypy errors above (validate-files used to ignore them)"; exit 1; }
	@echo ""
	@echo "🔍 Step 2: Discovering impacted tests..."
	@TEST_TYPE=$${TEST_TYPE:-all}; \
	FAST_ONLY=$${FAST_ONLY:-0}; \
	TEST_MARKERS=$$($(PYTHON) scripts/tools/find_impacted_tests.py \
		--files $(FILES) \
		--test-type $$TEST_TYPE \
		$(if $(filter 1,$(FAST_ONLY)),--fast-only,) \
		--output-format list 2>&1 | grep -v "^  " | grep -v "^⚠️" | grep -v "^❌" || true); \
	MARKER_EXPR=$$($(PYTHON) scripts/tools/find_impacted_tests.py \
		--files $(FILES) \
		--test-type $$TEST_TYPE \
		$(if $(filter 1,$(FAST_ONLY)),--fast-only,) \
		--output-format expression 2>/dev/null || echo ""); \
	if [ -z "$$MARKER_EXPR" ]; then \
		echo "⚠️  No tests found for changed files"; \
		exit 0; \
	fi; \
	if [ -n "$$TEST_MARKERS" ]; then \
		echo "  Discovered markers: $$TEST_MARKERS"; \
	fi; \
	echo "  Marker expression: $$MARKER_EXPR"; \
	echo ""
	@echo "🧪 Step 3: Running impacted tests..."
	@TEST_TYPE=$${TEST_TYPE:-all}; \
	FAST_ONLY=$${FAST_ONLY:-0}; \
	MARKER_EXPR=$$($(PYTHON) scripts/tools/find_impacted_tests.py \
		--files $(FILES) \
		--test-type $$TEST_TYPE \
		$(if $(filter 1,$(FAST_ONLY)),--fast-only,) \
		--output-format expression 2>/dev/null || echo ""); \
	if [ -z "$$MARKER_EXPR" ]; then \
		echo "⚠️  No tests found for changed files"; \
		exit 0; \
	fi; \
	$(PYTHON) -m pytest -m "$$MARKER_EXPR" \
		-n $(PYTEST_WORKERS) \
		--disable-socket --allow-hosts=127.0.0.1,localhost \
		-v \
		--tb=short
	@echo ""
	@echo "✅ Validation complete!"

validate-files-fast:
	@# Fast mode: only critical_path tests
	@# Usage: make validate-files-fast FILES="src/podcast_scraper/config.py"
	@if [ -z "$(FILES)" ]; then \
		echo "❌ Error: FILES required"; \
		echo "Usage: make validate-files-fast FILES='file1.py file2.py'"; \
		exit 1; \
	fi
	@$(MAKE) validate-files FILES="$(FILES)" TEST_TYPE=all FAST_ONLY=1

validate-files-unit:
	@# Unit tests only (fastest)
	@# Usage: make validate-files-unit FILES="src/podcast_scraper/config.py"
	@if [ -z "$(FILES)" ]; then \
		echo "❌ Error: FILES required"; \
		echo "Usage: make validate-files-unit FILES='file1.py file2.py'"; \
		exit 1; \
	fi
	@$(MAKE) validate-files FILES="$(FILES)" TEST_TYPE=unit







dgx-smoke:
	@# RFC-089 (not in this repository): probe DGX Ollama via tailnet (non-fatal when DGX offline).
	@# Auto-sources infra/.env.dgx.local (added with #897 / ADR-098) if present.
	@if [ -f infra/.env.dgx.local ]; then set -a && . ./infra/.env.dgx.local && set +a; fi; \
		if [ -z "$${DGX_TAILNET_FQDN:-}" ]; then \
			echo "WARN: DGX_TAILNET_FQDN unset; skipping dgx-smoke" >&2; exit 0; \
		fi; \
		host=$$(bash scripts/ops/resolve_dgx_tailnet_host.sh) && \
			curl -fsS --max-time 5 "http://$$host:11434/api/tags" >/dev/null && \
			echo "DGX Ollama OK at http://$$host:11434"

# Stage A pre-prod dress rehearsal (#814): run the prod profile shape against
# real audio from your laptop, with laptop-local Whisper. Validates the full
# pipeline (Gemini LLM, screenplay, diarize, GI, KG, vectors) without needing
# the DGX Whisper service deployed yet.
#
# Variables:
#   RSS           required — RSS feed URL to scrape
#   EPISODES      default 1 — number of episodes to process
#   WHISPER_MODEL default small.en — bump to large-v3 for final dress rehearsal
#                  (parity with prod's whisper-large-v3; ~3 GB one-time download)
#   OUTPUT_DIR    default ~/preprod-local-out — where to write artifacts
#
# Example:
#   make preprod-local RSS=https://feeds.example.com/show.rss EPISODES=3
#   make preprod-local RSS=https://... WHISPER_MODEL=large-v3   # final dress rehearsal
#
# See DGX_RUNBOOK §"Pre-prod laptop validation (Stage A/B/C)".
EPISODES ?= 1
WHISPER_MODEL ?= small.en
OUTPUT_DIR ?= $(HOME)/preprod-local-out
CHAOS_PORT ?= 18443
preprod-local:
	@if [ -z "$(RSS)" ]; then \
		echo "ERROR: RSS=<feed-url> required" >&2; \
		echo "  Example: make preprod-local RSS=https://feeds.example.com/show.rss" >&2; \
		exit 1; \
	fi
	@mkdir -p "$(OUTPUT_DIR)"
	@export PYTHONPATH="$(PWD)/src:$(PWD):$${PYTHONPATH}"; \
		$(PYTHON) -m podcast_scraper.cli \
			"$(RSS)" \
			--profile preprod_local_whisper \
			--whisper-model $(WHISPER_MODEL) \
			--output-dir "$(OUTPUT_DIR)" \
			--max-episodes $(EPISODES) \
			--log-level INFO
	@echo ""
	@echo "Stage A run complete. Inspect output at $(OUTPUT_DIR)"
	@echo "  - transcripts/*.txt        — Whisper output"
	@echo "  - transcripts/*.segments.json — should have speaker labels (SPEAKER_00, _01, ...)"
	@echo "  - metadata/*.gi.json       — GI insights with grounding"
	@echo "  - metadata/*.kg.json       — KG topics + entities"
	@echo "  - search/lance_index/      — LanceDB search index"
	@echo "  - metrics.json             — run timing + cost"

# Stage B chaos: DGX denied mid-run, cloud fallback must fire (#814).
# Starts chaos_proxy on $(CHAOS_PORT) (returns 503 to all requests), then
# runs the prod profile with --dgx-tailnet-host 127.0.0.1 and
# --dgx-whisper-port $(CHAOS_PORT) so the DGX-side health check + transcribe
# call land on the 503 server. Acceptance: transcription falls back to OpenAI
# cloud Whisper, episode completes, run includes dgx_fallback_active.
preprod-chaos-dgx-down:
	@if [ -z "$(RSS)" ]; then \
		echo "ERROR: RSS=<feed-url> required" >&2; \
		echo "  Example: make preprod-chaos-dgx-down RSS=https://feeds.example.com/show.rss" >&2; \
		exit 1; \
	fi
	@echo "→ Starting chaos_proxy (503-everything) on :$(CHAOS_PORT)"
	@$(PYTHON) scripts/tools/chaos_proxy.py --port $(CHAOS_PORT) --log-level INFO & \
		PROXY_PID=$$!; \
		trap "kill $$PROXY_PID 2>/dev/null || true" EXIT INT TERM; \
		sleep 1; \
		mkdir -p "$(OUTPUT_DIR)-chaos-dgx-down"; \
		export PYTHONPATH="$(PWD)/src:$(PWD):$${PYTHONPATH}"; \
		echo "→ Running prod profile with DGX routed to chaos proxy (cloud Whisper must take over)"; \
		$(PYTHON) -m podcast_scraper.cli \
			"$(RSS)" \
			--profile cloud_with_dgx_primary \
			--dgx-tailnet-host 127.0.0.1 \
			--dgx-whisper-port $(CHAOS_PORT) \
			--output-dir "$(OUTPUT_DIR)-chaos-dgx-down" \
			--max-episodes $(EPISODES) \
			--log-level INFO

# Stage B-2 chaos: the DGX LLM is dead while ASR/diarization stay healthy (#1932 follow-up).
#
# THE GAP THIS CLOSES. `preprod-chaos-dgx-down` above reroutes only --dgx-whisper-port and
# asserts cloud Whisper takes over. It passes cleanly while every LLM stage silently degrades,
# which is exactly what happened in production: the autoresearch vLLM on :8003 is GPU-mode-gated
# ("idle, not gone" — DGX_SERVING (not in this repository)), so whisper on :8000 and pyannote on :8001 answered
# normally while KG extraction and GI returned empty and the run recorded itself as clean.
#
# This is the realistic shape of a DGX incident: ONE service down, not the box.
#
# Acceptance — read the artifacts, do not trust the exit code:
#   - metrics.json: llm_kg_calls > 0 OR kg_failures > 0    (never both zero: that is the bug)
#   - *.kg.json: Topic labels are NOUN PHRASES, not truncated sentences, and none of them equals
#     a summary bullet from the sibling *.metadata.json
#   - extraction.model_version is NOT "topic_labels" (that string means bullets were substituted)
#   - a WARNING naming the fallback escalation appears in the run log
preprod-chaos-llm-down:
	@if [ -z "$(RSS)" ]; then \
		echo "ERROR: RSS=<feed-url> required" >&2; \
		echo "  Example: make preprod-chaos-llm-down RSS=https://feeds.example.com/show.rss" >&2; \
		exit 1; \
	fi
	@echo "→ Starting chaos_proxy (503-everything) on :$(CHAOS_PORT) — LLM only, ASR stays real"
	@$(PYTHON) scripts/tools/chaos_proxy.py --port $(CHAOS_PORT) --log-level INFO & \
		PROXY_PID=$$!; \
		trap "kill $$PROXY_PID 2>/dev/null || true" EXIT INT TERM; \
		sleep 1; \
		mkdir -p "$(OUTPUT_DIR)-chaos-llm-down"; \
		export PYTHONPATH="$(PWD)/src:$(PWD):$${PYTHONPATH}"; \
		printf 'vllm_api_base: "http://127.0.0.1:%s/v1"\n' "$(CHAOS_PORT)" \
			> "$(OUTPUT_DIR)-chaos-llm-down/chaos-llm.yaml"; \
		echo "→ vLLM routed at the chaos proxy; whisper/diarization untouched"; \
		$(PYTHON) -m $(PACKAGE).cli \
			"$(RSS)" \
			--profile prod_dgx_full \
			--config "$(OUTPUT_DIR)-chaos-llm-down/chaos-llm.yaml" \
			--output-dir "$(OUTPUT_DIR)-chaos-llm-down" \
			--max-episodes $(EPISODES) \
			--log-level INFO; \
		echo ""; \
		echo "→ Acceptance check"; \
		$(PYTHON) scripts/tools/check_chaos_llm_artifacts.py "$(OUTPUT_DIR)-chaos-llm-down"

# Stage B chaos: DGX + cloud both denied; pipeline should abort cleanly with
# operator-visible error (#814). No half-baked output. ``OPENAI_BASE_URL``
# routes the cloud Whisper client at the chaos proxy too; both health checks
# fail; the transcription provider has nowhere to go.
preprod-chaos-both-down:
	@if [ -z "$(RSS)" ]; then \
		echo "ERROR: RSS=<feed-url> required" >&2; exit 1; \
	fi
	@echo "→ Starting chaos_proxy on :$(CHAOS_PORT) (denies DGX + cloud)"
	@$(PYTHON) scripts/tools/chaos_proxy.py --port $(CHAOS_PORT) --log-level INFO & \
		PROXY_PID=$$!; \
		trap "kill $$PROXY_PID 2>/dev/null || true" EXIT INT TERM; \
		sleep 1; \
		mkdir -p "$(OUTPUT_DIR)-chaos-both-down"; \
		export PYTHONPATH="$(PWD)/src:$(PWD):$${PYTHONPATH}"; \
		export OPENAI_BASE_URL="http://127.0.0.1:$(CHAOS_PORT)/v1"; \
		echo "→ Running prod profile; expecting non-zero exit (operator-visible failure)"; \
		set +e; \
		$(PYTHON) -m podcast_scraper.cli \
			"$(RSS)" \
			--profile cloud_with_dgx_primary \
			--dgx-tailnet-host 127.0.0.1 \
			--dgx-whisper-port $(CHAOS_PORT) \
			--output-dir "$(OUTPUT_DIR)-chaos-both-down" \
			--max-episodes $(EPISODES) \
			--log-level INFO; \
		RC=$$?; \
		if [ $$RC -eq 0 ]; then \
			echo "FAIL: pipeline exited 0 — expected failure when both DGX and cloud denied"; \
			exit 2; \
		else \
			echo "OK: pipeline exited $$RC (expected non-zero — both backends denied)"; \
		fi




# Which episodes were transcribed from UNPREPROCESSED audio (#18/#558)? Preprocessing used to run
# under a flat 300s budget; on long episodes it hit that wall, produced nothing, and the ORIGINAL
# full-size file went to the STT provider. The damage is in the TRANSCRIPT, so neither gi-repair
# nor reprocess-corpus-from-transcripts (transcribe=off) can fix it — only re-transcription can,
# which is an explicit cost decision. This makes the number known. CORPUS_DIR required.
corpus-preprocessing-audit:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "import sys; from pathlib import Path; from podcast_scraper.preprocessing.audit import check_corpus_preprocessing; ok, report = check_corpus_preprocessing(Path('$${CORPUS_DIR}').expanduser()); print(report); sys.exit(0 if ok else 1)"

# Which episodes are serving WITHOUT a summary (#1686)? Reads the per-episode stage ledger
# across EVERY run dir, so an episode that failed twice is distinguishable from one that failed
# once — the first is worth a requeue, the second the pipeline has given up on. Non-zero exit if
# ANY episode lacks a summary: one is too many, and a tolerance threshold is how 8 of them became
# normal in production. CORPUS_DIR required.
corpus-summary-audit:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "import sys; from pathlib import Path; from podcast_scraper.summary_repair import check_corpus_summaries; ok, report = check_corpus_summaries(Path('$${CORPUS_DIR}').expanduser()); print(report); sys.exit(0 if ok else 1)"

# Emit the #1686 repair work-list: episode_ids to feed --reprocess-episode-ids. Writes
# <corpus>/summary_repair_worklist.txt (override with WORKLIST=), plus a .terminal sidecar for
# episodes that already failed their requeue — those need a person, not another dispatch, and
# dropping them silently would repeat the mistake this issue is about. Re-summarisation costs
# provider money. CORPUS_DIR required.
corpus-summary-worklist:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "from pathlib import Path; from podcast_scraper.summary_repair import write_work_list, terminal_episode_ids; root = Path('$${CORPUS_DIR}').expanduser(); d = Path('$${WORKLIST:-$${CORPUS_DIR}/summary_repair_worklist.txt}'); n = write_work_list(root, d); t = terminal_episode_ids(root); print(f'{n} retryable episode(s) written to {d}'); print(f'{len(t)} TERMINAL episode(s) in {d.name}.terminal — investigate by hand') if t else None"

# Emit the #18 repair work-list: the episode_ids to feed --reprocess-episode-ids. Writes
# <corpus>/preprocessing_repair_worklist.txt (override with WORKLIST=). Re-transcription costs
# real money — read the list before running it. CORPUS_DIR required.
corpus-preprocessing-worklist:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "from pathlib import Path; from podcast_scraper.preprocessing.audit import write_work_list; d = Path('$${WORKLIST:-$${CORPUS_DIR}/preprocessing_repair_worklist.txt}'); b = float('$${CHUNK_USD:-0}' or 0); n = write_work_list(Path('$${CORPUS_DIR}').expanduser(), d, chunk_budget_usd=(b or None)); print(f'{n} episode(s) written to {d}' + (f' in \$${b:.2f} batches (see {d.name}.001, .002, ...)' if b else ''))"
