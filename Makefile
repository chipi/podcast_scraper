# Auto-detect venv Python if .venv exists, otherwise use python3
ifeq ($(wildcard .venv/bin/python),)
PYTHON ?= python3
CODESPELL ?= codespell
else
PYTHON ?= .venv/bin/python
CODESPELL ?= .venv/bin/codespell
endif
PACKAGE = podcast_scraper

# Fixture version: resolved from tests/fixtures/FIXTURES_VERSION (single source of truth).
# Versioned fixture subdirs: tests/fixtures/{transcripts,audio}/$(FIXTURES_VERSION)/...
FIXTURES_VERSION ?= $(shell cat tests/fixtures/FIXTURES_VERSION 2>/dev/null | tr -d '[:space:]')

# Classic dot progress for fast layers (pyproject addopts no longer includes -q).
# PYTHONUNBUFFERED on fast pytest lines reduces line buffering in IDE/agent terminals.
PYTEST_PROGRESS_OPTS := -o console_output_style=classic

# GI/KG viewer (Vue + Vite + Playwright). Override if the app moves, e.g. ``make serve-ui WEB_VIEWER_DIR=apps/viewer``.
WEB_VIEWER_DIR ?= web/gi-kg-viewer

# Consumer Learning Player (Vue + Vite + Playwright; Epic 2 / RFC-099). Separate top-level app.
APP_DIR ?= app

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

.PHONY: help init init-no-ml venv-dev-init test-unit-dev-venv download-spacy-wheels format format-check lint lint-markdown lint-markdown-docs fix-md strip-doc-checkmarks strip-doc-emoji strip-docs type security security-bandit security-audit complexity complexity-track deadcode docstrings spelling spelling-docs quality check-unit-imports check-test-policy check-pricing-assumptions validate-gi-schema validate-kg-schema gil-quality-metrics diarization-quality diarization-quality compare-gil-runs kg-quality-metrics quality-metrics-ci fetch-ci-metrics fetch-ci-metrics-validate fetch-nightly-metrics validate-metrics-bundle build-metrics-dashboard-preview metrics-preview-check serve-metrics-dashboard metrics-dashboard-live deps-analyze deps-check deps-graph deps-graph-full call-graph flowcharts visualize release-docs-prep pre-release bump analyze-test-memory cleanup-processes check-zombie check-spotlight test-unit test-unit-sequential test-unit-no-ml test-integration test-integration-sequential test-integration-fast test-ci test-ci-fast test-e2e test-e2e-sequential test-e2e-fast verify-gil-offsets-after-acceptance preload-transformers-integration-summariesuality test-diarization test-nightly test test-sequential test-fast test-fast-no-py-e2e test-reruns test-track test-track-view test-openai test-openai-multi test-openai-all-feeds test-openai-real test-openai-real-multi test-openai-real-all-feeds test-openai-real-feed coverage coverage-check coverage-check-unit coverage-check-integration coverage-check-e2e coverage-check-combined merge-cov-fragments coverage-report coverage-enforce docs docs-check build _ci_body ci ci-fast ci-ui-fast ci-ui-full ci-ui-validation serve-for-validation ci-sequential ci-clean ci-nightly clean clean-cache clean-model-cache clean-all docker-build docker-build-fast docker-build-full docker-test docker-clean install-hooks preload-ml-models preload-ml-models-production hf-hub-smoke-test backup-cache backup-cache-dry-run backup-cache-list backup-cache-cleanup restore-cache restore-cache-dry-run metadata-generate source-index dataset-create dataset-smoke dataset-benchmark dataset-raw dataset-materialize run-promote baseline-create experiment-run ml-param-sweep autoresearch-score autoresearch-score-bundled silver-pairwise runs-list baselines-list run-compare runs-compare benchmark profile-freeze profile-diff profile-promote serve-gi-kg-viz test-ui test-ui-e2e build-viewer serve-app serve-app-dev test-app test-app-e2e build-app app-docker-build app-stack-config app-stack-up app-stack-down verify-gil-offsets-strict pipeline-validate transcription-sweep infra-plan infra-apply infra-recover drill-env delete-drill-hetzner-orphans drill-tofu-plan drill-tofu-apply drill-tofu-destroy

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
	@echo "  make validate-gi-schema [ARTIFACTS_DIR=path]  Validate gi.json files against GIL schema (strict)"
	@echo "  make validate-kg-schema [ARTIFACTS_DIR=path]  Validate kg.json files against KG schema (strict)"
	@echo "  make gil-quality-metrics [DIR=path]  PRD-017 metrics over .gi.json (optional --enforce via ARGS)"
	@echo "  make compare-gil-runs REF=run1 CAND=run2  Compare GIL gi.json stats between two run roots"
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
	@echo "  make serve-app           Consumer Learning Player dev server ($(APP_DIR); proxies /api → :8000)"
	@echo "  make serve-app-dev       API (mock OAuth) + Learning Player app in parallel — one-command local app env"
	@echo "  make test-app            Vitest unit tests + coverage gate for $(APP_DIR)"
	@echo "  make test-app-e2e        Playwright E2E for $(APP_DIR) (needs npm install + chromium in that dir)"
	@echo "  make build-app           Production Learning Player bundle (vue-tsc -b && vite build)"
	@echo "  make app-docker-build    Build the Learning Player Docker image"
	@echo "  make app-stack-up        Stack + Learning Player container (api proxied; APP_PORT, default 8081)"
	@echo "  make app-stack-down      Tear down the stack + Learning Player overlay"
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
	@echo "Infrastructure (RFC-082 prod VPS):"
	@echo "  make infra-plan      tofu init + plan against the prod Hetzner project (sources infra/.env.local; no state changes)"
	@echo "  make infra-apply     tofu init + plan + apply (CREATES REAL HETZNER RESOURCES; tofu prompts y/n before any change)"
	@echo "  make infra-recover   Recover from a failed apply: deletes orphan Hetzner resources by name, wipes stale state, re-runs apply with auto-ACL-import"
	@echo "  make drill-env       How to set HCLOUD_TOKEN_DRILL for the Hetzner drill project (local shell + make delete-drill-hetzner-orphans)"
	@echo "  make delete-drill-hetzner-orphans  Sources infra/.env.drill.local; deletes orphan drill Hetzner resources (partial apply cleanup)"
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
	@echo "  make stack-test-export      Stack-test: copy corpus_data volume → .stack-test-corpus/ for debug inspection"
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
	@echo "  make profile-freeze      Profiles: capture data/profiles/<VERSION>.yaml (PIPELINE_CONFIG=...; optional MONITOR=1)"
	@echo "  make profile-diff        Profiles: terminal diff of two profiles (FROM=v1 TO=v2)"
	@echo "  make profile-promote     Promote a working profile to data/profiles/references/"
	@echo "                            Usage: make profile-promote SOURCE=... PROMOTED_ID=... REASON=\"...\""
	@echo "  make run-compare         Streamlit UI: compare eval runs (pip install -e '.[compare]')"
	@echo "                            Usage: make run-compare [BASELINE=id]  (optional: default baseline in sidebar)"
	@echo "  make ml-param-sweep      Autoresearch Track B: ML hyperparameter ratchet (no API keys needed)"
	@echo "                            Usage: make ml-param-sweep MODEL=bart_led [MAX_FAILS=3] [MIN_GAIN=0.01] [DRY_RUN=1]"
	@echo "                            Models: bart_led, pegasus_led (see autoresearch/ml_param_tuning/param_space.yaml)"
	@echo "  make autoresearch-score  Autoresearch Track A: eval run + ROUGE + dual judges (scalar on stdout)"
	@echo "                            Usage: make autoresearch-score [CONFIG=...] [REFERENCE=...] [DRY_RUN=1]"
	@echo "  make silver-pairwise    Pairwise LLM judge between two silver candidate runs (winner on stdout)"
	@echo "                            Usage: make silver-pairwise CANDIDATE_A=<run_id> CANDIDATE_B=<run_id> [OUTPUT=path.json]"
	@echo "  make report-multi-run   Generate multi-run comparison report (baseline + N runs)"
	@echo "                            Usage: make report-multi-run [BASELINE_ID=...] RUN_IDS=id1,id2 REFERENCE_ID=ref [OUTPUT=...]"
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

# Shared markdownlint CLI args — keep lint-markdown and fix-md identical (and aligned with CI).
MARKDOWNLINT_CLI_ARGS = "**/*.md" \
	--ignore node_modules \
	--ignore "**/node_modules/**" \
	--ignore .venv \
	--ignore "**/.venv/**" \
	--ignore .build/site \
	--ignore "docs/wip/**" \
	--ignore "tests/fixtures/**" \
	--ignore "data/eval/runs/**" \
	--ignore "$(WEB_VIEWER_DIR)/playwright-report/**" \
	--ignore "$(WEB_VIEWER_DIR)/test-results/**" \
	--ignore "$(WEB_VIEWER_DIR)/validation-results/**" \
	--ignore "tests/stack-test/playwright-report/**" \
	--ignore "tests/stack-test/test-results/**" \
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

# Match CI lint job (python-app.yml): PYTHONPATH includes repo root so imports match Actions.
type:
	@echo "=== type (mypy): starting $$(date '+%Y-%m-%d %H:%M:%S') — may run several minutes with little output when clean ==="
	@export PYTHONPATH="$$PYTHONPATH:$(PWD)" && PYTHONUNBUFFERED=1 $(PYTHON) -m mypy --config-file pyproject.toml .

security: security-bandit security-audit

security-bandit:
	$(PYTHON) -m bandit -r . --exclude ./.venv,./.venv-dev,./infra/dgx/converge/.venv --skip B113,B108,B110,B310 --severity-level medium

security-audit:
	$(PYTHON) -m pip install --upgrade setuptools
	# Install ML dependencies to ensure they are audited

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
	@$(CODESPELL) src/ docs/ --skip="*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl"

spelling-docs:
	@echo "=== Spell Checking (Docs only) ==="
	@$(CODESPELL) docs/ --skip="*.pyc,*.json,*.xml,*.lock,*.mp3,*.whl"

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
	# Ignore CVE-2026-1839: transformers Trainer loads rng_state via torch.load without weights_only; fixed in 5.0.0rc3+.
	#   We pin transformers<5.0.0 (extractive QA / pipeline — see pyproject [ml]). Revisit when stable 5.x is adopted.
	# TODO(CVE-2026-1839): Remove --ignore-vuln after bumping transformers to a fixed 5.x release.
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
	# Ignore PYSEC-2025-189..197, PYSEC-2025-210, PYSEC-2026-139 (torch checkpoint /
	#   operator deserialization class). 11 advisories against torch <= 2.12.0 with
	#   NO fix versions (2.12.0 IS the latest). We only ``torch.load`` HuggingFace
	#   weights pinned in pyproject.toml; no user-supplied checkpoints.
	# TODO(torch PYSECs): drop ignores when upstream ships fixed versions and we
	#   bump the torch pin.
	#
	# Ignore PYSEC-2025-211..218 (transformers X-CLIP / Trainer / conversion-script
	#   deserialization, all RCE-class but require attacker-controlled checkpoint).
	#   8 advisories against transformers 4.x with NO fix in the 4.x line — fixes
	#   live only in the 5.x branch which we don't ship yet (sentence-transformers
	#   5.x allows transformers 4.41–5.x; we cap at <5 until extractive QA path
	#   is vendored — see [ml] in pyproject.toml).
	# TODO(transformers PYSECs): drop ignores after bumping transformers to a
	#   patched 5.x release (paired with the CVE-2026-1839 ignore above).
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
	# Note: If protobuf is updated to >=6.33.5 or >=7.0.0, this ignore can be removed
	# Note: en-core-web-sm is installed from GitHub (not PyPI), so it cannot be audited by pip-audit
	#       If it appears in audit output, it can be safely ignored as it's not from PyPI
	$(PYTHON) -m pip_audit --skip-editable \
		--ignore-vuln PYSEC-2022-42969 \
		--ignore-vuln CVE-2026-0994 \
		--ignore-vuln CVE-2026-4539 \
		--ignore-vuln CVE-2026-1839 \
		--ignore-vuln CVE-2025-69872 \
		--ignore-vuln CVE-2026-3219 \
		--ignore-vuln CVE-2026-6357 \
		--ignore-vuln PYSEC-2024-277 \
		--ignore-vuln PYSEC-2026-97 \
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
		--ignore-vuln PYSEC-2025-211 \
		--ignore-vuln PYSEC-2025-212 \
		--ignore-vuln PYSEC-2025-213 \
		--ignore-vuln PYSEC-2025-214 \
		--ignore-vuln PYSEC-2025-215 \
		--ignore-vuln PYSEC-2025-216 \
		--ignore-vuln PYSEC-2025-217 \
		--ignore-vuln PYSEC-2025-218 \
		--ignore-vuln PYSEC-2026-161 \
		--ignore-vuln MAL-2026-4750 \
		--ignore-vuln CVE-2025-3000
	# PYSEC-2026-161 (starlette<1.0.1, Host-header URL-path poisoning, GHSA-86qp-5c8j-p5mr):
	# Not exploitable in this codebase — grep -rn 'request.url.path' src/ is empty;
	# FastAPI routing uses the real request path, not the reconstructed URL. Traefik
	# also normalises Host upstream in prod (compose/docker-compose.prod.yml). Fix is in
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

docs-check: lint-markdown-docs spelling-docs docs
	@echo ""
	@echo "✓ Documentation validation complete (linting + spelling + build)"

# Coverage thresholds per layer (minimums to ensure balanced coverage)
# These are ambitious but achievable targets based on current coverage levels
# Combined threshold is enforced in CI; per-layer thresholds ensure no layer is neglected
COVERAGE_THRESHOLD_UNIT := 70          # Current: ~74% local, ~70% CI
COVERAGE_THRESHOLD_INTEGRATION := 42   # Raised 2026-04: integration-only line cov ~43% local
# E2E: full ``podcast_scraper`` tree in coverage denominator (``pyproject.toml`` only; no subtree omit).
# Target 40%; if local ``make coverage-check-e2e`` is below this, add pytest E2E until the gate passes.
COVERAGE_THRESHOLD_E2E := 40
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
	$(PYTHON) scripts/tools/check_self_hosted_runner_allowlist.py

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

validate-gi-schema:
	# Validate gi.json files against docs/architecture/gi/gi.schema.json (strict mode).
	# Usage: make validate-gi-schema [ARTIFACTS_DIR=path]. With no path, validates tests/fixtures (if any).
	# E2E tests that produce gi.json also run strict validation inline (ci-fast covers them).
	@if [ -n "$(ARTIFACTS_DIR)" ]; then \
		export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/validate_gi_schema.py "$(ARTIFACTS_DIR)"; \
	else \
		export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/validate_gi_schema.py tests/fixtures; \
	fi

validate-kg-schema:
	# Validate kg.json files against docs/architecture/kg/kg.schema.json (strict mode).
	# Usage: make validate-kg-schema [ARTIFACTS_DIR=path]. With no path, validates tests/fixtures (if any).
	@if [ -n "$(ARTIFACTS_DIR)" ]; then \
		export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/validate_kg_schema.py "$(ARTIFACTS_DIR)"; \
	else \
		export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/validate_kg_schema.py tests/fixtures; \
	fi

# GI/KG viewer v2 (#489): FastAPI + Vite. ``make init`` includes FastAPI via ``[dev]``; cd $(WEB_VIEWER_DIR) && npm install
.PHONY: serve serve-api serve-ui serve-app serve-app-dev serve-e2e-mock stack-build stack-build-llm stack-compose-validate stack-up stack-down stack-logs verify-stack-profiles stack-test-build stack-test-build-cloud stack-test-up stack-test-down stack-test-seed stack-test-playwright stack-test-export stack-test-ml stack-test-cloud-thin stack-test-ml-ci deploy-codespace restore-corpus restore-corpus-prod reprocess-corpus-from-transcripts corpus-compat-check index-two-tier enrich-relational-edges redo-diarization upgrade-status upgrade-check upgrade-dry-run upgrade-corpus upgrade-verify smoke-prod corpus-snapshot-manifest-validate corpus-snapshot-select-tag corpus-snapshot-select-tag-prod corpus-snapshot-selftest corpus-snapshot-integration
SERVE_OUTPUT_DIR ?= ./output
# Optional corpus-editing + jobs routes (health shows green when on). Override with SERVE_ARGS= to disable.
SERVE_ARGS ?= --enable-feeds-api --enable-operator-config-api --enable-jobs-api
# Default E2E mock RSS port (127.0.0.1:18765; override with E2E_MOCK_PORT when fixture URLs change).
# Deliberately not 8000 so ``serve-e2e-mock`` can run alongside ``serve-api`` (FastAPI default).
E2E_MOCK_PORT ?= 18765
serve:
	@echo "Running serve-api and serve-ui in parallel (Ctrl+C stops both)."
	@$(MAKE) -j2 serve-api serve-ui

serve-api:
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) -m $(PACKAGE).cli serve --output-dir "$(SERVE_OUTPUT_DIR)" $(SERVE_ARGS)

serve-ui:
	@cd $(WEB_VIEWER_DIR) && npm run dev

# Consumer Learning Player dev server (Vite). Proxies /api → 127.0.0.1:8000; run the API
# separately, or use ``serve-app-dev`` to run both with the local mock OAuth provider.
serve-app:
	@cd $(APP_DIR) && npm run dev

# One-command local app environment: the consumer API (with the dev/e2e mock OAuth provider
# + a dev session secret) and the Learning Player app, in parallel (Ctrl+C stops both).
# NEVER use APP_OAUTH_PROVIDER=mock outside local dev.
serve-app-dev:
	@echo "Running the consumer API (mock OAuth) + the Learning Player app in parallel (Ctrl+C stops both)."
	@APP_OAUTH_PROVIDER=mock APP_SESSION_SECRET=$${APP_SESSION_SECRET:-dev-secret} $(MAKE) -j2 serve-api serve-app

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

stack-build:
	@$(STACK_COMPOSE) build

stack-build-llm:
	@$(STACK_COMPOSE) --profile pipeline-llm build pipeline-llm

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
	@# story clean for the future GHCR publish step (RFC-081).
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
	@# Wipe the corpus volume both before (clean slate — a stale/partial LanceDB index
	@# from a prior run yields false "no_index" search failures; ephemeral CI runners
	@# never hit it, local re-runs do) and on EXIT (leave a clean laptop, no carryover).
	@set -u; \
	trap '$(MAKE) stack-test-down STACK_TEST_DOWN_VOLUMES=1 >/dev/null 2>&1 || true' EXIT; \
	$(MAKE) stack-test-down STACK_TEST_DOWN_VOLUMES=1 >/dev/null 2>&1 || true; \
	$(MAKE) stack-test-build \
		&& $(MAKE) stack-test-up \
		&& $(MAKE) stack-test-seed STACK_TEST_OPERATOR_VARIANT=ml \
		&& $(MAKE) stack-test-playwright

# RFC-081 Phase 1 — manual escape hatches for the deploy + restore
# workflows. Both fail loud if prerequisites (PAT / codespace name /
# backup repo) aren't wired yet.

# Trigger the ``deploy-codespace`` workflow on the named pre-prod
# codespace. Same code path as the auto-fired GHA on stack-test
# success; manual invocation is the operator's escape hatch when the
# auto-trigger didn't fire (e.g., stack-test workflow_run permission
# blip, or a forced redeploy after secrets change).
#
# Requires:
#   * gh auth login (or GH_TOKEN env)
#   * GH_TOKEN PAT must include ``codespaces`` scope (the default
#     ``GITHUB_TOKEN`` doesn't); store in ``CODESPACES_PAT`` GHA secret
#     and export when running this make target locally.
#   * ``CODESPACES_PAT_NAME`` (default ``podcast-scraper-preprod``) is
#     the codespace name; override via env.
deploy-codespace:
	@CS=$$($(MAKE) -s _resolve-codespace-name); \
	if [ -z "$$CS" ]; then exit 2; fi; \
	echo "Rebuilding codespace: $$CS"; \
	gh codespace rebuild --full --codespace "$$CS"

# Codespace lifecycle helpers (RFC-081 §Phase 1A operator wrappers).
#
# All four targets share name resolution via ``_resolve-codespace-name``:
#   1. ``$$CODESPACE_NAME`` env (full ``podcast-scraper-preprod-<suffix>``) — explicit override.
#   2. ``gh codespace list`` filtered by displayName ``podcast-scraper-preprod`` — default.
#
# Auth: same as deploy-codespace — gh CLI must be ``codespace``-scoped
# (``gh auth login -s codespace --web``). PAT-based auth in CI uses
# ``GH_TOKEN`` from the ``CODESPACES_PAT`` Actions secret (must be a
# **classic** PAT — fine-grained PATs are repo-scoped and 403 on
# ``/user/codespaces/...``).

_resolve-codespace-name:
	@if [ -n "$${CODESPACE_NAME:-}" ]; then \
		echo "$$CODESPACE_NAME"; \
	else \
		CS=$$(gh codespace list --json name,displayName -q \
			'.[] | select(.displayName=="podcast-scraper-preprod") | .name' 2>/dev/null | head -1); \
		if [ -z "$$CS" ]; then \
			echo "ERROR: no codespace with displayName=podcast-scraper-preprod found." >&2; \
			echo "       Set CODESPACE_NAME=<exact-name> or create the codespace first." >&2; \
			exit 2; \
		fi; \
		echo "$$CS"; \
	fi

# Wake / start the pre-prod codespace (no full rebuild — picks up updated
# Codespaces secrets at start). Use this after rotating GRAFANA_CLOUD_API_KEY
# / OPENAI_API_KEY / etc. so the running container's env reflects the new
# value (Codespaces secrets are baked into env at start, not read live).
codespace-start:
	@CS=$$($(MAKE) -s _resolve-codespace-name); \
	if [ -z "$$CS" ]; then exit 2; fi; \
	echo "Starting codespace: $$CS"; \
	gh api -X POST "/user/codespaces/$$CS/start" --jq '.state' >/dev/null && \
	echo "Start requested. Poll state with 'make codespace-status'."

# Suspend the pre-prod codespace (pauses billing; workspace state preserved).
# Use before stepping away — codespaces auto-suspend after 30 min idle anyway.
codespace-stop:
	@CS=$$($(MAKE) -s _resolve-codespace-name); \
	if [ -z "$$CS" ]; then exit 2; fi; \
	echo "Stopping codespace: $$CS"; \
	gh codespace stop --codespace "$$CS"

# Print current state (Available / ShuttingDown / Shutdown / Rebuilding / etc.)
codespace-status:
	@CS=$$($(MAKE) -s _resolve-codespace-name); \
	if [ -z "$$CS" ]; then exit 2; fi; \
	gh codespace list --json name,state,displayName,lastUsedAt \
		-q ".[] | select(.name==\"$$CS\") | \"\(.displayName) [\(.name)] state=\(.state) lastUsed=\(.lastUsedAt)\""

# Pull the codespace corpus to a local laptop directory via ``gh codespace cp``.
# Belt-and-suspenders backup before risky deploys (image rebuilds, profile flips,
# devcontainer changes). Pairs with ``codespace-restore-local`` which pushes a
# local backup back into the codespace.
#
# Default destination: ``$HOME/preprod_corpus_backup_<UTC date>/`` so multiple
# backups don't overwrite each other. Override with ``BACKUP_DIR=...``.
codespace-backup-local:
	@CS=$$($(MAKE) -s _resolve-codespace-name); \
	if [ -z "$$CS" ]; then exit 2; fi; \
	DEST="$${BACKUP_DIR:-$$HOME/preprod_corpus_backup_$$(date -u +%Y-%m-%d)}"; \
	mkdir -p "$$DEST"; \
	echo "Pulling corpus from codespace $$CS to $$DEST/ ..."; \
	gh codespace cp -e -c "$$CS" -r \
		'remote:/workspaces/podcast_scraper/.codespace_corpus' \
		"$$DEST/"; \
	echo ""; \
	echo "=== Verify backup ==="; \
	du -sh "$$DEST/.codespace_corpus" 2>/dev/null || echo "WARN: dir missing"; \
	gi=$$(find "$$DEST" -name '*.gi.json' 2>/dev/null | wc -l); \
	tx=$$(find "$$DEST" -name '*.txt' -path '*/transcripts/*' 2>/dev/null | wc -l); \
	echo "  gi.json artifacts: $$gi"; \
	echo "  transcripts:       $$tx"; \
	if [ "$$gi" = "0" ]; then \
		echo "ERROR: backup contains 0 gi.json artifacts; corpus may be empty." >&2; \
		exit 3; \
	fi; \
	echo "OK: backup written to $$DEST/.codespace_corpus/"

# Push a local laptop backup back into the codespace's corpus dir. Reverse of
# ``codespace-backup-local``. Idempotent: ``gh codespace cp`` overwrites
# existing files. Source defaults to the most recent backup under
# ``$HOME/preprod_corpus_backup_*``; override with ``BACKUP_DIR=...``.
codespace-restore-local:
	@CS=$$($(MAKE) -s _resolve-codespace-name); \
	if [ -z "$$CS" ]; then exit 2; fi; \
	if [ -n "$${BACKUP_DIR:-}" ]; then \
		SRC="$$BACKUP_DIR"; \
	else \
		SRC=$$(ls -1dt $$HOME/preprod_corpus_backup_* 2>/dev/null | head -1); \
		if [ -z "$$SRC" ]; then \
			echo "ERROR: no $$HOME/preprod_corpus_backup_* found. Override with BACKUP_DIR=..." >&2; \
			exit 2; \
		fi; \
	fi; \
	if [ ! -d "$$SRC/.codespace_corpus" ]; then \
		echo "ERROR: $$SRC/.codespace_corpus missing — not a valid backup." >&2; \
		exit 2; \
	fi; \
	echo "Restoring $$SRC/.codespace_corpus/ -> codespace $$CS:/workspaces/podcast_scraper/.codespace_corpus/"; \
	gh codespace cp -e -c "$$CS" -r \
		"$$SRC/.codespace_corpus" \
		'remote:/workspaces/podcast_scraper/'; \
	echo "OK: corpus restored into codespace."

# Trigger the cloud-side backup workflow (.github/workflows/backup-corpus.yml).
# workflow_dispatch only (no cron). Tarballs the codespace corpus and uploads
# to chipi/podcast_scraper-backup as a release asset. Matches workflow default:
# ``DRY_RUN=true`` skips upload; set ``DRY_RUN=false`` to publish.
codespace-backup-cloud:
	@DRY_RUN="$${DRY_RUN:-true}"; \
	echo "Dispatching backup-corpus.yml workflow (dry_run=$$DRY_RUN) ..."; \
	gh workflow run backup-corpus.yml --repo chipi/podcast_scraper -f dry_run=$$DRY_RUN; \
	sleep 3; \
	gh run list --workflow=backup-corpus.yml --repo chipi/podcast_scraper -L 1; \
	echo ""; \
	echo "Watch: gh run watch <id> --repo chipi/podcast_scraper"

# Corpus snapshot manifest helpers (RFC-084 / ADR-092 / #763).
corpus-snapshot-manifest-validate:
	@if [ -z "$(FILE)" ]; then \
		echo "FILE is required (path to snapshot.manifest.json)"; exit 2; \
	fi
	@bash scripts/ops/corpus_snapshot/validate_snapshot_manifest.sh "$(FILE)"

corpus-snapshot-select-tag:
	@BACKUP_REPO="$${PODCAST_BACKUP_REPO:-chipi/podcast_scraper-backup}"; \
	TAG_REGEX="$${TAG_REGEX:-^snapshot-[0-9]{8}$$}"; \
	export BACKUP_REPO TAG_REGEX PODCAST_BACKUP_TAG; \
	bash scripts/ops/corpus_snapshot/select_release_tag.sh

corpus-snapshot-selftest:
	@$(PYTHON) -m pytest tests/unit/scripts/test_corpus_snapshot_manifest.py tests/unit/scripts/test_stack_contract_restore_scripts.py -q && echo "MAKE_EXIT=0" || echo "MAKE_EXIT=$$?"

corpus-snapshot-integration:
	@$(PYTHON) -m pytest tests/unit/scripts/test_corpus_snapshot_manifest.py tests/unit/scripts/test_corpus_snapshot_integration.py tests/unit/scripts/test_stack_contract_restore_scripts.py -q && echo "MAKE_EXIT=0" || echo "MAKE_EXIT=$$?"

# Pull a corpus snapshot from chipi/podcast_scraper-backup (or PODCAST_BACKUP_REPO)
# and untar into ``.codespace_corpus/`` (the workspace-survives-suspend path that
# the codespace bind-mounts as the corpus root).
#
# Requires:
#   * gh auth login with read access to the backup repo (private by default).
#   * Run from inside the codespace OR with the workspace path overridden via
#     WORKSPACE_DIR (default: current dir).
#
corpus-snapshot-select-tag-prod:
	@BACKUP_REPO="$${PODCAST_BACKUP_REPO:-chipi/podcast_scraper-backup}"; \
	TAG_REGEX="$${TAG_REGEX:-^snapshot-prod-[0-9]{8}$$}"; \
	export BACKUP_REPO TAG_REGEX PODCAST_BACKUP_TAG; \
	bash scripts/ops/corpus_snapshot/select_release_tag.sh

# Codespace / pre-prod layout (``.codespace_corpus/`` in tarball). For prod VPS use
# ``restore-corpus-prod`` or ``prod-restore-corpus.yml`` / ``drill-restore-corpus.yml``.
restore-corpus:
	@WORKSPACE_DIR="$${WORKSPACE_DIR:-$$PWD}"; \
	export WORKSPACE_DIR PODCAST_BACKUP_REPO PODCAST_BACKUP_TAG TAG_REGEX='^snapshot-[0-9]{8}$$'; \
	bash scripts/ops/corpus_snapshot/restore_corpus_release.sh --layout codespace

# Prod VPS layout (top-level ``corpus/`` in tarball). Default parent ``/srv/podcast-scraper``
# when unset; override with ``WORKSPACE_DIR`` for local rehearsal.
restore-corpus-prod:
	@WORKSPACE_DIR="$${WORKSPACE_DIR:-/srv/podcast-scraper}"; \
	export WORKSPACE_DIR PODCAST_BACKUP_REPO PODCAST_BACKUP_TAG TAG_REGEX='^snapshot-prod-[0-9]{8}$$'; \
	bash scripts/ops/corpus_snapshot/restore_corpus_release.sh --layout prod

# Recompute GI/KG/search from on-disk transcripts without re-transcribing (#796).
# Requires CORPUS_DIR (corpus parent with feeds.spec.yaml + transcripts/).
reprocess-corpus-from-transcripts:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	test -f "$${CORPUS_DIR}/feeds.spec.yaml" || (echo "Missing $${CORPUS_DIR}/feeds.spec.yaml"; exit 1); \
	OP_CFG="$${CORPUS_DIR}/viewer_operator.yaml"; \
	test -f "$$OP_CFG" || OP_CFG="config/profiles/cloud_balanced.yaml"; \
	echo "Reprocessing $${CORPUS_DIR} (skip-existing + no-transcribe-missing)..."; \
	$(PYTHON) -m podcast_scraper.cli \
	  --config "$$OP_CFG" \
	  --feeds-spec "$${CORPUS_DIR}/feeds.spec.yaml" \
	  --output-dir "$${CORPUS_DIR}" \
	  --skip-existing \
	  --no-transcribe-missing; \
	echo "Optional: rebuild topic clusters when search/ index exists:"; \
	echo "  $(PYTHON) -m podcast_scraper.cli topic-clusters --output-dir $${CORPUS_DIR}"

# Report corpus/code compatibility for a on-disk corpus parent (#796 / #797).
corpus-compat-check:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -c "from pathlib import Path; from podcast_scraper.corpus_version import read_produced_by, assess_corpus_version_compat, MIN_SUPPORTED_CORPUS_CODE_VERSION; from podcast_scraper import __version__; root = Path('$${CORPUS_DIR}').expanduser().resolve(); pb = read_produced_by(root); ver, warn = assess_corpus_version_compat(pb); print(f'server={__version__} min_supported={MIN_SUPPORTED_CORPUS_CODE_VERSION}'); print(f'corpus_code_version={ver!r}'); print(f'produced_by={pb!r}'); import sys; (print(f'WARNING: {warn}') or sys.exit(1)) if warn else print('COMPAT OK')"

# Build the two-tier LanceDB index from corpus artifacts (RFC-090 Phase 2, follow-up
# B). Native path for corpora with no legacy index to migrate. CORPUS_DIR required.
index-two-tier:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli index-two-tier --output-dir "$${CORPUS_DIR}"

# Derive relational edges into each gi.json (#874): Podcast->HAS_EPISODE->Episode,
# Insight->MENTIONS->Entity, and Quote->SPOKEN_BY->Person (diarized episodes only).
# Idempotent cross-layer pass; re-run after new diarization (#876) to fill SPOKEN_BY.
enrich-relational-edges:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli enrich-edges --output-dir "$${CORPUS_DIR}"

# Re-diarize the whisper_transcription episodes (#876/#925) and cascade downstream.
# Diarization rewrites the transcript (offsets shift), so re-running those episodes
# re-does transcription+diarization+GI/KG/CIL; --reprocess-source scopes it to the
# whisper set so the already-diarized direct_download episodes are untouched.
# Diarization is driven entirely by PROFILE (must enable `diarize:`, e.g.
# config/profiles/local.yaml) -- never hardcoded here. HF_NET_ENV allows the gated
# pyannote weights to download on first run (needs HF_TOKEN). After the run, re-derive
# relational edges so corpus-wide SPOKEN_BY is filled.
#   make redo-diarization CORPUS_DIR=<corpus> PROFILE=config/profiles/local.yaml
redo-diarization:
	@test -n "$${CORPUS_DIR:-}" || { echo "CORPUS_DIR required (corpus parent path)"; exit 1; }; \
	test -n "$${PROFILE:-}" || { echo "PROFILE required (a diarization-enabled profile, e.g. config/profiles/local.yaml)"; exit 1; }; \
	test -f "$${CORPUS_DIR}/feeds.spec.yaml" || { echo "Missing $${CORPUS_DIR}/feeds.spec.yaml"; exit 1; }; \
	echo "Re-diarizing whisper_transcription episodes in $${CORPUS_DIR} under $${PROFILE}..."; \
	$(HF_NET_ENV) $(PYTHON) -m podcast_scraper.cli \
	  --config "$${PROFILE}" \
	  --feeds-spec "$${CORPUS_DIR}/feeds.spec.yaml" \
	  --output-dir "$${CORPUS_DIR}" \
	  --skip-existing \
	  --reprocess-source whisper_transcription; \
	echo "Re-deriving relational edges (corpus-wide SPOKEN_BY)..."; \
	$(PYTHON) -m podcast_scraper.cli enrich-edges --output-dir "$${CORPUS_DIR}"

# Phase 1 of staged re-diarization (#947): download + cache RAW audio for the existing
# corpus episodes only, WITHOUT transcribing/diarizing. --reprocess-existing-only scopes
# to on-disk GUIDs; --reprocess-source forces the existing whisper episodes back through
# download; --pipeline-stage download_only stops before transcription. The audio lands in
# the durable cache (default .cache/audio, external to the corpus) so phase 2
# (migrate-diarization) reprocesses off the cache with no feed re-fetch. Set
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

# Strict existing-only re-diarization MIGRATION (#876/#946). Like redo-diarization but
# adds --reprocess-existing-only: the episode set is restricted to GUIDs already on disk
# under each feed's run_*/metadata/. New feed items are dropped and max/offset/date caps
# are ignored, so the corpus NEVER grows -- a migration of existing data while ingestion
# is paused, not an ingest. Audio comes from the #947 raw-audio cache when present (run
# ``make download-audio`` first to prime it -> cache HIT, no feed fetch, and rolled-off
# episodes stay re-diarizable); otherwise it is re-fetched from the live feed enclosure
# and cached for next time. Aborts loudly if no on-disk GUIDs are found (mis-pointed
# CORPUS_DIR). Set AUDIO_CACHE_IN_CORPUS=1 to keep the cache inside the corpus.
#   make migrate-diarization CORPUS_DIR=<corpus> PROFILE=config/profiles/cloud_with_dgx_primary.yaml
migrate-diarization:
	@test -n "$${CORPUS_DIR:-}" || { echo "CORPUS_DIR required (corpus parent path)"; exit 1; }; \
	test -n "$${PROFILE:-}" || { echo "PROFILE required (a diarization-enabled profile, e.g. config/profiles/cloud_with_dgx_primary.yaml)"; exit 1; }; \
	test -f "$${CORPUS_DIR}/feeds.spec.yaml" || { echo "Missing $${CORPUS_DIR}/feeds.spec.yaml"; exit 1; }; \
	echo "Migrating (existing-only) whisper_transcription episodes in $${CORPUS_DIR} under $${PROFILE}..."; \
	$(HF_NET_ENV) $(PYTHON) -m podcast_scraper.cli \
	  --config "$${PROFILE}" \
	  --feeds-spec "$${CORPUS_DIR}/feeds.spec.yaml" \
	  --output-dir "$${CORPUS_DIR}" \
	  --skip-existing \
	  --reprocess-source whisper_transcription \
	  --reprocess-existing-only; \
	echo "Re-deriving relational edges (corpus-wide SPOKEN_BY)..."; \
	$(PYTHON) -m podcast_scraper.cli enrich-edges --output-dir "$${CORPUS_DIR}"

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
upgrade-corpus:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli upgrade run --corpus-dir "$${CORPUS_DIR}" --yes

upgrade-verify:
	@test -n "$${CORPUS_DIR:-}" || (echo "CORPUS_DIR required (corpus parent path)"; exit 1); \
	$(PYTHON) -m podcast_scraper.cli upgrade verify --corpus-dir "$${CORPUS_DIR}"

# Post-deploy prod smoke over Tailscale HTTPS (#797). Requires PROD_TAILNET_FQDN.
# Optional: SMOKE_CORPUS_PATH (in-container corpus root for API path=, e.g. /app/output on prod).
smoke-prod:
	@test -n "$${PROD_TAILNET_FQDN:-}" || (echo "PROD_TAILNET_FQDN required (MagicDNS FQDN)"; exit 1); \
	args="$$PROD_TAILNET_FQDN"; \
	if [ -n "$${SMOKE_CORPUS_PATH:-}" ]; then args="$$args --corpus-path $$SMOKE_CORPUS_PATH"; fi; \
	bash scripts/ops/post_deploy_smoke.sh $$args

# GHCR retention dry-run (#802). Requires gh CLI + package scopes.
ghcr-prune-dry-run:
	$(PYTHON) scripts/ops/ghcr_compute_retention.py

# Vitest unit tests for TypeScript utility logic (no browser needed)
test-ui:
	@echo "Vitest unit tests + coverage gate (gi-kg-viewer, #914)..."
	@cd $(WEB_VIEWER_DIR) && npm install && npm run test:coverage

# Playwright browser E2E (install browsers once: cd $(WEB_VIEWER_DIR) && npx playwright install firefox)
test-ui-e2e:
	@echo "Playwright E2E (gi-kg-viewer)..."
	@cd $(WEB_VIEWER_DIR) && npm install && npx playwright install firefox && npm run test:e2e

# Production viewer bundle: ``vue-tsc -b && vite build`` (catches strict-mode TS
# errors that ``vitest`` / ``playwright`` skip). Wired into every CI target so a
# vue-tsc regression cannot reach the stack-test Docker build path on main.
build-viewer:
	@echo "Production viewer bundle (vue-tsc -b && vite build)..."
	@cd $(WEB_VIEWER_DIR) && npm install && npm run build

# Consumer Learning Player — Vitest unit + coverage gate (mirrors ``test-ui``).
test-app:
	@echo "Vitest unit tests + coverage gate (Learning Player)..."
	@cd $(APP_DIR) && npm install && npm run test:coverage

# Consumer Learning Player — Playwright E2E (mobile + desktop projects).
# Install browsers once: cd $(APP_DIR) && npx playwright install chromium
test-app-e2e:
	@echo "Playwright E2E (Learning Player)..."
	@cd $(APP_DIR) && npm install && npx playwright install chromium && npm run test:e2e

# Production app bundle: ``vue-tsc -b && vite build`` (catches strict-mode TS errors that
# vitest/playwright skip). Run locally before push for app PRs (mirrors ``build-viewer``).
build-app:
	@echo "Production Learning Player bundle (vue-tsc -b && vite build)..."
	@cd $(APP_DIR) && npm install && npm run build

# Consumer Learning Player container (RFC-099 §10): its own nginx-served static image.
app-docker-build:
	@echo "Building the Learning Player Docker image..."
	@docker build -t podcast-scraper-learning-app:latest $(APP_DIR)

# Stack + consumer app overlay (api + viewer + learning-app on one network).
APP_STACK_COMPOSE ?= docker compose -f compose/docker-compose.stack.yml -f compose/docker-compose.app.yml
app-stack-config:
	@$(APP_STACK_COMPOSE) config >/dev/null && echo "compose config OK (stack + app overlay)"
app-stack-up:
	@echo "Stack + Learning Player up (app on APP_PORT, default 8081; api proxied)..."
	@$(APP_STACK_COMPOSE) up -d --build
app-stack-down:
	@$(APP_STACK_COMPOSE) down $(REMOVE_VOLUMES)

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

gil-quality-metrics:
	# PRD-017 GIL quality metrics over .gi.json (see scripts/tools/gil_quality_metrics.py).
	# Usage: make gil-quality-metrics DIR=path/to/run [ARGS='--enforce --json']
	@if [ -z "$(DIR)" ]; then \
		echo "DIR is required (e.g. make gil-quality-metrics DIR=./output)"; exit 2; \
	fi
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/gil_quality_metrics.py "$(DIR)" $(ARGS)

diarization-quality:
	# Corpus diarization / speaker-attribution quality (#876) — run after a re-diarization to
	# validate named host+guest, quote attribution + timestamps, and no network-org speaker
	# names. See scripts/tools/diarization_quality_metrics.py.
	# Usage: make diarization-quality CORPUS_DIR=path/to/corpus [ARGS='--enforce --json']
	@if [ -z "$(CORPUS_DIR)" ]; then \
		echo "CORPUS_DIR is required (e.g. make diarization-quality CORPUS_DIR=./output)"; exit 2; \
	fi
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/diarization_quality_metrics.py "$(CORPUS_DIR)" $(ARGS)

compare-gil-runs:
	# Compare GIL outcomes between two pipeline run directories (metadata/*.gi.json).
	# Usage: make compare-gil-runs REF=path/to/reference/run CAND=path/to/candidate/run
	# See scripts/tools/compare_gil_runs.py and docs/wip/gil-ml-vs-openai-outcome-benchmark.md
	@if [ -z "$(REF)" ] || [ -z "$(CAND)" ]; then \
		echo "Usage: make compare-gil-runs REF=path/to/reference/run CAND=path/to/candidate/run"; exit 2; \
	fi
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/compare_gil_runs.py "$(REF)" "$(CAND)"

kg-quality-metrics:
	# PRD-019 KG quality metrics over .kg.json (see scripts/tools/kg_quality_metrics.py).
	@if [ -z "$(DIR)" ]; then \
		echo "DIR is required (e.g. make kg-quality-metrics DIR=./output)"; exit 2; \
	fi
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/kg_quality_metrics.py "$(DIR)" $(ARGS)

quality-metrics-ci:
	# Same GIL+KG enforce as GitHub Actions test-unit job (committed fixtures).
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/gil_quality_metrics.py tests/fixtures/gil_kg_ci_enforce --enforce --strict-schema --fail-on-errors --min-extraction-coverage 1.0 --min-grounded-insight-rate 1.0 --min-quote-validity-rate 1.0 --min-avg-insights 1 --min-avg-quotes 1
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)/src" && $(PYTHON) scripts/tools/kg_quality_metrics.py tests/fixtures/gil_kg_ci_enforce --enforce --strict-schema --fail-on-errors --min-artifacts 1 --min-avg-nodes 1 --min-avg-edges 0 --min-extraction-coverage 1.0

cleanup-processes:
	# Clean up leftover Python/test processes from previous runs
	# Covers pytest workers, ML model probe processes, and worker calculator
	@echo "Cleaning up leftover test processes..."
	@pkill -f "pytest" 2>/dev/null || true
	# Dropped python.*podcast_scraper.*test: it matched cli serve when
	# --output-dir was under .test_outputs (test in test_outputs).
	@pkill -f "gw[0-9]" 2>/dev/null || true
	@pkill -f "python.*ml_model_cache_helpers" 2>/dev/null || true
	@pkill -f "python.*calculate_test_workers" 2>/dev/null || true
	@echo "Process cleanup complete"

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

test-unit: cleanup-processes
	# Unit tests: parallel execution for faster feedback
	# Parallelism: Uses unit-specific worker calculation (memory-aware, caps at 8)
	# Unit tests are lightweight (~100 MB per worker), so can use more workers
	# Network isolation enabled to match CI behavior and catch network dependency issues early
	# Note: cleanup-processes runs automatically via pytest fixture, but also called here for safety
	$(PYTHON) -m pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --disable-socket --allow-hosts=127.0.0.1,localhost

test-unit-no-ml: cleanup-processes
	# Runs import guard + unit tests with *current* $(PYTHON) (.venv if present).
	# For the same environment as GitHub test-unit, use: make venv-dev-init && make test-unit-dev-venv
	@echo "Verifying unit tests can import modules without ML dependencies..."
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(PYTHON) scripts/tools/check_unit_test_imports.py
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --disable-socket --allow-hosts=127.0.0.1,localhost

test-unit-dev-venv: cleanup-processes
	@test -x "$(VENVDEV_PY)" || { echo "Missing $(VENVDEV_PY). Run: make venv-dev-init [VENVDEV=$(VENVDEV)]"; exit 1; }
	@echo "Verifying imports (CI-style, .[dev] only)..."
	@export PYTHONPATH="${PYTHONPATH}:$(PWD)" && $(VENVDEV_PY) scripts/tools/check_unit_test_imports.py
	@echo "Running unit tests in $(VENVDEV)..."
	$(VENVDEV_PY) -m pytest tests/unit/ --cov=$(PACKAGE) --cov-report=term-missing -m 'not integration and not e2e' -n $(shell $(VENVDEV_PY) scripts/tools/calculate_test_workers.py --test-type unit --max-workers 8 2>/dev/null || echo 4) --disable-socket --allow-hosts=127.0.0.1,localhost

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
	$(PYTHON) -m pytest tests/integration/ -m "integration and critical_path and not ml_models" -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type integration --max-workers 5 2>/dev/null || echo 3) --cov=$(PACKAGE) --cov-report=term-missing --allow-hosts=127.0.0.1,localhost --reruns 2 --reruns-delay 1 --durations=20

test-ci:
	# CI test suite: parallel execution for speed
	# Includes: unit + critical path integration + critical path e2e (includes ML if models cached)
	# Uses conservative worker calculation (default type, caps at 5) for mixed test types
	# Note: Non-critical path tests run on main branch only
	$(PYTHON) -m pytest -m '(not integration and not e2e) or (integration and critical_path) or (e2e and critical_path)' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type default --max-workers 5 2>/dev/null || echo 3) --disable-socket --allow-hosts=127.0.0.1,localhost --cov=$(PACKAGE) --cov-report=term-missing

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

test-diarization: cleanup-processes
	# Speaker-diarization tests (RFC-058). Two tiers under the ``diarization`` marker:
	#   - integration: factory + provider wiring with pyannote MOCKED (always runs, no token).
	#   - e2e (tests/e2e/test_diarization_e2e.py): the REAL pyannote full flow — needs the
	#     gated models cached (``make preload-ml-models`` with an HF token that accepted the
	#     model terms) and skips when the model/token isn't available.
	# The e2e tier is selected by NO other target, so without this it never runs in CI/nightly.
	@echo "🎙️  Running pyannote diarization tests at $$(date '+%H:%M:%S')..."
	$(PYTHON) -m pytest tests/ -m diarization -v --tb=short --allow-hosts=127.0.0.1,localhost

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
	@E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/integration/ -m 'integration and critical_path' \
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
	@E2E_TEST_MODE=fast PYTHONUNBUFFERED=1 $(PYTHON) -m pytest tests/integration/ -m 'integration and critical_path' \
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

benchmark-acceptance:
	@# Generate performance benchmarking report from acceptance test results
	@if [ -z "$(SESSION_ID)" ]; then \
		echo "❌ Error: SESSION_ID is required"; \
		echo "Usage: make benchmark-acceptance SESSION_ID=<session_id> [COMPARE_BASELINE=...]"; \
		echo ""; \
		echo "Options:"; \
		echo "  SESSION_ID=id            Session ID (required, e.g., '20260208_101601')"; \
		echo "  OUTPUT_DIR=path          Output directory (default: .test_outputs/acceptance)"; \
		echo "  COMPARE_BASELINE=id      Baseline ID to compare against (optional)"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make benchmark-acceptance SESSION_ID=20260208_101601"; \
		echo "  make benchmark-acceptance SESSION_ID=20260208_101601 COMPARE_BASELINE=baseline_v1"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/acceptance/generate_performance_benchmark.py \
		--session-id $(SESSION_ID) \
		--output-dir "$(or $(OUTPUT_DIR),.test_outputs/acceptance)" \
		$(if $(COMPARE_BASELINE),--compare-baseline $(COMPARE_BASELINE)) \
		--output-format both \
		--log-level INFO
	@echo ""
	@echo "✓ Performance benchmark report generated"
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
	@E2E_TEST_MODE=multi_episode pytest tests/e2e/ --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD_E2E) -m 'e2e and not nightly' -n $(shell $(PYTHON) scripts/tools/calculate_test_workers.py --test-type e2e --max-workers 4 2>/dev/null || echo 2) --allow-hosts=127.0.0.1,localhost --reruns 3 --reruns-delay 1 -q

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

_ci_body: format-check lint lint-markdown type security complexity deadcode docstrings spelling check-test-policy test test-ui test-ui-e2e build-viewer test-app test-app-e2e build-app coverage-enforce docs build stack-test-ml-ci
	# Final gate is ``stack-test-ml-ci`` (build → up → seed → Playwright →
	# always-teardown). ml pipeline only — ~5-10 min, no API keys, no cloud
	# cost. Cloud-thin variant (``stack-test-cloud-thin``) is a separate
	# explicit target; public CI does not run it (recurring API cost) and
	# ``ci`` does not include it for the same reason locally.

# Sequential sub-makes + banners so long-quiet steps (especially mypy) never look like a hang.
ci-fast:
	# Note: ci-fast skips coverage-enforce and test-ui-e2e (Playwright). For viewer work use ci-ui-fast.
	@set -e; \
	echo ""; echo "=== ci-fast START $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""; \
	$(MAKE) cleanup-processes; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] format-check ==="; $(MAKE) format-check; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint ==="; $(MAKE) lint; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint-markdown ==="; $(MAKE) lint-markdown; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] type (mypy) ==="; $(MAKE) type; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] security ==="; $(MAKE) security; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] complexity ==="; $(MAKE) complexity; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] deadcode ==="; $(MAKE) deadcode; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] docstrings ==="; $(MAKE) docstrings; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] spelling ==="; $(MAKE) spelling; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] check-test-policy ==="; $(MAKE) check-test-policy; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] quality-metrics-ci ==="; $(MAKE) quality-metrics-ci; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] profile-drift-check ==="; $(MAKE) profile-drift-check; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] test-fast (pytest) ==="; $(MAKE) test-fast; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] corpus-snapshot-selftest ==="; $(MAKE) corpus-snapshot-selftest; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] test-ui ==="; $(MAKE) test-ui; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] build-viewer ==="; $(MAKE) build-viewer; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] docs ==="; $(MAKE) docs; \
	echo ""; echo "=== ci-fast [$$(date '+%Y-%m-%d %H:%M:%S')] build ==="; $(MAKE) build; \
	echo ""; echo "=== ci-fast DONE $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""

# Viewer-heavy gate: like ci-fast but skips Python tests/e2e and runs Playwright (longer than Vitest alone).
ci-ui-fast:
	# Note: ci-ui-fast skips coverage-enforce and Python tests/e2e; Playwright still needs browsers installed.
	@set -e; \
	echo ""; echo "=== ci-ui-fast START $$(date '+%Y-%m-%d %H:%M:%S') ==="; echo ""; \
	$(MAKE) cleanup-processes; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] format-check ==="; $(MAKE) format-check; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint ==="; $(MAKE) lint; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] lint-markdown ==="; $(MAKE) lint-markdown; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] type (mypy) ==="; $(MAKE) type; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] security ==="; $(MAKE) security; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] complexity ==="; $(MAKE) complexity; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] deadcode ==="; $(MAKE) deadcode; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] docstrings ==="; $(MAKE) docstrings; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] spelling ==="; $(MAKE) spelling; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] check-test-policy ==="; $(MAKE) check-test-policy; \
	echo ""; echo "=== ci-ui-fast [$$(date '+%Y-%m-%d %H:%M:%S')] quality-metrics-ci ==="; $(MAKE) quality-metrics-ci; \
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
	# Note: ci-ui-full = ci-ui-fast + stack-test-ml-ci. Requires Docker
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
# (e.g. ``.../viewer-validation-corpus/v2``). The raw ``feeds/<feed>/metadata/``
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
	@echo "Starting API + UI with SERVE_OUTPUT_DIR=$(PWD) (repo root)."
	@echo "Synthetic validation corpus root (pass this as CORPUS=):"
	@echo "  $(VIEWER_VALIDATION_CORPUS)"
	@SERVE_OUTPUT_DIR=$(PWD) $(MAKE) -j2 serve-api serve-ui

build-validation-index:
	# Build ALL search artifacts the Tier-3 walk needs, against the in-repo
	# synthetic validation corpus. Run this BEFORE ``make serve-for-validation``.
	# Unlocks the index-dependent specs (V1/V5 — Library/Digest handoffs — do NOT
	# need any of this; they pass on the path fix alone):
	#   1. LanceDB two-tier index (search/lance_index/) — the single search layer
	#      (native BM25 + dense + hybrid RRF). Required for V3 (semantic search →
	#      Show on graph); without it V3 is SKIPPED (test.skip on
	#      ``!indexJson.available``).
	#   2. topic_clusters.json — required for V2 (digest topic-band) and V4
	#      (dashboard topic-cluster chip). Without it the Intelligence tab shows
	#      "Topic clusters not yet built" → no chips → V4 fails.
	# Both live under <corpus>/search/ and are gitignored (binary,
	# embedding-model-hash-keyed, regenerable) — never committed.
	@CORPUS=$(if $(CORPUS),$(CORPUS),$(VIEWER_VALIDATION_CORPUS)); \
	echo "=== 1/2 Building LanceDB two-tier index at $$CORPUS/search/lance_index ==="; \
	$(PYTHON) -m podcast_scraper.cli index-two-tier \
		--output-dir $$CORPUS
	@CORPUS=$(if $(CORPUS),$(CORPUS),$(VIEWER_VALIDATION_CORPUS)); \
	echo "=== 2/2 Building topic_clusters.json at $$CORPUS/search ==="; \
	$(PYTHON) -m podcast_scraper.cli topic-clusters \
		--output-dir $$CORPUS \
		--threshold 0.35
	@echo ""
	@echo "Done — LanceDB two-tier + topic_clusters built. Now run:"
	@echo "  make serve-for-validation       (terminal 1)"
	@echo "  make ci-ui-validation CORPUS=$(VIEWER_VALIDATION_CORPUS)  (terminal 2)"

ci-clean: clean-all format-check lint lint-markdown type security complexity deadcode docstrings spelling check-test-policy preload-ml-models test test-ui test-ui-e2e build-viewer coverage-enforce docs build

ci-nightly: format-check lint lint-markdown type security complexity deadcode docstrings spelling check-test-policy preload-ml-models-production test-unit test-integration test-e2e test-nightly test-ui test-ui-e2e build-viewer coverage-enforce docs build
	@echo ""
	@echo "✓ Full nightly CI chain completed"

docker-build:
	@echo "Building Docker image (ML-enabled variant, production ML preload, default)..."
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg INSTALL_EXTRAS=ml \
		--build-arg PRELOAD_ML_MODELS=true \
		-t podcast-scraper:test \
		-f docker/pipeline/Dockerfile .

docker-build-llm:
	@echo "Building Docker image (LLM-only variant, ~200MB)..."
	@echo "This variant excludes ML dependencies for smaller size and faster builds..."
	@echo ""
	@DOCKER_BUILDKIT=1 docker build \
		--build-arg INSTALL_EXTRAS="" \
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

# Experiment commands (eval / benchmarks)
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
	@cmd="$(PYTHON) scripts/eval/data/generate_episode_metadata.py --input-dir $(INPUT_DIR)"; \
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
	@cmd="$(PYTHON) scripts/eval/data/generate_source_index.py --source-dir $(SOURCE_DIR)"; \
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
	cmd="$(PYTHON) scripts/eval/data/create_dataset_json.py --dataset-id $(DATASET_ID) --eval-dir $$EVAL_DIR --description \"$$DESCRIPTION\""; \
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
	$(PYTHON) scripts/eval/data/create_dataset_json.py \
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
	$(PYTHON) scripts/eval/data/create_dataset_json.py \
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
	$(PYTHON) scripts/eval/data/create_dataset_json.py \
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
	cmd="$(PYTHON) scripts/eval/data/materialize_dataset.py --dataset-id $(DATASET_ID) --output-dir $$OUTPUT_DIR"; \
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
	@cmd="$(PYTHON) scripts/eval/compare/promote_run.py --run-id $(RUN_ID) --as $(AS) --promoted-id $(PROMOTED_ID) --reason \"$(REASON)\""; \
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

registry-promote:
	@# Promote a baseline config.yaml into the code registry (model registry).
	@# Usage: make registry-promote BASELINE_ID=baseline_ml_dev_authority_smoke_v1 MODE_ID=ml_small_authority
	@if [ -z "$(BASELINE_ID)" ] || [ -z "$(MODE_ID)" ]; then \
		echo "❌ Error: BASELINE_ID and MODE_ID are required"; \
		echo ""; \
		echo "Usage: make registry-promote BASELINE_ID=baseline_ml_dev_authority_smoke_v1 MODE_ID=ml_small_authority"; \
		exit 1; \
	fi
	@echo "Promoting baseline $(BASELINE_ID) → mode $(MODE_ID)..."
	@$(PYTHON) scripts/registry/promote_baseline.py \
		--baseline-id $(BASELINE_ID) \
		--mode-id $(MODE_ID) \
		--baseline-dir data/eval/baselines/$(BASELINE_ID) \
		--registry-path src/podcast_scraper/providers/ml/model_registry.py
	@$(MAKE) format
	@echo "✓ Promotion complete. Review changes to src/podcast_scraper/providers/ml/model_registry.py"

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
	cmd="$(PYTHON) scripts/eval/data/materialize_baseline.py --baseline-id $$RUN_ID --dataset-id $(DATASET_ID) --output-dir data/eval/runs"; \
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
	@# scripts/eval/experiment/run_experiment.py does ``from scripts.eval.data...``
	@# so the repo root must be on sys.path. Mirror the pattern used by every
	@# other ``scripts/`` invocation in this Makefile (mypy / e2e mock / etc).
	@export PYTHONPATH="$$PYTHONPATH:$(PWD)"; \
	cmd="$(PYTHON) scripts/eval/experiment/run_experiment.py $(CONFIG)"; \
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

# Issue #510: frozen release profiles (see data/profiles/README.md)
profile-freeze:
	@# Usage: make profile-freeze VERSION=v2.6.0 PIPELINE_CONFIG=config/profiles/freeze/<provider>.yaml
	@# Optional: DATASET_ID=... OUTPUT=... SKIP_WARMUP=1 E2E_FEED=podcast1_multi_episode
	@# Optional: SAMPLE_INTERVAL=0.25 NO_STAGE_TRUTH=1 MONITOR=1 (live monitor ticks -> <VERSION>.monitor.log)
	@if [ -z "$(VERSION)" ]; then \
		echo "❌ Error: VERSION is required (e.g. VERSION=v2.6.0)"; \
		exit 1; \
	fi
	@if [ -z "$(PIPELINE_CONFIG)" ]; then \
		echo "❌ Error: PIPELINE_CONFIG is required (YAML of podcast_scraper Config fields)"; \
		echo "  Use one of config/profiles/freeze/<provider>.yaml (see config/profiles/freeze/README.md)."; \
		exit 1; \
	fi
	@if [ ! -f "$(PIPELINE_CONFIG)" ]; then \
		echo "❌ Error: PIPELINE_CONFIG not found: $(PIPELINE_CONFIG)"; \
		exit 1; \
	fi
	@cmd="$(PYTHON) scripts/eval/profile/freeze_profile.py --version $(VERSION) --pipeline-config $(PIPELINE_CONFIG)"; \
	if [ -n "$(DATASET_ID)" ]; then cmd="$$cmd --dataset-id $(DATASET_ID)"; fi; \
	if [ -n "$(OUTPUT)" ]; then cmd="$$cmd --output $(OUTPUT)"; fi; \
	if [ "$(SKIP_WARMUP)" = "1" ]; then cmd="$$cmd --skip-warmup"; fi; \
	if [ -n "$(E2E_FEED)" ]; then cmd="$$cmd --e2e-feed $(E2E_FEED)"; fi; \
	if [ -n "$(SAMPLE_INTERVAL)" ]; then cmd="$$cmd --sample-interval $(SAMPLE_INTERVAL)"; fi; \
	if [ "$(NO_STAGE_TRUTH)" = "1" ]; then cmd="$$cmd --no-stage-truth-snapshot"; fi; \
	if [ "$(MONITOR)" = "1" ]; then cmd="$$cmd --monitor"; fi; \
	eval $$cmd

profile-diff:
	@# Usage: make profile-diff FROM=v2.5.0 TO=v2.6.0  (paths: data/profiles/<tag>.yaml)
	@if [ -z "$(FROM)" ] || [ -z "$(TO)" ]; then \
		echo "❌ Error: FROM and TO are required (e.g. FROM=v2.5.0 TO=v2.6.0)"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/eval/profile/diff_profiles.py "data/profiles/$(FROM).yaml" "data/profiles/$(TO).yaml"

profile-promote:
	@# Usage: make profile-promote SOURCE=data/profiles/v2.6-wip-openai.yaml \
	@#        PROMOTED_ID=v2.6.0-openai REASON="Release v2.6.0 reference"
	@# Optional: NO_STAGE_TRUTH_REQUIRED=1 DRY_RUN=1
	@if [ -z "$(SOURCE)" ]; then \
		echo "❌ Error: SOURCE is required (path to working profile YAML)"; \
		exit 1; \
	fi
	@if [ -z "$(PROMOTED_ID)" ]; then \
		echo "❌ Error: PROMOTED_ID is required (e.g. v2.6.0-openai)"; \
		exit 1; \
	fi
	@if [ -z "$(REASON)" ]; then \
		echo "❌ Error: REASON is required (why this profile is being promoted)"; \
		exit 1; \
	fi
	@cmd="$(PYTHON) scripts/eval/profile/promote_profile.py \
		--source \"$(SOURCE)\" \
		--promoted-id \"$(PROMOTED_ID)\" \
		--reason \"$(REASON)\""; \
	if [ "$(NO_STAGE_TRUTH_REQUIRED)" = "1" ]; then cmd="$$cmd --no-stage-truth-required"; fi; \
	if [ "$(DRY_RUN)" = "1" ]; then cmd="$$cmd --dry-run"; fi; \
	eval $$cmd

ml-param-sweep:
	@# Autoresearch Track B: ML parameter ratchet loop (no LLM judges needed).
	@# Usage: make ml-param-sweep MODEL=bart_led [MAX_FAILS=3] [MIN_GAIN=0.01] [DRY_RUN=1]
	@# MODEL choices: bart_led, pegasus_led (defined in autoresearch/ml_param_tuning/param_space.yaml)
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ Error: MODEL is required (e.g. make ml-param-sweep MODEL=bart_led)"; \
		exit 1; \
	fi; \
	cmd="$(PYTHON) autoresearch/ml_param_tuning/sweep.py --model $(MODEL)"; \
	if [ -n "$(MAX_FAILS)" ]; then cmd="$$cmd --max-fails $(MAX_FAILS)"; fi; \
	if [ -n "$(MIN_GAIN)" ]; then cmd="$$cmd --min-gain $(MIN_GAIN)"; fi; \
	if [ "$(DRY_RUN)" = "1" ]; then cmd="$$cmd --dry-run"; fi; \
	if [ -n "$(LOG_LEVEL)" ]; then cmd="$$cmd --log-level $(LOG_LEVEL)"; fi; \
	echo "Running ML param sweep for MODEL=$(MODEL)..."; \
	eval $$cmd

autoresearch-score:
	@# Autoresearch Track A: reuse run_experiment + metrics, then optional LLM judges; final scalar on stdout.
	@# Usage: make autoresearch-score [CONFIG=path] [REFERENCE=id] [DRY_RUN=1] [LOG_LEVEL=INFO]
	@# Loads .env and optional .env.autoresearch from repo root (see autoresearch_track_a.load_local_dotenv_files).
	@# Env: AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY, AUTORESEARCH_JUDGE_* keys; optional AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1
	@# Optional: AUTORESEARCH_EVAL_N, AUTORESEARCH_SCORE_ROUGE_WEIGHT
	@cmd="$(PYTHON) autoresearch/initial_prompt_tuning/prompt_tuning/eval/score.py"; \
	if [ -n "$(CONFIG)" ]; then \
		echo "  Config: $(CONFIG)"; \
		cmd="$$cmd --config $(CONFIG)"; \
	else \
		echo "  Config: (default autoresearch_prompt_openai_smoke_bullets_v1.yaml)"; \
	fi; \
	if [ -n "$(JUDGE_CONFIG)" ]; then \
		echo "  Judge config: $(JUDGE_CONFIG)"; \
		cmd="$$cmd --judge-config $(JUDGE_CONFIG)"; \
	else \
		echo "  Judge config: (default judge_config.yaml — OpenAI + Anthropic)"; \
	fi; \
	if [ -n "$(REFERENCE)" ]; then \
		echo "  Reference: $(REFERENCE)"; \
		cmd="$$cmd --reference $(REFERENCE)"; \
	fi; \
	if [ -n "$(LOG_LEVEL)" ]; then \
		cmd="$$cmd --log-level $(LOG_LEVEL)"; \
	fi; \
	if [ "$(DRY_RUN)" = "1" ]; then \
		echo "  Mode: dry-run (ROUGE only via score-only, no judges)"; \
		cmd="$$cmd --dry-run"; \
	fi; \
	echo "Running autoresearch score harness..."; \
	eval $$cmd; \
	ec=$$?; \
	echo ""; \
	if [ $$ec -eq 0 ]; then \
		echo "✓ Scalar is the lone float line above (stdout)."; \
	else \
		echo "❌ autoresearch score failed (exit $$ec)."; \
	fi; \
	exit $$ec

autoresearch-score-bundled:
	@# Bundled prompt tuning: eval run + ROUGE + dual judges; final scalar on stdout.
	@# Usage: make autoresearch-score-bundled [CONFIG=path] [REFERENCE=id] [DRY_RUN=1] [LOG_LEVEL=INFO]
	@# Loads .env and optional .env.autoresearch from repo root.
	@# Env: AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY, AUTORESEARCH_JUDGE_* keys; optional AUTORESEARCH_ALLOW_PRODUCTION_KEYS=1
	@# Optional: AUTORESEARCH_EVAL_N, AUTORESEARCH_SCORE_ROUGE_WEIGHT
	@cmd="$(PYTHON) autoresearch/bundled_prompt_tuning/eval/score.py"; \
	if [ -n "$(CONFIG)" ]; then \
		echo "  Config: $(CONFIG)"; \
		cmd="$$cmd --config $(CONFIG)"; \
	else \
		echo "  Config: (default autoresearch_prompt_openai_bundled_smoke_bullets_v1.yaml)"; \
	fi; \
	if [ -n "$(REFERENCE)" ]; then \
		echo "  Reference: $(REFERENCE)"; \
		cmd="$$cmd --reference $(REFERENCE)"; \
	fi; \
	if [ -n "$(LOG_LEVEL)" ]; then \
		cmd="$$cmd --log-level $(LOG_LEVEL)"; \
	fi; \
	if [ "$(DRY_RUN)" = "1" ]; then \
		echo "  Mode: dry-run (ROUGE only via score-only, no judges)"; \
		cmd="$$cmd --dry-run"; \
	fi; \
	echo "Running autoresearch bundled score harness..."; \
	eval $$cmd; \
	ec=$$?; \
	echo ""; \
	if [ $$ec -eq 0 ]; then \
		echo "✓ Scalar is the lone float line above (stdout)."; \
	else \
		echo "❌ autoresearch-score-bundled failed (exit $$ec)."; \
	fi; \
	exit $$ec

silver-pairwise:
	@# Pairwise LLM judge: compare two silver candidate runs, print winner name on stdout.
	@# Usage: make silver-pairwise CANDIDATE_A=<run_id> CANDIDATE_B=<run_id> [DATASET=curated_5feeds_smoke_v1] [OUTPUT=path/to/results.json]
	@# Loads .env and .env.autoresearch for AUTORESEARCH_JUDGE_* keys.
	@# Example:
	@#   make silver-pairwise \
	@#     CANDIDATE_A=silver_candidate_openai_gpt54_smoke_v1 \
	@#     CANDIDATE_B=silver_candidate_anthropic_sonnet46_smoke_v1
	@if [ -z "$(CANDIDATE_A)" ] || [ -z "$(CANDIDATE_B)" ]; then \
		echo "❌ Error: CANDIDATE_A and CANDIDATE_B are required"; \
		echo ""; \
		echo "Usage: make silver-pairwise CANDIDATE_A=<run_id> CANDIDATE_B=<run_id>"; \
		exit 1; \
	fi
	@dataset=$${DATASET:-curated_5feeds_smoke_v1}; \
	cmd="$(PYTHON) scripts/eval/score/pairwise_judge.py \
		--candidate-a data/eval/runs/$(CANDIDATE_A) \
		--candidate-b data/eval/runs/$(CANDIDATE_B) \
		--transcripts data/eval/materialized/$$dataset"; \
	if [ -n "$(OUTPUT)" ]; then \
		cmd="$$cmd --output $(OUTPUT)"; \
	fi; \
	eval $$cmd

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
	@$(PYTHON) scripts/eval/compare/list_runs.py \
		$(if $(DATASET_ID),--dataset-id $(DATASET_ID))

baselines-list:
	@# List all baselines
	@# Usage: make baselines-list [DATASET_ID=dataset_id]
	@$(PYTHON) scripts/eval/compare/list_runs.py --baselines \
		$(if $(DATASET_ID),--dataset-id $(DATASET_ID))

run-compare:
	@# Streamlit run comparison UI (Issue #373)
	@# Usage: make run-compare [BASELINE=baseline_id]
	@echo "Launching run comparison tool..."
	@BASELINE=$(BASELINE) $(PYTHON) -m streamlit run tools/run_compare/app.py \
		--server.headless=false --server.port=8501

runs-compare:
	@# Compare two experiment runs
	@# Usage: make runs-compare RUN1=run_id1 RUN2=run_id2 [DATASET_ID=dataset_id] [OUTPUT=path/to/report.md]
	@if [ -z "$(RUN1)" ] || [ -z "$(RUN2)" ]; then \
		echo "❌ Error: RUN1 and RUN2 are required"; \
		echo "Usage: make runs-compare RUN1=run_id1 RUN2=run_id2"; \
		exit 1; \
	fi
	@$(PYTHON) scripts/eval/compare/compare_runs.py \
		--run1 $(RUN1) \
		--run2 $(RUN2) \
		$(if $(DATASET_ID),--dataset-id $(DATASET_ID)) \
		$(if $(OUTPUT),--output $(OUTPUT))

report-multi-run:
	@# Generate multi-run comparison report (baseline + N runs, vs-reference metrics)
	@# Usage: make report-multi-run [BASELINE_ID=id] RUN_IDS=id1,id2,... REFERENCE_ID=ref_id [OUTPUT=path] [TITLE=...] [LABELS=...]
	@# Default: BASELINE_ID=baseline_ml_prod_authority_smoke_v1 RUN_IDS=hybrid_ml_tier1_smoke_v1,hybrid_ml_tier2_qwen25_7b_smoke_v1 REFERENCE_ID=silver_gpt4o_smoke_v1
	@# Optional 32B tier2 (after ollama pull qwen2.5:32b): append hybrid_ml_tier2_qwen25_32b_smoke_v1 to RUN_IDS
	@REFERENCE_ID="$(REFERENCE_ID)"; \
	if [ -z "$$REFERENCE_ID" ]; then REFERENCE_ID=silver_gpt4o_smoke_v1; fi; \
	BASELINE_ID="$(BASELINE_ID)"; RUN_IDS="$(RUN_IDS)"; \
	if [ -z "$$BASELINE_ID" ] && [ -z "$$RUN_IDS" ]; then \
		BASELINE_ID=baseline_ml_prod_authority_smoke_v1; \
		RUN_IDS=hybrid_ml_tier1_smoke_v1,hybrid_ml_tier2_qwen25_7b_smoke_v1; \
		echo "Using defaults: BASELINE_ID=$$BASELINE_ID RUN_IDS=$$RUN_IDS REFERENCE_ID=$$REFERENCE_ID"; \
	fi; \
	cmd="$(PYTHON) scripts/eval/compare/multi_run_report.py --reference-id $$REFERENCE_ID"; \
	if [ -n "$$BASELINE_ID" ]; then cmd="$$cmd --baseline-id $$BASELINE_ID"; fi; \
	if [ -n "$$RUN_IDS" ]; then cmd="$$cmd --run-ids $$RUN_IDS"; fi; \
	if [ -n "$(OUTPUT)" ]; then cmd="$$cmd --output $(OUTPUT)"; else cmd="$$cmd --output docs/wip/multi_run_comparison.md"; fi; \
	if [ -n "$(TITLE)" ]; then cmd="$$cmd --title '$(TITLE)'"; fi; \
	if [ -n "$(LABELS)" ]; then cmd="$$cmd --labels '$(LABELS)'"; fi; \
	if [ -n "$(DATASET_ID)" ]; then cmd="$$cmd --dataset-id $(DATASET_ID)"; fi; \
	if [ -n "$(BASELINES_DIR)" ]; then cmd="$$cmd --baselines-dir $(BASELINES_DIR)"; fi; \
	if [ -n "$(RUNS_DIR)" ]; then cmd="$$cmd --runs-dir $(RUNS_DIR)"; fi; \
	eval $$cmd
	@echo "✓ Report generated. See OUTPUT path above."

embedding-provider-eval:
	@# Compare embedding providers on the operator's corpus (ADR-098 / #897).
	@# Uses gi.json SUPPORTED_BY edges as ground truth — no human annotation needed.
	@# Usage: make embedding-provider-eval CORPUS=./output [MODE=quote|transcript] [MAX_PAIRS=500] [OLLAMA_URL=http://...]
	@# MODE=quote (default): insight → supporting Quote node (short text, paraphrase quality).
	@# MODE=transcript: insight → episode's full transcript (long text, where MiniLM's 256-token cap truncates).
	@if [ -z "$(CORPUS)" ]; then \
		echo "ERROR: CORPUS is required (path to corpus root, parent of feeds/)." >&2; \
		echo "  Example: make embedding-provider-eval CORPUS=./output" >&2; \
		exit 1; \
	fi
	@$(PYTHON) scripts/eval_embedding_providers.py \
		--corpus "$(CORPUS)" \
		$(if $(MODE),--mode $(MODE)) \
		$(if $(MAX_PAIRS),--max-pairs $(MAX_PAIRS)) \
		$(if $(OLLAMA_URL),--ollama-base-url $(OLLAMA_URL))

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
	@$(PYTHON) scripts/eval/experiment/run_benchmark.py \
		--config $(CONFIG) \
		--baseline $(BASELINE) \
		$(if $(SMOKE),--smoke) \
		$(if $(ALL),--all) \
		$(if $(DATASETS),--datasets $(DATASETS)) \
		$(if $(REFERENCE),--reference $(REFERENCE)) \
		$(if $(OUTPUT_DIR),--output-dir $(OUTPUT_DIR)) \
		$(if $(LOG_LEVEL),--log-level $(LOG_LEVEL))

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
		$(PYTHON) -m flake8 --config .flake8 $$file || true; \
		$(PYTHON) -m mypy $$file || true; \
	done
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

pipeline-validate:
	@# Full pipeline validation: run providers through summary → GI → KG → bridge.
	@# Usage:
	@#   make pipeline-validate                            # all providers (cloud + local)
	@#   make pipeline-validate PROVIDER=gemini MODEL=gemini-2.5-flash-lite  # single provider
	@#   make pipeline-validate PV_ARGS="--all-cloud"      # all cloud only
	@#   make pipeline-validate PV_ARGS="--all-local"      # all Ollama only
	@if [ -n "$(PROVIDER)" ] && [ -n "$(MODEL)" ]; then \
		$(PYTHON) scripts/eval/experiment/pipeline_validate.py --provider "$(PROVIDER)" --model "$(MODEL)"; \
	elif [ -n "$(PV_ARGS)" ]; then \
		$(PYTHON) scripts/eval/experiment/pipeline_validate.py $(PV_ARGS); \
	else \
		$(PYTHON) scripts/eval/experiment/pipeline_validate.py --all; \
	fi

transcription-sweep:
	@# Transcription model sweep: compare local Whisper models on quality + speed.
	@# Usage:
	@#   make transcription-sweep                           # default 4 models × held-out episodes
	@#   make transcription-sweep MODELS=base.en,medium.en  # specific models
	@#   make transcription-sweep EPISODES=p01_e03          # single episode
	$(PYTHON) scripts/eval/experiment/transcription_sweep.py \
		--audio-dir tests/fixtures/audio/$(FIXTURES_VERSION) \
		--reference-dir data/eval/materialized/curated_5feeds_benchmark_v2 \
		--episodes "$${EPISODES:-p01_e03,p02_e03,p03_e03,p04_e03,p05_e03}" \
		--models "$${MODELS:-tiny.en,base.en,small.en,medium.en}"

# ============================================================================
# Infrastructure (RFC-082): provision Hetzner VPS via OpenTofu.
#
# Convenience wrappers around the infra/tofu sops-encrypted-state script.
# Both targets source secrets from infra/.env.local (gitignored) so the
# operator does not have to re-export HCLOUD_TOKEN / TS_API_KEY each time.
# See infra/.env.local.example for the required keys.
# ============================================================================
INFRA_ENV_FILE := infra/.env.local

infra-plan:
	@# Sources infra/.env.local + runs `tofu init && tofu plan`. No state changes.
	@if [ ! -f $(INFRA_ENV_FILE) ]; then \
		echo "ERROR: $(INFRA_ENV_FILE) is missing." >&2; \
		echo "Copy from infra/.env.local.example and fill in your secrets." >&2; \
		exit 1; \
	fi
	@set -a && . ./$(INFRA_ENV_FILE) && set +a && cd infra && ./tofu init && ./tofu plan

infra-recover:
	@# Idempotent recovery from a partial/failed apply: deletes orphan Hetzner
	@# resources (by name), wipes stale local state, re-runs apply, auto-imports
	@# the live ACL on conflict. Safe to run repeatedly.
	@bash infra/recover.sh

infra-apply:
	@# Sources infra/.env.local + runs `tofu init && plan && apply`. CREATES REAL HETZNER RESOURCES.
	@# tofu apply prompts y/n before any change, so this is safe to invoke; abort with Ctrl-C at the prompt.
	@if [ ! -f $(INFRA_ENV_FILE) ]; then \
		echo "ERROR: $(INFRA_ENV_FILE) is missing." >&2; \
		echo "Copy from infra/.env.local.example and fill in your secrets." >&2; \
		exit 1; \
	fi
	@set -a && . ./$(INFRA_ENV_FILE) && set +a && cd infra && ./tofu init && ./tofu plan && ./tofu apply

INFRA_DRILL_ENV_FILE := infra/.env.drill.local

# --- Local DR-drill OpenTofu (workspace `drill`, isolated from prod) — #1027 ---
# Source infra/.env.drill.local, map its secrets to the TF_VAR_* the config requires
# (same names as the GHA drill-infra-* jobs), and run the wrapper in drill mode
# (INFRA_WORKSPACE=drill → terraform.tfstate.enc.drill; the prod state is never touched).
# plan is read-only; apply/destroy prompt y/n. Requires the drill state present at
# infra/terraform/terraform.tfstate.enc.drill (from a drill apply run's
# `terraform-state-after-apply-drill` artifact, or a prior local drill apply).
drill-tofu-plan drill-tofu-apply drill-tofu-destroy: drill-tofu-%:
	@if [ ! -f $(INFRA_DRILL_ENV_FILE) ]; then \
		echo "ERROR: $(INFRA_DRILL_ENV_FILE) missing — run: make drill-env (and add TS_API_KEY, OPERATOR_SSH_PUBLIC_KEY, TAILNET_NAME)." >&2; \
		exit 1; \
	fi
	@set -a && . ./$(INFRA_DRILL_ENV_FILE) && set +a; \
	 : "$${HCLOUD_TOKEN_DRILL:?set HCLOUD_TOKEN_DRILL in $(INFRA_DRILL_ENV_FILE)}"; \
	 : "$${TS_API_KEY:?set TS_API_KEY in $(INFRA_DRILL_ENV_FILE)}"; \
	 : "$${OPERATOR_SSH_PUBLIC_KEY:?set OPERATOR_SSH_PUBLIC_KEY in $(INFRA_DRILL_ENV_FILE)}"; \
	 : "$${TAILNET_NAME:?set TAILNET_NAME in $(INFRA_DRILL_ENV_FILE)}"; \
	 export TF_VAR_hcloud_token="$$HCLOUD_TOKEN_DRILL"; \
	 export TF_VAR_tailscale_api_key="$$TS_API_KEY"; \
	 export TF_VAR_ssh_public_key="$$OPERATOR_SSH_PUBLIC_KEY"; \
	 export TF_VAR_tailscale_tailnet="$$TAILNET_NAME"; \
	 cd infra \
	   && yes yes | INFRA_WORKSPACE=drill ./tofu init \
	   && INFRA_WORKSPACE=drill ./tofu $* -var-file=terraform.drill.ci.tfvars
	@# `yes yes |` answers the one-time "migrate workspaces to local" prompt that
	@# tofu raises on a fresh checkout (the wrapper decrypts drill state into
	@# terraform.tfstate.d/drill/ before init, so tofu offers to adopt it). The
	@# prompt needs the literal word "yes"; -input=false would hard-error it.
	@# On an already-initialized .terraform/ there is no prompt and stdin is ignored.

dgx-smoke:
	@# RFC-089: probe DGX Ollama via tailnet (non-fatal when DGX offline).
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

# --- DGX config management (pyinfra) — see infra/dgx/converge/README.md -----
DGX_CONVERGE_DIR := infra/dgx/converge
DGX_CONVERGE_VENV := $(DGX_CONVERGE_DIR)/.venv
DGX_PYINFRA := $(DGX_CONVERGE_VENV)/bin/pyinfra
INFRA_DGX_ENV_FILE := infra/.env.dgx.local

# Source infra/.env.dgx.local if present, otherwise require DGX_TAILNET_FQDN in
# the calling shell. Bail with a friendly message + cp hint if neither is set.
define _dgx_env
	if [ -f $(INFRA_DGX_ENV_FILE) ]; then \
		set -a && . ./$(INFRA_DGX_ENV_FILE) && set +a; \
	fi; \
	if [ -z "$${DGX_TAILNET_FQDN:-}" ]; then \
		echo "ERROR: DGX_TAILNET_FQDN unset and $(INFRA_DGX_ENV_FILE) missing." >&2; \
		echo "  → cp infra/.env.dgx.local.example $(INFRA_DGX_ENV_FILE) and edit." >&2; \
		exit 1; \
	fi
endef

$(DGX_PYINFRA): $(DGX_CONVERGE_DIR)/requirements.txt
	@python3 -m venv $(DGX_CONVERGE_VENV)
	@$(DGX_CONVERGE_VENV)/bin/pip install --quiet --upgrade pip
	@$(DGX_CONVERGE_VENV)/bin/pip install --quiet -r $(DGX_CONVERGE_DIR)/requirements.txt
	@touch $(DGX_PYINFRA)

dgx-ssh-test:
	@# Confirm SSH from laptop to DGX works. Auto-sources $(INFRA_DGX_ENV_FILE).
	@# Two branches so we can quote the key path (it may contain spaces, e.g. NVIDIA Sync's nvsync.key).
	@$(_dgx_env); \
		host=$${DGX_SSH_HOST:-$$DGX_TAILNET_FQDN}; \
		user=$${DGX_SSH_USER:-root}; port=$${DGX_SSH_PORT:-22}; key=$${DGX_SSH_KEY:-}; \
		if [ -n "$$key" ] && [ -f "$$key" ]; then \
			echo "→ ssh -i \"$$key\" -p $$port $$user@$$host 'echo ok && uname -a'"; \
			ssh -i "$$key" -p "$$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$$user@$$host" 'echo ok && uname -a'; \
		else \
			if [ -n "$$key" ]; then echo "WARN: DGX_SSH_KEY set but $$key missing — falling back to ssh-agent / ~/.ssh/config." >&2; fi; \
			echo "→ ssh -p $$port $$user@$$host 'echo ok && uname -a'"; \
			ssh -p "$$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$$user@$$host" 'echo ok && uname -a'; \
		fi

# pyinfra: --sudo escalates non-root SSH users for privileged ops (systemd, /etc, /opt).
# If user is already root, --sudo is a no-op.
# NB: dgx-verify runs verify.py (assertions only, real execution — NOT --dry, which
# wouldn't actually probe DGX). dgx-deploy (#814) runs deploy.py to install
# faster-whisper-server convergently; both targets share the same inventory.
dgx-verify: $(DGX_PYINFRA)
	@# Read-only assertions, real execution. Fails on drift. Auto-sources $(INFRA_DGX_ENV_FILE).
	@$(_dgx_env); \
		cd $(DGX_CONVERGE_DIR) && .venv/bin/pyinfra --sudo -y inventory.py verify.py

# Convergent install on DGX (#814): faster-whisper-server systemd service,
# venv, env file, ACL-friendly bind. Idempotent — re-run is a no-op when
# nothing drifted. See infra/dgx/converge/deploy.py for the operations list
# and infra/dgx/converge/README.md for one-time SSH setup.
dgx-deploy: $(DGX_PYINFRA)
	@# Sudo required: writes /etc/systemd, /etc/faster-whisper, /opt/faster-whisper.
	@# Auto-sources $(INFRA_DGX_ENV_FILE). Run ``make dgx-verify`` afterwards.
	@$(_dgx_env); \
		cd $(DGX_CONVERGE_DIR) && .venv/bin/pyinfra --sudo -y inventory.py deploy.py

drill-env:
	@echo "DR drill Hetzner token (same scope as GitHub secret HCLOUD_TOKEN_DRILL):"
	@echo "  1. cp infra/.env.drill.local.example infra/.env.drill.local"
	@echo "  2. Edit infra/.env.drill.local — paste the drill-project API token."
	@echo "  3. Load into your current shell (make cannot modify the parent shell):"
	@echo "       set -a && . ./infra/.env.drill.local && set +a"
	@echo "  4. Or run orphan cleanup without step 3:"
	@echo "       make delete-drill-hetzner-orphans"

delete-drill-hetzner-orphans:
	@# Sources infra/.env.drill.local + runs scripts/ops/delete_drill_hetzner_orphans.sh (drill project only).
	@if [ ! -f $(INFRA_DRILL_ENV_FILE) ]; then \
		echo "ERROR: $(INFRA_DRILL_ENV_FILE) missing. Run: make drill-env" >&2; \
		exit 1; \
	fi; \
	 set -a && . ./$(INFRA_DRILL_ENV_FILE) && set +a; \
	 if [ -z "$${HCLOUD_TOKEN_DRILL:-}" ]; then \
		echo "ERROR: HCLOUD_TOKEN_DRILL not set after sourcing $(INFRA_DRILL_ENV_FILE)" >&2; \
		exit 1; \
	 fi; \
	 ./scripts/ops/delete_drill_hetzner_orphans.sh $(if $(filter 1 true yes,$(DRY_RUN)),--check-only,); \
	 echo "MAKE_EXIT=$$?"
