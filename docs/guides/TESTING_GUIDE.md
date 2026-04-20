# Testing Guide

> **Document Structure:**
>
> - **[Testing Strategy](../architecture/TESTING_STRATEGY.md)** - High-level philosophy, test pyramid, decision criteria; **Browser UI E2E (Playwright)** as an additive layer
> - **This document** - Quick reference, test execution commands
> - **[Unit Testing Guide](UNIT_TESTING_GUIDE.md)** - Unit test mocking patterns and isolation
> - **[Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md)** - Integration test mocking guidelines
> - **[RSS and feed ingestion](RSS_GUIDE.md)** - How RSS is fetched and parsed (helps when testing scraping vs transcript download)
> - **[E2E Testing Guide](E2E_TESTING_GUIDE.md)** - E2E server, real ML models, OpenAI mocking; **E2E feeds and server options** (feeds per mode, error injection, URLs); chaos tests (e.g. 404 audio) assert run index records failed episodes; **download vs multi-feed resilience** split ([Download resilience E2E](E2E_TESTING_GUIDE.md#download-resilience-e2e)); **browser E2E** for the GI/KG Vue viewer (`make test-ui-e2e`)
> - **[Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)** - What to test and prioritization

## Quick Reference

| Layer | Speed | Scope | ML/AI | Mocking |
| ------- | ------- | ------- | ------ | --------- |
| **Unit** | < 100ms | Single function | Mocked | All dependencies mocked |
| **Integration** | < 5s | Component interactions | Mocked | External services + ML/AI mocked |
| **E2E** | < 60s | Complete workflow | Real | No mocking (real everything) |
| **Browser UI E2E** | ~1-3 min (suite) | Vue viewer in Firefox (Playwright) | N/A | Vite + route/API mocks in specs |

**Unit tests and `pyproject` extras:** `tests/unit/` must **only** depend on **`[dev]`** -- never on `[ml]`, `[llm]`, `[compare]`, or `[server]`. CI `test-unit` installs `.[dev]` only, so any test requiring a non-`[dev]` extra will be silently skipped and never validated. If a test needs FastAPI, httpx, torch, spaCy, faiss, etc., move it to `tests/integration/` (where CI installs `.[dev,ml,llm,server]`). Do **not** use `pytest.importorskip()` in `tests/unit/` to work around missing extras. See [Unit Testing Guide -- Pyproject extras](UNIT_TESTING_GUIDE.md#pyproject-extras-what-unit-tests-may-depend-on) and [Testing Strategy -- Unit tests and optional extras](../architecture/TESTING_STRATEGY.md#unit-tests-and-optional-extras-pyproject).

**Automated policy enforcement:** Two scripts run in `make ci` and `make ci-fast` to
catch testing-policy violations before they reach CI:

| Script | Make target | What it checks |
| ------ | ----------- | -------------- |
| `check_unit_test_imports.py` | `make check-unit-imports` | Library modules import without ML deps at import time |
| `check_test_policy.py` | `make check-test-policy` | 3-tier ML/AI boundary rules (see table below) |

`check_test_policy.py` enforces four rules:

| Rule ID | Scope | Violation |
| ------- | ----- | --------- |
| U1-importorskip | `tests/unit/` | `pytest.importorskip()` -- move to `integration/` or mock |
| U2-available-guard | `tests/unit/` | `*_AVAILABLE` skip guards -- mock ML deps instead of skipping |
| I1-ml-models-marker | `tests/integration/` | `@pytest.mark.ml_models` -- real ML belongs in `tests/e2e/` |
| G1-empty-test-file | all `tests/` | Zero `test_` methods -- delete or add tests |

Run `make check-test-policy` locally after adding or moving tests. Pass `--fix-hint`
for remediation suggestions. Both scripts live in `scripts/tools/`.

**Decision Tree:**

1. Testing the **GI/KG viewer UI** in a real browser (graph, search shell, keyboard, theme)? --> **`make test-ui-e2e`** (Playwright). See [Browser E2E (GI / KG Viewer v2)](#browser-e2e-gi-kg-viewer-v2) and [Testing Strategy -- Browser UI E2E](../architecture/TESTING_STRATEGY.md#browser-ui-e2e-playwright).
2. Testing a complete **CLI / library / service** workflow? --> **E2E Test** (pytest)
3. Testing component interactions? --> **Integration Test**
4. Testing a single function? --> **Unit Test**

## Running Tests

### Default Commands

```bash

# Unit tests (parallel, network isolated)

make test-unit

# Same dependency set as CI test-unit (.[dev] only) — separate venv, leaves .venv unchanged

make venv-dev-init && make test-unit-dev-venv

# Integration tests (parallel, with reruns)

make test-integration

# E2E tests (parallel, with reruns)

make test-e2e

# All tests

make test
```

### Browser E2E (GI / KG Viewer v2) {#browser-e2e-gi-kg-viewer-v2}

Playwright drives a **real browser** against the Vue SPA. This stack is **orthogonal to pytest**:
specs are not collected by `pytest`, and **`make test` does not run them**. Strategically, it is
documented as an **additive** layer on the test pyramid — see
[Testing Strategy — Browser UI E2E (Playwright)](../architecture/TESTING_STRATEGY.md#browser-ui-e2e-playwright)
and [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md).

**Where this lives in the repo:** npm commands and `package.json` are under `web/gi-kg-viewer/`;
`make test-ui` / `make test-ui-e2e` run from the root. See
[Polyglot repository guide](POLYGLOT_REPO_GUIDE.md).

**E2E surface map:** [E2E_SURFACE_MAP.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) lists
viewer surfaces, entry paths, and stable selectors. Use it when **debugging** Playwright failures,
manual repros, or **agent-driven** browser tools (same a11y vocabulary as the tests). **Whenever you change viewer UX** (labels,
layout, routes, tokens, a11y names, or E2E flows), update artifacts in this order: **(1)** the
surface map, **(2)** `e2e/*.spec.ts` / `helpers.ts` / `fixtures.ts` and run **`make test-ui-e2e`**,
**(3)** [VIEWER_IA.md](../uxs/VIEWER_IA.md) if **shell IA** changed, then [UXS-001](../uxs/UXS-001-gi-kg-viewer.md) and/or the relevant [feature UXS](../uxs/index.md) if the **visual** / experience contract changed.
Full checklist: [E2E Testing Guide — When you change viewer UX](E2E_TESTING_GUIDE.md#when-you-change-viewer-ux-required-workflow)
([GitHub #509](https://github.com/chipi/podcast_scraper/issues/509)). Agent-browser loop:
[Agent-Browser Closed Loop Guide](AGENT_BROWSER_LOOP_GUIDE.md).

#### How it fits next to pytest

| Concern | Tool | Location |
| -------- | ---- | -------- |
| Viewer **TS utility logic** (parsing, merge, metrics, formatting) | **Vitest** | `web/gi-kg-viewer/src/utils/*.test.ts` |
| Viewer **HTTP API** (`/api/*`) -- pure logic helpers | pytest **unit** tests (no FastAPI needed) | `tests/unit/podcast_scraper/server/` (catalog, index staleness, pathutil) |
| Viewer **HTTP API** — wired app + real files | pytest **integration** | `tests/integration/server/` (`test_server_api.py`, `test_viewer_corpus_library.py`, `test_index_rebuild.py`, …) |
| Viewer **UI** (render, click, keyboard, graph container) | **Playwright** | `web/gi-kg-viewer/e2e/*.spec.ts` |
| Full **pipeline** / CLI / providers | pytest **E2E** | `tests/e2e/` |

#### Commands

```bash
make test-ui          # Vitest unit tests (fast, no browser)
make test-ui-e2e      # Playwright browser E2E (needs Firefox)
```

**`make test-ui`** runs `npm run test:unit` (Vitest) in `web/gi-kg-viewer`. Tests cover pure
TypeScript logic: artifact parsing, GI+KG merge, bridge-aware dedupe where implemented, metrics,
formatting, colors, visual groups,
and search-focus mapping. No browser or DOM required — runs in ~150 ms.

**`make test-ui-e2e`** runs `npm install` in `web/gi-kg-viewer`, installs the **Firefox** browser
for Playwright, and runs `npm run test:e2e`. Playwright’s **`webServer`** starts **Vite** on
**127.0.0.1:5174** so it does not collide with `npm run dev` on **5173**.

**What CI proves vs full stack:** That setup exercises the **Vue UI** in a real browser with
**Vite**; many specs **mock** `fetch` or rely on **offline** fixtures so the job stays fast.
It does **not** prove **`python -m podcast_scraper.cli serve`** (FastAPI + mounted **`dist/`** +
live **`/api/*`** on corpus files). Use **`serve`** / **`make serve`** for manual smoke of the
combined server, and **`tests/integration/server/`** (e.g. **`test_server_api.py`**) for pytest
coverage of a wired `create_app` and temp corpus.

For interactive debugging: `cd web/gi-kg-viewer && npx playwright test --ui` (see
[viewer README](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md)).

#### CI

GitHub Actions jobs:

- **`viewer-unit`** — runs `npm run test:unit` (Vitest, fast).
- **`viewer-e2e`** — runs `npm run test:e2e` (Playwright + Firefox) after the pytest E2E job
  that applies to the event (**`test-e2e-fast`** on PRs, **`test-e2e`** on push to
  `main` / `release/*`). **`coverage-unified`** waits on **`viewer-e2e`** so the merge report
  runs only after browser E2E has passed.

Both viewer jobs are in `.github/workflows/python-app.yml`. Touching `web/gi-kg-viewer/` in a
PR should include green runs for both (see
[CONTRIBUTING.md](https://github.com/chipi/podcast_scraper/blob/main/CONTRIBUTING.md)).

### GIL, KG, CIL, and semantic search validation {#gil-kg-cil-and-semantic-search-validation}

Use this table when you change **GIL**, **KG**, **`bridge.json`**, **CIL HTTP**, **FAISS
indexing**, or **search response shape**.

| Change area | Suggested checks |
| ----------- | ---------------- |
| GIL pipeline, `gi.json`, `gi` CLI | `make test-unit -k gi` (or scoped paths), `tests/e2e/test_gi_kg_cli_subprocess_e2e.py` (`gi validate` smoke); see [Testing Strategy — GIL Testing](../architecture/TESTING_STRATEGY.md#gil-testing-implemented--prd-017-rfc-049050) |
| KG pipeline, `kg` CLI | `tests/unit/kg/`, `tests/e2e/test_gi_kg_cli_subprocess_e2e.py` (`kg validate` / `kg inspect`) |
| `bridge.json` builder | `tests/unit/builders/test_bridge_builder.py`, `tests/integration/test_bridge_integration.py` |
| CIL query helpers / HTTP | `tests/unit/podcast_scraper/server/test_cil_queries.py`, `tests/integration/server/test_cil_api.py` |
| Transcript search + **lift** + offset math | `tests/unit/podcast_scraper/search/test_transcript_chunk_lift.py`, `test_gil_chunk_offset_verify.py`, `tests/integration/server/test_viewer_search.py` |
| Viewer merge / bridge TS | `make test-ui`; UX changes also `make test-ui-e2e` + [E2E surface map](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) |

**Corpus-level gate (indexed run only):** `make verify-gil-offsets-strict` runs
`verify-gil-chunk-offsets` with **`--strict`** and **`--min-overlap-rate 0.95`** (override
**`GIL_OFFSET_VERIFY_DIR`**). Use after acceptance or nightly jobs that produce
**`search/metadata.json`** + GI quotes, not as a substitute for pytest on every PR without
such a tree. Rationale: [GIL / KG / CIL cross-layer](GIL_KG_CIL_CROSS_LAYER.md).

**Single overview:** [GIL / KG / CIL cross-layer guide](GIL_KG_CIL_CROSS_LAYER.md).

**Nightly** (`.github/workflows/nightly.yml`): **`nightly-viewer-unit`** and
**`nightly-viewer-e2e`** run the same Vitest / Playwright commands on every scheduled or
`workflow_dispatch` run (no path filters). Vitest sits in the post–lint+build segment with
**`nightly-test-unit`**; Playwright runs after **`nightly-test-e2e`** completes successfully.

#### Writing and extending tests

- **Vitest specs:** `web/gi-kg-viewer/src/utils/*.test.ts` — co-located with source.
  Add a `.test.ts` file next to any new utility. Config: `vite.config.ts` `test` block.
- **Playwright specs:** `web/gi-kg-viewer/e2e/*.spec.ts` (e.g. offline graph, search mocks,
  dashboard, theme).
- **Playwright config:** `web/gi-kg-viewer/playwright.config.ts` (`testDir: ./e2e`,
  **Desktop Firefox**, `baseURL` / `webServer` on **5174**).
- **Shared helpers:** `e2e/fixtures.ts`, `e2e/helpers.ts`.

More detail: [E2E Testing Guide — Browser E2E (Playwright)](E2E_TESTING_GUIDE.md#browser-e2e-playwright),
[web/gi-kg-viewer/README.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md)
(section **Browser E2E (M7)**).

## Fast Variants (Critical Path Only)

```bash
make test-fast              # Unit + critical path integration + e2e
make test-integration-fast  # Critical path integration tests
make test-e2e-fast          # Critical path E2E tests
```

### Sequential (For Debugging)

For debugging test failures, run tests sequentially using pytest directly:

```bash

# Run all tests sequentially

pytest tests/ -n 0

# Run unit tests sequentially

pytest tests/unit/ -n 0

# Run integration tests sequentially

pytest tests/integration/ -n 0

# Run E2E tests sequentially

pytest tests/e2e/ -n 0
```

## Specific Tests

```bash

# Run specific test file

pytest tests/unit/podcast_scraper/test_config.py -v

# Run with marker

pytest tests/integration/ -m "integration and critical_path" -v

# Run with coverage

pytest tests/unit/ --cov=podcast_scraper --cov-report=term-missing
```

## Test Markers

| Marker | Purpose |
| -------- | --------- |
| `@pytest.mark.unit` | Unit tests |
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.e2e` | End-to-end tests |
| `@pytest.mark.critical_path` | Critical path tests (run in fast suite) |
| `@pytest.mark.nightly` | Nightly-only tests (excluded from regular CI) |
| `@pytest.mark.flaky` | May fail intermittently (gets reruns) |
| `@pytest.mark.serial` | Must run sequentially (rarely needed) |
| `@pytest.mark.ml_models` | Requires ML dependencies |
| `@pytest.mark.slow` | Slow-running tests |
| `@pytest.mark.network` | Hits external network |
| `@pytest.mark.llm` | Uses LLM APIs (excluded from nightly to avoid costs) |
| `@pytest.mark.openai` | Uses OpenAI specifically (subset of `llm`) |

## Network Isolation

Tests use network isolation appropriate to their layer:

- **Unit tests:** Full socket blocking (`--disable-socket --allow-hosts=127.0.0.1,localhost`)
- **Integration/E2E:** Host allowlist only (`--allow-hosts=127.0.0.1,localhost`)

## Parallel Execution

Tests run in parallel by default using `pytest-xdist`:

All tests run in parallel by default with memory-aware worker calculation. The
`@pytest.mark.serial` marker is registered but currently unused.

The Makefile automatically calculates the optimal number of workers based on:

- Available system memory
- CPU core count
- Test type (unit/integration/e2e have different memory requirements)
- Platform (more conservative on macOS)

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#memory-issues-with-ml-models) for details on memory-aware worker calculation.

> **Note:** The `@pytest.mark.serial` marker is rarely needed now. Global state cleanup
> fixtures in `conftest.py` reset shared state between tests, allowing most tests to run
> in parallel safely. Only use `serial` for tests with genuine resource conflicts.

### Memory Cleanup Best Practices

**Automatic Cleanup:**

The test suite includes an automatic cleanup fixture (`cleanup_ml_resources_after_test`) that:

- Limits PyTorch thread pools to prevent excessive thread spawning
- Cleans up the global preloaded ML provider
- **Finds and cleans up ALL SummaryModel and provider instances** created during tests (Issue #351)
- Forces garbage collection after integration/E2E tests

**Explicit Cleanup (Recommended):**

While automatic cleanup handles most cases, explicit cleanup is recommended for clarity and immediate memory release:

```python
from tests.conftest import cleanup_model, cleanup_provider

def test_something():
    # Create model directly
    model = summarizer.SummaryModel(...)
    try:
        # test code
    finally:
        cleanup_model(model)  # Explicit cleanup

def test_with_provider():
    # Create provider directly
    provider = create_summarization_provider(cfg)
    try:
        # test code
    finally:
        cleanup_provider(provider)  # Explicit cleanup
```

**Why Explicit Cleanup?**

1. **Immediate memory release** - Models are unloaded as soon as the test completes
2. **Clarity** - Makes it obvious that cleanup is happening
3. **Defensive** - Works even if automatic cleanup has issues
4. **Best practice** - Matches the pattern of resource management (try/finally)

**Helper Functions:**

- `cleanup_model(model)` - Unloads a SummaryModel instance
- `cleanup_provider(provider)` - Cleans up a provider instance (MLProvider, etc.)

Both functions are idempotent (safe to call multiple times) and handle None gracefully.

### Warning: `-s` Flag and Parallel Execution

**Do not use `-s` (no capture) with parallel tests** — it causes hangs due to tqdm
progress bars competing for terminal access.

```bash

# DON'T DO THIS (hangs)

pytest tests/e2e/ -s -n auto

# DO THIS INSTEAD

pytest tests/e2e/ -v -n auto     # Use -v for verbose output
pytest tests/e2e/ -s -n 0        # Or disable parallelism
make test-e2e-sequential         # Or use sequential target
```

See [Issue #176](https://github.com/chipi/podcast_scraper/issues/176) for details.

## Flaky Test Reruns

Integration and E2E tests use reruns:

```bash
pytest --reruns 3 --reruns-delay 1
```

## Flaky Test Markers

Some tests are marked with `@pytest.mark.flaky` to indicate they may fail intermittently
due to inherent non-determinism. These tests get automatic reruns.

### Why Some Tests Are Flaky

| Category | Tests | Root Cause |
| -------- | ----- | ---------- |
| **Whisper Transcription** | 4 | ML inference variability - audio transcription has natural variation |
| **Full Pipeline + Whisper** | 2 | Whisper timing + audio file I/O |
| **OpenAI Mock Integration** | 2 | Mock response parsing timing |
| **Full Pipeline + OpenAI** | 7 | Complex multi-component timing |

### Current Flaky Test Count: 15

| File | Count | Category |
| ---- | ----- | -------- |
| `test_basic_e2e.py` | 7 | Full pipeline with OpenAI mocks |
| `test_whisper_e2e.py` | 4 | Whisper inference variability |
| `test_full_pipeline_e2e.py` | 2 | Whisper transcription |
| `test_openai_provider_e2e.py` | 2 | OpenAI mock responses |

### Tests That Are NOT Flaky

The following categories are now **stable** and don't need flaky markers:

- **Episode selection (GitHub #521)** - `tests/e2e/test_episode_selection_e2e.py` (mock feed `podcast1_episode_selection`, Path 1 transcripts; one test is `critical_path` for fast E2E)
- **Transformers/spaCy model loading** - Uses offline mode (`HF_HUB_OFFLINE=1`)
- **ML model tests** - Explicit `summary_reduce_model` prevents cache misses
- **HTTP integration tests** (`tests/integration/rss/test_http_integration.py`, marker `integration_http`) — Local `http.server` only; autouse fixture resets `configure_http_policy` / `configure_downloader` and closes downloader sessions so urllib3 retry defaults do not stall 5xx tests. See [Integration Testing Guide — Real HTTP client integration](INTEGRATION_TESTING_GUIDE.md#real-http-client-integration-local-server).
- **Parallel execution** - Global state cleanup prevents race conditions

### Reducing Flakiness

If you encounter a flaky test, check these common causes:

1. **Network access** - Should be blocked via pytest-socket
2. **Model cache** - Run `make preload-ml-models` first
3. **Global state** - Ensure cleanup fixtures reset shared state
4. **Progress bars** - `TQDM_DISABLE=1` is set automatically in tests

See [Issue #177](https://github.com/chipi/podcast_scraper/issues/177) for investigation details.

## E2E Test Modes

Set via `E2E_TEST_MODE` environment variable:

| Mode | Episodes | Use Case |
| ------ | ---------- | ---------- |
| `fast` | 1 per test | Quick feedback |
| `multi_episode` | 5 per test | Full validation |
| `data_quality` | Multiple | Nightly only |

## ML Model Preloading

Tests require models to be pre-cached:

```bash
make preload-ml-models
```

See [E2E Testing Guide](E2E_TESTING_GUIDE.md) for model defaults.

## E2E Acceptance Tests

Acceptance tests allow you to run multiple configuration files sequentially, collect structured data (logs, outputs, timing, resource usage), and compare results against baselines. This is useful for:

- Running the same configs across different code versions to detect regressions
- Testing multiple configs with different RSS feeds or settings
- Validating system acceptance of different provider/model configurations
- Comparing performance metrics across runs

### Setting up acceptance configs

Optional full-pipeline YAML presets may live under **`config/acceptance/`** beside the tracked matrix. The repo tracks **`config/acceptance/README.md`**, **`config/acceptance/FAST_CONFIG.yaml`**, and **`config/acceptance/fragments/*.yaml`**.

1. **Create the folder:** `mkdir -p config/acceptance` (at project root).
2. **Copy example configs:** Use `config/examples/config.example.yaml` (or any example) as a template:
   `cp config/examples/config.example.yaml config/acceptance/config.my.myshow.yaml` (or a name that fits your feeds).
3. **Adjust for your definition of acceptance:** Edit the copied file(s)—RSS feed URLs, providers, model names, output paths, etc.—so they match what you consider “acceptance” for your use case. You can add multiple configs (e.g. one per show or per provider) and run them all with a pattern like `config/acceptance/*.yaml`.

**Multi-feed (GitHub #440):** Use **`feeds:`** / **`rss_urls:`** in operator YAML **or** **`--feeds-spec`** with a feeds document (RFC-077 shape; see **`config/examples/feeds.spec.example.yaml`**). With **`USE_FIXTURES=1`**, the acceptance runner replaces each external feed URL with a distinct local E2E fixture feed so the run stays offline.

**Append / resume (GitHub #444):** Copy that preset, set **`append: true`**, and re-run (stable **`run_append_*`** per feed). Pytest coverage: `tests/e2e/test_append_resume_e2e.py` (two CLI invocations, stable `run_append_*`, `index.json` 1.1.0). See [CONFIGURATION.md — Append / resume](../api/CONFIGURATION.md#append-resume-github-444).

**Episode selection (GitHub #521):** Pytest E2E regression for `--episode-order`, `--since` / `--until`, `--episode-offset`, and config overrides lives in **`tests/e2e/test_episode_selection_e2e.py`** (mock server feed **`podcast1_episode_selection`**, fixture **`tests/fixtures/rss/p01_episode_selection.xml`**). One test is marked **`critical_path`** so it runs under **`make test-e2e-fast`** / the E2E slice of **`make test-ci-fast`**. Integration coverage includes **`tests/integration/workflow/test_workflow_stages_integration.py`** (`prepare_episodes_from_feed`) and **`tests/integration/infrastructure/test_e2e_server.py`** (`TestE2EEpisodeSelectionFeed`). See [CONFIGURATION.md — Episode selection](../api/CONFIGURATION.md#episode-selection-github-521) and [E2E Testing Guide — E2E Feeds (RSS)](E2E_TESTING_GUIDE.md#e2e-feeds-rss).

**Corpus resolution + CLI (post–#505 / inspect hardening):** Unit tests include **`tests/unit/podcast_scraper/utils/test_corpus_episode_paths.py`** (YAML metadata, rglob fallback, parent search hint), **`tests/unit/podcast_scraper/utils/test_corpus_lock.py`**, **`TestKgSubcommandMultiFeed`** and **`TestGiSubcommand`** multi-feed **`gi inspect` / `kg inspect`** paths in **`tests/unit/podcast_scraper/test_cli.py`**, and viewer **`web/gi-kg-viewer/src/stores/shell.hints.test.ts`** for **`GET /api/artifacts`** `hints`. Playwright **`web/gi-kg-viewer/e2e/corpus-hints.spec.ts`** mirrors the hint banner (requires **`npx playwright install firefox`** locally / in CI).

**Full fast matrix with fixtures (smoke all acceptance presets offline):** `make test-acceptance-fixtures-fast` materializes each enabled row from **`config/acceptance/FAST_CONFIG.yaml`** into **`sessions/session_*/materialized/{id}.yaml`** and runs those configs. Uses **`USE_FIXTURES=1`**, disables auto analyze/benchmark, and CI sets a **1500s** per-config timeout (`TIMEOUT=...` to override locally). Same target runs on **main / release** pushes in CI (`test-acceptance-fixtures` job in `python-app.yml`).

Optional: use **`config/playground/`** for ad-hoc or one-off configs; run them with e.g. `make test-acceptance CONFIGS="config/playground/config.my.*.yaml"`.

### Running Acceptance Tests

```bash
# Run a single config file
make test-acceptance CONFIGS="config/examples/config.example.yaml"

# Run multiple configs (using glob patterns)
make test-acceptance CONFIGS="config/acceptance/*.yaml"

# Fast matrix from FAST_CONFIG.yaml (materialized YAMLs per row) + fixtures
make test-acceptance FROM_FAST_STEMS=1 USE_FIXTURES=1

# Same as above with CI-style defaults (no auto analyze/benchmark; default TIMEOUT=900; CI uses 1500)
make test-acceptance-fixtures-fast

# Save current runs as a baseline for future comparison
make test-acceptance CONFIGS="config/examples/config.example.yaml" SAVE_AS_BASELINE=baseline_v1

# Compare against an existing baseline
make test-acceptance CONFIGS="config/examples/config.example.yaml" COMPARE_BASELINE=baseline_v1

# Use fixture feeds (mock data) instead of real RSS feeds
make test-acceptance CONFIGS="config/examples/config.example.yaml" USE_FIXTURES=1

# Disable real-time log streaming (only save to files)
make test-acceptance CONFIGS="config/examples/config.example.yaml" NO_SHOW_LOGS=1

# Disable automatic analysis and benchmark reports
make test-acceptance CONFIGS="config/examples/config.example.yaml" NO_AUTO_ANALYZE=1 NO_AUTO_BENCHMARK=1
```

### Understanding Sessions vs Runs

**Session** = One execution of the acceptance test tool

- Triggered by a single command invocation
- Can process multiple config files sequentially
- Has a unique `session_id` (timestamp-based)
- Contains a summary of all runs in that session

**Run** = One execution of a single config file within a session

- Each config file you pass creates one run
- Has its own `run_id`, timing, exit code, logs, and outputs
- Runs execute sequentially within the session

**Example:**

If you run:

```bash
make test-acceptance CONFIGS="config1.yaml config2.yaml config3.yaml"
```

You get:

- **1 Session** (with `session_id = 20260208_101601`)
  - **3 Runs** (one for each config file)
    - `run_20260208_101601_123` (config1.yaml)
    - `run_20260208_101601_456` (config2.yaml)
    - `run_20260208_101601_789` (config3.yaml)

### Output Structure

Results are saved to `.test_outputs/acceptance/` by default:

```text
.test_outputs/acceptance/
├── sessions/
│   └── session_20260208_101601/          ← ONE SESSION
│       ├── session.json                   ← Summary of all runs
│       └── runs/
│           ├── run_20260208_101601_123/  ← RUN #1 (config1)
│           │   ├── config.original.yaml  ← Original config for this run
│           │   ├── config.yaml           ← Modified config used for execution
│           │   ├── run_data.json
│           │   ├── stdout.log
│           │   ├── stderr.log
│           │   └── ... (service outputs)
│           ├── run_20260208_101601_456/  ← RUN #2 (config2)
│           │   ├── config.original.yaml  ← Original config for this run
│           │   ├── config.yaml
│           │   └── ...
│           └── run_20260208_101601_789/  ← RUN #3 (config3)
│               ├── config.original.yaml  ← Original config for this run
│               ├── config.yaml
│               └── ...
└── baselines/
    └── baseline_v1/                       ← Saved baselines
        ├── baseline.json
        └── run_20260208_101601_123/      ← Copied run data
```

### Analyzing Results

Use the analysis script to generate reports:

```bash
# Basic analysis
make analyze-acceptance SESSION_ID=20260208_101601

# Comprehensive analysis with baseline comparison
make analyze-acceptance SESSION_ID=20260208_101601 MODE=comprehensive COMPARE_BASELINE=baseline_v1

# Or use the script directly
python scripts/acceptance/analyze_bulk_runs.py \
    --session-id 20260208_101601 \
    --output-dir .test_outputs/acceptance \
    --mode comprehensive \
    --compare-baseline baseline_v1
```

### Performance Benchmarking

Generate performance benchmarking reports that group runs by provider/model configuration:

```bash
# Generate benchmark report
make benchmark-acceptance SESSION_ID=20260208_101601

# Generate benchmark report with baseline comparison
make benchmark-acceptance SESSION_ID=20260208_101601 COMPARE_BASELINE=baseline_v1

# Or use the script directly
python scripts/acceptance/generate_performance_benchmark.py \
    --session-id 20260208_101601 \
    --output-dir .test_outputs/acceptance \
    --compare-baseline baseline_v1
```

The benchmark report includes:

- **Summary table** comparing all provider/model configurations
- **Performance metrics** per configuration (time per episode, throughput, memory)
- **Detailed analysis** for each configuration
- **Performance comparison** (fastest vs. slowest, memory usage)
- **Baseline comparison** (if `--compare-baseline` is provided):
  - Performance changes vs. baseline (time, throughput, memory)
  - Regression detection (20% slower, 100MB more memory)
  - Improvement detection (10% faster, 50MB less memory)
  - Detailed per-configuration comparison

**Baseline Comparison Features:**

- Compares provider/model configurations between current run and baseline
- Detects regressions (performance degradation)
- Detects improvements (performance gains)
- Shows percentage changes for all metrics
- Groups comparisons by provider/model (not just config name)

Reports are generated in both Markdown and JSON formats for easy review and programmatic analysis.

## Test Organization

**Unit tests** mirror the source tree (find the test for any source file mechanically).
**Integration tests** are organized by domain subsystem (providers, workflow, GI/KG, etc.).
**E2E tests** are flat by user scenario.

```text
tests/
├── unit/                    # Unit tests (fast, isolated)
│   ├── conftest.py          # Network/filesystem isolation
│   └── podcast_scraper/     # Per-module tests — mirrors src/ tree
├── integration/             # Integration tests — by domain subsystem
│   ├── conftest.py          # Shared fixtures
│   ├── providers/           # Provider factories, protocols, per-provider
│   │   ├── llm/            # LLM provider integration
│   │   ├── ml/             # ML model loading, embedding, QA, NLI
│   │   └── ollama/         # Ollama model-specific tests
│   ├── workflow/            # Orchestration, stages, metadata, resume
│   ├── gi/                  # GI/KG artifacts, evidence stack
│   ├── server/              # FastAPI app, viewer API
│   ├── search/              # FAISS indexing, corpus search
│   ├── rss/                 # RSS parsing, HTTP fetching
│   ├── eval/                # Evaluation framework
│   ├── infrastructure/      # Fixture mapping, infra
│   ├── tools/               # CLI tools
│   └── test_*.py            # Cross-cutting (filesystem, retry, cache)
├── e2e/                     # E2E tests — by user scenario
│   ├── fixtures/            # E2E server, HTTP server
│   └── test_*.py            # Complete workflow tests
└── conftest.py              # Shared fixtures, ML cleanup

web/gi-kg-viewer/            # Browser UI E2E (Playwright — not pytest)
├── e2e/                     # *.spec.ts
├── e2e/fixtures.ts          # Shared test fixtures
├── playwright.config.ts     # webServer (Vite :5174), Firefox
└── package.json             # test:e2e and other frontend scripts
```

See [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md#directory-organization) for
the domain-based layout rationale and per-folder contents.

## Coverage Thresholds

Per-tier thresholds enforced in CI (prevents regression):

| Tier | Threshold | Current |
| ---- | --------- | ------- |
| **Unit** | 70% | ~74% |
| **Integration** | 40% | ~42% |
| **E2E** | 40% | Full `podcast_scraper` tree in denominator; add pytest E2E until this gate passes |
| **Combined** | 70% | ~71%+ |

### Pytest E2E coverage (full package, no subtree omit)

**pytest E2E** jobs use the same **`pyproject.toml`** **`[tool.coverage.run]`** settings as unit and
integration (**`COVERAGE_THRESHOLD_E2E`** in the Makefile; **`--cov-fail-under`** in CI). There is
**no** separate `coverage-e2e.ini` that removes `server/`, `search/`, `gi/`, or other subtrees from
the fraction. The reported percentage is: lines hit by **`tests/e2e/`** divided by **all**
measurable lines under **`podcast_scraper`**.

**Why the E2E percentage is lower than unit/integration:** many modules (FastAPI app, FAISS
indexer, large GI helpers, eval harness) are **not** exercised on every pytest E2E run. That is
visible in the number: it is a **signal to add pytest E2E** (or subprocess workflows) for **key
capabilities**, not a reason to shrink the denominator.

**pytest E2E vs HTTP integration vs Playwright** (short):

- **pytest E2E** — Python workflows from CLI / pipeline / E2E server client; this is the tier this
  row’s threshold applies to.
- **HTTP integration** — FastAPI and routes via **TestClient** (and similar); fast boundary tests;
  does **not** replace maintainer pytest E2E obligations for capabilities that must be proven from
  the **real user entry point** you care about.
- **Playwright** — Browser UI only; **additive**; **not** an excuse to lower pytest E2E coverage or
  to skip pytest E2E for Python-only surfaces.

Authoritative narrative: [Testing Strategy — Pytest E2E vs HTTP integration vs browser E2E](../architecture/TESTING_STRATEGY.md#pytest-e2e-vs-http-integration-vs-browser-e2e-playwright).

**Note:** Local make targets now run with coverage:

```bash
make test-unit          # includes --cov
make test-integration   # includes --cov
make test-e2e           # includes --cov (E2E config)
```

## Test Count Targets

- **Unit tests:** 200+
- **Integration tests:** 50+
- **E2E tests:** 100+

## Layer-Specific Guides

For detailed implementation patterns:

- **[Unit Testing Guide](UNIT_TESTING_GUIDE.md)** - What to mock, isolation enforcement, test structure
- **[Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md)** - Mock vs real decisions, ML model usage
- **[E2E Testing Guide](E2E_TESTING_GUIDE.md)** - E2E server, OpenAI mocking, network guard

## What to Test

- **[Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)** - What to test based on the critical path,
  test prioritization, fast vs slow tests, `@pytest.mark.critical_path` marker

## Domain-Specific Testing

- **[Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md#testing-your-provider)** - Provider testing
  across all tiers (unit, integration, E2E), E2E server mock endpoints, testing checklist

## References

- [Testing Strategy](../architecture/TESTING_STRATEGY.md) - Overall testing philosophy (incl. Playwright layer)
- [ADR-066: Playwright for UI E2E](../adr/ADR-066-playwright-for-ui-e2e-testing.md) - Why Playwright for viewer v2
- [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md) - Prioritization
- [CI/CD Documentation](../ci/index.md) - GitHub Actions workflows
- [Architecture](../architecture/ARCHITECTURE.md) - Testing Notes section
