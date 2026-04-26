# Testing Strategy

> **Document Structure:**
>
> - **This document** - High-level strategy, test pyramid, decision criteria
> - **[Testing Guide](../guides/TESTING_GUIDE.md)** - Quick reference, test execution commands
> - **[Unit Testing Guide](../guides/UNIT_TESTING_GUIDE.md)** - Unit test mocking patterns and isolation
> - **[Integration Testing Guide](../guides/INTEGRATION_TESTING_GUIDE.md)** - Integration test mocking guidelines
> - **[E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md)** - pytest E2E (server, ML, mocks); **browser E2E** (Playwright / viewer v2)
> - **[Critical Path Testing Guide](../guides/CRITICAL_PATH_TESTING_GUIDE.md)** - What to test and prioritization

## Overview

This document defines the testing strategy for the
podcast scraper codebase. It establishes the test
pyramid approach, decision criteria for choosing test
types, and high-level testing patterns.

The system supports **9 providers** (1 local ML + 1 hybrid ML + 7 LLM: OpenAI,
Gemini, Anthropic, Mistral, DeepSeek, Grok, Ollama)
with per-provider unit, integration, and E2E tests.
All LLM providers use versioned Jinja2 prompt
templates managed by `PromptStore` (RFC-017).

**GIL**: The Grounded Insight Layer (GIL, PRD-017) has testing for insight/quote extraction,
grounding contract validation, and `gi.json` schema compliance. See **GIL Testing (Implemented)** below.

**CIL / bridge / search lift (RFC-072, #527–528):** Unit tests cover **`bridge.json`**
building, **`cil_queries`**, **transcript chunk lift**, and **GIL–FAISS offset**
verification helpers; FastAPI integration tests cover **CIL routes** and extended
**search** responses. The **strict offset gate** is a **Makefile** target
(`make verify-gil-offsets-strict`) run against a **concrete indexed corpus**, not part of
default `make ci-fast` — see [GIL / KG / CIL cross-layer guide](../guides/GIL_KG_CIL_CROSS_LAYER.md)
and [Testing Guide — GIL, KG, CIL, and semantic search](../guides/TESTING_GUIDE.md#gil-kg-cil-and-semantic-search-validation).

For detailed implementation guides per test layer,
see the layer-specific guides linked above.

## Problem Statement

Testing requirements and strategies were previously scattered across individual RFCs, making it difficult to:

- Understand the overall testing approach
- Ensure consistent testing patterns across modules
- Plan new test infrastructure
- Onboard new contributors to testing practices
- Track testing coverage and requirements

This unified testing strategy document provides a single source of truth for all testing decisions and requirements.

## Test Pyramid

The testing strategy follows a three-tier pyramid:

```text
        /\
       /E2E\          ← Few, realistic end-to-end tests
      /------\
     /Integration\    ← Moderate, focused integration tests
    /------------\
   /    Unit      \   ← Many, fast unit tests
  /----------------\
```

| Layer | Scope | Entry Point | HTTP | Fixtures | ML/AI Models |
| ----- | ----- | ----------- | ---- | -------- | ------------ |
| **Unit** | Individual functions/modules | Function/class level | Mocked | Mocked | Mocked |
| **Integration** | Component interactions | Component level | Local test server (or mocked) | Test fixtures | Mocked |
| **E2E** | Complete user workflows | User level (CLI/API) | Real HTTP client (local server) | Real data files | Real (the only place) |
| **Frontend unit** | TypeScript utility logic (parsing, merge, metrics) | Node (Vitest) | N/A | Vitest + inline fixtures | N/A |
| **Browser UI E2E** | Vue viewer in a real browser | Browser (Playwright) | Vite dev server (local) | Playwright + fixture JSON | N/A (no Python ML in UI tests) |

**Frontend unit** tests use **Vitest** (`make test-ui`) to validate pure TS logic in
`web/gi-kg-viewer/src/utils/*.test.ts` — fast (~150 ms), no browser.
**Browser UI E2E** is an **additive** layer: it uses **Playwright** under `web/gi-kg-viewer/`, not pytest, and is **not** counted in the ~230 pytest E2E tests. See **[Browser UI E2E (Playwright)](#browser-ui-e2e-playwright)** below and [ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md).

### Volume and default bias

**Goal:** Most tests are **unit** tests (~80%); **integration** (~14%) and **E2E** (~6%) are complementary layers—not copies of the same scenarios.

**Current distribution (~3,770 tests total):** ~3,000 unit | ~530 integration | ~230 E2E.

| Layer | Expectation |
| ----- | ----------- |
| **Unit** | **Default for new work.** Cover branches, errors, config, parsing, and provider behavior with mocks. This layer should have the **highest test count** (~80%) and run on every PR (fast). |
| **Integration** | **Cross-boundary validation.** Integration tests validate cross-module boundaries and provider interactions (factories, stages, HTTP client against a local server). Avoid re-testing every unit case again here. |
| **E2E** | **Non-redundant with integration.** E2E tests are kept non-redundant with integration tests—each covers a distinct full workflow (CLI / `run_pipeline` / critical paths) that proves the stack end-to-end. Do **not** add E2E for every new flag or branch—use unit tests for those. |

**Anti-pattern:** Asserting the same behavior three times (unit + integration + E2E) without a distinct guarantee at each layer. Prefer **one strong unit test** plus integration or E2E only where the layer adds real value (real I/O, multi-component flow, or user entry point).

### Decision Questions

1. **Am I testing a complete user workflow?** (CLI command, library API call, service API call)
   - **YES** → E2E Test (pytest) — unless the workflow is **only** the **Vue viewer in a browser** (graph, search panel, keyboard, theme), in which case → **Playwright** (`make test-ui-e2e`; see [Browser UI E2E (Playwright)](#browser-ui-e2e-playwright))
   - **NO** → Continue to question 2

2. **Am I testing how multiple components work together?** (RSS parser → Episode → Provider → File)
   - **YES** → Integration Test
   - **NO** → Continue to question 3

3. **Am I testing a single function/module in isolation?**
   - **YES** → Unit Test
   - **NO** → Review test scope and purpose

### Common Patterns

- **Component workflow** (RSS → Episode → Provider) → Integration Test
- **Complete CLI command** (`podcast-scraper <url>`) → E2E Test
- **Library API call** (`run_pipeline(config)`) → E2E Test
- **Error handling in pipeline** → Integration Test (if focused) or E2E Test (if complete workflow)
- **HTTP client behavior** → Integration Test (if isolated) or E2E Test (if in workflow)

### Multi-feed corpus, unified index, and corpus artifacts (#440 / #505 / #506)

- **Layout:** Two or more feeds under `<corpus_parent>/feeds/<stable_id>/…`; optional hybrid
  `<corpus_parent>/metadata/` alongside `feeds/` (indexed with per-feed metadata).
- **Unit tests (primary):**
  - `tests/unit/podcast_scraper/search/test_corpus_scope.py` — `discover_metadata_files` hybrid merge.
  - `tests/unit/podcast_scraper/search/test_indexer.py` — nested `feeds/…/metadata` and **hybrid**
    parent + feeds indexing; composite fingerprint keys (`index_fingerprint_scope_key`).
  - `tests/unit/podcast_scraper/workflow/test_corpus_operations.py` — manifest/summary JSON,
    `finished_at`, `finalize_multi_feed_batch` return value.
  - `tests/unit/podcast_scraper/test_service.py` — `ServiceResult.multi_feed_summary` (success,
    partial failure, **GitHub #559** ``multi_feed_strict`` when failures are soft-only).
  - `tests/unit/podcast_scraper/test_cli.py` — `corpus-status` parse/smoke; multi-feed CLI with
    injected `run_pipeline_fn`; **`gi inspect` / `kg inspect`** multi-feed **`--feed-id`** and
    ambiguous **`episode_id`** error paths (`TestGiSubcommand`, `TestKgSubcommandMultiFeed`).
  - `tests/unit/podcast_scraper/utils/test_corpus_episode_paths.py` — metadata-driven GI/KG paths,
    **YAML** metadata, **rglob** fallback without `.metadata.*`, **`corpus_search_parent_hint`**.
  - `tests/unit/podcast_scraper/utils/test_corpus_lock.py` — advisory **`.podcast_scraper.lock`**
    (`PODCAST_SCRAPER_CORPUS_LOCK`).
  - `tests/unit/podcast_scraper/gi/test_load.py` / `tests/unit/podcast_scraper/kg/test_kg_load.py` —
    **`find_*_by_episode_id`** across **`feeds/`** with **`feed_id`** disambiguation.
  - `web/gi-kg-viewer/src/stores/shell.hints.test.ts` (Vitest) — **`GET /api/artifacts`** `hints`;
    `web/gi-kg-viewer/e2e/corpus-hints.spec.ts` (Playwright; requires **`playwright install`**).
- **Integration tests:** `tests/integration/test_workflow_integration.py` — multi-feed CLI and
  `service.run` happy path (manifest + `corpus_run_summary.json` on disk); **partial feed failure**
  still writes both artifacts with `overall_ok: false` (CLI + service).
- **E2E:** `tests/e2e/test_service_api_e2e.py` — YAML `feeds:` + `multi_feed_summary`;
  `tests/e2e/test_basic_e2e.py` / `test_cli_subprocess_e2e.py` — multi-feed smoke where marked #440.
- **Multi-feed resilience E2E (#559 / #560):** **`multi_feed_strict`** (default lenient) and CLI
  **`--multi-feed-strict`** / **`--no-multi-feed-strict`** are covered in
  **`tests/e2e/test_multi_feed_resilience_e2e.py`** (corpus batch **`corpus_run_summary.json`**,
  per-feed **`failure_kind`**, strict vs lenient exit codes, transient RSS, corpus lock,
  multi-episode transcript skip + per-feed **`run.json`** metrics). HTTP/transcript retry and
  **`failure_summary`** in **`run.json`** where the pipeline records episode failures live in
  **`tests/e2e/test_download_resilience_e2e.py`**. Narrative and fast vs **`multi_episode`** mode:
  [E2E_TESTING_GUIDE.md — Download resilience E2E](../guides/E2E_TESTING_GUIDE.md#download-resilience-e2e).
- **Config note (#559):** Deprecated YAML/JSON key **`multi_feed_soft_fail_exit_zero`** is mapped
  during **`Config.model_validate`** on a dict only. Call **`Config(multi_feed_strict=...)`** in
  code; the legacy name is not a constructor keyword (model **`extra=forbid`**). No env alias.
- **Docs:** [CORPUS_MULTI_FEED_ARTIFACTS.md](../api/CORPUS_MULTI_FEED_ARTIFACTS.md), RFC-063 §5–§7,
  [CONFIGURATION.md](../api/CONFIGURATION.md#rss-and-multi-feed-corpus-github-440),
  [SEMANTIC_SEARCH_GUIDE.md](../guides/SEMANTIC_SEARCH_GUIDE.md).

### Unit Tests

- **Purpose**: Test individual functions/modules in isolation
- **Speed**: Fast (< 100ms each)
- **Scope**: Single module, mocked dependencies
- **Coverage**: High (target: ≥70% code coverage)
- **Examples**: Config validation, filename sanitization, URL normalization

### Integration Tests

- **Purpose**: Test interactions between multiple modules/components (component interactions, data flow)
- **Speed**: Moderate (< 5s each for fast tests)
- **Scope**: Multiple modules working together, real internal implementations
- **Entry Point**: Component-level (functions, classes, not user-facing APIs)
- **I/O Policy**:
  - **Allowed**: Real filesystem I/O (temp directories), real component interactions
  - **Mocked**: External services (HTTP APIs, external APIs) - mocked for speed/reliability
  - **Mocked**: All ML/AI models and APIs (Whisper, spaCy, transformers, OpenAI, Gemini, Ollama, etc.) -- real ML/AI is E2E only
  - **Optional**: Local HTTP server for HTTP client testing in isolation
- **Coverage**: Critical paths and edge cases, component interactions
- **Examples**: Provider factory with mocked ML, RSS parser to Episode to Provider to File output, HTTP client with local test server
- **Key Distinction**: Tests how our components wire together, not complete user workflows. ML/AI boundaries are always mocked. `@pytest.mark.ml_models` must not appear on integration tests.

### End-to-End Tests

- **Purpose**: Test complete user workflows from entry
  point to final output (CLI commands, library API
  calls, service API calls)
- **Speed**: Slow (< 60s each, may be minutes for
  full workflows)
- **Scope**: Full pipeline from entry point to output,
  real HTTP client, real data files, real ML models
- **Entry Point**: User-level (CLI commands,
  `run_pipeline()`, `service.run()`)
- **I/O Policy**:
  - **Allowed**: Real HTTP client with local HTTP
    server (no external network), real filesystem I/O,
    real data files
  - **Real implementations**: Use actual HTTP
    clients (no mocking), real file operations, real
    model loading
  - **Real ML models**: Use real Whisper, spaCy,
    and Transformers models (NO mocks)
  - **Real data files**: RSS feeds, transcripts,
    audio files from `tests/fixtures/e2e_server/`
  - **No external network**: All HTTP calls go to
    local server (network guard prevents external
    calls)
  - **No mocks**: E2E tests use real
    implementations throughout (no Whisper mocks,
    no ML model mocks)
- **Provider E2E tests**: Dedicated E2E test files
  exist for each of the 9 providers:
  `test_ml_models_e2e.py`,
  `test_hybrid_ml_provider_e2e.py` (includes Tier 1 MAP+REDUCE smoke and
  **layered cleaning** / `pattern` + `cleaning_hybrid_after_pattern`, Issue #419),
  `test_openai_provider_e2e.py`,
  `test_gemini_provider_e2e.py`,
  `test_anthropic_provider_e2e.py`,
  `test_mistral_provider_e2e.py`,
  `test_deepseek_provider_e2e.py`,
  `test_grok_provider_e2e.py`,
  `test_ollama_provider_e2e.py`. LLM providers use
  mock API endpoints served by the E2E HTTP server.
- **Coverage**: Complete user workflows,
  production-like scenarios
- **Examples**: CLI command
  (`podcast-scraper <rss_url>`) → Full pipeline →
  Output files, Library API
  (`run_pipeline(config)`) → Full pipeline →
  Output files
- **Key Distinction**: Tests complete user workflows
  with real implementations. NO mocks allowed — tests
  the system as users would use it.

### Pipeline Validation (manual, `make pipeline-validate`)

Full pipeline validation across all providers — runs summary → GI → KG →
bridge for each provider on the held-out dataset and checks pass/fail at
each stage. **Not CI-automated** (API costs + Ollama latency). Run manually
from the laptop when confidence is needed (pre-release, after provider code
changes, after dependency upgrades).

```bash
make pipeline-validate                              # all providers (cloud + local)
make pipeline-validate PROVIDER=gemini MODEL=gemini-2.5-flash-lite  # single
make pipeline-validate PV_ARGS="--all-cloud"        # 6 cloud providers
make pipeline-validate PV_ARGS="--all-local"        # Core 5 Ollama (ADR-077)
make pipeline-validate PV_ARGS="--local-fast"       # skip 35B model
```

**Pass/fail criteria per stage:**

| Stage | Criteria |
| ----- | -------- |
| Summary | ≥5 episodes, ≥500 chars avg |
| GI | ≥8 insights/ep, ≥50% grounding, ≥3 topics/ep |
| KG | ≥5 topics/ep (≤50 chars), ≥2 entities/ep |
| Bridge | ≥90% topic merge rate |

**Providers covered:** 6 cloud APIs (OpenAI, Gemini, Anthropic, DeepSeek,
Mistral, Grok) + Core 5 Ollama (qwen3.5:9b, llama3.1:8b, mistral:7b,
gemma2:9b, qwen3.5:35b). See ADR-077 for model selection rationale.

**When to run:**

- Before releasing a new version
- After touching provider code, GI/KG pipeline, bridge, or config wiring
- After upgrading dependencies (sentence-transformers, google-genai, etc.)
- When adding a new provider

Results saved to `data/eval/runs/_pipeline_validation/`.

Similarly, **`make transcription-sweep`** compares local Whisper models on
quality (WER) and speed for transcription evaluation.

### Browser UI E2E (Playwright) {#browser-ui-e2e-playwright}

The **GI/KG viewer v2** (`web/gi-kg-viewer/`) adds **browser-level** regression tests using
**Playwright** (TypeScript). This is **not** a fourth pytest marker: specs live in
`web/gi-kg-viewer/e2e/` and run via **`make test-ui-e2e`** (or `npm run test:e2e` in that
directory).

| Aspect | Browser UI E2E (Playwright) | pytest E2E (`tests/e2e/`) |
| ------ | ----------------------------- | ------------------------- |
| **Runner** | `@playwright/test` | pytest |
| **Entry point** | Browser → Vue SPA | CLI, `run_pipeline()`, `service.run()` |
| **Typical proof** | Layout, Cytoscape graph shell, search UI, a11y/keyboard, offline fixtures | Full pipeline, providers, disk artifacts |
| **Local server** | Playwright `webServer` starts **Vite** on **127.0.0.1:5174** | `e2e_server` fixture (RSS/audio/transcripts + LLM mocks) |
| **In `make test`?** | No | Yes |
| **CI** | Job **`viewer-e2e`** (`.github/workflows/python-app.yml`); required for docs publish path on main | `test-e2e`, `test-e2e-fast`, etc. |

**Python API for the viewer** (`GET /api/search`, `/api/explore`, `/api/corpus/*`,
`POST /api/index/rebuild`, etc.) is validated at two pytest layers:

- **Unit:** `tests/unit/podcast_scraper/server/` -- pure logic tests (`test_corpus_catalog.py`,
  `test_index_staleness.py`, `test_pathutil.py`, etc.) that do **not** import FastAPI or any
  non-`[dev]` package. These run in CI `test-unit` which installs `.[dev]` only.
- **Integration:** `tests/integration/server/` -- wired `create_app` with real filesystem layouts,
  `TestClient`, and no mocking of route internals (e.g. `test_server_api.py`,
  `test_viewer_corpus_library.py`, `test_index_rebuild.py`, `test_index_rebuild_gate.py`,
  `test_viewer_index_stats.py`). CI integration jobs install `.[dev,ml,llm,server]`.

### Unit tests and optional extras (`pyproject`) {#unit-tests-and-optional-extras-pyproject}

**Contract for `tests/unit/`:**

1. **Never require any non-`[dev]` extra** (`[ml]`, `[llm]`, `[compare]`, `[server]`, etc.).
   Real FAISS, Whisper, spaCy, FastAPI, httpx, cloud SDKs, etc. belong in **integration** or
   **E2E** tests (with the workflow installing the right extras). Use **mocks**,
   **`sys.modules` stubs**, or **lazy imports** in unit tests.
2. **`[dev]`** is the **only baseline** for `tests/unit/`. CI `test-unit` installs
   `pip install -e .[dev]` only. Anything declared under `[project.optional-dependencies].dev`
   in `pyproject.toml` (and its transitive wheels) is allowed; nothing else.
3. **Do not use `pytest.importorskip()` in `tests/unit/`** to guard non-`[dev]` imports.
   It causes silent skips in CI, meaning the test never validates anything. If a test needs
   FastAPI, httpx, faiss, torch, etc., move it to `tests/integration/` where CI installs
   the full extras (`.[dev,ml,llm,server]`). Do not pull ML into unit tests -- keep
   FAISS / torch / spacy out of `tests/unit/` except via mocks.

**Verification (two complementary scripts, both in `make ci` / `make ci-fast`):**

- `scripts/tools/check_unit_test_imports.py` (`make check-unit-imports`) -- ensures key
  library modules import without the heavy ML stack at import time.
- `scripts/tools/check_test_policy.py` (`make check-test-policy`) -- enforces the
  3-tier ML/AI boundary policy across the test corpus: no `pytest.importorskip()` in
  unit tests (U1), no `*_AVAILABLE` skip guards in unit tests (U2), no
  `@pytest.mark.ml_models` in integration tests (I1), and no empty test files (G1).
  Run with `--fix-hint` for remediation suggestions.

The same `[dev]`-only assumption applies to the full `pytest tests/unit/` job on CI.

**Detail:** [Unit Testing Guide -- Pyproject extras](../guides/UNIT_TESTING_GUIDE.md#pyproject-extras-what-unit-tests-may-depend-on).
Use **Playwright** when the risk is **client rendering or interaction**, not when a pure JSON
contract change is enough.

**References:** [Polyglot repository guide](../guides/POLYGLOT_REPO_GUIDE.md) (root vs `web/gi-kg-viewer/`),
[Testing Guide — Browser E2E](../guides/TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2),
[E2E Testing Guide — Playwright](../guides/E2E_TESTING_GUIDE.md#browser-e2e-playwright),
[ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md),
[viewer README](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md).

## Decision Criteria

The decision questions above provide a quick way to determine test type. For critical path prioritization, see [Critical Path Testing Guide](../guides/CRITICAL_PATH_TESTING_GUIDE.md). For detailed implementation guidelines, see [Testing Guide](../guides/TESTING_GUIDE.md).

**Quick Reference:**

- **Unit Test**: Single function/module in isolation, all dependencies mocked, `[dev]` only
- **Integration Test**: Multiple components working together, real internal implementations, all ML/AI mocked
- **E2E Test**: Complete user workflow from entry point to output, real HTTP client, real data files, real ML models (the only place for real ML/AI)
- **Browser UI E2E (Playwright)**: Vue viewer behavior in a real browser; see [Browser UI E2E (Playwright)](#browser-ui-e2e-playwright) and [Testing Guide — Browser E2E](../guides/TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2)

**Critical Path Priority**: If your test covers the critical path (RSS → Parse → Download/Transcribe → NER → Summarization → Metadata → Files), prioritize it. See [Critical Path Testing Guide](../guides/CRITICAL_PATH_TESTING_GUIDE.md) for details.

## Test Categories

### 1. Unit Tests

#### Configuration & Validation (`config.py`)

- **RFC-008**: Validate coercion logic, error messages, alias handling
- **RFC-007**: Test argument parsing edge cases (invalid speaker counts, unknown config keys)
- **Test Cases**:
  - Type coercion (string → int, validation failures)
  - Config file loading (JSON/YAML, invalid formats)
  - Default value application
  - Alias resolution (`rss` vs `rss_url`)

#### Filesystem Operations (`filesystem.py`)

- **RFC-004**: Sanitization edge cases, output derivation logic
- **Test Cases**:
  - Filename sanitization (special characters, reserved names)
  - Output directory derivation and validation
  - Run suffix generation
  - Path normalization across platforms

#### RSS Parsing (`rss_parser.py`)

- **RFC-002**: Varied RSS shapes, namespace differences, missing attributes
- **Test Cases**:
  - Namespace handling (Podcasting 2.0, standard RSS)
  - Relative URL resolution
  - Missing optional fields
  - Malformed XML handling
  - Edge cases (uppercase tags, mixed namespaces)

#### Transcript Downloads (`downloader.py`, `episode_processor.py`)

- **RFC-003**: Extension derivation edge cases
- **Test Cases**:
  - URL normalization (encoding, special characters)
  - Extension inference (from URL, Content-Type, declared type)
  - HTTP retry logic (unit test with mocked responses)
  - Transcript type preference ordering

#### Whisper Integration (`providers/ml/whisper_utils.py`)

- **RFC-005**: Mock Whisper library, loading paths, error handling
- **RFC-006**: Screenplay formatting with synthetic segments
- **Test Cases**:
  - Model loading (success, missing dependency, invalid model)
  - Screenplay formatting (gap handling, speaker rotation, aggregation)
  - Language parameter propagation
  - Model selection logic (`.en` variants for English)

#### Progress Reporting (`progress.py`)

- **RFC-009**: Noop factory, `set_progress_factory` behavior
- **Test Cases**:
  - Factory registration and replacement
  - Progress update calls
  - Context manager behavior

#### Speaker Detection (`speaker_detection.py`) - RFC-010

- **RFC-010**: NER extraction scenarios, host/guest distinction
- **Test Cases**:
  - Title-only detection (`"Alice interviews Bob"`)
  - Description-rich detection (multiple guest names)
  - Feed-level host inference
  - CLI override precedence
  - spaCy missing/disabled scenarios
  - Name capping when too many detected

#### Summarization (`summarizer.py`) - RFC-012

- **RFC-012**: Local transformer model integration, summary generation with map-reduce strategy
- **Test Cases**:
  - Model selection (explicit, auto-detection for MPS/CUDA/CPU)
  - Model loading and initialization on different devices
  - **Model integration tests** (marked as `@pytest.mark.slow` and `@pytest.mark.integration`):
    - Verify all models in `DEFAULT_SUMMARY_MODELS` can be loaded when configured
    - Test each model individually: `bart-large`, `bart-small`, `long`, `long-fast`
    - Catch dependency issues (e.g., missing protobuf for PEGASUS models)
    - Verify model and tokenizer are properly initialized
    - Test model unloading after loading
  - **Map-reduce strategy**:

```text
    - Map phase: chunking (word-based and token-based), chunk summarization
    - Reduce phase decision logic: single abstractive (≤800 tokens), mini map-reduce (800-4000 tokens), extractive (>4000 tokens)
    - Mini map-reduce: re-chunking combined summaries into 3-5 sections (650 words each), second map phase (summarize each section), final abstractive reduce
    - Extractive fallback behavior for extremely large combined summaries
```

- Summary generation with various text lengths
- Key takeaways extraction
- Text chunking for long transcripts
- Safe summarization error handling (OOM, missing dependencies)
- Memory optimization (CUDA/MPS)
- Model unloading and cleanup
- Integration with metadata generation pipeline

### Provider System (RFC-013, RFC-029)

- **RFC-013/RFC-029**: Protocol-based provider
  architecture for transcription, speaker detection,
  and summarization with 9 unified providers.
- **Provider Coverage**: Each provider has a
  consistent test structure:

| Provider | Unit Tests | Integration | E2E |
| --- | --- | --- | --- |
| MLProvider | `test_ml_provider.py`, `_lifecycle.py` | `test_provider_real_models.py` | `test_ml_models_e2e.py` |
| HybridMLProvider | `test_hybrid_ml_provider.py`, `_lifecycle.py` | `test_hybrid_ml_providers.py` | `test_hybrid_ml_provider_e2e.py` (layered `pattern` path, Issue #419) |
| OpenAIProvider | `test_openai_provider.py`, `_factory.py`, `_lifecycle.py` | `test_openai_providers.py` | `test_openai_provider_e2e.py` |
| GeminiProvider | `test_gemini_provider.py`, `_factory.py`, `_lifecycle.py` | `test_gemini_providers.py` | `test_gemini_provider_e2e.py` |
| AnthropicProvider | `test_anthropic_provider.py`, `_factory.py`, `_lifecycle.py` | `test_anthropic_providers.py` | `test_anthropic_provider_e2e.py` |
| MistralProvider | `test_mistral_provider.py`, `_factory.py`, `_lifecycle.py` | `test_mistral_providers.py` | `test_mistral_provider_e2e.py` |
| DeepSeekProvider | `test_deepseek_provider.py`, `_factory.py`, `_lifecycle.py` | `test_deepseek_providers.py` | `test_deepseek_provider_e2e.py` |
| GrokProvider | `test_grok_provider.py`, `_factory.py`, `_lifecycle.py` | `test_grok_providers.py` | `test_grok_provider_e2e.py` |
| OllamaProvider | `test_ollama_provider.py`, `_factory.py`, `_lifecycle.py` | `test_ollama_providers.py` | `test_ollama_provider_e2e.py` |

- **Provider Capabilities**: `test_capabilities.py`
  (unit) and `test_capabilities_e2e.py` (E2E) test
  `ProviderCapabilities` — which providers support
  which features (JSON mode, tool calls, streaming).

- **Test Cases**:
  - **Unit Tests (Standalone Provider)**: Each
    provider tested directly with all dependencies
    mocked. Tests: creation, initialization, protocol
    method implementation, error handling, cleanup,
    configuration validation.
  - **Unit Tests (Factory)**: Factory tests verify
    correct provider instantiation per config,
    protocol compliance, error handling.
  - **Unit Tests (Lifecycle)**: Provider lifecycle
    tests verify initialization, cleanup,
    resource management.
  - **Integration Tests**: Real provider
    implementations with mocked external services.
    Tests: provider factory, protocol compliance,
    component interactions, provider switching, error
    handling in workflow context.
  - **E2E Tests**: Real providers in full pipeline.
    LLM providers use E2E server mock endpoints.
    MLProvider uses real ML models (Whisper, spaCy,
    Transformers). Tests: complete workflows,
    error scenarios (API failures, rate limits).

### Prompt Management (RFC-017)

- **Prompt Store**: `test_prompt_store.py` (unit)
  tests versioned Jinja2 prompt template management.
- **Test Cases**:
  - Template loading per provider/task/version
  - Jinja2 variable rendering
  - Missing template error handling
  - Template version selection
  - All 9 providers have prompt directories
    (`prompts/<provider>/ner/`, `summarization/`)
  - Template validation (syntax, required variables)

#### Service API (`service.py`)

- **Public API**: Service interface for daemon/non-interactive use
- **Test Cases**:
  - `ServiceResult` dataclass (success/failure states, attributes)
  - `service.run()` with valid Config (success path)
  - `service.run()` with logging configuration
  - `service.run()` exception handling (returns failed ServiceResult)
  - `service.run_from_config_file()` with JSON config
  - `service.run_from_config_file()` with YAML config
  - `service.run_from_config_file()` with missing file (returns failed ServiceResult)
  - `service.run_from_config_file()` with invalid config (returns failed ServiceResult)
  - `service.run_from_config_file()` with Path objects
  - `service.main()` CLI entry point (success/failure exit codes)
  - `service.main()` version flag handling
  - `service.main()` missing config argument handling
  - Service API importability via `__getattr__`
  - ServiceResult equality and string representation
  - Integration with public API (`Config`, `load_config_file`, `run_pipeline`)

#### Reproducibility & Operational Hardening (Issue #379)

- **Determinism**: Seed-based reproducibility for `torch`, `numpy`, and `transformers`. Test that seeds are set correctly and outputs are consistent across runs.
- **Run Tracking**: Test run manifest creation (system state capture), run summary generation (manifest + metrics), and episode index generation (episode status tracking).
- **Failure Handling**: Test `--fail-fast` and `--max-failures` flags, episode failure tracking in metrics, and exit code behavior.
- **Retry Policies**: Test exponential backoff retry for transient errors (network failures, model loading errors), retry counts and delays, and fallback to cache clearing.
- **Timeout Enforcement**: Test transcription and summarization timeout enforcement, timeout error handling, and timeout configuration.
- **Security**: Test path validation (directory traversal prevention), model allowlist validation, safetensors format preference, and `trust_remote_code=False` enforcement.
- **Structured Logging**: Test JSON log formatting, log aggregation compatibility, and structured log fields.
- **Diagnostics**: Test `doctor` command checks (Python version, ffmpeg, write permissions, model cache, network connectivity).

**For detailed unit test execution commands, test file descriptions, fixtures, requirements, and coverage, see [Unit Testing Guide](../guides/UNIT_TESTING_GUIDE.md).**

### 2. Integration Tests

#### CLI Integration (`cli.py` + `workflow.py`)

- **RFC-001**: Success, dry-run, concurrency edge cases, error handling
- **RFC-007**: CLI happy path, invalid args, config file precedence
- **Test Cases**:
  - CLI argument parsing → Config validation → pipeline execution
  - Config file loading and precedence (CLI > config file)
  - Dry-run mode (no disk writes)
  - Error handling and exit codes
  - Dependency injection hooks (`apply_log_level_fn`, `run_pipeline_fn`)

#### Workflow Orchestration (`workflow.py`)

- **RFC-001**: End-to-end coordination, concurrency, cleanup
- **Test Cases**:
  - RSS fetch → episode parsing → transcript download → file writing
  - Concurrent transcript downloads (ThreadPoolExecutor)
  - Whisper job queuing and sequential processing
  - Temp directory cleanup
  - Skip-existing and clean-output flags

#### Episode Processing (`episode_processor.py`)

- **RFC-003**: HTTP response simulation with various headers
- **RFC-004**: Directory management interactions
- **Test Cases**:
  - Transcript download with various formats (VTT, SRT, JSON)
  - Media download for Whisper transcription
  - File naming and storage
  - Error handling (network failures, missing files)

#### Config + CLI + Workflow

- **RFC-008**: CLI + config files → Config instantiation
- **Test Cases**:
  - Config file loading → validation → pipeline execution
  - Config override precedence
  - Invalid config error handling

#### Whisper + Screenplay Formatting

- **RFC-006**: Screenplay flags → formatting → file output
- **RFC-010**: Detected speaker names → screenplay formatting
- **Test Cases**:
  - Whisper transcription → screenplay formatting with detected names
  - Speaker name override precedence
  - Language-aware model selection

**For detailed integration test execution commands, test file descriptions, fixtures, requirements,
and coverage, see [Integration Testing Guide](../guides/INTEGRATION_TESTING_GUIDE.md).**

### 3. End-to-End Tests

#### E2E Test Coverage Goals

**Critical Path Priority**: The critical path must have E2E tests for all three entry points
(CLI, Library API, Service API). See
[Critical Path Testing Guide](../guides/CRITICAL_PATH_TESTING_GUIDE.md) for details.

**Every major user-facing entry point should have at least one E2E test:**

1. **CLI Commands** - Each main CLI command should have E2E tests
2. **Library API Endpoints** - Each public API function should have E2E tests
3. **Critical User Scenarios** - Important workflows should have E2E tests

**What Doesn't Need E2E Tests:**

- Not every CLI flag combination needs an E2E test
- Every possible configuration value (tested in integration/unit tests)
- Edge cases in specific components (tested in integration tests)

**Rule of Thumb**: E2E tests should cover "as a user, I want to..." scenarios, not every possible configuration combination.

**For detailed E2E test execution commands and implementation, see [E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md).**

### Pytest E2E vs HTTP integration vs browser E2E (Playwright)

This section is about **roles**, so future changes do not confuse “we have a test” with “we closed
the loop at the right layer.”

| Layer | What it is | What it proves |
| ----- | ---------- | -------------- |
| **pytest E2E** (`tests/e2e/`, `pytest`, `@pytest.mark.e2e`) | Full workflows from **user-facing Python entry points** (CLI subprocess, `run_pipeline`, service-style runs, E2E HTTP **client** against the **E2E fixture server**). | That the **Python product path** from entry to artifacts behaves end-to-end for the scenarios we care about. **Coverage** for this job uses the **full** `podcast_scraper` package (see `pyproject.toml` `[tool.coverage.run]`); there is **no** subtree `omit` file tuned to make the percentage look higher. |
| **HTTP integration** (`tests/integration/server/`, FastAPI **TestClient** or similar) | Request/response and route wiring **in the Python process**, often without a separate long-lived server or browser. | Server modules, auth/middleware shapes, and API contracts **efficiently**. It is **not** a substitute for **pytest E2E** when the requirement is “operator/user ran the real CLI or full pipeline and this capability fired.” |
| **Browser UI E2E (Playwright)** | Real browser against the **Vue** dev server; specs under `web/gi-kg-viewer/e2e/`. | **UI** behavior (a11y strings, graph shell, client-side merge). It does **not** execute the pytest suite and does **not** give a pass on **pytest E2E** obligations for Python-only capabilities. |

**Policy (Python focus):** **Playwright is not an excuse** to skip pytest E2E or to shrink pytest E2E
coverage expectations for Python packages such as `server/`, `search/`, or `gi/`. If a **key
capability** must be proven at the **pytest E2E** layer, add or extend **`tests/e2e/`** (and expect
the **full-tree** E2E coverage percentage to rise as those paths execute). The **combined**
coverage gate (~70%) and integration jobs still matter; they complement E2E rather than replacing
it for that bar.

See also [Testing Guide — coverage thresholds](../guides/TESTING_GUIDE.md#coverage-thresholds) and
[Browser UI E2E (Playwright)](#browser-ui-e2e-playwright).

### E2E Test Tiers (Code Quality vs Data Quality)

E2E tests are organized into two tiers to balance fast CI feedback with
comprehensive validation. (The previous "Tier 2: Data Quality" tier was
retired — its concerns moved to ``data/eval/`` with versioned silver
references, byte-for-byte reproducible materialized inputs, and pairwise
LLM judging. See ``data/eval/README.md``.)

| Tier | Purpose | Episodes | Models | When | Makefile Target |
| ---- | ------- | -------- | ------ | ---- | --------------- |
| **Tier 1: Fast** | Code quality, critical path | 1 | Test (tiny/base) | Every PR | `test-e2e-fast` |
| **Tier 2: Nightly Full** | Production validation | 15 (5×3) | Production (base, BART-large, LED-large) | Nightly | `test-nightly` |

**Tier 1: Fast E2E Tests** (`@pytest.mark.e2e` + `@pytest.mark.critical_path`)

- 1 episode per test for fast feedback
- Test models (Whisper tiny.en, BART-base, LED-base)
- Run on every PR and push to main
- Focus: Does the code work correctly?

**Tier 2: Nightly Full Suite** (`@pytest.mark.nightly`)

- 15 episodes across 5 podcasts (p01-p05)
- Production models (Whisper base, BART-large-cnn, LED-large-16384)
- Run in nightly builds only
- Focus: Production-quality validation with real models
- **Memory / CI:** `make test-nightly` runs sequentially (no pytest-xdist). Parallel
  execution was removed because exit-code mismatches triggered a fallback path that
  re-ran the full suite, doubling wall time from ~75 min to ~3 h. Sequential also
  avoids OOM on `ubuntu-latest` where two workers each loading Whisper + large HF
  models exceeded available RAM.

**Key Principle:** Code quality tests (Tier 1) run on every PR. Data quality and nightly tests
(Tiers 2-3) run only in nightly builds to avoid slowing down CI/CD feedback.

**LLM/OpenAI Exclusion:** Tests marked `@pytest.mark.llm` are excluded from nightly builds
(`-m "not llm"`) to avoid API costs. See issue #183.

## Test Infrastructure

### Test Framework

- **Primary**: pytest (with unittest compatibility)
- **Fixtures**: pytest fixtures for test setup and teardown
- **Markers**: pytest markers for test categorization

### Mocking Strategy

- **Unit Tests**: Mock all external dependencies (HTTP, ML models, file system, API clients).
  `[dev]` only -- no non-`[dev]` extras allowed.
- **Integration Tests**: Mock all external services (HTTP APIs, external APIs) and all ML/AI
  models and APIs (Whisper, spaCy, Transformers, OpenAI, Gemini, Ollama, etc.). Use real
  internal implementations (real providers, real Config, real workflow logic). Real ML/AI
  inference never happens at this layer.
- **E2E Tests**: Use real implementations (HTTP client, ML models, file system) with local test
  server. For API providers, use E2E server mock endpoints instead of direct API calls. ML models
  (Whisper, spaCy, Transformers) are REAL -- no mocks allowed. This is the only layer where
  `@pytest.mark.ml_models` is valid.

**Provider Testing Strategy (9 providers):**

- **Unit Tests (Standalone Provider)**: Each of the
  9 providers tested directly with all dependencies
  mocked (API clients, ML models). Dedicated test
  files: `test_<provider>_provider.py`,
  `_factory.py`, `_lifecycle.py`.
- **Unit Tests (Factory)**: Test factories create
  correct unified providers for each
  `<provider>_provider` config value. Verify protocol
  compliance, test factory error handling.
- **Integration Tests**: Per-provider integration
  files (`test_<provider>_providers.py`) use real
  provider implementations with mocked external
  services. All ML/AI models and APIs are mocked.
- **E2E Tests**: Per-provider E2E files
  (`test_<provider>_provider_e2e.py`) use E2E server
  mock endpoints for LLM providers, real ML models
  for MLProvider. This is the only layer with real ML.
- **Key Principle**: Always verify protocol
  compliance, not class names. All providers
  implement the same protocol interfaces.
- **Prompt Templates**: LLM providers load prompts
  via `PromptStore`. Unit tests mock `PromptStore`;
  integration/E2E tests use real templates.
- **Test Organization**: See
  `docs/wip/PROVIDER_TEST_STRATEGY.md` for detailed
  test organization and separation.

### Test Organization

The test suite is organized into three main **pytest** categories, plus the **frontend** tree.
Each layer uses a different organizational axis:

- **`tests/unit/`** — Mirrors the `src/` tree 1:1 (find the test for any source file mechanically)
- **`tests/integration/`** — Organized by **domain subsystem** (providers, workflow, GI/KG, server, search, rss, etc.); cross-cutting tests stay in root. See [Integration Testing Guide — Directory Organization](../guides/INTEGRATION_TESTING_GUIDE.md#directory-organization)
- **`tests/e2e/`** — Flat by user scenario (CLI commands, service API, provider pipelines)
- **`web/gi-kg-viewer/src/utils/*.test.ts`** — Vitest unit tests for TS utility logic (`make test-ui`)
- **`web/gi-kg-viewer/e2e/`** — Playwright specs (`*.spec.ts`); **`web/gi-kg-viewer/playwright.config.ts`** — browser UI E2E (not collected by pytest)

### Test Markers

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (component interactions)
- `@pytest.mark.e2e` - End-to-end workflow tests
- `@pytest.mark.critical_path` - Critical path tests (run in fast suite)
- `@pytest.mark.serial` - Tests that must run sequentially (resource conflicts)
- `@pytest.mark.ml_models` - Requires ML dependencies (whisper, spacy, transformers) or uses real
  ML models

- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.network` - Tests that hit external network
- `@pytest.mark.multi_episode` - Multi-episode tests (nightly)
- `@pytest.mark.nightly` - Comprehensive nightly tests (15 episodes, production models)
- `@pytest.mark.llm` - Tests using LLM APIs (excluded
  from nightly to avoid costs)
- `@pytest.mark.openai` - Tests using OpenAI API
  specifically (subset of llm)
- `@pytest.mark.gil` - GIL extraction tests
- `@pytest.mark.grounding` - Grounding contract
  validation tests

**Execution Pattern**: Tests marked `serial` run first sequentially, then remaining tests run in
parallel with `-n auto`. All tests use network isolation via
`--disable-socket --allow-hosts=127.0.0.1,localhost`.

**For detailed test infrastructure information, see [Testing Guide](../guides/TESTING_GUIDE.md).**

## ML Quality Evaluation

Beyond functional testing, the project uses objective
metrics to evaluate the **quality** of ML-generated
outputs. These evaluations are performed using
dedicated scripts and a "golden" dataset.

### Evaluation Layers

| Layer | Focus | Metrics | Tools |
| :--- | :--- | :--- | :--- |
| **Cleaning** | Effective removal of ads/outro | Removal %, Brand detection | Automatic (via experiment runner) |
| **Summarization** | Accuracy and synthesis quality | ROUGE-1/2/L, Compression ratio | Automatic (via experiment runner) |
| **GIL** | Insight/quote extraction quality | Grounding rate, Quote verbatim %, Insight coverage | Per-provider comparison |

### Golden Datasets

Evaluation is performed against human-verified ground
truth data stored in `data/eval/`. This dataset is
versioned and frozen to provide a stable baseline for
comparison. See
[ADR-040](../adr/ADR-040-explicit-golden-dataset-versioning.md)
for details.

**GIL Golden Dataset**: A golden dataset for GIL
evaluation includes human-annotated insights,
quotes, and grounding links for a representative
set of episodes. This enables comparison across
extraction tiers (ML-only, Hybrid, Cloud LLM).

### Continuous Improvement

Quality evaluation is integrated into the
**[AI Quality & Experimentation Platform](../prd/PRD-007-ai-quality-experiment-platform.md)**
(PRD-007), which uses these metrics to gate new model
deployments and configuration changes. The complete
evaluation loop (runner, scorer, comparator) is
documented in the
**[Experiment Guide](../guides/EXPERIMENT_GUIDE.md)**
(Step 4: Evaluate Results).

## CI/CD Integration

### Continuous Integration Strategy

The CI/CD pipeline (GitHub Actions) implements a multi-layered validation strategy to balance fast feedback with comprehensive quality control.

#### 1. Every Pull Request (Per-Commit Feedback)

- **lint**: Fast formatting, code linting, markdown
  linting, and type checking (~2 min)
- **test-unit**: All unit tests with coverage,
  verified network isolation (~3-5 min)
- **test-integration-fast**: Critical path integration
  tests using test ML models (~5-8 min)
- **test-e2e-fast**: Critical path E2E tests (Tier 1)
  using test ML models (~8-12 min)
- **viewer-unit**: Vitest unit tests for `web/gi-kg-viewer` TS utils (`make test-ui`; ~150 ms)
- **viewer-e2e**: Playwright tests for `web/gi-kg-viewer` (`make test-ui-e2e`; Firefox, Vite on 5174)
- **build**: Package build validation (~2 min)
- **docs**: Documentation build validation (~3 min)

#### 2. Main Branch & Release Branches (Full Validation)

- **Unified Coverage**: Combines unit, integration,
  and E2E coverage reports into a single artifact.
- **test-integration**: Complete integration test
  suite.
- **test-e2e**: Complete E2E test suite (Tier 1).
- **Automated Metrics**: Collection of test health,
  code quality, and pipeline performance metrics.
- **Unified Dashboard**: Deployment of the
  [Metrics Dashboard](../ci/METRICS.md) to GitHub Pages.

#### 3. Nightly Comprehensive (Deep Analysis)

- **nightly-viewer-unit** / **nightly-viewer-e2e**: GI/KG viewer Vitest (parallel with unit) and
  Playwright after pytest E2E — same layout as `python-app.yml`; run on every scheduled/manual
  nightly (no path filters).
- **nightly-only-tests**: Full validation (Tier 3) using production-quality ML models (Whisper base, BART-large, LED-large).
- **Data Quality (Tier 2)**: Volume validation with multiple episodes.
- **Module Dependency Analysis**: Automated graph generation and coupling analysis.
- **Trend Tracking**: Historical metrics accumulation for long-term health monitoring.

#### Test Execution Pattern

- **Parallel Execution**: Most jobs run in parallel with reserved cores for system stability (`auto - 2`).
- **Flaky Test Resilience**: Unit, integration, and E2E jobs in GitHub Actions use automatic reruns (`--reruns 2 --reruns-delay 1`) alongside `pytest-json-report` where JSON is emitted.
- **Network Isolation**: Enforced via `pytest-socket` across all test tiers.
- **LLM Exclusion**: API-based tests (OpenAI) are excluded from nightly runs to avoid costs.

**For detailed test execution commands, parallel execution, and coverage configuration, see [Testing Guide](../guides/TESTING_GUIDE.md).**

## Testing Patterns

### Dependency Injection

- **CLI Testing**: Use `cli.main()` override callables (`apply_log_level_fn`, `run_pipeline_fn`, `logger`)
- **Workflow Testing**: Mock `run_pipeline` for CLI-focused tests
- **Benefit**: Test CLI behavior without executing full pipeline

### Mocking External Dependencies

- **HTTP**: Mock `requests.Session` and responses (unit/integration tests), use E2E server for E2E tests
- **Whisper**: Mock `whisper.load_model()` and `whisper.transcribe()` (unit and integration tests), use real models (E2E tests only)
- **ML Dependencies (spacy, torch, transformers)**:
  - **Unit Tests**: Must **not** require the **`[ml]`** extra -- mock or stub (`sys.modules`) before
    importing dependent modules; see [Unit tests and optional extras](#unit-tests-and-optional-extras-pyproject).
  - **Integration Tests**: Always mocked -- real ML inference is E2E only
  - **Verification**: CI runs `scripts/tools/check_unit_test_imports.py` (import-time check) and `scripts/tools/check_test_policy.py` (3-tier boundary rules) to ensure unit tests stay `[dev]`-only and ML models stay in E2E
- **File System**: Use `tempfile` for isolated test environments
- **API Providers** (OpenAI, Gemini, Anthropic,
  Mistral, DeepSeek, Grok, Ollama):
  - **Unit Tests**: Mock API clients (e.g., `OpenAI`,
    `genai`, `anthropic.Anthropic` classes)
  - **Integration Tests**: Mock API clients or use
    E2E server mock endpoints
  - **E2E Tests**: Use E2E server mock endpoints
    (real HTTP client, mock API responses). Dedicated
    mock fixtures: `openai_mock.py`,
    `gemini_mock_client.py`

### Test Isolation

- Each test uses `tempfile.TemporaryDirectory` for output
- Tests clean up after themselves
- No shared state between tests
- Mock external services (HTTP, file system)
- No network calls in unit tests (enforced by pytest plugin)
- No filesystem I/O in unit tests (enforced by pytest plugin, except tempfile operations)

### Error Testing

- Test validation errors (invalid config, malformed RSS)
- Test network failures (timeouts, connection errors)
- Test missing dependencies (Whisper, spaCy unavailable)
- Test edge cases (empty feeds, missing transcripts, invalid URLs)

## Test Requirements by Module

### `cli.py`

- [ ] Argument parsing (valid, invalid, edge cases)
- [ ] Config file loading and precedence
- [ ] Error handling and exit codes
- [ ] Dependency injection hooks
- [ ] Version flag behavior

### `config.py`

- [ ] Type coercion and validation
- [ ] Default value application
- [ ] Config file loading (JSON/YAML)
- [ ] Alias resolution
- [ ] Error messages

### `workflow.py`

- [ ] Pipeline orchestration
- [ ] Concurrent downloads
- [ ] Whisper job queuing
- [ ] Cleanup operations
- [ ] Dry-run mode

### `rss_parser.py`

- [ ] RSS parsing (various formats)
- [ ] Namespace handling
- [ ] URL resolution
- [ ] Episode creation
- [ ] Error handling (malformed XML)

### `downloader.py`

- [ ] HTTP session configuration
- [ ] Retry logic
- [ ] URL normalization
- [ ] Streaming downloads
- [ ] Error handling

### `episode_processor.py`

- [ ] Transcript download
- [ ] Media download
- [ ] File naming
- [ ] Whisper job creation
- [ ] Error handling

### `filesystem.py`

- [ ] Filename sanitization
- [ ] Output directory derivation
- [ ] Run suffix generation
- [ ] Path validation

### `providers/ml/whisper_utils.py`

- [ ] Model loading
- [ ] Transcription invocation
- [ ] Screenplay formatting
- [ ] Language handling
- [ ] Error handling (missing dependency)

### `speaker_detection.py` (RFC-010)

- [ ] NER extraction
- [ ] Host/guest distinction
- [ ] CLI override precedence
- [ ] Fallback behavior
- [ ] Caching logic

### `progress.py`

- [ ] Factory registration
- [ ] Progress updates
- [ ] Context manager behavior

### `summarizer.py` (RFC-012)

- [x] Model selection logic (explicit, auto-detection for MPS/CUDA/CPU)
- [x] Model loading and initialization
- [x] **Model integration tests** (all models in `DEFAULT_SUMMARY_MODELS` can be loaded)
- [ ] Summary generation
- [ ] Key takeaways generation
- [x] Text chunking for long transcripts
- [ ] Safe summarization with error handling
- [ ] Memory optimization (CUDA/MPS)
- [x] Model unloading
- [ ] Integration with metadata generation

### `service.py` (Public API)

- [x] `ServiceResult` dataclass
- [x] `service.run()` with valid Config
- [x] `service.run()` with logging configuration
- [x] `service.run()` exception handling
- [x] `service.run_from_config_file()` JSON/YAML
- [x] `service.run_from_config_file()` error cases
- [x] `service.main()` CLI entry point
- [x] `service.main()` version/missing config
- [x] Service API importability via `__getattr__`
- [x] ServiceResult equality and string repr
- [x] Integration with public API

### `providers/` (All 9 Providers)

- [x] MLProvider: unit, lifecycle, integration, E2E
- [x] HybridMLProvider: unit, lifecycle, integration, E2E
- [x] OpenAIProvider: unit, factory, lifecycle,
  integration, E2E
- [x] GeminiProvider: unit, factory, lifecycle,
  integration, E2E
- [x] AnthropicProvider: unit, factory, lifecycle,
  integration, E2E
- [x] MistralProvider: unit, factory, lifecycle,
  integration, E2E
- [x] DeepSeekProvider: unit, factory, lifecycle,
  integration, E2E
- [x] GrokProvider: unit, factory, lifecycle,
  integration, E2E
- [x] OllamaProvider: unit, factory, lifecycle,
  integration, E2E
- [x] Provider capabilities (`test_capabilities.py`)
- [x] Provider params (`test_provider_params.py`)

### `prompts/store.py` (Prompt Management)

- [x] Template loading per provider/task/version
- [x] Jinja2 variable rendering
- [x] Missing template error handling
- [x] Template version selection

### `schemas/summary_schema.py`

- [x] Summary schema validation
  (`test_summary_schema.py`)

### GIL Modules (RFC-049 — Implemented)

- [x] `gi/schema.py` — `gi.json` validation (unit: `test_schema.py`)
- [x] `gi/io.py` — gi.json read/write (unit: `test_io.py`)
- [x] Standalone schema validation: `scripts/tools/validate_gi_schema.py` and `make validate-gi-schema [ARTIFACTS_DIR=path]`; E2E tests that produce gi.json run strict validation inline (ci-fast gates on it).
- [x] KG artifacts: `scripts/tools/validate_kg_schema.py` and `make validate-kg-schema [ARTIFACTS_DIR=path]`; unit tests `test_kg_pipeline.py`, `test_kg_llm_extract.py`, `test_kg_schema.py`, `test_kg_contracts.py`; integration `test_kg_integration.py`, `test_kg_metadata_integration.py`; E2E subprocess smoke `test_gi_kg_cli_subprocess_e2e.py` (`kg validate`, `kg inspect`); acceptance via **`make test-acceptance`** (operator YAML paths you choose) and/or **`make test-acceptance-fixtures-fast`** (materialized **`MAIN_ACCEPTANCE_CONFIG.yaml`** matrix rows).
- [x] `gi/load.py` — load artifact, evidence spans, find by episode/insight id (unit: `test_load.py`)
- [x] `gi/explore.py` — scan, collect, topic filter (unit: `test_explore.py`)
- [x] `gi/pipeline.py` — build_artifact stub and grounded (unit: `test_pipeline.py`)
- [x] `gi/grounding.py` — find_grounded_quotes (unit: `test_grounding.py`; optional integration with real models)
- [x] GIL in workflow: metadata_generation writes gi.json when generate_gi true (unit: test_metadata_generation; integration/E2E pipeline tests as marked)
- [x] CLI `gi inspect`, `gi show-insight`, `gi explore` (unit: test_cli TestGiSubcommand; E2E subprocess: `test_gi_kg_cli_subprocess_e2e.py` for `gi validate`)
- [x] Three-tier extraction (ML-only, Hybrid, Cloud) — implemented (transformers, hybrid_ml, LLM providers)
- [ ] GIL extraction latency per tier — benchmarking planned

### `search/` (Semantic Corpus Search — RFC-061)

- [x] `search/chunker.py` — transcript chunking
  (unit: `test_chunker.py`)
- [x] `search/faiss_store.py` — FAISS vector store
  operations (unit: `test_faiss_store.py`)
- [x] `search/indexer.py` — corpus-wide index build,
  nested feeds, hybrid indexing, composite fingerprint
  keys (unit: `test_indexer.py`)
- [x] `search/corpus_scope.py` — `discover_metadata_files`
  hybrid merge, feed ID normalization
  (unit: `test_corpus_scope.py`)
- [x] `search/corpus_search.py` — shared search logic
  for CLI and HTTP
- [x] `search/corpus_similar.py` — episode-level
  similarity (unit: `test_corpus_similar.py`)
- [x] `search/cli_handlers.py` — CLI `podcast search`
  subcommand

### `server/` (Viewer API — RFC-062, RFC-067, RFC-068)

- [x] `server/app.py` — app factory, CORS, static
  mounting (unit: `test_viewer_*.py`)
- [x] `server/corpus_catalog.py` — filesystem-backed
  episode catalog (unit: `test_corpus_catalog.py`;
  integration: `test_viewer_corpus_library.py`)
- [x] `server/corpus_digest.py` — time-window digest
  selection and topic config
- [x] `server/index_rebuild.py` — background FAISS
  rebuild coordination (unit:
  `test_index_rebuild_gate.py`; integration:
  `test_index_rebuild.py`)
- [x] `server/index_staleness.py` — vector index
  freshness heuristics (unit:
  `test_index_staleness.py`)
- [x] `server/pathutil.py` — safe corpus path
  resolution and traversal prevention
- [x] `server/routes/` — route modules validated via
  FastAPI `TestClient` (unit) and wired `create_app`
  (integration in `tests/integration/server/`)

### `workflow/` Extensions (v2.5–v2.6)

- [x] `workflow/stages/` — stage-specific modules
  (setup, scraping, processing, transcription,
  metadata, summarization)
- [x] `workflow/corpus_operations.py` — multi-feed
  manifest and summary JSON (unit:
  `test_corpus_operations.py`)
- [x] `workflow/degradation.py` — graceful degradation
  policies for non-critical stage failures
- [x] `workflow/jsonl_emitter.py` — streaming JSONL
  metrics output
- [x] `workflow/run_manifest.py` /
  `workflow/run_summary.py` — run-level manifest
  capture and summary generation

## Test Pyramid Status

> **Note**: Test distribution numbers should be verified periodically by running test collection
> and counting tests by layer. Historical progress tracking was documented in
> `docs/wip/TEST_PYRAMID_ANALYSIS.md` and `docs/wip/TEST_PYRAMID_PLAN.md` (now consolidated here).

### Current State vs. Ideal Distribution

The test pyramid has reached a healthy shape that matches the ideal targets.

**Current Distribution (~3,770 tests, verified Apr 2026):**

- **Unit Tests**: ~3,000 (~80%) Target 70-80%
- **Integration Tests**: ~530 (~14%) Target 15-20%
- **E2E Tests**: ~230 (~6%) Target 5-10%

**Visual Representation:**

```text
        ╱╲
       ╱  ╲      E2E: ~6% (~230 tests)
      ╱    ╲
     ╱      ╲    Integration: ~14% (~530 tests)
    ╱        ╲
   ╱          ╲  Unit: ~80% (~3,000 tests)
  ╱____________╲
              Total: ~3,770 tests

Ideal Pyramid:
        ╱╲
       ╱  ╲      E2E: 5-10%
      ╱    ╲
     ╱      ╲    Integration: 15-20%
    ╱        ╲
   ╱          ╲  Unit: 70-80%
  ╱____________╲
```

The pyramid is now well-balanced. Historical issues (misclassified tests, missing unit coverage) have been addressed through the improvement phases below. Remaining maintenance items:

1. **Unit layer (~80%)**: Healthy. All core modules have unit tests. Continue adding unit tests as default for new work.
2. **Integration layer (~14%)**: Validates cross-module boundaries and provider interactions across all 9 providers.
3. **E2E layer (~6%)**: Covers complete user workflows (CLI, Library API, Service API) without duplicating integration-level scenarios.

### Goals and Targets

**Target Distribution (achieved as of Apr 2026):**

- **Unit Tests**: 70-80% (~3,000 tests)
- **Integration Tests**: 15-20% (~530 tests)
- **E2E Tests**: 5-10% (~230 tests)

**Success Metrics:**

- Unit test execution time: < 30 seconds
- Integration test execution time: < 5 minutes
- E2E test execution time: < 20 minutes
- Test coverage: Maintain ≥70%

### Improvement Strategy (Completed)

All four phases have been executed. The pyramid now matches the target distribution.

**Phase 1: Reclassify Misplaced Tests** Done

- Reclassified summarizer tests from E2E to unit.
- Reviewed and reclassified E2E tests that violated testing strategy definitions.

**Phase 2: Add Missing Unit Tests** Done

- Added unit tests for all core modules (`workflow.py`, `episode_processor.py`, `summarizer.py`, `speaker_detection.py`, `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py`, `cli.py`, `service.py`).

**Phase 3: Optimize Integration Layer** Done

- Moved component interaction tests from E2E to integration.
- Added integration tests for cross-module boundaries.

**Phase 4: Reduce E2E to True E2E** Done

- E2E layer now contains only true end-to-end user workflow tests.
- **Final Result**: Unit: ~80%, Integration: ~14%, E2E: ~6%

### Unit Test Coverage by Module

All previously-identified coverage gaps have been addressed. Key modules and their unit test status:

**Core Modules (covered):**

- `summarizer.py`: Text cleaning, chunking, validation functions
- `workflow.py`: Pipeline orchestration helpers
- `episode_processor.py`: Episode processing logic

**Supporting Modules (covered):**

- `speaker_detection.py`: Detection and scoring logic
- `cli.py` and `service.py`: Argument parsing and service logic
- `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py`: Utility functions

## Future Testing Enhancements

### E2E Test Infrastructure Improvements (Issue #14)

- [x] Local HTTP test server
- [x] Test audio file fixtures
- [x] Real Whisper integration tests
- [x] Test markers and CI integration

### Library API Tests (Issue #16)

- [x] `run_pipeline()` E2E tests
- [x] `load_config_file()` tests
- [x] Error handling tests
- [x] Return value validation

### GIL Testing (Implemented — PRD-017, RFC-049/050) {#gil-testing-implemented--prd-017-rfc-049050}

Testing for the Grounded Insight Layer follows the established test pyramid. Current coverage:

**Provider-based evidence:** The pipeline uses `find_grounded_quotes_via_providers` (in `gi/grounding.py`), which calls provider methods `extract_quotes` and `score_entailment`. These are unit-tested with mocked dependencies in `test_grounding.py` and `test_pipeline.py`; provider implementations (ML and LLM) are covered in `test_ml_provider.py`, `test_openai_provider.py`, and other provider test modules. Evidence stack integration (embedding, QA, NLI) is covered in `test_evidence_stack_integration.py`.

**Unit Tests:**

- [x] `gi/schema.py` — JSON schema validation (`test_schema.py`)
- [x] `gi/io.py` — read/write gi.json, `collect_gi_paths_from_inputs` (`test_io.py`)
- [x] `gi/quality_metrics.py` — PRD-017 aggregates (`test_gil_quality_metrics.py`)
- [x] `gi/corpus.py` — NDJSON / merged export (`test_gi_corpus.py`)
- [x] `kg/quality_metrics.py` — PRD-019 aggregates (`test_quality_metrics.py`)
- [x] `gi/load.py` — load artifact, evidence spans, find by episode/insight id (`test_load.py`)
- [x] `gi/explore.py` — scan, collect, topic filter (`test_explore.py`)
- [x] `gi/pipeline.py` — build_artifact stub, grounded via providers (including `create_gil_evidence_providers` / injected quote + entailment providers) (`test_pipeline.py`)
- [x] `gi/grounding.py` — find_grounded_quotes with mocked QA/NLI; find_grounded_quotes_via_providers with mock extract_quotes/score_entailment; pipeline_metrics evidence call counters (`test_grounding.py`)
- [x] Providers: extract_quotes and score_entailment (ML and LLM) unit-tested with mocked dependencies (`test_ml_provider.py`, `test_openai_provider.py`, etc.). Provider path and evidence method behaviour are covered in `test_grounding.py` and `test_pipeline.py`.
- [x] Workflow: generate_episode_metadata passes quote_extraction_provider and entailment_provider into build_artifact when generate_gi and gi_require_grounding true (`test_metadata_generation.py`)
- [x] CLI gi subcommand: parse, validate, export, inspect, show-insight, explore, query, exit codes (`test_cli.py`); config logging warnings for GIL stub insights and API summary + local evidence hybrid (`TestLogConfigurationGiStubWarning`, `TestLogConfigurationGilHybridWarning`)
- [x] CI fixtures: `tests/fixtures/gil_kg_ci_enforce` — GIL + KG quality metrics enforce (GitHub Actions + `make quality-metrics-ci`)
- [x] Bridge builder (`tests/unit/builders/test_bridge_builder.py`); CIL corpus logic (`test_cil_queries.py`); CIL HTTP (`tests/integration/server/test_cil_api.py`); search lift + offset verify (`tests/unit/podcast_scraper/search/test_transcript_chunk_lift.py`, `test_gil_chunk_offset_verify.py`); bridge wiring integration (`tests/integration/test_bridge_integration.py`)
- [x] Insight clustering (#599): unit tests for `collect_insight_rows_from_corpus`, `build_insight_clusters_payload` (`test_insight_clusters.py`); integration test for end-to-end corpus flow (`tests/integration/search/test_insight_clusters_cli.py`)
- [x] Cluster context expansion (#601): unit tests for `load_insight_clusters`, `expand_with_cluster_context`, `format_cluster_context` (`test_insight_cluster_context.py`); integration test for artifact → expand flow
- [x] Multi-quote extraction (#600): ML provider test updated for `answer_candidates(top_k=3)` mock; existing provider tests cover backward-compat `quote_text` fallback
- [x] Speaker flow validation (#598): integration tests for NER → host/guest detection → KG person injection (`tests/integration/providers/test_speaker_flow_integration.py` — 7 tests covering host+guest, host-only, description snippet, mentioned-not-guest, org-as-author). Speaker detection simplified from 1389→935 lines; 7 unit tests for removed scoring functions deleted

#### GIL and KG CI quality gates {#gil-and-kg-ci-quality-gates}

- **`make quality-metrics-ci`** — Runs `gil_quality_metrics.py` and `kg_quality_metrics.py` with **`--enforce`** (and strict schema flags) on `tests/fixtures/gil_kg_ci_enforce`. Listed in **`make help`**; depended on by **`make ci-fast`** in the repository `Makefile`.
- **GitHub Actions** — The `test-unit` job includes **GIL and KG quality metrics on CI fixtures (PRD-017 / PRD-019)** (same fixture tree; see `.github/workflows/python-app.yml`).
- **Optional local gates** — `make gil-quality-metrics DIR=<run_root>` and `make kg-quality-metrics DIR=<run_root>` with `ARGS='--enforce …'` for release or regression checks over a real run directory.
- **Critical-path E2E** — `tests/e2e/test_gi_kg_cli_subprocess_e2e.py` carries **`@pytest.mark.critical_path`** on GI/KG CLI subprocess smoke (included in `make test-fast` / `make ci-fast` per project defaults).

**Integration Tests:**

- [x] Evidence stack: load embedding, QA, NLI; encode, answer, entailment_score (`test_evidence_stack_integration.py`)
- [x] Optional: find_grounded_quotes with real models (skip when offline)
- [ ] `gi.json` → Postgres export (RFC-051) — not yet implemented

**E2E Tests:**

- [x] Subprocess GI/KG CLI smoke: `gi validate --strict`, `kg validate --strict`, `kg inspect --format json` on repo fixtures (`tests/e2e/test_gi_kg_cli_subprocess_e2e.py`, `critical_path`)
- [ ] Full pipeline E2E with `generate_gi` true → gi inspect, show-insight, explore (optional future; invalid-arg and inspect behaviors covered in unit `test_cli` today)

**Quality Evaluation (Planned):**

- Quote verbatim accuracy, grounding rate, insight coverage
- Cross-provider comparison using golden dataset

### Model Registry Testing (RFC-044, Implemented)

- `ModelRegistry` initialization and model lookup
- `ModelCapabilities` validation (token limits,
  device defaults)
- Registry used by summarizer and GIL extractor
- Invalid model name error handling

### Hybrid ML Platform Testing (RFC-042, Implemented)

- Hybrid MAP-REDUCE pipeline (LED map → FLAN-T5
  reduce)
- Extractive QA for quote extraction
- NLI for grounding validation
- Sentence embeddings for topic deduplication
- `StructuredExtractor` protocol compliance

### Performance Testing

- [ ] Benchmark large feed processing (1000+ episodes)
- [ ] Measure Whisper transcription performance
- [ ] Profile memory usage
- [ ] Test concurrent download limits
- [ ] GIL extraction latency per tier (benchmarking planned)

### Property-Based Testing

- [ ] Generate random RSS feeds
- [ ] Test filename sanitization with fuzzing
- [ ] Test URL normalization with edge cases
- [ ] `gi.json` schema fuzzing

## References

- **[Critical Path Testing Guide](../guides/CRITICAL_PATH_TESTING_GUIDE.md)**
  — What to test, prioritization
- **[Testing Guide](../guides/TESTING_GUIDE.md)**
  — Test execution, fixtures, coverage
- **[Unit Testing Guide](../guides/UNIT_TESTING_GUIDE.md)**
  — Mocking patterns and isolation
- **[Integration Testing Guide](../guides/INTEGRATION_TESTING_GUIDE.md)**
  — Integration test mocking guidelines
- **[E2E Testing Guide](../guides/E2E_TESTING_GUIDE.md)**
  — pytest E2E server, real ML models, API mocking; Playwright viewer stack
- **[ADR-066: Playwright for UI E2E](../adr/ADR-066-playwright-for-ui-e2e-testing.md)**
  — browser automation choice for viewer v2
- Test structure reorganization:
  `docs/rfc/RFC-018-test-structure-reorganization.md`
- CI workflow: `.github/workflows/python-app.yml`
- Related RFCs: RFC-001–RFC-018 (testing strategies),
  RFC-029 (unified providers), RFC-017 (prompt store)
- Implemented RFCs: RFC-042 (Hybrid ML), RFC-044 (Model
  Registry), RFC-049 (GIL Schema), RFC-050 (GIL Pipeline);
  Draft: RFC-051 (DB Projection)
- Related Issues: #14 (E2E testing), #16 (Library API
  E2E tests), #94 (src/ layout), #98 (Test structure
  reorganization)
- Architecture: `docs/architecture/ARCHITECTURE.md`
- Non-Functional Requirements: `docs/architecture/NON_FUNCTIONAL_REQUIREMENTS.md` (quality attributes validated by tests)
- Contributing guide: `CONTRIBUTING.md`
