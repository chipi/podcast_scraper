# E2E Testing Guide

> **See also:**
>
> - [Testing Strategy](../architecture/TESTING_STRATEGY.md) - High-level testing philosophy and test pyramid
> - [Testing Guide](TESTING_GUIDE.md) - Quick reference and test execution commands
> - [RSS and feed ingestion](RSS_GUIDE.md) - Production RSS path (HTTP, parsing, episode selection); contrasts with local `e2e_server` fixture feeds below

This guide covers **pytest** E2E test implementation: real HTTP client, E2E server, ML model
usage, and OpenAI mock endpoints.

For **where Playwright fits** in the overall strategy (pyramid, CI jobs, pytest vs browser), see
[Testing Strategy — Browser UI E2E (Playwright)](../architecture/TESTING_STRATEGY.md#browser-ui-e2e-playwright).

## Browser E2E (Playwright) {#browser-e2e-playwright}

The GI/KG **Vue** viewer (`web/gi-kg-viewer`) uses **Playwright** (TypeScript, **Firefox**), not
pytest. This section summarizes the **browser** stack only; everything below *Overview* in this
file remains **pytest** E2E.

| Topic | Detail |
| ----- | ------ |
| **Run from repo root** | `make test-ui-e2e` (`npm install`, `playwright install firefox`, `npm run test:e2e`) |
| **Run in package** | `cd web/gi-kg-viewer && npm run test:e2e` |
| **Config** | `web/gi-kg-viewer/playwright.config.ts` — `testDir: ./e2e`, `webServer` runs **Vite** on **127.0.0.1:5174** |
| **Specs** | `web/gi-kg-viewer/e2e/*.spec.ts` (+ `fixtures.ts`, `helpers.ts`) |
| **Surface map** | [E2E_SURFACE_MAP.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md) — surfaces, fixtures, stable Playwright selectors (update with UI/E2E changes) |
| **CI** | Workflow job **`viewer-e2e`** (same commands as `make test-ui-e2e`) |

### Debugging UI issues and interpreting failures

The surface map is the shared **contract for accessible names, regions, and user entry paths**. When
a Playwright assertion fails, when you reproduce a bug manually, or when an agent drives the app
via **Chrome DevTools MCP** or Playwright MCP (a11y snapshots), use
[E2E_SURFACE_MAP.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md)
to see what label or region should appear, which spec owns the surface, and how to disambiguate
controls that share a visible name. It does not replace UXS for visual design. For the full
agent-browser workflow (symmetry between reproduction and fix validation), see
[Agent-Browser Closed Loop Guide](AGENT_BROWSER_LOOP_GUIDE.md).

### When you change viewer UX (required workflow)

Applies to **humans and AI agents** editing `web/gi-kg-viewer/` (Vue UI: copy, layout, routes,
theme tokens, accessible names, or flows that Playwright exercises). Do **not** ship UI-only PRs
without walking this list in order:

1. **`e2e/E2E_SURFACE_MAP.md`** — Update if anything **E2E-visible** or **selector-related** changed
   (including `getByRole` strings, `#search-q`, `.graph-canvas`, file-picker vs list flows).
2. **Playwright** — Update `e2e/*.spec.ts`, `helpers.ts`, and/or `fixtures.ts`; run **`make test-ui-e2e`**.
3. **`docs/uxs/`** — Update **[UXS-001](../uxs/UXS-001-gi-kg-viewer.md)** when **shared** tokens, typography,
   or shell-wide rules change; update the relevant **[feature UXS](../uxs/index.md)** (Digest, Library,
   Graph, Search, Dashboard, …) when a **surface-specific** visual contract changes, even if tests still pass.

Also documented in [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) (*GI / KG browser viewer*),
[TESTING_GUIDE.md](TESTING_GUIDE.md) (*Browser E2E*), [UX specifications index](../uxs/index.md),
`.cursorrules` (*GI/KG viewer UX*), and `.ai-coding-guidelines.md` (*GI/KG browser viewer*).
| **vs pytest E2E** | pytest proves CLI/pipeline + `e2e_server`; Playwright proves **browser UX** (graph shell, search UI, a11y paths) |
| **vs FastAPI unit tests** | `tests/unit/podcast_scraper/server/test_viewer_*.py` cover **`/api/*`** JSON contracts; use Playwright when behavior depends on the **SPA** |
| **vs Vitest** | `web/gi-kg-viewer/src/utils/*.test.ts` cover **pure TS logic** (parsing, merge, metrics); `make test-ui` (~150 ms, no browser). Use Playwright for **rendered UI behavior** |

**Further reading:** [Polyglot repository guide](POLYGLOT_REPO_GUIDE.md) (root vs `web/gi-kg-viewer/`),
[Testing Guide — Browser E2E](TESTING_GUIDE.md#browser-e2e-gi-kg-viewer-v2),
[ADR-066](../adr/ADR-066-playwright-for-ui-e2e-testing.md),
[web/gi-kg-viewer/README.md](https://github.com/chipi/podcast_scraper/blob/main/web/gi-kg-viewer/README.md).

## Overview

**E2E tests** test complete user workflows with real implementations. No mocking allowed (except network isolation).

| Aspect | Requirement |
| -------- | ------------- |
| **Speed** | < 60 seconds per test |
| **Scope** | Complete user workflow |
| **Entry points** | CLI commands, `run_pipeline()`, `service.run()` |
| **HTTP** | Real client with local E2E server |
| **Filesystem** | Real file operations |
| **ML Models** | Real (Whisper, spaCy, Transformers) - NO mocks |

## Manual CLI runs against the fixture server

For **human** multi-feed checks without real RSS, use the same HTTP handler as pytest’s **`e2e_server`**:

1. From repo root (venv on **`PYTHONPATH`** includes repo root so **`tests.e2e`** resolves): **`make serve-e2e-mock`** (default port **18765**; override with **`E2E_MOCK_PORT`**).
2. In another terminal: **`python -m podcast_scraper.cli --config config/examples/manual_e2e_mock_five_podcasts.yaml`**

That YAML lists the five primary mock feeds (**`podcast1`**–**`podcast5`**) plus long-form
fixtures **`podcast7_sustainability`**, **`podcast8_solar`**, and **`podcast9_solo`** (p07–p09;
**p06** edge-case feed is intentionally omitted). Tracked path is under **`config/examples/`**;
you may keep a copy under **`config/manual/`** (gitignored except **`README.md`**). This is
**not** the same contract as CI pytest E2E (no network guard, you choose ML cost); it reuses
fixture XML/audio only.

## Core Principle: No Mocking

E2E tests use **real implementations throughout**:

- Real HTTP client (with local server)
- Real filesystem I/O
- Real ML models (Whisper, spaCy, Transformers)
- Real providers (MLProvider, OpenAIProvider)
- No external network (blocked by network guard)
- No Whisper mocks
- No ML model mocks

## E2E Server

**Ports:** Pytest’s **`e2e_server`** binds an **ephemeral** port (not fixed). The
FastAPI app from **`make serve-api`** defaults to **8000**. For manual runs, the
same fixture HTTP handler is exposed on a **fixed** port via
**`make serve-e2e-mock`**, default **18765** (**`E2E_MOCK_PORT`** in the Makefile),
so the RSS/mock API server can run alongside **`serve-api`** without colliding.

The `e2e_server` fixture provides a local HTTP server serving test fixtures:

```python
def test_basic_workflow(e2e_server):
    # Get URLs for test resources
    rss_url = e2e_server.urls.feed("podcast1")
    audio_url = e2e_server.urls.audio("p01_e01")
    transcript_url = e2e_server.urls.transcript("p01_e01")

    # Run complete workflow
    result = run_pipeline(rss_url, output_dir)
    assert result.success
```

### Available URLs

| Method | Returns |
| -------- | --------- |
| `e2e_server.urls.feed(podcast_name)` | RSS feed URL (e.g. `/feeds/podcast1/feed.xml`) |
| `e2e_server.urls.audio(episode_id)` | Audio file URL (e.g. `/audio/p01_e01.mp3`) |
| `e2e_server.urls.transcript(episode_id)` | Transcript URL (e.g. `/transcripts/p01_e01.txt`) |
| `e2e_server.urls.base()` | Server base URL |
| `e2e_server.urls.openai_api_base()` | OpenAI mock API base (`/v1`) |
| `e2e_server.urls.gemini_api_base()` | Gemini mock API base (`/v1beta`) |
| `e2e_server.urls.mistral_api_base()` | Mistral mock API base (`/v1`) |
| `e2e_server.urls.grok_api_base()` | Grok mock API base (`/v1`) |
| `e2e_server.urls.deepseek_api_base()` | DeepSeek mock API base (`/v1`) |
| `e2e_server.urls.ollama_api_base()` | Ollama mock API base (`/v1`) |
| `e2e_server.urls.anthropic_api_base()` | Anthropic mock API base (base URL, no `/v1`) |

### Download resilience E2E {#download-resilience-e2e}

**`tests/e2e/test_download_resilience_e2e.py`**

- Transient HTTP on transcript URLs (`set_transient_error` with `fail_count`), plus permanent `set_error_behavior`.
- `fetch_url` / downloader retry totals (`configure_downloader`, `http_retry_total` on `Config`).
- Single-feed pipeline: `run.json` may include **`failure_summary`** when some episodes fail (see `test_partial_failure_produces_summary`).
- Multi-feed isolation when one RSS feed is broken: second feed still runs; with **`multi_feed_strict=True`** the batch reports failure (`test_one_feed_down_others_continue`).

**`tests/e2e/test_multi_feed_resilience_e2e.py` (GitHub #560, offline only)**

- **`corpus_run_summary.json`** at the corpus parent: per-feed **`ok`**, **`error`**, **`failure_kind`** (soft vs hard, #559), **`overall_ok`**, schema **`1.1.0`** with **`batch_incidents`** (rollup of `corpus_incidents.jsonl` for that batch) and per-feed **`episode_incidents_unique`** so **`episodes_processed: 0`** with **`ok: true`** is not read as “no issues.”
- Lenient default vs **`multi_feed_strict`** / **`--multi-feed-strict`**: service and CLI exit semantics when all failures are soft-classified (RSS HTTP errors, unknown slug 404, wrong path under `/feeds/...`).
- **Unknown slug** and **wrong filename** under a known feed (both 404 on the mock server, no DNS).
- **Transient RSS 503** on one feed’s `feed.xml` with RSS retries; batch **`overall_ok`** true when retries succeed.
- **Corpus lock**: pre-acquire `LOCK_BASENAME`, assert a blocked `service.run`, then success after release.
- **Multi_episode mode** (`E2E_TEST_MODE=multi_episode`, not fast): two feeds, `max_episodes` greater than 1, transcript 404 on a shared fixture path; asserts per-feed **`metrics.json`** skipped counts and matching **`run.json`** **`metrics.episodes_skipped_total`** (skipped transcript is not always a run-index **failure**, so **`failure_summary`** may be absent).

Handler API: `E2EHTTPRequestHandler.set_transient_error(path, status=..., fail_count=...)` and `set_error_behavior(path, status=...)`. See [CONFIGURATION.md — Download resilience](../api/CONFIGURATION.md#download-resilience).

**Fast vs multi_episode:** tests marked **`critical_path`** run under `make test-e2e-fast` (`E2E_TEST_MODE=fast`). The multi-episode partial-failure case above skips when `E2E_TEST_MODE=fast`; run `make test-e2e` (multi_episode) for full coverage.

### E2E Feeds (RSS)

Feed names and RSS file mapping. Which feed name you can use depends on **test mode** (see [Test Modes](#test-modes)). For how the real pipeline fetches and parses RSS (retries, conditional GET, circuit breaker, multi-feed), see [RSS and feed ingestion](RSS_GUIDE.md).

**Full fixtures** (used in `data_quality` and `nightly`; mapping from `PODCAST_RSS_MAP`):

| Feed name | RSS file | Description |
| --------- | -------- | ----------- |
| `podcast1` | `p01_mtb.xml` | Main podcast (MTB) |
| `podcast2` | `p02_software.xml` | Software podcast |
| `podcast3` | `p03_scuba.xml` | Scuba podcast |
| `podcast4` | `p04_photo.xml` | Photo podcast |
| `podcast5` | `p05_investing.xml` | Investing podcast |
| `edgecases` | `p06_edge_cases.xml` | Edge-case episodes |
| `podcast1_multi_episode` | `p01_multi.xml` | 5 short episodes (multi-episode tests) |
| `podcast1_episode_selection` | `p01_episode_selection.xml` | 3 items, newest-first, all Path 1 transcripts (#521) |
| `podcast9_solo` | `p09_biohacking.xml` | Solo speaker (host only) |
| `podcast7_sustainability` | `p07_sustainability.xml` | Long-form (~15k words; Issue #283) |
| `podcast8_solar` | `p08_solar.xml` | Long-form (~20k words; Issue #283) |

**Fast fixtures** (used in `fast` and `multi_episode` when `set_use_fast_fixtures(True)`; mapping from `PODCAST_RSS_MAP_FAST`):

| Feed name | RSS file | Description |
| --------- | -------- | ----------- |
| `podcast1` | `p01_fast.xml` | 1 short episode (Path 2: transcription) |
| `podcast1_with_transcript` | `p01_fast_with_transcript.xml` | 1 episode with transcript URL (Path 1: download) |
| `podcast1_multi_episode` | `p01_multi.xml` | Same 5-episode feed |
| `podcast1_episode_selection` | `p01_episode_selection.xml` | Same as full map (episode selection E2E) |
| `podcast9_solo` | `p09_biohacking.xml` | Solo speaker |
| `podcast7_sustainability` | `p07_sustainability.xml` | Long-form |
| `podcast8_solar` | `p08_solar.xml` | Long-form |

#### Allowed feeds per test mode

Set automatically by `conftest` from `E2E_TEST_MODE`.

| Mode | Allowed feed names |
| ---- | ------------------ |
| `fast` | `podcast1`, `podcast1_with_transcript`, `podcast1_multi_episode`, `podcast1_episode_selection`, `podcast9_solo`, `podcast7_sustainability`, `podcast8_solar` |
| `multi_episode` | `podcast1_multi_episode`, `podcast1_episode_selection`, `podcast1_with_transcript`, `edgecases`, `podcast7_sustainability`, `podcast8_solar` |
| `nightly` | `podcast1`, `podcast2`, `podcast3`, `podcast4`, `podcast5`, `podcast1_episode_selection` (full fixtures) |
| `data_quality` | All feeds (None = allow all) |

Use `e2e_server.urls.feed("podcast1_multi_episode")` or `e2e_server.urls.feed("podcast1_episode_selection")` etc. Only feeds in the allowed set for the current mode are served; others return 404.

### E2E Server Options

The `e2e_server` fixture (and the handler class) support these options for controlling behavior:

**Error injection (chaos / failure testing):**

| Method | Description |
| ------ | ----------- |
| `e2e_server.set_error_behavior(url_path, status, delay=0.0)` | For a given path (e.g. `"/audio/p01_multi_e03.mp3"`), return HTTP `status` (e.g. 404, 500). Optional `delay` in seconds. |
| `e2e_server.clear_error_behavior(url_path)` | Remove error behavior for that path. |
| `e2e_server.reset()` | Clear all error behaviors and set allowed podcasts to None. |

Example: simulate 404 on audio so the run index records a failed episode:

```python
e2e_server.set_error_behavior("/audio/p01_multi_e03.mp3", 404)
# ... run pipeline ...
# assert index.json has one failed episode with error_type, error_message, error_stage
e2e_server.clear_error_behavior("/audio/p01_multi_e03.mp3")
```

**Allowed podcasts (advanced):**

| Method | Description |
| ------ | ----------- |
| `e2e_server.set_allowed_podcasts(podcasts)` | Restrict which feed names are served. `podcasts`: set of names or `None` for all. Normally set by conftest from `E2E_TEST_MODE`. |

**Fixture mode:**

- When **fast fixtures** are on, feeds resolve via `PODCAST_RSS_MAP_FAST` (e.g. `podcast1` → `p01_fast.xml`).
- When off (e.g. nightly/data_quality), feeds use `PODCAST_RSS_MAP` (e.g. `podcast1` → `p01_mtb.xml`).
- Conftest sets this from `E2E_TEST_MODE`; teardown clears error behaviors and resets fast-fixtures mode.

### Served Content

Content is served from `tests/fixtures/`:

- RSS feeds: `tests/fixtures/rss/*.xml`
- Audio files: `tests/fixtures/audio/*.mp3`
- Transcripts: `tests/fixtures/transcripts/*.txt`

## OpenAI Mock Endpoints

For API providers (OpenAI), the E2E server provides mock endpoints:

```python
def test_openai_provider(e2e_server):
    cfg = Config(
        rss_url=e2e_server.urls.feed("podcast1"),
        transcription_provider="openai",
        openai_api_key="sk-test123",
        openai_api_base=e2e_server.urls.openai_api_base(),  # Use mock
    )
    result = run_pipeline(cfg)
    assert result.success
```

### Mock Endpoints

| Endpoint | Purpose |
| ---------- | --------- |
| `/v1/chat/completions` | Summarization, speaker detection, GIL evidence (extract_quotes, score_entailment) |
| `/v1/audio/transcriptions` | Transcription |
| `/v1/messages` (Anthropic) | Summarization, speaker detection, GIL evidence (extract_quotes, score_entailment) |
| `/v1beta/models/{model}:generateContent` (Gemini) | Summarization, speaker detection, GIL evidence (extract_quotes, score_entailment) |

See `tests/e2e/fixtures/e2e_http_server.py` for implementation.

## ML Model Usage

E2E tests use **real ML models** - no mocking allowed.

### Test Model Defaults

Tests use smaller, faster models for speed:

| Component | Test Model | Production Model |
| ----------- | ------------ | ------------------ |
| Whisper | `tiny.en` | `base.en` |
| spaCy | `en_core_web_sm` | `en_core_web_sm` |
| Transformers MAP | `facebook/bart-base` | `facebook/bart-large-cnn` |
| Transformers REDUCE | `allenai/led-base-16384` | `allenai/led-large-16384` |

### Model Cache Requirements

Tests require models to be pre-cached:

```bash

# Preload all required models

make preload-ml-models
```

Use cache helpers to skip gracefully if not cached:

```python
from tests.integration.ml_model_cache_helpers import (
    require_whisper_model_cached,
    require_transformers_model_cached,
)

def test_with_real_models(e2e_server):
    require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)
    require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
    # Test with real models...
```

## Network Guard

E2E tests use network isolation to prevent external calls:

```bash
pytest tests/e2e/ --disable-socket --allow-hosts=127.0.0.1,localhost
```

If a test attempts external network access:

```text
SocketBlockedError: A]socket.socket call was blocked
```

## Test Patterns

### CLI E2E Test

```python
@pytest.mark.e2e
def test_cli_transcript_download(e2e_server, tmp_path):
    """Test CLI transcript download command."""
    rss_url = e2e_server.urls.feed("podcast1_with_transcript")

    result = subprocess.run([
        "podcast-scraper", rss_url,
        "--output-dir", str(tmp_path),
    ], capture_output=True)

    assert result.returncode == 0
    assert (tmp_path / "0001 - Episode 1.txt").exists()
```

### Library API E2E Test

```python
@pytest.mark.e2e
def test_run_pipeline(e2e_server, tmp_path):
    """Test run_pipeline() library API."""
    cfg = Config(
        rss_url=e2e_server.urls.feed("podcast1"),
        output_dir=str(tmp_path),
    )
    result = run_pipeline(cfg)
    assert result.success
```

### Service API E2E Test

```python
@pytest.mark.e2e
def test_service_run(e2e_server, tmp_path):
    """Test service.run() API."""
    cfg = Config(
        rss_url=e2e_server.urls.feed("podcast1"),
        output_dir=str(tmp_path),
    )
    result = service.run(cfg)
    assert result.success
```

### Full Pipeline with ML

```python
@pytest.mark.e2e
@pytest.mark.ml_models
def test_full_pipeline_with_summarization(e2e_server, tmp_path):
    """Test complete pipeline with real ML models."""
    require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)
    require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)

    cfg = Config(
        rss_url=e2e_server.urls.feed("podcast1"),
        output_dir=str(tmp_path),
        generate_summaries=True,
        summary_model=config.TEST_DEFAULT_SUMMARY_MODEL,
    )
    result = run_pipeline(cfg)
    assert result.success
    # Verify summary was generated
```

## Test Modes

E2E tests support different modes via the `E2E_TEST_MODE` environment variable (set by the Makefile). Mode controls which feeds are allowed and whether fast or full fixtures are used; see [E2E Feeds (RSS)](#e2e-feeds-rss) and [Allowed feeds per test mode](#allowed-feeds-per-test-mode).

| Mode | Episodes | Fixtures | Use Case |
| ------ | ---------- | --------- | ---------- |
| `fast` | 1 per test (via monkeypatch) | Fast | Quick feedback, critical path |
| `multi_episode` | No limit (e.g. 5) | Fast | Full validation |
| `nightly` | No limit (e.g. 15 across p01–p05) | Full | Nightly suite |
| `data_quality` | Multiple, all mock data | Full | Data quality / nightly |

Markers can override effective mode: tests marked `@pytest.mark.nightly` use nightly when `E2E_TEST_MODE` is unset; tests marked `@pytest.mark.critical_path` use fast when unset.

```bash
# Run with multi-episode mode
E2E_TEST_MODE=multi_episode make test-e2e

# Run fast E2E (critical path only, 1 episode per test)
make test-e2e-fast
```

### `make test-fast` / `make ci-fast` and E2E progress {#make-test-fast--make-ci-fast-and-e2e-progress}

The Makefile runs **two** pytest passes for critical-path E2E: tests **without** `@pytest.mark.ml_models` use parallel workers (`-n`); tests **with** `ml_models` run **sequentially** (`-n 1`). That avoids pytest-xdist showing a long flat progress bar while a single worker runs Whisper, spaCy, or Transformers (it looked like a hang around 70–80% even though work was still running). The ML phase can still take many minutes on CPU; ensure the Whisper test model is cached (`make preload-ml-models` or CI cache) so runs fail fast instead of downloading.

`make test-e2e-fast` uses the same split (`not ml_models` then `ml_models`).

## Test Files

| Purpose | Test File |
| --------- | ----------- |
| Network guard | `test_network_guard.py` |
| OpenAI mocking | `test_openai_mock.py` |
| E2E server | `test_e2e_server.py` |
| Fixture mapping | `test_fixture_mapping.py` |
| Basic workflows | `test_basic_e2e.py` |
| CLI commands | `test_cli_e2e.py` |
| Library API | `test_library_api_e2e.py` |
| Service API | `test_service_api_e2e.py` |
| Whisper | `test_whisper_e2e.py` |
| ML models | `test_ml_models_e2e.py` |
| Error handling | `test_error_handling_e2e.py` |
| Edge cases | `test_edge_cases_e2e.py` |
| HTTP behaviors | `test_http_behaviors_e2e.py` |
| Ollama providers | `test_ollama_provider_integration_e2e.py` |

## Running E2E Tests

```bash

# All E2E tests

make test-e2e

# Fast critical path (parallel non-ML, then sequential ML; see Test Modes above)

make test-e2e-fast

# Sequential (for debugging)

pytest tests/e2e/ -n 0

# Specific test file

pytest tests/e2e/test_basic_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost
```

## Test Markers

- `@pytest.mark.e2e` -- Required for all E2E tests
- `@pytest.mark.ml_models` -- Tests requiring real ML models (E2E only)
- `@pytest.mark.critical_path` -- Critical path tests (run in fast suite).
  See [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)

`@pytest.mark.ml_models` belongs **only** on E2E tests. `make check-test-policy`
(rule I1) enforces that integration tests do not carry this marker.

- `@pytest.mark.multi_episode` - Multi-episode tests
- `@pytest.mark.data_quality` - Data quality tests (nightly)

## Provider Testing

For provider-specific E2E testing (E2E server endpoints, full pipeline with providers):

→ **[Provider Implementation Guide - Testing Your Provider](PROVIDER_IMPLEMENTATION_GUIDE.md#testing-your-provider)**

Covers:

- E2E server mock endpoint implementation
- Provider works in full pipeline
- Multiple providers work together
- E2E test checklist for new providers

### Real API Testing (Manual Mode)

Some providers support real API testing for manual validation:

**Ollama (Local Server):**

```bash
# Prerequisites: Ollama installed and running
ollama serve  # Start server
ollama pull llama3.3:latest  # Pull models

# Run tests with real Ollama
USE_REAL_OLLAMA_API=1 \
pytest tests/e2e/test_ollama_provider_integration_e2e.py -v
```

**OpenAI/Gemini (Cloud APIs):**

```bash
# Set environment variable to use real APIs
USE_REAL_OPENAI_API=1 pytest tests/e2e/test_openai_provider_integration_e2e.py
USE_REAL_GEMINI_API=1 pytest tests/e2e/test_gemini_provider_integration_e2e.py
```

**Note:** Real API mode preserves test output for inspection and will incur costs for cloud APIs. See [Ollama Provider Guide](OLLAMA_PROVIDER_GUIDE.md) for detailed Ollama setup and troubleshooting.

## Coverage Targets

- **Total tests:** ~230
- **Focus:** Complete user workflows, production-like scenarios
- **Line coverage (pytest E2E):** Full `podcast_scraper` package in the coverage denominator (same
  `pyproject.toml` `[tool.coverage.run]` as other tiers; no subtree `omit` file). Threshold and CI
  wiring: [Testing Guide — coverage thresholds](TESTING_GUIDE.md#coverage-thresholds). Roles of
  pytest E2E vs HTTP integration vs Playwright: [Testing Strategy — layer roles](../architecture/TESTING_STRATEGY.md#pytest-e2e-vs-http-integration-vs-browser-e2e-playwright).
