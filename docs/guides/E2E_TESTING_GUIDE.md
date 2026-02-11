# E2E Testing Guide

> **See also:**
>
> - [Testing Strategy](../TESTING_STRATEGY.md) - High-level testing philosophy and test pyramid
> - [Testing Guide](TESTING_GUIDE.md) - Quick reference and test execution commands

This guide covers E2E test implementation: real HTTP client, E2E server, ML model usage, and OpenAI mock endpoints.

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

## Core Principle: No Mocking

E2E tests use **real implementations throughout**:

- ✅ Real HTTP client (with local server)
- ✅ Real filesystem I/O
- ✅ Real ML models (Whisper, spaCy, Transformers)
- ✅ Real providers (MLProvider, OpenAIProvider)
- ❌ No external network (blocked by network guard)
- ❌ No Whisper mocks
- ❌ No ML model mocks

## E2E Server

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

### E2E Feeds (RSS)

Feed names and RSS file mapping. Which feed name you can use depends on **test mode** (see [Test Modes](#test-modes)).

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
| `podcast9_solo` | `p09_biohacking.xml` | Solo speaker (host only) |
| `podcast7_sustainability` | `p07_sustainability.xml` | Long-form (~15k words; Issue #283) |
| `podcast8_solar` | `p08_solar.xml` | Long-form (~20k words; Issue #283) |

**Fast fixtures** (used in `fast` and `multi_episode` when `set_use_fast_fixtures(True)`; mapping from `PODCAST_RSS_MAP_FAST`):

| Feed name | RSS file | Description |
| --------- | -------- | ----------- |
| `podcast1` | `p01_fast.xml` | 1 short episode (Path 2: transcription) |
| `podcast1_with_transcript` | `p01_fast_with_transcript.xml` | 1 episode with transcript URL (Path 1: download) |
| `podcast1_multi_episode` | `p01_multi.xml` | Same 5-episode feed |
| `podcast9_solo` | `p09_biohacking.xml` | Solo speaker |
| `podcast7_sustainability` | `p07_sustainability.xml` | Long-form |
| `podcast8_solar` | `p08_solar.xml` | Long-form |

**Allowed feeds per test mode** (set automatically by `conftest` from `E2E_TEST_MODE`):

| Mode | Allowed feed names |
| ---- | ------------------ |
| `fast` | `podcast1`, `podcast1_with_transcript`, `podcast1_multi_episode`, `podcast9_solo`, `podcast7_sustainability`, `podcast8_solar` |
| `multi_episode` | `podcast1_multi_episode`, `podcast1_with_transcript`, `edgecases`, `podcast7_sustainability`, `podcast8_solar` |
| `nightly` | `podcast1`, `podcast2`, `podcast3`, `podcast4`, `podcast5` (full fixtures) |
| `data_quality` | All feeds (None = allow all) |

Use `e2e_server.urls.feed("podcast1_multi_episode")` etc. Only feeds in the allowed set for the current mode are served; others return 404.

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
| `/v1/chat/completions` | Summarization and speaker detection |
| `/v1/audio/transcriptions` | Transcription |

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

# Fast (excludes ml_models)

make test-e2e-fast

# Sequential (for debugging)

pytest tests/e2e/ -n 0

# Specific test file

pytest tests/e2e/test_basic_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost
```

## Test Markers

- `@pytest.mark.e2e` - Required for all E2E tests
- `@pytest.mark.ml_models` - Tests requiring real ML models
- `@pytest.mark.critical_path` - Critical path tests (run in fast suite).
  See [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)

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

- **Total tests:** 100+
- **Focus:** Complete user workflows, production-like scenarios
