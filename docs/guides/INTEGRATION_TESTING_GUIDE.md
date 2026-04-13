# Integration Testing Guide

> **See also:**
>
> - [Testing Strategy](../architecture/TESTING_STRATEGY.md) - High-level testing philosophy and test pyramid
> - [Testing Guide](TESTING_GUIDE.md) - Quick reference and test execution commands

This guide covers integration test implementation: what to mock vs use real, component
interaction testing, and mocking guidelines.

## Overview

**Integration tests** test component interactions with limited mocking. Use real internal
implementations, mock external dependencies.

| Aspect | Requirement |
| -------- | ------------- |
| **Speed** | < 5 seconds per test |
| **Scope** | Multiple components working together |
| **Internal implementations** | Real (Config, factories, providers, workflow) |
| **Filesystem** | Real (temp directories) |
| **External HTTP** | Mocked (or local test server) |
| **ML/AI models and APIs** | Always mocked (real ML/AI is E2E only) |

## Mocking Philosophy

### Always Mock

1. **HTTP Requests** (External Network)

   ```python
   @patch("podcast_scraper.rss.downloader.fetch_url")
   def test_component_workflow(self, mock_fetch):
       mock_fetch.return_value = b"<rss>...</rss>"
       # Test component interactions
   ```

2. **External API Calls** (OpenAI, etc.)

   ```python
   @patch("podcast_scraper.providers.openai.openai_provider.OpenAI")
   def test_openai_provider_integration(self, mock_client):

       # Mock API client, test provider integration

### Always Mock: ML/AI models and APIs

**All ML models** (Whisper, spaCy, Transformers) and **all AI APIs** (OpenAI, Gemini,
Ollama, etc.) are **always mocked** in integration tests. Real ML inference and real
API calls belong exclusively in E2E tests.

`@pytest.mark.ml_models` must not appear on integration tests. If a test loads a real
ML model or calls a real AI API, it is an E2E test and belongs in `tests/e2e/`.

```python
# Integration: mock ML, test component wiring
@pytest.mark.integration
def test_config_to_provider_creation(self):
    with patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"):
        provider = create_transcription_provider(cfg)
        # Test provider creation, not ML execution

# Integration: mock summarization, test workflow logic
@pytest.mark.integration
def test_summarization_workflow(self):
    with patch("podcast_scraper.providers.ml.summarizer.SummaryModel") as mock_model:
        mock_model.return_value.summarize.return_value = {"summary": "test"}
        # Test workflow orchestration with mocked ML
```

## Never Mock

1. **Internal Implementations**
   - Config, factories, providers, RSS parser, metadata generation
   - These are what we're testing

2. **Filesystem I/O**
   - Use `tempfile.TemporaryDirectory` for isolation
   - Test actual file operations

3. **Component Interactions**
   - Provider → metadata, workflow → providers
   - This is the integration we're testing

## Real ML Models Belong in E2E Only

**Do not use real ML models in integration tests.** If a test loads a real ML model
(Whisper, spaCy, Transformers) or calls a real AI API (OpenAI, Gemini, Ollama), it
belongs in `tests/e2e/` with `@pytest.mark.e2e` and `@pytest.mark.ml_models`.

`make check-test-policy` (rule I1-ml-models-marker) enforces this automatically.

Integration tests verify how *our* components wire together. The ML/AI boundary is
always a mock or stub at this layer.

## Test Patterns

### Component Workflow Test

```python
@pytest.mark.integration
def test_rss_to_provider_workflow(self):
    """Test RSS parsing → Episode creation → Provider processing."""
    # Use real internal implementations
    feed = parse_rss_feed(rss_content)
    episodes = create_episodes(feed)

    # Mock external HTTP
    with patch("podcast_scraper.rss.downloader.fetch_url") as mock_fetch:
        mock_fetch.return_value = b"transcript content"
        result = process_episodes(episodes, cfg)

    assert result.success
```

### Provider Integration Test (mocked ML)

```python
@pytest.mark.integration
def test_transcription_workflow(self):
    """Test transcription provider wiring with mocked Whisper."""
    with patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"):
        provider = create_transcription_provider(cfg)
        provider.initialize()
        # Verify provider creation and lifecycle, not ML output
```

### Local HTTP Server Test

```python
@pytest.mark.integration
def test_http_client_behavior(self, local_http_server):
    """Test HTTP client with local server."""
    url = local_http_server.url_for("/test")
    response = http_get(url, user_agent, timeout)
    assert response.status_code == 200
```

## Model Cache Helpers (E2E only)

Real-ML tests live in `tests/e2e/` (not here). They use cache helpers to skip
gracefully when models are not downloaded:

```python
from tests.integration.ml_model_cache_helpers import (
    require_whisper_model_cached,
    require_transformers_model_cached,
    require_spacy_model_cached,
)

@pytest.mark.e2e
@pytest.mark.ml_models
def test_with_real_models(self):
    require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)
    require_transformers_model_cached(config.TEST_DEFAULT_SUMMARY_MODEL, None)
    # Test with real models...
```

## Directory Organization

Integration tests are organized by **domain subsystem** — the area of functionality
being exercised — not by source module. This differs from unit tests, which mirror
the `src/` tree 1:1.

**Why domain-based?** An integration test for "provider factory creates Ollama
provider, initializes, and summarizes" spans `config.py`, `summarization/factory.py`,
`providers/ollama/`, and `prompts/store.py`. No single source module owns it. The
right grouping is the subsystem under test.

```text
tests/integration/
├── providers/               # Provider factories, protocols, error handling
│   ├── llm/                # LLM provider integration (Anthropic, OpenAI, …)
│   ├── ml/                 # ML model loading, embedding, QA, NLI, summarizer
│   └── ollama/             # Ollama model-specific tests
├── workflow/                # Orchestration, stages, resume, parallelism, metadata
├── gi/                      # GI artifacts, KG artifacts, evidence stack
├── server/                  # FastAPI viewer: wired app, corpus library, index rebuild/stats
├── search/                  # FAISS indexing, corpus search
├── rss/                     # RSS parsing, HTTP fetching
├── eval/                    # Evaluation framework
├── infrastructure/          # Fixture mapping, infra concerns
├── tools/                   # CLI tools
└── (root)                   # Cross-cutting: filesystem, retry, cache, audio
```

**Rules of thumb:**

- A domain folder is created when **3+ test files** share a subsystem.
- Truly cross-cutting tests (filesystem helpers, retry, transcript cache,
  audio preprocessing) stay in root — they span multiple subsystems.
- Each folder has an `__init__.py` (empty) for pytest collection.

### Comparison with unit test layout

| Aspect | Unit tests | Integration tests |
| ------ | ---------- | ----------------- |
| **Axis** | Source module (mirrors `src/`) | Domain subsystem |
| **Depth** | Deep (matches package nesting) | Shallow (1–2 levels) |
| **Finding tests** | "Where's the test for this file?" | "Where are tests for this subsystem?" |
| **Duplication** | 1:1 with source files | One folder may cover many source files |

## Test Files by Domain

### providers/

| Subfolder | Purpose | Example files |
| --------- | ------- | ------------- |
| `llm/` | LLM provider integration | `test_anthropic_providers.py`, `test_openai_providers.py` |
| `ml/` | ML model loading, embedding, QA, NLI, summarizer | `test_embedding_loader_integration.py`, `test_summarizer_integration.py` |
| `ollama/` | Ollama model-specific tests | `test_gemma2_9b_summary.py`, `test_llama3_1_8b_speaker.py` |
| (root) | Cross-provider: factories, protocols, capabilities | `test_capabilities_integration.py`, `test_fallback_behavior.py` |

### workflow/

| Purpose | Example files |
| ------- | ------------- |
| Orchestration and stages | `test_workflow_integration.py`, `test_workflow_stages_integration.py` |
| Metadata generation | `test_metadata_integration.py`, `test_kg_metadata_integration.py` |
| Resume and parallelism | `test_resume_behavior.py`, `test_parallel_summarization.py` |
| Queue and MPS | `test_bounded_queue_integration.py`, `test_mps_exclusive_integration.py` |

### gi/

| Purpose | Example files |
| ------- | ------------- |
| GI artifacts | `test_gi_integration.py` |
| KG artifacts | `test_kg_integration.py` |
| Evidence stack | `test_evidence_stack_integration.py` |

### Root (cross-cutting)

| Purpose | File |
| ------- | ---- |
| Filesystem helpers | `test_filesystem_integration.py` |
| Retry with metrics | `test_retry_integration.py` |
| Transcript cache | `test_transcript_cache_integration.py` |
| Audio preprocessing | `test_audio_preprocessing_integration.py` |
| Summary schema | `test_summary_schema_integration.py` |
| Protocol verification | `test_protocol_verification_integration.py` |

## Real HTTP client integration (local server)

`tests/integration/rss/test_http_integration.py` exercises `podcast_scraper.rss.downloader`
against a **local** `http.server` on `127.0.0.1` (marker `integration_http`). There is no
external network; pytest allows localhost sockets for this suite.

**Global downloader state:** The module uses thread-local `requests.Session` objects with
urllib3 `Retry` adapters. Production defaults retry many times on 5xx with exponential
backoff, which can make a test that hits a handler returning only 500 look hung. This
file uses an autouse fixture that calls `configure_http_policy()`, caps retries with
`configure_downloader(...)`, and `downloader.reset_http_sessions()` so each test builds
sessions with bounded retries. Teardown clears downloader overrides.

If you add integration tests that call `fetch_url` / `fetch_rss_feed_url` for real HTTP,
reuse the same pattern (or mock HTTP). See [CONFIGURATION.md — Download resilience](../api/CONFIGURATION.md#download-resilience) (threading and metrics) for how configuration applies to sessions.

## Running Integration Tests

```bash
# All integration tests
make test-integration

# Fast (excludes ml_models)
make test-integration-fast

# Sequential (for debugging)
pytest tests/integration/ -n 0

# Specific domain
pytest tests/integration/providers/ -v
pytest tests/integration/workflow/ -v
pytest tests/integration/gi/ -v

# Specific test file
pytest tests/integration/workflow/test_component_workflows.py -v
```

## Test Markers

- `@pytest.mark.integration` -- Required for all integration tests
- `@pytest.mark.critical_path` -- Critical path tests (run in fast suite).
  See [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)

`@pytest.mark.ml_models` must **not** appear on integration tests (enforced by
`make check-test-policy`, rule I1). Real-ML tests belong in `tests/e2e/`.

## Provider Testing

For provider-specific integration testing (E2E server mock endpoints, provider switching):

→ **[Provider Implementation Guide - Testing Your Provider](PROVIDER_IMPLEMENTATION_GUIDE.md#testing-your-provider)**

Covers:

- Provider works with E2E server mock endpoints
- Provider switching tests
- Error handling in workflow context
- Integration test checklist for new providers

## Coverage Targets

- **Total tests:** ~530
- **Focus:** Critical paths, component interactions, edge cases
