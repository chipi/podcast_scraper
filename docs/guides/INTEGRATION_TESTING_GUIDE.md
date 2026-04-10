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
| **ML Models** | Mocked for speed (unless testing ML workflow) |

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

### Conditionally Mock

**ML Models** (Whisper, spaCy, Transformers):

| Testing | Mock? | Marker |
| --------- | ------- | -------- |
| Non-ML workflows (config → provider creation) | ✅ Mock | None |
| ML workflow integration | ❌ Real | `@pytest.mark.ml_models` |

**Decision rule:** If test name contains "workflow" and involves ML → use real models.

```python

# Mock ML for speed (testing component wiring)

@pytest.mark.integration
def test_config_to_provider_creation(self):
    with patch("podcast_scraper.providers.ml.ml_provider._import_third_party_whisper"):
        provider = create_transcription_provider(cfg)
        # Test provider creation, not ML execution

# Real ML for workflow tests

@pytest.mark.integration
@pytest.mark.ml_models
def test_summarization_workflow(self):
    # Use real ML models for workflow testing
    summary_provider = create_summarization_provider(cfg)
    summary_provider.initialize()
    result = summary_provider.summarize(transcript)
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

## When to Use Real ML Models

Use real models with `@pytest.mark.ml_models` when:

1. Test is specifically testing ML workflow integration
2. Test name contains "workflow" and involves ML
3. Test validates actual model behavior
4. Test uses `require_*_model_cached()` helpers

Keep ML mocking when:

1. Testing non-ML component interactions
2. Testing error handling, configuration, or factory behavior
3. Test would be too slow with real models
4. Test doesn't need actual ML behavior

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

### Provider Integration Test

```python
@pytest.mark.integration
@pytest.mark.ml_models
def test_transcription_workflow(self):
    """Test real transcription provider in workflow."""
    require_whisper_model_cached(config.TEST_DEFAULT_WHISPER_MODEL)

    provider = create_transcription_provider(cfg)
    provider.initialize()
    try:
        result = provider.transcribe(audio_path)
        assert result.text
    finally:
        provider.cleanup()
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

## Model Cache Helpers

For tests using real ML models, use cache helpers to skip gracefully if models aren't cached:

```python
from tests.integration.ml_model_cache_helpers import (
    require_whisper_model_cached,
    require_transformers_model_cached,
    require_spacy_model_cached,
)

@pytest.mark.integration
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

- `@pytest.mark.integration` - Required for all integration tests
- `@pytest.mark.ml_models` - Tests requiring real ML models
- `@pytest.mark.critical_path` - Critical path tests (run in fast suite).
  See [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)

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
