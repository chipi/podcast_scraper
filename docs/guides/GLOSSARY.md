# Glossary

Key terms and concepts used throughout the podcast_scraper codebase and documentation.

---

## Architecture Terms

### Provider

A modular implementation of a specific capability (transcription, speaker detection,
summarization). Providers implement protocols and can be swapped without changing
calling code.

**Examples:** `WhisperTranscriptionProvider`, `OpenAITranscriptionProvider`

**See:** [Provider Implementation Guide](PROVIDER_IMPLEMENTATION_GUIDE.md)

### Protocol

A Python `typing.Protocol` that defines the interface a provider must implement.
Enables duck typing and dependency injection.

**Examples:** `TranscriptionProvider`, `SpeakerDetectionProvider`, `SummarizationProvider`

**See:** [Protocol Extension Guide](PROTOCOL_EXTENSION_GUIDE.md)

### Pipeline

The orchestrated sequence of operations that processes podcast episodes: RSS parsing →
transcript download → transcription → speaker detection → summarization → metadata.

**Entry point:** `run_pipeline()` in `src/podcast_scraper/__init__.py`

### Config

Pydantic model that holds all configuration options. Can be loaded from CLI arguments,
YAML/JSON files, or environment variables.

**See:** [Configuration Guide](../api/CONFIGURATION.md)

---

## Testing Terms

### Critical Path

The minimal set of tests that verify core functionality. Used for fast CI feedback.
Marked with `@pytest.mark.critical_path`.

**See:** [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md)

### E2E Test

End-to-end test that exercises the full system with real HTTP calls and actual ML
models. Uses the E2E test server for mocking external APIs.

**Marker:** `@pytest.mark.e2e`

**See:** [E2E Testing Guide](E2E_TESTING_GUIDE.md)

### E2E Server

A FastAPI test server (`tests/e2e/e2e_server.py`) that mocks external APIs (OpenAI,
RSS feeds) for E2E testing without hitting real endpoints.

### Integration Test

Test that verifies component interactions with mocked external dependencies. Uses
real internal code but mocks network, filesystem, and ML models.

**Marker:** `@pytest.mark.integration`

**See:** [Integration Testing Guide](INTEGRATION_TESTING_GUIDE.md)

### Unit Test

Test that verifies a single function or class in isolation. All dependencies are
mocked. Fast and deterministic.

**Marker:** `@pytest.mark.unit`

**See:** [Unit Testing Guide](UNIT_TESTING_GUIDE.md)

### Serial Marker

`@pytest.mark.serial` - Forces test to run sequentially (not in parallel). Used for
tests that consume significant memory or have shared state.

### Test Fixture

Pytest fixture that provides test data or setup. Defined in `conftest.py` files.

**Examples:** `sample_episode`, `mock_config`, `temp_output_dir`

---

## ML Terms

### Model Preloading

Process of downloading and caching ML models before running tests. Prevents network
calls during test execution.

**Command:** `make preload-ml-models`

**See:** [RFC-028](../rfc/RFC-028-ml-model-preloading-and-caching.md)

### Whisper

OpenAI's speech-to-text model used for transcription when podcasts don't provide
transcripts.

**Models:** `tiny`, `base`, `small`, `medium`, `large`

### spaCy

NLP library used for Named Entity Recognition (NER) in speaker detection.

**Model:** `en_core_web_sm`

### BART / LED

Transformer models used for summarization.

- **BART:** `facebook/bart-large-cnn` - Short document summarization
- **LED:** `allenai/led-base-16384` - Long document summarization

---

## CI/CD Terms

### Two-Tier Testing

CI strategy where PRs run fast critical-path tests, and main branch runs full test
suite.

**See:** [CI/CD](../ci/index.md)

### Path Filtering

GitHub Actions feature that only triggers workflows when specific files change.
Reduces unnecessary CI runs.

### Pre-commit Hook

Git hook that runs checks (formatting, linting) before allowing commits.

**Config:** `.pre-commit-config.yaml`

---

## File/Directory Terms

### `.test_outputs/`

Directory where test-generated files are written. Gitignored.

### `.build/`

Directory for build artifacts (docs site, distributions, coverage reports). Gitignored.

### `.cache/`

ML model cache directory within the project. Alternative to `~/.cache/`.

### `conftest.py`

Pytest configuration file containing fixtures and hooks. Can exist at multiple levels
(root, per-directory).

---

## Configuration Terms

### RSS Feed

XML feed that lists podcast episodes with metadata and media URLs.

### Podcast 2.0 Transcript Tag

RSS extension that provides transcript URLs directly in the feed XML.

### Output Directory

Where podcast_scraper writes downloaded transcripts, metadata, and summaries.

**CLI:** `--output-dir`

**Config:** `output_dir`

### Skip Existing

Option to skip processing episodes that already have output files.

**CLI:** `--skip-existing`

**Config:** `skip_existing: true`

---

## Code Style Terms

### Black

Python code formatter. Enforces consistent style.

**Line length:** 100 characters

### isort

Python import sorter. Groups and orders imports.

### flake8

Python linter for style and error checking.

### mypy

Static type checker for Python.

### markdownlint

Linter for Markdown files. Config in `.markdownlint.json`.

---

## Related Documentation

- [Architecture](../ARCHITECTURE.md) - System design overview
- [Development Guide](DEVELOPMENT_GUIDE.md) - Development workflow
- [Testing Guide](TESTING_GUIDE.md) - Test execution reference
