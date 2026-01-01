# Testing Guide

> **See also:**
>
> - [Testing Strategy](../TESTING_STRATEGY.md) for the overall testing strategy, decision criteria, and test pyramid concepts
> - [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md) for what to test based on the critical path and prioritization

This guide provides detailed implementation instructions for working with the test suite. It covers how to run tests,
what test files exist, available fixtures, requirements, and coverage details.

## Table of Contents

1. [Unit Test Implementation](#unit-test-implementation)
2. [Integration Test Implementation](#integration-test-implementation)
3. [E2E Test Implementation](#e2e-test-implementation)
4. [Test Execution Details](#test-execution-details)
5. [Test Infrastructure Details](#test-infrastructure-details)
6. [Network and Filesystem Isolation](#network-and-filesystem-isolation)
7. [Decision Trees and Edge Cases](#decision-trees-and-edge-cases)
8. [Current State vs. Ideal State](#current-state-vs-ideal-state)
9. [Migration Guidelines](#migration-guidelines)

---

## Unit Test Implementation

### Unit Test Execution

**Run All Unit Tests**:

````bash

# Run all unit tests

pytest tests/unit/ -v

# Or using Makefile

make test-unit
```text
```bash

# Configuration tests

pytest tests/unit/podcast_scraper/test_config.py -v

# Filesystem tests

pytest tests/unit/podcast_scraper/test_filesystem.py -v

# RSS parser tests

pytest tests/unit/podcast_scraper/test_rss_parser.py -v

# Downloader tests

pytest tests/unit/podcast_scraper/test_downloader.py -v

# Service API tests

pytest tests/unit/podcast_scraper/test_service.py -v

# Summarizer tests

pytest tests/unit/podcast_scraper/test_summarizer.py -v

# Speaker detection tests

pytest tests/unit/podcast_scraper/test_speaker_detection.py -v

# Metadata tests

pytest tests/unit/podcast_scraper/test_metadata.py -v

# Provider tests

pytest tests/unit/podcast_scraper/speaker_detectors/test_speaker_detector_provider.py -v
pytest tests/unit/podcast_scraper/summarization/test_summarization_provider.py -v
pytest tests/unit/podcast_scraper/transcription/test_transcription_provider.py -v

# OpenAI provider tests

pytest tests/unit/podcast_scraper/test_openai_providers.py -v

# Prompt store tests

pytest tests/unit/podcast_scraper/test_prompt_store.py -v

# Infrastructure tests (isolation, imports, etc.)

pytest tests/unit/test_network_isolation.py -v
pytest tests/unit/test_filesystem_isolation.py -v
pytest tests/unit/test_package_imports.py -v
```text

- **`test_config.py`** - Configuration model validation, type coercion, defaults
- **`test_filesystem.py`** - Filename sanitization, output directory derivation
- **`test_rss_parser.py`** - RSS parsing, namespace handling, URL resolution
- **`test_downloader.py`** - HTTP client, retry logic, URL normalization
- **`test_service.py`** - Service API (`service.run()`, `ServiceResult`)
- **`test_metadata.py`** - Metadata generation (JSON/YAML)

**ML/AI Module Tests**:

- **`test_summarizer.py`** - Summarization logic, model selection, chunking
- **`test_summarizer_edge_cases.py`** - Edge cases in summarization
- **`test_summarizer_security.py`** - Security considerations in summarization
- **`test_speaker_detection.py`** - Speaker detection (NER), host/guest distinction

**Provider Tests**:

- **`test_speaker_detector_provider.py`** - Speaker detector provider protocol
- **`test_summarization_provider.py`** - Summarization provider protocol
- **`test_transcription_provider.py`** - Transcription provider protocol
- **`test_openai_providers.py`** - OpenAI provider implementations (transcription, speaker detection, summarization)

**Provider Testing Patterns:**

- **Unit Tests**: Test provider creation, initialization, protocol methods, error handling, cleanup
- **Mock Strategy**: Mock API clients (for API providers), mock ML models (for local providers)
- **Example**: `test_openai_providers.py` mocks `OpenAI` client, tests provider initialization, transcription/summarization/speaker detection methods

**Infrastructure Tests**:

- **`test_network_isolation.py`** - Network call blocking enforcement
- **`test_filesystem_isolation.py`** - Filesystem I/O blocking enforcement
- **`test_package_imports.py`** - Package structure and importability
- **`test_protocol_definitions.py`** - Protocol definitions and compliance
- **`test_utilities.py`** - Shared test utilities
- **`test_api_versioning.py`** - API versioning tests
- **`test_prompt_store.py`** - Prompt store functionality

### Unit Test Fixtures

**Network and Filesystem Isolation**:

Unit tests automatically enforce isolation via pytest plugins in `tests/unit/conftest.py`:

- **Network isolation**: Blocks all network calls (except localhost for cache)
- **Filesystem isolation**: Blocks all filesystem I/O (except tempfile operations)
- **Exceptions**: `tempfile` operations, cache directories, site-packages access

**Mock Fixtures**:

- **ML dependency mocks**: Mock `spacy`, `torch`, `transformers` before importing dependent modules
- **HTTP mocks**: Mock `requests.Session` and responses
- **Whisper mocks**: Mock `whisper.load_model()` and `whisper.transcribe()`

### Unit Test Requirements

- **No ML dependencies**: Unit tests must run without ML packages installed (for CI speed)
- **Network isolation**: All network calls are blocked (enforced by pytest plugin)
- **Filesystem isolation**: All filesystem I/O is blocked (enforced by pytest plugin, except tempfile)
- **Fast execution**: Tests should complete in < 100ms each
- **Full mocking**: All external dependencies must be mocked

### Unit Test Coverage

**Total: 200+ unit tests**

**Test Distribution**:

- Core modules (config, filesystem, rss_parser, downloader): ~50 tests
- Service API: ~15 tests
- Summarizer: ~70 tests (including edge cases and security)
- Speaker detection: ~20 tests
- Providers (speaker, summarization, transcription): ~30 tests
- OpenAI providers: ~15 tests
- Infrastructure (isolation, imports, protocols): ~10 tests

**Coverage Target**: >80% code coverage overall, >90% for critical modules

---

## Integration Test Implementation

### Integration Test Execution

**Run All Integration Tests**:

```bash

# Run all integration tests

pytest tests/integration/ -v -m integration

# Or using Makefile

make test-integration
```text
```bash

# Component workflow tests

pytest tests/integration/test_component_workflows.py -v -m integration

# Full pipeline tests

pytest tests/integration/test_full_pipeline.py -v -m integration

# HTTP integration tests

pytest tests/integration/test_http_integration.py -v -m integration

# Provider integration tests

pytest tests/integration/test_provider_integration.py -v -m integration

# Provider real models tests

pytest tests/integration/test_provider_real_models.py -v -m integration

# Protocol compliance tests

pytest tests/integration/test_protocol_compliance.py -v -m integration
pytest tests/integration/test_protocol_compliance_extended.py -v -m integration

# Provider error handling tests

pytest tests/integration/test_provider_error_handling_extended.py -v -m integration

# OpenAI provider integration tests

pytest tests/integration/test_openai_provider_integration.py -v -m integration
pytest tests/integration/test_openai_providers.py -v -m integration

# Pipeline concurrent execution tests

pytest tests/integration/test_pipeline_concurrent.py -v -m integration

# Pipeline error recovery tests

pytest tests/integration/test_pipeline_error_recovery.py -v -m integration

# Parallel summarization tests

pytest tests/integration/test_parallel_summarization.py -v -m integration

# Metadata integration tests

pytest tests/integration/test_metadata_integration.py -v -m integration

# Fallback behavior tests

pytest tests/integration/test_fallback_behavior.py -v -m integration

# Stage 0 foundation tests

pytest tests/integration/test_provider_config_integration.py -v -m integration
```text

- **`test_component_workflows.py`** - Component interactions and data flow
- **`test_full_pipeline.py`** - Full pipeline execution (mocked HTTP)
- **`test_http_integration.py`** - HTTP client behavior with local test server
- **`test_metadata_integration.py`** - Metadata generation in pipeline context

**Provider Integration Tests**:

- **`test_provider_integration.py`** - Provider factory and protocol compliance
- **`test_provider_real_models.py`** - Real ML models in provider context
- **`test_protocol_compliance.py`** - Protocol compliance verification
- **`test_protocol_compliance_extended.py`** - Extended protocol compliance
- **`test_provider_error_handling_extended.py`** - Provider error handling scenarios
- **`test_openai_provider_integration.py`** - OpenAI provider integration (with E2E server mock endpoints)
- **`test_openai_providers.py`** - OpenAI provider implementations

**Provider Integration Testing Patterns:**

- **Real Providers**: Use actual provider implementations (not mocks)
- **Mocked External Services**: Mock HTTP APIs or use E2E server mock endpoints
- **Component Interactions**: Test how providers work with other components (Config, workflow, etc.)
- **Example**: `test_openai_provider_integration.py` uses real OpenAI providers with E2E server mock endpoints

**Pipeline Integration Tests**:

- **`test_pipeline_concurrent.py`** - Concurrent execution in pipeline
- **`test_pipeline_error_recovery.py`** - Error recovery in pipeline context
- **`test_parallel_summarization.py`** - Parallel summarization execution

**Other Integration Tests**:

- **`test_fallback_behavior.py`** - Fallback mechanisms (Whisper, etc.)
- **`test_provider_config_integration.py`** - Provider configuration and factory integration tests

### Integration Test Fixtures

**Local HTTP Server**:

Integration tests can use local HTTP servers for HTTP client testing:

```python

# Example: Using local test server for HTTP testing

def test_http_integration(local_http_server):
    url = local_http_server.url_for("/test")

    # Test HTTP client with local server

```yaml

- **Temp directories**: `tempfile.TemporaryDirectory` for isolated test runs
- **Real file operations**: Read/write files, create directories
- **Real component interactions**: Use actual Config, providers, workflow logic

**Mock External Services**:

- **HTTP APIs**: Mocked for speed/reliability (or use local test server)
- **External APIs**: OpenAI, etc. are mocked
- **ML Models**: May be mocked for speed, or real for model integration testing

### Integration Test Requirements

- **ML dependencies**: Integration tests require ML packages installed (for real model testing)
- **Real filesystem I/O**: Tests use real file operations in temp directories
- **Mocked external services**: HTTP APIs and external services are mocked (or use local test server)
- **Moderate speed**: Tests should complete in < 5s each for fast tests
- **Real component logic**: Use actual internal implementations (Config, providers, workflow)

### Integration Test Coverage

**Total: 50+ integration tests**

**Test Distribution**:

- Component workflows: ~10 tests
- Full pipeline: ~5 tests
- HTTP integration: ~5 tests
- Provider integration: ~10 tests
- Protocol compliance: ~5 tests
- Pipeline concurrent/error recovery: ~5 tests
- OpenAI providers: ~5 tests
- Other (fallback, metadata, etc.): ~5 tests

**Coverage Focus**: Critical paths, component interactions, edge cases in component context

---

## E2E Test Implementation

### E2E Test Execution

**Run All E2E Tests**:

```bash

# Run all E2E tests with network guard

pytest tests/e2e/ -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost
```text
```bash

# Network guard tests

pytest tests/e2e/test_network_guard.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# OpenAI mock tests

pytest tests/e2e/test_openai_mock.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# E2E server tests

pytest tests/e2e/test_e2e_server.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# Fixture mapping tests

pytest tests/e2e/test_fixture_mapping.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# Basic E2E tests

pytest tests/e2e/test_basic_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# CLI E2E tests

pytest tests/e2e/test_cli_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# Library API E2E tests

pytest tests/e2e/test_library_api_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# Service API E2E tests

pytest tests/e2e/test_service_api_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# Whisper E2E tests

pytest tests/e2e/test_whisper_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# ML models E2E tests

pytest tests/e2e/test_ml_models_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# Error handling E2E tests

pytest tests/e2e/test_error_handling_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# Edge cases E2E tests

pytest tests/e2e/test_edge_cases_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost

# HTTP behaviors E2E tests

pytest tests/e2e/test_http_behaviors_e2e.py -v -m e2e --disable-socket --allow-hosts=127.0.0.1,localhost
```text

- **`test_network_guard.py`** (3 tests)
  - Verifies network guard blocks external network calls
  - Verifies localhost connections are allowed
  - Verifies requests library external calls are blocked

- **`test_openai_mock.py`** (3 tests)
  - Verifies OpenAI providers use E2E server mock endpoints
  - Verifies OpenAI transcription provider works with E2E server
  - Verifies OpenAI summarization provider works with E2E server
  - Verifies OpenAI speaker detector works with E2E server

- **`test_openai_provider_integration_e2e.py`** (multiple tests)
  - Tests OpenAI providers in full pipeline workflows
  - Uses E2E server mock endpoints (real HTTP client, mock API responses)
  - Tests transcription, speaker detection, and summarization providers

- **`test_e2e_server.py`** (8 tests)
  - Verifies E2E server starts and stops correctly
  - Verifies URL helpers work correctly
  - Verifies RSS feeds are served correctly
  - Verifies audio files are served correctly
  - Verifies transcript files are served correctly
  - Verifies 404 handling for missing files
  - Verifies path traversal protection
  - Verifies range request support (206 Partial Content)

- **`test_fixture_mapping.py`** (7 tests)
  - Verifies fixture structure exists
  - Verifies all RSS files exist
  - Verifies podcast mapping is correct
  - Verifies RSS `<guid>` matches filename pattern
  - Verifies RSS `<enclosure>` URLs point to correct audio files
  - Verifies RSS `<podcast:transcript>` URLs (if present)
  - Verifies all episodes have corresponding files

**Workflow Tests**:

- **`test_basic_e2e.py`** (3 tests)
  - Basic CLI transcript download using real HTTP client
  - Basic library API pipeline using real HTTP client
  - Basic service API run using real HTTP client

- **`test_cli_e2e.py`** (12 tests)
  - CLI command tests for various workflows

- **`test_library_api_e2e.py`** (8 tests)
  - Library API (`run_pipeline()`) tests

- **`test_service_api_e2e.py`** (15 tests)
  - Service API (`service.run()`, `service.run_from_config_file()`, `service.main()`) tests

- **`test_whisper_e2e.py`** (4 tests)
  - Real Whisper transcription in full workflow

- **`test_ml_models_e2e.py`** (6 tests)
  - Real ML models (spaCy, Transformers) in full workflow

- **`test_error_handling_e2e.py`** (12 tests)
  - Error handling scenarios in complete workflows

- **`test_edge_cases_e2e.py`** (9 tests)
  - Edge cases in complete workflows

- **`test_http_behaviors_e2e.py`** (13 tests)
  - HTTP client behaviors in full workflow context

### E2E Test Fixtures

**E2E Server Fixture**:

The `e2e_server` fixture provides a local HTTP server that serves test fixtures:

```python
def test_example(e2e_server):
    rss_url = e2e_server.urls.feed("podcast1")
    audio_url = e2e_server.urls.audio("p01_e01")
    transcript_url = e2e_server.urls.transcript("p01_e01")

    # Use URLs in tests...

```text

**E2E Server Mock Endpoints**:

For API providers (e.g., OpenAI), the E2E server provides mock endpoints that return realistic API responses. Tests configure providers to use `e2e_server.urls.openai_api_base()` instead of the production API, allowing tests to run without real API calls.

**Example**:
```python
def test_openai_provider(e2e_server):
    cfg = create_test_config(
        transcription_provider="openai",
        openai_api_key="sk-test123",
        openai_api_base=e2e_server.urls.openai_api_base(),  # Use E2E server
    )
    # Test uses real HTTP client but hits mock endpoints
```
**Mock Endpoints**:
- `/v1/chat/completions` - For summarization and speaker detection
- `/v1/audio/transcriptions` - For transcription

See `tests/e2e/fixtures/e2e_http_server.py` for implementation details.

### E2E Test Requirements

- `pytest-socket` must be installed (in dev dependencies)
- Tests must be run with `--disable-socket --allow-hosts=127.0.0.1,localhost`
- Network guard will fail tests if external network is accessed
- All E2E tests use real HTTP client with `e2e_server` fixture (no HTTP mocking)

### ML Model Defaults for Tests

The test suite uses **smaller, faster models** for speed, while the production app uses **quality models** by default. This distinction is intentional and documented.

**Test Defaults** (used in CI/local dev for speed):
- **Whisper**: `tiny.en` (smallest, fastest English-only model)
- **spaCy**: `en_core_web_sm` (smallest model, installed as dependency)
- **Transformers MAP**: `facebook/bart-base` (small, ~500MB, fast)
- **Transformers REDUCE**: `allenai/led-base-16384` (long-context, needed for combined summaries)

**Production Defaults** (used in app runtime for quality):
- **Whisper**: `base.en` (better quality, matches app config default)
- **spaCy**: `en_core_web_sm` (same as tests)
- **Transformers MAP**: `facebook/bart-large-cnn` (large, ~2GB, better quality)
- **Transformers REDUCE**: `allenai/led-base-16384` (same as tests)

**Preloading**:
The `make preload-ml-models` command preloads both test and production defaults to ensure all models are cached. This allows:

- Fast test execution (using small models)
- Production quality (using large models)
- Flexibility to switch between models

See [Issue #143](https://github.com/chipi/podcast_scraper/issues/143) for detailed analysis and implementation.

### E2E Test Coverage

**Total: 103+ tests passing (all E2E tests with real HTTP client)**

**Test Distribution**:

- Network guard + OpenAI mocking: 6 tests
- E2E server infrastructure: 8 tests
- Fixture mapping: 7 tests
- Basic E2E tests: 3 tests
- CLI E2E tests: 12 tests
- Library API E2E tests: 8 tests
- Service API E2E tests: 15 tests
- Whisper E2E tests: 4 tests
- ML models E2E tests: 6 tests
- Error handling E2E tests: 12 tests
- Edge cases E2E tests: 9 tests
- HTTP behaviors E2E tests: 13 tests

**Implementation Status**: All E2E test infrastructure and coverage is complete. All E2E tests use real HTTP client with `e2e_server` fixture (no HTTP mocking). See [RFC-019](rfc/RFC-019-e2e-test-improvements.md) for implementation details.

---

## Test Execution Details

### Default Execution

**Default (unit tests only - fast feedback):**

```bash

# Run unit tests only (default pytest behavior)

pytest

# Or explicitly:

pytest tests/unit/
```text
```bash

# Unit tests only

pytest tests/unit/
make test-unit

# Integration tests only

pytest tests/integration/ -m integration
make test-integration

# Workflow E2E tests only

pytest tests/e2e/ -m e2e
make test-e2e

# All tests (excluding network tests)

pytest -m "not network"
make test
```text

### Parallel Execution

Tests run in parallel by default for faster feedback (2-4x speedup for unit tests, 3.4x for integration tests). This matches CI behavior:

```bash

# Default: parallel execution (auto-detects CPU count)

make test-unit
pytest -n auto

# Sequential execution (slower but clearer output, useful for debugging)

make test-unit-sequential
pytest  # No -n flag

# Run with specific number of workers

pytest -n 4

# Test suite-specific targets

make test-unit            # Unit tests (parallel)
make test-integration     # Integration tests (parallel)
make test-e2e             # E2E tests (parallel)
make test                 # All tests (parallel)
```yaml

- **Debugging test failures**: Sequential output is easier to read and debug
- **Investigating flaky tests**: Sequential execution can help identify timing issues
- **Resource-constrained environments**: If your machine is low on CPU/memory

**Note:** Parallel execution creates `.coverage.*` files (one per worker process) which are automatically merged into `.coverage`. These files are gitignored and can be cleaned with `make clean`.

### Verifying Test Markers

After changing pytest configuration (especially `pyproject.toml` `addopts`), verify markers work:

```bash

# Should collect integration tests

pytest tests/integration/ -m integration --collect-only -q | wc -l

# Should collect e2e tests

pytest tests/e2e/ -m e2e --collect-only -q | wc -l

# Should collect unit tests (default)

pytest tests/unit/ --collect-only -q | wc -l
```yaml

- Integration tests: > 50
- Workflow E2E tests: > 20
- Unit tests: > 100

If any count is 0, check for marker conflicts in `pyproject.toml` `addopts`. See `docs/wip/TEST_INFRASTRUCTURE_VALIDATION.md` for details.

### Flaky Test Reruns

```bash

# Retry failed tests (2 retries, 1 second delay)

pytest --reruns 2 --reruns-delay 1
make test-reruns

# Combine with parallel execution

pytest -n auto --reruns 2 --reruns-delay 1
```text

# - Network calls → NetworkCallDetectedError

# - Filesystem I/O → FilesystemIODetectedError

# - Exceptions: tempfile operations, cache directories

pytest tests/unit/

# Integration tests: Real filesystem I/O allowed, network calls are MOCKED

# - Real file operations in temp directories ✅

# - Real component interactions ✅

# - Network calls are mocked (for speed/reliability) ❌

pytest tests/integration/ -m integration

# E2E tests: Real network and filesystem I/O allowed

# - Real network calls (marked with @pytest.mark.network) ✅

# - Real filesystem I/O ✅

# - Real ML models ✅

# Note: Some E2E tests may still use mocks for fast feedback

pytest tests/e2e/ -m e2e

# Run E2E tests with real network calls

pytest tests/e2e/ -m "e2e and network"

```text

- **Target**: >80% code coverage overall
- **Critical Modules**: >90% (config, workflow, episode_processor)
- **Coverage Tools**: `coverage.py` with HTML reports

---

## Test Infrastructure Details

### Test Framework

- **Primary**: pytest (with unittest compatibility)
- **Fixtures**: pytest fixtures for test setup and teardown
- **Markers**: pytest markers for test categorization

### Mocking Strategy

- **HTTP Requests**: `unittest.mock.patch` with `MockHTTPResponse` fixtures (unit/integration tests), E2E server for E2E tests
- **Whisper Library**: Mock `whisper.load_model()` and `whisper.transcribe()` (unit tests), real models (integration/E2E tests)
- **File System**: `tempfile.TemporaryDirectory` for isolated test runs
- **ML Dependencies (spacy, torch, transformers)**:
  - **Unit Tests**: Must mock ML dependencies before importing modules that use them
  - **Integration Tests**: Real ML dependencies required and installed
  - **Why**: Unit tests run without ML dependencies in CI for speed; modules that import ML deps at top level will fail
  - **Solution**: Mock ML modules in `sys.modules` before importing dependent modules
- **API Providers (OpenAI, etc.)**:
  - **Unit Tests**: Mock `OpenAI` client class
  - **Integration Tests**: Mock API clients or use E2E server mock endpoints
  - **E2E Tests**: Use E2E server mock endpoints (real HTTP client, mock API responses)
  - **Why**: Prevents costs, rate limits, and flakiness from real API calls
  - **Solution**: Configure providers with `openai_api_base=e2e_server.urls.openai_api_base()` in tests

### Test Fixtures

- **RSS XML Samples**: Various feed structures, namespaces, edge cases
- **HTTP Response Mocks**: Realistic headers, content types, error responses
- **Whisper Mocks**: Fake model objects, transcription results
- **Test Audio Files**: Small (< 10s) audio files for E2E Whisper tests

### Test Organization

The test suite is organized into three main categories:

- **`tests/unit/`** - Unit tests per module
  - Fast, isolated tests (< 100ms each)
  - Fully mocked dependencies
  - No network calls (enforced by pytest plugin)
  - No filesystem I/O (enforced by pytest plugin, except tempfile operations)
  - Mirrors `src/podcast_scraper/` structure
  - Example: `tests/unit/podcast_scraper/test_config.py`

- **`tests/integration/`** - Integration tests
  - Test component interactions between multiple modules
  - Use **real internal implementations** (real Config, real providers, real workflow logic)
  - Use **real filesystem I/O** (temp directories, real file operations)
  - **Mock external services** (HTTP APIs, external APIs) for speed and reliability
  - Test how components work together, not just individual units
  - Example: `tests/integration/test_provider_integration.py`

- **`tests/e2e/`** - Workflow end-to-end tests
  - Test complete workflows from entry point to output
  - Test CLI commands, service mode, full pipelines
  - **Use real HTTP client** (with local server, no external network)
  - **Use real filesystem I/O** (real file operations, real output directories)
  - **Use real ML models** (Whisper, transformers, etc.)
  - **Full system testing**: Tests the system as users would use it
  - Slowest tests (may take seconds to minutes)
  - Example: `tests/e2e/test_basic_e2e.py`

**Shared Test Utilities:**

- **`tests/conftest.py`** - Shared fixtures and test utilities available to all tests
- **`tests/unit/conftest.py`** - Network and filesystem I/O isolation enforcement for unit tests

### Test Markers

- `@pytest.mark.integration` - Integration tests (test component interactions)
- `@pytest.mark.e2e` - Workflow end-to-end tests (test complete workflows)
- `@pytest.mark.network` - Tests that hit the network (off by default)
- `@pytest.mark.slow` - Slow-running tests (existing)
- `@pytest.mark.whisper` - Requires Whisper dependency (existing)
- `@pytest.mark.spacy` - Requires spaCy dependency (existing)
- `@pytest.mark.ml_models` - Requires ML model dependencies

**Marker Usage:**

- All integration tests must have `@pytest.mark.integration`
- All e2e tests must have `@pytest.mark.e2e`
- Unit tests should NOT have integration/e2e markers
- **E2E tests that make real network calls** should have `@pytest.mark.network`
- **Integration tests** typically mock network calls (for speed), so they usually don't need `@pytest.mark.network`
- **E2E tests** should use real HTTP client and be marked with `@pytest.mark.e2e`

---

## Network and Filesystem Isolation

### Network Isolation

Unit tests are automatically prevented from making network calls. The pytest plugin (`tests/unit/conftest.py`) blocks common network libraries:

- `requests.get()`, `requests.post()`, `requests.Session()` methods
- `urllib.request.urlopen()`
- `urllib3.PoolManager()`
- `socket.create_connection()`

If a unit test attempts a network call, it fails with `NetworkCallDetectedError`. Integration and e2e tests are not affected by network isolation.

### Filesystem I/O Isolation

Unit tests are automatically prevented from performing filesystem I/O operations. The pytest plugin (`tests/unit/conftest.py`) blocks:

- `open()` for file operations (outside temp directories)
- `os.makedirs()`, `os.remove()`, `os.unlink()`, `os.rmdir()`, `os.rename()`, etc.
- `shutil.copy()`, `shutil.move()`, `shutil.rmtree()`, etc.
- `Path.write_text()`, `Path.write_bytes()`, `Path.mkdir()`, `Path.unlink()`, `Path.rmdir()`, etc.

If a unit test attempts filesystem I/O, it fails with `FilesystemIODetectedError`.

**Exceptions (allowed in unit tests):**

- **`tempfile` operations**: `tempfile.mkdtemp()`, `tempfile.NamedTemporaryFile()` (designed for testing)
- **Operations within temp directories**: Automatically detected and allowed
- **Cache directories**: `~/.cache/`, `~/.local/share/`, etc. (for model loading)
- **Site-packages**: Read-only access to installed packages (e.g., spaCy models)
- **Python cache files**: `.pyc`, `__pycache__/` (created during imports)
- **`test_filesystem.py`**: Tests that need to test filesystem operations

**Why filesystem I/O isolation?**

- Ensures unit tests are fast and isolated
- Prevents tests from affecting each other through filesystem state
- Forces proper mocking of file operations
- Makes tests more deterministic and reproducible

Integration and e2e tests are not affected by filesystem I/O isolation.

---

## Decision Trees and Edge Cases

### Decision Tree

```text

Start: What are you testing?

├─ Is it testing a complete user workflow (CLI command, library API call, service API call)?
│  └─ YES → E2E Test
│     └─ Does it use real HTTP client without mocking?
│        └─ YES → E2E Test
│        └─ NO → Still E2E Test (but consider using real HTTP client)
│
├─ Is it testing how multiple components work together?
│  └─ YES → Integration Test
│     └─ Does it test complete pipeline from entry to output?
│        └─ YES → E2E Test (if it's a user workflow)
│        └─ NO → Integration Test
│
├─ Is it testing component interactions (RSS parser → Episode → Provider)?
│  └─ YES → Integration Test
│
├─ Is it testing error handling in pipeline context?
│  └─ Does it test complete workflow with errors?
│     └─ YES → E2E Test
│     └─ NO → Integration Test (if focused on specific error scenarios)
│
└─ Is it testing concurrent execution, thread safety, resource sharing?
   └─ Does it test in full pipeline context?
      └─ YES → E2E Test
      └─ NO → Integration Test

```text
**Question**: Both integration and E2E tests use local HTTP servers. What's the difference?

**Answer**:

- **Integration Tests**: Use local HTTP server to test HTTP client behavior in isolation (e.g., `test_http_integration.py`). Focus is on HTTP client functionality, not full workflow.
- **E2E Tests**: Use local HTTP server to test HTTP client in full workflow context. Focus is on complete workflow with real HTTP client.

**Decision**: If testing HTTP client behavior in isolation → Integration Test. If testing HTTP client in complete workflow → E2E Test.

#### Edge Case 2: Pipeline Error Handling

**Question**: Should error handling tests be integration or E2E?

**Answer**:

- **Integration Test**: If testing specific error scenarios with mocked HTTP
- **E2E Test**: If testing error handling in complete workflow with real HTTP client and real data files

**Decision**: If testing error handling in complete workflow with real HTTP client → E2E Test. If testing specific error scenarios with mocked HTTP → Integration Test.

#### Edge Case 3: Concurrent Execution

**Question**: Is concurrent execution testing integration or E2E?

**Answer**:

- **Integration Test**: If testing concurrent execution behavior in isolation
- **E2E Test**: If testing concurrent execution in complete workflow with real HTTP client and real data files

**Decision**: If testing concurrent execution in complete workflow → E2E Test. If testing concurrent execution behavior in isolation → Integration Test.

#### Edge Case 4: Real ML Models

**Question**: Integration tests can use real ML models. How is that different from E2E tests?

**Answer**:

- **Integration Tests**: Use real ML models to test model integration (e.g., `test_provider_real_models.py`). Focus is on model loading, initialization, and basic functionality.
- **E2E Tests**: Use real ML models in complete workflow context. Focus is on models working together in full pipelines.

**Decision**: If testing model integration in isolation → Integration Test. If testing models in complete workflow → E2E Test.

---

## Current State vs. Ideal State

### Current Implementation

**Current state** (as of this writing):

- ✅ **Unit tests**: Correctly isolated, no I/O, fully mocked
- ✅ **Integration tests**: Use real internal implementations, real filesystem I/O, local HTTP server for HTTP testing
- ✅ **E2E tests**: Use real HTTP client with `e2e_server` fixture (no HTTP mocking)

**Implementation Status**: All E2E test infrastructure and coverage is complete. All E2E tests use real HTTP client with `e2e_server` fixture (no HTTP mocking).

### Ideal State (Target)

**Target state** (what we should work toward):

- ✅ **Unit tests**: Fully isolated, no I/O (current state is correct)
- ✅ **Integration tests**: Real internal implementations, real filesystem I/O, local HTTP server for HTTP testing, mocked external APIs (current state is correct)
- ✅ **E2E tests**: Real HTTP client (with local server, no external network), real data files, real ML models in full workflow context (current state is correct)

**Benefits of real network calls in E2E tests:**

- Tests the system as users actually use it
- Catches integration issues with real APIs
- Validates actual HTTP handling, timeouts, retries
- More confidence in production readiness

**Trade-offs:**

- Slower test execution
- Potential flakiness from network issues
- Requires internet connection or test servers
- May need retry logic for transient failures

---

## Migration Guidelines

### When to Move a Test from Integration to E2E

Move a test to E2E if:

1. It tests a complete user workflow (CLI command, library API call)
2. It uses real HTTP client without mocking
3. It uses real data files (RSS feeds, transcripts, audio)
4. It tests complete pipeline from entry to output

### When to Keep a Test as Integration

Keep a test as integration if:

1. It tests component interactions without full pipeline
2. It focuses on specific scenarios (error handling, edge cases)
3. It uses mocked HTTP for speed
4. It tests component behavior in isolation

### Summary

**Integration Tests** = Component interactions, fast feedback, mocked external services

**E2E Tests** = Complete user workflows, real HTTP client, real data files, real ML models in full context

**Key Question**: "Am I testing how components work together, or am I testing a complete user workflow?"

- **Components together** → Integration Test
- **Complete user workflow** → E2E Test

---

## References

- [Testing Strategy](../TESTING_STRATEGY.md) - Overall testing strategy and decision criteria
- [Critical Path Testing Guide](CRITICAL_PATH_TESTING_GUIDE.md) - What to test based on the critical path and prioritization
- Test structure reorganization: [RFC-018](rfc/RFC-018-test-structure-reorganization.md)
- E2E test improvements: [RFC-019](rfc/RFC-019-e2e-test-improvements.md)
- CI workflow: `.github/workflows/python-app.yml`
- Architecture: [ARCHITECTURE.md](../ARCHITECTURE.md) (Testing Notes section)
- Contributing guide: `CONTRIBUTING.md` (Testing Requirements section)
````
