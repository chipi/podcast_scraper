# Testing Strategy

## Overview

This document defines a comprehensive testing strategy that ensures reliability, maintainability, and confidence in the podcast scraper codebase. It consolidates testing approaches across all modules, establishes test infrastructure requirements, and outlines CI/CD integration patterns.

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

## Quick Reference: Test Type Decision

**For detailed decision framework, see `docs/wip/TEST_BOUNDARY_DECISION_FRAMEWORK.md`**

### Key Distinction

| Test Type | What It Tests | Entry Point | HTTP Client | Data Files | ML Models |
| --------- | ------------- | ----------- | ----------- | ---------- | --------- |
| **Unit** | Individual functions/modules | Function/class level | Mocked | Mocked | Mocked |
| **Integration** | Component interactions | Component level | Local test server (or mocked) | Test fixtures | Real (optional) |
| **E2E** | Complete user workflows | User level (CLI/API) | Real HTTP client (local server) | Real data files | Real (in workflow) |

### Decision Questions

1. **Am I testing a complete user workflow?** (CLI command, library API call, service API call)
   - **YES** → E2E Test
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

### Unit Tests

- **Purpose**: Test individual functions/modules in isolation
- **Speed**: Fast (< 100ms each)
- **Scope**: Single module, mocked dependencies
- **Coverage**: High (target: >80% code coverage)
- **Examples**: Config validation, filename sanitization, URL normalization

### Integration Tests

- **Purpose**: Test interactions between multiple modules/components (component interactions, data flow)
- **Speed**: Moderate (< 5s each for fast tests)
- **Scope**: Multiple modules working together, real internal implementations
- **Entry Point**: Component-level (functions, classes, not user-facing APIs)
- **I/O Policy**:
  - ✅ **Allowed**: Real filesystem I/O (temp directories), real component interactions
  - ❌ **Mocked**: External services (HTTP APIs, external APIs) - mocked for speed/reliability
  - ✅ **Optional**: Local HTTP server for HTTP client testing in isolation
- **Coverage**: Critical paths and edge cases, component interactions
- **Examples**: Provider factory → provider implementation, RSS parser → Episode → Provider → File output, HTTP client with local test server
- **Key Distinction**: Tests how components work together, not complete user workflows

### End-to-End Tests

- **Purpose**: Test complete user workflows from entry point to final output (CLI commands, library API calls, service API calls)
- **Speed**: Slow (< 60s each, may be minutes for full workflows)
- **Scope**: Full pipeline from entry point to output, real HTTP client, real data files, real ML models
- **Entry Point**: User-level (CLI commands, `run_pipeline()`, `service.run()`)
- **I/O Policy**:
  - ✅ **Allowed**: Real HTTP client with local HTTP server (no external network), real filesystem I/O, real data files
  - ✅ **Real implementations**: Use actual HTTP clients (no mocking), real file operations, real model loading
  - ✅ **Real data files**: RSS feeds, transcripts, audio files from `tests/fixtures/e2e_server/`
  - ❌ **No external network**: All HTTP calls go to local server (network guard prevents external calls)
- **Coverage**: Complete user workflows, production-like scenarios
- **Examples**: CLI command (`podcast-scraper <rss_url>`) → Full pipeline → Output files, Library API (`run_pipeline(config)`) → Full pipeline → Output files
- **Key Distinction**: Tests complete user workflows, not just component interactions

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

#### Whisper Integration (`whisper_integration.py`)

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
    - Test each model individually: `default`, `fast`, `small`, `pegasus`, `pegasus-xsum`, `long`, `long-fast`
    - Catch dependency issues (e.g., missing protobuf for PEGASUS models)
    - Verify model and tokenizer are properly initialized
    - Test model unloading after loading
  - **Map-reduce strategy**:
    - Map phase: chunking (word-based and token-based), chunk summarization
    - Reduce phase decision logic: single abstractive (≤800 tokens), mini map-reduce (800-4000 tokens), extractive (>4000 tokens)
    - Mini map-reduce: re-chunking combined summaries into 3-5 sections (650 words each), second map phase (summarize each section), final abstractive reduce
    - Extractive fallback behavior for extremely large combined summaries
  - Summary generation with various text lengths
  - Key takeaways extraction
  - Text chunking for long transcripts
  - Safe summarization error handling (OOM, missing dependencies)
  - Memory optimization (CUDA/MPS)
  - Model unloading and cleanup
  - Integration with metadata generation pipeline

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

### 3. End-to-End Tests

#### Current State

- **Issue #14**: Current tests rely heavily on mocks; lack realistic E2E tests
- **Issue #16**: No E2E tests for library API (`run_pipeline()`)

#### E2E Test Infrastructure (Issue #14)

**Test Fixtures & Mock Servers**:

- Local HTTP server (`http.server` or `pytest-httpserver`)
- Serves realistic RSS feeds with various transcript formats
- Serves transcript files (VTT, SRT, JSON, plain text)
- Serves small test audio files (< 10 seconds)

**Realistic Whisper Testing**:

- Use actual Whisper models (smallest available, e.g., `tiny.en`)
- Small test audio files for fast transcription
- Gate expensive tests behind flag (`--run-whisper-e2e`)

**Test Scenarios**:

- **Full pipeline**: RSS fetch → episode parsing → transcript download → file writing
- **Whisper fallback**: RSS fetch → no transcript → media download → Whisper transcription → file writing
- **Error handling**: Network failures, malformed RSS, missing files
- **Edge cases**: Special characters in titles, relative URLs, multiple transcript formats
- **Configuration**: Various CLI/config combinations end-to-end

**Test Infrastructure**:

- Mark E2E tests with `@pytest.mark.e2e`
- Make E2E tests optional in CI (run on schedule or manual trigger)
- Ensure tests are isolated (clean temp directories, no side effects)
- Add fixtures for common test scenarios

#### Library API E2E Tests (Issue #16)

**Test Scenarios**:

1. **Basic Library Usage**: `Config` + `run_pipeline()` for transcript download
2. **Library with Config File**: `load_config_file()` + `run_pipeline()`
3. **Library with Whisper**: `Config` with `transcribe_missing=True` + `run_pipeline()`
4. **Library Error Handling**: Invalid config, network failures

**Implementation**:

- Use same test fixtures/mock servers as CLI E2E tests
- Mark tests with `@pytest.mark.e2e`
- Ensure isolation from CLI tests
- Test both success and error paths
- Verify return values (`count`, `summary`) match documented API

## Test Infrastructure

### Test Framework

- **Primary**: `unittest` (Python standard library)
- **Future Consideration**: `pytest` for better fixtures and markers

### Mocking Strategy

- **HTTP Requests**: `unittest.mock.patch` with `MockHTTPResponse` fixtures
- **Whisper Library**: Mock `whisper.load_model()` and `whisper.transcribe()`
- **File System**: `tempfile.TemporaryDirectory` for isolated test runs
- **ML Dependencies (spacy, torch, transformers)**:
  - **Unit Tests**: Must mock ML dependencies before importing modules that use them
  - **Integration Tests**: Real ML dependencies required and installed
  - **Why**: Unit tests run without ML dependencies in CI for speed; modules that import ML deps at top level will fail
  - **Solution**: Mock ML modules in `sys.modules` before importing dependent modules

**Network and Filesystem I/O Isolation for Unit Tests:**

- Unit tests are automatically prevented from making network calls and filesystem I/O operations
- A pytest plugin (`tests/unit/conftest.py`) blocks:
  - **Network libraries:**
    - `requests.get()`, `requests.post()`, `requests.Session()` methods
    - `urllib.request.urlopen()`
    - `urllib3.PoolManager()`
    - `socket.create_connection()`
  - **Filesystem operations:**
    - `open()` for file operations (outside temp directories)
    - `os.makedirs()`, `os.remove()`, `os.unlink()`, `os.rmdir()`, etc.
    - `shutil.copy()`, `shutil.move()`, `shutil.rmtree()`, etc.
    - `Path.write_text()`, `Path.write_bytes()`, `Path.mkdir()`, `Path.unlink()`, etc.
- If a unit test attempts a network call, it fails with `NetworkCallDetectedError`
- If a unit test attempts filesystem I/O, it fails with `FilesystemIODetectedError`
- **Exceptions (allowed in unit tests):**
  - `tempfile.mkdtemp()`, `tempfile.NamedTemporaryFile()` (designed for testing)
  - Operations within temp directories (detected automatically)
  - Cache directories (`~/.cache/`, `~/.local/share/`) for model loading
  - Site-packages (read-only access to installed packages)
  - Python cache files (`.pyc`, `__pycache__`) created during imports
  - `test_filesystem.py` tests (they need to test filesystem operations)
- Integration and workflow_e2e tests are not affected by network/filesystem isolation

### Test Fixtures

- **RSS XML Samples**: Various feed structures, namespaces, edge cases
- **HTTP Response Mocks**: Realistic headers, content types, error responses
- **Whisper Mocks**: Fake model objects, transcription results
- **Test Audio Files**: Small (< 10s) audio files for E2E Whisper tests

### Test Organization

The test suite is organized into three main categories (RFC-018):

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

- **`tests/workflow_e2e/`** - Workflow end-to-end tests
  - Test complete workflows from entry point to output
  - Test CLI commands, service mode, full pipelines
  - **Use real network calls** (marked with `@pytest.mark.network` for tests that hit real APIs)
  - **Use real filesystem I/O** (real file operations, real output directories)
  - **Use real ML models** (Whisper, transformers, etc.)
  - **Full system testing**: Tests the system as users would use it
  - Slowest tests (may take seconds to minutes)
  - **Note**: Some E2E tests may still use mocks for fast feedback, but full E2E tests should use real dependencies
  - Example: `tests/workflow_e2e/test_workflow_e2e.py`

**Shared Test Utilities:**

- **`tests/conftest.py`** - Shared fixtures and test utilities available to all tests
- **`tests/unit/conftest.py`** - Network and filesystem I/O isolation enforcement for unit tests

### Test Markers

- `@pytest.mark.integration` - Integration tests (test component interactions)
- `@pytest.mark.workflow_e2e` - Workflow end-to-end tests (test complete workflows)
- `@pytest.mark.network` - Tests that hit the network (off by default)
- `@pytest.mark.slow` - Slow-running tests (existing)
- `@pytest.mark.whisper` - Requires Whisper dependency (existing)
- `@pytest.mark.spacy` - Requires spaCy dependency (existing)

**Marker Usage:**

- All integration tests must have `@pytest.mark.integration`
- All workflow_e2e tests must have `@pytest.mark.workflow_e2e`
- Unit tests should NOT have integration/workflow_e2e markers
- **E2E tests that make real network calls** should have `@pytest.mark.network`
- **Integration tests** typically mock network calls (for speed), so they usually don't need `@pytest.mark.network`
- **E2E tests** should use real network calls and be marked with `@pytest.mark.network` when they do

## CI/CD Integration

### Continuous Integration Strategy

**On Every PR** (GitHub Actions):

- **`test-unit` job**: Fast unit tests (no ML deps), network and filesystem I/O isolation enforced, parallel execution
- **`test-integration` job**: Integration tests with ML dependencies, parallel execution, flaky test reruns
- **`test` job**: Full test suite (unit + integration) with coverage for PRs
- **`lint` job**: Formatting, linting, type checking, security scans
- **`docs` job**: Documentation build
- **`build` job**: Package build validation

**On Main Branch**:

- **`test-unit` job**: Unit tests (same as PR)
- **`test-integration` job**: Integration tests (same as PR)
- **`test-workflow-e2e` job**: Workflow E2E tests (runs only on main branch), parallel execution, flaky test reruns
- All other jobs run as on PRs

**Test Execution Strategy**:

- **Unit tests**: Run on every PR and push (fast feedback, ~30 seconds)
- **Integration tests**: Run on every PR and push (moderate speed, ~2-5 minutes)
- **Workflow E2E tests**: Run only on main branch pushes (slowest, ~5-10 minutes)
- **Parallel execution**: Enabled for all test jobs (`-n auto`)
- **Flaky test reruns**: Enabled for integration and workflow_e2e tests (`--reruns 2 --reruns-delay 1`)
- **Network isolation**: Enforced for unit tests only (automatic failure if network call detected)
  - Integration tests: Network calls are mocked (for speed/reliability)
  - E2E tests: Network calls are allowed (marked with `@pytest.mark.network`)
- **Filesystem I/O isolation**: Enforced for unit tests only (automatic failure if filesystem I/O detected, except tempfile operations)
  - Integration tests: Real filesystem I/O allowed (temp directories, real file operations)
  - E2E tests: Real filesystem I/O allowed (full file operations, real output directories)

### Test Execution

**Default (unit tests only - fast feedback):**

```bash
# Run unit tests only (default pytest behavior)
pytest

# Or explicitly:
pytest tests/unit/
```

**By test type:**

```bash
# Unit tests only
pytest tests/unit/
make test-unit

# Integration tests only
pytest tests/integration/ -m integration
make test-integration

# Workflow E2E tests only
pytest tests/workflow_e2e/ -m workflow_e2e
make test-workflow-e2e

# All tests (excluding network tests)
pytest -m "not network"
make test-all

# Network tests only (requires internet connection)
pytest -m network
make test-network
```

**Parallel execution (faster feedback):**

```bash
# Run tests in parallel (auto-detects CPU count)
pytest -n auto
make test-parallel

# Run with specific number of workers
pytest -n 4
```

**Verifying Marker Behavior:**

After changing pytest configuration (especially `pyproject.toml` `addopts`), verify markers work:

```bash
# Should collect integration tests
pytest tests/integration/ -m integration --collect-only -q | wc -l

# Should collect workflow_e2e tests
pytest tests/workflow_e2e/ -m workflow_e2e --collect-only -q | wc -l

# Should collect unit tests (default)
pytest tests/unit/ --collect-only -q | wc -l
```

**Expected minimum counts:**

- Integration tests: > 50
- Workflow E2E tests: > 20
- Unit tests: > 100

If any count is 0, check for marker conflicts in `pyproject.toml` `addopts`. See `docs/wip/TEST_INFRASTRUCTURE_VALIDATION.md` for details.

**Flaky test reruns:**

```bash
# Retry failed tests (2 retries, 1 second delay)
pytest --reruns 2 --reruns-delay 1
make test-reruns

# Combine with parallel execution
pytest -n auto --reruns 2 --reruns-delay 1
```

**Network and Filesystem I/O Policy:**

```bash
# Unit tests: Network and filesystem I/O are BLOCKED
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
pytest tests/workflow_e2e/ -m workflow_e2e

# Run E2E tests with real network calls
pytest tests/workflow_e2e/ -m "workflow_e2e and network"
```

## Test Boundary Decision Framework

For clear guidance on deciding whether a test should be an Integration Test or E2E Test, see:

- **`docs/wip/TEST_BOUNDARY_DECISION_FRAMEWORK.md`** - Comprehensive decision framework with criteria, decision trees, and examples

**Quick Reference**:

- **Integration Tests**: Test how components work together (component interactions, data flow)
- **E2E Tests**: Test complete user workflows (CLI commands, library API calls, full pipelines)

**Key Question**: "Am I testing how components work together, or am I testing a complete user workflow?"

- Components together → Integration Test
- Complete user workflow → E2E Test

## Current State vs. Ideal State

### Current Implementation

**Current state** (as of this writing):

- ✅ **Unit tests**: Correctly isolated, no I/O, fully mocked
- ✅ **Integration tests**: Use real internal implementations, real filesystem I/O, local HTTP server for HTTP testing
- ⚠️ **E2E tests**: Currently use mocked HTTP responses (`MockHTTPResponse`, `@patch` decorators) - **needs migration to real HTTP client**

**Why E2E tests currently use mocks:**

- Historical: Tests were written with mocks for speed and reliability
- Practical: Avoids flakiness from external services
- Trade-off: Faster tests but less realistic

### Ideal State (Target)

**Target state** (what we should work toward):

- ✅ **Unit tests**: Fully isolated, no I/O (current state is correct)
- ✅ **Integration tests**: Real internal implementations, real filesystem I/O, local HTTP server for HTTP testing, mocked external APIs (current state is correct)
- 🎯 **E2E tests**: Real HTTP client (with local server, no external network), real data files, real ML models in full workflow context (needs work - see `docs/wip/E2E_TEST_GAPS.md`)

**How to migrate E2E tests to real network calls:**

1. Mark E2E tests that should use real network with `@pytest.mark.network`
2. Remove `@patch` decorators for HTTP calls in those tests
3. Use real RSS feeds or test servers for network tests
4. Keep some fast E2E tests with mocks for quick feedback
5. Run full E2E tests with `pytest -m "workflow_e2e and network"` for comprehensive testing

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

### Network Isolation

Unit tests are automatically prevented from making network calls. The pytest plugin (`tests/unit/conftest.py`) blocks common network libraries:

- `requests.get()`, `requests.post()`, `requests.Session()` methods
- `urllib.request.urlopen()`
- `urllib3.PoolManager()`
- `socket.create_connection()`

If a unit test attempts a network call, it fails with `NetworkCallDetectedError`. Integration and workflow_e2e tests are not affected by network isolation.

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

Integration and workflow_e2e tests are not affected by filesystem I/O isolation.

**Coverage:**

```bash
# Run tests with coverage report
pytest --cov=podcast_scraper --cov-report=term-missing
make test
```

### Coverage Requirements

- **Target**: >80% code coverage overall
- **Critical Modules**: >90% (config, workflow, episode_processor)
- **Coverage Tools**: `coverage.py` with HTML reports

## Testing Patterns

### Dependency Injection

- **CLI Testing**: Use `cli.main()` override callables (`apply_log_level_fn`, `run_pipeline_fn`, `logger`)
- **Workflow Testing**: Mock `run_pipeline` for CLI-focused tests
- **Benefit**: Test CLI behavior without executing full pipeline

### Mocking External Dependencies

- **HTTP**: Mock `requests.Session` and responses
- **Whisper**: Mock `whisper.load_model()` and `whisper.transcribe()`
- **ML Dependencies (spacy, torch, transformers)**:
  - **Unit Tests**: Mock in `sys.modules` before importing dependent modules
  - **Integration Tests**: Real ML dependencies required
  - **Verification**: CI runs `scripts/check_unit_test_imports.py` to ensure modules can import without ML deps
- **File System**: Use `tempfile` for isolated test environments

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

### `whisper_integration.py`

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

- [x] `ServiceResult` dataclass (success/failure states, attributes)
- [x] `service.run()` with valid Config (success path)
- [x] `service.run()` with logging configuration
- [x] `service.run()` exception handling (returns failed ServiceResult)
- [x] `service.run_from_config_file()` with JSON config
- [x] `service.run_from_config_file()` with YAML config
- [x] `service.run_from_config_file()` with missing file (returns failed ServiceResult)
- [x] `service.run_from_config_file()` with invalid config (returns failed ServiceResult)
- [x] `service.run_from_config_file()` with Path objects
- [x] `service.main()` CLI entry point (success/failure exit codes)
- [x] `service.main()` version flag handling
- [x] `service.main()` missing config argument handling
- [x] Service API importability via `__getattr__`
- [x] ServiceResult equality and string representation
- [x] Integration with public API (`Config`, `load_config_file`, `run_pipeline`)

## Future Testing Enhancements

### E2E Test Infrastructure Improvements (Issue #14)

- [ ] Local HTTP test server
- [ ] Test audio file fixtures
- [ ] Real Whisper integration tests
- [ ] Test markers and CI integration

### Library API Tests (Issue #16)

- [ ] `run_pipeline()` E2E tests
- [ ] `load_config_file()` tests
- [ ] Error handling tests
- [ ] Return value validation

### Performance Testing

- [ ] Benchmark large feed processing (1000+ episodes)
- [ ] Measure Whisper transcription performance
- [ ] Profile memory usage
- [ ] Test concurrent download limits

### Property-Based Testing

- [ ] Generate random RSS feeds
- [ ] Test filename sanitization with fuzzing
- [ ] Test URL normalization with edge cases

## References

- Test structure reorganization: `docs/rfc/RFC-018-test-structure-reorganization.md`
- CI workflow: `.github/workflows/python-app.yml`
- Related RFCs: RFC-001 through RFC-018 (testing strategies and reorganization)
- Related Issues: #14 (E2E testing), #16 (Library API E2E tests), #94 (src/ layout), #98 (Test structure reorganization)
- Architecture: `docs/ARCHITECTURE.md` (Testing Notes section)
- Contributing guide: `CONTRIBUTING.md` (Testing Requirements section)
