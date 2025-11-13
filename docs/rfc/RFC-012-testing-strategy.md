# RFC-012: Testing Strategy

- **Status**: Draft
- **Authors**: GPT-5 Codex
- **Stakeholders**: Maintainers, contributors, CI/CD pipeline owners
- **Related PRDs**: `docs/prd/PRD-001-transcript-pipeline.md`, `docs/prd/PRD-002-whisper-fallback.md`, `docs/prd/PRD-003-user-interface-config.md`
- **Related Issues**: #14 (E2E testing), #16 (Library API E2E tests)

## Abstract

Define a comprehensive testing strategy that ensures reliability, maintainability, and confidence in the podcast scraper codebase. This RFC consolidates testing approaches across all modules, establishes test infrastructure requirements, and outlines CI/CD integration patterns.

## Problem Statement

Testing requirements and strategies are currently scattered across individual RFCs, making it difficult to:
- Understand the overall testing approach
- Ensure consistent testing patterns across modules
- Plan new test infrastructure
- Onboard new contributors to testing practices
- Track testing coverage and requirements

A unified testing strategy document provides a single source of truth for all testing decisions and requirements.

## Test Pyramid

The testing strategy follows a three-tier pyramid:

```
        /\
       /E2E\          ← Few, realistic end-to-end tests
      /------\
     /Integration\    ← Moderate, focused integration tests
    /------------\
   /    Unit      \   ← Many, fast unit tests
  /----------------\
```

### Unit Tests
- **Purpose**: Test individual functions/modules in isolation
- **Speed**: Fast (< 100ms each)
- **Scope**: Single module, mocked dependencies
- **Coverage**: High (target: >80% code coverage)
- **Examples**: Config validation, filename sanitization, URL normalization

### Integration Tests
- **Purpose**: Test interactions between modules with realistic data
- **Speed**: Moderate (< 5s each)
- **Scope**: Multiple modules, mocked external dependencies (HTTP, Whisper)
- **Coverage**: Critical paths and edge cases
- **Examples**: CLI argument parsing → Config → workflow execution, RSS parsing → episode creation

### End-to-End Tests
- **Purpose**: Test complete workflows with real dependencies
- **Speed**: Slow (< 60s each)
- **Scope**: Full pipeline, real HTTP servers, optional real Whisper
- **Coverage**: Happy paths and critical user scenarios
- **Examples**: Full transcript download pipeline, Whisper transcription with real audio

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

#### Whisper Integration (`whisper.py`)
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
- **spaCy**: Mock NER extraction for unit tests; optional real spaCy for integration

### Test Fixtures
- **RSS XML Samples**: Various feed structures, namespaces, edge cases
- **HTTP Response Mocks**: Realistic headers, content types, error responses
- **Whisper Mocks**: Fake model objects, transcription results
- **Test Audio Files**: Small (< 10s) audio files for E2E Whisper tests

### Test Organization
- **Current**: `test_podcast_scraper.py` (integration-focused suite)
- **Future**: Consider splitting into:
  - `tests/unit/` - Unit tests per module
  - `tests/integration/` - Integration tests
  - `tests/e2e/` - End-to-end tests
  - `tests/fixtures/` - Shared test fixtures

### Test Markers
- `@pytest.mark.e2e` - End-to-end tests (optional in CI)
- `@pytest.mark.whisper` - Requires Whisper dependency
- `@pytest.mark.spacy` - Requires spaCy dependency
- `@pytest.mark.slow` - Slow-running tests

## CI/CD Integration

### Continuous Integration Strategy

**On Every PR**:
- Run all unit tests
- Run integration tests (with mocks)
- Run linting and type checking
- Skip E2E tests (too slow for PR feedback)

**On Main Branch / Scheduled**:
- Run full test suite including E2E tests
- Run E2E tests with real Whisper (if available)
- Generate coverage reports

**Manual Trigger**:
- Full E2E test suite with `--run-whisper-e2e` flag
- Performance benchmarks

### Test Execution
```bash
# Unit and integration tests (fast, always run)
python -m pytest tests/unit tests/integration

# E2E tests (slow, optional)
pytest tests/e2e -m e2e

# Full suite with Whisper
pytest --run-whisper-e2e
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
- **spaCy**: Mock NER extraction for unit tests
- **File System**: Use `tempfile` for isolated test environments

### Test Isolation
- Each test uses `tempfile.TemporaryDirectory` for output
- Tests clean up after themselves
- No shared state between tests
- Mock external services (HTTP, file system)

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

### `whisper.py`
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

## Future Testing Enhancements

### E2E Test Infrastructure (Issue #14)
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

- Current test suite: `test_podcast_scraper.py`
- CI workflow: `.github/workflows/python-app.yml`
- Related RFCs: RFC-001 through RFC-010 (individual testing strategies)
- Related Issues: #14 (E2E testing), #16 (Library API E2E tests)
- Architecture: `docs/ARCHITECTURE.md` (Testing Notes section)

