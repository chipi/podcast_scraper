# Testing Strategy

> **See also:** [Testing Guide](TESTING_GUIDE.md) for detailed implementation instructions, test execution commands, test file descriptions, fixtures, and coverage details.

## Overview

This document defines the testing strategy for the podcast scraper codebase. It establishes the test pyramid approach, decision criteria for choosing test types, and high-level testing patterns. For detailed implementation instructions on how to run tests, what test files exist, and how to work with the test suite, see the [Testing Guide](TESTING_GUIDE.md).

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

## Decision Criteria

The decision questions above provide a quick way to determine test type. For detailed decision trees, edge cases, and migration guidelines, see [Testing Guide - Decision Trees and Edge Cases](TESTING_GUIDE.md#decision-trees-and-edge-cases).

**Quick Reference:**

- **Unit Test**: Single function/module in isolation, all dependencies mocked
- **Integration Test**: Multiple components working together, real internal implementations, mocked external services
- **E2E Test**: Complete user workflow from entry point to output, real HTTP client, real data files, real ML models

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

**For detailed unit test execution commands, test file descriptions, fixtures, requirements, and coverage, see [Testing Guide - Unit Test Implementation](TESTING_GUIDE.md#unit-test-implementation).**

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

**For detailed integration test execution commands, test file descriptions, fixtures, requirements, and coverage, see [Testing Guide - Integration Test Implementation](TESTING_GUIDE.md#integration-test-implementation).**

### 3. End-to-End Tests

#### E2E Test Coverage Goals

**Every major user-facing entry point should have at least one E2E test:**

1. **CLI Commands** - Each main CLI command should have E2E tests
2. **Library API Endpoints** - Each public API function should have E2E tests
3. **Critical User Scenarios** - Important workflows should have E2E tests

**What Doesn't Need E2E Tests:**

- Not every CLI flag combination needs an E2E test
- Every possible configuration value (tested in integration/unit tests)
- Edge cases in specific components (tested in integration tests)

**Rule of Thumb**: E2E tests should cover "as a user, I want to..." scenarios, not every possible configuration combination.

**For detailed E2E test execution commands, test file descriptions, fixtures, requirements, and coverage, see [Testing Guide - E2E Test Implementation](TESTING_GUIDE.md#e2e-test-implementation).**

## Test Infrastructure

### Test Framework

- **Primary**: pytest (with unittest compatibility)
- **Fixtures**: pytest fixtures for test setup and teardown
- **Markers**: pytest markers for test categorization

### Mocking Strategy

- **Unit Tests**: Mock all external dependencies (HTTP, ML models, file system)
- **Integration Tests**: Mock external services (HTTP APIs, external APIs), use real internal implementations
- **E2E Tests**: Use real implementations (HTTP client, ML models, file system) with local test server

### Test Organization

The test suite is organized into three main categories:

- **`tests/unit/`** - Unit tests per module (fast, isolated, fully mocked)
- **`tests/integration/`** - Integration tests (component interactions, real internal implementations, mocked external services)
- **`tests/workflow_e2e/`** - E2E tests (complete workflows, real HTTP client, real ML models)

### Test Markers

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.workflow_e2e` - E2E tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.ml_models` - Requires ML model dependencies

**For detailed test infrastructure information, including network/filesystem isolation, fixtures, and marker usage, see [Testing Guide - Test Infrastructure Details](TESTING_GUIDE.md#test-infrastructure-details).**

## CI/CD Integration

### Continuous Integration Strategy

**On Every PR** (GitHub Actions):

- **`test-unit` job**: Fast unit tests (no ML deps), network and filesystem I/O isolation enforced, parallel execution
- **`test-integration` job**: Integration tests with ML dependencies, parallel execution, flaky test reruns
- **`test-e2e-fast` job**: Fast E2E tests (excludes slow/ml_models), network guard active
- **`lint` job**: Formatting, linting, type checking, security scans
- **`docs` job**: Documentation build
- **`build` job**: Package build validation

**On Main Branch**:

- **`test-e2e-slow` job**: All E2E tests including slow/ml_models tests (runs only on main branch)
- All other jobs run as on PRs

**Test Execution Strategy**:

- **Unit tests**: Run on every PR and push (fast feedback, ~30 seconds)
- **Integration tests**: Run on every PR and push (moderate speed, ~2-5 minutes)
- **Fast E2E tests**: Run on every PR (excludes slow/ml_models, ~5-10 minutes)
- **Slow E2E tests**: Run only on main branch pushes (includes all E2E tests, ~15-20 minutes)
- **Parallel execution**: Enabled for all test jobs (`-n auto`)
- **Flaky test reruns**: Enabled for integration and E2E tests (`--reruns 2 --reruns-delay 1`)

**For detailed test execution commands, parallel execution, flaky test reruns, and coverage, see [Testing Guide - Test Execution Details](TESTING_GUIDE.md#test-execution-details).**

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

## Test Pyramid Status

### Current State vs. Ideal Distribution

The test pyramid shows our current distribution compared to the ideal:

**Current Distribution:**
- **Unit Tests**: ~41% (target: 70-80%) ⚠️ Too Low
- **Integration Tests**: ~27% (target: 15-20%) ⚠️ Too High
- **E2E Tests**: ~31% (target: 5-10%) ❌ Too High

**Visual Representation:**

```
Current Pyramid (Inverted):
        ╱╲
       ╱  ╲      E2E: 31% (should be 5-10%)
      ╱    ╲
     ╱      ╲    Integration: 27% (should be 15-20%)
    ╱        ╲
   ╱          ╲  Unit: 41% (should be 70-80%)
  ╱____________╲

Ideal Pyramid:
        ╱╲
       ╱  ╲      E2E: 5-10%
      ╱    ╲
     ╱      ╲    Integration: 15-20%
    ╱        ╲
   ╱          ╲  Unit: 70-80%
  ╱____________╲
```

### Key Issues

1. **Too Few Unit Tests**: Core business logic is being tested at E2E level instead of unit level
   - **Critical modules with zero unit tests**: `workflow.py`, `cli.py`, `service.py`, `episode_processor.py`, `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py`
   - **67 summarizer tests misclassified**: Currently at E2E level but test individual functions with mocked dependencies → should be unit tests
   - **Missing unit test coverage**: Many core functions in `summarizer.py`, `workflow.py`, and `speaker_detection.py` have no unit tests

2. **Too Many E2E Tests**: Many tests are misclassified
   - Tests that use function-level entry points with mocked dependencies should be unit tests
   - Tests that use component-level entry points should be integration tests
   - **Root cause**: Tests were written at E2E level for convenience, violating testing strategy definitions

3. **Integration Layer Underutilized**: Component interactions are often tested at E2E level
   - **Missing integration test coverage**: RSS Parser + Downloader, Downloader + Episode Processor, Progress Reporting + Workflow, Metrics + Workflow, Filesystem + Workflow
   - Some E2E tests should be integration tests (component-level entry points with mocked HTTP)

### Goals and Targets

**Target Distribution:**
- **Unit Tests**: 70-80% (~550-650 tests)
- **Integration Tests**: 15-20% (~120-150 tests)
- **E2E Tests**: 5-10% (~50-80 tests)

**Success Metrics:**
- Unit test execution time: < 30 seconds
- Integration test execution time: < 5 minutes
- E2E test execution time: < 20 minutes
- Test coverage: Maintain > 80%

### Improvement Strategy

**Phase 1: Reclassify Misplaced Tests** (High Priority)
- Move 67 summarizer tests from E2E to unit (they test individual functions with mocked dependencies)
- Review and reclassify E2E tests that violate testing strategy definitions
- **Expected Result**: Unit: ~50-51%, Integration: ~27-28%, E2E: ~22-23%

**Phase 2: Add Missing Unit Tests** (High Priority)
- Add unit tests for core functions currently untested:
  - `workflow.py` helper functions (8-10 tests)
  - `episode_processor.py` functions (5-8 tests)
  - `summarizer.py` core functions (45-65 tests)
  - `speaker_detection.py` functions (19-31 tests)
  - `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py` (17-28 tests)
  - `cli.py` and `service.py` (8-13 tests)
- **Target**: +150-200 new unit tests
- **Expected Result**: Unit: ~69-83%, Integration: ~27-28%, E2E: ~22-23%

**Phase 3: Optimize Integration Layer** (Medium Priority)
- Move component interaction tests from E2E to integration (~20-40 tests)
- Add missing integration tests for component-to-component interactions (~19-31 tests)
- **Expected Result**: Unit: ~69-83%, Integration: ~30-32%, E2E: ~16-19%

**Phase 4: Reduce E2E to True E2E** (Low Priority)
- Keep only true end-to-end user workflow tests
- Focus on complete user journeys (CLI commands, library API calls, service API calls)
- **Target**: ~50-80 true E2E tests (5-10% of total)
- **Expected Final Result**: Unit: ~70-80%, Integration: ~15-20%, E2E: ~5-10% ✅

### Priority Areas for Unit Test Coverage

**High Priority Modules:**
- `summarizer.py`: Text cleaning, chunking, validation functions (45-65 tests needed)
- `workflow.py`: Pipeline orchestration helpers (8-10 tests needed)
- `episode_processor.py`: Episode processing logic (5-8 tests needed)

**Medium Priority Modules:**
- `speaker_detection.py`: Detection and scoring logic (19-31 tests needed)
- `cli.py` and `service.py`: Argument parsing and service logic (8-13 tests needed)
- `preprocessing.py`, `progress.py`, `metrics.py`, `filesystem.py`: Utility functions (17-28 tests needed)

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

- **[Testing Guide](TESTING_GUIDE.md)** - Detailed implementation instructions, test execution, fixtures, and coverage
- Test structure reorganization: `docs/rfc/RFC-018-test-structure-reorganization.md`
- CI workflow: `.github/workflows/python-app.yml`
- Related RFCs: RFC-001 through RFC-018 (testing strategies and reorganization)
- Related Issues: #14 (E2E testing), #16 (Library API E2E tests), #94 (src/ layout), #98 (Test structure reorganization)
- Architecture: `docs/ARCHITECTURE.md` (Testing Notes section)
- Contributing guide: `CONTRIBUTING.md` (Testing Requirements section)
