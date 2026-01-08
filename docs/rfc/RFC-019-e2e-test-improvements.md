# RFC-019: E2E Test Infrastructure and Coverage Improvements

- **Status**: Complete
- **Authors**:
- **Stakeholders**: Maintainers, developers writing E2E tests, CI/CD pipeline maintainers, QA engineers
- **Related PRDs**:
  - `docs/prd/PRD-001-transcript-pipeline.md` (core pipeline)
  - `docs/prd/PRD-002-whisper-fallback.md` (Whisper transcription)
  - `docs/prd/PRD-003-user-interface-config.md` (CLI and config)
  - `docs/prd/PRD-004-metadata-generation.md` (metadata)
  - `docs/prd/PRD-005-episode-summarization.md` (summarization)
- **Related RFCs**:
  - `docs/rfc/RFC-018-test-structure-reorganization.md` (test structure - foundation)
  - `docs/rfc/RFC-020-integration-test-improvements.md` (integration test improvements - related work)
  - `docs/rfc/RFC-001-workflow-orchestration.md` (workflow tests)
  - `docs/rfc/RFC-003-transcript-downloads.md` (HTTP client)
  - `docs/rfc/RFC-005-whisper-integration.md` (Whisper tests)
- **Related Documents**:
  - `docs/TESTING_STRATEGY.md` - Overall testing strategy, test pyramid, and test boundary decision framework

## Abstract

This RFC defines a comprehensive plan to improve E2E (end-to-end) test infrastructure and coverage. The plan addresses critical gaps in the current E2E test suite by introducing a local HTTP server infrastructure, real data files, and comprehensive test coverage for all major user-facing entry points. The improvements ensure E2E tests verify complete user workflows using real HTTP clients, real data files, and real ML models, while maintaining strict network isolation.

**Key Improvements:**

- **Network Isolation**: Hard network guard prevents accidental external network calls
- **Local HTTP Server**: Reusable local HTTP server fixture serves real RSS feeds, transcripts, and audio files
- **Real Data Files**: Manually maintained test fixtures (RSS feeds, transcripts, audio files)
- **Comprehensive Coverage**: Every major CLI command and public API endpoint has E2E tests
- **Real Implementations**: E2E tests use real HTTP client, real data files, and real ML models (no mocking)
- **Clear Boundaries**: Clear decision framework distinguishes Integration tests from E2E tests

## Problem Statement

**Current Issues:**

1. **HTTP Client Not Tested with Real HTTP Stack**
   - E2E tests use `unittest.mock.patch` to mock `fetch_url`, bypassing the real HTTP client
   - Real HTTP behavior (headers, redirects, timeouts, retries, streaming) is not tested in E2E context
   - Cannot verify that the real HTTP client works correctly in full workflows

2. **No Real RSS Feed Testing**
   - RSS feeds are generated in-memory using helper functions
   - No testing with real-world RSS feed structures
   - No testing with various RSS feed formats (Podcasting 2.0, iTunes, standard RSS)
   - No testing with edge cases in real RSS feeds (special characters, relative URLs, missing fields)

3. **No Real Audio File Testing**
   - Audio files are mocked as byte strings (`b"FAKE AUDIO DATA"`)
   - No testing with real audio files (even small test files)
   - Whisper transcription is mocked in most E2E tests
   - Cannot verify that audio download and Whisper transcription work correctly

4. **Incomplete Coverage**
   - Not all major CLI commands have E2E tests
   - Not all public API endpoints have E2E tests
   - Missing E2E tests for error handling, edge cases, and HTTP behaviors

5. **Unclear Test Boundaries**
   - Ambiguity about what belongs in Integration tests vs E2E tests
   - No clear decision framework for test placement
   - Risk of tests being placed in wrong category

**Impact:**

- Cannot verify that real HTTP client works correctly in full workflows
- Cannot verify that RSS parsing works correctly with real-world feed structures
- Cannot verify that audio download and Whisper transcription work correctly
- Missing confidence in complete user workflows
- Difficult to maintain and extend E2E test suite

## Goals

1. **Network Isolation**: Hard network guard prevents accidental external network calls in E2E tests
2. **Real HTTP Client**: E2E tests use real HTTP client (`downloader.fetch_url`) without mocking
3. **Real Data Files**: E2E tests use real RSS feeds, transcripts, and audio files from fixtures
4. **Comprehensive Coverage**: Every major CLI command and public API endpoint has at least one E2E test
5. **Real ML Models**: E2E tests use real ML models (Whisper, spaCy, Transformers) in full workflow context
6. **Clear Boundaries**: Clear decision framework distinguishes Integration tests from E2E tests
7. **Maintainable Infrastructure**: Reusable HTTP server fixture and organized test fixtures
8. **CI/CD Integration**: E2E tests run in CI/CD with proper markers and validation

## Constraints & Assumptions

**Constraints:**

- E2E tests must **not** hit external networks (enforced with network guard)
- E2E tests must use real HTTP client (no mocking of `downloader.fetch_url`)
- E2E tests must use real data files (not in-memory generated data)
- E2E tests may be slower than integration tests (acceptable for comprehensive coverage)
- Test fixtures must be manually maintained (no generation scripts)

**Assumptions:**

- Local HTTP server is sufficient for E2E testing (no need for external network)
- Small test audio files (< 10 seconds) are sufficient for Whisper testing
- Real ML models (smallest available) are acceptable for E2E tests
- Manual fixture maintenance is acceptable (better than generation scripts)
- E2E tests can be marked as `slow` and `ml_models` for CI/CD optimization

## Design & Implementation

### 0. Network Guard + OpenAI Mocking (Stage 0) - 🚨 NON-NEGOTIABLE

**Goal**: Prevent accidental external network calls and mock OpenAI API calls in E2E tests.

**Network Guard Requirements (NON-NEGOTIABLE):**

- ❌ **NO external HTTP requests** should be made during E2E tests
- ✅ **Block outbound network** in tests (except localhost/127.0.0.1)
- ✅ **All RSS and audio** must be served from the local mock server
- ✅ **Fail hard** if a real URL is hit (immediate test failure with clear error message)

**Implementation:**

- Add `pytest-socket` to dev dependencies (recommended)
- Create `tests/e2e/conftest.py` with network blocking fixture (autouse=True)
- Block all outbound sockets except localhost/127.0.0.1
- Network guard must be active for ALL E2E tests (no opt-out)
- Test failure must occur immediately when external network call is attempted
- Error message must clearly indicate which URL was blocked and why
- Add test to verify network blocking works

**OpenAI API Mocking:**

- Create `tests/e2e/fixtures/openai_mock.py` with mock OpenAI client
- Implement realistic mock responses for all OpenAI provider methods:
  - `OpenAISummarizationProvider.summarize()` → mock summary response
  - `OpenAITranscriptionProvider.transcribe()` → mock transcription response
  - `OpenAISpeakerDetector.detect_speakers()` → mock speaker detection response
- Create pytest fixture `openai_mock` that patches OpenAI client initialization
- Document that OpenAI providers are mocked in E2E tests (intentional)

**Rationale:**

- Network guard ensures local fixtures are actually used
- OpenAI mocking prevents costs, rate limits, and flakiness
- E2E tests use real implementations for internal components (HTTP client, ML models) but mock external paid services

**Deliverables:**

- Network blocking fixture in `tests/e2e/conftest.py` (autouse=True)
- OpenAI mock fixture in `tests/e2e/fixtures/openai_mock.py`
- Test verifying network blocking (`test_network_guard.py`)
- Test verifying OpenAI mocking works
- Documentation of network guard and OpenAI mocking strategy

### Stage 1: HTTP Server Infrastructure

**Goal**: Create reusable local HTTP server fixture for E2E tests.

**Implementation:**

- Create `tests/e2e/fixtures/e2e_http_server.py` with:
  - `E2EHTTPServer` class (session-scoped)
  - `E2EHTTPRequestHandler` class
  - Security hardening (path traversal protection, URL encoding safety)
  - Request logging and debugging helpers
  - URL helper methods (`e2e_server.urls.feed()`, `e2e_server.urls.episode()`, etc.)
- Create `e2e_server` pytest fixture (session-scoped with function-scoped reset)
- Add HTTP behaviors:
  - Range requests support (206 Partial Content)
  - Proper headers (Content-Length, ETag, Last-Modified)
  - Configurable error scenarios (behavior registry)

**Deliverables:**

- `tests/e2e/fixtures/e2e_http_server.py`
- `e2e_server` pytest fixture in `tests/e2e/conftest.py`
- Basic test verifying server functionality

### Stage 2: Test Fixtures - Keep Flat Structure with URL Mapping

**Goal**: Use existing flat fixture structure with E2E server URL mapping (no file reorganization needed).

**RSS Linkage Requirements:**

For any episode:

- ✅ RSS `<guid>` == transcript filename == audio filename (e.g., `p01_e01`)
- ✅ RSS `<enclosure>` URL points to `/audio/pXX_eYY.mp3` (flat URL, matches existing structure)
- ✅ RSS `<podcast:transcript>` URL points to `/transcripts/pXX_eYY.txt` (flat URL, matches existing structure)

**Current Fixture Structure (Keep As-Is):**

````text
tests/fixtures/
├── rss/
│   ├── p01_mtb.xml
│   ├── p02_software.xml
│   ├── p03_scuba.xml
│   ├── p04_photo.xml
│   └── p05_investing.xml
├── audio/
│   ├── p01_e01.mp3
│   ├── p01_e02.mp3
│   ├── p01_e03.mp3
│   └── ... (p02-p05 episodes)
└── transcripts/
    ├── p01_e01.txt
    ├── p01_e02.txt
    ├── p01_e03.txt
    └── ... (p02-p05 episodes)
```text

- `/feeds/podcast1/feed.xml` → `tests/fixtures/rss/p01_mtb.xml` (mapping)
- `/audio/p01_e01.mp3` → `tests/fixtures/audio/p01_e01.mp3` (direct)
- `/transcripts/p01_e01.txt` → `tests/fixtures/transcripts/p01_e01.txt` (direct)

**Benefits:**

- ✅ No file reorganization needed - keep your existing flat structure
- ✅ No script changes - generation scripts continue to work
- ✅ RSS linkage already works - GUID `p01_e01` matches filename `p01_e01.mp3`
- ✅ Flexible routing - server maps any URL pattern to flat files

**Implementation:**

- Keep existing flat fixture structure (no reorganization)
- E2E server implements URL mapping logic:
  - Map `/feeds/{podcast}/feed.xml` → `rss/pXX_*.xml` (using podcast mapping)
  - Map `/audio/pXX_eYY.mp3` → `audio/pXX_eYY.mp3` (direct)
  - Map `/transcripts/pXX_eYY.txt` → `transcripts/pXX_eYY.txt` (direct)
- Update RSS feeds to use E2E server base URL (template-based, optional)
- Create edge case fixtures if needed (can stay in `rss/` folder or separate `edge_cases/`)

**Deliverables:**

- E2E server URL mapping implementation
- RSS feed updates (optional - can keep current URLs)
- Edge case fixtures (if needed)
- Documentation of URL mapping logic

### Stage 3: HTTP Server Integration with Fixtures

**Goal**: Integrate HTTP server with flat fixture structure and verify end-to-end file serving.

**Implementation:**

- Update `E2EHTTPRequestHandler` to serve files from existing `tests/fixtures/` structure
- Implement URL routing that maps to flat structure:

  **RSS Feed Routing:**
  - `/feeds/podcast1/feed.xml` → `tests/fixtures/rss/p01_mtb.xml`
  - `/feeds/podcast2/feed.xml` → `tests/fixtures/rss/p02_software.xml`
  - (Mapping: `podcast1` → `p01_mtb.xml`, `podcast2` → `p02_software.xml`, etc.)

  **Direct Flat URL Routing:**
  - `/audio/p01_e01.mp3` → `tests/fixtures/audio/p01_e01.mp3` (direct)
  - `/transcripts/p01_e01.txt` → `tests/fixtures/transcripts/p01_e01.txt` (direct)

  **Optional Hierarchical URL Routing:**
  - `/feeds/podcast1/episodes/episode1/audio.mp3` → `tests/fixtures/audio/p01_e01.mp3` (mapping)
  - `/feeds/podcast1/episodes/episode1/transcript.txt` → `tests/fixtures/transcripts/p01_e01.txt` (mapping)

- Implement path traversal protection (verify paths are within fixture root)
- Add URL helper methods to `e2e_server` fixture:
  - `e2e_server.urls.feed("podcast1")` → `http://localhost:8000/feeds/podcast1/feed.xml`
  - `e2e_server.urls.audio("p01_e01")` → `http://localhost:8000/audio/p01_e01.mp3`
  - `e2e_server.urls.transcript("p01_e01")` → `http://localhost:8000/transcripts/p01_e01.txt`

**URL Mapping Implementation Example:**

```python
class E2EHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """E2E HTTP server that maps URLs to flat fixture structure."""

    FIXTURE_ROOT = Path(__file__).parent.parent.parent / "fixtures"

    # Mapping: podcast name -> RSS filename

    PODCAST_RSS_MAP = {
        "podcast1": "p01_mtb.xml",
        "podcast2": "p02_software.xml",
        "podcast3": "p03_scuba.xml",
        "podcast4": "p04_photo.xml",
        "podcast5": "p05_investing.xml",
    }

    def do_GET(self):
        """Handle GET requests with URL mapping."""
        path = self.path.split('?')[0]  # Remove query string

```text

        # Route 1: RSS feeds

```python

        # /feeds/podcast1/feed.xml -> rss/p01_mtb.xml

```

        if path.startswith("/feeds/") and path.endswith("/feed.xml"):
            podcast_name = path.split("/")[2]  # Extract "podcast1"
            rss_file = self.PODCAST_RSS_MAP.get(podcast_name)
            if rss_file:
                file_path = self.FIXTURE_ROOT / "rss" / rss_file
                self._serve_file(file_path, content_type="application/xml")
                return

```python

        # /audio/p01_e01.mp3 -> audio/p01_e01.mp3

```

            self._serve_file(file_path, content_type="audio/mpeg")
            return

```python

### Stages 4-12: E2E Test Coverage

**Goal**: Create comprehensive E2E tests for all major user-facing entry points.

**Implementation:**

**Stage 4: Basic E2E Tests (Happy Paths)**

- Create `tests/e2e/test_basic_e2e.py` with basic CLI, library API, and service API tests
- Remove HTTP mocking from these tests
- Use `e2e_server` fixture to serve RSS feeds and transcripts

**Stage 5: CLI Command E2E Tests**

- Create `tests/e2e/test_cli_e2e.py` with tests for:

  - `podcast-scraper <rss_url>` - Basic transcript download
  - `podcast-scraper <rss_url> --transcribe-missing` - Whisper fallback
  - `podcast-scraper --config <config_file>` - Config file workflow
  - `podcast-scraper <rss_url> --dry-run` - Dry run
  - `podcast-scraper <rss_url> --generate-metadata` - Metadata generation
  - `podcast-scraper <rss_url> --generate-summaries` - Summarization
  - `podcast-scraper <rss_url> --transcribe-missing --generate-metadata --generate-summaries` - All features

**Stage 6: Library API E2E Tests**

- Create `tests/e2e/test_library_api_e2e.py` with tests for:

  - `run_pipeline(config)` - Basic pipeline
  - `run_pipeline(config)` with all features
  - `load_config_file(path)` + `run_pipeline()`

**Stage 7: Service API E2E Tests**

- Update `tests/e2e/test_service.py` to use real HTTP client
- Tests for `service.run(config)`, `service.run_from_config_file(path)`, `service.main()`

**Stage 8: Real Whisper Transcription E2E Tests**

- Create `tests/e2e/test_whisper_e2e.py` with real Whisper models and real audio files
- Mark tests as `@pytest.mark.slow` and `@pytest.mark.ml_models`

**Stage 9: Real ML Models E2E Tests**

- Create `tests/e2e/test_ml_models_e2e.py` with real spaCy and Transformers models
- Mark tests as `@pytest.mark.slow` and `@pytest.mark.ml_models`

**Stage 10: Error Handling E2E Tests**

- Create `tests/e2e/test_error_handling_e2e.py` with error scenarios
- Use `e2e_server` fixture with error scenarios (behavior registry)

**Stage 11: Edge Cases E2E Tests**

- Create `tests/e2e/test_edge_cases_e2e.py` with edge case scenarios
- Use edge case fixtures from `tests/fixtures/e2e_server/feeds/edge_cases/`

**Stage 12: HTTP Behaviors E2E Tests - Realistic Large File Testing**

- Create `tests/e2e/test_http_behaviors_e2e.py` with HTTP behavior scenarios
- Test Range requests, redirects, retries, timeouts
- **REQUIRED: Add realistic large file tests (exploit large audio files):**
  - `test_e2e_audio_streaming_with_range_requests` - Stream audio instead of downloading whole file
  - `test_e2e_audio_download_timeout_handling` - Enforce timeouts with large files
  - `test_e2e_audio_download_retry_logic` - Handle retries with large files
  - `test_e2e_audio_partial_reads` - Handle partial reads / range requests
- Use real large audio files from fixtures (not small test files)
- Verify streaming behavior, range request support, timeout handling, retry logic
- **Rationale**: Large audio files enable testing of real-world scenarios (streaming, timeouts, retries)

**Deliverables:**

- Comprehensive E2E test coverage for all major entry points
- All tests use real HTTP client (no mocking)
- All tests use real data files
- All tests pass

### Stage 13: Migration and Cleanup

**Goal**: Migrate existing E2E tests to use real HTTP client and remove mocks.

**Implementation:**

- Review all existing E2E tests in `tests/e2e/`
- Identify tests that still use HTTP mocking
- Migrate tests to use `e2e_server` fixture:

  - Remove `unittest.mock.patch` on `downloader.fetch_url`
  - Replace in-memory RSS generation with fixture files
  - Replace mocked audio with real audio files
  - Update tests to use real HTTP client

- Remove obsolete helper functions (if any)
- Update test documentation

**Deliverables:**

- All existing E2E tests migrated to real HTTP client
- All HTTP mocking removed from E2E tests
- All tests pass
- Documentation updated

### Stage 14: CI/CD Integration

**Goal**: Integrate E2E tests into CI/CD pipeline.

**Implementation:**

- Update `.github/workflows/python-app.yml`:

  - Add E2E test job (or update existing `test-e2e` job)
  - Ensure E2E tests run with proper markers (`-m e2e`)
  - Add test count validation for E2E tests
  - Configure slow tests to run on schedule or manual trigger

- Update `Makefile`:

  - Ensure `test-e2e` target runs E2E tests
  - Add `test-e2e-fast` target for fast E2E tests (exclude slow/ml_models)
  - Add `test-e2e-slow` target for slow E2E tests

- Update documentation

**Deliverables:**

- CI/CD pipeline updated
- Makefile targets updated
- Documentation updated
- E2E tests run in CI

## Key Decisions

1. **Local HTTP Server vs External Network**
   - **Decision**: Use local HTTP server with network guard (no external network)
   - **Rationale**: Faster, more reliable, no external dependencies, prevents accidental network calls

2. **Manual Fixtures vs Generation Scripts**
   - **Decision**: Manually maintain test fixtures (no generation scripts)
   - **Rationale**: Simpler, more maintainable, easier to review, fixtures are version-controlled

3. **Real ML Models vs Mocked Models**
   - **Decision**: Use real ML models (smallest available) in E2E tests
   - **Rationale**: E2E tests should verify complete workflows with real implementations

4. **Session-Scoped Server vs Function-Scoped Server**
   - **Decision**: Session-scoped server with function-scoped reset
   - **Rationale**: Faster test execution (server starts once), still provides test isolation

5. **HTTP Server Library Choice**
   - **Decision**: Use `http.server` (standard library) for consistency with integration tests
   - **Rationale**: No additional dependencies, consistent with existing integration test infrastructure

6. **Test Boundary Framework**
   - **Decision**: Create clear decision framework for Integration vs E2E tests
   - **Rationale**: Eliminates ambiguity, provides clear guidance for future test development

7. **Fixture Structure: Flat vs Hierarchical**
   - **Decision**: Keep existing flat structure, use URL mapping in E2E server
   - **Rationale**: No file reorganization needed, generation scripts continue to work, RSS linkage already works

8. **OpenAI API Mocking**
   - **Decision**: Mock OpenAI client in E2E tests (don't call real APIs)
   - **Rationale**: Fast, reliable, no cost, no rate limits, no flakiness. E2E tests should use real implementations for internal components but mock external paid services

## Alternatives Considered

1. **pytest-httpserver Library**
   - **Alternative**: Use `pytest-httpserver` library for HTTP server
   - **Rejected**: Adds dependency, `http.server` is sufficient and consistent with integration tests

2. **Fixture Generation Scripts**
   - **Alternative**: Use scripts to generate test fixtures from templates
   - **Rejected**: More complex, harder to maintain, manual fixtures are simpler and more reviewable

3. **External Network for E2E Tests**
   - **Alternative**: Allow E2E tests to hit external networks
   - **Rejected**: Slower, less reliable, introduces external dependencies, harder to test error scenarios

4. **Mocked ML Models in E2E Tests**
   - **Alternative**: Mock ML models in E2E tests for speed
   - **Rejected**: E2E tests should verify complete workflows with real implementations

5. **Function-Scoped Server**
   - **Alternative**: Create new HTTP server for each test
   - **Rejected**: Slower test execution, session-scoped with reset provides better performance

6. **Hierarchical Fixture Structure**
   - **Alternative**: Reorganize fixtures into hierarchical structure (`feeds/podcast1/episodes/episode1/`)
   - **Rejected**: Requires file reorganization, breaks generation scripts, unnecessary complexity. URL mapping achieves same result with flat structure.

7. **Real OpenAI API Calls in E2E Tests**
   - **Alternative**: Call real OpenAI APIs in E2E tests
   - **Rejected**: Costly, rate-limited, flaky, slow. Mock OpenAI client instead (tests provider integration without API costs)

## Testing Strategy

**E2E Test Coverage Goals:**

- **CLI Commands**: Every major CLI command has at least one E2E test
- **Library API**: Every public API endpoint has at least one E2E test
- **Service API**: Every service API endpoint has at least one E2E test
- **Critical Scenarios**: Happy paths, error handling, edge cases, HTTP behaviors

**Test Organization:**

- E2E tests in `tests/e2e/` directory
- Marked with `@pytest.mark.e2e`
- Slow tests marked with `@pytest.mark.slow`
- ML model tests marked with `@pytest.mark.ml_models`

**Test Execution:**

- Fast E2E tests run in CI on every commit
- Slow E2E tests run on schedule or manual trigger
- Network guard prevents accidental external network calls
- Test count validation ensures tests are collected and run

## Rollout & Monitoring

**Rollout Plan:**

1. **Stage 0-3**: Foundation (Network guard, HTTP server, fixtures) - 1-2 weeks
2. **Stage 4-7**: Core E2E tests (Basic, CLI, Library API, Service API) - 2-3 weeks
3. **Stage 8-12**: Advanced E2E tests (Whisper, ML models, errors, edge cases, HTTP behaviors) - 2-3 weeks
4. **Stage 13-14**: Migration, cleanup, CI/CD integration - 1-2 weeks

**Total Estimated Time**: 6-10 weeks (completed)

**Monitoring:**

- Track E2E test coverage (number of tests per entry point)
- Monitor E2E test execution time
- Track E2E test failures and flakiness
- Monitor network guard effectiveness (no external network calls)

**Success Criteria:**

1. ✅ All major CLI commands have at least one E2E test
2. ✅ All public API endpoints have at least one E2E test
3. ✅ All E2E tests use real HTTP client (no mocking)
4. ✅ All E2E tests use real data files (RSS feeds, transcripts, audio)
5. ✅ Network guard prevents external network calls
6. ✅ All E2E tests pass in CI/CD
7. ✅ Documentation is complete and up-to-date

## Relationship to Other Test RFCs

This RFC (RFC-019) is part of a comprehensive testing strategy that includes:

1. **RFC-018: Test Structure Reorganization** - Established the foundation by organizing tests into `unit/`, `integration/`, and `e2e/` directories, adding pytest markers, and enabling test execution control. This RFC builds upon that structure.

2. **RFC-020: Integration Test Improvements** - Completed comprehensive improvements to integration tests (182 tests across 15 files). While integration tests focus on component interactions, E2E tests (this RFC) focus on complete user workflows.

**Key Distinction:**

- **Integration Tests (RFC-020)**: Test how components work together (component interactions, data flow)
- **E2E Tests (RFC-019)**: Test complete user workflows (CLI commands, library API calls, full pipelines)

Together, these three RFCs provide:

- Clear test structure and boundaries (RFC-018) ✅ **Completed**
- Comprehensive component interaction testing (RFC-020) ✅ **Completed**
- Comprehensive user workflow testing (RFC-019) ✅ **Completed**

## References

- **Test Boundary Decision Framework**: `docs/TESTING_STRATEGY.md` (Test Boundary Decision Framework section)
- **Test Strategy**: `docs/TESTING_STRATEGY.md` - Overall testing strategy and test pyramid
- **Test Structure RFC**: `docs/rfc/RFC-018-test-structure-reorganization.md` (foundation)
- **Integration Test RFC**: `docs/rfc/RFC-020-integration-test-improvements.md` (related work)
- **Source Code**: `tests/e2e/`, `tests/fixtures/`
- **Fixture Specification**: `tests/fixtures/FIXTURES_SPEC.md` - How fixtures are generated

````
