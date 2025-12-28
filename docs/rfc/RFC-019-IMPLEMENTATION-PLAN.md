# RFC-019 Implementation Plan: E2E Test Infrastructure and Coverage Improvements

**Status**: Implementation Plan
**Related RFC**: `docs/rfc/RFC-019-e2e-test-improvements.md`
**Created**: 2025-01-XX

## Overview

This document provides a detailed, staged implementation plan for RFC-019. The plan breaks down the RFC into actionable tasks with clear deliverables, dependencies, and success criteria.

**Total Estimated Time**: 6-10 weeks (as per RFC-019)

## Implementation Stages

### Stage 0: Network Guard + OpenAI Mocking (🚨 NON-NEGOTIABLE)

**Priority**: CRITICAL - Must be done first
**Estimated Time**: 2-3 days
**Dependencies**: None

#### Tasks

1. **Add pytest-socket to dev dependencies**
   - Update `pyproject.toml` to include `pytest-socket>=1.0.0` in `[project.optional-dependencies.dev]`
   - Run `pip install -e .[dev]` to install

2. **Create E2E conftest.py with network guard**
   - Create `tests/workflow_e2e/conftest.py`
   - Implement `block_external_network` fixture (autouse=True)
   - Use `pytest-socket` to block all sockets except localhost/127.0.0.1
   - Ensure clear error messages when external network is accessed
   - Add fixture documentation

3. **Create OpenAI mock fixture**
   - Create `tests/workflow_e2e/fixtures/` directory
   - Create `tests/workflow_e2e/fixtures/openai_mock.py`
   - Implement mock OpenAI client with realistic responses:
     - Transcription: Return realistic transcript text
     - Summarization: Return realistic summary
     - Speaker detection: Return realistic speaker names
   - Create `openai_mock` pytest fixture that patches OpenAI client initialization
   - Document mocking strategy

4. **Create network guard verification test**
   - Create `tests/workflow_e2e/test_network_guard.py`
   - Test that external network calls are blocked
   - Test that localhost calls are allowed
   - Verify error messages are clear

5. **Create OpenAI mocking verification test**
   - Add test to verify OpenAI providers use mocked client
   - Verify mock responses are returned correctly

#### Deliverables

- ✅ `pytest-socket` added to dev dependencies
- ✅ `tests/workflow_e2e/conftest.py` with network guard fixture
- ✅ `tests/workflow_e2e/fixtures/openai_mock.py` with OpenAI mock fixture
- ✅ `tests/workflow_e2e/test_network_guard.py` - Network guard verification
- ✅ All existing E2E tests still pass (with network guard active)

#### Success Criteria

- Network guard blocks all external network calls
- Network guard allows localhost/127.0.0.1
- Clear error messages when external network is accessed
- OpenAI providers use mocked client in E2E tests
- All existing E2E tests pass (may need minor adjustments)

---

### Stage 1: HTTP Server Infrastructure

**Priority**: HIGH - Foundation for all E2E tests
**Estimated Time**: 3-4 days
**Dependencies**: Stage 0

#### Tasks

1. **Create E2E HTTP server module**
   - Create `tests/workflow_e2e/fixtures/e2e_http_server.py`
   - Implement `E2EHTTPServer` class (session-scoped)
   - Implement `E2EHTTPRequestHandler` class (extends `http.server.SimpleHTTPRequestHandler`)
   - Add security hardening:
     - Path traversal protection
     - URL encoding safety
     - Request validation
   - Add request logging and debugging helpers

2. **Implement URL helper class**
   - Create `E2EServerURLs` class
   - Methods: `feed(podcast_name)`, `audio(episode_id)`, `transcript(episode_id)`, `base_url()`
   - Attach to server instance as `e2e_server.urls`

3. **Create E2E server pytest fixture**
   - Add `e2e_server` fixture to `tests/workflow_e2e/conftest.py`
   - Session-scoped (starts once per test session)
   - Function-scoped reset (clean state between tests)
   - Auto-start server on first use
   - Auto-stop server after all tests

4. **Implement HTTP behaviors**
   - Range request support (206 Partial Content)
   - Proper headers (Content-Length, ETag, Last-Modified)
   - Configurable error scenarios (behavior registry)
   - Support for different content types

5. **Create basic server verification test**
   - Test server starts and stops correctly
   - Test URL helper methods
   - Test basic file serving
   - Test security (path traversal protection)

#### Deliverables

- ✅ `tests/workflow_e2e/fixtures/e2e_http_server.py` - Complete HTTP server implementation
- ✅ `e2e_server` pytest fixture in `tests/workflow_e2e/conftest.py`
- ✅ URL helper methods (`e2e_server.urls.feed()`, etc.)
- ✅ Basic test verifying server functionality
- ✅ Security tests (path traversal protection)

#### Success Criteria

- Server starts and stops correctly
- Server serves files correctly
- URL helpers work correctly
- Security protections work (path traversal blocked)
- Server handles range requests (206 Partial Content)

---

### Stage 2: Test Fixtures - URL Mapping

**Priority**: HIGH - Required for Stage 3
**Estimated Time**: 1-2 days
**Dependencies**: Stage 1

#### Tasks

1. **Verify fixture structure**
   - Confirm flat structure exists: `rss/`, `audio/`, `transcripts/`
   - Verify RSS files exist: `p01_mtb.xml`, `p02_software.xml`, etc.
   - Verify audio files exist: `p01_e01.mp3`, etc.
   - Verify transcript files exist: `p01_e01.txt`, etc.

2. **Create podcast mapping**
   - Define mapping: `podcast1` → `p01_mtb.xml`, `podcast2` → `p02_software.xml`, etc.
   - Document mapping in code comments
   - Add mapping to `E2EHTTPRequestHandler`

3. **Verify RSS linkage**
   - Check that RSS `<guid>` matches filename pattern (`p01_e01`)
   - Check that RSS `<enclosure>` URLs can be mapped to audio files
   - Check that RSS `<podcast:transcript>` URLs can be mapped to transcript files
   - Document RSS linkage requirements

4. **Optional: Update RSS feeds**
   - If needed, update RSS feeds to use E2E server base URL
   - Keep existing URLs as fallback
   - Use template-based approach if needed

#### Deliverables

- ✅ Podcast mapping defined and documented
- ✅ RSS linkage verified and documented
- ✅ Optional RSS feed updates (if needed)

#### Success Criteria

- All fixtures exist and are accessible
- Podcast mapping is clear and documented
- RSS linkage requirements are met
- RSS feeds can be served by E2E server

---

### Stage 3: HTTP Server Integration with Fixtures

**Priority**: HIGH - Enables real HTTP testing
**Estimated Time**: 3-4 days
**Dependencies**: Stage 1, Stage 2

#### Tasks

1. **Implement URL routing in E2E server**
   - RSS feed routing: `/feeds/{podcast}/feed.xml` → `rss/pXX_*.xml`
   - Direct flat URL routing: `/audio/pXX_eYY.mp3` → `audio/pXX_eYY.mp3`
   - Direct flat URL routing: `/transcripts/pXX_eYY.txt` → `transcripts/pXX_eYY.txt`
   - Implement path traversal protection

2. **Implement file serving**
   - Serve RSS feeds with correct content type (`application/xml`)
   - Serve audio files with correct content type (`audio/mpeg`)
   - Serve transcript files with correct content type (`text/plain`)
   - Handle missing files (404 errors)

3. **Add URL helper methods**
   - `e2e_server.urls.feed("podcast1")` → `http://localhost:8000/feeds/podcast1/feed.xml`
   - `e2e_server.urls.audio("p01_e01")` → `http://localhost:8000/audio/p01_e01.mp3`
   - `e2e_server.urls.transcript("p01_e01")` → `http://localhost:8000/transcripts/p01_e01.txt`

4. **Create integration tests**
   - Test RSS feed serving
   - Test audio file serving
   - Test transcript file serving
   - Test 404 handling
   - Test path traversal protection

#### Deliverables

- ✅ URL routing implemented in `E2EHTTPRequestHandler`
- ✅ File serving with correct content types
- ✅ URL helper methods working
- ✅ Integration tests for file serving

#### Success Criteria

- RSS feeds are served correctly
- Audio files are served correctly
- Transcript files are served correctly
- 404 errors are handled correctly
- Path traversal is blocked
- URL helpers return correct URLs

---

### Stage 4: Basic E2E Tests (Happy Paths)

**Priority**: HIGH - Core functionality
**Estimated Time**: 2-3 days
**Dependencies**: Stage 3

#### Tasks

1. **Create basic E2E test file**
   - Create `tests/workflow_e2e/test_basic_e2e.py`
   - Remove HTTP mocking from existing tests
   - Use `e2e_server` fixture instead of mocks

2. **Migrate basic CLI test**
   - Test: `podcast-scraper <rss_url>` - Basic transcript download
   - Remove `@patch("podcast_scraper.downloader.fetch_url")`
   - Use `e2e_server.urls.feed("podcast1")` for RSS URL
   - Verify transcript is downloaded correctly

3. **Migrate basic library API test**
   - Test: `run_pipeline(config)` - Basic pipeline
   - Remove HTTP mocking
   - Use `e2e_server` fixture
   - Verify pipeline completes successfully

4. **Migrate basic service API test**
   - Test: `service.run(config)` - Basic service execution
   - Remove HTTP mocking
   - Use `e2e_server` fixture
   - Verify service returns success

#### Deliverables

- ✅ `tests/workflow_e2e/test_basic_e2e.py` with basic happy path tests
- ✅ All tests use real HTTP client (no mocking)
- ✅ All tests use `e2e_server` fixture
- ✅ All tests pass

#### Success Criteria

- Basic CLI test works with real HTTP client
- Basic library API test works with real HTTP client
- Basic service API test works with real HTTP client
- No HTTP mocking in basic tests
- All tests pass

---

### Stage 5: CLI Command E2E Tests

**Priority**: HIGH - User-facing functionality
**Estimated Time**: 3-4 days
**Dependencies**: Stage 4

#### Tasks

1. **Create CLI E2E test file**
   - Create `tests/workflow_e2e/test_cli_e2e.py` (or update existing)
   - Organize tests by CLI command

2. **Implement CLI command tests**
   - `podcast-scraper <rss_url>` - Basic transcript download
   - `podcast-scraper <rss_url> --transcribe-missing` - Whisper fallback
   - `podcast-scraper --config <config_file>` - Config file workflow
   - `podcast-scraper <rss_url> --dry-run` - Dry run
   - `podcast-scraper <rss_url> --generate-metadata` - Metadata generation
   - `podcast-scraper <rss_url> --summarize` - Summarization
   - `podcast-scraper <rss_url> --transcribe-missing --generate-metadata --summarize` - All features

3. **Remove HTTP mocking**
   - Remove all `@patch("podcast_scraper.downloader.fetch_url")` decorators
   - Replace with `e2e_server` fixture usage
   - Update test assertions to use real HTTP responses

4. **Verify test coverage**
   - Each major CLI command has at least one test
   - Tests verify complete workflows
   - Tests use real HTTP client

#### Deliverables

- ✅ `tests/workflow_e2e/test_cli_e2e.py` with comprehensive CLI tests
- ✅ All CLI commands have E2E tests
- ✅ All tests use real HTTP client
- ✅ All tests pass

#### Success Criteria

- All major CLI commands have E2E tests
- All tests use real HTTP client (no mocking)
- All tests verify complete workflows
- All tests pass

---

### Stage 6: Library API E2E Tests

**Priority**: HIGH - Public API coverage
**Estimated Time**: 2-3 days
**Dependencies**: Stage 4

#### Tasks

1. **Create library API E2E test file**
   - Create `tests/workflow_e2e/test_library_api_e2e.py`
   - Organize tests by API function

2. **Implement library API tests**
   - `run_pipeline(config)` - Basic pipeline
   - `run_pipeline(config)` with all features (transcribe, metadata, summarize)
   - `load_config_file(path)` + `run_pipeline()` - Config file workflow

3. **Remove HTTP mocking**
   - Remove all HTTP mocking
   - Use `e2e_server` fixture
   - Verify return values match documented API

4. **Verify test coverage**
   - All public API functions have tests
   - Tests verify complete workflows
   - Tests use real HTTP client

#### Deliverables

- ✅ `tests/workflow_e2e/test_library_api_e2e.py` with library API tests
- ✅ All public API functions have E2E tests
- ✅ All tests use real HTTP client
- ✅ All tests pass

#### Success Criteria

- All public API functions have E2E tests
- All tests use real HTTP client (no mocking)
- All tests verify complete workflows
- All tests pass

---

### Stage 7: Service API E2E Tests

**Priority**: MEDIUM - Service API coverage
**Estimated Time**: 2-3 days
**Dependencies**: Stage 4

#### Tasks

1. **Update service API test file**
   - Update `tests/workflow_e2e/test_service.py`
   - Remove HTTP mocking from existing tests
   - Use `e2e_server` fixture

2. **Implement service API tests**
   - `service.run(config)` - Basic service execution
   - `service.run_from_config_file(path)` - Config file execution
   - `service.main()` - CLI entry point

3. **Remove HTTP mocking**
   - Remove all `@patch("podcast_scraper.downloader.fetch_url")` decorators
   - Replace with `e2e_server` fixture usage
   - Update test assertions

4. **Verify test coverage**
   - All service API functions have tests
   - Tests verify complete workflows
   - Tests use real HTTP client

#### Deliverables

- ✅ `tests/workflow_e2e/test_service.py` updated with real HTTP client
- ✅ All service API functions have E2E tests
- ✅ All tests use real HTTP client
- ✅ All tests pass

#### Success Criteria

- All service API functions have E2E tests
- All tests use real HTTP client (no mocking)
- All tests verify complete workflows
- All tests pass

---

### Stage 8: Real Whisper Transcription E2E Tests

**Priority**: MEDIUM - Real ML model testing
**Estimated Time**: 2-3 days
**Dependencies**: Stage 3

#### Tasks

1. **Create Whisper E2E test file**
   - Create `tests/workflow_e2e/test_whisper_e2e.py`
   - Mark tests as `@pytest.mark.slow` and `@pytest.mark.ml_models`

2. **Implement Whisper tests**
   - Test Whisper transcription with real audio files
   - Use smallest Whisper model (`tiny.en` or `base.en`)
   - Use small test audio files (< 10 seconds)
   - Verify transcription output is reasonable

3. **Test Whisper fallback workflow**
   - Test complete workflow: RSS → no transcript → audio download → Whisper → file output
   - Use `e2e_server` fixture for audio serving
   - Verify Whisper transcription is saved correctly

4. **Verify test coverage**
   - Whisper transcription is tested with real models
   - Whisper fallback workflow is tested end-to-end
   - Tests use real audio files

#### Deliverables

- ✅ `tests/workflow_e2e/test_whisper_e2e.py` with real Whisper tests
- ✅ Tests use real Whisper models
- ✅ Tests use real audio files
- ✅ Tests marked as slow/ml_models
- ✅ All tests pass

#### Success Criteria

- Whisper transcription works with real models
- Whisper fallback workflow works end-to-end
- Tests use real audio files
- Tests are marked appropriately (slow/ml_models)
- All tests pass

---

### Stage 9: Real ML Models E2E Tests

**Priority**: MEDIUM - Real ML model testing
**Estimated Time**: 2-3 days
**Dependencies**: Stage 3

#### Tasks

1. **Create ML models E2E test file**
   - Create `tests/workflow_e2e/test_ml_models_e2e.py`
   - Mark tests as `@pytest.mark.slow` and `@pytest.mark.ml_models`

2. **Implement spaCy tests**
   - Test speaker detection with real spaCy models
   - Use smallest spaCy model (`en_core_web_sm`)
   - Verify speaker detection works in full workflow

3. **Implement Transformers tests**
   - Test summarization with real Transformers models
   - Use smallest model (e.g., `facebook/bart-large-cnn`)
   - Verify summarization works in full workflow

4. **Test all models together**
   - Test complete workflow with all real models
   - Verify models work correctly together
   - Test model loading and cleanup

#### Deliverables

- ✅ `tests/workflow_e2e/test_ml_models_e2e.py` with real ML model tests
- ✅ Tests use real spaCy models
- ✅ Tests use real Transformers models
- ✅ Tests use real models in full workflows
- ✅ Tests marked as slow/ml_models
- ✅ All tests pass

#### Success Criteria

- Real ML models work in E2E tests
- Models work correctly in full workflows
- Tests are marked appropriately (slow/ml_models)
- All tests pass

---

### Stage 10: Error Handling E2E Tests

**Priority**: MEDIUM - Error scenario coverage
**Estimated Time**: 2-3 days
**Dependencies**: Stage 3

#### Tasks

1. **Create error handling E2E test file**
   - Create `tests/workflow_e2e/test_error_handling_e2e.py`

2. **Implement error scenario tests**
   - RSS feed returns 404
   - RSS feed returns 500
   - Transcript download fails (404, 500, timeout)
   - Audio download fails
   - Malformed RSS feed
   - Network timeout
   - Retry logic verification

3. **Use E2E server error scenarios**
   - Implement behavior registry in E2E server
   - Configure error scenarios (404, 500, timeout)
   - Test error handling in full workflow context

4. **Verify error handling**
   - Errors are handled gracefully
   - Error messages are clear
   - Partial failures don't break entire pipeline

#### Deliverables

- ✅ `tests/workflow_e2e/test_error_handling_e2e.py` with error handling tests
- ✅ E2E server supports error scenarios (behavior registry)
- ✅ All error scenarios are tested
- ✅ All tests pass

#### Success Criteria

- Error handling works correctly in E2E tests
- Error scenarios are tested comprehensively
- Error messages are clear
- All tests pass

---

### Stage 11: Edge Cases E2E Tests

**Priority**: LOW - Edge case coverage
**Estimated Time**: 2-3 days
**Dependencies**: Stage 3

#### Tasks

1. **Create edge cases E2E test file**
   - Create `tests/workflow_e2e/test_edge_cases_e2e.py`

2. **Implement edge case tests**
   - Special characters in episode titles
   - Relative URLs in RSS feeds
   - Missing optional fields in RSS feeds
   - Empty RSS feeds
   - Very long episode titles
   - Unicode characters in content

3. **Create edge case fixtures (if needed)**
   - Add edge case RSS feeds to `tests/fixtures/rss/` if needed
   - Document edge case scenarios

4. **Verify edge case handling**
   - Edge cases are handled gracefully
   - No crashes or unexpected behavior
   - Output is correct

#### Deliverables

- ✅ `tests/workflow_e2e/test_edge_cases_e2e.py` with edge case tests
- ✅ Edge case fixtures (if needed)
- ✅ All edge cases are tested
- ✅ All tests pass

#### Success Criteria

- Edge cases are handled correctly
- Edge case scenarios are tested comprehensively
- All tests pass

---

### Stage 12: HTTP Behaviors E2E Tests - Realistic Large File Testing

**Priority**: HIGH - Exploit large audio files
**Estimated Time**: 3-4 days
**Dependencies**: Stage 3

#### Tasks

1. **Create HTTP behaviors E2E test file**
   - Create `tests/workflow_e2e/test_http_behaviors_e2e.py`

2. **Implement realistic large file tests (REQUIRED)**
   - `test_e2e_audio_streaming_with_range_requests` - Stream audio instead of downloading whole file
     - Use large audio file from fixtures
     - Verify range requests are used
     - Verify streaming behavior works correctly
   - `test_e2e_audio_download_timeout_handling` - Enforce timeouts with large files
     - Configure timeout in E2E server
     - Verify timeout handling works correctly
     - Verify retry logic works
   - `test_e2e_audio_download_retry_logic` - Handle retries with large files
     - Configure retry behavior in E2E server
     - Verify retry logic works correctly
     - Verify partial failures are handled
   - `test_e2e_audio_partial_reads` - Handle partial reads / range requests
     - Use range requests to download partial audio
     - Verify partial reads work correctly
     - Verify range request support (206 Partial Content)

3. **Implement other HTTP behavior tests**
   - HTTP redirects
   - HTTP headers (User-Agent, Accept, etc.)
   - Content-Type handling
   - ETag and Last-Modified headers

4. **Use real large audio files**
   - Use actual large audio files from fixtures (not small test files)
   - Verify streaming, timeouts, retries work with real large files

#### Deliverables

- ✅ `tests/workflow_e2e/test_http_behaviors_e2e.py` with HTTP behavior tests
- ✅ Realistic large file tests (streaming, timeouts, retries, partial reads)
- ✅ Range request support (206 Partial Content)
- ✅ All tests use real large audio files
- ✅ All tests pass

#### Success Criteria

- Large file streaming works correctly
- Timeout handling works with large files
- Retry logic works with large files
- Partial reads / range requests work correctly
- All tests use real large audio files
- All tests pass

---

### Stage 13: Migration and Cleanup

**Priority**: HIGH - Remove all HTTP mocking
**Estimated Time**: 3-4 days
**Dependencies**: Stages 4-12

#### Tasks

1. **Review all existing E2E tests**
   - List all tests in `tests/workflow_e2e/`
   - Identify tests that still use HTTP mocking
   - Create migration checklist

2. **Migrate tests to real HTTP client**
   - Remove `@patch("podcast_scraper.downloader.fetch_url")` decorators
   - Replace in-memory RSS generation with fixture files
   - Replace mocked audio with real audio files
   - Update tests to use `e2e_server` fixture
   - Update test assertions

3. **Remove obsolete helper functions**
   - Identify helper functions that are no longer needed
   - Remove or deprecate obsolete functions
   - Update test documentation

4. **Verify all tests pass**
   - Run all E2E tests
   - Fix any issues
   - Ensure all tests use real HTTP client

5. **Update test documentation**
   - Document E2E test infrastructure
   - Document how to use `e2e_server` fixture
   - Document fixture structure

#### Deliverables

- ✅ All existing E2E tests migrated to real HTTP client
- ✅ All HTTP mocking removed from E2E tests
- ✅ Obsolete helper functions removed
- ✅ All tests pass
- ✅ Documentation updated

#### Success Criteria

- No HTTP mocking in E2E tests
- All tests use real HTTP client
- All tests use real data files
- All tests pass
- Documentation is complete

---

### Stage 14: CI/CD Integration

**Priority**: HIGH - Production readiness
**Estimated Time**: 2-3 days
**Dependencies**: Stages 0-13

#### Tasks

1. **Update GitHub Actions workflow**
   - Update `.github/workflows/python-app.yml`
   - Ensure E2E tests run with proper markers (`-m workflow_e2e`)
   - Add test count validation for E2E tests
   - Configure slow tests to run on schedule or manual trigger
   - Ensure network guard is active in CI

2. **Update Makefile**
   - Ensure `test-workflow-e2e` target runs E2E tests
   - Add `test-e2e-fast` target (exclude slow/ml_models)
   - Add `test-e2e-slow` target (include slow/ml_models)
   - Update documentation

3. **Add test count validation**
   - Add validation step to CI
   - Verify minimum number of E2E tests are collected
   - Fail CI if test count is too low

4. **Update documentation**
   - Document CI/CD integration
   - Document how to run E2E tests locally
   - Document test markers and their usage

#### Deliverables

- ✅ CI/CD pipeline updated
- ✅ Makefile targets updated
- ✅ Test count validation added
- ✅ Documentation updated
- ✅ E2E tests run in CI

#### Success Criteria

- E2E tests run in CI/CD
- Fast E2E tests run on every commit
- Slow E2E tests run on schedule or manual trigger
- Test count validation works
- All tests pass in CI

---

## Implementation Order

**Recommended order** (based on dependencies and priority):

1. **Stage 0** (CRITICAL) - Network guard + OpenAI mocking
2. **Stage 1** (HIGH) - HTTP server infrastructure
3. **Stage 2** (HIGH) - Test fixtures URL mapping
4. **Stage 3** (HIGH) - HTTP server integration with fixtures
5. **Stage 4** (HIGH) - Basic E2E tests (happy paths)
6. **Stage 5** (HIGH) - CLI command E2E tests
7. **Stage 6** (HIGH) - Library API E2E tests
8. **Stage 7** (MEDIUM) - Service API E2E tests
9. **Stage 12** (HIGH) - HTTP behaviors E2E tests (realistic large file testing)
10. **Stage 8** (MEDIUM) - Real Whisper transcription E2E tests
11. **Stage 9** (MEDIUM) - Real ML models E2E tests
12. **Stage 10** (MEDIUM) - Error handling E2E tests
13. **Stage 11** (LOW) - Edge cases E2E tests
14. **Stage 13** (HIGH) - Migration and cleanup
15. **Stage 14** (HIGH) - CI/CD integration

## Dependencies Graph

```
Stage 1 + Stage 2 → Stage 3 (Integration)
Stage 3 → Stage 4 (Basic Tests)
Stage 4 → Stage 5 (CLI Tests)
Stage 4 → Stage 6 (Library API Tests)
Stage 4 → Stage 7 (Service API Tests)
Stage 3 → Stage 8 (Whisper Tests)
Stage 3 → Stage 9 (ML Models Tests)
Stage 3 → Stage 10 (Error Handling Tests)
Stage 3 → Stage 11 (Edge Cases Tests)
Stage 3 → Stage 12 (HTTP Behaviors Tests)
Stages 4-12 → Stage 13 (Migration)
Stages 0-13 → Stage 14 (CI/CD)
```

**Coverage Metrics:**
- Number of E2E tests per entry point (CLI, Library API, Service API)
- Percentage of E2E tests using real HTTP client (target: 100%)
- Percentage of E2E tests using real data files (target: 100%)

**Quality Metrics:**
- E2E test execution time
- E2E test failure rate
- Network guard effectiveness (zero external network calls)
- Test count validation (minimum thresholds met)

**Completion Criteria:**
- ✅ All major CLI commands have E2E tests
- ✅ All public API endpoints have E2E tests
- ✅ All E2E tests use real HTTP client (no mocking)
- ✅ All E2E tests use real data files
- ✅ Network guard prevents external network calls
- ✅ All E2E tests pass in CI/CD
- ✅ Documentation is complete

## Risk Mitigation

**Risk 1: Network guard breaks existing tests**
- **Mitigation**: Stage 0 must be done first, fix any issues before proceeding
- **Mitigation**: Test network guard thoroughly before moving to Stage 1

**Risk 2: HTTP server implementation is complex**
- **Mitigation**: Start with simple implementation, iterate
- **Mitigation**: Use standard library (`http.server`) for consistency
- **Mitigation**: Test server thoroughly in Stage 1

**Risk 3: Migration is time-consuming**
- **Mitigation**: Migrate incrementally (Stage 4-7 first, then Stage 13)
- **Mitigation**: Keep old tests working during migration
- **Mitigation**: Test each migration step before proceeding

**Risk 4: Large file tests are slow**
- **Mitigation**: Mark tests as `@pytest.mark.slow`
- **Mitigation**: Run slow tests on schedule or manual trigger
- **Mitigation**: Use smallest possible large files for testing

## Notes

- **Network Guard**: Must be implemented first (Stage 0) - non-negotiable
- **HTTP Server**: Foundation for all E2E tests (Stage 1-3)
- **Migration**: Can be done incrementally (Stage 4-7, then Stage 13)
- **Large File Tests**: Critical for realistic testing (Stage 12)
- **CI/CD**: Final step to ensure production readiness (Stage 14)

## References

- **RFC-019**: `docs/rfc/RFC-019-e2e-test-improvements.md` - Complete RFC
- **Test Strategy**: `docs/TESTING_STRATEGY.md` - Test boundary decision framework
- **Fixture Spec**: `tests/fixtures/FIXTURES_SPEC.md` - Fixture generation specification
- **Current E2E Tests**: `tests/workflow_e2e/` - Existing E2E test files

