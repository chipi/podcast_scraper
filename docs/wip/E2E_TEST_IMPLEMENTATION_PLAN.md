# E2E Test Implementation Plan

## Overview

This document provides a staged plan for implementing comprehensive E2E tests that verify complete user workflows using real HTTP clients, real data files, and real ML models, while maintaining strict network isolation.

**Related Documents:**

- **`docs/wip/TEST_BOUNDARY_DECISION_FRAMEWORK.md`** - Criteria for Integration vs E2E tests

- **`docs/wip/E2E_TEST_GAPS.md`** - Analysis of current gaps

- **`docs/wip/E2E_HTTP_MOCKING_SERVER_PLAN.md`** - HTTP server infrastructure plan

- **`docs/TESTING_STRATEGY.md`** - Overall testing strategy

**Goal**: Every major user-facing entry point (CLI command, public API function) should have at least one E2E test covering the happy path and critical scenarios.

## Implementation Stages

### Stage 0: Network Guard (Foundation)

**Goal**: Prevent accidental external network calls in E2E tests.

**Tasks:**

1. Add `pytest-socket` to dev dependencies (or implement custom network blocker)

2. Create `tests/workflow_e2e/conftest.py` with network blocking fixture

3. Block all outbound sockets except localhost/127.0.0.1

4. Add test to verify network blocking works

5. Document network guard in test documentation

**Deliverables:**

- ✅ Network blocking fixture in `tests/workflow_e2e/conftest.py`

- ✅ Test verifying network blocking (`test_network_guard.py`)

- ✅ Documentation update

**Dependencies:** None

**Estimated Time:** 1-2 hours

---

### Stage 1: HTTP Server Infrastructure (Foundation)

**Goal**: Create reusable local HTTP server fixture for E2E tests.

**Tasks:**

1. Create `tests/workflow_e2e/fixtures/e2e_http_server.py` with:
   - `E2EHTTPServer` class (session-scoped)
   - `E2EHTTPRequestHandler` class
   - Security hardening (path traversal protection, URL encoding safety)
   - Request logging and debugging helpers
   - URL helper methods (`e2e_server.urls.feed()`, `e2e_server.urls.episode()`, etc.)

2. Create `e2e_server` pytest fixture (session-scoped with function-scoped reset)

3. Add HTTP behaviors:
   - Range requests support (206 Partial Content)
   - Proper headers (Content-Length, ETag, Last-Modified)
   - Configurable error scenarios (behavior registry)

4. Add request logging and pytest hook for log dumps on failure

5. Create basic test to verify server works

**Deliverables:**

- ✅ `tests/workflow_e2e/fixtures/e2e_http_server.py`

- ✅ `e2e_server` pytest fixture in `tests/workflow_e2e/conftest.py`

- ✅ Basic test verifying server functionality

- ✅ Documentation

**Dependencies:** Stage 0

**Estimated Time:** 4-6 hours

---

### Stage 2: Test Fixtures (Data Files)

**Goal**: Create manually maintained test fixtures (RSS feeds, transcripts, audio files).

**Tasks:**

1. Create `tests/fixtures/e2e_server/` directory structure:

   ```text

   tests/fixtures/e2e_server/
   ├── feeds/
   │   ├── podcast1/
   │   │   ├── feed.xml
   │   │   └── episodes/
   │   │       ├── episode1/
   │   │       │   ├── transcript.vtt
   │   │       │   ├── transcript.srt
   │   │       │   ├── transcript.json
   │   │       │   ├── transcript.txt
   │   │       │   └── audio.mp3
   │   │       └── episode2/
   │   │           └── ...
   │   ├── podcast2/
   │   │   └── ... (with relative URLs)
   │   └── edge_cases/
   │       ├── malformed_rss.xml
   │       ├── missing_transcript.xml
   │       └── special_chars.xml
   ├── transcripts/
   │   ├── sample.vtt
   │   ├── sample.srt
   │   └── sample.json
   ├── audio/
   │   ├── short_test.mp3  (< 10 seconds)
   │   └── short_test.m4a
   ├── manifest.json (optional documentation)
   └── README.md
   ```

2. Create RSS feed files:
   - `podcast1/feed.xml` - Standard RSS with absolute URLs, multiple episodes
   - `podcast2/feed.xml` - RSS with relative URLs (tests app resolution)
   - `edge_cases/malformed_rss.xml` - Intentionally malformed XML
   - `edge_cases/missing_transcript.xml` - RSS with no transcript URLs
   - `edge_cases/special_chars.xml` - RSS with special characters in titles/descriptions

3. Create transcript files:
   - VTT, SRT, JSON, TXT formats
   - Realistic content (not just "test transcript")

4. Create audio files:
   - Small test files (< 10 seconds) for Whisper testing
   - MP3 and M4A formats

5. Create `manifest.json` (optional) for documentation

6. Create `README.md` documenting fixture structure

**Deliverables:**

- ✅ Complete `tests/fixtures/e2e_server/` directory structure

- ✅ RSS feed files (standard, relative URLs, edge cases)

- ✅ Transcript files (multiple formats)

- ✅ Audio files (small test files)

- ✅ Documentation (`README.md`, optional `manifest.json`)

**Dependencies:** Stage 1

**Estimated Time:** 3-4 hours (manual work)

---

### Stage 3: HTTP Server Integration with Fixtures

**Goal**: Integrate HTTP server with fixture files and verify end-to-end file serving.

**Tasks:**

1. Update `E2EHTTPRequestHandler` to serve files from `tests/fixtures/e2e_server/`

2. Implement routing:
   - `/feeds/{podcast}/feed.xml` → serve RSS feed
   - `/feeds/{podcast}/episodes/{episode}/transcript.{ext}` → serve transcript
   - `/feeds/{podcast}/episodes/{episode}/audio.{ext}` → serve audio
   - `/transcripts/{filename}` → serve standalone transcripts
   - `/audio/{filename}` → serve standalone audio

3. Implement path traversal protection

4. Add URL helper methods to `e2e_server` fixture:
   - `e2e_server.urls.feed(podcast_name)` → return feed URL
   - `e2e_server.urls.episode_transcript(podcast, episode, format)` → return transcript URL
   - `e2e_server.urls.episode_audio(podcast, episode, format)` → return audio URL

5. Create tests verifying:
   - RSS feed serving
   - Transcript file serving
   - Audio file serving
   - Path traversal protection
   - URL helper methods

**Deliverables:**

- ✅ Updated `E2EHTTPRequestHandler` with file serving

- ✅ URL helper methods

- ✅ Tests verifying file serving

- ✅ Security tests (path traversal protection)

**Dependencies:** Stage 2

**Estimated Time:** 3-4 hours

---

### Stage 4: Basic E2E Tests (Happy Paths)

**Goal**: Create E2E tests for basic CLI commands and library API using real HTTP client.

**Tasks:**

1. Create `tests/workflow_e2e/test_basic_e2e.py` with:
   - `test_e2e_cli_basic_transcript_download` - Basic CLI command with transcript download
   - `test_e2e_library_api_basic` - Basic `run_pipeline()` call
   - `test_e2e_service_api_basic` - Basic `service.run()` call

2. Remove HTTP mocking from these tests (use real HTTP client)

3. Use `e2e_server` fixture to serve RSS feeds and transcripts

4. Verify:
   - Real HTTP client is used (no mocking)
   - Files are downloaded correctly
   - Output files are created correctly
   - Network guard prevents external calls

**Deliverables:**

- ✅ `tests/workflow_e2e/test_basic_e2e.py` with 3 basic E2E tests

- ✅ Tests use real HTTP client (no mocking)

- ✅ Tests use `e2e_server` fixture

- ✅ All tests pass

**Dependencies:** Stage 3

**Estimated Time:** 2-3 hours

---

### Stage 5: CLI Command E2E Tests

**Goal**: Create E2E tests for each major CLI command.

**Tasks:**

1. Create `tests/workflow_e2e/test_cli_e2e.py` (or update existing `test_cli.py`) with:
   - `test_e2e_cli_transcript_download` - `podcast-scraper <rss_url>`
   - `test_e2e_cli_whisper_fallback` - `podcast-scraper <rss_url> --transcribe-missing`
   - `test_e2e_cli_config_file` - `podcast-scraper --config <config_file>`
   - `test_e2e_cli_dry_run` - `podcast-scraper <rss_url> --dry-run`
   - `test_e2e_cli_metadata_generation` - `podcast-scraper <rss_url> --generate-metadata`
   - `test_e2e_cli_summarization` - `podcast-scraper <rss_url> --summarize`
   - `test_e2e_cli_all_features` - `podcast-scraper <rss_url> --transcribe-missing --generate-metadata --summarize`

2. Each test should:
   - Use real HTTP client (no mocking)
   - Use `e2e_server` fixture
   - Verify complete workflow from CLI command to output files
   - Verify exit codes and output

**Deliverables:**

- ✅ `tests/workflow_e2e/test_cli_e2e.py` with 7+ CLI E2E tests

- ✅ All major CLI commands covered

- ✅ All tests use real HTTP client

- ✅ All tests pass

**Dependencies:** Stage 4

**Estimated Time:** 4-5 hours

---

### Stage 6: Library API E2E Tests

**Goal**: Create E2E tests for public library API endpoints.

**Tasks:**

1. Create `tests/workflow_e2e/test_library_api_e2e.py` (or update existing `test_workflow_e2e.py`) with:
   - `test_e2e_run_pipeline_basic` - Basic `run_pipeline(config)`
   - `test_e2e_run_pipeline_whisper_fallback` - `run_pipeline()` with `transcribe_missing=True`
   - `test_e2e_run_pipeline_metadata` - `run_pipeline()` with `generate_metadata=True`
   - `test_e2e_run_pipeline_summarization` - `run_pipeline()` with `summarize=True`
   - `test_e2e_run_pipeline_all_features` - `run_pipeline()` with all features enabled
   - `test_e2e_load_config_file` - `load_config_file()` + `run_pipeline()`

2. Each test should:
   - Use real HTTP client (no mocking)
   - Use `e2e_server` fixture
   - Verify return values and output files

**Deliverables:**

- ✅ `tests/workflow_e2e/test_library_api_e2e.py` with 6+ library API E2E tests

- ✅ All major library API endpoints covered

- ✅ All tests use real HTTP client

- ✅ All tests pass

**Dependencies:** Stage 4

**Estimated Time:** 3-4 hours

---

### Stage 7: Service API E2E Tests

**Goal**: Create E2E tests for service API endpoints.

**Tasks:**

1. Update `tests/workflow_e2e/test_service.py` to use real HTTP client:
   - `test_e2e_service_run_basic` - Basic `service.run(config)`
   - `test_e2e_service_run_from_config_file` - `service.run_from_config_file(path)`
   - `test_e2e_service_run_with_features` - `service.run()` with all features
   - `test_e2e_service_main_cli` - `service.main()` CLI entry point

2. Remove HTTP mocking from these tests

3. Use `e2e_server` fixture

4. Verify `ServiceResult` structure and output files

**Deliverables:**

- ✅ Updated `tests/workflow_e2e/test_service.py` with real HTTP client

- ✅ All service API endpoints covered

- ✅ All tests use real HTTP client

- ✅ All tests pass

**Dependencies:** Stage 4

**Estimated Time:** 2-3 hours

---

### Stage 8: Real Whisper Transcription E2E Tests

**Goal**: Create E2E tests using real Whisper models with real audio files.

**Tasks:**

1. Create `tests/workflow_e2e/test_whisper_e2e.py` with:
   - `test_e2e_whisper_transcription_basic` - Real Whisper transcription with small audio file
   - `test_e2e_whisper_fallback_workflow` - Complete workflow with Whisper fallback
   - `test_e2e_whisper_screenplay_format` - Whisper with screenplay formatting
   - `test_e2e_whisper_speaker_detection` - Whisper with speaker detection

2. Use real audio files from `tests/fixtures/e2e_server/audio/`

3. Use real Whisper models (smallest available, e.g., `tiny.en`)

4. Mark tests as `@pytest.mark.slow` and `@pytest.mark.ml_models`

5. Verify:
   - Audio files are downloaded correctly
   - Whisper transcription produces valid output
   - Transcript files are created correctly

**Deliverables:**

- ✅ `tests/workflow_e2e/test_whisper_e2e.py` with 4+ Whisper E2E tests

- ✅ Tests use real Whisper models

- ✅ Tests use real audio files

- ✅ All tests pass

**Dependencies:** Stage 3 (audio files), Stage 4

**Estimated Time:** 3-4 hours

---

### Stage 9: Real ML Models E2E Tests

**Goal**: Create E2E tests using real ML models (spaCy, Transformers) in full workflow context.

**Tasks:**

1. Create `tests/workflow_e2e/test_ml_models_e2e.py` with:
   - `test_e2e_spacy_speaker_detection` - Real spaCy model in full workflow
   - `test_e2e_transformers_summarization` - Real Transformers model in full workflow
   - `test_e2e_all_ml_models_together` - All ML models in complete workflow

2. Use real ML models (smallest available)

3. Mark tests as `@pytest.mark.slow` and `@pytest.mark.ml_models`

4. Verify:
   - Models load correctly
   - Models work in full pipeline context
   - Output files contain expected results

**Deliverables:**

- ✅ `tests/workflow_e2e/test_ml_models_e2e.py` with 3+ ML model E2E tests

- ✅ Tests use real ML models

- ✅ Tests verify models in full workflow context

- ✅ All tests pass

**Dependencies:** Stage 4, Stage 8

**Estimated Time:** 3-4 hours

---

### Stage 10: Error Handling E2E Tests

**Goal**: Create E2E tests for error handling in complete workflows.

**Tasks:**

1. Create `tests/workflow_e2e/test_error_handling_e2e.py` with:
   - `test_e2e_malformed_rss_handling` - Malformed RSS in complete workflow
   - `test_e2e_missing_transcript_handling` - Missing transcript with Whisper fallback
   - `test_e2e_http_error_handling` - HTTP errors (404, 500, timeouts) in workflow
   - `test_e2e_network_timeout_handling` - Network timeout handling
   - `test_e2e_invalid_config_handling` - Invalid config error handling

2. Use `e2e_server` fixture with error scenarios (behavior registry)

3. Verify:
   - Errors are handled gracefully
   - Appropriate error messages are returned
   - Pipeline cleanup happens on error

**Deliverables:**

- ✅ `tests/workflow_e2e/test_error_handling_e2e.py` with 5+ error handling E2E tests

- ✅ Tests use real HTTP client with error scenarios

- ✅ All tests pass

**Dependencies:** Stage 3 (error scenarios), Stage 4

**Estimated Time:** 3-4 hours

---

### Stage 11: Edge Cases E2E Tests

**Goal**: Create E2E tests for edge cases in complete workflows.

**Tasks:**

1. Create `tests/workflow_e2e/test_edge_cases_e2e.py` with:
   - `test_e2e_relative_urls` - RSS feeds with relative URLs
   - `test_e2e_special_characters` - Special characters in titles/descriptions
   - `test_e2e_multiple_transcript_formats` - Multiple transcript formats in one feed
   - `test_e2e_large_episode_count` - Processing many episodes
   - `test_e2e_concurrent_processing` - Concurrent episode processing

2. Use edge case fixtures from `tests/fixtures/e2e_server/feeds/edge_cases/`

3. Verify:
   - Edge cases are handled correctly
   - Output files are created correctly
   - No crashes or data corruption

**Deliverables:**

- ✅ `tests/workflow_e2e/test_edge_cases_e2e.py` with 5+ edge case E2E tests

- ✅ Edge case fixtures created

- ✅ All tests pass

**Dependencies:** Stage 2 (edge case fixtures), Stage 4

**Estimated Time:** 3-4 hours

---

### Stage 12: HTTP Behaviors E2E Tests

**Goal**: Create E2E tests for HTTP behaviors (Range requests, redirects, retries) in full workflows.

**Tasks:**

1. Create `tests/workflow_e2e/test_http_behaviors_e2e.py` with:
   - `test_e2e_range_requests` - HTTP Range requests for audio streaming
   - `test_e2e_http_redirects` - HTTP redirects in workflow
   - `test_e2e_http_retries` - HTTP retry logic in workflow
   - `test_e2e_http_headers` - Custom headers in workflow
   - `test_e2e_http_timeouts` - HTTP timeout handling

2. Use `e2e_server` fixture with HTTP behavior scenarios

3. Verify:
   - HTTP behaviors work correctly in full workflow
   - Retry logic works correctly
   - Timeouts are handled gracefully

**Deliverables:**

- ✅ `tests/workflow_e2e/test_http_behaviors_e2e.py` with 5+ HTTP behavior E2E tests

- ✅ HTTP behavior scenarios implemented

- ✅ All tests pass

**Dependencies:** Stage 1 (HTTP behaviors), Stage 4

**Estimated Time:** 3-4 hours

---

### Stage 13: Migration and Cleanup

**Goal**: Migrate existing E2E tests to use real HTTP client and remove mocks.

**Tasks:**

1. Review all existing E2E tests in `tests/workflow_e2e/`

2. Identify tests that still use HTTP mocking

3. Migrate tests to use `e2e_server` fixture:
   - Remove `unittest.mock.patch` on `downloader.fetch_url`
   - Replace in-memory RSS generation with fixture files
   - Replace mocked audio with real audio files
   - Update tests to use real HTTP client

4. Remove obsolete helper functions (if any)

5. Update test documentation

6. Verify all E2E tests pass

**Deliverables:**

- ✅ All existing E2E tests migrated to real HTTP client

- ✅ All HTTP mocking removed from E2E tests

- ✅ All tests pass

- ✅ Documentation updated

**Dependencies:** All previous stages

**Estimated Time:** 4-6 hours

---

### Stage 14: CI/CD Integration

**Goal**: Integrate E2E tests into CI/CD pipeline.

**Tasks:**

1. Update `.github/workflows/python-app.yml`:
   - Add E2E test job (or update existing `test-workflow-e2e` job)
   - Ensure E2E tests run with proper markers (`-m workflow_e2e`)
   - Add test count validation for E2E tests
   - Configure slow tests to run on schedule or manual trigger

2. Update `Makefile`:
   - Ensure `test-workflow-e2e` target runs E2E tests
   - Add `test-e2e-fast` target for fast E2E tests (exclude slow/ml_models)
   - Add `test-e2e-slow` target for slow E2E tests

3. Update documentation:
   - Document how to run E2E tests
   - Document E2E test structure
   - Document fixture maintenance

**Deliverables:**

- ✅ CI/CD pipeline updated

- ✅ Makefile targets updated

- ✅ Documentation updated

- ✅ E2E tests run in CI

**Dependencies:** All previous stages

**Estimated Time:** 2-3 hours

---

## Summary

### Total Estimated Time: 40-55 hours

### Stage Dependencies

```text

Stage 0 (Network Guard)
  └─> Stage 1 (HTTP Server Infrastructure)
      └─> Stage 2 (Test Fixtures)
          └─> Stage 3 (HTTP Server Integration)
              └─> Stage 4 (Basic E2E Tests)
                  ├─> Stage 5 (CLI Command E2E Tests)
                  ├─> Stage 6 (Library API E2E Tests)
                  ├─> Stage 7 (Service API E2E Tests)
                  ├─> Stage 8 (Real Whisper E2E Tests)
                  ├─> Stage 9 (Real ML Models E2E Tests)
                  ├─> Stage 10 (Error Handling E2E Tests)
                  ├─> Stage 11 (Edge Cases E2E Tests)
                  └─> Stage 12 (HTTP Behaviors E2E Tests)
                      └─> Stage 13 (Migration and Cleanup)
                          └─> Stage 14 (CI/CD Integration)

```

### Coverage Goals

After completing all stages, we will have:

**CLI Commands:**

- ✅ `podcast-scraper <rss_url>` - Basic transcript download

- ✅ `podcast-scraper <rss_url> --transcribe-missing` - Whisper fallback

- ✅ `podcast-scraper --config <config_file>` - Config file workflow

- ✅ `podcast-scraper <rss_url> --dry-run` - Dry run

- ✅ `podcast-scraper <rss_url> --generate-metadata` - Metadata generation

- ✅ `podcast-scraper <rss_url> --summarize` - Summarization

- ✅ `podcast-scraper <rss_url> --transcribe-missing --generate-metadata --summarize` - All features

**Library API:**

- ✅ `run_pipeline(config)` - Basic pipeline

- ✅ `run_pipeline(config)` with all features

- ✅ `load_config_file(path)` + `run_pipeline()`

**Service API:**

- ✅ `service.run(config)` - Basic service API

- ✅ `service.run_from_config_file(path)` - Config file service API

- ✅ `service.main()` - CLI entry point

**Critical Scenarios:**

- ✅ Happy paths (all major workflows)

- ✅ Whisper fallback (real Whisper models)

- ✅ Real ML models in full workflows

- ✅ Error handling (malformed RSS, HTTP errors, timeouts)

- ✅ Edge cases (relative URLs, special characters, multiple formats)

- ✅ HTTP behaviors (Range requests, redirects, retries)

### Success Criteria

1. ✅ All major CLI commands have at least one E2E test

2. ✅ All public API endpoints have at least one E2E test

3. ✅ All E2E tests use real HTTP client (no mocking)

4. ✅ All E2E tests use real data files (RSS feeds, transcripts, audio)

5. ✅ Network guard prevents external network calls

6. ✅ All E2E tests pass in CI/CD

7. ✅ Documentation is complete and up-to-date
