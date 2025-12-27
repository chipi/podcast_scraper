# Test Boundary Decision Framework

## Overview

This document provides clear criteria and decision frameworks to determine whether a test should be an **Integration Test** or an **E2E Test**. The goal is to eliminate ambiguity and provide clear guidance for future test development.

## Key Distinction

**Integration Tests** = Test how **components work together** (component interactions, data flow between modules)

**E2E Tests** = Test **complete user workflows** from entry point to final output (CLI commands, library API calls, full pipelines)

## E2E Test Coverage Goals

### What Should Have E2E Tests

**Every major user-facing entry point should have at least one E2E test:**

1. **CLI Commands** - Each main CLI command should have E2E tests:
   - ✅ `podcast-scraper <rss_url>` - Basic transcript download
   - ✅ `podcast-scraper <rss_url> --transcribe-missing` - Whisper fallback workflow
   - ✅ `podcast-scraper --config <config_file>` - Config file workflow
   - ✅ `podcast-scraper <rss_url> --dry-run` - Dry run workflow
   - ✅ `podcast-scraper <rss_url> --generate-metadata` - Metadata generation workflow
   - ✅ `podcast-scraper <rss_url> --summarize` - Summarization workflow

2. **Library API Endpoints** - Each public API function should have E2E tests:
   - ✅ `run_pipeline(config)` - Main pipeline execution
   - ✅ `service.run(config)` - Service API execution
   - ✅ `service.run_from_config_file(path)` - Config file execution

3. **Critical User Scenarios** - Important workflows should have E2E tests:
   - ✅ Happy path (RSS → transcript download → file output)
   - ✅ Whisper fallback (RSS → no transcript → audio download → Whisper → file output)
   - ✅ Full pipeline with all features (RSS → transcript → metadata → summary → file output)
   - ✅ Error handling in complete workflow (malformed RSS, network errors, etc.)

### What Doesn't Need E2E Tests

**Not every CLI flag combination needs an E2E test:**

- ❌ Every possible `--max-episodes` value (tested in integration/unit tests)

- ❌ Every possible `--timeout` value (tested in integration/unit tests)

- ❌ Every possible config file format variation (tested in integration/unit tests)

- ❌ Edge cases in specific components (tested in integration tests)

**Rule of Thumb**: E2E tests should cover "as a user, I want to..." scenarios, not every possible configuration combination.

### Coverage Strategy

**E2E Test Coverage:**

- **Goal**: Every major user workflow has at least one E2E test

- **Focus**: Happy paths and critical user scenarios

- **Scope**: Complete workflows from entry point to output

- **Not Required**: Every flag combination, every edge case, every configuration option

**Integration Test Coverage:**

- **Goal**: Component interactions and edge cases

- **Focus**: How components work together, specific scenarios

- **Scope**: Component-level testing, not full user workflows

- **Covers**: Edge cases, error scenarios, configuration variations

**Example Coverage:**

| Scenario | E2E Test? | Integration Test? | Unit Test? |
| -------- | --------- | ----------------- | ---------- |
| `podcast-scraper <url>` (happy path) | ✅ Yes | ❌ No | ❌ No |
| `podcast-scraper <url> --max-episodes=5` | ✅ Yes (if different workflow) | ✅ Yes (config validation) | ✅ Yes (config parsing) |
| `podcast-scraper <url> --timeout=30` | ❌ No (same workflow) | ✅ Yes (timeout behavior) | ✅ Yes (timeout parsing) |
| RSS parsing with relative URLs | ❌ No | ✅ Yes | ✅ Yes |
| Error handling in pipeline | ✅ Yes (complete workflow) | ✅ Yes (specific errors) | ❌ No |
| Provider factory → Provider creation | ❌ No | ✅ Yes | ❌ No |
| Config file loading | ❌ No | ✅ Yes | ✅ Yes |

## Decision Criteria

### Integration Tests (`tests/integration/`)

**Use Integration Tests When:**

1. **Testing Component Interactions**
   - Testing how multiple internal components work together
   - Verifying data flow between components (e.g., RSS parser → Episode → Provider → File output)
   - Testing component integration without full pipeline execution

2. **Testing Internal Implementations**
   - Using real internal implementations (Config, factories, providers, workflow logic)
   - Testing real filesystem I/O (temp directories, real file operations)
   - Testing real component logic (not mocked)

3. **Mocking External Services**
   - Mocking HTTP calls (using local test server for speed/reliability)
   - Mocking external APIs (OpenAI, etc.)
   - Mocking ML models (for speed, or testing model integration separately)

4. **Fast Feedback**
   - Tests should run quickly (< 5s each for fast tests)
   - Focused on specific component interactions
   - Can run in parallel

5. **Isolated Scenarios**
   - Testing specific scenarios (error handling, edge cases, specific component combinations)
   - Not testing complete user workflows

**Examples:**

- `test_component_workflows.py` - RSS parsing → Episode creation → Provider usage

- `test_provider_integration.py` - Provider factory → Provider creation → Provider usage

- `test_http_integration.py` - Real HTTP client with local test server

- `test_pipeline_error_recovery.py` - Error handling in pipeline context

- `test_pipeline_concurrent.py` - Concurrent execution testing

### E2E Tests (`tests/workflow_e2e/`)

**Use E2E Tests When:**

1. **Testing Complete User Workflows**
   - Testing CLI commands (`podcast-scraper <rss_url>`)
   - Testing library API calls (`run_pipeline(config)`)
   - Testing service API calls (`service.run(config)`)
   - Testing complete pipelines from entry point to final output

2. **Testing Real HTTP Client in Full Context**
   - Using real HTTP client (`downloader.fetch_url`) without mocking
   - Using local HTTP server (but real HTTP stack, no external network)
   - Testing HTTP behavior in full workflow context (headers, redirects, timeouts, retries)

3. **Testing Real Data Files**
   - Using real RSS feed files (manually maintained in `tests/fixtures/e2e_server/`)
   - Using real transcript files (VTT, SRT, JSON)
   - Using real audio files (small test files for Whisper)

4. **Testing Real ML Models in Full Workflows**
   - Using real ML models (Whisper, spaCy, Transformers) in complete pipeline workflows
   - Testing model loading, initialization, and cleanup in full context
   - Testing that models work correctly together in full pipelines

5. **Testing Production-Like Scenarios**
   - Testing scenarios as users would actually use the system
   - Testing complete workflows with realistic data
   - Testing error recovery in full workflow context

6. **Slower, More Comprehensive**
   - Tests may be slower (< 60s each, may be minutes for full workflows)
   - More comprehensive coverage of complete workflows
   - May include performance/scale tests (marked as slow)

**Examples:**

- `test_workflow_e2e.py` - Complete CLI workflow from RSS URL to output files

- `test_service.py` - Complete service API workflows

- `test_cli.py` - Complete CLI command workflows

- Full pipeline tests with real HTTP server, real data files, real ML models

## Decision Tree

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

```

## Detailed Comparison

| Aspect | Integration Tests | E2E Tests |
| ------ | ------------------ | --------- |
| **Purpose** | Test component interactions | Test complete user workflows |
| **Entry Point** | Component-level (functions, classes) | User-level (CLI, library API, service API) |
| **Scope** | Multiple components working together | Full pipeline from entry to output |
| **HTTP Client** | Mocked or local test server (for speed) | Real HTTP client with local server (no external network) |
| **Data Files** | May use in-memory data or test fixtures | Real data files (RSS feeds, transcripts, audio) |
| **ML Models** | May be mocked (for speed) or real (for model testing) | Real ML models in full workflow context |
| **Speed** | Fast (< 5s each for fast tests) | Slower (< 60s each, may be minutes) |
| **Focus** | Component interactions, data flow | Complete workflows, user scenarios |
| **Examples** | Provider factory → Provider → Usage | CLI command → Full pipeline → Output files |

## Edge Cases and How to Handle Them

### Edge Case 1: HTTP Testing

**Question**: Both integration and E2E tests use local HTTP servers. What's the difference?

**Answer**:

- **Integration Tests**: Use local HTTP server to test HTTP client behavior in isolation (e.g., `test_http_integration.py`). Focus is on HTTP client functionality, not full workflow.

- **E2E Tests**: Use local HTTP server to test HTTP client in full workflow context. Focus is on complete workflow with real HTTP client.

**Decision**: If testing HTTP client behavior in isolation → Integration Test. If testing HTTP client in complete workflow → E2E Test.

### Edge Case 2: Pipeline Error Handling

**Question**: `test_pipeline_error_recovery.py` is in integration tests, but it tests pipeline errors. Should it be E2E?

**Answer**:

- **Current**: Integration test because it focuses on specific error scenarios and uses mocked HTTP.

- **Future**: Could be E2E test if it tests error handling in complete workflow with real HTTP client and real data files.

**Decision**: If testing error handling in complete workflow with real HTTP client → E2E Test. If testing specific error scenarios with mocked HTTP → Integration Test.

### Edge Case 3: Concurrent Execution

**Question**: `test_pipeline_concurrent.py` tests concurrent execution. Is it integration or E2E?

**Answer**:

- **Current**: Integration test because it focuses on concurrent execution behavior.

- **Could be E2E**: If testing concurrent execution in complete workflow with real HTTP client and real data files.

**Decision**: If testing concurrent execution in complete workflow → E2E Test. If testing concurrent execution behavior in isolation → Integration Test.

### Edge Case 4: Real ML Models

**Question**: Integration tests can use real ML models. How is that different from E2E tests?

**Answer**:

- **Integration Tests**: Use real ML models to test model integration (e.g., `test_provider_real_models.py`). Focus is on model loading, initialization, and basic functionality.

- **E2E Tests**: Use real ML models in complete workflow context. Focus is on models working together in full pipelines.

**Decision**: If testing model integration in isolation → Integration Test. If testing models in complete workflow → E2E Test.

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

## Examples

### Example 1: RSS Parsing

**Integration Test** (`test_component_workflows.py`):

```python
def test_rss_parsing_to_episode_workflow():
    """Test RSS parsing → Episode creation workflow."""
    rss_xml = build_rss_xml_with_transcript(...)
    feed = parse_rss_feed(rss_xml, base_url)
    episode = create_episode_from_feed_item(feed.items[0])
    assert episode.title == "Episode 1"
    # Tests component interaction, not full pipeline

```

**E2E Test** (`test_workflow_e2e.py`):

```python
def test_e2e_cli_transcript_download(e2e_server):
    """Test complete CLI workflow from RSS URL to output files."""
    rss_url = e2e_server.urls.feed("podcast1")
    exit_code = cli.main([rss_url, "--output-dir", tmpdir])
    assert exit_code == 0
    assert output_file.exists()
    # Tests complete user workflow

```

### Example 2: Error Handling

**Integration Test** (`test_pipeline_error_recovery.py`):

```python
def test_pipeline_handles_malformed_rss_feed():
    """Test pipeline handles malformed RSS."""
    # Uses mocked HTTP, focuses on error handling
    with patch("podcast_scraper.downloader.fetch_url") as mock_fetch:
        mock_fetch.return_value = create_rss_response(malformed_xml)
        # Test error handling in pipeline context

```

**E2E Test** (future):

```python
def test_e2e_malformed_rss_handling(e2e_server):
    """Test complete workflow handles malformed RSS."""
    # Uses real HTTP client, real data files
    rss_url = e2e_server.urls.feed("malformed_rss")
    result = run_pipeline(Config(rss=rss_url))
    # Test error handling in complete workflow

```

### Example 3: HTTP Client Testing

**Integration Test** (`test_http_integration.py`):

```python
def test_successful_http_request(test_http_server):
    """Test HTTP client with local test server."""
    response = fetch_url(f"{test_http_server.base_url}/success")
    assert response.status_code == 200
    # Tests HTTP client behavior in isolation

```

**E2E Test** (future):

```python
def test_e2e_http_client_in_workflow(e2e_server):
    """Test HTTP client in complete workflow."""
    rss_url = e2e_server.urls.feed("podcast1")
    result = run_pipeline(Config(rss=rss_url))
    # Tests HTTP client in full workflow context

```

## Summary

**Integration Tests** = Component interactions, fast feedback, mocked external services

**E2E Tests** = Complete user workflows, real HTTP client, real data files, real ML models in full context

**Key Question**: "Am I testing how components work together, or am I testing a complete user workflow?"

- **Components together** → Integration Test

- **Complete user workflow** → E2E Test

## E2E Test Coverage Goals

**Goal**: Every major user-facing entry point (CLI command, public API function) should have at least one E2E test covering the happy path and critical scenarios.

**Not Required**: Every CLI flag combination, every configuration option, or every edge case needs an E2E test. Those are covered by Integration and Unit tests.

**Coverage Strategy**:

- **E2E Tests**: Cover "as a user, I want to..." scenarios (happy paths, critical workflows)

- **Integration Tests**: Cover component interactions, edge cases, configuration variations

- **Unit Tests**: Cover individual functions, parsing, validation, edge cases

## Future Considerations

As we implement the E2E HTTP mocking server infrastructure:

1. **Integration tests** will continue to use local HTTP servers for HTTP client testing in isolation

2. **E2E tests** will use the E2E HTTP server (`e2e_server` fixture) for complete workflow testing

3. **Clear separation**: Integration tests focus on components, E2E tests focus on workflows

4. **Migration path**: Some integration tests may migrate to E2E tests when they test complete workflows with real HTTP client
