# RFC-054: Flexible E2E Mock Response Strategy

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, developers writing E2E tests, CI/CD pipeline maintainers
- **Related Issues**:
  - #135: Make e2e LLM API mocks realistic and comprehensive
  - #399: Provider-level concerns to harden
  - #401: DevEx / Ops Improvements
- **Related RFCs**:
  - `docs/rfc/RFC-019-e2e-test-improvements.md` (E2E test infrastructure - foundation)
  - `docs/rfc/RFC-020-integration-test-improvements.md` (integration test improvements)

## Abstract

This RFC defines a flexible strategy for E2E mock responses that supports both normal functional testing and advanced error handling scenarios. The design separates **non-functional concerns** (retries, timeouts, rate limits) from **functional responses** (API response structures), allowing tests to mix and match different scenarios without duplication.

**Key Principles:**

- **Default Normal Responses**: Most tests use realistic but simple responses for happy-path testing
- **Variety of Response Scenarios**: Advanced tests can select specific response profiles (errors, edge cases, special behaviors)
- **Centralized Non-Functional Logic**: Retry behavior, timeouts, rate limits handled in one place
- **Composable Design**: Tests can combine different functional responses with different non-functional behaviors
- **No Test Duplication**: Common patterns are reusable, not repeated in every test

## Problem Statement

**Current Issues:**

1. **All Tests See Same Responses**
   - Every test encounters identical mock responses
   - Cannot test error scenarios, edge cases, or advanced features
   - Hard to test retry logic, rate limiting, or timeout handling

2. **No Separation of Concerns**
   - Functional responses (API response structure) mixed with non-functional behavior (retries, delays)
   - Hard to test "normal response with retry" vs "error response with retry"
   - Changes to retry logic require updating every mock

3. **Limited Scenario Coverage**
   - Only happy-path responses are mocked
   - Cannot test 429 rate limits, 5xx errors, connection failures
   - Cannot test provider-specific behaviors or edge cases

4. **Duplication and Maintenance Burden**
   - Each test that needs a different scenario must duplicate mock setup
   - Common patterns (retry behavior, error responses) repeated across tests
   - Hard to maintain consistency across test suite

## Goals

1. **Normal Responses by Default**
   - Most tests get realistic, simple responses automatically
   - No configuration needed for basic happy-path testing
   - Fast test execution with minimal setup

2. **Variety of Response Scenarios**
   - Tests can select specific response profiles (errors, edge cases, special behaviors)
   - Support for all error types: 429, 5xx, connection failures, timeouts
   - Support for provider-specific behaviors and edge cases

3. **Centralized Non-Functional Logic**
   - Retry behavior, timeouts, rate limits configured in one place
   - Tests can override non-functional behavior without duplicating functional responses
   - Consistent retry logging and metrics across all tests

4. **Composable Design**
   - Tests can mix functional responses with non-functional behaviors
   - Example: "normal summarization response" + "retry on first attempt" + "rate limit after 2 retries"
   - No need to create separate mocks for every combination

## Design

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Request                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Mock Response Router                           │
│  - Determines response profile (normal/error/edge case)    │
│  - Applies non-functional behavior (retries/timeouts)      │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Response   │ │   Response   │ │   Response   │
│   Profiles   │ │   Profiles   │ │   Profiles   │
│  (Functional)│ │  (Functional)│ │  (Functional)│
│              │ │              │ │              │
│ - Normal     │ │ - Error      │ │ - Edge Case  │
│ - Success    │ │ - 429        │ │ - Partial    │
│ - Standard   │ │ - 5xx        │ │ - Degraded   │
└──────────────┘ └──────────────┘ └──────────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Non-Functional Behavior Layer                        │
│  - Retry logic (attempt tracking, exponential backoff)     │
│  - Timeout simulation (connect, read, overall)              │
│  - Rate limit handling (429 responses, retry-after headers) │
│  - Metrics tracking (retries, sleep time, tokens)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    HTTP Response                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Response Profiles (Functional)

Response profiles define the **functional content** of API responses (what data is returned).

**Base Profile: `normal`**
- Realistic API response structure matching provider documentation
- Standard success responses with proper fields
- Default for most tests (happy path)

**Error Profiles:**
- `rate_limit_429`: Returns 429 with Retry-After header
- `server_error_500`: Returns 500 Internal Server Error
- `server_error_503`: Returns 503 Service Unavailable
- `connection_error`: Simulates connection failure
- `timeout_error`: Simulates timeout (no response)

**Edge Case Profiles:**
- `empty_response`: Empty content (tests graceful handling)
- `malformed_json`: Invalid JSON structure (tests parsing)
- `partial_response`: Incomplete data (tests degradation)
- `large_response`: Very large payload (tests memory/performance)

**Provider-Specific Profiles:**
- `openai_verbose_json`: OpenAI verbose JSON format with segments
- `gemini_multimodal`: Gemini multimodal response structure
- `anthropic_streaming`: Anthropic streaming response format

#### 2. Non-Functional Behavior (Centralized)

Non-functional behavior controls **how** responses are delivered (retries, delays, timeouts).

**Retry Behavior:**
- `no_retries`: Always succeeds on first attempt
- `retry_once`: Fails first attempt, succeeds on retry
- `retry_twice`: Fails first two attempts, succeeds on third
- `retry_exhausted`: Always fails (tests retry exhaustion)
- `exponential_backoff`: Configurable backoff delays

**Timeout Behavior:**
- `no_timeout`: Immediate response
- `connect_timeout`: Simulates connection timeout
- `read_timeout`: Simulates read timeout
- `overall_timeout`: Simulates overall operation timeout

**Rate Limit Behavior:**
- `no_rate_limit`: No rate limiting
- `rate_limit_after_n`: Rate limit after N requests
- `rate_limit_retry_after`: Configurable Retry-After header

**Metrics Tracking:**
- Automatic tracking of retries, sleep time, tokens
- Consistent with production metrics format
- Supports validation in tests

#### 3. Response Router

The response router combines functional profiles with non-functional behavior.

**Configuration Methods:**

```python
# Test-level configuration (via fixture or decorator)
@pytest.mark.mock_profile("normal")  # Default
@pytest.mark.mock_profile("rate_limit_429")
@pytest.mark.mock_profile("server_error_500")

# Non-functional behavior
@pytest.mark.mock_retry("retry_once")
@pytest.mark.mock_timeout("no_timeout")
@pytest.mark.mock_rate_limit("no_rate_limit")

# Composable: mix and match
@pytest.mark.mock_profile("normal")
@pytest.mark.mock_retry("retry_twice")
@pytest.mark.mock_rate_limit("rate_limit_after_2")
```

**Programmatic Configuration:**

```python
def test_with_custom_behavior(e2e_server):
    # Configure specific endpoint
    e2e_server.set_response_profile(
        endpoint="/v1/chat/completions",
        profile="normal",
        retry="retry_once",
        rate_limit="no_rate_limit"
    )

    # Run test
    result = run_pipeline(...)
    assert result.success
```

### Implementation Structure

```
tests/e2e/fixtures/
├── mock_responses/
│   ├── __init__.py
│   ├── profiles.py              # Response profile definitions
│   │   ├── normal.py            # Normal/success responses
│   │   ├── errors.py            # Error responses (429, 5xx, etc.)
│   │   ├── edge_cases.py       # Edge case responses
│   │   └── provider_specific.py # Provider-specific formats
│   ├── behavior.py              # Non-functional behavior (retries, timeouts)
│   └── router.py                # Response router (combines profiles + behavior)
├── e2e_http_server.py          # Updated to use router
└── conftest.py                  # Test fixtures and markers
```

### Example Usage

#### Example 1: Normal Response (Default)

```python
def test_basic_summarization(e2e_server):
    # No configuration needed - uses default "normal" profile
    result = run_pipeline(...)
    assert result.success
    assert result.summary is not None
```

#### Example 2: Rate Limit with Retry

```python
@pytest.mark.mock_profile("rate_limit_429")
@pytest.mark.mock_retry("retry_twice")
def test_rate_limit_handling(e2e_server):
    # First two attempts get 429, third succeeds
    result = run_pipeline(...)
    assert result.success
    # Verify retry metrics were logged
    assert "provider_retry: provider=openai attempt=2" in logs
```

#### Example 3: Server Error with Graceful Degradation

```python
@pytest.mark.mock_profile("server_error_500")
@pytest.mark.mock_retry("retry_exhausted")
def test_graceful_degradation(e2e_server):
    # All retries fail, but transcript should still be saved
    result = run_pipeline(...)
    assert result.transcript is not None
    assert result.summary is None  # Summarization failed
    assert result.status == "degraded"
```

#### Example 4: Provider-Specific Format

```python
@pytest.mark.mock_profile("openai_verbose_json")
def test_openai_segments(e2e_server):
    # Tests OpenAI-specific verbose JSON format with segments
    result = run_pipeline(...)
    assert result.segments is not None
    assert len(result.segments) > 0
```

#### Example 5: Programmatic Configuration

```python
def test_dynamic_scenario(e2e_server):
    # Configure different behavior for different endpoints
    e2e_server.set_response_profile(
        endpoint="/v1/chat/completions",
        profile="normal",
        retry="retry_once"
    )
    e2e_server.set_response_profile(
        endpoint="/v1/audio/transcriptions",
        profile="normal",
        retry="no_retries"
    )

    result = run_pipeline(...)
    assert result.success
```

## Test Structure and Organization

### Current Test Organization

**Existing Tests (Backward Compatible):**
- All existing E2E tests continue to work unchanged
- They automatically use the default "normal" profile
- No changes needed to existing test code
- ~25-30 existing E2E test files continue as-is

**Example existing test (unchanged):**
```python
# tests/e2e/test_basic_e2e.py
def test_basic_transcript_download(e2e_server):
    # Uses default "normal" profile automatically
    result = run_pipeline(...)
    assert result.success
```

### New Test Organization

**New Test Files for Advanced Scenarios:**

We'll add **3-5 new test files** focused on error handling and advanced scenarios:

1. **`test_provider_error_handling_e2e.py`** (~10-15 tests)
   - Rate limit scenarios (429 errors)
   - Server error scenarios (5xx errors)
   - Connection failures
   - Timeout scenarios

2. **`test_provider_retry_behavior_e2e.py`** (~8-12 tests)
   - Retry on rate limits
   - Retry on transient errors
   - Retry exhaustion
   - Exponential backoff verification

3. **`test_provider_edge_cases_e2e.py`** (~6-10 tests)
   - Empty responses
   - Malformed JSON
   - Partial responses
   - Large responses

4. **`test_provider_graceful_degradation_e2e.py`** (~5-8 tests)
   - Summarization failure (transcript still saved)
   - Entity extraction failure (summary still produced)
   - Provider fallback scenarios

5. **`test_provider_metrics_tracking_e2e.py`** (~5-8 tests)
   - Token usage tracking
   - Cost calculation
   - Retry metrics
   - Rate limit sleep time

**Total New Tests: ~34-53 tests across 5 files**

### Test Structure Examples

#### Example 1: Error Handling Test File

```python
# tests/e2e/test_provider_error_handling_e2e.py

@pytest.mark.e2e
@pytest.mark.provider_errors
class TestProviderErrorHandling:
    """Test provider error handling scenarios."""

    @pytest.mark.mock_profile("rate_limit_429")
    @pytest.mark.mock_retry("retry_twice")
    def test_rate_limit_with_retry_succeeds(self, e2e_server):
        """Test that rate limit errors are retried and eventually succeed."""
        result = run_pipeline(...)
        assert result.success
        # Verify retry was logged
        assert "provider_retry: provider=openai attempt=2" in caplog.text

    @pytest.mark.mock_profile("rate_limit_429")
    @pytest.mark.mock_retry("retry_exhausted")
    def test_rate_limit_retry_exhausted(self, e2e_server):
        """Test that exhausted retries fail gracefully."""
        result = run_pipeline(...)
        assert not result.success
        assert "rate limit" in result.error_message.lower()

    @pytest.mark.mock_profile("server_error_500")
    @pytest.mark.mock_retry("retry_once")
    def test_server_error_with_retry(self, e2e_server):
        """Test that 500 errors are retried."""
        result = run_pipeline(...)
        assert result.success  # Succeeds on retry

    @pytest.mark.mock_profile("server_error_503")
    @pytest.mark.mock_retry("retry_exhausted")
    def test_service_unavailable_handling(self, e2e_server):
        """Test 503 Service Unavailable handling."""
        result = run_pipeline(...)
        assert not result.success
        assert result.status == "degraded"

    @pytest.mark.mock_profile("connection_error")
    @pytest.mark.mock_retry("retry_twice")
    def test_connection_error_handling(self, e2e_server):
        """Test connection error handling."""
        result = run_pipeline(...)
        # May succeed on retry or fail depending on scenario
        assert result.success or result.status == "degraded"
```

#### Example 2: Retry Behavior Test File

```python
# tests/e2e/test_provider_retry_behavior_e2e.py

@pytest.mark.e2e
@pytest.mark.provider_retries
class TestProviderRetryBehavior:
    """Test provider retry behavior and metrics."""

    @pytest.mark.mock_profile("normal")
    @pytest.mark.mock_retry("no_retries")
    def test_no_retries_needed(self, e2e_server):
        """Test normal flow with no retries."""
        result = run_pipeline(...)
        assert result.success
        # Verify no retry metrics
        assert result.metrics.retries == 0

    @pytest.mark.mock_profile("rate_limit_429")
    @pytest.mark.mock_retry("retry_once")
    def test_single_retry_succeeds(self, e2e_server):
        """Test single retry succeeds."""
        result = run_pipeline(...)
        assert result.success
        assert result.metrics.retries == 1
        assert result.metrics.rate_limit_sleep_sec > 0

    @pytest.mark.mock_profile("rate_limit_429")
    @pytest.mark.mock_retry("retry_twice")
    def test_multiple_retries_succeed(self, e2e_server):
        """Test multiple retries eventually succeed."""
        result = run_pipeline(...)
        assert result.success
        assert result.metrics.retries == 2

    @pytest.mark.mock_profile("rate_limit_429")
    @pytest.mark.mock_retry("exponential_backoff")
    def test_exponential_backoff(self, e2e_server):
        """Test exponential backoff delays."""
        result = run_pipeline(...)
        assert result.success
        # Verify backoff delays increase
        assert result.metrics.rate_limit_sleep_sec > 0
        # Verify retry logs show increasing delays
        assert "sleep=1.0" in caplog.text
        assert "sleep=2.0" in caplog.text
```

#### Example 3: Graceful Degradation Test File

```python
# tests/e2e/test_provider_graceful_degradation_e2e.py

@pytest.mark.e2e
@pytest.mark.graceful_degradation
class TestGracefulDegradation:
    """Test graceful degradation when components fail."""

    @pytest.mark.mock_profile("server_error_500")
    @pytest.mark.mock_retry("retry_exhausted")
    def test_summarization_failure_saves_transcript(self, e2e_server):
        """Test that transcript is saved even if summarization fails."""
        result = run_pipeline(...)
        # Transcript should be saved
        assert result.transcript is not None
        assert os.path.exists(result.transcript_path)
        # Summary should be None
        assert result.summary is None
        # Status should indicate degradation
        assert result.status == "degraded"

    @pytest.mark.mock_profile("normal")
    @pytest.mark.mock_profile("server_error_500", endpoint="/v1/chat/completions")
    def test_partial_failure_handling(self, e2e_server):
        """Test handling when one provider fails but others succeed."""
        # Transcription succeeds, summarization fails
        result = run_pipeline(...)
        assert result.transcript is not None
        assert result.summary is None
        assert result.status == "partial"
```

### Test Execution Patterns

**Default Execution (Most Tests):**
```bash
# All existing tests + new normal-scenario tests
pytest tests/e2e/ -m e2e

# ~25-30 existing test files (unchanged)
# ~5 new test files with normal scenarios
# Total: ~30-35 test files, ~100-150 tests
```

**Error Handling Tests (New):**
```bash
# Run only error handling tests
pytest tests/e2e/ -m "e2e and provider_errors"

# ~10-15 tests in test_provider_error_handling_e2e.py
```

**Retry Behavior Tests (New):**
```bash
# Run only retry behavior tests
pytest tests/e2e/ -m "e2e and provider_retries"

# ~8-12 tests in test_provider_retry_behavior_e2e.py
```

**All Advanced Scenarios:**
```bash
# Run all advanced scenario tests
pytest tests/e2e/ -m "e2e and (provider_errors or provider_retries or edge_cases or graceful_degradation)"

# ~34-53 new tests across 5 files
```

**CI/CD Execution:**
```bash
# Fast tests (normal scenarios only)
pytest tests/e2e/ -m "e2e and not (provider_errors or provider_retries or edge_cases)"

# Slow tests (advanced scenarios)
pytest tests/e2e/ -m "e2e and (provider_errors or provider_retries or edge_cases)"
```

### Test Count Summary

| Category | Test Files | Tests | Notes |
|----------|-----------|-------|-------|
| **Existing Tests** | ~25-30 | ~100-120 | Unchanged, use default "normal" profile |
| **Error Handling** | 1 | ~10-15 | New: rate limits, server errors, timeouts |
| **Retry Behavior** | 1 | ~8-12 | New: retry logic, backoff, metrics |
| **Edge Cases** | 1 | ~6-10 | New: empty, malformed, partial responses |
| **Graceful Degradation** | 1 | ~5-8 | New: partial failures, fallbacks |
| **Metrics Tracking** | 1 | ~5-8 | New: tokens, costs, retry metrics |
| **Total** | ~30-35 | ~134-173 | Mix of existing (unchanged) + new |

### Test Consumption Pattern

**Pattern 1: Default (No Configuration)**
```python
# 90% of tests - just work with defaults
def test_basic_functionality(e2e_server):
    result = run_pipeline(...)
    assert result.success
```

**Pattern 2: Decorator-Based (Simple Scenarios)**
```python
# 5-10% of tests - use decorators for specific scenarios
@pytest.mark.mock_profile("rate_limit_429")
@pytest.mark.mock_retry("retry_twice")
def test_rate_limit_handling(e2e_server):
    result = run_pipeline(...)
    assert result.success
```

**Pattern 3: Programmatic (Complex Scenarios)**
```python
# 1-2% of tests - programmatic configuration for complex scenarios
def test_custom_scenario(e2e_server):
    e2e_server.set_response_profile(
        endpoint="/v1/chat/completions",
        profile="normal",
        retry="retry_once"
    )
    e2e_server.set_response_profile(
        endpoint="/v1/audio/transcriptions",
        profile="rate_limit_429",
        retry="retry_twice"
    )
    result = run_pipeline(...)
    assert result.success
```

## Implementation Plan

### Phase 1: Response Profile System

**Goal**: Create base response profile system with normal responses.

- [ ] Create `mock_responses/profiles.py` with base profile classes
- [ ] Implement `normal` profile for all providers (OpenAI, Gemini, Anthropic, etc.)
- [ ] Update `e2e_http_server.py` to use profile system
- [ ] Add default profile selection (normal for all endpoints)
- [ ] Update existing tests to use profile system (backward compatible)

**Deliverables:**
- Response profile base classes
- Normal response implementations for all providers
- Integration with existing e2e server

### Phase 2: Non-Functional Behavior Layer

**Goal**: Centralize retry, timeout, and rate limit behavior.

- [ ] Create `mock_responses/behavior.py` with behavior classes
- [ ] Implement retry behavior (no_retries, retry_once, retry_twice, etc.)
- [ ] Implement timeout behavior (no_timeout, connect_timeout, read_timeout)
- [ ] Implement rate limit behavior (no_rate_limit, rate_limit_after_n)
- [ ] Add metrics tracking (retries, sleep time, tokens)
- [ ] Integrate with response router

**Deliverables:**
- Non-functional behavior classes
- Metrics tracking integration
- Retry logging (matches production format)

### Phase 3: Response Router

**Goal**: Combine profiles and behavior into unified router.

- [ ] Create `mock_responses/router.py` with router class
- [ ] Implement profile + behavior composition
- [ ] Add test-level configuration (pytest markers)
- [ ] Add programmatic configuration (e2e_server methods)
- [ ] Update e2e_http_server to use router

**Deliverables:**
- Response router implementation
- Pytest markers for test configuration
- Programmatic configuration API

### Phase 4: Error Profiles

**Goal**: Add error response profiles for comprehensive error testing.

- [ ] Implement error profiles (429, 5xx, connection, timeout)
- [ ] Add proper HTTP headers (Retry-After, etc.)
- [ ] Add error response formats matching real APIs
- [ ] Create tests for error scenarios

**Deliverables:**
- Error response profiles
- Error scenario tests
- Documentation

### Phase 5: Edge Case Profiles

**Goal**: Add edge case profiles for advanced testing.

- [ ] Implement edge case profiles (empty, malformed, partial, large)
- [ ] Add provider-specific profiles (OpenAI verbose JSON, Gemini multimodal, etc.)
- [ ] Create tests for edge cases
- [ ] Document edge case scenarios

**Deliverables:**
- Edge case response profiles
- Provider-specific profiles
- Edge case tests

### Phase 6: Integration and Documentation

**Goal**: Complete integration and comprehensive documentation.

- [ ] Update all existing tests to use new system (backward compatible)
- [ ] Create comprehensive test examples
- [ ] Document all profiles and behaviors
- [ ] Add migration guide for existing tests
- [ ] Update issue #135 with completion status

**Deliverables:**
- Updated test suite
- Comprehensive documentation
- Migration guide
- Test examples

## Benefits

1. **Flexibility**: Tests can easily select different response scenarios
2. **No Duplication**: Common patterns centralized, not repeated
3. **Maintainability**: Changes to retry logic in one place
4. **Composability**: Mix and match functional and non-functional behaviors
5. **Realistic Testing**: Error scenarios match real API behavior
6. **Comprehensive Coverage**: Can test all error types and edge cases
7. **Backward Compatible**: Existing tests continue to work (default to normal)

## Risks and Mitigations

**Risk**: Complexity increase in test infrastructure
- **Mitigation**: Clear separation of concerns, well-documented API, backward compatibility

**Risk**: Test execution time increase
- **Mitigation**: Default behavior is fast (normal responses, no retries), advanced scenarios opt-in

**Risk**: Maintenance burden of response profiles
- **Mitigation**: Profiles match real API documentation, centralized in one place

**Risk**: Test flakiness from complex scenarios
- **Mitigation**: Deterministic behavior, clear configuration, comprehensive documentation

## Alternatives Considered

### Alternative 1: Per-Test Mock Setup
- **Pros**: Simple, explicit
- **Cons**: Duplication, hard to maintain, no centralized logic

### Alternative 2: Configuration Files
- **Pros**: External configuration, easy to modify
- **Cons**: Less flexible, harder to programmatically configure

### Alternative 3: Separate Mock Servers
- **Pros**: Complete isolation
- **Cons**: More complex, harder to maintain, duplication

## Open Questions

1. Should response profiles be provider-agnostic or provider-specific?
   - **Proposal**: Provider-specific (matches real APIs), but with common base classes

2. How to handle provider-specific error formats?
   - **Proposal**: Provider-specific error profiles that match real error formats

3. Should metrics tracking be optional or always enabled?
   - **Proposal**: Always enabled (matches production), but can be validated or ignored in tests

4. How to handle streaming responses?
   - **Proposal**: Separate streaming profiles, handled in Phase 5

## Related Work

- Issue #135: Make e2e LLM API mocks realistic and comprehensive
- Issue #399: Provider-level concerns to harden (retry policy, timeouts)
- Issue #401: DevEx / Ops Improvements (structured logs, graceful degradation)
- RFC-019: E2E Test Infrastructure and Coverage Improvements

## Acceptance Criteria

- [ ] Response profile system implemented with normal responses
- [ ] Non-functional behavior layer implemented (retries, timeouts, rate limits)
- [ ] Response router implemented (combines profiles + behavior)
- [ ] Error profiles implemented (429, 5xx, connection, timeout)
- [ ] Edge case profiles implemented (empty, malformed, partial, large)
- [ ] Provider-specific profiles implemented (OpenAI, Gemini, Anthropic, etc.)
- [ ] Pytest markers for test configuration
- [ ] Programmatic configuration API
- [ ] All existing tests updated (backward compatible)
- [ ] Comprehensive documentation
- [ ] Test examples for all scenarios
- [ ] Issue #135 updated with completion status

## Timeline

- **Phase 1**: 1-2 weeks (Response Profile System)
- **Phase 2**: 1-2 weeks (Non-Functional Behavior Layer)
- **Phase 3**: 1 week (Response Router)
- **Phase 4**: 1-2 weeks (Error Profiles)
- **Phase 5**: 1-2 weeks (Edge Case Profiles)
- **Phase 6**: 1 week (Integration and Documentation)

**Total**: 6-10 weeks

---

**Status**: This RFC is in draft status and open for feedback. Implementation should begin after review and approval.
