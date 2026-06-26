# ADR-056: Composable E2E Mock Response Strategy

- **Status**: Superseded (the underlying RFC-054 work shipped in a different shape; see below)
- **Date**: 2026-02-05 (Proposed) · 2026-06-26 (Superseded)
- **Authors**: Podcast Scraper Team
- **Related RFCs**: [RFC-054](../rfc/RFC-054-e2e-mock-response-strategy.md) (now Completed via redirect — per-provider mock-client path, not the centralized router this ADR drafted)
- **Related Issues**: #135 (closed), #399 (closed), #401 (closed)

## Superseded — what actually shipped (2026-06-26)

The composable-router architecture this ADR proposed is **not what we built**. The three
linked issues all closed, but the work landed as:

- **Per-provider mock clients** (`tests/fixtures/mock_server/gemini_mock_client.py`,
  `mistral_mock_client.py`) carrying the functional response shapes per vendor.
- **Per-concern unit-test suites** for the non-functional behaviour:
  - `tests/unit/podcast_scraper/utils/test_retryable_errors.py`
  - `tests/unit/podcast_scraper/utils/test_retry_integration.py`
  - `tests/unit/podcast_scraper/utils/test_llm_circuit_breaker.py`
  - `tests/unit/podcast_scraper/utils/test_provider_metrics.py`
  - `tests/unit/podcast_scraper/test_download_resilience.py`

That's the same goals delivered (functional ↔ non-functional separation, 429 / 5xx /
timeout / retry / circuit-breaker coverage) without the shared response-profile router
this ADR sketched. The router would have been infrastructure overhead for a problem the
per-suite scaffolding already solves.

This ADR is preserved as the historical decision record. The shipping approach is
documented inline in RFC-054 (Status: Completed). No new decision is needed unless we
later consolidate the per-suite scaffolding into a shared router — that would warrant a
follow-on ADR.

## Context & Problem Statement

E2E tests use mock HTTP servers to simulate API responses from providers (OpenAI, Gemini, etc.). Currently, all tests see identical mock responses, making it impossible to test:

- Error scenarios (429 rate limits, 5xx errors, connection failures)
- Retry logic and exponential backoff
- Timeout handling
- Provider-specific behaviors and edge cases
- Non-functional concerns (retries, timeouts) mixed with functional responses

Without composable mocks:

- Tests can't verify error handling
- Retry logic can't be tested
- Each test that needs a different scenario must duplicate mock setup
- Common patterns (retry behavior, error responses) are repeated across tests

## Decision

We adopt a **Composable E2E Mock Response Strategy** that separates:

1. **Functional Responses** (API response structure): Normal responses, error responses, edge cases
2. **Non-Functional Behavior** (retries, timeouts, rate limits): Configured separately and composed with functional responses
3. **Response Router**: Composes functional responses with non-functional behavior based on test configuration
4. **Default Normal Responses**: Most tests get realistic but simple responses automatically (no configuration needed)

## Rationale

- **Test Variety**: Enables testing error scenarios, edge cases, and advanced features without duplication
- **Separation of Concerns**: Functional responses (what the API returns) are separate from non-functional behavior (how it behaves)
- **Composability**: Tests can mix "normal response" with "retry behavior" or "error response" with "timeout behavior"
- **Centralized Logic**: Non-functional behavior (retry logic, timeouts) is maintained in one place
- **Default Simplicity**: Most tests use realistic default responses automatically

## Alternatives Considered

1. **Per-Test Mock Setup**: Rejected as it leads to duplication and makes it hard to maintain consistency.
2. **Hardcoded Response Scenarios**: Rejected as it doesn't allow composition and requires creating separate mocks for every combination.
3. **No Mock Variety**: Rejected as it prevents testing error handling and retry logic.

## Consequences

- **Positive**:
  - Enables comprehensive error testing (429, 5xx, timeouts)
  - Tests can verify retry logic and exponential backoff
  - No duplication of common patterns
  - Centralized non-functional logic
  - Easy to add new response profiles
- **Negative**:
  - Initial implementation complexity
  - Tests need to understand the composition model
- **Neutral**:
  - Requires implementation of RFC-054

## Implementation Notes

- **Module**: `tests/e2e/` - E2E test infrastructure
- **Pattern**: Response Profile + Non-Functional Behavior composition
- **Architecture**:

  ```text
  Test Request → Mock Response Router →
  Response Profile (functional) + Non-Functional Behavior → Response
  ```

- **Response Profiles**: Normal, Error (429, 5xx), Edge Case (partial, degraded)
- **Non-Functional Behavior**: Retry logic, timeouts, rate limits
- **Default**: Most tests get realistic default responses automatically
- **Status**:  Draft RFC (RFC-054) - Not yet implemented

## References

- [RFC-054: Flexible E2E Mock Response Strategy](../rfc/RFC-054-e2e-mock-response-strategy.md)
- [Issue #135: Make e2e LLM API mocks realistic and comprehensive](https://github.com/chipi/podcast_scraper/issues/135)
- [Issue #399: Provider-level concerns to harden](https://github.com/chipi/podcast_scraper/issues/399)
- [Issue #401: DevEx / Ops Improvements](https://github.com/chipi/podcast_scraper/issues/401)
