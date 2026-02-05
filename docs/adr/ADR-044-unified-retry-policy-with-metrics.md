# ADR-044: Unified Retry Policy with Metrics

- **Status**: Accepted
- **Date**: 2026-02-05
- **Authors**: Podcast Scraper Team
- **Related Issues**: #399 (Provider-level concerns to harden)

## Context & Problem Statement

API-based providers (OpenAI, Gemini, Anthropic, etc.) can experience transient failures (429 rate limits, 5xx errors, connection resets). Each provider initially implemented its own retry logic with different behaviors, delays, and logging formats. This led to:

- Inconsistent retry behavior across providers
- "Provider A is flaky" myths when it was actually retry policy differences
- Duplicated retry logic in every provider
- No unified way to track retry counts and rate limit sleep time
- Inconsistent logging (some providers logged retries, others didn't)

## Decision

We adopt a **Unified Retry Policy with Metrics** using the `retry_with_metrics` utility function.

1. **All API-based providers use `retry_with_metrics()`** for all API calls (transcription, summarization, speaker detection).
2. **Exponential backoff**: Starts at `initial_delay` (default 1.0s), doubles each retry, capped at `max_delay` (default 30.0s).
3. **Rate limit detection**: Automatically detects 429 errors and respects `retry_after` headers when available.
4. **Unified logging**: Logs compact retry line: `provider_retry: provider=openai attempt=2 sleep=4.0 reason=429`.
5. **Metrics tracking**: Tracks retry counts and rate limit sleep time via `ProviderCallMetrics.record_retry()`.

## Rationale

- **Consistent Behavior**: All providers retry the same way, preventing "provider A is flaky" myths
- **Centralized Logic**: Retry logic is maintained in one place, not duplicated across providers
- **Visibility**: Unified logging format makes retries visible and debuggable
- **Metrics Integration**: Retry counts and sleep time are tracked in metrics for analysis
- **Rate Limit Handling**: Automatic detection and respect for `retry_after` headers prevents unnecessary retries

## Alternatives Considered

1. **Provider-Specific Retry Logic**: Rejected as it leads to inconsistency and duplication.
2. **Library-Based Retry (tenacity, backoff)**: Rejected as it doesn't integrate with our metrics system and doesn't provide unified logging.
3. **No Retry Logic**: Rejected as transient failures are common with API providers and users expect automatic retries.

## Consequences

- **Positive**:
  - Consistent retry behavior across all providers
  - Unified logging format for easy debugging
  - Metrics tracking for retry analysis
  - Automatic rate limit handling
  - Centralized maintenance (one place to update retry logic)
- **Negative**:
  - All providers must use the same retry policy (can't customize per-provider)
  - Slight overhead for providers that don't need retries (but minimal)
- **Neutral**:
  - Requires one-time migration of all providers to use `retry_with_metrics`

## Implementation Notes

- **Module**: `src/podcast_scraper/utils/provider_metrics.py` - `retry_with_metrics` function
- **Pattern**: Decorator-like wrapper function (not a decorator to allow metrics parameter)
- **Default Parameters**:
  - `max_retries=3`
  - `initial_delay=1.0`
  - `max_delay=30.0`
  - `retryable_exceptions=(Exception,)` (can be customized per call)
- **Rate Limit Detection**: Checks for "429", "rate limit", "quota", "resource exhausted" in error messages
- **All API Providers**: OpenAI, Gemini, Anthropic, Mistral, Grok, DeepSeek, Ollama
- **Logging Format**: `provider_retry: provider={name} attempt={n} sleep={sec} reason={reason}`

## References

- [Issue #399: Provider-level concerns to harden](https://github.com/chipi/podcast_scraper/issues/399)
- `src/podcast_scraper/utils/provider_metrics.py` - Implementation
- [ADR-043: Unified Provider Metrics Contract](ADR-043-unified-provider-metrics-contract.md) - Related metrics contract
