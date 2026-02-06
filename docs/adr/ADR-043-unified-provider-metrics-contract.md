# ADR-043: Unified Provider Metrics Contract

- **Status**: Accepted
- **Date**: 2026-02-05
- **Authors**: Podcast Scraper Team
- **Related Issues**: #399 (Provider-level concerns to harden)

## Context & Problem Statement

Providers (OpenAI, Gemini, Anthropic, local ML, etc.) have different capabilities and expose different metrics. Some providers track token usage and costs, while others (local ML) don't. The pipeline needed a way to collect consistent metrics across all providers without provider-specific branching logic.

Without a unified contract:

- Pipeline code had to check provider type before accessing metrics
- Logging format varied by provider
- Adding new metrics required updating multiple provider-specific code paths
- Provider comparison was difficult due to inconsistent metric availability

## Decision

We adopt a **Unified Provider Metrics Contract** using the `ProviderCallMetrics` dataclass.

1. **All providers must accept and populate a `ProviderCallMetrics` object** as a required parameter in `transcribe_with_segments()` and `summarize()` methods.
2. **Providers set `null` for unavailable metrics** (e.g., local ML providers set `prompt_tokens=None`).
3. **All providers call `call_metrics.finalize()`** before returning results.
4. **Pipeline logs standardized format** with consistent keys for all episodes, using `null` when metrics are unavailable.

## Rationale

- **Consistent Metrics Collection**: Pipeline can aggregate metrics without provider-specific branching
- **Standardized Logging**: All episodes log the same keys in the same order, enabling easy parsing and comparison
- **Provider Comparison**: Enables direct comparison of costs, retries, tokens across providers
- **Future-Proof**: Easy to add new metrics without changing provider interfaces
- **No Branching Logic**: Pipeline code doesn't need to check provider type before accessing metrics

## Alternatives Considered

1. **Provider-Specific Metrics Objects**: Rejected as it requires branching logic in pipeline code and makes provider comparison difficult.
2. **Optional Metrics Parameter**: Rejected as it doesn't enforce consistency and allows providers to skip metrics tracking.
3. **Separate Metrics Collection**: Rejected as it duplicates logic and makes it harder to track metrics at the call site.

## Consequences

- **Positive**:
  - Consistent metrics across all providers
  - Simplified pipeline code (no provider-specific branching)
  - Easy provider comparison (cost, performance, quality)
  - Standardized logging format for parsing and analysis
- **Negative**:
  - Providers must implement the contract even if they don't expose all metrics
  - Slight overhead for providers that don't track metrics (but minimal)
- **Neutral**:
  - Requires one-time migration of all providers to use the contract

## Implementation Notes

- **Module**: `src/podcast_scraper/utils/provider_metrics.py` - `ProviderCallMetrics` class
- **Pattern**: Required parameter pattern (no backward compatibility)
- **Standardized Logging Format**:

  ```text
  episode_metrics: audio_sec=X, transcribe_sec=Y, summary_sec=Z,
  retries=N, rate_limit_sleep_sec=W, prompt_tokens=A,
  completion_tokens=B, estimated_cost=C
  ```

- **All Providers**: OpenAI, Gemini, Anthropic, Mistral, Grok, DeepSeek, Ollama, MLProvider
- **Pipeline Integration**: `src/podcast_scraper/workflow/orchestration.py` - `_log_episode_metrics()`

## References

- [Issue #399: Provider-level concerns to harden](https://github.com/chipi/podcast_scraper/issues/399)
- `src/podcast_scraper/utils/provider_metrics.py` - Implementation
