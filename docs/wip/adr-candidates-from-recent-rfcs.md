# ADR Candidates from Recent RFCs

This document identifies architectural decisions from recently completed or added RFCs that should be extracted as ADRs.

## Summary

| ADR Candidate | Source RFC | Status | Priority | Description |
| --- | --- | --- | --- | --- |
| **Unified Provider Metrics Contract** | Recent implementation (provider metrics) | High | **1** | Standardized `ProviderCallMetrics` pattern for all providers |
| **Unified Retry Policy with Metrics** | Recent implementation (retry_with_metrics) | High | **2** | Centralized retry logic with exponential backoff and metrics tracking |
| **Composable E2E Mock Response Strategy** | RFC-054 | Medium | **3** | Separation of functional responses from non-functional behavior in tests |
| **Adaptive Summarization Routing** | RFC-053 | Medium | **4** | Rule-based routing with episode profiling for summarization strategies |
| **Centralized Model Registry** | RFC-044 | Medium | **5** | Single source of truth for model architecture limits |
| **Per-Episode Metrics Standardization** | Recent implementation | Low | **6** | Consistent metrics schema across providers (null for unavailable metrics) |

---

## 1. Unified Provider Metrics Contract

**Source**: Recent implementation (provider metrics tracking)

**Decision**: All providers must accept and populate a `ProviderCallMetrics` object, even if they don't expose all metrics (e.g., local ML providers don't have tokens, but still use the same interface).

**Rationale**:
- Enables consistent metrics collection across all providers
- Allows pipeline to aggregate metrics without provider-specific branching
- Standardizes logging format for per-episode metrics
- Makes it easy to add new metrics in the future

**Key Points**:
- `ProviderCallMetrics` is a required parameter (no backward compatibility)
- Providers set `null` for unavailable metrics (e.g., `prompt_tokens=None` for local ML)
- All providers call `call_metrics.finalize()` before returning
- Pipeline logs standardized format: `audio_sec`, `transcribe_sec`, `summary_sec`, `retries`, `rate_limit_sleep_sec`, `prompt_tokens`, `completion_tokens`, `estimated_cost`

**Related Code**:
- `src/podcast_scraper/utils/provider_metrics.py` - `ProviderCallMetrics` class
- All provider `transcribe_with_segments` and `summarize` methods

**Status**: ✅ Implemented

---

## 2. Unified Retry Policy with Metrics

**Source**: Recent implementation (`retry_with_metrics` utility)

**Decision**: All API-based providers use a centralized `retry_with_metrics` function for retry logic with exponential backoff, rate limit detection, and metrics tracking.

**Rationale**:
- Prevents "provider A is flaky" myths when it's actually retry policy differences
- Ensures consistent retry behavior across all providers
- Enables unified retry logging: `provider_retry: provider=openai attempt=2 sleep=4.0 reason=429`
- Tracks retry counts and rate limit sleep time in metrics

**Key Points**:
- Exponential backoff: starts at `initial_delay`, doubles each retry, capped at `max_delay`
- Rate limit detection: checks for "429", "rate limit", "quota", "resource exhausted"
- Respects `retry_after` header if available
- Logs compact retry line on each retry attempt
- Tracks metrics via `ProviderCallMetrics.record_retry()`

**Related Code**:
- `src/podcast_scraper/utils/provider_metrics.py` - `retry_with_metrics` function
- All API-based providers (OpenAI, Gemini, Anthropic, Mistral, Grok, DeepSeek, Ollama)

**Status**: ✅ Implemented

**Related Issues**: #399 (Provider-level concerns to harden)

---

## 3. Composable E2E Mock Response Strategy

**Source**: RFC-054 (Flexible E2E Mock Response Strategy)

**Decision**: E2E test mocks separate **functional responses** (API response structure) from **non-functional behavior** (retries, timeouts, rate limits), allowing tests to compose different scenarios without duplication.

**Rationale**:
- Enables testing error scenarios (429, 5xx, timeouts) without duplicating mock setup
- Allows tests to mix "normal response" with "retry behavior" or "error response" with "timeout behavior"
- Centralizes non-functional logic (retry behavior, timeouts) in one place
- Most tests get realistic default responses automatically (no configuration needed)

**Key Points**:
- **Response Profiles**: Define functional responses (normal, error, edge case)
- **Non-Functional Behavior**: Retry behavior, timeouts, rate limits configured separately
- **Response Router**: Composes functional responses with non-functional behavior
- **Default Normal**: Most tests use realistic but simple responses automatically
- **Composable**: Tests can combine different functional responses with different non-functional behaviors

**Architecture**:
```
Test Request → Mock Response Router → Response Profile (functional) + Non-Functional Behavior → Response
```

**Status**: 🟡 Draft RFC (not yet implemented)

**Related Issues**: #135, #399, #401

---

## 4. Adaptive Summarization Routing

**Source**: RFC-053 (Adaptive Summarization Routing Based on Episode Profiling)

**Decision**: Summarization pipeline uses **rule-based routing** with **episode profiling** to select optimal summarization strategies based on episode characteristics (duration, structure, content type).

**Rationale**:
- Different episode types (short vs long, dialogue vs monologue, technical vs narrative) benefit from different summarization strategies
- One-size-fits-all approach misses optimization opportunities
- Rule-based routing is deterministic and debuggable (logged for each episode)

**Key Points**:
- **Episode Profiling**: Analyzes duration, speaker count, turn-taking rate, entity density, numeric density, topic drift
- **Routing Rules**: Deterministic rules (e.g., `duration_minutes <= 15 and speaker_count <= 1` → `SHORT_MONOLOGUE`)
- **Strategy Selection**: Routes to appropriate strategy (short monologue, short dialogue, technical, long monologue, long dialogue, standard)
- **Model Roles**: Stable roles (Extractor, Summarizer, Reducer, Finalizer) compatible with RFC-042
- **Extraction-First**: All strategies produce structured intermediate outputs before reduction

**Routing Thresholds**:
- Token count < 2000 → Single-pass strategy
- Speaker turn rate > 2.0 turns/min → Dialogue strategy
- Entity density > 10.0 per 1000 tokens → Technical strategy
- Duration > 60 min + multiple speakers → Panel strategy

**Status**: 🟡 Draft RFC (not yet implemented)

**Related PRDs**: PRD-005 (Episode Summarization)

---

## 5. Centralized Model Registry

**Source**: RFC-044 (Model Registry for Architecture Limits)

**Decision**: Model architecture limits (e.g., `1024` for BART, `16384` for LED) are stored in a **centralized Model Registry** instead of hardcoded throughout the codebase.

**Rationale**:
- Eliminates maintenance burden: adding new models requires updating one place, not multiple files
- Prevents inconsistency: limits can't drift out of sync
- Provides single source of truth for model capabilities
- Enables extensibility: new models can be registered without code changes
- Supports dynamic detection with intelligent fallbacks

**Key Points**:
- **ModelCapabilities Dataclass**: Structured, type-safe model capability information
- **Registry Lookup**: O(1) lookup by model ID/alias
- **Pattern-Based Fallbacks**: Intelligent guessing for unknown models (e.g., "bart-*" → BART limits)
- **Safe Defaults**: Conservative defaults for unknown models
- **Extensibility**: Runtime registration for custom models
- **Model-Agnostic**: Handles both test and production models identically

**Status**: 🟡 Draft RFC (not yet implemented)

**Related RFCs**: RFC-029 (Provider Refactoring Consolidation)

---

## 6. Per-Episode Metrics Standardization

**Source**: Recent implementation (per-episode metrics logging)

**Decision**: Pipeline logs a **standardized metrics record** per episode with consistent keys, even when some providers don't expose certain metrics (use `null` instead of omitting keys).

**Rationale**:
- Enables provider comparison: can compare costs, retries, tokens across providers
- Simplifies parsing: no branching logic needed for different provider types
- Consistent logging format: `episode_metrics: audio_sec=X, transcribe_sec=Y, summary_sec=Z, retries=N, rate_limit_sleep_sec=W, prompt_tokens=A, completion_tokens=B, estimated_cost=C`

**Key Points**:
- **Standardized Keys**: `audio_sec`, `transcribe_sec`, `summary_sec`, `retries`, `rate_limit_sleep_sec`, `prompt_tokens`, `completion_tokens`, `estimated_cost`
- **Null for Unavailable**: Providers that don't expose tokens log `prompt_tokens=null` (not omitted)
- **Consistent Format**: All episodes log same keys in same order
- **Provider Comparison**: Enables easy comparison of providers (cost, performance, quality)

**Status**: ✅ Implemented

**Related Code**:
- `src/podcast_scraper/workflow/orchestration.py` - `_log_episode_metrics()`
- `src/podcast_scraper/workflow/metrics.py` - `EpisodeMetrics` dataclass

---

## Recommendations

### High Priority (Implement as ADRs)

1. **ADR-043: Unified Provider Metrics Contract** - Documents the `ProviderCallMetrics` pattern
2. **ADR-044: Unified Retry Policy with Metrics** - Documents the `retry_with_metrics` utility

### Medium Priority (Wait for RFC Implementation)

3. **ADR-045: Composable E2E Mock Response Strategy** - After RFC-054 is implemented
4. **ADR-046: Adaptive Summarization Routing** - After RFC-053 is implemented
5. **ADR-047: Centralized Model Registry** - After RFC-044 is implemented

### Low Priority (May Not Need ADR)

6. **Per-Episode Metrics Standardization** - This is more of an implementation detail than an architectural decision. Could be documented in ADR-043 as part of the metrics contract.

---

## Next Steps

1. **Create ADR-043** (Unified Provider Metrics Contract) - High priority, already implemented
2. **Create ADR-044** (Unified Retry Policy with Metrics) - High priority, already implemented
3. **Review RFC-054, RFC-053, RFC-044** - Create ADRs after implementation
4. **Update ADR index** - Add new ADRs to `docs/adr/index.md`
5. **Link ADRs to RFCs** - Update RFCs to reference their corresponding ADRs
