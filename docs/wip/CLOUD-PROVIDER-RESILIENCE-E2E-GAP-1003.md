# Cloud-LLM-provider resilience: E2E coverage audit (#1003 + 2026-06-15 close-out)

**Status:** Closed 2026-06-15. All gaps identified during the original audit
have been wired and proven by E2E tests. This doc is preserved as the
durable record of what was open at the audit and what closed it.

## TL;DR (after close-out)

- Self-hosted resilience: covered in `tests/e2e/test_tailnet_dgx_e2e.py`
  (5xx + watchdog-hang failover for whisper and diarize).
- Cloud LLM **guardrail** E2E: `tests/e2e/test_cloud_guardrails_e2e.py`
  (per-provider, all 3 failure modes).
- Cloud LLM **resilience** E2E:
  `tests/e2e/test_cloud_resilience_e2e.py` — per provider:
  permanent 5xx, transient 5xx retry-recovery, request timeout
  (now 4/4 cloud providers since Gemini SDK got its
  `summarization_timeout` plumbing).
- Cloud **FallbackAware** under real `GuardrailViolation`:
  `tests/e2e/test_cloud_guardrails_fallback_e2e.py` — primary trips
  guardrail, configured fallback's response reaches the caller.
- Ollama (the self-hosted-but-OpenAI-compatible third axis):
  `tests/e2e/test_ollama_guardrails_and_resilience_e2e.py` covers
  guardrail empty / thinking-prose, cleaning catch-and-degrade,
  permanent 5xx, transient 5xx, request timeout.
- Cost-attribution `triggered_guardrail=True` is now actually emitted
  at every cloud + Ollama summarize site (the field has existed since
  #1003 but no call site was passing `True` until this close-out).
- `retry_with_metrics` no longer retries on `GuardrailViolation`
  (correct semantics: a 200-OK-bad-content response will not become
  good content by retrying).
- Anthropic `max_tokens` / Gemini `MAX_TOKENS` finish-reason values
  are normalised to `"length"` at the per-SDK boundary so the
  guardrail trips on them (was a documented limitation in the
  original ADR-100 post-impl section).

## Matrix — coverage *after* close-out

|Provider             |Guardrail (200, bad shape)|Permanent 5xx|Transient 5xx → recover|Request timeout |Circuit breaker|Cleaning catch-and-degrade|
|---------------------|--------------------------|-------------|-----------------------|----------------|---------------|--------------------------|
|tailnet_dgx_whisper  |✓ (#999)                  |✓ (existing) |—                      |✓ (existing)    |✓ (existing)   |n/a                       |
|tailnet_dgx_diarize  |✓ (#999)                  |✓ (existing) |—                      |✓ (existing)    |✓ (existing)   |n/a                       |
|openai (chat)        |✓                         |✓            |✓                      |✓               |n/a            |✓                         |
|anthropic            |✓                         |✓            |✓                      |✓               |n/a            |✓                         |
|gemini               |✓                         |✓            |✓                      |✓ (plumbed)     |n/a            |✓                         |
|deepseek             |✓                         |✓            |✓                      |✓               |n/a            |✓                         |
|ollama               |✓                         |✓            |✓                      |✓               |n/a            |✓                         |

**Read this as:** "Guardrail" verifies the response-shape check fires
through the real SDK round-trip. "Permanent 5xx" verifies the broad
`except Exception` wraps transport errors into `ProviderRuntimeError`.
"Transient 5xx" verifies the SDK's retry recovers. "Request timeout"
verifies the `summarization_timeout` config plumbs through to the SDK's
httpx layer. "Cleaning catch-and-degrade" verifies the per-stage policy:
a guardrail trip in `clean_transcript` returns the original text rather
than failing the run.

## Why no cloud circuit breaker?

Cloud LLM providers' resilience is currently a single retry loop
(`retry_with_metrics` in `utils/provider_metrics.py`) on top of the SDK.
There is no per-cloud-provider `CircuitBreaker` instance like the
self-hosted whisper/diarize paths have. Whether that's a gap or a deliberate
choice — leaving the cloud SDK's own backoff/rate-limit handling to do
the work — is a separate design question. Out of scope for the
2026-06-15 close-out. The current behavior is: cloud SDKs handle their
own backoff; `retry_with_metrics` retries on retryable exception types;
no per-provider breaker state.

## What was open at the original audit (and how it closed)

Listed for historical reference. Each item below was a real gap; each
closed by code in the same batch.

1. **Transient-5xx retry-recovery, all cloud providers.** Closed by
   `TestTransient5xxRetryRecovery` in `test_cloud_resilience_e2e.py`.
2. **Watchdog/timeout, all cloud providers.** Closed by
   `TestRequestTimeoutE2E` in `test_cloud_resilience_e2e.py`; Gemini
   gained the underlying timeout plumbing in `gemini_provider.py`
   (`HttpOptions.timeout` from `get_http_timeout(cfg)`).
3. **FallbackAware routing under GuardrailViolation.** Closed by
   `test_cloud_guardrails_fallback_e2e.py`.
4. **Cost-event `triggered_guardrail=True` integration.** Closed at the
   summarize sites of all 4 cloud providers + Ollama; the cost event is
   emitted in BOTH the happy path (`triggered_guardrail=False`) and the
   `except GuardrailViolation` block (`triggered_guardrail=True`) before
   re-raising. Token data is captured up-front so the cost shape is
   identical between branches.
5. **Ollama resilience coverage.** Closed by
   `test_ollama_guardrails_and_resilience_e2e.py`.

Adjacent items that surfaced during the close-out and were fixed in the
same batch (not in the original audit list):

- Ollama `clean_transcript` had no `except GuardrailViolation` clause
  (the cloud-provider cleaning paths did); the wrap-into-PRE trap
  applied. Fixed; new test `test_cleaning_thinking_prose_degrades_gracefully`.
- `retry_with_metrics` would retry on `GuardrailViolation` because the
  default `retryable_exceptions=(Exception,)` is permissive and the
  bundled/GI/KG sites wrap the closure in `retry_with_metrics`. Fixed
  by adding a `GuardrailViolation` early-return to
  `utils/retryable_errors.py::is_retryable_error`.

## Second-round close-out (also 2026-06-15) — both scope-outs closed

Operator pushback on the two scope-outs above. Both closed in the same
batch:

### Cost-emit-with-flag at ALL guardrail-bearing call sites

Wired at every site (5 providers × 3-4 sites each) that calls
``check_chat_response``: summarize, summarize_bundled,
generate_insights, clean_transcript. Where the site was missing base
``llm_cost`` emission entirely (GI and cleaning across all providers
were going only to ``pipeline_metrics`` internals), base emission was
added with the right ``stage`` label (``gi`` / ``cleaning``). The
``triggered_guardrail`` flag now fires in BOTH happy and violation
branches.

Surfaced + fixed during the close-out:

- GI on every provider was silently catching ``GuardrailViolation``
  via ``except Exception: return []`` — ADR-100-incompatible (GI is
  fail-up). Now propagates to FallbackAware.
- Cleaning across cloud + Ollama had no llm_cost emission at all
  (only pipeline_metrics internal). Now emits with the right stage
  label.

### Per-cloud-provider CircuitBreaker

The substrate has existed since #697 (``LLMCircuitBreakerConfig`` +
integration in ``retry_with_metrics``). The wiring at the provider
level didn't. Closed by:

- Adding ``ProviderCallMetrics.set_breaker_config_from_cfg(cfg)`` —
  builds the breaker config from cfg when
  ``llm_circuit_breaker_enabled=True`` and attaches it to the metrics
  object.
- ``retry_with_metrics`` reads the breaker config off the metrics
  object as a fallback (preserves the existing explicit kwarg path).
- Each cloud provider's ``ProviderCallMetrics`` construction now calls
  ``set_breaker_config_from_cfg(self.cfg)`` immediately after
  ``set_provider_name``. Auto-wires every retry_with_metrics call
  through the provider — no per-site refactor of 84 call sites.

New E2E: ``test_cloud_circuit_breaker_e2e.py`` proves breaker trips
under a real 503 burst (against the mock server) when
``llm_circuit_breaker_enabled=True``, and stays unwired when the
config is default-off.
