# Cloud-LLM-provider resilience: E2E coverage audit (#1003 follow-up)

**Status:** Snapshot from 2026-06-15 audit during #1003 (cloud guardrails).
**Scope:** Cloud LLM providers' resilience paths exercised end-to-end via the
e2e mock server.

## TL;DR

- Self-hosted resilience is covered: `tests/e2e/test_tailnet_dgx_e2e.py` exercises
  5xx + watchdog-hang failover for the whisper and diarize paths.
- Cloud LLM resilience was a hole: the per-provider e2e suites
  (`test_{openai,anthropic,gemini,deepseek,ollama}_provider_e2e.py`) did not use
  `set_error_behavior` / `set_transient_error` at all. They covered happy path
  only.
- Baseline resilience coverage landed in this PR
  (`tests/e2e/test_cloud_resilience_e2e.py`): one test per cloud provider
  verifying permanent 5xx surfaces as `ProviderRuntimeError` through the real SDK.
- Gaps remaining (next-batch follow-up, **not** for this PR): transient-503
  retry-recovery, watchdog timeout, FallbackAware routing under real
  GuardrailViolation, cost-event `triggered_guardrail=True` integration.

## Matrix — coverage *after* this PR

|Provider             |Guardrail (200, bad shape)|Permanent 5xx|Transient 5xx → recover|Watchdog/timeout|Circuit breaker|
|---------------------|--------------------------|-------------|-----------------------|----------------|---------------|
|tailnet_dgx_whisper  |✓ (#999)                  |✓ (existing) |—                      |✓ (existing)    |✓ (existing)   |
|tailnet_dgx_diarize  |✓ (#999)                  |✓ (existing) |—                      |✓ (existing)    |✓ (existing)   |
|openai (chat)        |**✓ (this PR)**           |**✓ (this PR)**|gap                  |gap             |n/a — no breaker on cloud paths|
|anthropic            |**✓ (this PR)**           |**✓ (this PR)**|gap                  |gap             |n/a            |
|gemini               |**✓ (this PR)**           |**✓ (this PR)**|gap                  |gap             |n/a            |
|deepseek             |**✓ (this PR)**           |**✓ (this PR)**|gap                  |gap             |n/a            |
|ollama               |unit-only (#999)          |gap          |gap                    |gap             |n/a            |

**Read this as:** the column "Guardrail" verifies the response-shape check
fires through the real SDK round-trip. "Permanent 5xx" verifies the broad
`except Exception` wrap turns transport errors into `ProviderRuntimeError`
for the FallbackAware layer to catch. Remaining gaps are listed below.

## Why no cloud circuit breaker?

Cloud LLM providers' resilience is currently a single retry loop
(`retry_with_metrics` in `utils/provider_metrics.py`) on top of the SDK.
There is no per-cloud-provider `CircuitBreaker` instance like the
self-hosted whisper/diarize paths have. Whether that's a gap or a deliberate
choice — leaving the cloud SDK's own backoff/rate-limit handling to do the
work — is a separate design question. Out of scope here.

## Gaps to close in a follow-up PR

Each line below is a TaskCreate candidate, not a GH issue (per AGENTS.md
rule 15). Operator decides if/when to land.

1. **Transient-5xx retry-recovery, all 4 cloud providers.** Inject N=2 503s
   via `set_transient_error`, assert summarize succeeds on attempt 3
   (the retry_with_metrics default ladder). Confirms the underlying SDK
   surfaces the right exception classes for the retry decorator.

2. **Watchdog/timeout, all 4 cloud providers.** Inject a long delay via
   `set_error_behavior(..., delay=10.0)` and a tight client timeout in
   config. Assert the provider bails inside the watchdog grace. Today's
   timeout-from-config behavior on cloud SDKs is unverified at the
   end-to-end level.

3. **FallbackAware routing under GuardrailViolation.** This PR proves
   `GuardrailViolation` propagates out of each cloud provider's
   `summarize()` (was previously wrapped). The next layer — the
   `FallbackAwareSummarizationProvider` — needs an E2E test that confirms
   the violation is caught and the configured fallback is invoked.
   That requires a multi-provider config (`degradation_policy`), so it's
   a separate test file.

4. **Cost-event `triggered_guardrail=True` integration.** The
   `emit_llm_cost_event` field exists (unit-tested in
   `test_cloud_guardrails_wiring.py`). The provider sites that catch
   `GuardrailViolation` should emit a cost event with the flag set
   so cost-rollup can pivot on it. Today, the providers re-raise the
   violation *before* the cost emit point. Wiring this requires a
   small refactor in each provider's call site (capture cost-data
   prior to the guardrail check, emit-with-flag in the
   `except _guardrails.GuardrailViolation` block, then re-raise).

5. **Ollama resilience coverage.** Same matrix entries as the cloud
   providers, plus the `/api/generate` path. Ollama has unit-level
   guardrail coverage but no E2E.

## Why this isn't a critical-path blocker

The production fallback path is the
`FallbackAwareSummarizationProvider`. As long as each cloud provider raises
*something the fallback layer recognizes* (`GuardrailViolation` for bad shape,
`ProviderRuntimeError` for transport failure), routing works. This PR closes
the type-propagation gap for `GuardrailViolation` on the summarize path.

Numbers 1-4 above harden the verification layer but don't change runtime
behavior — they catch regressions earlier. They should land in the next
guardrails-batch follow-up.
