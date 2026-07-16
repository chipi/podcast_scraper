# ADR-113: Two fuses and an LLM-aware resilience layer

**Status:** Accepted
**Date:** 2026-07-15
**Deciders:** Marko
**Related:** ADR-100 (per-provider LLM circuit breaker), ADR-112 (registry as source of truth), the
RSS `http_policy` resilience layer

## Context

Three failures on the same day exposed that the LLM call path had far weaker safety than the RSS/HTTP
path, and that its retry logic was too coarse:

1. **A silent money runaway.** `gpt-5.5`'s bundled evidence call returned empty content, grounding
   fell back to scoring quote-pairs one at a time, and nothing bounded the loop — it made **~3,500
   successful LLM calls on ONE episode over an hour**, on an expensive model, for a `0-grounded`
   result. The existing circuit breaker never fired because every call SUCCEEDED (200 OK); a
   *failure* breaker cannot catch a storm of successful-but-wasteful calls.

2. **A terminal condition treated as retryable.** The Anthropic account hit its spend cap mid-run and
   returned `400 "You have reached your specified API usage limits"`. The binary retryable-or-not
   classifier lumped every "quota"/"limit" string into *retryable*, so the run either looped on it or
   died with a confusing crash instead of stopping cleanly with "you are out of money."

3. **Hammering an overloaded cheap model.** `gemini-2.5-flash-lite` is cheap and popular, so it 503s
   under load far more than heavier models. Tight, uniform retries make its storms worse; it needs a
   *more conservative* posture than other models — and the retry knobs were hardcoded at ~75 call
   sites.

## Decision

### 1. Two fuses (hard stops), distinct from the failure breaker

* **Count fuse** (`utils/llm_call_fuse.py`) — a per-episode and per-run ceiling on the *number* of
  LLM calls, across every provider including ollama/vllm (local models tie up the GPU, not free).
  Past the budget it raises `LLMCallBudgetExceeded` and aborts the run. Enforced once in
  `retry_with_metrics` (the chokepoint every provider call flows through), so no call site can escape
  it. Counts calls, not dollars, because call count is 100% reliable while our per-provider cost
  logging is not (yet).

* **Terminal fuse** (`utils/llm_error_taxonomy.py::LLMTerminalError`) — an out-of-money / no-access
  condition is not retryable, so it is treated like the count fuse: a clean, loud hard stop
  ("<provider>: no budget/credit left on this key — stopping"), never a retry or a raw crash.

The failure circuit breaker (ADR-100) stays, and is now **on by default**. The three mechanisms are
orthogonal: the breaker waits out *transient failures*, the count fuse stops *successful runaways*,
the terminal fuse stops *account death*.

### 2. An LLM-aware error taxonomy

`classify_llm_error` sorts every API error into four classes, checking TERMINAL first so an
out-of-money error is never mistaken for a look-alike rate limit:

* **RETRYABLE_OVERLOAD** — 5xx, Anthropic 529 `overloaded_error`, Gemini 503 `UNAVAILABLE`, timeouts
  → back off, break, retry.
* **RETRYABLE_RATE_LIMIT** — 429 + Retry-After, transient `RESOURCE_EXHAUSTED` → honour Retry-After.
* **TERMINAL** — `insufficient_quota`, 402 "Insufficient Balance", spend/usage cap, 401/403 auth →
  terminal fuse.
* **NON_RETRYABLE** — 400/404/422 our-fault → re-raise.

Signals come from what the seven providers actually return (including the exact 400 that stopped our
Anthropic account).

### 3. Per-model resilience profiles

`utils/llm_resilience.py` keys retries / backoff / breaker-threshold / cooldown by `(provider,
model)`, with a `DEFAULT` and per-model overrides. `gemini-2.5-flash-lite` gets a **conservative**
profile (6 retries, 2–60 s backoff, breaker trips at 2 and cools 60 s) so we wait our turn instead of
hammering it. The profile is resolved once and attached to the metrics object;
`retry_with_metrics` reads retries/backoff from it, overriding the call-site defaults — **nothing
hardcoded, no call-site churn.**

### Testing

Unit tests for the taxonomy (the real provider strings) and the fuse, plus a **mock-server**
integration test (a threaded `http.server` returning 503/429/402/quota, driven through the real
OpenAI SDK) mirroring the RSS resilience tests — so a provider SDK changing how it surfaces a status
fails a test rather than a production run.

## Alternatives considered

* **Extend the failure breaker to catch runaways.** It is failure-triggered; the runaway was all
  successes. A count is the only thing that catches it. **Rejected.**
* **Trip the fuse on estimated cost, not calls.** The honest metric, but our cost logging is
  incomplete (OpenAI/grok log nothing, Mistral undercounts ~12×), so a cost fuse would have blind
  spots. Call count is reliable now; cost can layer on once instrumentation is fixed. **Deferred.**
* **Extract one shared resilience module for HTTP + LLM.** The right long-term shape, but a real
  refactor of the load-bearing RSS path. We enriched the existing LLM breaker instead and left
  unification as a follow-up. **Deferred.**

## Consequences

**Good.** A runaway bounds at the budget and shouts. An out-of-money key stops cleanly with a
human-readable reason. flash-lite gets the patience it needs. Resilience is per-model and
configurable, not hardcoded.

**Cost / layering.** The per-model profiles live in `utils/` rather than the registry to avoid a
`utils → providers.ml` import cycle; materialising them into the registry (ADR-112) is a clean
follow-up.

**KNOWN GAP — operator control & observability (follow-up).** When a fuse blows or a breaker opens
there is today no operator-facing way to *see* or *act* on it:

* **Breakers self-heal** on a cooldown (open → half-open probe → close), so no action is needed — but
  there is no control to force-reset one before its cooldown.
* **Fuses abort the run.** There is no mid-run resume: the procedure is fix-the-cause-and-rerun (the
  fuse is per-run, so a new run starts fresh). Terminal conditions (add credit / raise cap) likewise
  just work on the next run.
* **No status view and no UI** — a blown fuse or open breaker is visible only in the logs.

The follow-up is a small operations surface: a status view of open breakers / blown fuses with
reasons, an operator "reset / acknowledge" control, and (optionally) resumable runs so a fixed
terminal condition can continue rather than restart. State is in-process module globals today; a
UI-driven reset would need it exposed and mutable through a control plane.
