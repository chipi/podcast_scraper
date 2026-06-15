# ADR-100: Response-shape guardrails for cloud LLM providers

- **Status**: Accepted
- **Date**: 2026-06-15
- **Authors**: Podcast Scraper Team
- **Related ADRs**: [ADR-099](ADR-099-response-shape-guardrails-for-self-deployed-services.md) (the self-hosted sibling; this ADR extends the same pattern to cloud APIs)
- **Related issues**: [#1003](https://github.com/chipi/podcast_scraper/issues/1003) (this implementation), [#999](https://github.com/chipi/podcast_scraper/issues/999) (the self-hosted precedent), [#1002](https://github.com/chipi/podcast_scraper/issues/1002) (threshold fine-tuning follow-up, shared with ADR-099)

## Context

ADR-099 shipped response-shape guardrails for self-hosted inference services
(Ollama, vLLM, whisper-openai, pyannote), closing the gap that
``providers/resilience`` (connection-level) cannot detect: a successful HTTP
response whose content is semantically corrupted.

The operator observed during the ADR-099 implementation:

> "as we work on this I am wondering if we should also implement this type of
> guardrails on my API LLM providers. you never know when one can hallucinate?"

The answer is yes. The cloud LLM providers we use (OpenAI, Anthropic, Gemini,
DeepSeek, Mistral, Grok, Groq) have the same structurally-invalid-response
failure modes as the self-hosted ones — we have direct evidence in our own
eval data:

- **Gemini-2.5-flash** with thinking mode: wildly inconsistent short outputs
  during `EVAL_PROMPT_LONG_V2_CROSS_PROVIDER_2026_06_14` — had to switch to
  `gemini-2.5-flash-lite` (production default) for stable measurements.
- **OpenAI / gpt-4o**: hits `finish_reason="length"` when prompts spill the
  budget. Surfaces as malformed JSON / mid-truncation at the next stage.
- **DeepSeek**: 3-2 split on `long_v2` in the cross-provider sweep with
  inconsistent shape — possible underlying response-shape variability.

The guardrail pattern from ADR-099 is provider-agnostic. The
`check_chat_response(content, *, service, finish_reason=None, expect_json=False)`
helper introduced in the precursor refactor was deliberately written for any
chat-completion-shaped API. The failure modes (empty, thinking-prose,
finish-reason-length, malformed JSON) are the same across self-hosted Ollama
and cloud OpenAI. **Only the SDK extraction differs per provider.**

This ADR captures the decisions for extending the pattern to cloud.

## Decision

**Wire the same `providers.guardrails.check_chat_response()` helper at the
content-producing call sites of the four mainstream cloud LLM providers** —
OpenAI, Anthropic, Gemini, DeepSeek — using the failure-handling pattern
established in ADR-099 (raise `GuardrailViolation`, caller's existing
exception path triggers fallback via the configured
`degradation_policy.fallback_provider_on_failure`).

### 1. One generic helper, kwarg-based service identifier

`check_chat_response(content, *, service="openai")` works for every
chat-completion provider. The `service` kwarg becomes the Prometheus label
and the `GuardrailViolation.service` attribute. **Pick a fixed short string
per provider** — `"openai"` / `"anthropic"` / `"gemini"` / `"deepseek"` /
`"mistral"` / `"grok"` / `"groq"` — and never embed deployment details
(no `"openai-via-azure"`, no `"gemini-prod"`).

No per-provider helper functions get added. Same code, different SDK
extraction at the call site.

### 2. SDK extraction stays at the call site

Each cloud provider's SDK has its own response shape:

| Provider | Content path | Finish reason path |
| --- | --- | --- |
| OpenAI | `response.choices[0].message.content` | `response.choices[0].finish_reason` |
| Anthropic | `response.content[0].text` | `response.stop_reason` |
| Gemini | `response.text` | `response.candidates[0].finish_reason` (when present) |
| DeepSeek | `response.choices[0].message.content` | `response.choices[0].finish_reason` |
| Mistral / Grok / Groq | OpenAI-compatible | same |

The caller extracts `content` and `finish_reason` first, then passes them to
the helper. Keeps the helper free of SDK dependencies (and free of the
breakage risk that SDK updates introduce).

### 3. Failure handling per stage, not per provider

ADR-099 routes self-hosted violations to the existing cloud fallback path
(transparent, single policy). Cloud calls differ — falling up to "the next
cloud provider" costs the operator money. The policy is per-stage:

| Pipeline stage | Default behaviour on guardrail violation |
| --- | --- |
| Summary (`summarize`) | Fail-up to `FallbackAwareSummarizationProvider`'s configured `degradation_policy.fallback_provider_on_failure`. Same path already triggered by connection-level failures. |
| Cleaning (`clean_transcript`) | **Graceful degradation** — return the original transcript. Mirrors the existing "if not cleaned: return text" pattern. Empty content is fine; thinking-prose markers in cleaning output are caught and fall to the original. |
| GI insights | Fail-up. The stage either produces structured insight data or skips with a logged degradation. |
| KG extraction | Fail-up. Same rationale as GI. |
| Speaker detection | Fail-up to spaCy local fallback if configured; otherwise skip the LLM-detected speaker enrichment and rely on transcript-side defaults. |

Per-provider overrides are explicitly **out of scope**. The failure
behaviour is owned by the stage's contract (does the downstream need
non-empty output?), not by who answered the call.

### 4. Cost attribution via `llm_cost` log event extension

Cloud-side guardrail violations have a property self-hosted ones don't:
**we paid for the response we then rejected**. Tracking that is operationally
material because:

- A cloud provider trending toward more guardrail violations is silently
  burning the operator's API budget.
- A fallback chain (Gemini fails → OpenAI takes over) means a single episode
  can incur charges from multiple providers; the operator wants to see that.

The existing `llm_cost` JSON log event already carries per-call cost data.
**Extend it with a `triggered_guardrail` boolean field** (default `false`;
set `true` only on the call that just raised `GuardrailViolation`). The
existing `corpus-cost` rollup picks this up automatically; new dashboards
can split paid-but-rejected spend without any new infrastructure.

No new Prometheus counter for cost-tracking. The cost data is already in
the log pipeline; pivoting on the new field is a downstream concern.

### 5. Telemetry: extend the existing counter, careful with cardinality

The Prometheus counter introduced in ADR-099 is
`inference_guardrail_violations_total{service, reason}`. Cloud violations
land on the same counter with `service` set per-provider. Cardinality math:

| Source | service values | reason values per service | Series added |
| --- | --- | --- | --- |
| Self-hosted (ADR-099) | 4 (whisper, ollama, vllm, pyannote) | ~2-3 each | ~10 |
| Cloud (this ADR) | 4 mainstream (openai, anthropic, gemini, deepseek) | ~4 each (empty, thinking-prose, finish-length, bad-json) | ~16 |
| **Total active series** | **~26** | | well under the 175-series DGX-side budget; <1% of Grafana free-tier 10k cap |

The Mistral / Grok / Groq providers get the same helper when they next get
PR work; estimated `+3 × 4 = 12` series later, still well within budget.

**Cardinality discipline enforced by code review** (same as ADR-099): only
`service` and `reason` are labels, both drawn from a fixed per-provider
enum. **Never** add high-cardinality fields (request ID, model name with
revision, response content) as Prometheus labels — those go in the
structured log body / Sentry context.

### 6. Refusal detection deferred to Phase 2

A potential 5th failure mode — cloud LLMs returning refusal text
("I can't help with that", "I'm not able to...", "As an AI...") — was
considered for Layer 1. **Deferred.**

- The false-positive risk is meaningful. A summary of a podcast about AI
  ethics legitimately includes phrases like "as an AI" without the response
  being a refusal.
- We have no measured cases of refusals tripping downstream issues today.
- Pattern matching the refusal openers requires per-provider tuning that
  ADR-099's "fixed thresholds, no configurability" rule discourages.

Filed as Phase 2 scope on #1003. If an operator-observed case surfaces
where a cloud refusal silently propagated downstream, that's the trigger
to revisit.

## Why this layer, not a different one

Considered alternatives:

| Alternative | Why rejected |
| --- | --- |
| Per-provider check helpers (`check_openai_response`, etc.) | Duplicate code; the failure modes are identical across all chat-completion APIs. The kwarg-service pattern keeps the helper count at 1. |
| New separate `cloud_guardrails/` module | Splits a tiny amount of logic across two modules. The check is provider-agnostic — `providers.guardrails.chat` is its right home. |
| Configurable thresholds per provider | Same fail-closed argument as ADR-099. Threshold fine-tuning lives in #1002 (evidence-based, in PR, with rate-data screenshots). |
| Provider-side proxies that rewrite bad responses | Out of scope; we don't control cloud providers and shouldn't pretend to. |

## Consequences

### Positive

- The guardrail layer covers every LLM call (self-hosted and cloud) under
  one mental model + one helper.
- Adding the next cloud provider becomes mechanical: 4 call-site changes,
  pick a service string, done. No new helper code.
- The same Prometheus counter, the same log shape, the same Sentry
  exception class. Downstream dashboards don't fork by provider.
- The cost-attribution extension makes cloud-side guardrail spend visible
  with no new infrastructure.

### Negative

- Cloud-fallback-on-violation costs money. A provider trending toward
  frequent violations doubles its bill (paid for the bad response, then
  paid the fallback provider for the retry). Mitigated by the cost log
  field — operator can spot the trend and switch defaults.
- Adding `provider` cardinality to the counter takes us from ~10 to ~26
  series. Still tiny but worth pinning at this size; #1002 owns the
  per-(service, reason) firing-rate evaluation.

### Risk

- **Risk:** the cleaning-stage graceful-degradation pattern misses a
  case where cloud returns thinking-prose-content as the "cleaned"
  result. Same risk we faced with Ollama in ADR-099, handled the same
  way: the guardrail call happens AFTER the empty-handling early return,
  catching the thinking-prose case without false-firing on legitimate
  empty.
- **Risk:** Anthropic's `content[0].text` shape may not exist if the
  response has no content blocks (rare, but possible). The caller
  defends with `getattr(...)` and treats absence as empty content —
  the helper handles `None` correctly via the empty-content path.

## Non-goals

- **No refusal detection in this ADR.** Phase 2 of #1003.
- **No reasoning-token tracking** (some thinking models return reasoning
  in a separate field; that's a different observability question, not a
  guardrail concern).
- **No provider-specific threshold knobs.** Per-stage failure-handling is
  the configurability axis; thresholds remain hardcoded per ADR-099.
- **No SDK-version compatibility shims.** If a cloud provider's SDK
  changes its response shape, the call-site extraction breaks loudly
  rather than silently. That's preferred — silent guardrail bypass is
  the failure mode this whole pattern exists to prevent.

## Initial thresholds (locked at ship time)

Same as ADR-099 — the helper is the same. Per-mode thresholds:

| Failure mode | Threshold | Notes |
| --- | --- | --- |
| Empty content | `content is None or content == ""` | Hard fail across all providers. |
| Thinking-prose markers | `content[:200]` contains `<think>` / `Okay, so I need to` / `Let me think` | First 200 chars only — avoids false-positive on transcripts that quote the marker later in the body. |
| Finish reason length | `finish_reason == "length"` | Indicates the model truncated mid-output. Structurally incomplete. |
| Bad JSON when expected | `expect_json=True` + `json.loads(content)` raises | Only when caller passes `expect_json=True`. |

Fine-tuning these against observed firing-rate data is tracked in #1002
(shared with ADR-099 — same thresholds apply, same evidence loop).

## Acceptance for the implementing PR (#1003)

- [ ] OpenAI, Anthropic, Gemini, DeepSeek provider call sites wire
  `providers.guardrails.check_chat_response(content, service=...)` at the
  content-producing stages (summary, cleaning, GI, KG, speaker as
  applicable).
- [ ] Mistral, Grok, Groq providers documented as remaining work with the
  same pattern; can ship in follow-up PRs.
- [ ] Cost-attribution: `llm_cost` log event extended with
  `triggered_guardrail` boolean field.
- [ ] Tests at the unit tier confirm the helper fires per-service for the
  4 failure modes; integration tests confirm the consumer-side fallback
  path triggers correctly per stage.
- [ ] Mock-server injection (from #999) reused for cloud E2E tests where
  practical.
- [ ] ADR-100 (this doc) referenced from the implementation site.

## Post-implementation updates (2026-06-15)

The implementing PR (#1003, commit `b0ee6c58`) closed the design as
specified above, but two facts surfaced during E2E validation that the
original draft assumed away. Both are now in the code; this section is
what a future reader needs to know.

### A. The `except GuardrailViolation: raise` clause

The design assumed `GuardrailViolation` would propagate naturally to the
`FallbackAwareSummarizationProvider` layer because "callers don't catch
unknown exception types." That assumption was wrong for the cloud
providers as written. Each provider's `summarize()` has a broad
`except Exception as exc` that wraps everything into
`ProviderAuthError` / `ProviderRuntimeError` for the
operator-facing error-classification system. `GuardrailViolation` got
silently re-typed and never reached the fallback layer.

The fix is one clause at every call site that calls
`check_chat_response`:

```python
except _guardrails.GuardrailViolation:
    raise  # ADR-100: let FallbackAware see the raw type, don't wrap
except Exception as exc:
    # existing error-classification block
```

Applied at: summarize on OpenAI / Anthropic / Gemini / DeepSeek. The
test that caught it was `test_cloud_guardrails_e2e.py` — the unit-level
wiring tests passed because they call the helper directly, not through
the provider's broad except. The cloud E2E was the only test that
exercised the full SDK → broad except → caller path.

The self-hosted whisper / diarize providers from ADR-099 already had
this clause (they were written knowing the wrap risk); the cloud
providers were not, because cloud providers' existing exception path
predates the guardrail design.

### B. Cleaning policy: catch-and-degrade, not propagate

Section 3 above specifies cleaning as "graceful degradation — return
the original transcript." The implementation lands that with an
explicit `except _guardrails.GuardrailViolation: return text` clause
in each provider's `clean_transcript`. The summarize and
GI / KG sites take the opposite policy (re-raise) because their
contracts demand non-empty output. **Don't paste-and-modify between
the two patterns** — they look identical but the trailing action
(`raise` vs `return text`) is the contract.

### C. Mock-server injection: extended to per-provider routes

`#999` shipped `inject_violation` for `/v1/chat/completions`,
`/v1/audio/transcriptions`, `/v1/diarize`, `/api/generate`. The
Anthropic and Gemini cloud providers use their own paths, so #1003
extended the injection vocabulary:

| Route | Violation types |
| --- | --- |
| `/v1/messages` (Anthropic) | `anthropic:empty_content`, `anthropic:thinking_prose`, `anthropic:max_tokens` |
| `/v1beta/generateContent` (Gemini) | `gemini:empty_content`, `gemini:thinking_prose`, `gemini:max_tokens` |

The `*:max_tokens` entries surface a known limitation: the helper trips
only on a literal `"length"` finish_reason. Anthropic's SDK returns
`"max_tokens"` and Gemini's returns `"MAX_TOKENS"`. Today the cloud
providers don't normalize these into `"length"`, so the helper
silently passes them through. The E2E tests document this as
a known gap (`test_cloud_guardrails_wiring.py::TestAnthropicWiring::test_max_tokens_stop_reason_treated_as_length`).
Normalizing at the provider call sites is a one-line follow-up.

### D. Cost-attribution: field exists, emit-with-flag deferred

The `emit_llm_cost_event(... triggered_guardrail: bool = False)`
extension lands in `workflow/cost_monitoring.py`. The cost-rollup
infrastructure can pivot on the field today. **What didn't land:**
the provider call sites don't yet emit a cost event with
`triggered_guardrail=True` in the `except GuardrailViolation`
block. The architectural reason: the cost-data (token counts, model,
estimated cost) is captured AFTER the guardrail check in the current
control flow. Emitting-with-flag requires a small refactor at each
call site (capture cost data BEFORE the check, emit-with-flag in
the except block, then re-raise). Tracked as item #4 in
`docs/wip/CLOUD-PROVIDER-RESILIENCE-E2E-GAP-1003.md`. Out of scope
for #1003; mechanics are now in place to make it a small follow-up
patch.

### E. Resilience-coverage audit (parallel review)

A second user-requested follow-up audit during the same review
flagged that prior to #1003, no cloud LLM provider had E2E
resilience coverage via the mock server's `set_error_behavior`
hooks. The matrix and follow-up gaps are documented in
`docs/wip/CLOUD-PROVIDER-RESILIENCE-E2E-GAP-1003.md`.
`tests/e2e/test_cloud_resilience_e2e.py` lands a per-provider
permanent-5xx baseline; transient-5xx retry, watchdog/timeout,
FallbackAware-routing-under-violation, and Ollama remain gaps for
a follow-up PR.

## References

- ADR-099 — the self-hosted precedent
- ADR-096 — the cloud fallback contract this builds on
- Originating eval reports:
  - `docs/guides/eval-reports/EVAL_PROMPT_LONG_V2_CROSS_PROVIDER_2026_06_14.md`
    (Gemini flash inconsistency, DeepSeek mixed)
  - `docs/guides/eval-reports/EVAL_REAL90_2026_06.md` (qwen3.5 thinking-budget;
    Ollama precedent but same shape across cloud thinking models)
- Code:
  - `providers/guardrails/chat.py` (the helper)
  - `providers/guardrails/exceptions.py` (`GuardrailViolation`)
  - `summarization/fallback.py` (FallbackAwareSummarizationProvider)
- Tracker: #1002 (threshold fine-tuning; shared)
EOF
