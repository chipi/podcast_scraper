# ADR-099: Response-shape guardrails for self-deployed inference services

- **Status**: Accepted
- **Date**: 2026-06-15
- **Authors**: Podcast Scraper Team
- **Related ADRs**: [ADR-096](ADR-096-dgx-spark-prod-primary-with-fallback.md) (DGX-as-primary fallback contract; this ADR extends the failure-mode classification that contract relies on)
- **Related issues**: [#999](https://github.com/chipi/podcast_scraper/issues/999) (this implementation), [#996](https://github.com/chipi/podcast_scraper/issues/996) (originating evidence), [#956](https://github.com/chipi/podcast_scraper/issues/956) (the connection-level resilience layer this ADR complements), [#1002](https://github.com/chipi/podcast_scraper/issues/1002) (threshold fine-tuning follow-up)
- **Related code**: `src/podcast_scraper/providers/tailnet_dgx/resilience.py` (existing layer this ADR extends), `infra/dgx/{whisper-server,pyannote-server}/` (services this defends)

## Context

The `tailnet_dgx.resilience` layer (#956) handles **connection-level failures** from self-deployed inference services on the DGX — timeouts, connection resets, circuit-breaker tripping. That's the right shape for the failure modes it was designed for, and the contract is documented in ADR-096 (DGX-as-primary with cloud fallback).

The 2026-06-15 #996 catastrophic-tail sweep (`EVAL_WHISPER_CONTENTION_AUTORESEARCH_2026_06_15.md`) measured **20% catastrophic failures** under vLLM contention on the GB10 box. Of those failures, **two of three failure modes are caught by the resilience layer** (hang, connection reset). The third — **"successful HTTP 200 response containing semantically corrupted content"** — is structurally invisible to the resilience layer because the transport completed normally.

The same systemic gap exists across every self-deployed inference service:

| Service | Observed silent-corruption mode | Source |
| --- | --- | --- |
| DGX whisper-openai | WER=1.000 garbage transcript under GPU contention | `EVAL_WHISPER_CONTENTION_2026_06` p03_e01; `EVAL_WHISPER_CONTENTION_AUTORESEARCH_2026_06_15` p08_e01 (in-band catastrophic) |
| DGX Ollama (qwen3.5 family) | Empty `content` because thinking-budget consumed all `num_predict` tokens before output emission | `EVAL_REAL90_2026_06`; `EVAL_PROMPT_LONG_V2_CROSS_PROVIDER_2026_06_14` Ollama re-run |
| DGX vLLM autoresearch | Malformed JSON, mid-truncation, prepended reasoning prose | `EVAL_PROMPT_LONG_V2_CROSS_PROVIDER_2026_06_14` cross-provider sweep |
| DGX pyannote | Hypothetical: single-speaker labels for multi-speaker audio under load | Not yet observed; preventive |

The pattern: **structurally-valid HTTP responses with semantically-wrong content propagate downstream as if real.** Whisper garbage → summarizer summarizes garbage → GI extracts entities from hallucinations → KG builds a graph from invented facts. The client at any layer of the stack has no way to know without a content-aware check.

This ADR captures the architectural decision for the defensive layer that closes the gap.

## Decision

**Add a generic response-shape guardrail layer co-located with the existing `tailnet_dgx.resilience` connection-level primitives**, with the following contract:

### 1. A single exception class — `GuardrailViolation`

Lives in `resilience.py` alongside `TimeoutLike`. The class carries enough context for downstream logging + Sentry capture:

```python
class GuardrailViolation(Exception):
    """A self-deployed service returned a successful HTTP response whose
    content fails a structural sanity check.

    Behavior: the existing fallback logic catches this as a sibling of
    TimeoutLike — DGX call counts as a failure, breaker records it, the
    consumer falls back to its cloud equivalent. No retry against DGX
    (a contention-driven garbage response is unlikely to fix itself
    on immediate retry; falling back is cheaper than re-pinging).
    """

    def __init__(self, service: str, reason: str, response_summary: str = ""):
        self.service = service        # whisper / ollama / vllm / pyannote
        self.reason = reason          # length_floor_violated / empty_content / ...
        self.response_summary = response_summary  # truncated repr for logging
```

### 2. Per-service callbacks invoked by a generic dispatch helper

```python
def check_response_shape(service: str, response: Any, **context: Any) -> None:
    """Raise GuardrailViolation if `response` for `service` fails its
    structural sanity check. Returns silently otherwise.

    `context` carries per-service hints (audio_duration_sec for whisper;
    expected_json_schema for vllm; etc.) so callbacks can do informed
    comparisons.
    """
```

The helper dispatches to per-service callbacks registered at module-load time. Adding a new service = adding a new callback registration. The list of services is small and reviewed at PR time — we do not need a plugin discovery system.

### 3. Hardcoded thresholds, not configurable

Thresholds are constants in code, not Pydantic config fields. **Rationale:**

- A guardrail that an operator can silently disable defeats its purpose. We have seen operators reach for "set the threshold to 0 to make CI green" patterns elsewhere; this layer is supposed to fail-closed.
- The configurability we WANT — adjusting thresholds against observed firing-rate data — is owned by the **fine-tuning tracker (#1002)**, not by profile config. When evidence accumulates that a threshold should change, the change ships in a PR with the rate-data screenshot attached.
- Initial thresholds are first-principles defaults (documented in #1002).

### 4. Telemetry — Prometheus counter + structured WARN log

Two pipelines, both within the existing observability budget:

| Pipeline | Shape | Cardinality / volume |
| --- | --- | --- |
| Prometheus counter `dgx_guardrail_violations_total{service, reason, host}` | One increment per `GuardrailViolation` raised | ~10 active series (4 services × ~2-3 reasons each + 1 host value); ~12 MB/month ingest at default 60s scrape |
| Structured WARN log via existing logger | Per-event payload: `service`, `reason`, `response_summary` (truncated to 200 chars), `audio_duration_sec` (when applicable) | ~500 bytes/event; at 20% catastrophic × ~100 calls/mo = ~10 KB/month |

Sentry capture of the exception happens via the existing error-reporting path (pyannote-server / whisper-server SDK init, see #942). **No separate breadcrumbs.** The exception IS the event; Sentry context comes from the existing scope.

**Cardinality discipline (enforced by code review):** the `reason` label is a fixed per-service enum. **Never** add high-cardinality fields (audio filename, request ID, response content) as Prometheus labels — those go in the log body and Sentry event context only. If the active series count for `dgx_guardrail_violations_total` ever climbs above ~50, that's a bug.

### 5. Mock-server test extension

The canonical E2E mock server at `tests/e2e/fixtures/e2e_http_server.py` already routes all four services' endpoints. It gains a test-only `inject_violation(route, violation_type)` classmethod that flags the next response on `route` to return a guardrail-violating payload. Test fixtures clear injections at teardown. **No production-path touch** — injection is class-state on the test handler.

This pattern is consistent with the existing `set_use_fast_fixtures` / `set_allowed_podcasts` classmethods on the same handler.

## Why this layer, not a different one

Other shapes were considered:

| Alternative | Why rejected |
| --- | --- |
| Per-consumer inline check (no shared layer) | Pattern would diverge across services; the next service added would not get the discipline for free; counter wiring would be reimplemented N times. |
| New separate `guardrails.py` module | Would split context-aware fallback handling across two modules. The existing `resilience.py` already owns the failure-mode taxonomy (`TimeoutLike`, breaker state); guardrails are a new failure-mode class in the same taxonomy. |
| Configurable thresholds per profile | Fail-closed discipline beats configurability for safety primitives. See § Decision item 3. |
| Sentry breadcrumbs to surface violations without raising | A breadcrumb without a raised exception means the downstream stage consumes the garbage anyway. The whole point is to raise → fall back → not consume the garbage. |
| Sample-rate-based golden-corpus audit (Layer 3 from #999) | Different layer. This ADR captures the cheapest defense (structural check at the boundary). Layer 3 / 4 are separate trackers if evidence shows the cheap layer is insufficient. |

## Consequences

### Positive

- The systemic gap is closed by a single small primitive (~50 lines of new code in `resilience.py`) that every current and future self-deployed service inherits.
- The fallback path is unchanged from the operator's perspective. `GuardrailViolation` behaves exactly like a 500 or timeout from the consumer's view: DGX fails, breaker counts, cloud picks up.
- The fine-tuning tracker (#1002) gives us a place to land threshold changes with evidence later, without that work blocking #999.

### Negative

- Adds one more failure-classification axis the resilience layer maintainers must keep coherent. Mitigated by co-locating with `TimeoutLike` so the two stay in sight together.
- Thresholds are hardcoded — when one needs to change, it ships in a code PR (not a config bump). For our deployment cadence (~weekly PRs), this is acceptable. If we ever needed sub-day threshold changes, that pattern would have to evolve.
- Sentry will count guardrail violations as errors, which may show up as a small ongoing alert noise. Mitigated by the existing `before_send` filter pattern from #942 — guardrails violations are *expected and handled* (fall back to cloud); if they're noisy enough to drown out real errors, the `before_send` filter can grow a per-fingerprint rate-limit for this exception class.

### Risk

- **Risk:** a guardrail false-fires on a legitimate edge case (e.g. a 30-second podcast clip with 90 seconds of silence triggers the whisper length floor). Customers experience an unnecessary cloud fallback.
  - **Mitigation:** thresholds are first-principles cautious (50% floor for whisper, not 80%). #1002 is the watchpoint — if we see false-fire rates >5%, tighten thresholds in a follow-up PR.
- **Risk:** an attacker manipulates input to provoke guardrail violations on every request, driving costs up via excessive cloud fallback.
  - **Mitigation:** out of threat model. Self-deployed services accept input from the same operator-controlled pipeline; no external attacker surface.

## Non-goals

- **This ADR does NOT cover Layer 2 / 3 / 4** from #999's broader framing (generic envelope-shape hook beyond the per-service callbacks; sample-rate golden-corpus audit; per-stack reliability profile). Those are separate future ADRs gated on evidence that Layer 1 is insufficient.
- **This ADR does NOT cover cloud providers' responses.** OpenAI, Anthropic, Gemini have their own structural validity contracts; we generally trust their SLAs. The architectural gap is specific to **self-deployed** inference.
- **This ADR does NOT change `tailnet_dgx.resilience`'s existing primitives.** `TimeoutLike`, `run_with_watchdog`, `CircuitBreaker`, `effective_timeout_sec` are unchanged. We extend the module; we don't refactor it.

## Initial thresholds (locked at ship time; tuning tracked in #1002)

| Service | Check | Threshold |
| --- | --- | --- |
| **Whisper** | Word-count floor | `word_count < int(duration_sec × 2.5 × 0.5)` (= 50% of expected speech rate) |
| **Ollama** | Empty content OR thinking-prose markers | `content == ""` OR contains any of: `<think>`, `Okay, so I need to`, `Let me think` |
| **vLLM** | JSON parse + finish_reason | When structured-output requested: JSON parse fails. Always: `finish_reason == "length"` |
| **Pyannote** | Empty segments for non-trivial audio | `segments == []` AND `duration_sec > 5.0` |

## Acceptance for the implementing PR (#999)

- [ ] `GuardrailViolation` exception class + `check_response_shape()` helper land in `resilience.py`
- [ ] All four service consumers wire the guardrail
- [ ] Prometheus counter `dgx_guardrail_violations_total{service, reason}` exposed
- [ ] Mock server `inject_violation()` classmethod added; existing E2E tests still green
- [ ] Unit tests: per-service guardrail fires on bad shape, does not fire on good shape
- [ ] Integration tests: existing `test_tailnet_dgx_*.py` patterns extended with guardrail scenarios
- [ ] E2E test: end-to-end fallback path triggered via injected violation, asserts cloud fallback fired
- [ ] ADR-099 (this doc) referenced from the implementation site

## References

- Originating eval reports:
  - `docs/guides/eval-reports/EVAL_WHISPER_CONTENTION_2026_06.md` § "2026-06-14 re-run"
  - `docs/guides/eval-reports/EVAL_WHISPER_CONTENTION_AUTORESEARCH_2026_06_15.md`
  - `docs/guides/eval-reports/EVAL_REAL90_2026_06.md` § "qwen3.5 burns budget on thinking"
  - `docs/guides/eval-reports/EVAL_PROMPT_LONG_V2_CROSS_PROVIDER_2026_06_14.md` § "Ollama"
- Existing resilience contract:
  - `docs/adr/ADR-096-dgx-spark-prod-primary-with-fallback.md`
  - `src/podcast_scraper/providers/tailnet_dgx/resilience.py`
- Observability substrate (Prometheus scrape + Sentry init):
  - `docs/guides/DGX_RUNBOOK.md` § "DGX observability"
  - `compose/grafana-agent.yaml` (`dgx-guardrail-violations` series ride on the existing pipeline-side remote_write)
EOF
)
