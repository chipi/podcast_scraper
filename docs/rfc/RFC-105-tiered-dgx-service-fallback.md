# RFC-105: Tiered fallback for DGX-backed pipeline stages

- **Status**: Proposed
- **Tracking issue**: #1198
- **Authors**: Marko
- **Stakeholders**: Core Pipeline, Providers, DGX Infra, Cost/Resilience
- **Related RFCs**:
  - `docs/rfc/RFC-046-materialization-architecture.md` (profiles/registry as source of truth)
- **Related work**:
  - `docs/guides/eval-reports/EVAL_MOSS_BAKEOFF_2026_07.md` (MOSS promoted as DGX transcription; #1174)
  - `#926` (the original lazy DGX-diarization → local fallback)
  - `#1177` (MOSS service) · `#952` (faster-whisper on :8000)

## Abstract

DGX-backed stages should degrade through an **ordered chain** of providers, not a single fallback.
This RFC defines a **provider-agnostic tiered fallback**: each stage (transcription, diarization,
summary/GI/KG) resolves to a primary provider plus an ordered list of fallbacks, tried in order on
failure, applied **uniformly** in one place instead of hand-wired inside individual providers.

## Problem statement

Fallback today is **per-provider and inconsistent**:

- `tailnet_dgx_whisper` self-wraps a single fallback (`transcription_fallback_provider`, default
  `openai`). `tailnet_dgx` diarization self-wraps its own lazy fallback (#926).
- The **`moss` provider has no fallback at all** — so promoting MOSS as the DGX transcription default
  (#1174) silently dropped the fallback; `transcription_fallback_provider` is dead config for it.
- The **LLM stages** (summary / GI / KG on Ollama :11434) have **no DGX-down fallback** — they fail
  if the box is unreachable.

And a **single** fallback level cannot distinguish the two failure modes a DGX profile actually
faces:

| Failure | What's reachable | Correct fallback |
| --- | --- | --- |
| One service crashes / its port is not in the tailnet ACL | rest of DGX is up | the **other DGX** provider for that stage |
| Whole DGX down / unreachable | nothing on the DGX | **cloud** (or local-mac) |

A profile that falls straight from MOSS to cloud pays for (and leaks to) the cloud even when
faster-whisper on the same box would have served it. A profile that falls only to DGX-whisper fails
outright when the box is down.

## Proposed design

### One ordered chain per stage, resolved once

Replace the singular `<stage>_fallback_provider` with an **ordered list**
`<stage>_fallback_providers: list[str]` — the chain tried after the primary. A generic
`FallbackChainProvider` wraps the primary + the chain and is constructed **once in each stage's
factory** (`transcription.factory`, `diarization.factory`, the LLM stage factory), so every provider
gets identical behaviour and the per-provider self-wrapping in `tailnet_dgx_whisper` /
`tailnet_dgx` diarization is retired.

```text
transcription (DGX prod):  moss → tailnet_dgx_whisper → openai
diarization   (DGX prod):  tailnet_dgx (pyannote) → deepgram
summary/gi/kg (DGX prod):  ollama(dgx) → gemini | openai
```

### Failure classification (do not cascade on bad input)

The chain advances only on **infrastructure** failures — connection refused, timeout, health-check
down, HTTP 5xx. A **content** failure (malformed audio, guardrail violation, a deterministic 4xx)
must **not** cascade: retrying the next tier wastes money and hides a real bug. Reuse the existing
`is_provider_audio_payload_limit_error` / guardrail classifications; add a small
`is_infra_failure(exc)` predicate the chain switches on.

### Health-aware skipping

Mirror `tailnet_dgx_whisper`'s existing preflight: if a tier exposes a health endpoint and it
reports down, skip straight to the next tier instead of eating the request timeout. Registry
`StageOption.endpoint` already carries the health surface.

### Config + registry

- Config: add `<stage>_fallback_providers: list[str]`. The legacy singular
  `<stage>_fallback_provider` maps to a **one-element chain** (full back-compat; no profile churn
  required to keep today's behaviour).
- Registry: a `ProfilePreset` may name a `<stage>_fallback` **list** of `StageOption` ids; the
  resolver emits the ordered provider list into the materialized profile. DGX presets get the chains
  above; cloud presets keep their single cloud fallback (a one-element chain).

## Migration

1. Land `FallbackChainProvider` + `is_infra_failure` + the config field (singular → 1-chain shim).
2. Move `tailnet_dgx_whisper` / `tailnet_dgx`-diarization fallback into the generic wrapper; delete
   the self-wrapping.
3. Add the LLM-stage chain (ollama → cloud) — new capability.
4. Set the DGX presets' chains in the registry; re-materialize.
5. Tests per tier: infra-failure cascades, content-failure does not, health-down skips, chain
   exhaustion raises the last error.

## Non-goals

- **Not** a load balancer or a circuit breaker with backoff windows — just ordered try-next-on-infra-
  failure. (A breaker can layer on later.)
- **Not** cross-stage orchestration (e.g. "if transcription fell to cloud, also move summary") — each
  stage's chain is independent.
- **Not** changing which providers win a bake-off — the chains encode existing eval verdicts.

## Alternatives considered

- **Keep per-provider self-wrapping, just add it to `moss`.** Fastest, but perpetuates the
  inconsistency (every new DGX provider re-implements fallback) and stays single-level — cannot
  express the DGX-whisper-then-cloud tier. Rejected as the durable answer; acceptable only as an
  interim stopgap if the chain work slips.
- **Pipeline-level try/except around each stage call.** Spreads fallback policy across the workflow
  instead of the provider layer; harder to test and to keep consistent with the registry.

## Open questions

- Diarization cloud tier: `deepgram` (paid, strong) vs `gemini` (already used for other stages)?
- LLM-stage cloud tier per profile: `gemini` (cloud_with_dgx) vs `openai`?
- Does `all-DGX / no-cloud` intent (some profiles forbid cloud) need a chain that ends at the
  last DGX tier and **fails closed** rather than reaching cloud? Likely yes — a per-profile
  `allow_cloud_fallback` flag.
