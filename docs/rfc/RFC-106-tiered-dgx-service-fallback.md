# RFC-106: Tiered fallback for DGX-backed pipeline stages

- **Status**: Implemented (all migration steps landed under #1198)
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

### The registry owns the chain; materialize emits it (source of truth)

The failover ladder is **registry-governed data**, exactly like the primary provider — not a
runtime default. This keeps the "profiles are views, the registry is the source of truth"
invariant (RFC-046) intact for fallback too.

- **`ProfilePreset`** gains a per-stage **`<stage>_fallback: list[str]`** — an ordered list of
  `StageOption` ids after the primary. Example (a DGX prod preset):

  ```python
  transcription="moss_transcribe_diarize",
  transcription_fallback=["tailnet_dgx_speaches_thread_b", "openai_whisper_1"],
  diarization="tailnet_dgx_diarization_community1",
  diarization_fallback=["deepgram_diarization_nova3"],
  ```

- **The resolver** maps that list of ids to their `StageOption.provider` values and emits
  **`<stage>_fallback_providers: list[str]`** into the materialized profile. The chain is therefore
  **present in the YAML** — no runtime inference, and `make profiles-check` guards it like every
  other registry-governed field.

- Because each tier is a `StageOption`, the chain is **eval-backed**: every entry carries its own
  `headline_metric`, `research_ref`, `realtime_multiple`, `resident_memory_gb`, and
  `tier` (`primary`/`fallback`). The ladder is a list of measured options, not opaque strings.

- **Config**: add `<stage>_fallback_providers: list[str]`. The legacy singular
  `<stage>_fallback_provider` maps to a **one-element chain** (full back-compat; no profile churn
  required to keep today's behaviour). Cloud presets keep their single cloud fallback (a
  one-element chain).

- **`allow_cloud_fallback`** (per profile): when false, the resolver refuses to emit a cloud
  `StageOption` into any chain — the ladder ends at the last on-prem tier and fails closed. This is
  what makes the `all-DGX / no-cloud` profiles safe to give a fallback at all.

## Migration — landed under #1198

1. ✅ Registry emits the chains (increment 1): `ProfilePreset.<stage>_fallback`, resolver emits
   `<stage>_fallback_providers`, `REGISTRY_GOVERNED_FIELDS` + config fields, `profiles-check` guard.
2. ✅ Transcription (increment 2a): `FallbackChainTranscriptionProvider` + `is_infra_failure`;
   `tailnet_dgx_whisper` self-wrap retired → pure DGX tier that raises. Closes the #1174 MOSS gap.
3. ✅ Diarization (increment 2b): `FallbackChainDiarizationProvider`; `tailnet_dgx` diarization
   self-wrap (local pyannote) retired onto the chain.
4. ✅ LLM/summary (increment 2c): `FallbackAwareSummarizationProvider` extended to an ordered
   chain, sourced from `summary_fallback_providers`; covers ollama and vLLM primaries.
5. ✅ Fail-closed (increment 3): `ProfilePreset.allow_cloud_fallback`; the resolver strips cloud
   tiers when False; airgapped presets declare it.

**Free-before-paid (operator directive).** Both stages prefer on-prem tiers before paid cloud:
`transcription: moss → dgx-whisper → local whisper → openai`;
`diarization: dgx pyannote → local pyannote → deepgram`. `is_infra_failure` cascades on
timeouts/5xx/connection-blips/guardrail-garbage and stops on content-deterministic failures
(payload-limit, non-429 4xx). The LLM chain keeps RFC-089's cascade-on-any contract.

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

## Open questions — resolved

- **Diarization cloud tier**: `deepgram`, but only after a free local in-process pyannote tier
  (`dgx pyannote → local pyannote → deepgram`) so a caller that can run pyannote never pays.
- **LLM-stage cloud tier**: `gemini` (the `cloud_balanced` summary tier) for the DGX prod profiles.
- **`all-DGX / no-cloud` fail-closed**: yes — implemented as `ProfilePreset.allow_cloud_fallback`
  (default True). When False the resolver drops cloud tiers from every chain; the airgapped presets
  set it.
