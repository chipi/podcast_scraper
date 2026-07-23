# ADR-122: Self-Hosted-Model Resilience Policy — backoff → trip → hold, selectable by run context

- **Status**: Accepted
- **Date**: 2026-07-21
- **Authors**: Podcast Scraper Team
- **Tracking issue**: [#1253](https://github.com/chipi/podcast_scraper/issues/1253)
- **Related ADRs**: [ADR-088](ADR-088-macos-local-ci-process-safety-for-ml-workloads.md) (process safety)
- **Related RFCs / prior art**: RFC-106 / #1198 (FallbackChain — pure DGX tier raises, chain owns
  the ladder), #954 (DGX watchdog + breaker), #876 (host/transcription resilience — "don't pile
  requests on an overloaded server")

## Context & Problem Statement

Validating the transcription bake-off for the 1000-episode reprocess
(`docs/wip/1000-EPISODES-REPROCESS-PLAN.md`, #1178/#1179), a **transient GPU contention** (a
lingering MOSS inference sharing the one DGX GPU) made the DGX whisper provider **time out on the
first call, immediately trip the circuit breaker, and give up** — after which every remaining
episode failed fast. The fuse worked *as designed*, but the design is tuned for a **live-serving**
pipeline (optimise availability: fail fast, fall over to another model, stay up). A **batch
reprocess** wants the opposite (optimise consistency: back off, hold the chosen model, pause on
sustained failure, never mix backends).

The platform already runs **seven** resilience mechanisms at different layers, and they blur
together. The ones relevant here:

| Layer | Mechanism | Problem class |
| --- | --- | --- |
| Transport | `hardened_http_client` (`resilience/sockets.py`) | connection reset/refused |
| Call | provider retry loop + watchdog (`deadlines.py`) | slow/hung request (**timeout**) |
| Circuit | `CircuitBreaker` (`resilience/breakers.py`) — `_whisper_breaker`, `_diarize_breaker` | endpoint wedged/down |
| Routing | `FallbackChain` (`resilience/fallback.py`) | provider unavailable → next model |

Two concrete defects for the reprocess context:

1. The call layer **treats a slow success like a wedged endpoint.** `whisper_provider._transcribe_via_dgx`
   `break`s on the first hard timeout (no backoff-retry), and `record_failure(hard=True)` trips the
   fuse **immediately** (bypassing `failure_threshold=2`). A transient GPU blip becomes a 5-minute
   DGX blackout for the whole batch.
2. The routing layer then **falls over to a different model** (`transcription_fallback_provider:
   whisper`, i.e. Mac-local whisper). For a controlled reprocess this silently produces a
   **corpus with mixed transcription backends** (inconsistent quality) — the opposite of what we want.

And **MOSS has no resilience at all** — its provider is a bare POST (no retry loop, no breaker),
so it is *less* protected than whisper/diarize, not more.

## Decision

Introduce a reusable **resilience policy** for the **self-hosted-model provider family** (DGX
whisper, DGX pyannote diarize, and MOSS), governing the call + circuit + routing layers, and
**selectable by run context**:

- **serve mode** → optimise availability. Unchanged from today: fail fast, trip early, fall over
  via `FallbackChain`. A live API serving users *should* degrade to cloud/local rather than fail.
- **reprocess mode** → optimise consistency. The new behaviour:
  1. **Timeout → exponential backoff + retry the *chosen* model.** The backoff (not a bare re-send)
     is what avoids piling requests on an overloaded server, so #876's objection is satisfied.
  2. **The fuse trips only after a policy threshold** of N failures-despite-backoff — not on the
     first hard timeout.
  3. **On a blown fuse: no cross-model fallover.** The chosen model is the only model. The batch
     **pauses and probes** the endpoint (half-open) until it recovers, up to a max-wait, then
     **alerts the operator** — it never switches backends.

The policy is a small abstraction in `resilience/` (a `ResiliencePolicy` carrying the knobs +
context) that the three self-hosted providers consume, rather than each provider re-implementing
retry/trip logic ad hoc. MOSS gains the policy it currently lacks.

### Policy knobs (config-tunable per run; these are the FINAL reprocess-mode defaults)

Finalized rather than left open: a reprocess runs on a DGX the operator dedicates (GPU exclusivity
is handled operationally — out of this ADR's scope), so contention is rare and these act as a
safety net, not a hot path. 3 retries over ~3.5 min ride out a transient co-tenant blip; a 15-min
pause-and-probe then survives a service restart without stalling the run forever. Tune from
observed behaviour, not a priori.

| Knob | Proposed default | Meaning |
| --- | --- | --- |
| `retries_before_trip` | 3 | backoff-retry cycles of the chosen model before the fuse trips |
| `backoff_schedule_sec` | 30, 60, 120 (×2, capped) | exponential wait between retries — rides out contention |
| `per_episode_max_wait_sec` | derived (duration-scaled timeout × retries) | ceiling before an episode's own attempts are abandoned |
| `on_open_max_wait_sec` | 900 (15 min) | how long the batch pauses-and-probes a blown endpoint before alerting the operator |

serve-mode defaults preserve today's behaviour (`retries_before_trip` effectively 1 on hard
timeout, fallover enabled).

## Alternatives Considered

- **Leave it as-is** and only fix the operator runbook. Rejected: the reprocess would keep
  producing mixed-backend corpora under transient contention, silently.
- **Remove `FallbackChain` fallover globally.** Rejected: the *live* pipeline genuinely wants
  fallover for availability. The behaviour must be context-selected, not deleted.
- **Patch the DGX whisper provider only.** Rejected: diarize has the same defect and MOSS has *no*
  resilience — a per-provider patch leaves the family inconsistent. A shared policy is the point.
- **Cover GPU exclusivity / service co-tenancy in the platform.** Explicitly out of scope — an
  operational concern the operator manages (which service owns the GPU, when).

## Consequences

- **Positive**: a reprocess rides out transient GPU contention on the chosen model, never mixes
  backends, and surfaces a genuinely-down endpoint to the operator instead of silently degrading.
  MOSS gains resilience. Resilience logic is centralised, not copy-pasted per provider.
- **Negative**: reprocess mode trades wall-clock for consistency — a contended DGX makes the batch
  *wait* rather than fall over, so a run can pause for minutes. That is the intended trade (a
  controlled reprocess is not latency-sensitive), bounded by `on_open_max_wait_sec`.
- **Neutral**: serve-mode behaviour is unchanged; the change is additive and context-gated.

## Implementation Notes

- **`resilience/`**: a `ResiliencePolicy` abstraction (knobs + `RunContext = serve | reprocess`) and
  a helper that wraps a "call the chosen model" callable with backoff-retry → trip-after-N → (serve:
  raise-for-chain | reprocess: hold-and-probe). Reuses the existing `CircuitBreaker`,
  `run_with_watchdog`, and the duration-scaled timeout.
- **Providers**: `tailnet_dgx/whisper_provider.py` (stop `break`-on-timeout; feed the policy),
  `tailnet_dgx/diarization_provider.py` (same), `providers/moss/moss_provider.py` (add the policy
  where there is none).
- **Run context**: derived from the profile — reprocess profiles (`reprocess_dgx_*`,
  `experiment_dgx_*`) select reprocess mode; serving profiles keep serve mode.
- **Tests (out-of-scope work is not merged without these)**: end-to-end coverage on the existing
  mock/stub server — extend `tests/integration/providers/test_tailnet_dgx_resilience_integration.py`'s
  `_DGXStubHandler` (`ok`/`hang`/`503`, add fail-N-then-recover sequences) and the LLM mock-server
  pattern (`tests/integration/utils/test_llm_resilience_mock_server.py`). Prove, with simulated
  responses: backoff-retry-then-succeed, trip-only-after-N, hold-no-fallover in reprocess mode,
  half-open recovery, and serve-mode fallover unchanged (regression guard).

## Decision update (2026-07-21): failure strategy is a first-class selector, and the LLM class joins the policy

The original decision tied behaviour to **run context** (serve ⇒ fail-fast/fallover,
reprocess ⇒ hold). Operating it surfaced two refinements, now implemented:

1. **The resolution behaviour is a named, selectable strategy — not a side-effect of context.**
   A single knob, `resilience_failure_strategy`, takes `failover | hold`:
   - **`failover`** — trip fast and fall over to the next provider in the configured chain
     (ASR: `FallbackChain`; LLM: the summary-fallback chain, e.g. DGX-vLLM → gemini). Availability.
   - **`hold`** — backoff-retry the *chosen* model, trip after N, pause-and-probe, then raise
     `ResilienceFuseOpenError` to halt the batch. No cross-model fallover. Consistency.

   Run context now only supplies the **default** (`serve → failover`, `reprocess → hold`); an
   explicit strategy in any profile/registry layer **overrides** it. This lets a reprocess run
   deliberately opt into gemini fallover (availability for that run), or a serve deployment choose
   hold — without a code change. Every existing profile is unchanged (the defaults reproduce the
   original serve/reprocess split). Resolved by `resolve_failure_strategy(cfg)` in
   `resilience/policy.py`; the four self-hosted providers and both transcription/diarization
   factories gate on `strategy is HOLD` rather than `run_context is REPROCESS`.

2. **The LLM class (summary / GI) is brought to parity under the same knob.** Previously ASR-only.
   - `failover`: `summarization/fallback.py` wraps the primary in the cross-LLM fallover chain
     (today's behaviour, #697's breaker still smooths 503 bursts with wait-and-resume).
   - `hold`: the fallover wrap is suppressed (chosen LLM only), and the per-provider
     `LLMCircuitBreaker` gains a **sustained-outage abort** — once an overload persists past
     `on_open_max_wait_sec` it raises `ResilienceFuseOpenError`, propagated as a hard-abort through
     `retry_with_metrics` so the workflow batch loop's `except ResilienceFuseOpenError: raise`
     halts the run identically to an ASR fuse. This is the LLM analogue of the ASR batch-halt.

Additionally, **every breaker TRIP now emits a guarded Sentry alert** (`capture_message`,
level=warning) at the closed→open transition — ASR `CircuitBreaker` (whisper/diarize/MOSS) and the
`LLMCircuitBreaker` alike — complementing the sustained-open escalation (`_emit_fuse_open_alert`,
level=error). A trip is a degradation; a held-and-never-recovered fuse is an outage.

Updated knob (in addition to the table above):

| Knob | Default | Meaning |
| --- | --- | --- |
| `resilience_failure_strategy` | derived: `serve → failover`, `reprocess → hold` | resolution strategy on chosen-model failure, honoured by ASR **and** LLM; overridable per profile/registry |

## Operational findings (2026-07-22): running the real reprocess pipeline surfaced a silent-default class

Validating the turbo ASR bake-off through the actual `migrate-diarization` make target (not a test
harness) exposed that a `reprocess_dgx_*` profile ran in **serve/failover**, not `hold` — its
transcription wrapped in a `FallbackChain`, so a DGX timeout would silently degrade to local whisper
and yield a **mixed-backend corpus**, the exact failure this ADR exists to prevent. Three linked
root causes, all fixed:

1. **Posture was name-derived only.** `reprocess_dgx_* → reprocess` was resolved from the profile
   *name*, which fires for `--profile <name>` but **not** for `--config <file>` — and the
   reprocess make targets load via `--config`. Fix: the reprocess profiles now **self-declare**
   `resilience_run_context: reprocess` + `resilience_failure_strategy: hold` (invocation-agnostic).
2. **`_build_config` dropped non-flag fields.** It carried `--config` YAML fields into the final
   Config via a *hand-maintained allowlist*; any field without an argparse flag (resilience_*,
   `transcript_cache_enabled`, …) silently reverted to its code default. Fix: it now carries **every**
   field the resolved config model set, killing the whole silent-default class in one place.
3. **The transcript cache replayed prior transcripts** on a model-swap reprocess (turbo replaying
   large-v3 by audio hash → the run "succeeded" doing nothing). Fix: `transcript_cache_enabled:
   false` on the reprocess profiles — a reprocess regenerates, it does not replay.

To make (1) systemic rather than per-profile, resilience posture is now a **registry-governed
field** (`REGISTRY_GOVERNED_FIELDS`): `make profiles-materialize` writes it into every profile and
`profiles-check` fails on drift. Regression: `test_cli_profile_routing` now round-trips the reprocess
profiles via **both** `--profile` and `--config`.

Lesson for the reprocess-once economics: a "runs green" reprocess is not evidence the *intended*
model ran — validate the ownership line (`transcription=…`, not `fallbackchain…`) and that
transcription actually executed (not a cache hit) before trusting a corpus.

## References

- Issue [#1253](https://github.com/chipi/podcast_scraper/issues/1253) — full scope + acceptance.
- `docs/wip/1000-EPISODES-REPROCESS-PLAN.md` — the arc this serves.
- [ADR-088](ADR-088-macos-local-ci-process-safety-for-ml-workloads.md), RFC-106/#1198, #954, #876.
