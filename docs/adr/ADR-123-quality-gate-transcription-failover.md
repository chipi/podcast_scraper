# ADR-123: Quality-gate transcription failover — re-route silently-incomplete outputs by coverage

- **Status**: Accepted
- **Date**: 2026-07-22
- **Authors**: Podcast Scraper Team
- **Tracking issue**: [#1258](https://github.com/chipi/podcast_scraper/issues/1258)
- **Related ADRs / RFCs**: [ADR-122](ADR-122-self-hosted-model-resilience-policy.md) (resilience
  policy: backoff→trip→hold), RFC-106 / #1198 (`FallbackChain` — infra failover), #1178/#1179 (the
  ASR bake-off that surfaced this).

## Context & Problem Statement

The #1178/#1179 ASR bake-off (`docs/wip/ASR-BAKEOFF-ISOLATED-2026-07-22.md`) found that
`large-v3-turbo` — the ~4× faster model we want for the v2→v3 reprocess — **silently drops large
spans of speech on long episodes, with no error raised.** On a 100-minute episode it transcribed
only **69% of the speech** (100 gaps, ~25 min dropped, 10,469 micro-segments) and scored 29.9 % WER
against large-v3; every episode ≤ 75 min was fine (92–97 % coverage, ~4–9 % WER). It reached the
outro (not a truncation) and did not repeat (not a hallucination) — a long-form VAD/segmentation
drop that returns a clean, coherent, materially **incomplete** transcript.

The two failover mechanisms we already have do **not** catch this:

- **`FallbackChain`** (RFC-106) advances only on an **exception** — a timeout, a 5xx, a connection
  reset. Turbo's call *succeeded*; nothing raised.
- **`hold`** (ADR-122) governs *infra-failure* behaviour in a reprocess (suppress fallover, halt on
  sustained failure). Also exception-driven; irrelevant to a successful-but-bad output.

So a turbo-built v3 corpus would lose ~a quarter of its longest — and content-richest — episodes,
invisibly.

## Decision

Introduce a third, distinct failover **mode**: **quality-gate transcription failover**, triggered by
a *successful-but-bad output* rather than an exception.

After a transcription succeeds, compute **coverage**:

```text
coverage = Σ(segment_end − segment_start) / audio_duration_seconds
```

from the provider's own segments (no reference transcript needed). If `coverage <
transcription_coverage_min`, the primary silently dropped speech — **re-transcribe the episode on a
failover model** (`transcription_coverage_failover_model`, e.g. large-v3) and use that result.

This is implemented as a wrapper — `CoverageGatedTranscriptionProvider` — mirroring
`FallbackChainTranscriptionProvider`, but gating on output coverage instead of catching an
exception. The winning model is recorded on the result (`model_used` / a `coverage_failover`
breadcrumb) so **per-episode provenance is preserved**.

### The failover taxonomy after this ADR

| mode | trigger | mechanism |
| --- | --- | --- |
| infra failover (RFC-106) | call **raised** (timeout / 5xx / conn reset) | `FallbackChain` advances tiers |
| `hold` (ADR-122) | reprocess consistency | *suppress* infra failover; halt on sustained failure |
| **quality failover (this)** | call **succeeded**, output **coverage < min** | `CoverageGated` re-transcribes on the failover model |

### Orthogonality to `hold`

`hold`'s guarantee is "no cross-model fallover **on infra failure** → no silent mixed-backend
corpus." Quality-failover *does* produce a mixed-backend corpus (turbo for most, large-v3 for the
low-coverage tail) — but this is **not** the failure `hold` forbids, because it is:

1. **explicit and declared** — the policy is "turbo, with large-v3 for episodes turbo demonstrably
   drops", set in the profile, not a silent runtime degradation; and
2. **recorded per episode** — each episode's metadata already carries its transcriber
   (`transcript_source` / `model_used`), so the corpus is mixed *by design and by record*.

The two are therefore **orthogonal triggers**: a reprocess profile can run `hold` on infra failure
**and** quality-failover on coverage. They compose — the coverage gate wraps the (possibly
hold-protected) primary.

### Knobs (registry-governed, materialized into profiles — same pattern as ADR-122)

| Knob | Default | Meaning |
| --- | --- | --- |
| `transcription_coverage_min` | `0.0` (gate OFF) | re-transcribe when coverage falls below this; a reprocess sets ~`0.85` |
| `transcription_coverage_failover_model` | `null` | the whisper model to re-transcribe with (e.g. `Systran/faster-whisper-large-v3`) |

Gate is active only when `coverage_min > 0` **and** a failover model is set. Both are
`REGISTRY_GOVERNED_FIELDS`; `profiles-check` guards drift.

## Alternatives Considered

- **Route by episode length up front** (turbo for short, large-v3 for long). Rejected as the
  primary mechanism: it re-runs *all* long episodes including the ones turbo handles fine, and the
  length threshold is unknown/​content-dependent. Coverage measures the *actual* failure per episode.
  (A length pre-filter could still be layered on later as an optimisation.)
- **Fix turbo's VAD/chunking so it never drops.** Out of scope here and not guaranteed to hold
  across all long/dense audio; the gate is the safety net regardless of upstream tuning.
- **Detect via WER against large-v3.** Requires transcribing every episode twice on both models —
  defeats the speed win. Coverage needs only the primary's own output.

## Consequences

- **Positive**: turbo's ~4× speed is usable for the reprocess without losing content on long
  episodes — only the flagged tail (~10 % on the bake-off sample) pays the slow path, so aggregate
  throughput stays well above all-large-v3 (~18× vs 7.1× on the sample). The failure is caught from
  the output alone, no reference needed. Provenance is preserved.
- **Negative**: a flagged episode is transcribed twice (turbo, then large-v3) — the turbo pass is
  "wasted" but is how the drop is detected; net still far faster than all-large-v3. Coverage is a
  proxy (silence-heavy audio reads as lower coverage); the threshold must be tuned on real data
  (validate on the 100-episode run).
- **Neutral**: gate defaults OFF (`coverage_min = 0.0`), so serving profiles are unchanged; it is
  opt-in via the reprocess presets.

## Implementation Notes

- `CoverageGatedTranscriptionProvider` in `providers/resilience/` — wraps `(primary_builder,
  failover_builder)`, computes coverage from `result["segments"]` and
  `episode_duration_seconds` (fallback: `probe_audio_duration_sec`), re-transcribes on the failover
  builder when below `coverage_min`, records provenance.
- Transcription factory wires it when `coverage_min > 0` + a failover model is set; the failover
  builder constructs the same provider type with `dgx_whisper_model` overridden to the failover
  model. Composes with the `hold` factory gate (ADR-122).
- Registry: two new `REGISTRY_GOVERNED_FIELDS`, emitted per preset; `make profiles-materialize`.
- Tests: a low-coverage stub triggers the re-route (and records provenance), a healthy one does not;
  a coverage-metric unit test.

## References

- Issue [#1258](https://github.com/chipi/podcast_scraper/issues/1258) — scope + acceptance.
- `docs/wip/ASR-BAKEOFF-ISOLATED-2026-07-22.md` — the ep6 evidence + coverage detector prototype.
- [ADR-122](ADR-122-self-hosted-model-resilience-policy.md), RFC-106/#1198.
