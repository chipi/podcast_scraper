# EVAL — transcript segment-time drift and the word-timestamp fix

**Issue:** #1173
**Date:** 2026-07-16
**Fixture harness:** `tests/integration/eval/segment_drift_harness.py`
**Prod-safety guard:** `podcast_scraper.evaluation.segment_time_consistency`
**Drift math:** `podcast_scraper.evaluation.segment_time_drift`

## TL;DR

A subtitle/player seeks by each transcript segment's `start`/`end`. Those times drifted from the
audio because the pipeline stored whisper's **segment-level** timestamps, which are coarse and
accumulate error on long audio. The fix (`transcription/word_timestamps.py`,
`apply_word_timestamps` / `apply_nested_word_timestamps`) rewrites each segment's `start`/`end`
from its **word-level** times, which stay accurate.

- **On real prod data (reuse, no re-transcription):** the shipped prod-v2 corpus
  (`tailnet_dgx_whisper`) has **coarse-quantized** segment times (durations pile up at exactly
  1.0 / 2.0 / 3.0 s) and **3293 physically-impossible segments** (>40 chars/sec) across
  **164/218 episodes**. Human speech tops out ≈ 25 cps; `"in the United States."` in 0.2 s = 105.
- **On the v3 fixtures (base.en, 8-fixture varied subset):** the refinement cuts pooled
  turn-boundary drift from **p95 4012 ms → 487 ms** (mean 643 → 191 ms) vs the pre-fix
  segment-level times. On the clean (non-garble) fixtures drift is ≤ 400 ms p95 / ≤ 1000 ms max.

## Root cause

Whisper emits a coarse `segments` list and, when asked for `word_timestamps`, an accurate flat
`words` list. The pipeline stored the segment times directly. Segment-level times are quantized
and drift up to tens of seconds on long episodes; word-level times are within ~0.05 s of the
audio. The DGX whisper provider that produced prod-v2 exhibited this as **round-second duration
quantization** — the smoking gun in the shipped `.segments.json`.

## How drift is measured (fixtures)

The 46 v3 fixtures carry an exact per-turn timeline: turn *i* was rendered as a separate `say`
aiff, so RTTM line *i*'s onset is turn *i*'s true audio start. The harness transcribes the mp3 with
word timestamps and, for each turn boundary, finds where that turn's first word lands.

**Measurement trap (documented so it is not re-hit):** naive char-position alignment reported
2–3 s of phantom drift — an *alignment* artifact, because ASR garble makes the transcript diverge
from the source text. A `difflib` word-sequence alignment with a 3-word anchor removes it. Even so,
`asr_garble` fixtures and ad boundaries keep rare multi-second outliers (whisper's own word times
are unreliable on garbled/ad audio), so **p95 is the reported metric**; the strict absolute
AC2 bound is validated on real prod audio (large-v3) via the DGX subset, not on synthetic garble.

## Acceptance

- **AC1** — drift harness over the fixtures (`segment_drift_harness.py`), difflib-aligned.
- **AC2** — refinement holds pooled p95 ≤ 500 ms and beats segment-level by ~8×; clean fixtures
  ≤ 1000 ms max. Absolute p95 ≤ 300 ms / max ≤ 1000 ms → DGX prod-audio subset (pending).
- **AC3** — `enforce_segment_time_consistency` (monotonic, in-bounds, impossible-cps) flags the
  pre-fix prod corpus; a re-run with the fixed providers clears it.
- **AC4** — `tests/integration/eval/test_segment_time_drift_fixtures.py` recomputes drift from a
  committed words-cache (no whisper) and guards against regression.
- **AC5** — this document.

## Follow-ups

- **DGX subset (#1173):** re-transcribe a small varied set of real `diar-prod90` episodes with the
  fixed `tailnet_dgx_whisper` (large-v3, word timestamps) to record the absolute real-audio drift
  and confirm the impossible-cps count drops to ~0.
- **Fixtures v4 (#1189):** a TTS with tighter word-onset alignment (or stored per-turn silence
  trims) would lower the `say` fixtures' drift floor and make the strict AC2 bound measurable on
  fixtures directly. Staying on `say` for now.
