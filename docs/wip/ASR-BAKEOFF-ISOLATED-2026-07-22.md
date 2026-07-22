# ASR bake-off — isolated, GPU-exclusive (turbo vs large-v3 vs MOSS)

**Date:** 2026-07-22
**Issue:** #1178 (turbo eval) / #1179 (ASR-diar deathmatch) — the transcription-model lock for the
v2→v3-on-DGX reprocess (`docs/wip/1000-EPISODES-REPROCESS-PLAN.md`, Phase 1·B).
**Dataset:** the 10 Hard Fork episodes (`prod_v3_10ep_v1`, feed `simplecast.com/l2i9YnTd`), 11.1 h
of audio, 0001–0010. Real episodes → all corpora live in git-ignored `.test_outputs/`.

## Method — why this run is trustworthy

Every prior turbo attempt was polluted or dead: the first 99-ep run **cascaded** (serve-mode fuse
tripped on episode 1 under GPU contention → 89/90 `circuit_open`), and the ad-hoc 10-ep table was
measured with transcription **and** diarization + MOSS all contending on the one GB10 GPU (which is
why its MOSS RTF swung 2.4→145.8 — garbage).

This run fixes both:

- **One GPU-bound task at a time.** Transcription-**only** (`diarize: false`), one model per run,
  **sequential** (turbo → large-v3 → MOSS). Validated GPU-exclusive via `nvidia-smi` (whisper at
  76–96 % util; MOSS/pyannote loaded-but-idle, not competing for compute). RTF is real and comparable.
- **Real pipeline, under `hold`.** Run through the shipped `podcast_scraper.cli` reprocess path
  (existing-only, audio from the #947 cache — no re-download), `resilience_failure_strategy: hold`
  (no cross-model fallover), `transcript_cache_enabled: false` (fresh transcription, no replay).
- **0 retries / 0 fuse trips** on all three runs — exclusive GPU meant `hold` never had to intervene.

## Speed (clean RTF — realtime factor, higher = faster)

| model | aggregate RTF | per-episode range | 11.1 h audio transcribed in |
| --- | ---: | --- | ---: |
| **turbo** (`deepdml/faster-whisper-large-v3-turbo-ct2`) | **30.5×** | 19.9× – 35.2× | **21.9 min** |
| large-v3 (`Systran/faster-whisper-large-v3`) | 7.1× | 6.4× – 8.2× | 94.3 min |
| MOSS (`moss`, transcription-only) | 2.6× | 2.3× – 2.9× | 254 min |

**turbo is ~4.3× faster than large-v3 and ~11.6× faster than MOSS.** Projected onto a 1000-episode
corpus at this corpus's ~66-min average: turbo ≈ 1.5 days, large-v3 ≈ 6.5 days, MOSS ≈ 17.6 days
(scales with the real corpus's average episode length — Hard Fork runs long).

## Quality (WER vs the large-v3 reference)

No human gold exists, so this measures **divergence from large-v3** (the un-distilled model), not
absolute accuracy. Lower = closer to the reference.

| model | aggregate WER vs lv3 | ex-episode-6 |
| --- | ---: | ---: |
| turbo | 9.0 % | **5.4 %** |
| MOSS | 5.7 % | 5.5 % |

Per-episode turbo WER is ~3.6–8.8 % **except episode 6** — the 100-minute Ezra Klein episode —
where **turbo hits 29.9 %** while MOSS stays at 6.4 %. That single episode drives turbo's aggregate
from ~5.4 % to 9.0 %.

### The turbo long-episode cliff (the finding that matters)

Episode 6 is the longest (6024 s ≈ 100 min) and is turbo's worst on **both** axes — slowest RTF
(19.9× vs its 33× median) **and** by far the worst WER (29.9 %). large-v3 and MOSS handle it fine.
This points to turbo **degrading on very long audio** (a known faster-whisper-turbo weakness /
possible chunk-boundary handling). For a 1000-episode corpus with many 90 min+ episodes this is a
real risk, not a rounding error — **it must be characterised before turbo is locked in.** Likely
follow-ups: check `MOSS_CONTEXT_MAX_DURATION`-style chunking for the turbo path, or cap turbo to
episodes under a length threshold and route long ones to large-v3.

## Recommendation (#1178 / #1179)

- **turbo is the speed winner by a wide, clean margin** (~4× vs large-v3) and reaches quality parity
  with large-v3 on normal-length episodes (~5.4 % WER). The 4× speedup is decisive for a 1000-ep
  reprocess.
- **The long-episode quality cliff (ep6) is now handled** — not by a length heuristic but by a
  **quality-gate failover** (ADR-120 / #1258, built): after each turbo transcription the pipeline
  measures coverage (Σ segment durations / audio duration) and re-transcribes any episode below
  `transcription_coverage_min` (0.85) on large-v3. The detector flags ep6 cleanly (69% vs 92–97%
  healthy) at a 10% failover rate; aggregate speed stays ~18× (still ~2.5× faster than all-large-v3).
  Live on `reprocess_dgx_turbo`. **Remaining validation:** confirm the failover rate + threshold on
  the 100-episode run before the reprocess.
- **MOSS is not the transcription default.** Slowest by far (2.6×) and this measures it
  transcription-only — MOSS's actual value is the **joint transcribe+diarize** single pass, which
  this bake-off deliberately isolated away. Evaluate MOSS on the diarization axis (#1170/#1179), not
  here.
- **large-v3 stays the reference / long-tail fallback.**

## Caveats (equal weight)

- **10 episodes, one show** (Hard Fork). Not yet the cross-show 20–30-podcast spread — the ep6 cliff
  especially needs confirming across more long episodes before it's a lock.
- **WER is vs large-v3, not human gold** — it measures *divergence from the reference model*, so
  "turbo 5.4 %" means "differs from large-v3 by 5.4 %", not "5.4 % word errors against truth".
- **MOSS transcription-only** understates MOSS — its design is joint diarization.
- **Diarization + downstream quality not measured here** — this is the ASR axis only.

## Infra findings surfaced by running the real pipeline (folded into #1253)

Running the actual reprocess pipeline (not a harness) to get these numbers exposed three
reprocess-breaking bugs, all fixed + regression-tested this session:

1. **Reprocess profiles ran serve/failover, not `hold`** — posture was name-derived
   (`reprocess_dgx_*`), which fires for `--profile` but not the make targets' `--config` load. A DGX
   timeout would have silently fallen over to local whisper → **mixed-backend corpus**. Fix: profiles
   self-declare `resilience_run_context: reprocess` + `resilience_failure_strategy: hold`.
2. **`_build_config` dropped every non-argparse `--config` field** to code defaults (a
   hand-maintained carry-list) — resilience_* and `transcript_cache_enabled` both silently reverted.
   Fix: carry all resolved config fields.
3. **Transcript cache replayed** the prior model's transcript on a model-swap (turbo run "succeeded"
   doing nothing). Fix: `transcript_cache_enabled: false` on reprocess.

Systematised: resilience posture is now a **registry-governed field** materialized into all 15
profiles (`profiles-check` guards drift). **Lesson: "runs green" ≠ the intended model ran** — check
the ownership line (`transcription=tailnetdgx…`, not `fallbackchain…`) and that transcription
actually executed (not a cache hit) before trusting a reprocessed corpus.

## Reproduce

Configs (git-ignored, transcription-only, `hold`, cache-off) in the session scratchpad
`asr-bakeoff/txonly_{turbo,lv3,moss}.yaml`; per-model run via the reprocess CLI (existing-only, from
the #947 audio cache); RTF from each run's `episode_metrics` lines; WER via `compute_wer.py`
(rapidfuzz word-level Levenshtein, large-v3 as reference).
