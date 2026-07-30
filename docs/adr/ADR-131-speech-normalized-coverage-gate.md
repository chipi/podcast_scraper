# ADR-131: Speech-normalized coverage for the transcription quality gate (revises ADR-123)

- **Status**: Accepted (approach **C** — reuse the diarizer's speech regions)
- **Date**: 2026-07-26
- **Authors**: Marko Dragoljevic
- **Related ADRs**: [ADR-123](ADR-123-quality-gate-transcription-failover.md) (the coverage-gate
  failover this revises), [ADR-122](ADR-122-self-hosted-model-resilience-policy.md) (`hold`)
- **Related**: #1178/#1179 (ASR bake-off), #1258 (coverage gate)

## Context & Problem Statement

ADR-123 gates a turbo→large-v3 failover on **coverage**:

```text
coverage = Σ(segment_end − segment_start) / audio_duration_seconds
```

The **denominator is the full audio duration**, so every second of non-speech — intro music, ad
breaks, stingers, silence — counts as "dropped speech." The metric therefore cannot tell a genuine
ASR failure apart from an episode that simply contains a lot of non-speech. It fires the failover on
the wrong signal.

### Evidence (measured on the real v2.3 turbo corpus, 90 episodes)

- **The gate never actually fired.** All 90 episodes are turbo (no `large-v3`/`Systran` `model_used`
  anywhere; the runs carried no `failover_model`). So the corpus is raw turbo output.
- **Recomputing `Σseg / audio_dur`: 11 / 90 (12%) fall below the 0.85 gate** — but on transcripts no
  model can improve, because the shortfall is non-speech:

  | episode | coverage | length |
  | --- | ---: | ---: |
  | Move Over Humans (WSJ) | **67.4%** | 20 min |
  | Tim Cook Built the Apple Empire | 71.3% | 21 min |
  | R.I.P. Spirit Airlines | 72.0% | 19 min |
  | … 8 more, 72–84% | | mostly 19–27 min |

  These are **short** NPR/WSJ/The-Daily-style episodes with intro music + ad breaks — not the
  long-episode cliff. On "Move Over Humans" large-v3 also reached only ~75.9%, so ~24% is music both
  models correctly skip and turbo's *real* speech drop was ~9%. Under the current gate all 11 would
  failover to large-v3 (4× slower) and recover nothing.

### The proposed metric, validated on the same corpus

Recomputing `speech_coverage = Σ(transcript) / Σ(pyannote speech)` (denominator from the persisted
diarization caches) on the WSJ feed — the false-positive cluster:

| episode | raw coverage | **speech coverage** |
| --- | ---: | ---: |
| Move Over Humans | 76.1% | **96.9%** |
| Tim Cook Built the Apple Empire | 79.9% | **103.6%** |
| The College Student Who Defeated… | 79.7% | **99.4%** |
| … all 10 WSJ episodes | 76–89% | **97–105%** |

Every episode raw-coverage would have failed over shows **~100% speech coverage** — turbo
transcribed essentially all the *speech*; the shortfall was non-speech both models skip. (Values
slightly over 100% are transcript/diarization boundary overlap; clamp at 1.0.) This is the direct
empirical confirmation that the speech-normalized denominator removes the false positives.

### But the gate catches a real, different failure

The genuine turbo problem is the **long-episode cliff**: the 100-min Ezra Klein bake-off episode
scored **29.9% WER** vs large-v3's ~6% — a true long-form speech drop. That episode *should*
failover. A correct metric must still catch it while passing the 11 music-heavy ones. The two are
distinguishable only once non-speech is out of the denominator: the cliff has low **speech**
coverage; the music-heavy episodes have high speech coverage.

## Decision

Gate on **speech coverage**, not raw audio coverage:

```text
speech_coverage = Σ(segment_end − segment_start) / speech_seconds
```

where `speech_seconds` is the duration of **actual speech** in the audio (non-speech removed).
Failover when `speech_coverage < transcription_speech_coverage_min`. The failover *mode* (ADR-123's
`CoverageGatedTranscriptionProvider` wrapper, the `hold` orthogonality, the per-episode provenance
breadcrumb) is unchanged — only the metric's denominator changes, plus the breadcrumb now records
**both** raw and speech coverage.

### The implementation fork (the decision that needs your call)

The gate runs **client-side** (in `fallback.py`), but the whisper model is a **remote DGX server** —
so `speech_seconds` is not free. `ffmpeg silencedetect` (already a dependency) is **not** sufficient:
it is energy-based and detects *silence*, not *music*, and music is exactly the false-positive case.
The viable options:

| # | approach | new dep? | catches music? | where VAD runs | cost |
| --- | --- | --- | --- | --- | --- |
| **A** | **silero-vad client-side** | **yes** (silero-vad, ~2 MB torch model) | yes | laptop CPU | ~1–3 s/ep; keeps the gate exactly where ADR-123 put it |
| **B** | **DGX whisper server enables `vad_filter`, returns `speech_seconds`** | no (client) | yes | DGX (where faster-whisper already is) | server-side change to homelab infra |
| **C** | **reuse pyannote diarization as the VAD** (Σ speaker-speech / audio) | no | yes (music has no speaker) | DGX (already running) | move the gate downstream of diarization; re-transcribe-only on failover (diarization stays valid) |

- **A** is the smallest change to *this* repo but adds a client runtime dependency (needs approval
  per the no-new-deps rule).
- **B** is architecturally cleanest (VAD where the model is, zero client cost) but touches the DGX
  whisper server, i.e. homelab infra outside this repo.
- **C** adds no dependency and reuses a signal we already compute, but relocates the gate from the
  transcription-provider wrapper to the orchestration level (after diarization) — a larger refactor
  of ADR-123's clean wrapper design. Diarization is audio-based, so it is valid for both the turbo
  and the (possible) large-v3 transcript; only transcription is re-run on failover.

**Chosen: C.** Keeps the pipeline fully local (the v2→v3 goal), adds nothing, and reuses the speech
regions the diarizer already produces.

**C is provider-agnostic, not pyannote-specific.** Every diarization provider (Deepgram, MOSS,
Gemini, pyannote, tailnet-DGX) returns the same `DiarizationResult(segments=[start,end,speaker])`, so
`speech_seconds = Σ(merged segments)` reads off the common interface regardless of which diarizer
runs. Two consequences to design for:

1. **Diarization must be ON.** When `cfg.diarize` is off, or the diarizer returns no speaker turns,
   there is no speech denominator — the gate then **falls back to the raw-coverage gate** (ADR-123's
   original metric) so behaviour never regresses; the speech-normalized path is a strict upgrade that
   engages only when diarization is present.
2. **The denominator inherits the diarizer's speech detection.** A word-level diarizer (Deepgram)
   gives slightly tighter spans than a turn-level one (pyannote); a diarizer that labelled music as a
   speaker would inflate the denominator. All current providers do internal VAD and the threshold
   carries margin (clean ≈ 100%, real drops far lower), so the gate is robust — but its denominator
   quality does track the configured diarizer. Full diarizer-independence would require A (silero-vad).

### Threshold recalibration (required, not optional)

The current `0.85` was set on the *raw* metric and is meaningless on speech coverage (normal
episodes will jump to ~0.98). Before enabling, recompute `speech_coverage` on the **bake-off 18**
(which have both turbo and large-v3, plus the ep6 cliff) and pick a `transcription_speech_coverage_min`
that (a) failovers the Ezra Klein cliff and (b) passes "Move Over Humans." Record the chosen value
and the two anchor points.

## Alternatives considered

- **Keep raw coverage, lower the threshold.** Rejected: no single raw threshold separates the ep6
  cliff (69% raw, real drop) from "Move Over Humans" (67% raw, non-speech) — they overlap. The
  confound is structural, not a threshold-tuning problem.
- **Always transcribe on both and compare (turbo vs large-v3).** Rejected: defeats the 4× speed
  purpose of turbo — you would run large-v3 on every episode.
- **ffmpeg `silencedetect`.** Rejected as the sole signal: misses music, the main false-positive.

## Non-Goals

- Not removing the gate — the long-episode cliff is real and must still failover.
- Not changing the failover *mode* — ADR-123's wrapper, `hold` orthogonality, and provenance
  breadcrumb stay; only the denominator and the recorded fields change.
- Not a new ASR model or a turbo replacement.

## Consequences

- **~12% fewer false failovers** on the corpus profile measured — music/ad-heavy episodes stay on
  turbo (correctly) instead of paying 4× for large-v3 that recovers nothing.
- **The gate measures what it claims** — real speech drop, not audio composition.
- **No re-transcription of the v2.3 corpus needed** — the 11 flagged episodes are fine (non-speech);
  this only changes what the *next* reprocess does.
- **Breadcrumb gains `speech_coverage` + `speech_seconds`** alongside the existing raw coverage, so
  provenance shows both.
- One of: a new client dep (A), a DGX server change (B), or a gate relocation (C) — the open decision.

## Validation plan

1. Unit: speech-coverage on a synthetic segments+VAD fixture (speech-only, music-padded, real-drop).
2. Recompute speech coverage on the bake-off 18 → set the threshold; assert ep6 fails and Move Over
   Humans passes.
3. Smoke: reprocess "Move Over Humans" (must NOT failover) and the Ezra Klein episode (must failover).
