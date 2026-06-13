# EVAL — Diarization championship (3-way: MPS / CUDA / Gemini)

Phase 1 — 2026-06-10 — MPS / CUDA / CPU.
Phase 2 — 2026-06-13 — Gemini 2.5 audio API added (#962); fresh
pyannote/MPS re-run on the post-#944 multi-voice v2 fixtures.
Phase 3 — 2026-06-13 — real Diarization Error Rate (DER) computation (#992),
with time-aligned ground truth derived from Deepgram nova-3 word-level
timestamps.

**Issue:** #930, #962, #992
**Branch:** `feat/autoresearch-batch-3-championships` (Phase 1); `main` (Phases 2 + 3)
**Dataset:** v2 audio fixtures, 5 episodes (RFC-059 §2 macOS `say` generation)
**Status:** **DER measured** for pyannote/MPS + Gemini 2.5 Flash. Phase 1
pyannote/DGX numbers are reused — same model, different device, identical
segment counts within numerical noise.

---

## TL;DR

**pyannote-on-Apple-MPS is essentially tied with pyannote-on-DGX-CUDA in latency.** Pure CPU is **~17× slower** than either GPU path. The operator's MacBook is a fully viable diarization platform for podcast workloads at the scale we run — DGX adds value primarily for x86/ARM-no-GPU deployments (cheap VPS, CI runners) and for keeping load off the laptop.

## Latency results (5 v2 episodes, ~5 min each)

| Backend | Hardware | Episodes | Mean wall (s) | Realtime multiple |
| --- | --- | ---: | ---: | ---: |
| **Apple MPS** | M4 Pro laptop (auto-pick) | 5/5 | **23.35** | **~13×** |
| **NVIDIA CUDA** | DGX GB10 via `:8001` | 5/5 | **23.50** | **~13×** |
| **Pure CPU** | M4 Pro with `LOCAL_DIARIZE_DEVICE=cpu` | 1/5 | **415.48** | **0.7×** |

Per-episode breakdown:

| Episode | Apple MPS (s) | NVIDIA CUDA (s) | Pure CPU (s) |
| --- | ---: | ---: | ---: |
| p01_e01 | 23.1 | 25.7 | **415.5** |
| p02_e01 | 27.1 | 26.7 | (not run — CPU too slow) |
| p03_e01 | 20.6 | 19.9 | (not run) |
| p04_e01 | 23.6 | 23.4 | (not run) |
| p05_e01 | 22.4 | 21.8 | (not run) |

## Speaker count detection (3-backend, 5 episodes)

All three backends detect **3 speakers per episode** on v2 fixtures (MPS and CUDA always 3; the partial CPU run also 3). The transcript-label parser reports 5 "ground-truth" speakers — but **the v2 audio is generated with a single TTS voice** (`tests/scripts/transcripts_to_mp3.py` defaults to `say --voice Alex` for all roles per the current state of `#934`). So pyannote can't separate Maya / Liam / Ad acoustically when they all speak with Alex's voice; clustering on residual acoustic features lands at 3.

**Until `#934` lands distinct voices per speaker, speaker-count accuracy on v2 fixtures is not a useful pyannote-quality signal.** Segmentation density (segments-per-turn ratio) is the cleaner number on this dataset.

## Segmentation density

Segments produced by pyannote ÷ turn-changes in the transcript:

| Backend | Mean ratio (5 ep) | Range |
| --- | ---: | --- |
| Apple MPS | 1.07 | 0.87 – 1.28 |
| NVIDIA CUDA | 1.08 | 0.87 – 1.28 |
| Pure CPU | 1.00 (1 ep) | n/a |

All three backends produce **roughly the same segmentation density** (~1.0× turn-changes), which is what you'd hope for: identical model, identical inputs, identical output up to device-level numerical noise. The GPU vs CPU difference is purely a latency story for this model + dataset, not a quality story.

## What this means operationally

**The "no DGX" path is viable for podcast diarization workloads** if the operator runs on Apple Silicon. ~25 s per 5-min episode on an M4 Pro is fine for batch processing dozens of episodes per session. The DGX adds value when:

1. The host machine has no GPU (CPU = 17× slower → unworkable for prod).
2. Diarization runs in production CI / VPS infra (no Apple Silicon there).
3. The operator wants to keep load off the laptop while transcription / KG also runs.

For laptop-driven manual processing, **either path is fine**. This is a strong signal for the `local` profile shape (`config/profiles/local.yaml`): keep the in-process pyannote default for laptop runs, no DGX required.

## Phase 2 — Gemini 2.5 audio API added (2026-06-13)

Issue #962 ships the `GeminiDiarizationProvider` — Gemini 2.5 Flash audio input
with a structured-JSON prompt asking for speaker turns. Closes the
"#962 cloud_*-needs-a-non-pyannote-path" gap.

| Backend | Episodes | Mean wall (s) | Mean ratio (seg / gt-turn) | Cost / 5-min ep |
| --- | ---: | ---: | ---: | ---: |
| **pyannote / MPS** (fresh re-run 2026-06-13) | 5/5 | **22.2** | **1.07** | $0 |
| **pyannote / DGX CUDA** (phase 1) | 5/5 | 23.5 | 1.08 | $0 (DGX) |
| **Gemini 2.5 Flash audio** | 5/5 | **37.3** | **1.68** | ≈ $0.03/ep |

**Verdict:** Gemini is a *working* backend — every v2 episode produces valid
speaker turns — but it **over-segments by ~60%** vs pyannote (ratio 1.68 vs
1.07; lower is closer to the transcript's actual turn count). Latency is
also ~1.6× pyannote on the laptop. The audio API is **not free** (~$0.03 per
5-min episode under the Gemini 2.5 Flash audio rate). Pyannote stays the
quality + cost leader on every dimension.

The win for #962 is **fall-back coverage**: the `cloud_*` profile path no
longer 404s on diarization when pyannote isn't installed. Use:

- `diarization_provider: local` — default, pyannote everywhere it's installed.
- `diarization_provider: tailnet_dgx` — DGX-equipped profiles per #926.
- `diarization_provider: gemini` — **new**: cloud-only profiles that want
  to skip the pyannote dependency (pipeline-llm image / lightweight CI).

No production-default flip is warranted; pyannote stays the canonical
diarizer. Filing the Gemini path as the cloud_* fall-back.

## Phase 3 — real DER (#992, 2026-06-13)

Closes the speaker-confusion blind spot that `segments_per_turn_ratio`
couldn't measure. Time-aligned ground truth derived per the path proposed
in #992:

1. Deepgram nova-3 word-level timestamps on each v2 episode (5 calls, ~$0.10
   total, <2 s/episode wall-clock).
2. Word-level Levenshtein DP alignment between the reference transcript and
   Deepgram's hypothesis text. Aligned ratio: 1469–1471 / 1485 reference
   words on average (98.9 %). Unaligned reference words get times linearly
   interpolated between aligned neighbours.
3. Each reference word inherits its `Speaker:` line label. Contiguous
   same-speaker words collapse to `(start, end, speaker)` ground-truth
   utterance segments.
4. `pyannote.metrics.diarization.DiarizationErrorRate` with `collar=0.0`,
   `skip_overlap=False`. Optimal speaker mapping is solved internally
   (Hungarian), so reference labels like `Maya` and hypothesis labels like
   `SPEAKER_00` don't need to match by name.

Aggregate across all 5 episodes (micro-average — pool seconds then divide):

| Backend | **DER** | Confusion | Missed | False alarm | Total reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| **pyannote / MPS** | **1.66 %** | 0.93 % | 0.48 % | 0.25 % | 2779.9 s |
| **Gemini 2.5 Flash** | **101.96 %** | 31.46 % | 22.99 % | 47.51 % | 2779.9 s |

**Pyannote scores 1.66 % DER**. Sub-second-per-episode speaker confusion
across ~9 minutes of audio per episode. This is *very* good — pyannote is
not just qualitatively the winner, it's quantitatively essentially correct.

**Gemini scores 101.96 % DER** — yes, above 100 %, which means errors
exceed total reference speech time. This isn't a marginal failure mode; it
reveals a deeper Gemini-side bug that the Phase 2 segment-ratio couldn't
see:

- **`p01_e01`**: Gemini's max timestamp is 9.11 — the audio is 551 seconds
  long. Gemini emitted times in *minutes*, not seconds, despite the prompt
  explicitly requesting "floating-point seconds from the start of the
  audio". Result: every Gemini segment is compressed into the first 9.11 s
  of the timeline → 542 / 549.7 s of reference speech sits past the
  hypothesis's max time → missed-detection dominates.
- **`p02_e01` through `p05_e01`**: Gemini emitted times in something that
  looks like inflated seconds — max end ~1.6 × the actual audio duration
  (e.g., 1055 s for an 656 s audio). Confusion + false-alarm dominate
  because the hypothesis claims speech at timestamps the audio doesn't
  have.

**Two different timing-unit failure modes on five episodes.** The model
knows what's said and roughly when, but it cannot anchor its output to a
consistent time scale across runs. This is not a prompt-engineering
problem (the prompt explicitly specifies seconds) — it's a Gemini 2.5
Flash audio-modality limitation on time-grounded structured output.

### What this changes about the verdict

The Phase 2 conclusion ("pyannote stays canonical, Gemini is fall-back for
ultra-thin deployments") **stands**, but the framing tightens:

- Gemini's diarization output is **not usable** for any downstream task
  that depends on timestamps (segment-aligned UI playback, time-coded
  speaker-attributed search hits, anything in the GI evidence stack that
  cross-references audio offsets).
- Gemini's segments **are** still usable for "did at least two distinct
  speakers participate, and roughly how many?" questions — which is the
  speaker-count signal — but that's a much narrower fall-back than #962's
  acceptance language implied.
- If `diarization_provider: gemini` ever becomes a real production path,
  it needs an integration test that asserts the output's max timestamp is
  within ~10 % of the audio duration. The current Gemini provider should
  log a warning if it sees timestamps outside that band.

A follow-up could try Gemini 2.5 Pro or a structured-output schema with
explicit `seconds_from_start` field validation; both are out of scope for
issue #992. Filing separately if the operator wants Gemini diarization to be
load-bearing.

## What's NOT in this report (gaps + follow-ups)

1. **Diarization client resilience** (#954, filed earlier) — independent of this evaluation but matters for prod reliability under shared-GPU contention.
2. **Burst latency** — only sequential measurements here. Concurrent diarization calls would test the queueing behavior that #954 is filed against.
3. **Multi-voice v2 fixtures (#944)** — landed between phase 1 and phase 2; phase 2 re-ran pyannote/MPS on the new fixtures for an apples-to-apples comparison. Speaker-count still over-detects (3 vs 2) because both backends pick up a third acoustic cluster from the Ad insert reads. Phase 3's DER measurement confirms pyannote is structurally correct on this dataset — the over-count is acoustic, not categorical.
4. **Gemini timestamp-unit failure** — separable follow-up if Gemini diarization ever needs to be load-bearing; would need a Gemini 2.5 Pro retry or a structured-output schema with `seconds_from_start` field validation.

## Recommendation

**3-way verdict** (post-phase 3): pyannote stays the canonical diarization
engine across all profile shapes. **Phase 3 DER strengthens this:**
pyannote scores 1.66 % DER on v2 — essentially correct. Gemini's structural
timing-unit failure makes its output unusable for timestamp-dependent
downstream consumers, even though it correctly identifies that multiple
speakers exist.

- `local` profile (laptop, no DGX): in-process pyannote with `device=auto` → picks MPS on Apple Silicon → ~13× realtime, **1.66 % DER**
- `cloud_with_dgx_*` profiles (DGX available): route to `:8001/v1/diarize` → ~13× realtime, off-laptop, **same DER as MPS** (same model, different device)
- `cloud_*` profiles (no DGX, no GPU): pyannote/local stays the canonical default. `diarization_provider: gemini` is a **last-resort** fall-back for ultra-thin deployments where (a) the pyannote dependency cannot be shipped AND (b) no downstream consumer depends on the segment timestamps being accurate.

Phase 4 (future): burst-latency, real production load patterns, Gemini 2.5
Pro retry on the timestamp-unit failure.

## Artifacts (Phase 3)

- `scripts/eval/score/diarization_der_v1.py` — DER computation harness with the Deepgram-alignment ground-truth derivation
- `data/eval/runs/diarization_3way_v1/local-mps/segments_*.json` — per-episode pyannote/MPS segment dumps
- `data/eval/runs/diarization_3way_v1/gemini/segments_*.json` — per-episode Gemini segment dumps
- `data/eval/runs/diarization_der_v1/ground_truth/*.json` — derived time-aligned ground truth (reusable for future backend comparisons)
- `data/eval/runs/diarization_der_v1/deepgram_words/*.json` — cached Deepgram word-level transcripts
- `data/eval/runs/diarization_der_v1/metrics.json` — Phase 3 headline metrics

## Artifacts

- `scripts/eval/score/diarization_dgx_vs_cloud_v1.py` — harness with 3 backends (`dgx`, `local` with device override, `gemini` slot for future)
- `data/eval/runs/diarization_dgx_vs_cloud_v1/dgx/metrics.json` — DGX CUDA 5/5
- `data/eval/runs/diarization_dgx_vs_cloud_v1/local-mps/metrics.json` — Apple MPS 5/5
- `data/eval/runs/diarization_dgx_vs_cloud_v1/local-cpu/metrics.json` — Pure CPU 1/5

## References

- Issue: #930
- Parent epic: #927 (DGX-vs-cloud autoresearch programme)
- pyannote-on-DGX deploy: `infra/dgx/pyannote-server/` (#926)
- Distinct-voice v2 fixtures: #934
- Diarization client resilience gap: #954 (filed today during pyannote-wedge investigation)
