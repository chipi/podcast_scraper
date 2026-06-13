# EVAL — Diarization championship (3-way: MPS / CUDA / Gemini)

Phase 1 — 2026-06-10 — MPS / CUDA / CPU.
Phase 2 — 2026-06-13 — Gemini 2.5 audio API added (#962); fresh
pyannote/MPS re-run on the post-#944 multi-voice v2 fixtures.

**Issue:** #930, #962
**Branch:** `feat/autoresearch-batch-3-championships` (Phase 1); `main` (Phase 2)
**Dataset:** v2 audio fixtures, 5 episodes (RFC-059 §2 macOS `say` generation)
**Status:** **3-way panel COMPLETE** as of 2026-06-13. Gemini speaker detection
re-measured on the current multi-voice v2 fixtures (#944) alongside a fresh
pyannote/MPS run for apples-to-apples comparison.

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

## What's NOT in this report (gaps + follow-ups)

1. **Proper DER** (Diarization Error Rate) — requires time-aligned speaker ground truth, which v2 fixtures don't ship. Could be derived from whisper word-level timestamps as an alignment proxy. Filed for follow-up.
2. **Diarization client resilience** (#954, filed today) — independent of this evaluation but matters for prod reliability under shared-GPU contention.
3. **Burst latency** — only sequential measurements here. Concurrent diarization calls would test the queueing behavior that #954 is filed against.
4. **Multi-voice v2 fixtures (#944)** — landed between phase 1 and phase 2; phase 2 re-ran pyannote/MPS on the new fixtures for an apples-to-apples comparison. Speaker-count still over-detects (3 vs 2) because both backends pick up a third acoustic cluster from the Ad insert reads. Real DER (item 1) is the next signal-strength bump.

## Recommendation

**3-way verdict** (post-phase 2): pyannote stays the canonical diarization
engine across all profile shapes. Gemini is now a wired fall-back for
cloud-only profiles that want to skip the pyannote install:

- `local` profile (laptop, no DGX): in-process pyannote with `device=auto` → picks MPS on Apple Silicon → ~13× realtime
- `cloud_with_dgx_*` profiles (DGX available): route to `:8001/v1/diarize` → ~13× realtime, off-laptop
- `cloud_*` profiles (no DGX, no GPU): pyannote/local stays the canonical default; `diarization_provider: gemini` is the **explicit fall-back** for ultra-thin deployments that don't ship the pyannote dependency (over-segments ~60% vs pyannote and costs ~$0.03/episode, but does work end-to-end).

Phase 3 (when the gaps above close): real DER measurement, burst-latency.

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
