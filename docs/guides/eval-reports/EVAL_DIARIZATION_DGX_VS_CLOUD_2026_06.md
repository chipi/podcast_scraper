# EVAL — Diarization championship phase 1 (3-way: MPS / CUDA / CPU), 2026-06-10

**Issue:** #930
**Branch:** `feat/autoresearch-batch-3-championships`
**Dataset:** v2 audio fixtures, 5 episodes (RFC-059 §2 macOS `say` generation)
**Status:** Phase 1 complete (3 GPU/CPU backends measured). Gemini speaker-detection comparison is a follow-up — no Gemini speech provider in repo yet.

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

## What's NOT in this report (gaps + follow-ups)

1. **Gemini speech speaker detection** — no Gemini speech provider in this repo. `cloud_balanced.yaml` references it but the provider class needs wiring. Filed as part of the broader cloud-provider-pluggability work (no separate ticket yet).
2. **Proper DER** (Diarization Error Rate) — requires time-aligned speaker ground truth, which v2 fixtures don't ship. Could be derived from whisper word-level timestamps as an alignment proxy. Filed for follow-up.
3. **Distinct voices in v2 fixtures** (#934) — until this lands, speaker-count is the wrong signal. Documented above.
4. **Diarization client resilience** (#954, filed today) — independent of this evaluation but matters for prod reliability under shared-GPU contention.
5. **Burst latency** — only sequential measurements here. Concurrent diarization calls would test the queueing behavior that #954 is filed against.

## Recommendation

**Phase 1 verdict**: keep pyannote as the diarization engine across all profile shapes. Device selection picks itself:

- `local` profile (laptop, no DGX): in-process pyannote with `device=auto` → picks MPS on Apple Silicon → ~13× realtime
- `cloud_with_dgx_*` profiles (DGX available): route to `:8001/v1/diarize` → ~13× realtime, off-laptop
- `cloud_*` profiles (no DGX, no GPU): currently `speaker_detector_provider: gemini` — keep until either (a) Gemini speech is wired and benchmarked, or (b) the operator decides cost > convenience and migrates to a cloud pyannote service

Phase 2 (when the gaps above close): real DER measurement vs Gemini speaker detection, burst-latency, fixture quality once #934 voices land.

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
