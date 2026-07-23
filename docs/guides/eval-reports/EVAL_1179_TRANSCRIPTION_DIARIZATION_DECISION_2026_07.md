# EVAL — #1179 transcription + diarization default-profile decision (2026-07)

Closes the #1179 "deathmatch" epic: the single default-profile ASR (transcription) and diarization
choice for the v2→v3 reprocess, folding the separate reports into one decision. This is the
**model lock** for the reprocess (reprocess-once economics — pick before the run, not after).

## Transcription (ASR)

**Decision:**

| vertical | primary | fallback / secondary | notes |
| --- | --- | --- | --- |
| **DGX (self-hosted)** | **`large-v3-turbo`** (speaches :8000, ~25× realtime) | large-v3 (ADR-123 coverage failover); MOSS (accurate-but-slow) | turbo picked on SPEED; MOSS is the accurate DGX option but 2.9× |
| **Cloud** | **`openai-whisper-1`** | — | best accuracy on real human GT; Deepgram is diarization-only |

**Evidence:** the first ASR bake-off scored against **real human verbatim transcripts** (not a
model-as-silver): 80,000 Hours n=10 (single-show) + Lex Fridman n=2 (cross-show). openai-whisper-1
best on both shows; large-v3 worst on both (→ **#1273** serving-anomaly follow-up). Full report:
[EVAL_ASR_5MODEL_BAKEOFF_2026_07](EVAL_ASR_5MODEL_BAKEOFF_2026_07.md). turbo's long-episode coverage
cliff is handled by the ADR-123 gate.

**Confidence — stated honestly:** the *decision* is locked; the *evidence* is thin (n=10 + n=2, MOSS
not cross-show-tested, #1273 open). turbo-primary is a SPEED call for the ~4-GPU-day 1000-ep run
(turbo ~0.8d vs large-v3 ~4d), not an accuracy claim.

## Diarization

**Decision: pyannote `community-1`** (DGX `:8001`), pyannote/local + Deepgram as fallbacks. In the
DGX profiles as `tailnet_dgx_diarization_community1`.

**Evidence:**
[EVAL_DIARIZATION_31_VS_COMMUNITY1_RTTM_2026_07](EVAL_DIARIZATION_31_VS_COMMUNITY1_RTTM_2026_07.md)
(community-1 vs pyannote 3.1 on RTTM truth) and
[EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06](EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md) (DGX vs cloud).
MOSS does transcribe+diarize in one pass but **lost diarization to pyannote on real audio**, so we
run turbo (transcription) + pyannote community-1 (diarization) rather than MOSS for both.

## Default-profile result (what the reprocess runs)

- **DGX reprocess** (`reprocess_dgx_turbo`): transcription = turbo (+ coverage gate → large-v3),
  diarization = pyannote community-1, `hold` resilience (ADR-122), governance on (ADR-124).
- **DGX serving** (`prod_dgx_*`): same transcription/diarization, serve/failover.
- **Cloud** (`cloud_*`): transcription = openai-whisper-1, diarization = Deepgram nova-3.

All materialized from the registry (SSOT) + drift-checked; `dgx_whisper_model` now governed so a
turbo profile can't silently run large-v3.

## Open follow-ups (do NOT block the reprocess model-lock)

- **#1273** — large-v3 speaches/int8 serving anomaly (it's beaten by its own turbo distillation) +
  revisit whether large-v3 is the right coverage-failover target.
- **MOSS cross-show** — MOSS was only measured on 80k (n=10), not Lex; its 2nd-place accuracy is
  single-show. Optional; the turbo-vs-MOSS decision is speed-driven regardless.

## Status

Transcription: decided + in the registry + pushed (`feat/1000-episodes`). Diarization: decided + in
the profiles. **#1179 can close** — the reprocess model-lock is set; remaining items are tracked
follow-ups (#1273), not gates.
