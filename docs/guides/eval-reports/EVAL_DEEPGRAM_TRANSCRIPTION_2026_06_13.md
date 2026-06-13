# Eval: Deepgram nova-3 transcription on v2 fixtures (#979/#980/#981 follow-up)

**Date:** 2026-06-13
**Ticket:** registry-materialization follow-up — closes the
`transcription_provider: deepgram` gap for `cloud_quality` opt-in.
**Companion:** [EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md](EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md)
(Whisper tiny/base accent-WER baseline), [EVAL_TRANSCRIPTION_3WAY_2026_06.md](EVAL_TRANSCRIPTION_3WAY_2026_06.md) (cloud/MPS/DGX comparison).

## TL;DR

Deepgram nova-3 measured against the same 5 v2 audio episodes that #906
Tier 3 used. **Mean WER 2.48% / 1.2s per episode — best accuracy AND best
latency across every model we've measured on v2.**

| Model | Mean WER | Min | Max | Mean lat/ep | Approx cost/ep |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny.en (TEST_DEFAULT) | 10.93% | 4.13% | 23.14% | 7.5s | $0 (local) |
| base.en (PROD_DEFAULT laptop) | 3.92% | 2.34% | 6.18% | 11.7s | $0 (local) |
| **Deepgram nova-3** | **2.48%** | **1.52%** | **4.49%** | **1.2s** | ≈ $0.022 (≈5min audio @ $0.0043/min) |

Nova-3 wins on **every** episode against both Whisper tiers. Especially
strong on the accent-stress case (p04_e01: Daniel en-GB + Kathy fr-CA) —
**4.49%** vs base.en's 5.25% vs tiny.en's 23.14%.

This closes the missing-research-ref gap that blocked `cloud_quality`
profile opt-in to the registry (per `docs/wip/RESEARCH_POWERED_REGISTRY_PLAN.md`).

---

## Method

Mirror of `whisper_accent_wer_v1.py` exactly so headlines stay comparable:

- **Same 5 v2 episodes** used by #906 Tier 3: p01_e01, p02_e01, p03_e01,
  p04_e01, p05_e01. Same 9-voice accent coverage (US-en × 4, en-IN, en-AU,
  it-IT, en-GB × 2, fr-CA × 1, en-IE × 1).
- **Same ground truth** — v2 transcripts (generated FROM the audio via
  macOS `say` per RFC-059 §2).
- **Same WER routine + transcript normalisation** — reuses helpers from
  `whisper_accent_wer_v1.py` so a row in this report is directly
  substitutable for a row in the Tier 3 table.
- **Deepgram options**: `model=nova-3`, `smart_format=False`,
  `punctuate=True`, `language=en`. `smart_format=False` keeps the
  hypothesis text in the same shape as raw Whisper output (no
  paragraph/sentence reflow) so the WER comparison is apples-to-apples.

Script: `scripts/eval/score/deepgram_transcription_wer_v1.py`.

## Per-episode

| Episode | Voices | tiny.en | base.en | **nova-3** |
| --- | --- | ---: | ---: | ---: |
| p01_e01 | Samantha (US-en) + Fred (US-en) | 6.34% | 6.18% | **2.21%** |
| p02_e01 | Alex (US-en) + Isha (en-IN) | 4.48% | 2.83% | **2.17%** |
| p03_e01 | Karen (en-AU) + Luca (it-IT) | 6.51% | 2.98% | **2.01%** |
| p04_e01 | **Daniel (en-GB) + Kathy (fr-CA)** | **23.14%** | **5.25%** | **4.49%** |
| p05_e01 | Moira (en-IE) + Oliver (en-GB) | 4.13% | 2.34% | **1.52%** |

- Nova-3 wins **every** episode. Average gain over base.en: ~37% relative WER reduction.
- The accent-stress case (p04_e01) is where the spread is biggest — nova-3
  cuts another ~14% off base.en's already-strong number, and is ~5× lower
  than tiny.en.
- Latency: nova-3 is **~10× faster** than base.en on the same hardware
  budget (the request side, not the inference side — Deepgram does the
  inference on their servers). In a `cloud_quality` profile that already
  trades laptop battery for cloud spend, that's an unambiguous win.

## Cost

Deepgram nova-3 is **$0.0043/min** (pre-pay) per the 2026 pricing sheet.
The 5-episode v2 sample is ~30 min total audio → ~**$0.13** for the eval
run. For a 1-hour podcast episode (typical production), nova-3 ≈ **$0.26
per episode**, vs OpenAI Whisper-1 at $0.006/min ≈ $0.36/hour episode.
Within 30% of Whisper API cost, well below the quality-gap warranted by
the WER difference.

## What this unblocks

The `cloud_quality.yaml` profile picks `transcription_provider: deepgram`
with `deepgram_model: nova-3` as its transcription default. Previously the
registry had no `_TRANSCRIPTION_OPTIONS` entry for Deepgram because no
eval justified the choice — Deepgram diarization was in
`EVAL_DIARIZATION_DGX_VS_CLOUD_2026_06.md` but transcription wasn't.
This report fills that gap.

Next steps (separate work):

1. Add `deepgram_nova_3` to `_TRANSCRIPTION_OPTIONS` (registry materialization).
2. Add `cloud_quality` to `_PROFILE_PRESETS`.
3. Add `profile: cloud_quality` to `config/profiles/cloud_quality.yaml`
   so the drift test covers it.

## Acceptance

- [x] Same 5 v2 episodes as #906 Tier 3; same WER routine.
- [x] Nova-3 outperforms both tiny.en and base.en on every episode.
- [x] Deepgram cost order-of-magnitude documented.
- [x] Script committed (`scripts/eval/score/deepgram_transcription_wer_v1.py`).
- [x] Metrics JSON: `data/eval/runs/baseline_deepgram_transcription_wer_v1/metrics.json`.

## Out of scope

- Deepgram nova-2 / nova-1 vs nova-3 — newer is monotonically better per
  Deepgram's own benchmarks; not worth measuring laptop-side without a
  cost or accuracy signal pointing back to the older models.
- Cross-language coverage — v2 fixtures are en-dominant; non-en performance
  was sampled in #906 Tier 3 (Italian, Spanish, French Canadian guest voices)
  and remains the structural reason base.en spikes on p04.
- Latency-with-network-failure variance — deferred to the `#956` DGX-over-
  Tailscale resilience work-stream, where retry behaviour for every cloud
  transcriber gets shared treatment.

## Reproduction

```bash
mkdir -p data/eval/runs/baseline_deepgram_transcription_wer_v1
export $(grep -E '^DEEPGRAM_API_KEY=' .env)
PYTHONPATH=. .venv/bin/python scripts/eval/score/deepgram_transcription_wer_v1.py \
    --audio-dir tests/fixtures/audio/v2 \
    --transcripts-dir tests/fixtures/transcripts/v2 \
    --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
    --model nova-3 \
    --output data/eval/runs/baseline_deepgram_transcription_wer_v1
```
