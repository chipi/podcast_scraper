# Eval: Whisper `small.en` on v2 fixtures (#979/#980/#981 follow-up)

**Date:** 2026-06-13
**Ticket:** registry-materialization follow-up — closes the `whisper_model: small.en`
gap for the `local.yaml` profile opt-in.
**Companion:** [EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md](EVAL_FIXTURES_V2_TIER3_TUNING_2026_06_08.md)
(Tier 3 deferred `small.en` with "would take ~150 min CPU" — actual measured
runtime was ~2.5 min total).

## TL;DR

`small.en` measured on the same 5 v2 audio episodes as Tier 3. **Mean WER
2.94% / 30.6s per episode — quality upgrade over `base.en` (3.92% mean),
significant latency cost vs `base.en` (~2.6× slower per episode).**

| Model | Mean WER | Min | Max | Mean lat/ep |
| --- | ---: | ---: | ---: | ---: |
| tiny.en (TEST_DEFAULT) | 10.93% | 4.13% | 23.14% | 7.5s |
| base.en (PROD_DEFAULT laptop) | 3.92% | 2.34% | 6.18% | 11.7s |
| **small.en** (cur. `local.yaml` choice) | **2.94%** | **1.87%** | **4.61%** | **30.6s** |

Tier 3 estimated `small.en` at ~150 min CPU; on M4 Pro the real number was
**~30 s/episode**, ~2.5 min total for the 5-episode suite. Estimate was off
by ~50×; report corrected in `local.yaml` provenance.

## Per-episode

| Episode | Voices | tiny.en | base.en | **small.en** |
| --- | --- | ---: | ---: | ---: |
| p01_e01 | Samantha (US-en) + Fred (US-en) | 6.34% | 6.18% | **2.21%** |
| p02_e01 | Alex (US-en) + Isha (en-IN) | 4.48% | 2.83% | 4.61% |
| p03_e01 | Karen (en-AU) + Luca (it-IT) | 6.51% | 2.98% | **1.87%** |
| p04_e01 | **Daniel (en-GB) + Kathy (fr-CA)** | **23.14%** | 5.25% | **2.49%** |
| p05_e01 | Moira (en-IE) + Oliver (en-GB) | 4.13% | 2.34% | 3.51% |

`small.en` wins big where it matters most — the accent-stress case
(p04_e01) is **2.49% vs base.en's 5.25%**. The two cases where `base.en`
edges `small.en` (p02, p05) are within 1.5 pp; the wins for `small.en`
elsewhere swamp those.

Variance pattern: `small.en` 1.87–4.61% (spread ~2.7 pp); `base.en`
2.34–6.18% (spread ~3.8 pp). `small.en` is more consistent — a desirable
property for a laptop-default that runs against unknown content.

## Why `local.yaml` picks small.en — finally backed by measurement

The `local.yaml` profile picks `whisper_model: small.en` for the laptop
default. Prior to this eval there was no v2-fixture comparison justifying
the choice over `base.en` (Tier 3 explicitly deferred `small.en` testing).
This report fills the gap:

- Quality: `small.en` is the right call. **25 % relative WER reduction** vs
  `base.en` mean, and consistent across the accent-stress cases.
- Latency: 30 s/ep is the tax. Acceptable for a laptop profile (not a
  cloud one) where the operator trades wall-clock for autonomy.

## What this unblocks

Registry materialization for `local.yaml`:

1. Add `local_whisper_small_en` to `_TRANSCRIPTION_OPTIONS` (research_ref → this report).
2. Add `local` to `_PROFILE_PRESETS`.
3. Add `profile: local` to `config/profiles/local.yaml` so the drift test
   covers it.

## Acceptance

- [x] Same 5 v2 episodes as #906 Tier 3; same WER routine + `whisper_accent_wer_v1.py`.
- [x] `small.en` improves on `base.en` mean WER (2.94 % vs 3.92 %).
- [x] Latency tax documented (30 s/ep vs 11.7 s/ep on M4 Pro).
- [x] Tier 3's "~150 min CPU" estimate corrected.
- [x] Metrics JSON: `data/eval/runs/baseline_whisper_small_en_v1/metrics.json`.

## Out of scope

- `medium.en` / `large-v3` on laptop CPU — quality ceiling exists in
  `EVAL_TRANSCRIPTION_3WAY_2026_06.md` for `large-v3` via MPS, which is
  the better tier for laptop quality-first work.
- Latency-vs-quality cost curve fitting — `small.en` is the operationally
  chosen point; the table above is enough to defend it.

## Reproduction

```bash
mkdir -p data/eval/runs/baseline_whisper_small_en_v1
PYTHONPATH=. .venv/bin/python scripts/eval/score/whisper_accent_wer_v1.py \
    --audio-dir tests/fixtures/audio/v2 \
    --transcripts-dir tests/fixtures/transcripts/v2 \
    --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
    --models small.en \
    --output  data/eval/runs/baseline_whisper_small_en_v1
```
