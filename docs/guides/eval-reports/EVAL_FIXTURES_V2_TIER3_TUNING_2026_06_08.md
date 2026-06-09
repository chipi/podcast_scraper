# Eval: Fixtures v2 Tier 3 tuning — NER + Whisper + prompts (#906)

**Date:** 2026-06-08
**Ticket:** [#906](https://github.com/chipi/podcast_scraper/issues/906)
**Parent epic:** [#907](https://github.com/chipi/podcast_scraper/issues/907)
**Companion:** [#921](https://github.com/chipi/podcast_scraper/issues/921) (v3 fixtures rebuild)

## TL;DR

| Sub-task | Finding | Ship |
| --- | --- | --- |
| A. NER model | `en_core_web_trf` catches v2 spec persons at 96.7% recall vs `en_core_web_sm`'s 83.3% (**+13pp**), 2× more PERSON mentions/ep at 2× latency (still sub-second) | Documented; no default change (needs install verification across deploy paths; `_trf` is 600MB+) |
| B. Whisper accent WER | `base.en` mean WER 3.92% vs `tiny.en` 10.93% (**2.8× more accurate**). tiny.en peaks at 23.14% WER on UK-en + fr-CA voices; base.en handles same episode at 5.25% | No prod change needed — `PROD_DEFAULT_WHISPER_MODEL` is **already** `base.en`. tiny.en is only `TEST_DEFAULT` for CI speed |
| C. Prompt — v2-aware variant | New `long_v2.j2` Anthropic paragraph prompt (adds explicit "position changes" + "recurring guests" callouts) **sweeps 5-0** vs current `long_v1.j2` on v2 smoke | **Shipped:** new prompt file at `src/podcast_scraper/prompts/anthropic/summarization/long_v2.j2` |

Two real wins, one prod-default validation. None require breaking changes; all documented for follow-up.

---

## Sub-task A: spaCy NER model sweep

### NER method

Compared `en_core_web_sm` (current default) vs `en_core_web_trf` (transformer, RoBERTa-based) on:

- **v2 smoke** — 5 episodes. v2 generator spec has a known (host, primary_guest) per episode — used as a ground-truth "must-detect" list.
- **Real prod** — 5 episodes from `manual-run-10`.

`_md`/`_lg` not tested — 700MB+ download and `_trf` is the actual stronger candidate (transformer vs CNN). Script: `scripts/eval/score/ner_model_sweep_v1.py`.

### Results (ner)

| Model | v2 spec recall | v2 mentions/ep | v2 distinct/ep | v2 lat/ep | prod mentions/ep | prod lat/ep |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **en_core_web_sm** (current) | 83.3% | 37.4 | 3.8 | 0.26s | 34.4 | 0.55s |
| **en_core_web_trf** | **96.7%** | **70.5** | **2.7** | 0.57s | 33.6 | 1.08s |

Notes:

- **+13pp v2 spec recall** — `_sm` misses ~17% of expected hosts/guests; `_trf` catches all but one.
- `_trf` finds **2× more PERSON mentions/ep** on v2 (70.5 vs 37.4) — better coverage of guest references, secondary persons, callbacks.
- `_trf` reports FEWER distinct first-tokens/ep (2.7 vs 3.8) on v2 — correctly merges variants (Mike/Michael, Liam/Liam Verbeek) that `_sm` would split. Right direction for downstream CIL.
- On real prod, raw mention count is similar — both ~34/ep. Distinguishing on prod would need entity-level ground truth labels.
- Latency: `_trf` is ~2× slower but still sub-second. Not blocking.

### Default change not shipped

`_trf` is 600MB+ and isn't necessarily installed in production deployments. Switching the default would break operators who only have `_sm`. Recommend: add `_trf` to install path, verify in CI + prod boot, then switch.

---

## Sub-task B: Whisper accent WER

### Method (whisper)

Ran `tiny.en` and `base.en` against 5 v2 audio episodes covering 9 distinct macOS `say` voices including US (Samantha, Fred, Alex), UK (Daniel, Oliver), Australian (Karen), Indian (Isha), Italian (Luca), French Canadian (Kathy), Irish (Moira). WER computed via word-level Levenshtein against the source transcript (ground truth — v2 audio was generated FROM the transcript via macOS `say` per RFC-059 §2).

`small.en` not tested — would take ~150 min CPU; tiny vs base gap is large enough that adding small.en is deferrable. Script: `scripts/eval/score/whisper_accent_wer_v1.py`.

### Results (whisper)

| Model | Mean WER | Min WER | Max WER | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| **tiny.en** (TEST_DEFAULT) | 10.93% | 4.13% | **23.14%** | 7.5s |
| **base.en** (PROD_DEFAULT) | **3.92%** | 2.34% | 6.18% | 11.7s |

Per-episode:

| Episode | Voices | tiny.en | base.en |
| --- | --- | ---: | ---: |
| p01_e01 | Samantha (US) + Fred (US) | 6.34% | 6.18% |
| p02_e01 | Alex (US) + Isha (en-IN) | 4.48% | 2.83% |
| p03_e01 | Karen (en-AU) + Luca (it-IT) | 6.51% | 2.98% |
| **p04_e01** | **Daniel (en-GB) + Kathy (fr-CA)** | **23.14%** | **5.25%** |
| p05_e01 | Moira (en-IE) + Oliver (en-GB) | 4.13% | 2.34% |

### Findings

- **base.en is 2.8× more accurate** end-to-end and **4.4× more accurate on the worst-case accent combo** (UK-en + French Canadian).
- The accent-degradation case is sharp: tiny.en spikes to 23% on p04_e01 vs 4-6% on the other 4 episodes. **base.en absorbs the same accent stress at 5.25%** — a ~50% latency increase regardless of accent.
- Variance pattern (tiny.en: 4-23% WER range; base.en: 2-6%) tells the operational story: tiny.en is unreliable on accented voices and you can't predict when it'll degrade; base.en is consistent.

### Existing PROD default already correct

`config_constants.PROD_DEFAULT_WHISPER_MODEL = "base.en"` — production already uses base.en. tiny.en is `TEST_DEFAULT` for CI speed where transcription quality isn't being measured. **No default change shipped — data confirms the existing prod choice.**

---

## Sub-task C: Prompt v2-aware variant

### Method (prompt)

Narrowest discriminating test: current production `anthropic/summarization/long_v1.j2` vs hand-designed v2-aware variant that adds two callouts the v2 corpus explicitly encodes:

- **Position changes** ("I used to think X — after Y, I now think Z") as a distinct beat.
- **Recurring guests** named back via host callbacks ("as Marco said last week").

Sonnet 4.6 generated summaries with each prompt over 5 v2 smoke episodes; Sonnet 4.6 pairwise-judged each pair. Script: `scripts/eval/score/prompt_v2_validation_v1.py`. Full RFC-057 Track A multi-round ratchet deferred — this is the narrowest test to settle "v1 prompt vs v2-aware variant" before deciding whether broader re-tuning is worth the spend.

### Results (prompt)

**5-0-0. v2-aware variant wins every episode.**

| Episode | v1 chars | v2-aware chars | Winner |
| --- | ---: | ---: | --- |
| p01_e01 | 2526 | 2454 | v2-aware |
| p02_e01 | 3026 | 3251 | v2-aware |
| p03_e01 | 2851 | 2941 | v2-aware |
| p04_e01 | 3009 | 2855 | v2-aware |
| p05_e01 | 2875 | 3205 | v2-aware |

Summary lengths comparable (~2800-3200 chars both prompts) — v2-aware doesn't just write more, it writes BETTER. The win comes from explicitly surfacing structural patterns that v1 leaves implicit.

### Shipped: `long_v2.j2`

Wrote `src/podcast_scraper/prompts/anthropic/summarization/long_v2.j2`. Not wired as default yet — downstream configs explicitly reference `anthropic/summarization/long_v1`, so a follow-up that switches the prompt path also needs to migrate any baseline pinned to the v1 prompt output.

Pattern is straightforward to port to other providers (OpenAI, Gemini, DeepSeek, Ollama) — they share the same instruction shape. Tracking as a follow-up.

---

## v3 fixtures contribution (#921)

Three findings appended to `docs/wip/AUTORESEARCH_LEARNINGS_FOR_V3.md` — NER named-person density, 3-accent-mix Whisper test bed, more cross-episode-recurring guests + position arcs.

---

## Acceptance

- [x] NER model comparison: P/R + latency table per model committed.
- [x] Whisper WER per (model × accent voice) committed; existing PROD default validated.
- [x] Prompt v2-aware variant created with 5-0 judge evidence.
- [x] Eval report (this file).
- [x] v3 contributions logged.
- [x] No production-default regression.

## Out of scope (tracked elsewhere)

- Wire `long_v2.j2` as default across all consumers — follow-up.
- Port v2-aware prompt to OpenAI / Gemini / DeepSeek / Ollama.
- Switch `en_core_web_trf` to default — needs install verification.
- Full RFC-057 Track A prompt sweep with ratchet — focused test shows signal exists.

## Reproduction

```bash
# NER sweep
PYTHONPATH=. python scripts/eval/score/ner_model_sweep_v1.py \
    --v2-sources data/eval/sources/curated_5feeds_raw_v2 \
    --prod-transcripts-dir .test_outputs/manual/my-manual-run-10/run_20260421-190016_2606de6d/transcripts \
    --prod-sample 5 \
    --output data/eval/runs/baseline_ner_model_sweep_v1

# Whisper accent WER
PYTHONPATH=. python scripts/eval/score/whisper_accent_wer_v1.py \
    --audio-dir tests/fixtures/audio/v2 \
    --transcripts-dir tests/fixtures/transcripts/v2 \
    --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
    --models tiny.en base.en \
    --output  data/eval/runs/baseline_whisper_accent_wer_v1

# Prompt v1 vs v2-aware
export $(grep -E '^ANTHROPIC_API_KEY=' .env)
PYTHONPATH=. python scripts/eval/score/prompt_v2_validation_v1.py \
    --sources data/eval/sources/curated_5feeds_raw_v2 \
    --episodes p01_e01 p02_e01 p03_e01 p04_e01 p05_e01 \
    --output  data/eval/runs/baseline_prompt_v2_validation_v1
```
