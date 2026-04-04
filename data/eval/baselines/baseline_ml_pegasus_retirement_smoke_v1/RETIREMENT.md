# Pegasus+LED Retirement — baseline_ml_pegasus_retirement_smoke_v1

**Date:** 2026-04-03
**Decision:** Retire `ml_prod_authority_v1` (Pegasus-CNN + LED-base) as ML prod baseline
**Successor:** `baseline_ml_longt5_led_smoke_v1` (LongT5+LED)

## Evidence

| Config | avg tokens | ROUGE-L | Embed | Notes |
| --- | --- | --- | --- | --- |
| baseline ml prod authority smoke v1 | 54.2 | 10.7% | 52.0% | Original prod, vs silver gpt4o |
| baseline ml prod authority smoke v2 | ~71 | - | - | Relaxed constraints, still broken |
| baseline ml dev authority (BART+LED) | 217.2 | 18.1% | 72.4% | Dev model, better than prod |

## Root Cause

**Layer 1 — Wrong pretraining domain:** Pegasus was trained on CNN/DailyMail news
with Gap Sentence Generation (GSG). GSG selects "most important" sentences based on
news article structure: dense, non-repetitive, one topic per sentence. Podcast
transcripts are the opposite — conversational, repetitive, one idea stretched over
many sentences with filler. Pegasus reads a podcast chunk and extracts the most
repeated phrases as "key sentences", producing near-identical summaries for every
chunk.

**Layer 2 — LED ngram exhaustion:** LED reduce concatenates all chunk summaries as
input. With `no_repeat_ngram_size=3`, LED tracks every 3-gram it has generated and
refuses to repeat it. When the input is dense with repeated phrases (from layer 1),
LED exhausts its entire 3-gram budget within ~55-70 tokens and stops. This is LED
behaving correctly — it cannot generate non-redundant text from redundant input.

**Layer 3 — Params cannot fix this:** Setting `no_repeat_ngram_size=6` or higher
only slightly delays the exhaustion (55 → 71 tokens). Setting it to 0 produces
longer output but full of near-duplicate sentences. The problem is in the map model,
not the reduce model. No combination of LED params produces quality output from
Pegasus-generated chunk summaries on podcast content.

## Why Not Deleted

Pegasus is a strong model for its intended domain (news). When news is added as a
content type, Pegasus should be re-evaluated — it may be the right map model there.
The `ml_prod_authority_v1` mode is kept in the registry with `deprecated_at` and
`deprecation_reason` fields rather than removed (registry is append-only).

## What Was Tried

- `baseline_ml_prod_authority_smoke_v1`: original prod params
- `baseline_ml_prod_authority_smoke_v2`: relaxed `no_repeat_ngram_size=6`,
  `length_penalty=1.2`, `early_stopping=true` — output improved 55 → 71 tokens,
  root cause unchanged
- `param_space.yaml` pegasus\_led section was drafted but never executed —
  param tuning was correctly ruled out after root cause analysis
