# Eval: cleaning_v3 vs cleaning_v4 — broader 15-episode judge (#989)

**Date:** 2026-06-13
**Ticket:** [#989](https://github.com/chipi/podcast_scraper/issues/989)
**Companion:** [EVAL_FIXTURES_V2_TIER2_TUNING_2026_06_08.md](EVAL_FIXTURES_V2_TIER2_TUNING_2026_06_08.md)
(#905 Tier 2; 5-episode sample that surfaced the v3 preference but flagged
"broader judge pass" as the gate before flipping production)

## TL;DR

**v3 wins 15 / 15 episodes** under a position-bias-neutralised Sonnet 4.6
pairwise judge across the full v2 fixture set (`p[1-5]_e[1-3]`). Both A/B
orderings agree on every episode — there is no positional bias and no
ambiguity. The #989 "≥ 60 % v3 wins" gate passes by a 40 pp margin.

Production default is flipped from `cleaning_v4` to `cleaning_v3` in the
four hardcoded operational fallbacks. Historical `ModeConfiguration`
entries in `model_registry.py` (promoted between 2026-02 and 2026-04) are
**left at `cleaning_v4`** — they record what was promoted at that time and
are immutable per the materialize-decisions discipline; a future promoted
mode pointing at `cleaning_v3` would be a new `ModeConfiguration` entry
with a new `mode_id` + `promoted_at` timestamp.

## Method

For each of 15 v2 episodes:

1. Read raw transcript source from
   `data/eval/sources/curated_5feeds_raw_v2/feed-<feed>/<episode>.txt`.
2. Apply `cleaning_v3` and `cleaning_v4` from the registered
   `preprocessing.profiles`.
3. Ask Sonnet 4.6 to pairwise-judge A vs B (anonymised) twice — once with
   A = v3 and B = v4, once with the labels swapped. Same rubric #905 used:
   removed sponsor / ad / intro / outro / meta-commentary, preserved
   substantive content, kept speaker labels, no invention or paraphrase.
4. **Consensus rule**: if both orderings agree, that's the verdict. If
   they disagree, the episode is tagged `TIE_POSITIONAL`.

Position-swapping is the key methodology delta from #905's 5-episode run.
It catches the "Sonnet always picks A" failure mode that single-order
sampling would miss.

## Cost / time

30 Sonnet 4.6 calls (15 episodes × 2 orderings) at ~$0.02/call ≈ **$0.60**.
Sequential wall-clock 4 minutes (with 0.5 s between calls for rate
politeness).

## Results

```text
  p01_e01: order1= v3 order2= v3 consensus=v3
  p01_e02: order1= v3 order2= v3 consensus=v3
  p01_e03: order1= v3 order2= v3 consensus=v3
  p02_e01: order1= v3 order2= v3 consensus=v3
  p02_e02: order1= v3 order2= v3 consensus=v3
  p02_e03: order1= v3 order2= v3 consensus=v3
  p03_e01: order1= v3 order2= v3 consensus=v3
  p03_e02: order1= v3 order2= v3 consensus=v3
  p03_e03: order1= v3 order2= v3 consensus=v3
  p04_e01: order1= v3 order2= v3 consensus=v3
  p04_e02: order1= v3 order2= v3 consensus=v3
  p04_e03: order1= v3 order2= v3 consensus=v3
  p05_e01: order1= v3 order2= v3 consensus=v3
  p05_e02: order1= v3 order2= v3 consensus=v3
  p05_e03: order1= v3 order2= v3 consensus=v3

v3 wins:  15/15 (100.0%)
v4 wins:   0/15 (0.0%)
ties:      0/15 (0.0%)
60% gate: PASS — production flip to v3 is indicated
```

This is meaningfully stronger than #905's 10 W / 0 L / 5 T on the smaller
sample. The 5 ties in #905 collapse to v3 wins when position-bias is
controlled — the ties weren't genuine ties, they were the same Sonnet
verdict reading differently depending on which slot v3 occupied.

## What was flipped

The four hardcoded operational fallbacks:

1. `src/podcast_scraper/preprocessing/profiles.py:417` — `DEFAULT_PROFILE`
2. `src/podcast_scraper/providers/ml/summarizer.py:2143` — function arg default
3. `src/podcast_scraper/providers/ml/ml_provider.py:1382` — priority-chain fallback
4. `src/podcast_scraper/providers/ml/hybrid_ml_provider.py:454` — priority-chain fallback

All four now read `"cleaning_v3"`. Inline comments cite #989 + this report.

## What was NOT flipped (intentional)

The 5 `ModeConfiguration` entries in `model_registry.py` that pin
`preprocessing_profile="cleaning_v4"` — these are historical promotions
(`ml_small_authority`, `ml_prod_authority_v1`, `ml_bart_led_autoresearch_v1`,
two hybrid variants) with `promoted_at` timestamps between 2026-02-12 and
2026-04-04. They record what configuration produced specific baseline metrics
at the time of promotion. Flipping retroactively would misrepresent the
record. The discipline equivalent of the recently-shipped registry
materialization principle: registry entries are frozen at their
`promoted_at` timestamp; a new winning configuration becomes a new entry,
not a retroactive edit.

If a future autoresearch finding promotes a `cleaning_v3`-based mode,
that's a new `ModeConfiguration` with a fresh `mode_id`.

## Acceptance

- [x] 15-episode sample run across v2 fixtures
- [x] Position-bias-neutralised (every episode judged with A=v3 and A=v4)
- [x] Pass the ≥ 60 % gate (100 % achieved)
- [x] Default flipped in 4 operational fallbacks
- [x] Historical `ModeConfiguration` entries preserved with explicit rationale

## Reproduction

```bash
mkdir -p data/eval/runs/cleaning_v3_vs_v4_broader_judge_v1
export $(grep -E '^ANTHROPIC_API_KEY=' .env)
PYTHONPATH=. .venv/bin/python \
    scripts/eval/score/cleaning_v3_vs_v4_broader_judge_v1.py \
    --sources data/eval/sources/curated_5feeds_raw_v2 \
    --output  data/eval/runs/cleaning_v3_vs_v4_broader_judge_v1
```
