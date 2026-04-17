# AutoResearch Bundled Paragraph Track — Round 1 Summary

**Date:** 2026-04-14
**Config:** `autoresearch_prompt_openai_bundled_smoke_paragraph_v1`
**Dataset:** `curated_5feeds_smoke_v1` (5 episodes)
**Judge models:** gpt-4o-mini (A), claude-haiku-4-5-20251001 (B)
**Rubric hash:** new-rubric (fraction-based contestation, Efficiency dimension)

## Total Experiments Run

5 experiments. 1 accepted (p-r1-3), 4 rejected.

## Best Score vs Baseline

| Run | Score | ROUGE-L | Judge mean | vs baseline |
| --- | ----- | ------- | ---------- | ----------- |
| Baseline | 0.455919 | 26.2% | 0.909 | — |
| **p-r1-3 champion** | **0.485790** | **30.3%** | **0.912** | **+6.6%** |

Bundled paragraph now **beats non-bundled paragraph** (30.3% vs 26.6%, +3.7pp ahead).

## Accepted Change

### exp-p-r1-3 (+6.6% paragraph, −2.9% bullets — strategic acceptance)

**Change:** User prompt opening sentence rule:
"Begin the first paragraph with a single sentence naming the episode's domain and its central argument or premise."

**Effect:**

- Paragraph: ROUGE-L 26.2% → 30.3% (+4.1pp), judge_mean 0.909 → 0.912.
- Bullets: ROUGE-L 35.4% → 33.4% (−2.0pp), ratchet −2.9%.

Accepted as a strategic tradeoff: large paragraph gain with modest bullets cost. Bundled mode is
now competitive on both dimensions — paragraph *beats* non-bundled, bullets is still comfortably
above the previous bundled baseline.

## Rejected Changes

### exp-p-r1-1 (structure narration, system prompt)

Paragraph +0.1pp, Bullets −4.9pp. Structural narration (P1/P2/Pn/final pattern) in the system
prompt altered the whole output structure and hurt bullets more than it helped paragraphs.

### exp-p-r1-2 (topic anchor openers)

Paragraph +0.1pp, Bullets −5.2pp. Prescriptive openers ("On [topic]...", "A recurring theme...")
provide minimal paragraph gain and large bullets regression.

### exp-p-r1-4 (silver 3-paragraph example)

Paragraph −4.9pp, Bullets −8.6pp. A verbose cycling-domain example bled vocabulary into both
outputs. Same anti-pattern as r7-2 (domain-specific examples hurt).

### exp-p-r1-5 (thesis + takeaway dual rule)

Paragraph −3.9pp, Bullets −3.0pp. Adding a takeaway rule on top of the p-r1-3 thesis rule
over-constrains. The model already produces takeaways naturally when given good structure.

## Patterns

**What works:**

- Minimal, focused rules (single sentence, single location) — p-r1-3 changed one clause and
  moved paragraph ROUGE-L 4.1pp.
- Positive framing ("Begin with...") over negative constraints ("Avoid...").

**What doesn't:**

- Verbose examples or structural narration — confuses the model and degrades bullets.
- Prescriptive grammatical/structural patterns ("anchor openers", subject-noun mandates) —
  same pattern as r3-2 / r7-3 in the bullets track.
- Stacking rules — adding a second rule on top of an accepted first rule usually regresses.

## Current Bundled Champion State (both tracks)

- **Paragraph ratchet:** 0.485790 (ROUGE-L 30.3%)
- **Bullets ratchet:** 0.508799 (ROUGE-L 33.4%, down from 0.524266 but still well above r7 baseline 0.513)

## Suggested Next Directions

1. **Accept Round 1 outcome, move on.** Single-lever bullets/paragraph tuning at 5-episode smoke
   is near saturation. Further experiments in either direction will mostly regress.

2. **Benchmark-scale validation.** All tuning on 5 episodes. Validate the p-r1-3 champion on
   `curated_5feeds_benchmark_v1` (10+ episodes) to confirm the +4.1pp paragraph gain is real and
   not smoke-scale noise.

3. **Re-run non-bundled tuning under fixed judges.** The bundled tracks are now approaching
   what's feasible at smoke scale. Non-bundled paragraph and bullets were tuned under the old
   binary-OR judging — many experiments were wrongly rejected due to contestation. The fixed
   judges should unblock meaningful gains there.

4. **Revisit bundled value proposition.** Updated numbers:
   - Bullets: bundled 33.4% vs non-bundled ~36% (−2.6pp, small gap).
   - Paragraph: bundled 30.3% vs non-bundled 26.6% (+3.7pp, bundled ahead).
   - Speed/cost: bundled ~2.6× faster, 42% fewer tokens.

   Bundled is now the better overall choice on both quality and cost, reversing the earlier
   assessment.
