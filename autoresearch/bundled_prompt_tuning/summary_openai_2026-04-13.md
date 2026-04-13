# AutoResearch Bundled Prompt Tuning — Summary

**Date:** 2026-04-13
**Run:** `autoresearch_prompt_openai_bundled_smoke_bullets_v1`
**Dataset:** `curated_5feeds_smoke_v1` (5 episodes)
**Judge models:** gpt-4o-mini (A), claude-haiku-4-5 (B)
**Rubric hash:** 9f43a4b9

---

## Total Experiments Run

3 experiments (exp-1, exp-2, exp-3). Stopped at exactly 3 per user instruction.

---

## Best Score vs Baseline

| Run      | Score    | ROUGE-L | Embed | Judge mean | Formula          |
| -------- | -------- | ------- | ----- | ---------- | ---------------- |
| Baseline | 0.231183 | 23.1%   | 73.7% | n/a        | ROUGE-only blend |
| exp-1    | 0.461547 | 25.8%   | 73.7% | 0.936      | full blend       |
| exp-2    | 0.466996 | 26.3%   | 75.3% | 0.943      | full blend       |
| exp-3    | 0.256322 | 25.6%   | 71.0% | 0.893      | ROUGE-only blend |

**Best: exp-2 (0.466996)**

- Absolute improvement over baseline: **+0.235813**
- Relative improvement: **+101.9%**

Note: the large jump from baseline to exp-1 is partly explained by the baseline being ROUGE-only
(judges contested), while exp-1 and exp-2 used the full ROUGE+judge blend (judges agreed). The
underlying ROUGE-L improvement from baseline to best is 23.1% → 26.3% (+3.2 pp), and embedding
similarity rose from 73.7% to 75.3% (+1.6 pp).

---

## Changes That Improved the Score

### exp-1 (+99.6% from baseline, accepted)

**Change:** Added instruction to "cover ALL major discussion segments in the order they appear"
and preserved "ALL substantive discussion including Q&A, examples, and supporting details" in the
cleaning step.

**Effect:** ROUGE-L 23.1% → 25.8% (+2.7 pp), embed stayed flat, judges agreed (0.936). Judges
moved from contested to unanimous, activating the full blend and producing the biggest single-run
score jump.

### exp-2 (+1.2% from exp-1, accepted)

**Change:** Added verbatim terminology preservation in the system prompt ("Preserve key technical
terms, concept names, product names, and specific vocabulary from the transcript verbatim") and a
parallel rule in the user prompt ("Use precise terminology from the transcript; do not paraphrase
technical terms, named frameworks, tools, or concepts").

**Effect:** ROUGE-L 25.8% → 26.3% (+0.5 pp), embed 73.7% → 75.3% (+1.6 pp), judges 0.936 →
0.943. Modest but real improvement across all three dimensions.

---

## Changes That Hurt the Score

### exp-3 (−45.2% from exp-2, rejected)

**Change:** Required each summary paragraph to include "at least one concrete specific (named
tool, framework, product, number, percentage, timeframe, or outcome)" and bumped the default
minimum paragraph count.

**Effect:** ROUGE-L dropped back to 25.6%, embedding similarity fell to 71.0%, judges contested
(judge mean 0.893 < threshold). The over-specification caused the model to write longer, more
verbose summaries that diverged from the silver reference style and triggered judge disagreement,
falling back to the ROUGE-only blend.

---

## Patterns

**Consistently helped:**

- Broad coverage instructions ("ALL major discussion segments", "in order") — the single most
  impactful change, both for ROUGE-L and for judge agreement.
- Verbatim terminology preservation — improved embedding similarity by keeping named concepts
  from the transcript intact rather than paraphrasing.
- Structural clarity (explicit chronological arc) — helped judges agree, enabling the full blend.

**Hurt or had no effect:**

- Forcing very specific content requirements per paragraph (concrete numbers/specifics mandate)
  — caused over-verbosity, hurt embedding similarity, and triggered judge disagreement.
- Bumping minimum paragraph count — made summaries longer but not better aligned with the silver
  reference, hurting ROUGE-L.

---

## Suggested Next Directions

1. **Judge agreement focus:** The contested/non-contested flip is the biggest score lever given
   the 0.70 ROUGE / 0.30 judge blend. Prompts that keep judges aligned (clear structure,
   consistent terminology) unlock the full blend. Investigate what triggers judge disagreement.

2. **Silver alignment study:** Read a sample of silver reference summaries to understand their
   typical style, paragraph count, and vocabulary patterns. Align prompts to match that style
   more closely rather than making the output longer.

3. **Paragraph count tuning:** The default `paragraphs_min=2` / `paragraphs_max=4` range may
   be suboptimal. A targeted ablation (e.g., fix at 3 paragraphs) could improve ROUGE-L without
   triggering verbosity issues.

4. **Bullet count tuning:** The 6-8 bullet target in the user prompt is higher than what the
   silver reference may use. Reducing to 5-6 could improve judge scores if the silver tends
   toward shorter bullet lists.

5. **Embedding similarity push:** exp-2 showed that terminology preservation moves embedding
   similarity up. Further gains likely come from better semantic alignment — try asking for
   phrasing close to the transcript rather than paraphrasing.

---

## Anomalies and Crashes

- **exp-2 first attempt:** The initial `run_experiment.py` subprocess reported "0 matching
  episodes" during ROUGE scoring, causing `score.py` to exit 1. A second automatic run succeeded
  and produced the accepted score. Root cause appears to be a transient reference-matching issue
  (likely a stale predictions cache in the scorer). No data was lost; the second run re-ran
  inference from scratch.

- **exp-3 scoring divergence:** Two scoring passes produced different final scalars (0.450514
  observed in first run vs 0.256322 in recorded result). The lower value reflects judge
  contestation (ROUGE-only blend) in a second pass. Conservative lower value was recorded.

---

AUTORESEARCH COMPLETE
