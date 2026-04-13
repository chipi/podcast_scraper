# AutoResearch Bundled Prompt Tuning — Summary

**Date:** 2026-04-13
**Run:** `autoresearch_prompt_openai_bundled_smoke_bullets_v1`
**Dataset:** `curated_5feeds_smoke_v1` (5 episodes)
**Judge models:** gpt-4o-mini (A), claude-haiku-4-5-20251001 (B)
**Rubric hash:** 9f43a4b9

---

## Total Experiments Run

9 experiments total (exp-1 through exp-4 in r1; exp-r2-1 through exp-r2-3 in r2, early-stopped after 3
consecutive failures). r2 used corrected judges (claude-haiku-4-5-20251001) after r1 judge issue resolved.

---

## Best Score vs Baseline

| Run       | Score    | ROUGE-L | Embed | Judge mean | Formula          |
| --------- | -------- | ------- | ----- | ---------- | ---------------- |
| Baseline  | 0.231183 | 23.1%   | 73.7% | n/a        | ROUGE-only blend |
| exp-1     | 0.461547 | 25.8%   | 73.7% | 0.936      | full blend       |
| exp-2     | 0.466996 | 26.3%   | 75.3% | 0.943      | full blend       |
| exp-3     | 0.256322 | 25.6%   | 71.0% | 0.893      | ROUGE-only blend |
| exp-4     | 0.474687 | 27.3%   | 76.7% | 0.946      | full blend       |
| exp-r2-1  | 0.458081 | 25.3%   | 77.2% | 0.936      | full blend       |
| exp-r2-2  | 0.254719 | 25.5%   | 75.3% | 0.893      | ROUGE-only blend |
| exp-r2-3  | 0.282039 | 28.2%   | 72.5% | 0.933      | ROUGE-only blend |

**Best: exp-4 (0.474687) — champion unchanged after r2**

- Absolute improvement over baseline: **+0.243504**
- Relative improvement: **+105.3%**

Note: the large jump from baseline to exp-1 is partly explained by the baseline being ROUGE-only
(judges contested), while accepted experiments used the full ROUGE+judge blend (judges agreed). The
underlying ROUGE-L improvement from baseline to best is 23.1% → 27.3% (+4.2 pp), and embedding
similarity rose from 73.7% to 76.7% (+3.0 pp).

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

### exp-4 (+1.6% from exp-2, accepted)

**Change:** Replaced the verbose chronological-arc paragraph instruction with a tighter conciseness
rule: capped summary at 2–3 paragraphs (down from 2–4) and added an explicit "no filler or
repetition" directive in both system and user prompts.

**Effect:** Average tokens fell from ~592 → 474, ROUGE-L improved 26.3% → 27.3% (+1.0 pp), embed
75.3% → 76.7% (+1.4 pp), judge_mean 0.943 → 0.946. Judges remained in agreement; tighter outputs
better matched the silver reference style.

---

## Changes That Hurt the Score

### exp-3 (−45.2% from exp-2, rejected)

**Change:** Added verbose "follow the episode's chronological arc paragraph by paragraph" language
and required each paragraph to anchor to "a named tool, framework, product, number, percentage,
timeframe, or outcome."

**Effect:** ROUGE-L dropped to 25.6%, embedding fell to 71.0%, judges contested (judge mean 0.893).
The over-specification caused the model to write longer (avg 592 tokens), more verbose summaries that
diverged from the silver reference style and triggered judge disagreement, forcing ROUGE-only blend.

### Round 2 (exp-r2-1 through exp-r2-3, early stop)

All 3 r2 experiments rejected; 3-consecutive-fail early stop triggered. exp-r2-4 and exp-r2-5 not run.

**exp-r2-1** (−3.5%): Fixing exactly 2 paragraphs. Judges agreed (0.936) but ROUGE-L dropped from 27.3% → 25.3%.
Two paragraphs is too tight; meaningful content was compressed out.

**exp-r2-2** (−46.3%): Fixing exactly 3 paragraphs. Judges contested (0.893) → ROUGE-only blend.
Longer output diverged from silver reference style and triggered judge disagreement.

**exp-r2-3** (−40.6%): Reducing bullets to 5-6. ROUGE-L *improved* to 28.2% but embedding dropped
from 76.7% → 72.5% and judges contested. Fewer bullets hurt semantic coverage enough to lose judge agreement.

**Key r2 finding:** The 2-3 paragraph range in exp-4 is already at the optimal point. Fixing either
end of the range makes things worse. Judge contestation is easily triggered by length changes — the
silver reference style is narrow and length mismatches cause divergence.

---

## Patterns

**Consistently helped:**

- Broad coverage instructions ("ALL major discussion segments", "in order") — the single most
  impactful change, both for ROUGE-L and for judge agreement.
- Verbatim terminology preservation — improved embedding similarity by keeping named concepts
  from the transcript intact rather than paraphrasing.
- Output length constraints (paragraph cap + no-filler rule) — tighter outputs match the silver
  reference more closely, improving both ROUGE-L and embed scores.

**Hurt or had no effect:**

- Forcing very specific content requirements per paragraph (concrete numbers/specifics mandate)
  — caused over-verbosity, hurt embedding similarity, and triggered judge disagreement.
- "Chronological arc" framing when redundant — adding explicit narrative-progression language on
  top of an already-present "in order" instruction made outputs longer without improving coverage.
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

3. **Paragraph count tuning:** The current 2–3 paragraph cap improved results vs 2–4. A targeted
   ablation fixing at exactly 2 or exactly 3 paragraphs could surface the optimal target.

4. **Bullet count tuning:** The 6–8 bullet target in the user prompt may be higher than what the
   silver reference uses. Reducing to 5–6 could improve judge scores.

5. **Embedding similarity push:** Terminology preservation (exp-2) and conciseness (exp-4) both
   moved embed up. Further gains likely come from asking for phrasing closer to the transcript,
   or from experimenting with stricter word-level anchoring in bullets.

---

## Anomalies and Crashes

- **Hook/linter interference:** During session 2, a hook or linter modified
  `bundled_clean_summary_user_v1.j2` between reads (lines 19–20 changed to add bullet specificity
  rules). This meant the exp-2 commit included both the intended verbatim-terminology addition
  AND the linter's bullet specificity change. Their individual contributions cannot be fully
  separated without a controlled re-run.

- **exp-3 scoring divergence (session 1):** Two scoring passes produced different final scalars
  (0.450514 in first run vs 0.256322 recorded). The lower value reflects judge contestation
  (ROUGE-only blend). Conservative lower value was recorded.

- **r2 judge fallback (prior session):** r2 experiments run in a prior Chat-tab session used
  `claude-haiku-4-5` (short alias, now invalid) which caused the Anthropic judge to throw and
  silently fall back to running gpt-4o-mini twice. Those 5 rows were discarded and re-run with
  corrected model ID `claude-haiku-4-5-20251001`. The silent fallback has been removed from
  `autoresearch_track_a.py` — Anthropic judge failures now raise immediately.

---

AUTORESEARCH COMPLETE
