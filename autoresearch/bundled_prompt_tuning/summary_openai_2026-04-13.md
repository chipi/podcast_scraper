# AutoResearch Bundled Prompt Tuning — Summary

**Date:** 2026-04-13
**Run:** `autoresearch_prompt_openai_bundled_smoke_bullets_v1`
**Dataset:** `curated_5feeds_smoke_v1` (5 episodes)
**Judge models:** gpt-4o-mini (A), claude-haiku-4-5-20251001 (B)
**Rubric hash:** 9f43a4b9

---

## Total Experiments Run

14 experiments total (exp-1 through exp-4 in r1; exp-r2-1 through exp-r2-3 in r2, early-stopped; exp-r3-1
through exp-r3-5 in r3, all 5 planned experiments run). r2 and r3 used corrected judges
(claude-haiku-4-5-20251001) after r1 judge issue resolved.

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
| exp-r3-1  | 0.484212 | 28.6%   | n/a   | 0.946      | full blend       |
| exp-r3-2  | 0.472618 | 27.0%   | n/a   | 0.946      | full blend       |
| exp-r3-3  | 0.504330 | 31.2%   | n/a   | 0.953      | full blend       |
| exp-r3-4  | 0.501324 | 30.2%   | n/a   | 0.966      | full blend       |
| exp-r3-5  | 0.483498 | 28.2%   | n/a   | 0.953      | full blend       |

**Best: exp-r3-3 (0.504330) — champion after r3**

- Absolute improvement over baseline: **+0.273147**
- Relative improvement: **+118.2%**
- Improvement over previous r1 champion (exp-4): **+0.029643 (+6.2%)**

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

## Round 3 Results

All 5 planned r3 experiments run. 2 accepted, 3 rejected.

### exp-r3-1 (+2.0% from r1 champion, accepted)

**Change:** Added 3 silver-quality example bullets to the system prompt as a style target — showing
the exact density, em-dash precision, and "X rather than Y" contrast pattern of the silver reference.

**Effect:** ROUGE-L 27.3% → 28.6% (+1.3 pp), judge_mean steady at 0.946. Judges agreed. Direct
style exemplars closed some vocabulary gap.

### exp-r3-2 (−2.4% from r3-1 champion, rejected)

**Change:** Added anchor-lead instruction requiring each bullet to open with the specific
concept/technique as grammatical subject; explicitly banned "The episode", "Speakers", "The host",
"One key" openers in both system and user prompts.

**Effect:** ROUGE-L dropped 28.6% → 27.0%. The prescriptive subject-noun rule constrained the model
too much and produced less fluent bullets. Judge scores unchanged (0.946). Prompt constraints that
force grammatical structure hurt more than they help.

### exp-r3-3 (+4.2% from r3-1 champion, accepted)

**Change:** Added prose style narration to the system prompt: "dense and noun-anchored; prefer 'X
does Y'; use em-dash or semicolon to pack a second layer of precision; favor 'X rather than Y'
contrasts; every bullet stands alone as a self-contained insight."

**Effect:** ROUGE-L 28.6% → 31.2% (+2.6 pp), judge_mean 0.946 → 0.953. Judges agreed. This was
the highest single-experiment gain in r3. Style narration (describing the *type* of writing) works
better than grammatical prescriptions (exp-r3-2) or structural rules.

### exp-r3-4 (−0.6% from r3-3 champion, rejected)

**Change:** Model upgrade gpt-4o-mini → gpt-4o (YAML change only, prompts unchanged).

**Effect:** ROUGE-L dropped from 31.2% → 30.2% despite judge_mean rising from 0.953 → 0.966.
gpt-4o produces outputs that judges rate more highly but that are slightly less aligned with the
silver reference vocabulary. At rouge_weight=0.70, the ROUGE drop outweighs the judge gain.
Marginal at this eval scale; worth re-testing on a larger dataset before ruling out.

### exp-r3-5 (−4.1% from r3-3 champion, rejected)

**Change:** Restructured user prompt output rules into 3 numbered Tasks (Task 1 title, Task 2 summary,
Task 3 bullets) to help the model budget attention per section.

**Effect:** ROUGE-L dropped 31.2% → 28.2%. Restructuring the prompt into tasks fragmented the
existing instruction flow, and the model lost context between the tasks. The flat bullet-list format
of the original user prompt is more effective than numbered sections.

---

## Suggested Next Directions

1. **Style narration iteration (r4):** exp-r3-3 showed that style narration outperforms all other
   r3 strategies. Next: extend the narration with bad-vs-good bullet examples ("avoid: 'The episode
   discusses X' — prefer: 'X enables Y because Z'"), or add a style description for the summary
   paragraph to match the silver's dense, thesis-first structure.

2. **Few-shot for title + summary + bullets combined:** r3-1 showed few-shot examples help for
   bullets. Try adding a complete few-shot block (title + summary + bullets) showing all three
   components together, rather than bullets only.

3. **gpt-4o on a larger dataset:** r3-4 was marginal on 5 episodes (ROUGE-L small drop, judge
   quality up). Worth re-testing on the full benchmark — at scale, gpt-4o's higher capability may
   produce clearer gains, and the ROUGE-L vs judge tradeoff may flip.

4. **Benchmark-scale validation:** All tuning so far on the 5-episode smoke set is noisy. Running
   the r3-3 champion prompt on the full benchmark would confirm whether the +6.2% gain holds or
   is smoke-set variance.

5. **Summary paragraph alignment:** All r3 gains came from bullets (silver has bullets-only
   reference). The summary paragraph contributes noise to ROUGE. Aligning the summary paragraph
   more closely with what judges reward (clear thesis, concise structure) could push judge_mean
   above 0.953 and unlock further score gains.

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
