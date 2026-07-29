# Eval — does #1191 salience ranking track insight quality? (100-ep, 2026-07-29)

**Verdict: NO for ranking, YES (bounded) for de-truncation.** Salience barely predicts independent
quality (Spearman 0.16); the value-gate tier that feeds it barely discriminates (CORE 3.06 vs USEFUL
2.95 on a 1–5 scale). Sorting by salience is *harmless* (tiny gain at top-3, ~flat elsewhere) so it
stays the default display order, but **nothing should be built on it** — the deferred search-boost
and operator `routing_tag` filter chip stay deferred. The real, measured win is **de-truncation**:
insights the old top-5 cap cut (ranks 6–8) are as good as the top-5; the tail beyond 8 is weaker.

## Method

- Corpus: `prod-v2.4-100ep` — 105 episodes, **866 insights** (mean 8.2/ep, max 13), all carrying
  `salience/rank/routing_tag/tier`. Gemini generated them.
- Judge: **claude-sonnet-5** (cross-vendor — Gemini generated, so the judge is disjoint), scalar
  rubric 1–5 on SUBSTANCE (specific / informative / non-obvious). Text-only. Cost **$1.06**, 0 errors.
- Harness: `scratchpad_insight_eval.py`. Reconstructs both arms (salience-topN vs extraction-topN)
  from the one run — no baseline re-run needed, since de-truncation stores every insight + its rank.

## Results

| Metric | Value | Reading |
| --- | --- | --- |
| Spearman(salience, judge) | **0.16** | near-zero — salience does not predict quality |
| Spearman(tier, judge) | **0.24** | also weak |
| mean judge, tier 2 (USEFUL) / tier 3 (CORE) | 2.95 / **3.06** | tiers barely separate quality |
| A/B top-3: salience vs extraction | **+0.12** | small help at the very top |
| A/B top-5 / top-8 / top-10 | +0.05 / **−0.04** / −0.01 | evaporates, slightly negative by 8 |
| de-trunc: within-top5 / beyond-top5 | 2.99 / **3.05** | ranks 6–8 as good as top-5 → cap-5 cut good insights |
| de-trunc: within-top8 / beyond-top8 | 3.06 / **2.63** | the tail beyond 8 is genuinely weaker |
| overall mean judge | **3.02 / 5** | "real but generic" — quality ceiling is extraction, not order |

## Why the ranking is weak

`salience = f(tier, grounded, surfaceable)`. Its dominant input, the value-gate **tier**, does not
discriminate (2.95 vs 3.06), so ordering by salience is close to reshuffling. This is not a bug in
the plumbing — it is that the value gate's tier is a coarse keep/route signal, not a fine quality
score. Making ranking matter requires a better tier — i.e. **ADR-133's deferred rubric research is
now required, not optional.** This eval is the evidence.

## routing_tag is NOT a quality signal (audit sub-finding)

Judge score by `routing_tag`: **`connect` 3.12 (41% ≥4-good) > `surface` 2.99 (33%)**. So the
`surface`/`connect` split does **not** track quality — it tracks **attribution** (`surface` =
spoken by a named person; `connect` = unattributed, `surfaceable=False`, kept for KG/threading).
Consequences: (1) the deferred operator `routing_tag` filter chip is **doubly unjustified** —
filtering by it would hide *better* insights; (2) high-quality **unattributed** insights (the 178
`connect`) are surfaced on **no** attributed surface today (the server's `surfaceable` gate excludes
them from insight lists + perspectives) — a real product gap, but surfacing them as "someone's
insight" breaks the attribution model, so it's a roadmap question, not a quick fix.

## Actions taken / recommended

1. **Keep salience-sort as the default order** (harmless, small top gain) — done in surfaces 1/3/4.
2. **Player fold 6 → 8** — ranks 6–8 are good; 8 (= `gi_surface_default_limit`) is the right cut.
3. **Search boost + operator filter chip: stay deferred** — salience (0.16) isn't trustworthy
   enough to gate/boost on. Revisit only if the rubric research lands a discriminating tier.
4. **Next real lever is insight *quality*** (extraction + value-gate rubric), not ranking. Overall
   3.0/5 says the ceiling is what we extract, not how we order it.

## NOT covered / caveats

- **Single judge, single rubric.** A second judge vendor or a grounding-aware rubric could shift the
  absolute means; a 0.16 correlation is unlikely to flip to "trustworthy," but the tier means (2.95
  vs 3.06) could move. Not run.
- **Text-only judging.** Grounding is a separate stored flag, not fed to the judge here.
- **One corpus, 105 episodes**, tech/finance-heavy + a few narrative shows. Not a general claim.
- **De-truncation win is quality-neutral-to-positive to ~8, negative beyond.** We did NOT measure
  whether the extra 6–8 insights improve downstream KG connectivity / threads — only their own
  substance.
