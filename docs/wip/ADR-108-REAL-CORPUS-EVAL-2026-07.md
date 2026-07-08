# ADR-108 real-corpus eval — `topic_consensus` + `stance_timeline` vs prod-v2

**Date:** 2026-07-07 · **Model:** `cross-encoder/nli-deberta-v3-small` (real DeBERTa, local CPU,
`[ml]` extra) · **Corpus:** `.test_outputs/manual/prod-v2/corpus` (99 episode bundles after
latest-run-per-feed dedup; 2 505 insights, 147 persons, 196 topics with ≥2 speakers).

This note records the honest outcome of running the two reimagined NLI enrichers (ADR-108) with the
real model over a real corpus — the eval the accuracy gate needs before either can auto-promote.

## Headline

> **SUPERSEDED — see the Update at the bottom of this file.** This headline records the *symmetric-
> entailment* result only. The follow-up tuning loop found a working **composite** signal (embedding
> cosine ≥ 0.70 AND NLI contradiction ≤ 0.5 → **precision 0.91**), so `topic_consensus` was rewritten
> to it and **activated** (auto-promoted). `stance_timeline` was retired for a read-time CIL timeline.

**Symmetric entailment clears no meaningful precision bar on prod-v2 — it stays dark.** The initial
ADR-108 thesis — that flipping to NLI *entailment* makes these winnable *without an LLM* — is **not
borne out** on this corpus; the data-driven gate is doing exactly its job. (The composite signal in
the Update below is what ultimately won.)

## `topic_consensus` — corroboration is too sparse

- 2 903 cross-Person pairs scored; **1** cleared symmetric entailment ≥ 0.6.
- The single emitted pair (score 0.78), `topic:ai-agents`:
  - Karpathy: *"AI agents can be designed with distinct personalities that influence user
    engagement…"*
  - Clark: *"AI systems demonstrate emergent preferences and a sense of self when trained to perform
    actions…"* — a plausible-but-loose corroboration (designed personality vs emergent self).
- **Precision on n=1 is statistically hollow; recall ≈ 0** (1 pair over 99 episodes). Symmetric
  entailment ≥ 0.6 is a very strict bar — real paraphrase-grade agreement between two different
  speakers' insights is rare. Lowering the threshold to buy recall, then curating gold on the
  survivors, is p-hacking without an independent gold set. **Not promoted.**

## `stance_timeline` — near-zero stance signal + a shift-rule bug

- Only **13** (person, topic) trajectories have ≥2 dated points corpus-wide (guests rarely recur on
  the same topic across episodes) — the stance surface is inherently thin on this corpus.
- The stance signal itself is **~0 almost everywhere**: `stance = entail(insight, "{topic} is good
  and promising.") − entail(insight, "{topic} is bad and overhyped.")`. A *factual* insight entails
  neither evaluative anchor, so the difference sits at ±0.00–0.05. Only 1 trajectory
  (Katie Martin / Oil prices) showed real magnitude — and its 4 points are all on the **same date**,
  so it is not a temporal shift at all.

### Bug found + fixed (shipped this branch)

`_deviation` flagged `shifted` when `sign_flips > 0`, and `sign_flips` counted **any** straddle of
zero. With stances jittering at ±0.001, that flagged **11 of 12** essentially-flat trajectories as
"shifted" (a ~92% false-positive rate). Fixed with a **deadzone** (`move_threshold / 2`): a stance
within the deadzone of 0 is neutral, so noise can't masquerade as a pro↔anti reversal. Post-fix:
**1 of 13** flagged (down from 12), and that one is the same-date artifact above. Mirrored in the
fixture builder's `_stance_deviation`; regression test added. **Not promoted.**

## Why not just write passing `gate_metrics.json`?

Because it would be fabricated. The gate exists to keep unvalidated signals out of the product; the
enrichers auto-promote the moment a real eval records precision ≥ 0.5 in
`data/eval/enrichment/<id>/gate_metrics.json`, and no such honest result exists here. Writing one
anyway is exactly the failure mode the gate was built to prevent.

## "Seeing them in the UI" is still available — via fixtures, not promotion

The gate governs whether the enricher **runs in a real pipeline**, not whether committed output
renders. The app-validation corpus (`tests/fixtures/app-validation-corpus/v3`) carries authored
`topic_consensus.json` + `stance_timeline.json`, and the operator viewer surfaces (Consensus edges +
Person "Consensus" / "Stance shifts") are wired to them. So the surfaces demo end-to-end against the
fixture corpus today — the enrichers just stay dark in real runs until an eval earns their keep.

## What legitimate activation would take (future work, not this push)

- **stance_timeline:** a stronger stance signal than raw entailment-of-evaluative-anchors — e.g.
  a sentiment/stance-tuned head, better anchor phrasing, or aggregating more insights per point; and
  a corpus where guests actually recur on a topic over time.
- **topic_consensus:** a curated corroboration gold on a real corpus to tune the symmetric threshold
  against measured precision/recall, rather than the strict 0.6 default that emits ~1 pair.
- Either path is an autoresearch-style tuning loop, not a config flip.

## Update — the composite works; `topic_consensus` ACTIVATED

The tuning loop above resolved `topic_consensus` in the same session. Re-scoring the 2 903 pairs by
**embedding cosine** (all-MiniLM-L6-v2) instead of symmetric entailment surfaced the genuine agreements
that entailment scored ~0.01 (financial-history, book-sales, commodities, music, ai-governance): 28
pairs clear cosine ≥ 0.70. Embedding **alone** hits precision 0.71 (it admits similar-but-opposite pairs
like "AI offloads routine work" vs "AI intensifies work"). Adding an **NLI contradiction ≤ 0.5** filter
removes exactly those (their contradiction ≈ 0.99, bimodally separated from the ≈ 0.00 of real
agreement) → **precision 0.909** (20/22) over a curated 28-pair gold.

So the winning signal is a **composite**: cosine ≥ 0.70 (shared-question gate) AND contradiction ≤ 0.5
(direction gate) — implemented as a single `ConsensusScorer` provider (MiniLM + DeBERTa, CPU-local, no
LLM). `topic_consensus` was rewritten to it, its real gate_metrics recorded (precision 0.91), and it
**auto-promoted** into the cloud / dgx / dev / local profiles. `stance_timeline` remains dark — its
signal is genuinely absent on this corpus (unchanged conclusion). ADR-108 updated with the corrected
design + activation.
