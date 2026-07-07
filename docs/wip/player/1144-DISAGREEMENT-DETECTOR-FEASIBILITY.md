# #1144 stance-level disagreement detector — build + measurement record

- **Status**: **Built 2026-07-07 — no-LLM path measured non-viable; enricher ships gated dark**
- **Issue**: #1144 (product goal inherited from #1106: replace the over-firing
  `nli_contradiction` with a stance-level disagreement detector)
- **Context**: RFC-088 enrichment layer; LEARNING-PLATFORM-GAP-ANALYSIS-2026-07 gap item 2
- **Constraint**: operator directive — **no new LLM-dependent enrichers**

---

## TL;DR

Built the no-LLM stance-level detector and **measured** it against a proper gold set. The
answer is **no** — no-LLM DeBERTa NLI cannot do this task at any granularity. The enricher
is wired + gate-guarded and ships **dark** (excluded from registry/profiles/UI by its
accuracy gate); `perspectives` (#1146) remains the live no-LLM multi-speaker surface.

This reverses the earlier "BUILD it, mechanism proven" framing (2026-07-07 AM) with data.
Two claims in that earlier note were wrong:

1. **"The mechanism is proven — the v3 panel is detected."** It conflated *surfacing* with
   *detecting*. `perspectives` (#1146) **surfaces** the Cho/Bessent panel (both stances side
   by side). A no-LLM disagreement **scorer** does **not detect** it — the designed
   disagreement scores 0.30 symmetric contradiction, below any usable threshold.
2. **"Gate on a shared question → fixes the over-firing."** The shared-question gate is
   exactly what needs an LLM. Without one, there is no gate — only DeBERTa contradiction,
   which fires on topic-adjacency indiscriminately.

## What was built (all no-LLM, on `main` after this branch)

- `enrichment/enrichers/stance_disagreement.py` — corpus-scope ML-tier enricher. Aggregates
  each speaker's insights on a topic into one stance, scores the pair with the injected
  `NliScorer` (the CPU DeBERTa already used by `nli_contradiction`) **symmetrically** (both
  directions must clear the threshold), emits pairs above threshold. Wired everywhere a real
  ML enricher is (`enrichers/__init__`, `eval/admission.known_enricher_manifests`,
  `ml_wiring`, `registry` hint, `profile_sets` candidate) + a `FixedNliScorer` unit suite.
- `data/eval/enrichment/disagreement/gold_v1.jsonl` — the regression bar: the 40 prod-v2
  silver pairs (0 real disagreements) **plus** the designed v3 Cho-vs-Bessent debate
  (known positive, `disagree`) and the Cho-vs-Fischer topic-adjacent hard negative
  (`no_shared_question`).
- `scripts/eval/score/disagreement_stance_eval_v1.py` — scores the real DeBERTa output vs
  the gold, reports stance-aggregate (shipping signal) + atomic-max (diagnostic), writes
  the `gate_metrics.json` the accuracy gate reads. Local-only (real model, never CI —
  [[feedback_no_llm_in_ci]]).
- `data/eval/enrichment/stance_disagreement/gate_metrics.json` — the measured result
  (`precision 0.0`), which keeps the enricher dark.

## The measurement (2026-07-07, `cross-encoder/nli-deberta-v3-small`)

Gold = 42 pairs (1 real disagreement: Cho vs Bessent).

| Signal | Cho vs Bessent (the real disagreement) | Best precision | Recall |
| --- | --- | --- | --- |
| **stance-aggregate** (shipping) | 0.304 — below threshold, **missed** | **0%** | 0% |
| **atomic-max** (diagnostic) | 0.999 — caught | **10%** | 100% |

The atomic-max row is the tell: the real disagreement scores 0.999, but so do the
`no_shared_question` negatives — Clark/Bobrowsky 0.999, Amy Lawrence/Tim Cook 0.998,
Kravitz/Younis 0.961. **DeBERTa fires "contradiction" on any two substantive claims that
share a topic** — zero discrimination between genuine opposition and topic-adjacency. No
threshold, and no combination of the two signals, separates the positive from the false
positives, because they sit at the same score. This is the same failure that gave
`nli_contradiction` 0% precision (#1106), now confirmed at the stance level too.

**Conclusion:** the shared-question gate (does A engage B's *proposition*, not just B's
*topic*?) is a semantic judgment DeBERTa structurally lacks. It needs an LLM, which #1144
rules out. There is no viable no-LLM detector.

## Bonus finding — candidate generation also misses the designed positive

Independent of the scorer: the harvester (`min_insights=2`, per-topic `ABOUT` linking,
inherited from nli's index) does **not even form** the Cho-vs-Bessent pair. In the debate
episode (`p05_e04`) Bessent's concentration stance is split across `topic:risk-management`
(1 insight) + `topic:systems-thinking` (1), so he has <2 on either topic and is filtered
out. The pair the harvester surfaces instead is the cross-episode, ambiguous Cho-vs-Fischer
(systems-risk vs diversification — different *questions*). So even with a perfect scorer,
the current candidate generator would miss the cleanest designed disagreement. Recorded in
`gold_v1.jsonl` (the Cho-vs-Bessent row is added directly, with a note) so the scorer can be
measured on the real positive regardless.

## Disposition (operator, 2026-07-07): keep the enricher **gated dark**

- The enricher + eval + gold stay in the tree as a wired, gate-guarded **framework
  placeholder + regression harness**. The manifest `accuracy_gate` (precision ≥ 0.5) + the
  measured `gate_metrics.json` (precision 0.0) exclude it everywhere at once.
- **Auto-promotion contract:** if a future *non-LLM* scorer ever records precision ≥ 0.5
  against `gold_v1.jsonl`, the gate admits it with **no code edit** (guarded by
  `test_admission_would_promote_if_a_future_scorer_cleared_the_bar`).
- Until then, `perspectives` (#1146) is the honest no-LLM surface: it shows both stances
  without claiming "disagreement", so it cannot fabricate one.

## The chain of evidence (unchanged, for the record)

1. **#1106 (`nli_contradiction`)** — atomic-insight NLI over-fired: 0% precision on prod-v2
   (150-pair Opus silver). Enricher disabled/gated.
2. **#1144 hypothesis** — disagreement lives at the **stance** level, not the atomic level.
   `disagreement_stance_harvest_v1.py` builds the candidate pairs; the earlier Opus silver
   (`silver_prodv2_v1.jsonl`, n=40) found 0 `disagree` / 9 `agree` / 31 `no_shared_question`
   — a property of the small curated prod-v2 corpus, which is why the v3 designed positive
   was added to the gold.
3. **Measured (this record)** — stance-level, no-LLM: 0% precision (aggregate) / 10%
   (atomic-max). The stance-level hypothesis does not survive contact with a real scorer
   under the no-LLM constraint.
