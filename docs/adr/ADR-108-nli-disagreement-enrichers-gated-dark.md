# ADR-108: NLI-based disagreement enrichers gated dark pending a precise scorer

- **Status**: Accepted (2026-07-07)
- **Date**: 2026-07-07
- **Authors**: Marko Dragoljevic, Claude (Opus 4.8)
- **Related ADRs**:
  - [ADR-104](ADR-104-enrichment-layer-boundary-vs-kg-direct.md) — the enrichment-layer
    boundary these enrichers live behind.
- **Related RFCs**:
  - [RFC-088](../rfc/RFC-088-enrichment-layer-architecture.md) — enrichment layer + the
    accuracy-gate amendment (`RFC-088-ENRICHER-ACCURACY-GATE-2026-07.md`) this ADR relies on.

## Context

RFC-088 shipped `nli_contradiction` (cross-Person Insight *contradiction* pairs per Topic via
a DeBERTa cross-encoder `NliScorer`), and #1144 built `stance_disagreement` (a stance-level,
**no-LLM** successor that aggregates each speaker's stance and scores the pair symmetrically).
Both aim to surface the high-signal claim *"these two speakers disagree."*

The accuracy evals killed both:

- **`nli_contradiction`** — measured **0% precision** on prod-v2 (150-pair Opus silver, #1106).
  The cross-encoder scores "contradiction" on merely *topic-adjacent* insight pairs.
- **`stance_disagreement`** — measured **0% precision** (stance-aggregate) / 10% (atomic-max)
  vs the #1144 gold (`gold_v1.jsonl` = 40 prod-v2 pairs + the designed v3 Cho-vs-Bessent
  debate positive, #1148). DeBERTa cannot separate genuine opposition from topic-adjacency at
  any granularity — the positive (0.999 atomic-max) is indistinguishable from the
  `no_shared_question` negatives (0.96–0.999).

The load-bearing piece is a **shared-question gate** (do A and B engage the same contested
*proposition*, not just the same *topic*?). That is a semantic judgment DeBERTa structurally
lacks; it needs an LLM. The operator ruled out new LLM-dependent enrichers.

## Decision

Both enrichers stay **wired and registered** but **gated dark** by the data-driven accuracy
gate (`enrichment/eval/admission.py` → `profile_sets._admit`): each manifest declares
`accuracy_gate(precision ≥ 0.5)`, and `data/eval/enrichment/<id>/gate_metrics.json` records the
measured 0%, so the gate excludes them from the registry → profiles → UI. No enricher is
disabled by a hand-edit; the **data** excludes them.

The **live** cross-speaker surface is instead **`perspectives`** (#1146) — a CIL query over the
GI that shows each speaker's take side by side without asserting "disagreement." It makes no
precision claim, so it cannot fabricate one, and it sidesteps the gate entirely (it is not an
enricher). `stance_disagreement` is retained as a gate-guarded **framework placeholder +
regression harness**: if a future *non-LLM* scorer clears `gold_v1.jsonl`, it auto-promotes
with no code edit.

## Consequences

- No fabricated "contradiction"/"disagreement" pairs ever reach an operator or end user — the
  #1106 failure mode (0% precision surfaced as fact) cannot recur.
- The viewer retains **orphaned render code** for `nli_contradiction` (person "Contradictions"
  list, graph edges panel) that shows nothing while the enricher is dark — a known cleanup
  candidate, kept for now in case a viable scorer lands.
- The gate is the durable firewall: adding a disagreement detector is now a *data* exercise
  (clear the gold), not a code toggle.

## Alternatives considered

- **Ship the imprecise detector** — rejected; recreates #1106 (fabricated disagreements).
- **An offline LLM judge** (the shared-question gate as an LLM prompt at enrichment time,
  never CI) — rejected by operator constraint (no new LLM-dependent enrichers).
- **Delete the enrichers** — rejected; the wired-but-gated placeholder + `gold_v1.jsonl`
  regression bar are cheap and let a future scorer auto-promote.
