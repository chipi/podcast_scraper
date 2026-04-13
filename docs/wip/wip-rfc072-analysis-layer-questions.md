# RFC-072 Phase 6 — Analysis Layer: Open Questions

**Date:** 2026-04-12
**Status:** Thinking artifact, not a work ticket
**Context:** RFC-072 Phases 1-4 build the data foundation (CIL, bridge, cross-layer
queries). Phase 5 adds search lift. Phase 6 is the analysis layer that operates on
the collected data to detect position changes, contradictions, and controversy. This
note captures the questions to answer before scoping a separate RFC.

---

## What "Position Change Detection" Actually Means

The Position Tracker (Phase 4) assembles a chronological arc of Insights by a person
on a topic. But it does not automatically detect that Insight A from 2023 contradicts
Insight B from 2025. What does detection look like?

**Candidate definitions:**

- **Entailment/contradiction between Insight text pairs.** Given two Insight texts
  from the same person on the same topic, classify the relationship as
  entailment, contradiction, or neutral. This is a standard NLI task.
- **Temporal sentiment/stance shift.** Track whether the person's position on a topic
  moves from positive to negative (or vice versa) over time. This requires a stance
  or sentiment signal per Insight, not just pairwise comparison.
- **LLM narrative.** Feed the full position arc to an LLM and ask it to narrate the
  shift: "In 2023, X believed Y. By 2025, they had shifted to Z. The pivot appears
  to have happened around episode E." This is the most user-friendly output but the
  hardest to evaluate.

**Key question:** Is the goal to *detect* change (binary: did it change?) or to
*narrate* change (structured: what changed, when, and how)? The answer determines
the approach.

---

## What "Controversy Radar" Requires

The simplest version: collect all Insights on `topic:X` from different persons, then
flag pairs where person A's Insight contradicts person B's Insight.

**Open questions:**

- Is NLI sufficient? Standard NLI models (e.g. cross-encoder on MNLI) classify
  (premise, hypothesis) as entailment/contradiction/neutral. Insight texts are not
  premise-hypothesis pairs — they are two independent claims. Does NLI still work, or
  do you need a different framing?
- Is semantic similarity + polarity a better signal? Two Insights can be semantically
  similar (same topic, same framing) but opposite in stance. Embedding distance alone
  does not capture polarity.
- Scale: for a corpus with 50 persons and 100 topics, the pairwise comparison space
  is large. How to prune? Only compare Insights on the same topic? Only compare
  Insights with high confidence?

---

## Candidate Approaches

| Approach | Pros | Cons | Eval difficulty |
| --- | --- | --- | --- |
| NLI cross-encoder (e.g. DeBERTa-v3-base-mnli-fever-anli) | Well-studied, fast, no API cost | Insight texts are not natural NLI pairs; may need prompt wrapping | Medium — need labeled Insight pairs |
| LLM-as-judge (e.g. GPT-4o, Claude) | Handles nuance, can narrate | API cost, latency, harder to evaluate systematically | Hard — subjective judgments |
| Embedding distance + stance classifier | Separates similarity from polarity | Requires a stance classifier (not off-the-shelf for arbitrary topics) | Hard — stance labels needed |
| Hybrid: NLI for detection, LLM for narration | Best of both — fast filter, rich output | Two systems to maintain | Medium |

---

## What Eval Data Would Be Needed

To test any of these approaches, you need:

1. **Labeled Insight pairs** — 50-100 pairs of Insights from the same person on the
   same topic, manually labeled as "same position," "shifted position," or
   "contradictory." Source: the eval corpus after Phases 1-4 land.
2. **Cross-person contradiction pairs** — 50-100 pairs of Insights from different
   persons on the same topic, manually labeled as "agree," "disagree," or "unrelated."
3. **Position arc narratives** — 10-20 manually written "here is how person X's
   thinking on topic Y evolved" summaries, to evaluate LLM narration quality.

This eval data does not exist yet. Building it is part of scoping the RFC.

---

## When to Start

After Phases 1-4 land and you have real cross-layer query results to look at. The
position arcs and guest briefs will reveal whether the Insight quality and topic
consistency are good enough to support analysis, or whether extraction quality
(Known Limitations 2 and 3 in RFC-072) needs to improve first.

Do not start the analysis RFC until you have run the Phase 4 queries against a real
corpus and looked at the output.

---

## References

- RFC-072 Phase 6 and Known Limitation 4 (Analysis layer not included)
- RFC-072 Vision section (Position Tracker, Guest Brief)
- RFC-072 Known Limitations 2-3 (Name variation, Topic deduplication)
