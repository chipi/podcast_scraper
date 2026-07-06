# #1144 stance-level disagreement detector — build decision

- **Status**: **Decided 2026-07-07 — BUILD a precise stance-level detector**
- **Issue**: #1144 (product goal inherited from #1106: replace the over-firing
  `nli_contradiction` with a stance-level disagreement detector)
- **Context**: RFC-088 enrichment layer; LEARNING-PLATFORM-GAP-ANALYSIS-2026-07 gap item 2

---

## TL;DR

Build it. Cross-person disagreement between named experts on a **shared question** is
high-signal, specific information — precisely the kind of thing the platform should
surface when it exists. Two facts settle the direction:

1. **The mechanism is proven.** The injected panel in the v3 fixture (Daniel Cho
   *diversify* vs Scott Bessent *concentrate* on risk-management, #1148) is detected and
   surfaced end-to-end — the detect-and-attribute chain works.
2. **The 0/40 spike result is a *corpus* property, not a feature verdict.** The spike ran
   on a small, curated corpus of mostly single-host / agreeable interviews, where two
   speakers land on the same *topic* but rarely the same *contested question* (31/40
   `no_shared_question`). As the corpus grows / diversifies (debate-format, point-counter-
   point, cross-show revisits), real disagreement appears — and it's worth catching.

The spike's real lesson is a **design constraint, not a stop sign**: the detector must
gate on a **shared question / proposition**, because that's exactly what separates genuine
disagreement from the topic-adjacency false positives that gave `nli_contradiction` 0%
precision. That gate is what we build.

## The chain of evidence

1. **#1106 (`nli_contradiction`)** — atomic-insight NLI over-fired: 0% precision on
   prod-v2 (150-pair Opus silver). The cross-encoder scores "contradiction" on merely
   topic-adjacent insight pairs. Enricher disabled in all profiles; the accuracy gate
   (RFC-088 amendment) keeps it out until an eval records precision ≥ 0.5.

2. **#1144 hypothesis** — real disagreement lives at the **stance** level, not the
   atomic-insight level: does speaker A's *overall position* on a topic oppose B's?
   `scripts/eval/score/disagreement_stance_harvest_v1.py` builds the candidate pairs
   (per topic with ≥2 speakers ×≥2 insights, aggregate each speaker's stance);
   `disagreement_stance_silver_v1.py` has Opus judge stance-vs-stance with an explicit
   `no_shared_question` category for the "same topic, different facet" case.

3. **Spike result** (`data/eval/enrichment/disagreement/silver_prodv2_v1.jsonl`, n=40):

   | label | count | meaning |
   | --- | --- | --- |
   | `no_shared_question` | 31 | same topic, different *question* — no common proposition |
   | `agree` | 9 | aligned positions |
   | `disagree` | **0** | genuine opposition |

   Representative rationales: *"A discusses technical enterprise AI deployment while B
   discusses Meta's market position — addressing no common proposition."* The pairs are
   real; the low `disagree` count reflects the **corpus** (small, curated, mostly
   agreeable interviews), not the value of the signal.

## Why so few disagreements here (and why that's not a stop sign)

Curated podcasts today are mostly single-host or *agreeable* interviews. Two speakers
land on the same *topic* but rarely the same *contested question*. That's a property of
this corpus mix — as it grows/diversifies (debate-format, point-counterpoint, cross-show
revisits, the injected v3 panel), genuine disagreement appears. The spike's durable
lesson is the **design constraint**: gate on a shared question, or you get the
topic-adjacency false positives that gave `nli_contradiction` 0% precision.

## Design — precise stance-level disagreement enricher

A new ML-tier enricher `stance_disagreement`, structured like `nli_contradiction` but
operating at the **stance** level with a **shared-question gate**:

1. **Candidates** — reuse the harvest logic: per topic, group insights by speaker; for
   each cross-speaker pair where both have ≥2 insights (enough to *have* a stance),
   aggregate each speaker's insights into a stance.
2. **Judge (the gate)** — a `StanceDisagreementScorer` decides, for a pair:
   `shared_question` (do they engage a common proposition?) → if no, drop; else
   `a_position` / `b_position` / `label ∈ {disagree, agree}` + `strength`. This is the
   silver prompt (`disagreement_stance_silver_v1.py`) promoted to an injected scorer —
   an offline LLM provider at enrichment time (never CI: [[feedback_no_llm_in_ci]]).
3. **Emit** — only `label == disagree` pairs, with `shared_question`, both positions,
   strength, and the SPOKEN_BY-attributed persons — so the viewer can show
   *"A says X, B says the opposite"* with evidence.
4. **Gate** — an `accuracy_gate` (like `nli_contradiction`'s precision ≥ 0.5) scored
   against the Opus silver set, so the enricher only enters the registry/profiles/UI
   once it's precise. Deterministic in CI (silver is pre-computed).

Perspectives (#1146) remains the always-on multi-speaker surface; `stance_disagreement`
adds the sharper "these two *disagree*" claim on top, only where the gate is confident.
