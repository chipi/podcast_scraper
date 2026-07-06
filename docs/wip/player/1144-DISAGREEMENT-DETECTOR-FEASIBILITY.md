# #1144 stance-level disagreement detector — feasibility decision

- **Status**: **Decided 2026-07-07 — do NOT build; close/defer (feasibility-negative)**
- **Issue**: #1144 (product goal inherited from #1106: replace the over-firing
  `nli_contradiction` with a stance-level disagreement detector)
- **Context**: RFC-088 enrichment layer; LEARNING-PLATFORM-GAP-ANALYSIS-2026-07 gap item 2

---

## TL;DR

The #1144 feasibility spike **already answered the feasibility question, negatively.**
On the real prod-v2 corpus, a strong reference labeler (Opus 4.8) judged 40 viable
cross-speaker stance pairs and found **0 disagreements** — 31 `no_shared_question`,
9 `agree`, 0 `disagree` (mean confidence 0.79). The stance-level disagreement signal
the detector would surface is **near-absent in real corpora**, so building it (an
offline-LLM enrichment tier + RFC + enricher + scorer) is **not justified**.

**Recommendation: close/defer #1144.** The product value ("what do different experts
say on this topic?") is already delivered by **perspectives (#1146, shipped)** — a
multi-speaker view *without* a disagreement claim. Revisit only if a corpus that
actually contains cross-person disagreement (e.g. debate-format shows) enters scope.

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
   real, the labels are confident; there is simply nothing to detect.

## Why the signal is absent (and why that's expected)

Curated podcasts are mostly single-host or *agreeable* interviews: a host and a guest,
or two experts elaborating a shared view. Two speakers land on the same *topic* but
rarely on the same *contested question* — so the honest label is "different facet," not
"disagree." This is the #1106 lesson one level up: **topic co-occurrence is not
disagreement**, and neither is stance co-occurrence.

## What this means for the build

- **Do not build** the stance-disagreement enricher / offline-LLM tier / RFC. The
  premise fails its own pre-registered gate ("if most pairs are no_shared_question/agree,
  reconsider before building").
- The **fixture** corpus (v3) carries one *engineered* disagreement (the injected
  Daniel-Cho-diversify vs Scott-Bessent-concentrate panel, #1148) — useful to exercise
  `nli_contradiction` + perspectives end-to-end, but it is a *test* signal, not evidence
  that real corpora contain disagreement.
- **Perspectives (#1146)** already ships the defensible product surface: it shows each
  speaker's take on a topic side-by-side, letting the *user* judge — no machine claim of
  "these two disagree" that the data can't support.

## Reopen criteria

Revisit #1144 only if the corpus mix changes to include content where cross-person
disagreement is actually present and frequent — e.g. debate-format or point/counterpoint
shows — and a fresh harvest+silver on that corpus shows a non-trivial `disagree` rate.
Until then, closing #1144 is the honest, cost-correct call.
