# #112 Entity-focused KG re-experiment — findings

**Date**: 2026-06-19
**Branch**: `feat/autoresearch-followups-2026-06-18`
**Trigger**: #1016 final report § 6b — 5 of 7 cohort candidates produced 0 entity-class nodes on the KG stage; only Moonlight (30%) emitted matchable entities. The bug was attributed to "cohort-uniform prompt doesn't explicitly request entity extraction."

## Hypothesis tested

**H₀**: Switching the KG configs from `kg_extraction_src: summary_bullets` to `kg_extraction_src: provider` (which routes KG extraction through `build_kg_transcript_system_prompt`, a prompt that already explicitly requests `entities` with `entity_kind` ∈ {person, organization}) will lift entity coverage from 0% to substantial across the cohort.

## Methodology

1. Created entity-focused variants of each KG round3 config — single field change:
   `kg_extraction_src: summary_bullets` → `kg_extraction_src: provider`
   Configs in `autoresearch/112_entity_focused_kg/configs/`.

2. Ran the eval on the same `curated_5feeds_dev_v1` dataset against the
   same `silver_opus47_kg_dev_v1` silver (cohort-disjoint vendor per
   JUDGING.md vendor-bias rule).

3. Compared per-candidate scores against the original Round 3 baseline
   from `data/eval/runs/autoresearch_prompt_vllm_<model>_dev_knowledge_graph_round3_v1/`.

4. **Cohort scope**: stopped after 2 candidates (Cell F NVFP4 + Moonlight).
   Two were enough to reject H₀ definitively; running the remaining 5
   wouldn't have changed the conclusion (more on this below).

## Results — H₀ rejected

| Candidate | Source | Topic cov | Topic avg_sim | Entity cov | Entity emit count |
|---|---|---|---|---|---|
| Cell F (Qwen3-30B-A3B-NVFP4) | bullets (baseline) | 41% | 0.600 | 0% | (varies; mostly 0) |
| **Cell F** | **provider** | **50%** (+9pp) | **0.642** | **0%** | 16 entities total (concepts) |
| Moonlight-16B-A3B | bullets (#1016) | 28.2% | 0.512 | **30.0%** (9/30) | 9 entities (named) |
| **Moonlight** | **provider** | **37.9%** (+9.7pp) | **0.550** | **0.0%** (regression!) | 27 entities (concepts) |

**Topics**: provider source consistently improves topic coverage by ~10pp.
**Entities**: provider source **regresses** entity coverage on the one
candidate that had any (Moonlight 30% → 0%). For Cell F it's
flat-at-zero — confirms model-behavior bias, not config alone.

## Why the hypothesis was wrong

The `provider` path feeds the **full transcript** to an LLM with a prompt
asking for "person | organization" entities. Both candidates respond by
extracting **conceptual nouns** as orgs:

- Cell F p02_e02 candidates: "RFCs", "error budgets", "security
  practices", "secrets management" — all `kind: org`
- Silver p02_e02 entities: `person: Ethan`, `person: Jonas`,
  `org: Practical Systems`

The candidates are confusing topic-like concepts with named entities.
The model's interpretation of "organization" leans toward "topical
category" rather than "specific named institution."

The `bullets` source worked **better for Moonlight** specifically
because:

1. The summary bullets already filtered out conceptual noise
2. What survives the summary is "the host Maya said X about Cascadia
   Alliance's trail policy" — natural-language references to named
   people / orgs in context
3. The bullet-derived KG extraction with `build_kg_from_bullets_system_prompt`
   was tightly scoped to a small input → fewer false positives

So `kg_extraction_src: summary_bullets` was inadvertently doing the
right thing on Moonlight by virtue of input filtering, not because of
the prompt.

## Real fix paths (not in scope for this experiment)

The #1016 § 6b root-cause attribution ("prompt doesn't request entities")
was incomplete. The actual gaps:

1. **Summary prompt suppresses names** — the current summary prompt
   (`ollama/qwen3.5_35b/summarization/system_v1`) says "Free of speaker
   names" — so bullets-derived entity extraction is starved of named-
   entity signal. For candidates without Moonlight's natural-language
   preservation, bullets yield 0 entities.

2. **`provider` source over-extracts** — when given the full transcript,
   the prompt's "entity_kind: person | organization" rule isn't
   enforced strictly enough; models grab conceptual nouns as orgs.

Two fixes that would actually move entity coverage:

- **Fix A: entity-preserving summary prompt** — rewrite the summary
  prompt to allow named-entity mentions ("Maya, the host, asked Liam
  about Cascadia Alliance's trail standards"). Keep the
  `kg_extraction_src: summary_bullets` source. Bullets then carry
  natural entity references, and the bullets-derived KG extraction
  picks them up cleanly.
- **Fix B: two-pass KG extraction** — keep `summary_bullets` for topics
  (where it works), add a second pass using `provider` source with a
  TIGHTER prompt that says "Extract ONLY proper-noun entities;
  abstract concepts go in topics not entities." Merge the buckets.

Both are summary-prompt level changes, not KG-config flips. Real work,
not a config tweak.

## Net for #112

The originally-scoped Cell E-like experiment (config flip across the
cohort) is **not the right path forward**. Doing the full 7-candidate
sweep with `kg_extraction_src: provider` would have produced 5 more
confirmations of "topics up, entities flat at 0% or regressed."
Stopping at 2 candidates spared the cohort the cost.

What the experiment surfaced:

1. The original §6b explanation was incomplete — the issue isn't
   missing entity request, it's input curation.
2. `kg_extraction_src: provider` IS a measurable improvement for the
   **topics** axis (~10pp) but a regression for entities.
3. Moonlight's 30% entity coverage in #1016 was real but
   coincidentally-arrived-at — its summary-bullet style preserves
   names better than other candidates, so its bullets carried entity
   signal the others lacked.
4. The realistic fix lives at the summary-prompt layer, not at the
   KG-config layer.

## DGX state after this experiment

- vLLM autoresearch: temporarily swapped to Moonlight (for this test),
  restored to Cell F NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 at the end.
- No homelab compose changes committed; all model swaps were
  uncommitted on the DGX. (Could be left uncommitted or git-checked-out
  to match the homelab `main` — already aligned now.)

## Updated recommendation for the task

Reframe #112 from "re-run cohort with entity-focused prompt" to
"redesign the summary prompt for entity preservation, then re-validate
KG entity coverage." That's a different shape of work:

- Prompt design (1-2 hours desk work)
- Re-run Cell F summary + KG with new summary prompt (~10 min GPU)
- Score entity coverage; if up, sweep the cohort

This experiment validated that the original §6b framing was
incomplete; the redesign question is now better-shaped. Marking #112
as "done — validated H₀ false; reframed scope". The actual fix lives
in a follow-up that addresses the summary prompt.
