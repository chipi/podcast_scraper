# #1191 — route-and-tag insights (durable redesign) — decomposition & plan

**Status:** planning (no code yet). **Design source:** [`GI_WHAT_TO_SURFACE.md`](GI_WHAT_TO_SURFACE.md).
**Interim already shipped (separate):** the duration-scaled insight cap + fuse recalibration
(`a01377c3`, tagged `#1191`) is grafted onto `feat/hardening-27` as a bounded stopgap. **This plan is
the proper redesign that removes the cap entirely.** Sized as an **epic** (comparable to the #1169
speaker arc).

---

## Problem statement (one paragraph)

The GI pipeline hard-caps insights per episode (`GI_MAX_INSIGHTS_CEILING`, enforced as
`min(max_insights, ceiling)` in all 6 providers). A count cap **baked into the pipeline is baked into
the corpus**: once a 100-episode build cuts at N, insight N+1 is gone and unrecoverable without
reprocessing everything. It truncates *real, gated-good* insights on long/dense episodes, and it hides
the very signal we want (an 8-insight news roundup vs a 19-insight interview *should* differ). The fix
is **route → classify → rank → tag, never truncate** — "first N" becomes a **view-time** decision.

## The target shape (from the design doc)

```text
1. EXTRACT     chunked                              -> ~50 candidates
2. ATTRIBUTE   speaker, from the quote's offset     -> free, deterministic   [shipped, #1169]
3. CLASSIFY    the pinned judge, one batched call   -> SURFACE / CONNECT / DROP
4. NOVELTY     embeddings vs the corpus             -> free
5. RANK        novelty x speaker-role x type        -> NO truncation
6. GROUND      verbatim quotes (top ~20)            -> the expensive stage
```

- **SURFACE** = `STANCE` / `ARGUMENT` / `EXPERIENCE` — what a person said you can't get elsewhere.
- **CONNECT** = `EVENT` (deals/launches/numbers) — corpus connective tissue → KG + threads; **never**
  competes for a UI slot.
- **DROP** = `TRIVIA` / `GENERIC` / `FILLER`.

Key correction the redesign carries: the classifier is the **value gate's judge** (pinned, vendor-
disjoint per #939), and its **rubric** is the actual bug — today its top tier is "a decision, deal,
launch, number", i.e. *a definition of news*, so it scores the least-novel thing top. Route-and-tag
needs the judge to give each claim a SURFACE/CONNECT/DROP label with the **speaker + role** as anchor.

---

## Component breakdown (7)

| # | Component | Files (starting point) | Delivers | Risk |
| --- | --- | --- | --- | --- |
| 1 | **Schema/contract** — add `tier` (surface/connect/drop), `routing_tag` (stance/argument/experience/event/…), `rank`, `salience` | `gi/contracts.py`, `docs/architecture/gi/gi.schema.json`, KG schema, both validators | the fields everything else writes/reads | **corpus migration** for existing artifacts |
| 2 | **Classify stage** — SURFACE/CONNECT/DROP via the pinned judge with a **better rubric** | `gi/value_gate.py` (judge), rubric/prompt, registry pins | the label that decides routing | rubric crispness (design "Open": judge still ~44% STANCE) |
| 3 | **Novelty stage** — corpus-novelty via embeddings | partial infra in `search/…/about_edges.py` | near-dup kill / redundancy rank | corpus-level speaker canonicalisation needed for guest weighting |
| 4 | **Rank stage** — composite novelty × speaker-role × type × grounded | new stage in `gi/pipeline.py` | ordered list, no cut | ranking must be "honest" (design §No fixed cutoff) |
| 5 | **Remove provider truncation** — drop `min(…, ceiling)` ×6; `gi_max_insights` → safety-only | the 6 GI providers | corpus stops truncating | keep extraction bounded by token budget (chunking/bundling stays) |
| 6 | **Viewer** — ranked first-look + expand | `web/gi-kg-viewer` | "first N" as a UI decision | builds on #1048 ranked-topic overview |
| 7 | **Eval + metrics** — per-tier metrics + re-baseline | eval harness, metrics | trustworthy long-form numbers | "no cap" shifts counts / `avg_insight_nodes`; **re-baseline** |

**Fuse coupling:** more surfaced insights ⇒ more per-episode grounding calls; keep the LLM call-fuse
limits in step (the interim recalibration is the current floor — re-measure with a per-episode
`llm_calls` counter, ties #1180).

---

## Suggested sequencing

1. **See the problem (repro).** A local demonstration that a long/dense episode loses real
   gated-good insights to `min(max_insights, ceiling)` — the cut is permanent in the corpus. This is
   the "see it before you fix it" gate and becomes a regression fixture.
2. **Foundation: #1 schema/contract + #2 classify-tag.** Freeze the contract (tier/tag/rank/salience)
   and land the classify stage behind a flag, writing tags without yet changing truncation. Migration
   path decided here (new fields nullable/defaulted so old artifacts read).
3. **#3 novelty + #4 rank** (both "free"), still non-truncating and flagged.
4. **#5 remove truncation** — only once 1-4 land and the fuse/eval (7) can absorb the count change.
5. **#6 viewer** ranked first-look, then **#7 eval re-baseline**.

Grounding-cost decision (design §Stage order): today we ground all ~50 then gate; the redesign grounds
the **top ~20** and keeps what grounds (~4× cheaper) — but grounding is itself a quality filter, so
selecting before grounding risks surfacing something unevidenced. Resolve when #4 lands.

## Non-goals / explicitly out of scope

- Not a "rank and keep top 12" — **any** pipeline cutoff is rejected by the design.
- Not a cleaning problem — the data says the DROP bucket is nearly empty (0% filler, 1.6% trivia); the
  dilution is 37% EVENTs (right content, wrong shape) → **routing**, not cleaning.
- LoRA / model training out of scope (the classifier is a pinned-judge rubric change, not a new model).

## Open questions (from the design doc)

- The judge rubric still isn't crisp (≈44% STANCE even independent); next iteration feeds it the
  **speaker + role** as the missing anchor.
- Which local model is the judge — a bake-off decides.
- Guest-vs-host weighting needs corpus-level **speaker canonicalisation** (phonetic + edit-distance
  merge of ASR name variants).
