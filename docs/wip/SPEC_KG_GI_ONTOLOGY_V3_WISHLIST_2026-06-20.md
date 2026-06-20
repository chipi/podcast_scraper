# KG + GI ontology — v3 wishlist (post-v2 forward look, 2026-06-20)

**Status**: WIP wishlist, not a spec. Companion to rounds 1-3 of the
v2 spec. Records what v2 explicitly leaves on the table so we can
spot v2 choices that would lock v3 out, and so we have a destination
to point at when contributors ask "where does this fit?"

**Trigger**: Operator asked "should we do v3 after #1037 lands, and
what else might we be missing across other PRDs/RFCs?" Two
questions; two answers below.

---

## TL;DR — recommendation on sequencing

**Don't write a v3 spec now. Let v2 ship, then v3 emerges from real
data.** My honest read of the broader RFC/PRD audit:

- v2's ontology is sufficient for **every** planned downstream surface
  (Search PRD-031, Position Tracker PRD-028, Person Profile PRD-029,
  Topic Entity View PRD-026, Enriched Search PRD-027, Relational
  Query Layer RFC-094, LITM Context Packs RFC-093)
- The real v3 gates are **tooling and infrastructure**, not ontology:
  enrichment workers (RFC-088), MCP tool layer (RFC-095), entity
  resolver (#849), ML query router (RFC-092), corpus-impact surface
- Trying to spec v3 ontology changes before v2 ships in production
  is premature — the surface that drives the change doesn't exist yet

**Recommended sequencing**:

1. Wait for #1037 to merge cleanly
2. Branch `feat/corpus-ontology-v2` for #1036 chunk 1 (docs only —
   `docs/architecture/corpus/ontology.md`, zero risk)
3. Ship v2 chunks 1-9 over the following weeks
4. **Use v2 in production for at least 2-4 weeks** — let real query
   patterns + viewer usage surface what's actually missing
5. THEN write v3 spec, grounded in observed gaps, not theoretical ones

This doc parks the wishlist so we can spot v2 choices that would
foreclose v3 options. It's a destination map, not a spec.

---

## What v2 already covers (the "no need to defer" list)

The Explore audit confirmed v2's ontology is sufficient for the
flagship use cases — all of these ship on v2 with no additional
ontology work:

| Surface | PRD/RFC | v2 sufficient? |
|---|---|---|
| Position Tracker | PRD-028 | ✓ (round-3 added `insight_type` + `position_hint` for this) |
| Person Profile | PRD-029 | ✓ |
| Topic Entity View (base) | PRD-026 | ✓ (enricher layer is orthogonal — needed for temporal velocity, cooccurrence, grounding-rate badges, but the base view works on v2 alone) |
| Relational Query Layer | RFC-094 | ✓ (entirely enabled by v2 cross-layer edges) |
| LITM Context Packs (base) | RFC-093 | ✓ (degrades gracefully on missing optional fields) |
| Hybrid Search backend | RFC-090 | ✓ (RFC-090 implementation details, no new ontology fields) |
| Search Product (intent routing) | PRD-031 v1 | ✓ (rules-based router; ML upgrade is RFC-092) |

This is actually a strong validation of v2's scope. The ontology
foundation v2 lays is the right surface area; everything else is
infrastructure built on top.

---

## v3 wishlist — concretely deferred items

Grouped by category. Each item is either: (a) needs new infrastructure
not yet built, or (b) needs ontology work we explicitly chose not to
do in v2.

### Category 1 — Ontology evolution (the things v2 leaves on the table)

| # | Item | Why deferred from v2 | What it unlocks |
|---|---|---|---|
| 1 | **Topic semantic deduplication** | RFC-049 v1.1 deferral; corpus-level work | Cleaner browse / search — "AI Regulation" and "AI Policy" merge |
| 2 | **Cross-episode Person merging** (alias registry) | RFC-072 KL2 — corpus-level work | "Sam Altman" + "Samuel Altman" + "@sama" resolve to same Person |
| 3 | **CONTRADICTS edges between Insights** | RFC-049 Resolved Q1; needs NLI pipeline addition | Contradiction badges in Search (PRD-031 OQ-3), LITM packs (RFC-093 top_contradiction) |
| 4 | **Cross-episode Insight deduplication** | RFC-049 line 291 — deferred to v1.1, never landed | Reduced duplicate insights in Position Tracker timelines |

### Category 2 — Enrichment infrastructure (RFC-088)

The enrichment layer is what makes Topic Entity View (PRD-026) +
parts of LITM packs (RFC-093) useful. It's not ontology work — it's
a separate infrastructure decision:

| # | Item | Where it surfaces |
|---|---|---|
| 5 | **`temporal_velocity` enricher** — monthly mention histograms + trend direction per topic/person | PRD-026 temporal view, PRD-028 timeline |
| 6 | **`topic_cooccurrence` enricher** — pairs/triples of topics that co-occur in episodes | PRD-026 related-topics surface |
| 7 | **`grounding_rate` enricher** — % of Insights for a topic/person that are grounded | PRD-026 grounding badge, PRD-031 quality filter |
| 8 | **`corpus_coverage` enricher** — what topics / people / shows are under-represented | PRD-031 corpus_coverage intent, RFC-093 coverage_gaps in packs |

These are corpus-scope batch workers, run periodically. RFC-088
defines the architecture; implementation has not begun.

### Category 3 — Query-time infrastructure

| # | Item | Status | What it unlocks |
|---|---|---|---|
| 9 | **Entity resolver** (#849) — maps free-text "Maya" / "Sam Altman" to `person:` canonical IDs | Not built; #849 is a recon ticket | PRD-031 entity_lookup intent, RFC-094 string-keyed queries |
| 10 | **ML query intent router** (RFC-092) — ONNX classifier upgrading rules-based v1 routing | RFC-092 drafted, not built; gated on eval data | PRD-031 v2 intent routing |
| 11 | **Natural-language query layer** | RFC-050/056 deferred; PRD-031 v3 listed | Q&A surface ("what does Maya say about braking?") |
| 12 | **Char-offset alignment verification** | RFC-072 Known Limitation 1; hard blocker for Enriched Search | PRD-027 ship gate; chunk-to-Insight lift |

### Category 4 — Agent / tool layer

| # | Item | Status | What it unlocks |
|---|---|---|---|
| 13 | **MCP tool layer** (RFC-095) | Not started; generic MCP infrastructure | RFC-093 LITM packs as MCP tool, RFC-094 relational queries as MCP endpoints |
| 14 | **Agent-facing query endpoints** | Depends on #13 | Programmatic corpus access for downstream agents |

### Category 5 — UX features (Search V3 etc.)

These aren't ontology work but worth tracking:

| # | Item | Source |
|---|---|---|
| 15 | Saved searches | PRD-031 V3 |
| 16 | Search result export ("research bundle") | PRD-031 V3 |
| 17 | Search as MCP tool endpoint | PRD-031 V3 (depends on #13) |
| 18 | Cross-show topic view (multi-corpus) | PRD-026 / 028 / 029 all single-corpus only |

---

## v2 choices that could lock v3 capabilities — preservation checklist

If we get any of these wrong in v2, we close v3 doors. Pre-flight
checklist for v2 chunks:

| Invariant | Why it matters | v3 capabilities at risk if violated |
|---|---|---|
| **Grounding contract** — `Insight.grounded ⇔ ≥1 SUPPORTED_BY edge` to verbatim Quote | Viewer UX, enriched search (PRD-027 FR3.3 — "never synthesise from ungrounded Insights") | Every downstream surface |
| **Quote `char_start`/`char_end`/`timestamp_*_ms`** required, not optional | FAISS chunk-to-Quote matching (PRD-027), viewer jump-to-moment, Position Tracker quote display | Enriched Search, Position Tracker, viewer timeline |
| **Canonical IDs** — `person:{slug}`, `topic:{slug}`, `org:{slug}` consistent across kg.json + gi.json + bridge.json | Relational queries (RFC-094), CIL queries (RFC-072), graph traversals | All cross-layer features |
| **SPOKEN_BY emitted whenever diarization aligns** | Position Tracker quote attribution, Person Profile speech-vs-mention split | Position Tracker, Person Profile |
| **`bridge.json` per episode** | Cross-episode identity foundation | Position Tracker, Person Profile, all CIL queries |
| **`ABOUT (Insight → Topic)` edges emitted** | Topic Entity View (PRD-026 FR4.1), relational queries (RFC-094 `who_said`) | Topic browsing, all topic-keyed surfaces |
| **`MENTIONS_PERSON` / `MENTIONS_ORG` edges emitted** | Search entity_lookup intent, Person Profile mention count, RFC-094 `insights_about` | Search + Person/Org surfaces |
| **`insight_type` + `position_hint` fields on Insight** | Position Tracker timeline + filtering, Search temporal intent | Position Tracker, temporal Search |

All eight are already committed in round-3 of v2. This checklist is
a guard-rail for the chunk-by-chunk implementation: if any chunk's
implementation deviates from these, that's a v3-foreclosure event.

---

## Sequencing rationale — why not write v3 spec now

Three reasons:

1. **v2 isn't shipped.** Specs that aren't grounded in shipped
   behavior tend to be wrong in the details. The author (me) hasn't
   seen the viewer with cross-layer edges live yet. Neither have you.
   What v3 wants is best discovered AFTER v2 surfaces real friction.

2. **Most v3 items are infrastructure, not ontology.** The wishlist
   above shows ~70% of the items don't need ontology changes — they
   need enricher workers, MCP layer, entity resolver, ML router.
   Each of those is its own RFC + decision cycle. Bundling them into
   "v3 ontology" would be a category error.

3. **Two items genuinely depend on v2 outcomes**:
   - Topic semantic dedup → wait until v2's emitted topics give us a
     concrete corpus to measure dedup against
   - Cross-episode Person merging → wait until v2's bridge artifact
     gives us name variants to reconcile

   Spec them before we have the input data and we'll guess wrong.

**Better pattern**: this wishlist doc lives alongside the v2 spec
trio. As v2 chunks ship, we capture observations here. When v2 is
stable + in production, the wishlist + observations become the v3
spec input.

---

## Concrete next-step plan (the actual recommendation)

1. **#1037 lands cleanly** — security-quality CI now green per
   commit `388f932a`. Wait for the rest of the checks to complete.
2. **Tag the merge** — the autoresearch-followups thread closes
   when #1037 merges. Mark #1033 / #1034 / #1035 / #1036 / #1037
   as the snapshot.
3. **Branch `feat/corpus-ontology-v2`** off main once #1037 is in.
4. **Ship #1036 chunk 1** — `docs/architecture/corpus/ontology.md`
   as the unified ontology doc. Zero-risk doc-only PR. Lets us
   iterate on the textual ontology before touching code.
5. **Ship chunks 2-9** at whatever cadence makes sense — schemas,
   pipelines, migration, silvers, viewer, deprecation. Each chunk
   is independently bisectable.
6. **Observe in production for 2-4 weeks** — track real user
   interaction with the new ontology via the viewer + search.
   Note any places where the data shape forces ugly UX or
   inefficient queries.
7. **Write SPEC_KG_GI_ONTOLOGY_V3 round-1** with the observations
   as input. By then, RFC-088 + RFC-095 + #849 progress will also
   inform what's reasonable to scope.

**My honest read**: v2 is a lot of work but well-scoped. After it
ships, the next big thing is probably the **enrichment layer
(RFC-088)** — not v3 ontology. Topic Entity View + LITM packs both
want enrichers more than they want new node types. The ontology
foundation is right; the missing piece is the batch-worker tier on
top.

---

## What to revisit in this doc when v2 ships

Pull this doc back up and update the wishlist with:

- Which items got more important / less important based on real use
- Which items we accidentally solved during v2 implementation
- Which items the operator changed their mind about
- New items that emerged from v2 in production

This isn't ceremony — it's the spec-iteration discipline that round-3
already proved works (round-2 had blind spots that the RFC audit
caught + round-3 corrected). v3 should get the same treatment.

---

## Cross-references

- v2 spec lineage: rounds 1-3 in `docs/wip/SPEC_KG_GI_ONTOLOGY_*`
- v2 anchor issue: #1036 (now reflects round-3 framing in body)
- Autoresearch programme epic: #907 (umbrella, contains #1036)
- Current branch / PR: #1037 (feat/autoresearch-followups-2026-06-18)
- Adjacent RFCs/PRDs surveyed in the round-3 audit:
  - PRD-017, PRD-019, PRD-021, PRD-022, PRD-023, PRD-024, PRD-025,
    PRD-026, PRD-027, PRD-028, PRD-029, PRD-031, PRD-032
  - RFC-049, RFC-050, RFC-055, RFC-056, RFC-062, RFC-072, RFC-088,
    RFC-090, RFC-091, RFC-092, RFC-093, RFC-094, RFC-095
