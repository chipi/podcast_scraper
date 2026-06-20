# Unified KG + GI ontology v2 — round-3 (after RFC/PRD audit, 2026-06-20)

**Status**: WIP draft. Continuation of round-1 + round-2 after a full
read of PRD-017, PRD-019, RFC-049/050, RFC-055/056, RFC-072, RFC-091.
Round-2 was written from the ontology.md summaries alone and missed
significant prior decisions — round-3 corrects that.

**TL;DR of what I missed in round-2**:

- RFC-072 (Canonical Identity Layer + Cross-Layer Bridge) **already
  shipped the infrastructure** I framed as "future work":
  canonical `person:` / `org:` / `topic:` IDs, per-episode `bridge.json`,
  `CorpusGraph` in `src/podcast_scraper/search/corpus_graph.py`,
  EntityResolver in `src/podcast_scraper/identity/resolver.py`. The
  identity merge of `speaker:` → `person:` is done.
- RFC-091 **explicitly REJECTED** KG proximity as a retrieval signal
  after empirical A/B testing showed −0.018 to −0.170 nDCG hits.
  Whatever we propose in v2 must not lean on co-occurrence as a
  ranking signal.
- RFC-072 introduced two additive v1.1 GIL fields — `insight_type`
  and `position_hint` — that power flagship use cases (Position
  Tracker, Person Profile). My round-2 didn't reference them.
- The original PRDs explicitly named "merging KG into gi.json" as a
  **Non-Goal** (PRD-019 line 26). Round-2's "Direction B unified
  ontology" reverses an explicit decision. We owe an explicit
  justification, not just a redesign.
- Entity extraction was **deferred to v1.1** in PRD-017 / RFC-049 and
  v1.1 never shipped. #1035 effectively brought entity extraction to
  production quality — so the deferral is now an artifact of
  documentation, not capability. v2 should formally retire the
  deferral.

This doc reframes the round-2 proposal against what's actually shipped
+ what the original design wanted but never delivered.

---

## 1. What's actually shipped vs round-2's framing

Round-2 treated these as gaps; the RFC audit says they're done or near-done:

| Round-2 "gap" | Actual state | Where it lives |
|---|---|---|
| KG↔GI shared id space | **Shipped** via RFC-072 CIL. `person:` / `org:` / `topic:` are canonical, owned by neither layer | `src/podcast_scraper/identity/slugify.py` + migration scripts |
| Bridge between artifacts | **Shipped** as per-episode `bridge.json` declaring all canonical IDs and their GIL/KG references | RFC-072 §2 |
| Cross-layer corpus graph | **Shipped** in-memory as `CorpusGraph` | `src/podcast_scraper/search/corpus_graph.py` (Slice B of #849) |
| Entity resolver | **Shipped** | `src/podcast_scraper/identity/resolver.py` (Slice A of #849) |
| `speaker:` → `person:` migration | **Shipped** in RFC-072; legacy artifacts still readable | `scripts/migrate_gi_speaker_to_person.py` |
| KG `entity:person:` → `person:` migration | **Shipped** | `scripts/migrate_kg_entity_ids.py` |

**What this means for round-2's "Direction A — minimal bridge"**:
Direction A is essentially already done. I framed it as a low-risk
forward sketch; in reality it's mostly behind us. The bridge edges
RFC-072 named (`ABOUT`, `SPEAKER_OF` via SPOKEN_BY + SUPPORTED_BY
composition, `IN_EPISODE`, `COVERS`, `MENTIONED_IN`) are queryable
today via the CorpusGraph layer — they just aren't materialized in
the per-episode artifact files.

**The real remaining work is**: turn the cross-layer queries that
RFC-072 makes possible into shipped per-artifact data so the viewer
can show them without rebuilding the corpus graph at request time.
That's a much smaller scope than round-2 implied.

---

## 2. The grounding-asymmetry paradox (round-2 missed this)

The agent flagged it cleanly:

> GIL requires verbatim quote evidence (the grounding contract).
> KG only requires co-occurrence (MENTIONS).
> Merging them would force all entities into the grounding contract
> (bloat) — or allow ungrounded entities (contradiction).

Round-2's proposal effectively said "ship Insight → MENTIONS_PERSON →
Person edges in GI." That's fine *as long as* the MENTIONS_PERSON edge
doesn't pretend to be a grounding edge. We have to be explicit:

| Edge | Grounding requirement | Why |
|---|---|---|
| `Insight → SUPPORTED_BY → Quote` | **Required** — verbatim contract | The whole reason GI exists |
| `Insight → ABOUT → Topic` | **Not required** — descriptive | An Insight is *about* something whether or not a quote names it |
| `Insight → MENTIONS_PERSON → Person` | **Not required** — descriptive | Same |
| `Insight → MENTIONS_ORG → Organization` | **Not required** — descriptive | Same |
| KG `MENTIONS` (Topic/Person/Org → Episode) | **Not required** — discovery | Pre-existing co-occurrence semantics |

Two-tier contract: **grounding edges (SUPPORTED_BY) prove a claim;
descriptive edges (ABOUT, MENTIONS_*) classify it.** Both can be
present on the same Insight; only the first one decides
`grounded=true/false`. The viewer's `hideUngroundedInsights` filter
still operates only on the grounding edge.

This is a real design constraint I didn't write down in round-2. v2's
schema must declare which edges are "evidentiary" vs "descriptive" so
future agents (and future me) don't blur the line.

**Implication**: section 2 of round-2 ("The grounding contract —
preserved exactly") was right in spirit but under-specified. Round-3
adds the two-tier edge contract.

---

## 3. The flagship use cases round-2 didn't anchor to

RFC-072 (the bridge) was built to enable two specific products:

### Position Tracker

> Trace how a Person's stated positions on a Topic evolved across
> episodes (and across time).

Requires:
- `Person` canonical identity (✓ shipped)
- `Topic` canonical identity (✓ shipped)
- `Insight → ABOUT → Topic` edge (not shipped yet — round-2 proposes it)
- `Insight → MENTIONS_PERSON → Person` edge (not shipped — round-2 proposes it)
- `Insight.position_hint` (0.0–1.0, derived from Quote timestamp / Episode duration) — **introduced in RFC-072 as v1.1 additive field, not in round-2**
- Episode `publish_date` (✓ shipped)

### Person Profile

> Single page showing everything we know about a Person:
> insights they voiced, quotes attributable to them, topics they've
> discussed, episodes they appeared in, organizations they're
> associated with.

Requires the same data foundations as Position Tracker plus:
- `Insight.insight_type` (claim | recommendation | observation |
  question | unknown) — **introduced in RFC-072 as v1.1 additive
  field, not in round-2**

These flagship use cases are what the operator's "viewer graph +
search/list UI" answer (round-1 Q2) actually means in concrete terms.
v2 should anchor on shipping these two surfaces end-to-end.

---

## 4. The punch list of "deferred but never finished" original-intent work

From the agent's audit, in priority order based on what enables the
flagship use cases:

### High-priority (blocks Position Tracker / Person Profile)

1. **`insight_type` field on Insight** (RFC-072 §2a) — additive v1.1.
   Requires LLM extraction change (classify each Insight into the
   enum). Empirical data on classification quality not yet collected.

2. **`position_hint` field on Insight** (RFC-072 §2b) — additive v1.1.
   **No LLM change needed** — pure arithmetic on existing
   Quote.timestamp_start_ms / Episode.duration_ms. Requires
   Episode.duration_ms to be set (RSS or audio duration), which is
   inconsistent today.

3. **`ABOUT` edge (Insight → Topic)** — defined in GI ontology but
   never emitted by default pipeline. Round-2 proposed post-pass
   semantic match; same approach holds.

4. **`MENTIONS_PERSON` + `MENTIONS_ORG` edges (Insight → Person/Org)**
   — defined as the GI `MENTIONS` edge in the ontology but Entity
   was deferred to v1.1 so it never fired. With #1035 NER pre-pass
   shipping Person/Org with 97-100% recall, the receiver exists now
   and these edges can light up.

### Medium-priority (enables analysis layers, not blocking flagships)

5. **Cross-episode Person identity merging** (RFC-072 Known
   Limitation 2). "Sam Altman" vs "Samuel Altman" → two `person:`
   slugs today. Future work: corpus-level alias registry. Not
   blocking flagship use cases but matters when scaling to multi-show
   corpora.

6. **Topic semantic deduplication** (RFC-049 Resolved Q1; RFC-072
   Known Limitation 3). "AI Regulation" vs "AI Policy" → two
   `topic:` slugs today. Future work: embedding similarity merge.
   `search/topic_clusters.py` Pareto-0.75 is the existing lever.

7. **Show / Podcast as a first-class node type with `FROM_SHOW` edges**.
   RFC-091 named this; #849 descoped it. Today: `Episode.podcast_id`
   is a string, `Episode.feed_id` is a string, no Show/Podcast node.
   v2 should formalize.

### Lower-priority (post-v2)

8. **Phase 5 chunk-to-Insight lift** (RFC-072 Known Limitation 1).
   Blocked on char-offset alignment between GIL Quotes and FAISS
   chunks. Separate workstream; v2 shouldn't try to unblock it.

9. **Contradiction / CONTRADICTS edges between Insights**
   (RFC-049 Resolved Q1, RFC-072 KL5). NLI-driven; requires a
   pipeline addition that doesn't exist today. Future analysis layer.

10. **Natural-language query translation** (RFC-050 Resolved Q2;
    RFC-056). Wait until structured queries (Insight Explorer)
    validate.

---

## 5. Reframed schema diffs (round-2 schema sketches + audit corrections)

### kg.schema.json — v2.0 (unchanged from round-2)

```diff
  nodes:
- - Episode | Topic | Entity(kind: person|org)
+ - Episode | Topic | Person | Organization
+ - Podcast  # NEW — RFC-091 'FROM_SHOW' needed it, #849 descoped

  edges:
- - MENTIONS (Topic|Entity → Episode)
+ - MENTIONS (Topic|Person|Organization → Episode)
+ - HAS_EPISODE (Podcast → Episode)
  - RELATED_TO (still reserved, still not emitted)
```

### gi.schema.json — v3.0 (round-2 + v1.1 additive fields)

```diff
  nodes:
- - Podcast | Episode | Person | Topic | Insight | Quote (Entity deferred)
+ - Podcast | Episode | Person | Organization | Topic | Insight | Quote

  Insight properties (additive):
+ - insight_type: enum [claim, recommendation, observation, question, unknown]
+ - position_hint: number 0.0-1.0  # derived from Quote timestamps + Episode duration

  edges:
  - HAS_EPISODE         (Podcast → Episode)
  - SPOKE_IN            (Person → Episode)
  - HAS_INSIGHT         (Episode → Insight)
  - SUPPORTED_BY        (Insight → Quote)   [grounding edge — required for grounded=true]
  - SPOKEN_BY           (Quote → Person)
- - MENTIONS            (Insight → Entity, was defined but Entity deferred)
+ - ABOUT               (Insight → Topic)            [descriptive — new]
+ - MENTIONS_PERSON     (Insight → Person)           [descriptive — new]
+ - MENTIONS_ORG        (Insight → Organization)     [descriptive — new]
```

### Edge-classification metadata (new, missing in round-2)

The schema should label each edge as `evidentiary` or `descriptive`:

```json
"edge_class": {
  "SUPPORTED_BY": "evidentiary",  // grounding contract
  "HAS_EPISODE": "structural",
  "SPOKE_IN": "structural",
  "HAS_INSIGHT": "structural",
  "SPOKEN_BY": "attribution",
  "ABOUT": "descriptive",
  "MENTIONS_PERSON": "descriptive",
  "MENTIONS_ORG": "descriptive",
  "MENTIONS": "discovery"  // KG's co-occurrence edge
}
```

This isn't a runtime field — it's metadata in the schema doc so the
viewer can style + filter consistently and future contributors know
which edges are grounding-load-bearing.

---

## 6. Updated migration chunks (round-2 was over-scoped)

Many of round-2's chunks shrink because the CIL / bridge work shipped.

### Chunk 1 — unified ontology doc + acknowledge what's shipped

- `docs/architecture/corpus/ontology.md` (new) — captures CIL
  (RFC-072), grounding two-tier contract (this doc §2), v1.1
  additive fields, edge classification
- Marks `docs/architecture/kg/ontology.md` v1.2 + `docs/architecture/gi/ontology.md`
  v2.0 as superseded with pointers
- Explicitly notes: "the ID space + bridge + CorpusGraph already shipped
  via RFC-072; this doc unifies the textual ontology, not the runtime"

**Risk**: zero. Docs only.

### Chunk 2 — schemas v2.0 (KG) + v3.0 (GI)

- `kg.schema.json` v2.0: Entity → Person + Organization + Podcast
- `gi.schema.json` v3.0: add `insight_type` + `position_hint` properties +
  ABOUT / MENTIONS_PERSON / MENTIONS_ORG edges
- Permissive transition: both schemas accept legacy + new shape
- Add edge classification metadata

**Risk**: low.

### Chunk 3 — KG pipeline emits Person + Organization + Podcast

- `kg/pipeline.py::_append_topics_and_entities_from_partial`:
  replace Entity emission with Person + Organization (no `kind`
  discriminator)
- Emit Podcast node + HAS_EPISODE edge using `Episode.podcast_id`
  + `Episode.feed_id` (reuses existing data, no new extraction)
- Migrate `kg/filters.py::repair_entity_kind` → decide node type
  not property

**Risk**: medium. Test fixtures regenerate.

### Chunk 4 — GI pipeline emits the three descriptive cross-layer edges

- `gi/pipeline.py`: after Insight extraction, run post-pass against
  the episode's Topic / Person / Org nodes (read from bridge.json
  or KG artifact) and emit ABOUT / MENTIONS_PERSON / MENTIONS_ORG
  edges with confidence scores
- Reuse the `search/topic_clusters.py` similarity threshold
  pattern (0.75)
- Same architectural pattern as #1035 NER pre-pass (deterministic
  helper around LLM call)

**Risk**: medium.

### Chunk 5 — Add `insight_type` and `position_hint` emission

- `gi/pipeline.py`: extend extraction prompt to classify each Insight
  into the `insight_type` enum
- Compute `position_hint` as arithmetic post-pass
  (`mean(Quote.timestamp_start_ms) / Episode.duration_ms`)
- Requires `Episode.duration_ms` to be set — backfill audit needed
  for which feeds currently expose duration

**Risk**: medium. Two prompt changes + duration backfill story.

### Chunk 6 — Migration scripts for existing corpora

- `scripts/migrate_kg_entity_to_person_org.py`: legacy
  `Entity(kind=person)` → `Person`, `Entity(kind=org)` →
  `Organization`. Pattern from RFC-072 migration scripts.
- `scripts/backfill_gi_insight_type.py`: optional — runs LLM
  classification on existing Insights without re-extracting them
- `scripts/compute_gi_position_hints.py`: pure arithmetic — works
  on every existing artifact with Quote timestamps

**Risk**: medium.

### Chunk 7 — Silver rebuild (full, per Q-2.3)

- Regenerate `silver_opus47_kg_dev_v1` shape: Persons + Organizations
  + Topics (drop the Entity discriminator)
- Regenerate `silver_opus47_gi_dev_v1` shape: add Topic references
  + Person mentions to insights for ABOUT / MENTIONS_PERSON
  measurement; add `insight_type` for classification accuracy
- Same for `silver_sonnet46_*_benchmark_v2`
- Score scripts updated for new shape
- Re-baseline KG topic / entity / ABOUT / MENTIONS_PERSON coverage

**Risk**: medium-high. The full rebuild is ~half-day of LLM time
on dev_v1 + benchmark_v2.

### Chunk 8 — Viewer updates

- `web/gi-kg-viewer/src/types/artifact.ts`: new node + edge types
- Graph stylesheet handles Person / Organization / Podcast
- Position Tracker view (new) — uses `insight_type` + `position_hint`
- Person Profile view (new) — uses all Insight → MENTIONS_PERSON
- Grounding contract visualization unchanged (dashed border, filter)
- Edge styling differentiates evidentiary / descriptive / discovery

**Risk**: medium-high. Two new UI surfaces (Position Tracker, Person
Profile) on top of schema rewiring.

### Chunk 9 — Drop legacy support in schemas

- After bake time, schemas v3 (KG) + v4 (GI) enforce new shape only
- Migration scripts deprecated

**Risk**: low.

**Compared to round-2 (8 chunks): added chunk 5 + chunk 8 to
explicitly own the v1.1 fields + flagship UI surfaces, kept the
same chunk count but reshaped scope.**

---

## 7. Decisions we're reversing — with justification

Round-2 quietly reversed several explicit prior decisions. Round-3
should make these explicit so reviewers (you, future contributors)
can see the why.

| Prior decision | Source | Why we're reversing |
|---|---|---|
| "Replacing GIL or merging KG into `gi.json`" was Non-Goal | PRD-019 line 26 | We're NOT merging artifacts (still two files). We ARE unifying the ontology + emitting cross-layer edges. The Non-Goal was about file structure, not ontology consolidation. |
| Entity extraction deferred to v1.1 | PRD-017 Non-Goal 1; RFC-049 Resolved Q2 | #1035 NER pre-pass shipped entity extraction at 97-100% recall. The capability gap that justified deferral no longer exists. v1.1 retired by acknowledgement. |
| "Cross-links between artifacts are optional and out of scope for v1" | RFC-055 §5 line 108 | RFC-072 made cross-layer joins explicit via `bridge.json`. v2 makes the join edges part of the per-artifact shape so the viewer doesn't have to read the bridge separately. |
| "KG is not a rename of GIL...do not interchange `gi` and `kg`" | PRD-019 line 45 | Still respect. KG and GI remain separately toggleable, separately extracted, separate file outputs. We're aligning the *ontology* not the operational naming. |
| KG proximity as retrieval signal | RFC-091 (rejected 2026-06-03) | Not reversing — confirming. v2 explicitly does NOT propose proximity-based ranking. Relational queries are the value, exactly as RFC-091 concluded. |

The headline reversal — "we're touching things the original design
said were out of scope" — is real. The justification is empirical:
#1035 retired one deferral reason (entity recall), RFC-072 retired
another (cross-layer plumbing). The remaining work is now
formalization, not net-new capability.

---

## 8. The five "never shipped but originally intended" items, surfaced

The agent's punch list. Round-3 incorporates each:

| Item | Original source | v2 chunk that picks it up |
|---|---|---|
| Entity extraction at v1 status | PRD-017 v1.1 deferral | Chunks 2 + 3 + 6 (formal Person/Org node types) |
| Topic semantic normalization | RFC-049 v1.1 deferral | Deferred to v3 — out of scope for v2 |
| `ABOUT (Insight → Topic)` edge | GI schema defined, never emitted | Chunk 4 |
| `MENTIONS (Insight → Entity)` edge | GI schema defined, Entity deferred | Chunk 4 (renamed MENTIONS_PERSON / MENTIONS_ORG) |
| `Insight → CONTRADICTS → Insight` | RFC-049 Resolved Q1 deferred | Deferred to v3 — needs NLI pipeline addition |
| Cross-episode Person merge / alias registry | RFC-072 KL2 | Deferred to v3 — needs corpus-level work |
| Phase 5 chunk-to-Insight lift | RFC-072 KL1 | Out of scope — char-offset alignment blocker |
| Natural-language query translation | RFC-050/056 deferred | Out of scope — wait for structured queries to validate |

**v2 scope = pick up the 4 things that are achievable today (Person/Org
node types, ABOUT, MENTIONS_PERSON, MENTIONS_ORG, insight_type +
position_hint) + Show/Podcast first-class.** Everything else marked
"deferred to v3" or "out of scope" stays parked.

---

## 9. Open questions for round-3

The big ones are still settled (grounding contract, cross-layer bridge,
unified Person, viewer + search surface). These are the new mechanical
calls that came out of the audit:

### Q-3.1 — Should `Insight.insight_type` extraction land in v2 or wait?

Option A: emit `insight_type` (LLM classification) as part of v2.
Cleanly enables Person Profile out of the gate. Cost: prompt change
+ measurement work.
Option B: emit only `position_hint` in v2 (arithmetic, no LLM
change). Insight_type lands in v2.1 once we measure classification
accuracy.

**My lean**: A. The prompt-change cost is small once you're touching
the GI extraction prompt anyway (chunk 4). Position Tracker can
ship in v2 with `position_hint` alone, but Person Profile is much
weaker without `insight_type`.

**?** Preference?

### Q-3.2 — How do we handle Episodes without `duration_ms`?

**Locked decision (operator review)**: 4-step fallback waterfall.
The "missing duration" scenario is real but rare and recoverable
from data already on disk.

**Empirical state of the corpus** (sampled from
`.test_outputs/manual/**/*.gi.json`, 477 files):
- 411 / 477 = **86% coverage** today
- 66 misses concentrated in older runs (`run_20260330-*` batch — 0%
  coverage) and one indie feed (BOOKstore Economics — RSS lacks the
  iTunes namespace entirely)
- Modern feeds since ~April 2026 consistently expose `<itunes:duration>`

**Four realistic root causes**:
1. **Publisher doesn't emit `<itunes:duration>`** — most common. Some
   indie podcasts use bare-bones RSS generators that skip the iTunes
   namespace. Older episodes in a feed's archive (pre-2015) may
   predate the publisher adopting it.
2. **Malformed value** — `<itunes:duration>about 45 min</itunes:duration>`
   or empty tag. Parser returns `None` cleanly (existing exception
   handler at `rss/parser.py:472`).
3. **Video feeds** — use `<media:content duration="...">` etc.
   instead of iTunes namespace. We don't read alternative tags today.
4. **Legacy / archive runs** — older test corpora where we never
   bothered with duration extraction.

**Fallback waterfall** (recovers ~100% in practice):

| Step | Source | Coverage estimate |
|---|---|---|
| 1 | `Episode.duration_ms` from RSS (existing) | ~86% |
| 2 | Last segment's `end` × 1000 from `*.segments.json` | ~99.9% (every transcribed episode has segments; verified `seg.end=1963.68s` matches episode length) |
| 3 | `max(Quote.timestamp_end_ms)` across the episode's quotes | lower-bound; preserves ordering for Position Tracker |
| 4 | Skip `position_hint` emission (optional field) | < 0.1% edge case |

**Why this matters**: I originally framed `position_hint` as
needing an ffprobe-backfill workstream. That's wrong — the
transcript pipeline already produces `*.segments.json` with a
last-segment `end` field that gives true episode duration to within
a second. **No new data extraction needed.** Helper lives in
`gi/pipeline.py` as a new function with the 4-step waterfall;
schema field stays `Optional` for the < 0.1% case.

**Locked**: ✓ — implemented in chunk 5 alongside `insight_type`.

### Q-3.3 — Do we add Podcast/Show node now or wait?

Round-2 deferred this. The audit surfaced that RFC-091 wanted a
`FROM_SHOW` edge but #849 descoped it. With operator's "viewer
graph" answer, the Show as a first-class clickable node makes
viewer UX cleaner ("click Singletrack Sessions → see all episodes").

**My lean**: yes, add in v2 chunk 3. Cheap (data already on Episode),
viewer benefit clear.

**?** OK?

### Q-3.4 — Should the unified ontology doc explicitly mention RFC-091?

I think yes — the doc should state that the unified ontology is
"relational, not proximity-based" up front, with the RFC-091 decision
record as the citation. Avoids a future contributor proposing
proximity-based retrieval again.

**?** OK?

### Q-3.5 — Migration sequencing — corpora before or after viewer?

Chunk 8 (viewer) needs the new node + edge types to render. Chunk 6
(migration script) rewrites existing corpora. Order:

- Migrate corpora first → viewer rolls out against existing data
- Viewer rolls out first → handles both legacy + new shape, migration
  happens lazily

**My lean**: corpora first. Cleaner state, no dual-shape complexity in
the viewer code. The audit showed RFC-072 migrations followed this
pattern and shipped cleanly.

**?** OK?

---

## 10. Synthesis: what v2 ACTUALLY is

After the audit, v2 is much more bounded than round-2 implied:

> v2 is a **formalization PR** that takes the cross-layer
> infrastructure RFC-072 already shipped + the entity recall #1035
> retired the deferral for + the flagship use cases RFC-072 named
> and turns them into a **consolidated ontology doc + a per-artifact
> shape that the viewer can render directly**.

It is NOT:
- A new graph database
- A merger of `kg.json` and `gi.json` files
- A new extraction pipeline
- A change to the grounding contract semantics
- A retrieval-side change (RFC-091 stands)

It IS:
- A unified ontology document (the original PRD-019 + PRD-017
  framed them as separate; the consolidated doc captures what's
  actually shared post-RFC-072)
- Schema diff: Entity → Person + Organization; v1.1 fields land;
  cross-layer edges emit per-artifact
- Pipeline edits: KG emits Podcast nodes; GI emits ABOUT /
  MENTIONS_PERSON / MENTIONS_ORG via post-pass semantic match
- Two new viewer surfaces (Position Tracker, Person Profile) the
  CIL was originally built to enable but never delivered
- Migration scripts (small, RFC-072 pattern)
- Silver rebuild + re-baseline (full)

**Total effort**: I'd estimate 9 chunks landing in roughly two
3-chunk waves + a final viewer wave, against the prior round-2
estimate of "multi-week." The actual delta from today is smaller
than I framed.

---

## 11. Suggested next loop

Same pattern as before:

1. You answer Q-3.1 through Q-3.5 (or push back on framing)
2. I update #1036 issue body with this round-3 framing — the
   chunked plan, the explicit reversal acknowledgements, the
   v1.1 field landing, the Podcast node decision
3. Round-4 (if needed) writes the actual chunk 1 (the
   `docs/architecture/corpus/ontology.md` doc) as a real
   ship-ready file
4. Branch + chunked execution begins

---

## Appendix A — what the RFC audit changed in my mental model

Concrete things I had wrong in round-2:

1. I said "ids span artifacts but nothing declares them shared" —
   wrong. RFC-072 + `bridge.json` declares them shared.
2. I said "no Quote→Topic link" — partially wrong. The CorpusGraph
   composes this in-memory via Quote → SUPPORTED_BY-reverse →
   Insight → ABOUT → Topic. Just not materialized in the artifact.
3. I said "Show/Podcast boundary fuzzy" — accurate, but I missed
   that RFC-091 had a concrete plan (FROM_SHOW edge) that #849
   descoped.
4. I said "no Person-Insight bridge" — partially wrong. The synth
   `SPEAKER_OF` shortcut exists in CorpusGraph (composes SPOKEN_BY +
   SUPPORTED_BY). The materialization is what's missing.
5. I framed "Direction A — minimal bridge" as a forward sketch. It's
   substantially already shipped.
6. I missed `insight_type` and `position_hint` entirely. Both are
   from RFC-072 and matter for flagship use cases.
7. I missed RFC-091's empirical rejection of KG-proximity-as-retrieval.

## Appendix B — references the audit pulled

- PRD-017 (Grounded Insight Layer) — `docs/prd/PRD-017-grounded-insight-layer.md`
- PRD-019 (Knowledge Graph Layer) — `docs/prd/PRD-019-knowledge-graph-layer.md`
- RFC-049 (GI core) — `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- RFC-050 (GI use cases) — `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md`
- RFC-055 (KG core) — `docs/rfc/RFC-055-knowledge-graph-layer-core.md`
- RFC-056 (KG use cases) — `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md`
- **RFC-072 (CIL + cross-layer bridge)** — `docs/rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md` — the most important read for round-2's gaps
- RFC-091 (KG proximity decision record) — `docs/rfc/RFC-091-*.md`

Spec lineage on this branch:
- Round-1: `SPEC_KG_GI_ONTOLOGY_REVIEW_2026-06-20.md`
- Round-2: `SPEC_KG_GI_ONTOLOGY_V2_2026-06-20.md`
- Round-3 (this doc): `SPEC_KG_GI_ONTOLOGY_V2_ROUND3_2026-06-20.md`
- GH anchor: #1036 (filed before round-3 audit — body needs update
  with the new framing once you sign off)
