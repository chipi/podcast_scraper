# Unified KG + GI ontology — round-2 sketch (2026-06-20)

**Status**: WIP. Continuation of `SPEC_KG_GI_ONTOLOGY_REVIEW_2026-06-20.md`
after operator's locked-in answers:

1. **Grounding contract = load-bearing** → preserve `Insight ← SUPPORTED_BY → Quote` invariant
2. **Cross-layer bridge = yes** → ids span artifacts; emit cross-layer edges
3. **Unified Person concept** → no Entity-vs-Person split

Plus the implicit answer to #2 (user-facing surface): **viewer graph +
search/list UI**. I checked the viewer code while drafting — grounding
isn't just a data invariant, it's visible UX (`hideUngroundedInsights`
filter, ungrounded insights get dashed-border styling). That's a hard
constraint, not a "nice to keep."

This round proposes a concrete unified ontology + a chunked migration
plan. Same iterative tone as round-1 — push back on any of it.

---

## 1. The shape we're heading toward

### Node types (proposed v2)

| Node | Purpose | Replaces |
|---|---|---|
| **Podcast** | Feed-level container | (KG never had; GI deferred) |
| **Episode** | Anchor (one per episode) | Kept |
| **Person** | Anyone named or speaking — host, guest, mentioned-but-absent, diarized speaker | Merges `KG.Entity(kind=person)` + `GI.Person` |
| **Organization** | Orgs, companies, shows-as-orgs | Renames `KG.Entity(kind=org)` |
| **Topic** | What an episode is about (one definition shared across layers) | Same name, single concept |
| **Insight** | Claim / takeaway extracted from episode content | Kept from GI |
| **Quote** | Verbatim transcript span (the grounding contract) | Kept from GI |

What's new vs today:
- `Person` is one canonical node (not split across files)
- `Organization` is a first-class type (not buried under `Entity.kind=org`)
- `Topic` has one definition (KG and GI silver use the same shape now)

What's gone:
- `Entity` (replaced by `Person` + `Organization`)
- The `kind`/`entity_kind` discriminator (now a type-level distinction)
- The "deferred to v1.1" placeholder in GI

### Edge types (proposed v2)

Grouped by intent. **Bold edges = load-bearing for grounding or UI**.

**Structural / containment**
| Edge | From → To | Notes |
|---|---|---|
| HAS_EPISODE | Podcast → Episode | Today: KG knows `podcast_id` string but emits no edge; GI ontology defined the edge but doesn't emit it. Fix: actually emit it. |
| MENTIONS | Topic\|Person\|Organization → Episode | KG's existing discovery edge. Survives unchanged. |

**Speaker attribution**
| Edge | From → To | Notes |
|---|---|---|
| SPOKE_IN | Person → Episode | Person was a diarized speaker in this episode. |
| **SPOKEN_BY** | **Quote → Person** | Speaker attribution on a quote. Required when diarization aligns. |

**Insight grounding (the contract)**
| Edge | From → To | Notes |
|---|---|---|
| HAS_INSIGHT | Episode → Insight | Episode contains insight. |
| **SUPPORTED_BY** | **Insight → Quote** | **Hard invariant: `Insight.grounded=true` ⇔ ≥1 SUPPORTED_BY edge.** Viewer styles ungrounded insights with a dashed border. |

**Cross-layer bridges (new, what this round adds)**
| Edge | From → To | Notes |
|---|---|---|
| ABOUT | Insight → Topic | Insight is *about* a Topic. Replaces today's silent gap. |
| MENTIONS_PERSON | Insight → Person | Insight refers to a Person. The #874 / RFC-091 cross-layer edge that never landed. |
| MENTIONS_ORG | Insight → Organization | Same, for orgs. |

**Topic relations** (optional, evolve later)
| Edge | From → To | Notes |
|---|---|---|
| RELATED_TO | Topic ↔ Topic | Already reserved in KG schema. Defer to a follow-up; not needed for v2. |

### What the viewer can do that it can't today

| User intent | Graph traversal |
|---|---|
| "Show me everything about Maya" | One node click → her node has SPOKE_IN, MENTIONS_PERSON(reverse), MENTIONS(reverse) all incident — single page renders everything |
| "What does the host say about braking?" | `person:maya ← SPOKEN_BY ← Quote ← SUPPORTED_BY ← Insight → ABOUT → topic:braking-technique` — graph query, no out-of-band join |
| "List insights mentioning Singletrack Sessions" | `org:singletrack-sessions ← MENTIONS_ORG ← Insight` — direct filter |
| "Browse episodes by topic" | `topic:braking-technique ← MENTIONS ← Episode` — same as today, but now Topic identity is shared across silver + KG + GI |

The big shift: today the viewer does the "what episodes mention Maya"
join in code by string-matching `person:maya` across `kg.json` and
`gi.json`. With the v2 ontology and a shared id space, that's a graph
walk — one cypher-like query.

---

## 2. The grounding contract — preserved exactly

**Invariant (unchanged from v1)**:

> `Insight.grounded = true` IF AND ONLY IF there exists ≥1
> `SUPPORTED_BY` edge from that Insight to a Quote node, AND that
> Quote's `text` equals `transcript[char_start:char_end]` verbatim.

What the v2 changes don't touch:
- Quote.text must be verbatim — no paraphrasing
- char_start/char_end remain required
- timestamp_start_ms / timestamp_end_ms remain required (the viewer
  jumps to these)
- Quote → Person attribution via SPOKEN_BY when diarization aligns

What we ADD on top of the contract (no weakening):
- An ungrounded Insight (`grounded=false`) can now still have `ABOUT`
  edges to Topics and `MENTIONS_PERSON` / `MENTIONS_ORG` edges to
  Persons / Orgs. Those edges aren't evidence — they're descriptive.
  The grounding line stays cleanly drawn at SUPPORTED_BY.

This means the viewer's `hideUngroundedInsights` filter still works
exactly as it does today: dashed border for `grounded=false`, toggle
hides them.

---

## 3. Concrete schema diff sketches

Two files (`kg.schema.json` + `gi.schema.json`) become "two artifact
shapes over one shared ontology." That's simpler than "one merged
schema" because KG and GI run at different times (KG: ~15 s/ep, GI:
~60 s/ep with QA+NLI) and you'd rather not block one on the other.

### kg.schema.json — v2.0 (sketch)

```diff
  nodes:
- - Episode | Topic | Entity(kind: person|org)
+ - Episode | Topic | Person | Organization
+   # Note: Topic / Person / Organization id strings are shared with gi.json

  edges:
  - MENTIONS (Topic|Person|Organization → Episode)
  - RELATED_TO (reserved, still not emitted)
```

Node-level changes:
- `Entity` type goes away
- `Person` replaces `Entity(kind=person)`; same `person:{slug}` id pattern
- `Organization` replaces `Entity(kind=org)`; same `org:{slug}` id pattern
- `Topic` unchanged
- `Episode` unchanged

### gi.schema.json — v3.0 (sketch)

```diff
  nodes:
- - Podcast | Episode | Person | Topic | Insight | Quote (Entity deferred)
+ - Podcast | Episode | Person | Organization | Topic | Insight | Quote

  edges:
  - HAS_EPISODE  (already defined, now actually emitted)
  - SPOKE_IN     (Person → Episode, unchanged)
  - HAS_INSIGHT  (Episode → Insight, unchanged)
  - SUPPORTED_BY (Insight → Quote, unchanged — the contract)
  - SPOKEN_BY    (Quote → Person, unchanged)
- - MENTIONS     (Insight → Entity, was defined but Entity deferred)
+ - ABOUT             (Insight → Topic — newly emitted)
+ - MENTIONS_PERSON   (Insight → Person — newly emitted)
+ - MENTIONS_ORG      (Insight → Organization — newly emitted)
```

Why three flavors of MENTIONS_* (per type) rather than one polymorphic
MENTIONS (Insight → Person|Org)?

- Cleaner graph queries — UI can filter by edge type alone
- Avoids the `kind`/`entity_kind` discriminator pattern that bit us
  in KG v1
- The viewer can style by edge type instead of by target node type

### Shared id contract

We codify what RFC-072 already started:

| Type | ID pattern | Where defined |
|---|---|---|
| Episode | `episode:{episode_id}` | Anchor — same string across files |
| Podcast | `podcast:{slug(rss_url) or stable id}` | New, declare both shapes can emit |
| Person | `person:{slug(name)}` | Already canonical in both schemas |
| Organization | `org:{slug(name)}` | Already canonical in KG; declare GI uses same |
| Topic | `topic:{slug(label)}` | Already canonical in KG; declare GI ABOUT edges use same |
| Insight | `insight:{16-hex}` | Stays GI-only |
| Quote | `quote:{16-hex}` | Stays GI-only |

**Validation rule (new)**: when GI emits an `ABOUT` / `MENTIONS_PERSON`
/ `MENTIONS_ORG` edge, the target id must be a `topic:` / `person:` /
`org:` id, but the target node may or may not be present in the same
GI file — the corpus layer joins them. This is the cross-layer bridge
contract.

---

## 4. Migration plan (chunked, à la #1034)

Each chunk = isolated commit, independently bisectable, doesn't break
the prior commit's shape.

### Chunk 1 — define + freeze the new ontology docs

- Write `docs/architecture/corpus/ontology.md` (or extend KG's) — one
  doc describing the unified node/edge types and shared id contract
- Mark `docs/architecture/kg/ontology.md` v1.2 as superseded; add
  pointer
- Mark `docs/architecture/gi/ontology.md` v2.0 as superseded; add
  pointer
- No code changes yet — pure docs

**Risk**: zero. Docs only.

### Chunk 2 — schemas accept both old + new shape

- `kg.schema.json` v2.0: accepts `Entity` (legacy) OR `Person` + `Organization` (new)
- `gi.schema.json` v3.0: accepts the deferred-Entity language (legacy)
  OR the new ABOUT/MENTIONS_PERSON/MENTIONS_ORG edges
- Validation passes both shapes for the transition window

**Risk**: low. Permissive schemas don't break anything.

### Chunk 3 — pipeline emits new shape (KG side)

- `kg/pipeline.py::_append_topics_and_entities_from_partial`: replace
  the Entity-with-kind emission with separate Person + Organization
  node types
- `kg/filters.py::repair_entity_kind`: split into
  `repair_person_or_org` (decides node type, not a property)
- Test fixtures get regenerated for v2 shape
- v1 reads still validate against v2 schema (per chunk 2)

**Risk**: medium. Lots of test churn but the contract is mechanical.

### Chunk 4 — pipeline emits new shape (GI side)

- `gi/pipeline.py::build_artifact`: emit ABOUT edge per insight by
  matching the insight's referenced topics (the LLM already mentions
  them in the insight text; can extract via post-pass or ask in the
  prompt)
- Emit MENTIONS_PERSON / MENTIONS_ORG edges using the NER pre-pass
  output already wired in #1035 (Persons/Orgs are already discovered;
  just emit edges to them)
- Insight nodes don't need new properties — just new edges out

**Risk**: medium. New edge emission, but the underlying entity recall
is already at 100% thanks to #1035 NER pre-pass.

### Chunk 5 — migration script for existing corpora

- `scripts/migrate_kg_entity_to_person_org.py`: walks corpus, rewrites
  legacy `Entity(kind=person)` → `Person`, `Entity(kind=org)` →
  `Organization`. Same as `migrate_kg_entity_ids.py` (RFC-072) pattern.
- Optional: a forward-walk that adds ABOUT / MENTIONS_* edges to
  existing GI files where we can recover them via NER + LLM re-pass
  (this is its own cost decision; defer)

**Risk**: medium. Migration touches stored data; needs dry-run mode.

### Chunk 6 — silvers refresh

- Regenerate `silver_opus47_kg_dev_v1` shape: replace `entities[]` with
  `persons[]` + `organizations[]`
- Add `topics[]` to `silver_opus47_gi_dev_v1` so we can measure ABOUT
  edge coverage
- Score scripts (`score_kg_topic_coverage.py`, etc.) updated for new
  shape — but per the #1035 phase 3 finding, the scorer also needs to
  handle the new node types
- Re-baseline #1033/#1035/#116/#113 numbers under the new shape
  (some numbers will move because we measure different things now)

**Risk**: medium-high. This is where we lose direct comparability with
prior cohort sweeps. Worth doing — but worth flagging.

### Chunk 7 — viewer updates

- Type definitions (`web/gi-kg-viewer/src/types/artifact.ts`) updated
  for new node + edge types
- Graph stylesheet handles Person / Organization nodes (was Entity)
- Search / list / filter UI updated to use new concepts
- Grounding visualization (dashed border) stays unchanged

**Risk**: medium. UI is a separate concern but the API surface is
small.

### Chunk 8 — drop legacy support in schemas (after some bake time)

- After all corpora are migrated, schemas v3 enforce new shape only
- Migration scripts marked deprecated
- Docs cleanup

**Risk**: low if 1–7 went well.

---

## 5. Open questions (round-2)

The big ones are settled. These are the smaller mechanical calls:

### Q-2.1 — Where does the unified ontology doc live?

Three options:
- `docs/architecture/corpus/ontology.md` (new, neutral)
- Extend `docs/architecture/kg/ontology.md` to cover both
- Extend `docs/architecture/gi/ontology.md` to cover both

I'd lean toward **`docs/architecture/corpus/ontology.md` (new)** because
it's the cleanest "this is the corpus model" doc, and both KG/GI
ontology files become pointers to it.

**?** OK with that?

### Q-2.2 — How aggressive on the ABOUT edge?

Two flavors of emitting `Insight → ABOUT → Topic`:

- **(a) Post-pass match**: after extracting insights, run a semantic
  similarity match against the episode's topics, emit ABOUT edges with
  confidence scores
- **(b) Prompt the LLM to emit it**: ask the LLM during insight
  extraction to also tag each insight with topic ids it relates to

(b) is more direct but more brittle (prompt drift). (a) is more
mechanical but adds a step. Probably (a) for v2 minimum, with a
flag to upgrade to (b) later.

**?** Preference?

### Q-2.3 — Silver migration: incremental or full rebuild?

If we want the new ontology to be measurable, the silvers have to
include the new edge types. Options:
- **Incremental**: add `persons` + `organizations` arrays + topic
  references to insights as separate enrichment passes on existing
  silvers
- **Full rebuild**: regenerate silvers from scratch using Opus 4.7
  with a new prompt that produces the unified shape

Incremental is faster but inherits old labeling biases. Full rebuild
is cleaner but is a half-day of LLM time on the silver.

**?** Worth the full rebuild?

### Q-2.4 — Do we ship as RFC, ADR, or just commit?

Given the size (8 chunks, schema changes, migration script, silver
refresh, viewer updates), this feels like RFC territory. Specifically
an RFC that supersedes RFC-055 (KG core) + RFC-072 (entity migration)
+ touches RFC-049 (GI core).

But you said earlier you're allergic to ceremony, so maybe an ADR
referenced from the round-2 doc is enough, with the doc itself living
in `docs/architecture/corpus/ontology.md`.

**?** Your call on process.

### Q-2.5 — Sequencing with other work

This is a multi-week effort (8 chunks). The current branch
(`feat/autoresearch-followups-2026-06-18`) has 18 unpushed commits
spanning #1033/#1034/#1035/#116/#113 + this draft. Suggested
sequencing:

- **Push current branch first** (close the autoresearch-followups
  thread cleanly)
- **Start v2 ontology on a new branch** (`feat/corpus-ontology-v2`?)
- **Chunk 1+2 (docs + schemas) first** — they unblock everything else
  and ship low-risk

**?** OK with that order?

---

## 6. What I'm NOT proposing

Worth being explicit about scope boundaries:

- **Not merging the two artifact files.** `kg.json` + `gi.json` stay
  separate at the filesystem level; they just share the same node id
  space and ontology
- **Not introducing a graph database.** This is still flat-JSON
  artifacts + a viewer that builds the graph in memory. RFC-091's
  rejection of KG-proximity-as-retrieval stands
- **Not changing the summary stage.** The summary → bullets → metadata
  flow is separate; this only touches KG + GI extraction
- **Not touching the corpus index / LanceDB.** Vector indexing reads
  from these files but doesn't care about the schema beyond the topic
  / entity labels (which keep working)
- **Not adding new node types beyond Person / Organization.** No
  Place, no Event, no Date. Could come later; not in v2
- **Not changing the grounding contract semantics.** Quote.text stays
  verbatim, char_start/char_end stay required, `grounded` is still
  the boolean ground truth

---

## 7. Suggested next loop

1. You answer Q-2.1 through Q-2.5 (or push back on the framing)
2. I draft chunk 1 (the unified ontology doc) as a real file under
   `docs/architecture/corpus/ontology.md` — concrete, ship-ready
3. You review chunk 1 in detail, we iterate
4. Once chunk 1 lands, chunks 2–8 follow with the same review pattern

Worth saying: I think the right end-state here is meaningfully better
than where we are, and it's a clean line of thinking to follow now
while the autoresearch story is fresh. Branch the v2 ontology work,
ship current branch, take this in stride.

---

## Appendix — references

- Round-1 doc: `docs/wip/SPEC_KG_GI_ONTOLOGY_REVIEW_2026-06-20.md`
- Operator answers (this session):
  - Q1: grounding contract = load-bearing
  - Q2: viewer graph + search/list UI = primary surface
  - Q3: cross-layer bridge yes
  - Q4: unified Person concept
- Viewer evidence for grounding-as-UX:
  - `web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts:402` — ungrounded
    dashed border
  - `web/gi-kg-viewer/src/stores/graphFilters.ts:61` —
    `setHideUngrounded` filter
- Current schemas:
  - `docs/architecture/kg/kg.schema.json` (v1.2)
  - `docs/architecture/gi/gi.schema.json` (v2.0)
