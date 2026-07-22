# Unified Corpus Ontology (v2)

**Status:** v2 (RFC-097, 2026-06-20). Consolidates the prior KG v1.2
ontology and GIL v2.0 ontology into a single per-corpus shape. The
canonical [`docs/architecture/kg/ontology.md`](../kg/ontology.md) and
[`docs/architecture/gi/ontology.md`](../gi/ontology.md) are superseded
for v2.0+ but retained for archaeology.

**Source of truth.** All `kg.json` and `gi.json` artifacts MUST conform
to this ontology and the companion schemas
([`kg.schema.json`](../kg/kg.schema.json),
[`gi.schema.json`](../gi/gi.schema.json)). For design lineage see
[RFC-097](../../rfc/RFC-097-unified-kg-gi-ontology-v2.md),
[RFC-072](../../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)
(CIL + bridge), [RFC-049](../../rfc/RFC-049-grounded-insight-layer-core.md)
(GIL core), [RFC-055](../../rfc/RFC-055-knowledge-graph-layer-core.md)
(KG core), and [PRD-017](../../prd/PRD-017-grounded-insight-layer.md) /
[PRD-019](../../prd/PRD-019-knowledge-graph-layer.md).

---

## Headline invariants

1. **The graph is relational, not proximity-based.** v2 explicitly does
   not propose proximity-based ranking. The decision record is
   [RFC-091 (2026-06-03)](../../rfc/RFC-091-kg-proximity-signal.md) —
   empirical A/B testing showed −0.018 to −0.170 nDCG hits when
   co-occurrence was used as a retrieval signal. KG `MENTIONS` edges
   exist for display + relational queries (RFC-094), not ranking.
2. **The grounding contract is load-bearing.**
   `Insight.grounded = true ⇔ ≥1 SUPPORTED_BY → Quote` edge. The
   viewer styles ungrounded Insights with a dashed border
   (`web/gi-kg-viewer/src/utils/cyGraphStylesheet.ts:407` (selector `node[type = "Insight"].insight-ungrounded`)) and
   `hideUngroundedInsights` is a user-facing filter. Only the
   `SUPPORTED_BY` edge promotes `grounded=true`; no descriptive edge
   does.
3. **`kg.json` and `gi.json` remain separate files on disk.** v2
   unifies the **ontology**, not the file layout. The two pipelines
   stay independently toggleable (preserves
   [PRD-019 line 45](../../prd/PRD-019-knowledge-graph-layer.md)).
4. **Canonical IDs span artifacts.** `person:{slug}` / `org:{slug}` /
   `topic:{slug}` are owned by neither layer; they live in
   [`bridge.json`](../../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md)
   and are emitted identically by KG and GI.

---

## Two-tier edge contract

Every edge type carries an `edge_class` metadata field declared in
the schema (not on each edge instance). This tells the viewer's
stylesheet, the relational query layer, and future contributors
which edges are grounding-load-bearing vs. classification vs.
discovery.

| `edge_class` | Meaning | Examples |
| --- | --- | --- |
| **evidentiary** | Proves an Insight is grounded — load-bearing for the grounding contract. | `SUPPORTED_BY` |
| **descriptive** | Classifies / labels an Insight. Does **not** prove. Does **not** promote `grounded=true`. | `ABOUT`, `MENTIONS_PERSON`, `MENTIONS_ORG` |
| **discovery** | Co-occurrence; useful for browse + relational queries but **not** for ranking (RFC-091). | KG `MENTIONS`, `RELATED_TO` (reserved) |
| **structural** | Parent / child / containment, and durable person↔show affiliation. | `HAS_EPISODE`, `HAS_INSIGHT`, `HOSTS`, `GUESTS_ON` |
| **attribution** | Who-said-what. | `SPOKE_IN`, `SPOKEN_BY` |

**Hard rule.** Only `edge_class: evidentiary` edges decide
`Insight.grounded`. A future descriptive edge (e.g., `MENTIONS_TOPIC`,
should we add it) does not change this contract.

---

## Node types

| Node | `edge_class` interactions | Required properties | Optional |
| --- | --- | --- | --- |
| `Podcast` | source of `HAS_EPISODE`; target of `HOSTS` / `GUESTS_ON` | `title`, `rss_url` | `publisher` |
| `Episode` | target of `HAS_EPISODE`, source of `HAS_INSIGHT` | `podcast_id`, `title`, `publish_date` (ISO-8601) | `audio_url`, `duration_ms`, `feed_id` |
| `Person` | target of `SPOKEN_BY` / `MENTIONS_PERSON`, source of `SPOKE_IN` / `HOSTS` / `GUESTS_ON` | `name` | `aliases[]` |
| `Organization` | target of `MENTIONS_ORG` / KG `MENTIONS` | `name` | `aliases[]` |
| `Topic` | target of `ABOUT` / KG `MENTIONS` | `label`, `slug` | `aliases[]`, `description` |
| `Insight` | source of `SUPPORTED_BY` / `ABOUT` / `MENTIONS_PERSON` / `MENTIONS_ORG`, target of `HAS_INSIGHT` | `text`, `episode_id`, `grounded` | `confidence`, `insight_type`, `position_hint` |
| `Quote` | target of `SUPPORTED_BY`, source of `SPOKEN_BY` | `text` (**verbatim**), `episode_id`, `char_start`, `char_end`, `timestamp_start_ms`, `timestamp_end_ms`, `transcript_ref` | `speaker_id` (`person:{slug}` when diarization aligned) |

**v1 → v2 transition.** The legacy `Entity` node with `kind: person | org`
discriminator is replaced by first-class `Person` and `Organization`
node types. Permissive schemas (v2.0 KG / v3.0 GI) accept both shapes
during the bake window; chunk 9 of the v2 migration drops legacy
support (gated on ADR-101).

---

## Edge types

| Edge | From → To | `edge_class` | Required | Notes |
| --- | --- | --- | --- | --- |
| `HAS_EPISODE` | `Podcast → Episode` | structural | — | NEW in v2 (was descoped from #849; now first-class) |
| `SPOKE_IN` | `Person → Episode` | attribution | — | — |
| `HAS_INSIGHT` | `Episode → Insight` | structural | — | — |
| `SUPPORTED_BY` | `Insight → Quote` | **evidentiary** | — | **The grounding contract.** ≥1 such edge ⇒ `Insight.grounded = true`. |
| `SPOKEN_BY` | `Quote → Person` | attribution | — | Only emitted when diarization aligns. |
| `ABOUT` | `Insight → Topic` | descriptive | — | NEW in v2 emission (defined in v1, never emitted). |
| `MENTIONS_PERSON` | `Insight → Person` | descriptive | — | NEW in v2; replaces deferred v1.1 `MENTIONS → Entity` for Persons. |
| `MENTIONS_ORG` | `Insight → Organization` | descriptive | — | NEW in v2; replaces deferred v1.1 `MENTIONS → Entity` for Orgs. |
| `MENTIONS` (KG) | `Topic` / `Person` / `Organization` → `Episode` | discovery | — | KG co-occurrence edge. **Not** retained for retrieval (RFC-091). |
| `RELATED_TO` | `Topic ↔ Topic` | discovery | — | Reserved in schema; not emitted by the default pipeline. |

**Descriptive edges carry an optional `confidence` (0.0–1.0) property**
from the semantic-match post-pass that emits them.

---

## Identity (ID) rules

| Node | ID pattern | Scope |
| --- | --- | --- |
| `Podcast` | `podcast:{slug}` | Global (slug of feed title/rss host) |
| `Episode` | `episode:{episode_id}` | Per-feed (RSS GUID family); shared between KG and GI |
| `Person` | `person:{slug(name)}` | Global; merged across episodes in combined graphs. **Legacy**: `entity:person:{slug}` and `speaker:{slug}` — migrate with `scripts/migrate_kg_entity_to_person_org.py` and `scripts/migrate_gi_speaker_to_person.py`. |
| `Organization` | `org:{slug(name)}` | Global. **Legacy**: `entity:organization:{slug}` — migrate with `scripts/migrate_kg_entity_to_person_org.py`. |
| `Topic` | `topic:{slug(label)}` | Global; identical key in KG and GI artifacts |
| `Insight` | `insight:{16-hex}` | Per-episode; SHA-256 over `(episode_id, index, insight_text prefix)` |
| `Quote` | `quote:{16-hex}` | Per-episode; SHA-256 over `(episode_id, quote_index, text prefix, char_start, char_end)` |

**Slug rule.** Lowercase, hyphenated, max length capped in
`src/podcast_scraper/identity/slugify.py`. Non-empty in all artifacts.

**Cross-layer joining.** Per-episode `bridge.json` declares the
canonical IDs the episode's KG and GI artifacts share. The
`CorpusGraph` (`src/podcast_scraper/search/corpus_graph.py`) composes
cross-layer queries in-memory. v2 additionally **materializes** the
descriptive cross-layer edges (ABOUT / MENTIONS_PERSON / MENTIONS_ORG)
in per-artifact JSON, so the viewer + relational query layer
(RFC-094) can render the relationships without rebuilding the graph.

### `MENTIONS_PERSON` emit paths

Two routes produce typed `Insight → Person` edges. Both feed the same
JSON shape; the viewer cannot tell them apart at read time.

1. **LLM emit** (cloud_thin, cloud_balanced, local_dgx_*) — the
   summarization LLM directly returns mentioned-person spans at
   insight-emit time; the bridging layer types them as
   `MENTIONS_PERSON` against KG Person ids.
2. **Post-pass enrichment** (airgapped, airgapped_thin) — the LLM is
   summary-only (BART-small / SummLlama), so insights are emitted
   without explicit mention spans. The `apply_typed_mentions_and_rewrite_gi`
   post-pass (`gi/relational_edges.py`) walks every Insight's text
   and matches against the KG Person index. Two matchers run in
   order:

   - **Whole-word regex** on each KG Person `name`. Cheap and
     deterministic given the text. Misses paraphrased fragments
     ("Maya" when the KG entry is "Maya Hutchinson").
   - **spaCy NER pass** (opt-in via `cfg.gi_typed_mentions_use_ner`,
     default off; default-on for airgapped + airgapped_thin
     profiles). Extracts PERSON spans and matches them by
     token-subset against KG Person names. Reuses the spaCy model
     already loaded for speaker detection — no extra load cost.

   **Shared-surname disambiguation.** When a single-token span
   (e.g. "Trump") token-subset-matches MULTIPLE KG Persons (Donald
   Trump AND Eric Trump in the same artifact), the resolver refuses
   to pick one and emits no edge. Multi-token spans resolve
   normally. Validated on the prod-v2 corpus (209 GI files,
   2026-06-24): operator-labelled 50-row sample showed 47 TP / 3
   AMBIGUOUS / 0 FP. See ADR-102 for the `_retro_audit` audit-trail
   pattern used when sweeping the post-pass over existing corpora.

### `MENTIONS_ORG` + `Organization` under LLM-free profiles

`Organization` nodes under airgapped / airgapped_thin land via the
**KG ORG post-pass** (`gi_typed_mentions_use_ner: true` flips on the
GI side; `kg_organizations_use_ner: true` lands the KG nodes — #1058
chunk 1). Walks each Insight's text with spaCy NER, extracts `ORG`
spans, and adds an `Organization` node per unique slug. The `MENTIONS_ORG`
edges then land via the same NER pass that emits `MENTIONS_PERSON`
(extended in #1058 chunk 2 to filter `PERSON | ORG`, picking the typed
edge from the entity-index `kind`).

### Cross-show concept Topics under LLM-free profiles

Per-show `Topic` labels under airgapped are derived from BART /
SummLlama bullets — they're literally bullet text, so two shows
discussing the same concept get different labels and never link.
A corpus-level post-pass (`kg_topic_corpus_clustering: true`,
issue #1058 chunk 3) clusters all per-show Topic labels via
sentence-transformers cosine similarity (the same MiniLM model the
ABOUT-edge layer uses). Clusters that span ≥2 distinct episodes
produce a synthetic `concept:topic-{slug}` Topic (`is_concept: true`)
plus `RELATED_TO` edges from every member Topic. The concept-Topic
lives in each source KG artifact (duplicated with stable id for
idempotency) so `corpus_graph` resolves the edge target on any join.

Together with the live NER ORG post-pass and the GI MENTIONS_ORG
extension above, this means an airgapped corpus carries the
data every connectivity surface needs — including
`cross_show_synthesis` — without ever calling a cloud LLM.

The corpus-level pass auto-fires from the workflow orchestrator
(`Step 16`, after every per-episode finalize) on profiles where
`kg_topic_corpus_clustering: true`. It can also be triggered
manually for re-runs and experimentation:

```bash
.venv/bin/python -m podcast_scraper cluster-corpus-topics \
    --output-dir <corpus>
```

---

## Insight properties (v2 additive)

Both fields are **Optional** and land additively (RFC-072 §2 v1.1
specification, shipped by RFC-097 v2 chunk 5).

### `insight_type`

LLM-classified enum on each `Insight`. Powers Position Tracker
filtering and Person Profile categorization.

| Value | Meaning | Example |
| --- | --- | --- |
| `claim` | factual assertion | "AI regulation will lag innovation by 3–5 years" |
| `recommendation` | prescriptive | "Founders should ship before raising" |
| `observation` | descriptive without assertion | "Conferences this year felt smaller" |
| `question` | interrogative | "But is the model actually getting better?" |
| `unknown` | classifier couldn't decide | — |

Provider-quality measured per-provider before silver regen (chunk 5
gate of v2 implementation).

### `position_hint`

Float in `[0.0, 1.0]`. Mean Quote start time relative to episode
duration. Powers Position Tracker timeline + temporal ordering.

**4-step computation waterfall** (`src/podcast_scraper/gi/position_hint.py`):

1. `Episode.duration_ms` from RSS `<itunes:duration>` (~86% prod
   corpus coverage today)
2. Last segment's `end × 1000` from `*.segments.json` (~99.9% — every
   transcribed episode has segments)
3. `max(Quote.timestamp_end_ms)` across the episode's Quotes
   (lower-bound; preserves ordering)
4. Skip emission (field is Optional; < 0.1% edge case)

`position_hint = mean(Quote.timestamp_start_ms) / duration_ms`.

---

## Required artifact root fields

Both `kg.json` and `gi.json` carry:

- `schema_version` — KG: `2.0`+ ; GI: `3.0`+ for v2 shape. Permissive
  validators accept legacy `1.x` / `2.0` during the v2 bake window.
- `model_version` — Model identifier used for extraction (GI only;
  KG includes equivalent under `extraction.model_version`).
- `prompt_version` — Prompt version (enables A/B testing).
- `episode_id` — Episode identifier (shared between KG and GI).
- `nodes[]` — All nodes.
- `edges[]` — All edges.

**KG-specific** (`kg.json`):

- `extraction.extracted_at` — ISO-8601 timestamp.
- `extraction.transcript_ref` — Relative transcript path or label
  used in extraction.

---

## Provenance

ML-derived nodes (Insight, Quote, Topic, Person, Organization)
SHOULD carry:

- `confidence` (0.0–1.0) — Extraction certainty (not factual truth).

The root artifact MUST carry `model_version` + `prompt_version`.

---

## Minimal worked example

A grounded Insight + descriptive edges + insight_type + position_hint:

```json
{
  "schema_version": "3.0",
  "model_version": "claude-opus-4-7",
  "prompt_version": "v3.1",
  "episode_id": "episode:abc123",
  "nodes": [
    {"id": "podcast:the-journal", "type": "Podcast",
     "properties": {"title": "The Journal", "rss_url": "https://feeds.example.com/the-journal"}},
    {"id": "episode:abc123", "type": "Episode",
     "properties": {"podcast_id": "podcast:the-journal", "title": "AI Regulation",
                    "publish_date": "2026-02-03T00:00:00Z", "duration_ms": 2700000}},
    {"id": "person:sam-altman", "type": "Person", "properties": {"name": "Sam Altman"}},
    {"id": "org:openai", "type": "Organization", "properties": {"name": "OpenAI"}},
    {"id": "topic:ai-regulation", "type": "Topic",
     "properties": {"label": "AI Regulation", "slug": "ai-regulation"}},
    {"id": "insight:episode:abc123:a1b2c3d4", "type": "Insight",
     "properties": {"text": "AI regulation will lag innovation by 3-5 years",
                    "episode_id": "episode:abc123", "grounded": true,
                    "insight_type": "claim", "position_hint": 0.34},
     "confidence": 0.85},
    {"id": "quote:episode:abc123:e5f6g7h8", "type": "Quote",
     "properties": {"text": "Regulation will lag innovation by 3-5 years. That's my prediction.",
                    "episode_id": "episode:abc123", "speaker_id": "person:sam-altman",
                    "char_start": 10234, "char_end": 10302,
                    "timestamp_start_ms": 918000, "timestamp_end_ms": 933000,
                    "transcript_ref": "transcript.json"}}
  ],
  "edges": [
    {"type": "HAS_EPISODE",     "from": "podcast:the-journal",                "to": "episode:abc123"},
    {"type": "SPOKE_IN",        "from": "person:sam-altman",                  "to": "episode:abc123"},
    {"type": "HAS_INSIGHT",     "from": "episode:abc123",                     "to": "insight:episode:abc123:a1b2c3d4"},
    {"type": "SUPPORTED_BY",    "from": "insight:episode:abc123:a1b2c3d4",    "to": "quote:episode:abc123:e5f6g7h8"},
    {"type": "SPOKEN_BY",       "from": "quote:episode:abc123:e5f6g7h8",      "to": "person:sam-altman"},
    {"type": "ABOUT",           "from": "insight:episode:abc123:a1b2c3d4",    "to": "topic:ai-regulation",
     "properties": {"confidence": 0.79}},
    {"type": "MENTIONS_PERSON", "from": "insight:episode:abc123:a1b2c3d4",    "to": "person:sam-altman",
     "properties": {"confidence": 0.92}},
    {"type": "MENTIONS_ORG",    "from": "insight:episode:abc123:a1b2c3d4",    "to": "org:openai",
     "properties": {"confidence": 0.71}}
  ]
}
```

The Insight above is **grounded** because it has ≥1 `SUPPORTED_BY`
edge. The `ABOUT` + `MENTIONS_PERSON` + `MENTIONS_ORG` edges classify
it but **do not** affect the `grounded` field. Removing the
`SUPPORTED_BY` edge would force `grounded: false` and trip the
viewer's dashed-border styling.

---

## Known limitations carried over (parked for v3)

- **Cross-layer Podcast id derivation**: the KG pipeline derives
  `podcast:{slug(feed_id)}` from the workflow's `feed_id` (e.g.
  `rss_example_com_abc123`), while the GI post-pass
  `add_episode_show_edges` derives `podcast:{slug(show_title)}` from the
  feed's `title` (e.g. `the-journal`). These may produce different
  canonical IDs for the same show. v2 ships both because each layer is
  the cheaper source-of-truth for its own emission; v3 should converge
  both on `podcast:{slug(title)}` once feed title is threaded through
  the workflow's KG call. Workaround today: keep the KG `MENTIONS`
  edges + GI `HAS_EPISODE` edges independent and rely on the bridge
  artifact for cross-layer joins.
- **Per-provider insight_extraction prompts** (`prompts/{anthropic,
  gemini,openai,deepseek,grok,mistral,ollama}/insight_extraction/v1.j2`)
  still emit plain-text insights with no structured `insight_type`.
  Outputs degrade to `insight_type: "unknown"` via the pipeline's
  default normalizer. The megabundle / extraction-bundled path (updated
  in chunk 5) does emit structured `insight_type`. Updating each
  per-provider prompt is a v2.1 follow-up (additive, schema-safe).

## Versioning + supersession

| Schema family | v1.x | v2.x | v3.x |
| --- | --- | --- | --- |
| KG | `1.0` / `1.1` / `1.2` (legacy `Entity(kind=...)`) | **`2.0` (this doc)** — `Person` + `Organization` + `Podcast` first-class; `HAS_EPISODE`; `edge_class` metadata | reserved (legacy rejection per ADR-101, chunk 9) |
| GI | `1.0` (no `Person`); `2.0` (`Person` + `insight_type`/`position_hint` defined but not emitted) | n/a | **`3.0` (this doc)** — `Organization`; `ABOUT` + `MENTIONS_PERSON` + `MENTIONS_ORG` emitted; `insight_type` + `position_hint` shipped; `edge_class` metadata |

Permissive transition: schemas v2.0 (KG) + v3.0 (GI) accept both
legacy and new shape. Chunk 9 of the RFC-097 migration drops legacy
support after ≥2–4 weeks of production operation under v2 — captured
in [ADR-101](../../adr/ADR-101-drop-legacy-kg-gi-shape.md).

**Where the old docs went.**
[`docs/architecture/kg/ontology.md`](../kg/ontology.md) and
[`docs/architecture/gi/ontology.md`](../gi/ontology.md) are
superseded by this doc for v2.0+ but retained as v1 archaeology.
Bump `schema_version`, the relevant JSON Schema, and this file
together for any future breaking change.

---

## Related

- [RFC-097](../../rfc/RFC-097-unified-kg-gi-ontology-v2.md) — design + chunked plan + cross-doc closures
- [RFC-072](../../rfc/RFC-072-canonical-identity-layer-cross-layer-bridge.md) — CIL + bridge (the infrastructure this doc formalizes)
- [RFC-091](../../rfc/RFC-091-kg-proximity-signal.md) — KG proximity rejection decision record
- [RFC-094](../../rfc/RFC-094-search-powered-surfaces-query-layer.md) — relational query layer over the v2-materialized edges
- [PRD-028](../../prd/PRD-028-position-tracker.md) / [PRD-029](../../prd/PRD-029-person-profile.md) — viewer surfaces that consume `insight_type` + `position_hint`
- [GIL Guide](../../guides/GROUNDED_INSIGHTS_GUIDE.md) / [KG Guide](../../guides/KNOWLEDGE_GRAPH_GUIDE.md) — user-facing references
