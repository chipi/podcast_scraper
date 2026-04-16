# RFC-072: Canonical Identity Layer & Cross-Layer Bridge

- **Status**: Draft
- **Authors**: Marko
- **Stakeholders**: Core team
- **Related PRDs**:
  - `docs/prd/PRD-017-grounded-insight-layer.md`
  - `docs/prd/PRD-019-knowledge-graph-layer.md`
  - `docs/prd/PRD-021-semantic-corpus-search.md`
- **Related ADRs**:
  - `docs/adr/ADR-052-separate-gil-and-kg-artifact-layers.md`
  - `docs/adr/ADR-053-grounding-contract-for-evidence-backed-insights.md`
  - `docs/adr/ADR-061-faiss-phase-1-with-post-filter-metadata.md`
  - `docs/adr/ADR-062-sentence-boundary-transcript-chunking.md`
- **Related RFCs**:
  - `docs/rfc/RFC-049-grounded-insight-layer-core.md`
  - `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md` (single-layer consumption;
    cross-layer use cases now live here)
  - `docs/rfc/RFC-055-knowledge-graph-layer-core.md`
  - `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md` (single-layer consumption;
    cross-layer use cases now live here)
  - `docs/rfc/RFC-061-semantic-corpus-search.md`
  - `docs/rfc/RFC-073-enrichment-layer-architecture.md` -- enrichment layer; provides
    `grounding_rate`, `topic_cooccurrence`, `temporal_velocity`, and
    `nli_contradiction` enrichers that feed PRDs 026--029
- **Related Documents**:
  - `docs/architecture/gi/ontology.md`
  - `docs/architecture/kg/ontology.md`
- **GitHub (implementation tracking)**:
  This RFC is the authoritative spec. The issues below were derived from it; if an
  issue body drifts, reconcile the issue to match the RFC.
  - [Issue #524](https://github.com/chipi/podcast_scraper/issues/524) — Phase 1
    (canonical slugifier, wire builders; no artifact ID migration yet)
  - [Issue #525](https://github.com/chipi/podcast_scraper/issues/525) — Phase 2
    (ontology migration **and** GIL v1.1 `insight_type` + `position_hint` per Sections
    2a–2b; viewer compat)
  - [Issue #526](https://github.com/chipi/podcast_scraper/issues/526) — Phase 3
    (`bridge.json` + viewer awareness)
  - [Issue #527](https://github.com/chipi/podcast_scraper/issues/527) — Phase 4
    (cross-layer query API)
  - [Issue #528](https://github.com/chipi/podcast_scraper/issues/528) — Phase 5
    (chunk-to-Insight lift; gated on char-offset verification)

## Implementation status (repository)

Phases **1–5** are implemented in the current tree: canonical identity slugging and
builders; per-episode **`bridge.json`** emission in the metadata pipeline; **CIL** read
routes under **`/api/persons/*`** and **`/api/topics/*`**; semantic-search **transcript
lift** (`lifted`) and **`verify-gil-chunk-offsets`** with **`merged_episode_gi_paths`**
for feed-nested metadata; shared path rules in **`builders/rfc072_artifact_paths.py`**
and GIL edge-type normalisation in **`gi/edge_normalization.py`**. This RFC remains the
normative design; if GitHub issue bodies drift, reconcile them to match this document.

**Status** (header above): treat as **Draft** until stakeholders sign off; behaviour is
nevertheless exercised in CI (acceptance + offset verification) and tests.

---

## Abstract

GIL, KG, and semantic search were built as independent layers. Each is solid on its
own: GIL delivers grounded insights backed by verbatim quotes; KG maps entities, topics,
and typed relationships; semantic search finds relevant transcript passages by meaning.
But the most valuable things the system can do require all three working together — and
today they cannot, because they use divergent identity schemes for the same real-world
concepts and have no explicit join path between them.

This RFC introduces a **Canonical Identity Layer (CIL)** — a shared vocabulary of global
identifiers for persons, organisations, and topics — and a **per-episode bridge artifact**
(`bridge.json`) that makes cross-layer joins explicit at write time. Together, they form
the foundation for two flagship use cases that demonstrate the full value of the combined
stack:

1. **Position Tracker** — how a person's thinking on a topic evolved across episodes,
   grounded in timestamped quotes.
2. **Person Profile** — a structured overview of a guest's corpus presence: topics
   discussed, Insight counts per topic, episode appearances, and cross-navigation.

Alongside the CIL, this RFC introduces two additive GIL v1.1 fields:

- **`insight_type`** — an optional classification on every Insight node (`claim`,
  `recommendation`, `observation`, `question`, `unknown`) that makes the flagship
  query patterns type-aware: the Position Tracker can filter to claims, and the Person
  Profile can surface positions by statement type.
- **`position_hint`** — a normalised float (`0.0`-`1.0`) derived from supporting Quote
  timestamps and episode duration, giving each Insight a temporal position within its
  episode. No extraction change needed — pure arithmetic on existing data.

The CIL, bridge, `insight_type`, and `position_hint` work entirely on the existing
filesystem JSON artifact stack. No database prerequisite. GIL and KG remain separate
artifacts with independent schemas (per ADR-052). The bridge is the seam, not a merge.

---

## Problem Statement

GIL and KG were designed in parallel and independently developed their own identity
schemes for the same real-world concepts:

| Concept | GIL ID | KG ID |
| --- | --- | --- |
| A person | `speaker:{slug}` | `entity:person:{slug}` |
| An organisation | *(not modelled)* | `entity:organization:{slug}` |
| A topic | `topic:{slug}` *(deferred)* | `topic:{slug}` |

These schemes are close but not the same. The `entity:` prefix in KG is an
implementation detail of that layer leaking into what should be a shared identity space.
`speaker:` in GIL captures only the *speaking* role of a person, not their existence as
a real-world entity. The result: the same person appears under two different IDs in two
artifacts, and the system has no way to join across them without fragile string
heuristics.

Semantic search (RFC-061, ADR-061/062) compounds the problem. The FAISS index stores
chunks with episode-level metadata but no references to GIL Insights or KG entities.
Retrieved chunks cannot be lifted to structured, attributed results without a resolution
path.

This siloing blocks the most valuable use cases the current stack is capable of:

1. **Opinion and claim tracking** — "What does guest X actually believe about topic Y,
   and has it changed across episodes?" Requires resolving a person identity stably
   across episodes and joining their Insights by topic.

2. **Controversy radar** — "Which guests in my corpus hold contradictory views on the
   same topic?" Requires shared `topic:{slug}` as a stable join key across Insights
   from different speakers across different episodes.

3. **Follow-the-thread** — "How has the conversation about topic X evolved across 50
   episodes, chronologically?" Requires `topic:{slug}` anchoring both KG MENTIONS edges
   (which episodes) and GIL ABOUT/Insight nodes (what was actually said).

4. **Person profile** — "What has person X talked about across all episodes in my
   corpus, and how deeply?" Requires `person:{slug}` as a corpus-wide anchor joining
   Insights, Quotes, Topics, and episode metadata.

None of these require a database. They require **a shared identity contract** and a
**lightweight bridge artifact** that makes the join explicit at write time rather than
guessed at query time.

---

## Vision: Flagship Use Cases

The CIL and bridge are not ends in themselves. They are the foundation for concrete
features that demonstrate the value of GIL + KG + semantic search working as a
coherent system. This section defines the two flagship use cases in detail: what they
produce, who uses them, what each layer contributes, and where the current boundaries
are.

### Flagship 1: Position Tracker

**The question it answers:** "How has person X's thinking on topic Y evolved across
episodes — and when did it change?"

**Target users:** Journalists researching a public figure's evolving stance. Researchers
tracking how expert opinion shifts over time. Curious listeners who want more than
"find me episodes about X."

**Why it matters:** The difference between a glorified quote search and a Position
Tracker is *narrative structure over time*. The output is not "here are 5 things Satya
Nadella said about AI" — it is "here is how his thinking shifted from 2023 to 2025,
and here is the moment it changed, grounded in his own words."

**What each layer contributes:**

| Layer | Contribution |
| --- | --- |
| CIL | `person:satya-nadella` + `topic:ai-safety` as stable join keys across episodes |
| GIL | Insights attributed to that person on that topic, each grounded in timestamped quotes. `insight_type` (Section 2a) enables filtering to `claim` type Insights for a focused position arc. `position_hint` (Section 2b) adds intra-episode ordering so Insights within a single episode show argumentative progression — setup vs conclusion. |
| KG | Episode publish dates, entity metadata (role, org affiliation) |
| Semantic Search | Entry point — user searches a person + topic, FAISS returns chunks, chunks lift to Insights via the bridge |
| Bridge | The per-episode join that makes "all Insights by person X on topic Y" a single corpus-wide query |

**Target output shape:**

```json
{
  "person": {
    "id": "person:satya-nadella",
    "display_name": "Satya Nadella"
  },
  "topic": {
    "id": "topic:ai-safety",
    "display_name": "AI Safety"
  },
  "position_arc": [
    {
      "episode_id": "episode:abc123",
      "publish_date": "2023-06-15T00:00:00Z",
      "podcast_title": "Lex Fridman Podcast",
      "insights": [
        {
          "id": "insight:a1b2c3d4",
          "text": "AI safety is important but should not slow down innovation",
          "insight_type": "claim",
          "position_hint": 0.15,
          "grounded": true,
          "confidence": 0.88,
          "supporting_quotes": [
            {
              "text": "We need to move fast. Safety is a priority, but paralysis is not safety.",
              "timestamp_start_ms": 1200000,
              "timestamp_end_ms": 1215000
            }
          ]
        }
      ]
    },
    {
      "episode_id": "episode:def456",
      "publish_date": "2025-02-20T00:00:00Z",
      "podcast_title": "Hard Fork",
      "insights": [
        {
          "id": "insight:e5f6g7h8",
          "text": "AI safety requires mandatory external audits before deployment",
          "insight_type": "claim",
          "position_hint": 0.85,
          "grounded": true,
          "confidence": 0.91,
          "supporting_quotes": [
            {
              "text": "I've changed my mind. We need third-party audits. Full stop.",
              "timestamp_start_ms": 840000,
              "timestamp_end_ms": 852000
            }
          ]
        }
      ]
    }
  ],
  "position_count": 2,
  "episode_count": 2,
  "date_range": {
    "earliest": "2023-06-15",
    "latest": "2025-02-20"
  }
}
```

**What the bridge enables:** The `position_arc` is assembled by scanning `bridge.json`
files across all episodes, filtering for episodes where both `person:satya-nadella` and
`topic:ai-safety` appear in GIL, then loading each episode's `gi.json` to extract the
relevant Insights and Quotes. Without the bridge, this requires loading every `gi.json`
in the corpus. With the bridge, the scan is a fast filter on small files.

**Current boundary — what the bridge does NOT do:**

Position *change detection* -- recognising that the 2025 Insight contradicts the 2023
Insight -- is a separate capability for a future analysis layer. The bridge makes the
*collection* possible; the analysis layer adds the *intelligence*.

### Flagship 2: Person Profile

**The question it answers:** "What has person X talked about across this corpus, and
how deeply?"

**Target users:** Podcast listeners and researchers exploring a guest's corpus
footprint. Producers prepping for a recording session who want to see what a guest
has discussed publicly across all episodes.

**Why it matters:** Without a person-anchored view, users must search episode by
episode or scan graph nodes to understand a guest's presence. A structured profile
generated from the corpus shows the full picture instantly: which topics they discuss,
how many Insights exist per topic, and where to drill deeper.

**What each layer contributes:**

| Layer | Contribution |
| --- | --- |
| CIL | `person:` as the anchor for the entire profile |
| GIL | All Insights and Quotes attributed to this person across all episodes. `insight_type` (Section 2a) enables surfacing `claim` and `recommendation` types as known positions. `position_hint` (Section 2b) serves as a tiebreaker -- later statements on a topic are more likely to represent the guest's settled position. |
| KG | Entity relationships (org affiliation, co-appearances), topic associations, episode metadata |
| Semantic Search | Optional -- "find episodes where this guest was discussed but was not present" (mentions without appearance) |
| Bridge | The join that makes "everything person X has said across 50 episodes" a single query |

**Target output shape:**

```json
{
  "path": "/path/to/corpus",
  "person_id": "person:sam-altman",
  "person_display_name": "Sam Altman",
  "appearances": 8,
  "date_range": {
    "earliest": "2023-01-10",
    "latest": "2026-03-15"
  },
  "topic_display_names": {
    "topic:ai-regulation": "AI Regulation",
    "topic:open-source": "Open Source"
  },
  "topics": {
    "topic:ai-regulation": {
      "insight_count": 5,
      "insights": []
    },
    "topic:open-source": {
      "insight_count": 3,
      "insights": []
    }
  }
}
```

**Current boundary -- what the bridge does NOT do:**

Cross-podcast coverage (what the guest said on *other* shows) requires the corpus to
contain multiple podcasts. The CIL design supports this -- slugs are feed-independent --
but the value scales with corpus breadth.

### Use Cases Enabled but Not Flagshipped

The CIL and bridge also enable these use cases, which are not detailed here but become
straightforward once the foundation is in place:

- **Controversy radar** -- surface topic + person pairs where Insights from different
  persons contradict each other. Requires a future analysis layer.
- **Follow-the-thread** — chronological topic evolution across episodes. A simpler
  variant of the Position Tracker without the person filter.
- **Gap analysis** — topics that appear in audience questions (if captured) but have
  zero Insights in the corpus. Requires an external signal (audience questions) but the
  bridge provides the "what has been covered" side.
- **Timestamped claim verification** — flag factual claims in Insights and link to
  external sources. Requires an external fact-checking signal but the bridge provides
  the structured claim inventory.

---

## Goals

1. **Define a Canonical Identity Layer (CIL)** with three global ID namespaces —
   `person:`, `org:`, `topic:` — owned by neither GIL nor KG, referenced by both.

2. **Migrate GIL** to replace `speaker:{slug}` with `person:{slug}` (clean-slate
   tolerance confirmed; additive migration path provided for existing artifacts).
   Simultaneously add two additive v1.1 fields on Insight nodes: `insight_type`
   (Section 2a) for type-aware query patterns, and `position_hint` (Section 2b) for
   intra-episode temporal ordering — both improving the flagship use cases.

3. **Migrate KG** to replace `entity:person:{slug}` and `entity:organization:{slug}`
   with `person:{slug}` and `org:{slug}` respectively.

4. **Emit a `bridge.json` artifact per episode** during the existing pipeline build
   pass, declaring all canonical IDs that appear in that episode and their GIL/KG
   references — no database required, works entirely on the filesystem artifact stack.

5. **Define a slug contract** — one canonical slugifier, one codebase location, used by
   both GIL and KG builders — so `"Lex Fridman"` always produces `person:lex-fridman`
   regardless of which layer emits it.

6. **Enable chunk-to-Insight lifting** in semantic search results by aligning FAISS
   chunk metadata `char_start`/`char_end` offsets with GIL Quote offsets (both reference
   the same transcript text), making structured enrichment of search results possible
   without pipeline changes to FAISS.

7. **Provide the data foundation for the Position Tracker and Person Profile** — the
   bridge makes the cross-layer, cross-episode joins these features require possible
   with the current architecture.

---

## Constraints & Assumptions

**Constraints:**

- Must work entirely on the filesystem JSON artifact stack — no database prerequisite.
  Postgres projection (ADR-054) is a future consumer of this, not a dependency.
- `bridge.json` is a **third artifact** per episode — `gi.json` and `kg.json` schemas
  are updated (clean slate tolerance confirmed) but remain structurally independent.
- GIL and KG remain separate layers (per ADR-052). The bridge joins them without
  merging them. The merge decision is explicitly deferred — real usage of the bridge
  will inform whether separation continues to earn its keep.
- Semantic search layer (FAISS index) is **not rebuilt** as part of this RFC. The bridge
  enables lifting at query time; FAISS metadata alignment is a follow-up (see Known
  Limitations).
- Slugifier must be deterministic, idempotent, and produce non-empty strings.
- The flagship use cases (Position Tracker, Person Profile) define the *data foundation*
  and *query patterns*. Advanced analysis capabilities are a future extension.
- `insight_type` and `position_hint` are additive (GIL v1.1). Existing `gi.json`
  artifacts without these fields remain valid — consumers treat missing `insight_type`
  as `"unknown"` and missing `position_hint` as `null`. No migration script needed for
  either field. `position_hint` requires `episode_duration_ms` from the KG Episode
  node; it is `null` when duration is unavailable.

**Assumptions:**

- GIL and KG are both produced per episode by the existing build pipeline; `bridge.json`
  is emitted in the same pass after both are written.
- Transcript character offsets used in GIL Quote nodes and FAISS chunk metadata reference
  the same normalised transcript text. **This must be verified before implementing the
  chunk-to-Insight lift** (see Known Limitations, section 1).
- `topic:{slug}` is already effectively shared (both layers use the same pattern); this
  RFC formalises it rather than changes it.
- Name normalisation happens *before* slugification. The LLM extraction step in both
  GIL and KG builders is responsible for producing a consistent canonical name string
  (e.g. "Sam Altman" not "Sam" or "Samuel Altman"). The slugifier is deterministic for
  identical inputs but does not perform fuzzy matching.

---

## Design & Implementation

### 1. Canonical Identity Layer (CIL) — Shared Namespace

The CIL defines three global ID prefixes. These are not owned by GIL or KG. They are
a shared vocabulary that both layers reference.

| Prefix | Represents | Example |
| --- | --- | --- |
| `person:{slug}` | A real-world individual | `person:lex-fridman` |
| `org:{slug}` | A real-world organisation | `org:openai` |
| `topic:{slug}` | An abstract subject or theme | `topic:ai-regulation` |

**What does NOT change:**

- `episode:{episode_id}` — already shared, unchanged.
- `podcast:{slug}` — already shared, unchanged.
- `insight:{16-hex}`, `quote:{16-hex}` — GIL-internal IDs, unchanged.

**Canonical slugifier contract:**

One function, one location: `podcast_scraper/identity/slugify.py`.
Both the GIL builder and KG builder import from this module. No local slug
implementations.

```python
# podcast_scraper/identity/slugify.py

import re
import unicodedata


def slugify(text: str) -> str:
    """
    Canonical slugifier for CIL identifiers.

    Behaviour:
    - Unicode normalise (NFKD), strip non-ASCII (so "Jose" from "Jose",
      "Bjork" from "Bjork" — diacritics are intentionally dropped for
      stable cross-system compatibility)
    - Lowercase
    - Replace whitespace and punctuation with hyphens
    - Collapse consecutive hyphens
    - Strip leading/trailing hyphens
    - Raise ValueError if result is empty (preserves original input
      in error message for diagnostics)
    """
    original = text
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    if not text:
        raise ValueError(
            f"Slug is empty after normalisation of: {original!r}"
        )
    return text


def person_id(name: str) -> str:
    return f"person:{slugify(name)}"


def org_id(name: str) -> str:
    return f"org:{slugify(name)}"


def topic_id(label: str) -> str:
    return f"topic:{slugify(label)}"
```

---

### 2. GIL Ontology Migration

**Change:** Replace `speaker:{slug}` with `person:{slug}` throughout `gi.json`.

The `Speaker` node type is renamed to `Person` in the GIL ontology. The grounding
contract, Quote properties, SPOKEN_BY edges, and all other GIL structures are unchanged.

| Before | After |
| --- | --- |
| `"id": "speaker:lex-fridman"` | `"id": "person:lex-fridman"` |
| `"type": "Speaker"` | `"type": "Person"` |
| `Quote.speaker_id: "speaker:lex-fridman"` | `Quote.speaker_id: "person:lex-fridman"` |

**`aliases` field** is retained on the Person node (was on Speaker) — important for
entity resolution when names appear with variations across episodes.

**Migration of existing `gi.json` artifacts:** A one-time migration script
(`scripts/migrate_gi_speaker_to_person.py`) rewrites `speaker:` prefixes to `person:`
in all existing artifacts. Idempotent (safe to re-run). Back up the corpus before
running.

#### 2a. Insight Type Enum (GIL v1.1 — Additive)

An optional `insight_type` field is added to every Insight node in `gi.json`. This is
an additive v1.1 change — no existing field is removed or renamed, and existing
artifacts remain valid (missing `insight_type` is treated as `"unknown"`).

**Why it matters for the flagships:**

- **Position Tracker**: A chronological arc of Insights is useful, but an arc filtered
  to `claim` type Insights is dramatically more useful. "I wonder if AI will replace
  jobs" (question) carries different weight than "AI will replace 40% of jobs by 2030"
  (claim) when tracking how a person's *stated positions* evolve. Without `insight_type`,
  the consumer must mentally classify every Insight in the arc.
- **Person Profile**: The topic overview benefits from distinguishing `claim` and
  `recommendation` type Insights -- those are the positions a user wants to see when
  exploring a person's corpus footprint.

**Enum values:**

| Value | Semantics | Example |
| --- | --- | --- |
| `claim` | A stated position, belief, or factual assertion | "AI regulation will lag innovation by 3-5 years" |
| `recommendation` | An explicit suggestion or call to action | "Companies should adopt mandatory external audits" |
| `observation` | A descriptive or analytical remark without a stance | "The EU has moved faster than the US on AI policy" |
| `question` | A question posed by the speaker (rhetorical or genuine) | "Can we really trust self-regulation?" |
| `unknown` | Default — not yet classified (existing artifacts, extraction failure) | *(implicit for pre-v1.1 artifacts)* |

**Schema change in `gi.json` Insight node:**

```json
{
  "id": "insight:a1b2c3d4",
  "type": "Insight",
  "properties": {
    "text": "AI regulation will significantly lag behind the pace of innovation",
    "confidence": 0.92,
    "grounded": true,
    "insight_type": "claim"
  }
}
```

**Extraction prompt change:** The LLM extraction step in the GIL builder receives a
small addition asking it to classify each Insight into one of the five types. This is
a prompt engineering change (RFC-052 territory, autoresearch candidate) — not a schema
breaking change. The classification is expected to be high-accuracy for `claim` vs
`question` (structurally distinct) and moderate-accuracy for `claim` vs `observation`
(requires understanding stance). Accuracy can be measured and improved via the existing
eval framework.

**Backward compatibility:** Existing `gi.json` artifacts without `insight_type` are
treated as `"unknown"` by all consumers. No migration script needed — the field is
optional. New pipeline runs emit it; old artifacts are valid without it.

**Relationship to grounding contract (ADR-053):** `insight_type` is orthogonal to the
grounding contract. A `claim` type Insight still requires a supporting Quote with
`grounded: true`. The type classifies *what kind of statement* the Insight represents,
not *how well it is evidenced*.

#### 2b. Position Hint (GIL v1.1 — Additive, Derived)

An optional `position_hint` field is added to every Insight node in `gi.json`. Unlike
`insight_type`, this field requires **no extraction change** — it is purely derived from
data that already exists (Quote timestamps and episode duration).

**The problem it solves:**

Insights do not inherit any temporal position from their supporting Quotes, even though
Quotes carry `timestamp_start_ms` and `timestamp_end_ms`. All Insights from a single
episode look temporally equivalent. But in practice, the last thing someone says on a
topic often carries more weight than the first — a guest might spend 20 minutes building
toward a conclusion. For the Position Tracker, knowing *where* in an episode an Insight
occurred is the difference between "here are 5 things they said" and "here is the
progression of their argument, and here is the conclusion."

**Why it matters for the flagships:**

- **Position Tracker**: When assembling a chronological arc across episodes, Insights
  within a single episode can be ordered by `position_hint` to show argumentative
  progression. A `claim` Insight at `position_hint: 0.85` (late in the episode) is
  likely a conclusion; one at `0.12` (early) is likely framing or setup. This adds
  intra-episode narrative structure to the inter-episode timeline.
- **Person Profile**: When exploring a person's topic footprint, `position_hint`
  helps identify later, more settled statements vs early exploratory remarks.

**Computation:**

```python
def compute_position_hint(
    supporting_quotes: list[dict], episode_duration_ms: int | None
) -> float | None:
    if not episode_duration_ms or not supporting_quotes:
        return None
    mean_start = sum(
        q["timestamp_start_ms"] for q in supporting_quotes
    ) / len(supporting_quotes)
    return round(min(mean_start / episode_duration_ms, 1.0), 2)
```

Returns a normalised float from `0.0` (start of episode) to `1.0` (end of episode),
clamped to `1.0` (guards against bad data where a Quote timestamp exceeds the reported
episode duration), rounded to two decimal places. Returns `null` if
`episode_duration_ms` is missing or if the Insight has no supporting Quotes with
timestamps.

**Schema change in `gi.json` Insight node (combined with `insight_type`):**

```json
{
  "id": "insight:a1b2c3d4",
  "type": "Insight",
  "properties": {
    "text": "AI regulation will significantly lag behind the pace of innovation",
    "confidence": 0.92,
    "grounded": true,
    "insight_type": "claim",
    "position_hint": 0.85
  }
}
```

**Where `episode_duration_ms` comes from:** The KG `Episode` node already carries
`duration_ms` in its properties (sourced from RSS feed metadata or audio file
inspection). The GIL builder reads this value at build time to compute `position_hint`
for each Insight. If the Episode node lacks `duration_ms`, `position_hint` is `null`.

**Backward compatibility:** Identical to `insight_type` — existing artifacts without
`position_hint` are treated as `null` by all consumers. No migration script needed.
New pipeline runs emit it; old artifacts are valid without it.

**No extraction change needed:** This is a pure post-processing step in the GIL
builder. The Quote timestamps and episode duration are already available at build time.
No LLM call, no prompt change, no autoresearch candidate — just arithmetic.

---

### 3. KG Ontology Migration

**Change:** Replace `entity:person:{slug}` and `entity:organization:{slug}` with
`person:{slug}` and `org:{slug}`.

| Before | After |
| --- | --- |
| `"id": "entity:person:lex-fridman"` | `"id": "person:lex-fridman"` |
| `"entity_kind": "person"` | `"kind": "person"` *(property retained)* |
| `"id": "entity:organization:openai"` | `"id": "org:openai"` |
| `"entity_kind": "organization"` | `"kind": "org"` *(property retained)* |

The `kind` property is retained on Entity nodes for consumers who need it. The `entity:`
prefix is removed from the ID — it was a KG implementation detail, not a semantic truth.

**Migration of existing `kg.json` artifacts:** Script
`scripts/migrate_kg_entity_ids.py` rewrites `entity:person:` to `person:` and
`entity:organization:` to `org:` in all existing artifacts. Idempotent. Back up the
corpus before running.

---

### 4. Bridge Artifact — `bridge.json`

A third artifact emitted per episode, alongside `gi.json` and `kg.json`, in the same
pipeline build pass.

**Purpose:** Declare all canonical IDs that appear in this episode, with their type,
display name, aliases, and which layers reference them. Enables cross-layer joins at
query time without a database.

**Schema (inline):**

```json
{
  "schema_version": "1.0",
  "episode_id": "episode:abc123",
  "emitted_at": "2026-04-12T10:00:00Z",
  "identities": [
    {
      "id": "person:lex-fridman",
      "type": "person",
      "display_name": "Lex Fridman",
      "aliases": ["Lex", "Lexar Fridman"],
      "sources": {
        "gi": true,
        "kg": true
      }
    },
    {
      "id": "org:openai",
      "type": "org",
      "display_name": "OpenAI",
      "aliases": ["Open AI", "OpenAI Inc"],
      "sources": {
        "gi": false,
        "kg": true
      }
    },
    {
      "id": "topic:ai-regulation",
      "type": "topic",
      "display_name": "AI Regulation",
      "aliases": [],
      "sources": {
        "gi": true,
        "kg": true
      }
    }
  ]
}
```

**Field definitions:**

| Field | Type | Description |
| --- | --- | --- |
| `schema_version` | string | `"1.0"` |
| `episode_id` | string | Episode anchor, same as `gi.json` and `kg.json` |
| `emitted_at` | ISO-8601 | Pipeline run timestamp |
| `identities` | array | All canonical IDs appearing in this episode |
| `id` | string | Canonical CIL ID (`person:`, `org:`, `topic:`) |
| `type` | enum | `person` / `org` / `topic` |
| `display_name` | string | Human-readable name (for UI, not identity) |
| `aliases` | string[] | Known variant spellings seen in this episode |
| `sources.gi` | boolean | Whether this ID appears in `gi.json` for this episode. A topic with `sources.gi: false` is KG-known but not yet GIL-enriched (no ABOUT edges from Insights) — this is a valid and expected state in the current pipeline, since GIL topic support is deferred in v1. Query patterns that need GIL Insights (e.g. Pattern A) correctly skip these episodes. |
| `sources.kg` | boolean | Whether this ID appears in `kg.json` for this episode |

**What the bridge does NOT contain:**

- Insight or Quote IDs — those are GIL-internal, not shared.
- Chunk IDs — those are semantic search-internal.
- Any content or text — pure identity registry, no intelligence payload.

**Emit logic (pipeline):**

```python
# podcast_scraper/builders/bridge_builder.py

from datetime import datetime


def build_bridge(
    episode_id: str, gi_artifact: dict, kg_artifact: dict
) -> dict:
    identities: dict[str, dict] = {}

    for node in gi_artifact.get("nodes", []):
        _collect_identity(identities, node, source="gi")

    for node in kg_artifact.get("nodes", []):
        _collect_identity(identities, node, source="kg")

    return {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "emitted_at": datetime.utcnow().isoformat() + "Z",
        "identities": list(identities.values()),
    }


def _collect_identity(
    identities: dict[str, dict], node: dict, source: str
) -> None:
    nid = node["id"]
    if not nid.startswith(("person:", "org:", "topic:")):
        return
    kind = nid.split(":")[0]
    node_aliases = node.get("properties", {}).get("aliases", [])
    display = (
        node.get("properties", {}).get("name")
        or node.get("properties", {}).get("label", "")
    )

    if nid in identities:
        existing = identities[nid]
        existing["sources"][source] = True
        existing["aliases"] = list(
            set(existing["aliases"]) | set(node_aliases)
        )
    else:
        identities[nid] = {
            "id": nid,
            "type": kind,
            "display_name": display,
            "aliases": list(node_aliases),
            "sources": {"gi": source == "gi", "kg": source == "kg"},
        }
```

---

### 5. Cross-Layer Query Patterns (No Database)

With `bridge.json` in place, cross-layer queries work by loading the relevant
per-episode artifacts from the filesystem and joining on canonical IDs. No database
required.

These patterns directly support the flagship use cases.

**Pattern A: Position Tracker query — Insights by person + topic across episodes**

The optional `insight_types` filter defaults to `("claim",)` for a focused position
arc. Pass `None` to include all types (useful for exploratory views).

```python
def position_arc(
    corpus_dir: Path,
    target_person: str,
    target_topic: str,
    insight_types: tuple[str, ...] | None = ("claim",),
) -> list[dict]:
    results = []
    for episode_dir in corpus_dir.iterdir():
        bridge_path = episode_dir / "bridge.json"
        gi_path = episode_dir / "gi.json"
        kg_path = episode_dir / "kg.json"
        if not all(p.exists() for p in [bridge_path, gi_path, kg_path]):
            continue

        bridge = json.loads(bridge_path.read_text())
        id_set = {i["id"] for i in bridge["identities"] if i["sources"]["gi"]}
        if target_person not in id_set or target_topic not in id_set:
            continue

        gi = json.loads(gi_path.read_text())
        kg = json.loads(kg_path.read_text())

        spoken_quotes = {
            e["from"] for e in gi["edges"]
            if e["type"] == "SPOKEN_BY" and e["to"] == target_person
        }
        about_topic = {
            e["from"] for e in gi["edges"]
            if e["type"] == "ABOUT" and e["to"] == target_topic
        }
        relevant_insights = about_topic & {
            e["from"] for e in gi["edges"]
            if e["type"] == "SUPPORTED_BY" and e["to"] in spoken_quotes
        }

        episode_node = next(
            (n for n in kg["nodes"] if n["type"] == "Episode"), None
        )
        publish_date = (
            episode_node["properties"].get("publish_date")
            if episode_node else None
        )

        insights = [n for n in gi["nodes"] if n["id"] in relevant_insights]
        if insight_types:
            insights = [
                n for n in insights
                if n.get("properties", {}).get("insight_type", "unknown")
                in insight_types
            ]
        insights.sort(
            key=lambda n: n.get("properties", {}).get("position_hint") or 0.0
        )
        if insights:
            results.append({
                "episode_id": bridge["episode_id"],
                "publish_date": publish_date,
                "insights": insights,
            })

    return sorted(results, key=lambda r: r["publish_date"] or "")
```

**Pattern B: Person Profile query — all Insights by a person, grouped by topic**

Insights are grouped by topic and annotated with `insight_type` and `position_hint`.
The profile consumer uses `insight_type` to distinguish statement types: `claim` and
`recommendation` are the positions a user wants to see; `observation` and `question`
provide supporting context. `position_hint` identifies later, more settled statements.

```python
def person_profile(corpus_dir: Path, target_person: str) -> dict:
    by_topic: dict[str, list] = {}
    all_quotes: list[dict] = []

    for episode_dir in corpus_dir.iterdir():
        bridge_path = episode_dir / "bridge.json"
        gi_path = episode_dir / "gi.json"
        if not bridge_path.exists() or not gi_path.exists():
            continue

        bridge = json.loads(bridge_path.read_text())
        gi_ids = {
            i["id"] for i in bridge["identities"]
            if i["sources"]["gi"]
        }
        if target_person not in gi_ids:
            continue

        gi = json.loads(gi_path.read_text())

        spoken_quotes = {
            e["from"] for e in gi["edges"]
            if e["type"] == "SPOKEN_BY" and e["to"] == target_person
        }
        supported_insights = {
            e["from"] for e in gi["edges"]
            if e["type"] == "SUPPORTED_BY" and e["to"] in spoken_quotes
        }

        for edge in gi["edges"]:
            if edge["type"] == "ABOUT" and edge["from"] in supported_insights:
                topic_id = edge["to"]
                insight_node = next(
                    (n for n in gi["nodes"] if n["id"] == edge["from"]),
                    None,
                )
                if insight_node:
                    props = insight_node.get("properties", {})
                    by_topic.setdefault(topic_id, []).append({
                        "episode_id": bridge["episode_id"],
                        "insight": insight_node,
                        "insight_type": props.get("insight_type", "unknown"),
                        "position_hint": props.get("position_hint"),
                    })

        for qid in spoken_quotes:
            quote_node = next(
                (n for n in gi["nodes"] if n["id"] == qid), None
            )
            if quote_node:
                all_quotes.append({
                    "episode_id": bridge["episode_id"],
                    "quote": quote_node,
                })

    return {
        "person_id": target_person,
        "topics": by_topic,
        "quotes": all_quotes,
    }
```

**Pattern C: Topic timeline — follow-the-thread across episodes**

Like Pattern A, supports an optional `insight_types` filter (defaults to all types,
since topic timelines often benefit from the full picture) and sorts Insights within
each episode by `position_hint`.

```python
def topic_timeline(
    corpus_dir: Path,
    target_topic: str,
    insight_types: tuple[str, ...] | None = None,
) -> list[dict]:
    results = []
    for episode_dir in corpus_dir.iterdir():
        bridge_path = episode_dir / "bridge.json"
        gi_path = episode_dir / "gi.json"
        kg_path = episode_dir / "kg.json"
        if not all(p.exists() for p in [bridge_path, gi_path, kg_path]):
            continue

        bridge = json.loads(bridge_path.read_text())
        topic_ids = {i["id"] for i in bridge["identities"]}
        if target_topic not in topic_ids:
            continue

        gi = json.loads(gi_path.read_text())
        kg = json.loads(kg_path.read_text())

        about_insights = {
            e["from"] for e in gi["edges"]
            if e["type"] == "ABOUT" and e["to"] == target_topic
        }
        insights = [n for n in gi["nodes"] if n["id"] in about_insights]
        if insight_types:
            insights = [
                n for n in insights
                if n.get("properties", {}).get("insight_type", "unknown")
                in insight_types
            ]
        insights.sort(
            key=lambda n: n.get("properties", {}).get("position_hint") or 0.0
        )

        episode_node = next(
            (n for n in kg["nodes"] if n["type"] == "Episode"), None
        )
        publish_date = (
            episode_node["properties"].get("publish_date")
            if episode_node else None
        )

        if insights:
            results.append({
                "episode_id": bridge["episode_id"],
                "publish_date": publish_date,
                "insights": insights,
            })

    return sorted(results, key=lambda r: r["publish_date"] or "")
```

---

### 6. Semantic Search — Chunk-to-Insight Lift Path

The FAISS index (RFC-061) stores chunks with `episode_id`, `char_start`, `char_end`
metadata. GIL Quote nodes also store `episode_id`, `char_start`, `char_end` against the
same transcript text.

**The lift:** A search result chunk can be matched to a Quote node if their
`(episode_id, char_start, char_end)` ranges overlap. The matched Quote then yields
its Insight (via SUPPORTED_BY edges) and the Insight yields its Person (via
SPOKEN_BY to Person) and Topic (via ABOUT to Topic).

This enrichment is a **query-time operation** on existing artifacts — no FAISS rebuild,
no pipeline change. The bridge is the final piece that makes the Person and Topic IDs
resolvable to display names.

**Prerequisite:** Verify that GIL Quote char offsets and FAISS chunk char offsets use the
same transcript normalisation. See Known Limitations, section 1.

**Enriched search result shape (target — contingent on Phase 5 verification):**

```json
{
  "chunk_text": "Regulation will lag innovation by 3-5 years...",
  "semantic_score": 0.91,
  "episode_id": "episode:abc123",
  "episode_title": "AI Regulation with Lex Fridman",
  "publish_date": "2026-02-03T00:00:00Z",
  "lifted": {
    "insight": {
      "id": "insight:a1b2c3d4",
      "text": "AI regulation will significantly lag behind the pace of innovation",
      "insight_type": "claim",
      "position_hint": 0.85,
      "grounded": true
    },
    "speaker": {
      "id": "person:lex-fridman",
      "display_name": "Lex Fridman"
    },
    "topic": {
      "id": "topic:ai-regulation",
      "display_name": "AI Regulation"
    },
    "quote": {
      "timestamp_start_ms": 120000,
      "timestamp_end_ms": 135000
    }
  }
}
```

The `lifted` block is contingent on Phase 5 char-offset verification. If offsets
diverge, a mapping layer is needed before this enrichment is available.

---

## Key Decisions

1. **`person:` not `entity:person:` as the canonical prefix**
   - **Decision**: `person:{slug}` is the canonical form. `entity:` was a KG
     implementation detail.
   - **Rationale**: A person is a person regardless of which layer references them.
     Neither GIL nor KG should own the identity namespace. `person:` is semantically
     honest and shorter.

2. **Bridge as a third artifact, not embedded in gi.json or kg.json**
   - **Decision**: `bridge.json` is standalone per episode.
   - **Rationale**: Keeps `gi.json` and `kg.json` schemas clean and independently
     versioned. The bridge can evolve without touching either ontology. It is the
     explicit seam between layers.

3. **No database prerequisite**
   - **Decision**: All cross-layer joins work on filesystem JSON artifacts.
   - **Rationale**: Postgres projection (ADR-054) is a future consumer of this work,
     not a prerequisite. The bridge must deliver value with the current stack.

4. **One slugifier, one location**
   - **Decision**: `podcast_scraper/identity/slugify.py` is the single source of truth.
   - **Rationale**: Silent ID divergence (two slugifiers producing different outputs for
     the same name) is the most likely failure mode of the bridge. One function
     eliminates this class of bug.

5. **Clean-slate migration with idempotent scripts**
   - **Decision**: Migrate existing artifacts via one-time scripts rather than supporting
     both old and new ID formats.
   - **Rationale**: Supporting both formats indefinitely adds query complexity and
     testing surface. The corpus is small enough for a clean migration.

6. **GIL and KG remain separate — merge decision deferred**
   - **Decision**: The bridge joins the layers without merging them. Whether to merge
     GIL and KG into a single artifact is explicitly deferred.
   - **Rationale**: The bridge provides the join capability. Real usage will reveal
     whether the separation continues to earn its keep (some queries only need one
     layer) or creates unnecessary friction (every query loads both). Let data from
     building the flagship use cases inform the merge decision.

7. **Flagship use cases define the data foundation, not the analysis layer**
   - **Decision**: Position Tracker and Person Profile define the query patterns and
     output shapes. Advanced analysis capabilities are future extensions.
   - **Rationale**: The bridge is the prerequisite. Building the collection and
     assembly layer first, then layering analysis on top, avoids coupling the
     foundation to a specific analysis approach.

8. **`insight_type` as an additive GIL v1.1 field, not deferred to the analysis layer**
   - **Decision**: Add an optional `insight_type` enum (`claim`, `recommendation`,
     `observation`, `question`, `unknown`) to every Insight node in `gi.json`. Default
     `unknown` so existing artifacts remain valid without migration.
   - **Rationale**: The Position Tracker and Person Profile are qualitatively better when
     they can distinguish claims from observations and questions. Without `insight_type`,
     the consumer must mentally classify every Insight — which defeats the purpose of
     structured data. The field is additive (no breakage), the extraction prompt change
     is small and measurable, and emitting it now means new pipeline runs accumulate
     typed Insights before Phase 4 implementation. Deferring it to Phase 4 or to a
     future analysis layer would require re-extracting all artifacts.

9. **`position_hint` as a derived GIL v1.1 field, computed from existing data**
   - **Decision**: Add an optional `position_hint` float (`0.0`-`1.0`) to every Insight
     node, derived from the mean `timestamp_start_ms` of supporting Quotes divided by
     `episode_duration_ms`. Default `null` when duration is missing.
   - **Rationale**: Insights within a single episode are temporally flat — a framing
     remark at minute 2 and a concluding claim at minute 55 look identical. The Position
     Tracker needs intra-episode ordering to show argumentative progression, and the
     Person Profile benefits from distinguishing settled positions. Unlike
     `insight_type`, this requires no extraction change — it is pure arithmetic on data
     already present at build time (Quote timestamps + Episode duration). Zero cost to
     emit, meaningful improvement to both flagships.

---

## Alternatives Considered

1. **Runtime slug matching without a bridge artifact**
   - **Description**: At query time, infer that `speaker:lex-fridman` and
     `entity:person:lex-fridman` refer to the same entity by stripping prefixes and
     comparing slugs.
   - **Pros**: No pipeline change, no new artifact.
   - **Cons**: Fragile — depends on both layers having used the same slugifier (which
     they currently do not guarantee). Fails silently. Cannot capture aliases. No
     explicit record of which layers contributed data for a given identity.
   - **Why Rejected**: The bridge is cheap to emit and makes the join explicit and
     auditable.

2. **Merging GIL and KG into a single unified artifact**
   - **Description**: One `graph.json` per episode containing all nodes and edges from
     both layers.
   - **Pros**: Simpler query — one artifact to load.
   - **Cons**: Conflates two distinct semantic contracts (grounded evidence vs. entity
     discovery). Makes independent versioning harder. ADR-052 explicitly separates them
     for good reasons.
   - **Why Rejected**: The bridge achieves the join without sacrificing the separation.
     The merge decision is deferred to be informed by real usage.

3. **Postgres projection as the bridge**
   - **Description**: Use the future relational projection (ADR-054) as the cross-layer
     join layer.
   - **Pros**: More powerful query capabilities.
   - **Cons**: Requires database infrastructure not yet in place. Blocks value on a
     prerequisite.
   - **Why Rejected**: The filesystem bridge delivers value now. The Postgres projection
     will be a natural consumer of `bridge.json` when built.

4. **Finishing RFC-050/056 use cases independently first**
   - **Description**: Complete the single-layer use case RFCs before building
     cross-layer capabilities.
   - **Pros**: Each layer is fully specified before integration.
   - **Cons**: The most valuable use cases are cross-layer. Finishing single-layer
     consumption RFCs pulls effort toward inward-looking features when the real value
     is in the join. Single-layer consumption (CLI inspect, schema validation) is
     already well-defined in those RFCs.
   - **Why Rejected**: Cross-layer use cases are prioritised. RFC-050 and RFC-056
     retain their single-layer consumption patterns; cross-layer use cases live here.

---

## Known Limitations

These are not deferred "nice to haves." They are real constraints on how well the
flagship use cases work in practice. Each is scoped to a specific failure mode with a
mitigation path.

### 1. Char offset alignment (blocks semantic search lift)

GIL Quote `char_start`/`char_end` and FAISS chunk `char_start`/`char_end` must
reference the same normalised transcript text. If they diverge (different whitespace
normalisation, BOM handling, encoding differences), the chunk-to-Insight lift in
Section 6 will produce incorrect matches.

**Mitigation:** Empirical verification on eval corpus before implementing Phase 5.
If offsets diverge, a mapping layer that re-normalises one coordinate space to the
other is needed.

**Blast radius:** Blocks the semantic search entry point for Position Tracker and
Person Profile. The filesystem-based query patterns (Sections 5A-5C) are unaffected.

### 2. Name variation across episodes (degrades cross-episode joins)

The slugifier is deterministic for identical input strings, but the real problem is
that input varies: "Sam Altman" vs "Samuel Altman" vs "Sam" (no surname). These
produce different slugs and therefore different `person:` IDs. The same person appears
as multiple identities, and cross-episode queries miss data.

**Mitigation:** Name normalisation is the responsibility of the LLM extraction step
in both GIL and KG builders (stated in Assumptions). Improving extraction prompt
quality (RFC-052) is the primary lever. A corpus-level alias registry that merges
`person:sam-altman` and `person:samuel-altman` into a single canonical identity is a
future capability — the bridge's per-episode `aliases` field provides the raw material
for building it.

**Blast radius:** Proportional to how often names vary across episodes. Position
Tracker and Person Profile degrade gracefully — they return partial results rather than
wrong results. The user sees Insights for "Sam Altman" but misses Insights attributed
to "Samuel Altman."

### 3. Topic deduplication (degrades topic-based queries)

KG topics are extracted from summary bullets or LLM extraction and may produce slightly
different labels for the same concept across episodes ("AI Regulation" vs "AI Policy"
vs "Artificial Intelligence Governance"). Each produces a different `topic:` slug.
Follow-the-thread and controversy radar queries miss data when the same concept has
multiple slugs.

**Mitigation:** Like name variation, this is primarily an extraction quality issue.
Semantic deduplication of topic slugs (embedding similarity, LLM-based canonicalisation)
is a future capability. The bridge records what the pipeline emits; improving what the
pipeline emits is the lever.

**Blast radius:** Same as name variation — partial results, not wrong results. Topic
timeline shows the thread for `topic:ai-regulation` but misses episodes that used
`topic:ai-policy` for the same concept.

### 4. Episode duration availability (degrades `position_hint` coverage)

`position_hint` depends on `episode_duration_ms` from the KG Episode node. This value
is sourced from RSS feed metadata (`<itunes:duration>`) or audio file inspection. If
the feed omits duration or the audio was not inspected, `position_hint` is `null` for
all Insights in that episode.

**Mitigation:** Most podcast RSS feeds include `<itunes:duration>`. For feeds that do
not, a one-time audio inspection pass can backfill the value. The field degrades
gracefully — query patterns treat `null` as `0.0` for sorting purposes, so Insights
without a hint sort to the beginning rather than disappearing.

**Blast radius:** Proportional to how many episodes lack duration metadata. Position
Tracker loses intra-episode ordering for those episodes but still shows the
inter-episode chronological arc. Person Profile loses the `position_hint` signal but
still ranks by topic Insight count.

### 5. Analysis layer not included (limits flagship use case depth)

The Position Tracker can assemble a chronological arc of Insights. It cannot
automatically detect that a later Insight contradicts an earlier one. The Person
Profile lists topics and Insight counts but does not rank or curate them
analytically.

**Mitigation:** A future analysis layer may add automated interpretation. This
RFC ensures the data is joinable; interpretation is a separate concern.

**Blast radius:** The flagship use cases deliver value without analysis --
a chronological position arc is useful even without automated change detection,
and a topic overview is useful without analytical ranking.

---

## Relationship to RFC-050 and RFC-056

RFC-050 (GIL Use Cases) and RFC-056 (KG Use Cases) were written when GIL and KG were
being developed independently. Their single-layer consumption patterns remain valid:

- **RFC-050**: `gi inspect`, `gi show-insight`, `gi explore` CLI commands; per-episode
  GIL inspection; schema validation; Insight Explorer query within GIL.
- **RFC-056**: `kg validate`, `kg inspect`, `kg export`, `kg entities` CLI commands;
  entity roll-up; topic co-occurrence; structured export.

The **cross-layer use cases** — opinion tracking, guest intelligence, controversy
detection, follow-the-thread — now live in this RFC (RFC-072). They require the CIL
and bridge, which did not exist when RFC-050/056 were drafted.

RFC-050 and RFC-056 do not need to be "finished" for the cross-layer vision to proceed.
Their single-layer patterns are useful for inspection and debugging. The cross-layer
patterns defined here are where the product value lives.

---

## Testing Strategy

**Test Coverage:**

- **Unit tests for slugifier**: Determinism, idempotency, edge cases (unicode, empty
  string, all-punctuation, very long names, diacritics). Located in
  `tests/unit/identity/test_slugify.py`.
- **Unit tests for bridge builder**: Given synthetic `gi.json` + `kg.json`, assert
  correct `bridge.json` output. Cover: person in GI only, org in KG only, topic in
  both, aliases merged from both layers correctly. Located in
  `tests/unit/builders/test_bridge_builder.py`.
- **Integration tests**: Run pipeline on eval episodes, assert `bridge.json` is
  emitted, assert all `person:` and `org:` IDs in `bridge.json` match IDs in `gi.json`
  and `kg.json`. Located in `tests/integration/test_bridge_integration.py`.
- **Migration script tests**: Assert idempotency and correctness of
  `migrate_gi_speaker_to_person.py` and `migrate_kg_entity_ids.py` against fixture
  artifacts.
- **Cross-layer query tests**: Assert that Position Tracker and Person Profile query
  patterns return correct results against a synthetic multi-episode corpus with known
  identities.
- **Insight type tests**: Assert that `insight_type` is emitted on Insight nodes in
  new pipeline runs. Assert that the Position Tracker `insight_types` filter correctly
  includes/excludes Insights by type. Assert that the Person Profile annotates each
  Insight with its type. Assert backward compatibility: `gi.json` artifacts without
  `insight_type` are treated as `"unknown"` by all query patterns.
- **Position hint tests**: Assert that `position_hint` is computed correctly from
  supporting Quote timestamps and episode duration. Cover edge cases: no supporting
  Quotes (returns `null`), missing `episode_duration_ms` (returns `null`), single
  Quote, multiple Quotes (mean of `timestamp_start_ms`). Assert that the Position
  Tracker sorts Insights within each episode by `position_hint`. Assert backward
  compatibility: artifacts without `position_hint` are treated as `null`.

**Success Criteria:**

1. All `gi.json` artifacts use `person:{slug}` — no `speaker:` prefix remains.
2. All `kg.json` artifacts use `person:{slug}` and `org:{slug}` — no `entity:` prefix
   remains.
3. Every episode directory in the corpus contains a valid `bridge.json`.
4. `bridge.json` identities are a superset of all `person:`, `org:`, `topic:` IDs in
   `gi.json` and `kg.json` for that episode.
5. Slugifier produces identical output for the same input regardless of which module
   calls it.
6. Position Tracker query returns a chronologically ordered arc of Insights for a
   given person + topic across a multi-episode test corpus.
7. Person Profile query returns all Insights grouped by topic for a given person across
   a multi-episode test corpus.
8. New pipeline runs emit `insight_type` and `position_hint` on every Insight node
   (where data is available). Pre-existing artifacts without these fields are handled
   gracefully (`insight_type` defaults to `"unknown"`, `position_hint` defaults to
   `null`).

---

## Rollout & Monitoring

**Phase 1 — Slugifier and CIL namespace (prerequisite):**

Implement `podcast_scraper/identity/slugify.py`. Update both GIL and KG builders to
import from it. No artifact changes yet — just align the slug function.

**Phase 2 — Ontology migration + GIL v1.1 fields (`insight_type`, `position_hint`):**

Update GIL ontology (`speaker:` to `person:`). Add `insight_type` field to Insight
nodes in the GIL builder extraction prompt (default `"unknown"` for backward
compatibility). Add `position_hint` computation to the GIL builder post-processing
(derived from Quote timestamps and episode duration — no extraction change). Update KG
ontology (`entity:person:` to `person:`, `entity:organization:` to `org:`). Update
schemas. Run migration scripts against existing artifacts (after backup). Update all
tests. Existing artifacts without `insight_type` or `position_hint` remain valid — no
migration needed for these fields.

**Phase 3 — Bridge artifact:**

Implement `bridge_builder.py`. Wire into pipeline after GI and KG build passes. Emit
`bridge.json` per episode. Validate against eval corpus.

**Phase 4 — Flagship query patterns and API:**

Implement Position Tracker and Person Profile query patterns. Expose via API:

- `GET /persons/{person_id}/positions?topic={topic_id}` — position arc for a person
  on a topic.
- `GET /persons/{person_id}/brief` — person profile (topics, Insight counts,
  appearances).
- `GET /topics/{topic_id}/timeline` — topic evolution across episodes.
- `GET /topics/{topic_id}/persons` — all persons who discuss a topic.
- `GET /persons/{person_id}/topics` — all topics a person discusses.

**Phase 5 — Semantic search lift (follow-up):**

After verifying char offset alignment (Known Limitations, section 1), add
chunk-to-Insight lift to search result enrichment. No FAISS rebuild required.

**`grounding_rate` source note:** The `grounding_rate` statistic used in PRD-026
(Topic Entity View, person chip badge) and PRD-029 (Person Profile) comes from
RFC-073's `grounding_rate` corpus enricher, not from an analysis layer pass.

---

## GI/KG viewer (RFC-062) alignment

RFC-072 **type and ID** changes (`Person` instead of `Speaker`, `person:` / `org:` prefixes,
`bridge.json` where wired) are implemented in the SPA under `web/gi-kg-viewer/src/utils/`
(for example `parsing.ts`, `visualGroup.ts`, `mergeGiKg.ts`, `searchFocus.ts`). Co-located
**Vitest** suites (`*.test.ts`) cover canonical id display, graph grouping, and merge
behaviour.

---

## Benefits

1. **Cross-episode person intelligence**: Query all Insights, quotes, and topics for
   any guest across the entire corpus — the foundation for the Position Tracker and
   Person Profile.
2. **Topic threading**: Follow any topic chronologically across episodes, grounded in
   verbatim quotes — enables follow-the-thread navigation.
3. **Richer semantic search**: Search results lift from raw chunks to structured,
   attributed, timestamped Insights — a qualitatively different user experience.
4. **DB-free**: Full value delivered on the current filesystem artifact stack. Postgres
   projection (ADR-054) becomes a natural next step, not a prerequisite.
5. **Future-proof**: `person:`, `org:`, `topic:` as shared global IDs will cleanly
   survive the Postgres projection, the graph viewer (ADR-065), and any future content
   type additions.
6. **Coherent vision**: GIL, KG, and semantic search stop being three independent
   layers and start being a system that delivers use cases none of them could deliver
   alone.
7. **Deferred merge**: GIL and KG remain separate (per ADR-052) while the bridge
   provides the join. Real usage informs whether to merge later.
8. **Type-aware Insights**: `insight_type` makes the Position Tracker and Person
   Profile qualitatively better by distinguishing claims from observations and
   questions — the difference between a structured position arc and a noisy quote dump.
9. **Intra-episode ordering**: `position_hint` adds temporal structure within episodes
   so the Position Tracker can show argumentative progression (setup to conclusion) and
   the Person Profile can identify later, more settled statements. Zero extraction
   cost — derived from existing Quote timestamps.

---

## Migration Path

1. **Phase 1**: Deploy slugifier. No artifact changes.
2. **Phase 2**: Back up corpus. Run migration scripts. Add `insight_type` to GIL
   builder extraction prompt and `position_hint` computation to GIL builder
   post-processing (new runs emit both; old artifacts stay valid without them).
   Validate full corpus. Bump `gi.json` `schema_version` to `2.0`, `kg.json`
   `schema_version` to `1.2`.
3. **Phase 3**: Deploy bridge builder. Rebuild all episode artifacts (pipeline re-run
   or standalone bridge builder against existing artifacts).
4. **Phase 4**: Implement flagship query patterns. Expose API endpoints. Update
   Cytoscape viewer (ADR-065) to navigate by canonical ID.
5. **Phase 5**: Enable search result lifting post char-offset verification.

---

## References

- `docs/architecture/gi/ontology.md` — GIL ontology (Speaker to Person migration)
- `docs/architecture/kg/ontology.md` — KG ontology (entity: prefix removal)
- `docs/adr/ADR-052-separate-gil-and-kg-artifact-layers.md` — separation rationale
- `docs/adr/ADR-053-grounding-contract-for-evidence-backed-insights.md`
- `docs/adr/ADR-061-faiss-phase-1-with-post-filter-metadata.md`
- `docs/adr/ADR-062-sentence-boundary-transcript-chunking.md`
- `docs/rfc/RFC-049-grounded-insight-layer-core.md`
- `docs/rfc/RFC-050-grounded-insight-layer-use-cases.md` — single-layer GIL consumption
- `docs/rfc/RFC-055-knowledge-graph-layer-core.md`
- `docs/rfc/RFC-056-knowledge-graph-layer-use-cases.md` — single-layer KG consumption
- `docs/rfc/RFC-061-semantic-corpus-search.md`
