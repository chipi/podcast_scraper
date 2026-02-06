# Grounded Insight Layer Ontology (v1)

## Status

v1 (implementation-ready)

## Purpose

Define the canonical ontology contract for the **Grounded Insight Layer (GIL)**:

- Node & edge types
- Required properties
- Identity (ID) rules
- **Grounding contract** (the 2025 moat)
- Provenance & evidence requirements

This document is the source of truth for contributors. All `kg.json` outputs MUST conform
to this ontology and the companion schema (`docs/kg/kg.schema.json`).

---

## Design Principles

- **Insight-centric**: Focus on takeaways (Insights) and evidence (Quotes), not just claims
- **Evidence first-class**: Quotes are nodes, not metadata—enables trust and navigation
- **Explicit grounding**: Every Insight must declare `grounded=true/false`
- **Stable IDs**: Global concepts must have stable identifiers across episodes
- **Entities deferred**: Entity extraction deferred to v1.1 to focus on core value

---

## The Grounding Contract (Critical)

The grounding contract is what makes GIL trustworthy:

### Hard Rules (Invariants)

1. **Every Quote MUST be verbatim**
   - `Quote.text` must exactly match `transcript[char_start:char_end]`
   - No paraphrasing, no summarization, no rewording
   - Timestamps must correspond to the quoted span

2. **Every Insight MUST have explicit grounding status**
   - `grounded=true`: Insight has ≥1 `SUPPORTED_BY` edge to a Quote
   - `grounded=false`: Insight is extracted but lacks supporting quote (rare, but honest)

3. **SUPPORTED_BY edges are evidence links**
   - An Insight can have multiple supporting Quotes
   - Each Quote provides evidence for the Insight's validity

### Why This Matters

- **Trust**: Users know exactly which Insights have evidence
- **Quality Metrics**: System can measure `% insights grounded` and `quote validity rate`
- **RAG Applications**: Downstream systems can filter for grounded-only Insights
- **Debugging**: Ungrounded Insights are visible, not hidden

---

## Top-Level Concepts

### Node Types (v1)

| Node Type    | Description                                          |
| ------------ | ---------------------------------------------------- |
| Podcast      | A podcast feed                                       |
| Episode      | A single podcast episode                             |
| Speaker      | A person speaking (optional if no diarization)       |
| Topic        | An abstract subject discussed (lightweight)          |
| **Insight**  | A key takeaway / conclusion extracted from content   |
| **Quote**    | Verbatim transcript span used as evidence            |

### Node Types (v1.1 - Deferred)

| Node Type | Description                                        |
| --------- | -------------------------------------------------- |
| Entity    | Person, company, product, place (deferred to v1.1) |

### Edge Types (v1)

| Edge             | From -> To         | Description                         |
| ---------------- | ------------------ | ----------------------------------- |
| HAS_EPISODE      | Podcast -> Episode | Podcast contains episode            |
| SPOKE_IN         | Speaker -> Episode | Speaker participated                |
| **HAS_INSIGHT**  | Episode -> Insight | Episode contains insight            |
| **SUPPORTED_BY** | Insight -> Quote   | Quote provides evidence for insight |
| **SPOKEN_BY**    | Quote -> Speaker   | Speaker said the quote              |
| ABOUT            | Insight -> Topic   | Insight is about topic              |
| RELATED_TO       | Topic <-> Topic    | Semantic relationship (optional)    |

---

## Common Properties

### Required on all nodes

- `id` (string) - Unique identifier
- `type` (enum) - Node type
- `properties` (object) - Type-specific properties

### Required on all edges

- `type` (enum) - Edge type
- `from` (node id) - Source node
- `to` (node id) - Target node

### Provenance (required for ML-derived content)

Any node produced by ML extraction SHOULD include:

- `confidence` (0.0–1.0) - Extraction certainty (not factual truth)

The root `kg.json` file MUST include:

- `model_version` - Model identifier used for extraction
- `prompt_version` - Prompt version used (enables A/B testing)

---

## Identity Rules (IDs)

### Episode-scoped IDs

- Episode ID must be stable and derived from RSS entry GUID if available
- Insight ID should be episode-scoped to avoid accidental global merging
- Quote ID should be episode-scoped and content-based

Recommended format:

- `episode:<rss_guid>`
- `insight:<episode_id>:<sha1(text_normalized)>`
- `quote:<episode_id>:<sha1(text)>` or `quote:<episode_id>:<char_start>-<char_end>`

### Global IDs (deduplicated across episodes)

These must be stable across episodes:

- Speaker: `speaker:<slug(name)>`
- Topic: `topic:<slug(label)>`

Deduplication: Extraction and resolution are separate steps. The extractor may emit
provisional IDs, but the resolver should converge to stable IDs over time.

---

## Node Definitions

### Podcast

**Definition:** A podcast feed.

Required properties:

- `title` (string)
- `rss_url` (string)

Optional:

- `publisher` (string)

---

### Episode

**Definition:** A single podcast episode.

Required properties:

- `podcast_id` (string)
- `title` (string)
- `publish_date` (ISO date-time string)

Optional:

- `audio_url` (string)
- `duration_ms` (integer)

---

### Speaker

**Definition:** A person speaking in the episode.

Required properties:

- `name` (string)

Optional:

- `aliases` (string[])

---

### Topic

**Definition:** An abstract subject discussed. Lightweight and mergeable.

Required properties:

- `label` (string)

Optional:

- `aliases` (string[])

---

### Insight (NEW)

**Definition:** A key takeaway or conclusion extracted from episode content.

Unlike traditional "claims," Insights:

- Focus on **what users want to know** (takeaways)
- Have explicit **grounding status** (`grounded=true/false`)
- Link to supporting **Quote** nodes for evidence

Required properties:

- `text` (string) - The insight statement (can be rephrased for clarity)
- `episode_id` (string) - Episode where insight was extracted
- `grounded` (boolean) - Whether insight has ≥1 supporting quote

Optional:

- `confidence` (number, 0.0-1.0) - Extraction confidence

---

### Quote (NEW)

**Definition:** A verbatim transcript span that serves as evidence.

Making Quote a first-class node enables:

- Evidence-backed retrieval (Insight → Quote → timestamp)
- Trust verification (users can check Quote against transcript)
- Quality metrics (quote validity rate)
- Speaker attribution when available

Required properties:

- `text` (string) - **Verbatim** text from transcript (no paraphrasing!)
- `episode_id` (string) - Episode containing the quote
- `char_start` (integer) - Character start in transcript text
- `char_end` (integer) - Character end in transcript text
- `timestamp_start_ms` (integer) - Timestamp start (milliseconds)
- `timestamp_end_ms` (integer) - Timestamp end (milliseconds)
- `transcript_ref` (string) - Reference to transcript artifact

Optional:

- `speaker_id` (string, nullable) - Speaker who said the quote (if diarization available)

---

### Entity (v1.1 - Deferred)

**Definition:** A real-world named entity. Deferred to v1.1.

Required properties:

- `name` (string)
- `entity_type` (enum: person, company, product, place, org, event, other)

Optional:

- `external_ids` (object, e.g. `{ "wikidata": "Q..." }`)

---

## Edge Definitions

### HAS_EPISODE (Podcast → Episode)

**Required properties:** none

### SPOKE_IN (Speaker → Episode)

**Required properties:** none

### HAS_INSIGHT (Episode → Insight) (NEW)

**Required properties:** none

### SUPPORTED_BY (Insight → Quote) (NEW)

**Definition:** Links an Insight to a Quote that provides evidence for it.

**Required properties:** none (Quote already carries evidence/provenance)

**Semantics:** If an Insight has ≥1 `SUPPORTED_BY` edge, it is `grounded=true`.

### SPOKEN_BY (Quote → Speaker) (NEW)

**Definition:** Links a Quote to the Speaker who said it.

**Required properties:** none

**Note:** Only present if speaker diarization is available.

### ABOUT (Insight → Topic)

**Definition:** Links an Insight to a Topic it discusses.

Optional properties:

- `confidence` (number, 0.0-1.0)

### RELATED_TO (Topic ↔ Topic)

**Definition:** Semantic relationship between topics.

Optional properties:

- `confidence` (number, 0.0-1.0)

---

## Required Output Artifact: kg.json

Each episode output folder contains a `kg.json` capturing nodes/edges for the episode.

### Root-level fields

- `schema_version` (string, required) - Schema version (e.g., "1.0")
- `model_version` (string, required) - Model used for extraction
- `prompt_version` (string, required) - Prompt version used
- `episode_id` (string, required) - Episode identifier
- `nodes` (array, required) - All nodes
- `edges` (array, required) - All edges

### Guidance

- Episode-local Insight and Quote nodes live here
- Global nodes (Topic, Speaker) may be referenced or introduced
- The logical full GIL is the union of all episode `kg.json` files
- Every Insight must have `grounded` field set explicitly

---

## Minimal Example

```json
{
  "schema_version": "1.0",
  "model_version": "gpt-4.1-mini-2026-01-xx",
  "prompt_version": "v2.1",
  "episode_id": "episode:abc123",
  "nodes": [
    {
      "id": "podcast:the-journal",
      "type": "Podcast",
      "properties": {
        "title": "The Journal",
        "rss_url": "https://feeds.example.com/the-journal"
      }
    },
    {
      "id": "episode:abc123",
      "type": "Episode",
      "properties": {
        "podcast_id": "podcast:the-journal",
        "title": "AI Regulation",
        "publish_date": "2026-02-03T00:00:00Z"
      }
    },
    {
      "id": "speaker:sam-altman",
      "type": "Speaker",
      "properties": {
        "name": "Sam Altman"
      }
    },
    {
      "id": "topic:ai-regulation",
      "type": "Topic",
      "properties": {
        "label": "AI Regulation"
      }
    },
    {
      "id": "insight:episode:abc123:a1b2c3d4",
      "type": "Insight",
      "properties": {
        "text": "AI regulation will significantly lag behind the pace of innovation",
        "episode_id": "episode:abc123",
        "grounded": true
      },
      "confidence": 0.85
    },
    {
      "id": "quote:episode:abc123:e5f6g7h8",
      "type": "Quote",
      "properties": {
        "text": "Regulation will lag innovation by 3–5 years. That's my prediction.",
        "episode_id": "episode:abc123",
        "speaker_id": "speaker:sam-altman",
        "char_start": 10234,
        "char_end": 10302,
        "timestamp_start_ms": 120000,
        "timestamp_end_ms": 135000,
        "transcript_ref": "transcript.json"
      }
    },
    {
      "id": "quote:episode:abc123:i9j0k1l2",
      "type": "Quote",
      "properties": {
        "text": "We'll see laws that are already outdated when they pass.",
        "episode_id": "episode:abc123",
        "speaker_id": "speaker:sam-altman",
        "char_start": 10890,
        "char_end": 10945,
        "timestamp_start_ms": 142000,
        "timestamp_end_ms": 148000,
        "transcript_ref": "transcript.json"
      }
    }
  ],
  "edges": [
    {
      "type": "HAS_EPISODE",
      "from": "podcast:the-journal",
      "to": "episode:abc123"
    },
    {
      "type": "SPOKE_IN",
      "from": "speaker:sam-altman",
      "to": "episode:abc123"
    },
    {
      "type": "HAS_INSIGHT",
      "from": "episode:abc123",
      "to": "insight:episode:abc123:a1b2c3d4"
    },
    {
      "type": "SUPPORTED_BY",
      "from": "insight:episode:abc123:a1b2c3d4",
      "to": "quote:episode:abc123:e5f6g7h8"
    },
    {
      "type": "SUPPORTED_BY",
      "from": "insight:episode:abc123:a1b2c3d4",
      "to": "quote:episode:abc123:i9j0k1l2"
    },
    {
      "type": "SPOKEN_BY",
      "from": "quote:episode:abc123:e5f6g7h8",
      "to": "speaker:sam-altman"
    },
    {
      "type": "SPOKEN_BY",
      "from": "quote:episode:abc123:i9j0k1l2",
      "to": "speaker:sam-altman"
    },
    {
      "type": "ABOUT",
      "from": "insight:episode:abc123:a1b2c3d4",
      "to": "topic:ai-regulation",
      "properties": {
        "confidence": 0.79
      }
    }
  ]
}
```

---

## Grounding Example

The above example shows a **grounded** Insight:

```text
Insight: "AI regulation will significantly lag behind the pace of innovation"
  grounded: true
  confidence: 0.85

  SUPPORTED_BY → Quote 1: "Regulation will lag innovation by 3–5 years..."
  SUPPORTED_BY → Quote 2: "We'll see laws that are already outdated..."
```

An **ungrounded** Insight would look like:

```json
{
  "id": "insight:episode:abc123:x9y8z7w6",
  "type": "Insight",
  "properties": {
    "text": "European approach may become the global standard",
    "episode_id": "episode:abc123",
    "grounded": false
  },
  "confidence": 0.45
}
```

Note: No `SUPPORTED_BY` edges exist for this Insight, and `grounded=false` is explicit.

---

## Version History

| Version | Date       | Changes                                               |
| ------- | ---------- | ----------------------------------------------------- |
| 1.0     | 2026-02-06 | Initial v1: Insight + Quote model, grounding contract |
