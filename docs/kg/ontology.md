# Podcast Knowledge Graph Ontology (v1)

## Status

v1 (lean, implementation-ready)

## Purpose

Define the canonical ontology contract for the Podcast Knowledge Graph (PKG):
  •  Node & edge types
  •  Required properties
  •  Identity (ID) rules
  •  Provenance & evidence requirements

This document is the source of truth for contributors. All kg.json outputs MUST conform to this ontology and the companion schema (schemas/kg.schema.json).

⸻

## Design Principles

  •  Minimal ontology: only what is required for attribution + aggregation + evidence.
  •  Evidence-first: anything extracted from ML must be traceable to transcript evidence.
  •  Stable IDs: global concepts must have stable identifiers across episodes.

⸻

## Top-Level Concepts

### Node Types (v1)

  •  Podcast
  •  Episode
  •  Speaker
  •  Topic
  •  Entity
  •  Claim

Edge Types (v1)
  •  HAS_EPISODE (Podcast → Episode)
  •  SPOKE_IN (Speaker → Episode)
  •  DISCUSSES (Episode → Topic)
  •  MENTIONS (Episode → Entity)
  •  ASSERTS (Speaker → Claim)
  •  ABOUT (Claim → Topic | Entity)
  •  RELATED_TO (Topic ↔ Topic)

⸻

## Common Properties

### Required on all nodes

  •  id (string)
  •  type (enum)
  •  properties (object; type-specific)

Required on all edges
  •  type (enum)
  •  from (node id)
  •  to (node id)
  •  properties (object; type-specific)

Provenance (required for ML-derived content)

Any node/edge produced by ML extraction MUST include:
  •  confidence (0.0–1.0)
  •  evidence object containing:
  •  episode_id
  •  transcript_ref (pointer to transcript artifact)
  •  char_start / char_end (span in transcript text)
  •  timestamp_start_ms / timestamp_end_ms
  •  extraction_method (e.g. llm, ner, rules)
  •  model_version

Note: confidence is extraction certainty, not factual truth.

⸻

## Identity Rules (IDs)

### Episode-scoped IDs

  •  Episode ID must be stable and derived from RSS entry GUID if available.
  •  Claim ID should be episode-scoped to avoid accidental global merging.

Recommended:
  •  episode:<rss_guid>
  •  claim:<episode_id>:<sha1(text_normalized)>

Global IDs (deduplicated across episodes)

These must be stable across episodes:
  •  Speaker
  •  Topic
  •  Entity

Recommended normalization:
  •  Speakers: speaker:<slug(name)> (optionally include podcast namespace if collisions occur)
  •  Topics: topic:<slug(label)>
  •  Entities: entity:<type>:<slug(name)>

Deduplication: Extraction and resolution are separate. The extractor may emit provisional IDs, but the resolver should converge to stable IDs over time.

⸻

## Node Definitions

### Podcast

**Definition:** A podcast feed.

Required properties:
  •  title (string)
  •  rss_url (string)

Optional:
  •  publisher (string)

⸻

### Episode

**Definition:** A single podcast episode.

Required properties:
  •  podcast_id (string)
  •  title (string)
  •  publish_date (ISO date-time string)

Optional:
  •  audio_url (string)
  •  duration_ms (integer)

⸻

### Speaker

**Definition:** A person speaking in the episode.

Required properties:
  •  name (string)

Optional:
  •  aliases (string[])

⸻

### Topic

**Definition:** An abstract subject discussed.

Required properties:
  •  label (string)

Optional:
  •  aliases (string[])

⸻

### Entity

**Definition:** A real-world named entity.

Required properties:
  •  name (string)
  •  entity_type (enum: person, company, product, place, org, event, other)

Optional:
  •  external_ids (object, e.g. { "wikidata": "Q..." })

⸻

### Claim

**Definition:** A declarative statement attributed to a speaker.

Required properties:
  •  text (string)
  •  speaker_id (string)
  •  episode_id (string)
  •  timestamp_start_ms (integer)
  •  timestamp_end_ms (integer)

Required provenance (see above):
  •  confidence
  •  evidence

⸻

## Edge Definitions

### HAS_EPISODE (Podcast → Episode)

**Required properties:** none

### SPOKE_IN (Speaker → Episode)

**Required properties:** none

### DISCUSSES (Episode → Topic)

Required properties:
  •  confidence
  •  evidence

MENTIONS (Episode → Entity)

Required properties:
  •  confidence
  •  evidence

ASSERTS (Speaker → Claim)

**Required properties:** none (Claim already carries evidence/provenance)

### ABOUT (Claim → Topic | Entity)

Required properties:
  •  confidence
  •  evidence

RELATED_TO (Topic ↔ Topic)

Required properties:
  •  confidence
  •  evidence

⸻

## Required Output Artifact: kg.json

Each episode output folder contains a kg.json capturing nodes/edges introduced or referenced by the episode.

Guidance:
  •  Episode-local Claim nodes live here.
  •  Global nodes (Topic, Entity, Speaker) may be referenced or introduced.
  •  The logical full KG is the union of all episode kg.json files.

⸻

## Minimal Example

```json
{
  "schema_version": "1.0",
  "episode_id": "episode:abc123",
  "nodes": [
    {"id": "podcast:the-journal", "type": "Podcast", "properties": {"title": "The Journal", "rss_url": "https://..."}},
    {"id": "episode:abc123", "type": "Episode", "properties": {"podcast_id": "podcast:the-journal", "title": "AI Regulation", "publish_date": "2026-02-03T00:00:00Z"}},
    {"id": "speaker:sam-altman", "type": "Speaker", "properties": {"name": "Sam Altman"}},
    {"id": "topic:ai-regulation", "type": "Topic", "properties": {"label": "AI Regulation"}},
    {
      "id": "claim:episode:abc123:5d41402abc4b2a76b9719d911017c592",
      "type": "Claim",
      "properties": {
        "text": "Regulation will lag innovation by 3–5 years.",
        "speaker_id": "speaker:sam-altman",
        "episode_id": "episode:abc123",
        "timestamp_start_ms": 120000,
        "timestamp_end_ms": 135000
      },
      "confidence": 0.82,
      "evidence": {
        "episode_id": "episode:abc123",
        "transcript_ref": "transcript.json",
        "char_start": 10234,
        "char_end": 10321,
        "timestamp_start_ms": 120000,
        "timestamp_end_ms": 135000,
        "extraction_method": "llm",
        "model_version": "gpt-4.1-mini-2026-01-xx"
      }
    }
  ],
  "edges": [
    {"type": "HAS_EPISODE", "from": "podcast:the-journal", "to": "episode:abc123", "properties": {}},
    {"type": "SPOKE_IN", "from": "speaker:sam-altman", "to": "episode:abc123", "properties": {}},
    {
      "type": "ASSERTS",
      "from": "speaker:sam-altman",
      "to": "claim:episode:abc123:5d41402abc4b2a76b9719d911017c592",
      "properties": {}
    },
    {
      "type": "ABOUT",
      "from": "claim:episode:abc123:5d41402abc4b2a76b9719d911017c592",
      "to": "topic:ai-regulation",
      "properties": {
        "confidence": 0.79,
        "evidence": {
          "episode_id": "episode:abc123",
          "transcript_ref": "transcript.json",
          "char_start": 10234,
          "char_end": 10321,
          "timestamp_start_ms": 120000,
          "timestamp_end_ms": 135000,
          "extraction_method": "llm",
          "model_version": "gpt-4.1-mini-2026-01-xx"
        }
      }
    }
  ]
}
```
