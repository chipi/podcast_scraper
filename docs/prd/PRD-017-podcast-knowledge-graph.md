# PRD-017: Podcast Knowledge Graph (PKG)

- **Status**: 📋 Draft
- **Authors**: Podcast Scraper Team
- **Related RFCs**: RFC-049 (Core Concepts), RFC-050 (Use Cases)
- **Related Documents**:
  - `docs/kg/ontology.md` - Human-readable ontology specification
  - `docs/kg/kg.schema.json` - Machine-readable schema

## Summary

The **Podcast Knowledge Graph (PKG)** transforms podcast content from isolated, linear transcripts into a structured, queryable knowledge graph that captures entities, topics, claims, and relationships. This enables deep semantic querying, cross-podcast comparison, and downstream AI applications while preserving attribution, timestamps, and evidence.

Each podcast episode incrementally updates a shared world model rather than producing standalone outputs, enabling non-linear exploration of podcast content and supporting future RAG, analytics, and trend detection use cases.

## Background & Context

Podcast content is rich in insights, opinions, and factual knowledge, but is currently consumed as linear, unstructured text (transcripts, summaries). This makes it difficult to:

- Compare ideas across episodes and podcasts
- Track how opinions and topics evolve over time
- Attribute claims to speakers with evidence
- Query knowledge in a precise, explainable way

The existing podcast scraper produces transcripts and summaries, but loses relational meaning. The Knowledge Graph addresses this by extracting structured knowledge from transcripts while preserving attribution, timestamps, and confidence scores.

**How it relates to existing features:**

- **Transcripts** (PRD-001): KG extraction operates on existing transcript outputs
- **Summarization** (PRD-005): KG complements summaries by providing structured, queryable knowledge
- **Metadata** (PRD-004): KG data is co-located with existing metadata artifacts
- **Speaker Detection** (PRD-008): KG leverages detected speakers for attribution

## Goals

1. **Extract Structured Knowledge**: Identify entities, topics, and claims from transcripts with ML-based extraction
2. **Preserve Attribution**: Link all extracted knowledge to speakers, episodes, and transcript evidence
3. **Enable Non-Linear Exploration**: Support graph-based queries that go beyond keyword search
4. **Support Future Use Cases**: Provide foundation for RAG, analytics, and trend detection applications

## Non-Goals (v1)

- Perfect factual correctness (confidence scores indicate extraction certainty, not truth)
- Real-time processing (batch processing per episode)
- Full ontology coverage of all domains (minimal v1 ontology focused on core concepts)
- Automated truth validation (evidence-backed extraction, not fact-checking)
- Sentiment analysis (deferred to post-v1)
- Disagreement/contradiction modeling (deferred to post-v1)
- Trend detection and temporal analytics (deferred to post-v1)
- Advanced entity intelligence dashboards (deferred to post-v1)
- Real-time or streaming ingestion (deferred to post-v1)

## Personas

- **Knowledge Workers & Researchers**: Need to quickly find and compare insights across multiple podcast episodes without listening to hours of audio
- **Founders, Investors, Analysts**: Require credible, attributable claims for due diligence and market intelligence
- **Developers Building AI Agents**: Need structured, queryable knowledge for RAG systems and downstream applications

## User Stories

- _As a researcher, I can query a topic and retrieve all related episodes and speakers so that I can understand how a subject is discussed across podcasts._
- _As an analyst, I can extract concrete statements with original context so that I can cite claims with evidence._
- _As a developer, I can query the knowledge graph programmatically so that I can build AI agents on top of podcast data._
- _As a knowledge worker, I can understand what a speaker believes across episodes so that I can assess credibility and consistency._

## Functional Requirements

### FR1: Knowledge Extraction

- **FR1.1**: Extract topics from episode transcripts with confidence scores and evidence
- **FR1.2**: Extract entities (persons, companies, products, places) from transcripts with entity type classification
- **FR1.3**: Extract declarative claims from transcripts, attributing each claim to a speaker
- **FR1.4**: Attach timestamp ranges and transcript spans to all extracted knowledge

### FR2: Graph Construction

- **FR2.1**: Deduplicate nodes (topics, entities, speakers) across episodes using stable IDs
- **FR2.2**: Create relationships between nodes (DISCUSSES, MENTIONS, ASSERTS, ABOUT, RELATED_TO)
- **FR2.3**: Attach confidence scores and provenance metadata to all ML-derived nodes and edges
- **FR2.4**: Maintain episode-local Claim nodes while referencing global nodes (Topic, Entity, Speaker)

### FR3: Storage & Output

- **FR3.1**: Generate `kg.json` file per episode in the episode output directory
- **FR3.2**: Co-locate KG data with existing artifacts (transcript.json, summary.json, metadata.json)
- **FR3.3**: Ensure all `kg.json` files conform to the machine-readable schema (`docs/kg/kg.schema.json`)
- **FR3.4**: Include schema version in each `kg.json` file for evolution tracking

### FR4: Query & Consumption (v1-Scoped)

- **FR4.1**: Support structured queries over KG data (topic, speaker, claim)
- **FR4.2**: Return evidence-backed results with links to transcript spans
- **FR4.3**: Enable programmatic access to KG data via JSON files
- **FR4.4**: Support basic graph traversals (Topic → Episode → Speaker, Speaker → Claim → Topic)

## Success Metrics

- **Extraction Coverage**: >=80% of episodes successfully converted into graph updates
- **Knowledge Density**: Average >=5 topics and >=3 entities extracted per episode
- **Attribution Accuracy**: >=90% claim attribution accuracy (manual sampling)
- **Query Usefulness**: Positive developer feedback on query patterns and output shapes
- **Schema Compliance**: 100% of generated `kg.json` files pass schema validation

## Dependencies

- **PRD-001**: Transcript Acquisition Pipeline (KG operates on transcripts)
- **PRD-004**: Metadata Generation (KG data co-located with metadata)
- **PRD-005**: Episode Summarization (KG complements summaries)
- **PRD-008**: Speaker Name Detection (KG uses detected speakers for attribution)
- **RFC-049**: Core KG Concepts & Data Model (defines ontology and storage)
- **RFC-050**: KG Use Cases & Consumption (defines query patterns and integration)

## Constraints & Assumptions

**Constraints:**

- Must not require global graph storage in v1 (logical union of per-episode files)
- Must be backward compatible with existing output directory structure
- Must complete extraction within reasonable time bounds (similar to summarization)
- Must not break existing workflows when disabled

**Assumptions:**

- ML models (LLMs, NER) are available for extraction
- Transcripts are available before KG extraction
- Users have sufficient storage for per-episode `kg.json` files
- Schema validation can be manual initially, automated later via CI

## Design Considerations

### Ontology Design

- **Option A**: Minimal ontology (Podcast, Episode, Speaker, Topic, Entity, Claim)
  - **Pros**: Simple, focused, easier to implement and validate
  - **Cons**: May need expansion for advanced use cases
  - **Decision**: Option A (minimal v1 ontology, expandable via RFC updates)

### Storage Strategy

- **Option A**: Per-episode `kg.json` files (logical union for global graph)
  - **Pros**: Easy debugging, natural sharding, reprocessable, co-located with outputs
  - **Cons**: Global queries require scanning multiple files
  - **Decision**: Option A (per-episode storage, global queries deferred to post-v1)

### Claim Normalization

- **Option A**: Episode-scoped Claim IDs (no global deduplication)
  - **Pros**: Simpler, avoids accidental merging of similar claims
  - **Cons**: Cannot detect repeated claims across episodes
  - **Decision**: Option A (episode-scoped in v1, global deduplication deferred)

## Integration with Existing Features

The Knowledge Graph enhances the pipeline by:

- **Transcript Processing**: KG extraction runs after transcript acquisition, using transcript content as input
- **Metadata Generation**: KG data is co-located with metadata artifacts in the same episode directory
- **Summarization**: KG complements summaries by providing structured, queryable knowledge alongside narrative summaries
- **Speaker Detection**: KG uses detected speakers (from PRD-008) for claim attribution and speaker-topic associations

## Example Output

```json
{
  "schema_version": "1.0",
  "episode_id": "episode:abc123",
  "nodes": [
    {
      "id": "podcast:the-journal",
      "type": "Podcast",
      "properties": {
        "title": "The Journal",
        "rss_url": "https://..."
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
    {
      "type": "HAS_EPISODE",
      "from": "podcast:the-journal",
      "to": "episode:abc123",
      "properties": {}
    },
    {
      "type": "SPOKE_IN",
      "from": "speaker:sam-altman",
      "to": "episode:abc123",
      "properties": {}
    },
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

## Open Questions

1. How to model disagreement between claims? (Deferred to post-v1)
2. How to represent uncertainty vs opinion? (Deferred to post-v1)
3. Should claims be normalized or episode-specific? (Decision: Episode-specific in v1)
4. When does global graph storage become necessary? (Post-v1 consideration)
5. How aggressively should entity resolution be applied in v1? (TBD during implementation)

## Related Work

- **RFC-049**: Core KG Concepts & Data Model - Defines ontology, storage, and schema
- **RFC-050**: KG Use Cases & Consumption - Defines query patterns and integration
- **docs/kg/ontology.md**: Human-readable ontology specification
- **docs/kg/kg.schema.json**: Machine-readable schema for validation

## Release Checklist

- [ ] PRD reviewed and approved
- [ ] RFC-049 created with technical design (Core Concepts)
- [ ] RFC-050 created with technical design (Use Cases)
- [ ] Ontology specification (`docs/kg/ontology.md`) finalized
- [ ] Schema specification (`docs/kg/kg.schema.json`) finalized
- [ ] Implementation completed
- [ ] Tests cover extraction, graph construction, and query patterns
- [ ] Documentation updated (README, config examples)
- [ ] Integration with transcript pipeline verified
- [ ] Schema validation integrated (manual or CI)
